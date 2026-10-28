"""Axis 2: is a query's shortlist enriched for models that carry the query's alteration?

Scored per disease and alteration (``resources/mutation_tasks.yaml``). The queries are the
tumours of one disease, the pool the models of the same disease, each labelled mutant or
wild type for one alteration. For every query the share s_k of its k nearest pool models
that are mutant is computed in the embedding, and summarised by the **AUROC**: the
probability that a mutant query's s_k exceeds a wild-type query's, ties counting one half.
Mutant and wild-type queries share the disease and the pool, so 0.5 means the neighbours
carry no information about the alteration beyond the disease, and a difference in
prevalence between the two datasets cancels.

Uncertainty comes from resampling the queries, which leaves the embedding alone. Query
clusters are bootstrapped for the 95% intervals: a TCGA tumour's tissue source site, any
other tumour's patient, so the samples that share a batch move together. Queries and pool
are sorted by sample id first, so every method sees the same resamples.

Two bounds come from the labels alone: ``random`` (AUROC 0.5) and ``oracle`` (every query
retrieves models of its own status first).

Each pair's primary analyses are also summarised by one AUROC: the mean over the analyses
of each cancer, then over the cancers, a cancer with n query patients weighing n^0.5, as a
label does on the ontology axis. Pairs stay apart, as they ask different questions (seen in
training or not). Its interval resamples the query clusters within each cancer, once for
all of its analyses, so analyses sharing patients move together. Each method's aggregate
is compared with the baseline's on the same resamples (``retrieval.compare``), and so is each
analysis at the primary depth, those differences BH-corrected within each tier and pair.
"""

from collections import defaultdict
from collections.abc import Callable
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.stats import false_discovery_control

from mapping_benchmark.mutation_labels import validate_labels, validate_sample_ids
from mapping_benchmark.mutation_tasks import Analysis, Disease, Tasks
from mapping_benchmark.retrieval import (
    LABEL_WEIGHT,
    Bound,
    Interval,
    bootstrap,
    compare,
    interval,
    nearest_neighbours,
    resampling_units,
)


class KResult(BaseModel):
    auroc: float
    auroc_ci: Interval | None = None
    # One AUROC per resample, in resample order, for comparisons with other methods.
    auroc_draws: list[float] = []


# Cohorts are fixed before loading any embedding method.


def select_case_side(
    labels: pd.DataFrame,
    units: pd.DataFrame,
    disease: Disease,
    dataset: str,
    analysis: Analysis,
    side: str,
) -> tuple[pd.DataFrame, dict, list[str]]:
    """Filter one case side and account for each exclusion before scoring."""
    rows = units[units.dataset.eq(dataset) & units.oncotree_code.isin(disease.oncotree)]
    reasons = []
    count = {
        "units": len(rows),
        "replicates_excluded": int(
            (
                labels.dataset.eq(dataset)
                & labels.oncotree_code.isin(disease.oncotree)
                & ~labels.representative
            ).sum()
        ),
    }
    count["msi_unknown"] = int(rows.MSI.isna().sum())

    # Excluding positive screens does not turn the remaining unknowns into MSS.
    before = len(rows)
    if disease.exclude_msi_positive:
        rows = rows[~rows.msi_positive_screen]
    count["msi_positive_excluded"] = before - len(rows)

    before = len(rows)
    if analysis.stratum != "all":
        rows = rows[rows.MSI.eq(int(analysis.stratum == "MSI")).fillna(False)]
    count["stratum_excluded"] = before - len(rows)

    # Only explicit 0/1 labels enter either side of the analysis.
    column = "MSI_MANTIS" if side == "query" and analysis.query_mantis_only else analysis.biomarker
    if analysis.query_mantis_only and side == "query" and dataset != "tcga":
        reasons.append("MANTIS-only sensitivity requires TCGA queries")
    count.update(
        positive=int(rows[column].eq(1).sum()),
        control=int(rows[column].eq(0).sum()),
        unknown=int(rows[column].isna().sum()),
    )
    known = rows[rows[column].notna()]
    count["patients"] = int(known.patient_id.nunique())

    return known, count, reasons


def select_cases(labels: pd.DataFrame, tasks: Tasks) -> list[dict]:
    """Prespecified cohorts shared by all methods and bounds, including failed tasks."""
    validate_labels(labels, tasks)
    units = labels[labels.representative].sort_index(kind="stable")
    cases = []

    for disease_name, disease in tasks.diseases.items():
        for pair_name, pair in tasks.pairs.items():
            for analysis_name, analysis in disease.analyses.items():
                sides, counts = {}, {}
                reasons = []
                for side, dataset in (("query", pair.query), ("pool", pair.database)):
                    known, count, side_reasons = select_case_side(
                        labels,
                        units,
                        disease,
                        dataset,
                        analysis,
                        side,
                    )
                    sides[side] = known.index.tolist()
                    counts[side] = count
                    reasons.extend(side_reasons)

                rule = tasks.eligibility
                for side, status, minimum in (
                    ("query", "positive", rule.mutant_queries),
                    ("query", "control", rule.wildtype_queries),
                    ("pool", "positive", rule.mutant_pool),
                    ("pool", "control", rule.wildtype_pool),
                ):
                    n = counts[side][status]
                    if n < minimum:
                        reasons.append(f"{side} {status}: {n} < {minimum}")

                n_pool = counts["pool"]["positive"] + counts["pool"]["control"]
                available_k = [k for k in tasks.k if k <= n_pool]
                cases.append(
                    {
                        "disease": disease_name,
                        "pair": pair_name,
                        "analysis": analysis_name,
                        **analysis.model_dump(),
                        "query_dataset": pair.query,
                        "database_dataset": pair.database,
                        "eligible": not reasons,
                        "ineligible_reasons": reasons,
                        "available_k": available_k,
                        "unavailable_k": [k for k in tasks.k if k > n_pool],
                        "counts": counts,
                        "query_ids": sides["query"],
                        "pool_ids": sides["pool"],
                    }
                )
    return cases


# Embedding sample alignment


def load_embedding(folder: Path) -> tuple[pd.DataFrame, np.ndarray]:
    embedding_samples = pd.read_csv(
        folder / "latents.csv",
        index_col=0,
        dtype={"sample_id": str, "dataset": str},
        keep_default_na=False,
    )
    validate_sample_ids(embedding_samples)
    latents = np.load(folder / "latents.npy", mmap_mode="r", allow_pickle=False)
    if latents.ndim != 2 or len(latents) != len(embedding_samples) or latents.shape[1] == 0:
        raise ValueError(f"latent dimensions do not match embedding samples in {folder}")
    if not np.issubdtype(latents.dtype, np.number) or not np.isfinite(latents).all():
        raise ValueError(f"latents must be finite numeric values: {folder}")
    return embedding_samples, latents


def validate_alignment(
    embedding_samples: pd.DataFrame, cases: list[dict], method: str
) -> pd.Series:
    """Fail on missing samples or cross-dataset IDs instead of changing method cohorts."""
    indexed = embedding_samples.set_index("sample_id")
    for case in cases:
        if not case["eligible"] or not case["available_k"]:
            continue
        for side, dataset in (("query", case["query_dataset"]), ("pool", case["database_dataset"])):
            ids = case[f"{side}_ids"]
            if missing := set(ids) - set(indexed.index):
                raise ValueError(
                    f"{method} lacks {len(missing)} fixed-cohort samples: {sorted(missing)[:5]}"
                )
            if not indexed.loc[ids, "dataset"].eq(dataset).all():
                raise ValueError(f"{method}: dataset mismatch in {side} cohort")
    return pd.Series(np.arange(len(embedding_samples)), index=embedding_samples.sample_id)


# Evaluation and testing families

SCORE_COLUMNS = [
    "disease",
    "pair",
    "analysis",
    "biomarker",
    "stratum",
    "tier",
    "method",
    "bound",
    "k",
    "primary_depth",
    "n_query",
    "query_positive",
    "n_query_patients",
    "n_pool",
    "pool_positive",
    "auroc",
    "auroc_ci_low",
    "auroc_ci_high",
    "baseline",
    "delta",
    "delta_ci_low",
    "delta_ci_high",
    "delta_p",
    "delta_q",
    "fdr_family",
]


def score_record(
    case: dict, method: str, bound: str | None, k: int, metric: KResult, tasks: Tasks
) -> dict:
    record = {
        key: case[key] for key in ("disease", "pair", "analysis", "biomarker", "stratum", "tier")
    }
    record.update(
        method=method,
        bound=bound,
        k=k,
        primary_depth=k == tasks.primary_k,
        n_query=len(case["query_ids"]),
        n_pool=len(case["pool_ids"]),
        query_positive=case["counts"]["query"]["positive"],
        pool_positive=case["counts"]["pool"]["positive"],
        n_query_patients=case["counts"]["query"]["patients"],
        auroc=metric.auroc,
    )
    record["auroc_ci_low"], record["auroc_ci_high"] = metric.auroc_ci or (None, None)
    return record


def correct_multiplicity(scores: pd.DataFrame, tasks: Tasks) -> pd.DataFrame:
    """BH q-values of the differences from the baseline, for each tier at the primary depth.

    The differences are corrected within each pair of a tier, as the pairs are compared with
    the baseline one by one, so a pair with clear differences does not lower the bar for
    another. Sensitivity analyses and other depths keep raw p-values only.
    """
    scores = scores.copy()
    scores["delta_q"] = np.nan
    scores["fdr_family"] = None
    for tier in ("primary", "exploratory"):
        family = scores.tier.eq(tier) & scores.k.eq(tasks.primary_k) & scores.bound.isna()
        scores.loc[family, "fdr_family"] = f"{tier}@{tasks.primary_k}"
        for _, cells in scores[family & scores.delta_p.notna()].groupby("pair"):
            scores.loc[cells.index, "delta_q"] = false_discovery_control(
                cells.delta_p.to_numpy(float), method="bh"
            )
    return scores[SCORE_COLUMNS]


AGGREGATE_COLUMNS = [
    "pair",
    "method",
    "bound",
    "k",
    "primary_depth",
    "diseases",
    "n_analyses",
    "auroc",
    "auroc_ci_low",
    "auroc_ci_high",
    "baseline",
    "delta",
    "delta_ci_low",
    "delta_ci_high",
    "delta_p",
]


def cancer_sizes(labels: pd.DataFrame, cases: list[dict]) -> dict[str, dict[str, int]]:
    """Query patients per pair and cancer, over the cancer's eligible primary analyses."""
    ids = defaultdict(set)
    for case in cases:
        if case["tier"] == "primary" and case["eligible"]:
            ids[case["pair"], case["disease"]].update(case["query_ids"])
    sizes = defaultdict(dict)
    for (pair, disease), samples in ids.items():
        sizes[pair][disease] = labels.loc[sorted(samples), "patient_id"].nunique()
    return dict(sizes)


def disease_weights(diseases, sizes: dict[str, int]) -> np.ndarray:
    """Each cancer weighs its query patients^0.5, shared equally among its analyses."""
    names, code, counts = np.unique(np.asarray(diseases), return_inverse=True, return_counts=True)
    weights = np.array([sizes[d] for d in names], float) ** LABEL_WEIGHT
    return (weights / weights.sum())[code] / counts[code]


def aggregate(
    scores: pd.DataFrame,
    draws: dict,
    tasks: Tasks,
    sizes: dict[str, dict[str, int]],
    baseline: str | None = None,
) -> pd.DataFrame:
    """One AUROC per pair, method and depth over the primary analyses.

    ``draws`` holds each method's aggregate per resample, by (pair, method, k). At the
    primary depth, every method but the baseline is also compared with it.
    """
    records = []
    primary = scores[scores.tier.eq("primary")]
    for (pair, method, k), cells in primary.groupby(["pair", "method", "k"], sort=False):
        low, high = interval(draws.get((pair, method, k), [])) or (None, None)
        records.append(
            {
                "pair": pair,
                "method": method,
                "bound": cells.bound.iloc[0],
                "k": k,
                "primary_depth": k == tasks.primary_k,
                "diseases": "+".join(sorted(cells.disease.unique())),
                "n_analyses": len(cells),
                "auroc": float(disease_weights(cells.disease, sizes[pair]) @ cells.auroc),
                "auroc_ci_low": low,
                "auroc_ci_high": high,
            }
        )
    observed = {(r["pair"], r["method"], r["k"]): r["auroc"] for r in records}
    for r in records:
        key, reference = (r["pair"], r["method"], r["k"]), (r["pair"], baseline, r["k"])
        if r["primary_depth"] and r["method"] != baseline and key in draws and reference in draws:
            r.update(baseline=baseline, **compare(draws[key], draws[reference],
                                                  r["auroc"] - observed[reference]))
    return pd.DataFrame(records, columns=AGGREGATE_COLUMNS)


def evaluate(
    labels: pd.DataFrame,
    tasks: Tasks,
    methods: dict[str, Path],
    progress: Callable[[str], None] | None = None,
    baseline: str | None = None,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """Select common cohorts, score all methods and bounds, then correct testing families.

    Also returns each pair's primary analyses as one AUROC per method (``aggregate``).
    At the primary depth, each analysis and each aggregate is compared with the
    ``baseline`` method's.
    """
    cases = select_cases(labels, tasks)
    sizes = cancer_sizes(labels, cases)
    records, draws = score_bounds(labels, tasks, cases, progress), {}

    # Each method must supply the exact same samples for every eligible cohort.
    for method, folder in methods.items():
        method_records, method_draws = score_method(
            labels, tasks, cases, sizes, method, folder, progress
        )
        records.extend(method_records)
        draws |= method_draws

    # Each analysis against the baseline's on the same resamples; the draws go no further.
    cell = itemgetter("disease", "pair", "analysis")
    reference = {cell(r): r for r in records if r["method"] == baseline and "draws" in r}
    for r in records:
        ref = reference.get(cell(r))
        if ref and r["method"] != baseline and "draws" in r:
            r.update(baseline=baseline, **compare(r["draws"], ref["draws"],
                                                  r["auroc"] - ref["auroc"]))

    # Correct across the declared run after collecting all methods.
    scores = correct_multiplicity(pd.DataFrame(records, columns=SCORE_COLUMNS), tasks)
    return cases, scores, aggregate(scores, draws, tasks, sizes, baseline)


def score_bounds(
    labels: pd.DataFrame,
    tasks: Tasks,
    cases: list[dict],
    progress: Callable[[str], None] | None = None,
) -> list[dict]:
    """Calculate label-only references and report ineligible cohorts."""
    records = []

    # Reference bounds depend only on the fixed labels and pool composition.
    for case in cases:
        if not case["eligible"]:
            if progress:
                progress(
                    f"{case['disease']}/{case['pair']}/{case['analysis']}: "
                    + "; ".join(case["ineligible_reasons"])
                )
            continue
        q_column = "MSI_MANTIS" if case["query_mantis_only"] else case["biomarker"]
        query_positive = labels.loc[case["query_ids"], q_column].to_numpy(bool)
        pool_positive = labels.loc[case["pool_ids"], case["biomarker"]].to_numpy(bool)
        for bound in ("random", "oracle"):
            for k, metric in evaluate_alteration(
                None,
                query_positive,
                pool_positive,
                np.array(case["query_ids"]),
                case["available_k"],
                0,
                tasks.seed,
                bound=bound,
            ).items():
                records.append(score_record(case, bound.title(), bound, k, metric, tasks))
    return records


def score_method(
    labels: pd.DataFrame,
    tasks: Tasks,
    cases: list[dict],
    sizes: dict[str, dict[str, int]],
    method: str,
    folder: Path,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict]:
    """Align one embedding to the fixed cohorts, then score each available depth.

    Also returns each pair's aggregate AUROC per resample, by (pair, method, k).
    """
    records = []
    primary = defaultdict(list)
    embedding_samples, latents = load_embedding(folder)
    positions = validate_alignment(embedding_samples, cases, method)
    graphs = {}
    for case in cases:
        if not case["eligible"] or not case["available_k"]:
            continue
        query_ids, pool_ids = case["query_ids"], case["pool_ids"]
        key = (tuple(query_ids), tuple(pool_ids))
        if key not in graphs:
            graphs[key] = nearest_neighbours(
                latents[positions.loc[query_ids].to_numpy()],
                latents[positions.loc[pool_ids].to_numpy()],
                max(case["available_k"]),
            )

        q_column = "MSI_MANTIS" if case["query_mantis_only"] else case["biomarker"]
        query_samples = labels.loc[query_ids]
        query_positive = query_samples[q_column].to_numpy(bool)
        pool_positive = labels.loc[pool_ids, case["biomarker"]].to_numpy(bool)
        clusters = resampling_units(query_samples.patient_id)
        metrics = evaluate_alteration(
            graphs[key],
            query_positive,
            pool_positive,
            clusters,
            case["available_k"],
            tasks.bootstrap,
            tasks.seed,
        )
        for k, metric in metrics.items():
            records.append(score_record(case, method, None, k, metric, tasks))
            if k == tasks.primary_k and metric.auroc_draws:
                records[-1]["draws"] = metric.auroc_draws
            if case["tier"] == "primary":
                counts = pool_positive[graphs[key][:, :k]].sum(axis=1)
                primary[case["pair"], k].append((case["disease"], counts, query_positive, clusters))
        if progress:
            progress(f"{method}: {case['disease']}/{case['pair']}/{case['analysis']} scored")
    draws = {
        (pair, method, k): pooled_draws(
            cells, disease_weights([c[0] for c in cells], sizes[pair]), tasks.bootstrap, tasks.seed
        )
        for (pair, k), cells in primary.items()
    }
    return records, draws


# --- Scores -------------------------------------------------------------------------------


def auroc(counts: np.ndarray, mutant: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Mann-Whitney AUROC of integer scores (mutant neighbours), ties at one half.

    ``weights`` give each query a multiplicity, as a bootstrap resample does. NaN when one
    class has no weight. Bincounts rather than sklearn's roc_auc_score, which is ~50x
    slower and runs once per bootstrap resample.
    """
    w = np.ones(len(counts)) if weights is None else np.asarray(weights, float)
    levels = int(counts.max()) + 1 if len(counts) else 1
    pos = np.bincount(counts[mutant], weights=w[mutant], minlength=levels)
    neg = np.bincount(counts[~mutant], weights=w[~mutant], minlength=levels)
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def pooled_draws(cells: list[tuple], weights: np.ndarray, n: int, seed: int) -> np.ndarray:
    """The weighted mean AUROC over ``n`` joint resamples, each cancer resampled within itself.

    ``cells`` holds each analysis's (disease, mutant-neighbour counts, query labels, query
    clusters), ``weights`` each analysis's weight. One resample draws every cluster once for
    all analyses of its cancer, so a cluster drawn twice counts twice in each analysis it is
    in; the same clusters and seed give every method the same resamples.
    """
    ends = np.cumsum([len(c[1]) for c in cells])[:-1]

    def statistic(w: np.ndarray) -> float:
        parts = np.split(w, ends)
        return float(weights @ [auroc(c[1], c[2], part) for c, part in zip(cells, parts)])

    strata = np.concatenate([[c[0]] * len(c[1]) for c in cells])
    clusters = np.concatenate([c[3] for c in cells])
    return bootstrap(statistic, strata, clusters, n, np.random.default_rng(seed))


def bound_metrics(
    bound: Bound, query_mutant: np.ndarray, pool_mutant: np.ndarray, k: int
) -> KResult:
    if bound == "random":
        return KResult(auroc=0.5)
    # The oracle hands every query the pool models of its own status first.
    n, n_mutant = len(pool_mutant), int(pool_mutant.sum())
    best, worst = min(k, n_mutant) / k, max(0, k - (n - n_mutant)) / k
    return KResult(auroc=1.0 if best > worst else 0.5)


def evaluate_alteration(
    neighbours: np.ndarray | None,
    query_mutant: np.ndarray,
    pool_mutant: np.ndarray,
    clusters: np.ndarray,
    k_values: list[int],
    n_bootstrap: int,
    seed: int,
    bound: Bound | None = None,
) -> dict[int, KResult]:
    """Every k's metrics for one alteration; ``neighbours`` is unused for a bound."""
    query_mutant, pool_mutant = query_mutant.astype(bool), pool_mutant.astype(bool)
    if query_mutant.all() or not query_mutant.any() or pool_mutant.all() or not pool_mutant.any():
        return {}
    out = {}
    for k in k_values:
        if k > len(pool_mutant):
            raise ValueError(f"k={k} exceeds the pool of {len(pool_mutant)}")
        if bound:
            out[k] = bound_metrics(bound, query_mutant, pool_mutant, k)
            continue
        counts = pool_mutant[neighbours[:, :k]].sum(axis=1)
        # A resample that lacks mutant or wild-type queries has no AUROC and is skipped.
        draws = bootstrap(
            lambda w: auroc(counts, query_mutant, w),
            np.zeros(len(clusters)),
            clusters,
            n_bootstrap,
            np.random.default_rng(seed),
        )
        out[k] = KResult(auroc=auroc(counts, query_mutant), auroc_ci=interval(draws),
                         auroc_draws=draws.tolist())
    return out
