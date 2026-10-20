"""Axis 2: is a query's shortlist enriched for models that carry the query's alteration?

Scored per disease and alteration (``resources/mutation_tasks.yaml``). The queries are the
tumours of one disease, the pool the models of the same disease, each labelled mutant or
wild type for one alteration. For every query the share s_k of its k nearest pool models
that are mutant is computed in the embedding, and summarised by the **AUROC**: the
probability that a mutant query's s_k exceeds a wild-type query's, ties counting one half.
Mutant and wild-type queries share the disease and the pool, so 0.5 means the neighbours
carry no information about the alteration beyond the disease, and a difference in
prevalence between the two datasets cancels.

Uncertainty comes from two resamplings that leave the embedding alone. Patients are
bootstrapped (a patient's aliquots move together) for the 95% intervals, and the pool's
labels are permuted for the p-values. A permutation keeps the neighbour graph, so a pool
model that is everyone's neighbour is part of the null. Queries and pool are sorted by
sample id first, so every method sees the same resamples and the same permutations.

Two bounds come from the labels alone: ``random`` (AUROC 0.5) and ``oracle`` (every query
retrieves models of its own status first).
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.stats import false_discovery_control

from mapping_benchmark.mutation_labels import validate_labels, validate_sample_ids
from mapping_benchmark.mutation_tasks import Analysis, Disease, Tasks
from mapping_benchmark.retrieval import Bound, nearest_neighbours

Interval = tuple[float, float]


class KResult(BaseModel):
    auroc: float
    auroc_ci: Interval | None = None
    auroc_p: float | None = None


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
    "auroc_p",
    "auroc_q",
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
        auroc_p=metric.auroc_p,
    )
    record["auroc_ci_low"], record["auroc_ci_high"] = metric.auroc_ci or (None, None)
    return record


def correct_multiplicity(scores: pd.DataFrame, tasks: Tasks) -> pd.DataFrame:
    """Separate BH families for each tier, at the declared primary depth.

    Each family spans all scored diseases, pairs and methods in this run. Sensitivity
    analyses and other depths retain raw p-values; they cannot enter the primary family.
    """
    scores = scores.copy()
    scores["auroc_q"] = np.nan
    scores["fdr_family"] = None
    for tier in ("primary", "exploratory"):
        family = scores.tier.eq(tier) & scores.k.eq(tasks.primary_k) & scores.bound.isna()
        scores.loc[family, "fdr_family"] = f"{tier}@{tasks.primary_k}"
        keep = family & scores.auroc_p.notna()
        if keep.any():
            scores.loc[keep, "auroc_q"] = false_discovery_control(
                scores.loc[keep, "auroc_p"].to_numpy(float), method="bh"
            )
    return scores[SCORE_COLUMNS]


def evaluate(
    labels: pd.DataFrame,
    tasks: Tasks,
    methods: dict[str, Path],
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    """Select common cohorts, score all methods and bounds, then correct testing families."""
    cases = select_cases(labels, tasks)
    records = score_bounds(labels, tasks, cases, progress)

    # Each method must supply the exact same samples for every eligible cohort.
    for method, folder in methods.items():
        records.extend(score_method(labels, tasks, cases, method, folder, progress))

    # Correct across the declared run after collecting all methods.
    scores = pd.DataFrame(records, columns=SCORE_COLUMNS)
    return cases, correct_multiplicity(scores, tasks)


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
    method: str,
    folder: Path,
    progress: Callable[[str], None] | None = None,
) -> list[dict]:
    """Align one embedding to the fixed cohorts, then score each available depth."""
    records = []
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
        metrics = evaluate_alteration(
            graphs[key],
            query_positive,
            pool_positive,
            query_samples.patient_id.to_numpy(str),
            case["available_k"],
            tasks.bootstrap,
            tasks.permutations,
            tasks.seed,
        )
        for k, metric in metrics.items():
            records.append(score_record(case, method, None, k, metric, tasks))
        if progress:
            progress(f"{method}: {case['disease']}/{case['pair']}/{case['analysis']} scored")
    return records


# --- Scores -------------------------------------------------------------------------------


def auroc(counts: np.ndarray, mutant: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Mann-Whitney AUROC of integer scores (mutant neighbours), ties at one half.

    ``weights`` give each query a multiplicity, as a bootstrap resample does. NaN when one
    class has no weight. Bincounts rather than sklearn's roc_auc_score, which is ~50x
    slower and runs ~2,000 times per alteration in the bootstrap and permutations.
    """
    w = np.ones(len(counts)) if weights is None else np.asarray(weights, float)
    levels = int(counts.max()) + 1 if len(counts) else 1
    pos = np.bincount(counts[mutant], weights=w[mutant], minlength=levels)
    neg = np.bincount(counts[~mutant], weights=w[~mutant], minlength=levels)
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def score(
    neighbours: np.ndarray, query_mutant: np.ndarray, pool_mutant: np.ndarray, k: int
) -> float:
    """AUROC at depth k."""
    return auroc(pool_mutant[neighbours[:, :k]].sum(axis=1), query_mutant)


def bootstrap(
    neighbours: np.ndarray,
    query_mutant: np.ndarray,
    pool_mutant: np.ndarray,
    patients: np.ndarray,
    k: int,
    n: int,
    rng: np.random.Generator,
) -> Interval | None:
    """95% percentile interval of the AUROC over patient-cluster resamples of the queries.

    A resample that lacks mutant or wild-type queries has no AUROC and is skipped.
    """
    counts = pool_mutant[neighbours[:, :k]].sum(axis=1)
    _, cluster = np.unique(patients, return_inverse=True)
    n_clusters = cluster.max() + 1
    draws = []
    for _ in range(n):
        weights = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)[
            cluster
        ].astype(float)
        draws.append(auroc(counts, query_mutant, weights))
    draws = np.asarray(draws, float)
    draws = draws[np.isfinite(draws)]
    return tuple(float(v) for v in np.percentile(draws, [2.5, 97.5])) if len(draws) else None


def permutation_p(
    neighbours: np.ndarray,
    query_mutant: np.ndarray,
    pool_mutant: np.ndarray,
    k: int,
    observed: float,
    n: int,
    rng: np.random.Generator,
) -> float:
    """One-sided p-value of the AUROC against shuffled pool labels."""
    exceed = sum(
        score(neighbours, query_mutant, rng.permutation(pool_mutant), k) >= observed - 1e-12
        for _ in range(n)
    )
    return (1 + exceed) / (1 + n)


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
    patients: np.ndarray,
    k_values: list[int],
    n_bootstrap: int,
    n_permutations: int,
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
        auc = score(neighbours, query_mutant, pool_mutant, k)
        rng = np.random.default_rng(seed)
        ci = bootstrap(neighbours, query_mutant, pool_mutant, patients, k, n_bootstrap, rng)
        p = (
            permutation_p(neighbours, query_mutant, pool_mutant, k, auc, n_permutations, rng)
            if n_permutations
            else None
        )
        out[k] = KResult(auroc=auc, auroc_ci=ci, auroc_p=p)
    return out
