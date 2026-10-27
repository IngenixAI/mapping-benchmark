"""Collect every result JSON into one table.

``summary.csv`` has one row per (axis, pair, disease, alteration, method, k, metric):
simGIC on the ontology axis and precision on the donor axis, each with its 95% interval,
and AUROC on the mutation axis, with its 95% interval, permutation p-value and the common
evaluator's Benjamini-Hochberg q-value. Corrections remain separate for
primary/exploratory tiers across the declared run. Each mutation pair's primary analyses
also get one row per method and k, metric ``mean AUROC``: the mean per cancer, then over
cancers, with its interval.
``summary.md`` shows tables per axis and tier, methods as columns and bounds last.
A star marks q < 0.05; unavailable/ineligible scores are shown as —, with counts and reasons.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BOUNDS = ["Random", "Oracle"]
SCORE = {"ontology": "simGIC", "donor": "precision"}
Q_STAR = 0.05


def rows(results_dir: Path, inputs: list[Path]):
    """Read only the result files requested by the current workflow."""
    for path in sorted(inputs):
        result = json.loads(path.read_text())
        if "cases" in result and "tasks" in result and "scores" in result:
            yield from mutation_rows(result)
            continue
        parts = path.resolve().relative_to(results_dir.resolve()).parts
        axis, pair = parts[:2]
        method = path.stem
        if axis in SCORE:
            for k, metrics in result["metrics"].items():
                ci = metrics.get("precision_ci") or (np.nan, np.nan)
                yield {"axis": axis, "pair": pair, "method": method, "k": int(k),
                       "metric": SCORE[axis], "score": round(metrics["precision"], 4),
                       "ci_low": round(ci[0], 4), "ci_high": round(ci[1], 4),
                       "eligible": True, "n_query": result["n_query"],
                       "n_database": result["n_database"]}


def mutation_rows(result: dict):
    """Read all cancers from the common result and preserve its correction families."""
    cases = {(c["disease"], c["pair"], c["analysis"]): c for c in result["cases"]}
    seen = set()

    def base(case, method, k):
        biomarker = result["tasks"]["biomarkers"].get(case["biomarker"])
        return {
            "axis": "mutation", "pair": case["pair"], "disease": case["disease"],
            "alteration": case["analysis"], "metric": "AUROC",
            "class": "+".join(biomarker["classes"]) if biomarker else "status",
            "tier": case["tier"], "stratum": case["stratum"], "method": method, "k": k,
            "primary_depth": k == result["tasks"]["primary_k"], "eligible": case["eligible"],
            "n_query": len(case["query_ids"]), "n_database": len(case["pool_ids"]),
            "n_query_mutant": case["counts"]["query"]["positive"],
            "n_database_mutant": case["counts"]["pool"]["positive"],
            "ineligible_reason": "; ".join(case["ineligible_reasons"]),
            "statistics_source": "shared", "fdr_family": None,
        }

    for score in result["scores"]:
        key = (score["disease"], score["pair"], score["analysis"])
        seen.add((*key, score["method"], score["k"]))
        yield base(cases[key], score["method"], score["k"]) | {
            "score": score["auroc"], "ci_low": score["auroc_ci_low"],
            "ci_high": score["auroc_ci_high"], "p_value": score.get("auroc_p"),
            "q_value": score.get("auroc_q"), "fdr_family": score["fdr_family"],
        }
    for row in result.get("aggregate", []):
        yield {"axis": "mutation", "pair": row["pair"], "disease": row["diseases"],
               "alteration": f"{row['n_analyses']} primary analyses", "metric": "mean AUROC",
               "tier": "primary", "stratum": "all", "method": row["method"], "k": row["k"],
               "primary_depth": row["primary_depth"], "eligible": True, "score": row["auroc"],
               "ci_low": row["auroc_ci_low"], "ci_high": row["auroc_ci_high"],
               "statistics_source": "shared"}
    for key, case in cases.items():
        for method in [*result["methods"], *BOUNDS]:
            for k in result["tasks"]["k"]:
                if (*key, method, k) not in seen:
                    yield base(case, method, k) | {"score": np.nan, "q_value": np.nan,
                                                  "p_value": np.nan}


COLUMNS = ["axis", "pair", "disease", "alteration", "class", "method", "k", "metric", "score",
           "ci_low", "ci_high", "p_value", "q_value", "eligible", "n_query", "n_query_mutant",
           "n_database", "n_database_mutant", "tier", "stratum", "primary_depth",
           "fdr_family", "statistics_source", "ineligible_reason"]


def markdown(summary: pd.DataFrame) -> str:
    sections = {"ontology": _retrieval, "mutation": _mutation, "donor": _retrieval}
    out = []
    for axis, section in sections.items():
        frame = summary[summary["axis"] == axis]
        if not frame.empty:
            out += section(axis, frame)
    return "\n".join(out)


def _retrieval(axis: str, frame: pd.DataFrame) -> list[str]:
    return [f"## {axis.capitalize()} axis ({SCORE[axis]}@k)\n",
            _table(_pivot(frame, ["pair", "k"]))]


def _mutation(_: str, frame: pd.DataFrame) -> list[str]:
    out = []
    pooled = frame[frame["metric"] == "mean AUROC"].rename(columns={"disease": "cancers"})
    if not pooled.empty:
        pooled = pooled.assign(score=[
            f"{s:.3f}" + (f" [{lo:.3f}, {hi:.3f}]" if pd.notna(lo) else "")
            for s, lo, hi in zip(pooled["score"], pooled["ci_low"], pooled["ci_high"])])
        out.append("## Mutation axis: primary analyses per pair, cancers weighted n^0.5 "
                   "(AUROC@k)\n")
        out.append(_table(_pivot(pooled, ["pair", "cancers", "k"], aggfunc="first")))
        out.append("Mean AUROC per cancer, then over cancers; [95% interval] from joint "
                   "cluster resamples within each cancer.\n")
    frame = frame[frame["metric"] == "AUROC"]
    for tier, cells in frame.groupby("tier", sort=False):
        cells = cells.copy()
        cells["score"] = [_cell(s, q, e) for s, q, e in zip(
            cells["score"], cells["q_value"].fillna(1.0), cells["eligible"].astype(bool))]
        out.append(f"## Mutation axis: {tier} (AUROC@k)\n")
        out.append(_table(_pivot(cells, ["pair", "disease", "alteration", "stratum", "k"],
                                 aggfunc="first")))
    out.append("\\* Benjamini-Hochberg q < 0.05 against shuffled pool labels. "
               "Primary and exploratory families are separate; shared results retain the "
               "runner's q-values. Unavailable and ineligible scores are shown as —.\n")
    counts = frame[frame["method"] == "Random"]
    counts = counts[["pair", "disease", "alteration", "class", "tier", "stratum",
                     "n_query_mutant", "n_query", "n_database_mutant", "n_database",
                     "eligible", "ineligible_reason"]].astype(
        {"n_query_mutant": int, "n_query": int, "n_database_mutant": int, "n_database": int})
    out.append("## Mutation axis: mutant / scored samples per cell\n")
    out.append(_table(counts.drop_duplicates()))
    return out


def _cell(score: float, q: float, eligible: bool) -> str:
    if pd.isna(score):
        return "—"
    text = f"{score:.3f}" + ("*" if eligible and q < Q_STAR else "")
    return text if eligible else f"({text})"


def _pivot(frame: pd.DataFrame, index: list[str], aggfunc="mean") -> pd.DataFrame:
    table = frame.pivot_table(index=index, columns="method", values="score", aggfunc=aggfunc)
    methods = [m for m in table.columns if m not in BOUNDS] + \
              [b for b in BOUNDS if b in table.columns]
    return table[methods].reset_index()


def _table(frame: pd.DataFrame) -> str:
    cells = [[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
             for row in frame.itertuples(index=False)]
    lines = ["| " + " | ".join(frame.columns) + " |",
             "|" + "|".join("---" for _ in frame.columns) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in cells]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True,
                        help="Current workflow result files; excludes stale files from older runs")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    summary = pd.DataFrame(list(rows(args.results_dir, args.inputs)))
    if summary.empty:
        raise SystemExit(f"no results under {args.results_dir}")
    summary = summary.reindex(columns=COLUMNS)
    text_columns = ["disease", "alteration", "class", "tier", "stratum", "ineligible_reason"]
    summary[text_columns] = summary[text_columns].fillna("")
    summary = summary.sort_values(
        ["axis", "pair", "disease", "alteration", "k", "metric", "method"], ignore_index=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.csv, index=False)
    text = markdown(summary)
    args.markdown.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
