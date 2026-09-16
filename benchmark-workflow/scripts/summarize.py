"""Collect every result JSON into one table.

``summary.csv`` has one row per (axis, pair, cohort, method, k) with the axis's score:
simGIC@k on the ontology axis, Jaccard@k on the mutation axis, precision@k on the donor
axis. ``summary.md`` shows the same numbers as one table per axis, methods as columns
with the two bounds last.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

BOUNDS = ["Random", "Oracle"]
SCORE = {"ontology": "simGIC", "mutation": "Jaccard", "donor": "precision"}


def rows(results_dir: Path):
    for axis in SCORE:
        for path in sorted((results_dir / axis).glob("**/*.json")):
            parts = path.relative_to(results_dir / axis).with_suffix("").parts
            pair, cohort, method = (parts[0], "", parts[1]) if len(parts) == 2 else parts
            result = json.loads(path.read_text())
            for k, metrics in result["metrics"].items():
                yield {
                    "axis": axis, "pair": pair, "cohort": cohort, "method": method,
                    "k": int(k),
                    "score": round(metrics.get("precision", metrics.get("mean_jaccard")), 4),
                    "n_query": result["n_query"], "n_database": result["n_database"],
                }


def markdown(summary: pd.DataFrame) -> str:
    out = []
    for axis, score in SCORE.items():
        frame = summary[summary["axis"] == axis]
        if frame.empty:
            continue
        index = ["pair", "cohort", "k"] if axis == "mutation" else ["pair", "k"]
        table = frame.pivot_table(index=index, columns="method", values="score")
        methods = [m for m in table.columns if m not in BOUNDS] + \
                  [b for b in BOUNDS if b in table.columns]
        table = table[methods].reset_index()
        out.append(f"## {axis.capitalize()} axis ({score}@k)\n")
        out.append(_table(table))
    return "\n".join(out)


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
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    summary = pd.DataFrame(list(rows(args.results_dir)))
    if summary.empty:
        raise SystemExit(f"no results under {args.results_dir}")
    summary = summary.sort_values(["axis", "pair", "cohort", "k", "method"], ignore_index=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.csv, index=False)
    text = markdown(summary)
    args.markdown.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
