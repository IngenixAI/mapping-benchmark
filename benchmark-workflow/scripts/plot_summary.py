"""One figure with the three axes: each row a bar from Random to Oracle, methods as marks."""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

AXES = [("ontology", "simGIC@k"), ("mutation", "Jaccard@k"), ("donor", "precision@k")]
MARKERS = ["o", "D", "^", "v", "s", "P", "X", "*"]
COLOURS = ["#3a6fd4", "#e08a1e", "#1e8e63", "#c74b96", "#c0392b", "#7f7f7f", "#8c564b"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="summary.csv")
    parser.add_argument("--output", type=Path, required=True, help="PDF; a PNG is written too")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary, keep_default_na=False)
    summary["row"] = summary["pair"].str.replace("_to_", " → ")
    summary.loc[summary["cohort"] != "", "row"] += " · " + summary["cohort"]
    methods = [m for m in summary["method"].unique() if m not in ("Random", "Oracle")]
    style = {m: (MARKERS[i % len(MARKERS)], COLOURS[i % len(COLOURS)])
             for i, m in enumerate(methods)}

    panels = [(axis, label, summary[summary["axis"] == axis]) for axis, label in AXES]
    panels = [(axis, label, f) for axis, label, f in panels if not f.empty]
    heights = [max(1.0, f["row"].nunique() * 0.55 + 0.6) for _, _, f in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, sum(heights)), squeeze=False,
                             gridspec_kw={"height_ratios": heights})
    for ax, (axis, label, frame) in zip(axes[:, 0], panels):
        k = frame["k"].max()
        frame = frame[frame["k"] == k]
        rows = list(dict.fromkeys(frame["row"]))
        for y, row in enumerate(rows):
            cell = frame[frame["row"] == row].set_index("method")["score"]
            lo, hi = cell.get("Random", 0.0), cell.get("Oracle", 1.0)
            ax.barh(y, hi - lo, left=lo, height=0.6, color="#ececec", edgecolor="#555")
            ax.text(hi + 0.01, y, f"{hi:.2f}", va="center", fontsize=8)
            for method in methods:
                if method in cell:
                    ax.plot(cell[method], y, style[method][0], color=style[method][1],
                            markersize=7, label=method if y == 0 else None)
        ax.set_yticks(range(len(rows)), rows)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.08)
        ax.set_xlabel(label.replace("@k", f"@{k}"))
        ax.set_title(f"{axis.capitalize()} axis", loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Wrote {args.output} and {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
