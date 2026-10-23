"""Overview and per-cancer figures with one fixed worst-to-best method order.

Ontology/donor use Random-to-Oracle bars. Mutation uses AUROC and bootstrap intervals.
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mapping_benchmark.mutation_visualization import model_ranking

matplotlib.use("Agg")

# axis, the metric plotted, its axis label
BARS = [("ontology", "simGIC", "simGIC@k"), ("donor", "precision", "precision@k")]
MARKERS = ["o", "D", "^", "v", "s", "P", "X", "*"]
COLOURS = ["#3a6fd4", "#e08a1e", "#1e8e63", "#c74b96", "#c0392b", "#7f7f7f", "#8c564b",
           "#17a2b8"]
MUTED = "#777"


def plot_bars(fig, panels, methods, style) -> None:
    """One panel per axis, one row per pair: the Random-to-Oracle bar and each method."""
    heights = [max(1.0, f["row"].nunique() * 0.55 + 0.6) for _, _, f in panels]
    axes = fig.subplots(len(panels), 1, squeeze=False,
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


def plot_mutation(fig, frame, methods, style, title="Mutation axis") -> None:
    """AUROC with its bootstrap 95% interval; filled marks have BH q < 0.05.

    One row per disease and alteration, most mutant queries first; one column per pair.
    Ineligible cells and unavailable depths are omitted. Missing methods have no point.
    Tier/stratum labels use the shared task definitions.
    """
    primary = frame.get("primary_depth", pd.Series(False, index=frame.index)).eq(True)
    if primary.any():
        frame = frame[primary]
    k = frame["k"].max()
    frame = frame[frame["k"].eq(k) & frame["eligible"].eq(True) & frame["score"].notna()]
    if frame.empty:
        ax = fig.subplots()
        ax.text(.5, .5, "No eligible analyses at this depth.", ha="center", va="center")
        ax.axis("off")
        return
    frame = frame.assign(cell=frame["disease"] + " " + frame["alteration"] + " "
                         + frame["class"].fillna("") + " "
                         + frame.get("stratum", "all") + " " + frame.get("tier", ""))
    cells = frame.groupby("cell")["n_query_mutant"].max().sort_values(ascending=False).index
    pairs = list(dict.fromkeys(frame["pair"]))
    axes = fig.subplots(len(cells), len(pairs), sharex=True, sharey=True, squeeze=False)
    scored = frame[frame["method"].isin(methods)]
    low_ci = scored["ci_low"].min()
    low = min(0.4, np.floor(low_ci * 10) / 10) if np.isfinite(low_ci) else 0.4
    for i, name in enumerate(cells):
        for j, pair in enumerate(pairs):
            ax = axes[i, j]
            cell = frame[(frame["pair"] == pair) & (frame["cell"] == name)].set_index("method")
            if cell.empty:
                ax.set_visible(False)
                continue
            r = cell.iloc[0]
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=1, zorder=0)
            ax.set_title(f"{r.disease} · {r.alteration} · {r.tier}\n"
                         f"queries {r.n_query_mutant:.0f}/{r.n_query - r.n_query_mutant:.0f}"
                         f" · models {r.n_database_mutant:.0f}/"
                         f"{r.n_database - r.n_database_mutant:.0f}",
                         loc="left", fontsize=7, color=MUTED)
            if i == 0:
                ax.annotate(pair.replace("_to_", " → "), (0.5, 1.28), xycoords="axes fraction",
                            ha="center", fontsize=11, fontweight="bold")
            for y, method in enumerate(methods):
                if method not in cell.index or pd.isna(cell.loc[method, "score"]):
                    continue
                m = cell.loc[method]
                marker, colour = style[method]
                errors = ([[max(0, m.score - m.ci_low)], [max(0, m.ci_high - m.score)]]
                          if np.isfinite(m.ci_low) and np.isfinite(m.ci_high) else None)
                ax.errorbar(m.score, y, xerr=errors,
                            fmt=marker, color=colour, markersize=5, elinewidth=1.2,
                            markerfacecolor=colour if m.q_value < 0.05 else "white")
    axes[0, 0].set_yticks(range(len(methods)), methods)
    axes[0, 0].set_ylim(len(methods) - 0.5, -0.5)
    axes[0, 0].set_xlim(low, 1)
    caption = (
        f"AUROC@{k} · Worst → best by global mean primary AUROC (equal cell weights). "
        "Whiskers: 95% CI · Filled: BH q < 0.05 · Dashed: random. "
        "Counts: positive/control. Ineligible analyses omitted."
    )
    fig.supxlabel(textwrap.fill(caption, int(fig.get_figwidth() * 16)), fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="summary.csv")
    parser.add_argument("--output", type=Path, required=True, help="PDF; a PNG is written too")
    parser.add_argument("--mutation-output", type=Path, required=True,
                        help="PDF of the mutation axis; a PNG is written too")
    parser.add_argument("--mutation-cancer-dir", type=Path,
                        help="Per-cancer summary.pdf/.png directory; default: OUTPUT/mutation")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    summary["row"] = summary["pair"].str.replace("_to_", " → ")
    methods = [m for m in summary["method"].unique() if m not in ("Random", "Oracle")]
    mutation = summary[(summary["axis"] == "mutation") & (summary["metric"] == "AUROC")]
    primary = mutation[mutation["primary_depth"].eq(True)]
    primary_k = int(primary.k.iloc[0]) if not primary.empty else int(mutation.k.max()) \
        if not mutation.empty else 10
    ranking = model_ranking(mutation.rename(columns={"score": "auroc"}), primary_k, methods)
    methods = [row["method"] for row in ranking]
    style = {m: (MARKERS[i % len(MARKERS)], COLOURS[i % len(COLOURS)])
             for i, m in enumerate(sorted(methods))}

    panels = [(axis, label, summary[(summary["axis"] == axis) & (summary["metric"] == metric)])
              for axis, metric, label in BARS]
    panels = [(axis, label, f) for axis, label, f in panels if not f.empty]
    height = sum(max(1.0, f["row"].nunique() * 0.55 + 0.6) for _, _, f in panels)
    fig = plt.figure(figsize=(8, max(2, height + 0.5)), layout="constrained")
    if panels:
        plot_bars(fig, panels, methods, style)
        handles, labels = fig.axes[0].get_legend_handles_labels()
        if labels:
            fig.legend(handles, labels, loc="outside upper right", ncol=len(labels), fontsize=8,
                       frameon=False)
    else:
        ax = fig.subplots()
        ax.text(.5, .5, "Ontology and donor results were not included in this summary.",
                ha="center", va="center")
        ax.axis("off")
    save(fig, args.output)

    if not mutation.empty:
        cancer_dir = args.mutation_cancer_dir or args.mutation_output.parent / "mutation"
        for disease, frame, path in [
            (None, mutation, args.mutation_output),
            *((d, f, cancer_dir / d / "summary.pdf") for d, f in mutation.groupby("disease")),
        ]:
            visible = frame[frame["eligible"].eq(True) & frame["score"].notna()]
            if visible["primary_depth"].eq(True).any():
                visible = visible[visible["primary_depth"].eq(True)]
            rows = visible.groupby(["disease", "alteration"]).ngroups
            pairs = max(1, visible["pair"].nunique())
            fig = plt.figure(figsize=(max(5, pairs * 4),
                                      max(2, rows * (len(methods) * 0.17 + 0.5) + .9)),
                             layout="constrained")
            title = f"Mutation axis · {disease}" if disease else "Mutation axis"
            plot_mutation(fig, frame, methods, style, title)
            save(fig, path)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Wrote {path} and {path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
