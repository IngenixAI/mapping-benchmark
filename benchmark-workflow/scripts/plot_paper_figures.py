"""The results figures, sized for the NeurIPS text width (5.5 in), in the Ingenix style.

Part of the main workflow, for whichever methods ``summary.csv`` holds; the methods of the
paper keep their fixed colours and markers, any other method is grey with its own marker.
``fig3_aggregated`` puts the three axes in one figure, in the paper's order: ontology and
donor identity as random-to-oracle intervals, and the mutation axis between them as one mean
AUROC per pair (cancers weighted by n^0.5). Given ``comparisons.csv``, a method that the paired test could not tell
from the baseline is drawn hollow there, and a dashed line marks the baseline's score.
``fig3_expanded_ontology_values`` breaks the ontology score down by query diagnosis, as each
method's own share of the random-to-oracle interval, and ``fig3_expanded_mutation`` shows
each primary mutation analysis against the baseline; both mark the verdicts BH-corrected
within each pair. Within a cancer, analyses and diagnoses run from the baseline's highest
score to its lowest. Each is written as PDF and PNG. Every plotted number comes from the
CSV; a 95% interval is drawn wherever the row carries ``ci_low``/``ci_high``.
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

from mapping_benchmark.retrieval import MIN_TESTED_CLUSTERS

matplotlib.use("Agg")

# Ingenix chrome: brand text, grey and purple tint (ingenix.ai, the white paper and poster).
PURPLE, PALE = "#4F00B9", "#F3EEFB"
INK, INK2, MUTED = "#202020", "#5C5C5C", "#898989"
GRID, AXIS, BAND = "#E6E1EE", "#C9C3D6", PALE

# Colour and marker follow the method, never its rank. Brand purple and amber lead, then
# brand lilac and red, extended with blue and teal; dataviz-validated in the plotted order
# (worst adjacent CVD dE 20.3, normal vision 20.8). Amber is under 3:1 on white, so
# identity is also carried by marker and fixed position.
STYLE = {
    "scVI": ("scVI", PURPLE, "D"),
    "MOBER": ("MOBER", "#E89B00", "^"),
    "PCA": ("PCA", "#A27BE8", "o"),
    "Celligner": ("Celligner", "#C0392B", "v"),
    "Bulkformer": ("BulkFormer", "#1F6FD1", "h"),
    "Geneformer": ("Geneformer", "#14A38B", "p"),
    "Corr-hvg": ("Correlation", "#52514E", "s"),  # no-training baseline
}
BASELINE = "Corr-hvg"
BASELINE_DASH = (0, (2, 1.5))
EXTRA_MARKERS = "XP*<>d"  # methods without a fixed style
PAIRS = {
    "tcga_to_depmap": "TCGA → DepMap",
    "metaprism_to_depmap": "MetaPRISM → DepMap",
    "tcga_to_metaprism": "TCGA → MetaPRISM",
    "tcga_to_pdxe": "TCGA → PDXE",
}
DISEASES = ["LUAD", "COAD", "SKCM", "BRCA"]
ALTERATIONS = {
    "KRAS_MSS": "KRAS (MSS)",
    "PIK3CA_core": "PIK3CA",
    "PIK3CA_extended": "PIK3CA (extended)",
    "TP53_lof": "TP53 LoF",
    "BRAF_V600E": "BRAF V600E",
    "MSI_MANTIS_sensitivity": "MSI (MANTIS)",
}
ALPHA = 0.05

plt.rcParams.update(
    {
        # Commissioner is the brand typeface (OFL; static instances in ~/.local/share/fonts).
        # A list, not "sans-serif", so glyphs it lacks (→, ∩) fall back per character.
        "font.family": ["Commissioner", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "text.color": INK,
        "xtick.color": INK2,
        "axes.edgecolor": AXIS,
        "xtick.major.size": 2,
        "xtick.major.width": 0.5,
        "xtick.major.pad": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

W = 5.5  # figure width = NeurIPS text width (in)
CANVAS_H = 12.0  # drawing canvas; only the used height is saved
LEFT = 0.04  # left margin (in)
RIGHT = 0.12  # right margin (in)
LABEL_W = 0.92  # method-label column of the aggregated panels (in)
GAP = 0.13  # gap between facets of one aggregated row (in)
SPLIT = 0.16  # extra gap before the donor facet (in)
PAD = 0.5  # y padding inside an aggregated facet, in rows
BASE_GAP = 0.3  # extra gap before the baseline row (and between disease groups), in rows
COL_GAP = 0.22  # gap between the two mutation columns (in)
MUT_LABEL_W = 0.84  # analysis-label column of each mutation column (in)
DISEASE_W = 0.31  # disease code width inside that label column (in)
BLOCK_FILL = 0.86  # share of an analysis block covered by the dodged methods + band
BLOCK_GAP = 0.4  # extra gap between analysis blocks, in blocks, so whiskers group by analysis


def alt_label(a):
    # An analysis without a display name keeps its id, spaced.
    s = ALTERATIONS.get(a, a.replace("_nonsilent", " non-silent")).replace("_", " ")
    return textwrap.fill(s, 9, break_long_words=False, break_on_hyphens=False)


def load(summary: Path):
    d = pd.read_csv(summary)
    ont = d[(d.axis == "ontology") & (d.metric == "simGIC") & (d.k == 10)]
    don = d[(d.axis == "donor") & (d.metric == "precision") & (d.k == 1)]
    mut = d[(d.axis == "mutation") & (d.metric == "AUROC") & (d.k == 10) & d.eligible]
    assert set(ont.method) == set(don.method) == set(mut.method), "methods differ between axes"
    extra = sorted(set(ont.method) - set(STYLE) - {"Random", "Oracle"})
    for i, m in enumerate(extra):
        # ponytail: one grey for every new method, told apart by marker and label; give a
        # method that stays its own validated colour in STYLE
        STYLE[m] = (m, MUTED, EXTRA_MARKERS[i % len(EXTRA_MARKERS)])
    return d, ont, don, mut


def method_order(ont, don, mut):
    """Learned methods by mean fraction of the random-to-oracle interval (each axis weighted
    equally; mutation = primary tier), then the baseline."""
    key = ["axis", "pair", "disease", "alteration"]
    df = pd.concat([ont, don, mut[mut.tier == "primary"]]).fillna({"disease": "", "alteration": ""})
    b = df[df.method.isin(["Random", "Oracle"])].pivot_table(
        index=key, columns="method", values="score"
    )
    x = df[df.method.isin(STYLE)].join(b, on=key)
    x["frac"] = (x.score - x.Random) / (x.Oracle - x.Random)
    per_axis = x.groupby(["axis", "method"]).frac.mean().unstack(0)
    print(
        "Mean fraction of random-to-oracle interval:\n",
        per_axis.assign(mean=per_axis.mean(1)).round(3),
    )
    base = [BASELINE] if BASELINE in per_axis.index else []
    return list(per_axis.mean(1).drop(base).sort_values(ascending=False).index) + base


def row_positions(order):
    """y of the method rows: unit steps, with an extra gap before the baseline (last)."""
    n = len(order)
    return np.arange(n) + np.where(np.arange(n) == n - 1, BASE_GAP * (order[-1] == BASELINE), 0)


def add_axes(fig, x, top, w, h):
    """Axes placed in inches, `top` measured from the top edge of the figure."""
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([x / fw, (fh - top - h) / fh, w / fw, h / fh])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_yticks([])
    return ax


def text(fig, x, top, s, va="top", **kw):
    """Figure text at x inches from the left and `top` inches from the top edge."""
    fw, fh = fig.get_size_inches()
    return fig.text(x / fw, 1 - top / fh, s, va=va, **kw)


def panel(fig, x, top, letter, title):
    """Panel heading: the letter in brand purple, the title in ink."""
    text(fig, x, top, letter, weight="bold", size=8.5, color=PURPLE)
    text(fig, x + 0.16, top, title, weight="bold", size=8)


def finish_x(ax, xlim, ticks, show, labelled=None):
    ax.set_xlim(xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" if labelled is None or t in labelled else "" for t in ticks])
    if show:
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_linewidth(0.5)
    else:
        ax.tick_params(bottom=False, labelbottom=False)


def bound_lines(ax, rnd, orc):
    ax.axvline(rnd, color=MUTED, lw=0.9, zorder=1)
    ax.axvline(orc, color=INK, lw=1.2, zorder=1)


def point(ax, x, y, m, filled, ms):
    _, col, mk = STYLE[m]
    ax.plot(
        x,
        y,
        ls="none",
        marker=mk,
        ms=ms,
        mfc=col if filled else "white",
        mec=col,
        mew=0.6,
        zorder=3,
    )


# --------------------------------------------------------------------------- A-C: aggregated
def method_labels(fig, ax_top, h, order):
    """The method names, one per row, left of a row of facets."""
    ys = row_positions(order)
    lab = add_axes(fig, LEFT, ax_top, LABEL_W, h)
    lab.set_ylim(ys[-1] + PAD, -PAD)
    lab.axis("off")
    for m, y in zip(order, ys):
        lab.text(
            0.97,
            y,
            STYLE[m][0],
            ha="right",
            va="center",
            style="italic" if m == BASELINE else "normal",
        )


def draw_facet(fig, x, ax_top, w, h, s, order, xlim, ticks, labelled=None):
    """One pair: methods as rows on the random-to-oracle band, each with its 95% CI.
    ``s`` is that pair's rows, indexed by method; a method whose ``verdict`` is undecided is
    drawn hollow."""
    ys = row_positions(order)
    ax = add_axes(fig, x, ax_top, w, h)
    ax.set_ylim(ys[-1] + PAD, -PAD)
    rnd, orc = s.score.Random, s.score.Oracle
    ax.axvspan(rnd, orc, color=BAND, lw=0, zorder=0)
    for y in ys:
        ax.axhline(y, color=GRID, lw=0.4, zorder=0.5)
    bound_lines(ax, rnd, orc)
    if BASELINE in s.index:
        ax.axvline(s.score[BASELINE], color=STYLE[BASELINE][1], lw=0.6, ls=BASELINE_DASH, zorder=1)
    undecided = s.get("verdict", pd.Series(dtype=object)).eq("undecided")
    for m, y in zip(order, ys):
        col = STYLE[m][1]
        lo, hi = s.ci_low[m], s.ci_high[m]
        if pd.notna(lo) and pd.notna(hi):  # the 95% CI, drawn as in panel C
            ax.hlines(y, lo, hi, color=col, lw=0.9, zorder=2)
        hollow = undecided.get(m, False)
        ax.plot(s.score[m], y, marker=STYLE[m][2], ms=4.2, mfc="white" if hollow else col,
                mec=col if hollow else "white", mew=0.7 if hollow else 0.5, zorder=3)
    finish_x(ax, xlim, ticks, True, labelled)


def draw_row(fig, top, order, row_h, facets, groups):
    """One row of facets that share the method rows; returns its bottom (in).

    ``facets`` holds (rows by method, xlim, ticks, title, subtitle or None) per facet,
    ``groups`` (letter, title, first and last facet, x label) per panel. A panel after the
    first starts a little further right.
    """
    ys = row_positions(order)
    h = (ys[-1] + 2 * PAD) * row_h
    lines = max(f[3].count("\n") + 1 + (f[4].count("\n") + 1 if f[4] else 0) for f in facets)
    ax_top = top + 0.15 + 0.12 * lines + 0.03
    starts = {g[2] for g in groups[1:]}
    n = len(facets)
    fw = (W - LEFT - LABEL_W - RIGHT - SPLIT * len(starts) - GAP * (n - 1)) / n
    xs = list(LEFT + LABEL_W + np.arange(n) * (fw + GAP)
              + SPLIT * np.cumsum([i in starts for i in range(n)]))

    method_labels(fig, ax_top, h, order)
    for (frame, xlim, ticks, title, subtitle), x in zip(facets, xs):
        draw_facet(fig, x, ax_top, fw, h, frame, order, xlim, ticks)
        text(fig, x + fw / 2, top + 0.15, title, ha="center", linespacing=1.1)
        if subtitle:
            below = top + 0.15 + 0.12 * (title.count("\n") + 1) + 0.01
            text(fig, x + fw / 2, below, subtitle, ha="center", color=INK2, linespacing=1.1)
    for letter, name, first, last, label in groups:
        text(fig, (xs[first] + xs[last] + fw) / 2, ax_top + h + 0.17, label, ha="center",
             color=INK2, linespacing=1.1)
        panel(fig, LEFT if first == 0 else xs[first] - 0.14, top, letter, name)
    return ax_top + h + 0.2 + 0.12 * max(g[4].count("\n") + 1 for g in groups)


# --------------------------------------------------------------------------- C: mutation
def mutation_rows(sub):
    """One pair's analyses by disease, then from the baseline's highest AUROC to its lowest (the
    methods' mean without a baseline); block centres, spaced apart, with disease gaps."""
    by = sub[sub.method == BASELINE] if BASELINE in set(sub.method) else sub[
        sub.method.isin(STYLE)]
    r = by.groupby(["disease", "alteration"]).score.mean().reset_index()
    rank = {d: i for i, d in enumerate(DISEASES)}
    r = r.assign(o=r.disease.map(lambda d: rank.get(d, len(DISEASES))))
    r = r.sort_values(["o", "disease", "score"], ascending=[True, True, False])
    r = r[["disease", "alteration"]]
    new = r.disease.ne(r.disease.shift()).to_numpy()
    new[0] = False
    return r.assign(y=np.arange(len(r)) * (1 + BLOCK_GAP) + BASE_GAP * np.cumsum(new))


def draw_pair(fig, x0, col_w, top, sub, order, block_h, xlim, ticks, show_x):
    """One pair: an analysis block per row, the seven methods dodged on a shared AUROC axis.

    A point is filled when it differs from the baseline (q < 0.05), whose score is a dashed
    line through the block.
    """
    rows = mutation_rows(sub)
    ylim = (rows.y.max() + 0.5, -0.5)
    h = (ylim[0] - ylim[1]) * block_h
    lab = add_axes(fig, x0, top, MUT_LABEL_W, h)
    lab.set_ylim(ylim)
    lab.axis("off")
    first = rows.disease.ne(rows.disease.shift())
    for r, f in zip(rows.itertuples(), first):
        if f:
            lab.text(0, r.y, r.disease, va="center", color=INK2)
        lab.text(
            DISEASE_W / MUT_LABEL_W, r.y, alt_label(r.alteration), va="center", linespacing=1.0
        )

    ax = add_axes(fig, x0 + MUT_LABEL_W, top, col_w - MUT_LABEL_W, h)
    ax.set_ylim(ylim)
    rnd, orc = (sub[sub.method == b].score.unique() for b in ("Random", "Oracle"))
    assert len(rnd) == len(orc) == 1, "mutation bounds are not constant within a tier"
    rnd, orc = rnd[0], orc[0]
    for t in ticks:
        ax.axvline(t, color=GRID, lw=0.35, zorder=0.5)
    for y in (rows.y.to_numpy()[1:] + rows.y.to_numpy()[:-1]) / 2:  # rule between analysis blocks
        ax.axhline(y, color=GRID, lw=0.5, zorder=0.5)
    for y in rows.y:  # one shaded random-to-oracle band per analysis; white gaps separate blocks
        ax.add_patch(
            Rectangle(
                (rnd, y - BLOCK_FILL / 2 - 0.04),
                orc - rnd,
                BLOCK_FILL + 0.08,
                color=BAND,
                lw=0,
                zorder=0,
            )
        )
    bound_lines(ax, rnd, orc)
    base = rows.merge(sub[sub.method == BASELINE], on=["disease", "alteration"])
    if len(base):
        half = BLOCK_FILL / 2 + 0.04
        ax.vlines(base.score, base.y - half, base.y + half, color=STYLE[BASELINE][1], lw=0.6,
                  linestyles=[BASELINE_DASH], zorder=1)
    pos = row_positions(order)
    off = (pos - pos[-1] / 2) * (BLOCK_FILL / max(pos[-1], 1))
    for m, o in zip(order, off):
        col = STYLE[m][1]
        s = rows.merge(sub[sub.method == m], on=["disease", "alteration"], validate="1:1")
        y = s.y + o
        ax.hlines(
            y,
            s.ci_low.clip(lower=xlim[0]),
            s.ci_high.clip(upper=xlim[1]),
            color=col,
            lw=0.75,
            zorder=2,
        )
        for yy in y[s.ci_low < xlim[0]]:  # truncated interval: arrowhead at the axis edge
            ax.annotate(
                "",
                (xlim[0], yy),
                (xlim[0] + 0.02, yy),
                annotation_clip=False,
                arrowprops=dict(
                    arrowstyle="-|>", color=col, lw=0.75, mutation_scale=4, shrinkA=0, shrinkB=0
                ),
            )
        assert (s.ci_high <= xlim[1]).all()
        sig = (s.delta_q < ALPHA).to_numpy() | (m == BASELINE)
        point(ax, s.score[sig], y[sig], m, True, 2.9)
        point(ax, s.score[~sig], y[~sig], m, False, 2.9)
    finish_x(ax, xlim, ticks, show_x)
    if show_x:
        text(
            fig, x0 + (MUT_LABEL_W + col_w) / 2, top + h + 0.16, "AUROC@10", ha="center", color=INK2
        )
    return h


def draw_column(fig, x0, col_w, top, tier_df, order, block_h, xlim, ticks):
    """Stack the pairs of one tier; x-axis under the last. Returns the new top (in)."""
    pairs = [p for p in PAIRS if p in set(tier_df.pair)]
    for i, pair in enumerate(pairs):
        top += 0.06 if i else 0
        text(fig, x0, top, PAIRS[pair], weight="bold")
        top += 0.14
        top += draw_pair(
            fig,
            x0,
            col_w,
            top,
            tier_df[tier_df.pair == pair],
            order,
            block_h,
            xlim,
            ticks,
            i == len(pairs) - 1,
        )
    return top + 0.3


def mutation_xlim(df, column="score"):
    """From 0.1 below the lowest point estimate (or ``column``) to the oracle; longer CIs get
    arrowheads."""
    lo = np.floor(df[df.method.isin(STYLE)][column].min() * 10) / 10
    return (lo - 0.01, 1.01), list(np.arange(lo, 1.0 + 1e-9, 0.1).round(1))


def n_analyses(df):
    return df[["pair", "disease", "alteration"]].drop_duplicates().shape[0]


def baseline_counts(tier_df, order):
    """Analyses where each method is better / worse than the baseline (q < 0.05)."""
    t = tier_df[tier_df.delta_q < ALPHA]
    return {m: f"{(t[t.method == m].delta > 0).sum()}↑ {(t[t.method == m].delta < 0).sum()}↓"
            for m in order if m != BASELINE}


def legend(fig, top, order, counts=None, sig=None, nonsig=None, baseline=False):
    """Method key (colour + marker, fixed order, row-major) and an encoding key below it.

    ``sig`` and ``nonsig`` label filled and hollow points; without ``sig`` neither is keyed.
    """
    fh = fig.get_size_inches()[1]
    ncol, n = 4, len(order)
    idx = [
        r * ncol + c for c in range(ncol) for r in range(2) if r * ncol + c < n
    ]  # row-major fill
    handles = [
        Line2D(
            [],
            [],
            color=STYLE[m][1],
            lw=0.75,
            marker=STYLE[m][2],
            ms=3.6,
            mfc=STYLE[m][1],
            mec=STYLE[m][1],
        )
        for m in order
    ]
    labels = [STYLE[m][0] + (f" ({counts[m]})" if m in (counts or {}) else "") for m in order]
    leg = fig.legend(
        [handles[i] for i in idx],
        [labels[i] for i in idx],
        loc="upper center",
        bbox_to_anchor=(0.5, 1 - top / fh),
        ncol=ncol,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.6,
        handletextpad=0.5,
        borderaxespad=0,
        labelspacing=0.35,
    )
    if order[-1] == BASELINE:
        leg.get_texts()[idx.index(n - 1)].set_style("italic")
    lines = {
        "random expectation": Line2D([], [], color=MUTED, lw=0.9),
        "oracle": Line2D([], [], color=INK, lw=1.2),
    }
    if baseline:
        lines[f"{STYLE[BASELINE][0].lower()} baseline"] = Line2D(
            [], [], color=STYLE[BASELINE][1], lw=0.6, ls=BASELINE_DASH
        )
    points = {}
    if sig:
        points[sig] = Line2D([], [], ls="none", marker="o", ms=3.6, mfc=INK2, mec=INK2)
        points[nonsig] = Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=INK2,
                                mew=0.6)
    # Beside the baseline's key, the point keys get a row of their own.
    rows = [r for r in ([lines, points] if baseline else [lines | points]) if r]
    for i, row in enumerate(rows):
        fig.legend(
            list(row.values()),
            list(row),
            loc="upper center",
            bbox_to_anchor=(0.5, 1 - (top + 0.3 + 0.15 * i) / fh),
            ncol=len(row),
            frameon=False,
            handlelength=1.6,
            columnspacing=1.4,
            handletextpad=0.45,
            borderaxespad=0,
        )
    return top + 0.47 + 0.15 * (len(rows) - 1)


def draw_mutation_columns(fig, primary, order, top):
    """The primary analyses against the baseline, TCGA → DepMap left and the MetaPRISM pairs
    right."""
    col_w = (W - LEFT - RIGHT - COL_GAP) / 2
    xlim, ticks = mutation_xlim(primary)
    left = primary.pair == list(PAIRS)[0]
    return max(
        draw_column(fig, x, col_w, top, primary[rows], order, 0.29, xlim, ticks)
        for x, rows in ((LEFT, left), (LEFT + col_w + COL_GAP, ~left))
    )


# --------------------------------------------------------------------------- expanded ontology
MIN_QUERIES = 10  # the MetaPRISM diagnoses shown: most have a handful of patients
CELL_H = 0.105  # heatmap row height (in)
HEAT_LABEL_W = 1.72  # diagnosis-name column (in)
N_W = 0.27  # query-count column (in)
GROUP_GAP = 0.1  # between the pairs' column groups (in)
BAR_GAP = 0.06  # before the baseline's column, after the method cells (in)
# Each method's own share: bright enough for black arrows on every cell (contrast >= 5:1),
# hue running yellow, green, aqua, blue to lilac as lightness falls from L* 95 to 60.
HEAT = LinearSegmentedColormap.from_list("share", [
    "#FFF3B0", "#D9F0A3", "#A6DBA0", "#7FCDBB", "#74A9CF", "#9E9AC8", "#A27BE8"])
HEAT.set_under("#BDBDBD")  # below random
HEAT.set_bad("none")


def up(c):  # an arrow, as triangles are MOBER's and Celligner's markers
    return dict(marker="$↑$", ms=3.6, mfc=c, mec=c, mew=0)


def down(c):
    return dict(marker="$↓$", ms=3.6, mfc=c, mec=c, mew=0)


def ontology_shares(d):
    """simGIC@10 per query diagnosis as a share of that diagnosis's random-to-oracle interval,
    and each method's q-value against the baseline (NaN where not tested)."""
    per = d[(d.axis == "ontology") & (d.metric == "simGIC per cancer") & (d.k == 10)]
    b = per.pivot_table(index=["pair", "disease"], columns="method", values="score")
    q = per.set_index(["pair", "disease", "method"]).delta_q.unstack()
    return b.sub(b.Random, axis=0).div(b.Oracle - b.Random, axis=0), q.reindex(b.index)


def draw_heatmap(fig, top, shares, q, pairs, rows, names, sizes, order):
    """Query diagnoses as rows; per pair, each method's own share (an arrow where it differs
    from the baseline, hatching where not tested), then the baseline's share. Returns the
    bottom (in).
    """
    methods = [m for m in order if m != BASELINE]
    n = len(methods) + 1  # cells per group
    cell_w = (W - LEFT - HEAT_LABEL_W - N_W - RIGHT - 2 * GROUP_GAP - 3 * BAR_GAP) / (3 * n)
    bw = cell_w  # the baseline's column
    gw, head = len(methods) * cell_w + BAR_GAP + bw, 0.6
    h = len(rows) * CELL_H
    x0 = LEFT + HEAT_LABEL_W
    for i, pair in enumerate(pairs):
        gx = x0 + i * (gw + GROUP_GAP)
        s = shares.loc[pair].reindex(rows)
        base = s.get(BASELINE, pd.Series(np.nan, index=rows))
        ax = add_axes(fig, gx, top + head, len(methods) * cell_w, h)
        ax.set_xticks([])
        values, cmap, norm = s[methods], HEAT, Normalize(0, 1)
        ax.pcolormesh(np.arange(len(methods) + 1), np.arange(len(rows) + 1), values.to_numpy(),
                      cmap=cmap, norm=norm, edgecolors="white", linewidth=0.6)
        pq = q.loc[pair].reindex(rows)[methods]
        # An arrow marks a decided cell, white where white contrasts more than ink (WCAG
        # luminance).
        decided = (pq < ALPHA).to_numpy()
        sign = np.sign(s[methods].sub(base, axis=0).to_numpy())
        marks = [(decided & (sign > 0), up), (decided & (sign < 0), down)]
        rgb = cmap(norm(values.to_numpy()))[..., :3]
        rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        dark = rgb @ [0.2126, 0.7152, 0.0722] < 0.21
        for on_dark, colour in ((False, INK), (True, "white")):
            for where, style in marks:
                yy, xx = np.nonzero(where & (dark == on_dark))
                ax.plot(xx + 0.5, yy + 0.5, ls="none", **style(colour))
        for j in np.flatnonzero(pq.isna().all(axis=1).to_numpy()):
            ax.add_patch(Rectangle((0, j), len(methods), 1, fill=False, hatch="//////",
                                   ec="white", lw=0))
        ax.set_xlim(0, len(methods))
        ax.set_ylim(len(rows), 0)

        bx = add_axes(fig, gx + len(methods) * cell_w + BAR_GAP, top + head, bw, h)
        bx.pcolormesh([0, 1], np.arange(len(rows) + 1), base.to_numpy()[:, None], cmap=HEAT,
                      norm=Normalize(0, 1), edgecolors="white", linewidth=0.6)
        bx.set_ylim(len(rows), 0)
        bx.set_xticks([])

        text(fig, gx + gw / 2, top, PAIRS[pair], ha="center", weight="bold")
        for m, x in [*zip(methods, gx + (np.arange(len(methods)) + 0.5) * cell_w),
                     (BASELINE, gx + gw - bw / 2)]:
            text(fig, x, top + head - 0.04, STYLE[m][0] if m in STYLE else m, rotation=90,
                 ha="center", va="bottom", size=6, style="italic" if m == BASELINE else "normal")
    nx = x0 + len(pairs) * gw + (len(pairs) - 1) * GROUP_GAP + N_W
    text(fig, nx, top + head - 0.04, "n", ha="right", va="bottom", size=6, color=INK2)
    for j, r in enumerate(rows):
        y = top + head + (j + 0.5) * CELL_H
        tested = q.loc[pairs].xs(r, level="disease").notna().any().any()
        text(fig, x0 - 0.05, y, names[r] + ("" if tested else "†"), ha="right", va="center",
             size=6)
        text(fig, nx, y, f"{sizes[r]:,}", ha="right", va="center", size=6, color=INK2)
    return top + head + h


def draw_expanded_ontology(fig, d, samples, order, top):
    """TCGA queries against the three databases, then the larger MetaPRISM diagnoses; each
    method as its own share."""
    shares, q = ontology_shares(d)
    labelled = samples[samples.mondo_id.ne("")]
    names = labelled.groupby("mondo_id").mondo_name.first()

    def block(top, dataset, pairs, keep):
        sizes = labelled[labelled.dataset == dataset].mondo_id.value_counts()
        rows = [r for r in sizes.index if keep(sizes[r])]
        # By the baseline's own share over the pairs, so its bars fall down the page.
        by = [BASELINE] if BASELINE in shares else order
        mean = shares.loc[pairs, by].groupby(level="disease").mean().mean(axis=1)
        rows = sorted(rows, key=lambda r: -mean[r])
        return draw_heatmap(fig, top, shares, q, pairs, rows, names, sizes, order), sizes, rows

    base = STYLE[BASELINE][0].lower() if BASELINE in STYLE else BASELINE
    text(fig, LEFT, top, textwrap.fill(
        "simGIC@10 per query diagnosis: each method, as a share of that diagnosis's "
        "random-to-oracle interval; rows by the baseline's share", 112),
        color=INK2, linespacing=1.2)
    tcga = [p for p in PAIRS if p.startswith("tcga_to")]
    bottom, _, _ = block(top + 0.35, "tcga", tcga, lambda n: True)
    mp_top = bottom + 0.2
    bottom, sizes, rows = block(
        mp_top, "metaprism", ["metaprism_to_depmap"], lambda n: n >= MIN_QUERIES
    )

    # The key, in the space right of the single MetaPRISM group.
    kx = LEFT + HEAT_LABEL_W + (W - LEFT - HEAT_LABEL_W - N_W - RIGHT) / 3 + N_W + 0.35
    kw = W - RIGHT - kx - 0.1  # room for the centred end labels
    note = (
        f"MetaPRISM: the {len(rows)} diagnoses with ≥ {MIN_QUERIES} patients "
        f"({sizes[rows].sum()} of {sizes.sum()} queries). n: query samples per diagnosis. "
        "Grey: below random. Arrows: BH q < 0.05 within the pair; unmarked: undecided."
    )
    text(fig, kx, mp_top + 0.62, textwrap.fill(note, 38), color=INK2, linespacing=1.2)
    marks = {f"better than the {base} baseline": Line2D([], [], ls="none", **up(INK)),
             f"worse than the {base} baseline": Line2D([], [], ls="none", **down(INK))}
    marks |= {f"† not tested: < {MIN_TESTED_CLUSTERS} resampling clusters":
             Rectangle((0, 0), 1, 1, fc=MUTED, ec="white", hatch="//////", lw=0)}
    fh = fig.get_size_inches()[1]
    fig.legend(list(marks.values()), list(marks), loc="upper left", frameon=False,
               bbox_to_anchor=(kx / W, 1 - (mp_top + 1.38) / fh), borderaxespad=0,
               handlelength=1.2, handletextpad=0.5, fontsize=6, labelcolor=INK2)
    cax = add_axes(fig, kx, mp_top + 2.05, kw, 0.09)
    bar = fig.colorbar(ScalarMappable(Normalize(0, 1), HEAT), cax=cax, orientation="horizontal",
                       extend="min")
    bar.set_ticks([0, 0.5, 1], labels=["0\nrandom", "0.5", "1\noracle"])
    bar.outline.set_visible(False)
    cax.tick_params(length=2, width=0.5, pad=1.5, labelsize=6, colors=INK2)
    text(fig, kx, mp_top + 1.87, "share of the random-to-oracle interval", size=6, color=INK2)
    return bottom + 0.05


def save(fig, output_dir, name, used_h):
    """Save only the used top `used_h` inches of the canvas."""
    box = Bbox.from_bounds(0, CANVAS_H - used_h, W, used_h)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(output_dir / f"{name}.{ext}", facecolor="white", bbox_inches=box, **kw)
    print(f"{name}: {W} x {used_h:.2f} in")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path, help="results/summary.csv")
    parser.add_argument(
        "--samples", required=True, type=Path, help="data/raw/expression_samples.csv"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--comparisons", type=Path, help="results/comparisons.csv: verdicts against the baseline"
    )
    args = parser.parse_args()

    d, ont, don, mut = load(args.summary)
    order = method_order(ont, don, mut)
    print("Method order:", [STYLE[m][0] for m in order])
    primary = mut[mut.tier == "primary"]
    print("Eligible primary analyses:", n_analyses(primary))

    # ---- Aggregated: A ontology, B mutation as one AUROC per pair, C donor identity
    agg = d[(d.axis == "mutation") & (d.metric == "mean AUROC") & (d.k == 10)]
    keys = {"sig": None}
    if args.comparisons:
        verdicts = pd.read_csv(args.comparisons).set_index(["axis", "pair", "k", "method"])
        ont, don, agg = (f.join(verdicts.verdict, on=["axis", "pair", "k", "method"])
                         for f in (ont, don, agg))
        keys = {"sig": f"differs from the {STYLE[BASELINE][0].lower()} baseline (Holm p < 0.05)",
                "nonsig": "undecided"}
    fig = plt.figure(figsize=(W, CANVAS_H))
    top = legend(fig, 0.03, order, baseline=BASELINE in order, **keys)
    ont_max = np.ceil(ont.score.max() * 10) / 10
    ontology = [
        (ont[ont.pair == p].set_index("method"), (0, ont_max),
         list(np.arange(0, ont_max + 1e-9, 0.3).round(1)), PAIRS[p].replace(" → ", " →\n"), None)
        for p in PAIRS if p in set(ont.pair)
    ]
    top = draw_row(fig, top + 0.03, order, 0.094, ontology,
                   [("A", "Disease ontology", 0, len(ontology) - 1,
                     "ontological agreement (simGIC@10)")]) + 0.02
    # Mutation from just below random (0.5), or below the lowest interval if one reaches past it.
    xlim = (min(0.5, agg[agg.method.isin(STYLE)].ci_low.min()) - 0.02, 1.01)
    second = []
    for p in [p for p in PAIRS if p in set(agg.pair)]:
        s = agg[agg.pair == p].set_index("method")
        second.append((s, xlim, [0.5, 0.75, 1], PAIRS[p].replace(" → ", " →\n"),
                       f"{s.disease.iloc[0].replace('+', ', ')}\n{s.alteration.iloc[0]}"))
    second.append((don.set_index("method"), (-0.03, 1.05), [0, 0.5, 1],
                   "APHRODYTHE\nhuman → PDX", None))
    bottom = draw_row(fig, top, order, 0.094, second,
                      [("B", "Mutation", 0, len(second) - 2,
                        "mutation enrichment (mean AUROC@10)"),
                       ("C", "Donor identity", len(second) - 1, len(second) - 1,
                        "donor match rate\n(precision@1)")])
    save(fig, args.output_dir, "fig3_aggregated", bottom)

    # ---- Expanded: ontology per query diagnosis; each primary mutation analysis
    samples = pd.read_csv(args.samples, dtype=str, keep_default_na=False)
    fig = plt.figure(figsize=(W, CANVAS_H))
    save(fig, args.output_dir, "fig3_expanded_ontology_values",
         draw_expanded_ontology(fig, d, samples, order, 0.04))
    fig = plt.figure(figsize=(W, CANVAS_H))
    base = STYLE[BASELINE][0].lower()
    top = legend(fig, 0.03, order, baseline_counts(primary, order), baseline=BASELINE in order,
                 sig=f"differs from the {base} baseline (BH q < 0.05)", nonsig="undecided")
    note = textwrap.fill(
        f"{n_analyses(primary)} eligible primary analyses; within a cancer by the {base} "
        f"baseline's AUROC, highest first. Counts in the key: analyses better (↑) "
        f"or worse (↓) than the {base} baseline.", 108)
    text(fig, LEFT, top + 0.04, note, color=INK2, linespacing=1.2)
    save(
        fig,
        args.output_dir,
        "fig3_expanded_mutation",
        draw_mutation_columns(fig, primary, order, top + 0.13 + 0.12 * (note.count("\n") + 1)),
    )


if __name__ == "__main__":
    main()
