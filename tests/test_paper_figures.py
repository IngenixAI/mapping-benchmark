"""The results figures render for any method set, not only the paper's seven."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "benchmark-workflow/scripts/plot_paper_figures.py"
spec = importlib.util.spec_from_file_location("plot_paper_figures", SCRIPT)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)

ONTOLOGY_PAIRS = ["tcga_to_depmap", "metaprism_to_depmap", "tcga_to_metaprism", "tcga_to_pdxe"]


def summary():
    """PCA and a method without a fixed style, no baseline, and a cancer outside the paper."""
    rows = []
    for method in ["PCA", "NewMethod", "Random", "Oracle"]:
        bound = method in ("Random", "Oracle")
        simgic = {"Random": 0.1, "Oracle": 0.6}.get(method, 0.3)
        auroc = {"Random": 0.5, "Oracle": 1.0}.get(method, 0.7)

        def row(score, **kw):
            ci = {} if bound else {"ci_low": score - 0.05, "ci_high": score + 0.05}
            rows.append({"method": method, "score": score, "eligible": True, **ci, **kw})

        for pair in ONTOLOGY_PAIRS:
            row(simgic, axis="ontology", pair=pair, k=10, metric="simGIC")
            for label in ("MONDO:1", "MONDO:2"):
                row(
                    simgic,
                    axis="ontology",
                    pair=pair,
                    k=10,
                    metric="simGIC per cancer",
                    disease=label,
                )
        row(simgic, axis="donor", pair="aphrodite", k=1, metric="precision")
        for pair in ONTOLOGY_PAIRS[:2]:
            for disease, alteration, tier in (
                ("LUAD", "KRAS", "primary"),
                ("NEW", "GENE_x", "primary"),
                ("NEW", "GENE_y", "exploratory"),
            ):
                row(
                    auroc,
                    axis="mutation",
                    pair=pair,
                    disease=disease,
                    alteration=alteration,
                    tier=tier,
                    k=10,
                    metric="AUROC",
                    delta=None if bound else -0.1,
                    delta_q=None if bound else 0.01,
                )
            row(
                auroc,
                axis="mutation",
                pair=pair,
                disease="LUAD+NEW",
                tier="primary",
                k=10,
                alteration="2 primary analyses",
                metric="mean AUROC",
            )
    return pd.DataFrame(rows)


def test_every_figure_renders_without_the_paper_methods(tmp_path, monkeypatch):
    summary().to_csv(tmp_path / "summary.csv", index=False)
    pd.DataFrame(
        [("tcga", "MONDO:1", "one")] * 20
        + [("tcga", "MONDO:2", "two")] * 20
        + [("metaprism", "MONDO:1", "one")] * 12
        + [("metaprism", "MONDO:2", "two")] * 3,
        columns=["dataset", "mondo_id", "mondo_name"],
    ).to_csv(tmp_path / "samples.csv")
    # Verdicts against a baseline that is not among the methods: undecided ones are hollow.
    pd.DataFrame(
        [("ontology", p, 10, "PCA", "undecided") for p in ONTOLOGY_PAIRS]
        + [("donor", "aphrodite", 1, "NewMethod", "better"),
           ("mutation", "tcga_to_depmap", 10, "PCA", "worse")],
        columns=["axis", "pair", "k", "method", "verdict"],
    ).to_csv(tmp_path / "comparisons.csv", index=False)
    out = tmp_path / "figures"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--summary",
            str(tmp_path / "summary.csv"),
            "--samples",
            str(tmp_path / "samples.csv"),
            "--comparisons",
            str(tmp_path / "comparisons.csv"),
            "--output-dir",
            str(out),
        ],
    )
    plot.main()
    names = ["fig3_aggregated", "fig3_expanded_ontology_values", "fig3_expanded_mutation"]
    assert sorted(p.name for p in out.iterdir()) == sorted(
        f"{n}.{e}" for n in names for e in ("pdf", "png")
    )
    assert plot.STYLE["NewMethod"][1] == plot.MUTED
