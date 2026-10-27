"""Write the benchmark report and reproducible artifacts after scoring succeeds."""

import hashlib
import html
import importlib.metadata
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mapping_benchmark.mutation_tasks import Tasks
from mapping_benchmark.mutation_visualization import write_explorer


def digest(path: Path) -> dict:
    with path.open("rb") as f:
        checksum = hashlib.file_digest(f, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": checksum, "bytes": path.stat().st_size}


def cohort_counts(cases: list[dict]) -> pd.DataFrame:
    """One row per case side, with explicit exclusion counts and eligibility reasons."""
    counts = []
    for case in cases:
        for side, values in case["counts"].items():
            counts.append(
                {
                    key: case[key]
                    for key in ("disease", "pair", "analysis", "biomarker", "tier", "stratum")
                }
                | {
                    "side": side,
                    **values,
                    "eligible": case["eligible"],
                    "reason": "; ".join(case["ineligible_reasons"]),
                    "unavailable_k": ",".join(map(str, case["unavailable_k"])),
                }
            )
    return pd.DataFrame(counts)


def report_protocol(tasks: Tasks) -> list[str]:
    """Describe the declared cohort rules, uncertainty and testing families."""
    return [
        "<!doctype html><html lang='en'><meta charset='utf-8'>",
        "<title>Biomarker retrieval benchmark</title>",
        "<style>body{font:15px system-ui;margin:32px;color:#183038}"
        "table{border-collapse:collapse;font-size:12px}"
        "td,th{padding:6px;border:1px solid #ccd7d9}"
        "p{max-width:1100px;line-height:1.5}.scroll{overflow-x:auto}</style>",
        "<h1>Biomarker retrieval benchmark</h1>",
        f"<p>Declared primary depth: {tasks.primary_k}. Cohorts require "
        f"{tasks.eligibility.mutant_queries} positive / "
        f"{tasks.eligibility.wildtype_queries} control queries and "
        f"{tasks.eligibility.mutant_pool} positive / "
        f"{tasks.eligibility.wildtype_pool} control pool units. "
        "Ineligible cohorts are retained with exclusion reasons and receive no scores. "
        "Depths larger than the labelled pool are unavailable, without substituting k.</p>",
        "<p>Strict configured OncoTree codes apply to both sides. One representative "
        "expression row is selected by sample ID per sequenced tumour, model or patient. "
        "MSI does not require mutation-call availability. Missing or conflicting MSI "
        "evidence remains unknown; MSS strata require explicit negative MSI evidence. "
        "MetaPRISM MSI is unavailable in the pinned source.</p>",
        "<p>Mutation controls are provisional noncarriers in samples with "
        "somatic variant-call records. "
        "Call-set presence does not establish gene-level sequencing coverage. "
        "Each biomarker declares controls: alteration-negative, protein-altering-free, "
        "or mutation-free, plus whether known homozygous deletions are withheld. "
        "Unknown required annotations are withheld. Disease definitions can exclude "
        "positive MSI screens; this does not call remaining unknown samples MSS. "
        "A selected positive takes precedence over other gene calls. "
        "Sequence consequences do not establish biallelic or functional loss.</p>",
        f"<p>Intervals use {tasks.bootstrap} cluster bootstrap draws (a TCGA tumour's "
        "tissue source site, any other tumour's patient), conditional "
        f"on the fixed model catalogue. One-sided p-values use {tasks.permutations} "
        "permutations of pool labels and the plus-one correction. BH q-values are "
        "computed separately for primary and exploratory tiers, across all scored "
        "diseases, pairs and methods at the primary depth in this run. Sensitivity "
        "analyses and other depths have raw p-values only. "
        "Exploratory results support hypothesis generation.</p>",
        "<p>AUROC random reference: 0.5. Oracle values use each case's actual labelled pool. "
        "All methods use identical per-case sample IDs; missing embeddings fail the run.</p>",
    ]


def write_report(
    output: Path, cases: list[dict], scores: pd.DataFrame, tasks: Tasks, methods: list[str]
) -> None:
    count_table = cohort_counts(cases)
    count_table.to_csv(output / "counts.csv", index=False)

    parts = report_protocol(tasks)
    parts.append("<p><a href='explorer.html'>Open the interactive mutation explorer</a>: "
                 "filter cancer, dataset pair, analysis, tier and shortlist depth.</p>")

    for tier in ("primary", "exploratory", "sensitivity"):
        table = scores[scores.tier.eq(tier) & scores.k.eq(tasks.primary_k)]
        parts.extend(
            [
                f"<h2>{html.escape(tier.title())} at k={tasks.primary_k}</h2>",
                "<div class='scroll'>",
                table.round(4).to_html(index=False),
                "</div>",
            ]
        )

    # Keep the actual declared biomarker definitions beside the results.
    rules = [{"biomarker": n, **a.model_dump()} for n, a in tasks.biomarkers.items()]
    parts.extend(
        [
            "<h2>Cohort counts and eligibility</h2><div class='scroll'>",
            count_table.to_html(index=False),
            "</div>",
            "<h2>Declared biomarkers</h2><div class='scroll'>",
            pd.DataFrame(rules).to_html(index=False),
            "</div>",
            "<p>Artifacts: <a href='scores.csv'>scores.csv</a>, "
            "<a href='counts.csv'>counts.csv</a>, "
            "<a href='results.json'>results.json</a>, "
            "<a href='sample_labels.csv'>sample_labels.csv</a>, "
            "<a href='variant_evidence.csv'>variant_evidence.csv</a>, "
            "<a href='manifest.json'>manifest.json</a>.</p></html>",
        ]
    )
    (output / "report.html").write_text("\n".join(parts))
    write_explorer(output / "explorer.html", cases, scores, tasks, methods)


def build_manifest(tasks: Tasks, methods: list[str], provenance: dict) -> dict:
    """Record the exact inputs, implementation and environment for this run."""
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    code = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    template = Path(__file__).with_name("templates") / "mutation_explorer.html"
    code["templates/mutation_explorer.html"] = digest(template)
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": tasks.model_dump(),
        "methods": methods,
        **provenance,
        "git_revision": revision,
        "git_dirty": dirty,
        "code": code,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("mapping-benchmark", "numpy", "pandas", "scipy", "scikit-learn", "pydantic")
        },
        "fdr_policy": "BH on AUROC separately by tier (primary/exploratory), "
        "all diseases/pairs/methods at primary_k in this run",
    }


def write_results(
    output: Path,
    labels: pd.DataFrame,
    evidence: pd.DataFrame,
    cases: list[dict],
    scores: pd.DataFrame,
    tasks: Tasks,
    methods: list[str],
    provenance: dict,
) -> None:
    """Stage all artifacts; publish the manifest last so hashes verify completion."""
    manifest = build_manifest(tasks, methods, provenance)

    # Complete every computation before publishing. The manifest is the final commit
    # marker; its output hashes allow readers to detect any interrupted publication.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mutation-", dir=output.parent) as tmp:
        stage = Path(tmp)
        labels.to_csv(stage / "sample_labels.csv")
        evidence.to_csv(stage / "variant_evidence.csv", index=False)
        scores.to_csv(stage / "scores.csv", index=False)

        # pandas JSON conversion turns missing/NaN values into standards-compliant nulls.
        result = {
            "schema_version": 1,
            "tasks": tasks.model_dump(),
            "methods": methods,
            "cases": cases,
            "scores": json.loads(scores.to_json(orient="records")),
        }
        (stage / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        write_report(stage, cases, scores, tasks, methods)

        # Hash staged artifacts before publishing the final completion marker.
        manifest["outputs"] = {
            p.name: {k: v for k, v in digest(p).items() if k != "path"}
            for p in sorted(stage.iterdir())
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))

        output.mkdir(parents=True, exist_ok=True)
        for p in sorted(stage.iterdir(), key=lambda p: p.name == "manifest.json"):
            os.replace(p, output / p.name)
