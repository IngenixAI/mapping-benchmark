"""Run the common mutation/MSI benchmark: labels -> cohorts/scores -> artifacts.

Cancer-specific biology is declared in mutation_tasks.yaml. Every disease follows
these same steps, and every method uses the same cohort IDs.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from mapping_benchmark import alterations
from mapping_benchmark.mutation import evaluate
from mapping_benchmark.mutation_labels import build_labels, validate_labels
from mapping_benchmark.mutation_report import digest, write_results
from mapping_benchmark.mutation_tasks import Tasks, load_tasks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", type=Path, default=Path("benchmark-workflow/resources/mutation_tasks.yaml")
    )
    parser.add_argument(
        "--sources", type=Path, default=Path("benchmark-workflow/resources/mutation_sources.yaml")
    )
    parser.add_argument(
        "--expression-samples", type=Path, default=Path("data/raw/expression_samples.csv")
    )
    parser.add_argument(
        "--benchmark-config", type=Path, default=Path("benchmark-workflow/config.yaml")
    )
    parser.add_argument("--models-root", type=Path, default=Path("results/models"))
    parser.add_argument(
        "--methods", help="Comma-separated method folders; default: benchmark config"
    )
    parser.add_argument("--diseases", help="Comma-separated task IDs; default: all configured")
    parser.add_argument("--pairs", help="Comma-separated pair IDs; default: all configured")
    parser.add_argument(
        "--labels",
        type=Path,
        help="Optional supplied label snapshot; otherwise rebuild from raw sources",
    )
    parser.add_argument("--evidence", type=Path, help="Variant evidence for supplied --labels")
    parser.add_argument("--bounds-only", action="store_true")
    parser.add_argument(
        "--baseline", help="Method every other method's pair aggregate is compared with"
    )
    parser.add_argument("--bootstrap", type=int, help="Override task bootstrap count")
    parser.add_argument("--permutations", type=int, help="Override task permutation count")
    parser.add_argument("--output", type=Path, default=Path("results/mutation"))
    args = parser.parse_args(argv)
    if args.evidence and not args.labels:
        parser.error("--evidence requires --labels")
    if args.bounds_only and args.methods:
        parser.error("--methods and --bounds-only are mutually exclusive")

    return args


def selected_tasks(tasks: Tasks, diseases: str | None, pairs: str | None) -> Tasks:
    config = tasks.model_dump()
    for key, selection in (("diseases", diseases), ("pairs", pairs)):
        if selection is not None:
            names = selection.split(",")
            if len(set(names)) != len(names) or any(n not in config[key] for n in names):
                raise ValueError(f"unknown or duplicate {key}: {selection}")
            config[key] = {n: config[key][n] for n in names}
    return Tasks.model_validate(config)


def select_methods(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve the selected method names to existing embedding folders."""
    if args.bounds_only:
        return {}

    names = (
        args.methods.split(",")
        if args.methods
        else list(yaml.safe_load(args.benchmark_config.read_text())["methods"])
    )
    if not names or len(set(names)) != len(names):
        raise ValueError("methods must be nonempty and unique")
    if any(Path(n).name != n or n in {".", "..", "Random", "Oracle"} for n in names):
        raise ValueError("methods must be folder names, excluding Random and Oracle")

    return {name: args.models_root / name for name in names}


def read_label_inputs(
    args: argparse.Namespace, tasks: Tasks
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build source labels, or read a supplied snapshot, and describe its provenance."""
    label_inputs = {}
    sources = None
    if args.labels:
        label_inputs["labels"] = digest(args.labels)
        labels = pd.read_csv(
            args.labels, index_col="sample_id", dtype={"sample_id": str}, keep_default_na=True
        )
        validate_labels(labels, tasks)
        if args.evidence:
            label_inputs["evidence"] = digest(args.evidence)
            evidence = pd.read_csv(args.evidence)
        else:
            evidence = pd.DataFrame(
                columns=[*alterations.CALL_COLUMNS, "dataset", "biomarker", "gene"]
            )
        label_origin = "supplied snapshot; biomarker definitions must match the supplied labels"
    else:
        label_inputs.update(
            sources=digest(args.sources), expression_samples=digest(args.expression_samples)
        )
        sources = alterations.load_sources(args.sources)
        expression_samples = pd.read_csv(
            args.expression_samples, index_col=0, dtype=str, keep_default_na=False
        )
        labels, evidence = build_labels(expression_samples, sources, tasks)
        label_origin = "derived from pinned raw sources"

    provenance = {
        "inputs": label_inputs,
        "sources": sources.model_dump() if sources else None,
        "label_origin": label_origin,
        "variant_evidence_available": bool(args.evidence) or not args.labels,
    }
    return labels, evidence, provenance


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # 1. Validate the shared specification and choose methods.
    tasks = selected_tasks(load_tasks(args.tasks), args.diseases, args.pairs)
    overrides = {
        key: getattr(args, key)
        for key in ("bootstrap", "permutations")
        if getattr(args, key) is not None
    }
    tasks = Tasks.model_validate(tasks.model_dump() | overrides)
    methods = select_methods(args)
    if args.baseline and args.baseline not in methods:
        raise ValueError(f"baseline {args.baseline!r} is not among the methods")

    inputs = {"tasks": digest(args.tasks), "runner": digest(Path(__file__))}
    if not args.methods and not args.bounds_only:
        inputs["benchmark_config"] = digest(args.benchmark_config)
    for name, folder in methods.items():
        inputs[f"{name}/embedding_samples"] = digest(folder / "latents.csv")
        inputs[f"{name}/latents"] = digest(folder / "latents.npy")

    # 2. Build source labels or validate a supplied snapshot.
    labels, evidence, provenance = read_label_inputs(args, tasks)
    provenance["inputs"].update(inputs)

    # 3. Select common cohorts, score all methods and correct testing families.
    cases, scores, aggregate = evaluate(
        labels, tasks, methods, progress=lambda s: print(s, flush=True), baseline=args.baseline
    )

    # 4. Publish tables, report and provenance only after scoring succeeds.
    write_results(
        args.output, labels, evidence, cases, scores, aggregate, tasks, list(methods), provenance
    )
    print(f"Report: {args.output / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
