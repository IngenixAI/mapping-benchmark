"""Score one method, or one bound, on one pair of the ontology or donor axis.

Reads the method's ``latents.npy`` and the matching ``latents.csv`` (one row per latent,
with ``dataset`` and the label column), splits them into the query and database datasets
and writes ``mapping_benchmark.retrieval.evaluate``'s result as JSON, a method's with its
cluster-bootstrap 95% interval. A bound reads the labels only, so ``--latents`` is not
needed.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mapping_benchmark import ontology
from mapping_benchmark.retrieval import Pool, Relevance, evaluate, resampling_units


def load_pools(
    samples: pd.DataFrame, latents: np.ndarray | None, query: str, database: str, label_column: str
) -> tuple[Pool, Pool]:
    """The two sides of a pair. Without latents the features are zero-width."""
    if latents is None:
        latents = np.empty((len(samples), 0))
    if len(latents) != len(samples):
        raise ValueError(f"{len(latents)} latents but {len(samples)} samples rows")
    labels = samples[label_column].fillna("").astype(str).to_numpy()
    # A TCGA patient's aliquots share the barcode's first 12 characters, a MetaPRISM
    # patient's samples their MP number; the paired study names the donor. TCGA patients
    # are then resampled by tissue source site.
    patients = (samples["sample_id"].str.extract(r"^(TCGA-\w\w-\w{4}|MP-\d+)", expand=False)
                .fillna(samples["donor_id"]).fillna(samples["sample_id"]).to_numpy())
    clusters = resampling_units(patients)
    pools = []
    for dataset in (query, database):
        mask = (samples["dataset"] == dataset).to_numpy()
        if not mask.any():
            raise ValueError(f"no samples of dataset {dataset!r}")
        pools.append(Pool(latents[mask], labels[mask], clusters[mask]))
    return pools[0], pools[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--latents", type=Path, help="Not needed with --bound")
    parser.add_argument("--query-dataset", required=True)
    parser.add_argument("--database-dataset", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--relevance", choices=["binary", "simgic"], required=True)
    parser.add_argument("--ontology", type=Path, help="Ontology parquet, for simgic")
    parser.add_argument("--k", required=True, help="Comma-separated k values")
    parser.add_argument("--bound", choices=["oracle", "random"])
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="Cluster resamples for a method's 95%% interval; 0 for none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not (args.latents or args.bound):
        parser.error("--latents is required unless --bound is given")

    samples = pd.read_csv(args.samples, index_col=0)
    latents = np.load(args.latents) if args.latents and not args.bound else None
    query, database = load_pools(
        samples, latents, args.query_dataset, args.database_dataset, args.label_column
    )
    graph = ontology.load(args.ontology) if args.relevance == "simgic" else None
    result = evaluate(
        query,
        database,
        Relevance(args.relevance, graph),
        [int(k) for k in args.k.split(",")],
        bound=args.bound,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    for k, m in sorted(result.metrics.items()):
        ci = f" [{m.precision_ci[0]:.4f}, {m.precision_ci[1]:.4f}]" if m.precision_ci else ""
        print(
            f"{args.bound or 'method'} {args.query_dataset}->{args.database_dataset} "
            f"k={k}: precision={m.precision:.4f}{ci} "
            f"({result.n_query} queries, {result.n_database} database samples)"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
