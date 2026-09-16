"""Score one method, or one bound, on one pair of the ontology or donor axis.

Reads the method's ``latents.npy`` and the matching ``latents.csv`` (one row per latent,
with ``dataset`` and the label column), splits them into the query and database datasets
and writes ``mapping_benchmark.retrieval.evaluate``'s result as JSON. A bound reads the
labels only, so ``--latents`` is not needed.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mapping_benchmark import ontology
from mapping_benchmark.retrieval import Pool, Relevance, evaluate


def load_pools(metadata: pd.DataFrame, latents: np.ndarray | None, query: str,
               database: str, label_column: str) -> tuple[Pool, Pool]:
    """The two sides of a pair. Without latents the features are zero-width."""
    if latents is None:
        latents = np.empty((len(metadata), 0))
    if len(latents) != len(metadata):
        raise ValueError(f"{len(latents)} latents but {len(metadata)} metadata rows")
    labels = metadata[label_column].fillna("").astype(str).to_numpy()
    pools = []
    for dataset in (query, database):
        mask = (metadata["dataset"] == dataset).to_numpy()
        if not mask.any():
            raise ValueError(f"no samples of dataset {dataset!r}")
        pools.append(Pool(latents[mask], labels[mask]))
    return pools[0], pools[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--latents", type=Path, help="Not needed with --bound")
    parser.add_argument("--query-dataset", required=True)
    parser.add_argument("--database-dataset", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--relevance", choices=["binary", "simgic"], required=True)
    parser.add_argument("--ontology", type=Path, help="Ontology parquet, for simgic")
    parser.add_argument("--k", required=True, help="Comma-separated k values")
    parser.add_argument("--bound", choices=["oracle", "random"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not (args.latents or args.bound):
        parser.error("--latents is required unless --bound is given")

    metadata = pd.read_csv(args.metadata, index_col=0)
    latents = np.load(args.latents) if args.latents and not args.bound else None
    query, database = load_pools(metadata, latents, args.query_dataset,
                                 args.database_dataset, args.label_column)
    graph = ontology.load(args.ontology) if args.relevance == "simgic" else None
    result = evaluate(query, database, Relevance(args.relevance, graph),
                      [int(k) for k in args.k.split(",")], bound=args.bound)

    for k, m in sorted(result.metrics.items()):
        print(f"{args.bound or 'method'} {args.query_dataset}->{args.database_dataset} "
              f"k={k}: precision={m.precision:.4f} "
              f"({result.n_query} queries, {result.n_database} database samples)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
