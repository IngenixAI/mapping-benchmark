"""Score one method, or one bound, on one pair and cohort of the mutation axis.

Samples without mutation calls, outside the cohort, or (on the query side) with no
mutation in the gene panel are left out; the panel comes from ``mutation_panel.py``.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mapping_benchmark import mutation, ontology
from mapping_benchmark.retrieval import Pool, Relevance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--latents", type=Path, help="Not needed with --bound")
    parser.add_argument("--query-dataset", required=True)
    parser.add_argument("--database-dataset", required=True)
    parser.add_argument("--query-mutations", required=True, type=Path)
    parser.add_argument("--database-mutations", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path, help="panel_genes.yaml")
    parser.add_argument("--cohorts", required=True, type=Path, help="mutation_cohorts.yaml")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--label-column", required=True, help="MONDO term column")
    parser.add_argument("--ontology", required=True, type=Path, help="For the oracle")
    parser.add_argument("--k", required=True, help="Comma-separated k values")
    parser.add_argument("--bound", choices=["oracle", "random"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not (args.latents or args.bound):
        parser.error("--latents is required unless --bound is given")

    metadata = pd.read_csv(args.metadata, index_col=0)
    latents = (np.load(args.latents) if args.latents and not args.bound
               else np.empty((len(metadata), 0)))
    panel = yaml.safe_load(args.panel.read_text())["genes"]
    cohort = mutation.load_cohorts(args.cohorts)[args.cohort]

    sides = {}
    for side, dataset, path in (("query", args.query_dataset, args.query_mutations),
                                ("database", args.database_dataset, args.database_mutations)):
        matrix = pd.read_parquet(path, columns=panel)
        meta = metadata[metadata["dataset"] == dataset]
        rows = latents[(metadata["dataset"] == dataset).to_numpy()]
        keep = meta["sample_id"].astype(str).isin(matrix.index).to_numpy()
        keep &= meta[args.label_column].fillna("").astype(str).ne("").to_numpy()
        if cohort:
            keep &= meta[args.label_column].isin(cohort).to_numpy()
        muts = matrix.loc[meta["sample_id"].astype(str)[keep].to_numpy()].to_numpy(bool)
        if side == "query":
            # A query with no mutation on the panel has an undefined Jaccard.
            non_empty = muts.any(axis=1)
            keep[np.flatnonzero(keep)[~non_empty]] = False
            muts = muts[non_empty]
        sides[side] = (Pool(rows[keep], meta[args.label_column].to_numpy()[keep]), muts)
        print(f"{side} {dataset}: {int(keep.sum())} of {len(meta)} samples in cohort "
              f"{args.cohort!r} with mutations on the {len(panel)}-gene panel")

    (query, q_mut), (database, db_mut) = sides["query"], sides["database"]
    if len(query.labels) == 0 or len(database.labels) == 0:
        raise SystemExit("no samples left to score")
    disease = Relevance("simgic", ontology.load(args.ontology)) if args.bound == "oracle" else None
    result = mutation.evaluate(query, database, q_mut, db_mut,
                               [int(k) for k in args.k.split(",")],
                               bound=args.bound, disease=disease)

    for k, m in sorted(result.metrics.items()):
        print(f"{args.bound or 'method'} k={k}: jaccard={m.mean_jaccard:.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
