"""Derive the gene panel of the mutation axis, and list the query samples it leaves empty.

The panel is the set of genes every mutation dataset has a column for, so that a Jaccard
between two datasets is over genes both were assayed on. ``panel_genes.yaml`` holds it;
``excluded_queries.yaml`` lists the query samples without mutation calls and those with
no mutated gene on the panel (an undefined Jaccard), which the evaluation drops.
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

HEADER = "# Derived by mutation_panel.py - do not edit.\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutation-dir", required=True, type=Path,
                        help="Directory holding <dataset>.parquet")
    parser.add_argument("--datasets", required=True,
                        help="Comma-separated datasets the panel is shared by")
    parser.add_argument("--query-datasets", required=True,
                        help="Comma-separated datasets used as queries")
    parser.add_argument("--metadata", required=True, type=Path, help="Exported metadata CSV")
    parser.add_argument("--panel-out", required=True, type=Path)
    parser.add_argument("--excluded-out", required=True, type=Path)
    args = parser.parse_args()

    datasets = sorted(args.datasets.split(","))
    parquets = {ds: args.mutation_dir / f"{ds}.parquet" for ds in datasets}
    schemas = [pq.read_schema(path) for path in parquets.values()]
    panel = sorted(set.intersection(*(
        set(s.names) - set(s.pandas_metadata["index_columns"]) for s in schemas
    )))
    print(f"Panel: {len(panel)} genes shared by {datasets}")
    args.panel_out.parent.mkdir(parents=True, exist_ok=True)
    args.panel_out.write_text(HEADER + yaml.safe_dump(
        {"n_genes": len(panel), "datasets": datasets, "genes": panel}, sort_keys=False))

    metadata = pd.read_csv(args.metadata, index_col=0, dtype=str, keep_default_na=False)
    empty: dict[str, list] = {}
    unassayed: dict[str, list] = {}
    for ds in sorted(args.query_datasets.split(",")):
        matrix = pd.read_parquet(parquets[ds], columns=panel)
        assayed = set(matrix.index.astype(str))
        empty_ids = set(matrix.index[matrix.sum(axis=1) == 0].astype(str))
        ids = metadata.loc[metadata["dataset"] == ds, "sample_id"]
        empty[ds] = sorted(ids[ids.isin(empty_ids)])
        unassayed[ds] = sorted(ids[~ids.isin(assayed)])
        print(f"{ds}: {len(empty[ds])} empty on the panel, {len(unassayed[ds])} without "
              f"calls, of {len(ids)} samples")
    args.excluded_out.write_text(HEADER + yaml.safe_dump({
        "n_empty_on_panel": {ds: len(v) for ds, v in empty.items()},
        "n_unassayed": {ds: len(v) for ds, v in unassayed.items()},
        "empty_on_panel": empty,
        "unassayed": unassayed,
    }, sort_keys=False))


if __name__ == "__main__":
    main()
