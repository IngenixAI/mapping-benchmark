"""Download the benchmark's inputs and write them in the common layout.

See ``mapping_benchmark.datasets`` for what is written.
"""

import argparse
from pathlib import Path

from mapping_benchmark import datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="resources/datasets.yaml")
    parser.add_argument("--expression", type=Path, required=True, help="Output h5ad")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Output CSV with the h5ad's obs")
    parser.add_argument("--ontology", type=Path, required=True, help="Output MONDO parquet")
    parser.add_argument("--mutation-dir", type=Path, required=True)
    args = parser.parse_args()

    spec = datasets.load_spec(args.spec)
    datasets.export_ontology(spec, args.ontology)
    combined = datasets.export_expression(spec, args.expression)
    combined.obs.to_csv(args.metadata)
    for name in datasets.mutation_datasets(spec):
        datasets.export_mutations(spec, name, args.mutation_dir / f"{name}.parquet")


if __name__ == "__main__":
    main()
