"""Subset the exported h5ad to the gene panel, and optionally to some datasets and to log1p.

Adds the ``data_source`` column the methods use as the batch label and stores the
expression matrix dense, which is what they expect.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Exported h5ad")
    parser.add_argument("--genes", required=True, help="Selected genes CSV")
    parser.add_argument("--output", required=True, help="Output h5ad")
    parser.add_argument("--datasets", default="",
                        help="Comma-separated datasets to keep; default every dataset")
    parser.add_argument("--log1p", action="store_true", help="Store log1p(TPM) instead of TPM")
    args = parser.parse_args()

    adata = sc.read_h5ad(args.input)
    if args.datasets:
        adata = adata[adata.obs["dataset"].isin(args.datasets.split(","))]
    genes = pd.read_csv(args.genes)["gene"].tolist()
    adata = adata[:, genes].copy()
    adata.obs["data_source"] = adata.obs["dataset"].values

    x = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    adata.X = np.log1p(x, dtype=np.float32) if args.log1p else x.astype(np.float32)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output)
    print(f"Wrote {adata.n_obs} samples x {adata.n_vars} genes to {args.output}")
    print(adata.obs["dataset"].value_counts().to_string())


if __name__ == "__main__":
    main()
