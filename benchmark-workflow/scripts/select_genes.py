"""Select the gene panel shared by every method.

Ranks genes by pooled within-dataset variance of log1p expression, over the `--datasets` given
(the training datasets), so the panel never sees data the models are not fitted on.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def select_genes(expression: np.ndarray, datasets: np.ndarray, n_genes: int) -> np.ndarray:
    """Column indices of the `n_genes` most variable genes WITHIN datasets, on log1p expression.

    Pooled within-dataset variance = sum_d (n_d - 1) var_d / sum_d (n_d - 1).
    """
    weighted = np.zeros(expression.shape[1], dtype=np.float64)
    dof = 0
    for dataset in np.unique(datasets):
        rows = datasets == dataset
        n = int(rows.sum())
        weighted += np.log1p(expression[rows]).var(axis=0, ddof=1) * (n - 1)
        dof += n - 1
    return np.argsort(weighted / dof)[::-1][:n_genes].copy()


def main():
    parser = argparse.ArgumentParser(description="Select the shared gene panel")
    parser.add_argument("--input", required=True, help="Combined benchmark h5ad")
    parser.add_argument("--output", required=True, help="Output CSV with gene names")
    parser.add_argument("--datasets", required=True,
                        help="Comma-separated dataset values to pool variance within")
    parser.add_argument("--n-genes", type=int, default=2000, help="Number of genes to select")
    args = parser.parse_args()

    wanted = [d.strip() for d in args.datasets.split(",")]

    print(f"Loading {args.input}...")
    adata = sc.read_h5ad(args.input)
    mask = adata.obs["dataset"].isin(wanted).to_numpy()
    if not mask.any():
        raise SystemExit(
            f"none of {wanted} in dataset: "
            f"{sorted(adata.obs['dataset'].unique())}"
        )
    sub = adata[mask]
    print(f"Pooling within {wanted}: {sub.n_obs} samples x {sub.n_vars} genes")

    x = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
    index = select_genes(x, sub.obs["dataset"].to_numpy().astype(str), args.n_genes)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"gene": adata.var_names[index]}).to_csv(output, index=False)
    print(f"Wrote {len(index)} genes (of {adata.n_vars}) to {output}")


if __name__ == "__main__":
    main()
