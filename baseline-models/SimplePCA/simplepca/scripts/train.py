"""Fit PCA on log1p(TPM) of the training data and save it."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True, help="Input h5ad, TPM in X")
    parser.add_argument("--output-dir", required=True, help="Directory for the model")
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    adata = sc.read_h5ad(args.train_file)
    print(f"Fitting PCA on {adata.n_obs} samples x {adata.n_vars} genes")
    x = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    pca = PCA(n_components=args.n_components, random_state=args.random_seed).fit(np.log1p(x))
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, output / "pca_model.joblib")
    adata.var_names.to_frame(name="gene").to_csv(output / "features.csv", index=False)
    print(f"Saved model to {output}")


if __name__ == "__main__":
    main()
