"""Embed an h5ad with a fitted PCA: latents.npy, plus the samples' obs as latents.csv."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scanpy as sc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Directory from simplepca-train")
    parser.add_argument("--data-file", required=True, help="Input h5ad, TPM in X")
    parser.add_argument("--output", required=True, help="Output .npy")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    pca = joblib.load(model_dir / "pca_model.joblib")
    genes = pd.read_csv(model_dir / "features.csv")["gene"].tolist()

    adata = sc.read_h5ad(args.data_file)
    if list(adata.var_names) != genes:
        raise SystemExit("the data does not carry the gene panel the model was fitted on")
    x = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    latents = pca.transform(np.log1p(x))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, latents)
    adata.obs.to_csv(output.with_suffix(".csv"))
    print(f"Wrote {latents.shape} latents to {output}")


if __name__ == "__main__":
    main()
