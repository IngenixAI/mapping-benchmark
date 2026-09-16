#!/usr/bin/env python3
"""Snakemake adapter: h5ad + a saved model dir in, latents.npy + latents.csv out.

`model.embed` does the whole job, including the gene-panel check.
"""

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc

from scvi_baseline.model import embed


def main():
    parser = argparse.ArgumentParser(description="Extract scVI latent representations")
    parser.add_argument("--model-dir", required=True, help="Model directory from scvi-train")
    parser.add_argument("--data-file", required=True, help="h5ad to project (linear TPM in X)")
    parser.add_argument("--output-latents", required=True, help="Output .npy")
    parser.add_argument("--output-metadata", required=True, help="Output .csv")
    args = parser.parse_args()

    print(f"Loading {args.data_file}...")
    adata = sc.read_h5ad(args.data_file)
    print(f"  {adata.n_obs} samples x {adata.n_vars} genes")

    x = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    latent = embed(args.model_dir, x, adata.var_names.to_numpy())
    print(f"  Latent shape: {latent.shape}")

    latents_path = Path(args.output_latents)
    latents_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(latents_path, latent)

    metadata_path = Path(args.output_metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    adata.obs.to_csv(metadata_path)
    print(f"Wrote {latents_path} and {metadata_path}")


if __name__ == "__main__":
    main()
