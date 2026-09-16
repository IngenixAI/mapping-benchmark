#!/usr/bin/env python3
"""Snakemake adapter: h5ad in, a saved scVI model dir out.

All modelling lives in scvi_baseline/model.py; this only translates I/O conventions.
"""

import argparse

import numpy as np
import scanpy as sc

from scvi_baseline.model import BATCH_KEY, train


def main():
    parser = argparse.ArgumentParser(description="Train the scVI baseline on an h5ad")
    parser.add_argument("--train-file", required=True, help="Input h5ad with linear TPM in X")
    parser.add_argument("--output-dir", required=True, help="Directory for the saved model")
    parser.add_argument("--n-latent", type=int, default=64, help="Latent dimensions")
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-hidden", type=int, default=256)
    parser.add_argument("--gene-likelihood", choices=["nb", "zinb", "poisson"], default="nb")
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--kl-warmup-epochs", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.train_file}...")
    adata = sc.read_h5ad(args.train_file)
    print(f"  {adata.n_obs} samples x {adata.n_vars} genes")
    if BATCH_KEY not in adata.obs.columns:
        raise SystemExit(f"{BATCH_KEY!r} not in obs: {list(adata.obs.columns)}")

    x = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    train(
        x,
        adata.var_names.to_numpy(),
        adata.obs[BATCH_KEY].astype(str).to_numpy(),
        args.output_dir,
        max_epochs=args.max_epochs,
        seed=args.random_seed,
        n_latent=args.n_latent,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        gene_likelihood=args.gene_likelihood,
        dropout_rate=args.dropout_rate,
        kl_warmup_epochs=args.kl_warmup_epochs,
    )


if __name__ == "__main__":
    main()
