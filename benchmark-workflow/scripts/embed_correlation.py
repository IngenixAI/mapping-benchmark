"""A model-free baseline: the gene panel's log1p(TPM), row-standardised, as the latents.

Nothing is fitted. Between two row-standardised profiles over n genes the squared
Euclidean distance is 2n(1 - r), with r their Pearson correlation, so Euclidean retrieval
ranks the database exactly as correlation distance does.

Each sample is transformed independently; evaluation-cohort means are never estimated.
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np


def embed(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=np.float64)
    x -= x.mean(axis=1, keepdims=True)
    scale = x.std(axis=1, keepdims=True)
    if not np.isfinite(x).all() or np.any(scale == 0):
        raise ValueError("correlation requires finite profiles with nonzero variance")
    return (x / scale).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file",
        required=True,
        type=Path,
        help="Every sample on the gene panel, in log1p(TPM)",
    )
    parser.add_argument("--output-latents", required=True, type=Path)
    parser.add_argument("--output-samples", required=True, type=Path)
    args = parser.parse_args()

    data = ad.read_h5ad(args.data_file)
    x = data.X.toarray() if hasattr(data.X, "toarray") else data.X
    latents = embed(x)
    args.output_latents.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_latents, latents)
    data.obs.to_csv(args.output_samples)
    print(f"{latents.shape[0]} samples x {latents.shape[1]} genes")


if __name__ == "__main__":
    main()
