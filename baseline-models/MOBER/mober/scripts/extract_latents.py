#!/usr/bin/env python3
"""Embed an h5ad with a trained MOBER model: the encoder's mean, one row per sample.

Writes ``latents.npy`` and, beside it, ``latents.csv`` with the samples' obs.
"""

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader, TensorDataset

from mober.core.projection import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Output directory of `mober train`")
    parser.add_argument("--data-file", required=True, help="Input h5ad, log1p(TPM) in X")
    parser.add_argument("--output", required=True, help="Output .npy")
    parser.add_argument("--batch-size", type=int, default=1600)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, features, _ = load_model(str(Path(args.model_dir) / "models"), device)
    adata = sc.read_h5ad(args.data_file)
    if list(adata.var_names) != list(features):
        raise SystemExit("the data does not carry the gene panel the model was fitted on")
    x = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)

    net = model.module if isinstance(model, torch.nn.DataParallel) else model
    net.eval()
    latents = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)),
                                   batch_size=args.batch_size):
            latents.append(net.encoder(batch.to(device))[0].cpu().numpy())
    latents = np.concatenate(latents)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, latents)
    adata.obs.to_csv(output.with_suffix(".csv"))
    print(f"Wrote {latents.shape} latents to {output}")


if __name__ == "__main__":
    main()
