# PCA baseline

Principal component analysis of log1p(TPM), fitted on the training datasets and applied
to every sample. The simplest embedding the benchmark scores.

```bash
uv sync --locked                      # in this directory
uv run simplepca-train --train-file training.h5ad --output-dir model/ --n-components 64
uv run simplepca-project --model-dir model/ --data-file expression.h5ad --output latents.npy
```

`simplepca-project` writes `latents.npy` (samples × components) and `latents.csv` (the
input's `obs`, one row per latent). The data must carry the gene panel the model was fitted
on; the benchmark's `prepare_data` step guarantees that.
