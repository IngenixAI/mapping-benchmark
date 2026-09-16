# scVI baseline

[scVI](https://docs.scvi-tools.org/) (Lopez et al. 2018), a variational autoencoder with
a negative-binomial likelihood, applied to bulk RNA-seq. It is fed unrounded linear TPM
with the dataset of origin as the batch key; the batch index never enters the latent.
The embedding is the encoder's mean, L2-normalised so that Euclidean nearest neighbours
rank like cosine similarity.

```bash
uv sync --locked                      # in this directory
uv run scvi-train --train-file training.h5ad --output-dir model/ \
    --n-latent 64 --n-layers 1 --n-hidden 256 --dropout-rate 0 --gene-likelihood nb \
    --max-epochs 300 --kl-warmup-epochs 100 --random-seed 42
uv run scvi-project --model-dir model/ --data-file expression.h5ad \
    --output-latents latents.npy --output-metadata latents.csv
```

`scvi-train` needs the `data_source` obs column and writes scvi-tools' own model format
plus `genes.json`, the gene panel it was fitted on; `scvi-project` refuses data on any
other panel. Training prints the final ELBO, the latent spread and how far each dataset's
samples sit from the others, which is a quick read on whether the batches mixed.
