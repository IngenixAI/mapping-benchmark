"""The scVI baseline: `train` fits and saves, `embed` loads and encodes.

Input is unrounded linear TPM; scVI log-transforms the encoder input itself. Embeddings
are the encoder's mean, L2-normalised.
"""

import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import scvi

LAYER = "expression"
META_FILE = "genes.json"
# Never enters the latent: scVI's encoder drops the batch index unless `encode_covariates=True`.
BATCH_KEY = "data_source"


def _anndata(expression, genes, datasets) -> ad.AnnData:
    x = np.asarray(expression, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D (n, G) matrix, got shape {x.shape}")
    if x.shape[1] != len(genes):
        raise ValueError(f"{x.shape[1]} columns but {len(genes)} gene names")
    adata = ad.AnnData(X=x, obs=pd.DataFrame({BATCH_KEY: pd.Categorical(datasets)}))
    adata.var_names = [str(g) for g in genes]
    adata.layers[LAYER] = x
    return adata


def train(
    expression,
    genes,
    datasets,
    output_dir: str,
    n_latent: int = 64,
    n_layers: int = 1,
    n_hidden: int = 256,
    gene_likelihood: str = "nb",
    dropout_rate: float = 0.0,
    max_epochs: int = 300,
    kl_warmup_epochs: int = 100,
    seed: int = 42,
) -> None:
    scvi.settings.seed = seed
    datasets = np.asarray(datasets, dtype=str)
    adata = _anndata(expression, genes, datasets)

    scvi.model.SCVI.setup_anndata(adata, layer=LAYER, batch_key=BATCH_KEY)
    fitted = scvi.model.SCVI(
        adata,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        gene_likelihood=gene_likelihood,
        dropout_rate=dropout_rate,
    )
    fitted.train(max_epochs=max_epochs, plan_kwargs={"n_epochs_kl_warmup": kl_warmup_epochs})
    elbo = fitted.history["elbo_train"]["elbo_train"]
    print(f"ran {len(elbo)} epochs (KL warmup {kl_warmup_epochs}); "
          f"final ELBO train {elbo.iloc[-1]:.1f}")

    os.makedirs(output_dir, exist_ok=True)
    fitted.save(output_dir, overwrite=True, save_anndata=False)
    with open(os.path.join(output_dir, META_FILE), "w") as f:
        json.dump({
            "genes": [str(g) for g in genes],
            "batch_category": str(sorted(set(datasets))[0]),
        }, f, indent=2)

    latent = fitted.get_latent_representation()
    spread = float(latent.std(axis=0).mean())
    print(f"fit scVI on {adata.shape} ({len(set(datasets))} datasets); "
          f"latent {latent.shape}, per-dim sd {spread:.4g}; saved to {output_dir}")
    if spread < 0.05:
        print(
            f"  WARNING: latent sd {spread:.4g} — posterior collapse, "
            "retrieval will be at chance"
        )
    # own/nearest-other centroid distance: ~1.0 = well mixed, << 1 = datasets still cluster apart.
    centroids = {d: latent[datasets == d].mean(axis=0) for d in sorted(set(datasets))}
    for dataset, centre in centroids.items():
        rows = latent[datasets == dataset]
        own = np.linalg.norm(rows - centre, axis=1).mean()
        other = min(np.linalg.norm(rows - c, axis=1).mean()
                    for d, c in centroids.items() if d != dataset)
        print(f"  {dataset:16s} n={int((datasets == dataset).sum()):6d} "
              f"own/nearest-other centroid dist = {own / other:.3f}")


def embed(model_dir: str, expression, genes) -> np.ndarray:
    """(n, G) linear TPM -> (n, n_latent) embedding, for any dataset, seen in training or not."""
    with open(os.path.join(model_dir, META_FILE)) as f:
        meta = json.load(f)
    # The gene axis is positional, so a panel reselected since training would silently embed the
    # wrong genes.
    got = [str(g) for g in genes]
    if got != meta["genes"]:
        raise ValueError(
            f"gene axis mismatch: the data does not carry the panel this model was fitted on "
            f"({len(got)} vs {len(meta['genes'])} genes; first differing position "
            f"{next((i for i, (a, b) in enumerate(zip(got, meta['genes'])) if a != b), 'n/a')}). "
            f"Re-run training against the current selected_genes.csv."
        )
    adata = _anndata(expression, got, [meta["batch_category"]] * len(expression))
    fitted = scvi.model.SCVI.load(model_dir, adata=adata)
    latent = np.asarray(fitted.get_latent_representation(), dtype=np.float32)
    # Unit-norm rows: Euclidean nearest neighbours then rank like cosine similarity.
    norms = np.linalg.norm(latent, axis=1, keepdims=True)
    return latent / np.maximum(norms, np.finfo(latent.dtype).eps)
