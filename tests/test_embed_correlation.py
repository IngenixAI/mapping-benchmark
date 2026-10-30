"""The model-free baseline: each expression profile transformed independently."""

import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "benchmark-workflow" / "scripts" / "embed_correlation.py"


def run(tmp_path: Path, x: np.ndarray, datasets: list[str], *extra: str) -> np.ndarray:
    obs = pd.DataFrame(
        {"sample_id": [f"s{i}" for i in range(len(x))], "dataset": datasets},
        index=[f"s{i}" for i in range(len(x))],
    )
    ad.AnnData(x.astype(np.float32), obs=obs).write_h5ad(tmp_path / "data.h5ad")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data-file",
            str(tmp_path / "data.h5ad"),
            "--output-latents",
            str(tmp_path / "out.npy"),
            "--output-samples",
            str(tmp_path / "out.csv"),
            *extra,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    written = pd.read_csv(tmp_path / "out.csv", index_col=0)
    assert written["sample_id"].tolist() == obs.index.tolist()
    return np.load(tmp_path / "out.npy")


def test_euclidean_distance_ranks_like_correlation(tmp_path):
    x = np.random.default_rng(0).gamma(2.0, size=(6, 50))
    z = run(tmp_path, x, ["tcga"] * 6).astype(np.float64)
    r = np.corrcoef(x)
    d2 = ((z[:, None] - z[None]) ** 2).sum(-1)
    np.testing.assert_allclose(d2, 2 * x.shape[1] * (1 - r), atol=1e-3)


def test_embedding_does_not_depend_on_the_other_samples(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 40))
    alone = run(tmp_path, x[:1], ["metaprism"])
    together = run(tmp_path, x, ["metaprism"] * 4)
    np.testing.assert_array_equal(alone[0], together[0])
