# MOBER

Adapted from [Novartis/MOBER](https://github.com/Novartis/MOBER), the Multi Origin Batch Effect Remover. The upstream [MIT license](LICENSE) and copyright notice are retained.

## Run in this benchmark

Follow the [root README](../../README.md) for installation and the full workflow. From the repository root, the MOBER environment can also be installed and its commands inspected with:

```bash
uv sync --locked --directory baseline-models/MOBER
uv run --frozen --directory baseline-models/MOBER mober train --help
uv run --frozen --directory baseline-models/MOBER mober-extract-latents --help
```

## Local adaptations

- **Packaging:** `pyproject.toml`/Hatchling and `uv.lock` define the environment and the `mober` and `mober-extract-latents` commands. The upstream `setup.py` is removed.
- **Embedding:** the benchmark scores the encoder's posterior mean, written by the added `mober-extract-latents` command, rather than a decoded expression profile. Encoding to the mean rather than sampling makes it deterministic for a fixed model and input.
- **Training:** the benchmark trains on log1p(TPM) of the gene panel with a 10% validation split, patience 100 and at most 3000 epochs; settings live in [the benchmark configuration](../../benchmark-workflow/config.yaml). Progress is shown as a progress bar.
- **Logging:** MLflow tracking and its command-line options are removed; metrics are written to files under the output directory.

These are benchmark adaptations, not a claim to reproduce every published MOBER result.

## Reference

If you use MOBER, cite the [upstream manuscript](https://doi.org/10.1101/2022.09.07.506964):

```bibtex
@article{Dimitrieva2022.09.07.506964,
  author = {Dimitrieva, Slavica and Janssens, Rens and Li, Gang and Szalata, Artur and Gopal, Raja and Parmar, Chintan and Kauffmann, Audrey and Durand, Eric Y.},
  title = {Biologically relevant integration of transcriptomics profiles from cancer cell lines, patient-derived xenografts and clinical tumors using deep learning},
  year = {2022},
  doi = {10.1101/2022.09.07.506964},
  journal = {bioRxiv}
}
```
