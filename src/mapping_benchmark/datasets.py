"""The benchmark's inputs: one HuggingFace dataset, exported into a common layout.

``resources/datasets.yaml`` names the repository, the ontology tables and the expression
files. This module downloads them and writes:

* one expression h5ad with every dataset, on the shared gene panel, whose ``obs`` has the
  columns ``sample_id``, ``dataset``, ``mondo_id``, ``mondo_name`` and ``donor_id`` (empty
  where a dataset has none);
* the MONDO graph as a parquet with the columns ``id``, ``name``, ``parents``.

The mutation axis does not read this repository: its labels are derived from raw tables
by ``mapping_benchmark.alterations``.

One thing is not in the spec because it is derived from the source's own columns: how
the paired human/PDX study splits into patients' tumours and xenografts. See
:data:`DERIVED`.
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download
from loguru import logger
from pydantic import BaseModel, ConfigDict

OBS_COLUMNS = ["sample_id", "dataset", "mondo_id", "mondo_name", "donor_id"]


class OntologySpec(BaseModel, extra="forbid"):
    """The MONDO table: its columns, and the obs columns holding each sample's term."""

    file: str
    id: str
    name: str
    parents: str
    sample_id: str
    sample_name: str


class DatasetSpec(BaseModel, extra="forbid"):
    expression: str
    keep: dict[str, str] = {}
    exclude: list[str] = []


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hf_repo: str
    hf_revision: str
    ontology: OntologySpec
    datasets: dict[str, DatasetSpec]

    def download(self, file: str) -> Path:
        return Path(hf_hub_download(self.hf_repo, file, repo_type="dataset",
                                    revision=self.hf_revision))


def load_spec(path: str | Path) -> Spec:
    with open(path) as f:
        return Spec.model_validate(yaml.safe_load(f))


# --- Derived obs columns ----------------------------------------------------------------

def _gse317901(obs: pd.DataFrame) -> dict:
    # Patient and sample type live in one text field: "Patient hnc0002, human sample" or
    # "Patient hnc0002, pdx sample, biological replicate a". A PDX whose patient has no
    # human sample can never be retrieved by its own donor, so it is typed apart and left
    # out of the database.
    patient = obs["tissue"].str.extract(r"Patient (hnc\d+)", expand=False)
    human = obs["tissue"].str.contains("human sample")
    paired = patient.isin(patient[human])
    sample_type = np.where(human, "human", np.where(paired, "pdx", "pdx_unpaired"))
    return {"donor_id": patient, "sample_type": sample_type}


DERIVED = {"gse317901_tpm.h5ad": _gse317901}


# --- Export -----------------------------------------------------------------------------

def standardise(adata: ad.AnnData, name: str, spec: DatasetSpec,
                ontology: OntologySpec) -> ad.AnnData:
    """Filter one source h5ad and rewrite its obs into the common columns."""
    obs = adata.obs.copy()
    for column, values in DERIVED.get(Path(spec.expression).name, lambda _: {})(obs).items():
        obs[column] = values
    if "donor_id" not in obs:
        obs["donor_id"] = ""

    keep = np.ones(adata.n_obs, dtype=bool)
    for column, value in spec.keep.items():
        keep &= (obs[column].astype(str) == value).to_numpy()
    keep &= ~obs.index.isin(spec.exclude)

    out = pd.DataFrame({
        "sample_id": obs.index.astype(str),
        "dataset": name,
        "mondo_id": obs[ontology.sample_id].astype(str),
        "mondo_name": obs[ontology.sample_name].astype(str),
        "donor_id": obs["donor_id"].astype(str),
    }, index=obs.index)[OBS_COLUMNS]

    adata = adata[keep].copy()
    adata.obs = out[keep].replace("nan", "")
    logger.info(f"  {name}: {adata.n_obs} of {len(keep)} samples kept")
    return adata


def export_expression(spec: Spec, output: Path) -> ad.AnnData:
    adatas = []
    for name, dataset in spec.datasets.items():
        logger.info(f"Loading {name} from {dataset.expression}")
        adata = ad.read_h5ad(spec.download(dataset.expression))
        adatas.append(standardise(adata, name, dataset, spec.ontology))
    combined = ad.concat(adatas, join="inner")
    combined.obs_names_make_unique()
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(output)
    logger.info(f"Wrote {combined.n_obs} samples x {combined.n_vars} genes to {output}")
    return combined


def export_ontology(spec: Spec, output: Path) -> None:
    """Rewrite the MONDO table into ``id``, ``name`` and ``parents`` (a JSON list)."""
    onto = spec.ontology
    df = pd.read_parquet(spec.download(onto.file)).drop_duplicates(onto.id)
    parents = df[onto.parents].map(_parent_list)
    out = pd.DataFrame({"id": df[onto.id].astype(str), "name": df[onto.name].astype(str),
                        "parents": parents})
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    logger.info(f"Wrote {len(out)} MONDO terms to {output}")


def _parent_list(value) -> str:
    """The parents cell as a JSON list, whether the table stores a JSON string or an array."""
    if isinstance(value, str):
        return value
    return json.dumps([] if value is None else [str(v) for v in value])
