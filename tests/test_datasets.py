"""Standardising the source h5ads into the common obs columns."""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mapping_benchmark import datasets

RESOURCES = Path(__file__).parents[1] / "benchmark-workflow" / "resources"
MONDO = datasets.OntologySpec(file="m.parquet", id="mondo_id", name="name",
                              parents="direct_ancestors_ids",
                              sample_id="ontology_disease_matched_id",
                              sample_name="ontology_disease_matched_name")


def source(index, **columns) -> ad.AnnData:
    obs = pd.DataFrame(columns, index=pd.Index(index))
    obs["ontology_disease_matched_id"] = "MONDO:1"
    obs["ontology_disease_matched_name"] = "one"
    return ad.AnnData(X=np.zeros((len(index), 2), dtype=np.float32), obs=obs)


def test_the_real_spec_validates():
    spec = datasets.load_spec(RESOURCES / "datasets.yaml")
    assert set(spec.datasets) >= {"tcga", "depmap", "metaprism", "pdxe",
                                  "aphrodite_human", "aphrodite_pdx"}
    assert datasets.mutation_datasets(spec) == ["tcga", "depmap", "metaprism"]
    assert spec.mutations is not None
    assert Path(spec.datasets["aphrodite_pdx"].expression).name in datasets.DERIVED


def test_keep_and_exclude_filter_the_samples_and_obs_gets_the_common_columns():
    adata = source(["TCGA-AA-0001-01A", "TCGA-AA-0002-11A", "TCGA-AA-0003-01A"],
                   condition=["Tumor", "Normal", "Tumor"])
    spec = datasets.DatasetSpec(expression="tcga_tpm.h5ad", keep={"condition": "Tumor"},
                                exclude=["TCGA-AA-0003-01A"])
    out = datasets.standardise(adata, "tcga", spec, MONDO)
    assert out.obs["sample_id"].tolist() == ["TCGA-AA-0001-01A"]
    assert out.obs.columns.tolist() == datasets.OBS_COLUMNS
    assert out.obs["mondo_id"].tolist() == ["MONDO:1"] and out.obs["donor_id"].tolist() == [""]


def test_mutation_matrix_is_assayed_samples_by_assayed_genes():
    mutated = pd.DataFrame({"dataset": ["d", "d", "d", "other"],
                            "sample_id": ["s1", "s1", "s9", "s1"],
                            "gene_id_ensembl": ["g1", "g2", "g1", "g1"]})
    samples = pd.DataFrame({"dataset": ["d", "d", "other"], "sample_id": ["s1", "s2", "s1"]})
    genes = pd.DataFrame({"dataset": ["d", "d", "d"], "gene_id_ensembl": ["g1", "g2", "g3"]})
    matrix = datasets.mutation_matrix(mutated, samples, genes, "d")
    assert matrix.index.tolist() == ["s1", "s2"]      # s9 is not an assayed sample: ignored
    assert matrix.columns.tolist() == ["g1", "g2", "g3"]
    assert matrix.loc["s1"].tolist() == [1, 1, 0]
    assert matrix.loc["s2"].tolist() == [0, 0, 0]      # assayed, nothing mutated: all-zero row
    assert matrix.dtypes.iloc[0] == np.int8


def test_the_paired_study_splits_into_human_and_paired_pdx():
    tissue = ["Patient hnc0002, human sample",
              "Patient hnc0002, pdx sample, biological replicate a",
              "Patient hnc0002, pdx sample, biological replicate b",
              "Patient hnc0099, pdx sample, biological replicate a"]   # no human sample
    adata = source(["H2", "X2a", "X2b", "X99"], tissue=tissue)
    human = datasets.standardise(adata, "aphrodite_human", datasets.DatasetSpec(
        expression="gse317901_tpm.h5ad", keep={"sample_type": "human"}), MONDO)
    pdx = datasets.standardise(adata, "aphrodite_pdx", datasets.DatasetSpec(
        expression="gse317901_tpm.h5ad", keep={"sample_type": "pdx"}), MONDO)
    assert human.obs["donor_id"].tolist() == ["hnc0002"]
    assert pdx.obs["sample_id"].tolist() == ["X2a", "X2b"]
    assert pdx.obs["donor_id"].tolist() == ["hnc0002", "hnc0002"]


def test_export_ontology_writes_json_parents(tmp_path, monkeypatch):
    table = pd.DataFrame({"mondo_id": ["MONDO:0", "MONDO:1", "MONDO:1"],
                          "name": ["root", "leaf", "leaf"],
                          "direct_ancestors_ids": ["[]", '["MONDO:0"]', '["MONDO:0"]']})
    table.to_parquet(tmp_path / "m.parquet")
    spec = datasets.Spec(hf_repo="x", hf_revision="y", ontology=MONDO, datasets={})
    monkeypatch.setattr(datasets.Spec, "download", lambda self, file: tmp_path / file)
    datasets.export_ontology(spec, tmp_path / "out.parquet")
    out = pd.read_parquet(tmp_path / "out.parquet")
    assert out.columns.tolist() == ["id", "name", "parents"]
    assert len(out) == 2 and json.loads(out["parents"][1]) == ["MONDO:0"]


def test_spec_rejects_unknown_keys():
    with pytest.raises(ValueError):
        datasets.DatasetSpec(expression="x.h5ad", alignment="old")
