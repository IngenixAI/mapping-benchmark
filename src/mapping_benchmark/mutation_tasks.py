"""Validated, disease-independent specification of biomarker retrieval benchmarks."""

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from mapping_benchmark import alterations

Dataset = Literal["tcga", "depmap", "metaprism"]
Tier = Literal["primary", "exploratory", "sensitivity"]
Stratum = Literal["all", "MSS", "MSI"]


class Eligibility(BaseModel, extra="forbid"):
    """The fewest mutant and wild-type samples a pair needs on each side to be reported."""

    mutant_queries: int = Field(gt=0)
    wildtype_queries: int = Field(gt=0)
    mutant_pool: int = Field(gt=0)
    wildtype_pool: int = Field(gt=0)


class Biomarker(BaseModel, extra="forbid"):
    gene: str
    ensembl: str = Field(pattern=r"^ENSG\d+$")
    classes: list[alterations.Class] = Field(min_length=1)
    hotspots: alterations.Hotspots | None = None
    controls: Literal["protein_altering_free", "alteration_negative", "mutation_free"] = (
        "protein_altering_free"
    )
    exclude_deletions: bool = True
    exclude_protein_pattern: str | None = None
    note: str = ""

    @model_validator(mode="after")
    def validate_rule(self):
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("duplicate alteration classes")
        if "hotspot" in self.classes and (
            self.hotspots is None or not any(self.hotspots.model_dump().values())
        ):
            raise ValueError("hotspot biomarkers require nonempty hotspots")
        if self.exclude_protein_pattern is not None:
            re.compile(self.exclude_protein_pattern)
        if self.hotspots is not None:
            if any(re.fullmatch(r"[A-Z][1-9]\d*", c) is None for c in self.hotspots.codons):
                raise ValueError("hotspot codons must be a residue and positive position, e.g. G12")
            if any(alterations.parse(c).kind == "other" for c in self.hotspots.changes):
                raise ValueError("hotspot changes must be normalized protein changes, e.g. V600E")
            if any(
                start <= 0 or end < start
                for start, end in self.hotspots.deletions + self.hotspots.insertions
            ):
                raise ValueError("hotspot ranges must be positive and ascending")
        return self


class Analysis(BaseModel, extra="forbid"):
    biomarker: str
    stratum: Stratum = "all"
    tier: Tier
    # Only TCGA has the unfiltered MANTIS annotation; the pool still uses strict MSI.
    query_mantis_only: bool = False

    @model_validator(mode="after")
    def validate_msi(self):
        if self.biomarker == "MSI" and self.stratum != "all":
            raise ValueError("MSI cannot be scored within a single MSI stratum")
        if self.query_mantis_only and (self.biomarker != "MSI" or self.tier != "sensitivity"):
            raise ValueError("MANTIS-only calls are restricted to MSI sensitivity analyses")
        return self


class Disease(BaseModel, extra="forbid"):
    oncotree: list[str] = Field(min_length=1)
    analyses: dict[str, Analysis] = Field(min_length=1)
    exclude_msi_positive: bool = False


class Pair(BaseModel, extra="forbid"):
    query: Dataset
    database: Dataset

    @model_validator(mode="after")
    def distinct_sides(self):
        if self.query == self.database:
            raise ValueError("query and database datasets must differ")
        return self


class MSI(BaseModel, extra="forbid"):
    mantis_threshold: float = Field(default=0.4, ge=0, allow_inf_nan=False)
    sensor_threshold: float = Field(default=3.5, ge=0, allow_inf_nan=False)


class Tasks(BaseModel, extra="forbid"):
    schema_version: Literal[1] = 1
    k: list[int] = Field(default=[1, 5, 10], min_length=1)
    primary_k: int = Field(default=10, gt=0)
    bootstrap: int = Field(default=1000, ge=0)
    permutations: int = Field(default=1000, ge=0)
    seed: int = Field(default=0, ge=0)
    eligibility: Eligibility
    msi: MSI = Field(default_factory=MSI)
    depmap_oncotree: dict[str, str | None] = Field(default_factory=dict)
    biomarkers: dict[str, Biomarker] = Field(min_length=1)
    diseases: dict[str, Disease] = Field(min_length=1)
    pairs: dict[str, Pair] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_tasks(self):
        if any(k <= 0 for k in self.k) or len(set(self.k)) != len(self.k):
            raise ValueError("k must contain distinct positive depths")
        if self.primary_k not in self.k:
            raise ValueError("primary_k must appear in k")

        # Biomarkers may share a gene, but must agree on its source identifier.
        genes = {}
        for name, biomarker in self.biomarkers.items():
            if name in {
                "MSI",
                "MSI_MANTIS",
                "MSI_DEPMAP",
                "sample_id",
                "dataset",
                "patient_id",
                "mutation_sample_id",
                "unit_id",
                "oncotree_code",
                "has_variant_calls",
                "deleted",
                "representative",
                "msi_conflict",
                "msi_source",
                "msi_status",
                "msi_score_mantis",
                "msi_score_sensor",
                "msi_positive_screen",
            }:
                raise ValueError(f"reserved biomarker name: {name}")
            if biomarker.gene in genes and genes[biomarker.gene] != biomarker.ensembl:
                raise ValueError(f"conflicting Ensembl ids for {biomarker.gene}")
            genes[biomarker.gene] = biomarker.ensembl

        # Every analysis must refer to a declared biomarker or the built-in MSI label.
        for disease in self.diseases.values():
            for analysis in disease.analyses.values():
                if analysis.biomarker != "MSI" and analysis.biomarker not in self.biomarkers:
                    raise ValueError(f"undefined biomarker: {analysis.biomarker}")

        # IDs become output keys and command-line selections; forbid ambiguous names.
        for names in (
            self.biomarkers,
            self.diseases,
            self.pairs,
            *(d.analyses for d in self.diseases.values()),
        ):
            if any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", n) is None for n in names):
                raise ValueError("IDs must use letters, digits, underscore, dot or hyphen")
        return self

    @property
    def genes(self) -> dict[str, str]:
        """Source readers need the union of biomarker genes, independent of disease."""
        return {biomarker.gene: biomarker.ensembl for biomarker in self.biomarkers.values()}


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys rather than silently replacing a biomarker or disease."""

    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate YAML mapping key")
        return super().construct_mapping(node, deep=deep)


def load_tasks(path: str | Path) -> Tasks:
    return Tasks.model_validate(yaml.load(Path(path).read_text(), Loader=UniqueKeyLoader))
