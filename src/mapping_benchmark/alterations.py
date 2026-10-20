"""Shared mutation rules for TCGA, DepMap and the full published MetaPRISM MAF.

Source annotations are translated to common consequence terms. ``lof`` means a
truncating, essential splice-site, start-loss or stop-loss call (VEP's HIGH-impact coding
terms); ``nonsilent`` also includes missense and in-frame changes. Neither rule proves
loss of gene activity.
``hotspot`` matches the configured protein changes. Missing annotations needed by a
rule give unknown status. A known positive can still establish a sample's label.

DepMap phase sets with multiple frameshifts whose net length change is a multiple
of three are uncertain; their combined protein effect is not reconstructed.
"""

import base64
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

Class = Literal["hotspot", "nonsilent", "lof"]
CALL_COLUMNS = ["sample_id", "gene_id_ensembl", "protein_change", "consequence", "uncertain"]

LOF_TERMS = {"stop_gained", "frameshift_variant", "splice_site", "splice_acceptor_variant",
             "splice_donor_variant", "start_lost", "stop_lost"}
INFRAME_TERMS = {"missense_variant", "inframe_deletion", "inframe_insertion"}
NONSILENT_TERMS = LOF_TERMS | INFRAME_TERMS | {"protein_altering_variant"}
# Consequences that alone establish a negative for these sequence-based rules.
OTHER_TERMS = {"synonymous_variant", "splice_region_variant", "intron_variant",
               "3_prime_UTR_variant", "5_prime_UTR_variant", "upstream_gene_variant",
               "downstream_gene_variant", "non_coding_transcript_variant",
               "non_coding_transcript_exon_variant", "intergenic_variant",
               "stop_retained_variant", "start_retained_variant"}
MAF_TERMS = {
    "Missense_Mutation": "missense_variant", "Nonsense_Mutation": "stop_gained",
    "Frame_Shift_Del": "frameshift_variant", "Frame_Shift_Ins": "frameshift_variant",
    "Splice_Site": "splice_site", "Translation_Start_Site": "start_lost",
    "In_Frame_Del": "inframe_deletion", "In_Frame_Ins": "inframe_insertion",
    "Nonstop_Mutation": "stop_lost", "Silent": "synonymous_variant",
    "Splice_Region": "splice_region_variant", "Intron": "intron_variant",
    "3'UTR": "3_prime_UTR_variant", "5'UTR": "5_prime_UTR_variant",
    "3'Flank": "downstream_gene_variant", "5'Flank": "upstream_gene_variant",
    "RNA": "non_coding_transcript_variant", "IGR": "intergenic_variant",
    "lincRNA": "non_coding_transcript_variant",
}

class Hotspots(BaseModel, extra="forbid"):
    """The selected, functionally relevant protein changes of one gene.

    ``codons`` takes any missense at that residue (``G12``), ``changes`` exactly that change
    (``L858R``); ``deletions`` and ``insertions`` take an in-frame deletion, or an in-frame
    insertion or duplication, whose first residue lies in one of the inclusive codon ranges.
    """

    codons: list[str] = []
    changes: list[str] = []
    deletions: list[tuple[int, int]] = []
    insertions: list[tuple[int, int]] = []


# --- Protein changes ----------------------------------------------------------------------

@dataclass(frozen=True)
class Change:
    """One parsed protein change. ``kind`` is sub, del, ins, delins, dup or other."""

    kind: str
    residue: str = ""
    start: int = 0
    end: int = 0
    inserted: str = ""


_SPAN = r"([A-Z])(\d+)(?:_([A-Z])(\d+))?"
_PATTERNS = [
    ("sub", re.compile(r"^([A-Z])(\d+)([A-Z*=])$")),
    ("delins", re.compile(rf"^{_SPAN}delins([A-Z*]+)$")),
    ("del", re.compile(rf"^{_SPAN}del$")),
    ("ins", re.compile(rf"^{_SPAN}ins([A-Z*]+)$")),
    ("dup", re.compile(rf"^{_SPAN}dup$")),
]


def normalise_protein_change(value) -> str:
    """``p.R213Ter`` -> ``R213*``; empty for a missing value."""
    if not isinstance(value, str):
        return ""
    return value.strip().removeprefix("p.").replace("Ter", "*")


def parse(change: str) -> Change:
    for kind, pattern in _PATTERNS:
        m = pattern.match(change)
        if not m:
            continue
        residue, start = m.group(1), int(m.group(2))
        if kind == "sub":
            return Change("sub", residue, start, start, m.group(3))
        end = int(m.group(4)) if m.group(4) else start
        inserted = m.group(5) if kind in ("delins", "ins") else ""
        return Change(kind, residue, start, end, inserted)
    return Change("other")


def is_hotspot(change: str, rule: Hotspots) -> bool:
    if change in rule.changes:
        return True
    c = parse(change)
    if c.kind == "sub" and c.inserted not in ("=", "*", c.residue):
        if f"{c.residue}{c.start}" in rule.codons:
            return True
    deleted = c.end - c.start + 1
    if any(lo <= c.start <= hi for lo, hi in rule.deletions):
        if c.kind == "del" or (c.kind == "delins" and len(c.inserted) < deleted):
            return True
    if any(lo <= c.start <= hi for lo, hi in rule.insertions):
        if c.kind in ("ins", "dup") or (c.kind == "delins" and len(c.inserted) > deleted):
            return True
    return False


# --- Sources ------------------------------------------------------------------------------

def from_maf(table: pd.DataFrame) -> pd.DataFrame:
    """Translate MAF classes into the consequence terms used by the shared rules."""
    return pd.DataFrame({
        "sample_id": table["sample_id"].astype(str),
        "gene_id_ensembl": table["gene_id_ensembl"].astype(str),
        "protein_change": table["aa_change"].map(normalise_protein_change),
        "consequence": table["variant_classification"].map(MAF_TERMS).fillna(""),
        "uncertain": False,
    })[CALL_COLUMNS].reset_index(drop=True)


def ambiguous_frameshifts(table: pd.DataFrame) -> np.ndarray:
    """Flag phase sets that could restore the reading frame; do not infer a protein change.

    Mark every call of the gene in that phase set uncertain. A separate known positive
    call can still establish a positive label.
    """
    key = ["Consensus_model_id", "gene_id_ensembl", "ps", "gt"]
    frameshifts = table[table["variant_info"].fillna("").str.contains("frameshift_variant")
                       & table["ps"].notna()].copy()
    frameshifts["shift"] = frameshifts["alt"].str.len() - frameshifts["ref"].str.len()
    groups = frameshifts.groupby(key)["shift"].agg(["size", "sum"])
    ambiguous = groups.index[(groups["size"] > 1) & (groups["sum"] % 3 == 0)]
    return pd.MultiIndex.from_frame(table[key]).isin(ambiguous)


def from_depmap(table: pd.DataFrame) -> pd.DataFrame:
    """Calls from DepMap's ``somatic_mutations_raw``, whose consequence is a VEP term list."""
    return pd.DataFrame({
        "sample_id": table["Consensus_model_id"].astype(str),
        "gene_id_ensembl": table["gene_id_ensembl"].astype(str),
        "protein_change": table["protein_change"].map(normalise_protein_change),
        "consequence": table["variant_info"].fillna("").astype(str),
        "uncertain": ambiguous_frameshifts(table),
    })[CALL_COLUMNS].reset_index(drop=True)


# --- Labels -------------------------------------------------------------------------------

def status(calls: pd.DataFrame, cls: Class, hotspots: Hotspots | None = None) -> pd.Series:
    """Per-call True, False or NA, using only the annotations needed by ``cls``."""
    if cls not in ("hotspot", "nonsilent", "lof"):
        raise ValueError(f"unknown alteration class: {cls}")
    if cls == "hotspot" and hotspots is None:
        raise ValueError("hotspot status requires a hotspot definition")

    def classify(change, consequence, uncertain):
        if uncertain:
            return pd.NA
        terms = set(consequence.split("&"))
        if cls == "hotspot":
            if is_hotspot(change, hotspots):
                return True
            if parse(change).kind != "other":
                return False
            # Missing/unreadable missense or in-frame HGVS could hide a hotspot.
            if terms & (INFRAME_TERMS | {"protein_altering_variant"}):
                return pd.NA
            return False if terms & (LOF_TERMS | OTHER_TERMS) else pd.NA
        positive = LOF_TERMS if cls == "lof" else NONSILENT_TERMS
        if terms & positive:
            return True
        # A generic coding/protein-altering annotation cannot exclude truncation.
        if terms & {"coding_sequence_variant", "protein_altering_variant"}:
            return pd.NA
        return False if terms & (INFRAME_TERMS | OTHER_TERMS) else pd.NA

    return pd.Series([classify(c, t, u) for c, t, u in
                      calls[["protein_change", "consequence", "uncertain"]].itertuples(
                          index=False, name=None)], index=calls.index, dtype="boolean")


def carries(calls: pd.DataFrame, cls: Class, hotspots: Hotspots | None = None) -> np.ndarray:
    """The known positive calls; use :func:`status` to retain unknown annotations."""
    return status(calls, cls, hotspots).fillna(False).to_numpy(bool)

def column(gene: str, cls: Class) -> str:
    """The label column of one alteration: ``KRAS_hotspot``."""
    return f"{gene}_{cls}"


# --- Raw inputs ---------------------------------------------------------------------------

MAF_FIELDS = {"Tumor_Sample_Barcode": "sample_id", "Gene": "gene_id_ensembl",
              "Variant_Classification": "variant_classification", "HGVSp_Short": "aa_change"}


class PublishedMAF(BaseModel, extra="forbid"):
    """The original study's full somatic MAF, pinned by SHA-256 and cached locally."""

    url: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    http_user: str | None = None  # public Nextcloud shares use the literal user 'anonymous'

    def download(self, cache: Path = Path("data/raw/mutation_sources")) -> Path:
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / f"{self.sha256}.maf.gz"
        if target.exists():
            contents = target.read_bytes()
        else:
            headers = {}
            if self.http_user is not None:
                credentials = base64.b64encode(f"{self.http_user}:".encode()).decode()
                headers["Authorization"] = f"Basic {credentials}"
            with urlopen(Request(self.url, headers=headers), timeout=30) as response:
                contents = response.read()
        if hashlib.sha256(contents).hexdigest() != self.sha256:
            raise ValueError(f"checksum mismatch for published MAF: {target}")
        if not target.exists():
            target.write_bytes(contents)
        return target

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        table = pd.read_csv(self.download(), sep="\t", comment="#", dtype=str,
                            usecols=list(MAF_FIELDS))
        table = table.rename(columns=MAF_FIELDS)
        table["gene_id_ensembl"] = table["gene_id_ensembl"].str.split(".").str[0]
        return table[list(MAF_FIELDS.values()) if columns is None else columns]


class Table(BaseModel, extra="forbid"):
    """One table of a HuggingFace dataset at a pinned revision, in one or more shards."""

    repo: str
    revision: str
    files: list[str]

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.concat([
            pd.read_parquet(hf_hub_download(self.repo, f, repo_type="dataset",
                                            revision=self.revision), columns=columns)
            for f in self.files
        ], ignore_index=True)


class Sources(BaseModel, extra="forbid"):
    """``resources/mutation_sources.yaml``: every raw table the labels are built from."""

    tcga_mutations: Table
    tcga_clinical: Table
    depmap_mutations: Table
    depmap_subtypes: Table
    metaprism_mutations: PublishedMAF
    metaprism_clinical: Table
    oncotree: Table
    depmap_copy_number: Table
    depmap_features: Table
    cell_lines: Table


def load_sources(path: str | Path) -> Sources:
    with open(path) as f:
        return Sources.model_validate(yaml.safe_load(f))
