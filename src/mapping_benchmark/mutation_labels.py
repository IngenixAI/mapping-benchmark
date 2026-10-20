"""Build source-grounded mutation and MSI labels for every configured cancer.

Readers identify samples with variant-call records before selecting task genes.
This establishes call-set availability, not sequencing depth or gene callability.
"""

import json
from typing import Literal

import numpy as np
import pandas as pd

from mapping_benchmark import alterations
from mapping_benchmark.mutation_tasks import Tasks

# Sample inventory and label construction


def validate_sample_ids(samples: pd.DataFrame) -> None:
    """Check the identifiers shared by expression and embedding sample tables."""
    for column in ("sample_id", "dataset"):
        if column not in samples or samples[column].isna().any() or samples[column].eq("").any():
            raise ValueError(f"sample table requires nonempty {column}")
    if samples.sample_id.duplicated().any():
        raise ValueError("duplicate sample_id in sample table")


def build_labels(
    expression_samples: pd.DataFrame, sources: alterations.Sources, tasks: Tasks
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All expression rows, selected variant evidence, and explicit MSI unknowns.

    Representatives are chosen by ID before consulting any biomarker label. TCGA aliquots
    share a sequenced tumour, DepMap rows share a model, MetaPRISM rows share a patient.
    Deduplication is across each dataset; a sequenced unit can never enter a pool twice.
    """
    validate_sample_ids(expression_samples)
    datasets = sorted({d for pair in tasks.pairs.values() for d in (pair.query, pair.database)})
    codes = {c for disease in tasks.diseases.values() for c in disease.oncotree}
    frames, evidence = [], []

    for dataset in datasets:
        # Readers expect expression IDs in both the series values and its index.
        ids = expression_samples.loc[expression_samples.dataset.eq(dataset), "sample_id"]
        if ids.empty:
            raise ValueError(f"expression_samples has no samples for configured dataset {dataset}")
        ids = pd.Series(ids.to_numpy(), index=ids.to_numpy(), dtype=str)
        # Match expression IDs to the source and restrict to the configured diseases.
        samples, calls = READERS[dataset](ids, sources, tasks)
        samples = samples[samples.oncotree_code.isin(codes)].copy()
        samples["dataset"] = dataset
        samples["unit_id"] = samples.mutation_sample_id if dataset == "tcga" else samples.patient_id

        # MSI evidence is independent of whether mutation-call records are available.
        samples = add_msi_labels(samples, dataset, sources, tasks)

        # Apply the same declared biomarkers to each dataset.
        samples, selected = label_biomarkers(samples, calls, tasks)
        evidence.append(selected.assign(dataset=dataset))

        # Choose one representative by ID; labels never influence this choice.
        samples = samples.sort_index(kind="stable")
        valid_unit = samples.unit_id.notna() & samples.unit_id.ne("")
        samples["representative"] = valid_unit & ~samples.unit_id.duplicated()
        frames.append(samples)

    labels = pd.concat(frames)
    labels.index.name = "sample_id"
    validate_labels(labels, tasks)
    return labels, pd.concat(evidence, ignore_index=True)[
        [*alterations.CALL_COLUMNS, "dataset", "biomarker", "gene"]
    ]


def validate_labels(labels: pd.DataFrame, tasks: Tasks) -> None:
    required = {
        "dataset",
        "patient_id",
        "unit_id",
        "oncotree_code",
        "has_variant_calls",
        "representative",
        "MSI",
        "MSI_MANTIS",
        *tasks.biomarkers,
    }
    if missing := required - set(labels):
        raise ValueError(f"labels lack columns: {sorted(missing)}")

    # Screen presence is boolean; biological MSI status may still be unknown.
    if any(d.exclude_msi_positive for d in tasks.diseases.values()):
        if "msi_positive_screen" not in labels:
            raise ValueError("msi_positive_screen is required for positive-screen exclusion")
        if labels.msi_positive_screen.isna().any() or not pd.api.types.is_bool_dtype(
            labels.msi_positive_screen
        ):
            raise ValueError("msi_positive_screen must be boolean")

    if labels.index.has_duplicates or labels.index.isna().any() or labels.index.isin([""]).any():
        raise ValueError("labels require unique nonempty sample IDs")
    for column in ("has_variant_calls", "representative"):
        if labels[column].isna().any() or not pd.api.types.is_bool_dtype(labels[column]):
            raise ValueError(f"{column} must be boolean")

    # Binary labels allow an explicit unknown value.
    for column in ("MSI", "MSI_MANTIS", *tasks.biomarkers):
        if not labels[column].dropna().isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0, 1, or unknown")
    if (labels[list(tasks.biomarkers)].notna().any(axis=1) & ~labels.has_variant_calls).any():
        raise ValueError("samples without variant-call records cannot have mutation labels")

    # Replicates may remain in the table, but only one row represents each unit.
    reps = labels[labels.representative]
    for column in ("unit_id", "patient_id", "dataset"):
        if reps[column].isna().any() or reps[column].eq("").any():
            raise ValueError(f"representatives require nonempty {column}")
    if reps.duplicated(["dataset", "unit_id"]).any():
        raise ValueError("duplicate representative sequencing units")


# Mutation labels and positive variant evidence


def label_biomarkers(
    samples: pd.DataFrame, calls: pd.DataFrame, tasks: Tasks
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply declared biomarkers to source samples and retain the positive variant evidence."""
    samples = samples.copy()
    evidence = []
    for name, biomarker in tasks.biomarkers.items():
        samples[name], positive = mutation_label(
            samples,
            calls,
            biomarker.ensembl,
            biomarker.gene,
            biomarker.classes,
            biomarker.hotspots,
            biomarker.exclude_protein_pattern,
            controls=biomarker.controls,
            exclude_deletions=biomarker.exclude_deletions,
        )
        positive = positive[positive.sample_id.isin(samples.mutation_sample_id)]
        evidence.append(positive.assign(biomarker=name, gene=biomarker.gene))
    return samples, pd.concat(evidence, ignore_index=True)


def mutation_label(
    samples: pd.DataFrame,
    calls: pd.DataFrame,
    ensembl: str,
    gene: str,
    classes: list[str],
    hotspots: alterations.Hotspots | None = None,
    exclude_protein_pattern: str | None = None,
    *,
    controls: Literal[
        "protein_altering_free", "alteration_negative", "mutation_free"
    ] = "protein_altering_free",
    exclude_deletions: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    """Selected variants versus controls under one explicit, disease-independent policy.

    Other coding variants, unresolved annotations and known homozygous deletions make
    noncarriers unknown. Synonymous-only calls can establish a noncarrier control.
    A separate known selected variant still establishes a positive label.
    """
    rows = calls[calls["gene_id_ensembl"].eq(ensembl)].copy()
    selected = pd.Series(False, index=rows.index, dtype="boolean")
    for cls in classes:
        selected |= alterations.status(rows, cls, hotspots)
    excluded = pd.Series(False, index=rows.index)
    if exclude_protein_pattern:
        excluded = rows["protein_change"].str.contains(
            exclude_protein_pattern, regex=True, na=False
        )
        selected &= ~excluded

    # A known selected variant takes precedence over other calls in this gene.
    evidence = rows[selected.fillna(False)].copy()
    hit = samples["mutation_sample_id"].isin(evidence["sample_id"])

    # Decide which other calls disqualify a sample from the control group.
    if controls == "mutation_free":
        noncontrol = pd.Series(True, index=rows.index)
    elif controls == "protein_altering_free":
        noncontrol = alterations.status(rows, "nonsilent").fillna(True)
    elif controls == "alteration_negative":
        noncontrol = selected.isna()
    else:
        raise ValueError(f"unknown control policy: {controls}")

    noncontrol |= excluded
    other = samples["mutation_sample_id"].isin(rows.loc[noncontrol, "sample_id"])
    deleted = samples["deleted"].fillna("").str.split(";").map(lambda g: gene in g)

    # Missing call records and ambiguous noncarriers remain unknown, never zero.
    label = hit.astype("Int64")
    label = label.mask(
        ~samples["has_variant_calls"] | (~hit & (other | (deleted & exclude_deletions)))
    )
    return label, evidence.drop_duplicates()


# MSI annotations


def add_msi_labels(
    samples: pd.DataFrame, dataset: str, sources: alterations.Sources, tasks: Tasks
) -> pd.DataFrame:
    """Join source MSI annotations, preserving missing and conflicting calls."""
    if dataset == "tcga":
        clinical = sources.tcga_clinical.read(
            [
                "sample_id",
                "msi_score_mantis",
                "msi_score_sensor",
            ]
        ).set_index("sample_id")
        annotations = clinical.reindex(samples.mutation_sample_id).set_axis(samples.index)
        samples = samples.join(annotations)
        samples = samples.join(
            tcga_msi(samples.msi_score_mantis, samples.msi_score_sensor, **tasks.msi.model_dump())
        )
        samples["msi_source"] = "MANTIS; available MSIsensor cross-check"

    elif dataset == "depmap":
        cell_lines = sources.cell_lines.read(["cvcl_id", "msi_status"]).set_index("cvcl_id")
        annotations = cell_lines.reindex(samples.patient_id).set_axis(samples.index)
        samples = samples.join(annotations)
        samples = samples.join(depmap_msi(samples.msi_positive_screen, samples.msi_status))
        samples["MSI_MANTIS"] = pd.Series(pd.NA, index=samples.index, dtype="Int64")
        samples["msi_source"] = "DepMap MSI tag; Cellosaurus explicit MSI/MSS"

    else:
        # MetaPRISM has no MSI annotation in these inputs. A false screen flag only
        # means no recorded positive screen; it cannot establish an MSS label.
        for column in ("MSI", "MSI_MANTIS"):
            samples[column] = pd.Series(pd.NA, index=samples.index, dtype="Int64")
        samples["msi_conflict"] = False
        samples["msi_source"] = "unavailable in pinned source"

    return samples


def tcga_msi(
    mantis: pd.Series,
    sensor: pd.Series,
    mantis_threshold: float = 0.4,
    sensor_threshold: float = 3.5,
) -> pd.DataFrame:
    """MANTIS primary calls, withholding disagreements with available MSIsensor calls.

    MSIsensor is a cross-check, not a fallback when MANTIS is missing. Both use the
    original tool thresholds. Agreement between algorithms is not independent clinical
    validation. The unfiltered MANTIS call is retained for sensitivity analysis.
    """
    mantis = pd.to_numeric(mantis, errors="coerce")
    sensor = pd.to_numeric(sensor, errors="coerce")
    # MANTIS is a distance score, not a probability; TCGA values can exceed one.
    mantis = mantis.where(mantis.ge(0) & mantis.lt(float("inf")))
    sensor = sensor.where(sensor.ge(0) & sensor.lt(float("inf")))
    m = mantis.ge(mantis_threshold).astype("Int64").where(mantis.notna())
    s = sensor.ge(sensor_threshold).astype("Int64").where(sensor.notna())
    conflict = (m.notna() & s.notna() & m.ne(s)).fillna(False)
    return pd.DataFrame({"MSI": m.mask(conflict), "MSI_MANTIS": m, "msi_conflict": conflict})


def depmap_msi(tag: pd.Series, cellosaurus: pd.Series) -> pd.DataFrame:
    """Positive DepMap tags plus explicit Cellosaurus MSI/MSS, excluding conflicts.

    Absence of a positive-only subtype tag cannot establish MSS. MSI-low and mixed
    Cellosaurus descriptions are withheld because they do not establish an MSI-high
    versus MSS comparison.
    """
    text = cellosaurus.fillna("")
    stable = text.str.startswith("Stable (MSS)")
    unstable = text.str.startswith(("Instable (MSI)", "Instable (MSI-high)"))
    mixed = text.str.contains("MSI-low|MSS.*Instable|MSI.*Stable", regex=True)
    conflict = (tag & stable) | mixed
    label = pd.Series(pd.NA, index=tag.index, dtype="Int64")
    label.loc[tag | unstable] = 1
    label.loc[stable & ~tag] = 0
    label = label.mask(conflict)
    return pd.DataFrame(
        {"MSI": label, "MSI_DEPMAP": tag.astype("Int64").where(tag), "msi_conflict": conflict}
    )


# Source readers: full call-set availability is checked before gene selection.

MAF_COLUMNS = ["sample_id", "gene_id_ensembl", "variant_classification", "aa_change"]
DEPMAP_COLUMNS = [
    "Consensus_model_id",
    "is_default_entry_for_model",
    "gene_id_ensembl",
    "protein_change",
    "variant_info",
    "ref",
    "alt",
    "gt",
    "ps",
]
# Relative copy number below which a line has lost the gene on both alleles. DepMap's values
# for STK11 and KEAP1 sit below 0.08 or above 0.36, so the exact cut barely matters.
DELETED = 0.25


def tcga(ids: pd.Series, sources: alterations.Sources, tasks: Tasks):
    table = sources.tcga_mutations.read(MAF_COLUMNS)
    clinical = sources.tcga_clinical.read(
        ["sample_id", "cancer_type_oncotree", "msi_score_mantis"]
    ).set_index("sample_id")

    # TCGA expression aliquots share a 15-character sequenced-tumour barcode.
    key = ids.str[:15]
    samples = pd.DataFrame(
        {
            "patient_id": ids.str[:12],
            "mutation_sample_id": key,
            "oncotree_code": clinical["cancer_type_oncotree"].reindex(key).to_numpy(),
            "has_variant_calls": key.isin(set(table["sample_id"])),
            "msi_positive_screen": (
                clinical["msi_score_mantis"].reindex(key) >= tasks.msi.mantis_threshold
            ).to_numpy(),
            "deleted": "",
        },
        index=ids.index,
    )
    return samples, alterations.from_maf(_genes(table, tasks))


def depmap(ids: pd.Series, sources: alterations.Sources, tasks: Tasks):
    table = sources.depmap_mutations.read(DEPMAP_COLUMNS)
    table = table[table["is_default_entry_for_model"]]

    # Select the deepest disease code, then apply declared source corrections.
    subtypes = sources.depmap_subtypes.read()
    levels = sources.oncotree.read(["oncotree_code", "oncotree_level"])
    coded = subtypes.merge(levels, left_on="Subtype", right_on="oncotree_code")
    deepest = (
        coded.sort_values(["oncotree_level", "oncotree_code"], kind="stable")
        .groupby("Consensus_model_id")["oncotree_code"]
        .last()
    )
    oncotree = deepest.reindex(ids).set_axis(ids.index)
    overridden = ids.isin(tasks.depmap_oncotree)
    oncotree[overridden] = ids[overridden].map(tasks.depmap_oncotree)

    msi = set(subtypes.loc[subtypes["Subtype"] == "MSI", "Consensus_model_id"])
    samples = pd.DataFrame(
        {
            "patient_id": ids,
            "mutation_sample_id": ids,
            "oncotree_code": oncotree,
            "has_variant_calls": ids.isin(set(table["Consensus_model_id"])),
            "msi_positive_screen": ids.isin(msi),
            "deleted": _deleted(ids, sources, tasks),
        },
        index=ids.index,
    )
    return samples, alterations.from_depmap(_genes(table, tasks))


def _deleted(ids: pd.Series, sources: alterations.Sources, tasks: Tasks) -> pd.Series:
    """Per line, the task genes lost on both alleles: by DepMap's relative copy number, or
    a homozygous gene deletion in Cellosaurus (NCI-H358's TP53, which copy_number lacks)."""
    symbol = {e: g for g, e in tasks.genes.items()}
    features = sources.depmap_features.read(["copy_number_features"]).iloc[0, 0]
    at = [i for i, e in enumerate(features) if e in symbol]
    table = sources.depmap_copy_number.read()
    copies = pd.DataFrame(
        np.stack([row[at] for row in table["omics_features"]]),
        index=table["Consensus_model_id"],
        columns=[symbol[features[i]] for i in at],
    )

    lost = copies.lt(DELETED).stack()
    pairs = set(lost.index[lost])

    # Cellosaurus can record deletions absent from the copy-number matrix.
    lines = sources.cell_lines.read(["cvcl_id", "mutations"]).dropna()
    for cvcl, entries in lines[lines["cvcl_id"].isin(set(ids))].itertuples(index=False):
        pairs |= {
            (cvcl, m["gene"])
            for m in json.loads(entries)
            if m["mutation_type"] == "Gene deletion" and m["zygosity"] == "Homozygous"
        }
    return ids.map(lambda i: ";".join(g for g in sorted(tasks.genes) if (i, g) in pairs))


def metaprism(ids: pd.Series, sources: alterations.Sources, tasks: Tasks):
    table = sources.metaprism_mutations.read(MAF_COLUMNS)
    clinical = sources.metaprism_clinical.read(
        ["patient_id", "sample_id", "cancer_type_oncotree"]
    ).set_index("patient_id")

    # Link an RNA sample to the patient's sequenced DNA sample.
    patient = ids.str.extract(r"^(MP-\d+)", expand=False)
    key = clinical["sample_id"].reindex(patient)
    samples = pd.DataFrame(
        {
            "patient_id": patient,
            "mutation_sample_id": key.to_numpy(),
            "oncotree_code": clinical["cancer_type_oncotree"].reindex(patient).to_numpy(),
            "has_variant_calls": (key.notna() & key.isin(set(table["sample_id"]))).to_numpy(),
            # No positive MSI annotation is available; strict MSI stays unknown.
            "msi_positive_screen": False,
            "deleted": "",
        },
        index=ids.index,
    )
    return samples, alterations.from_maf(_genes(table, tasks))


READERS = {"tcga": tcga, "depmap": depmap, "metaprism": metaprism}


def _genes(table: pd.DataFrame, tasks: Tasks) -> pd.DataFrame:
    return table[table["gene_id_ensembl"].isin(set(tasks.genes.values()))]
