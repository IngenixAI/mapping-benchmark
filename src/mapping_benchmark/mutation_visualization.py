"""Shared display ordering and a standalone explorer of the scored mutation cases."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mapping_benchmark.mutation_tasks import Tasks

RANKING_RULE = (
    "Methods run from worst to best by mean AUROC across eligible primary "
    "cancer/pair/analysis cells at the primary depth. Each cell has equal weight. "
    "The order stays fixed across cancers, filters and metrics. "
    "Ties are alphabetical; methods without primary scores appear first as unranked."
)


def model_ranking(
    scores: pd.DataFrame, primary_k: int, methods: list[str]
) -> list[dict]:
    """Rank observed primary AUROCs, excluding bounds and other depths/tiers.

    Scores use the common evaluator's column names. Missing estimates do not become
    zeros. This is a descriptive display order, not a statistical method comparison.
    """
    methods = list(dict.fromkeys(m for m in methods if m not in {"Random", "Oracle"}))
    scores = scores.assign(auroc=pd.to_numeric(scores.auroc, errors="coerce"))
    selected = scores[
        scores.method.isin(methods)
        & scores.tier.eq("primary")
        & scores.k.eq(primary_k)
        & np.isfinite(scores.auroc)
    ]
    means = selected.groupby("method").auroc.agg(["mean", "size"])
    ranking = [
        {
            "method": method,
            "mean_auroc": float(means.loc[method, "mean"]) if method in means.index else None,
            "cells": int(means.loc[method, "size"]) if method in means.index else 0,
        }
        for method in methods
    ]
    return sorted(
        ranking,
        key=lambda r: (
            r["mean_auroc"] is not None,
            r["mean_auroc"] if r["mean_auroc"] is not None else -np.inf,
            r["method"].casefold(),
        ),
    )


def write_explorer(
    path: Path, cases: list[dict], scores: pd.DataFrame, tasks: Tasks, methods: list[str]
) -> None:
    """Embed exact counts, statistics and fixed ordering; no network or server needed."""
    ranking = model_ranking(scores, tasks.primary_k, methods)
    payload = {
        "cases": [{k: v for k, v in c.items() if k not in {"query_ids", "pool_ids"}}
                  for c in cases],
        "scores": json.loads(scores.to_json(orient="records")),
        "ranking": ranking,
        "ranking_rule": RANKING_RULE,
        "primary_k": tasks.primary_k,
        "k": tasks.k,
        "diseases": list(tasks.diseases),
        "pairs": list(tasks.pairs),
        "biomarkers": {name: b.model_dump() for name, b in tasks.biomarkers.items()},
    }
    # JSON is data, even if a supplied label contains HTML/script delimiters.
    data = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("templates") / "mutation_explorer.html"
    path.write_text(template.read_text().replace("__BENCHMARK_DATA__", data))
