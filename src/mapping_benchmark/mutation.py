"""Axis 2: Jaccard@k of driver-gene mutation sets.

Each sample's mutations are a binary vector over a shared gene panel. A retrieved
neighbour is worth the Jaccard overlap between its mutation set and the query's, and
Jaccard@k is the mean over the k nearest neighbours, averaged over queries. Scored within
a cohort (breast, lung or every sample), so the question is whether, among samples of
one disease, the embedding finds those with a similar mutational profile.

Two bounds are scored without an embedding:

* ``oracle`` is the best ranking that respects disease similarity: database samples are
  ordered by simGIC to the query's MONDO term, ties broken by Jaccard, so it is what a
  perfect ontology match could achieve at best.
* ``random`` is the expectation over a uniformly random draw, the query's mean Jaccard
  over the whole database.
"""

from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel

from mapping_benchmark.retrieval import Bound, Pool, Relevance, nearest_neighbours


class MutationKResult(BaseModel):
    mean_jaccard: float
    per_label_jaccard: dict[str, float] = {}


class MutationResult(BaseModel):
    metrics: dict[int, MutationKResult]
    bound: Bound | None = None
    n_query: int
    n_database: int
    n_genes: int


def load_cohorts(path: str | Path) -> dict[str, set[str]]:
    """Cohort name -> the MONDO terms it keeps; an empty set keeps every sample."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {name: set(terms or []) for name, terms in raw.items()}


def jaccard_matrix(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """``(n_query, n_database)`` pairwise Jaccard over two boolean gene matrices.

    One matmul for the intersections rather than a three-way broadcast. Two empty sets
    score 0.
    """
    q, d = query.astype(np.int32), database.astype(np.int32)
    intersection = q @ d.T
    union = q.sum(1)[:, None] + d.sum(1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros(intersection.shape), where=union > 0)


def evaluate(query: Pool, database: Pool, query_mutations: np.ndarray,
             database_mutations: np.ndarray, k_values: list[int],
             bound: Bound | None = None, disease: Relevance | None = None) -> MutationResult:
    """Jaccard@k for every k, for an embedding or for one of the two bounds.

    ``query.features`` and ``database.features`` are read only for an embedding; the
    labels are the MONDO terms the oracle ranks by and the per-label results group by.
    """
    if query_mutations.shape[1] != database_mutations.shape[1]:
        raise ValueError("query and database mutation matrices differ in gene count")
    jaccard = jaccard_matrix(query_mutations.astype(bool), database_mutations.astype(bool))
    max_k = max(k_values)

    if bound == "random":
        score = {k: jaccard.mean(axis=1) for k in k_values}
    elif bound == "oracle":
        if disease is None:
            raise ValueError("the oracle needs a disease relevance to rank by")
        ranked = _disease_ranked(jaccard, query.labels, database.labels, disease)
        score = {k: ranked[:, k - 1] for k in k_values}
    else:
        neighbours = nearest_neighbours(query.features, database.features, max_k)
        gains = np.take_along_axis(jaccard, neighbours, axis=1)
        score = {k: gains[:, :k].mean(axis=1) for k in k_values}

    labels = np.unique(query.labels)
    return MutationResult(
        metrics={
            k: MutationKResult(
                mean_jaccard=float(s.mean()),
                per_label_jaccard={str(lab): float(s[query.labels == lab].mean())
                                   for lab in labels},
            )
            for k, s in score.items()
        },
        bound=bound, n_query=len(query.labels), n_database=len(database.labels),
        n_genes=int(query_mutations.shape[1]),
    )


def _disease_ranked(jaccard: np.ndarray, q_labels: np.ndarray, db_labels: np.ndarray,
                    disease: Relevance) -> np.ndarray:
    """``(n_query, n_database)``: mean Jaccard of the top-k disease-ranked rows, per k.

    Column ``k - 1`` is the score at depth k. Rows are ordered by disease similarity to the
    query's label; within a run of equal similarity they are ordered by the query's own
    Jaccard, best first, which makes this the best ranking consistent with the disease
    ordering.
    """
    q_vocab, q_code = np.unique(q_labels, return_inverse=True)
    db_vocab, db_code = np.unique(db_labels, return_inverse=True)
    similarity = disease.matrix(list(q_vocab), list(db_vocab))[:, db_code]

    out = np.empty_like(jaccard)
    depths = np.arange(1, jaccard.shape[1] + 1)
    for code, row in enumerate(similarity):
        queries = q_code == code
        order = np.argsort(-row, kind="stable")
        ranked = jaccard[queries][:, order]
        tie = row[order]
        bounds = np.flatnonzero(np.r_[True, tie[1:] != tie[:-1], True])
        for start, stop in zip(bounds[:-1], bounds[1:]):
            ranked[:, start:stop] = -np.sort(-ranked[:, start:stop], axis=1)
        out[queries] = np.cumsum(ranked, axis=1) / depths
    return out
