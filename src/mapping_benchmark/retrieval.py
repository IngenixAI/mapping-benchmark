"""Axes 1 and 3: graded precision@k of nearest-neighbour retrieval.

For every query sample the k nearest database samples are retrieved in the embedding
(Euclidean distance), and each neighbour is worth the *relevance* of its label to the
query's label: simGIC between two MONDO terms on the ontology axis, 1 or 0 for the same
or a different donor on the donor axis. precision@k is the mean relevance of the k
neighbours, averaged over the queries of each label and then over labels, a label with
n queries weighing sqrt(n) so that populous cancer types count more but not
proportionally so.

Two bounds are scored from the labels alone, which puts every method on a common scale:

* ``oracle`` returns, for each query, the most relevant database samples its label admits.
* ``random`` is the expectation over a uniformly random draw of k database samples.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from loguru import logger
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors

from mapping_benchmark.ontology import Ontology

LABEL_WEIGHT = 0.5
Bound = Literal["oracle", "random"]
Kind = Literal["binary", "simgic"]


class Relevance:
    """How much a database label is worth to a query label, in [0, 1]."""

    def __init__(self, kind: Kind, ontology: Ontology | None = None) -> None:
        if kind == "simgic" and ontology is None:
            raise ValueError("simgic relevance needs an ontology")
        self.kind = kind
        self.ontology = ontology

    def matrix(self, query_labels: list[str], database_labels: list[str]) -> np.ndarray:
        """``(len(query_labels), len(database_labels))`` relevances."""
        if self.kind == "binary":
            return np.equal(np.array(query_labels)[:, None],
                            np.array(database_labels)[None, :]).astype(np.float64)
        onto = self.ontology
        for side, labels in (("query", query_labels), ("database", database_labels)):
            unknown = [t for t in labels if t not in onto]
            if len(unknown) == len(labels):
                raise ValueError(f"no {side} label is a term of the ontology "
                                 f"(e.g. {labels[0]!r})")
            if unknown:
                logger.warning(f"{len(unknown)} {side} label(s) not in the ontology score 0 "
                               f"against every other term: {unknown[:5]}")
        out = np.zeros((len(query_labels), len(database_labels)))
        for i, a in enumerate(query_labels):
            for j, b in enumerate(database_labels):
                out[i, j] = onto.simgic(a, b)
        return out


@dataclass
class Pool:
    """One side of a retrieval task: an embedding row and a label per sample."""

    features: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels).astype(str)
        if len(self.features) != len(self.labels):
            raise ValueError("features and labels differ in length")

    def labelled(self) -> "Pool":
        """Rows with a non-empty label."""
        keep = self.labels != ""
        return Pool(self.features[keep], self.labels[keep])


class KResult(BaseModel):
    precision: float
    per_label_precision: dict[str, float] = {}


class RetrievalResult(BaseModel):
    metrics: dict[int, KResult]
    relevance: Kind
    bound: Bound | None = None
    n_query: int
    n_database: int
    n_query_labels: int
    n_database_labels: int


def nearest_neighbours(query: np.ndarray, database: np.ndarray, k: int) -> np.ndarray:
    """``(n_query, k)`` database row indices, nearest first."""
    nn = NearestNeighbors(metric="euclidean", n_jobs=-1).fit(database)
    return nn.kneighbors(query, n_neighbors=k)[1]


def evaluate(query: Pool, database: Pool, relevance: Relevance,
             k_values: list[int], bound: Bound | None = None) -> RetrievalResult:
    """precision@k for every k, for an embedding or for one of the two bounds."""
    query, database = query.labelled(), database.labelled()
    q_vocab, q_code = np.unique(query.labels, return_inverse=True)
    db_vocab, db_code = np.unique(database.labels, return_inverse=True)
    if len(db_vocab) < 2:
        raise ValueError("the database carries a single label; every ranking scores alike")

    rel = relevance.matrix(list(q_vocab), list(db_vocab))
    max_k = max(k_values)
    db_counts = np.bincount(db_code, minlength=len(db_vocab))

    # (n_query, max_k): what the neighbour at each rank is worth to its query.
    if bound == "oracle":
        gains = _ideal_gains(rel, db_counts, max_k)[q_code]
    elif bound == "random":
        gains = np.repeat((rel @ db_counts / db_counts.sum())[q_code, None], max_k, axis=1)
    else:
        neighbours = nearest_neighbours(query.features, database.features, max_k)
        gains = rel[q_code[:, None], db_code[neighbours]]

    return RetrievalResult(
        metrics={k: precision_at_k(gains[:, :k], query.labels) for k in k_values},
        relevance=relevance.kind, bound=bound,
        n_query=len(query.labels), n_database=len(database.labels),
        n_query_labels=len(q_vocab), n_database_labels=len(db_vocab),
    )


def precision_at_k(gains: np.ndarray, labels: np.ndarray) -> KResult:
    """Mean relevance per query, averaged per label, then over labels with weight n^0.5."""
    per_query = gains.mean(axis=1)
    vocab, sizes = np.unique(labels, return_counts=True)
    per_label = np.array([per_query[labels == label].mean() for label in vocab])
    weights = sizes.astype(float) ** LABEL_WEIGHT
    return KResult(precision=float(np.average(per_label, weights=weights)),
                   per_label_precision=dict(zip(vocab.tolist(), per_label.tolist())))


def _ideal_gains(rel: np.ndarray, db_counts: np.ndarray, k: int) -> np.ndarray:
    """``(n_query_labels, k)``: the best relevance each rank could hold, best first.

    Bounded by how many database rows carry each label: a perfect label carried by two
    samples fills two slots, after which the next best label takes over.
    """
    supply = np.minimum(db_counts, k)
    out = np.zeros((rel.shape[0], k))
    for i, row in enumerate(rel):
        best = np.argsort(-row)
        gains = np.repeat(row[best], supply[best])[:k]
        out[i, :gains.size] = gains
    return out
