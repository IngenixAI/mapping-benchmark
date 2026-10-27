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

A method's 95% interval resamples the query clusters within each label, as the mutation
axis does within each cancer. A cluster is a TCGA tumour's tissue source site or any other
tumour's patient, so the samples that share a batch move together. The database stays fixed,
every label keeps its weight and its number of clusters, and every method sees the same
resamples, so two methods can be compared on them (``compare``). A label of one cluster
has no spread of its own to resample. It joins one stratum shared by all such labels, where
a label's weight follows its draws, so a patient drawn twice counts twice. On the donor axis
that is every label, as each is one donor; on MetaPRISM, every diagnosis of one patient.
A label of at least ``MIN_TESTED_CLUSTERS`` clusters also keeps its own score per resample,
so that methods can be compared label by label; with fewer clusters, a cluster bootstrap's
interval is too narrow to test on.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors

from mapping_benchmark.ontology import Ontology

LABEL_WEIGHT = 0.5
MIN_CLUSTERS = 2  # a label of one cluster has no spread of its own to resample
MIN_TESTED_CLUSTERS = 10  # fewest clusters for a label's own comparison between methods
Bound = Literal["oracle", "random"]
Kind = Literal["binary", "simgic"]
Interval = tuple[float, float]


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
    """One side of a retrieval task: an embedding row, a label and a cluster per sample.

    The cluster is the unit a bootstrap resamples. Without clusters every sample is its own.
    """

    features: np.ndarray
    labels: np.ndarray
    clusters: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels).astype(str)
        self.clusters = (np.arange(len(self.labels)) if self.clusters is None
                         else np.asarray(self.clusters))
        if not len(self.features) == len(self.labels) == len(self.clusters):
            raise ValueError("features, labels and clusters differ in length")

    def labelled(self) -> "Pool":
        """Rows with a non-empty label."""
        keep = self.labels != ""
        return Pool(self.features[keep], self.labels[keep], self.clusters[keep])


class KResult(BaseModel):
    precision: float
    precision_ci: Interval | None = None
    # One score per resample, in resample order, for comparisons with other methods.
    precision_draws: list[float] = []
    per_label_precision: dict[str, float] = {}
    # Each label of MIN_TESTED_CLUSTERS or more: its score per resample, in resample order.
    per_label_draws: dict[str, list[float]] = {}


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


def evaluate(query: Pool, database: Pool, relevance: Relevance, k_values: list[int],
             bound: Bound | None = None, n_bootstrap: int = 0,
             seed: int = 0) -> RetrievalResult:
    """precision@k for every k, for an embedding or for one of the two bounds.

    An embedding's precision carries a 95% interval over ``n_bootstrap`` cluster
    resamples; a bound, which reads no embedding, carries none.
    """
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
        metrics={k: precision_at_k(gains[:, :k], query.labels, query.clusters,
                                   0 if bound else n_bootstrap, seed) for k in k_values},
        relevance=relevance.kind, bound=bound,
        n_query=len(query.labels), n_database=len(database.labels),
        n_query_labels=len(q_vocab), n_database_labels=len(db_vocab),
    )


def precision_at_k(gains: np.ndarray, labels: np.ndarray, clusters: np.ndarray,
                   n_bootstrap: int = 0, seed: int = 0) -> KResult:
    """Mean relevance per query, averaged per label, then over labels with weight n^0.5."""
    per_query = gains.mean(axis=1)
    vocab, code, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    per_label = np.array([per_query[labels == label].mean() for label in vocab])
    weights = sizes.astype(float) ** LABEL_WEIGHT
    ci, draws, label_draws = None, [], {}
    if n_bootstrap:
        # A label's weight n^0.5 spread over its n queries. Each stratum keeps the weight of
        # its labels; the labels of a single cluster share stratum -1.
        query_weight = weights[code] / sizes[code]
        n_clusters = np.array([len(np.unique(clusters[code == i])) for i in range(len(vocab))])
        strata = np.where(n_clusters[code] >= MIN_CLUSTERS, code, -1)
        _, stratum = np.unique(strata, return_inverse=True)
        share = np.bincount(stratum, query_weight)
        # A tested label is a stratum of its own, so every resample draws it.
        tested = n_clusters >= MIN_TESTED_CLUSTERS

        def statistic(w: np.ndarray) -> np.ndarray:
            own = (np.bincount(code, w * per_query, len(vocab))[tested]
                   / np.bincount(code, w, len(vocab))[tested])
            w = w * query_weight
            pooled = share @ (np.bincount(stratum, w * per_query) / np.bincount(stratum, w)) \
                / share.sum()
            return np.r_[pooled, own]

        draws = bootstrap(statistic, strata, clusters, n_bootstrap, np.random.default_rng(seed))
        draws, label_draws = draws[:, 0], dict(zip(vocab[tested].tolist(),
                                                   np.round(draws[:, 1:].T, 6).tolist()))
        ci = interval(draws)
    return KResult(precision=float(np.average(per_label, weights=weights)), precision_ci=ci,
                   precision_draws=np.round(draws, 6).tolist(),
                   per_label_precision=dict(zip(vocab.tolist(), per_label.tolist())),
                   per_label_draws=label_draws)


def resampling_units(patients) -> np.ndarray:
    """A TCGA patient's tissue source site (barcode ``TCGA-XX``), any other patient as is.

    A site's tumours share their collection and processing batch, so they are resampled
    together.
    """
    patients = pd.Series(np.asarray(patients, str))
    return patients.str.extract(r"^(TCGA-\w\w)-", expand=False).fillna(patients).to_numpy()


def bootstrap(statistic: Callable[[np.ndarray], float | np.ndarray], strata: np.ndarray,
              clusters: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``statistic(weights)``, a number or an array, over ``n`` resamples of the clusters
    within each stratum.

    Each resample draws every stratum's clusters with replacement, as many as it has, and
    gives each row its cluster's multiplicity as a weight, so a cluster's rows move together.
    A cluster belongs to one stratum. Strata and clusters are taken in sorted order, so the
    same rows and seed give the same resamples in any row order, and two methods scored on
    the same rows can be compared resample by resample. A resample without a score is NaN.
    """
    _, s = np.unique(strata, return_inverse=True)
    _, c = np.unique(clusters, return_inverse=True)
    nested, cluster = np.unique(s * (c.max() + 1) + c, return_inverse=True)
    stratum = nested // (c.max() + 1)
    size = np.bincount(stratum)
    first = np.cumsum(size) - size
    draws = []
    for _ in range(n):
        drawn = first[stratum] + (rng.random(len(nested)) * size[stratum]).astype(int)
        draws.append(statistic(np.bincount(drawn, minlength=len(nested))[cluster].astype(float)))
    return np.array(draws, float)


def interval(draws) -> Interval | None:
    """95% percentile interval of the resamples that have a score."""
    draws = np.asarray(draws, float)
    draws = draws[np.isfinite(draws)]
    return tuple(float(v) for v in np.percentile(draws, [2.5, 97.5])) if len(draws) else None


def compare(draws, reference, delta: float) -> dict:
    """A method against a reference scored on the same resamples.

    ``delta`` is the observed difference; the interval and the two-sided p-value come from
    the difference in each resample, the p-value with the plus-one correction per tail.
    """
    d = np.asarray(draws, float) - np.asarray(reference, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return {}
    low, high = interval(d)
    tail = min((d <= 0).sum(), (d >= 0).sum())
    return {"delta": delta, "delta_ci_low": low, "delta_ci_high": high,
            "delta_p": min(1.0, 2 * (tail + 1) / (len(d) + 1))}


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
