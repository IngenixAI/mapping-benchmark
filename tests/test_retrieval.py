"""precision@k under binary and simGIC relevance, and the two bounds."""

import numpy as np
import pytest

from mapping_benchmark.retrieval import (
    LABEL_WEIGHT,
    MIN_CLUSTERS,
    Pool,
    Relevance,
    bootstrap,
    evaluate,
    precision_at_k,
    resampling_units,
)

# Fixed expectations; do not regenerate from the implementation under test.
EXPECTED_PRECISION = {1: 1.0, 3: 0.3888888888888889, 5: 0.5666666666666665}


def pool(labels, coords) -> Pool:
    return Pool(np.array(coords, dtype=float)[:, None], np.array(labels))


def test_binary_relevance_reproduces_the_fixed_precisions():
    # Interleaved so that retrieval is wrong often enough to move the numbers off 1.0.
    labels = ["a"] * 6 + ["b"] * 6
    coords = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]
    result = evaluate(pool(labels, coords), pool(labels, [c + 0.25 for c in coords]),
                      Relevance("binary"), [1, 3, 5])
    for k, expected in EXPECTED_PRECISION.items():
        assert result.metrics[k].precision == pytest.approx(expected)
    assert result.n_query_labels == result.n_database_labels == 2


def test_labels_are_weighted_by_the_square_root_of_their_size():
    # 16 "a" queries all retrieve their own label; 4 "b" queries all retrieve "a".
    query = pool(["a"] * 16 + ["b"] * 4, list(range(16)) + [100, 101, 102, 103])
    database = pool(["a"] * 16 + ["b"] * 4, list(range(16)) + [200, 201, 202, 203])
    m = evaluate(query, database, Relevance("binary"), [1]).metrics[1]
    assert LABEL_WEIGHT == 0.5
    assert m.per_label_precision == {"a": 1.0, "b": 0.0}
    assert m.precision == pytest.approx(4 / 6)   # weights 4 and 2, not 16 and 4


def test_simgic_relevance_scores_a_parent_above_a_sibling(toy):
    matrix = Relevance("simgic", toy).matrix(["lung_ad"],
                                             ["lung_ad", "carcinoma", "lung_sq", "melanoma"])
    same, parent, sibling, other = matrix[0]
    assert same == 1.0 and parent > sibling > other == 0.0


def test_a_vocabulary_off_the_ontology_is_refused(toy):
    with pytest.raises(ValueError, match="no query label"):
        Relevance("simgic", toy).matrix(["breast"], ["lung_ad"])
    # One unknown label among known ones scores 0 and is only warned about.
    assert Relevance("simgic", toy).matrix(["lung_ad"], ["lung_ad", "nope"])[0].tolist() == [1, 0]


def test_the_two_bounds_bracket_a_method():
    # 4 "a" and 6 "b" on each side: a random draw hits "a" 4/10 of the time, and the
    # oracle fills 4 of 5 slots for an "a" query and 5 of 5 for a "b" query.
    labels = ["a"] * 4 + ["b"] * 6
    query, database = pool(labels, range(10)), pool(labels, [c + 0.25 for c in range(10)])
    rel = Relevance("binary")
    oracle = evaluate(query, database, rel, [5], bound="oracle").metrics[5]
    random = evaluate(query, database, rel, [5], bound="random").metrics[5]
    method = evaluate(query, database, rel, [5]).metrics[5]

    assert oracle.per_label_precision == {"a": pytest.approx(0.8), "b": pytest.approx(1.0)}
    assert random.per_label_precision == {"a": pytest.approx(0.4), "b": pytest.approx(0.6)}
    assert random.precision <= method.precision <= oracle.precision


def test_bounds_read_no_features():
    labels = ["a", "a", "b", "b"]
    empty = Pool(np.empty((4, 0)), np.array(labels))
    result = evaluate(empty, empty, Relevance("binary"), [1], bound="oracle")
    assert result.metrics[1].precision == 1.0


def test_unlabelled_rows_are_dropped_and_a_single_label_database_is_refused():
    query = pool(["a", "a", ""], [0, 1, 2])
    database = pool(["a", "b", ""], [0, 1, 2])
    result = evaluate(query, database, Relevance("binary"), [1])
    assert (result.n_query, result.n_database) == (2, 2)
    with pytest.raises(ValueError, match="single label"):
        evaluate(query, pool(["a", "a"], [0, 1]), Relevance("binary"), [1])


def test_interval_counts_a_donor_drawn_twice_as_two_donors():
    # Each query is its own donor and label, as on the donor axis, so the interval must be
    # that of a plain mean over resampled donors; recomputed label weights would shrink it.
    rng = np.random.default_rng(1)
    hits = rng.random(30) < 0.4
    donors = np.array([f"d{i:02d}" for i in range(30)])
    # A hit sits on its own donor's xenograft, a miss on the next donor's.
    at = np.where(hits, np.arange(30), (np.arange(30) + 1) % 30)
    query = pool(donors, at)
    query.clusters = donors
    database = pool(donors, np.arange(30))
    result = evaluate(query, database, Relevance("binary"), [1], n_bootstrap=200, seed=3)
    draw = np.random.default_rng(3)
    expected = np.percentile(
        [hits[(draw.random(30) * 30).astype(int)].mean() for _ in range(200)], [2.5, 97.5])
    m = result.metrics[1]
    assert m.precision == pytest.approx(hits.mean())
    assert m.precision_ci == pytest.approx(tuple(expected))
    bound = evaluate(query, database, Relevance("binary"), [1], "oracle", n_bootstrap=200)
    assert bound.metrics[1].precision_ci is None


def test_resamples_keep_each_stratum_and_move_a_cluster_together():
    strata = np.array(["a", "a", "a", "b", "b", "b"])
    clusters = np.array(["x", "x", "y", "z", "w", "v"])
    seen = []
    draws = bootstrap(lambda w: seen.append(w) or w @ np.arange(6), strata, clusters, 50,
                      np.random.default_rng(0))
    for w in seen:
        assert w[0] == w[1]  # cluster x's two rows
        assert w[0] + w[2] == 2 and w[3:].sum() == 3  # a draws its 2 clusters, b its 3
    # The same rows in another order give the same resamples.
    order = np.array([5, 2, 0, 4, 1, 3])
    shuffled = bootstrap(lambda w: w @ np.arange(6)[order], strata[order], clusters[order], 50,
                         np.random.default_rng(0))
    assert np.array_equal(draws, shuffled)


def test_label_shares_stay_fixed_unless_a_label_is_short_of_clusters():
    # Every "a" query scores 1, every "b" query 0, so only the labels' shares could move.
    n = MIN_CLUSTERS
    gains = np.r_[np.ones(n), np.zeros(n)][:, None]
    labels = np.array(["a"] * n + ["b"] * n)
    m = precision_at_k(gains, labels, np.arange(2 * n), 200, seed=1)
    assert m.precision == pytest.approx(0.5)
    assert m.precision_ci == pytest.approx((0.5, 0.5))
    # Labels of one query each share one stratum, and their weights follow the draws.
    few = precision_at_k(np.array([[1.0], [0.0], [1.0], [0.0]]), np.array(list("abcd")),
                         np.arange(4), 200, seed=1)
    low, high = few.precision_ci
    assert low < 0.5 < high


def test_a_tcga_patient_is_resampled_with_its_tissue_source_site():
    assert resampling_units(["TCGA-AB-1234", "TCGA-AB-5678", "MP-12", "d01"]).tolist() == [
        "TCGA-AB", "TCGA-AB", "MP-12", "d01"]
