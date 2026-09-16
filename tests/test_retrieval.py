"""precision@k under binary and simGIC relevance, and the two bounds."""

import numpy as np
import pytest

from mapping_benchmark.retrieval import LABEL_WEIGHT, Pool, Relevance, evaluate

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
