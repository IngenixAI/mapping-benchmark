"""Jaccard@k and its two bounds, on a matrix small enough to check by hand."""

import numpy as np
import pytest

from mapping_benchmark import mutation
from mapping_benchmark.retrieval import Pool, Relevance

# Three queries, four database rows.
QUERY = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=bool)
DATABASE = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1]], dtype=bool)
Q_LABELS = np.array(["a", "a", "b"])
DB_LABELS = np.array(["a", "a", "b", "b"])


def _bound(bound, k_values=(1, 2, 4), q_labels=Q_LABELS, disease=Relevance("binary")):
    return mutation.evaluate(
        Pool(np.empty((3, 0)), q_labels), Pool(np.empty((4, 0)), DB_LABELS),
        QUERY, DATABASE, list(k_values), bound=bound, disease=disease,
    )


def test_jaccard_matrix_matches_hand_computation():
    jac = mutation.jaccard_matrix(QUERY, DATABASE)
    assert np.allclose(jac[0], [1.0, 1 / 3, 0.0, 0.5])    # {g1,g2} against each row
    assert np.allclose(jac[1], [0.5, 0.5, 0.0, 0.25])     # {g1}
    assert np.allclose(jac[2], 0.0)                       # empty set, no division by zero


def test_the_oracle_takes_the_best_rows_of_the_nearest_disease_first():
    oracle = _bound("oracle")
    # Same-disease rows come first, best Jaccard first: 1.0, 0.5 and 0 for the three queries.
    assert np.isclose(oracle.metrics[1].mean_jaccard, (1.0 + 0.5 + 0.0) / 3)
    assert np.isclose(oracle.metrics[2].mean_jaccard, (2 / 3 + 0.5 + 0.0) / 3)
    # At k = n_database every row is included, which is what the random draw averages.
    assert np.isclose(oracle.metrics[4].mean_jaccard, _bound("random").metrics[4].mean_jaccard)


def test_a_graded_disease_relevance_reorders_the_database():
    class Graded:
        # "r" is the closest disease to both query labels, "q" halfway, "p" nearly unrelated.
        def matrix(self, query_labels, database_labels):
            return np.tile([0.1, 0.5, 0.9], (len(query_labels), 1))

    oracle = mutation.evaluate(
        Pool(np.empty((3, 0)), Q_LABELS), Pool(np.empty((4, 0)), np.array(["p", "q", "q", "r"])),
        QUERY, DATABASE, [2], bound="oracle", disease=Graded(),
    )
    # k=2 takes "r" then the better of the two "q" rows, never "p", though row 0 is worth
    # 1.0 to query 0.
    assert np.isclose(oracle.metrics[2].mean_jaccard,
                      ((0.5 + 1 / 3) / 2 + (0.25 + 0.5) / 2 + 0.0) / 3)


def test_random_is_the_database_mean_and_flat_in_k():
    random = _bound("random")
    expected = mutation.jaccard_matrix(QUERY, DATABASE).mean()
    assert np.isclose(random.metrics[1].mean_jaccard, expected)
    assert len({m.mean_jaccard for m in random.metrics.values()}) == 1


def test_an_embedding_that_ranks_the_identical_row_first_beats_chance():
    query = Pool(np.array([[0.0], [1.0], [2.0]]), Q_LABELS)
    database = Pool(np.array([[0.4], [1.4], [2.4], [3.0]]), DB_LABELS)
    method = mutation.evaluate(query, database, QUERY, DATABASE, [1, 2])
    random = _bound("random", (1, 2))
    for k in (1, 2):
        assert method.metrics[k].mean_jaccard >= random.metrics[k].mean_jaccard
    assert set(method.metrics[1].per_label_jaccard) == {"a", "b"}
    assert method.n_genes == 4


def test_the_oracle_needs_a_disease_relevance():
    with pytest.raises(ValueError, match="oracle needs"):
        _bound("oracle", disease=None)


def test_cohorts_load_as_sets_and_pan_is_empty(tmp_path):
    (tmp_path / "c.yaml").write_text("breast:\n  - MONDO:1\n  - MONDO:2\npan: []\n")
    assert mutation.load_cohorts(tmp_path / "c.yaml") == {"breast": {"MONDO:1", "MONDO:2"},
                                                            "pan": set()}
