import json

import pandas as pd
import pytest

from mapping_benchmark import ontology


def test_ancestors_are_the_closure_and_empty_off_graph(toy):
    assert toy.ancestors("lung_ad") == {"lung_ad", "carcinoma", "root"}
    assert toy.ancestors("nope") == frozenset()


def test_information_content_is_higher_for_rarer_terms(toy):
    assert toy.ic("lung_ad") > toy.ic("carcinoma") > toy.ic("root") == 0.0


def test_simgic_orders_identical_parent_sibling_and_unrelated(toy):
    same = toy.simgic("lung_ad", "lung_ad")
    parent = toy.simgic("lung_ad", "carcinoma")
    sibling = toy.simgic("lung_ad", "lung_sq")
    other = toy.simgic("lung_ad", "melanoma")
    assert same == 1.0
    assert parent > sibling > other == 0.0   # melanoma shares only the root, worth nothing


def test_simgic_matches_the_formula(toy):
    ic = toy.information_content
    shared = ic["carcinoma"] + ic["root"]
    only_a, only_b = ic["lung_ad"], ic["lung_sq"]
    assert toy.simgic("lung_ad", "lung_sq") == pytest.approx(shared / (shared + only_a + only_b))


def test_off_graph_terms_score_zero_unless_identical(toy):
    assert toy.simgic("EFO:1", "EFO:1") == 1.0
    assert toy.simgic("EFO:1", "lung_ad") == 0.0


def test_load_reads_json_parents_and_drops_dangling_ones(tmp_path):
    frame = pd.DataFrame({
        "id": ["root", "a", "b"],
        "name": ["root", "A", "B"],
        "parents": [json.dumps([]), json.dumps(["root"]), json.dumps(["a", "MISSING"])],
    })
    frame.to_parquet(tmp_path / "onto.parquet")
    loaded = ontology.load(tmp_path / "onto.parquet")
    assert loaded.names["a"] == "A"
    assert loaded.parents["b"] == {"a"}
    assert loaded.ancestors("b") == {"b", "a", "root"}
