"""The derived gene panel and the query samples it leaves out."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT = Path(__file__).parents[1] / "benchmark-workflow" / "scripts" / "mutation_panel.py"
SPEC = importlib.util.spec_from_file_location("mutation_panel", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

# q1 is mutated in a shared gene, q2 only in a gene the database lacks, q3 has no row.
QUERY = pd.DataFrame({"ENSG1": [1, 0], "ENSG2": [0, 0], "ENSG_QUERY_ONLY": [0, 1]},
                     index=pd.Index(["q1", "q2"], name="patient_id")).astype(np.int8)
DATABASE = pd.DataFrame({"ENSG1": [1], "ENSG2": [1]},
                        index=pd.Index(["d1"], name="patient_id")).astype(np.int8)


def _run(tmp_path, monkeypatch):
    (tmp_path / "mut").mkdir()
    QUERY.to_parquet(tmp_path / "mut" / "qds.parquet")
    DATABASE.to_parquet(tmp_path / "mut" / "dds.parquet")
    pd.DataFrame({"sample_id": ["q1", "q2", "q3", "d1"],
                  "dataset": ["qds", "qds", "qds", "dds"]},
                 index=["q1", "q2", "q3", "d1"]).to_csv(tmp_path / "metadata.csv")
    monkeypatch.setattr("sys.argv", [
        "mutation_panel.py", "--mutation-dir", str(tmp_path / "mut"),
        "--datasets", "qds,dds", "--query-datasets", "qds",
        "--metadata", str(tmp_path / "metadata.csv"),
        "--panel-out", str(tmp_path / "panel.yaml"),
        "--excluded-out", str(tmp_path / "excluded.yaml"),
    ])
    MODULE.main()
    return (yaml.safe_load((tmp_path / "panel.yaml").read_text()),
            yaml.safe_load((tmp_path / "excluded.yaml").read_text()))


def test_panel_is_the_intersection_of_the_datasets(tmp_path, monkeypatch):
    panel, _ = _run(tmp_path, monkeypatch)
    assert panel["genes"] == ["ENSG1", "ENSG2"]
    assert panel["n_genes"] == 2
    assert panel["datasets"] == ["dds", "qds"]


def test_empty_and_unassayed_queries_are_listed_apart(tmp_path, monkeypatch):
    _, excluded = _run(tmp_path, monkeypatch)
    assert excluded["empty_on_panel"] == {"qds": ["q2"]}
    assert excluded["unassayed"] == {"qds": ["q3"]}
    assert excluded["n_empty_on_panel"] == {"qds": 1}
