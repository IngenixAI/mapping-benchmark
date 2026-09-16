import pytest

from mapping_benchmark.ontology import Ontology

# A toy DAG:  root -> {carcinoma -> {lung_ad, lung_sq}, melanoma}
EDGES = {"carcinoma": {"root"}, "melanoma": {"root"},
         "lung_ad": {"carcinoma"}, "lung_sq": {"carcinoma"}}


@pytest.fixture
def toy() -> Ontology:
    return Ontology(parents=EDGES,
                    names={t: t for t in ("root", "carcinoma", "melanoma", "lung_ad", "lung_sq")})
