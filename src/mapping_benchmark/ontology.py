"""The disease ontology, information content and simGIC.

Two disease terms are compared by their *ancestor closures*, the term itself plus every
term above it in the graph. simGIC (Pesquita et al. 2008) is the information content the
two closures share over the information content of their union:

    simGIC(A, B) = IC(A ∩ B) / (IC(A ∩ B) + IC(A \\ B) + IC(B \\ A))
    IC(t) = -ln p(t),  p(t) = share of the ontology's terms at or below t

Identical terms score 1. Terms sharing nothing but the root score 0. A pair one level
apart, such as lung adenocarcinoma and non-small cell lung carcinoma, scores in between,
which is what lets a neighbour annotated at a different depth still earn credit.

The graph is a DAG, so a term can be reached by several paths; every measure here is a
set operation on closures, so the number of paths never enters.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd


class Ontology:
    """Direct-parent edges and names, with information content and simGIC on top."""

    def __init__(self, parents: dict[str, set[str]], names: dict[str, str]) -> None:
        self.parents = parents
        self.names = names
        self._ancestors: dict[str, frozenset[str]] = {}
        self._ic: dict[str, float] | None = None

    def __contains__(self, term: str) -> bool:
        return term in self.names

    def ancestors(self, term: str) -> frozenset[str]:
        """The term plus every term above it; empty for a term the graph does not carry."""
        if term not in self.names:
            return frozenset()
        if term not in self._ancestors:
            seen, stack = set(), [term]
            while stack:
                node = stack.pop()
                if node not in seen:
                    seen.add(node)
                    stack.extend(self.parents.get(node, ()))
            self._ancestors[term] = frozenset(seen)
        return self._ancestors[term]

    @property
    def information_content(self) -> dict[str, float]:
        """``-ln(terms at or below t / terms)`` for every term; 0 at the root."""
        if self._ic is None:
            below: dict[str, int] = defaultdict(int)
            for term in self.names:
                for ancestor in self.ancestors(term):
                    below[ancestor] += 1
            total = len(self.names)
            self._ic = {t: -math.log(n / total) for t, n in below.items()}
        return self._ic

    def ic(self, term: str) -> float:
        return self.information_content.get(term, 0.0)

    def simgic(self, a: str, b: str) -> float:
        """simGIC of two terms in [0, 1]; 1 for identical terms, 0 when one is off-graph."""
        if a == b:
            return 1.0
        up_a, up_b = self.ancestors(a), self.ancestors(b)
        ic = self.information_content
        union = sum(ic.get(t, 0.0) for t in up_a | up_b)
        if not union:
            return 0.0
        shared = sum(ic.get(t, 0.0) for t in up_a & up_b)
        return min(shared / union, 1.0)


def load(path: str | Path) -> Ontology:
    """Read an ontology parquet with columns ``id``, ``name`` and ``parents``.

    ``parents`` holds each term's direct parents as a JSON list. ``datasets.export_ontology``
    writes this layout from the HuggingFace table; a parent that is not itself a term is
    dropped so that every node in a closure has a seat in the IC denominator.
    """
    df = pd.read_parquet(path)
    names = dict(zip(df["id"], df["name"]))
    parents = {
        term: {p for p in json.loads(raw) if p in names}
        for term, raw in zip(df["id"], df["parents"])
    }
    return Ontology(parents, names)
