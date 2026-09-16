"""Build the interactive simGIC explorer: the MONDO graph the benchmark reaches, click two
terms to see their ancestor cones and the score.

simGIC is IC(A ∩ B) / IC(A ∪ B) over the two ancestor closures, so the number is only
readable next to the two cones it is a ratio of. The page draws every annotated term plus
its full closure, sizes each node by its information content, and paints the union of two
picked cones in three colours: shared, only A, only B. A second page draws the full
term-by-term matrix, clustered into blocks. Graph, weights and score all come from
``mapping_benchmark.ontology``, the module the benchmark scores with.

Usage, after the workflow has downloaded the data:
    uv run python docs/simgic-explorer/build.py
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from mapping_benchmark.ontology import Ontology, load

HERE = Path(__file__).parent
SWEEPS = 6   # barycentre passes; the ordering stops moving well before this


def depths(nodes: set[str], parents: dict[str, set[str]]) -> dict[str, int]:
    """Longest path from a root, so every parent is drawn strictly above its children."""
    depth: dict[str, int] = {}
    busy: set[str] = set()

    def of(node: str) -> int:
        if node in depth:
            return depth[node]
        busy.add(node)
        above = [of(p) for p in parents.get(node, set()) & nodes if p not in busy]
        busy.discard(node)
        depth[node] = 1 + max(above) if above else 0
        return depth[node]

    return {node: of(node) for node in nodes}


def layout(nodes: set[str], parents: dict[str, set[str]]) -> dict[str, tuple[float, int]]:
    """(column, row) per node: row is the depth, column a barycentre order within the row."""
    row = depths(nodes, parents)
    kids: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        for parent in parents.get(node, set()) & nodes:
            kids[parent].append(node)
    layers: dict[int, list[str]] = defaultdict(list)
    for node in sorted(nodes, key=lambda n: (row[n], n)):
        layers[row[node]].append(node)
    x = {node: float(i) for layer in layers.values() for i, node in enumerate(layer)}

    def sweep(order, neighbours) -> None:
        for level in order:
            layer = layers[level]
            key = {n: (sum(x[m] for m in neighbours.get(n, ()) if m in x)
                       / len(neighbours[n]) if neighbours.get(n) else x[n]) for n in layer}
            layer.sort(key=lambda n: (key[n], n))
            for i, node in enumerate(layer):
                x[node] = float(i)

    up = {n: [p for p in parents.get(n, set()) & nodes] for n in nodes}
    for _ in range(SWEEPS):
        sweep(sorted(layers), up)
        sweep(sorted(layers, reverse=True), kids)
    widest = max(len(layer) for layer in layers.values())
    for layer in layers.values():
        shift = (widest - len(layer)) / 2
        for node in layer:
            x[node] += shift
    return {node: (x[node], row[node]) for node in nodes}


def view(onto: Ontology, counts: pd.Series) -> dict:
    """The graph induced by the annotated terms and their closures, laid out, with IC."""
    terms = set(counts.index)
    nodes: set[str] = set()
    for term in terms:
        nodes |= onto.ancestors(term)
    place = layout(nodes, onto.parents)
    order = sorted(nodes, key=lambda n: (place[n][1], place[n][0], n))
    index = {node: i for i, node in enumerate(order)}
    return {
        "label": "Disease — MONDO",
        "samples": int(counts.sum()),
        "nodes": [
            {"id": node, "name": onto.names.get(node, node), "ic": round(onto.ic(node), 4),
             "x": round(place[node][0], 1), "y": place[node][1],
             "p": sorted(index[p] for p in onto.parents.get(node, set()) & nodes),
             "n": int(counts.get(node, 0)), "annotated": node in terms}
            for node in order
        ],
    }


def check(onto: Ontology, page: dict) -> None:
    """The page's numbers must be the evaluator's: recompute simGIC from what shipped."""
    nodes = page["nodes"]
    up = {}
    for i in range(len(nodes)):
        seen, stack = set(), [i]
        while stack:
            j = stack.pop()
            if j not in seen:
                seen.add(j)
                stack.extend(nodes[j]["p"])
        up[i] = seen

    def ratio(a: int, b: int) -> float:
        shared = sum(nodes[i]["ic"] for i in up[a] & up[b])
        union = sum(nodes[i]["ic"] for i in up[a] | up[b])
        return min(shared / union, 1.0) if union else 0.0

    for a in range(0, len(nodes), max(1, len(nodes) // 50)):
        for b in (0, len(nodes) // 3, len(nodes) - 1):
            if a != b:
                want = onto.simgic(nodes[a]["id"], nodes[b]["id"])
                assert abs(ratio(a, b) - want) < 1e-3, (nodes[a]["id"], nodes[b]["id"])


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
__BODY__</body>
</html>
"""


def render(template: Path, out: Path, data: dict) -> None:
    body = template.read_text().replace("__THEME__", (HERE / "theme.css").read_text())
    body = body.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out.write_text(PAGE.replace("__BODY__", body), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", type=Path, default=Path("data/raw/mondo.parquet"))
    parser.add_argument("--metadata", type=Path, default=Path("data/raw/metadata.csv"))
    parser.add_argument("--label-column", default="mondo_id")
    parser.add_argument("--out", type=Path, default=HERE / "simgic_explorer.html")
    parser.add_argument("--matrix-out", type=Path, default=HERE / "simgic_matrix.html")
    args = parser.parse_args()

    onto = load(args.ontology)
    labels = pd.read_csv(args.metadata, index_col=0, dtype=str, keep_default_na=False)
    counts = labels.loc[labels[args.label_column] != "", args.label_column].value_counts()
    counts = counts[[t in onto for t in counts.index]]
    page = view(onto, counts)
    check(onto, page)
    print(f"{len(page['nodes'])} nodes for {len(counts)} annotated terms, "
          f"{int(counts.sum())} samples; ln N = {math.log(len(onto.names)):.2f}")
    data = {"views": {"mondo": page}}
    render(HERE / "explorer.html", args.out, data)
    render(HERE / "matrix.html", args.matrix_out, data)


if __name__ == "__main__":
    main()
