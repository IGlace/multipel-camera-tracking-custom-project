"""Graph rendering helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def save_graph_debug_figure(graph: nx.Graph, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))

    positions = {}
    labels = {}
    for idx, (node_id, data) in enumerate(graph.nodes(data=True)):
        node = data.get("node")
        if node is not None and hasattr(node, "center_norm"):
            x, y = node.center_norm
            positions[node_id] = (x, 1.0 - y)
            camera_id = getattr(node, "camera_id", "cam")
            labels[node_id] = f"{camera_id}:{idx}"
        else:
            positions[node_id] = (idx, 0)
            labels[node_id] = str(node_id)

    edge_scores = [float(data.get("score", 0.0)) for _, _, data in graph.edges(data=True)]
    widths = [1.0 + 2.5 * score for score in edge_scores] if edge_scores else 1.0

    nx.draw_networkx_nodes(graph, positions, node_size=700, node_color="#8ecae6")
    if graph.number_of_edges() > 0:
        nx.draw_networkx_edges(graph, positions, width=widths, edge_color=edge_scores, edge_cmap=plt.cm.viridis)
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=8)

    edge_labels = {
        (u, v): f"{float(data.get('score', 0.0)):.2f}"
        for u, v, data in graph.edges(data=True)
    }
    if edge_labels:
        nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, font_size=7)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
