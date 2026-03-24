"""Frame-level graph construction helpers."""

from __future__ import annotations

import networkx as nx

from mcmt_core.features.edges import EdgeFeatureBuilder
from mcmt_core.features.nodes import NodeRecord


def build_spatial_frame_graph(nodes: list[NodeRecord], edge_builder: EdgeFeatureBuilder) -> nx.Graph:
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node.node_id, node=node)
    for i, source in enumerate(nodes):
        for target in nodes[i + 1 :]:
            if source.camera_id == target.camera_id:
                continue
            edge = edge_builder.build(source, target)
            graph.add_edge(source.node_id, target.node_id, edge=edge, score=edge.score, features=edge.features)
    return graph
