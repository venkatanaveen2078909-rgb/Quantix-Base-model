from typing import Any, Dict, List

import networkx as nx


def build_vrp_graph(data: Dict[str, Any]) -> nx.Graph:
    """
    Builds a NetworkX graph for a Vehicle Routing Problem (VRP) from raw logistics data.
    """
    G = nx.Graph()
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    # Add nodes (locations, depots, customers)
    for node in nodes:
        G.add_node(
            node["id"], demand=node.get("demand", 0), type=node.get("type", "customer")
        )

    # Add edges (distances, costs)
    for edge in edges:
        G.add_edge(
            edge["source"],
            edge["target"],
            weight=edge.get("distance", 1.0),
            cost=edge.get("cost", 1.0),
        )

    return G


def extract_distance_matrix(G: nx.Graph) -> List[List[float]]:
    """
    Extracts a distance matrix from a NetworkX graph.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    matrix = [[float("inf")] * n for _ in range(n)]

    for i in range(n):
        matrix[i][i] = 0.0

    for u, v, data in G.edges(data=True):
        i = nodes.index(u)
        j = nodes.index(v)
        weight = data.get("weight", float("inf"))
        matrix[i][j] = weight
        matrix[j][i] = weight

    return matrix
