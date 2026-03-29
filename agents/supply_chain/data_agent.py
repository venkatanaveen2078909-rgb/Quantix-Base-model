from typing import Any, Dict

from utils.graph_builder import build_vrp_graph, extract_distance_matrix


class DataProcessingAgent:
    def process(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cleans and normalizes raw logistics data, converting it into a graph structure.
        """
        nodes = raw_data.get("nodes", [])
        edges = raw_data.get("edges", [])

        valid_node_ids = set()
        for node in nodes:
            if "id" in node:
                valid_node_ids.add(node["id"])

        cleaned_edges = []
        for edge in edges:
            if (
                edge.get("source") in valid_node_ids
                and edge.get("target") in valid_node_ids
            ):
                if edge.get("distance", 1.0) < 0:
                    edge["distance"] = abs(edge["distance"])
                cleaned_edges.append(edge)

        cleaned_data = {"nodes": nodes, "edges": cleaned_edges}

        G = build_vrp_graph(cleaned_data)
        distance_matrix = extract_distance_matrix(G)

        return {
            "graph": G,
            "distance_matrix": distance_matrix,
            "num_nodes": len(nodes),
            "cleaned_data": cleaned_data,
        }
