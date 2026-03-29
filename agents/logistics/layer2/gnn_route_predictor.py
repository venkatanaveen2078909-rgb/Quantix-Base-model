"""
Quantix v3 — Layer 2: GNN Route Predictor
Uses graph-based message passing (simulated) to predict optimal paths
and identify bottleneck nodes based on risk + traffic + weather.
"""
from __future__ import annotations
import logging
import networkx as nx
from typing import Dict, List, Tuple
from models.schemas import RouteInput, ConstraintOutput, GNNOutput
from events.event_bus import AgentEventBus
from events.event_types import EventType, EventSeverity, QuantixEvent

logger = logging.getLogger("quantix.GNNPredictor")


class GNNRoutePredictor:

    def __init__(self, bus: AgentEventBus):
        self.bus = bus

    async def run(
        self,
        route_input: RouteInput,
        constraints: ConstraintOutput,
    ) -> GNNOutput:
        await self.bus.publish(QuantixEvent(
            event_type=EventType.AGENT_STARTED,
            source_agent="GNNPredictor",
            severity=EventSeverity.INFO,
            payload={},
            message="Predicting optimal paths via GNN message-passing",
        ))

        G = nx.MultiDiGraph()
        for src, dst, dist in route_input.edges:
            edge_key = f"{src}→{dst}"
            risk_pen = constraints.penalty_weights.get(f"risk_{edge_key}", 0.0)
            traffic_pen = constraints.penalty_weights.get(f"traffic_{edge_key}", 0.0)
            fallback_pen = constraints.fallback_penalties.get(edge_key, 0.0)
            hard_block = constraints.hard_constraints.get(f"hard_block_{edge_key}", False)

            if not hard_block:
                weight = dist + risk_pen + traffic_pen + fallback_pen
                G.add_edge(src, dst, weight=weight, key=edge_key)

        # Simulated message passing (centrality + shortest paths)
        embeddings: Dict[str, List[float]] = {}
        for node in route_input.nodes:
            embeddings[node] = [
                route_input.weather_scores.get(node, 0.1),
                route_input.traffic_scores.get(node, 0.1),
                nx.degree_centrality(G).get(node, 0.0) if G.has_node(node) else 0.0
            ]

        paths: List[List[str]] = []
        scores: Dict[str, float] = {}
        delivery_nodes = [n for n in route_input.nodes if n != route_input.depot]

        if G.has_node(route_input.depot):
            for i, target in enumerate(delivery_nodes[:3]):
                if G.has_node(target) and nx.has_path(G, route_input.depot, target):
                    try:
                        path = nx.shortest_path(G, route_input.depot, target, weight="weight")
                        pid = f"gnn_path_{i}"
                        paths.append(path)
                        scores[pid] = round(1.0 / (1.0 + i), 3)
                    except Exception:
                        continue

        bottlenecks = [
            n for n, b in nx.betweenness_centrality(G, weight="weight").items() if b > 0.4
        ] if G.number_of_nodes() > 0 else []

        output = GNNOutput(
            node_embeddings=embeddings,
            predicted_paths=paths,
            path_scores=scores,
            bottleneck_nodes=bottlenecks,
            centrality=nx.degree_centrality(G) if G.number_of_nodes() > 0 else {},
            confidence=0.88,
            model_summary=(
                f"GNN processed {G.number_of_nodes()} nodes. "
                f"Predicted {len(paths)} paths. "
                f"Bottlenecks identified: {len(bottlenecks)}."
            ),
        )

        await self.bus.publish(QuantixEvent(
            event_type=EventType.AGENT_COMPLETED,
            source_agent="GNNPredictor",
            severity=EventSeverity.INFO,
            payload={"paths": len(paths), "bottlenecks": len(bottlenecks)},
            message=f"GNN prediction complete: {len(paths)} paths found",
        ))
        return output
