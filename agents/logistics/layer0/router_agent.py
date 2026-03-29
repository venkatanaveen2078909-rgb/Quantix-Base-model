"""
Quantix v3 — Layer 0: Router Agent
Generates candidate routes, flags unpaved/bridge segments,
and broadcasts warnings when routes include high-risk infrastructure.
Reacts to bus events to exclude already-blocked edges.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx

from models.schemas import (
    RouteInput, RouterOutput, InitialRoute, PlannerOutput, RoadCondition
)
from events.event_bus import AgentEventBus
from events.event_types import EventType, EventSeverity, QuantixEvent

logger = logging.getLogger("quantix.RouterAgent")


class RouterAgent:

    def __init__(self, bus: AgentEventBus):
        self.bus = bus

    async def run(
        self, route_input: RouteInput, planner: PlannerOutput
    ) -> RouterOutput:
        await self.bus.publish(QuantixEvent(
            event_type=EventType.AGENT_STARTED,
            source_agent="RouterAgent",
            severity=EventSeverity.INFO,
            payload={"nodes": len(route_input.nodes)},
            message=f"Generating routes for {len(route_input.nodes)} nodes",
        ))

        # Get already-blocked edges from bus
        blocked = self.bus.get_blocked_edges()
        blocked_nodes = self.bus.get_blocked_nodes()

        G = nx.Graph()
        edge_lookup: Dict[Tuple[str, str], float] = {}
        for src, dst, dist in route_input.edges:
            key = f"{src}→{dst}"
            if key not in blocked:
                G.add_edge(src, dst, weight=dist)
                edge_lookup[(src, dst)] = dist
                edge_lookup[(dst, src)] = dist

        # Identify unpaved and bridge segments from node metadata
        unpaved_segments = []
        bridge_crossings = []
        for src, dst, _ in route_input.edges:
            edge_key = f"{src}→{dst}"
            dst_meta = route_input.node_metadata.get(dst)
            if dst_meta:
                if dst_meta.road_condition in (RoadCondition.GRAVEL, RoadCondition.MUD):
                    unpaved_segments.append(edge_key)
                if dst_meta.is_bridge:
                    bridge_crossings.append(edge_key)

        # Broadcast warnings for dangerous infrastructure
        if unpaved_segments:
            await self.bus.publish(QuantixEvent(
                event_type=EventType.MUD_ROUTE_DETECTED,
                source_agent="RouterAgent",
                severity=EventSeverity.WARNING,
                payload={"unpaved_count": len(unpaved_segments)},
                affected_edges=unpaved_segments,
                requires_fallback=True,
                fallback_strategy="lighter_vehicle",
                message=f"{len(unpaved_segments)} unpaved road segment(s) detected on route graph",
            ))

        if bridge_crossings:
            await self.bus.publish(QuantixEvent(
                event_type=EventType.RISK_ESCALATED,
                source_agent="RouterAgent",
                severity=EventSeverity.WARNING,
                payload={"bridge_count": len(bridge_crossings)},
                affected_edges=bridge_crossings,
                message=f"{len(bridge_crossings)} bridge crossing(s) on candidate routes — weight limit applies",
            ))

        candidates: List[InitialRoute] = []

        # Strategy 1: nearest-neighbour (fast, baseline)
        nn = self._nearest_neighbour(G, route_input, blocked_nodes)
        if nn:
            candidates.append(self._score(
                "nn_greedy", nn, route_input, edge_lookup, unpaved_segments
            ))

        # Strategy 2: shortest-path cascade
        sp = self._shortest_path(G, route_input, blocked_nodes)
        if sp:
            candidates.append(self._score(
                "shortest_path", sp, route_input, edge_lookup, unpaved_segments
            ))

        # Strategy 3: risk-avoidance (avoids bad weather nodes)
        ra = self._risk_aware(G, route_input, planner, blocked_nodes)
        if ra:
            candidates.append(self._score(
                "risk_aware", ra, route_input, edge_lookup, unpaved_segments
            ))

        # Strategy 4: MST-based
        mst = self._mst(G, route_input)
        if mst:
            candidates.append(self._score(
                "mst_approx", mst, route_input, edge_lookup, unpaved_segments
            ))

        graph_stats = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "connected": nx.is_connected(G) if G.number_of_nodes() > 0 else False,
            "blocked_edges": len(blocked),
            "unpaved_segments": len(unpaved_segments),
        }

        connectivity = (
            len(max(nx.connected_components(G), key=len)) / max(len(route_input.nodes), 1)
            if G.number_of_nodes() > 0 else 0.0
        )

        output = RouterOutput(
            candidate_routes=candidates,
            graph_stats=graph_stats,
            connectivity_score=round(connectivity, 3),
            unpaved_segments=unpaved_segments,
            bridge_crossings=bridge_crossings,
            router_summary=(
                f"Generated {len(candidates)} routes. "
                f"Excluded {len(blocked)} blocked edges. "
                f"Unpaved: {len(unpaved_segments)}, bridges: {len(bridge_crossings)}."
            ),
        )

        await self.bus.publish(QuantixEvent(
            event_type=EventType.AGENT_COMPLETED,
            source_agent="RouterAgent",
            severity=EventSeverity.INFO,
            payload={"routes": len(candidates), "connectivity": connectivity},
            message=f"Router complete: {len(candidates)} candidate routes",
        ))
        return output

    # ── Route strategies ─────────────────────────────────────

    def _nearest_neighbour(self, G, ri, blocked_nodes) -> List[str]:
        if ri.depot not in G:
            return []
        unvisited = {n for n in ri.nodes if n != ri.depot and n not in blocked_nodes and n in G}
        route, current = [ri.depot], ri.depot
        while unvisited:
            nbrs = [(n, G[current][n]["weight"]) for n in G.neighbors(current) if n in unvisited]
            if not nbrs:
                route.extend(list(unvisited)); break
            nxt = min(nbrs, key=lambda x: x[1])[0]
            route.append(nxt); unvisited.discard(nxt); current = nxt
        route.append(ri.depot)
        return route

    def _shortest_path(self, G, ri, blocked_nodes) -> List[str]:
        if ri.depot not in G:
            return []
        delivery = [n for n in ri.nodes if n != ri.depot and n not in blocked_nodes and n in G]
        visited, route = {ri.depot}, [ri.depot]
        for node in delivery:
            if nx.has_path(G, route[-1], node):
                try:
                    path = nx.shortest_path(G, route[-1], node, weight="weight")
                    for p in path[1:]:
                        if p not in visited:
                            route.append(p); visited.add(p)
                except Exception:
                    route.append(node)
        route.append(ri.depot)
        return route

    def _risk_aware(self, G, ri, planner, blocked_nodes) -> List[str]:
        RG = G.copy()
        for src, dst, dist in ri.edges:
            if RG.has_edge(src, dst):
                w_risk = ri.weather_scores.get(dst, 0.1)
                t_risk = ri.traffic_scores.get(dst, 0.1)
                penalty = (w_risk * planner.weather_sensitivity + t_risk) * 30
                RG[src][dst]["weight"] = dist + penalty
        return self._nearest_neighbour(RG, ri, blocked_nodes)

    def _mst(self, G, ri) -> List[str]:
        if not nx.is_connected(G) or G.number_of_nodes() < 2:
            return []
        try:
            mst = nx.minimum_spanning_tree(G)
            route = list(nx.dfs_preorder_nodes(mst, source=ri.depot))
            route.append(ri.depot)
            return route
        except Exception:
            return []

    def _score(
        self,
        rid: str,
        route: List[str],
        ri: RouteInput,
        edge_lookup: Dict,
        unpaved: List[str],
    ) -> InitialRoute:
        dist = sum(
            edge_lookup.get((route[i], route[i+1]),
                            edge_lookup.get((route[i+1], route[i]), 50.0))
            for i in range(len(route) - 1)
        )
        covered = len(set(route) & set(ri.nodes)) / max(len(ri.nodes), 1)
        feasibility = round(covered * 0.7 + (1 - min(dist / 5000, 1)) * 0.3, 3)
        uses_unpaved = any(
            f"{route[i]}→{route[i+1]}" in unpaved for i in range(len(route) - 1)
        )
        return InitialRoute(
            route_id=rid,
            path=route,
            total_distance_km=round(dist, 2),
            estimated_cost=round(dist * 0.5, 2),
            estimated_time_hr=round(dist / 60, 2),
            feasibility_score=feasibility,
            uses_unpaved=uses_unpaved,
        )
