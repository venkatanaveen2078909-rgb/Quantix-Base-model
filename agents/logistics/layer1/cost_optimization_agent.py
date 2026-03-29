"""
Quantix v3 — Layer 1: Cost Optimization Agent
Computes baseline costs and adjusts for fallback penalties from event bus.
LISTENS TO: FallbackEngine (penalty_map applied to edge costs)
"""
from __future__ import annotations
import logging
from typing import Dict, Any
from models.schemas import CostInput, RouteInput, CostOptimizationOutput
from events.event_bus import AgentEventBus
from events.event_types import EventType, EventSeverity, QuantixEvent

logger = logging.getLogger("quantix.CostAgent")


class CostOptimizationAgent:

    def __init__(self, bus: AgentEventBus):
        self.bus = bus
        self._penalty_map: Dict[str, float] = {}
        bus.subscribe(EventType.FALLBACK_ACTIVATED, self._on_fallback)
        bus.subscribe(EventType.ROUTE_BLOCKED,      self._on_route_event)
        bus.subscribe(EventType.VEHICLE_STUCK,      self._on_route_event)

    async def _on_fallback(self, ev: QuantixEvent):
        penalties = ev.payload.get("penalty_map", {})
        self._penalty_map.update(penalties)

    async def _on_route_event(self, ev: QuantixEvent):
        for edge in ev.affected_edges:
            self._penalty_map[edge] = max(self._penalty_map.get(edge, 1.0), 2.0)

    async def run(self, cost_input: CostInput, route_input: RouteInput) -> CostOptimizationOutput:
        await self.bus.publish(QuantixEvent(
            event_type=EventType.AGENT_STARTED,
            source_agent="CostAgent",
            severity=EventSeverity.INFO,
            payload={},
            message="Computing baseline costs with fallback adjustments",
        ))

        total_dist = sum(cost_input.route_distances.values())
        fuel_cost = total_dist * cost_input.fuel_cost_per_km
        driver_cost = (total_dist / 60) * cost_input.driver_cost_per_hour
        warehouse_cost = cost_input.warehouse_cost_per_day * cost_input.total_warehousing_days
        fleet_overhead = cost_input.num_trucks * 150.0
        baseline = round(fuel_cost + driver_cost + warehouse_cost + fleet_overhead, 2)

        # Build edge cost coefficients with fallback penalties
        edge_coeffs: Dict[str, float] = {}
        for src, dst, dist in route_input.edges:
            edge_key = f"{src}→{dst}"
            base_coeff = dist / max(total_dist, 1)
            penalty_mult = self._penalty_map.get(edge_key, 1.0)
            edge_coeffs[edge_key] = round(base_coeff * penalty_mult, 4)

        # Total penalty adjustment summary
        penalty_adj = {k: v for k, v in self._penalty_map.items()}

        output = CostOptimizationOutput(
            total_baseline_cost=baseline,
            cost_breakdown={
                "fuel_cost": round(fuel_cost, 2),
                "driver_cost": round(driver_cost, 2),
                "warehouse_cost": round(warehouse_cost, 2),
                "fleet_overhead": round(fleet_overhead, 2),
                "fallback_penalties_applied": len(penalty_adj),
            },
            optimization_objective={
                "edge_cost_coefficients": edge_coeffs,
                "minimize": "total_route_cost",
                "penalty_edges": list(penalty_adj.keys()),
            },
            potential_savings_pct=round(18.0 + len(penalty_adj) * 0.5, 2),
            penalty_adjustments=penalty_adj,
            analysis_summary=(
                f"Baseline cost: ${baseline:,.2f}. "
                f"Fallback penalty edges: {len(penalty_adj)}. "
                f"Estimated savings potential: {18.0 + len(penalty_adj)*0.5:.1f}%."
            ),
        )
        await self.bus.publish(QuantixEvent(
            event_type=EventType.AGENT_COMPLETED,
            source_agent="CostAgent",
            severity=EventSeverity.INFO,
            payload={"baseline": baseline},
            message=f"CostAgent done: baseline=${baseline:,.2f}",
        ))
        return output
