"""Quantix v3 — Layer 1: Demand Intelligence Agent"""
from __future__ import annotations
import logging, math
from typing import Dict, List
from models.schemas import RouteInput, SupplyChainInput, CostInput, DemandOutput
from events.event_bus import AgentEventBus
from events.event_types import EventType, EventSeverity, QuantixEvent

logger = logging.getLogger("quantix.DemandAgent")

class DemandIntelligenceAgent:
    def __init__(self, bus: AgentEventBus):
        self.bus = bus
        self._demand_surge_factor = 1.0
        bus.subscribe(EventType.ROUTE_BLOCKED,  self._on_blockage)
        bus.subscribe(EventType.STOCK_SHORTAGE, self._on_shortage)

    async def _on_blockage(self, ev: QuantixEvent):
        self._demand_surge_factor += 0.10   # blockages create urgent re-orders

    async def _on_shortage(self, ev: QuantixEvent):
        self._demand_surge_factor += 0.20

    async def run(self, ri: RouteInput, sc: SupplyChainInput, ci: CostInput) -> DemandOutput:
        delivery_nodes = [n for n in ri.nodes if n != ri.depot]
        forecasts: Dict[str, float] = {}
        variability: Dict[str, float] = {}
        surge_nodes, priority = [], []

        for node in delivery_nodes:
            base = (sc.demand_forecasts.get(node, 100.0) + ci.delivery_loads.get(node, 100.0)) / 2
            forecast = round(base * self._demand_surge_factor, 2)
            forecasts[node] = forecast
            variability[node] = round(forecast * 0.12, 2)
            if forecast > ci.truck_capacity * 0.8:
                surge_nodes.append(node)
            if ri.delivery_deadlines.get(node, float("inf")) < 24:
                priority.append(node)

        if surge_nodes:
            await self.bus.publish(QuantixEvent(
                event_type=EventType.DEMAND_SURGE,
                source_agent="DemandAgent",
                severity=EventSeverity.WARNING,
                payload={"surge_nodes": surge_nodes, "factor": self._demand_surge_factor},
                affected_nodes=surge_nodes,
                message=f"Demand surge x{self._demand_surge_factor:.2f} at {len(surge_nodes)} node(s)",
            ))

        fleet_rec = max(math.ceil(sum(forecasts.values()) / ci.truck_capacity), ci.num_trucks)
        return DemandOutput(
            demand_forecasts=forecasts,
            demand_variability=variability,
            surge_nodes=surge_nodes,
            priority_deliveries=priority,
            fleet_recommendation=fleet_rec,
            analysis_summary=(
                f"Demand surge factor: x{self._demand_surge_factor:.2f}. "
                f"Surge nodes: {len(surge_nodes)}. Fleet recommendation: {fleet_rec}."
            ),
        )
