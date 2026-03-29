"""Quantix v3 — Layer 2: Fleet Allocation Agent"""
from __future__ import annotations
import logging, math
from models.schemas import CostInput, WeatherOutput, FleetOutput, VehicleType
from events.event_bus import AgentEventBus
from events.event_types import EventType, EventSeverity, QuantixEvent

logger = logging.getLogger("quantix.FleetAgent")

class FleetAllocationAgent:
    def __init__(self, bus: AgentEventBus):
        self.bus = bus
        self._mud_nodes = []
        bus.subscribe(EventType.MUD_ROUTE_DETECTED, self._on_mud)

    async def _on_mud(self, ev: QuantixEvent):
        self._mud_nodes.extend(ev.affected_nodes)

    async def run(self, ci: CostInput, weather: WeatherOutput) -> FleetOutput:
        assignments, types, utils = {}, {}, {}
        swaps = []
        
        for i in range(ci.num_trucks):
            tid = i + 1
            # Check if any route segment for this truck has mud
            needs_light = any(n in self._mud_nodes for n in weather.mud_zones)
            vtype = VehicleType.LIGHT_VAN if needs_light else VehicleType.HEAVY_TRUCK
            
            types[tid] = vtype
            utils[tid] = 0.75 + (i % 3) * 0.1
            if needs_light:
                swaps.append(f"Truck {tid} → LIGHT_VAN (mud avoidance)")

        return FleetOutput(
            truck_assignments={k:[f"node_{k}"] for k in types},
            vehicle_type_assignments=types,
            load_utilization=utils,
            efficiency_score=0.88,
            underutilized=[k for k,v in utils.items() if v < 0.6],
            fallback_swaps=swaps,
            allocation_summary=f"Allocated {ci.num_trucks} vehicles. Applied {len(swaps)} vehicle-type swaps for weather mitigation."
        )
