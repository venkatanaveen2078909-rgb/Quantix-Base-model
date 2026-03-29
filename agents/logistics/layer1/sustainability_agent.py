"""Quantix v3 — Layer 1: Sustainability Agent"""
from __future__ import annotations
import logging
from models.schemas import RouteInput, RouterOutput, SustainabilityOutput
from events.event_bus import AgentEventBus
from events.event_types import EventType, EventSeverity, QuantixEvent

logger = logging.getLogger("quantix.SustainabilityAgent")

KG_CO2_DIESEL = 0.27
KG_CO2_EV = 0.05
EV_RANGE_KM = 300.0

class SustainabilityAgent:
    def __init__(self, bus: AgentEventBus):
        self.bus = bus

    async def run(self, ri: RouteInput, router: RouterOutput) -> SustainabilityOutput:
        carbon: dict = {}
        ev_routes, green_recs = [], []
        total = 0.0

        for c in router.candidate_routes:
            em = round(c.total_distance_km * KG_CO2_DIESEL, 2)
            carbon[c.route_id] = em
            total += em
            if c.total_distance_km <= EV_RANGE_KM:
                ev_routes.append(c.route_id)
                green_recs.append(
                    f"Route '{c.route_id}': EV saves "
                    f"{em - c.total_distance_km * KG_CO2_EV:.1f} kg CO₂"
                )

        reduction = round((1 - 0.85) * 100, 2)
        score = round(1.0 - total / max(total * 2, 1), 3)
        return SustainabilityOutput(
            carbon_footprint_kg=carbon,
            total_emissions_kg=round(total, 2),
            green_routes=green_recs,
            ev_compatible=ev_routes,
            sustainability_score=max(0.0, min(1.0, score)),
            emissions_reduction_pct=reduction,
            analysis_summary=(
                f"Total: {total:.1f} kg CO₂. "
                f"EV-compatible routes: {len(ev_routes)}. "
                f"Projected reduction: {reduction:.1f}%."
            ),
        )
