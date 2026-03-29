"""
Quantix v3 — Layer 1: Traffic Intelligence Agent
Detects congestion, stuck vehicles, accidents, and road blockages.
LISTENS TO: WeatherAgent (mud → adds congestion)
PUBLISHES TO: RouteRiskAgent, ConstraintBuilder, FleetAgent
"""
from __future__ import annotations
import logging
from typing import Dict, List

from models.schemas import RouteInput, TrafficOutput, WeatherOutput
from events.event_bus import AgentEventBus
from events.event_types import EventType, EventSeverity, QuantixEvent

logger = logging.getLogger("quantix.TrafficAgent")

CONGESTION_HIGH = 0.65
STUCK_RISK_THRESHOLD = 0.55   # mud score above this → stuck vehicle risk


class TrafficIntelligenceAgent:

    def __init__(self, bus: AgentEventBus):
        self.bus = bus
        self._extra_congestion: Dict[str, float] = {}
        self._events_published: List[str] = []

        # React to weather events: mud/rain adds congestion
        bus.subscribe(EventType.HEAVY_RAIN_DETECTED, self._on_heavy_rain)
        bus.subscribe(EventType.MUD_ROUTE_DETECTED,  self._on_mud_route)
        bus.subscribe(EventType.FLOOD_ZONE_DETECTED, self._on_flood_zone)
        bus.subscribe(EventType.SNOWSTORM_DETECTED,  self._on_snowstorm)
        bus.subscribe(EventType.ACCIDENT_REPORTED,   self._on_accident)

    async def _on_heavy_rain(self, ev: QuantixEvent):
        for edge in ev.affected_edges:
            self._extra_congestion[edge] = max(
                self._extra_congestion.get(edge, 0), 0.30
            )
        logger.info(f"[TrafficAgent] Heavy rain: adding congestion to {len(ev.affected_edges)} edges")

    async def _on_mud_route(self, ev: QuantixEvent):
        for edge in ev.affected_edges:
            self._extra_congestion[edge] = max(
                self._extra_congestion.get(edge, 0), 0.45
            )

    async def _on_flood_zone(self, ev: QuantixEvent):
        for edge in ev.affected_edges:
            self._extra_congestion[edge] = 1.0   # effectively blocked

    async def _on_snowstorm(self, ev: QuantixEvent):
        for key in list(self._extra_congestion):
            self._extra_congestion[key] = 0.80

    async def _on_accident(self, ev: QuantixEvent):
        for edge in ev.affected_edges:
            self._extra_congestion[edge] = max(
                self._extra_congestion.get(edge, 0), 0.70
            )

    async def run(
        self, route_input: RouteInput, weather_output: WeatherOutput
    ) -> TrafficOutput:
        await self.bus.publish(QuantixEvent(
            event_type=EventType.AGENT_STARTED,
            source_agent="TrafficAgent",
            severity=EventSeverity.INFO,
            payload={},
            message="Analyzing traffic patterns",
        ))

        congestion: Dict[str, float] = {}
        peak_penalties: Dict[str, float] = {}
        windows: Dict[str, str] = {}
        incidents: List[str] = []
        blocked_edges: List[str] = []
        stuck_vehicles: List[str] = []

        for src, dst, dist in route_input.edges:
            edge_key = f"{src}→{dst}"
            base_c = route_input.traffic_scores.get(dst, 0.15)
            extra = self._extra_congestion.get(edge_key, 0.0)
            c = round(min(base_c * 1.3 + extra + (dist / 2000) * 0.05, 1.0), 3)
            congestion[edge_key] = c

            if c > CONGESTION_HIGH:
                incidents.append(
                    f"High congestion on {edge_key}: {c:.0%} — "
                    f"{'weather-induced' if extra > 0 else 'traffic demand'}"
                )

            # Stuck vehicle detection: mud + heavy traffic + heavy truck
            mud_risk = weather_output.weather_risk_scores.get(dst, 0)
            road = weather_output.road_conditions.get(dst)
            from models.schemas import RoadCondition
            if road == RoadCondition.MUD and c > STUCK_RISK_THRESHOLD:
                stuck_vehicles.append(edge_key)

        # Publish vehicle-stuck events
        if stuck_vehicles:
            await self.bus.publish(QuantixEvent(
                event_type=EventType.VEHICLE_STUCK,
                source_agent="TrafficAgent",
                severity=EventSeverity.CRITICAL,
                payload={"stuck_edges": stuck_vehicles},
                affected_edges=stuck_vehicles,
                requires_fallback=True,
                fallback_strategy="alternate_route",
                message=(
                    f"VEHICLE STUCK: {len(stuck_vehicles)} segment(s) where mud + congestion "
                    "creates large-vehicle stuck scenario. Rerouting required."
                ),
            ))
            self._events_published.append("VEHICLE_STUCK")
            blocked_edges.extend(stuck_vehicles)

        # Congestion spike
        high_cong = [e for e, c in congestion.items() if c > 0.75]
        if high_cong:
            await self.bus.publish(QuantixEvent(
                event_type=EventType.CONGESTION_SPIKE,
                source_agent="TrafficAgent",
                severity=EventSeverity.WARNING,
                payload={"edges": high_cong, "count": len(high_cong)},
                affected_edges=high_cong,
                requires_fallback=len(high_cong) > 2,
                fallback_strategy="night_window",
                message=f"Congestion spike on {len(high_cong)} edges — consider night delivery window",
            ))
            self._events_published.append("CONGESTION_SPIKE")

        # Peak-hour penalties
        for node in route_input.nodes:
            t = route_input.traffic_scores.get(node, 0.1)
            w = weather_output.weather_risk_scores.get(node, 0.1)
            peak_penalties[node] = round(1.0 + (t + w * 0.3) * 0.5, 3)
            windows[node] = "10:00-16:00" if t > 0.5 or w > 0.5 else "07:00-19:00"

        levels = list(congestion.values())
        avg_c = sum(levels) / max(len(levels), 1)
        overall = (
            "critical" if avg_c > 0.75 else
            "heavy"    if avg_c > 0.55 else
            "moderate" if avg_c > 0.35 else "light"
        )

        output = TrafficOutput(
            congestion_scores=congestion,
            peak_hour_penalties=peak_penalties,
            recommended_windows=windows,
            incidents=incidents,
            blocked_edges=blocked_edges,
            stuck_vehicles=stuck_vehicles,
            overall_level=overall,
            analysis_summary=(
                f"Traffic: {overall.upper()}. "
                f"Incidents: {len(incidents)}. "
                f"Stuck vehicles: {len(stuck_vehicles)}. "
                f"Blocked edges: {len(blocked_edges)}."
            ),
            events_published=self._events_published,
        )

        await self.bus.publish(QuantixEvent(
            event_type=EventType.AGENT_COMPLETED,
            source_agent="TrafficAgent",
            severity=EventSeverity.INFO,
            payload={"overall": overall, "incidents": len(incidents)},
            message=f"TrafficAgent done: {overall} — {len(stuck_vehicles)} stuck vehicles",
        ))
        return output
