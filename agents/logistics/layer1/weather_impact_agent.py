"""
Quantix v3 — Layer 1: Weather Impact Agent
Analyzes weather conditions, detects mud zones, flood zones,
and publishes cascading events for downstream agents.
COMMUNICATES WITH: TrafficAgent (mud slows traffic),
                   RouteRiskAgent (high weather = high risk),
                   FleetAgent (mud = lighter vehicle needed)
"""
from __future__ import annotations
import logging
import math
from typing import Dict, List

from models.schemas import (
    RouteInput, WeatherOutput, WeatherCondition, RoadCondition, VehicleType
)
from events.event_bus import AgentEventBus
from events.event_types import EventType, EventSeverity, QuantixEvent

logger = logging.getLogger("quantix.WeatherAgent")

# Rain → mud threshold: weather score above this on non-paved segment = mud risk
MUD_THRESHOLD = 0.50
FLOOD_THRESHOLD = 0.75
STORM_THRESHOLD = 0.80


class WeatherImpactAgent:

    def __init__(self, bus: AgentEventBus):
        self.bus = bus
        self._events_published: List[str] = []

        # Subscribe to vehicle_stuck: if a vehicle is stuck, flag that node as dangerous
        bus.subscribe(EventType.VEHICLE_STUCK, self._on_vehicle_stuck)

    async def _on_vehicle_stuck(self, event: QuantixEvent):
        """React to a stuck vehicle — escalate those edges to CLOSED."""
        logger.warning(f"[WeatherAgent] Reacting to VEHICLE_STUCK on {event.affected_edges}")
        if event.affected_edges:
            await self.bus.publish(QuantixEvent(
                event_type=EventType.ROAD_CLOSED,
                source_agent="WeatherAgent:Reactor",
                severity=EventSeverity.CRITICAL,
                payload={"reason": "large vehicle stuck blocking road"},
                affected_edges=event.affected_edges,
                affected_nodes=event.affected_nodes,
                requires_fallback=True,
                fallback_strategy="alternate_route",
                message=(
                    f"Road closed: large vehicle stuck on "
                    f"{event.affected_edges} due to weather conditions"
                ),
            ))
            self._events_published.append("ROAD_CLOSED:reactor")

    async def run(self, route_input: RouteInput) -> WeatherOutput:
        await self.bus.publish(QuantixEvent(
            event_type=EventType.AGENT_STARTED,
            source_agent="WeatherAgent",
            severity=EventSeverity.INFO,
            payload={"weather": route_input.current_weather.value},
            message=f"Analyzing weather: {route_input.current_weather.value}",
        ))

        weather = route_input.current_weather
        risk_scores: Dict[str, float] = {}
        road_conditions: Dict[str, RoadCondition] = {}
        severe_zones, mud_zones, flood_zones = [], [], []
        vehicle_restrictions: Dict[str, VehicleType] = {}

        for node in route_input.nodes:
            base_w = route_input.weather_scores.get(node, 0.1)
            meta = route_input.node_metadata.get(node)
            base_road = meta.road_condition if meta else RoadCondition.PAVED

            # Compute weather-adjusted risk
            weather_multiplier = {
                WeatherCondition.CLEAR:       1.0,
                WeatherCondition.RAIN:        1.5,
                WeatherCondition.HEAVY_RAIN:  2.2,
                WeatherCondition.STORM:       3.0,
                WeatherCondition.SNOW:        2.5,
                WeatherCondition.FOG:         1.6,
                WeatherCondition.HIGH_WIND:   1.8,
            }.get(weather, 1.0)

            risk = round(min(base_w * weather_multiplier, 1.0), 3)
            risk_scores[node] = risk

            # Determine road condition under weather
            actual_condition = base_road
            if weather in (WeatherCondition.HEAVY_RAIN, WeatherCondition.STORM):
                if base_road in (RoadCondition.GRAVEL, RoadCondition.PAVED) and risk > MUD_THRESHOLD:
                    actual_condition = RoadCondition.MUD
                elif base_road == RoadCondition.GRAVEL and risk > FLOOD_THRESHOLD:
                    actual_condition = RoadCondition.FLOODED
            elif weather == WeatherCondition.SNOW:
                actual_condition = RoadCondition.ICY
            road_conditions[node] = actual_condition

            # Classify zones
            if actual_condition == RoadCondition.MUD:
                mud_zones.append(node)
                vehicle_restrictions[node] = VehicleType.LIGHT_VAN
            if actual_condition == RoadCondition.FLOODED:
                flood_zones.append(node)
                vehicle_restrictions[node] = VehicleType.MOTORBIKE
            if risk > 0.65:
                severe_zones.append(node)

        # ── Publish events based on findings ─────────────────

        if weather in (WeatherCondition.HEAVY_RAIN, WeatherCondition.STORM):
            affected_edges = [
                f"{s}→{d}" for s, d, _ in route_input.edges
                if d in mud_zones or s in mud_zones
            ]
            await self.bus.publish(QuantixEvent(
                event_type=EventType.HEAVY_RAIN_DETECTED,
                source_agent="WeatherAgent",
                severity=EventSeverity.WARNING,
                payload={"weather": weather.value, "mud_nodes": mud_zones},
                affected_nodes=mud_zones,
                affected_edges=affected_edges,
                requires_fallback=len(mud_zones) > 0,
                fallback_strategy="lighter_vehicle",
                message=(
                    f"Heavy rain detected. {len(mud_zones)} node(s) transitioning to mud. "
                    "Recommend lighter vehicles on affected segments."
                ),
            ))
            self._events_published.append("HEAVY_RAIN_DETECTED")

        if mud_zones:
            mud_edges = [
                f"{s}→{d}" for s, d, _ in route_input.edges
                if d in mud_zones or s in mud_zones
            ]
            await self.bus.publish(QuantixEvent(
                event_type=EventType.MUD_ROUTE_DETECTED,
                source_agent="WeatherAgent",
                severity=EventSeverity.WARNING,
                payload={"mud_nodes": mud_zones, "count": len(mud_zones)},
                affected_nodes=mud_zones,
                affected_edges=mud_edges,
                requires_fallback=True,
                fallback_strategy="lighter_vehicle",
                message=(
                    f"{len(mud_zones)} mud zone(s) identified: {mud_zones}. "
                    "Heavy trucks risk getting stuck — switch to lighter vehicles."
                ),
            ))
            self._events_published.append("MUD_ROUTE_DETECTED")

        if flood_zones:
            flood_edges = [
                f"{s}→{d}" for s, d, _ in route_input.edges
                if d in flood_zones or s in flood_zones
            ]
            await self.bus.publish(QuantixEvent(
                event_type=EventType.FLOOD_ZONE_DETECTED,
                source_agent="WeatherAgent",
                severity=EventSeverity.CRITICAL,
                payload={"flood_nodes": flood_zones},
                affected_nodes=flood_zones,
                affected_edges=flood_edges,
                requires_fallback=True,
                fallback_strategy="alternate_route",
                message=f"FLOOD ZONE DETECTED: {flood_zones}. Route these nodes via alternate path.",
            ))
            self._events_published.append("FLOOD_ZONE_DETECTED")

        if weather == WeatherCondition.SNOW:
            await self.bus.publish(QuantixEvent(
                event_type=EventType.SNOWSTORM_DETECTED,
                source_agent="WeatherAgent",
                severity=EventSeverity.CRITICAL,
                payload={"icy_nodes": list(road_conditions.keys())},
                requires_fallback=True,
                fallback_strategy="warehouse_hold",
                message="Snowstorm active: all roads icy — warehouse hold recommended for non-essential cargo",
            ))
            self._events_published.append("SNOWSTORM_DETECTED")

        if weather == WeatherCondition.HIGH_WIND:
            await self.bus.publish(QuantixEvent(
                event_type=EventType.HIGH_WIND_DETECTED,
                source_agent="WeatherAgent",
                severity=EventSeverity.WARNING,
                payload={"wind_affected": list(risk_scores.keys())},
                requires_fallback=True,
                fallback_strategy="split_load",
                message="High winds: overloaded/tall trucks face instability risk — consider load splitting",
            ))
            self._events_published.append("HIGH_WIND_DETECTED")

        avg_risk = sum(risk_scores.values()) / max(len(risk_scores), 1)
        disruption_prob = round(1 - math.exp(-avg_risk * 2), 3)

        output = WeatherOutput(
            weather_risk_scores=risk_scores,
            road_conditions=road_conditions,
            severe_zones=severe_zones,
            mud_zones=mud_zones,
            flood_zones=flood_zones,
            disruption_probability=disruption_prob,
            current_condition=weather,
            vehicle_restrictions=vehicle_restrictions,
            analysis_summary=(
                f"Weather: {weather.value}. Mud zones: {len(mud_zones)}, "
                f"Flood zones: {len(flood_zones)}, Severe: {len(severe_zones)}. "
                f"Disruption prob: {disruption_prob:.0%}."
            ),
            events_published=self._events_published,
        )

        await self.bus.publish(QuantixEvent(
            event_type=EventType.AGENT_COMPLETED,
            source_agent="WeatherAgent",
            severity=EventSeverity.INFO,
            payload={"mud": len(mud_zones), "flood": len(flood_zones)},
            message=f"WeatherAgent done: {len(self._events_published)} event(s) published",
        ))
        return output
