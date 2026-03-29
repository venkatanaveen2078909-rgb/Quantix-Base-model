"""
Quantix v3 — Layer 1: Risk Intelligence Agent
Merges traffic + weather + historical data into unified route risk scores.
Reacts to stuck-vehicle and road-blocked events by escalating affected edges.
LISTENS TO: TrafficAgent (stuck vehicles → risk spike)
            WeatherAgent (flood/mud → risk spike)
PUBLISHES TO: ConstraintBuilder (high-risk edges as hard constraints)
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import RouteInput, RouteRiskOutput, RiskLevel
from events.event_bus import AgentEventBus
from events.event_types import EventType, EventSeverity, QuantixEvent

logger = logging.getLogger("quantix.RiskAgent")


def _risk_level(score: float) -> RiskLevel:
    if score < 0.25: return RiskLevel.LOW
    if score < 0.50: return RiskLevel.MEDIUM
    if score < 0.75: return RiskLevel.HIGH
    return RiskLevel.CRITICAL


class RiskIntelligenceAgent:

    def __init__(self, bus: AgentEventBus):
        self.bus = bus
        self.llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Quantix Risk Intelligence AI. Analyze logistical risks."),
            ("human", "Route Data: {data}. Provide a 1-sentence qualitative summary.")
        ])
        self._escalated_edges: Dict[str, float] = {}
        self._events_published: List[str] = []

        bus.subscribe(EventType.VEHICLE_STUCK,    self._on_critical_event)
        bus.subscribe(EventType.ROUTE_BLOCKED,    self._on_critical_event)
        bus.subscribe(EventType.ROAD_CLOSED,      self._on_critical_event)
        bus.subscribe(EventType.FLOOD_ZONE_DETECTED, self._on_critical_event)
        bus.subscribe(EventType.ACCIDENT_REPORTED,self._on_critical_event)

    async def _on_critical_event(self, ev: QuantixEvent):
        for edge in ev.affected_edges:
            self._escalated_edges[edge] = max(
                self._escalated_edges.get(edge, 0.0), 0.95
            )
        logger.warning(
            f"[RiskAgent] Escalating risk on {ev.affected_edges} "
            f"due to {ev.event_type.value}"
        )

    async def run(
        self,
        route_input: RouteInput,
        traffic_out,
        weather_out,
    ) -> RouteRiskOutput:
        await self.bus.publish(QuantixEvent(
            event_type=EventType.AGENT_STARTED,
            source_agent="RiskAgent",
            severity=EventSeverity.INFO,
            payload={},
            message="Computing unified route risk scores",
        ))

        risk_scores: Dict[str, float] = {}
        high_risk: List[str] = []
        mud_prone: List[str] = []
        flood_risk: List[str] = []

        for src, dst, dist in route_input.edges:
            edge_key = f"{src}→{dst}"

            traffic   = traffic_out.congestion_scores.get(edge_key, 0.1)
            weather   = weather_out.weather_risk_scores.get(dst, 0.1)
            delay     = route_input.historical_delays.get(dst, 0.1)
            escalated = self._escalated_edges.get(edge_key, 0.0)

            # Weighted composite + any escalation from bus events
            base = traffic * 0.35 + weather * 0.35 + delay * 0.15 + escalated * 0.15
            risk = round(min(base, 1.0), 3)
            risk_scores[edge_key] = risk

            if risk > 0.60:
                high_risk.append(edge_key)
            if dst in weather_out.mud_zones:
                mud_prone.append(edge_key)
            if dst in weather_out.flood_zones:
                flood_risk.append(dst)

        avg_risk = sum(risk_scores.values()) / max(len(risk_scores), 1)
        overall = _risk_level(avg_risk)

        # Publish escalation for critical risk
        if overall in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            await self.bus.publish(QuantixEvent(
                event_type=EventType.RISK_ESCALATED,
                source_agent="RiskAgent",
                severity=EventSeverity.CRITICAL if overall == RiskLevel.CRITICAL else EventSeverity.WARNING,
                payload={"overall_risk": overall.value, "high_risk_count": len(high_risk)},
                affected_edges=high_risk,
                requires_fallback=overall == RiskLevel.CRITICAL,
                fallback_strategy="alternate_route",
                message=(
                    f"Overall route risk {overall.value}: {len(high_risk)} high-risk segments. "
                    "Constraint builder should add penalties."
                ),
            ))
            self._events_published.append("RISK_ESCALATED")

        # Build alternate routes (top risk-avoiding paths)
        alt_routes = [
            [route_input.depot] +
            [n for n in route_input.nodes if n != route_input.depot
             and risk_scores.get(f"{route_input.depot}→{n}", 1.0) < 0.5]
            for _ in range(2)
        ][:2]

        output = RouteRiskOutput(
            route_risk_scores=risk_scores,
            overall_risk_level=overall,
            alternate_routes=[r for r in alt_routes if len(r) > 1],
            high_risk_edges=high_risk,
            mud_prone_edges=mud_prone,
            flood_risk_nodes=flood_risk,
            optimization_constraints={
                "time_penalties": {
                    n: risk_scores.get(f"{route_input.depot}→{n}", 0.1) * 50
                    for n in route_input.nodes
                },
                "hard_block_edges": list(self.bus.get_blocked_edges()),
            },
            analysis_summary=(
                f"Risk: {overall.value}. High-risk segments: {len(high_risk)}, "
                f"mud-prone: {len(mud_prone)}, flood-risk nodes: {len(flood_risk)}."
            ),
            events_published=self._events_published,
        )

        await self.bus.publish(QuantixEvent(
            event_type=EventType.AGENT_COMPLETED,
            source_agent="RiskAgent",
            severity=EventSeverity.INFO,
            payload={"risk": overall.value},
            message=f"RiskAgent done: {overall.value} — {len(high_risk)} high-risk edges",
        ))
        return output
