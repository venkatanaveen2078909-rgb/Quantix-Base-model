"""
Quantix v3 — Layer 1: Supply Chain Risk Agent
Evaluates supplier vulnerability and publishes supply-chain failure events.
LISTENS TO: TrafficAgent (blocked routes → supplier delays)
"""
from __future__ import annotations
import logging
from typing import Dict, List
from models.schemas import SupplyChainInput, SupplyChainRiskOutput, RiskLevel
from events.event_bus import AgentEventBus
from events.event_types import EventType, EventSeverity, QuantixEvent

logger = logging.getLogger("quantix.SCRiskAgent")


def _risk_level(s):
    if s < 0.25: return RiskLevel.LOW
    if s < 0.50: return RiskLevel.MEDIUM
    if s < 0.75: return RiskLevel.HIGH
    return RiskLevel.CRITICAL


class SupplyChainRiskAgent:

    def __init__(self, bus: AgentEventBus):
        self.bus = bus
        self._route_delays: float = 0.0
        self._events_published: List[str] = []
        bus.subscribe(EventType.ROUTE_BLOCKED,   self._on_route_blocked)
        bus.subscribe(EventType.VEHICLE_STUCK,   self._on_route_blocked)
        bus.subscribe(EventType.ROAD_CLOSED,     self._on_route_blocked)

    async def _on_route_blocked(self, ev: QuantixEvent):
        # Blocked routes add delay to inbound supply chain
        self._route_delays += 0.15
        logger.info(f"[SCRiskAgent] Route blockage adds supply delay factor +0.15")

    async def run(self, sc_input: SupplyChainInput) -> SupplyChainRiskOutput:
        await self.bus.publish(QuantixEvent(
            event_type=EventType.AGENT_STARTED,
            source_agent="SCRiskAgent",
            severity=EventSeverity.INFO,
            payload={"suppliers": len(sc_input.suppliers)},
            message="Analyzing supply chain risk",
        ))

        vuln: List[str] = []
        disruptions: List[str] = []
        stockout: Dict[str, float] = {}

        reliability_scores = []
        for s in sc_input.suppliers:
            rel = sc_input.supplier_reliability.get(s, 0.8)
            delay = sc_input.shipment_delays.get(s, 0.0) + self._route_delays
            adjusted = max(0.0, rel - delay * 0.4)
            reliability_scores.append(adjusted)

            if adjusted < 0.6:
                vuln.append(s)
                disruptions.append(f"Supplier '{s}' reliability dropped to {adjusted:.2f} (delays+weather)")

            stock = sc_input.warehouse_stock.get(s, 100.0)
            demand = sc_input.demand_forecasts.get(s, 80.0)
            stockout_r = max(0.0, min(1.0, (demand - stock) / max(demand, 1)))
            stockout[s] = round(stockout_r, 3)

        avg_rel = sum(reliability_scores) / max(len(reliability_scores), 1)
        sc_score = round(1.0 - avg_rel, 3)
        risk = _risk_level(sc_score)

        # Publish failure events for vulnerable suppliers
        if vuln:
            await self.bus.publish(QuantixEvent(
                event_type=EventType.SUPPLIER_FAILURE,
                source_agent="SCRiskAgent",
                severity=EventSeverity.WARNING,
                payload={"vulnerable": vuln, "count": len(vuln)},
                requires_fallback=risk in (RiskLevel.HIGH, RiskLevel.CRITICAL),
                fallback_strategy="alternate_supplier",
                message=f"{len(vuln)} vulnerable supplier(s): {vuln}. Consider backup sourcing.",
            ))
            self._events_published.append("SUPPLIER_FAILURE")

        high_stockout = [s for s, r in stockout.items() if r > 0.7]
        if high_stockout:
            await self.bus.publish(QuantixEvent(
                event_type=EventType.STOCK_SHORTAGE,
                source_agent="SCRiskAgent",
                severity=EventSeverity.WARNING,
                payload={"nodes": high_stockout},
                requires_fallback=True,
                fallback_strategy="alternate_supplier",
                message=f"Stock shortage risk at: {high_stockout}",
            ))
            self._events_published.append("STOCK_SHORTAGE")

        output = SupplyChainRiskOutput(
            supply_chain_risk_score=sc_score,
            risk_level=risk,
            vulnerable_suppliers=vuln,
            potential_disruptions=disruptions,
            stockout_risk=stockout,
            optimization_constraints={
                "preferred_suppliers": [
                    s for s in sc_input.suppliers if s not in vuln
                ],
            },
            analysis_summary=(
                f"SC Risk: {risk.value}. "
                f"Vulnerable: {len(vuln)}/{len(sc_input.suppliers)}. "
                f"Route delay factor: +{self._route_delays:.2f}."
            ),
            events_published=self._events_published,
        )
        await self.bus.publish(QuantixEvent(
            event_type=EventType.AGENT_COMPLETED,
            source_agent="SCRiskAgent",
            severity=EventSeverity.INFO,
            payload={"risk": risk.value},
            message=f"SCRiskAgent done: {risk.value}",
        ))
        return output
