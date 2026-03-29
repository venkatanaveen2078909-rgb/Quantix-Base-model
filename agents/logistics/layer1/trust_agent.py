"""Quantix v3 — Layer 1: Trust Agent"""
from __future__ import annotations
import logging
from typing import List
from models.schemas import SupplyChainInput, CostInput, TrustOutput
from events.event_bus import AgentEventBus
from events.event_types import EventType, EventSeverity, QuantixEvent

logger = logging.getLogger("quantix.TrustAgent")

class TrustAgent:
    def __init__(self, bus: AgentEventBus):
        self.bus = bus
        self._events_published: List[str] = []

    async def run(self, sc: SupplyChainInput, ci: CostInput) -> TrustOutput:
        supplier_trust, flags, audit, blacklisted = {}, [], [], []

        for s in sc.suppliers:
            rel = sc.supplier_reliability.get(s, 0.8)
            delay = sc.shipment_delays.get(s, 0.0)
            trust = round(max(0.0, rel - delay * 0.3), 3)
            supplier_trust[s] = trust
            if trust < 0.6:
                flags.append(f"Supplier '{s}' trust LOW ({trust:.2f})")
                audit.append(f"[AUDIT] Flag '{s}' for review")
            if trust < 0.4:
                blacklisted.append(s)
                await self.bus.publish(QuantixEvent(
                    event_type=EventType.SUPPLIER_BLACKLISTED,
                    source_agent="TrustAgent",
                    severity=EventSeverity.CRITICAL,
                    payload={"supplier": s, "trust": trust},
                    requires_fallback=True,
                    fallback_strategy="alternate_supplier",
                    message=f"Supplier '{s}' BLACKLISTED (trust={trust:.2f})",
                ))
                self._events_published.append("SUPPLIER_BLACKLISTED")

        driver_trust = {
            f"driver_{i+1}": round(min(0.85 + (abs(hash(f"d{i}")) % 100) / 1000, 1.0), 3)
            for i in range(ci.num_trucks)
        }
        overall = round(
            (sum(supplier_trust.values()) + sum(driver_trust.values()))
            / max(len(supplier_trust) + len(driver_trust), 1), 3
        )

        return TrustOutput(
            driver_trust=driver_trust,
            supplier_trust=supplier_trust,
            compliance_flags=flags,
            blacklisted=blacklisted,
            overall_trust=overall,
            audit_trail=audit,
            analysis_summary=(
                f"Trust: {overall:.2f}. "
                f"Flags: {len(flags)}. Blacklisted: {len(blacklisted)}."
            ),
            events_published=self._events_published,
        )
