"""
Quantix v3 — Real-World Event Types
All events agents can broadcast and subscribe to.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class EventSeverity(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"
    FALLBACK = "FALLBACK"      # triggers re-routing cascade


class EventType(str, Enum):
    # ── Weather Events ─────────────────────────────────────────
    HEAVY_RAIN_DETECTED       = "HEAVY_RAIN_DETECTED"
    MUD_ROUTE_DETECTED        = "MUD_ROUTE_DETECTED"
    FLOOD_ZONE_DETECTED       = "FLOOD_ZONE_DETECTED"
    SNOWSTORM_DETECTED        = "SNOWSTORM_DETECTED"
    HIGH_WIND_DETECTED        = "HIGH_WIND_DETECTED"
    WEATHER_CLEARED           = "WEATHER_CLEARED"

    # ── Traffic / Road Events ───────────────────────────────────
    VEHICLE_STUCK             = "VEHICLE_STUCK"          # large truck lodged on route
    ROUTE_BLOCKED             = "ROUTE_BLOCKED"          # complete blockage
    CONGESTION_SPIKE          = "CONGESTION_SPIKE"
    ACCIDENT_REPORTED         = "ACCIDENT_REPORTED"
    ROAD_CLOSED               = "ROAD_CLOSED"
    CONGESTION_CLEARED        = "CONGESTION_CLEARED"

    # ── Supply Chain Events ─────────────────────────────────────
    SUPPLIER_FAILURE          = "SUPPLIER_FAILURE"
    STOCK_SHORTAGE            = "STOCK_SHORTAGE"
    DEMAND_SURGE              = "DEMAND_SURGE"
    SUPPLIER_RESTORED         = "SUPPLIER_RESTORED"

    # ── Fleet / Vehicle Events ──────────────────────────────────
    TRUCK_BREAKDOWN           = "TRUCK_BREAKDOWN"
    TRUCK_OVERLOADED          = "TRUCK_OVERLOADED"
    DRIVER_UNAVAILABLE        = "DRIVER_UNAVAILABLE"
    FLEET_REBALANCED          = "FLEET_REBALANCED"

    # ── Risk Escalation ─────────────────────────────────────────
    RISK_ESCALATED            = "RISK_ESCALATED"
    ROUTE_REROUTED            = "ROUTE_REROUTED"
    CONSTRAINT_ADDED          = "CONSTRAINT_ADDED"
    FALLBACK_ACTIVATED        = "FALLBACK_ACTIVATED"

    # ── Trust / Compliance ──────────────────────────────────────
    SUPPLIER_BLACKLISTED      = "SUPPLIER_BLACKLISTED"
    COMPLIANCE_VIOLATION      = "COMPLIANCE_VIOLATION"

    # ── System ──────────────────────────────────────────────────
    AGENT_STARTED             = "AGENT_STARTED"
    AGENT_COMPLETED           = "AGENT_COMPLETED"
    PIPELINE_CHECKPOINT       = "PIPELINE_CHECKPOINT"


@dataclass
class QuantixEvent:
    """Single event message passed between agents."""
    event_type: EventType
    source_agent: str
    severity: EventSeverity
    payload: Dict[str, Any]
    affected_nodes: List[str] = field(default_factory=list)
    affected_edges: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    event_id: str = field(default_factory=lambda: __import__("uuid").uuid4().hex[:8])
    requires_fallback: bool = False
    fallback_strategy: Optional[str] = None     # e.g. "alternate_route", "lighter_vehicle"
    message: str = ""

    def __str__(self) -> str:
        return (
            f"[{self.severity.value}] {self.source_agent} → {self.event_type.value} "
            f"| nodes={self.affected_nodes} | {self.message}"
        )
