"""
Quantix v3 — Agent Event Bus
Pub/sub message broker that lets every agent broadcast events and
subscribe to events from other agents. This is the nervous system
that enables real-world inter-agent communication and cascading
fallback responses.

Architecture:
  publish(event)          → appends to global log + notifies subscribers
  subscribe(type, fn)     → registers async callback
  get_events(type?)       → query event history
  get_fallback_triggers() → returns events that require rerouting
"""
from __future__ import annotations
import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from events.event_types import EventType, EventSeverity, QuantixEvent

logger = logging.getLogger("quantix.EventBus")


class AgentEventBus:
    """
    Singleton-style event bus shared across all agents in a pipeline run.
    Each orchestrator run creates a fresh bus instance.
    """

    def __init__(self):
        self._log: List[QuantixEvent] = []
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._fallback_triggers: List[QuantixEvent] = []
        self._blocked_edges: set = set()
        self._blocked_nodes: set = set()
        self._active_fallbacks: List[str] = []

    # ── Pub / Sub ─────────────────────────────────────────────

    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """Register an async or sync callback for an event type."""
        self._subscribers[event_type].append(callback)

    async def publish(self, event: QuantixEvent) -> None:
        """
        Broadcast an event. Automatically:
        - Logs it
        - Notifies all subscribers
        - Tracks blocked edges/nodes for fallback
        - Marks fallback triggers
        """
        self._log.append(event)
        icon = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨", "FALLBACK": "🔄"}.get(event.severity.value, "•")
        logger.info(f"{icon}  BUS ← {event}")

        # Track infrastructure blockages
        if event.event_type in (
            EventType.ROUTE_BLOCKED, EventType.ROAD_CLOSED,
            EventType.VEHICLE_STUCK, EventType.FLOOD_ZONE_DETECTED
        ):
            for edge in event.affected_edges:
                self._blocked_edges.add(edge)
            for node in event.affected_nodes:
                self._blocked_nodes.add(node)

        # Mark fallback triggers
        if event.requires_fallback or event.severity in (
            EventSeverity.CRITICAL, EventSeverity.FALLBACK
        ):
            self._fallback_triggers.append(event)
            if event.fallback_strategy:
                self._active_fallbacks.append(event.fallback_strategy)

        # Notify subscribers
        cbs = self._subscribers.get(event.event_type, [])
        for cb in cbs:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception as e:
                logger.error(f"Subscriber error for {event.event_type}: {e}")

    # ── Queries ───────────────────────────────────────────────

    def get_all_events(self) -> List[QuantixEvent]:
        return list(self._log)

    def get_events_by_type(self, event_type: EventType) -> List[QuantixEvent]:
        return [e for e in self._log if e.event_type == event_type]

    def get_events_by_severity(self, severity: EventSeverity) -> List[QuantixEvent]:
        return [e for e in self._log if e.severity == severity]

    def get_fallback_triggers(self) -> List[QuantixEvent]:
        return list(self._fallback_triggers)

    def is_edge_blocked(self, edge_key: str) -> bool:
        return edge_key in self._blocked_edges

    def is_node_blocked(self, node: str) -> bool:
        return node in self._blocked_nodes

    def get_blocked_edges(self) -> set:
        return set(self._blocked_edges)

    def get_blocked_nodes(self) -> set:
        return set(self._blocked_nodes)

    def get_active_fallbacks(self) -> List[str]:
        return list(self._active_fallbacks)

    def has_critical_events(self) -> bool:
        return any(
            e.severity in (EventSeverity.CRITICAL, EventSeverity.FALLBACK)
            for e in self._log
        )

    def summary(self) -> Dict[str, Any]:
        by_type: Dict[str, int] = defaultdict(int)
        for e in self._log:
            by_type[e.event_type.value] += 1
        return {
            "total_events": len(self._log),
            "fallback_triggers": len(self._fallback_triggers),
            "blocked_edges": len(self._blocked_edges),
            "blocked_nodes": len(self._blocked_nodes),
            "active_fallbacks": self._active_fallbacks,
            "event_breakdown": dict(by_type),
        }
