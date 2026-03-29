"""
Quantix v3 — Layer 1: Constraint Builder Agent
Aggregates ALL intelligence outputs (including bus events) into
a unified constraint matrix for the quantum optimizer.
This agent is the LAST in the parallel layer — it reads the full event log.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Tuple
from models.schemas import (
    RouteInput, CostInput, PlannerOutput, RouteRiskOutput,
    TrafficOutput, WeatherOutput, TrustOutput, DemandOutput, ConstraintOutput
)
from events.event_bus import AgentEventBus
from events.event_types import EventType, EventSeverity, QuantixEvent

logger = logging.getLogger("quantix.ConstraintBuilder")


class ConstraintBuilderAgent:

    def __init__(self, bus: AgentEventBus):
        self.bus = bus

    async def run(
        self,
        route_input: RouteInput,
        cost_input: CostInput,
        planner: PlannerOutput,
        risk: RouteRiskOutput,
        traffic: TrafficOutput,
        weather: WeatherOutput,
        trust: TrustOutput,
        demand: DemandOutput,
        fallback_plan: Any,
    ) -> ConstraintOutput:
        await self.bus.publish(QuantixEvent(
            event_type=EventType.AGENT_STARTED,
            source_agent="ConstraintBuilder",
            severity=EventSeverity.INFO,
            payload={},
            message="Assembling constraint matrix from all intelligence outputs",
        ))

        hard: Dict[str, Any] = {}
        soft: Dict[str, Any] = {}
        penalties: Dict[str, float] = {}
        fallback_pen: Dict[str, float] = {}
        conflicts: List[Tuple[str, str]] = []

        # ── Hard constraints from planner ───────────────────
        for i, hc in enumerate(planner.hard_constraints):
            hard[f"planner_{i}"] = hc

        hard["truck_capacity"] = cost_input.truck_capacity
        hard["num_trucks"] = cost_input.num_trucks

        # ── Time window deadlines ─────────────────────────
        for node, dl in route_input.delivery_deadlines.items():
            hard[f"deadline_{node}"] = dl
            penalties[f"deadline_violation_{node}"] = 60.0 / max(dl, 0.1)

        # ── Hard blocks from bus (stuck vehicles, flood, closure) ──
        for edge in self.bus.get_blocked_edges():
            hard[f"hard_block_{edge}"] = True
            penalties[f"hard_block_{edge}"] = 9999.0
            await self.bus.publish(QuantixEvent(
                event_type=EventType.CONSTRAINT_ADDED,
                source_agent="ConstraintBuilder",
                severity=EventSeverity.INFO,
                payload={"constraint": f"hard_block_{edge}"},
                affected_edges=[edge],
                message=f"Hard constraint added: edge {edge} BLOCKED",
            ))

        for node in self.bus.get_blocked_nodes():
            hard[f"hard_block_node_{node}"] = True
            penalties[f"hard_block_node_{node}"] = 9999.0

        # ── Soft: traffic congestion ─────────────────────
        for edge, cong in traffic.congestion_scores.items():
            if cong > 0.45:
                soft[f"traffic_{edge}"] = cong
                penalties[f"traffic_{edge}"] = cong * 25.0

        # ── Soft: weather risk ───────────────────────────
        for node, wr in weather.weather_risk_scores.items():
            if wr > 0.35:
                soft[f"weather_{node}"] = wr
                penalties[f"weather_{node}"] = wr * 20.0

        # ── Soft: route risk ─────────────────────────────
        for edge, risk_s in risk.route_risk_scores.items():
            soft[f"risk_{edge}"] = risk_s
            penalties[f"risk_{edge}"] = risk_s * 30.0 * planner.weight_risk

        # ── Conflict detection ───────────────────────────
        for edge in risk.high_risk_edges:
            dst = edge.split("→")[-1]
            if dst in weather.flood_zones:
                conflicts.append((f"high_risk_{edge}", f"flood_zone_{dst}"))
            if edge in traffic.stuck_vehicles:
                conflicts.append((f"high_risk_{edge}", f"vehicle_stuck_{edge}"))

        # ── Trust blacklists ─────────────────────────────
        for s in trust.blacklisted:
            hard[f"blacklist_{s}"] = True
            penalties[f"blacklist_{s}"] = 9999.0

        # ── Demand priority rewards ──────────────────────
        for node in demand.priority_deliveries:
            soft[f"priority_{node}"] = True
            penalties[f"priority_{node}"] = -15.0   # negative = reward

        # ── Fallback plan penalties ──────────────────────
        if fallback_plan:
            # Check if fallback_plan is a dict or FallbackPlan object
            penalty_map = fallback_plan.get("penalty_map", {}) if isinstance(fallback_plan, dict) else getattr(fallback_plan, 'penalty_map', {})
            for edge, mult in penalty_map.items():
                fallback_pen[edge] = mult * 10.0
                penalties[edge] = max(penalties.get(edge, 0), fallback_pen[edge])

        total = len(hard) + len(soft)

        output = ConstraintOutput(
            hard_constraints=hard,
            soft_constraints=soft,
            penalty_weights=penalties,
            fallback_penalties=fallback_pen,
            conflict_pairs=conflicts,
            total_count=total,
            constraint_summary=(
                f"Total constraints: {total} ({len(hard)} hard, {len(soft)} soft). "
                f"Conflicts: {len(conflicts)}. "
                f"Fallback penalties applied: {len(fallback_pen)}. "
                f"Hard-blocked edges: {len(self.bus.get_blocked_edges())}."
            ),
        )

        await self.bus.publish(QuantixEvent(
            event_type=EventType.AGENT_COMPLETED,
            source_agent="ConstraintBuilder",
            severity=EventSeverity.INFO,
            payload={"total": total, "conflicts": len(conflicts)},
            message=f"Constraint matrix ready: {total} constraints, {len(conflicts)} conflicts",
        ))
        return output
