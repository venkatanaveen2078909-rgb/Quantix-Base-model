"""Quantix v3 — Complexity Agent"""
from __future__ import annotations
import logging
from models.schemas import ConstraintOutput, ComplexityReport, ComplexityClass, SolverType, SolverTier
from events.event_bus import AgentEventBus
from events.event_types import EventType, EventSeverity, QuantixEvent

logger = logging.getLogger("quantix.Complexity")

class ComplexityAgent:
    def __init__(self, bus: AgentEventBus):
        self.bus = bus

    async def run(self, constraints: ConstraintOutput) -> ComplexityReport:
        num_v = constraints.total_count * 2
        num_c = len(constraints.hard_constraints) + len(constraints.soft_constraints)
        density = round(num_c / max(num_v, 1), 2)
        
        c_class = (
            ComplexityClass.CRITICAL if num_v > 100 or num_c > 50 else
            ComplexityClass.COMPLEX  if num_v > 50 or num_c > 30 else
            ComplexityClass.MODERATE if num_v > 20 else ComplexityClass.SIMPLE
        )
        
        solver = (
            SolverType.DWAVE_LEAP if c_class == ComplexityClass.CRITICAL else
            SolverType.QAOA if c_class == ComplexityClass.COMPLEX else
            SolverType.CLASSICAL_MILP
        )
        tier = (
            SolverTier.HIGH if c_class == ComplexityClass.CRITICAL else
            SolverTier.MEDIUM if c_class == ComplexityClass.COMPLEX else SolverTier.LOW
        )

        return ComplexityReport(
            complexity_class=c_class,
            num_variables=num_v,
            num_constraints=num_c,
            num_conflicts=len(constraints.conflict_pairs),
            constraint_density=density,
            recommended_solver=solver,
            recommended_tier=tier,
            rationale=f"Classified as {c_class.value} due to {num_v} variables and {num_c} constraints. "
                      f"Conflict density: {density}."
        )
