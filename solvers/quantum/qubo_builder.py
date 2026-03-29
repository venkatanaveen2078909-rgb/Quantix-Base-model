"""Quantix v3 — QUBO Builder"""
from __future__ import annotations
import logging
from typing import Dict, Tuple
from models.schemas import ConstraintOutput, ComplexityReport, QUBOProblem
from events.event_bus import AgentEventBus
from events.event_types import EventType, EventSeverity, QuantixEvent

logger = logging.getLogger("quantix.QUBOBuilder")

class QUBOBuilder:
    def __init__(self, bus: AgentEventBus):
        self.bus = bus

    async def run(self, constraints: ConstraintOutput, report: ComplexityReport) -> QUBOProblem:
        qubo: Dict[Tuple[int, int], float] = {}
        v_map: Dict[str, int] = {}
        
        # Map constraints to binary variables
        idx = 0
        for name in constraints.hard_constraints:
            v_map[name] = idx; idx += 1
        for name in constraints.soft_constraints:
            if name not in v_map:
                v_map[name] = idx; idx += 1
        
        # Simple QUBO formulation: linear weights on diagonal
        for name, weight in constraints.penalty_weights.items():
            if name in v_map:
                v_idx = v_map[name]
                qubo[(v_idx, v_idx)] = weight
        
        # Quadratic penalties for conflicts
        for c1, c2 in constraints.conflict_pairs:
            if c1 in v_map and c2 in v_map:
                i, j = v_map[c1], v_map[c2]
                qubo[(min(i,j), max(i,j))] = 50.0

        return QUBOProblem(
            qubo_matrix=qubo,
            variable_map=v_map,
            num_variables=len(v_map),
            metadata={"solver": report.recommended_solver.value, "tier": report.recommended_tier.value}
        )
