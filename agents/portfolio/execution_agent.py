from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from solvers.unified_dispatch import UnifiedSolverDispatch
from utils.helpers import get_logger


@dataclass
class ExecutionResult:
    weights: List[float]
    solver_used: str
    objective_value: float
    performance_metrics: Dict[str, Any]
    bitstring: Optional[List[int]]


class ExecutionAgent:
    """Run the selected solver backend and normalize output shape."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

        # ✅ Unified solver initialization
        self.solver_dispatch = UnifiedSolverDispatch()

    def execute(
        self,
        solver_key: str,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        constraints: Dict[str, Any],
        alpha_signs: np.ndarray,
        qubo: Optional[Dict] = None,
    ) -> ExecutionResult:

        # Pass solver preference (classical, qaoa, annealer)
        from utils.helpers import run_async
        raw_result = run_async(self.solver_dispatch.solve_portfolio(
            expected_returns=expected_returns,
            covariance=covariance,
            constraints=constraints,
            alpha_signs=alpha_signs,
            qubo=qubo,
            preference=solver_key
        ))
        
        # Normalize output for ExecutionResult
        return ExecutionResult(
            weights=raw_result.get("weights", []),
            solver_used=raw_result.get("solver_used", "unknown"),
            objective_value=raw_result.get("objective_value", raw_result.get("cost", 0.0)),
            performance_metrics=raw_result.get("performance_metrics", {}),
            bitstring=raw_result.get("bitstring") or raw_result.get("solution")
        )