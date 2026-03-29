from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from solvers.classical_solver import ClassicalSolver
from solvers.qaoa_solver import QAOASolver
from solvers.annealer_solver import AnnealerSolver
from utils.logging_utils import get_logger
from quantix_types.solver_types import SolverOutput


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

        # ✅ Correct solver initialization
        self.classical_solver = ClassicalSolver()
        self.qaoa_solver = QAOASolver()
        self.annealer_solver = AnnealerSolver()

    def execute(
        self,
        solver_key: str,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        constraints: Dict[str, Any],
        alpha_signs: np.ndarray,
        qubo: Optional[Dict] = None,
    ) -> ExecutionResult:

        # ---------------- CLASSICAL ----------------
        if solver_key == "classical":
            raw: SolverOutput = self.classical_solver.solve(
                expected_returns, covariance, constraints, alpha_signs
            )

            return ExecutionResult(
                weights=raw["weights"],
                solver_used="classical",
                objective_value=raw["objective_value"],
                performance_metrics={
                    "time_taken": raw["runtime"],
                    "iterations": raw["iterations"],
                    "backend": raw["backend"],
                    "status": raw["status"],
                },
                bitstring=None,
            )

        # ---------------- VALIDATION ----------------
        if qubo is None:
            raise ValueError("qubo must be provided for qaoa and annealer solvers")

        n_vars = len(expected_returns)

        # ---------------- QAOA ----------------
        if solver_key == "qaoa":
            raw: SolverOutput = self.qaoa_solver.solve(qubo, n_vars, alpha_signs)
            solver_used = "hybrid_qaoa"

        # ---------------- ANNEALER ----------------
        elif solver_key == "annealer":
            raw: SolverOutput = self.annealer_solver.solve(qubo, n_vars, alpha_signs)
            solver_used = "quantum_annealer"

        else:
            raise ValueError(f"unsupported solver key: {solver_key}")

        return ExecutionResult(
            weights=raw["weights"],
            solver_used=solver_used,
            objective_value=raw["objective_value"],
            performance_metrics={
                "time_taken": raw["runtime"],
                "iterations": raw["iterations"],
                "backend": raw["backend"],
                "status": raw["status"],
            },
            bitstring=raw["bitstring"],
        )