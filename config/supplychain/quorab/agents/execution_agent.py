import os
import sys
from typing import Any, Dict, Optional

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ExecutionAgent:
    def __init__(self):
        pass

    def execute(
        self,
        model_type: str,
        analysis_result: Dict[str, Any],
        qubo: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Runs the selected solver.
        """
        if model_type == "classical":
            from solvers.classical_solver import ClassicalSolver

            solver = ClassicalSolver()
            distance_matrix = analysis_result["distance_matrix"]
            num_vehicles = 1
            depot_index = 0
            return solver.solve(distance_matrix, num_vehicles, depot_index)

        elif model_type == "qaoa":
            if qubo is None:
                raise ValueError("QUBO matrix is required for QAOA solver")
            from solvers.qaoa_solver import QAOASolver

            solver = QAOASolver()
            return solver.solve(qubo)

        elif model_type == "annealer":
            if qubo is None:
                raise ValueError("QUBO matrix is required for Annealer solver")
            from solvers.annealer_solver import AnnealerSolver

            solver = AnnealerSolver()
            return solver.solve(qubo)

        else:
            raise ValueError(f"Unknown model type: {model_type}")
