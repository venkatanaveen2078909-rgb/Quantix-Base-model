import os
import sys
from typing import Any, Dict, Optional

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from solvers.unified_dispatch import UnifiedSolverDispatch
from .qubo_generator import SCQUBOGenerator
from utils.helpers import run_async

class ExecutionAgent:
    def __init__(self):
        self.solver_dispatch = UnifiedSolverDispatch()
        self.qubo_gen = SCQUBOGenerator()

    def execute(
        self,
        model_type: str,
        analysis_result: Dict[str, Any],
        qubo_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        
        distance_matrix = analysis_result["distance_matrix"]
        num_nodes = analysis_result["num_nodes"]
        
        # If quantum path and no QUBO, generate one
        qubo_obj = None
        if model_type in ["qaoa", "annealer"]:
             qubo_obj = self.qubo_gen.generate(num_nodes, distance_matrix)

        # Pass solver preference (classical, qaoa, annealer)
        result = run_async(self.solver_dispatch.solve_supply_chain(
            distance_matrix=distance_matrix,
            qubo=qubo_obj,
            preference=model_type
        ))
        
        return result
