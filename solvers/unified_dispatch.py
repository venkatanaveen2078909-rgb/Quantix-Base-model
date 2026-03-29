"""
Unified Solver Dispatch (v3)
Handles Classical, Hybrid QAOA, and D-Wave paths for all domains.
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Tuple

from models.schemas import SolverType, SolverTier, SolverDecision, QUBOProblem
from utils.helpers import get_logger

logger = get_logger("UnifiedSolver")

# Import solvers lazily to avoid heavy dependency load if not used
class UnifiedSolverDispatch:
    def __init__(self):
        from solvers.quantum.adaptive_solver import AdaptiveSolverDispatch
        self.quantum_dispatch = AdaptiveSolverDispatch()

    async def solve_logistics(
        self, 
        distance_matrix: List[List[float]], 
        num_vehicles: int, 
        depot_index: int,
        qubo: Optional[QUBOProblem] = None,
        preference: Optional[str] = None
    ) -> Dict[str, Any]:
        
        # 1. Map string preference to SolverType enum
        pref_enum = None
        if preference:
            if preference.lower() in ["classical", "milp"]: pref_enum = SolverType.CLASSICAL_MILP
            elif preference.lower() == "qaoa": pref_enum = SolverType.QAOA
            elif preference.lower() in ["dwave", "annealer", "leap"]: pref_enum = SolverType.DWAVE_LEAP
            elif preference.lower() == "sa": pref_enum = SolverType.SIMULATED_ANNEALING

        num_vars = qubo.num_variables if qubo else len(distance_matrix)**2
        
        # 2. Decide solver path
        if pref_enum == SolverType.CLASSICAL_MILP or (not pref_enum and num_vars < 20):
            from solvers.logistics.vrp_solver import ClassicalSolver
            solver = ClassicalSolver()
            result = solver.solve(distance_matrix, num_vehicles, depot_index)
            result["solver_used"] = SolverType.CLASSICAL_MILP.value
            return result
        
        if not qubo:
             raise ValueError("QUBO must be provided for Quantum/Hybrid paths in Logistics")

        # Use Adaptive Dispatch for QAOA/D-Wave
        sample, energy, decision, elapsed = self.quantum_dispatch.solve(qubo, domain="logistics", preference=pref_enum)
        return {
            "solution": sample,
            "cost": energy,
            "solver_used": decision.solver.value,
            "performance_metrics": {
                "time_taken": elapsed / 1000.0,
                "tier": decision.tier.value,
                "reason": decision.reason
            }
        }

    async def solve_supply_chain(
        self, 
        distance_matrix: List[List[float]], 
        qubo: Optional[QUBOProblem] = None,
        preference: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. Map string preference to SolverType enum
        pref_enum = None
        if preference:
            if preference.lower() in ["classical", "milp"]: pref_enum = SolverType.CLASSICAL_MILP
            elif preference.lower() == "qaoa": pref_enum = SolverType.QAOA
            elif preference.lower() in ["dwave", "annealer", "leap"]: pref_enum = SolverType.DWAVE_LEAP
            elif preference.lower() == "sa": pref_enum = SolverType.SIMULATED_ANNEALING

        num_vars = qubo.num_variables if qubo else len(distance_matrix)**2
        
        # 2. Decide solver path
        if pref_enum == SolverType.CLASSICAL_MILP or (not pref_enum and num_vars < 20):
            from solvers.logistics.vrp_solver import ClassicalSolver 
            solver = ClassicalSolver()
            return solver.solve(distance_matrix, 1, 0)
            
        if not qubo:
            raise ValueError("QUBO must be provided for Quantum/Hybrid paths in Supply Chain")

        sample, energy, decision, elapsed = self.quantum_dispatch.solve(qubo, domain="supply_chain", preference=pref_enum)
        return {
            "solution": sample,
            "cost": energy,
            "solver_used": decision.solver.value,
            "performance_metrics": {
                "time_taken": elapsed / 1000.0,
                "tier": decision.tier.value,
                "reason": decision.reason
            }
        }

    async def solve_portfolio(
        self,
        expected_returns: Any,
        covariance: Any,
        constraints: Dict[str, Any],
        alpha_signs: Any,
        qubo: Optional[QUBOProblem] = None,
        preference: Optional[str] = None
    ) -> Dict[str, Any]:
        
        # 1. Map string preference to SolverType enum
        pref_enum = None
        if preference:
            if preference.lower() in ["classical", "milp", "slsqp"]: pref_enum = SolverType.CLASSICAL_MILP
            elif preference.lower() == "qaoa": pref_enum = SolverType.QAOA
            elif preference.lower() in ["dwave", "annealer", "leap"]: pref_enum = SolverType.DWAVE_LEAP
            elif preference.lower() == "sa": pref_enum = SolverType.SIMULATED_ANNEALING

        num_vars = qubo.num_variables if qubo else len(expected_returns)
        
        # 2. Decide solver path
        if pref_enum == SolverType.CLASSICAL_MILP or (not pref_enum and num_vars < 20):
            from solvers.financial.portfolio_solver import ClassicalSolver
            solver = ClassicalSolver()
            raw = solver.solve(expected_returns, covariance, constraints, alpha_signs)
            raw["solver_used"] = SolverType.CLASSICAL_MILP.value
            return raw

        if not qubo:
            raise ValueError("QUBO must be provided for Quantum/Hybrid paths in Portfolio")

        sample, energy, decision, elapsed = self.quantum_dispatch.solve(qubo, domain="portfolio", preference=pref_enum)
        
        return {
            "solution": sample,
            "cost": energy,
            "solver_used": decision.solver.value,
            "performance_metrics": {
                "time_taken": elapsed / 1000.0,
                "tier": decision.tier.value,
                "reason": decision.reason
            }
        }
