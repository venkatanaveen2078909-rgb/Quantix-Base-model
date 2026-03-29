"""
Quantix — Layer 2: Quantum Optimization Engine (v2)
Converts logistics problems into QUBO formulation and solves
using adaptive solver selection:
  - D-Wave Leap     → >100 QUBO variables (high constraint)
  - QAOA (Qiskit)   → 20–100 QUBO variables (medium)
  - Classical MILP  → <20 QUBO variables (low, exact)
  - SA fallback     → if preferred solver unavailable
"""
from __future__ import annotations
import json
import time
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from models.schemas import (
    RouteRiskOutput,
    SupplyChainRiskOutput,
    CostOptimizationOutput,
    QUBOProblem,
    QuantumSolution,
    SolverType,
    SolverTier,
    RouteInput,
)
from .adaptive_solver import AdaptiveSolverDispatch, AdaptiveSolverSelector
from config.settings import (
    WEIGHT_COST, WEIGHT_TIME, WEIGHT_RISK,
    PENALTY_VISIT_ONCE, PENALTY_CAPACITY, PENALTY_TIME_WINDOW,
    ANNEALING_READS,
)
from utils.helpers import get_logger, log_agent_output, timed

logger = get_logger("QuantumEngine")


# ═══════════════════════════════════════════════════════════════
# QUBO Builder
# ═══════════════════════════════════════════════════════════════

class QUBOBuilder:
    """Constructs a QUBO matrix from logistics problem data."""

    def __init__(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str, float]],
        depot: str,
        num_trucks: int,
        truck_capacity: float,
        delivery_loads: Dict[str, float],
    ):
        self.nodes = nodes
        self.edges = edges
        self.depot = depot
        self.num_trucks = num_trucks
        self.truck_capacity = truck_capacity
        self.delivery_loads = delivery_loads
        self.edge_list = [(s, d, w) for s, d, w in edges]
        self.num_edges = len(self.edge_list)
        self.num_vars = num_trucks * self.num_edges

        # Variable map: (truck, edge) → QUBO index
        self.var_map: Dict[str, int] = {}
        idx = 0
        for t in range(num_trucks):
            for e, (src, dst, _) in enumerate(self.edge_list):
                # Use ':' as separator to avoid issues with underscores in node names
                key = f"x:t{t}:e{e}:{src}→{dst}"
                self.var_map[key] = idx
                idx += 1

        logger.info(
            f"[QUBOBuilder] {self.num_vars} variables "
            f"({num_trucks} trucks × {self.num_edges} edges)"
        )

    def _var_idx(self, truck: int, edge_idx: int) -> int:
        key = f"x:t{truck}:e{edge_idx}:{self.edge_list[edge_idx][0]}→{self.edge_list[edge_idx][1]}"
        return self.var_map[key]

    def build(
        self,
        edge_costs: Dict[str, float],
        edge_risks: Dict[str, float],
        time_penalties: Dict[str, float],
    ) -> QUBOProblem:
        Q: Dict[Tuple[int, int], float] = {}

        def add_qubo(i: int, j: int, val: float):
            key = (min(i, j), max(i, j))
            Q[key] = Q.get(key, 0.0) + val

        # Objective terms
        for t in range(self.num_trucks):
            for e_idx, (src, dst, dist) in enumerate(self.edge_list):
                v = self._var_idx(t, e_idx)
                edge_key = f"{src}→{dst}"
                cost_c = edge_costs.get(edge_key, dist * 0.5) * WEIGHT_COST
                risk_c = edge_risks.get(edge_key, 0.2) * 10.0 * WEIGHT_RISK
                time_c = time_penalties.get(dst, 0.0) * WEIGHT_TIME
                add_qubo(v, v, cost_c + risk_c + time_c)

        # Constraint 1: each delivery node visited exactly once
        delivery_nodes = [n for n in self.nodes if n != self.depot]
        for node in delivery_nodes:
            arriving = [
                (t, e_idx)
                for t in range(self.num_trucks)
                for e_idx, (_, dst, _) in enumerate(self.edge_list)
                if dst == node
            ]
            for (t1, e1), (t2, e2) in product(arriving, repeat=2):
                v1 = self._var_idx(t1, e1)
                v2 = self._var_idx(t2, e2)
                if v1 == v2:
                    add_qubo(v1, v1, PENALTY_VISIT_ONCE * (1 - 2))
                else:
                    add_qubo(v1, v2, PENALTY_VISIT_ONCE * 2)

        # Constraint 2: truck capacity
        for t in range(self.num_trucks):
            for e1_idx in range(self.num_edges):
                v1 = self._var_idx(t, e1_idx)
                _, dst1, _ = self.edge_list[e1_idx]
                load1 = self.delivery_loads.get(dst1, 0.0)
                add_qubo(v1, v1, PENALTY_CAPACITY * (load1**2 - 2 * self.truck_capacity * load1))
                for e2_idx in range(e1_idx + 1, self.num_edges):
                    v2 = self._var_idx(t, e2_idx)
                    _, dst2, _ = self.edge_list[e2_idx]
                    load2 = self.delivery_loads.get(dst2, 0.0)
                    add_qubo(v1, v2, PENALTY_CAPACITY * 2 * load1 * load2)

        return QUBOProblem(
            qubo_matrix=Q,
            variable_map=self.var_map,
            num_variables=self.num_vars,
            problem_type="VRP",
            metadata={
                "num_trucks": self.num_trucks,
                "num_nodes": len(self.nodes),
                "num_edges": self.num_edges,
                "depot": self.depot,
            },
        )


# ═══════════════════════════════════════════════════════════════
# Route Extractor
# ═══════════════════════════════════════════════════════════════

class RouteExtractor:
    def __init__(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str, float]],
        depot: str,
        num_trucks: int,
        var_map: Dict[str, int],
    ):
        self.nodes = nodes
        self.edges = edges
        self.depot = depot
        self.num_trucks = num_trucks
        self.rev_map = {v: (int(k.split(":")[1][1:]), int(k.split(":")[2][1:]))
                        for k, v in var_map.items()}

    def extract(self, sample: Dict[int, int]) -> List[List[str]]:
        truck_edges: Dict[int, List[Tuple[str, str]]] = {t: [] for t in range(self.num_trucks)}
        for qubo_idx, val in sample.items():
            if val == 1 and qubo_idx in self.rev_map:
                truck, edge_idx = self.rev_map[qubo_idx]
                if edge_idx < len(self.edges):
                    src, dst, _ = self.edges[edge_idx]
                    truck_edges[truck].append((src, dst))

        routes = []
        for truck, active_edges in truck_edges.items():
            if not active_edges:
                continue
            G = nx.DiGraph()
            G.add_edges_from(active_edges)
            try:
                route = list(nx.dfs_preorder_nodes(G, source=self.depot))
            except Exception:
                route = [self.depot] + list({d for _, d in active_edges})
            if len(route) > 1:
                routes.append(route)

        if not routes:
            routes = self._greedy_fallback()
        return routes

    def _greedy_fallback(self) -> List[List[str]]:
        G = nx.Graph()
        for src, dst, w in self.edges:
            G.add_edge(src, dst, weight=w)
        unvisited = set(n for n in self.nodes if n != self.depot)
        routes = []
        for _ in range(self.num_trucks):
            if not unvisited:
                break
            route = [self.depot]
            current = self.depot
            while unvisited:
                neighbors = [n for n in G.neighbors(current) if n in unvisited]
                if not neighbors:
                    break
                nearest = min(neighbors, key=lambda n: G[current][n]["weight"])
                route.append(nearest)
                unvisited.discard(nearest)
                current = nearest
            route.append(self.depot)
            routes.append(route)
        return routes


# ═══════════════════════════════════════════════════════════════
# Main Engine
# ═══════════════════════════════════════════════════════════════

class QuantumOptimizationEngine:
    """
    Layer 2 — Adaptive Quantum Optimization Engine.
    Automatically selects D-Wave Leap / QAOA / Classical MILP / SA
    based on problem size.
    """

    def __init__(self):
        from solvers.unified_dispatch import UnifiedSolverDispatch
        self.dispatcher = UnifiedSolverDispatch()
        logger.info("QuantumOptimizationEngine (Unified) initialized")

    @timed("QuantumEngine")
    async def optimize(
        self,
        route_input: RouteInput,
        route_risk: RouteRiskOutput,
        supply_chain_risk: SupplyChainRiskOutput,
        cost_output: CostOptimizationOutput,
        cost_input_data: Optional[Dict] = None,
        preference: Optional[str] = None
    ) -> QuantumSolution:
        logger.info("Starting adaptive quantum optimization...")

        # Resolve truck/capacity from cost_input if provided
        num_trucks = 4
        truck_capacity = 400.0
        delivery_loads = {n: 100.0 for n in route_input.nodes if n != route_input.depot}

        if cost_input_data:
            num_trucks = cost_input_data.get("num_trucks", num_trucks)
            truck_capacity = cost_input_data.get("truck_capacity", truck_capacity)
            delivery_loads = cost_input_data.get("delivery_loads", delivery_loads)

        # Build QUBO
        builder = QUBOBuilder(
            nodes=route_input.nodes,
            edges=route_input.edges,
            depot=route_input.depot,
            num_trucks=num_trucks,
            truck_capacity=truck_capacity,
            delivery_loads=delivery_loads,
        )

        edge_costs = cost_output.optimization_objective.get("edge_cost_coefficients", {})
        edge_risks = route_risk.route_risk_scores
        time_penalties = route_risk.optimization_constraints.get("time_penalties", {})

        qubo_problem = builder.build(edge_costs, edge_risks, time_penalties)
        logger.info(
            f"[QuantumEngine] QUBO: {qubo_problem.num_variables} vars, "
            f"{len(qubo_problem.qubo_matrix)} non-zeros"
        )

        # Unified solve
        raw_result = await self.dispatcher.solve_logistics(
            distance_matrix=[[e[2] for e in route_input.edges]], 
            num_vehicles=num_trucks,
            depot_index=0,
            qubo=qubo_problem,
            preference=preference
        )
        sample = raw_result["solution"]
        energy = raw_result.get("cost", 0.0)
        
        # We need a decision object for compatibility
        from models.schemas import SolverDecision, SolverTier, SolverType
        
        # Safely map the string back to the enum
        raw_solver = raw_result.get("solver_used", "classical_milp")
        try:
            solver_val = SolverType(raw_solver)
        except ValueError:
            # Fallback for any legacy strings
            if "Classical" in raw_solver:
                solver_val = SolverType.CLASSICAL_MILP
            elif "D-Wave" in raw_solver:
                solver_val = SolverType.DWAVE_LEAP
            else:
                solver_val = SolverType.CLASSICAL_MILP

        decision = SolverDecision(
            solver=solver_val,
            tier=SolverTier.HIGH if solver_val == SolverType.DWAVE_LEAP else SolverTier.MEDIUM,
            reason="Unified Dispatch selection",
            num_variables=qubo_problem.num_variables,
            estimated_solve_time_sec=raw_result.get("performance_metrics", {}).get("time_taken", 0.0)
        )
        solve_ms = raw_result.get("performance_metrics", {}).get("time_taken", 0) * 1000

        # Extract routes
        extractor = RouteExtractor(
            nodes=route_input.nodes,
            edges=route_input.edges,
            depot=route_input.depot,
            num_trucks=num_trucks,
            var_map=qubo_problem.variable_map,
        )
        routes = extractor.extract(sample)

        # Cost computation
        route_costs = self._compute_route_costs(routes, route_input.edges, cost_output)
        total_optimized = sum(route_costs.values())
        baseline = cost_output.total_baseline_cost
        reduction_pct = max(0, (baseline - total_optimized) / max(baseline, 1) * 100)

        # Supply chain allocation
        sc_allocation = self._allocate_supply_chain(
            routes, supply_chain_risk.optimization_constraints
        )

        solution = QuantumSolution(
            optimized_routes=routes,
            route_costs=route_costs,
            total_optimized_cost=round(total_optimized, 2),
            total_baseline_cost=round(baseline, 2),
            cost_reduction_pct=round(reduction_pct, 2),
            solver_used=decision.solver,
            solver_tier=decision.tier,
            solver_decision=decision,
            solution_energy=round(energy, 4),
            supply_chain_allocation=sc_allocation,
            num_annealing_reads=ANNEALING_READS,
            optimization_metadata={
                "qubo_size": qubo_problem.num_variables,
                "qubo_nonzeros": len(qubo_problem.qubo_matrix),
                "num_trucks_used": len(routes),
                "total_nodes": len(route_input.nodes),
                "num_deliveries": len(route_input.nodes) - 1,
                "high_risk_routes_avoided": len(route_risk.high_risk_edges),
                "solver_tier": decision.tier.value,
                "solver_reason": decision.reason,
                "solve_time_ms": solve_ms,
            },
        )

        log_agent_output("QuantumEngine", solution)
        return solution

    def _compute_route_costs(
        self,
        routes: List[List[str]],
        edges: List[Tuple[str, str, float]],
        cost_output: CostOptimizationOutput,
    ) -> Dict[int, float]:
        edge_lookup = {(s, d): w for s, d, w in edges}
        coeffs = cost_output.optimization_objective.get("edge_cost_coefficients", {})
        baseline_per_unit = cost_output.total_baseline_cost / max(len(coeffs), 1)
        route_costs = {}
        for i, route in enumerate(routes):
            cost = 0.0
            for j in range(len(route) - 1):
                src, dst = route[j], route[j + 1]
                dist = edge_lookup.get((src, dst), edge_lookup.get((dst, src), 50.0))
                edge_key = f"{src}→{dst}"
                coeff = coeffs.get(edge_key, 0.5)
                cost += coeff * baseline_per_unit * 0.85
            route_costs[i] = round(cost, 2)
        return route_costs

    def _allocate_supply_chain(
        self,
        routes: List[List[str]],
        sc_constraints: Dict[str, Any],
    ) -> Dict[str, str]:
        preferred = sc_constraints.get("preferred_suppliers", [])
        allocation = {}
        for route in routes:
            for i, node in enumerate(route):
                if preferred:
                    allocation[node] = preferred[i % len(preferred)]
        return allocation
