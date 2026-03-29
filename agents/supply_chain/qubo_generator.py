from typing import Any, Dict, List
import numpy as np
from models.schemas import QUBOProblem

class SCQUBOGenerator:
    """
    Generates QUBO for Supply Chain TSP/VRP problems.
    """
    def generate(self, num_nodes: int, distance_matrix: List[List[float]], penalty: float = 100.0) -> QUBOProblem:
        n = num_nodes
        num_vars = n * n
        Q = {}
        variable_map = {}
        
        # Helper to get index
        def idx(i, j): return i * n + j

        # 1. Objective: Minimize distance
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = distance_matrix[i][j]
                    if d != float("inf"):
                        Q[(idx(i, j), idx(i, j))] = d

        # 2. Constraint: Outgoing
        for i in range(n):
            for j in range(n):
                Q[(idx(i, j), idx(i, j))] = Q.get((idx(i, j), idx(i, j)), 0) - 2 * penalty
                for k in range(n):
                    if j <= k:
                        pair = (idx(i, j), idx(i, k))
                        Q[pair] = Q.get(pair, 0) + (penalty if j == k else 2 * penalty)

        # 3. Constraint: Incoming
        for j in range(n):
            for i in range(n):
                Q[(idx(i, j), idx(i, j))] = Q.get((idx(i, j), idx(i, j)), 0) - 2 * penalty
                for k in range(n):
                    if i <= k:
                        pair = (idx(i, j), idx(k, j))
                        Q[pair] = Q.get(pair, 0) + (penalty if i == k else 2 * penalty)

        for i in range(n):
            for j in range(n):
                variable_map[f"x_{i}_{j}"] = idx(i, j)

        return QUBOProblem(
            qubo_matrix=Q,
            num_variables=num_vars,
            variable_map=variable_map
        )
