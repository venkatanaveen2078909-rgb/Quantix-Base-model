from utils.qubo_utils import create_qubo_matrix
import os
import sys
from typing import Any, Dict

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class QUBOBaseGeneratorAgent:
    def generate(self, encoded_result: Dict[str, Any]) -> np.ndarray:
        """
        Converts the problem into QUBO format.
        Includes objective function and penalty terms.
        x_ij = 1 if edge from i to j is used, else 0.
        Variables are flattened: index = i * n + j
        """
        n = encoded_result["num_nodes"]
        distance_matrix = encoded_result["distance_matrix"]
        penalties = encoded_result["encoded_constraints"]

        P = penalties["visit_once_penalty"]

        num_vars = n * n
        Q = create_qubo_matrix(num_vars)

        # 1. Objective Function: Minimize total distance
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = distance_matrix[i][j]
                    if dist != float("inf"):
                        idx = i * n + j
                        Q[idx, idx] += dist

        # 2. Constraint: Each node is visited exactly once (outgoing)
        # Sum_j x_ij = 1 => Penalty * (Sum_j x_ij - 1)^2
        for i in range(n):
            for j in range(n):
                idx1 = i * n + j
                Q[idx1, idx1] -= 2 * P
                for k in range(n):
                    idx2 = i * n + k
                    if idx1 <= idx2:
                        Q[idx1, idx2] += P if idx1 == idx2 else 2 * P

        # 3. Constraint: Each node is visited exactly once (incoming)
        # Sum_i x_ij = 1 => Penalty * (Sum_i x_ij - 1)^2
        for j in range(n):
            for i in range(n):
                idx1 = i * n + j
                Q[idx1, idx1] -= 2 * P
                for k in range(n):
                    idx2 = k * n + j
                    if idx1 <= idx2:
                        Q[idx1, idx2] += P if idx1 == idx2 else 2 * P

        return Q
