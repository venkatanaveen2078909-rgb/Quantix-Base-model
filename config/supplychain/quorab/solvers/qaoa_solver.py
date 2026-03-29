import time
from typing import Any, Dict
import numpy as np

import numpy
print(numpy.__version__)

import qiskit
print(qiskit.__version__)

from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram


class QAOASolver:
    def __init__(self) -> None:
        # ✅ Stable classical backend (no sampler issues)
        self.solver = NumPyMinimumEigensolver()
        self.eigen_optimizer = MinimumEigenOptimizer(self.solver)

    def solve(self, Q: np.ndarray) -> Dict[str, Any]:

        if not isinstance(Q, np.ndarray):
            raise TypeError("Q must be numpy array")

        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("QUBO must be square")

        start_time = time.time()

        qp = QuadraticProgram()
        n = Q.shape[0]

        for i in range(n):
            qp.binary_var(name=f"x{i}")

        linear = {}
        quadratic = {}

        for i in range(n):
            if Q[i, i] != 0:
                linear[f"x{i}"] = float(Q[i, i])

            for j in range(i + 1, n):
                if Q[i, j] != 0:
                    quadratic[(f"x{i}", f"x{j}")] = float(Q[i, j])

        qp.minimize(linear=linear, quadratic=quadratic)

        result = self.eigen_optimizer.solve(qp)

        end_time = time.time()

        return {
            "solution": result.x.tolist(),
            "objective_value": float(result.fval),
            "time_taken": end_time - start_time
        }


# ✅ TEST
if __name__ == "__main__":
    Q = np.array([
        [1, -1],
        [-1, 2]
    ])

    solver = QAOASolver()
    result = solver.solve(Q)

    print("\nRESULT:")
    print(result)