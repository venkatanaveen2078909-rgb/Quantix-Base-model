import numpy as np
from typing import cast, List

from solvers.annealer_solver import AnnealerSolver
from solvers.classical_solver import ClassicalSolver
from solvers.qaoa_solver import QAOASolver


def test_classical_solver_outputs_weights() -> None:
    solver = ClassicalSolver()
    mu = np.array([0.1, 0.12, 0.05])
    cov = np.array(
        [
            [0.02, 0.01, 0.0],
            [0.01, 0.03, 0.01],
            [0.0, 0.01, 0.015],
        ]
    )
    constraints = {
        "risk_tolerance_lambda": 1.0,
        "allow_shorting": False,
        "min_weight": 0.0,
        "max_weight": 1.0,
        "cardinality_limit": None,
        "risk_threshold": None,
    }

    out = solver.solve(mu, cov, constraints)

    # ✅ FIX: cast weights
    weights = cast(List[float], out["weights"])

    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 1e-6


def test_qaoa_and_annealer_fallbacks() -> None:
    qubo = {(0, 0): -1.0, (1, 1): -0.5, (0, 1): 0.2}
    signs = np.array([1.0, -1.0])

    qaoa = QAOASolver().solve(qubo=qubo, n_vars=2, signs=signs)
    anneal = AnnealerSolver().solve(qubo=qubo, n_vars=2, signs=signs)

    # ✅ FIX: cast weights
    q_weights = cast(List[float], qaoa["weights"])
    a_weights = cast(List[float], anneal["weights"])

    assert len(q_weights) == 2
    assert len(a_weights) == 2
    assert "backend" in qaoa
    assert "backend" in anneal