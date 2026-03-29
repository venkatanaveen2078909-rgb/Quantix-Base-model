from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from utils.portfolio_qubo_utils import qubo_dict_to_matrix


@dataclass
class QuboResult:
    q_matrix: np.ndarray
    qubo_dict: Dict[Tuple[int, int], float]
    explanation: str
    cardinality_target: int


class QUBOGenerator:
    """Generate QUBO encoding for risk-return portfolio selection."""

    def generate(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        risk_lambda: float,
        cardinality_limit: int | None,
        alpha_signs: np.ndarray,
    ) -> QuboResult:
        n = len(expected_returns)
        if n == 0:
            raise ValueError("expected_returns must not be empty")

        k = int(cardinality_limit) if cardinality_limit else max(1, min(10, n // 2 if n > 2 else n))
        k = max(1, min(k, n))

        signs = np.where(alpha_signs == 0, 1.0, alpha_signs.astype(float))
        lam = float(risk_lambda)

        budget_penalty = 6.0
        invalid_penalty = 2.0

        qubo: Dict[Tuple[int, int], float] = {}

        def add_coeff(i: int, j: int, value: float) -> None:
            key = (i, j) if i <= j else (j, i)
            qubo[key] = qubo.get(key, 0.0) + float(value)

        # Risk term under signed equal-weight decoding w_i = sign_i * x_i / k.
        for i in range(n):
            for j in range(i, n):
                coeff = lam * covariance[i, j] * signs[i] * signs[j] / (k * k)
                if abs(coeff) > 0:
                    add_coeff(i, j, coeff)

        # Return term -mu^T w.
        for i in range(n):
            coeff = -expected_returns[i] * signs[i] / k
            add_coeff(i, i, coeff)

        # Budget/cardinality penalty A * (sum(x_i) - k)^2.
        for i in range(n):
            add_coeff(i, i, budget_penalty * (1 - (2 * k)))
            for j in range(i + 1, n):
                add_coeff(i, j, 2.0 * budget_penalty)

        # Invalid state penalty for neutral assets if they leak into the set.
        for i in range(n):
            if alpha_signs[i] == 0:
                add_coeff(i, i, invalid_penalty)

        q_matrix = qubo_dict_to_matrix(qubo, n)
        explanation = (
            "QUBO = lambda*risk_term - return_term + penalties. "
            "Risk and return are encoded with signed selection variables and equal-weight decoding. "
            "Budget/cardinality is enforced via quadratic penalty (sum(x)-k)^2. "
            "Neutral assets receive additional diagonal penalties."
        )
        return QuboResult(
            q_matrix=q_matrix,
            qubo_dict=qubo,
            explanation=explanation,
            cardinality_target=k,
        )
