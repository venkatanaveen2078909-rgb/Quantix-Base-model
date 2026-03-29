from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
from scipy.optimize import minimize

from utils.logging_utils import get_logger
from quantix_types.solver_types import SolverOutput


class ClassicalSolver:
    """Continuous mean-variance optimization with deterministic constraints."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def solve(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        constraints: Dict[str, float | bool | int | None],
        alpha_signs: Optional[np.ndarray] = None,
    ) -> SolverOutput:   # ✅ FIXED

        n_assets = expected_returns.shape[0]

        lam = float(constraints.get("risk_tolerance_lambda") or 1.0)
        budget = float(constraints.get("budget") or 1.0)
        allow_shorting = bool(constraints.get("allow_shorting") or False)
        min_weight = float(constraints.get("min_weight") or 0.0)
        max_weight = float(constraints.get("max_weight") or 1.0)
        cardinality_limit = constraints.get("cardinality_limit")
        risk_threshold = constraints.get("risk_threshold")

        if alpha_signs is None:
            alpha_signs = np.ones(n_assets)
        else:
            alpha_signs = np.where(alpha_signs == 0, 1.0, alpha_signs.astype(float))

        if allow_shorting:
            bounds = [(-max_weight, max_weight) for _ in range(n_assets)]
        else:
            lower = max(0.0, min_weight)
            bounds = [(lower, max_weight) for _ in range(n_assets)]

        def objective(w: np.ndarray) -> float:
            risk = float(w.T @ covariance @ w)
            ret = float(expected_returns @ w)
            l2 = 1e-6 * float(np.sum(np.square(w)))
            return lam * risk - ret + l2

        linear_constraint = {"type": "eq", "fun": lambda w: float(np.sum(w) - budget)}
        nonlinear_constraints = [linear_constraint]

        if risk_threshold is not None:
            threshold = float(risk_threshold)
            nonlinear_constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: float(threshold - (w.T @ covariance @ w)),
                }
            )

        # Initial weights
        w0 = np.ones(n_assets, dtype=float) / n_assets
        w0 = w0 * alpha_signs

        if allow_shorting:
            norm = np.sum(np.abs(w0))
            w0 = w0 / norm if norm > 0 else w0
            w0 = w0 - np.mean(w0) + (budget / n_assets)
            w0 = w0 / np.sum(w0) * budget
        else:
            w0 = np.maximum(w0, 0.0)
            w0 = w0 / np.sum(w0) * budget

        start = time.perf_counter()

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=nonlinear_constraints,
            options={"maxiter": 500, "ftol": 1e-9, "disp": False},
        )

        runtime = time.perf_counter() - start

        if not result.success:
            self.logger.warning("Fallback triggered: %s", result.message)
            weights = w0.copy()
            status = "fallback"
        else:
            weights = result.x.astype(float)
            status = "success"

        # Cardinality constraint
        if cardinality_limit:
            k = int(cardinality_limit)
            if 0 < k < n_assets:
                order = np.argsort(np.abs(weights))[::-1]
                mask = np.zeros(n_assets)
                mask[order[:k]] = 1.0
                weights = weights * mask

                if allow_shorting:
                    if np.sum(np.abs(weights)) > 0:
                        weights = weights / np.sum(np.abs(weights))
                    weights = weights - np.mean(weights) + (budget / n_assets)

                if np.sum(weights) != 0:
                    weights = weights / np.sum(weights) * budget

        objective_value = float(objective(weights))

        # ✅ FINAL RETURN (CORRECT)
        return {
            "weights": weights.tolist(),
            "objective_value": objective_value,
            "runtime": float(runtime),
            "iterations": int(getattr(result, "nit", 0)),
            "status": status,
            "backend": "scipy-slsqp",
            "bitstring": [],   # classical → no bitstring
        }