from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from utils.finance_utils import (
    compute_expected_return,
    compute_portfolio_risk,
    compute_sharpe_ratio,
    diversification_score,
    equal_weight_portfolio,
)


@dataclass
class ReportResult:
    json_output: Dict[str, Any]
    summary: str


class ReportAgent:
    """Build stable JSON output and business-readable summary."""

    def generate(
        self,
        asset_ids: List[str],
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        solver_used: str,
        performance_metrics: Dict[str, Any],
        selection_rationale: str,
        constraint_summary: List[str],
    ) -> ReportResult:
        portfolio_return = compute_expected_return(weights, expected_returns)
        portfolio_risk = compute_portfolio_risk(weights, covariance)
        sharpe = compute_sharpe_ratio(portfolio_return, portfolio_risk)

        benchmark_weights = equal_weight_portfolio(len(asset_ids), allow_shorting=False)
        benchmark_return = compute_expected_return(benchmark_weights, expected_returns)
        benchmark_risk = compute_portfolio_risk(benchmark_weights, covariance)

        return_improvement = 0.0
        if benchmark_return != 0:
            return_improvement = ((portfolio_return - benchmark_return) / abs(benchmark_return)) * 100.0

        risk_reduction = 0.0
        if benchmark_risk != 0:
            risk_reduction = ((benchmark_risk - portfolio_risk) / abs(benchmark_risk)) * 100.0

        selected_assets = [asset_id for asset_id, weight in zip(asset_ids, weights) if abs(weight) > 1e-12]
        selected_weights = [float(weight) for weight in weights if abs(weight) > 1e-12]

        output = {
            "selected_assets": selected_assets,
            "weights": selected_weights,
            "expected_return": float(portfolio_return),
            "risk": float(portfolio_risk),
            "sharpe_ratio": float(sharpe),
            "solver_used": solver_used,
            "performance_metrics": {
                "time_taken": float(performance_metrics.get("time_taken", 0.0)),
                "iterations": int(performance_metrics.get("iterations", 0)),
                "backend": str(performance_metrics.get("backend", "unknown")),
                "status": str(performance_metrics.get("status", "unknown")),
            },
            "business_impact": {
                "return_improvement_%": float(return_improvement),
                "risk_reduction_%": float(risk_reduction),
                "diversification_score": float(diversification_score(weights)),
            },
        }

        summary = self._build_summary(
            selected_assets=selected_assets,
            solver_used=solver_used,
            selection_rationale=selection_rationale,
            constraints=constraint_summary,
            portfolio_return=portfolio_return,
            portfolio_risk=portfolio_risk,
            benchmark_return=benchmark_return,
            benchmark_risk=benchmark_risk,
        )

        output["human_readable_summary"] = summary
        return ReportResult(json_output=output, summary=summary)

    def _build_summary(
        self,
        selected_assets: List[str],
        solver_used: str,
        selection_rationale: str,
        constraints: List[str],
        portfolio_return: float,
        portfolio_risk: float,
        benchmark_return: float,
        benchmark_risk: float,
    ) -> str:
        selected_str = ", ".join(selected_assets) if selected_assets else "none"
        constraints_text = "; ".join(constraints)
        return (
            f"Selected assets: {selected_str}. "
            f"Solver used: {solver_used} ({selection_rationale}). "
            f"Constraints applied: {constraints_text}. "
            f"Portfolio expected return={portfolio_return:.6f}, risk={portfolio_risk:.6f}. "
            f"Equal-weight benchmark return={benchmark_return:.6f}, risk={benchmark_risk:.6f}."
        )
