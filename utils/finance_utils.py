from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_expected_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    """Compute expected portfolio return mu^T w."""
    return float(np.dot(expected_returns, weights))


def compute_portfolio_risk(weights: np.ndarray, covariance: np.ndarray) -> float:
    """Compute portfolio variance w^T Sigma w."""
    return float(weights.T @ covariance @ weights)


def compute_sharpe_ratio(expected_return: float, risk_variance: float, risk_free_rate: float = 0.0) -> float:
    """Compute Sharpe ratio from expected return and variance."""
    vol = float(np.sqrt(max(risk_variance, 0.0)))
    if vol == 0:
        return 0.0
    return float((expected_return - risk_free_rate) / vol)


def diversification_score(weights: np.ndarray) -> float:
    """Compute normalized diversification score based on inverse HHI."""
    abs_w = np.abs(weights)
    total = abs_w.sum()
    if total == 0:
        return 0.0
    p = abs_w / total
    hhi = float(np.sum(np.square(p)))
    n = len(weights)
    if n <= 1:
        return 1.0
    min_hhi = 1.0 / n
    return float((1.0 - hhi) / (1.0 - min_hhi))


def equal_weight_portfolio(n_assets: int, allow_shorting: bool = False) -> np.ndarray:
    """Build an equal-weight benchmark portfolio."""
    if n_assets <= 0:
        raise ValueError("n_assets must be positive")
    if allow_shorting:
        # Deterministic neutral benchmark for long/short mode.
        half = n_assets // 2
        weights = np.ones(n_assets, dtype=float)
        weights[:half] = -1.0
        weights = weights / np.sum(np.abs(weights))
        return weights
    return np.ones(n_assets, dtype=float) / n_assets


def build_weights_map(asset_ids: List[str], weights: np.ndarray) -> Dict[str, float]:
    """Map asset ids to scalar weights."""
    return {asset_id: float(weight) for asset_id, weight in zip(asset_ids, weights)}
