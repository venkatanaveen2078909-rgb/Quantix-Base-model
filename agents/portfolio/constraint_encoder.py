from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EncodedConstraints:
    definitions: Dict[str, Any]
    symbolic_expressions: List[str]


class ConstraintEncoder:
    """Translate user constraints into symbolic and machine-readable forms."""

    def encode(
        self,
        n_assets: int,
        constraints: Dict[str, Any],
    ) -> EncodedConstraints:
        budget = float(constraints.get("budget", 1.0))
        min_weight = float(constraints.get("min_weight", 0.0))
        max_weight = float(constraints.get("max_weight", 1.0))
        allow_shorting = bool(constraints.get("allow_shorting", False))
        cardinality_limit: Optional[int] = constraints.get("cardinality_limit")
        risk_threshold = constraints.get("risk_threshold")

        if not allow_shorting and min_weight < 0:
            raise ValueError("min_weight cannot be negative when allow_shorting is False")
        if max_weight <= 0:
            raise ValueError("max_weight must be positive")
        if min_weight > max_weight:
            raise ValueError("min_weight cannot exceed max_weight")
        if cardinality_limit is not None and cardinality_limit > n_assets:
            raise ValueError("cardinality_limit cannot exceed number of assets")

        symbolic = [
            f"budget: sum(w_i) = {budget}",
            f"bounds: {min_weight} <= w_i <= {max_weight}",
            f"shorting: {'enabled' if allow_shorting else 'disabled'}",
        ]
        if cardinality_limit is not None:
            symbolic.append(f"cardinality: sum(z_i) <= {cardinality_limit}")
        if risk_threshold is not None:
            symbolic.append(f"risk: w^T Sigma w <= {float(risk_threshold)}")

        definitions: Dict[str, Any] = {
            "budget": budget,
            "weight_bounds": {
                "min_weight": min_weight,
                "max_weight": max_weight,
            },
            "allow_shorting": allow_shorting,
            "cardinality_limit": cardinality_limit,
            "risk_threshold": risk_threshold,
            "n_assets": n_assets,
        }

        return EncodedConstraints(definitions=definitions, symbolic_expressions=symbolic)
