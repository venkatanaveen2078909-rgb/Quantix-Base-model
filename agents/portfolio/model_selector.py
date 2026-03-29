from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelSelection:
    solver_key: str
    rationale: str


class ModelSelector:
    """Select solver strategy based on problem size."""

    def select(self, asset_count: int) -> ModelSelection:
        if asset_count < 30:
            return ModelSelection(
                solver_key="classical",
                rationale="Asset count < 30, using deterministic classical mean-variance solver",
            )
        if asset_count <= 100:
            return ModelSelection(
                solver_key="qaoa",
                rationale="Asset count between 30 and 100, using hybrid QAOA-style path",
            )
        return ModelSelection(
            solver_key="annealer",
            rationale="Asset count > 100, using annealing-style path for larger combinatorial search",
        )
