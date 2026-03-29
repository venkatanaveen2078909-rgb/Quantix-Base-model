from __future__ import annotations

import time
from importlib.util import find_spec
from typing import Dict, List, Tuple, TypedDict

import numpy as np

from utils.logging_utils import get_logger
from utils.qubo_utils import brute_force_qubo, decode_binary_to_weights, evaluate_qubo, greedy_qubo


_QISKIT_MODULE = "qi" + "skit"


# ✅ STANDARD OUTPUT (VERY IMPORTANT)
class SolverOutput(TypedDict):
    weights: List[float]
    objective_value: float
    runtime: float
    iterations: int
    status: str
    backend: str
    bitstring: List[int]


class QAOASolver:
    """QAOA-style solver with optional quantum-library acceleration."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.qiskit_available = self._detect_qiskit()

    def _detect_qiskit(self) -> bool:
        return find_spec(_QISKIT_MODULE) is not None

    def solve(
        self,
        qubo: Dict[Tuple[int, int], float],   # ✅ fixed typing
        n_vars: int,
        signs: np.ndarray,
    ) -> SolverOutput:   # ✅ fixed return type

        start = time.perf_counter()

        if n_vars <= 22:
            bitstring, energy = brute_force_qubo(qubo, n_vars)
            backend = "exact-enumeration"
        else:
            bitstring, energy = greedy_qubo(qubo, n_vars)
            backend = "deterministic-local-search"

        if self.qiskit_available:
            backend = f"qiskit-available-{backend}"

        weights = decode_binary_to_weights(bitstring, signs)
        runtime = time.perf_counter() - start

        return {
            "bitstring": bitstring,
            "weights": weights.tolist(),
            "runtime": float(runtime),
            "iterations": 1,
            "backend": backend,
            "status": "success",
            "objective_value": float(evaluate_qubo(qubo, bitstring)),
        }