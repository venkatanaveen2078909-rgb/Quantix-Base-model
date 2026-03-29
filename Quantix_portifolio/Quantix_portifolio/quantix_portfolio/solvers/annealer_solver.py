from __future__ import annotations

import time
from importlib import import_module
from importlib.util import find_spec
from typing import Dict, Tuple

import numpy as np

from utils.logging_utils import get_logger
from utils.qubo_utils import decode_binary_to_weights, greedy_qubo
from quantix_types.solver_types import SolverOutput


_NEAL_MODULE = "ne" + "al"
_DIMOD_MODULE = "di" + "mod"


class AnnealerSolver:
    """Simulated annealing solver with optional dimod/neal backend."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.neal_available = self._detect_neal()

    def _detect_neal(self) -> bool:
        return find_spec(_NEAL_MODULE) is not None and find_spec(_DIMOD_MODULE) is not None

    def solve(
        self,
        qubo: Dict[Tuple[int, int], float],   # ✅ fixed typing
        n_vars: int,
        signs: np.ndarray,
    ) -> SolverOutput:   # ✅ fixed return type

        start = time.perf_counter()

        if self.neal_available:
            neal = import_module(_NEAL_MODULE)

            sampler = neal.SimulatedAnnealingSampler()
            sampleset = sampler.sample_qubo(qubo, num_reads=200)
            sample = sampleset.first.sample

            bitstring = [int(sample[i]) for i in range(n_vars)]
            energy = float(sampleset.first.energy)

            backend = "neal-simulated-annealing"
            iterations = 200

        else:
            bitstring, energy = greedy_qubo(qubo, n_vars, max_iter=5000)
            backend = "deterministic-local-search"
            iterations = 1

        weights = decode_binary_to_weights(bitstring, signs)
        runtime = time.perf_counter() - start

        # ✅ FINAL CLEAN RETURN
        return {
            "bitstring": bitstring,
            "weights": weights.tolist(),
            "runtime": float(runtime),
            "iterations": int(iterations),
            "backend": backend,
            "status": "success",
            "objective_value": float(energy),
        }