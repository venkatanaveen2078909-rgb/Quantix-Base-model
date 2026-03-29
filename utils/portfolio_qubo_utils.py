from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Tuple

import numpy as np


QuboDict = Dict[Tuple[int, int], float]


def qubo_dict_to_matrix(qubo: QuboDict, n_vars: int) -> np.ndarray:
    """Convert a QUBO dictionary into a dense matrix representation."""
    q = np.zeros((n_vars, n_vars), dtype=float)
    for (i, j), value in qubo.items():
        q[i, j] += value
        if i != j:
            q[j, i] += value
    return q


def evaluate_qubo(qubo: QuboDict, bitstring: Iterable[int]) -> float:
    """Evaluate QUBO energy for a binary vector."""
    x = list(bitstring)
    energy = 0.0
    for (i, j), coeff in qubo.items():
        energy += coeff * x[i] * x[j]
    return float(energy)


def brute_force_qubo(qubo: QuboDict, n_vars: int) -> Tuple[List[int], float]:
    """Solve a small QUBO exactly by exhaustive enumeration."""
    best_x: List[int] = [0] * n_vars
    best_energy = float("inf")
    for candidate in product([0, 1], repeat=n_vars):
        energy = evaluate_qubo(qubo, candidate)
        if energy < best_energy:
            best_energy = energy
            best_x = list(candidate)
    return best_x, best_energy


def greedy_qubo(qubo: QuboDict, n_vars: int, max_iter: int = 2000) -> Tuple[List[int], float]:
    """Deterministic local-search fallback for larger QUBOs."""
    x = [0] * n_vars
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        base_energy = evaluate_qubo(qubo, x)
        for idx in range(n_vars):
            x[idx] = 1 - x[idx]
            candidate_energy = evaluate_qubo(qubo, x)
            if candidate_energy + 1e-12 < base_energy:
                base_energy = candidate_energy
                improved = True
            else:
                x[idx] = 1 - x[idx]
    return x, evaluate_qubo(qubo, x)


def decode_binary_to_weights(bitstring: List[int], signs: np.ndarray) -> np.ndarray:
    """Decode binary selection variables into signed equal weights."""
    x = np.array(bitstring, dtype=float)
    active = int(np.sum(x))
    if active == 0:
        return np.zeros_like(x)
    weights = x / active
    return weights * signs
