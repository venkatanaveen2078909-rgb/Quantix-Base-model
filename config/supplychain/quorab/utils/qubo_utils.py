from typing import Dict, Tuple

import numpy as np


def create_qubo_matrix(num_vars: int) -> np.ndarray:
    """
    Initializes an empty QUBO matrix.
    """
    return np.zeros((num_vars, num_vars))


def add_linear_term(Q: np.ndarray, index: int, coefficient: float):
    """
    Adds a linear term to the QUBO matrix (diagonal element).
    """
    Q[index, index] += coefficient


def add_quadratic_term(Q: np.ndarray, index1: int, index2: int, coefficient: float):
    """
    Adds a quadratic term to the QUBO matrix (off-diagonal element).
    Assuming upper triangular formulation.
    """
    if index1 > index2:
        index1, index2 = index2, index1
    if index1 != index2:
        Q[index1, index2] += coefficient


def dict_to_matrix(
    qubo_dict: Dict[Tuple[int, int], float], num_vars: int
) -> np.ndarray:
    """
    Converts a QUBO dictionary to a numpy matrix.
    """
    Q = np.zeros((num_vars, num_vars))
    for (i, j), val in qubo_dict.items():
        if i > j:
            i, j = j, i
        if i == j:
            Q[i, i] += val
        else:
            Q[i, j] += val
    return Q


def matrix_to_dict(Q: np.ndarray) -> Dict[Tuple[int, int], float]:
    """
    Converts a QUBO matrix to a dictionary format suitable for D-Wave or Qiskit.
    """
    qubo_dict = {}
    n = Q.shape[0]
    for i in range(n):
        for j in range(i, n):
            if Q[i, j] != 0:
                qubo_dict[(i, j)] = float(Q[i, j])
    return qubo_dict
