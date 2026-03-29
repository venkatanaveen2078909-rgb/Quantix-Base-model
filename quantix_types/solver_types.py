from typing import TypedDict, List


class SolverOutput(TypedDict):
    weights: List[float]
    objective_value: float
    runtime: float
    iterations: int
    status: str
    backend: str
    bitstring: List[int]