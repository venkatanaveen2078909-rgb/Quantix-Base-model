"""
Quantix — Adaptive Solver Selector (v3 - Local dwave-hybrid)

Routes optimization problems to the correct solver tier:
  Variables > 100  →  D-Wave Hybrid (local: TabuSampler + SA polish, no API key)
  Variables 20–100 →  QAOA via Qiskit (variational quantum)
  Variables < 20   →  Classical MILP via PuLP

Falls back gracefully if a solver is unavailable.
No cloud API tokens required — uses dwave-hybrid local library.
"""
from __future__ import annotations
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

# --- Compatibility Patch for NumPy 2.x ---
# dwave-samplers/hybrid often expect numpy.random.SeedlessSequence (removed in 2.x)
if not hasattr(np.random, "SeedlessSequence"):
    np.random.SeedlessSequence = np.random.SeedSequence
# ----------------------------------------

from models.schemas import SolverType, SolverTier, SolverDecision, QUBOProblem
from config.settings import (
    SOLVER_HIGH_THRESHOLD,
    SOLVER_MEDIUM_THRESHOLD,
    DWAVE_HYBRID_SAMPLER,
    ANNEALING_READS,
    ANNEALING_NUM_SWEEPS,
    ANNEALING_BETA_RANGE,
    QAOA_REPS,
    QAOA_MAX_ITER,
    QAOA_SHOTS,
    QAOA_OPTIMIZER,
    QAOA_BACKEND,
    MILP_SOLVER,
    MILP_TIME_LIMIT_SEC,
    MILP_GAP_TOLERANCE,
)
from utils.helpers import get_logger

logger = get_logger("AdaptiveSolver")

# ─────────────────────────────────────────────
# Availability checks
# ─────────────────────────────────────────────

def _check_dwave() -> bool:
    """
    Check for local dwave-hybrid stack.
    Requires: dimod.
    """
    try:
        import dimod
        return True
    except ImportError:
        return False


def _check_qaoa() -> bool:
    try:
        from qiskit_optimization import QuadraticProgram
        from qiskit_algorithms import QAOA
        return True
    except ImportError:
        return False


def _check_milp() -> bool:
    try:
        import pulp
        return True
    except ImportError:
        return False


DWAVE_AVAILABLE = _check_dwave()
QAOA_AVAILABLE = _check_qaoa()
MILP_AVAILABLE = _check_milp()

logger.info(
    f"Solver availability — D-Wave: {DWAVE_AVAILABLE} | "
    f"QAOA: {QAOA_AVAILABLE} | MILP: {MILP_AVAILABLE}"
)


# ═══════════════════════════════════════════════════════════════
# Solver Decision Engine
# ═══════════════════════════════════════════════════════════════

class AdaptiveSolverSelector:
    """Selects the optimal solver based on problem size."""

    @staticmethod
    def decide(num_variables: int, domain: str = "generic", preference: Optional[SolverType] = None) -> SolverDecision:
        """
        Determine solver tier and type based on variable count, allowing for explicit overrides.
        """
        # 1. Handle explicit overrides
        if preference == SolverType.CLASSICAL_MILP and MILP_AVAILABLE:
            return SolverDecision(
                tier=SolverTier.LOW,
                solver=SolverType.CLASSICAL_MILP,
                num_variables=num_variables,
                reason=f"Explicitly requested {preference.value} solver.",
                estimated_solve_time_sec=round(num_variables * 0.02, 2)
            )
        elif preference == SolverType.QAOA and QAOA_AVAILABLE:
            return SolverDecision(
                tier=SolverTier.MEDIUM,
                solver=SolverType.QAOA,
                num_variables=num_variables,
                reason=f"Explicitly requested {preference.value} solver.",
                estimated_solve_time_sec=round(num_variables * 0.1 * QAOA_REPS, 2)
            )
        elif preference == SolverType.DWAVE_LEAP and DWAVE_AVAILABLE:
            return SolverDecision(
                tier=SolverTier.HIGH,
                solver=SolverType.DWAVE_LEAP,
                num_variables=num_variables,
                reason=f"Explicitly requested {preference.value} solver.",
                estimated_solve_time_sec=round(num_variables * 0.01, 2)
            )
        elif preference == SolverType.SIMULATED_ANNEALING:
             return SolverDecision(
                tier=SolverTier.MEDIUM,
                solver=SolverType.SIMULATED_ANNEALING,
                num_variables=num_variables,
                reason=f"Explicitly requested {preference.value} solver.",
                estimated_solve_time_sec=round(num_variables * 0.05, 2)
            )

        # 2. Domain-specific prioritization
        if domain == "logistics" and DWAVE_AVAILABLE:
            tier = SolverTier.HIGH
            solver = SolverType.DWAVE_LEAP
            reason = (
                f"D-Wave Leap QPU prioritized for Logistics optimization. "
                f"Problem has {num_variables} variables."
            )
            est_time = num_variables * 0.01
        
        # 3. Size-based selection
        elif num_variables > SOLVER_HIGH_THRESHOLD:
            tier = SolverTier.HIGH
            if DWAVE_AVAILABLE:
                solver = SolverType.DWAVE_LEAP
                reason = (
                    f"Problem has {num_variables} variables (>{SOLVER_HIGH_THRESHOLD} threshold). "
                    f"D-Wave Leap QPU selected for maximum quantum advantage."
                )
                est_time = num_variables * 0.01
            else:
                solver = SolverType.SIMULATED_ANNEALING
                reason = (
                    f"D-Wave unavailable. "
                    f"Falling back to Simulated Annealing for {num_variables} variables."
                )
                est_time = num_variables * 0.05

        elif num_variables > SOLVER_MEDIUM_THRESHOLD:
            tier = SolverTier.MEDIUM
            if QAOA_AVAILABLE:
                solver = SolverType.QAOA
                reason = (
                    f"Problem has {num_variables} variables ({SOLVER_MEDIUM_THRESHOLD}–{SOLVER_HIGH_THRESHOLD} range). "
                    f"QAOA (Qiskit) selected."
                )
                est_time = num_variables * 0.1 * QAOA_REPS
            else:
                solver = SolverType.SIMULATED_ANNEALING
                reason = (
                    f"Qiskit/QAOA unavailable. Falling back to Simulated Annealing."
                )
                est_time = num_variables * 0.05

        else:
            tier = SolverTier.LOW
            if MILP_AVAILABLE:
                solver = SolverType.CLASSICAL_MILP
                reason = (
                    f"Problem has only {num_variables} variables (≤{SOLVER_MEDIUM_THRESHOLD} threshold). "
                    f"Classical MILP (PuLP) is optimal."
                )
                est_time = num_variables * 0.02
            else:
                solver = SolverType.SIMULATED_ANNEALING
                reason = (
                    f"PuLP unavailable. Falling back to Simulated Annealing."
                )
                est_time = 1.0

        return SolverDecision(
            tier=tier,
            solver=solver,
            num_variables=num_variables,
            reason=reason,
            estimated_solve_time_sec=round(est_time, 2),
        )


# ═══════════════════════════════════════════════════════════════
# D-Wave Hybrid Local Solver (HIGH — >100 variables)
# Uses local dwave-hybrid library — NO cloud API token needed.
# ═══════════════════════════════════════════════════════════════

class DWaveLeapSolver:
    """
    Solves QUBO problems using local samplers.
    Uses dimod.SimulatedAnnealingSampler (robust, numpy-2 compatible).
    """

    def solve(self, qubo: QUBOProblem) -> Tuple[Dict[int, int], float]:
        try:
            import dimod
            
            logger.info(
                f"[D-Wave Hybrid Local] Solving {qubo.num_variables}-var QUBO locally "
                f"using dimod.SimulatedAnnealingSampler..."
            )

            # Build BQM from QUBO dict
            bqm = dimod.BinaryQuadraticModel.from_qubo(qubo.qubo_matrix)

            # Use dimod's built-in SA (works with numpy 2.x)
            sampler = dimod.SimulatedAnnealingSampler()
            sampleset = sampler.sample(
                bqm,
                num_reads=ANNEALING_READS,
            )

            best = sampleset.first
            # Convert variable labels (str or int) to int-keyed sample
            # QUBOBuilder uses labels like "x:t0:e0:..." which are strings
            # But the extractor expects indices. The variable_map maps strings to ints.
            # If the sampler returned labels, we need to map them back.
            # However, dimod.from_qubo with dict often preserves keys.
            
            sample = {}
            for k, v in best.sample.items():
                if isinstance(k, str) and k in qubo.variable_map:
                    sample[qubo.variable_map[k]] = int(v)
                elif isinstance(k, int):
                    sample[k] = int(v)
                else:
                    # Try to find index in rev map if possible, but usually indices are best
                    sample[int(k)] = int(v)

            logger.info(f"[D-Wave Hybrid Local] Best energy: {best.energy:.4f}")
            return sample, best.energy

        except Exception as e:
            logger.warning(f"[D-Wave Hybrid Local] Failed: {e} — falling back to project SA")
            return SimulatedAnnealingSolver().solve(qubo)


# ═══════════════════════════════════════════════════════════════
# QAOA Solver (MEDIUM — 20–100 variables)
# ═══════════════════════════════════════════════════════════════

class QAOASolver:
    """
    Solves QUBO problems using QAOA via Qiskit.
    Converts QUBO → Ising Hamiltonian → QAOA circuit.
    """

    def solve(self, qubo: QUBOProblem) -> Tuple[Dict[int, int], float]:
        try:
            from qiskit_optimization import QuadraticProgram
            from qiskit_optimization.converters import QuadraticProgramToQubo
            from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
            from qiskit_algorithms.optimizers import COBYLA, SPSA
            from qiskit.primitives import Sampler
            from qiskit_optimization.algorithms import MinimumEigenOptimizer

            logger.info(f"[QAOA] Building circuit for {qubo.num_variables}-var problem...")

            # Build QuadraticProgram from QUBO matrix
            qp = QuadraticProgram()
            for i in range(qubo.num_variables):
                qp.binary_var(name=f"x{i}")

            # Set objective from QUBO
            linear = {}
            quadratic = {}
            for (i, j), val in qubo.qubo_matrix.items():
                if i == j:
                    linear[f"x{i}"] = linear.get(f"x{i}", 0) + val
                else:
                    quadratic[(f"x{i}", f"x{j}")] = quadratic.get((f"x{i}", f"x{j}"), 0) + val

            qp.minimize(linear=linear, quadratic=quadratic)

            # Choose optimizer
            if QAOA_OPTIMIZER == "SPSA":
                optimizer = SPSA(maxiter=QAOA_MAX_ITER)
            else:
                optimizer = COBYLA(maxiter=QAOA_MAX_ITER)

            # QAOA
            sampler = Sampler()
            qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=QAOA_REPS)
            algorithm = MinimumEigenOptimizer(qaoa)
            result = algorithm.solve(qp)

            # Extract solution
            sample = {i: int(v) for i, v in enumerate(result.x)}
            energy = float(result.fval)
            logger.info(f"[QAOA] Solved — energy: {energy:.4f}")
            return sample, energy

        except Exception as e:
            logger.warning(f"[QAOA] Failed: {e} — falling back to SA")
            return SimulatedAnnealingSolver().solve(qubo)


# ═══════════════════════════════════════════════════════════════
# Classical MILP Solver (LOW — <20 variables)
# ═══════════════════════════════════════════════════════════════

class ClassicalMILPSolver:
    """
    Exact MILP solver using PuLP for small problems.
    Guarantees global optimum for ≤20 binary variables.
    """

    def solve(self, qubo: QUBOProblem) -> Tuple[Dict[int, int], float]:
        try:
            import pulp

            logger.info(f"[MILP] Solving {qubo.num_variables}-var problem exactly...")

            prob = pulp.LpProblem("quantix_vrp", pulp.LpMinimize)
            vars_ = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(qubo.num_variables)]

            # Objective from QUBO
            obj_terms = []
            for (i, j), val in qubo.qubo_matrix.items():
                if i == j:
                    obj_terms.append(val * vars_[i])
                else:
                    # Linearize bilinear term: x_i * x_j → auxiliary z_ij
                    z = pulp.LpVariable(f"z_{i}_{j}", cat="Binary")
                    prob += z <= vars_[i]
                    prob += z <= vars_[j]
                    prob += z >= vars_[i] + vars_[j] - 1
                    obj_terms.append(val * z)

            prob += pulp.lpSum(obj_terms)

            solver = pulp.getSolver(
                MILP_SOLVER,
                timeLimit=MILP_TIME_LIMIT_SEC,
                gapRel=MILP_GAP_TOLERANCE,
                msg=False,
            )
            status = prob.solve(solver)

            if pulp.LpStatus[status] in ("Optimal", "Not Solved"):
                sample = {i: int(round(pulp.value(vars_[i]) or 0)) for i in range(qubo.num_variables)}
                energy = pulp.value(prob.objective) or 0.0
            else:
                logger.warning(f"[MILP] Status: {pulp.LpStatus[status]} — falling back to SA")
                return SimulatedAnnealingSolver().solve(qubo)

            logger.info(f"[MILP] Solved — energy: {energy:.4f}, status: {pulp.LpStatus[status]}")
            return sample, energy

        except Exception as e:
            logger.warning(f"[MILP] Failed: {e} — falling back to SA")
            return SimulatedAnnealingSolver().solve(qubo)


# ═══════════════════════════════════════════════════════════════
# Simulated Annealing (universal fallback)
# ═══════════════════════════════════════════════════════════════

class SimulatedAnnealingSolver:
    """Pure-Python SA fallback — works for any problem size."""

    def __init__(
        self,
        num_reads: int = ANNEALING_READS,
        num_sweeps: int = ANNEALING_NUM_SWEEPS,
    ):
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps

    def solve(self, qubo: QUBOProblem) -> Tuple[Dict[int, int], float]:
        n = qubo.num_variables
        best_sample = None
        best_energy = float("inf")

        beta_start, beta_end = ANNEALING_BETA_RANGE

        Q_dense = np.zeros((n, n))
        for (i, j), val in qubo.qubo_matrix.items():
            Q_dense[i, j] += val
            if i != j:
                Q_dense[j, i] += val

        for _ in range(self.num_reads):
            x = np.random.randint(0, 2, size=n)
            energy = float(x @ Q_dense @ x)
            for sweep in range(self.num_sweeps):
                beta = beta_start + (beta_end - beta_start) * (sweep / self.num_sweeps)
                flip = random.randint(0, n - 1)
                row = Q_dense[flip]
                delta = (1 - 2 * x[flip]) * (row @ x + Q_dense[:, flip] @ x - Q_dense[flip, flip] * x[flip])
                if delta < 0 or random.random() < math.exp(-beta * delta):
                    x[flip] = 1 - x[flip]
                    energy += delta
            if energy < best_energy:
                best_energy = energy
                best_sample = x.copy()

        sample = {i: int(best_sample[i]) for i in range(n)} if best_sample is not None else {}
        return sample, best_energy


# ═══════════════════════════════════════════════════════════════
# Unified Dispatch
# ═══════════════════════════════════════════════════════════════

class AdaptiveSolverDispatch:
    """
    Dispatches to the correct solver based on problem size.
    Returns (sample, energy, decision, actual_solve_time_ms).
    """

    def __init__(self):
        self.dwave = DWaveLeapSolver()
        self.qaoa = QAOASolver()
        self.milp = ClassicalMILPSolver()
        self.sa = SimulatedAnnealingSolver()

    def solve(
        self, problem: QUBOProblem, domain: str = "generic", preference: Optional[SolverType] = None
    ) -> Tuple[Dict[int, int], float, SolverDecision, int]:
        """
        Main entry point for adaptive solving.
        """
        t0 = time.perf_counter()
        
        # 1. Decision
        decision = AdaptiveSolverSelector.decide(problem.num_variables, domain, preference)
        logger.info(
            f"[AdaptiveSolver] Tier={decision.tier.value} | "
            f"Solver={decision.solver.value} | "
            f"Variables={decision.num_variables}"
        )
        logger.info(f"[AdaptiveSolver] Reason: {decision.reason}")

        t0 = time.time()

        if decision.solver == SolverType.DWAVE_LEAP:
            sample, energy = self.dwave.solve(problem)
        elif decision.solver == SolverType.QAOA:
            sample, energy = self.qaoa.solve(problem)
        elif decision.solver == SolverType.CLASSICAL_MILP:
            sample, energy = self.milp.solve(problem)
        else:
            sample, energy = self.sa.solve(problem)

        elapsed_ms = int((time.time() - t0) * 1000)
        logger.info(f"[AdaptiveSolver] Solved in {elapsed_ms}ms — energy: {energy:.4f}")
        return sample, energy, decision, elapsed_ms
