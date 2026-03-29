"""
Quantix — Configuration & Settings (v3 - Production Ready)
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────────
# LLM (Groq / OpenAI Compatible)
# ─────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

# Backward compatibility (if some code still uses OPENAI_API_KEY)
OPENAI_API_KEY: str = GROQ_API_KEY

# ─────────────────────────────────────────────
# Database (PostgreSQL Async)
# ─────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:root@localhost:5432/quantix_db"
)

DATABASE_POOL_SIZE: int = int(os.getenv("DATABASE_POOL_SIZE", "10"))
DATABASE_MAX_OVERFLOW: int = int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))

# ─────────────────────────────────────────────
# Adaptive Solver Thresholds
# ─────────────────────────────────────────────
SOLVER_HIGH_THRESHOLD: int = int(os.getenv("SOLVER_HIGH_THRESHOLD", "20"))
SOLVER_MEDIUM_THRESHOLD: int = int(os.getenv("SOLVER_MEDIUM_THRESHOLD", "5"))

# ─────────────────────────────────────────────
# D-Wave Hybrid (Local — no API token required)
# ─────────────────────────────────────────────
# Sampler to use for HIGH-tier problems: 'Tabu' (default) | 'SA'
DWAVE_HYBRID_SAMPLER: str = os.getenv("DWAVE_HYBRID_SAMPLER", "Tabu")

ANNEALING_READS: int = int(os.getenv("ANNEALING_READS", "100"))
ANNEALING_NUM_SWEEPS: int = int(os.getenv("ANNEALING_NUM_SWEEPS", "500"))

ANNEALING_BETA_RANGE: tuple = (
    float(os.getenv("ANNEALING_BETA_START", "0.1")),
    float(os.getenv("ANNEALING_BETA_END", "4.0")),
)

# ─────────────────────────────────────────────
# QAOA (Hybrid Quantum)
# ─────────────────────────────────────────────
QAOA_REPS: int = int(os.getenv("QAOA_REPS", "3"))
QAOA_MAX_ITER: int = int(os.getenv("QAOA_MAX_ITER", "300"))
QAOA_OPTIMIZER: str = os.getenv("QAOA_OPTIMIZER", "COBYLA")
QAOA_SHOTS: int = int(os.getenv("QAOA_SHOTS", "1024"))
QAOA_BACKEND: str = os.getenv("QAOA_BACKEND", "statevector_simulator")

# ─────────────────────────────────────────────
# Classical Solver (MILP)
# ─────────────────────────────────────────────
MILP_SOLVER: str = os.getenv("MILP_SOLVER", "PULP_CBC_CMD")
MILP_TIME_LIMIT_SEC: int = int(os.getenv("MILP_TIME_LIMIT_SEC", "60"))
MILP_GAP_TOLERANCE: float = float(os.getenv("MILP_GAP_TOLERANCE", "0.01"))

# ─────────────────────────────────────────────
# Optimization Weights
# ─────────────────────────────────────────────
WEIGHT_COST: float = float(os.getenv("WEIGHT_COST", "0.4"))
WEIGHT_TIME: float = float(os.getenv("WEIGHT_TIME", "0.3"))
WEIGHT_RISK: float = float(os.getenv("WEIGHT_RISK", "0.3"))

# QUBO Penalties
PENALTY_VISIT_ONCE: float = float(os.getenv("PENALTY_VISIT_ONCE", "100.0"))
PENALTY_CAPACITY: float = float(os.getenv("PENALTY_CAPACITY", "50.0"))
PENALTY_TIME_WINDOW: float = float(os.getenv("PENALTY_TIME_WINDOW", "30.0"))

# ─────────────────────────────────────────────
# API Config
# ─────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_WORKERS: int = int(os.getenv("API_WORKERS", "4"))
API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"

# ─────────────────────────────────────────────
# Security
# ─────────────────────────────────────────────
SECRET_KEY: str = os.getenv("SECRET_KEY", "quantix-secret-change-in-production")
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")  # json | text