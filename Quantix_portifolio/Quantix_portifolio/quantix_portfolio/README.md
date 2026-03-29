# Quantix Systems - Hybrid Quantum-Classical Portfolio Optimization Engine

Quantix Systems is a modular backend for portfolio optimization with a deterministic classical path and optional quantum-inspired paths.

It supports:
- financial data ingestion from JSON or CSV,
- alpha generation using Piotroski F-Score and optional SUE,
- mean-variance optimization under constraints,
- classical, hybrid QAOA-style, and annealing-style solver backends,
- stable JSON output plus business-readable narrative reporting.

## Architecture

- agents/: pipeline orchestration logic
- solvers/: solver backends and fallback behavior
- utils/: shared validation, logging, finance math, and QUBO helpers
- api/: FastAPI interface
- sample_data/: runnable demo files
- tests/: unit tests for core components

## Pipeline Flow

1. DataAgent loads CSV/JSON and computes expected returns and covariance when missing.
2. AlphaAgent computes Piotroski signals and optional SUE.
3. ModelSelector chooses solver based on investable universe size.
4. ConstraintEncoder converts user constraints into symbolic + machine form.
5. QUBOGenerator encodes the objective for hybrid/annealer paths.
6. ExecutionAgent runs the selected backend.
7. ReportAgent returns stable JSON metrics and a human-readable summary.

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run API

```bash
uvicorn api.main:app --reload
```

Service URL: http://127.0.0.1:8000

## API Endpoints

- GET /health
    - Returns status check.
- GET /schema
    - Returns OptimizeRequest JSON schema.
- POST /optimize
    - Runs full optimization pipeline.

## Sample Request

Use CSV paths from sample_data for end-to-end execution:

```json
{
    "market_data_path": "sample_data/portfolio.csv",
    "fundamentals_path": "sample_data/fundamentals.csv",
    "earnings_path": "sample_data/earnings.csv",
    "constraints": {
        "budget": 1.0,
        "risk_tolerance_lambda": 1.0,
        "min_weight": 0.0,
        "max_weight": 0.5,
        "allow_shorting": false,
        "cardinality_limit": 4
    }
}
```

## Output Contract

The optimize endpoint returns:

```json
{
    "selected_assets": ["AAPL", "MSFT"],
    "weights": [0.54, 0.46],
    "expected_return": 0.0123,
    "risk": 0.0021,
    "sharpe_ratio": 0.26,
    "solver_used": "classical",
    "performance_metrics": {
        "time_taken": 0.013,
        "iterations": 34,
        "backend": "scipy-slsqp",
        "status": "success"
    },
    "business_impact": {
        "return_improvement_%": 8.4,
        "risk_reduction_%": 4.1,
        "diversification_score": 0.71
    },
    "human_readable_summary": "..."
}
```

## Tests

```bash
pytest -q
```

## Quantum Backends and Fallbacks

- qaoa_solver.py:
    - detects qiskit when installed,
    - falls back to deterministic exact/local QUBO search.
- annealer_solver.py:
    - uses neal/dimod simulated annealing when installed,
    - falls back to deterministic local search when unavailable.

This guarantees reproducible behavior without optional quantum dependencies.
