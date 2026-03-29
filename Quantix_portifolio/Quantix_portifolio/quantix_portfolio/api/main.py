from __future__ import annotations

from typing import Any, Dict

import numpy as np
from fastapi import FastAPI, HTTPException

from agents.alpha_agent import AlphaAgent
from agents.constraint_encoder import ConstraintEncoder
from agents.data_agent import DataAgent
from agents.execution_agent import ExecutionAgent
from agents.model_selector import ModelSelector
from agents.qubo_generator import QUBOGenerator
from agents.report_agent import ReportAgent
from utils.logging_utils import configure_logging, get_logger
from utils.validation import ConstraintInput, OptimizeRequest, OptimizeResponse


configure_logging()
logger = get_logger("api.main")

app = FastAPI(title="Quantix Systems API", version="1.0.0")


def run_pipeline(payload: OptimizeRequest) -> Dict[str, Any]:
    data_agent = DataAgent()
    alpha_agent = AlphaAgent()
    model_selector = ModelSelector()
    constraint_encoder = ConstraintEncoder()
    qubo_generator = QUBOGenerator()
    execution_agent = ExecutionAgent()
    report_agent = ReportAgent()

    processed = data_agent.process(
        market_data=[item.model_dump() for item in payload.market_data] if payload.market_data else None,
        fundamentals_data=payload.fundamentals_data,
        earnings_data=[item.model_dump() for item in payload.earnings_data] if payload.earnings_data else None,
        market_data_path=payload.market_data_path,
        fundamentals_path=payload.fundamentals_path,
        earnings_path=payload.earnings_path,
    )

    alpha_result = alpha_agent.generate(
        asset_ids=processed.asset_ids,
        fundamentals_df=processed.fundamentals_df,
        earnings_df=processed.earnings_df,
    )

    investable_ids = alpha_result.investable_asset_ids
    if not investable_ids:
        raise ValueError("No investable assets after alpha filtering")

    expected_returns = processed.expected_returns.loc[investable_ids].to_numpy(dtype=float)
    covariance = processed.covariance.loc[investable_ids, investable_ids].to_numpy(dtype=float)
    alpha_signs = np.array([alpha_result.signals.get(asset_id, 1) for asset_id in investable_ids], dtype=float)

    selection = model_selector.select(len(investable_ids))

    constraints_model: ConstraintInput = payload.constraints
    constraints_dict = constraints_model.model_dump()
    encoded = constraint_encoder.encode(len(investable_ids), constraints_dict)

    qubo_result = None
    if selection.solver_key in {"qaoa", "annealer"}:
        qubo_result = qubo_generator.generate(
            expected_returns=expected_returns,
            covariance=covariance,
            risk_lambda=constraints_model.risk_tolerance_lambda,
            cardinality_limit=constraints_model.cardinality_limit,
            alpha_signs=alpha_signs,
        )

    execution_result = execution_agent.execute(
        solver_key=selection.solver_key,
        expected_returns=expected_returns,
        covariance=covariance,
        constraints=constraints_dict,
        alpha_signs=alpha_signs,
        qubo=qubo_result.qubo_dict if qubo_result else None,
    )

    weights = np.array(execution_result.weights, dtype=float)
    report = report_agent.generate(
        asset_ids=investable_ids,
        weights=weights,
        expected_returns=expected_returns,
        covariance=covariance,
        solver_used=execution_result.solver_used,
        performance_metrics=execution_result.performance_metrics,
        selection_rationale=selection.rationale,
        constraint_summary=encoded.symbolic_expressions,
    )

    return report.json_output


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/schema")
def schema() -> Dict[str, Any]:
    return OptimizeRequest.model_json_schema()


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(payload: OptimizeRequest) -> Dict[str, Any]:
    try:
        result = run_pipeline(payload)
        return result
    except Exception as exc:
        logger.exception("Optimization failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
