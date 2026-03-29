import numpy as np
from fastapi import APIRouter, HTTPException
from typing import Any, Dict

from models.schemas import PortfolioOptimizeRequest, PortfolioOptimizeResponse
from agents.portfolio.alpha_agent import AlphaAgent
from agents.portfolio.constraint_encoder import ConstraintEncoder
from agents.portfolio.data_agent import DataAgent
from agents.portfolio.execution_agent import ExecutionAgent
from agents.portfolio.model_selector import ModelSelector
from agents.portfolio.qubo_generator import QUBOGenerator
from agents.portfolio.report_agent import ReportAgent
from utils.helpers import get_logger

router = APIRouter(prefix="/portfolio", tags=["portfolio"])
logger = get_logger("api.portfolio")

def run_portfolio_pipeline(payload: PortfolioOptimizeRequest) -> Dict[str, Any]:
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

    constraints_dict = payload.constraints.model_dump()
    encoded = constraint_encoder.encode(len(investable_ids), constraints_dict)

    qubo_result = None
    if selection.solver_key in {"qaoa", "annealer"}:
        qubo_result = qubo_generator.generate(
            expected_returns=expected_returns,
            covariance=covariance,
            risk_lambda=payload.constraints.risk_tolerance_lambda,
            cardinality_limit=payload.constraints.cardinality_limit,
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

@router.post("/optimize", response_model=PortfolioOptimizeResponse)
async def optimize_portfolio(payload: PortfolioOptimizeRequest):
    try:
        result = run_portfolio_pipeline(payload)
        return result
    except Exception as exc:
        logger.exception("Portfolio optimization failed")
        raise HTTPException(status_code=400, detail=str(exc))
