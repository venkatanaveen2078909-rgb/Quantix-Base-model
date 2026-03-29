import asyncio
import logging
import json
import os
import pandas as pd
from typing import Dict, Any

from models.schemas import (
    RouteInput, SupplyChainInput, CostInput, 
    PortfolioOptimizeRequest, SCOptimizationRequest
)
from orchestrator.langgraph_orchestrator import QuantixOrchestrator
from db.database import create_tables, AsyncSessionLocal

# Domain-specific agent imports for standalone verification
from agents.supply_chain.data_agent import DataProcessingAgent as SCDataAgent
from agents.supply_chain.analyzer_agent import ProblemAnalyzerAgent as SCAnalyzer
from agents.supply_chain.model_selector import ModelSelectorAgent as SCSelector
from agents.supply_chain.execution_agent import ExecutionAgent as SCExecution
from agents.supply_chain.report_agent import BusinessReportAgent as SCReporter

from agents.portfolio.data_agent import DataAgent as PortDataAgent
from agents.portfolio.alpha_agent import AlphaAgent as PortAlphaAgent
from agents.portfolio.model_selector import ModelSelector as PortSelector
from agents.portfolio.execution_agent import ExecutionAgent as PortExecution
from agents.portfolio.report_agent import ReportAgent as PortReporter

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("UnifiedVerify")

async def verify_logistics():
    logger.info("--- Testing Logistics Domain (V3 Orchestrator) ---")
    from utils.data_ingestion import load_dataset
    dataset = load_dataset("data/sample_dataset_v3.json")
    
    orchestrator = QuantixOrchestrator()
    from models.schemas import QuantixState
    state = QuantixState(
        route_input=dataset["route_input"],
        supply_chain_input=dataset["supply_chain_input"],
        cost_input=dataset["cost_input"],
    )
    
    await create_tables()
    async with AsyncSessionLocal() as db:
        final_state = await orchestrator.run(state, db)
        if final_state.status == "completed":
            logger.info("✅ Logistics Verification Successful")
            return True
        else:
            logger.error(f"❌ Logistics Verification Failed: {final_state.errors}")
            return False

async def verify_supply_chain():
    logger.info("--- Testing Supply Chain Domain ---")
    with open("data/supply_chain_logistics.json", "r") as f:
        data = json.load(f)
    
    try:
        data_agent = SCDataAgent()
        processed = data_agent.process(data)
        
        analyzer = SCAnalyzer()
        analysis = analyzer.process(processed)
        
        selector = SCSelector()
        model = selector.select_model(analysis)
        
        execution = SCExecution()
        res = execution.execute(model, analysis)
        
        reporter = SCReporter()
        report = reporter.generate_report(res)
        
        logger.info("✅ Supply Chain Verification Successful")
        return True
    except Exception as e:
        logger.error(f"❌ Supply Chain Verification Failed: {str(e)}")
        return False

async def verify_portfolio():
    logger.info("--- Testing Portfolio Domain ---")
    try:
        data_agent = PortDataAgent()
        processed = data_agent.process(
            market_data_path="data/portfolio_prices.csv",
            fundamentals_path="data/portfolio_fundamentals.csv",
            earnings_path="data/portfolio_earnings.csv"
        )
        
        alpha_agent = PortAlphaAgent()
        alpha_res = alpha_agent.generate(processed.asset_ids, processed.fundamentals_df, processed.earnings_df)
        
        selector = PortSelector()
        selection = selector.select(len(alpha_res.investable_asset_ids))
        
        execution = PortExecution()
        import numpy as np
        investable_ids = alpha_res.investable_asset_ids
        returns = processed.expected_returns.loc[investable_ids].to_numpy()
        cov = processed.covariance.loc[investable_ids, investable_ids].to_numpy()
        
        res = execution.execute(
            solver_key=selection.solver_key,
            expected_returns=returns,
            covariance=cov,
            constraints={"budget": 1.0, "risk_tolerance_lambda": 1.0},
            alpha_signs=np.ones(len(investable_ids))
        )
        
        reporter = PortReporter()
        report = reporter.generate(
            asset_ids=investable_ids,
            weights=np.array(res.weights),
            expected_returns=returns,
            covariance=cov,
            solver_used=res.solver_used,
            performance_metrics=res.performance_metrics,
            selection_rationale=selection.rationale,
            constraint_summary=["budget: 1.0"]
        )
        
        logger.info("✅ Portfolio Verification Successful")
        return True
    except Exception as e:
        logger.error(f"❌ Portfolio Verification Failed: {str(e)}")
        return False

async def main():
    results = {}
    results["logistics"] = await verify_logistics()
    results["supply_chain"] = await verify_supply_chain()
    results["portfolio"] = await verify_portfolio()
    
    print("\n" + "="*40)
    print("      UNIFIED VERIFICATION SUMMARY")
    print("="*40)
    for domain, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{domain:<15}: {status}")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(main())
