from fastapi import APIRouter, HTTPException
from models.schemas import SCOptimizationRequest
from agents.supply_chain.data_agent import DataProcessingAgent
from agents.supply_chain.analyzer_agent import ProblemAnalyzerAgent
from agents.supply_chain.model_selector import ModelSelectorAgent
from agents.supply_chain.constraint_encoder import ConstraintEncoderAgent
from agents.supply_chain.qubo_generator import SCQUBOGenerator
from agents.supply_chain.execution_agent import ExecutionAgent
from agents.supply_chain.report_agent import BusinessReportAgent

router = APIRouter(prefix="/supply-chain", tags=["supply-chain"])

@router.post("/optimize")
async def optimize_supply_chain(request: SCOptimizationRequest):
    try:
        raw_data = {
            "nodes": [n for n in request.nodes],
            "edges": [e for e in request.edges],
        }

        # 1. Data Processing
        data_agent = DataProcessingAgent()
        processed_data = data_agent.process(raw_data)

        # 2. Problem Analyzer
        analyzer = ProblemAnalyzerAgent()
        analysis_result = analyzer.process(processed_data)

        # 3. Model Selector
        selector = ModelSelectorAgent()
        model_type = selector.select_model(analysis_result)

        # 4. Constraint Encoding
        encoder = ConstraintEncoderAgent()
        encoded_result = encoder.encode(analysis_result)

        # 5. QUBO Generator
        qubo = None
        if model_type in ["qaoa", "annealer"]:
            qubo_gen = SCQUBOGenerator()
            qubo = qubo_gen.generate(analysis_result["num_nodes"], analysis_result["distance_matrix"])

        # 6. Execution
        execution_agent = ExecutionAgent()
        execution_result = execution_agent.execute(
            model_type, analysis_result, qubo)

        # 7. Business Report
        report_agent = BusinessReportAgent()
        final_report = report_agent.generate_report(execution_result)

        return final_report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
