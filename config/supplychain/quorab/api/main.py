import os
import sys
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..agents.report_agent import BusinessReportAgent
from ..agents.qubo_generator import QUBOBaseGeneratorAgent
from ..agents.model_selector import ModelSelectorAgent
from ..agents.execution_agent import ExecutionAgent
from ..agents.data_agent import DataProcessingAgent
from ..agents.constraint_encoder import ConstraintEncoderAgent
from ..agents.analyzer_agent import ProblemAnalyzerAgent


app = FastAPI(title="Quantix Systems Optimization API", version="1.0.0")


class OptimizationRequest(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


@app.post("/optimize")
async def optimize_supply_chain(request: OptimizationRequest):
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
            qubo_gen = QUBOBaseGeneratorAgent()
            qubo = qubo_gen.generate(encoded_result)

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
