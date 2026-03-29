import asyncio
import json
import unittest.mock as mock
from orchestrator.langgraph_orchestrator import QuantixOrchestrator
from models.schemas import QuantixState, RouteInput, SupplyChainInput, CostInput

async def debug():
    print("--- Debugging Logistics Orchestrator (Mock DB) ---")
    
    with open("data/sample_dataset_v3.json", "r") as f:
        data = json.load(f)
    
    state = QuantixState(
        route_input=RouteInput(**data["route_input"]),
        supply_chain_input=SupplyChainInput(**data["supply_chain_input"]),
        cost_input=CostInput(**data["cost_input"]),
    )
    
    orchestrator = QuantixOrchestrator()
    
    # Mocking the DB session completely
    mock_db = mock.AsyncMock()
    
    print("Running orchestrator...")
    try:
        final_state = await orchestrator.run(state, mock_db)
        print(f"Status: {final_state.status}")
        if final_state.status == "failed":
             print(f"Error Log: {final_state.agent_logs}")
        else:
             print("Success!")
             print(f"Report: {final_state.executive_report[:200]}...")
    except Exception as e:
        import traceback
        print("Orchestrator CRASHED:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug())
