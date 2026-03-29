from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from db.database import get_db
from db.repository import OptimizationRepository
from models.schemas import QuantixState, RouteInput, SupplyChainInput, CostInput
from orchestrator.langgraph_orchestrator import QuantixOrchestrator

router = APIRouter(prefix="/logistics", tags=["logistics"])
orchestrator = QuantixOrchestrator()

@router.post("/optimize", response_model=dict)
async def run_optimization(
    route_input: RouteInput,
    supply_chain_input: SupplyChainInput,
    cost_input: CostInput,
    solver_preference: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    state = QuantixState(
        route_input=route_input,
        supply_chain_input=supply_chain_input,
        cost_input=cost_input,
        solver_preference=solver_preference,
    )
    final_state = await orchestrator.run(state, db)
    if str(final_state.status).lower() == "failed":
        raise HTTPException(status_code=500, detail="Optimization failed")
    return {
        "run_id": final_state.run_id,
        "status": final_state.status,
        "executive_report": final_state.executive_report,
    }

@router.get("/history", response_model=List[dict])
async def get_history(limit: int = 10, db: AsyncSession = Depends(get_db)):
    repo = OptimizationRepository(db)
    runs = await repo.list_runs(limit=limit)
    return [
        {
            "id": str(r.id),
            "status": r.status,
            "created_at": r.created_at.isoformat(),
            "cost_reduction": r.cost_reduction_pct,
            "roi": r.roi_pct,
        }
        for r in runs
    ]

@router.get("/run/{run_id}", response_model=dict)
async def get_run_details(run_id: str, db: AsyncSession = Depends(get_db)):
    repo = OptimizationRepository(db)
    run = await repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "id": str(run.id),
        "status": run.status,
        "executive_report": run.executive_report,
        "quantum_solution": run.quantum_solution,
        "agent_logs": run.agent_logs,
    }
