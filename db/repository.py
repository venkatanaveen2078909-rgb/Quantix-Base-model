"""
Quantix — Database Repository (v2)
CRUD operations for optimization runs and audit logs.

This module is imported by the FastAPI app at startup, so it must define
`OptimizationRepository` at module import time.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import OptimizationRun, SolverAudit
from models.schemas import QuantixState, SolverDecision
from utils.helpers import get_logger

logger = get_logger("Repository")


class OptimizationRepository:
    """Repository for optimization run persistence."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_run(self, state: QuantixState) -> OptimizationRun:
        """Persist a new optimization run from initial state."""
        run = OptimizationRun(
            id=UUID(state.run_id),
            created_at=datetime.fromisoformat(state.created_at),
            status=state.status.value,
            num_nodes=len(state.route_input.nodes) if state.route_input else None,
            num_edges=len(state.route_input.edges) if state.route_input else None,
            num_suppliers=len(state.supply_chain_input.suppliers)
            if state.supply_chain_input
            else None,
            num_trucks=state.cost_input.num_trucks if state.cost_input else None,
        )
        self.db.add(run)
        await self.db.flush()
        return run

    async def update_run(self, state: QuantixState) -> Optional[OptimizationRun]:
        """Update run record with current pipeline state."""
        result = await self.db.execute(
            select(OptimizationRun).where(OptimizationRun.id == UUID(state.run_id))
        )
        run = result.scalar_one_or_none()
        if not run:
            logger.warning(f"Run {state.run_id} not found for update")
            return None

        run.status = state.status.value
        run.current_step = state.current_step
        run.errors = state.errors

        if state.route_risk_output:
            run.route_risk_level = state.route_risk_output.overall_risk_level.value
            run.route_risk_output = state.route_risk_output.model_dump()

        if state.sc_risk_output:
            run.supply_chain_risk_level = state.sc_risk_output.risk_level.value
            run.supply_chain_risk_output = state.sc_risk_output.model_dump()

        # Schema uses `cost_output`, while the DB uses `cost_optimization_output`.
        if state.cost_output:
            run.baseline_cost = state.cost_output.total_baseline_cost
            run.cost_optimization_output = state.cost_output.model_dump()

        if state.quantum_solution:
            qs = state.quantum_solution
            run.solver_used = qs.solver_used.value
            run.solver_tier = qs.solver_tier.value
            run.qubo_variables = qs.optimization_metadata.get("qubo_size")
            run.solution_energy = qs.solution_energy
            run.optimized_cost = qs.total_optimized_cost
            run.cost_reduction_pct = qs.cost_reduction_pct
            run.quantum_solution = qs.model_dump()

        if state.logistics_strategy:
            run.logistics_strategy = state.logistics_strategy.model_dump()

        if state.roi_analysis:
            roi = state.roi_analysis
            run.annual_savings = roi.annual_cost_savings_usd
            run.payback_months = roi.payback_period_months
            run.five_year_npv = roi.five_year_npv_usd
            run.roi_pct = roi.roi_pct
            run.roi_analysis = roi.model_dump()

        if state.executive_report:
            run.executive_report = state.executive_report.model_dump()
            run.completed_at = datetime.utcnow()

        run.agent_logs = state.agent_logs

        await self.db.flush()
        return run

    async def get_run(self, run_id: str) -> Optional[OptimizationRun]:
        result = await self.db.execute(
            select(OptimizationRun).where(OptimizationRun.id == UUID(run_id))
        )
        return result.scalar_one_or_none()

    async def list_runs(
        self,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> List[OptimizationRun]:
        query = select(OptimizationRun).order_by(desc(OptimizationRun.created_at))
        if status:
            query = query.where(OptimizationRun.status == status)
        query = query.limit(limit).offset(offset)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def log_solver_audit(
        self,
        run_id: str,
        decision: SolverDecision,
        dwave_available: bool,
        qaoa_available: bool,
        actual_solve_time_ms: Optional[int] = None,
    ) -> SolverAudit:
        audit = SolverAudit(
            run_id=UUID(run_id),
            num_variables=decision.num_variables,
            tier_selected=decision.tier.value,
            solver_selected=decision.solver.value,
            reason=decision.reason,
            dwave_available=dwave_available,
            qaoa_available=qaoa_available,
            actual_solve_time_ms=actual_solve_time_ms,
        )
        self.db.add(audit)
        await self.db.flush()
        return audit

    async def get_solver_stats(self) -> dict:
        """Aggregate solver usage statistics."""
        result = await self.db.execute(
            select(
                OptimizationRun.solver_used,
                func.count(OptimizationRun.id).label("count"),
                func.avg(OptimizationRun.cost_reduction_pct).label("avg_reduction"),
            ).group_by(OptimizationRun.solver_used)
        )
        rows = result.all()
        return {
            row.solver_used: {
                "count": row.count,
                "avg_cost_reduction_pct": round(row.avg_reduction or 0, 2),
            }
            for row in rows
        }