"""
Quantix — Multi-Agent Orchestrator (v2)
Powered by LangGraph for DAG orchestration and state management.
"""
from __future__ import annotations
from typing import Dict, Optional

from langgraph.graph import StateGraph, END
from sqlalchemy.ext.asyncio import AsyncSession

from models.schemas import QuantixState, OptimizationStatus
from agents.logistics.layer1.route_risk_agent import RouteRiskAgent
from agents.logistics.layer1.supply_chain_risk_agent import SupplyChainRiskAgent
from agents.logistics.layer1.cost_optimization_agent import CostOptimizationAgent
from agents.logistics.layer3.business_insight_agents import StrategyAgent, ROIAgent, ExecutiveAgent
from solvers.quantum.quantum_engine import QuantumOptimizationEngine
from db.repository import OptimizationRepository
from utils.helpers import get_logger, create_log_entry
from events.event_bus import AgentEventBus

logger = get_logger("Orchestrator")


class QuantixOrchestrator:
    """Orchestrates the multi-agent logistics optimization pipeline."""

    def __init__(self):
        self._builder = StateGraph(QuantixState)
        self._setup_graph()
        self.graph = self._builder.compile()

        # Shared event bus instance (dependency-injected into bus-aware agents)
        self.bus = AgentEventBus()

        # Initialize agents
        self.route_agent = RouteRiskAgent()  # this agent version does not require bus
        self.sc_agent = SupplyChainRiskAgent(bus=self.bus)
        self.cost_agent = CostOptimizationAgent(bus=self.bus)
        self.quantum_engine = QuantumOptimizationEngine()
        self.strategy_agent = StrategyAgent()
        self.roi_agent = ROIAgent()
        self.exec_agent = ExecutiveAgent()

    def _setup_graph(self):
        # Nodes
        self._builder.add_node("analyze_risks", self._node_layer1)
        self._builder.add_node("quantum_optimize", self._node_layer2)
        self._builder.add_node("generate_insights", self._node_layer3)

        # Edges
        self._builder.set_entry_point("analyze_risks")
        self._builder.add_edge("analyze_risks", "quantum_optimize")
        self._builder.add_edge("quantum_optimize", "generate_insights")
        self._builder.add_edge("generate_insights", END)

    async def _node_layer1(self, state: QuantixState) -> Dict:
        logger.info("--- Entering Layer 1: Risk & Cost Analysis ---")
        logs = []

        route_out = await self.route_agent.run(state.route_input)
        logs.append(create_log_entry("RouteRiskAgent", "success", route_out.analysis_summary))

        sc_out = await self.sc_agent.run(state.supply_chain_input)
        logs.append(create_log_entry("SupplyChainAgent", "success", sc_out.analysis_summary))

        # CostOptimizationAgent requires both cost_input and route_input
        cost_out = await self.cost_agent.run(state.cost_input, state.route_input)
        logs.append(create_log_entry("CostAgent", "success", cost_out.analysis_summary))

        return {
            "route_risk_output": route_out,
            "sc_risk_output": sc_out,
            "cost_output": cost_out,
            "agent_logs": state.agent_logs + logs,
            "current_step": "layer1_complete",
        }

    async def _node_layer2(self, state: QuantixState) -> Dict:
        logger.info("--- Entering Layer 2: Quantum Optimization ---")
        try:
            solution = await self.quantum_engine.optimize(
                state.route_input,
                state.route_risk_output,
                state.sc_risk_output,
                state.cost_output,
                state.cost_input.model_dump() if state.cost_input else None,
                preference=state.solver_preference
            )
            return {
                "quantum_solution": solution,
                "current_step": "layer2_complete",
                "agent_logs": state.agent_logs
                + [create_log_entry("QuantumEngine", "success", "Optimization complete")],
            }
        except Exception as e:
            logger.error(f"Quantum optimization failed: {e}")
            return {"status": OptimizationStatus.FAILED, "errors": state.errors + [str(e)]}

    async def _node_layer3(self, state: QuantixState) -> Dict:
        logger.info("--- Entering Layer 3: Business Insights ---")
        if not state.quantum_solution:
            logger.warning("No quantum solution found. Skipping Layer 3 insights.")
            return {"status": OptimizationStatus.FAILED}

        strategy = await self.strategy_agent.run(state.quantum_solution)
        roi = await self.roi_agent.run(state.quantum_solution)
        report = await self.exec_agent.run(strategy, roi, state.quantum_solution)

        return {
            "logistics_strategy": strategy,
            "roi_analysis": roi,
            "executive_report": report,
            "status": OptimizationStatus.COMPLETED,
            "current_step": "pipeline_complete",
            "agent_logs": state.agent_logs
            + [create_log_entry("BusinessAgents", "success", "Insights generated")],
        }

    async def run(
        self, initial_state: QuantixState, db: Optional[AsyncSession] = None
    ) -> QuantixState:
        logger.info(f"Starting Quantix Pipeline [id={initial_state.run_id}]")
        repo = OptimizationRepository(db) if db else None

        if repo:
            await repo.create_run(initial_state)

        final_state_dict = await self.graph.ainvoke(initial_state)
        final_state = initial_state.model_copy(update=final_state_dict)

        if repo:
            await repo.update_run(final_state)
            await db.commit()

        logger.info(f"Quantix Pipeline Completed [id={final_state.run_id}]")
        return final_state