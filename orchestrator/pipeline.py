"""
Quantix v3 — Core Orchestrator Pipeline
Manages the 9-stage asynchronous execution of all agents.
"""
from __future__ import annotations
import asyncio
import logging
from typing import Dict, Any, List

from models.schemas import (
    QuantixState, RouteInput, SupplyChainInput, CostInput,
    OptimizationStatus, QuantumSolution, SolverType, SolverTier
)
from events.event_bus import AgentEventBus
from events.event_types import EventType, EventSeverity, QuantixEvent
from fallback.fallback_engine import FallbackEngine, FallbackPlan

# Agent Imports
from agents.logistics.layer0.planner_agent import PlannerAgent
from agents.logistics.layer0.router_agent import RouterAgent
from agents.logistics.layer1.weather_impact_agent import WeatherImpactAgent
from agents.logistics.layer1.traffic_intelligence_agent import TrafficIntelligenceAgent
from agents.logistics.layer1.risk_intelligence_agent import RiskIntelligenceAgent
from agents.logistics.layer1.supply_chain_risk_agent import SupplyChainRiskAgent
from agents.logistics.layer1.cost_optimization_agent import CostOptimizationAgent
from agents.logistics.layer1.demand_intelligence_agent import DemandIntelligenceAgent
from agents.logistics.layer1.sustainability_agent import SustainabilityAgent
from agents.logistics.layer1.trust_agent import TrustAgent
from agents.logistics.layer1.constraint_builder_agent import ConstraintBuilderAgent
from agents.logistics.layer2.gnn_route_predictor import GNNRoutePredictor
from agents.logistics.layer2.delay_prediction_agent import DelayPredictionAgent
from agents.logistics.layer2.rl_routing_agent import RLRoutingAgent
from agents.logistics.layer2.fleet_allocation_agent import FleetAllocationAgent
from agents.logistics.complexity.complexity_agent import ComplexityAgent
from solvers.quantum.quantum_engine import QuantumOptimizationEngine
from agents.logistics.layer3.logistics_strategy_agent import LogisticsStrategyAgent
from agents.logistics.layer3.roi_analysis_agent import ROIAnalysisAgent
from agents.logistics.layer3.scenario_simulation_agent import ScenarioSimulationAgent
from agents.logistics.layer3.executive_report_agent import ExecutiveReportAgent

logger = logging.getLogger("quantix.Pipeline")


class QuantixPipeline:
    def __init__(self):
        self.bus = AgentEventBus()
        self.fallback = FallbackEngine(self.bus)
        self.q_engine = QuantumOptimizationEngine()

    async def run(
        self,
        ri: RouteInput,
        sc: SupplyChainInput,
        ci: CostInput,
        initial_events: List[QuantixEvent] = None
    ) -> QuantixState:
        state = QuantixState(route_input=ri, supply_chain_input=sc, cost_input=ci)
        state.status = OptimizationStatus.RUNNING

        try:
            # ── Step 1: Fallback Pre-process ─────────────────
            state.current_step = "fallback_init"
            plan = await self.fallback.process_and_cascade(initial_events or [])
            state.fallback_plan = plan.to_dict()

            # ── Step 2: Layer 0 (Planning & Routing) ──────────
            state.current_step = "layer0"
            planner = PlannerAgent(self.bus)
            router = RouterAgent(self.bus)
            state.planner_output = await planner.run(ri, sc, ci)
            state.router_output = await router.run(ri, state.planner_output)

            # Step 3: Layer 1 (Parallel Intelligence) ───────
            state.current_step = "layer1"
            weather_a = WeatherImpactAgent(self.bus)
            traffic_a = TrafficIntelligenceAgent(self.bus)
            risk_a = RiskIntelligenceAgent(self.bus)
            sc_risk_a = SupplyChainRiskAgent(self.bus)
            cost_a = CostOptimizationAgent(self.bus)
            demand_a = DemandIntelligenceAgent(self.bus)
            sustain_a = SustainabilityAgent(self.bus)
            trust_a = TrustAgent(self.bus)

            # Weather must run first as Traffic depends on its output object
            weather_out = await weather_a.run(ri)
            state.weather_output = weather_out

            # Concurrent execution for independent agents
            traffic_out, sc_risk_out, cost_out, sustain_out, trust_out = await asyncio.gather(
                traffic_a.run(ri, weather_out),
                sc_risk_a.run(sc),
                cost_a.run(ci, ri),
                sustain_a.run(ri, state.router_output),
                trust_a.run(sc, ci)
            )
            # Re-run components that need cross-outputs if necessary, or rely on bus reactivity
            # For simplicity, we assume agents are bus-reactive.
            risk_out = await risk_a.run(ri, traffic_out, weather_out)
            demand_out = await demand_a.run(ri, sc, ci)
            
            state.weather_output = weather_out
            state.traffic_output = traffic_out
            state.sc_risk_output = sc_risk_out
            state.cost_output = cost_out
            state.sustainability_output = sustain_out
            state.trust_output = trust_out
            state.route_risk_output = risk_out
            state.demand_output = demand_out

            # Constraint Building (Final Stage of Layer 1)
            builder = ConstraintBuilderAgent(self.bus)
            state.constraint_output = await builder.run(
                ri, ci, state.planner_output, risk_out, traffic_out, weather_out, trust_out, demand_out, state.fallback_plan
            )

            # ── Step 4: Layer 2 (Prediction & RL) ─────────────
            state.current_step = "layer2"
            gnn = GNNRoutePredictor(self.bus)
            delay = DelayPredictionAgent(self.bus)
            rl = RLRoutingAgent(self.bus)
            fleet = FleetAllocationAgent(self.bus)

            gnn_out, delay_out, rl_out, fleet_out = await asyncio.gather(
                gnn.run(ri, state.constraint_output),
                delay.run(ri, weather_out, traffic_out),
                rl.run(ri, state.constraint_output),
                fleet.run(ci, weather_out)
            )
            state.gnn_output = gnn_out
            state.delay_output = delay_out
            state.rl_output = rl_out
            state.fleet_output = fleet_out

            # ── Step 5: Quantum Optimization ──────────────────
            state.current_step = "quantum"
            
            # Using the unified QuantumOptimizationEngine
            state.quantum_solution = await self.q_engine.optimize(
                route_input=ri,
                route_risk=state.route_risk_output,
                supply_chain_risk=state.sc_risk_output,
                cost_output=state.cost_output
            )

            # ── Step 6: Layer 3 (Business & Reporting) ────────
            state.current_step = "layer3"
            strat_a = LogisticsStrategyAgent(self.bus)
            roi_a = ROIAnalysisAgent(self.bus)
            scen_a = ScenarioSimulationAgent(self.bus)
            rep_a = ExecutiveReportAgent(self.bus)

            state.logistics_strategy = await strat_a.run(state.quantum_solution)
            state.roi_analysis = await roi_a.run(state.quantum_solution)
            state.scenario_output = await scen_a.run(state.quantum_solution)
            state.executive_report = await rep_a.run(
                state.logistics_strategy, state.roi_analysis, state.scenario_output, sustain_out, state.quantum_solution
            )

            state.status = OptimizationStatus.COMPLETED
            state.current_step = "done"

        except Exception as e:
            logger.exception("Pipeline failed")
            state.status = OptimizationStatus.FAILED
            state.errors.append(str(e))

        return state
