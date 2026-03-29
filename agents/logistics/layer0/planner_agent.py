"""
Quantix v3 — Layer 0: Planner Agent
Understands task, detects weather/road context, and sets optimization
weights + weather_sensitivity for all downstream agents.
"""
from __future__ import annotations
import logging
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import (
    RouteInput, SupplyChainInput, CostInput,
    PlannerOutput, RiskLevel, WeatherCondition
)
from events.event_bus import AgentEventBus
from events.event_types import EventType, EventSeverity, QuantixEvent

logger = logging.getLogger("quantix.PlannerAgent")


class PlannerAgent:

    def __init__(self, bus: AgentEventBus):
        self.bus = bus
        self.llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Quantix Strategic Planner AI."),
            ("human", "Analyze task context: {context}. Recommend optimization weights.")
        ])

    async def run(
        self,
        route_input: RouteInput,
        sc_input: SupplyChainInput,
        cost_input: CostInput,
    ) -> PlannerOutput:
        await self.bus.publish(QuantixEvent(
            event_type=EventType.AGENT_STARTED,
            source_agent="PlannerAgent",
            severity=EventSeverity.INFO,
            payload={},
            message="Planner analyzing task context",
        ))

        avg_reliability = (
            sum(sc_input.supplier_reliability.values())
            / max(len(sc_input.supplier_reliability), 1)
        )
        has_deadlines = bool(route_input.delivery_deadlines)
        num_nodes = len(route_input.nodes)
        weather = route_input.current_weather

        # Determine weather sensitivity
        weather_sensitivity = {
            WeatherCondition.CLEAR:      0.1,
            WeatherCondition.RAIN:       0.55,
            WeatherCondition.HEAVY_RAIN: 0.85,
            WeatherCondition.STORM:      0.95,
            WeatherCondition.SNOW:       0.90,
            WeatherCondition.FOG:        0.65,
            WeatherCondition.HIGH_WIND:  0.75,
        }.get(weather, 0.5)

        # Adjust weights based on context
        if weather in (WeatherCondition.HEAVY_RAIN, WeatherCondition.STORM, WeatherCondition.SNOW):
            wc, wt, wr = 0.20, 0.25, 0.55    # risk-dominant in bad weather
            obj = "risk"
            hard = [
                "Avoid mud-prone unpaved roads",
                "Avoid flood zones",
                "Each delivery node visited exactly once",
                "Truck capacity not exceeded",
            ]
            soft = [
                "Prefer paved roads under rain conditions",
                "Prefer lighter vehicles on low-clearance routes",
            ]
            risk_tol = RiskLevel.HIGH
        elif has_deadlines and avg_reliability < 0.7:
            wc, wt, wr = 0.25, 0.40, 0.35
            obj = "balanced"
            hard = ["Time windows must be satisfied", "Truck capacity not exceeded"]
            soft = ["Minimize risk on critical delivery segments"]
            risk_tol = RiskLevel.HIGH
        elif has_deadlines:
            wc, wt, wr = 0.20, 0.50, 0.30
            obj = "time"
            hard = ["Time windows must be satisfied", "Truck capacity not exceeded"]
            soft = ["Minimize cost on non-critical segments"]
            risk_tol = RiskLevel.MEDIUM
        elif num_nodes > 15:
            wc, wt, wr = 0.50, 0.20, 0.30
            obj = "cost"
            hard = ["Each delivery node visited exactly once", "Truck capacity not exceeded"]
            soft = ["Prefer shorter routes where feasible"]
            risk_tol = RiskLevel.MEDIUM
        else:
            wc, wt, wr = 0.35, 0.30, 0.35
            obj = "balanced"
            hard = ["Each delivery node visited exactly once", "Truck capacity not exceeded"]
            soft = ["Balance load across fleet"]
            risk_tol = RiskLevel.MEDIUM

        output = PlannerOutput(
            task_summary=(
                f"Optimize {num_nodes} delivery nodes with {cost_input.num_trucks} trucks. "
                f"Weather: {weather.value}. Supplier reliability: {avg_reliability:.0%}. "
                f"Time constraints: {'YES' if has_deadlines else 'NO'}."
            ),
            primary_objective=obj,
            weight_cost=wc,
            weight_time=wt,
            weight_risk=wr,
            hard_constraints=hard,
            soft_constraints=soft,
            risk_tolerance=risk_tol,
            weather_sensitivity=weather_sensitivity,
            planning_notes=(
                f"Weather sensitivity set to {weather_sensitivity:.2f} for {weather.value}. "
                f"Objective: {obj.upper()}."
            ),
        )

        await self.bus.publish(QuantixEvent(
            event_type=EventType.AGENT_COMPLETED,
            source_agent="PlannerAgent",
            severity=EventSeverity.INFO,
            payload={"objective": obj, "weather_sensitivity": weather_sensitivity},
            message=f"Plan ready — objective={obj}, risk_tolerance={risk_tol.value}",
        ))
        logger.info(f"[PlannerAgent] Done: {output.task_summary}")
        return output
