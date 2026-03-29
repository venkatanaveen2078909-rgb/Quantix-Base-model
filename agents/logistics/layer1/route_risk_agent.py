"""
Quantix — Layer 1: Route Risk Agent
Analyzes route safety, traffic, and weather using LangChain tools.
"""
from __future__ import annotations
import json
from typing import Any, Dict, List

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from models.schemas import RouteInput, RouteRiskOutput, RiskLevel
from utils.helpers import get_logger, timed, log_agent_output, score_to_risk_level
from config.settings import LLM_MODEL

logger = get_logger("RouteRiskAgent")


class RouteRiskAgent:
    """Assess topographical and environmental risks for logistics routes."""

    def __init__(self):
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a logistics risk analyst. Evaluate route risks based on provided data."),
            ("human", "Analyze these routes: {route_data}. Identify high-risk edges and weather/traffic impacts.")
        ])

    @timed("RouteRiskAgent")
    async def run(self, input_data: RouteInput) -> RouteRiskOutput:
        logger.info(f"Analyzing risks for {len(input_data.nodes)} nodes")

        # Mock analysis logic (in production, use LLM + Tool calling)
        risk_scores = {}
        high_risk_edges = []

        for src, dst, dist in input_data.edges:
            edge_key = f"{src}→{dst}"
            # Combine traffic/weather/history
            traffic = input_data.traffic_scores.get(dst, 0.1)
            weather = input_data.weather_scores.get(dst, 0.1)
            delay = input_data.historical_delays.get(dst, 0.1)

            base_risk = (traffic * 0.4 + weather * 0.4 + delay * 0.2)
            risk_scores[edge_key] = round(base_risk, 3)

            if base_risk > 0.6:
                high_risk_edges.append(edge_key)

        avg_risk = sum(risk_scores.values()) / max(len(risk_scores), 1)
        overall_level = RiskLevel[score_to_risk_level(avg_risk)]

        output = RouteRiskOutput(
            route_risk_scores=risk_scores,
            overall_risk_level=overall_level,
            high_risk_edges=high_risk_edges,
            optimization_constraints={
                "time_penalties": {n: risk_scores.get(f"some→{n}", 0.1) * 50 for n in input_data.nodes}
            },
            analysis_summary=f"Analyzed {len(risk_scores)} edges. Found {len(high_risk_edges)} high-risk segments."
        )

        log_agent_output("RouteRiskAgent", output)
        return output
