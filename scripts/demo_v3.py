"""
Quantix v3 — Verification Demo
Runs the full 9-stage pipeline with simulated fallback events.
"""
from __future__ import annotations
import asyncio
import logging
import json
from datetime import datetime

from orchestrator.pipeline import QuantixPipeline
from models.schemas import (
    RouteInput, SupplyChainInput, CostInput, NodeMetadata,
    RoadCondition, WeatherCondition, VehicleType
)
from events.event_types import EventType, EventSeverity, QuantixEvent

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("QuantixDemo")

from utils.data_ingestion import load_dataset


async def run_demo():
    logger.info("🚀 Starting Quantix v3 Verification Demo")
    
    # 1. Load Data from dataset file
    dataset = load_dataset("data/sample_dataset_v3.json")
    ri = dataset["route_input"]
    sc = dataset["supply_chain_input"]
    ci = dataset["cost_input"]

    # 2. Define Initial Real-World Events (Simulated Incident)
    initial_events = [
        QuantixEvent(
            event_type=EventType.HEAVY_RAIN_DETECTED,
            source_agent="ExternalSensor",
            severity=EventSeverity.WARNING,
            payload={"intensity": "heavy"},
            affected_nodes=["B"],
            affected_edges=["A→B"],
            message="Heavy rain reported over gravel segment B"
        )
    ]

    # 3. Execute Pipeline
    pipeline = QuantixPipeline()
    state = await pipeline.run(ri, sc, ci, initial_events)

    # 4. Results Summary
    if state.status == "completed":
        logger.info("✅ Pipeline Completed Successfully!")
        report = state.executive_report
        print("\n" + "="*50)
        print("          EXECUTIVE SUMMARY")
        print("="*50)
        print(f"Summary: {report.executive_summary}")
        print("\nKey Findings:")
        for f in report.key_findings:
            print(f"  - {f}")
        print("\nStrategic Actions:")
        for a in report.strategic_recommendations:
            print(f"  - {a}")
        print("\nROI Highlights:")
        print(f"  - Cost Reduction: {report.optimization_highlights['reduction']}%")
        print(f"  - Annual Savings: ${report.optimization_highlights['savings']:,.2f}")
        print("="*50)
    else:
        logger.error(f"❌ Pipeline Failed: {state.errors}")

if __name__ == "__main__":
    asyncio.run(run_demo())
