"""
Quantix — Pipeline Verification Demo
Runs full end-to-end optimization using real Mumbai logistics data.
"""
from __future__ import annotations
import asyncio

from models.schemas import QuantixState
from orchestrator.langgraph_orchestrator import QuantixOrchestrator
from db.database import AsyncSessionLocal, create_tables
from utils.data_ingestion import load_mumbai_data


async def main():
    print("Starting Quantix Logistics AI — Mumbai Dataset...")

    await create_tables()

    import json
    import os

    # Load local JSON dataset from data folder as requested
    json_path = os.path.join("data", "sample_dataset_v3.json")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r") as f:
        dataset = json.load(f)

    # dataset is already a dict with route_input, supply_chain_input, cost_input
    # but we need to convert them to Pydantic models for the state
    from models.schemas import RouteInput, SupplyChainInput, CostInput
    
    route_in = RouteInput(**dataset["route_input"])
    sc_in = SupplyChainInput(**dataset["supply_chain_input"])
    cost_in = CostInput(**dataset["cost_input"])

    orchestrator = QuantixOrchestrator()
    initial_state = QuantixState(
        route_input=route_in,
        supply_chain_input=sc_in,
        cost_input=cost_in,
    )

    print("\nOrchestrating multi-agent pipeline...")
    async with AsyncSessionLocal() as db:
        final_state = await orchestrator.run(initial_state, db)
        await db.commit()

    if str(final_state.status).lower() == "failed":
        print("Pipeline failed:", final_state.errors)
        return

    print("\n" + "=" * 60)
    print("  MUMBAI LOGISTICS OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Status: {final_state.status}")

    if final_state.quantum_solution:
        sol = final_state.quantum_solution
        print(f"Solver Used:     {sol.solver_used} ({sol.solver_tier})")
        print(f"Baseline Cost:   ₹{sol.total_baseline_cost:,.2f}")
        print(f"Optimized Cost:  ₹{sol.total_optimized_cost:,.2f}")
        print(f"Cost Reduction:  {sol.cost_reduction_pct}%")
        print(f"\nOptimized Routes ({len(sol.optimized_routes)} trucks):")
        for i, route in enumerate(sol.optimized_routes):
            print(f"  Vehicle {i + 1}: {' → '.join(route)}")

    if final_state.executive_report:
        print("\nEXECUTIVE SUMMARY")
        print(final_state.executive_report.executive_summary)

    print("\n" + "=" * 60)
    print("Optimization Complete.")


if __name__ == "__main__":
    asyncio.run(main())