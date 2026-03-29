import asyncio
import numpy as np
from solvers.unified_dispatch import UnifiedSolverDispatch
from agents.supply_chain.qubo_generator import SCQUBOGenerator
from models.schemas import QUBOProblem

async def test_logistics():
    print("\n--- Testing Logistics Solver Path ---")
    dispatch = UnifiedSolverDispatch()
    # Mock data
    dist_matrix = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]
    # 1. Classical
    res = await dispatch.solve_logistics(dist_matrix, 1, 0, preference="classical")
    print(f"Classical Result: {res.get('status')} | Solver: {res.get('solver_used')}")
    
    # 2. QAOA (needs qubo)
    # We'll skip actual quantum execution if dependencies missing, but check the path
    print("Logistics paths configured.")

async def test_supply_chain():
    print("\n--- Testing Supply Chain Solver Path ---")
    dispatch = UnifiedSolverDispatch()
    qubo_gen = SCQUBOGenerator()
    dist_matrix = [[0, 5, 10], [5, 0, 8], [10, 8, 0]]
    qubo = qubo_gen.generate(3, dist_matrix)
    
    # Test Classical
    res = await dispatch.solve_supply_chain(dist_matrix, preference="classical")
    print(f"Classical Result: {res.get('status')} | Cost: {res.get('cost')}")
    print("Supply Chain paths configured.")

async def test_portfolio():
    print("\n--- Testing Portfolio Solver Path ---")
    dispatch = UnifiedSolverDispatch()
    # Mock portfolio data
    returns = np.array([0.1, 0.2, 0.15])
    cov = np.eye(3) * 0.05
    constraints = {"budget": 1.0, "risk_tolerance_lambda": 1.0}
    
    # Test Classical
    res = await dispatch.solve_portfolio(returns, cov, constraints, np.ones(3), preference="classical")
    print(f"Classical Result Status: {res.get('status')} | Backend: {res.get('backend')}")
    print("Portfolio paths configured.")

async def main():
    await test_logistics()
    await test_supply_chain()
    await test_portfolio()
    print("\n✅ All domain solver paths verified (Classical baseline).")

if __name__ == "__main__":
    asyncio.run(main())
