from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from config.settings import DATABASE_URL
import uuid
from datetime import datetime

# ✅ BASE
class Base(DeclarativeBase):
    pass

# ✅ MODELS
class OptimizationRun(Base):
    __tablename__ = "optimization_runs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String, default="pending")
    current_step = Column(String, default="start")
    errors = Column(JSON, default=list)
    
    # Inputs
    num_nodes = Column(Integer, nullable=True)
    num_edges = Column(Integer, nullable=True)
    num_suppliers = Column(Integer, nullable=True)
    num_trucks = Column(Integer, nullable=True)
    
    # Layer Outputs
    route_risk_level = Column(String, nullable=True)
    route_risk_output = Column(JSON, nullable=True)
    supply_chain_risk_level = Column(String, nullable=True)
    supply_chain_risk_output = Column(JSON, nullable=True)
    
    baseline_cost = Column(Float, nullable=True)
    cost_optimization_output = Column(JSON, nullable=True)
    
    # Quantum Solution
    solver_used = Column(String, nullable=True)
    solver_tier = Column(String, nullable=True)
    qubo_variables = Column(Integer, nullable=True)
    solution_energy = Column(Float, nullable=True)
    optimized_cost = Column(Float, nullable=True)
    cost_reduction_pct = Column(Float, nullable=True)
    quantum_solution = Column(JSON, nullable=True)
    
    # Business Logic
    logistics_strategy = Column(JSON, nullable=True)
    annual_savings = Column(Float, nullable=True)
    payback_months = Column(Float, nullable=True)
    five_year_npv = Column(Float, nullable=True)
    roi_pct = Column(Float, nullable=True)
    roi_analysis = Column(JSON, nullable=True)
    executive_report = Column(JSON, nullable=True)
    
    agent_logs = Column(JSON, default=list)

class SolverAudit(Base):
    __tablename__ = "solver_audits"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("optimization_runs.id"))
    num_variables = Column(Integer)
    tier_selected = Column(String)
    solver_selected = Column(String)
    reason = Column(String)
    dwave_available = Column(Boolean, default=False)
    qaoa_available = Column(Boolean, default=False)
    actual_solve_time_ms = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

# ✅ ENGINE
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True
)

# ✅ SESSION
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# ✅ DEPENDENCY
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# ✅ CREATE TABLES
async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)