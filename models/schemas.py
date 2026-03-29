"""
Quantix v3 — Shared Data Models
Complete schema for the full 9-stage pipeline with inter-agent events.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import uuid


# ═══════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════
from enum import Enum

class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    partial = "partial"

class RiskLevel(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

class SolverType(str, Enum):
    DWAVE_LEAP           = "dwave_leap"
    QAOA                 = "qaoa"
    CLASSICAL_MILP       = "classical_milp"
    SIMULATED_ANNEALING  = "simulated_annealing"

class SolverTier(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"

class OptimizationStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    PARTIAL   = "partial"

class ComplexityClass(str, Enum):
    SIMPLE   = "simple"
    MODERATE = "moderate"
    COMPLEX  = "complex"
    CRITICAL = "critical"

class WeatherCondition(str, Enum):
    CLEAR     = "CLEAR"
    RAIN      = "RAIN"
    HEAVY_RAIN= "HEAVY_RAIN"
    STORM     = "STORM"
    SNOW      = "SNOW"
    FOG       = "FOG"
    HIGH_WIND = "HIGH_WIND"

class RoadCondition(str, Enum):
    PAVED       = "PAVED"
    GRAVEL      = "GRAVEL"
    MUD         = "MUD"
    FLOODED     = "FLOODED"
    ICY         = "ICY"
    CLOSED      = "CLOSED"

class VehicleType(str, Enum):
    HEAVY_TRUCK  = "HEAVY_TRUCK"
    MEDIUM_VAN   = "MEDIUM_VAN"
    LIGHT_VAN    = "LIGHT_VAN"
    MOTORBIKE    = "MOTORBIKE"


# ═══════════════════════════════════════════
# INPUT MODELS
# ═══════════════════════════════════════════

class NodeMetadata(BaseModel):
    node_id: str
    road_condition: RoadCondition = RoadCondition.PAVED
    elevation_m: float = 0.0
    is_bridge: bool = False
    max_vehicle_weight_tons: float = 20.0
    urban: bool = False

class RouteInput(BaseModel):
    nodes: List[str]
    edges: List[Tuple[str, str, float]]
    traffic_scores: Dict[str, float]         = Field(default_factory=dict)
    weather_scores: Dict[str, float]         = Field(default_factory=dict)
    historical_delays: Dict[str, float]      = Field(default_factory=dict)
    delivery_deadlines: Dict[str, float]     = Field(default_factory=dict)
    depot: str
    node_metadata: Dict[str, NodeMetadata]   = Field(default_factory=dict)
    current_weather: WeatherCondition        = WeatherCondition.CLEAR
    weather_forecast_hrs: Dict[str, str]     = Field(default_factory=dict)  # node → forecast

class SupplyChainInput(BaseModel):
    suppliers: List[str]
    supplier_reliability: Dict[str, float]
    shipment_delays: Dict[str, float]  = Field(default_factory=dict)
    warehouse_stock: Dict[str, float]
    demand_forecasts: Dict[str, float]
    lead_times: Dict[str, float]       = Field(default_factory=dict)

class CostInput(BaseModel):
    fuel_cost_per_km: float
    driver_cost_per_hour: float
    warehouse_cost_per_day: float
    num_trucks: int
    truck_capacity: float
    delivery_loads: Dict[str, float]
    route_distances: Dict[str, float]
    total_warehousing_days: int = 1
    vehicle_types: Dict[int, VehicleType] = Field(default_factory=dict)


# ═══════════════════════════════════════════
# LAYER 0 — PLANNER + ROUTER
# ═══════════════════════════════════════════

class PlannerOutput(BaseModel):
    task_summary: str
    primary_objective: str = "balanced"
    weight_cost: float = 0.4
    weight_time: float = 0.3
    weight_risk: float = 0.3
    hard_constraints: List[str]  = Field(default_factory=list)
    soft_constraints: List[str]  = Field(default_factory=list)
    risk_tolerance: RiskLevel    = RiskLevel.MEDIUM
    weather_sensitivity: float   = 0.5     # 0=ignore, 1=max caution
    planning_notes: str          = ""

class InitialRoute(BaseModel):
    route_id: str
    path: List[str]
    total_distance_km: float
    estimated_cost: float
    estimated_time_hr: float
    feasibility_score: float
    uses_unpaved: bool           = False
    max_vehicle_weight_tons: float = 20.0

class RouterOutput(BaseModel):
    candidate_routes: List[InitialRoute]
    graph_stats: Dict[str, Any]  = Field(default_factory=dict)
    connectivity_score: float    = 1.0
    unpaved_segments: List[str]  = Field(default_factory=list)
    bridge_crossings: List[str]  = Field(default_factory=list)
    router_summary: str          = ""


# ═══════════════════════════════════════════
# LAYER 1 — PARALLEL INTELLIGENCE
# ═══════════════════════════════════════════

class RouteRiskOutput(BaseModel):
    route_risk_scores: Dict[str, float]
    overall_risk_level: RiskLevel
    alternate_routes: List[List[str]]   = Field(default_factory=list)
    high_risk_edges: List[str]          = Field(default_factory=list)
    mud_prone_edges: List[str]          = Field(default_factory=list)
    flood_risk_nodes: List[str]         = Field(default_factory=list)
    optimization_constraints: Dict[str, Any] = Field(default_factory=dict)
    analysis_summary: str               = ""
    events_published: List[str]         = Field(default_factory=list)

class TrafficOutput(BaseModel):
    congestion_scores: Dict[str, float]
    peak_hour_penalties: Dict[str, float]
    recommended_windows: Dict[str, str]
    incidents: List[str]                = Field(default_factory=list)
    blocked_edges: List[str]            = Field(default_factory=list)
    stuck_vehicles: List[str]           = Field(default_factory=list)
    overall_level: str                  = "moderate"
    analysis_summary: str               = ""
    events_published: List[str]         = Field(default_factory=list)

class WeatherOutput(BaseModel):
    weather_risk_scores: Dict[str, float]
    road_conditions: Dict[str, RoadCondition]
    severe_zones: List[str]             = Field(default_factory=list)
    mud_zones: List[str]                = Field(default_factory=list)
    flood_zones: List[str]              = Field(default_factory=list)
    disruption_probability: float       = 0.0
    current_condition: WeatherCondition = WeatherCondition.CLEAR
    vehicle_restrictions: Dict[str, VehicleType] = Field(default_factory=dict)
    analysis_summary: str               = ""
    events_published: List[str]         = Field(default_factory=list)

class SupplyChainRiskOutput(BaseModel):
    supply_chain_risk_score: float      = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    vulnerable_suppliers: List[str]     = Field(default_factory=list)
    potential_disruptions: List[str]    = Field(default_factory=list)
    stockout_risk: Dict[str, float]     = Field(default_factory=dict)
    optimization_constraints: Dict[str, Any] = Field(default_factory=dict)
    analysis_summary: str               = ""
    events_published: List[str]         = Field(default_factory=list)

class CostOptimizationOutput(BaseModel):
    total_baseline_cost: float
    cost_breakdown: Dict[str, Any]
    optimization_objective: Dict[str, Any]
    potential_savings_pct: float
    penalty_adjustments: Dict[str, float] = Field(default_factory=dict)
    analysis_summary: str               = ""

class DemandOutput(BaseModel):
    demand_forecasts: Dict[str, float]
    demand_variability: Dict[str, float]
    surge_nodes: List[str]              = Field(default_factory=list)
    priority_deliveries: List[str]      = Field(default_factory=list)
    fleet_recommendation: int           = 4
    analysis_summary: str               = ""

class SustainabilityOutput(BaseModel):
    carbon_footprint_kg: Dict[str, float]
    total_emissions_kg: float           = 0.0
    green_routes: List[str]             = Field(default_factory=list)
    ev_compatible: List[str]            = Field(default_factory=list)
    sustainability_score: float         = 0.5
    emissions_reduction_pct: float      = 0.0
    analysis_summary: str               = ""

class TrustOutput(BaseModel):
    driver_trust: Dict[str, float]
    supplier_trust: Dict[str, float]
    compliance_flags: List[str]         = Field(default_factory=list)
    blacklisted: List[str]              = Field(default_factory=list)
    overall_trust: float                = 1.0
    audit_trail: List[str]              = Field(default_factory=list)
    analysis_summary: str               = ""
    events_published: List[str]         = Field(default_factory=list)

class ConstraintOutput(BaseModel):
    hard_constraints: Dict[str, Any]    = Field(default_factory=dict)
    soft_constraints: Dict[str, Any]    = Field(default_factory=dict)
    penalty_weights: Dict[str, float]   = Field(default_factory=dict)
    fallback_penalties: Dict[str, float]= Field(default_factory=dict)
    conflict_pairs: List[Tuple[str, str]]= Field(default_factory=list)
    total_count: int                    = 0
    constraint_summary: str             = ""


# ═══════════════════════════════════════════
# PREDICTION LAYER
# ═══════════════════════════════════════════

class GNNOutput(BaseModel):
    node_embeddings: Dict[str, List[float]]
    predicted_paths: List[List[str]]
    path_scores: Dict[str, float]
    bottleneck_nodes: List[str]         = Field(default_factory=list)
    centrality: Dict[str, float]        = Field(default_factory=dict)
    confidence: float                   = 0.85
    model_summary: str                  = ""

class DelayOutput(BaseModel):
    predicted_delays: Dict[str, float]
    delay_probs: Dict[str, float]
    critical_segments: List[str]        = Field(default_factory=list)
    total_expected_delay_min: float     = 0.0
    on_time_probability: float          = 1.0
    model_summary: str                  = ""


# ═══════════════════════════════════════════
# RL LAYER
# ═══════════════════════════════════════════

class RLRoutingOutput(BaseModel):
    rl_routes: List[List[str]]
    policy_scores: Dict[str, float]
    improvement_pct: float              = 0.0
    convergence_episodes: int           = 0
    avoided_edges: List[str]            = Field(default_factory=list)
    rl_summary: str                     = ""

class FleetOutput(BaseModel):
    truck_assignments: Dict[int, List[str]]
    vehicle_type_assignments: Dict[int, VehicleType]
    load_utilization: Dict[int, float]
    efficiency_score: float             = 0.0
    underutilized: List[int]            = Field(default_factory=list)
    overflow_risk: bool                 = False
    fallback_swaps: List[str]           = Field(default_factory=list)
    allocation_summary: str             = ""


# ═══════════════════════════════════════════
# COMPLEXITY + FALLBACK
# ═══════════════════════════════════════════

class ComplexityReport(BaseModel):
    complexity_class: ComplexityClass
    num_variables: int
    num_constraints: int
    num_conflicts: int
    constraint_density: float
    recommended_solver: SolverType
    recommended_tier: SolverTier
    rationale: str                      = ""
    warnings: List[str]                 = Field(default_factory=list)


# ═══════════════════════════════════════════
# QUANTUM LAYER
# ═══════════════════════════════════════════

class SolverDecision(BaseModel):
    tier: SolverTier
    solver: SolverType
    num_variables: int
    reason: str
    estimated_solve_time_sec: float

class QUBOProblem(BaseModel):
    qubo_matrix: Dict[Tuple[int, int], float]
    variable_map: Dict[str, int]
    num_variables: int
    problem_type: str                   = "VRP"
    metadata: Dict[str, Any]           = Field(default_factory=dict)

class QuantumSolution(BaseModel):
    optimized_routes: List[List[str]]
    route_costs: Dict[int, float]
    total_optimized_cost: float
    total_baseline_cost: float
    cost_reduction_pct: float
    solver_used: SolverType
    solver_tier: SolverTier
    solver_decision: Optional[SolverDecision] = None
    solution_energy: float
    supply_chain_allocation: Dict[str, str] = Field(default_factory=dict)
    num_annealing_reads: int            = 0
    fallback_applied: bool              = False
    fallback_strategies: List[str]      = Field(default_factory=list)
    optimization_metadata: Dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════
# BUSINESS LAYER
# ═══════════════════════════════════════════

class LogisticsStrategy(BaseModel):
    recommended_routes: List[Dict[str, Any]]
    warehouse_strategy: str
    supply_chain_restructuring: List[str]
    risk_mitigation_actions: List[str]
    fallback_playbook: List[str]        = Field(default_factory=list)
    implementation_timeline: str        = ""

class ROIAnalysis(BaseModel):
    annual_cost_savings_usd: float
    cost_reduction_pct: float
    delivery_time_reduction_pct: float
    efficiency_improvement_pct: float
    payback_period_months: float
    five_year_npv_usd: float
    roi_pct: float
    fallback_cost_impact_usd: float     = 0.0
    key_drivers: List[str]              = Field(default_factory=list)

class Scenario(BaseModel):
    name: str
    description: str
    cost_delta_pct: float
    time_delta_pct: float
    risk_delta_pct: float
    emissions_delta_pct: float
    recommended: bool                   = False

class ScenarioOutput(BaseModel):
    scenarios: List[Scenario]
    best: str
    worst: str
    sensitivity: Dict[str, float]       = Field(default_factory=dict)
    simulation_summary: str             = ""

class ExecutiveReport(BaseModel):
    executive_summary: str
    key_findings: List[str]
    strategic_recommendations: List[str]
    risk_overview: str
    fallback_summary: str               = ""
    optimization_highlights: Dict[str, Any]
    roi_summary: Dict[str, Any]
    scenario_summary: Dict[str, Any]    = Field(default_factory=dict)
    sustainability_summary: Dict[str, Any] = Field(default_factory=dict)
    event_log_summary: Dict[str, Any]   = Field(default_factory=dict)
    charts_data: Dict[str, Any]         = Field(default_factory=dict)
    timestamp: str                      = ""


# ═══════════════════════════════════════════
# FULL ORCHESTRATOR STATE
# ═══════════════════════════════════════════

class QuantixState(BaseModel):
    run_id: str    = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str= Field(default_factory=lambda: datetime.utcnow().isoformat())

    # Raw inputs
    route_input:        Optional[RouteInput]         = None
    supply_chain_input: Optional[SupplyChainInput]   = None
    cost_input:         Optional[CostInput]          = None
    solver_preference:  Optional[str]                = None

    # Layer 0
    planner_output:     Optional[PlannerOutput]      = None
    router_output:      Optional[RouterOutput]       = None

    # Layer 1 (parallel)
    route_risk_output:  Optional[RouteRiskOutput]    = None
    traffic_output:     Optional[TrafficOutput]      = None
    weather_output:     Optional[WeatherOutput]      = None
    sc_risk_output:     Optional[SupplyChainRiskOutput] = None
    cost_output:        Optional[CostOptimizationOutput] = None
    demand_output:      Optional[DemandOutput]       = None
    sustainability_output: Optional[SustainabilityOutput] = None
    trust_output:       Optional[TrustOutput]        = None
    constraint_output:  Optional[ConstraintOutput]   = None

    # Fallback
    fallback_plan:      Optional[Dict[str, Any]]     = None
    real_world_events:  List[str]                    = Field(default_factory=list)

    # Prediction layer
    gnn_output:         Optional[GNNOutput]          = None
    delay_output:       Optional[DelayOutput]        = None

    # RL layer
    rl_output:          Optional[RLRoutingOutput]    = None
    fleet_output:       Optional[FleetOutput]        = None

    # Complexity
    complexity_report:  Optional[ComplexityReport]   = None

    # Quantum
    qubo_problem:       Optional[QUBOProblem]        = None
    quantum_solution:   Optional[QuantumSolution]    = None

    # Business
    logistics_strategy: Optional[LogisticsStrategy]  = None
    roi_analysis:       Optional[ROIAnalysis]        = None
    scenario_output:    Optional[ScenarioOutput]     = None
    executive_report:   Optional[ExecutiveReport]    = None

    # Pipeline metadata
    current_step: str                                = "start"
    status: OptimizationStatus                       = OptimizationStatus.PENDING
    errors: List[str]                               = Field(default_factory=list)
    agent_logs: List[Dict[str, Any]]                = Field(default_factory=list)
    retry_count: int                                = 0
    max_retries: int                                = 3

    class Config:
        arbitrary_types_allowed = True


# ═══════════════════════════════════════════
# PORTFOLIO DOMAIN MODELS
# ═══════════════════════════════════════════

class PortfolioConstraintInput(BaseModel):
    budget: float = Field(default=1.0, gt=0)
    risk_tolerance_lambda: float = Field(default=1.0, ge=0)
    min_weight: float = Field(default=0.0)
    max_weight: float = Field(default=1.0, gt=0)
    allow_shorting: bool = Field(default=False)
    cardinality_limit: Optional[int] = Field(default=None, gt=0)
    risk_threshold: Optional[float] = Field(default=None, gt=0)

class PortfolioMarketAssetInput(BaseModel):
    asset_id: str
    asset_name: Optional[str] = None
    returns: Optional[List[float]] = None
    historical_prices: Optional[List[float]] = None
    expected_return: Optional[float] = None

class PortfolioEarningsInput(BaseModel):
    asset_id: str
    expected_earnings: float
    actual_earnings: float
    earnings_std_dev: Optional[float] = Field(default=1.0, gt=0)

class PortfolioOptimizeRequest(BaseModel):
    market_data: Optional[List[PortfolioMarketAssetInput]] = None
    fundamentals_data: Optional[List[Dict[str, Any]]] = None
    earnings_data: Optional[List[PortfolioEarningsInput]] = None
    constraints: PortfolioConstraintInput = Field(default_factory=PortfolioConstraintInput)
    market_data_path: Optional[str] = None
    fundamentals_path: Optional[str] = None
    earnings_path: Optional[str] = None

class PortfolioOptimizeResponse(BaseModel):
    selected_assets: List[str]
    weights: List[float]
    expected_return: float
    risk: float
    sharpe_ratio: float
    solver_used: str
    performance_metrics: Dict[str, Any]
    business_impact: Dict[str, float]
    human_readable_summary: str


# ═══════════════════════════════════════════
# SUPPLY CHAIN DOMAIN MODELS
# ═══════════════════════════════════════════

class SCOptimizationRequest(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
