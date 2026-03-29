"""
Quantix — Mumbai Logistics Data Ingestion
Loads real-world Mumbai logistics data from the mumbai_logistics_v2 dataset
and maps it to Quantix RouteInput, SupplyChainInput, and CostInput schemas.
"""
from __future__ import annotations
import json
import os
import random
from typing import Dict, Any, List, Tuple

BASE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "mumbai_logistics_v2", "mumbai_logistics_v2"
)

CORE_DIR = os.path.join(BASE_DATA_DIR, "1. CORE LOGISTICS ENTITIES")
NETWORK_DIR = os.path.join(BASE_DATA_DIR, "2. NETWORK & ROUTING DATA")
REALTIME_DIR = os.path.join(BASE_DATA_DIR, "4. DYNAMIC & REAL-TIME DATA")


def _load(folder: str, filename: str) -> Any:
    path = os.path.join(folder, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _weather_to_quantix(condition: str) -> str:
    """Map Mumbai weather condition strings to Quantix WeatherCondition enum values."""
    mapping = {
        "clear": "CLEAR",
        "sunny": "CLEAR",
        "rain": "RAIN",
        "heavy rain": "HEAVY_RAIN",
        "storm": "STORM",
        "snow": "SNOW",
        "fog": "FOG",
        "haze": "FOG",
        "mist": "FOG",
        "wind": "HIGH_WIND",
        "high wind": "HIGH_WIND",
    }
    return mapping.get(condition.lower().strip(), "CLEAR")


def load_mumbai_data(max_nodes: int = 50) -> Dict[str, Any]:
    """
    Load the Mumbai logistics dataset and return Quantix-compatible inputs.

    Args:
        max_nodes: Limit on delivery nodes to include (default 50 for manageable runs).
                   Set to 500 for the full dataset (requires significant memory/compute).

    Returns:
        dict with keys: route_input, supply_chain_input, cost_input
    """
    # ── 1. Depot ──────────────────────────────────────────────────────────────
    depot = _load(CORE_DIR, "A_Depot_Warehouse_Information.json")
    depot_id = depot["depot_id"]  # "DEPOT_001"

    # ── 2. Delivery Nodes (B) ─────────────────────────────────────────────────
    locations: List[Dict] = _load(CORE_DIR, "B_Delivery_Locations_Nodes.json")
    locations = locations[:max_nodes]  # Limit for performance

    node_ids = [depot_id] + [loc["location_id"] for loc in locations]

    # Build traffic scores from access restriction data + delivery type
    traffic_scores: Dict[str, float] = {depot_id: 0.1}
    delivery_deadlines: Dict[str, float] = {}

    DELIVERY_TYPE_SCORES = {
        "express": 0.8,
        "priority": 0.7,
        "same_day": 0.6,
        "standard": 0.3,
    }
    DELIVERY_DEADLINES = {
        "priority": 12.0,
        "express": 13.0,
        "same_day": 20.0,
    }

    for loc in locations:
        loc_id = loc["location_id"]
        d_type = loc.get("delivery_type", "standard")
        peak = loc["access_restrictions"].get("peak_hour_restriction", False)
        narrow = loc["access_restrictions"].get("narrow_streets", False)

        base = DELIVERY_TYPE_SCORES.get(d_type, 0.3)
        score = round(min(base + (0.15 if peak else 0) + (0.1 if narrow else 0), 1.0), 2)
        traffic_scores[loc_id] = score

        if d_type in DELIVERY_DEADLINES:
            delivery_deadlines[loc_id] = DELIVERY_DEADLINES[d_type]

    # ── 3. Distance Matrix (E) → edges ────────────────────────────────────────
    # Distance matrix is 501x501 (DEPOT + 500 LOCs); use first max_nodes+1 rows
    dist_matrix_data = _load(NETWORK_DIR, "E_Distance_Matrix.json")
    matrix: List[List[float]] = dist_matrix_data.get("matrix", [])
    matrix_nodes: List[str] = dist_matrix_data.get("node_ids", node_ids)

    # Build edges from distance matrix (values already in km)
    edges: List[Tuple[str, str, float]] = []
    route_distances: Dict[str, float] = {}

    num = min(len(node_ids), len(matrix))
    for i in range(num):
        for j in range(num):
            if i != j:
                src = node_ids[i]
                dst = node_ids[j]
                try:
                    dist_km = float(matrix[i][j])
                except (IndexError, TypeError, ValueError):
                    dist_km = 0.0
                if dist_km > 0:
                    edges.append((src, dst, round(dist_km, 2)))
                    route_distances[f"{src}→{dst}"] = round(dist_km, 2)

    # ── Fallback if matrix not in expected format ──────────────────────────────
    if not edges:
        for i, src in enumerate(node_ids):
            for j, dst in enumerate(node_ids):
                if i != j:
                    dist_km = round(random.uniform(2, 40), 2)
                    edges.append((src, dst, dist_km))
                    route_distances[f"{src}→{dst}"] = dist_km

    # ── 4. Real-Time Traffic + Weather (J) ────────────────────────────────────
    realtime = _load(REALTIME_DIR, "J_Real_Time_Traffic_Weather.json")
    weather_raw = realtime.get("weather_conditions", {}).get("condition", "clear")
    current_weather = _weather_to_quantix(weather_raw)

    # Apply congestion_level from traffic data to override traffic_scores where available
    traffic_data = realtime.get("traffic_data", [])
    for i, seg in enumerate(traffic_data):
        if i < len(locations):
            loc_id = locations[i]["location_id"]
            traffic_scores[loc_id] = round(
                max(traffic_scores.get(loc_id, 0.3), seg["congestion_level"]), 2
            )

    # ── 5. Vehicles (D) → cost + fleet info ───────────────────────────────────
    vehicles: List[Dict] = _load(CORE_DIR, "D_Vehicle_Fleet_Information.json")
    available_vehicles = [v for v in vehicles if v["current_status"] == "available"]
    if not available_vehicles:
        available_vehicles = vehicles[:10]

    avg_cost_per_km = round(
        sum(v["operating_costs"]["cost_per_km"] for v in available_vehicles) / len(available_vehicles), 2
    )
    avg_cost_per_hour = round(
        sum(v["operating_costs"]["cost_per_hour"] for v in available_vehicles) / len(available_vehicles), 2
    )
    avg_fixed_cost = round(
        sum(v["operating_costs"]["fixed_cost_per_day"] for v in available_vehicles) / len(available_vehicles), 2
    )
    avg_capacity = round(
        sum(v["capacity"]["max_weight"] for v in available_vehicles) / len(available_vehicles), 2
    )
    num_trucks = min(len(available_vehicles), 20)

    # Delivery loads from location service time (proxy for package weight/effort)
    delivery_loads: Dict[str, float] = {}
    for loc in locations:
        service_min = loc.get("service_time", 15)
        delivery_loads[loc["location_id"]] = round(service_min * 2.5, 1)  # rough kg estimate

    # ── 6. Supply Chain (from supplier = vehicle zones → reliability proxy) ────
    zones = list(set(v.get("assigned_zone", "Mumbai") for v in available_vehicles))
    supplier_reliability = {
        zone: round(
            sum(1 for v in available_vehicles if v.get("assigned_zone") == zone and v["current_status"] == "available") /
            max(sum(1 for v in available_vehicles if v.get("assigned_zone") == zone), 1), 2
        )
        for zone in zones
    }
    warehouse_stock = {zone: round(random.uniform(200, 1000), 1) for zone in zones}
    demand_forecasts = {zone: round(random.uniform(100, 600), 1) for zone in zones}

    # ── 7. Build Quantix input dicts ──────────────────────────────────────────
    from models.schemas import (
        RouteInput, SupplyChainInput, CostInput, WeatherCondition
    )

    route_input = RouteInput(
        nodes=node_ids,
        edges=edges,
        depot=depot_id,
        traffic_scores=traffic_scores,
        weather_scores={},  # Mumbai dataset doesn't have per-node weather
        current_weather=WeatherCondition(current_weather),
        delivery_deadlines=delivery_deadlines,
    )

    supply_chain_input = SupplyChainInput(
        suppliers=zones,
        supplier_reliability=supplier_reliability,
        warehouse_stock=warehouse_stock,
        demand_forecasts=demand_forecasts,
    )

    cost_input = CostInput(
        fuel_cost_per_km=avg_cost_per_km,
        driver_cost_per_hour=avg_cost_per_hour,
        warehouse_cost_per_day=avg_fixed_cost,
        num_trucks=num_trucks,
        truck_capacity=avg_capacity,
        delivery_loads=delivery_loads,
        route_distances=route_distances,
    )

    print(f"[DataIngestion] Loaded Mumbai dataset:")
    print(f"  Depot: {depot_id} ({depot['name']})")
    print(f"  Nodes: {len(node_ids)} ({max_nodes} delivery + 1 depot)")
    print(f"  Edges: {len(edges)}")
    print(f"  Vehicles: {num_trucks} available (of {len(vehicles)} total)")
    print(f"  Weather: {current_weather} (raw: '{weather_raw}')")
    print(f"  Supplier zones: {zones}")

    return {
        "route_input": route_input,
        "supply_chain_input": supply_chain_input,
        "cost_input": cost_input,
    }


def load_dataset(file_path: str) -> Dict[str, Any]:
    """
    Generic JSON dataset loader (used for sample_dataset.json files).
    For Mumbai real-time data use load_mumbai_data() instead.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    from models.schemas import RouteInput, SupplyChainInput, CostInput

    route_in = RouteInput(**data["route_input"])
    sc_in = SupplyChainInput(**data["supply_chain_input"])
    cost_data = data["cost_input"]
    cost_data["route_distances"] = {
        f"{s}→{d}": w for s, d, w in route_in.edges
    }
    cost_in = CostInput(**cost_data)

    return {
        "route_input": route_in,
        "supply_chain_input": sc_in,
        "cost_input": cost_in,
    }
