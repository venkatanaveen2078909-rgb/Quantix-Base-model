import time
from typing import Any, Dict, List
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

class SupplyChainClassicalSolver:
    """
    Classical solver for Supply Chain problems (TSP/VRP).
    Based on OR-Tools.
    """
    def solve(
        self, distance_matrix: List[List[float]], num_vehicles: int = 1, depot_index: int = 0
    ) -> Dict[str, Any]:
        start_time = time.time()
        scaled_matrix = [
            [int(d * 1000) if d != float("inf") else 999999 for d in row]
            for row in distance_matrix
        ]
        manager = pywrapcp.RoutingIndexManager(len(scaled_matrix), num_vehicles, depot_index)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            return scaled_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        
        solution = routing.SolveWithParameters(search_parameters)
        solve_time = time.time() - start_time

        if solution:
            routes = []
            for vehicle_id in range(num_vehicles):
                index = routing.Start(vehicle_id)
                route = []
                while not routing.IsEnd(index):
                    route.append(manager.IndexToNode(index))
                    index = solution.Value(routing.NextVar(index))
                route.append(manager.IndexToNode(index))
                routes.append(route)
            return {
                "status": "SUCCESS",
                "routes": routes,
                "cost": solution.ObjectiveValue() / 1000.0,
                "solver_used": "Classical (OR-Tools)",
                "runtime": solve_time
            }
        return {"status": "FAILED", "error": "No solution found"}
