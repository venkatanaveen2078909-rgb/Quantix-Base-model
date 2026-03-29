import time
from typing import Any, Dict, List

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


class ClassicalSolver:
    def __init__(self):
        pass

    def solve(
        self, distance_matrix: List[List[float]], num_vehicles: int, depot_index: int
    ) -> Dict[str, Any]:
        """
        Solves the Vehicle Routing Problem using Google OR-Tools.
        """
        start_time = time.time()

        # Scaling distance matrix to integers as OR-Tools requires integers
        scaled_matrix = [
            [int(d * 1000) if d != float("inf") else 999999 for d in row]
            for row in distance_matrix
        ]

        manager = pywrapcp.RoutingIndexManager(
            len(scaled_matrix), num_vehicles, depot_index
        )
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return scaled_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(
            distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        dimension_name = "Distance"
        routing.AddDimension(transit_callback_index, 0,
                             3000000, True, dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        solution = routing.SolveWithParameters(search_parameters)

        solve_time = time.time() - start_time

        if solution:
            return self._extract_solution(
                manager, routing, solution, num_vehicles, solve_time
            )
        else:
            return {"error": "No solution found"}

    def _extract_solution(
        self, manager, routing, solution, num_vehicles, solve_time
    ) -> Dict[str, Any]:
        routes = []
        total_distance = 0
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            route.append(manager.IndexToNode(index))
            routes.append(route)
            total_distance += route_distance

        return {
            "solution": {"status": "SUCCESS"},
            "routes": routes,
            "cost": total_distance / 1000.0,
            "solver_used": "Classical (OR-Tools)",
            "performance_metrics": {"time_taken": solve_time, "iterations": 1},
        }
