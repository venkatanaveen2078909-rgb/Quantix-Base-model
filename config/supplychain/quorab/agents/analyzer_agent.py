from typing import Any, Dict


class ProblemAnalyzerAgent:
    def process(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the problem type and extracts variables and constraints.
        """
        num_nodes = processed_data["num_nodes"]

        problem_type = "TSP" if num_nodes <= 10 else "VRP"

        variables_count = num_nodes * num_nodes

        constraints = [
            "Visit every node exactly once",
            "Depart from depot and return to depot",
            "Capacity constraints (if applicable)",
        ]

        return {
            "problem_type": problem_type,
            "variables_count": variables_count,
            "constraints": constraints,
            "num_nodes": num_nodes,
            "distance_matrix": processed_data["distance_matrix"],
        }
