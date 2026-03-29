from typing import Any, Dict


class ModelSelectorAgent:
    def select_model(self, analysis_result: Dict[str, Any]) -> str:
        """
        Chooses solver based on number of variables.
        < 50 -> Classical
        50-200 -> Hybrid QAOA
        > 200 -> D-Wave Annealer
        """
        var_count = analysis_result["variables_count"]

        if var_count < 50:
            return "classical"
        elif 50 <= var_count <= 200:
            return "qaoa"
        else:
            return "annealer"
