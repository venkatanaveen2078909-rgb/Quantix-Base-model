import numpy as np

from agents.report_agent import ReportAgent


def test_report_agent_output_schema() -> None:
    agent = ReportAgent()

    asset_ids = ["A", "B", "C"]
    weights = np.array([0.5, 0.5, 0.0])
    mu = np.array([0.1, 0.12, 0.05])
    cov = np.array(
        [
            [0.02, 0.01, 0.0],
            [0.01, 0.03, 0.01],
            [0.0, 0.01, 0.015],
        ]
    )

    result = agent.generate(
        asset_ids=asset_ids,
        weights=weights,
        expected_returns=mu,
        covariance=cov,
        solver_used="classical",
        performance_metrics={"time_taken": 0.02, "iterations": 12, "backend": "test", "status": "success"},
        selection_rationale="small universe",
        constraint_summary=["sum(w)=1", "0<=w<=1"],
    )

    out = result.json_output
    assert "selected_assets" in out
    assert "weights" in out
    assert "performance_metrics" in out
    assert "business_impact" in out
    assert "human_readable_summary" in out
