from pathlib import Path

from agents.data_agent import DataAgent


def test_data_agent_process_csv_paths(sample_paths: dict[str, Path]) -> None:
    data_agent = DataAgent()

    processed = data_agent.process(
        market_data_path=str(sample_paths["market"]),
        fundamentals_path=str(sample_paths["fundamentals"]),
        earnings_path=str(sample_paths["earnings"]),
    )

    assert len(processed.asset_ids) > 0
    assert processed.returns_df.shape[0] >= 5
    assert processed.covariance.shape == (len(processed.asset_ids), len(processed.asset_ids))
    assert "ROA" in processed.fundamentals_df.columns
