from pathlib import Path

import pandas as pd

from agents.alpha_agent import AlphaAgent


def test_alpha_agent_piotroski_and_signals(sample_paths: dict[str, Path]) -> None:
    fundamentals = pd.read_csv(sample_paths["fundamentals"])
    earnings = pd.read_csv(sample_paths["earnings"])

    alpha_agent = AlphaAgent()
    asset_ids = fundamentals["asset_id"].astype(str).tolist()

    result = alpha_agent.generate(asset_ids, fundamentals, earnings)

    assert len(result.piotroski_scores) == len(asset_ids)
    assert all(0 <= score <= 9 for score in result.piotroski_scores.values())
    assert set(result.signals.keys()) == set(asset_ids)
    assert all(signal in {-1, 0, 1} for signal in result.signals.values())
    assert isinstance(result.sue_scores, dict)
    assert set(result.sue_scores.keys()).issubset(set(asset_ids))
    assert len(result.investable_asset_ids) >= 1
