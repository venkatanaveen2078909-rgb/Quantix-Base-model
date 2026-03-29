from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from utils.helpers import get_logger
from utils.portfolio_validation import REQUIRED_FUNDAMENTAL_COLUMNS, validate_fundamentals_dataframe


@dataclass
class AlphaResult:
    investable_asset_ids: List[str]
    signals: Dict[str, int]
    piotroski_scores: Dict[str, int]
    sue_scores: Dict[str, float]
    metadata: Dict[str, str]


class AlphaAgent:
    """Generate alpha signals and filter the investable universe."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def generate(
        self,
        asset_ids: List[str],
        fundamentals_df: pd.DataFrame,
        earnings_df: pd.DataFrame,
    ) -> AlphaResult:
        """Compute Piotroski and SUE signals and return filtered assets."""
        fundamentals = fundamentals_df.copy()

        has_all_fundamentals = all(col in fundamentals.columns for col in REQUIRED_FUNDAMENTAL_COLUMNS)
        if not fundamentals.empty and has_all_fundamentals:
            validate_fundamentals_dataframe(fundamentals)
            scores = self._compute_piotroski_scores(fundamentals)
        else:
            # Default neutral score when fundamentals are unavailable.
            scores = {asset_id: 5 for asset_id in asset_ids}

        signals = {}
        for asset_id in asset_ids:
            score = int(scores.get(asset_id, 5))
            if score >= 8:
                signals[asset_id] = 1
            elif score <= 1:
                signals[asset_id] = -1
            else:
                signals[asset_id] = 0

        sue_scores = self._compute_sue_scores(earnings_df)

        investable = [asset_id for asset_id, signal in signals.items() if signal != 0]
        if not investable:
            # Deterministic fallback: pick top half by Piotroski score as long signals.
            ranked = sorted(asset_ids, key=lambda aid: scores.get(aid, 0), reverse=True)
            fallback_count = max(1, len(ranked) // 2)
            investable = ranked[:fallback_count]
            for aid in investable:
                signals[aid] = 1

        self.logger.info("Alpha agent selected %s investable assets out of %s", len(investable), len(asset_ids))
        return AlphaResult(
            investable_asset_ids=investable,
            signals=signals,
            piotroski_scores={asset_id: int(scores.get(asset_id, 5)) for asset_id in asset_ids},
            sue_scores=sue_scores,
            metadata={
                "selection_rule": "Piotroski >= 8 long, <= 1 short; fallback top-half long when no strong signals",
                "sue_rule": "SUE = (actual - expected) / std_dev when earnings data available",
            },
        )

    def _compute_piotroski_scores(self, fundamentals_df: pd.DataFrame) -> Dict[str, int]:
        df = fundamentals_df.copy().fillna(0.0)
        signals_df = pd.DataFrame(
            {
                "asset_id": df["asset_id"].astype(str),
                "s1": (df["ROA"] > 0).astype(int),
                "s2": (df["CFO"] > 0).astype(int),
                "s3": (df["Delta_ROA"] > 0).astype(int),
                "s4": (df["CFO"] > df["ROA"]).astype(int),
                "s5": (df["Delta_Leverage"] < 0).astype(int),
                "s6": (df["Delta_Liquidity"] > 0).astype(int),
                "s7": (df["Equity_Issued"] == 0).astype(int),
                "s8": (df["Delta_Margin"] > 0).astype(int),
                "s9": (df["Delta_Turnover"] > 0).astype(int),
            }
        )
        signals_df["score"] = signals_df[[f"s{i}" for i in range(1, 10)]].sum(axis=1)
        return dict(zip(signals_df["asset_id"], signals_df["score"].astype(int)))

    def _compute_sue_scores(self, earnings_df: pd.DataFrame) -> Dict[str, float]:
        if earnings_df.empty:
            return {}
        required = {"asset_id", "expected_earnings", "actual_earnings"}
        if not required.issubset(set(earnings_df.columns)):
            return {}

        df = earnings_df.copy()
        if "earnings_std_dev" not in df.columns:
            df["earnings_std_dev"] = 1.0
        df["earnings_std_dev"] = pd.to_numeric(df["earnings_std_dev"], errors="coerce").fillna(1.0)
        df.loc[df["earnings_std_dev"] <= 0, "earnings_std_dev"] = 1.0

        sue = (df["actual_earnings"] - df["expected_earnings"]) / df["earnings_std_dev"]
        return dict(zip(df["asset_id"].astype(str), sue.astype(float)))
