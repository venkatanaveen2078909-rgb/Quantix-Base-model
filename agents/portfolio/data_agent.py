from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from utils.helpers import get_logger
from utils.portfolio_validation import validate_market_dataframe


@dataclass
class ProcessedData:
    asset_ids: List[str]
    asset_names: List[str]
    returns_df: pd.DataFrame
    expected_returns: pd.Series
    covariance: pd.DataFrame
    fundamentals_df: pd.DataFrame
    earnings_df: pd.DataFrame


class DataAgent:
    """Load, clean, validate, and structure market/fundamental data."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def process(
        self,
        market_data: Optional[Sequence[Dict[str, Any]]] = None,
        fundamentals_data: Optional[Sequence[Dict[str, Any]]] = None,
        earnings_data: Optional[Sequence[Dict[str, Any]]] = None,
        market_data_path: Optional[str] = None,
        fundamentals_path: Optional[str] = None,
        earnings_path: Optional[str] = None,
    ) -> ProcessedData:
        """Return validated and ready-to-optimize structured data."""
        market_df = self._load_market_data(market_data, market_data_path)
        fundamentals_df = self._load_optional_data(fundamentals_data, fundamentals_path)
        earnings_df = self._load_optional_data(earnings_data, earnings_path)

        validate_market_dataframe(market_df)
        clean_market = self._clean_market_data(market_df)

        returns_df = self._build_returns_matrix(clean_market)
        expected_returns = self._compute_expected_returns(clean_market, returns_df)
        covariance = self._compute_covariance(clean_market, returns_df)

        asset_ids = list(returns_df.columns)
        asset_names = self._extract_asset_names(clean_market, asset_ids)

        fundamentals_df = self._normalize_optional_dataframe(fundamentals_df, asset_ids)
        earnings_df = self._normalize_optional_dataframe(earnings_df, asset_ids)

        self.logger.info("Processed %s assets and %s return periods", len(asset_ids), len(returns_df))
        return ProcessedData(
            asset_ids=asset_ids,
            asset_names=asset_names,
            returns_df=returns_df,
            expected_returns=expected_returns,
            covariance=covariance,
            fundamentals_df=fundamentals_df,
            earnings_df=earnings_df,
        )

    def _load_market_data(
        self,
        market_data: Optional[Sequence[Dict[str, Any]]],
        market_data_path: Optional[str],
    ) -> pd.DataFrame:
        if market_data_path:
            return self._read_path(market_data_path)

        if not market_data:
            raise ValueError("market_data or market_data_path must be provided")

        records: List[Dict[str, Any]] = []
        for item in market_data:
            asset_id = str(item["asset_id"])
            asset_name = str(item.get("asset_name", asset_id))
            expected_return = item.get("expected_return")
            returns = item.get("returns")
            prices = item.get("historical_prices")

            if returns and isinstance(returns, list):
                for idx, value in enumerate(returns):
                    records.append(
                        {
                            "asset_id": asset_id,
                            "asset_name": asset_name,
                            "period": idx,
                            "return": float(value),
                            "expected_return": expected_return,
                        }
                    )
            elif prices and isinstance(prices, list):
                for idx in range(1, len(prices)):
                    prev_p = float(prices[idx - 1])
                    curr_p = float(prices[idx])
                    ret = 0.0 if prev_p == 0 else (curr_p / prev_p) - 1.0
                    records.append(
                        {
                            "asset_id": asset_id,
                            "asset_name": asset_name,
                            "period": idx,
                            "return": ret,
                            "expected_return": expected_return,
                        }
                    )
            else:
                row: Dict[str, Any] = {
    "asset_id": asset_id,
    "asset_name": asset_name,
}
                if expected_return is not None:
                    row["expected_return"] = float(expected_return)
                records.append(row)

        return pd.DataFrame(records)

    def _load_optional_data(
        self,
        data: Optional[Sequence[Dict[str, Any]]],
        path: Optional[str],
    ) -> pd.DataFrame:
        if path:
            return self._read_path(path)
        if data:
            return pd.DataFrame(list(data))
        return pd.DataFrame()

    def _read_path(self, path_value: str) -> pd.DataFrame:
        path = Path(path_value)
        if not path.exists():
            raise ValueError(f"data path does not exist: {path_value}")
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".json":
            return pd.read_json(path)
        raise ValueError(f"unsupported file format: {path.suffix}")

    def _clean_market_data(self, market_df: pd.DataFrame) -> pd.DataFrame:
        df = market_df.copy()
        if "asset_name" not in df.columns:
            df["asset_name"] = df["asset_id"].astype(str)

        if "period" not in df.columns:
            df["period"] = df.groupby("asset_id").cumcount()

        if "return" not in df.columns:
            if "returns" in df.columns:
                df["return"] = pd.to_numeric(df["returns"], errors="coerce")
            elif "price" in df.columns:
                df["price"] = pd.to_numeric(df["price"], errors="coerce")
                df["return"] = (
                    df.sort_values(["asset_id", "period"])
                    .groupby("asset_id")["price"]
                    .pct_change()
                    .fillna(0.0)
                )

        df["return"] = pd.to_numeric(df["return"], errors="coerce")
        df["return"] = df["return"].fillna(df["return"].median() if not df["return"].dropna().empty else 0.0)

        if "expected_return" in df.columns:
            df["expected_return"] = pd.to_numeric(df["expected_return"], errors="coerce")

        return df

    def _build_returns_matrix(self, market_df: pd.DataFrame) -> pd.DataFrame:
        returns_df = market_df.pivot_table(
            index="period",
            columns="asset_id",
            values="return",
            aggfunc="mean",
        ).sort_index()
        returns_df = returns_df.replace([np.inf, -np.inf], np.nan)
        returns_df = returns_df.interpolate(limit_direction="both").fillna(0.0)
        return returns_df

    def _compute_expected_returns(self, market_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.Series:
        if "expected_return" in market_df.columns and market_df["expected_return"].notna().any():
            exp_ret = market_df.groupby("asset_id")["expected_return"].mean()
            exp_ret = exp_ret.reindex(returns_df.columns)
            return exp_ret.fillna(returns_df.mean())
        return returns_df.mean()

    def _compute_covariance(self, market_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
        if "covariance_matrix" in market_df.columns:
            self.logger.warning("covariance_matrix column found but external covariance parsing is skipped")
        cov = returns_df.cov()
        if cov.isnull().values.any():
            cov = cov.fillna(0.0)
        epsilon = 1e-8
        cov = cov + np.eye(cov.shape[0]) * epsilon
        return cov

    def _extract_asset_names(self, market_df: pd.DataFrame, asset_ids: List[str]) -> List[str]:
        names = market_df.groupby("asset_id")["asset_name"].first()
        return [str(names.get(asset_id, asset_id)) for asset_id in asset_ids]

    def _normalize_optional_dataframe(self, df: pd.DataFrame, asset_ids: List[str]) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame({"asset_id": asset_ids})
        result = df.copy()
        if "asset_id" not in result.columns:
            raise ValueError("optional data must contain 'asset_id'")
        result["asset_id"] = result["asset_id"].astype(str)
        result = result.drop_duplicates(subset=["asset_id"], keep="last")
        merged = pd.DataFrame({"asset_id": asset_ids}).merge(result, on="asset_id", how="left")
        return merged
