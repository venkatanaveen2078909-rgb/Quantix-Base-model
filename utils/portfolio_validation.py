from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator


REQUIRED_FUNDAMENTAL_COLUMNS = [
    "asset_id",
    "ROA",
    "CFO",
    "Delta_ROA",
    "Delta_Leverage",
    "Delta_Liquidity",
    "Equity_Issued",
    "Delta_Margin",
    "Delta_Turnover",
]


class ConstraintInput(BaseModel):
    budget: float = Field(default=1.0, gt=0)
    risk_tolerance_lambda: float = Field(default=1.0, ge=0)
    min_weight: float = Field(default=0.0)
    max_weight: float = Field(default=1.0, gt=0)
    allow_shorting: bool = Field(default=False)
    cardinality_limit: Optional[int] = Field(default=None, gt=0)
    risk_threshold: Optional[float] = Field(default=None, gt=0)

    @field_validator("max_weight")
    @classmethod
    def validate_bounds(cls, value: float) -> float:
        if value > 1.0:
            raise ValueError("max_weight must be <= 1.0")
        return value


class MarketAssetInput(BaseModel):
    asset_id: str
    asset_name: Optional[str] = None
    returns: Optional[List[float]] = None
    historical_prices: Optional[List[float]] = None
    expected_return: Optional[float] = None


class EarningsInput(BaseModel):
    asset_id: str
    expected_earnings: float
    actual_earnings: float
    earnings_std_dev: Optional[float] = Field(default=1.0, gt=0)


class OptimizeRequest(BaseModel):
    market_data: Optional[List[MarketAssetInput]] = None
    fundamentals_data: Optional[List[Dict[str, Any]]] = None
    earnings_data: Optional[List[EarningsInput]] = None
    constraints: ConstraintInput = Field(default_factory=ConstraintInput)
    market_data_path: Optional[str] = None
    fundamentals_path: Optional[str] = None
    earnings_path: Optional[str] = None


class OptimizeResponse(BaseModel):
    selected_assets: List[str]
    weights: List[float]
    expected_return: float
    risk: float
    sharpe_ratio: float
    solver_used: str
    performance_metrics: Dict[str, Any]
    business_impact: Dict[str, float]
    human_readable_summary: str


def validate_market_dataframe(df: pd.DataFrame) -> None:
    """Validate market data for minimum fields needed by the pipeline."""
    if "asset_id" not in df.columns:
        raise ValueError("market data must contain 'asset_id'")
    if "return" not in df.columns and "returns" not in df.columns and "price" not in df.columns:
        raise ValueError("market data must include one of: return, returns, price")


def validate_fundamentals_dataframe(df: pd.DataFrame) -> None:
    """Validate fundamental fields needed for Piotroski F-Score."""
    missing = [col for col in REQUIRED_FUNDAMENTAL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"fundamentals data missing required columns: {missing}")
