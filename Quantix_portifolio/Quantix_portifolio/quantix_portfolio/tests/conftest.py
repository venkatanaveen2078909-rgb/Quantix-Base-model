from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


# Ensure project root is importable even when pytest is run from parent folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.validation import REQUIRED_FUNDAMENTAL_COLUMNS


def _is_market_file(path: Path) -> bool:
    try:
        columns = set(pd.read_csv(path, nrows=1).columns)
    except Exception:
        return False
    return "asset_id" in columns and bool({"return", "returns", "price"} & columns)


def _is_fundamentals_file(path: Path) -> bool:
    try:
        columns = set(pd.read_csv(path, nrows=1).columns)
    except Exception:
        return False
    return set(REQUIRED_FUNDAMENTAL_COLUMNS).issubset(columns)


def _is_earnings_file(path: Path) -> bool:
    required = {"asset_id", "expected_earnings", "actual_earnings"}
    try:
        columns = set(pd.read_csv(path, nrows=1).columns)
    except Exception:
        return False
    return required.issubset(columns)


@pytest.fixture(scope="session")
def sample_paths() -> dict[str, Path]:
    """Locate sample CSVs by schema so tests survive file/folder renames."""
    market_path: Path | None = None
    fundamentals_path: Path | None = None
    earnings_path: Path | None = None

    candidate_dirs = sorted([p for p in PROJECT_ROOT.glob("sample_data*") if p.is_dir()])
    for directory in candidate_dirs:
        for csv_path in directory.glob("*.csv"):
            if market_path is None and _is_market_file(csv_path):
                market_path = csv_path
            if fundamentals_path is None and _is_fundamentals_file(csv_path):
                fundamentals_path = csv_path
            if earnings_path is None and _is_earnings_file(csv_path):
                earnings_path = csv_path

    if market_path is None or fundamentals_path is None or earnings_path is None:
        raise RuntimeError(
            "Could not locate required sample CSV files by schema in sample_data* directories"
        )

    return {
        "market": market_path,
        "fundamentals": fundamentals_path,
        "earnings": earnings_path,
    }
