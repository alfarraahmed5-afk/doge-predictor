"""Root conftest.py — shared pytest fixtures for all test modules.

Loads all seven synthetic fixture Parquet files from
``tests/fixtures/doge_sample_data/`` as session-scoped fixtures so each file
is read from disk only once per test run.

Fixture list:
    doge_trending_bull    : 500-row DOGEUSDT uptrend DataFrame
    doge_trending_bear    : 500-row DOGEUSDT downtrend DataFrame
    doge_ranging          : 500-row DOGEUSDT tight-range DataFrame
    doge_decoupled        : 500-row DOGEUSDT decoupled-from-BTC DataFrame
    doge_mania            : 200-row DOGEUSDT 10x-parabolic DataFrame
    btc_aligned           : 500-row BTCUSDT DataFrame aligned to DOGE timestamps
    funding_rates_sample  : 200-row 8h funding-rate DataFrame

All OHLCV fixtures have columns:
    open_time (int), open (float), high (float), low (float), close (float),
    volume (float), symbol (str), era (str)

The funding fixture has columns:
    timestamp_ms (int), funding_rate (float), symbol (str)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

_FIXTURE_DIR: Path = Path(__file__).parent / "fixtures" / "doge_sample_data"


# ---------------------------------------------------------------------------
# OHLCV fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def doge_trending_bull() -> pd.DataFrame:
    """500-row DOGEUSDT uptrend fixture (EMA20 > EMA50 > EMA200 by end)."""
    return pd.read_parquet(_FIXTURE_DIR / "dogeusdt_1h_trending_bull.parquet")


@pytest.fixture(scope="session")
def doge_trending_bear() -> pd.DataFrame:
    """500-row DOGEUSDT downtrend fixture (EMA20 < EMA50 < EMA200 by end)."""
    return pd.read_parquet(_FIXTURE_DIR / "dogeusdt_1h_trending_bear.parquet")


@pytest.fixture(scope="session")
def doge_ranging() -> pd.DataFrame:
    """500-row DOGEUSDT tight-range fixture (low ATR, low BB width)."""
    return pd.read_parquet(_FIXTURE_DIR / "dogeusdt_1h_ranging.parquet")


@pytest.fixture(scope="session")
def doge_decoupled() -> pd.DataFrame:
    """500-row DOGEUSDT fixture with BTC log-return correlation < 0.20."""
    return pd.read_parquet(_FIXTURE_DIR / "dogeusdt_1h_decoupled.parquet")


@pytest.fixture(scope="session")
def doge_mania() -> pd.DataFrame:
    """200-row DOGEUSDT 10x parabolic-mania fixture."""
    return pd.read_parquet(_FIXTURE_DIR / "dogeusdt_1h_mania.parquet")


@pytest.fixture(scope="session")
def btc_aligned() -> pd.DataFrame:
    """500-row BTCUSDT fixture aligned to the DOGE timestamp grid."""
    return pd.read_parquet(_FIXTURE_DIR / "btcusdt_1h_aligned.parquet")


# ---------------------------------------------------------------------------
# Funding rate fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def funding_rates_sample() -> pd.DataFrame:
    """200-row 8h-cadence DOGEUSDT funding rate fixture."""
    return pd.read_parquet(_FIXTURE_DIR / "funding_rates_sample.parquet")


# ---------------------------------------------------------------------------
# Convenience fixture: all five DOGE regime DataFrames in one dict
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def all_doge_fixtures(
    doge_trending_bull: pd.DataFrame,
    doge_trending_bear: pd.DataFrame,
    doge_ranging: pd.DataFrame,
    doge_decoupled: pd.DataFrame,
    doge_mania: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return all five DOGE regime fixtures keyed by regime name.

    Returns:
        Dict mapping ``'trending_bull'``, ``'trending_bear'``,
        ``'ranging'``, ``'decoupled'``, ``'mania'`` to their DataFrames.
    """
    return {
        "trending_bull": doge_trending_bull,
        "trending_bear": doge_trending_bear,
        "ranging": doge_ranging,
        "decoupled": doge_decoupled,
        "mania": doge_mania,
    }
