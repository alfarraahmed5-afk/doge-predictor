"""Unit tests for src/features/htf_features.py and src/features/orderbook_features.py.

MANDATORY: HTF lookahead boundary tests.
    The core invariant: a 4h (or 1d) candle is visible at 1h time T
    only after that candle has fully closed (lookup_key = open_time + interval_ms).

MANDATORY SPECIFIC TEST:
    At 2022-01-01 15:00 UTC (T0+15h):
        htf_4h_rsi == RSI of 4h bar that closed at 12:00 (NOT the bar closing at 16:00)
    Verified by asserting rsi[15h] == rsi[12h] and rsi[16h] != rsi[15h].

Test groups:
    TestHTFMandatoryBoundary  — MANDATORY 2022-01-01 15:00 boundary test
    TestHTFLookaheadGuard4h   — structural constant-within-bar property
    TestHTFLookaheadGuard1d   — 1d bar boundary correctness
    TestHTFAtDistance          — ATH distance properties (fixed ATH)
    TestHTFValues              — Indicator value sanity
    TestHTFInputValidation     — Missing-column error paths
    TestHTFOrderBook           — orderbook_features.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.htf_features import (
    HTF_FEATURE_NAMES,
    _4H_MS,
    _1D_MS,
    compute_htf_features,
)
from src.features.orderbook_features import compute_orderbook_features

# ---------------------------------------------------------------------------
# Constants and shared helpers
# ---------------------------------------------------------------------------

_1H_MS: int = 3_600_000
_T0: int = 1_640_995_200_000  # 2022-01-01 00:00:00 UTC


def _make_1h(n: int, start_ms: int = _T0, seed: int = 1) -> pd.DataFrame:
    """Generate a synthetic 1h OHLCV DataFrame with *n* rows."""
    rng = np.random.default_rng(seed)
    times = np.array([start_ms + i * _1H_MS for i in range(n)], dtype=np.int64)
    close = 0.10 * np.cumprod(1 + rng.normal(0.0002, 0.005, n))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.uniform(1e6, 5e6, n)
    return pd.DataFrame(
        {
            "open_time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": "DOGEUSDT",
            "era": "training",
        }
    )


def _make_4h(n: int, start_ms: int = _T0, seed: int = 2) -> pd.DataFrame:
    """Generate a synthetic 4h OHLCV DataFrame with *n* rows."""
    rng = np.random.default_rng(seed)
    times = np.array([start_ms + i * _4H_MS for i in range(n)], dtype=np.int64)
    close = 0.10 * np.cumprod(1 + rng.normal(0.0008, 0.010, n))
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    volume = rng.uniform(5e6, 2e7, n)
    return pd.DataFrame(
        {
            "open_time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _make_1d(n: int, start_ms: int = _T0, seed: int = 3) -> pd.DataFrame:
    """Generate a synthetic 1d OHLCV DataFrame with *n* rows."""
    rng = np.random.default_rng(seed)
    times = np.array([start_ms + i * _1D_MS for i in range(n)], dtype=np.int64)
    close = 0.10 * np.cumprod(1 + rng.normal(0.002, 0.020, n))
    high = close * (1 + np.abs(rng.normal(0, 0.015, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.010, n))
    return pd.DataFrame(
        {
            "open_time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


# ---------------------------------------------------------------------------
# TestHTFMandatoryBoundary — MANDATORY
# ---------------------------------------------------------------------------


class TestHTFMandatoryBoundary:
    """MANDATORY: Verify the specific 2022-01-01 15:00 UTC boundary.

    At open_time = 2022-01-01 15:00 UTC (T0 + 15h):
        - 4h bar [12:00, 16:00) is NOT yet closed  → must NOT be used.
        - 4h bar [08:00, 12:00) IS closed           → lookup_key = 12:00 <= 15:00.
        - Therefore htf_4h_rsi at 15:00 == RSI from the [08:00, 12:00) bar.

    Equivalent assertions:
        rsi[15h] == rsi[12h]   (both rows use the same [08:00–12:00] closed bar)
        rsi[16h] != rsi[15h]   ([12:00–16:00] bar closes at 16:00; RSI updates)
    """

    # Shared fixtures for this class (enough prior 4h history for RSI warmup)
    _PRIOR_BARS = 25

    def _build_result(self) -> pd.DataFrame:
        """Build the HTF feature DataFrame used across all boundary tests."""
        offset = self._PRIOR_BARS * _4H_MS
        df_4h = _make_4h(self._PRIOR_BARS + 10, start_ms=_T0 - offset, seed=99)
        df_1h = _make_1h(20, start_ms=_T0, seed=77)
        df_1d = _make_1d(5, start_ms=_T0 - 5 * _1D_MS, seed=55)
        return compute_htf_features(df_1h, df_4h, df_1d)

    def test_rsi_at_1500_equals_rsi_at_1200(self):
        """MANDATORY: htf_4h_rsi at 2022-01-01 15:00 == RSI at 12:00.

        Both 12:00 and 15:00 fall inside the half-open interval [12:00, 16:00)
        with respect to visible closed bars, so both see the [08:00–12:00] bar.
        """
        result = self._build_result()

        # Row 12 → open_time = T0 + 12h = 2022-01-01 12:00 UTC
        # Row 15 → open_time = T0 + 15h = 2022-01-01 15:00 UTC
        rsi_at_1200 = result["htf_4h_rsi"].iloc[12]
        rsi_at_1500 = result["htf_4h_rsi"].iloc[15]

        assert not np.isnan(rsi_at_1200), (
            "RSI at 12:00 must be non-NaN with 25 prior 4h bars (RSI warmup = 14)"
        )
        assert rsi_at_1200 == rsi_at_1500, (
            f"MANDATORY BOUNDARY TEST FAILED: "
            f"htf_4h_rsi at 15:00 ({rsi_at_1500:.6f}) != "
            f"htf_4h_rsi at 12:00 ({rsi_at_1200:.6f}). "
            f"At 15:00 the [12:00–16:00] bar is NOT yet closed; "
            f"both rows must reflect the [08:00–12:00] bar."
        )

    def test_rsi_at_1500_not_equal_to_rsi_at_1600(self):
        """MANDATORY: htf_4h_rsi at 15:00 != RSI computed on the bar closing at 16:00.

        The bar [12:00–16:00] first becomes visible at 16:00 (lookup_key=16:00).
        Its RSI is first observable at row 16, and must differ from row 15.
        """
        result = self._build_result()

        rsi_at_1500 = result["htf_4h_rsi"].iloc[15]
        rsi_at_1600 = result["htf_4h_rsi"].iloc[16]

        assert rsi_at_1500 != rsi_at_1600, (
            f"MANDATORY BOUNDARY TEST FAILED: "
            f"htf_4h_rsi at 16:00 ({rsi_at_1600:.6f}) == "
            f"htf_4h_rsi at 15:00 ({rsi_at_1500:.6f}). "
            f"At 16:00 the [12:00–16:00] bar just closed; RSI must update."
        )

    def test_rsi_constant_from_1200_to_1500(self):
        """MANDATORY: rows 12:00, 13:00, 14:00, 15:00 all have the same RSI.

        All four 1h candles fall within the half-open interval [12:00, 16:00)
        for visible closed bars and must reflect the [08:00–12:00] bar.
        """
        result = self._build_result()

        rsi_vals = [result["htf_4h_rsi"].iloc[h] for h in range(12, 16)]

        assert all(v == rsi_vals[0] for v in rsi_vals), (
            f"MANDATORY BOUNDARY TEST FAILED: "
            f"htf_4h_rsi must be constant for rows 12–15 (same closed 4h bar). "
            f"Got: {rsi_vals}"
        )


# ---------------------------------------------------------------------------
# TestHTFLookaheadGuard4h
# ---------------------------------------------------------------------------


class TestHTFLookaheadGuard4h:
    """Structural tests for the 4h lookahead guard."""

    def test_htf_4h_constant_within_bar(self):
        """htf_4h_rsi is constant for all 4 consecutive 1h rows inside one bar."""
        n_4h = 80
        df_1h = _make_1h(n_4h * 4)
        df_4h = _make_4h(n_4h)
        df_1d = _make_1d(n_4h // 6 + 5)

        result = compute_htf_features(df_1h, df_4h, df_1d)
        rsi = result["htf_4h_rsi"].to_numpy()

        for i in range(8, n_4h * 4 - 4, 4):
            group = rsi[i : i + 4]
            if not np.any(np.isnan(group)):
                assert np.all(group == group[0]), (
                    f"htf_4h_rsi changed within a 4h bar at 1h rows [{i},{i+4}): {group}"
                )

    def test_htf_4h_nan_before_first_close(self):
        """1h rows 0–3 must have NaN htf_4h_rsi when 4h data starts at T0."""
        df_1h = _make_1h(20, start_ms=_T0)
        df_4h = _make_4h(10, start_ms=_T0)
        df_1d = _make_1d(5, start_ms=_T0)

        result = compute_htf_features(df_1h, df_4h, df_1d)

        for i in range(4):
            assert np.isnan(result["htf_4h_rsi"].iloc[i]), (
                f"Row {i} (T0+{i}h) must be NaN — first 4h bar not yet closed. "
                f"Got {result['htf_4h_rsi'].iloc[i]}"
            )

    def test_htf_4h_non_nan_with_prior_history(self):
        """With prior 4h history, row 0 is non-NaN (prior bar closes before T0)."""
        prior_bars = 25
        df_4h = _make_4h(prior_bars + 5, start_ms=_T0 - prior_bars * _4H_MS)
        df_1h = _make_1h(20, start_ms=_T0)
        df_1d = _make_1d(5, start_ms=_T0 - 5 * _1D_MS)

        result = compute_htf_features(df_1h, df_4h, df_1d)

        # The 4h bar at T0 - 4h has lookup_key = T0 → visible at row 0
        rsi_row0 = result["htf_4h_rsi"].iloc[0]
        assert not np.isnan(rsi_row0), (
            f"Row 0 should be non-NaN with prior 4h history, got {rsi_row0}"
        )

    def test_htf_4h_value_varies_across_bars(self):
        """htf_4h_rsi is not constant across all bars (guard actually updates)."""
        df_1h = _make_1h(320, seed=10)
        df_4h = _make_4h(80, seed=11)
        df_1d = _make_1d(15, seed=12)

        result = compute_htf_features(df_1h, df_4h, df_1d)
        rsi = result["htf_4h_rsi"].dropna()
        assert rsi.std() > 0.5, "htf_4h_rsi has no variation — guard may be broken"

    def test_htf_4h_column_names_present(self):
        """All expected 4h feature columns must be present."""
        result = compute_htf_features(_make_1h(40), _make_4h(20), _make_1d(5))
        for col in ("htf_4h_rsi", "htf_4h_trend", "htf_4h_bb_pctb"):
            assert col in result.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# TestHTFLookaheadGuard1d
# ---------------------------------------------------------------------------


class TestHTFLookaheadGuard1d:
    """Verify 1d features do not use unclosed daily bars."""

    def test_htf_1d_constant_within_day(self):
        """htf_1d_trend is constant for all 24 consecutive 1h rows in a day."""
        n_1d = 30
        n_1h = n_1d * 24
        df_1h = _make_1h(n_1h, seed=20)
        df_4h = _make_4h(n_1h // 4 + 5, seed=21)
        df_1d = _make_1d(n_1d, seed=22)

        result = compute_htf_features(df_1h, df_4h, df_1d)
        trend = result["htf_1d_trend"].to_numpy()

        for i in range(24, n_1h - 24, 24):
            group = trend[i : i + 24]
            if not np.any(np.isnan(group)):
                assert np.all(group == group[0]), (
                    f"htf_1d_trend changed within a 1d bar at 1h rows [{i},{i+24})"
                )

    def test_htf_1d_columns_present(self):
        result = compute_htf_features(_make_1h(50), _make_4h(15), _make_1d(5))
        assert "htf_1d_trend" in result.columns
        assert "htf_1d_return" in result.columns


# ---------------------------------------------------------------------------
# TestHTFAtDistance
# ---------------------------------------------------------------------------


class TestHTFAtDistance:
    """ATH distance feature: log(doge_ath_price / close) — always >= 0."""

    def test_ath_distance_always_nonnegative(self):
        """ath_distance = log(ATH/close) is always >= 0 for close <= ATH."""
        result = compute_htf_features(_make_1h(200), _make_4h(60), _make_1d(10))
        dist = result["ath_distance"].to_numpy()
        assert np.all(dist >= 0.0), f"ath_distance has negative values: min={dist.min():.6f}"

    def test_ath_distance_zero_when_close_equals_ath(self):
        """log(ATH / ATH) = 0."""
        from src.config import doge_settings

        ath = doge_settings.doge_ath_price
        n = 10
        times = np.array([_T0 + i * _1H_MS for i in range(n)], dtype=np.int64)
        df_1h = pd.DataFrame(
            {
                "open_time": times,
                "open": np.full(n, ath),
                "high": np.full(n, ath * 1.001),
                "low": np.full(n, ath * 0.999),
                "close": np.full(n, ath),
                "volume": np.ones(n) * 1e6,
                "symbol": "DOGEUSDT",
                "era": "training",
            }
        )
        result = compute_htf_features(df_1h, _make_4h(5), _make_1d(3))
        assert np.allclose(result["ath_distance"].to_numpy(), 0.0, atol=1e-9), (
            "ath_distance must be ~0 when close == ATH"
        )

    def test_ath_distance_increases_as_close_falls(self):
        """Lower close prices produce larger ath_distance."""
        n = 5
        times = np.array([_T0 + i * _1H_MS for i in range(n)], dtype=np.int64)
        closes = np.array([0.70, 0.50, 0.30, 0.20, 0.10])
        df_1h = pd.DataFrame(
            {
                "open_time": times,
                "open": closes,
                "high": closes * 1.01,
                "low": closes * 0.99,
                "close": closes,
                "volume": np.ones(n) * 1e6,
                "symbol": "DOGEUSDT",
                "era": "training",
            }
        )
        result = compute_htf_features(df_1h, _make_4h(3), _make_1d(2))
        dist = result["ath_distance"].to_numpy()
        assert np.all(np.diff(dist) > 0), (
            f"ath_distance must increase as close decreases. Got: {dist}"
        )

    def test_ath_distance_matches_formula(self):
        """ath_distance must exactly equal log(cfg.doge_ath_price / close)."""
        from src.config import doge_settings

        df_1h = _make_1h(20)
        result = compute_htf_features(df_1h, _make_4h(10), _make_1d(5))
        expected = np.log(doge_settings.doge_ath_price / df_1h["close"].to_numpy())
        assert np.allclose(result["ath_distance"].to_numpy(), expected, atol=1e-9), (
            "ath_distance does not match log(cfg.doge_ath_price / close)"
        )

    def test_ath_distance_column_present(self):
        result = compute_htf_features(_make_1h(30), _make_4h(10), _make_1d(5))
        assert "ath_distance" in result.columns


# ---------------------------------------------------------------------------
# TestHTFValues
# ---------------------------------------------------------------------------


class TestHTFValues:
    """Sanity checks on indicator value ranges."""

    def test_htf_4h_rsi_in_range(self):
        """RSI values must be in [0, 100] (or NaN during warmup)."""
        result = compute_htf_features(_make_1h(200, seed=30), _make_4h(60, seed=31), _make_1d(10))
        rsi = result["htf_4h_rsi"].dropna().to_numpy()
        assert np.all(rsi >= 0.0), f"RSI below 0: min={rsi.min():.2f}"
        assert np.all(rsi <= 100.0), f"RSI above 100: max={rsi.max():.2f}"

    def test_htf_4h_trend_valid_values(self):
        """htf_4h_trend must only be -1, 0, or +1 (0 only during EMA warmup)."""
        result = compute_htf_features(_make_1h(200), _make_4h(60), _make_1d(10))
        unique = set(result["htf_4h_trend"].dropna().astype(int).unique())
        assert unique.issubset({-1, 0, 1}), f"Unexpected htf_4h_trend values: {unique}"

    def test_htf_1d_trend_valid_values(self):
        """htf_1d_trend must only be -1, 0, or +1 (NaN only during warmup)."""
        result = compute_htf_features(_make_1h(200), _make_4h(60), _make_1d(15))
        unique = set(result["htf_1d_trend"].dropna().astype(int).unique())
        assert unique.issubset({-1, 0, 1}), f"Unexpected htf_1d_trend values: {unique}"

    def test_all_htf_feature_names_present(self):
        """All six canonical HTF feature names must be in output."""
        result = compute_htf_features(_make_1h(200), _make_4h(60), _make_1d(10))
        for col in HTF_FEATURE_NAMES:
            assert col in result.columns, f"Missing HTF feature: {col}"

    def test_output_is_copy(self):
        """compute_htf_features must not modify the input DataFrame in-place."""
        df_1h = _make_1h(50)
        original_cols = set(df_1h.columns)
        compute_htf_features(df_1h, _make_4h(20), _make_1d(5))
        assert set(df_1h.columns) == original_cols

    def test_row_count_preserved(self):
        """Output must have the same number of rows as input doge_1h."""
        df_1h = _make_1h(96)
        result = compute_htf_features(df_1h, _make_4h(30), _make_1d(5))
        assert len(result) == len(df_1h)


# ---------------------------------------------------------------------------
# TestHTFInputValidation
# ---------------------------------------------------------------------------


class TestHTFInputValidation:
    """Missing-column error handling."""

    def test_missing_1h_open_time(self):
        with pytest.raises(ValueError, match="doge_1h missing columns"):
            compute_htf_features(_make_1h(20).drop(columns=["open_time"]), _make_4h(5), _make_1d(5))

    def test_missing_4h_close(self):
        with pytest.raises(ValueError, match="doge_4h missing columns"):
            compute_htf_features(_make_1h(20), _make_4h(10).drop(columns=["close"]), _make_1d(5))

    def test_missing_1d_close(self):
        with pytest.raises(ValueError, match="doge_1d missing columns"):
            compute_htf_features(_make_1h(20), _make_4h(5), _make_1d(5).drop(columns=["close"]))


# ---------------------------------------------------------------------------
# TestHTFOrderBook
# ---------------------------------------------------------------------------


class TestHTFOrderBook:
    """Unit tests for compute_orderbook_features."""

    def _sample_book(self) -> dict:
        return {
            "bids": [["0.10050", "5000"], ["0.10040", "3000"], ["0.10030", "2000"]],
            "asks": [["0.10060", "2000"], ["0.10070", "1000"], ["0.10080", "500"]],
        }

    def test_returns_expected_keys(self):
        result = compute_orderbook_features(self._sample_book())
        assert "bid_ask_spread" in result
        assert "order_book_imbalance" in result

    def test_bid_ask_spread_positive(self):
        result = compute_orderbook_features(self._sample_book())
        assert result["bid_ask_spread"] > 0.0

    def test_bid_ask_spread_formula(self):
        book = {"bids": [["0.10000", "1000"]], "asks": [["0.10010", "1000"]]}
        result = compute_orderbook_features(book)
        mid = (0.10000 + 0.10010) / 2.0
        expected = (0.10010 - 0.10000) / mid
        assert abs(result["bid_ask_spread"] - expected) < 1e-9

    def test_imbalance_positive_when_bid_heavy(self):
        book = {"bids": [["0.100", "9000"]], "asks": [["0.101", "1000"]]}
        assert compute_orderbook_features(book)["order_book_imbalance"] > 0.0

    def test_imbalance_negative_when_ask_heavy(self):
        book = {"bids": [["0.100", "1000"]], "asks": [["0.101", "9000"]]}
        assert compute_orderbook_features(book)["order_book_imbalance"] < 0.0

    def test_imbalance_zero_when_balanced(self):
        book = {"bids": [["0.100", "1000"]], "asks": [["0.101", "1000"]]}
        assert abs(compute_orderbook_features(book)["order_book_imbalance"]) < 1e-9

    def test_imbalance_in_range(self):
        """Imbalance must always be in [-1, +1]."""
        rng = np.random.default_rng(42)
        bids = [[str(round(0.1 - i * 0.0001, 6)), str(int(rng.integers(100, 10000)))] for i in range(10)]
        asks = [[str(round(0.1001 + i * 0.0001, 6)), str(int(rng.integers(100, 10000)))] for i in range(10)]
        result = compute_orderbook_features({"bids": bids, "asks": asks})
        assert -1.0 <= result["order_book_imbalance"] <= 1.0

    def test_empty_order_book_returns_zeros(self):
        result = compute_orderbook_features({"bids": [], "asks": []})
        assert result["bid_ask_spread"] == 0.0
        assert result["order_book_imbalance"] == 0.0

    def test_missing_keys_returns_zeros(self):
        result = compute_orderbook_features({})
        assert result["bid_ask_spread"] == 0.0
        assert result["order_book_imbalance"] == 0.0

    def test_uses_top_10_levels_only(self):
        """Only the top 10 bid/ask levels affect the imbalance calculation."""
        # Build 15 levels with huge volume at level 11 (index 10)
        bids = [[str(round(0.1 - i * 0.00001, 6)), "1000"] for i in range(15)]
        asks = [[str(round(0.10001 + i * 0.00001, 6)), "1000"] for i in range(15)]
        bids[10] = [bids[10][0], "99999999"]
        asks[10] = [asks[10][0], "99999999"]

        result_all = compute_orderbook_features({"bids": bids, "asks": asks})
        result_top10 = compute_orderbook_features({"bids": bids[:10], "asks": asks[:10]})
        assert abs(
            result_all["order_book_imbalance"] - result_top10["order_book_imbalance"]
        ) < 1e-9
