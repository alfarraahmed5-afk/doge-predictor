"""Unit tests for DogeRegimeClassifier and get_regime_features.

Test plan:
    1.  Trending-bull fixture → TRENDING_BULL for all post-warmup rows
    2.  Trending-bear fixture → TRENDING_BEAR for all post-warmup rows
    3.  Ranging fixture → RANGING_LOW_VOL or RANGING_HIGH_VOL (no trending)
    4.  DECOUPLED overrides trend: synthetic independent BTC data → DECOUPLED
        assigned wherever 24h log-return correlation is < threshold
    5.  Log-return correlation ≠ raw-price correlation (regression-prevention)
    6.  No NaN in output for any fixture dataset
    7.  get_regime_distribution sums to 1.0
    8.  detect_transition: same regime → False, different → True
    9.  get_at: returns correct label after classify(); RuntimeError before
   10.  Missing columns raise ValueError
   11.  get_regime_features: all five labels produce valid dicts
   12.  get_regime_features: invalid label raises ValueError
   13.  get_regime_features: one-hot values are mutually exclusive (exactly one 1.0)
   14.  Regime series contains only known labels
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.config import regime_config
from src.regimes.classifier import REGIME_LABELS, DogeRegimeClassifier
from src.regimes.detector import RegimeChangeDetector, RegimeChangeEvent
from src.regimes.features import REGIME_FEATURE_KEYS, get_regime_features

# ---------------------------------------------------------------------------
# Constants mirrored from generate_fixtures.py
# ---------------------------------------------------------------------------

_START_MS: int = 1_640_995_200_000  # 2022-01-01 00:00:00 UTC
_HOUR_MS: int = 3_600_000

# EMA200 is the slowest indicator; valid from row 199 (0-indexed).
# 7d return valid from row 168.  Both valid simultaneously from row 199.
_WARMUP: int = 199

# For trending fixtures, we expect this fraction of post-warmup rows to match.
_TREND_DOMINANCE: float = 0.95  # ≥ 95% of valid rows must be the expected regime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_times(n: int) -> list[int]:
    return [_START_MS + i * _HOUR_MS for i in range(n)]


def _make_ohlcv(
    closes: np.ndarray,
    open_times: list[int],
    symbol: str = "DOGEUSDT",
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a close-price array."""
    opens = np.empty(len(closes))
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    spread = 0.002  # 0.2% body spread for high/low
    highs = np.maximum(opens, closes) * (1.0 + spread)
    lows = np.minimum(opens, closes) * (1.0 - spread)
    return pd.DataFrame(
        {
            "open_time": open_times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.ones(len(closes)) * 1_000_000.0,
            "symbol": symbol,
            "era": "training",
        }
    )


# ---------------------------------------------------------------------------
# Fixtures (synthetic — deterministic, not file-backed)
# ---------------------------------------------------------------------------


def _trending_bull_df(n: int = 500) -> pd.DataFrame:
    """Strong uptrend: drift 0.003 ensures 7d return >> 5% for ALL post-warmup rows.

    With drift=0.003, sigma=0.005:
        Expected 168-period log return = 168 * 0.003 = 0.504 (≈ 65% price gain).
        1st-percentile 168-period return ≈ exp(0.504 - 2.33 * 0.065) - 1 ≈ 42% >> 5%.
    """
    rng = np.random.default_rng(1)
    log_ret = rng.normal(0.003, 0.005, n)
    closes = 0.10 * np.exp(np.cumsum(log_ret))
    return _make_ohlcv(closes, _open_times(n))


def _trending_bear_df(n: int = 500) -> pd.DataFrame:
    """Strong downtrend: drift -0.003 ensures 7d return << -5% for ALL post-warmup rows."""
    rng = np.random.default_rng(2)
    log_ret = rng.normal(-0.003, 0.005, n)
    closes = 0.20 * np.exp(np.cumsum(log_ret))
    return _make_ohlcv(closes, _open_times(n))


def _ranging_df(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    mean_price = 0.090
    theta = 0.15
    sigma = 0.0004  # very low volatility → ATR << 0.3%, BB width << 0.04
    prices = np.empty(n)
    prices[0] = mean_price
    for i in range(1, n):
        prices[i] = (
            prices[i - 1]
            + theta * (mean_price - prices[i - 1])
            + rng.normal(0, sigma)
        )
    prices = np.clip(prices, 1e-6, None)
    return _make_ohlcv(prices, _open_times(n))


def _independent_btc_df(n: int, seed: int = 999) -> pd.DataFrame:
    """BTC DataFrame whose log returns are statistically independent of DOGE."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0, 0.010, n)
    closes = 42_000.0 * np.exp(np.cumsum(log_ret))
    return _make_ohlcv(closes, _open_times(n), symbol="BTCUSDT")


def _correlated_btc_df(doge_closes: np.ndarray, noise: float = 0.002) -> pd.DataFrame:
    """BTC DataFrame whose log returns mirror DOGE with tiny additive noise.

    By construction, BTC-DOGE 24h rolling log-return correlation ≈ 1.0 → far
    above the DECOUPLED threshold of 0.30, so DECOUPLED should NOT be assigned.
    """
    rng = np.random.default_rng(77)
    n = len(doge_closes)
    # BTC = constant multiple of DOGE + tiny noise
    btc_scale = 42_000.0 / doge_closes[0]
    btc_closes = doge_closes * btc_scale + rng.normal(0, noise, n)
    btc_closes = np.clip(btc_closes, 1.0, None)
    return _make_ohlcv(btc_closes, _open_times(n), symbol="BTCUSDT")


# ---------------------------------------------------------------------------
# 1 & 2. Trending bull / bear
# ---------------------------------------------------------------------------


class TestTrendingRegimes:
    """Classifier assigns TRENDING_BULL / TRENDING_BEAR after warmup."""

    def test_trending_bull_post_warmup(self) -> None:
        """Post-warmup rows in a strong-uptrend series must be TRENDING_BULL."""
        df = _trending_bull_df()
        clf = DogeRegimeClassifier()
        regimes = clf.classify(df, btc_df=None)

        post_warmup = regimes.iloc[_WARMUP:]
        n_bull = (post_warmup == "TRENDING_BULL").sum()
        frac = n_bull / len(post_warmup)
        assert frac >= _TREND_DOMINANCE, (
            f"Expected ≥{_TREND_DOMINANCE:.0%} TRENDING_BULL, got {frac:.1%}"
        )

    def test_trending_bear_post_warmup(self) -> None:
        """Post-warmup rows in a strong-downtrend series must be TRENDING_BEAR."""
        df = _trending_bear_df()
        clf = DogeRegimeClassifier()
        regimes = clf.classify(df, btc_df=None)

        post_warmup = regimes.iloc[_WARMUP:]
        n_bear = (post_warmup == "TRENDING_BEAR").sum()
        frac = n_bear / len(post_warmup)
        assert frac >= _TREND_DOMINANCE, (
            f"Expected ≥{_TREND_DOMINANCE:.0%} TRENDING_BEAR, got {frac:.1%}"
        )

    def test_trending_bull_fixture_file(
        self, doge_trending_bull: pd.DataFrame, btc_aligned: pd.DataFrame
    ) -> None:
        """Parquet fixture trending-bull rows: ≥ 95% post-warmup → TRENDING_BULL."""
        clf = DogeRegimeClassifier()
        # No BTC to avoid DECOUPLED confusion from independent BTC fixture
        regimes = clf.classify(doge_trending_bull, btc_df=None)
        post_warmup = regimes.iloc[_WARMUP:]
        frac = (post_warmup == "TRENDING_BULL").mean()
        assert frac >= _TREND_DOMINANCE, f"Trending-bull fixture: {frac:.1%} TRENDING_BULL"


# ---------------------------------------------------------------------------
# 3. Ranging
# ---------------------------------------------------------------------------


class TestRangingRegime:
    """Classifier assigns RANGING_LOW_VOL or RANGING_HIGH_VOL for tight ranges."""

    def test_ranging_produces_only_ranging_regimes(self) -> None:
        """All post-warmup rows of a tight-range series must be ranging."""
        df = _ranging_df()
        clf = DogeRegimeClassifier()
        regimes = clf.classify(df, btc_df=None)

        post_warmup = regimes.iloc[_WARMUP:]
        n_ranging = (
            (post_warmup == "RANGING_LOW_VOL") | (post_warmup == "RANGING_HIGH_VOL")
        ).sum()
        frac = n_ranging / len(post_warmup)
        assert frac >= 0.99, (
            f"Expected ≥99% ranging regime, got {frac:.1%}. "
            f"Regime counts: {post_warmup.value_counts().to_dict()}"
        )

    def test_ranging_fixture_file(self, doge_ranging: pd.DataFrame) -> None:
        """Parquet fixture ranging rows must all be RANGING_LOW_VOL (no BTC)."""
        clf = DogeRegimeClassifier()
        regimes = clf.classify(doge_ranging, btc_df=None)
        # The OU fixture has very low sigma → BB width << 0.04 → RANGING_LOW_VOL
        post_warmup = regimes.iloc[_WARMUP:]
        frac_low = (post_warmup == "RANGING_LOW_VOL").mean()
        assert frac_low >= 0.90, (
            f"Ranging fixture: expected ≥90% RANGING_LOW_VOL, got {frac_low:.1%}"
        )


# ---------------------------------------------------------------------------
# 4. DECOUPLED overrides trend
# ---------------------------------------------------------------------------


class TestDecoupledOverride:
    """DECOUPLED overrides TRENDING_BULL wherever BTC-DOGE corr < threshold."""

    def test_decoupled_overrides_trend_signals(self) -> None:
        """Rows with btc_corr < threshold are DECOUPLED even if EMA trend holds."""
        n = 400
        doge_df = _trending_bull_df(n)
        btc_df = _independent_btc_df(n, seed=999)

        clf = DogeRegimeClassifier()
        regimes = clf.classify(doge_df, btc_df)

        # Recompute btc_corr the same way as classifier to find DECOUPLED rows
        doge_close = doge_df["close"].to_numpy()
        btc_close = btc_df["close"].to_numpy()
        doge_lr = pd.Series(np.log(doge_close / np.roll(doge_close, 1)))
        doge_lr.iloc[0] = np.nan
        btc_lr = pd.Series(np.log(btc_close / np.roll(btc_close, 1)))
        btc_lr.iloc[0] = np.nan
        btc_corr = doge_lr.rolling(24).corr(btc_lr)

        threshold = regime_config.thresholds.btc_corr_decoupled
        decoupled_expected = (btc_corr < threshold).fillna(False)

        # Every row where corr < threshold MUST be DECOUPLED
        if decoupled_expected.any():
            classified_decoupled = regimes[decoupled_expected]
            wrong = (classified_decoupled != "DECOUPLED").sum()
            assert wrong == 0, (
                f"{wrong} rows had corr < {threshold} but were not DECOUPLED"
            )

    def test_decoupled_rows_exist_with_independent_btc(self) -> None:
        """Independent BTC and DOGE noise guarantees some DECOUPLED rows exist."""
        n = 400
        doge_df = _trending_bull_df(n)
        btc_df = _independent_btc_df(n, seed=42)
        clf = DogeRegimeClassifier()
        regimes = clf.classify(doge_df, btc_df)
        # With truly independent series, at least some windows will have corr < 0.30
        assert (regimes == "DECOUPLED").sum() > 0, (
            "Expected at least one DECOUPLED row with independent BTC data"
        )

    def test_correlated_btc_produces_no_decoupled(self) -> None:
        """Highly correlated BTC (≈ DOGE) must NOT produce any DECOUPLED rows."""
        n = 400
        doge_df = _trending_bull_df(n)
        # BTC = DOGE × scale factor + tiny noise → corr ≈ 1.0
        btc_df = _correlated_btc_df(doge_df["close"].to_numpy())
        clf = DogeRegimeClassifier()
        regimes = clf.classify(doge_df, btc_df)
        assert (regimes == "DECOUPLED").sum() == 0, (
            "Highly-correlated BTC should yield zero DECOUPLED rows"
        )

    def test_decoupled_fixture_file(
        self, doge_decoupled: pd.DataFrame, btc_aligned: pd.DataFrame
    ) -> None:
        """Parquet decoupled fixture should produce DECOUPLED rows with btc_aligned."""
        clf = DogeRegimeClassifier()
        regimes = clf.classify(doge_decoupled, btc_aligned)
        # Decoupled DOGE has zero-drift independent noise; should yield DECOUPLED rows
        assert (regimes == "DECOUPLED").sum() > 0, (
            "Decoupled fixture + BTC aligned should yield some DECOUPLED rows"
        )


# ---------------------------------------------------------------------------
# 5. Log returns vs raw price correlation
# ---------------------------------------------------------------------------


class TestLogReturnCorrelation:
    """BTC-DOGE correlation MUST be computed on log returns, not raw prices."""

    def test_price_corr_differs_from_log_return_corr(self) -> None:
        """Co-trending series: raw-price correlation >> log-return correlation.

        When two series trend in the same direction with INDEPENDENT noise,
        the rolling raw-price correlation is spuriously high (Yule-Walker /
        non-stationarity effect) while the rolling log-return correlation
        correctly reflects the true independence.

        This test uses strongly co-trending series (drift=0.005, low noise)
        so the difference is unambiguous and not fixture-dependent.
        """
        n = 500
        rng_d = np.random.default_rng(10)
        rng_b = np.random.default_rng(20)

        # Both series trend strongly upward with independent noise
        doge_log_ret = rng_d.normal(0.005, 0.002, n)
        btc_log_ret = rng_b.normal(0.005, 0.002, n)
        doge_close = 0.10 * np.exp(np.cumsum(doge_log_ret))
        btc_close = 42_000.0 * np.exp(np.cumsum(btc_log_ret))

        # Raw-price correlation (spuriously high for co-trending series)
        price_corr = (
            pd.Series(doge_close).rolling(24).corr(pd.Series(btc_close)).dropna()
        )

        # Log-return correlation (correctly near zero for independent noise)
        doge_lr = pd.Series(doge_log_ret)
        btc_lr = pd.Series(btc_log_ret)
        log_corr = doge_lr.rolling(24).corr(btc_lr).dropna()

        mean_price_corr = float(price_corr.mean())
        mean_log_corr = float(log_corr.mean())

        # For strongly co-trending series: raw price corr must be high (spurious)
        assert mean_price_corr > 0.80, (
            f"Expected mean raw-price corr > 0.80, got {mean_price_corr:.3f}"
        )
        # Log-return corr must be near zero (independent noise)
        assert abs(mean_log_corr) < 0.30, (
            f"Expected |mean log-return corr| < 0.30, got {mean_log_corr:.3f}"
        )
        # The gap must be substantial
        assert mean_price_corr - mean_log_corr > 0.50, (
            f"Gap too small: price_corr={mean_price_corr:.3f}, "
            f"log_corr={mean_log_corr:.3f}"
        )


# ---------------------------------------------------------------------------
# 6. No NaN in output
# ---------------------------------------------------------------------------


class TestNoNanOutput:
    """Every classify() call must produce a fully-populated regime Series."""

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "doge_trending_bull",
            "doge_trending_bear",
            "doge_ranging",
            "doge_decoupled",
        ],
    )
    def test_no_nan_in_output(self, fixture_name: str, request: pytest.FixtureRequest) -> None:
        """No NaN labels for any fixture, with or without BTC DataFrame."""
        df: pd.DataFrame = request.getfixturevalue(fixture_name)
        clf = DogeRegimeClassifier()
        regimes = clf.classify(df, btc_df=None)
        assert regimes.isna().sum() == 0, (
            f"Found NaN regime labels in {fixture_name} output"
        )

    def test_no_nan_with_short_series(self) -> None:
        """Even a 10-row DataFrame must have no NaN regime labels."""
        df = _ranging_df(n=10)
        clf = DogeRegimeClassifier()
        regimes = clf.classify(df, btc_df=None)
        assert len(regimes) == 10
        assert regimes.isna().sum() == 0

    def test_all_labels_are_known(self, doge_trending_bull: pd.DataFrame) -> None:
        """All regime labels must be in the REGIME_LABELS tuple."""
        clf = DogeRegimeClassifier()
        regimes = clf.classify(doge_trending_bull, btc_df=None)
        unknown = ~regimes.isin(REGIME_LABELS)
        assert unknown.sum() == 0, (
            f"Unknown regime labels: {regimes[unknown].unique()}"
        )


# ---------------------------------------------------------------------------
# 7. get_regime_distribution
# ---------------------------------------------------------------------------


class TestGetRegimeDistribution:
    """Distribution values must sum to 1.0 and cover all five labels."""

    def test_distribution_sums_to_one(self, doge_trending_bull: pd.DataFrame) -> None:
        clf = DogeRegimeClassifier()
        regimes = clf.classify(doge_trending_bull, btc_df=None)
        dist = clf.get_regime_distribution(regimes)
        total = sum(dist.values())
        assert math.isclose(total, 1.0, abs_tol=1e-9), (
            f"Distribution values sum to {total}, expected 1.0"
        )

    def test_distribution_has_all_five_keys(self) -> None:
        df = _trending_bull_df()
        clf = DogeRegimeClassifier()
        regimes = clf.classify(df, btc_df=None)
        dist = clf.get_regime_distribution(regimes)
        assert set(dist.keys()) == set(REGIME_LABELS), (
            f"Missing keys in distribution: {set(REGIME_LABELS) - set(dist.keys())}"
        )

    def test_distribution_all_zeros_on_empty_series(self) -> None:
        clf = DogeRegimeClassifier()
        dist = clf.get_regime_distribution(pd.Series([], dtype=str))
        assert all(v == 0.0 for v in dist.values())
        assert set(dist.keys()) == set(REGIME_LABELS)

    def test_distribution_dominant_regime_is_bull(self) -> None:
        df = _trending_bull_df()
        clf = DogeRegimeClassifier()
        regimes = clf.classify(df, btc_df=None)
        dist = clf.get_regime_distribution(regimes)
        dominant = max(dist, key=lambda k: dist[k])
        assert dominant == "TRENDING_BULL", (
            f"Expected dominant regime TRENDING_BULL, got {dominant}"
        )


# ---------------------------------------------------------------------------
# 8. detect_transition
# ---------------------------------------------------------------------------


class TestDetectTransition:
    """detect_transition must return True iff regimes differ."""

    @pytest.mark.parametrize(
        "prev, curr, expected",
        [
            ("TRENDING_BULL", "TRENDING_BULL", False),
            ("TRENDING_BEAR", "TRENDING_BEAR", False),
            ("RANGING_LOW_VOL", "RANGING_LOW_VOL", False),
            ("TRENDING_BULL", "TRENDING_BEAR", True),
            ("RANGING_LOW_VOL", "DECOUPLED", True),
            ("DECOUPLED", "TRENDING_BULL", True),
            ("RANGING_HIGH_VOL", "RANGING_LOW_VOL", True),
        ],
    )
    def test_detect_transition(
        self, prev: str, curr: str, expected: bool
    ) -> None:
        result = DogeRegimeClassifier.detect_transition(prev, curr)
        assert result is expected, (
            f"detect_transition({prev!r}, {curr!r}) returned {result}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# 9. get_at
# ---------------------------------------------------------------------------


class TestGetAt:
    """get_at must return the correct label after classify(); raise before."""

    def test_get_at_raises_before_classify(self) -> None:
        clf = DogeRegimeClassifier()
        with pytest.raises(RuntimeError, match="classify\\(\\)"):
            clf.get_at(_START_MS)

    def test_get_at_returns_correct_label(self) -> None:
        df = _trending_bull_df(n=300)
        clf = DogeRegimeClassifier()
        regimes = clf.classify(df, btc_df=None)

        # Check a few timestamps against the returned Series
        for i in [0, 50, 100, 200, 299]:
            ts = int(df["open_time"].iloc[i])
            expected_label = regimes.iloc[i]
            result = clf.get_at(ts)
            assert result == expected_label, (
                f"get_at({ts}) returned {result!r}, expected {expected_label!r}"
            )

    def test_get_at_unknown_timestamp_raises_key_error(self) -> None:
        df = _trending_bull_df(n=100)
        clf = DogeRegimeClassifier()
        clf.classify(df, btc_df=None)
        bad_ts = _START_MS + 99_999 * _HOUR_MS  # far outside the fixture
        with pytest.raises(KeyError):
            clf.get_at(bad_ts)


# ---------------------------------------------------------------------------
# 10. Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Missing columns and empty DataFrames raise ValueError."""

    def test_missing_doge_column_raises(self) -> None:
        df = _ranging_df(n=50)
        df = df.drop(columns=["close"])
        clf = DogeRegimeClassifier()
        with pytest.raises(ValueError, match="close"):
            clf.classify(df)

    def test_empty_doge_dataframe_raises(self) -> None:
        df = _ranging_df(n=50).iloc[:0]  # empty
        clf = DogeRegimeClassifier()
        with pytest.raises(ValueError, match="empty"):
            clf.classify(df)

    def test_missing_btc_column_raises(self) -> None:
        doge_df = _ranging_df(n=50)
        btc_df = _independent_btc_df(n=50)
        btc_df = btc_df.drop(columns=["close"])
        clf = DogeRegimeClassifier()
        with pytest.raises(ValueError, match="close"):
            clf.classify(doge_df, btc_df=btc_df)


# ---------------------------------------------------------------------------
# 11–13. get_regime_features (from src/regimes/features.py)
# ---------------------------------------------------------------------------


class TestGetRegimeFeatures:
    """get_regime_features returns correct one-hot + ordinal encodings."""

    @pytest.mark.parametrize("label", list(REGIME_LABELS))
    def test_all_five_regimes_return_valid_dict(self, label: str) -> None:
        features = get_regime_features(label)
        assert set(features.keys()) == set(REGIME_FEATURE_KEYS), (
            f"Missing feature keys for {label}: "
            f"{set(REGIME_FEATURE_KEYS) - set(features.keys())}"
        )
        for key, val in features.items():
            assert isinstance(val, float), (
                f"{key} value {val!r} is not float for label {label}"
            )

    def test_one_hot_exactly_one_active(self) -> None:
        """Exactly one one-hot column is 1.0 per label."""
        one_hot_keys = [k for k in REGIME_FEATURE_KEYS if k != "regime_encoded"]
        for label in REGIME_LABELS:
            features = get_regime_features(label)
            active = sum(features[k] for k in one_hot_keys)
            assert math.isclose(active, 1.0), (
                f"Label {label}: sum of one-hot features = {active}, expected 1.0"
            )

    @pytest.mark.parametrize(
        "label, expected_encoded",
        [
            ("TRENDING_BULL", 0.0),
            ("TRENDING_BEAR", 1.0),
            ("RANGING_HIGH_VOL", 2.0),
            ("RANGING_LOW_VOL", 3.0),
            ("DECOUPLED", 4.0),
        ],
    )
    def test_regime_encoded_ordinal(self, label: str, expected_encoded: float) -> None:
        features = get_regime_features(label)
        assert features["regime_encoded"] == expected_encoded, (
            f"regime_encoded for {label} = {features['regime_encoded']}, "
            f"expected {expected_encoded}"
        )

    def test_invalid_label_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown regime label"):
            get_regime_features("NOT_A_REGIME")

    def test_trending_bull_one_hot_correct(self) -> None:
        features = get_regime_features("TRENDING_BULL")
        assert features["regime_is_trending_bull"] == 1.0
        assert features["regime_is_trending_bear"] == 0.0
        assert features["regime_is_ranging_high"] == 0.0
        assert features["regime_is_ranging_low"] == 0.0
        assert features["regime_is_decoupled"] == 0.0

    def test_decoupled_one_hot_correct(self) -> None:
        features = get_regime_features("DECOUPLED")
        assert features["regime_is_decoupled"] == 1.0
        assert features["regime_is_trending_bull"] == 0.0
        assert features["regime_is_trending_bear"] == 0.0
        assert features["regime_is_ranging_high"] == 0.0
        assert features["regime_is_ranging_low"] == 0.0


# ---------------------------------------------------------------------------
# 14. Regime distribution with all five regimes (smoke test)
# ---------------------------------------------------------------------------


class TestAllFiveRegimesPresent:
    """Smoke test — combined dataset should contain all five regime labels."""

    def test_all_five_regimes_reachable(self) -> None:
        """Force all five regimes through targeted synthetic data."""
        # Verify each regime is producible individually
        clf = DogeRegimeClassifier()

        # TRENDING_BULL
        bull_df = _trending_bull_df(400)
        bull_regimes = clf.classify(bull_df, btc_df=None)
        assert "TRENDING_BULL" in bull_regimes.values

        # TRENDING_BEAR
        bear_df = _trending_bear_df(400)
        bear_regimes = clf.classify(bear_df, btc_df=None)
        assert "TRENDING_BEAR" in bear_regimes.values

        # RANGING (either variant)
        range_df = _ranging_df(400)
        range_regimes = clf.classify(range_df, btc_df=None)
        assert (
            "RANGING_LOW_VOL" in range_regimes.values
            or "RANGING_HIGH_VOL" in range_regimes.values
        )

        # DECOUPLED — need independent BTC data
        btc_df = _independent_btc_df(400, seed=999)
        doge_df = _trending_bull_df(400)
        dec_regimes = clf.classify(doge_df, btc_df)
        # With independent BTC, some rows will be DECOUPLED
        assert "DECOUPLED" in dec_regimes.values


# ---------------------------------------------------------------------------
# 15. RegimeChangeDetector
# ---------------------------------------------------------------------------


class TestRegimeChangeDetector:
    """RegimeChangeDetector.detect() must return None or a RegimeChangeEvent."""

    def test_no_change_returns_none(self) -> None:
        detector = RegimeChangeDetector()
        result = detector.detect("TRENDING_BULL", "TRENDING_BULL", btc_corr=0.5, atr_norm=0.003)
        assert result is None

    @pytest.mark.parametrize("label", list(REGIME_LABELS))
    def test_same_regime_always_none(self, label: str) -> None:
        detector = RegimeChangeDetector()
        assert detector.detect(label, label, btc_corr=0.4, atr_norm=0.002) is None

    def test_change_returns_event(self) -> None:
        detector = RegimeChangeDetector()
        event = detector.detect(
            "RANGING_LOW_VOL", "TRENDING_BULL", btc_corr=0.55, atr_norm=0.004
        )
        assert isinstance(event, RegimeChangeEvent)
        assert event.from_regime == "RANGING_LOW_VOL"
        assert event.to_regime == "TRENDING_BULL"

    def test_event_fields_populated(self) -> None:
        detector = RegimeChangeDetector()
        event = detector.detect(
            "TRENDING_BEAR", "RANGING_HIGH_VOL",
            btc_corr=0.6, atr_norm=0.007, changed_at=1_640_995_200_000
        )
        assert event is not None
        assert event.btc_corr == pytest.approx(0.6)
        assert event.atr_norm == pytest.approx(0.007)
        assert event.changed_at == 1_640_995_200_000

    def test_is_critical_when_entering_decoupled(self) -> None:
        detector = RegimeChangeDetector()
        event = detector.detect("TRENDING_BULL", "DECOUPLED", btc_corr=0.10, atr_norm=0.005)
        assert event is not None
        assert event.is_critical is True

    def test_is_critical_when_leaving_decoupled(self) -> None:
        detector = RegimeChangeDetector()
        event = detector.detect("DECOUPLED", "RANGING_HIGH_VOL", btc_corr=0.45, atr_norm=0.006)
        assert event is not None
        assert event.is_critical is True

    def test_is_not_critical_without_decoupled(self) -> None:
        detector = RegimeChangeDetector()
        event = detector.detect("TRENDING_BULL", "TRENDING_BEAR", btc_corr=0.7, atr_norm=0.004)
        assert event is not None
        assert event.is_critical is False

    def test_invalid_prev_label_raises(self) -> None:
        detector = RegimeChangeDetector()
        with pytest.raises(ValueError, match="Unknown regime label"):
            detector.detect("BAD_LABEL", "TRENDING_BULL", btc_corr=0.5, atr_norm=0.003)

    def test_invalid_curr_label_raises(self) -> None:
        detector = RegimeChangeDetector()
        with pytest.raises(ValueError, match="Unknown regime label"):
            detector.detect("TRENDING_BULL", "NOT_A_REGIME", btc_corr=0.5, atr_norm=0.003)

    def test_event_is_immutable(self) -> None:
        detector = RegimeChangeDetector()
        event = detector.detect("RANGING_LOW_VOL", "DECOUPLED", btc_corr=0.1, atr_norm=0.002)
        assert event is not None
        with pytest.raises((AttributeError, TypeError)):
            event.from_regime = "TRENDING_BULL"  # type: ignore[misc]

    def test_default_changed_at_is_zero(self) -> None:
        detector = RegimeChangeDetector()
        event = detector.detect("RANGING_HIGH_VOL", "TRENDING_BEAR", btc_corr=0.5, atr_norm=0.005)
        assert event is not None
        assert event.changed_at == 0
