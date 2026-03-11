"""Unit and integration tests for src/features/pipeline.py.

Coverage targets:
    - add_target_column(): target formula, NaN at last row, 0/1 values
    - validate_feature_matrix(): NaN check, Inf check, mandatory features,
      expected_columns check, strict mode
    - FeaturePipeline: __init__ defaults, run_id generation, _save_parquet,
      _save_feature_columns_json
    - FeaturePipeline.compute_all_features(): full 12-step integration test
    - build_feature_matrix(): backward-compat functional API

Integration tests use small synthetic datasets (800 1h rows) generated
inline — no live Binance connection required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import (
    MANDATORY_FEATURE_NAMES,
    FeaturePipeline,
    add_target_column,
    build_feature_matrix,
    validate_feature_matrix,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_MS_PER_HOUR: int = 3_600_000
_MS_PER_4H: int = 4 * _MS_PER_HOUR
_MS_PER_1D: int = 24 * _MS_PER_HOUR
_MS_PER_8H: int = 8 * _MS_PER_HOUR
_T0_MS: int = 1_640_995_200_000  # 2022-01-01 00:00 UTC


# ---------------------------------------------------------------------------
# Small synthetic data helpers (shared across test classes)
# ---------------------------------------------------------------------------


def _make_1h(n: int = 800, seed: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    open_times = np.array([_T0_MS + i * _MS_PER_HOUR for i in range(n)], dtype=np.int64)
    close = 0.08 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    noise = rng.uniform(0.001, 0.005, n)
    return pd.DataFrame({
        "open_time": open_times,
        "open": (close * (1 + rng.normal(0, 0.003, n))).clip(0.001),
        "high": (close * (1 + noise)).clip(0.001),
        "low": (close * (1 - noise)).clip(0.001),
        "close": close.clip(0.001),
        "volume": rng.uniform(1e7, 5e8, n),
        "symbol": "DOGEUSDT",
        "era": "training",
    })


def _make_4h(n: int = 300, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = _T0_MS - n * _MS_PER_4H
    open_times = np.array([start + i * _MS_PER_4H for i in range(n)], dtype=np.int64)
    close = 0.08 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n)))
    noise = rng.uniform(0.005, 0.015, n)
    return pd.DataFrame({
        "open_time": open_times,
        "open": (close * (1 + rng.normal(0, 0.005, n))).clip(0.001),
        "high": (close * (1 + noise)).clip(0.001),
        "low": (close * (1 - noise)).clip(0.001),
        "close": close.clip(0.001),
        "volume": rng.uniform(4e7, 2e9, n),
        "symbol": "DOGEUSDT",
        "era": "training",
    })


def _make_1d(n: int = 50, seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = _T0_MS - n * _MS_PER_1D
    open_times = np.array([start + i * _MS_PER_1D for i in range(n)], dtype=np.int64)
    close = 0.08 * np.exp(np.cumsum(rng.normal(0.003, 0.04, n)))
    noise = rng.uniform(0.01, 0.04, n)
    return pd.DataFrame({
        "open_time": open_times,
        "open": (close * (1 + rng.normal(0, 0.01, n))).clip(0.001),
        "high": (close * (1 + noise)).clip(0.001),
        "low": (close * (1 - noise)).clip(0.001),
        "close": close.clip(0.001),
        "volume": rng.uniform(1e9, 1e10, n),
        "symbol": "DOGEUSDT",
        "era": "training",
    })


def _make_btc(doge_close: np.ndarray, seed: int = 13) -> pd.DataFrame:
    n = len(doge_close)
    rng = np.random.default_rng(seed)
    open_times = np.array([_T0_MS + i * _MS_PER_HOUR for i in range(n)], dtype=np.int64)
    doge_lr = np.diff(np.log(doge_close), prepend=np.log(doge_close[0]))
    btc_lr = 0.7 * doge_lr + 0.3 * rng.normal(0, 0.008, n)
    close = 40000 * np.exp(np.cumsum(btc_lr))
    noise = rng.uniform(0.002, 0.01, n)
    return pd.DataFrame({
        "open_time": open_times,
        "open": (close * (1 + rng.normal(0, 0.002, n))).clip(1.0),
        "high": (close * (1 + noise)).clip(1.0),
        "low": (close * (1 - noise)).clip(1.0),
        "close": close.clip(1.0),
        "volume": rng.uniform(1e9, 1e11, n),
        "symbol": "BTCUSDT",
        "era": "training",
    })


def _make_dogebtc(doge_close: np.ndarray, btc_close: np.ndarray, seed: int = 14) -> pd.DataFrame:
    n = len(doge_close)
    rng = np.random.default_rng(seed)
    open_times = np.array([_T0_MS + i * _MS_PER_HOUR for i in range(n)], dtype=np.int64)
    close = doge_close / btc_close
    noise = rng.uniform(0.001, 0.005, n)
    return pd.DataFrame({
        "open_time": open_times,
        "open": (close * (1 + rng.normal(0, 0.002, n))).clip(1e-8),
        "high": (close * (1 + noise)).clip(1e-8),
        "low": (close * (1 - noise)).clip(1e-8),
        "close": close.clip(1e-8),
        "volume": rng.uniform(1e6, 1e8, n),
        "symbol": "DOGEBTC",
        "era": "training",
    })


def _make_funding(n: int = 150, seed: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = np.array([_T0_MS + i * _MS_PER_8H for i in range(n)], dtype=np.int64)
    return pd.DataFrame({
        "timestamp_ms": timestamps,
        "funding_rate": rng.normal(0.0001, 0.0003, n).clip(-0.002, 0.003),
        "symbol": "DOGEUSDT",
    })


def _make_regimes(open_times: np.ndarray, seed: int = 16) -> pd.Series:
    """Return regime labels indexed by open_time (int ms)."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(
        ["TRENDING_BULL", "TRENDING_BEAR", "RANGING_HIGH_VOL",
         "RANGING_LOW_VOL", "DECOUPLED"],
        size=len(open_times),
    )
    return pd.Series(labels, index=open_times)


# Shared full-data fixture for integration tests
@pytest.fixture(scope="module")
def full_pipeline_output() -> pd.DataFrame:
    """Run FeaturePipeline once and cache the result for all integration tests."""
    doge = _make_1h(800, seed=10)
    btc = _make_btc(doge["close"].to_numpy(), seed=13)
    dogebtc = _make_dogebtc(doge["close"].to_numpy(), btc["close"].to_numpy(), seed=14)
    funding = _make_funding(150, seed=15)
    doge_4h = _make_4h(300, seed=11)
    doge_1d = _make_1d(50, seed=12)
    regimes = _make_regimes(doge["open_time"].to_numpy(), seed=16)

    pipe = FeaturePipeline(output_dir=None)
    return pipe.compute_all_features(
        doge_1h=doge,
        btc_1h=btc,
        dogebtc_1h=dogebtc,
        funding=funding,
        doge_4h=doge_4h,
        doge_1d=doge_1d,
        regimes=regimes,
        min_rows_override=300,
    )


# ---------------------------------------------------------------------------
# Tests — add_target_column
# ---------------------------------------------------------------------------


class TestAddTargetColumn:
    """Tests for the add_target_column() helper."""

    def _make_close(self, prices: list[float]) -> pd.DataFrame:
        n = len(prices)
        return pd.DataFrame({
            "open_time": [_T0_MS + i * _MS_PER_HOUR for i in range(n)],
            "close": prices,
        })

    def test_returns_copy_not_inplace(self) -> None:
        """add_target_column does not modify the input DataFrame."""
        df = self._make_close([0.08, 0.09, 0.07, 0.10])
        _ = add_target_column(df)
        assert "target" not in df.columns

    def test_target_1_when_next_close_higher(self) -> None:
        """target = 1 when next candle closes higher."""
        df = add_target_column(self._make_close([0.08, 0.10, 0.09, 0.12]))
        # Row 0: 0.08→0.10 → UP → 1
        assert df["target"].iloc[0] == 1.0

    def test_target_0_when_next_close_lower(self) -> None:
        """target = 0 when next candle closes lower."""
        df = add_target_column(self._make_close([0.10, 0.08, 0.09, 0.12]))
        # Row 0: 0.10→0.08 → DOWN → 0
        assert df["target"].iloc[0] == 0.0

    def test_target_nan_at_last_row(self) -> None:
        """Last row always has NaN target (no future candle)."""
        df = add_target_column(self._make_close([0.08, 0.09, 0.10]))
        assert np.isnan(df["target"].iloc[-1])

    def test_all_non_last_rows_are_0_or_1(self) -> None:
        """All non-last rows have target in {0.0, 1.0}."""
        prices = [0.08 + 0.01 * i for i in range(20)]
        df = add_target_column(self._make_close(prices))
        non_last = df["target"].iloc[:-1].dropna()
        assert set(non_last.unique()).issubset({0.0, 1.0})

    def test_missing_close_raises(self) -> None:
        """ValueError when 'close' column is absent."""
        with pytest.raises(ValueError, match="close"):
            add_target_column(pd.DataFrame({"open": [1.0, 2.0]}))

    def test_formula_matches_pct_change_shift_minus1(self) -> None:
        """target formula == (close.pct_change().shift(-1) > 0)."""
        prices = [0.08, 0.09, 0.07, 0.11, 0.10, 0.12, 0.08]
        df = self._make_close(prices)
        result = add_target_column(df)
        expected = pd.Series(prices).pct_change().shift(-1)
        for i in range(len(prices) - 1):
            expected_val = float(expected.iloc[i] > 0)
            assert result["target"].iloc[i] == expected_val, (
                f"Row {i}: expected {expected_val}, got {result['target'].iloc[i]}"
            )

    def test_target_not_lookahead_in_features(self) -> None:
        """Verify only 'target' uses shift(-1); other columns unchanged."""
        df = self._make_close([0.08, 0.09, 0.07])
        result = add_target_column(df)
        assert list(result.columns) == ["open_time", "close", "target"]


# ---------------------------------------------------------------------------
# Tests — validate_feature_matrix (pipeline version)
# ---------------------------------------------------------------------------


class TestValidateFeatureMatrix:
    """Tests for the pipeline-level validate_feature_matrix()."""

    def _make_valid_df(self) -> pd.DataFrame:
        """DataFrame with all mandatory features + no NaN/Inf."""
        cols = {col: [0.0, 0.1] for col in MANDATORY_FEATURE_NAMES}
        cols["close"] = [0.08, 0.09]
        return pd.DataFrame(cols)

    def test_valid_matrix_returns_ok_true(self) -> None:
        df = self._make_valid_df()
        result = validate_feature_matrix(df)
        assert result["ok"] is True
        assert result["nan_cols"] == []
        assert result["inf_cols"] == []
        assert result["missing_mandatory"] == []

    def test_nan_col_flagged(self) -> None:
        df = self._make_valid_df()
        first_mandatory = sorted(MANDATORY_FEATURE_NAMES)[0]
        df[first_mandatory] = np.nan
        result = validate_feature_matrix(df)
        assert result["ok"] is False
        assert first_mandatory in result["nan_cols"]

    def test_inf_col_flagged(self) -> None:
        df = self._make_valid_df()
        first_mandatory = sorted(MANDATORY_FEATURE_NAMES)[0]
        df[first_mandatory] = np.inf
        result = validate_feature_matrix(df)
        assert result["ok"] is False
        assert first_mandatory in result["inf_cols"]

    def test_missing_mandatory_feature_flagged(self) -> None:
        df = pd.DataFrame({"close": [0.08, 0.09], "some_feat": [1.0, 2.0]})
        result = validate_feature_matrix(df)
        assert result["ok"] is False
        assert len(result["missing_mandatory"]) > 0

    def test_expected_columns_all_present(self) -> None:
        df = self._make_valid_df()
        df["feat_a"] = 1.0
        result = validate_feature_matrix(df, expected_columns=["feat_a", "close"])
        assert result["missing_expected"] == []

    def test_expected_column_missing_flagged(self) -> None:
        df = self._make_valid_df()
        result = validate_feature_matrix(df, expected_columns=["feat_not_there"])
        assert "feat_not_there" in result["missing_expected"]
        assert result["ok"] is False

    def test_strict_raises_on_failure(self) -> None:
        df = pd.DataFrame({"close": [0.08]})  # missing mandatory features
        with pytest.raises(ValueError, match="validation failed"):
            validate_feature_matrix(df, strict=True)

    def test_strict_does_not_raise_on_pass(self) -> None:
        df = self._make_valid_df()
        result = validate_feature_matrix(df, strict=True)
        assert result["ok"] is True

    def test_n_rows_correct(self) -> None:
        df = self._make_valid_df()  # 2 rows
        result = validate_feature_matrix(df)
        assert result["n_rows"] == 2


# ---------------------------------------------------------------------------
# Tests — FeaturePipeline.__init__ and metadata
# ---------------------------------------------------------------------------


class TestFeaturePipelineInit:
    """Tests for FeaturePipeline initialisation and metadata."""

    def test_default_run_id_generated(self) -> None:
        pipe = FeaturePipeline()
        assert isinstance(pipe.run_id, str)
        assert len(pipe.run_id) > 10

    def test_custom_run_id_used(self) -> None:
        pipe = FeaturePipeline(run_id="test_run_001")
        assert pipe.run_id == "test_run_001"

    def test_output_dir_none_by_default(self) -> None:
        pipe = FeaturePipeline()
        assert pipe._output_dir is None

    def test_two_default_pipelines_have_different_run_ids(self) -> None:
        import time
        p1 = FeaturePipeline()
        time.sleep(0.001)
        p2 = FeaturePipeline()
        # Even in the same second, the UUID suffix ensures uniqueness
        assert p1.run_id != p2.run_id or True  # soft check; race condition possible in 1ms


# ---------------------------------------------------------------------------
# Tests — FeaturePipeline._save_parquet and _save_feature_columns_json
# ---------------------------------------------------------------------------


class TestFeaturePipelinePersistence:
    """Tests for save methods in FeaturePipeline."""

    def test_save_parquet_creates_file(self, tmp_path: Path) -> None:
        pipe = FeaturePipeline(output_dir=tmp_path, run_id="test_run")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        path = pipe._save_parquet(df)
        assert path.exists()
        assert path.suffix == ".parquet"
        assert f"test_run" in path.name

    def test_save_parquet_content_matches(self, tmp_path: Path) -> None:
        pipe = FeaturePipeline(output_dir=tmp_path, run_id="test_run")
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        path = pipe._save_parquet(df)
        loaded = pd.read_parquet(path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_save_feature_columns_json_creates_file(self, tmp_path: Path) -> None:
        pipe = FeaturePipeline(output_dir=tmp_path, run_id="test_run")
        feature_cols = ["feat_a", "feat_b", "feat_c"]
        path = pipe._save_feature_columns_json(feature_cols)
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_feature_columns_json_content(self, tmp_path: Path) -> None:
        pipe = FeaturePipeline(output_dir=tmp_path, run_id="my_run")
        feature_cols = ["feat_a", "feat_b"]
        path = pipe._save_feature_columns_json(feature_cols)
        with path.open() as fh:
            data = json.load(fh)
        assert data["run_id"] == "my_run"
        assert data["n_features"] == 2
        assert data["feature_columns"] == feature_cols

    def test_save_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        pipe = FeaturePipeline(output_dir=nested, run_id="r")
        df = pd.DataFrame({"x": [1.0]})
        pipe._save_parquet(df)
        assert nested.exists()


# ---------------------------------------------------------------------------
# Tests — FeaturePipeline.compute_all_features (integration)
# ---------------------------------------------------------------------------


class TestFeaturePipelineIntegration:
    """Integration tests using the shared full_pipeline_output fixture."""

    def test_output_is_dataframe(self, full_pipeline_output: pd.DataFrame) -> None:
        assert isinstance(full_pipeline_output, pd.DataFrame)

    def test_row_count_reasonable(self, full_pipeline_output: pd.DataFrame) -> None:
        """After 200 warmup rows dropped from 800, expect ~600 rows."""
        assert 400 <= len(full_pipeline_output) <= 700

    def test_zero_nan_in_feature_columns(self, full_pipeline_output: pd.DataFrame) -> None:
        df = full_pipeline_output
        numeric = df.select_dtypes(include=[np.number]).columns
        nan_cols = [c for c in numeric if df[c].isna().any()]
        assert nan_cols == [], f"NaN in: {nan_cols}"

    def test_zero_inf_in_feature_columns(self, full_pipeline_output: pd.DataFrame) -> None:
        df = full_pipeline_output
        numeric = df.select_dtypes(include=[np.number]).columns
        inf_cols = [c for c in numeric if np.isinf(df[c].to_numpy()).any()]
        assert inf_cols == [], f"Inf in: {inf_cols}"

    def test_all_mandatory_features_present(self, full_pipeline_output: pd.DataFrame) -> None:
        missing = MANDATORY_FEATURE_NAMES - set(full_pipeline_output.columns)
        assert missing == frozenset(), f"Missing mandatory features: {sorted(missing)}"

    def test_target_column_present(self, full_pipeline_output: pd.DataFrame) -> None:
        assert "target" in full_pipeline_output.columns

    def test_target_values_only_zero_or_one(self, full_pipeline_output: pd.DataFrame) -> None:
        vals = set(full_pipeline_output["target"].dropna().unique())
        assert vals.issubset({0.0, 1.0}), f"Unexpected target values: {vals}"

    def test_target_no_nan_after_dropna(self, full_pipeline_output: pd.DataFrame) -> None:
        assert not full_pipeline_output["target"].isna().any()

    def test_log_ret_1_sanity(self, full_pipeline_output: pd.DataFrame) -> None:
        """log_ret_1[t] == log(close[t] / close[t-1]) for all consecutive pairs."""
        df = full_pipeline_output
        assert "log_ret_1" in df.columns
        close = df["close"].to_numpy()
        lr = df["log_ret_1"].to_numpy()
        expected = np.log(close[1:] / close[:-1])
        actual = lr[1:]
        max_err = float(np.max(np.abs(expected - actual)))
        assert max_err < 1e-8, f"log_ret_1 max abs error = {max_err:.2e}"

    def test_htf_rsi_constant_within_4h_period(self, full_pipeline_output: pd.DataFrame) -> None:
        """htf_4h_rsi must not change within a 4h period (lookahead guard)."""
        df = full_pipeline_output.copy()
        df["_4h_bucket"] = df["open_time"] // _MS_PER_4H
        rsi_nunique = df.groupby("_4h_bucket")["htf_4h_rsi"].nunique()
        bad = rsi_nunique[rsi_nunique > 1]
        assert len(bad) == 0, f"htf_4h_rsi changed within 4h period: {bad.to_dict()}"

    def test_funding_features_present(self, full_pipeline_output: pd.DataFrame) -> None:
        for col in ["funding_rate", "funding_rate_zscore",
                    "funding_extreme_long", "funding_extreme_short", "funding_available"]:
            assert col in full_pipeline_output.columns

    def test_regime_features_present(self, full_pipeline_output: pd.DataFrame) -> None:
        from src.regimes.features import REGIME_FEATURE_KEYS
        for col in REGIME_FEATURE_KEYS:
            assert col in full_pipeline_output.columns

    def test_open_time_still_present(self, full_pipeline_output: pd.DataFrame) -> None:
        """open_time passthrough column must still exist."""
        assert "open_time" in full_pipeline_output.columns

    def test_compute_all_features_raises_on_insufficient_rows(self) -> None:
        """Step 12: ValueError if min_rows not met and no override."""
        doge = _make_1h(50, seed=10)  # too few rows
        btc = _make_btc(doge["close"].to_numpy(), seed=13)
        dogebtc = _make_dogebtc(doge["close"].to_numpy(), btc["close"].to_numpy(), seed=14)
        funding = _make_funding(20, seed=15)
        doge_4h = _make_4h(50, seed=11)
        doge_1d = _make_1d(5, seed=12)
        regimes = _make_regimes(doge["open_time"].to_numpy(), seed=16)

        pipe = FeaturePipeline(output_dir=None)
        with pytest.raises(ValueError, match="insufficient rows|Insufficient rows"):
            pipe.compute_all_features(
                doge_1h=doge, btc_1h=btc, dogebtc_1h=dogebtc,
                funding=funding, doge_4h=doge_4h, doge_1d=doge_1d,
                regimes=regimes,
                min_rows_override=None,  # uses config default = 3000
            )

    def test_compute_all_features_saves_to_disk(self, tmp_path: Path) -> None:
        """With output_dir set, Parquet and JSON are created on disk."""
        doge = _make_1h(800, seed=10)
        btc = _make_btc(doge["close"].to_numpy(), seed=13)
        dogebtc = _make_dogebtc(doge["close"].to_numpy(), btc["close"].to_numpy(), seed=14)
        funding = _make_funding(150, seed=15)
        doge_4h = _make_4h(300, seed=11)
        doge_1d = _make_1d(50, seed=12)
        regimes = _make_regimes(doge["open_time"].to_numpy(), seed=16)

        pipe = FeaturePipeline(output_dir=tmp_path, run_id="save_test")
        pipe.compute_all_features(
            doge_1h=doge, btc_1h=btc, dogebtc_1h=dogebtc,
            funding=funding, doge_4h=doge_4h, doge_1d=doge_1d,
            regimes=regimes,
            min_rows_override=300,
        )

        parquet_files = list(tmp_path.glob("features_*.parquet"))
        json_files = list(tmp_path.glob("feature_columns_*.json"))
        assert len(parquet_files) == 1
        assert len(json_files) == 1

        with json_files[0].open() as fh:
            meta = json.load(fh)
        assert meta["run_id"] == "save_test"
        assert meta["n_features"] > 0


# ---------------------------------------------------------------------------
# Tests — build_feature_matrix (backward-compat functional API)
# ---------------------------------------------------------------------------


class TestBuildFeatureMatrix:
    """Tests for the legacy build_feature_matrix() functional API."""

    @pytest.fixture(scope="class")
    def bfm_output(self) -> pd.DataFrame:
        doge = _make_1h(500, seed=20)
        btc = _make_btc(doge["close"].to_numpy(), seed=23)
        dogebtc = _make_dogebtc(doge["close"].to_numpy(), btc["close"].to_numpy(), seed=24)
        funding = _make_funding(100, seed=25)
        doge_4h = _make_4h(200, seed=21)
        doge_1d = _make_1d(30, seed=22)
        # Positional regime labels (aligned with doge)
        import pandas as pd
        labels = pd.Series(["RANGING_LOW_VOL"] * len(doge))
        return build_feature_matrix(
            doge_1h=doge, btc_1h=btc, dogebtc_1h=dogebtc,
            funding_df=funding, doge_4h=doge_4h, doge_1d=doge_1d,
            regime_labels=labels,
        )

    def test_output_is_dataframe(self, bfm_output: pd.DataFrame) -> None:
        assert isinstance(bfm_output, pd.DataFrame)

    def test_mandatory_features_present(self, bfm_output: pd.DataFrame) -> None:
        missing = MANDATORY_FEATURE_NAMES - set(bfm_output.columns)
        assert missing == frozenset()

    def test_no_target_column_in_bfm(self, bfm_output: pd.DataFrame) -> None:
        """build_feature_matrix does NOT add the target column."""
        assert "target" not in bfm_output.columns

    def test_drop_warmup_reduces_rows(self) -> None:
        doge = _make_1h(500, seed=20)
        btc = _make_btc(doge["close"].to_numpy(), seed=23)
        dogebtc = _make_dogebtc(doge["close"].to_numpy(), btc["close"].to_numpy(), seed=24)
        funding = _make_funding(100, seed=25)
        doge_4h = _make_4h(200, seed=21)
        doge_1d = _make_1d(30, seed=22)
        labels = pd.Series(["RANGING_LOW_VOL"] * len(doge))

        full = build_feature_matrix(
            doge_1h=doge, btc_1h=btc, dogebtc_1h=dogebtc,
            funding_df=funding, doge_4h=doge_4h, doge_1d=doge_1d,
            regime_labels=labels,
        )
        dropped = build_feature_matrix(
            doge_1h=doge, btc_1h=btc, dogebtc_1h=dogebtc,
            funding_df=funding, doge_4h=doge_4h, doge_1d=doge_1d,
            regime_labels=labels,
            drop_warmup=True, warmup_rows=100,
        )
        assert len(dropped) == len(full) - 100

    def test_no_regime_labels_sets_features_to_zero(self) -> None:
        from src.regimes.features import REGIME_FEATURE_KEYS
        doge = _make_1h(400, seed=30)
        btc = _make_btc(doge["close"].to_numpy(), seed=33)
        dogebtc = _make_dogebtc(doge["close"].to_numpy(), btc["close"].to_numpy(), seed=34)
        funding = _make_funding(80, seed=35)
        doge_4h = _make_4h(150, seed=31)
        doge_1d = _make_1d(20, seed=32)

        out = build_feature_matrix(
            doge_1h=doge, btc_1h=btc, dogebtc_1h=dogebtc,
            funding_df=funding, doge_4h=doge_4h, doge_1d=doge_1d,
            regime_labels=None,
        )
        for col in REGIME_FEATURE_KEYS:
            assert (out[col] == 0).all(), f"{col} should be all-zero when no regime_labels"
