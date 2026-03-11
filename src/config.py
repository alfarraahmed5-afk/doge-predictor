"""Configuration module for doge_predictor.

This is the ONLY place configuration is loaded. All other modules import
Settings instances from here. Configuration is loaded once at startup and
never re-read mid-pipeline.

Usage::

    from src.config import settings, doge_settings, regime_config, rl_config

All YAML files are read from the ``config/`` directory relative to the project
root (the directory that contains this file's parent package).
"""

from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------

# src/ lives one level below the project root.
_SRC_DIR: Path = Path(__file__).parent
_PROJECT_ROOT: Path = _SRC_DIR.parent
_CONFIG_DIR: Path = _PROJECT_ROOT / "config"


def _load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML file from the config/ directory.

    Args:
        filename: Name of the YAML file (e.g. ``"settings.yaml"``).

    Returns:
        Parsed YAML contents as a plain Python dict.

    Raises:
        FileNotFoundError: If the file does not exist at the expected path.
    """
    path = _CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Sub-models — settings.yaml
# ---------------------------------------------------------------------------


class ProjectSettings(BaseModel):
    """Top-level project metadata."""

    name: str = "doge_predictor"
    version: str = "1.0.0"
    seed: int = 42
    log_level: str = "INFO"


class DatabaseSettings(BaseModel):
    """TimescaleDB / PostgreSQL connection parameters.

    Sensitive credentials are read from environment variables, not from the
    YAML file directly.
    """

    host: str = "localhost"
    port: int = 5432
    name: str = "doge_predictor"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 5
    pool_timeout: int = 30

    @property
    def url(self) -> str:
        """SQLAlchemy connection URL."""
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class MLflowSettings(BaseModel):
    """MLflow experiment tracking configuration."""

    tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    experiment_name: str = "doge_predictor"


class PathSettings(BaseModel):
    """Canonical filesystem paths (all relative to project root)."""

    data_root: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    features_dir: Path = Path("data/features")
    checkpoints_dir: Path = Path("data/checkpoints")
    regimes_dir: Path = Path("data/regimes")
    predictions_dir: Path = Path("data/predictions")
    replay_buffers_dir: Path = Path("data/replay_buffers")
    models_dir: Path = Path("mlruns/models")

    def resolve(self, root: Path = _PROJECT_ROOT) -> "PathSettings":
        """Return a copy with all paths resolved to absolute paths.

        Args:
            root: Project root to resolve relative paths against.

        Returns:
            New PathSettings with absolute Path objects.
        """
        return PathSettings(
            data_root=root / self.data_root,
            raw_dir=root / self.raw_dir,
            processed_dir=root / self.processed_dir,
            features_dir=root / self.features_dir,
            checkpoints_dir=root / self.checkpoints_dir,
            regimes_dir=root / self.regimes_dir,
            predictions_dir=root / self.predictions_dir,
            replay_buffers_dir=root / self.replay_buffers_dir,
            models_dir=root / self.models_dir,
        )


class LoggingSettings(BaseModel):
    """Loguru logging configuration."""

    format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
    )
    rotation: str = "100 MB"
    retention: str = "30 days"
    log_dir: Path = Path("logs")


# ---------------------------------------------------------------------------
# Global settings model
# ---------------------------------------------------------------------------


class Settings(BaseModel):
    """Root configuration model loaded from config/settings.yaml.

    This is the only source of truth for global parameters.
    """

    project: ProjectSettings = Field(default_factory=ProjectSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


# ---------------------------------------------------------------------------
# Sub-models — doge_settings.yaml
# ---------------------------------------------------------------------------


class IndicatorSettings(BaseModel):
    """Technical indicator periods loaded from config/doge_settings.yaml.

    All feature modules read their period constants from this model so that
    no magic numbers appear in ``src/``.
    """

    sma_periods: list[int] = Field(default_factory=lambda: [7, 21, 50, 200])
    ema_periods: list[int] = Field(default_factory=lambda: [7, 14, 21, 50, 200])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    bb_squeeze_threshold: float = 0.04
    atr_period: int = 14
    stoch_fastk: int = 14
    stoch_slowk: int = 3
    stoch_slowd: int = 3
    ichimoku_tenkan: int = 9
    ichimoku_kijun: int = 26
    ichimoku_senkou_b: int = 52
    ichimoku_displacement: int = 26
    volume_ma_period: int = 20
    cmf_period: int = 20
    obv_ema_span: int = 20
    rolling_vol_windows: list[int] = Field(default_factory=lambda: [6, 12, 24, 48, 168])
    rolling_skew_period: int = 24
    rolling_kurt_period: int = 24
    log_return_periods: list[int] = Field(
        default_factory=lambda: [1, 3, 6, 12, 24, 48, 168]
    )
    momentum_periods: list[int] = Field(default_factory=lambda: [6, 12, 24, 48])


class WalkForwardSettings(BaseModel):
    """Walk-forward cross-validation parameters."""

    training_window_days: int = 180
    validation_window_days: int = 30
    step_size_days: int = 7
    min_training_rows: int = 3000


class RiskSettings(BaseModel):
    """Position sizing and fee parameters."""

    base_risk_pct: float = 0.01
    reduced_risk_pct: float = 0.005
    taker_fee: float = 0.001
    slippage_min: float = 0.0002
    slippage_max: float = 0.0008
    round_number_size_reduction: float = 0.30
    max_drawdown_halt: float = 0.25


class AcceptanceGates(BaseModel):
    """Minimum performance thresholds that must all pass before deployment."""

    directional_accuracy_oos: float = 0.54
    sharpe_annualized: float = 1.0
    sharpe_per_regime: float = 0.8
    max_drawdown: float = 0.20
    calmar_ratio: float = 0.6
    profit_factor: float = 1.3
    win_rate: float = 0.45
    min_trade_count: int = 150
    decoupled_max_drawdown: float = 0.15


class BacktestSettings(BaseModel):
    """Backtesting execution parameters."""

    fill_price: str = "next_open"
    taker_fee: float = 0.001
    slippage_min: float = 0.0002
    slippage_max: float = 0.0008
    position_size_pct: float = 0.01
    reduced_position_size_pct: float = 0.005
    max_drawdown_halt: float = 0.25


class DogeSettings(BaseModel):
    """DOGE-specific configuration loaded from config/doge_settings.yaml."""

    symbol: str = "DOGEUSDT"
    secondary_symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT", "DOGEBTC"])
    primary_interval: str = "1h"
    secondary_intervals: list[str] = Field(default_factory=lambda: ["4h", "1d"])
    training_start_date: str = "2022-01-01"
    context_start_date: str = "2019-07-01"
    indicators: IndicatorSettings = Field(default_factory=IndicatorSettings)
    walk_forward: WalkForwardSettings = Field(default_factory=WalkForwardSettings)
    round_number_levels: list[float] = Field(
        default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.00]
    )
    round_number_proximity_pct: float = 0.005
    risk: RiskSettings = Field(default_factory=RiskSettings)
    default_confidence_threshold: float = 0.62
    btc_crash_threshold: float = -0.04
    volume_spike_threshold: float = 3.0
    funding_rate_extreme_long: float = 0.001
    funding_rate_extreme_short: float = -0.0005
    correlation_windows: list[int] = Field(default_factory=lambda: [12, 24, 168])
    dogebtc_momentum_windows: list[int] = Field(default_factory=lambda: [6, 24, 48])
    volume_rolling_window: int = 20
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    acceptance_gates: AcceptanceGates = Field(default_factory=AcceptanceGates)
    freshness_check_multiplier: int = 2
    inference_lookback_candles: int = 500


# ---------------------------------------------------------------------------
# Sub-models — regime_config.yaml
# ---------------------------------------------------------------------------


class RegimeDefinition(BaseModel):
    """Configuration for a single market regime."""

    confidence_threshold: float
    position_size_multiplier: float
    description: str = ""


class RegimeThresholds(BaseModel):
    """Numeric thresholds used by DogeRegimeClassifier."""

    btc_corr_decoupled: float = 0.30
    bb_width_low: float = 0.04
    bb_width_high: float = 0.06
    atr_low: float = 0.003
    atr_high: float = 0.005
    roll7d_bull: float = 0.05
    roll7d_bear: float = -0.05


class RegimeConfig(BaseModel):
    """Full regime configuration loaded from config/regime_config.yaml."""

    thresholds: RegimeThresholds = Field(default_factory=RegimeThresholds)
    regimes: dict[str, RegimeDefinition] = Field(default_factory=dict)
    precedence: list[str] = Field(
        default_factory=lambda: [
            "DECOUPLED",
            "TRENDING_BULL",
            "TRENDING_BEAR",
            "RANGING_HIGH_VOL",
            "RANGING_LOW_VOL",
        ]
    )
    min_history_candles: int = 200

    def get_confidence_threshold(self, regime_label: str) -> float:
        """Return the confidence threshold for the given regime label.

        Args:
            regime_label: One of the five regime keys (e.g. ``"TRENDING_BULL"``).

        Returns:
            Confidence threshold float.

        Raises:
            KeyError: If regime_label is not a recognised regime.
        """
        return self.regimes[regime_label].confidence_threshold

    def get_position_size_multiplier(self, regime_label: str) -> float:
        """Return the position size multiplier for the given regime label.

        Args:
            regime_label: One of the five regime keys.

        Returns:
            Position size multiplier float (1.0 = full size, 0.5 = half).

        Raises:
            KeyError: If regime_label is not a recognised regime.
        """
        return self.regimes[regime_label].position_size_multiplier


# ---------------------------------------------------------------------------
# Sub-models — rl_config.yaml
# ---------------------------------------------------------------------------


class HorizonSettings(BaseModel):
    """Configuration for a single prediction horizon."""

    candles: int
    reward_weight: float
    punish_weight: float
    decay_constant: float
    label: str


class RewardSettings(BaseModel):
    """Reward function parameters."""

    direction_correct: float = 1.0
    direction_wrong: float = -1.0
    direction_flat: float = 0.1
    calibration_correct_min: float = 1.0
    calibration_correct_max: float = 2.0
    calibration_wrong_min: float = -1.0
    calibration_wrong_max: float = -3.0
    decoupled_decay_multiplier: float = 0.5


class ReplayBufferSettings(BaseModel):
    """Replay buffer configuration."""

    max_size_per_horizon: int = 10000
    priority_threshold: float = 2.0
    priority_oversample: int = 3
    min_per_regime: int = 100
    min_samples_to_train: int = 500
    checkpoint_interval_hours: int = 6


class PredictionStoreSettings(BaseModel):
    """TimescaleDB prediction store settings."""

    table_name: str = "doge_predictions"
    replay_table_name: str = "doge_replay_buffer"
    retention_days: int = 365


class CurriculumStageAdvancement(BaseModel):
    """Advancement criteria for a curriculum stage."""

    min_days: int
    min_accuracy: float
    min_mean_reward: float


class CurriculumStage(BaseModel):
    """A single curriculum stage definition."""

    label: str
    horizons: list[str]
    advancement_criteria: CurriculumStageAdvancement | None = None


class CurriculumSettings(BaseModel):
    """Curriculum manager configuration."""

    stages: dict[int, CurriculumStage]
    starting_stage: int = 1


class SelfTrainingSchedule(BaseModel):
    """Scheduled self-training trigger."""

    day_of_week: str = "sunday"
    hour_utc: int = 2


class SelfTrainingTriggers(BaseModel):
    """Event-driven self-training triggers."""

    buffer_fill_pct: float = 0.80
    rolling_7d_mean_reward: float = 0.0
    new_regime_predictions: int = 50


class SelfTrainingSettings(BaseModel):
    """Self-training loop configuration."""

    schedule: SelfTrainingSchedule = Field(default_factory=SelfTrainingSchedule)
    triggers: SelfTrainingTriggers = Field(default_factory=SelfTrainingTriggers)
    min_cooldown_hours: int = 48
    min_batch_size: int = 32
    max_batch_size: int = 256


class RLConfig(BaseModel):
    """Full RL self-teaching configuration loaded from config/rl_config.yaml."""

    horizons: dict[str, HorizonSettings] = Field(default_factory=dict)
    reward: RewardSettings = Field(default_factory=RewardSettings)
    replay_buffer: ReplayBufferSettings = Field(default_factory=ReplayBufferSettings)
    prediction_store: PredictionStoreSettings = Field(
        default_factory=PredictionStoreSettings
    )
    curriculum: CurriculumSettings
    self_training: SelfTrainingSettings = Field(default_factory=SelfTrainingSettings)


# ---------------------------------------------------------------------------
# Loader functions — parse YAML into typed Pydantic models
# ---------------------------------------------------------------------------


def _build_settings() -> Settings:
    """Load and validate config/settings.yaml, overriding DB credentials from env.

    Returns:
        Validated :class:`Settings` instance.
    """
    raw = _load_yaml("settings.yaml")

    # Override database credentials from environment variables if present
    db_section = raw.get("database", {})
    db_section["host"] = os.getenv("DB_HOST", db_section.get("host", "localhost"))
    db_section["port"] = int(os.getenv("DB_PORT", str(db_section.get("port", 5432))))
    db_section["name"] = os.getenv("DB_NAME", db_section.get("name", "doge_predictor"))
    db_section["user"] = os.getenv("DB_USER", db_section.get("user", "postgres"))
    db_section["password"] = os.getenv("DB_PASSWORD", db_section.get("password", ""))
    raw["database"] = db_section

    return Settings.model_validate(raw)


def _build_doge_settings() -> DogeSettings:
    """Load and validate config/doge_settings.yaml.

    Returns:
        Validated :class:`DogeSettings` instance.
    """
    raw = _load_yaml("doge_settings.yaml")
    return DogeSettings.model_validate(raw)


def _build_regime_config() -> RegimeConfig:
    """Load and validate config/regime_config.yaml.

    Returns:
        Validated :class:`RegimeConfig` instance.
    """
    raw = _load_yaml("regime_config.yaml")

    # Convert nested regime dicts to RegimeDefinition instances
    if "regimes" in raw:
        raw["regimes"] = {
            label: RegimeDefinition.model_validate(defn)
            for label, defn in raw["regimes"].items()
        }

    return RegimeConfig.model_validate(raw)


def _build_rl_config() -> RLConfig:
    """Load and validate config/rl_config.yaml.

    Returns:
        Validated :class:`RLConfig` instance.
    """
    raw = _load_yaml("rl_config.yaml")

    # Convert horizon dicts
    if "horizons" in raw:
        raw["horizons"] = {
            label: HorizonSettings.model_validate(h)
            for label, h in raw["horizons"].items()
        }

    # Convert curriculum stages
    if "curriculum" in raw and "stages" in raw["curriculum"]:
        stages_raw = raw["curriculum"]["stages"]
        stages: dict[int, CurriculumStage] = {}
        for stage_num, stage_data in stages_raw.items():
            advancement_raw = stage_data.get("advancement_criteria")
            if advancement_raw is not None:
                stage_data["advancement_criteria"] = (
                    CurriculumStageAdvancement.model_validate(advancement_raw)
                )
            stages[int(stage_num)] = CurriculumStage.model_validate(stage_data)
        raw["curriculum"]["stages"] = stages

    return RLConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Seed application
# ---------------------------------------------------------------------------


def _apply_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value (read from config, not hardcoded).
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    try:
        import torch  # noqa: PLC0415
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch is optional at import time (not available in some envs)


# ---------------------------------------------------------------------------
# Module-level singletons — load once, import everywhere
# ---------------------------------------------------------------------------

settings: Settings = _build_settings()
doge_settings: DogeSettings = _build_doge_settings()
regime_config: RegimeConfig = _build_regime_config()
rl_config: RLConfig = _build_rl_config()

# Apply global seed immediately after loading config
_apply_global_seed(settings.project.seed)

# Resolve paths to absolute (done after singleton construction)
settings.paths = settings.paths.resolve(_PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Public accessor — returns the pre-built singleton (useful for DI / testing)
# ---------------------------------------------------------------------------


def get_settings() -> Settings:
    """Return the global :class:`Settings` singleton.

    The singleton is built once at module import time. This function is a
    thin accessor that enables dependency-injection patterns and is used by
    the Phase 1 quality gate smoke test.

    Returns:
        The application-wide :class:`Settings` instance.
    """
    return settings


__all__ = [
    "Settings",
    "DogeSettings",
    "IndicatorSettings",
    "RegimeConfig",
    "RLConfig",
    "settings",
    "doge_settings",
    "regime_config",
    "rl_config",
    "get_settings",
    "_PROJECT_ROOT",
    "_CONFIG_DIR",
]
