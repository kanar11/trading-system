"""Declarative, validated run configuration.

A single typed object (:class:`RunConfig`) captures an entire backtest run:
data source, strategy selection and parameters, costs, risk controls, and
walk-forward settings. It can be built from defaults, loaded from a YAML file,
or overridden via ``QUANTBT_*`` environment variables, and it converts cleanly
into the dataclass configs used by the lower-level modules.

This is the reproducibility backbone: a run is fully described by its config,
which can be serialised back to YAML and stored next to the results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from quantbt.regime.detector import RegimeConfig
from quantbt.risk.manager import RiskConfig
from quantbt.validation.walk_forward import WalkForwardConfig

StrategyName = Literal["momentum", "mean-reversion", "adaptive"]


class _Model(BaseModel):
    """Base for nested config models: reject unknown keys, validate on assign."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DataParams(_Model):
    """Where price data comes from."""

    ticker: str = "SPY"
    start: str = "2010-01-01"
    end: str | None = None


class MomentumParams(_Model):
    """Momentum strategy parameters."""

    lookback: int = Field(default=200, ge=1)
    threshold: float = Field(default=0.005, ge=0.0)
    use_sma_filter: bool = True


class MeanReversionParams(_Model):
    """Mean reversion (Bollinger + RSI) parameters."""

    bb_window: int = Field(default=20, ge=1)
    bb_std: float = Field(default=2.0, gt=0.0)
    rsi_period: int = Field(default=14, ge=1)
    rsi_oversold: float = Field(default=30.0, ge=0.0, le=100.0)
    rsi_overbought: float = Field(default=70.0, ge=0.0, le=100.0)
    use_rsi_filter: bool = True

    @model_validator(mode="after")
    def _check_rsi_levels(self) -> MeanReversionParams:
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError(
                f"rsi_oversold ({self.rsi_oversold}) must be below "
                f"rsi_overbought ({self.rsi_overbought})."
            )
        return self


class RegimeParams(_Model):
    """Regime-detection parameters for the adaptive strategy."""

    adx_period: int = Field(default=14, ge=1)
    adx_trending_threshold: float = Field(default=25.0, ge=0.0)
    adx_weak_threshold: float = Field(default=20.0, ge=0.0)
    hurst_window: int = Field(default=100, ge=2)
    hurst_trending_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    hurst_mr_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    vol_window: int = Field(default=20, ge=1)
    vol_percentile: float = Field(default=75.0, ge=0.0, le=100.0)
    smoothing_window: int = Field(default=5, ge=1)

    def to_dataclass(self) -> RegimeConfig:
        return RegimeConfig(**self.model_dump())


class BacktestParams(_Model):
    """Backtest engine parameters (costs and volatility targeting)."""

    transaction_cost: float = Field(default=0.001, ge=0.0)
    vol_target: float | None = Field(default=0.15, ge=0.0)
    vol_window: int = Field(default=20, ge=1)


class RiskParams(_Model):
    """Risk-control parameters. Set ``enabled=False`` to disable all controls."""

    enabled: bool = True
    stop_loss: float | None = Field(default=0.05, ge=0.0)
    take_profit: float | None = Field(default=0.10, ge=0.0)
    trailing_stop: float | None = Field(default=0.03, ge=0.0)
    max_position: float = Field(default=1.0, gt=0.0)
    daily_loss_limit: float | None = Field(default=0.02, ge=0.0)

    def to_dataclass(self) -> RiskConfig | None:
        """Return a RiskConfig, or None when risk controls are disabled."""
        if not self.enabled:
            return None
        return RiskConfig(
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            trailing_stop=self.trailing_stop,
            max_position=self.max_position,
            daily_loss_limit=self.daily_loss_limit,
        )


class WalkForwardParams(_Model):
    """Walk-forward validation parameters. ``enabled`` switches the run mode."""

    enabled: bool = False
    in_sample_days: int = Field(default=504, ge=1)
    out_of_sample_days: int = Field(default=126, ge=1)
    step_days: int | None = Field(default=None, ge=1)

    def to_dataclass(self) -> WalkForwardConfig:
        return WalkForwardConfig(
            in_sample_days=self.in_sample_days,
            out_of_sample_days=self.out_of_sample_days,
            step_days=self.step_days,
        )


class OutputParams(_Model):
    """Where results go and how chatty the run is."""

    output_dir: str = "results"
    verbose: bool = False


class RunConfig(BaseSettings):
    """Top-level configuration for a single backtest run.

    Resolution order (lowest to highest precedence): model defaults, values
    from a YAML file (via :meth:`from_yaml`), then ``QUANTBT_*`` environment
    variables (nested with ``__``, e.g. ``QUANTBT_DATA__TICKER=AAPL``).
    """

    model_config = SettingsConfigDict(
        env_prefix="QUANTBT_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    strategy: StrategyName = "momentum"
    data: DataParams = Field(default_factory=DataParams)
    momentum: MomentumParams = Field(default_factory=MomentumParams)
    mean_reversion: MeanReversionParams = Field(default_factory=MeanReversionParams)
    regime: RegimeParams = Field(default_factory=RegimeParams)
    backtest: BacktestParams = Field(default_factory=BacktestParams)
    risk: RiskParams = Field(default_factory=RiskParams)
    walk_forward: WalkForwardParams = Field(default_factory=WalkForwardParams)
    output: OutputParams = Field(default_factory=OutputParams)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> RunConfig:
        """Load config from a YAML file, with optional top-level overrides.

        Fields present in the file (or in ``overrides``) take precedence; any
        field left unset falls back to a ``QUANTBT_*`` environment variable and
        then to the model default.
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config file {path} must contain a YAML mapping.")
        return cls(**{**raw, **overrides})

    def to_yaml(self, path: str | Path) -> None:
        """Serialise the resolved config to YAML (for reproducibility)."""
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )

    def strategy_label(self) -> str:
        """Human-readable strategy name (e.g. 'Mean Reversion')."""
        return self.strategy.replace("-", " ").title()
