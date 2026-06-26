"""Tests for the declarative RunConfig layer."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from quantbt.config import RunConfig
from quantbt.regime.detector import RegimeConfig
from quantbt.risk.manager import RiskConfig
from quantbt.validation.walk_forward import WalkForwardConfig


def test_defaults_construct() -> None:
    cfg = RunConfig()
    assert cfg.strategy == "momentum"
    assert cfg.data.ticker == "SPY"
    assert cfg.risk.enabled is True
    assert cfg.backtest.vol_target == 0.15


def test_strategy_label() -> None:
    assert RunConfig(strategy="mean-reversion").strategy_label() == "Mean Reversion"


def test_extra_keys_forbidden() -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"data": {"ticker": "AAPL", "bogus": 1}})


def test_rsi_levels_validated() -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"mean_reversion": {"rsi_oversold": 80, "rsi_overbought": 20}})


def test_negative_lookback_rejected() -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"momentum": {"lookback": 0}})


def test_risk_to_dataclass_enabled() -> None:
    rc = RunConfig().risk.to_dataclass()
    assert isinstance(rc, RiskConfig)
    assert rc.stop_loss == 0.05


def test_risk_to_dataclass_disabled() -> None:
    cfg = RunConfig.model_validate({"risk": {"enabled": False}})
    assert cfg.risk.to_dataclass() is None


def test_regime_and_walk_forward_to_dataclass() -> None:
    cfg = RunConfig()
    assert isinstance(cfg.regime.to_dataclass(), RegimeConfig)
    wf = cfg.walk_forward.to_dataclass()
    assert isinstance(wf, WalkForwardConfig)
    assert wf.in_sample_days == 504


def test_yaml_round_trip(tmp_path: Path) -> None:
    cfg = RunConfig.model_validate({"strategy": "adaptive", "data": {"ticker": "QQQ"}})
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    loaded = RunConfig.from_yaml(path)
    assert loaded.strategy == "adaptive"
    assert loaded.data.ticker == "QQQ"


def test_from_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        RunConfig.from_yaml(path)


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANTBT_DATA__TICKER", "TSLA")
    monkeypatch.setenv("QUANTBT_STRATEGY", "mean-reversion")
    cfg = RunConfig()
    assert cfg.data.ticker == "TSLA"
    assert cfg.strategy == "mean-reversion"
