"""Tests for CLI argument parsing and config building."""

from pathlib import Path

from quantbt.cli import build_config, parse_args


def test_parse_args_defaults_are_none() -> None:
    args = parse_args([])
    assert args.ticker is None
    assert args.strategy is None
    assert args.no_risk is None
    assert args.walk_forward is None


def test_build_config_defaults() -> None:
    cfg = build_config(parse_args([]))
    assert cfg.strategy == "momentum"
    assert cfg.data.ticker == "SPY"
    assert cfg.risk.enabled is True


def test_build_config_overrides() -> None:
    args = parse_args(
        ["--strategy", "mean-reversion", "--ticker", "AAPL", "--no-risk", "--lookback", "50"]
    )
    cfg = build_config(args)
    assert cfg.strategy == "mean-reversion"
    assert cfg.data.ticker == "AAPL"
    assert cfg.risk.enabled is False
    assert cfg.momentum.lookback == 50


def test_build_config_walk_forward_flag() -> None:
    cfg = build_config(parse_args(["--walk-forward", "--wf-is-days", "300"]))
    assert cfg.walk_forward.enabled is True
    assert cfg.walk_forward.in_sample_days == 300


def test_build_config_yaml_base_with_override(tmp_path: Path) -> None:
    yaml_path = tmp_path / "base.yaml"
    yaml_path.write_text(
        "strategy: adaptive\ndata:\n  ticker: QQQ\n  start: '2015-01-01'\n",
        encoding="utf-8",
    )
    # CLI flag overrides the file's ticker; file's start is preserved.
    cfg = build_config(parse_args(["--config", str(yaml_path), "--ticker", "IWM"]))
    assert cfg.strategy == "adaptive"
    assert cfg.data.ticker == "IWM"
    assert cfg.data.start == "2015-01-01"
