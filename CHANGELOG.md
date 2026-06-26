# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Declarative, validated run configuration (`quantbt.config.RunConfig`, pydantic):
  defaults < `QUANTBT_*` env vars < YAML file < CLI flags. Serialised to
  `results/run_config.yaml` for reproducibility.
- Strategy registry (`quantbt.strategy.registry`): `@register` decorator and
  `build_strategy` dispatch; CLI strategy choices derive from the registry.
- Console entry point `quantbt-backtest` (pip-installable package).
- GitHub Actions CI (ruff + mypy + pytest/coverage on a 3.11/3.12 matrix),
  pre-commit hooks, and branch coverage with an 85% floor.
- `configs/example.yaml` sample configuration.

### Changed
- Renamed the importable package `src` → `quantbt`; the project is now an
  installable package with a `[build-system]`.
- `main.py` reduced to a thin shim over `quantbt.cli:main`.
- `.gitattributes` added for LF normalisation.

## [0.4.0] - earlier

### Added
- Momentum, mean-reversion, and adaptive (regime-based) strategies.
- Cost-aware backtest engine with volatility targeting.
- Risk management layer (stop-loss, take-profit, trailing stop, position and
  daily-loss limits).
- Walk-forward validation, regime detection (ADX + Hurst), trade-level
  analytics, parameter sweep, and performance metrics.
