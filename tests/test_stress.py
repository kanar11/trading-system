"""Tests for scenario stress testing."""

import numpy as np
import pandas as pd
import pytest

from src.risk import factor_scenario_pnl, scenario_pnl


def test_asset_scenario_pnl_by_hand() -> None:
    weights = pd.Series({"aaa": 0.6, "bbb": 0.4})
    scenarios = pd.DataFrame({"aaa": [-0.10, 0.05], "bbb": [-0.20, 0.0]}, index=["crash", "rally"])
    pnl = scenario_pnl(weights, scenarios)
    assert pnl.name == "pnl"
    assert list(pnl.index) == ["crash", "rally"]
    assert pnl["crash"] == pytest.approx(0.6 * -0.10 + 0.4 * -0.20)  # -14%
    assert pnl["rally"] == pytest.approx(0.03)
    assert pnl.idxmin() == "crash"


def test_hedged_book_is_immune_to_common_shocks() -> None:
    weights = pd.Series({"aaa": 0.5, "bbb": -0.5})
    scenarios = pd.DataFrame({"aaa": [-0.15], "bbb": [-0.15]}, index=["gap"])
    assert scenario_pnl(weights, scenarios)["gap"] == pytest.approx(0.0)


def test_short_book_profits_from_a_crash() -> None:
    weights = pd.Series({"aaa": -1.0})
    scenarios = pd.DataFrame({"aaa": [-0.2]}, index=["crash"])
    assert scenario_pnl(weights, scenarios)["crash"] == pytest.approx(0.2)


def test_factor_scenarios_propagate_through_betas() -> None:
    weights = pd.Series({"aaa": 0.6, "bbb": 0.4})
    betas = pd.DataFrame({"mkt": [1.2, 0.8], "rates": [0.1, -0.3]}, index=["aaa", "bbb"])
    shocks = pd.DataFrame({"mkt": [-0.10], "rates": [0.02]}, index=["risk_off"])
    pnl = factor_scenario_pnl(weights, betas, shocks)
    exposure_mkt = 0.6 * 1.2 + 0.4 * 0.8
    exposure_rates = 0.6 * 0.1 + 0.4 * -0.3
    assert pnl["risk_off"] == pytest.approx(exposure_mkt * -0.10 + exposure_rates * 0.02)


def test_factor_version_matches_asset_version_via_b() -> None:
    weights = pd.Series({"aaa": 0.7, "bbb": 0.3})
    betas = pd.DataFrame({"mkt": [1.0, 0.5]}, index=["aaa", "bbb"])
    factor_shocks = pd.DataFrame({"mkt": [-0.08, 0.04]}, index=["down", "up"])
    via_factors = factor_scenario_pnl(weights, betas, factor_shocks)
    # equivalent per-asset shocks: shock_a = beta_a * f
    asset_shocks = pd.DataFrame(
        factor_shocks["mkt"].to_numpy()[:, None] * betas["mkt"].to_numpy()[None, :],
        index=factor_shocks.index,
        columns=betas.index,
    )
    via_assets = scenario_pnl(weights, asset_shocks)
    assert np.allclose(via_factors.to_numpy(), via_assets.to_numpy())


def test_beta_neutral_book_ignores_factor_shocks() -> None:
    weights = pd.Series({"aaa": 0.5, "bbb": -0.5})
    betas = pd.DataFrame({"mkt": [1.0, 1.0]}, index=["aaa", "bbb"])
    shocks = pd.DataFrame({"mkt": [-0.25]}, index=["meltdown"])
    assert factor_scenario_pnl(weights, betas, shocks)["meltdown"] == pytest.approx(0.0)


def test_factor_column_order_is_irrelevant() -> None:
    weights = pd.Series({"aaa": 1.0})
    betas = pd.DataFrame({"mkt": [1.0], "rates": [0.5]}, index=["aaa"])
    shocks_a = pd.DataFrame({"mkt": [-0.1], "rates": [0.02]}, index=["s"])
    shocks_b = shocks_a[["rates", "mkt"]]
    a = factor_scenario_pnl(weights, betas, shocks_a)
    b = factor_scenario_pnl(weights, betas, shocks_b)
    assert a["s"] == pytest.approx(b["s"])


def test_bad_inputs_raise() -> None:
    weights = pd.Series({"aaa": 1.0})
    scenarios = pd.DataFrame({"bbb": [-0.1]}, index=["s"])
    with pytest.raises(ValueError, match="missing shock columns"):
        scenario_pnl(weights, scenarios)
    with pytest.raises(ValueError, match="must not be empty"):
        scenario_pnl(weights, pd.DataFrame())
    with pytest.raises(ValueError, match="finite"):
        scenario_pnl(pd.Series({"aaa": np.nan}), pd.DataFrame({"aaa": [-0.1]}, index=["s"]))
    betas = pd.DataFrame({"mkt": [1.0]}, index=["aaa"])
    with pytest.raises(ValueError, match="betas missing"):
        factor_scenario_pnl(
            pd.Series({"zzz": 1.0}), betas, pd.DataFrame({"mkt": [-0.1]}, index=["s"])
        )
    with pytest.raises(ValueError, match="match the beta columns"):
        factor_scenario_pnl(weights, betas, pd.DataFrame({"x": [-0.1]}, index=["s"]))
