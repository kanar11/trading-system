"""Tests for quoted/effective/realized spread and price impact."""

import numpy as np
import pandas as pd
import pytest

from src.execution import (
    effective_spread,
    price_impact,
    quoted_spread,
    realized_spread,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02 09:30", periods=n, freq="min")


def test_buy_at_the_ask_pays_the_quoted_spread() -> None:
    idx = _idx(1)
    bid = pd.Series([100.0], index=idx)
    ask = pd.Series([101.0], index=idx)
    mid = (bid + ask) / 2
    quoted = quoted_spread(bid, ask)
    effective = effective_spread(ask, mid, side=1)
    # crossing the full half-spread on a buy costs exactly the quoted spread
    assert effective.iloc[0] == pytest.approx(quoted.iloc[0])
    assert effective.name == "effective_spread"


def test_sell_below_mid_is_a_positive_cost() -> None:
    idx = _idx(2)
    price = pd.Series([99.8, 100.4], index=idx)
    mid = pd.Series([100.0, 100.0], index=idx)
    out = effective_spread(price, mid, side=-1)
    assert out.iloc[0] == pytest.approx(0.4)  # sold under the mid: cost
    assert out.iloc[1] == pytest.approx(-0.8)  # sold above the mid: improvement


def test_identity_effective_equals_realized_plus_impact() -> None:
    rng = np.random.default_rng(8)
    idx = _idx(200)
    mid = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.0005, 200)), index=idx)
    price = mid * (1 + rng.normal(0, 0.0004, 200))
    future_mid = mid * (1 + rng.normal(0, 0.0006, 200))
    sides = pd.Series(rng.choice([-1.0, 1.0], 200), index=idx)
    eff = effective_spread(price, mid, sides)
    real = realized_spread(price, future_mid, sides)
    imp = price_impact(mid, future_mid, sides)
    assert np.allclose(eff.to_numpy(), real.to_numpy() + imp.to_numpy())


def test_adverse_selection_shows_as_positive_impact() -> None:
    idx = _idx(1)
    mid = pd.Series([100.0], index=idx)
    future_mid = pd.Series([100.3], index=idx)  # market ran after the buy
    assert price_impact(mid, future_mid, side=1).iloc[0] == pytest.approx(0.6)
    # for the liquidity provider the earned spread shrinks accordingly
    price = pd.Series([100.2], index=idx)
    assert realized_spread(price, future_mid, side=1).iloc[0] == pytest.approx(-0.2)


def test_relative_quoted_spread_divides_by_mid() -> None:
    idx = _idx(1)
    bid = pd.Series([99.5], index=idx)
    ask = pd.Series([100.5], index=idx)
    out = quoted_spread(bid, ask, relative=True)
    assert out.iloc[0] == pytest.approx(1.0 / 100.0)


def test_side_series_is_applied_per_trade() -> None:
    idx = _idx(2)
    price = pd.Series([100.4, 99.6], index=idx)
    mid = pd.Series([100.0, 100.0], index=idx)
    sides = pd.Series([1.0, -1.0], index=idx)
    out = effective_spread(price, mid, sides)
    assert np.allclose(out.to_numpy(), [0.8, 0.8])  # both crossed the spread


def test_bad_inputs_raise() -> None:
    idx = _idx(2)
    price = pd.Series([100.0, 100.0], index=idx)
    mid = pd.Series([100.0, 100.0], index=idx)
    with pytest.raises(ValueError, match="side values"):
        effective_spread(price, mid, side=0)
    with pytest.raises(ValueError, match="index"):
        effective_spread(price, mid, pd.Series([1.0], index=idx[:1]))
    with pytest.raises(ValueError, match="same index"):
        effective_spread(price.iloc[:1], mid, side=1)
    with pytest.raises(ValueError, match="positive"):
        effective_spread(pd.Series([100.0, np.nan], index=idx), mid, side=1)
    with pytest.raises(ValueError, match="ask must be >= bid"):
        quoted_spread(pd.Series([101.0, 100.0], index=idx), pd.Series([100.5, 100.5], index=idx))
