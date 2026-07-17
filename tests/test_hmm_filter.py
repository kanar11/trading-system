"""Tests for causal HMM forward filtering."""

import numpy as np
import pandas as pd
import pytest

from src.regime import (
    HMMConfig,
    filter_hmm_probabilities,
    filtered_hmm_states,
    fit_gaussian_hmm,
)


def _switching_series(seed: int = 0) -> tuple[pd.Series, np.ndarray]:
    """Two clearly separated regimes: calm around +0.1%, wild around -0.2%."""
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.001, 0.004, 250)
    wild = rng.normal(-0.002, 0.025, 250)
    labels = np.array([1] * 250 + [0] * 250)  # mean-sorted: wild has the LOWER mean
    idx = pd.date_range("2021-01-04", periods=500, freq="B")
    return pd.Series(np.concatenate([calm, wild]), index=idx), labels


def _fitted(series: pd.Series, n_states: int = 2):  # type: ignore[no-untyped-def]
    return fit_gaussian_hmm(series.to_numpy(), HMMConfig(n_states=n_states))


def test_rows_are_probability_distributions() -> None:
    series, _ = _switching_series()
    probs = filter_hmm_probabilities(series, _fitted(series))
    assert probs.shape == (500, 2)
    assert list(probs.columns) == [0, 1]
    assert np.allclose(probs.sum(axis=1).to_numpy(), 1.0)
    assert probs.index.equals(series.index)


def test_filter_is_strictly_causal() -> None:
    series, _ = _switching_series()
    model = _fitted(series)
    full = filter_hmm_probabilities(series, model)
    prefix = filter_hmm_probabilities(series.iloc[:200], model)
    # the first 200 rows are bit-identical: the future never leaks in
    assert np.array_equal(full.iloc[:200].to_numpy(), prefix.to_numpy())


def test_filter_tracks_the_regime_switch() -> None:
    series, labels = _switching_series()
    states = filtered_hmm_states(series, _fitted(series))
    assert states.name == "state"
    # well after the switch the causal filter must sit in the low-mean state
    agreement = float((states.to_numpy()[300:] == labels[300:]).mean())
    assert agreement > 0.8


def test_filtered_differs_from_smoothed_posterior() -> None:
    series, _ = _switching_series()
    model = _fitted(series)
    filtered = filter_hmm_probabilities(series, model).to_numpy()
    # smoothing conditions on the future, filtering does not — on noisy data
    # around the switch they cannot coincide
    assert not np.allclose(filtered, model.posterior)


def test_out_of_sample_filtering_works() -> None:
    series, labels = _switching_series()
    train, test = series.iloc[:350], series.iloc[350:]
    model = _fitted(train)
    oos_states = filtered_hmm_states(test, model)
    assert len(oos_states) == 150
    assert float((oos_states.to_numpy() == labels[350:]).mean()) > 0.8


def test_single_state_model_is_certain() -> None:
    series, _ = _switching_series()
    model = _fitted(series.iloc[:100], n_states=1)
    probs = filter_hmm_probabilities(series.iloc[:100], model)
    assert np.allclose(probs.to_numpy(), 1.0)


def test_bad_inputs_raise() -> None:
    series, _ = _switching_series()
    model = _fitted(series)
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match="empty"):
        filter_hmm_probabilities(empty, model)
    with_nan = series.copy()
    with_nan.iloc[5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        filter_hmm_probabilities(with_nan, model)
