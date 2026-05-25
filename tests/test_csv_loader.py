"""Tests for the local CSV OHLCV loader."""

import numpy as np
import pandas as pd
import pytest

from src.data.csv_loader import load_csv_ohlcv


def _write_csv(tmp_path, rows: pd.DataFrame, name: str = "data.csv") -> str:
    p = tmp_path / name
    rows.to_csv(p, index=False)
    return str(p)


def _sample_rows(date_label: str = "Date") -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    return pd.DataFrame(
        {
            date_label: dates,
            "Open": np.arange(100, 110, dtype=float),
            "High": np.arange(101, 111, dtype=float),
            "Low": np.arange(99, 109, dtype=float),
            "Close": np.arange(100, 110, dtype=float),
            "Adj Close": np.arange(99, 109, dtype=float),
            "Volume": np.arange(1_000_000, 1_010_000, 1_000),
        }
    )


def test_loads_basic_csv(tmp_path):
    path = _write_csv(tmp_path, _sample_rows())
    df = load_csv_ohlcv(path)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 10
    assert isinstance(df.index, pd.DatetimeIndex)


def test_auto_detects_alternative_date_column(tmp_path):
    for label in ["Datetime", "Timestamp", "Time"]:
        path = _write_csv(tmp_path, _sample_rows(label), name=f"{label}.csv")
        df = load_csv_ohlcv(path)
        assert len(df) == 10


def test_explicit_date_col(tmp_path):
    rows = _sample_rows("trade_day")
    path = _write_csv(tmp_path, rows)
    df = load_csv_ohlcv(path, date_col="trade_day")
    assert len(df) == 10


def test_use_adj_close_replaces_close(tmp_path):
    path = _write_csv(tmp_path, _sample_rows())
    df_raw = load_csv_ohlcv(path, use_adj_close=False)
    df_adj = load_csv_ohlcv(path, use_adj_close=True)
    assert (df_adj["close"] != df_raw["close"]).all()
    # adj close = close - 1 in our fixture
    assert (df_adj["close"] == df_raw["close"] - 1).all()


def test_date_range_filter(tmp_path):
    path = _write_csv(tmp_path, _sample_rows())
    df = load_csv_ohlcv(path, start="2020-01-03", end="2020-01-08")
    assert df.index.min() >= pd.Timestamp("2020-01-03")
    assert df.index.max() <= pd.Timestamp("2020-01-08")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv_ohlcv(tmp_path / "nope.csv")


def test_empty_file_raises(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("Date,Open,High,Low,Close,Volume\n")
    with pytest.raises(ValueError, match="empty"):
        load_csv_ohlcv(p)


def test_missing_required_columns_raises(tmp_path):
    rows = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=3, freq="B"),
            "Open": [1, 2, 3],
            "Close": [1, 2, 3],
            # missing high, low, volume
        }
    )
    path = _write_csv(tmp_path, rows)
    with pytest.raises(ValueError, match="missing required"):
        load_csv_ohlcv(path)


def test_filters_empty_result_raises(tmp_path):
    path = _write_csv(tmp_path, _sample_rows())
    with pytest.raises(ValueError, match="No rows left"):
        load_csv_ohlcv(path, start="2099-01-01")
