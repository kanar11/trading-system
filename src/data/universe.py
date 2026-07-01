"""Pre-defined trading universes (baskets of tickers).

Constant lists of tickers commonly used as research baskets. Keeping
them centralised means strategies / portfolio backtests can refer to
them by name rather than hard-coding hundreds of strings inline.

These are *static snapshots*; for live S&P 500 constituents you'd
want a properly maintained data source (Wikipedia, FactSet, etc.).
The point here is to ship usable defaults for back-research.
"""

from __future__ import annotations

# Mega-cap US tech
FAANG = ["META", "AAPL", "AMZN", "NFLX", "GOOGL"]
FAANG_PLUS = FAANG + ["MSFT", "NVDA", "TSLA"]

# Dow 30 (2024 composition)
DOW30 = [
    "AAPL",
    "AMGN",
    "AXP",
    "BA",
    "CAT",
    "CRM",
    "CSCO",
    "CVX",
    "DIS",
    "DOW",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "JPM",
    "KO",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "V",
    "VZ",
    "WBA",
    "WMT",
]

# SPDR US sector ETFs (11 sectors)
SECTOR_ETFS = {
    "XLB": "Materials",
    "XLC": "Communication Services",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}

# Broad-market and factor benchmarks
BENCHMARKS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "DIA": "Dow Jones Industrial Avg",
    "EFA": "MSCI EAFE",
    "EEM": "MSCI Emerging Markets",
    "AGG": "US Aggregate Bonds",
    "GLD": "Gold",
    "TLT": "20+ Year Treasuries",
    "USO": "Crude Oil",
}

# Common factor ETFs for attribution
FACTOR_ETFS = {
    "MTUM": "Momentum",
    "VLUE": "Value",
    "QUAL": "Quality",
    "SIZE": "Size",
    "USMV": "Min-Volatility",
    "HDV": "High Dividend",
}


def _universe_table() -> dict[str, list[str]]:
    """Map of built-in universe name -> ticker list."""
    return {
        "faang": FAANG,
        "faang_plus": FAANG_PLUS,
        "dow30": DOW30,
        "sectors": list(SECTOR_ETFS.keys()),
        "benchmarks": list(BENCHMARKS.keys()),
        "factors": list(FACTOR_ETFS.keys()),
    }


def get_universe(name: str) -> list[str]:
    """Look up a built-in universe by name.

    Args:
        name: One of ``"faang"``, ``"faang_plus"``, ``"dow30"``,
            ``"sectors"``, ``"benchmarks"``, ``"factors"``.

    Returns:
        List of ticker strings.

    Raises:
        KeyError: If the name is unknown.
    """
    key = name.strip().lower().replace("-", "_")
    table = _universe_table()
    if key not in table:
        raise KeyError(f"Unknown universe {name!r}. Available: {sorted(table.keys())}")
    return list(table[key])  # defensive copy


def list_universes() -> list[str]:
    """Return the sorted names of the built-in universes."""
    return sorted(_universe_table())


def combine_universes(*names: str, dedupe: bool = True) -> list[str]:
    """Concatenate several built-in universes into one ticker list.

    Order-preserving. With ``dedupe`` (default), the first occurrence of each
    ticker is kept and later duplicates dropped.

    Raises:
        KeyError: If any name is unknown.
    """
    combined: list[str] = []
    for name in names:
        combined.extend(get_universe(name))
    if not dedupe:
        return combined
    seen: set[str] = set()
    unique: list[str] = []
    for ticker in combined:
        if ticker not in seen:
            seen.add(ticker)
            unique.append(ticker)
    return unique


def in_universe(ticker: str, name: str) -> bool:
    """Return whether ``ticker`` (case-insensitive) is in the named universe.

    Raises:
        KeyError: If the universe name is unknown.
    """
    symbol = ticker.strip().upper()
    return symbol in {t.upper() for t in get_universe(name)}
