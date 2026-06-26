"""Thin entry point for the backtesting pipeline.

The implementation lives in :mod:`quantbt.cli`; this shim keeps ``python main.py``
working. Prefer the installed console script ``quantbt-backtest``.
"""

from quantbt.cli import main

if __name__ == "__main__":
    main()
