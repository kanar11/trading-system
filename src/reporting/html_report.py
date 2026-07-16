"""Self-contained HTML performance report.

The matplotlib tearsheet (:mod:`src.reporting.tearsheet`) needs a display;
this is the shareable artefact — a single HTML string with no external
assets, images or scripts, assembled from the package's own tables:
the head-to-head statistics of :func:`src.reporting.benchmark.
benchmark_comparison` (or its strategy-only column), the calendar
year × month grid of :func:`src.reporting.periodic.monthly_returns_table`
and the worst drawdown episodes of :func:`src.reporting.drawdowns.
drawdown_table`. Drop the string into a file and it renders anywhere —
CI artefacts, email, a wiki.

Direct-import module::

    from src.reporting.html_report import html_report, save_html_report
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.reporting.benchmark import _RELATIVE_ROWS, benchmark_comparison
from src.reporting.drawdowns import drawdown_table
from src.reporting.periodic import monthly_returns_table

_STYLE = """
body { font-family: Segoe UI, Arial, sans-serif; margin: 2em; color: #222; }
h1 { border-bottom: 2px solid #444; padding-bottom: 0.2em; }
h2 { margin-top: 1.6em; color: #333; }
table { border-collapse: collapse; margin: 0.6em 0; }
th, td { border: 1px solid #bbb; padding: 4px 10px; text-align: right; }
th { background: #f0f0f0; }
.meta { color: #666; font-size: 0.9em; }
""".strip()


def html_report(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Strategy report",
    periods_per_year: int = 252,
    top_drawdowns: int = 5,
) -> str:
    """Assemble a self-contained HTML performance report.

    Args:
        returns: Per-bar strategy returns on a DatetimeIndex.
        benchmark: Optional benchmark returns on the same index; adds the
            benchmark column and the relative block to the summary.
        title: Report headline.
        periods_per_year: Bars per year for annualisation.
        top_drawdowns: Episodes shown in the drawdown table.

    Returns:
        A complete ``<html>`` document as a string (inline CSS, no
        external assets).

    Raises:
        ValueError: If ``returns`` is empty or the benchmark is misaligned
            (raised by the underlying tables).
    """
    if len(returns) == 0:
        raise ValueError("returns must not be empty.")

    summary = benchmark_comparison(
        returns,
        benchmark if benchmark is not None else returns,
        periods_per_year=periods_per_year,
    )
    if benchmark is None:
        # self-comparison fills the relative block with degenerate values
        # (beta 1, TE 0) — drop those rows, keep the absolute statistics
        absolute_rows = [r for r in summary.index if r not in _RELATIVE_ROWS]
        summary = summary.loc[absolute_rows, ["strategy"]]

    monthly = monthly_returns_table(returns)
    drawdowns = drawdown_table(returns, top_n=top_drawdowns)

    def _table(frame: pd.DataFrame) -> str:
        return frame.to_html(float_format=lambda x: f"{x:.4f}", na_rep="—", border=0)

    first, last = returns.index[0], returns.index[-1]
    parts = [
        f"<style>{_STYLE}</style>",
        f"<h1>{title}</h1>",
        f'<p class="meta">{len(returns)} bars, {first:%Y-%m-%d} to {last:%Y-%m-%d}.'
        f" Benchmark: {'included' if benchmark is not None else 'none'}.</p>",
        "<h2>Summary statistics</h2>",
        _table(summary),
        "<h2>Monthly returns</h2>",
        _table(monthly),
        f"<h2>Top {top_drawdowns} drawdowns</h2>",
        _table(drawdowns),
    ]
    body = "\n".join(parts)
    head = "<html><head><meta charset='utf-8'></head><body>"
    return f"<!DOCTYPE html>\n{head}\n{body}\n</body></html>"


def save_html_report(
    path: str | Path,
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Strategy report",
    periods_per_year: int = 252,
    top_drawdowns: int = 5,
) -> Path:
    """Write :func:`html_report` to ``path`` (UTF-8) and return the path."""
    target = Path(path)
    document = html_report(
        returns,
        benchmark=benchmark,
        title=title,
        periods_per_year=periods_per_year,
        top_drawdowns=top_drawdowns,
    )
    target.write_text(document, encoding="utf-8")
    return target
