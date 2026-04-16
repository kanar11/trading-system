"""Grid search convenience script.

Thin wrapper around src.reporting.sweep.run_sweep — kept for backwards
compatibility. For full control use:
    python -m src.reporting.sweep
"""

import logging

from src.reporting.sweep import run_sweep

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    run_sweep()
