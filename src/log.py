# src/log.py
"""
Centralised logging configuration for the battery project.

Usage
-----
Entry points (main.py, run_full_pipeline.py __main__):

    from src.log import setup_logging
    setup_logging(output_dir=Path("outputs/run_xyz"))

Every other module:

    import logging
    logger = logging.getLogger(__name__)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMATTER = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def setup_logging(
    output_dir: Path | None = None,
    level: int = logging.INFO,
) -> None:
    """Configure the root logger. Safe to call multiple times (no-op after first call)."""
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_FORMATTER)
    root.addHandler(ch)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
        fh.setFormatter(_FORMATTER)
        root.addHandler(fh)
