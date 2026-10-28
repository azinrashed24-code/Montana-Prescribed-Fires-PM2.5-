"""Package locations. Scripts read data/ and never overwrite it.

PACKAGE_ROOT is this file's parent folder (the repo root), not os.getcwd().
`python code/run_analysis.py` therefore works from any working directory.
"""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA = PACKAGE_ROOT / "data"
RESULTS = PACKAGE_ROOT / "results"
FIGURES = RESULTS / "figures"
VALIDATION = PACKAGE_ROOT / "validation"
DERIVED = PACKAGE_ROOT / "derived"
MAP = DATA / "map"
GAMM_CSV_DIR = DATA / "gamm" / "csv"


def ensure_results() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    VALIDATION.mkdir(parents=True, exist_ok=True)


def first_existing(*paths: Path) -> Path | None:
    """Return the first path that exists, else None."""
    for path in paths:
        if path.exists():
            return path
    return None
