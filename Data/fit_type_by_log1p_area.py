"""Fire type × log(1 + burned area), same covariates as the primary model."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from paths import ensure_results
from run_analysis import interaction


def main() -> None:
    ensure_results()
    table = interaction()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
