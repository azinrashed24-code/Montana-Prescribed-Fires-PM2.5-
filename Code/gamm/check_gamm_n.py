"""Check hourly GAMM input size without fitting bam().

Primary M4 sample: 379,433 hours / 761 fires / 21 monitors.
This check is sample counts only. Coefficients, fitted smooths, and AIC
were not refit here and remain outside the 111-check verification.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
GAMM_GZ = PKG / "data" / "gamm" / "csv" / "step05_gamm_model_data.csv.gz"
GAMM_CSV = PKG / "data" / "gamm" / "csv" / "step05_gamm_model_data.csv"

EXPECTED = {"n": 379433, "n_fires": 761, "n_monitors": 21}


def gamm_path() -> Path:
    if GAMM_GZ.exists():
        return GAMM_GZ
    if GAMM_CSV.exists():
        return GAMM_CSV
    raise FileNotFoundError(
        "Missing data/gamm/csv/step05_gamm_model_data.csv.gz "
        "(uncompressed .csv is also accepted)."
    )


def check_gamm_n() -> dict:
    path = gamm_path()
    df = pd.read_csv(path, usecols=["FireID", "MonitorID"], low_memory=False)
    got = {
        "n": int(len(df)),
        "n_fires": int(df["FireID"].nunique()),
        "n_monitors": int(df["MonitorID"].nunique()),
        "path": str(path.relative_to(PKG)).replace("\\", "/"),
        "bytes": int(path.stat().st_size),
    }
    return got


def main() -> int:
    got = check_gamm_n()
    ok = (
        got["n"] == EXPECTED["n"]
        and got["n_fires"] == EXPECTED["n_fires"]
        and got["n_monitors"] == EXPECTED["n_monitors"]
    )
    print(
        f"GAMM hourly table: {got['path']} "
        f"({got['bytes'] / (1024 * 1024):.2f} MB)"
    )
    print(
        f"  n={got['n']:,} fires={got['n_fires']} monitors={got['n_monitors']} "
        f"{'PASS' if ok else 'FAIL'}"
    )
    print(
        "  This validation checks sample counts only. Coefficients, fitted "
        "smooths, and AIC were not refit by this report."
    )
    if not ok:
        print("  expected 379,433 / 761 / 21")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
