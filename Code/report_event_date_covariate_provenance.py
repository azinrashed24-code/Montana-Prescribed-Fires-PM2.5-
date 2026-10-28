"""Standalone event-date covariate provenance report.

    python code/report_event_date_covariate_provenance.py

Run from the package root (or any working directory). Resolves data/ from
this file, not cwd. Does not fit OLS. Does not replace the name-year 4.55
table. Writes validation/event_date_covariate_provenance.txt.

RX complete-case wind/downwind/season/area are compared to Stage B.
WF wind is compared to the name-year modeling table (Stage B stores no
wildfire wind). Fuel is fire-year grain.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fit_event_date import write_event_date_covariate_provenance  # noqa: E402
from paths import VALIDATION  # noqa: E402

COMMAND = "python code/report_event_date_covariate_provenance.py"


def main() -> int:
    summary = write_event_date_covariate_provenance()
    out = VALIDATION / "event_date_covariate_provenance.txt"
    rx_n = int(summary.get("rx_n", -1))
    wf_n = int(summary.get("wf_n", -1))
    rx_ok = (
        int(summary.get("rx_wind_match", -1)) == rx_n
        and int(summary.get("rx_down_match", -1)) == rx_n
        and int(summary.get("rx_season_match", -1)) == rx_n
        and int(summary.get("rx_area_match", -1)) == rx_n
        and rx_n == 3856
    )
    wf_ok = int(summary.get("wf_nameyear_wind_match", -1)) == wf_n and wf_n == 309
    fuel_ok = int(summary.get("fuel_has_date", 1)) == 0
    print(f"Command: {COMMAND}")
    print(f"Wrote {out.as_posix() if out.exists() else out}")
    print(
        f"RX vs Stage B wind/downwind/season/area: "
        f"{summary.get('rx_wind_match')} / {rx_n}"
    )
    print(
        f"WF wind vs name-year modeling_station: "
        f"{summary.get('wf_nameyear_wind_match')} / {wf_n}"
    )
    print(
        "Fuel grain: fire-year only"
        if fuel_ok
        else "Fuel grain: date column present (unexpected)"
    )
    ok = rx_ok and wf_ok and fuel_ok and out.exists()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
