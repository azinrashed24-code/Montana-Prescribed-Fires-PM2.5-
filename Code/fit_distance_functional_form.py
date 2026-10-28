"""Linear distance vs log(1+distance) vs restricted cubic spline (4 df).

Primary model uses linear distance. This script reports AIC and
the wildfire coefficient under the three distance forms on the same
complete-case sample. It does not replace the primary adjusted table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analysis import OUTCOMES, complete_case, fit_clustered, load_modeling, wildfire_term
from paths import RESULTS, ensure_results

# Precompute log(1+distance). Do not put np.log1p() in the patsy formula
# (patsy does not see the Python name np).
SPECS = [
    ("Linear distance (primary)", "Distance_km"),
    ("log(1 + distance)", "log1p_distance_km"),
    ("Restricted cubic spline, 4 df", "cr(Distance_km, df=4)"),
]


def formula(outcome: str, distance_term: str) -> str:
    return (
        f"{outcome} ~ C(fire_cohort, Treatment(reference='WFIGS_RX'))"
        " + log1p_area_ha"
        " + C(Dominant_FuelGroup, Treatment(reference='Grass'))"
        " + C(met_season, Treatment(reference='Fall'))"
        f" + {distance_term}"
        " + downwind_during"
        " + wind_speed_ms_during_mean"
    )


def main() -> None:
    ensure_results()
    st = load_modeling()
    rows = []
    for label, col in OUTCOMES:
        d = complete_case(st, col)
        d = d.copy()
        d["log1p_distance_km"] = np.log1p(pd.to_numeric(d["Distance_km"], errors="coerce"))
        for spec_name, term in SPECS:
            m = fit_clustered(formula(col, term), d)
            w = wildfire_term(m)
            lo, hi = m.conf_int().loc[w]
            rows.append(
                {
                    "Outcome": label,
                    "Distance form": spec_name,
                    "Wildfire estimate": float(m.params[w]),
                    "CI low": float(lo),
                    "CI high": float(hi),
                    "AIC": float(m.aic),
                    "n": int(len(d)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "Table_S2_distance_functional_form.csv", index=False)
    print(out.to_string(index=False))
    print("Wrote", RESULTS / "Table_S2_distance_functional_form.csv")


if __name__ == "__main__":
    main()
