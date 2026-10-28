"""Fit the primary adjusted wildfire vs prescribed-fire models."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analysis import FORMULA, OUTCOMES, complete_case, fit_clustered, format_p, load_modeling, wildfire_term
from paths import RESULTS, ensure_results


def main() -> None:
    ensure_results()
    st = load_modeling()
    rows = []
    for label, col in OUTCOMES:
        d = complete_case(st, col)
        m = fit_clustered(FORMULA.format(outcome=col), d)
        term = wildfire_term(m)
        lo, hi = m.conf_int().loc[term]
        rows.append(
            {
                "Outcome": label,
                "Fire-type contrast": "Wildfire vs prescribed fire",
                "Estimate, ug m-3": round(float(m.params[term]), 2),
                "95% CI, ug m-3": f"{lo:.2f}–{hi:.2f}",
                "p-value": format_p(float(m.pvalues[term])),
                "Linkages, n": int(len(d)),
                "Fires, n": int(d["fire_id"].nunique()),
                "Covariates": "Fire size, distance, fuel, season, wind, downwind",
            }
        )
    import pandas as pd

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "Table_adjusted_wildfire_vs_rx.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
