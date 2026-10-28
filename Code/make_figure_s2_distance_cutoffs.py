"""Figure S2: distance-cutoff sensitivity (50/100/150/200/300 km).

Table S2 is the distance-form AIC table (linear vs log1p vs spline).
This figure holds the east sector and varies only the distance cap.
The 300 km row is the primary 4,186 / 1,547 sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analysis import FORMULA, OUTCOMES, complete_case, fit_clustered, load_modeling, wildfire_term  # noqa: E402
from paths import FIGURES, RESULTS, ensure_results  # noqa: E402

DPI = 150
CUTOFFS = [50, 100, 150, 200, 300]


def make_figure_s2() -> pd.DataFrame:
    ensure_results()
    st = load_modeling()
    rows = []
    for label, col in OUTCOMES:
        d0 = complete_case(st, col)
        for cap in CUTOFFS:
            d = d0[d0["Distance_km"] <= cap].copy()
            rec = {
                "outcome": label,
                "distance_cutoff_km": cap,
                "n_linkages": int(len(d)),
                "n_fire_years": int(d["fire_id"].nunique()) if len(d) else 0,
                "is_primary": cap == 300,
            }
            if len(d) < 80 or d["fire_cohort"].nunique() < 2:
                rec.update({"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan})
            else:
                m = fit_clustered(FORMULA.format(outcome=col), d)
                term = wildfire_term(m)
                lo, hi = m.conf_int().loc[term]
                rec.update({"estimate": float(m.params[term]), "ci_low": float(lo), "ci_high": float(hi)})
            rows.append(rec)
    res = pd.DataFrame(rows)
    prim = res[(res["outcome"] == "During minus Before") & (res["is_primary"])].iloc[0]
    if int(prim["n_linkages"]) != 4186 or int(prim["n_fire_years"]) != 1547:
        raise SystemExit(f"S2 300 km is {int(prim['n_linkages'])}/{int(prim['n_fire_years'])}, expected 4186/1547")
    res.to_csv(RESULTS / "Figure_S2_distance_cutoff_coefficients.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), dpi=DPI, sharey=True)
    colors = ["#0f766e", "#b45309", "#7c3aed"]
    for ax, (label, _), color in zip(axes, OUTCOMES, colors):
        sub = res[res["outcome"] == label].sort_values("distance_cutoff_km")
        y = np.arange(len(sub))
        ax.axvline(0, color="#94a3b8", lw=1.0)
        for i, prim_flag in enumerate(sub["is_primary"]):
            if bool(prim_flag):
                ax.axhspan(i - 0.38, i + 0.38, color="#fef3c7", zorder=0)
        ax.errorbar(
            sub["estimate"],
            y,
            xerr=[sub["estimate"] - sub["ci_low"], sub["ci_high"] - sub["estimate"]],
            fmt="o",
            color=color,
            capsize=3,
        )
        ax.set_yticks(
            y,
            [f"≤{int(k)} km  (n={int(n):,})" for k, n in zip(sub["distance_cutoff_km"], sub["n_linkages"])],
        )
        ax.set_title(label, fontsize=9.5, fontweight="600")
        ax.set_xlabel("Wildfire − prescribed (µg m$^{-3}$)")
        ax.grid(axis="x", color="#e2e8f0")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Distance cutoff")
    fig.suptitle("Figure S2. Distance-cutoff sensitivity (east sector held)", fontsize=11, fontweight="600")
    fig.tight_layout()
    png = FIGURES / "Figure_S2_distance_cutoffs.png"
    fig.savefig(png, facecolor="white", bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print("Wrote", png)
    return res


def main() -> None:
    print(make_figure_s2().to_string(index=False))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
