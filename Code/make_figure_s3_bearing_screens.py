"""Figure S3: geographic bearing-screen sensitivity, nested on the primary sample.

Distance is held at ≤300 km. The highlighted 45–135° row is the Methods 2.4
complete-case sample (During minus Before: 4,186 linkages / 1,547 fires).
Narrower screens subset that sample. Broader screens add monitors outside
45–135° that remain within 300 km of the same primary fires.
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

from analysis import (  # noqa: E402
    FORMULA,
    OUTCOMES,
    complete_case,
    fit_clustered,
    load_modeling,
    wildfire_term,
)
from paths import DATA, FIGURES, RESULTS, VALIDATION, ensure_results, first_existing  # noqa: E402

DPI = 150
SCREENS = [
    ("75°–105° (narrower)", 75.0, 105.0, False),
    ("60°–120° (narrower)", 60.0, 120.0, False),
    ("45°–135° (primary)", 45.0, 135.0, True),
    ("30°–150° (broader)", 30.0, 150.0, False),
    ("0°–180° (eastern half)", 0.0, 180.0, False),
    ("No directional filter", 0.0, 360.0, False),
]
KEEP = [
    "fire_cohort",
    "Fire_Name",
    "Fire_Year",
    "aqs_code",
    "fire_id",
    "Distance_km",
    "Bearing_deg",
    "area_ha",
    "log1p_area_ha",
    "Dominant_FuelGroup",
    "met_season",
    "downwind_during",
    "wind_speed_ms_during_mean",
    "delta_During_minus_Before",
    "delta_MaxDailyDuring_minus_Before",
    "delta_After_minus_Before",
]


def link_key(d: pd.DataFrame) -> pd.Series:
    return (
        d["fire_cohort"].astype(str)
        + "|"
        + d["Fire_Name"].astype(str)
        + "|"
        + d["Fire_Year"].astype(str)
        + "|"
        + d["aqs_code"].astype(str)
    )


def pass_screen(bearing: pd.Series, bmin: float, bmax: float) -> pd.Series:
    b = pd.to_numeric(bearing, errors="coerce")
    if bmin == 0.0 and bmax == 360.0:
        return b.notna()
    return (b >= bmin) & (b <= bmax)


def prepare_cache(primary: pd.DataFrame) -> pd.DataFrame:
    cache_path = first_existing(DATA / "bearing_panel.csv.gz", DATA / "bearing_panel.csv")
    if cache_path is None:
        raise FileNotFoundError("Missing data/bearing_panel.csv.gz")
    cache = pd.read_csv(cache_path, low_memory=False)
    cache["Distance_km"] = pd.to_numeric(cache["Distance_km"], errors="coerce")
    cache["Bearing_deg"] = pd.to_numeric(cache["Bearing_deg"], errors="coerce")
    cache["downwind_during"] = pd.to_numeric(cache["downwind_during"], errors="coerce").fillna(0)
    cache["fire_id"] = (
        cache["fire_cohort"].astype(str)
        + "|"
        + cache["Fire_Name"].astype(str)
        + "|"
        + cache["Fire_Year"].astype(str)
    )
    fire_cov = (
        primary.groupby("fire_id", as_index=False)
        .agg(
            area_ha=("area_ha", "median"),
            log1p_area_ha=("log1p_area_ha", "median"),
            Dominant_FuelGroup=("Dominant_FuelGroup", "first"),
            met_season=("met_season", "first"),
            wind_speed_ms_during_mean=("wind_speed_ms_during_mean", "median"),
        )
    )
    cache = cache.merge(fire_cov, on="fire_id", how="left", suffixes=("", "_p"))
    cache["in_primary_fire"] = cache["fire_id"].isin(set(primary["fire_id"]))
    cache["key"] = link_key(cache)
    cache["fire_cohort"] = pd.Categorical(cache["fire_cohort"], categories=["WFIGS_RX", "MTBS_wildfire"])
    return cache


def screen_sample(primary: pd.DataFrame, cache: pd.DataFrame, outcome: str, bmin: float, bmax: float) -> pd.DataFrame:
    prim = primary.copy()
    prim["key"] = link_key(prim)
    if bmin >= 45.0 and bmax <= 135.0 and not (bmin == 0.0 and bmax == 360.0):
        return complete_case(prim[pass_screen(prim["Bearing_deg"], bmin, bmax)], outcome)
    if bmin == 0.0 and bmax == 360.0:
        east_part = prim
        extra_bear = ~((cache["Bearing_deg"] >= 45.0) & (cache["Bearing_deg"] <= 135.0))
    else:
        east_part = prim[pass_screen(prim["Bearing_deg"], max(bmin, 45.0), min(bmax, 135.0))]
        extra_bear = pass_screen(cache["Bearing_deg"], bmin, bmax) & (
            (cache["Bearing_deg"] < 45.0) | (cache["Bearing_deg"] > 135.0)
        )
    extra = cache[extra_bear & cache["in_primary_fire"] & ~cache["key"].isin(set(prim["key"]))].copy()
    extra = extra[extra[outcome].notna()]
    extra = extra[extra["Distance_km"].notna() & (extra["Distance_km"] <= 300) & (extra["Distance_km"] > 0)]
    extra = extra[extra["area_ha"].notna() & extra["Dominant_FuelGroup"].notna()]
    extra = extra[extra["wind_speed_ms_during_mean"].notna() & extra["met_season"].notna()]
    extra["downwind_during"] = extra["downwind_during"].fillna(0)
    extra["log1p_area_ha"] = extra["log1p_area_ha"].fillna(np.log1p(extra["area_ha"]))
    cols = [c for c in KEEP if c in east_part.columns]
    extra_cols = [c for c in cols if c in extra.columns]
    out = pd.concat([east_part[cols], extra[extra_cols]], ignore_index=True)
    return complete_case(out, outcome)


def fit_screens(primary_by_outcome: dict[str, pd.DataFrame], cache: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, col in OUTCOMES:
        prim = primary_by_outcome[col]
        for name, bmin, bmax, is_primary in SCREENS:
            d = screen_sample(prim, cache, col, bmin, bmax)
            rec = {
                "outcome": label,
                "screen": name,
                "bearing_min": bmin,
                "bearing_max": bmax,
                "is_primary": is_primary,
                "n_linkages": int(len(d)),
                "n_fire_years": int(d["fire_id"].nunique()) if len(d) else 0,
                "n_rx": int((d["fire_cohort"] == "WFIGS_RX").sum()) if len(d) else 0,
                "n_wf": int((d["fire_cohort"] == "MTBS_wildfire").sum()) if len(d) else 0,
            }
            if len(d) < 80 or d["fire_cohort"].nunique() < 2:
                rec.update({"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan})
            else:
                m = fit_clustered(FORMULA.format(outcome=col), d)
                term = wildfire_term(m)
                lo, hi = m.conf_int().loc[term]
                rec.update(
                    {
                        "estimate": float(m.params[term]),
                        "ci_low": float(lo),
                        "ci_high": float(hi),
                        "p_value": float(m.pvalues[term]),
                    }
                )
            rows.append(rec)
            print(f"  {label:28s} {name:28s} n={rec['n_linkages']:5d} fires={rec['n_fire_years']:4d}")
    return pd.DataFrame(rows)


def plot_figure(res: pd.DataFrame, out_png: Path) -> None:
    fig = plt.figure(figsize=(14.0, 5.1), dpi=DPI)
    gs = fig.add_gridspec(1, 6, width_ratios=[3.2, 1.2, 3.2, 1.2, 3.2, 1.2], wspace=0.08)
    colors = {
        "During minus Before": "#0f766e",
        "Peak daily minus Before": "#b45309",
        "After minus Before": "#7c3aed",
    }
    screen_order = [s[0] for s in SCREENS]
    lo_all = pd.to_numeric(res["ci_low"], errors="coerce")
    hi_all = pd.to_numeric(res["ci_high"], errors="coerce")
    x_lo = float(np.nanmin(lo_all.to_numpy())) if lo_all.notna().any() else 0.0
    x_hi = float(np.nanmax(hi_all.to_numpy())) if hi_all.notna().any() else 1.0
    x_lo = min(0.0, x_lo)
    span = max(x_hi - x_lo, 1.0)
    shared_xlim = (x_lo - 0.03 * span, x_hi + 0.05 * span)
    for j, (label, _) in enumerate(OUTCOMES):
        ax = fig.add_subplot(gs[0, 2 * j])
        axn = fig.add_subplot(gs[0, 2 * j + 1], sharey=ax)
        sub = res[res["outcome"] == label].copy()
        sub["screen"] = pd.Categorical(sub["screen"], categories=screen_order, ordered=True)
        sub = sub.sort_values("screen")
        y = np.arange(len(sub))
        est = sub["estimate"].to_numpy()
        lo = sub["ci_low"].to_numpy()
        hi = sub["ci_high"].to_numpy()
        ax.axvline(0, color="#94a3b8", lw=1.0, zorder=0)
        for i, prim in enumerate(sub["is_primary"]):
            if bool(prim):
                ax.axhspan(i - 0.38, i + 0.38, color="#fef3c7", zorder=0)
                axn.axhspan(i - 0.38, i + 0.38, color="#fef3c7", zorder=0)
        ax.errorbar(est, y, xerr=[est - lo, hi - est], fmt="o", color=colors[label], ecolor=colors[label], elinewidth=1.4, capsize=3.5, markersize=6.5, zorder=2)
        ax.set_yticks(y)
        if j == 0:
            ax.set_yticklabels(list(sub["screen"]), fontsize=8)
            ax.set_ylabel("Directional screen")
        else:
            ax.tick_params(labelleft=False)
        ax.set_ylim(-0.7, len(sub) - 0.3)
        ax.set_xlim(*shared_xlim)
        ax.set_xlabel("Wildfire − prescribed\ncoefficient (µg m$^{-3}$)")
        ax.set_title(label, fontsize=10, fontweight="600")
        ax.grid(axis="x", color="#e2e8f0", lw=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        axn.set_xlim(0, 1)
        axn.set_xticks([])
        axn.tick_params(left=False, labelleft=False)
        for spine in axn.spines.values():
            spine.set_visible(False)
        axn.set_title("Sample", fontsize=8, fontweight="600", color="#334155", pad=6)
        for i, r in enumerate(sub.itertuples(index=False)):
            note = ""
            if bool(r.is_primary):
                note = "  primary"
            axn.text(0.08, y[i], f"n={int(r.n_linkages):,}\n{int(r.n_fire_years):,} fires{note}", va="center", ha="left", fontsize=6.6, color="#475569", linespacing=1.1)
        axn.set_ylim(-0.7, len(sub) - 0.3)
    fig.suptitle(
        "Figure S3. Sensitivity of the wildfire-versus-prescribed coefficient\nto geographic bearing screens (distance held at ≤300 km)",
        fontsize=11,
        fontweight="600",
        y=0.98,
    )
    fig.subplots_adjust(left=0.14, right=0.99, top=0.84, bottom=0.16, wspace=0.12)
    fig.savefig(out_png, facecolor="white", bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def make_figure_s3() -> pd.DataFrame:
    ensure_results()
    st = load_modeling()
    primary = {col: complete_case(st, col) for _, col in OUTCOMES}
    d0 = primary["delta_During_minus_Before"]
    print(f"Primary During minus Before: {len(d0)} linkages / {d0['fire_id'].nunique()} fires")
    cache = prepare_cache(d0)
    res = fit_screens(primary, cache)
    prim_row = res[(res["outcome"] == "During minus Before") & (res["is_primary"])].iloc[0]
    if int(prim_row["n_linkages"]) != 4186 or int(prim_row["n_fire_years"]) != 1547:
        raise SystemExit(
            f"Highlighted 45-135 is {int(prim_row['n_linkages'])}/{int(prim_row['n_fire_years'])}, expected 4186/1547"
        )
    res.to_csv(RESULTS / "Figure_S3_bearing_screen_coefficients.csv", index=False)
    png = FIGURES / "Figure_S3_bearing_screens.png"
    plot_figure(res, png)
    (FIGURES / "Figure_S3_caption.txt").write_text(
        "Figure S3. Sensitivity of adjusted wildfire-versus-prescribed fire PM2.5 contrasts "
        "to geographic bearing screens. Distance is held at ≤300 km in every row. The "
        "highlighted 45–135° east-sector screen is the primary Methods 2.4 sample "
        "(During minus Before: 4,186 linkages / 1,547 fires). Narrower screens subset that "
        "sample. Broader screens and no directional filter add monitors outside 45–135° that "
        "remain within 300 km of the same primary fires. Extra pairs inherit fire-level "
        "area, fuel, season, and wind from the primary name-year sample; pair-specific "
        "fields on extras are distance, bearing, outcomes, and panel downwind "
        "(see validation/s3_expanded_sample_covariates.txt). Models use the same covariate "
        "list as the main adjusted table. East sector is a geographic selection rule; "
        "meteorological downwind is a covariate.\n",
        encoding="utf-8",
    )
    (VALIDATION / "s3_expanded_sample_covariates.txt").write_text(
        "Figure S3 expanded-sample covariates (broader screens and no directional filter).\n"
        "\n"
        "data/bearing_panel.csv.gz columns: fire_cohort, Fire_Name, Fire_Year, aqs_code,\n"
        "Distance_km, Bearing_deg, during/peak/after outcomes, downwind_during.\n"
        "The panel does not store wind_speed_ms_during_mean, Dominant_FuelGroup,\n"
        "met_season, or area_ha.\n"
        "\n"
        "Narrower screens (75–105 and 60–120) subset the primary 4,186-row complete-case\n"
        "sample. Broader screens add extra fire–monitor pairs from the panel that stay\n"
        "within 300 km of the same primary fires. Those extra rows inherit fire-level\n"
        "area, fuel, season, and wind from the primary sample (median wind, first fuel,\n"
        "first season by fire_id = cohort|name|year). Downwind, distance, bearing, and\n"
        "outcomes for extras come from the panel; missing downwind is coded 0.\n"
        "\n"
        "Verified from packaged files: extra pairs do not have independently computed\n"
        "during-window wind or event-specific fuel at the extra monitor. Fire-level\n"
        "covariates are copied from the name-year primary sample. Pair-specific fields\n"
        "on extras are distance, bearing, outcomes, and panel downwind only.\n",
        encoding="utf-8",
    )
    print("Wrote", png)
    return res


def main() -> None:
    make_figure_s3()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
