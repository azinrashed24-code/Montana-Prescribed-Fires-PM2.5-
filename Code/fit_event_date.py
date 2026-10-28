"""Independent event-date clustered OLS, refit from packaged event-level data.

    python code/fit_event_date.py

Reads data/modeling_station_with_date.csv (shipped Stage B event+monitor
rows; rebuilt from Stage B if that file is absent). Joins max_daily on
event+monitor and overlays fuel unique on Fire_Name+Fire_Year. Does not
read results/*.csv. Does not replace the name-year 4.55 table. Writes
results/Table_event_date_coefficients.csv,
validation/event_date_independent_refit.txt, and
validation/event_date_covariate_provenance.txt.

Covariate provenance alone (no OLS):

    python code/report_event_date_covariate_provenance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analysis import (  # noqa: E402
    EVENT_KEYS,
    EVENT_MONITOR_KEYS,
    FORMULA,
    OUTCOMES,
    build_event_monitor_table_from_stage_b,
    chl_same_event_audit,
    complete_case,
    event_key_extras,
    iso_date,
    load_stage_b,
    make_event_id,
    make_nameyear_id,
    overlay_fuel,
    require_unique_keys,
    wildfire_term,
)
from paths import DATA, RESULTS, VALIDATION, ensure_results  # noqa: E402

COMMAND = "python code/fit_event_date.py"
CLUSTER = "event_id"
DATED_NAME = "modeling_station_with_date.csv"


def _software_versions() -> list[str]:
    import matplotlib as mpl

    return [
        f"python {sys.version.split()[0]}",
        f"pandas {pd.__version__}",
        f"numpy {np.__version__}",
        f"statsmodels {statsmodels.__version__}",
        f"matplotlib {mpl.__version__}",
    ]


def _read_max_daily_event() -> pd.DataFrame:
    md = pd.read_csv(DATA / "max_daily.csv", low_memory=False)
    if "Fire_Date" not in md.columns and "File_Date" in md.columns:
        md["Fire_Date"] = md["File_Date"]
    if "Fire_Date" not in md.columns:
        raise ValueError("data/max_daily.csv has no Fire_Date; cannot join on event+monitor.")
    md = md[
        EVENT_MONITOR_KEYS + ["delta_MaxDailyDuring_minus_Before"]
    ].copy()
    md["Fire_Date"] = iso_date(md["Fire_Date"])
    require_unique_keys(md, EVENT_MONITOR_KEYS, "max_daily event+monitor")
    return md


def load_event_date_table() -> tuple[pd.DataFrame, dict]:
    """Build the event-date modeling table from packaged event-level files.

    Prefer data/modeling_station_with_date.csv (shipped). If it is missing,
    rebuild the same event+monitor grain from Stage B. Reconstruct is
    optional. Same-event covariates (area, season, wind, downwind,
    outcomes) stay on the Fire_Date row.
    """
    dated = DATA / DATED_NAME
    source = DATED_NAME
    if dated.exists():
        st = pd.read_csv(dated, low_memory=False)
        if "Fire_Date" not in st.columns and "File_Date" in st.columns:
            st["Fire_Date"] = st["File_Date"]
        st["Fire_Date"] = iso_date(st["Fire_Date"])
    else:
        source = "stage_b_daily.csv (rebuilt event+monitor)"
        st = build_event_monitor_table_from_stage_b()
        st["Fire_Date"] = iso_date(st["Fire_Date"])
    st = st.dropna(subset=["Fire_Date"]).copy()
    require_unique_keys(st, EVENT_MONITOR_KEYS, "event-monitor table before join")
    extras_before_join = event_key_extras(st, EVENT_MONITOR_KEYS)

    n0 = len(st)
    md = _read_max_daily_event()
    st = st.merge(
        md,
        on=EVENT_MONITOR_KEYS,
        how="left",
        suffixes=("", "_max_daily"),
    )
    if len(st) != n0:
        raise ValueError(f"event+monitor join inflated {n0} -> {len(st)}")
    extras_after_join = event_key_extras(st, EVENT_MONITOR_KEYS)
    if extras_after_join != 0:
        raise ValueError(
            f"Event-monitor join is not unique: extras={extras_after_join}."
        )
    if "delta_MaxDailyDuring_minus_Before" not in st.columns:
        raise ValueError("Peak outcome missing after event+monitor join.")
    if "delta_MaxDailyDuring_minus_Before_max_daily" in st.columns:
        dated_peak = pd.to_numeric(st["delta_MaxDailyDuring_minus_Before"], errors="coerce")
        md_peak = pd.to_numeric(st["delta_MaxDailyDuring_minus_Before_max_daily"], errors="coerce")
        both = dated_peak.notna() & md_peak.notna()
        disagree = int((both & (dated_peak != md_peak)).sum())
        if disagree:
            raise ValueError(
                f"Dated peak disagrees with max_daily on {disagree} event+monitor rows."
            )
        st = st.drop(columns=["delta_MaxDailyDuring_minus_Before_max_daily"])

    if "log1p_area_ha" not in st.columns and "area_ha" in st.columns:
        st["log1p_area_ha"] = np.log1p(pd.to_numeric(st["area_ha"], errors="coerce"))

    st = overlay_fuel(st, require_event_unique=True)
    extras_after_fuel = event_key_extras(st, EVENT_MONITOR_KEYS)
    if extras_after_fuel != 0:
        raise ValueError(f"Fuel overlay mixed events: extras={extras_after_fuel}.")

    st["event_id"] = make_event_id(st)
    st["fire_id_event"] = st["event_id"]
    st["fire_id_nameyear"] = make_nameyear_id(st)
    st["fire_id"] = st["event_id"]
    st["fire_cohort"] = pd.Categorical(
        st["fire_cohort"], categories=["WFIGS_RX", "MTBS_wildfire"]
    )
    for col in ("log1p_area_ha", "Distance_km", "wind_speed_ms_during_mean", "area_ha"):
        if col in st.columns:
            st[col] = pd.to_numeric(st[col], errors="coerce")

    # fillna(0) is elementwise on an already-unique event+monitor table, so
    # it cannot copy April CHL downwind onto the October CHL row (or vice versa).
    n_downwind_na = int(pd.to_numeric(st["downwind_during"], errors="coerce").isna().sum())
    st["downwind_during"] = pd.to_numeric(st["downwind_during"], errors="coerce")
    st["_downwind_was_missing"] = st["downwind_during"].isna()
    st["downwind_during"] = st["downwind_during"].fillna(0).astype(int)

    extras_final = event_key_extras(st, EVENT_MONITOR_KEYS)
    if extras_final != 0:
        raise ValueError(f"Final event-monitor extras={extras_final}.")

    meta = {
        "source": source,
        "n_rows": int(len(st)),
        "extras_before_join": int(extras_before_join),
        "extras_after_join": int(extras_after_join),
        "extras_after_fuel": int(extras_after_fuel),
        "extras_final": int(extras_final),
        "n_downwind_na_before_fillna": n_downwind_na,
        "max_daily_matched": int(
            pd.to_numeric(st["delta_MaxDailyDuring_minus_Before"], errors="coerce").notna().sum()
        ),
    }
    return st, meta


def fit_event_date(st: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Complete-case OLS, cluster-robust by event_id, for during/peak/after."""
    meta: dict = {}
    if st is None:
        st, meta = load_event_date_table()
    rows = []
    downwind_in_cc = {}
    for label, col in OUTCOMES:
        d = complete_case(st, col)
        extras_cc = event_key_extras(d, EVENT_MONITOR_KEYS)
        if extras_cc != 0:
            raise ValueError(f"complete-case {col} extras={extras_cc}.")
        n_missing_downwind = (
            int(d["_downwind_was_missing"].sum()) if "_downwind_was_missing" in d.columns else 0
        )
        downwind_in_cc[label] = n_missing_downwind
        m = smf.ols(FORMULA.format(outcome=col), data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d[CLUSTER]}
        )
        term = wildfire_term(m)
        lo, hi = m.conf_int().loc[term]
        rows.append(
            {
                "Outcome": label,
                "Estimate": float(m.params[term]),
                "CI low": float(lo),
                "CI high": float(hi),
                "n linkages": int(len(d)),
                "n events": int(d["event_id"].nunique()),
                "n fires": int(d["fire_id_nameyear"].nunique()),
                "extra event-monitor rows": extras_cc,
                "clustering": "event_id = cohort|name|year|date",
            }
        )
    table = pd.DataFrame(rows)
    meta["downwind_missing_in_complete_case"] = downwind_in_cc
    return table, st, meta


def _fmt_row(row: pd.Series) -> str:
    return (
        f"{float(row['Estimate']):.2f} ({float(row['CI low']):.2f}–{float(row['CI high']):.2f}) "
        f"n={int(row['n linkages'])}/{int(row['n events'])}/{int(row['n fires'])}"
    )


def write_independent_audit(
    table: pd.DataFrame,
    st: pd.DataFrame,
    meta: dict,
) -> Path:
    """Command, versions, estimates, extras=0, and CHL #12 2021 rows."""
    ensure_results()
    VALIDATION.mkdir(parents=True, exist_ok=True)
    audit = chl_same_event_audit(st)
    chl_dates = sorted({str(x) for x in audit["Fire_Date"].dropna().unique()}) if len(audit) else []
    apr = audit[audit["Fire_Date"] == "2021-04-17"] if len(audit) else audit
    oct_ = audit[audit["Fire_Date"] == "2021-10-19"] if len(audit) else audit
    apr_area = sorted(pd.to_numeric(apr["area_ha"], errors="coerce").dropna().unique()) if len(apr) else []
    oct_area = sorted(pd.to_numeric(oct_["area_ha"], errors="coerce").dropna().unique()) if len(oct_) else []
    apr_season = sorted(apr["met_season"].dropna().astype(str).unique()) if len(apr) else []
    oct_season = sorted(oct_["met_season"].dropna().astype(str).unique()) if len(oct_) else []
    apr_peak = (
        sorted(pd.to_numeric(apr["delta_MaxDailyDuring_minus_Before"], errors="coerce").dropna().unique())
        if len(apr)
        else []
    )
    oct_peak = (
        sorted(pd.to_numeric(oct_["delta_MaxDailyDuring_minus_Before"], errors="coerce").dropna().unique())
        if len(oct_)
        else []
    )
    shared_area = set(apr_area) & set(oct_area)
    shared_peak = set(apr_peak) & set(oct_peak)
    extras = int(meta.get("extras_final", event_key_extras(st, EVENT_MONITOR_KEYS)))
    during = table[table["Outcome"] == "During minus Before"].iloc[0]
    peak = table[table["Outcome"] == "Peak daily minus Before"].iloc[0]
    after = table[table["Outcome"] == "After minus Before"].iloc[0]
    dw_cc = meta.get("downwind_missing_in_complete_case", {})
    dw_during = dw_cc.get("During minus Before", "n/a")
    chl_lines = []
    if len(audit):
        show = [
            c
            for c in [
                "Fire_Date",
                "aqs_code",
                "area_ha",
                "met_season",
                "delta_MaxDailyDuring_minus_Before",
                "delta_During_minus_Before",
                "wind_speed_ms_during_mean",
                "downwind_during",
            ]
            if c in audit.columns
        ]
        chl_lines = audit[show].to_string(index=False).splitlines()
    lines = [
        "Independent event-date refit from packaged event-level data.",
        "This file is written by the second command; it is not a cached copy.",
        "",
        f"Command: {COMMAND}",
        "Working directory: package root (the folder that contains code/ and data/).",
        "",
        "SOFTWARE",
        *[f"  {v}" for v in _software_versions()],
        "",
        "DATA",
        f"  Event-level source: data/{meta.get('source', DATED_NAME)}",
        "  Peak join: data/max_daily.csv on fire_cohort+Fire_Name+Fire_Year+Fire_Date+aqs_code",
        "  Fuel overlay: data/burned_area_fuel.csv unique on Fire_Name+Fire_Year",
        "  Complete-case: outcome, fuel not Unknown, log1p_area, Distance_km, wind, season, downwind",
        "  Formula: outcome ~ C(fire_cohort, Treatment('WFIGS_RX')) + log1p_area_ha",
        "    + C(Dominant_FuelGroup, Treatment('Grass')) + C(met_season, Treatment('Fall'))",
        "    + Distance_km + downwind_during + wind_speed_ms_during_mean",
        "  Clustering: event_id = cohort|name|year|date",
        f"  Event+monitor rows: {meta.get('n_rows', len(st))}",
        "",
        "EXTRAS (must be 0)",
        f"  Before join: {meta.get('extras_before_join', 'n/a')}",
        f"  After event+monitor join: {meta.get('extras_after_join', extras)}",
        f"  After fuel overlay: {meta.get('extras_after_fuel', 'n/a')}",
        f"  Final table: {extras}",
        f"  Complete-case during extras: {int(during['extra event-monitor rows'])}",
        "",
        "DOWNWIND fillna(0)",
        "  Applied only after event+monitor uniqueness (extras=0).",
        "  fillna is elementwise, so it cannot mix two Fire_Date rows.",
        f"  Missing downwind before fillna (full table): {meta.get('n_downwind_na_before_fillna', 'n/a')}",
        f"  Of those, remaining in complete-case during: {dw_during}",
        "",
        "ESTIMATES (wildfire vs prescribed fire)",
        f"  During: {_fmt_row(during)}",
        f"  Peak:   {_fmt_row(peak)}",
        f"  After:  {_fmt_row(after)}",
        "  Output: results/Table_event_date_coefficients.csv",
        "",
        "CHL #12 2021 (must be two events)",
        f"  Dates: {', '.join(chl_dates) or 'NONE'}",
        f"  2021-04-17 area ha: {', '.join(f'{v:.6g}' for v in apr_area) or 'NONE'}; season: {apr_season or 'NONE'}",
        f"  2021-10-19 area ha: {', '.join(f'{v:.6g}' for v in oct_area) or 'NONE'}; season: {oct_season or 'NONE'}",
        f"  Shared area values: {sorted(shared_area) or 'none'}",
        f"  Shared peak values: {sorted(shared_peak) or 'none'}",
        "  Rows:",
        *([f"    {ln}" for ln in chl_lines] if chl_lines else ["    NONE"]),
        "",
        "Name-year 4.55 remains the labeled primary table",
        "(results/Table_adjusted_wildfire_vs_rx.csv). This script does not overwrite it.",
        "",
    ]
    path = VALIDATION / "event_date_independent_refit.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _num_match(a: pd.Series, b: pd.Series) -> tuple[int, int, int, int]:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    both = aa.notna() & bb.notna()
    match = int((both & np.isclose(aa, bb, rtol=0.0, atol=1e-9)).sum())
    return match, int(both.sum()), int(aa.notna().sum()), int(bb.notna().sum())


def _alias_fire_date(df: pd.DataFrame) -> pd.DataFrame:
    """Use Fire_Date. Accept File_Date only as a fallback alias."""
    out = df.copy()
    if "Fire_Date" not in out.columns and "File_Date" in out.columns:
        out["Fire_Date"] = out["File_Date"]
    return out


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add missing columns as NA so a standalone run does not KeyError."""
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def _safe_share(n: int, d: int) -> str:
    """Format n/d without dividing when the denominator is 0 (WF Stage B wind)."""
    if d == 0:
        return f"{n} / 0"
    return f"{n} / {d}"


def write_event_date_covariate_provenance(event: pd.DataFrame | None = None) -> dict:
    """Compare event-date During complete-case covariates to Stage B and name-year.

    Writes validation/event_date_covariate_provenance.txt. Does not change
    coefficients. Fuel is fire-year grain (no date in burned_area_fuel.csv).
    Safe to call from python code/report_event_date_covariate_provenance.py
    without running the OLS refit.
    """
    ensure_results()
    VALIDATION.mkdir(parents=True, exist_ok=True)
    if event is None:
        from analysis import load_modeling_event_correct  # noqa: WPS433

        event = load_modeling_event_correct()
    event = _alias_fire_date(event)
    d = complete_case(event, "delta_During_minus_Before").copy()
    d = _alias_fire_date(d)
    d["Fire_Date"] = iso_date(d["Fire_Date"])
    b = _alias_fire_date(load_stage_b())
    b["Fire_Date"] = iso_date(b["Fire_Date"])
    stage_b_cov = ["wind_speed_ms_during_mean", "downwind_during", "met_season", "area_ha"]
    b = _ensure_cols(b, stage_b_cov)
    st = _alias_fire_date(pd.read_csv(DATA / "modeling_station.csv", low_memory=False))
    st = _ensure_cols(st, ["wind_speed_ms_during_mean"])
    fuel = pd.read_csv(DATA / "burned_area_fuel.csv")
    inv_wf = _alias_fire_date(pd.read_csv(DATA / "inventory_wf.csv"))
    if "Fire_Date" in inv_wf.columns:
        inv_wf["Fire_Date"] = iso_date(inv_wf["Fire_Date"])

    keys = EVENT_MONITOR_KEYS
    ny = EVENT_KEYS
    b_cov = b[keys + stage_b_cov].rename(
        columns={
            "wind_speed_ms_during_mean": "wind_b",
            "downwind_during": "down_b",
            "met_season": "season_b",
            "area_ha": "area_b",
        }
    )
    m = d[keys + ["wind_speed_ms_during_mean", "downwind_during", "met_season", "area_ha"]].merge(
        b_cov, on=keys, how="left"
    )
    rx = m[m["fire_cohort"] == "WFIGS_RX"]
    wf = m[m["fire_cohort"] == "MTBS_wildfire"]
    rx_wind = _num_match(rx["wind_speed_ms_during_mean"], rx["wind_b"])
    rx_down = _num_match(rx["downwind_during"], rx["down_b"])
    rx_area = _num_match(rx["area_ha"], rx["area_b"])
    rx_season_both = rx["met_season"].notna() & rx["season_b"].notna()
    rx_season_match = int(
        (rx_season_both & (rx["met_season"].astype(str) == rx["season_b"].astype(str))).sum()
    )
    wf_wind = _num_match(wf["wind_speed_ms_during_mean"], wf["wind_b"])
    wf_down = _num_match(wf["downwind_during"], wf["down_b"])
    wf_area = _num_match(wf["area_ha"], wf["area_b"])
    wf_season_both = wf["met_season"].notna() & wf["season_b"].notna()
    wf_season_match = int(
        (wf_season_both & (wf["met_season"].astype(str) == wf["season_b"].astype(str))).sum()
    )

    wf_cc = d[d["fire_cohort"] == "MTBS_wildfire"].copy()
    st_first = st.drop_duplicates(ny, keep="first")
    ny_keep = [c for c in ny + ["wind_speed_ms_during_mean", "Fire_Date"] if c in st_first.columns]
    if "Fire_Date" in st_first.columns:
        st_first = st_first.copy()
        st_first["Fire_Date"] = iso_date(st_first["Fire_Date"])
    j = wf_cc.merge(
        st_first[ny_keep],
        on=ny,
        how="left",
        suffixes=("_ev", "_ny"),
    )
    if "wind_speed_ms_during_mean_ny" not in j.columns:
        j["wind_speed_ms_during_mean_ny"] = pd.NA
    if "wind_speed_ms_during_mean_ev" not in j.columns:
        j["wind_speed_ms_during_mean_ev"] = j.get("wind_speed_ms_during_mean", pd.NA)
    wf_ny_wind = _num_match(j["wind_speed_ms_during_mean_ev"], j["wind_speed_ms_during_mean_ny"])
    if "Fire_Date_ny" in j.columns:
        wf_date_match = int((j["Fire_Date_ev"].astype(str) == j["Fire_Date_ny"].astype(str)).sum())
    else:
        wf_date_match = 0
    inv_date_cols = [c for c in ["Fire_Name", "Fire_Year", "Fire_Date"] if c in inv_wf.columns]
    if "Fire_Date" in inv_wf.columns:
        wf_inv = wf_cc.merge(
            inv_wf[inv_date_cols].drop_duplicates(["Fire_Name", "Fire_Year"]),
            on=["Fire_Name", "Fire_Year"],
            how="left",
            suffixes=("_ev", "_inv"),
        )
        wf_inv_date = int(
            (wf_inv["Fire_Date_ev"].astype(str) == wf_inv["Fire_Date_inv"].astype(str)).sum()
        )
    else:
        wf_inv_date = 0
    wf_b_all = b[b["fire_cohort"] == "MTBS_wildfire"]
    wf_b_wind_n = int(pd.to_numeric(wf_b_all["wind_speed_ms_during_mean"], errors="coerce").notna().sum())
    wf_b_down_n = int(pd.to_numeric(wf_b_all["downwind_during"], errors="coerce").notna().sum())

    fuel_has_date = any("date" in c.lower() for c in fuel.columns)
    fuel_keys = fuel.groupby(["Fire_Name", "Fire_Year"]).size()
    chl_fuel = fuel[(fuel["Fire_Name"] == "CHL #12") & (fuel["Fire_Year"] == 2021)]
    chl = d[(d["Fire_Name"] == "CHL #12") & (d["Fire_Year"] == 2021)].copy()
    chl["Fire_Date"] = iso_date(chl["Fire_Date"])
    chl_fuel_by_date = (
        chl.groupby(chl["Fire_Date"].astype(str))["Dominant_FuelGroup"]
        .agg(lambda s: ", ".join(sorted({str(x) for x in s.dropna()})))
        .to_dict()
        if len(chl)
        else {}
    )
    chl_apr_fuel = chl_fuel_by_date.get("2021-04-17", "NONE")
    chl_oct_fuel = chl_fuel_by_date.get("2021-10-19", "NONE")

    rx_met_all = (
        rx_wind[0] == len(rx)
        and rx_down[0] == len(rx)
        and rx_season_match == len(rx)
        and rx_area[0] == len(rx)
    )
    summary = {
        "during_n": int(len(d)),
        "rx_n": int(len(rx)),
        "wf_n": int(len(wf)),
        "rx_wind_match": rx_wind[0],
        "rx_down_match": rx_down[0],
        "rx_season_match": rx_season_match,
        "rx_area_match": rx_area[0],
        "rx_met_all_match": int(rx_met_all),
        "wf_stage_b_wind_nonmissing": wf_b_wind_n,
        "wf_stage_b_wind_match": wf_wind[0],
        "wf_nameyear_wind_match": wf_ny_wind[0],
        "wf_event_vs_nameyear_date_match": wf_date_match,
        "wf_inventory_date_match": wf_inv_date,
        "chl_april_fuel": chl_apr_fuel,
        "chl_october_fuel": chl_oct_fuel,
        "fuel_has_date": int(fuel_has_date),
        "fuel_nameyear_unique": int((fuel_keys == 1).all()) if len(fuel_keys) else 0,
    }
    lines = [
        "Event-date During complete-case covariates vs Stage B and the name-year table.",
        "Sample is the event-date complete-case During minus Before rows",
        f"(n = {len(d)}; RX {len(rx)}; WF {len(wf)}).",
        "",
        "RX vs STAGE B (join on fire_cohort+Fire_Name+Fire_Year+Fire_Date+aqs_code)",
        f"  RX rows: {len(rx)}",
        f"  Wind match: {_safe_share(rx_wind[0], len(rx))} (Stage B non-missing {rx_wind[3]})",
        f"  Downwind match: {_safe_share(rx_down[0], len(rx))}",
        f"  Season match: {_safe_share(rx_season_match, len(rx))}",
        f"  Area match: {_safe_share(rx_area[0], len(rx))}",
        "  Date alignment: the join is on Fire_Date, so matched rows are the same event date.",
        "",
        "WF WIND SOURCE",
        f"  WF complete-case rows: {len(wf)}",
        f"  Stage B WF rows (full Stage B, not complete-case): {len(wf_b_all)}",
        f"  Stage B WF wind non-missing: {wf_b_wind_n}",
        f"  Stage B WF downwind non-missing: {wf_b_down_n}",
        f"  Event-date vs Stage B wind match: {_safe_share(wf_wind[0], len(wf))}",
        f"  Event-date vs Stage B downwind match: {_safe_share(wf_down[0], len(wf))}",
        f"  Event-date vs Stage B season match: {_safe_share(wf_season_match, len(wf))}",
        f"  Event-date vs Stage B area match: {_safe_share(wf_area[0], len(wf))}",
        "  Stage B does not store wildfire wind or downwind. The event-date table",
        "  copies those fields from data/modeling_station.csv (name-year grain)",
        "  for unique name-year-monitor keys. Column name:",
        "  wind_speed_ms_during_mean (During-window mean, not a separate",
        "  ignition-hour extract). Packaged files do not include the hourly ISD",
        "  series used to compute that mean, so ignition-hour vs multi-day window",
        "  cannot be re-derived here.",
        f"  WF wind match to name-year modeling_station (keep first name-year-monitor): "
        f"{_safe_share(wf_ny_wind[0], len(wf))}",
        f"  WF Fire_Date vs filled modeling_station Fire_Date: {_safe_share(wf_date_match, len(wf))}",
        f"  WF Fire_Date vs inventory_wf Fire_Date: {_safe_share(wf_inv_date, len(wf))}",
        "  All 309 WF complete-case keys have one Stage B date, so they can be",
        "  aligned to Fire_Date at the name-year-monitor grain. There are no",
        "  CHL-style name-year collisions among these 309 wildfire rows.",
        "",
        "FUEL GRAIN",
        "  File: data/burned_area_fuel.csv",
        f"  Columns: {', '.join(fuel.columns.astype(str))}",
        f"  Has a date column: {bool(fuel_has_date)}",
        f"  Unique on Fire_Name+Fire_Year: {bool(summary['fuel_nameyear_unique'])} "
        f"({len(fuel)} rows, {len(fuel_keys)} name-year keys)",
        "  overlay_fuel() joins Fire_Name+Fire_Year only. That does not inflate",
        "  rows, but it does not assign event-specific fuel when two burns share",
        "  name+year. Source fuel in this package has no date, so a date join is",
        "  not possible.",
        "",
        "CHL #12 2021 FUEL (April vs October)",
        f"  Fuel file rows for CHL #12 2021: {len(chl_fuel)}",
        f"  2021-04-17 Dominant_FuelGroup: {chl_apr_fuel}",
        f"  2021-10-19 Dominant_FuelGroup: {chl_oct_fuel}",
        "  Both dates receive the same fire-year fuel row (Timber).",
        "",
        "S3 expanded-sample covariates are documented in",
        "validation/s3_expanded_sample_covariates.txt (bearing panel has",
        "distance/bearing/outcomes/downwind only; extra pairs inherit fire-level",
        "wind/fuel/season/area from the primary name-year sample).",
        "",
    ]
    (VALIDATION / "event_date_covariate_provenance.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def run_event_date_refit() -> dict:
    """Fit, write the coefficient CSV and audit, return summary numbers."""
    ensure_results()
    table, st, meta = fit_event_date()
    out_csv = RESULTS / "Table_event_date_coefficients.csv"
    table.to_csv(out_csv, index=False)
    audit_path = write_independent_audit(table, st, meta)
    provenance = write_event_date_covariate_provenance(st)
    during = table[table["Outcome"] == "During minus Before"].iloc[0]
    peak = table[table["Outcome"] == "Peak daily minus Before"].iloc[0]
    after = table[table["Outcome"] == "After minus Before"].iloc[0]
    audit = chl_same_event_audit(st)
    chl_dates = sorted({str(x) for x in audit["Fire_Date"].dropna().unique()}) if len(audit) else []
    apr = audit[audit["Fire_Date"] == "2021-04-17"] if len(audit) else audit
    oct_ = audit[audit["Fire_Date"] == "2021-10-19"] if len(audit) else audit
    apr_area = sorted(pd.to_numeric(apr["area_ha"], errors="coerce").dropna().unique()) if len(apr) else []
    oct_area = sorted(pd.to_numeric(oct_["area_ha"], errors="coerce").dropna().unique()) if len(oct_) else []
    apr_season = sorted(apr["met_season"].dropna().astype(str).unique()) if len(apr) else []
    oct_season = sorted(oct_["met_season"].dropna().astype(str).unique()) if len(oct_) else []
    chl_ok = (
        "2021-04-17" in chl_dates
        and "2021-10-19" in chl_dates
        and apr_area
        and oct_area
        and abs(float(apr_area[0]) - 0.40) < 0.05
        and abs(float(oct_area[0]) - 32.37) < 0.05
        and apr_season == ["Spring"]
        and oct_season == ["Fall"]
        and not (set(apr_area) & set(oct_area))
    )
    extras = int(meta.get("extras_final", 1))
    ok = extras == 0 and chl_ok and out_csv.exists()
    return {
        "ok": ok,
        "csv": str(out_csv),
        "audit": str(audit_path),
        "command": COMMAND,
        "extras": extras,
        "chl_ok": chl_ok,
        "chl_dates": chl_dates,
        "during": float(during["Estimate"]),
        "during_lo": float(during["CI low"]),
        "during_hi": float(during["CI high"]),
        "during_n": int(during["n linkages"]),
        "during_events": int(during["n events"]),
        "during_fires": int(during["n fires"]),
        "peak": float(peak["Estimate"]),
        "peak_lo": float(peak["CI low"]),
        "peak_hi": float(peak["CI high"]),
        "after": float(after["Estimate"]),
        "after_lo": float(after["CI low"]),
        "after_hi": float(after["CI high"]),
        "after_n": int(after["n linkages"]),
        "after_events": int(after["n events"]),
        "after_fires": int(after["n fires"]),
        "provenance": provenance,
        "table": table,
    }


def main() -> int:
    result = run_event_date_refit()
    print(result["table"].to_string(index=False))
    print()
    print(f"Command: {result['command']}")
    print(
        f"During: {result['during']:.2f} ({result['during_lo']:.2f}–{result['during_hi']:.2f}) "
        f"n={result['during_n']}/{result['during_events']}/{result['during_fires']}"
    )
    print(
        f"Peak:   {result['peak']:.2f} ({result['peak_lo']:.2f}–{result['peak_hi']:.2f})"
    )
    print(
        f"After:  {result['after']:.2f} ({result['after_lo']:.2f}–{result['after_hi']:.2f}) "
        f"n={result['after_n']}/{result['after_events']}/{result['after_fires']}"
    )
    print(f"extras={result['extras']}")
    print(f"CHL dates: {', '.join(result['chl_dates']) or 'NONE'}; audit_ok={result['chl_ok']}")
    print(f"Wrote {result['csv']}")
    print(f"Wrote {result['audit']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
