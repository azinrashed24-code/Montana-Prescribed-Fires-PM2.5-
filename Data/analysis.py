"""Shared loaders and the primary clustered OLS recipe.

Two labeled paths
  A. Name-year fire identifier (load_modeling, primary):
     fire_id = cohort|name|year; many-to-many-capable join on
     fire_cohort + Fire_Name + Fire_Year + aqs_code. This path does
     not use Fire_Date. Reproducible name-year results:
     4.55 / 7.68 / 11.01, n = 4,186 / 1,547.
  B. Event-date identifier (load_modeling_event_correct):
     uniqueness is fire_cohort + Fire_Name + Fire_Year + Fire_Date +
     aqs_code. event_id if used in Python is derived as those fields
     joined, not a column in the CSV. Dates and collision covariates
     come from Stage B (data/stage_b_daily.csv /
     data/modeling_station_with_date.csv).
     Clustering is by event_id so the two 2021 CHL #12 dates are
     separate clusters. Coefficients live in
     results/Table_S_event_id_sensitivity.csv.

load_modeling_unique_keys() remains a name+year+monitor collapse
(~4.47, n = 4,156 / 1,547) and is not the event-date model.

Complete case: non-missing outcome, fuel (not Unknown), log1p_area,
distance, wind speed, season, and downwind. No extra Distance_km > 0 filter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from paths import DATA  # noqa: E402

FORMULA = (
    "{outcome} ~ C(fire_cohort, Treatment(reference='WFIGS_RX'))"
    " + log1p_area_ha"
    " + C(Dominant_FuelGroup, Treatment(reference='Grass'))"
    " + C(met_season, Treatment(reference='Fall'))"
    " + Distance_km"
    " + downwind_during"
    " + wind_speed_ms_during_mean"
)

FORMULA_INTERACTION = (
    "{outcome} ~ C(fire_cohort, Treatment(reference='WFIGS_RX')) * log1p_area_ha"
    " + C(Dominant_FuelGroup, Treatment(reference='Grass'))"
    " + C(met_season, Treatment(reference='Fall'))"
    " + Distance_km"
    " + downwind_during"
    " + wind_speed_ms_during_mean"
)

OUTCOMES = [
    ("During minus Before", "delta_During_minus_Before"),
    ("Peak daily minus Before", "delta_MaxDailyDuring_minus_Before"),
    ("After minus Before", "delta_After_minus_Before"),
]

REQ = [
    "log1p_area_ha",
    "Distance_km",
    "wind_speed_ms_during_mean",
    "Dominant_FuelGroup",
    "met_season",
    "downwind_during",
]

# Name-year fire–monitor grain (primary). Clustering on the 4.55 path
# stays fire_id = cohort|name|year and does not include Fire_Date.
EVENT_KEYS = ["fire_cohort", "Fire_Name", "Fire_Year", "aqs_code"]
NAME_YEAR_MONITOR = ["Fire_Name", "Fire_Year", "aqs_code"]
# Event-correct grain: one burn date is one event.
EVENT_ID_PARTS = ["fire_cohort", "Fire_Name", "Fire_Year", "Fire_Date"]
EVENT_MONITOR_KEYS = ["fire_cohort", "Fire_Name", "Fire_Year", "Fire_Date", "aqs_code"]
EVENT_DATE_KEYS = ["Fire_Name", "Fire_Year", "Fire_Date"]


def event_key_extras(df: pd.DataFrame, keys: list[str] | None = None) -> int:
    """Rows beyond distinct event keys (0 means the table is unique)."""
    use = keys or EVENT_KEYS
    if df.empty:
        return 0
    present = [c for c in use if c in df.columns]
    if len(present) < len(use):
        return 0
    return int(len(df) - df[present].drop_duplicates().shape[0])


def require_unique_keys(
    df: pd.DataFrame,
    keys: list[str],
    label: str,
) -> pd.DataFrame:
    """Raise if keys are not unique. Do not collapse collisions with keep='first'."""
    extras = event_key_extras(df, keys)
    if extras != 0:
        raise ValueError(
            f"{label} is not unique on {keys}: {extras} extra rows. "
            "Refusing keep='first' collapse."
        )
    return df


def iso_date(series: pd.Series) -> pd.Series:
    """Normalize dates to ISO YYYY-MM-DD strings (NA stays NA)."""
    parsed = pd.to_datetime(series, errors="coerce")
    out = parsed.dt.strftime("%Y-%m-%d")
    return out.astype("object").where(parsed.notna(), other=pd.NA)


def make_event_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["fire_cohort"].astype(str)
        + "|"
        + df["Fire_Name"].astype(str)
        + "|"
        + df["Fire_Year"].astype(str)
        + "|"
        + df["Fire_Date"].astype(str)
    )


def make_nameyear_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["fire_cohort"].astype(str)
        + "|"
        + df["Fire_Name"].astype(str)
        + "|"
        + df["Fire_Year"].astype(str)
    )


def _read_max_daily() -> pd.DataFrame:
    """Primary-path peak column only. Fire_Date is ignored on the 4.55 path."""
    return pd.read_csv(DATA / "max_daily.csv", low_memory=False)[
        [
            "fire_cohort",
            "Fire_Name",
            "Fire_Year",
            "aqs_code",
            "delta_MaxDailyDuring_minus_Before",
        ]
    ]


def _read_fuel() -> pd.DataFrame:
    fuel = pd.read_csv(DATA / "burned_area_fuel.csv")[
        ["Fire_Name", "Fire_Year", "dominant_fuel"]
    ].dropna(subset=["dominant_fuel"])
    require_unique_keys(fuel, ["Fire_Name", "Fire_Year"], "data/burned_area_fuel.csv")
    return fuel.rename(columns={"dominant_fuel": "_ba_fuel"})


def overlay_fuel(st: pd.DataFrame, *, require_event_unique: bool = False) -> pd.DataFrame:
    """Attach LANDFIRE fuel after the modeling grain is already set.

    burned_area_fuel.csv has no Fire_Date. It is unique on Fire_Name +
    Fire_Year, so fuel is fire-year grain, not event-specific. Two burns
    that share name+year (CHL #12 2021 April vs October) receive the same
    fuel row. Overlay onto an already-unique event-monitor table so the
    merge cannot cartesian-expand. Fuel itself is not an event key.
    """
    if require_event_unique:
        extras = event_key_extras(st, EVENT_MONITOR_KEYS)
        if extras != 0:
            raise ValueError(
                f"Fuel overlay expects unique event-monitor keys; extras={extras}."
            )
    fuel = _read_fuel()
    n0 = len(st)
    out = st.merge(fuel, on=["Fire_Name", "Fire_Year"], how="left")
    if len(out) != n0:
        raise ValueError(f"Fuel overlay inflated {n0} -> {len(out)}")
    out["Dominant_FuelGroup"] = out["_ba_fuel"]
    return out.drop(columns=["_ba_fuel"])


DATED_MODELING_NAME = "modeling_station_with_date.csv"


def load_stage_b() -> pd.DataFrame:
    """Read Stage B. Fire_Date must already be on the file.

    Does not fill dates from inventory or max_daily.
    """
    b = pd.read_csv(DATA / "stage_b_daily.csv", low_memory=False)
    if "Fire_Date" not in b.columns and "File_Date" in b.columns:
        b["Fire_Date"] = b["File_Date"]
    if "Fire_Date" not in b.columns:
        raise ValueError("data/stage_b_daily.csv has no Fire_Date column")
    b["Fire_Date"] = iso_date(b["Fire_Date"])
    missing = int(b["Fire_Date"].isna().sum())
    if missing:
        raise ValueError(
            f"Stage B has {missing} missing Fire_Date values; "
            "not filling from inventory or max_daily."
        )
    return b


def stage_b_n_dates(b: pd.DataFrame | None = None) -> pd.Series:
    """Distinct Stage B Fire_Date counts per name-year-monitor key."""
    if b is None:
        b = load_stage_b()
    return b.groupby(EVENT_KEYS, dropna=False)["Fire_Date"].nunique().rename("n_dates")


def diagnose_stage_b_dates(st: pd.DataFrame | None = None, b: pd.DataFrame | None = None) -> dict:
    """Match rates of Stage B as the date source for modeling_station."""
    if st is None:
        st = pd.read_csv(DATA / "modeling_station.csv", low_memory=False)
        st = st.drop(columns=[c for c in ("Fire_Date", "Fire_Date_note") if c in st.columns])
    if b is None:
        b = load_stage_b()
    n_dates = stage_b_n_dates(b)
    st_keys = st[EVENT_KEYS].drop_duplicates()
    keyed = st_keys.merge(n_dates, on=EVENT_KEYS, how="left")
    keyed["n_dates"] = keyed["n_dates"].fillna(0).astype(int)
    st_rows = st.merge(n_dates, on=EVENT_KEYS, how="left")
    st_rows["n_dates"] = st_rows["n_dates"].fillna(0).astype(int)
    matched_rows = int((st_rows["n_dates"] > 0).sum())
    chl = b[(b["Fire_Name"] == "CHL #12") & (b["Fire_Year"] == 2021)].copy()
    chl_dates = sorted({str(x) for x in chl["Fire_Date"].dropna().unique()})
    chl_monitors = sorted({str(x) for x in chl["aqs_code"].dropna().unique()})
    chl_pairs = (
        chl.groupby(["Fire_Date", "aqs_code"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["Fire_Date", "aqs_code"])
    )
    multi = n_dates[n_dates > 1].reset_index()
    return {
        "modeling_rows": int(len(st)),
        "modeling_keys": int(len(st_keys)),
        "stage_b_rows": int(len(b)),
        "stage_b_nameyear_monitor_keys": int(b[EVENT_KEYS].drop_duplicates().shape[0]),
        "stage_b_event_monitor_keys": int(b[EVENT_MONITOR_KEYS].drop_duplicates().shape[0]),
        "stage_b_fire_date_missing": int(b["Fire_Date"].isna().sum()),
        "modeling_rows_matching_stage_b": matched_rows,
        "keys_with_0_dates": int((keyed["n_dates"] == 0).sum()),
        "keys_with_1_date": int((keyed["n_dates"] == 1).sum()),
        "keys_with_gt1_dates": int((keyed["n_dates"] > 1).sum()),
        "rows_with_0_dates": int((st_rows["n_dates"] == 0).sum()),
        "rows_with_1_date": int((st_rows["n_dates"] == 1).sum()),
        "rows_with_gt1_dates": int((st_rows["n_dates"] > 1).sum()),
        "chl_dates": chl_dates,
        "chl_monitors": chl_monitors,
        "chl_stage_b_rows": int(len(chl)),
        "chl_pairs": chl_pairs,
        "ambiguous_keys": multi,
        "n_dates": n_dates,
    }


def add_fire_date_to_modeling_station(
    st: pd.DataFrame, b: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Add Fire_Date on unique name-year-monitor keys. Ambiguous keys stay NA.

    Does not split the 8,615-row name-year table. Both dates for a collision
    key are stored in Fire_Date_note.
    """
    if b is None:
        b = load_stage_b()
    out = st.drop(columns=[c for c in ("Fire_Date", "Fire_Date_note") if c in st.columns]).copy()
    n_dates = stage_b_n_dates(b)
    single = (
        b.merge(n_dates[n_dates == 1].reset_index()[EVENT_KEYS], on=EVENT_KEYS)[
            EVENT_KEYS + ["Fire_Date"]
        ].drop_duplicates(EVENT_KEYS)
    )
    notes = (
        b.groupby(EVENT_KEYS, dropna=False)["Fire_Date"]
        .agg(lambda s: ";".join(sorted({str(x) for x in s.dropna()})))
        .rename("Fire_Date_note")
        .reset_index()
    )
    out = out.merge(single, on=EVENT_KEYS, how="left")
    out = out.merge(notes, on=EVENT_KEYS, how="left")
    return out


def build_event_monitor_table_from_stage_b() -> pd.DataFrame:
    """One row per Stage B event+monitor, with same-event covariates.

    Unique name-year-monitor keys keep modeling_station covariates (including
    wildfire wind, which Stage B does not carry) and take Fire_Date and peak
    from Stage B. Keys with two Stage B dates are replaced by the Stage B
    event+monitor rows so area, season, wind, downwind, distance, and
    outcomes belong to one date.
    """
    st = pd.read_csv(DATA / "modeling_station.csv", low_memory=False)
    st = st.drop(columns=[c for c in ("Fire_Date", "Fire_Date_note") if c in st.columns])
    b = load_stage_b()
    n_dates = stage_b_n_dates(b)
    uniq_keys = n_dates[n_dates == 1].reset_index()[EVENT_KEYS]
    amb_keys = n_dates[n_dates > 1].reset_index()[EVENT_KEYS]

    date_u = b.merge(uniq_keys, on=EVENT_KEYS)[EVENT_KEYS + ["Fire_Date"]].copy()
    require_unique_keys(date_u, EVENT_KEYS, "Stage B unique-key dates")
    peak_cols = EVENT_KEYS + ["Fire_Date", "delta_MaxDailyDuring_minus_Before"]
    peak_u = b.merge(uniq_keys, on=EVENT_KEYS)[peak_cols].copy()
    require_unique_keys(peak_u, EVENT_MONITOR_KEYS, "Stage B unique-key peaks")

    uniq = st.merge(n_dates, on=EVENT_KEYS, how="left")
    uniq = uniq.loc[uniq["n_dates"] == 1].drop(columns=["n_dates"])
    uniq = uniq.merge(date_u, on=EVENT_KEYS, how="left")
    if "delta_MaxDailyDuring_minus_Before" in uniq.columns:
        uniq = uniq.drop(columns=["delta_MaxDailyDuring_minus_Before"])
    uniq = uniq.merge(peak_u, on=EVENT_MONITOR_KEYS, how="left")

    extra_cols = [c for c in st.columns if c not in b.columns and c not in EVENT_KEYS]
    # Name-year extras (e.g. wildfire wind) are one value per name-year-monitor.
    # Ambiguous keys still take event-level Stage B covariates for area/season/
    # wind/downwind/outcomes; extras from modeling_station are attached by
    # name-year-monitor only where Stage B does not carry them.
    st_first = st.drop_duplicates(EVENT_KEYS, keep="first")
    amb = b.merge(amb_keys, on=EVENT_KEYS)
    amb_extras = event_key_extras(amb, EVENT_MONITOR_KEYS)
    if amb_extras != 0:
        raise ValueError(
            f"Stage B ambiguous event+monitor rows are not unique: extras={amb_extras}"
        )
    if extra_cols:
        amb = amb.merge(st_first[EVENT_KEYS + extra_cols], on=EVENT_KEYS, how="left")
    amb["log1p_area_ha"] = np.log1p(pd.to_numeric(amb["area_ha"], errors="coerce"))

    cols: list[str] = []
    for frame in (uniq, amb):
        for col in frame.columns:
            if col not in cols:
                cols.append(col)
    pieces = []
    for frame in (uniq, amb):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            frame = frame.copy()
            for col in missing:
                frame[col] = pd.NA
        pieces.append(frame[cols])
    out = pd.concat(pieces, ignore_index=True)
    out["Fire_Date"] = iso_date(out["Fire_Date"])
    out = out.dropna(subset=["Fire_Date"]).copy()
    require_unique_keys(out, EVENT_MONITOR_KEYS, "Stage B event+monitor table")
    return out


def chl_same_event_audit(event: pd.DataFrame) -> pd.DataFrame:
    """CHL #12 2021 rows: area, season, wind, and peaks by date."""
    chl = event[(event["Fire_Name"] == "CHL #12") & (event["Fire_Year"] == 2021)].copy()
    if chl.empty:
        return chl
    chl["Fire_Date"] = iso_date(chl["Fire_Date"])
    keep = [
        c
        for c in [
            "Fire_Date",
            "aqs_code",
            "area_ha",
            "log1p_area_ha",
            "met_season",
            "Distance_km",
            "wind_speed_ms_during_mean",
            "downwind_during",
            "Dominant_FuelGroup",
            "delta_During_minus_Before",
            "delta_After_minus_Before",
            "delta_MaxDailyDuring_minus_Before",
        ]
        if c in chl.columns
    ]
    return chl[keep].sort_values(["Fire_Date", "aqs_code"]).reset_index(drop=True)


def _finalize_modeling(st: pd.DataFrame) -> pd.DataFrame:
    if "fire_id" not in st.columns:
        st["fire_id"] = make_nameyear_id(st)
    if "fire_id_nameyear" not in st.columns:
        st["fire_id_nameyear"] = make_nameyear_id(st)
    st["fire_cohort"] = pd.Categorical(
        st["fire_cohort"], categories=["WFIGS_RX", "MTBS_wildfire"]
    )
    # Primary recipe: missing downwind_during is coded 0 (not downwind).
    # None of those missing-alignment rows are in the 4,186 complete-case
    # sample; they already fail other complete-case filters. See
    # validation/downwind_missingness.txt.
    st["downwind_during"] = pd.to_numeric(st["downwind_during"], errors="coerce")
    st["_downwind_was_missing"] = st["downwind_during"].isna()
    st["downwind_during"] = st["downwind_during"].fillna(0).astype(int)
    for col in ("log1p_area_ha", "Distance_km", "wind_speed_ms_during_mean", "area_ha"):
        if col in st.columns:
            st[col] = pd.to_numeric(st[col], errors="coerce")
    return st


def load_modeling(*, unique_event_keys: bool = False) -> pd.DataFrame:
    """Load the daily modeling table.

    unique_event_keys=False (default, primary): the name-year join, including
    the many-to-many max_daily match. Fire_Date on modeling_station is
    ignored. That is the fillna(0) recipe that yields 4.55 / 7.68 / 11.01
    and n = 4,186 / 1,547. Collapsing keys changes n
    (see load_modeling_unique_keys).

    unique_event_keys=True: drop duplicate EVENT_KEYS in modeling_station and
    max_daily *before* the join so the merge cannot inflate 8,615 → 8,663.
    After that join, fire_cohort+Fire_Name+Fire_Year+aqs_code is unique.
    fire_id clustering remains cohort|name|year.
    """
    st = pd.read_csv(DATA / "modeling_station.csv", low_memory=False)
    max_daily = _read_max_daily()
    if unique_event_keys:
        st = st.drop_duplicates(EVENT_KEYS, keep="first")
        max_daily = max_daily.drop_duplicates(EVENT_KEYS, keep="first")
    st = st.merge(max_daily, on=EVENT_KEYS, how="left")
    st = overlay_fuel(st)
    if unique_event_keys:
        st = st.drop_duplicates(EVENT_KEYS, keep="first")
    return _finalize_modeling(st)


def load_modeling_unique_keys() -> pd.DataFrame:
    """Unique fire_cohort+name+year+monitor table. Not the primary spec."""
    return load_modeling(unique_event_keys=True)


def load_modeling_event_correct() -> pd.DataFrame:
    """Event-correct table: unique on fire_cohort + Fire_Name + Fire_Year + Fire_Date + aqs_code.

    event_id if used in Python is derived as those fields joined, not a
    column in the CSV. Clustering is by that derived event_id.

    Reads data/modeling_station_with_date.csv when present (shipped Stage B
    event+monitor table). If that file is missing, rebuilds the same table
    from data/stage_b_daily.csv. Reconstruct is optional regeneration, not
    a required first step. Does not look up dates from inventory. Fuel is
    overlaid after the event+monitor grain is unique.
    """
    dated = DATA / DATED_MODELING_NAME
    if dated.exists():
        st = pd.read_csv(dated, low_memory=False)
        if "Fire_Date" not in st.columns and "File_Date" in st.columns:
            st["Fire_Date"] = st["File_Date"]
        st["Fire_Date"] = iso_date(st["Fire_Date"])
    else:
        st = build_event_monitor_table_from_stage_b()
    st = st.dropna(subset=["Fire_Date"]).copy()
    require_unique_keys(
        st,
        EVENT_MONITOR_KEYS,
        "event-monitor table before peak join",
    )
    if "delta_MaxDailyDuring_minus_Before" not in st.columns:
        b = load_stage_b()
        peak = b[EVENT_MONITOR_KEYS + ["delta_MaxDailyDuring_minus_Before"]].copy()
        require_unique_keys(peak, EVENT_MONITOR_KEYS, "Stage B peak event+monitor")
        st = st.merge(peak, on=EVENT_MONITOR_KEYS, how="left")
        require_unique_keys(st, EVENT_MONITOR_KEYS, "event-monitor table after peak join")
    if "log1p_area_ha" not in st.columns and "area_ha" in st.columns:
        st["log1p_area_ha"] = np.log1p(pd.to_numeric(st["area_ha"], errors="coerce"))
    st = overlay_fuel(st, require_event_unique=True)
    st["event_id"] = make_event_id(st)
    st["fire_id_event"] = st["event_id"]
    st["fire_id_nameyear"] = make_nameyear_id(st)
    st["fire_id"] = st["event_id"]
    return _finalize_modeling(st)


def complete_case(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    d = df[df[outcome].notna()].copy()
    d = d[d["Dominant_FuelGroup"].notna()]
    d = d[d["Dominant_FuelGroup"] != "Unknown"]
    for col in REQ:
        d = d[d[col].notna()]
    return d


def complete_case_with_area(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    d = complete_case(df, outcome)
    d = d[d["area_ha"].notna() & (d["area_ha"] > 0)]
    return d


def fit_clustered(formula: str, data: pd.DataFrame, cluster: str = "fire_id"):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data[cluster]}
    )


def fit_path_rows(
    df: pd.DataFrame,
    path: str,
    clustering: str,
    extra_keys: list[str],
    cluster: str = "fire_id",
) -> list[dict]:
    """Complete-case clustered OLS for each outcome on one modeling path."""
    rows = []
    for label, col in OUTCOMES:
        d = complete_case(df, col)
        m = fit_clustered(FORMULA.format(outcome=col), d, cluster=cluster)
        term = wildfire_term(m)
        lo, hi = m.conf_int().loc[term]
        n_fires = (
            int(d["fire_id_nameyear"].nunique())
            if "fire_id_nameyear" in d.columns
            else int(d["fire_id"].nunique())
        )
        if "event_id" in d.columns:
            n_events = int(d["event_id"].nunique())
        else:
            n_events = n_fires
        rows.append(
            {
                "path": path,
                "Outcome": label,
                "Estimate": float(m.params[term]),
                "CI low": float(lo),
                "CI high": float(hi),
                "n linkages": int(len(d)),
                "n events": n_events,
                "n fires": n_fires,
                "extra event-monitor rows": event_key_extras(d, extra_keys),
                "clustering": clustering,
            }
        )
    return rows


def wildfire_term(model) -> str:
    for name in model.params.index:
        if "MTBS_wildfire" in name and ":" not in name:
            return name
    raise KeyError("wildfire main-effect term not found")


def interaction_term(model) -> str:
    for name in model.params.index:
        if "MTBS_wildfire" in name and ":" in name and "log1p_area_ha" in name:
            return name
        if "log1p_area_ha" in name and ":" in name:
            return name
    raise KeyError("fire-type × log(1+area) term not found")


def format_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"
