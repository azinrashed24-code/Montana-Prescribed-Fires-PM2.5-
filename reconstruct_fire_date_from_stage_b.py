"""Optional regeneration of the dated event+monitor table from Stage B.

    python code/reconstruct_fire_date_from_stage_b.py

data/modeling_station_with_date.csv is shipped in this package. A clean
checkout does not need this script. Run it only to regenerate that file
from data/stage_b_daily.csv. It does not overwrite data/modeling_station.csv
and it is not a required first step.

Reads data/stage_b_daily.csv (must already have Fire_Date). Does not fill
dates from inventory or max_daily. Does not delete files.

Writes:
  data/modeling_station_with_date.csv (one row per event+monitor)
  data/modeling_station_ambiguous_dates.csv
  validation/stage_b_fire_date_reconstruction.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analysis import (  # noqa: E402
    EVENT_KEYS,
    EVENT_MONITOR_KEYS,
    build_event_monitor_table_from_stage_b,
    diagnose_stage_b_dates,
    load_stage_b,
)
from paths import DATA, VALIDATION, ensure_results  # noqa: E402


def ambiguous_sidecar(st: pd.DataFrame, b: pd.DataFrame, diag: dict) -> pd.DataFrame:
    keys = diag["ambiguous_keys"]
    if keys.empty:
        return pd.DataFrame(
            columns=EVENT_KEYS + ["n_dates", "Fire_Date_list", "n_modeling_copies"]
        )
    copies = st.groupby(EVENT_KEYS).size().rename("n_modeling_copies").reset_index()
    dates = (
        b.merge(keys[EVENT_KEYS], on=EVENT_KEYS)
        .groupby(EVENT_KEYS)["Fire_Date"]
        .agg(lambda s: ";".join(sorted({str(x) for x in s.dropna()})))
        .rename("Fire_Date_list")
        .reset_index()
    )
    out = keys.merge(dates, on=EVENT_KEYS, how="left").merge(copies, on=EVENT_KEYS, how="left")
    return out.sort_values(EVENT_KEYS).reset_index(drop=True)


def write_diagnosis(diag: dict, event_rows: int) -> None:
    ensure_results()
    VALIDATION.mkdir(parents=True, exist_ok=True)
    chl_pair_lines = []
    pairs = diag["chl_pairs"]
    if pairs is not None and len(pairs):
        for _, r in pairs.iterrows():
            chl_pair_lines.append(f"    {r['Fire_Date']}  {r['aqs_code']}  n={int(r['n'])}")
    amb = diag["ambiguous_keys"]
    amb_lines = []
    if amb is not None and len(amb):
        for _, r in amb.sort_values(EVENT_KEYS).iterrows():
            amb_lines.append(
                f"    {r['fire_cohort']} | {r['Fire_Name']} | {int(r['Fire_Year'])} | "
                f"{r['aqs_code']}: {int(r['n_dates'])} dates"
            )
    lines = [
        "Stage B as the Fire_Date source for the event-date modeling table.",
        "",
        "Name-year grain: fire_cohort + Fire_Name + Fire_Year + aqs_code.",
        "  File: data/modeling_station.csv (8,615 rows). This reconstruct",
        "  script does not overwrite that file. load_modeling() ignores Fire_Date.",
        "Event-date grain: the name-year grain plus Fire_Date.",
        "  File: data/modeling_station_with_date.csv (shipped; this script",
        "  optionally regenerates it and does not mutate modeling_station.csv).",
        "Dates are taken from data/stage_b_daily.csv only.",
        "Inventory and max_daily are not used to fill missing dates.",
        "",
        "MATCH RATES",
        f"  modeling_station rows: {diag['modeling_rows']}",
        f"  modeling_station name-year-monitor keys: {diag['modeling_keys']}",
        f"  Stage B rows: {diag['stage_b_rows']}",
        f"  Stage B name-year-monitor keys: {diag['stage_b_nameyear_monitor_keys']}",
        f"  Stage B event+monitor keys: {diag['stage_b_event_monitor_keys']}",
        f"  Stage B Fire_Date missing: {diag['stage_b_fire_date_missing']}",
        f"  modeling rows whose name-year-monitor key is in Stage B: "
        f"{diag['modeling_rows_matching_stage_b']} / {diag['modeling_rows']}",
        "",
        "DISTINCT Fire_Date COUNTS IN STAGE B (modeling keys)",
        f"  0 dates: {diag['keys_with_0_dates']}",
        f"  1 date: {diag['keys_with_1_date']}",
        f"  >1 dates: {diag['keys_with_gt1_dates']}",
        "",
        "DISTINCT Fire_Date COUNTS IN STAGE B (modeling rows)",
        f"  0 dates: {diag['rows_with_0_dates']}",
        f"  1 date: {diag['rows_with_1_date']}",
        f"  >1 dates: {diag['rows_with_gt1_dates']}",
        "",
        "CONFLICTS (same name+year+monitor, two Stage B dates)",
        f"  Ambiguous keys: {diag['keys_with_gt1_dates']}",
        f"  Unambiguous keys (one Stage B date): {diag['keys_with_1_date']}",
        *amb_lines,
        "",
        "CHL #12 2021 IN STAGE B",
        f"  Dates: {', '.join(diag['chl_dates']) or 'NONE'}",
        f"  Monitors: {', '.join(diag['chl_monitors']) or 'NONE'}",
        f"  Stage B rows: {diag['chl_stage_b_rows']}",
        "  Both 2021-04-17 and 2021-10-19 are present for the same name-year.",
        *chl_pair_lines,
        "",
        "FILES WRITTEN",
        "  data/modeling_station.csv: not written. That is the name-year table.",
        f"  data/modeling_station_with_date.csv: {event_rows} event+monitor rows,",
        "  used by the event-date path only.",
        "  data/modeling_station_ambiguous_dates.csv: name-year-monitor keys",
        "  with more than one Stage B date.",
        "  The name-year 8,615-row table is not split and is not overwritten.",
        "",
    ]
    (VALIDATION / "stage_b_fire_date_reconstruction.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    ensure_results()
    st = pd.read_csv(DATA / "modeling_station.csv", low_memory=False)
    st_for_diag = st.drop(columns=[c for c in ("Fire_Date", "Fire_Date_note") if c in st.columns])
    b = load_stage_b()
    diag = diagnose_stage_b_dates(st_for_diag, b)

    event = build_event_monitor_table_from_stage_b()
    event.to_csv(DATA / "modeling_station_with_date.csv", index=False)

    side = ambiguous_sidecar(st_for_diag, b, diag)
    side.to_csv(DATA / "modeling_station_ambiguous_dates.csv", index=False)

    write_diagnosis(diag, len(event))
    print(
        "modeling_station.csv: not overwritten "
        f"({len(st)} name-year rows remain as-is)."
    )
    print(
        f"modeling_station_with_date.csv: {len(event)} event+monitor rows; "
        f"unique={event[EVENT_MONITOR_KEYS].drop_duplicates().shape[0]}"
    )
    print(
        f"Stage B match: {diag['modeling_rows_matching_stage_b']}/"
        f"{diag['modeling_rows']} rows; keys 0/1/>1 dates = "
        f"{diag['keys_with_0_dates']}/{diag['keys_with_1_date']}/"
        f"{diag['keys_with_gt1_dates']}"
    )
    print(f"CHL #12 2021 Stage B dates: {', '.join(diag['chl_dates']) or 'NONE'}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
