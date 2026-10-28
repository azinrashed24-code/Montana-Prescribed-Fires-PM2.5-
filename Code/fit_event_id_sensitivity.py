"""Event-date identifier from the Stage B event+monitor reconstruction.

    python code/fit_event_id_sensitivity.py

Does not replace the name-year 4.55 / 7.68 / 11.01 table. Reads
data/modeling_station_with_date.csv (shipped; rebuilt from Stage B if
absent). Writes results/Table_S_event_id_sensitivity.csv and
results/event_vs_name_year_comparison.txt.
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
    chl_same_event_audit,
    complete_case,
    event_key_extras,
    fit_path_rows,
    load_modeling,
    load_modeling_event_correct,
)
from paths import RESULTS, ensure_results  # noqa: E402

# Name-year path (reproducible alternate). Do not change these.
EXPECTED_NAMEYEAR = {
    "during": 4.55,
    "during_lo": 2.91,
    "during_hi": 6.19,
    "during_n": 4186,
    "during_fires": 1547,
    "peak": 7.68,
    "peak_lo": 5.24,
    "peak_hi": 10.12,
    "after": 11.01,
    "after_lo": 6.61,
    "after_hi": 15.40,
    "after_n": 4191,
    "after_fires": 1548,
}

# Event-corrected path after same-event Stage B reconstruction.
EXPECTED_EVENT = {
    "during": 4.45,
    "during_lo": 2.82,
    "during_hi": 6.09,
    "during_n": 4165,
    "during_events": 1549,
    "during_fires": 1548,
    "peak": 7.57,
    "peak_lo": 5.14,
    "peak_hi": 10.00,
    "after": 11.00,
    "after_lo": 6.60,
    "after_hi": 15.40,
    "after_n": 4170,
    "after_events": 1550,
    "after_fires": 1549,
    "join_extras": 0,
    "chl_april": "2021-04-17",
    "chl_october": "2021-10-19",
    "chl_april_ha": 0.404686,
    "chl_october_ha": 32.37488,
}


def event_id_sensitivity_table() -> tuple:
    """Refit both paths from data/ CSVs. Does not read the results CSV."""
    primary = load_modeling()
    event = load_modeling_event_correct()
    rows = []
    rows.extend(
        fit_path_rows(
            primary,
            "A. name-year join (primary)",
            "fire_id = cohort|name|year",
            EVENT_KEYS,
            cluster="fire_id",
        )
    )
    rows.extend(
        fit_path_rows(
            event,
            "B. event-date join (sensitivity)",
            "event_id = cohort|name|year|date",
            EVENT_MONITOR_KEYS,
            cluster="event_id",
        )
    )
    return pd.DataFrame(rows), primary, event


def event_diagnostics(event) -> dict:
    extras_table = event_key_extras(event, EVENT_MONITOR_KEYS)
    d = complete_case(event, "delta_During_minus_Before")
    extras_cc = event_key_extras(d, EVENT_MONITOR_KEYS)
    audit = chl_same_event_audit(event)
    chl_dates = sorted({str(x) for x in audit["Fire_Date"].dropna().unique()}) if len(audit) else []
    apr = audit[audit["Fire_Date"] == EXPECTED_EVENT["chl_april"]] if len(audit) else audit
    oct_ = audit[audit["Fire_Date"] == EXPECTED_EVENT["chl_october"]] if len(audit) else audit
    shared_area = set(apr["area_ha"].dropna()) & set(oct_["area_ha"].dropna()) if len(audit) else set()
    shared_wind = (
        set(apr["wind_speed_ms_during_mean"].dropna()) & set(oct_["wind_speed_ms_during_mean"].dropna())
        if len(audit)
        else set()
    )
    shared_peak = (
        set(apr["delta_MaxDailyDuring_minus_Before"].dropna())
        & set(oct_["delta_MaxDailyDuring_minus_Before"].dropna())
        if len(audit)
        else set()
    )
    return {
        "event_join_extras": extras_table,
        "event_cc_extras": extras_cc,
        "chl_dates": chl_dates,
        "chl_audit": audit,
        "chl_shared_area": shared_area,
        "chl_shared_wind": shared_wind,
        "chl_shared_peak": shared_peak,
        "chl_april_area": sorted(apr["area_ha"].dropna().unique()) if len(apr) else [],
        "chl_october_area": sorted(oct_["area_ha"].dropna().unique()) if len(oct_) else [],
        "chl_april_season": sorted(apr["met_season"].dropna().astype(str).unique()) if len(apr) else [],
        "chl_october_season": sorted(oct_["met_season"].dropna().astype(str).unique()) if len(oct_) else [],
    }


def _row(table: pd.DataFrame, path: str, outcome: str) -> pd.Series:
    return table[(table["path"] == path) & (table["Outcome"] == outcome)].iloc[0]


def _fmt_nums(values) -> str:
    if not values:
        return "none"
    return ", ".join(f"{float(v):.6g}" for v in values)


def write_event_vs_name_year_comparison(table: pd.DataFrame, diag: dict) -> Path:
    """Present both estimates and assumptions. Does not choose a primary model."""
    ny_d = _row(table, "A. name-year join (primary)", "During minus Before")
    ny_p = _row(table, "A. name-year join (primary)", "Peak daily minus Before")
    ny_a = _row(table, "A. name-year join (primary)", "After minus Before")
    ev_d = _row(table, "B. event-date join (sensitivity)", "During minus Before")
    ev_p = _row(table, "B. event-date join (sensitivity)", "Peak daily minus Before")
    ev_a = _row(table, "B. event-date join (sensitivity)", "After minus Before")
    diff = float(ev_d["Estimate"]) - float(ny_d["Estimate"])
    ev_lo, ev_hi = float(ev_d["CI low"]), float(ev_d["CI high"])
    ny_lo, ny_hi = float(ny_d["CI low"]), float(ny_d["CI high"])
    extras = int(diag["event_join_extras"])
    ev_excludes0 = ev_lo > 0
    ny_excludes0 = ny_lo > 0
    lines = [
        "Name-year join vs event-corrected (Stage B) join.",
        "Both estimates are reported. Choosing which model is primary",
        "requires methodological review; this script does not replace 4.55.",
        "",
        "A. NAME-YEAR (labeled primary in this package; fire_id = cohort|name|year)",
        "  Assumptions: join and cluster on name+year; Fire_Date is not used;",
        "  same-name same-year burns (CHL #12 2021) share one fire_id and can",
        "  inflate the join. Reproducible complete-case During sample is 4,186 / 1,547.",
        f"  During: {float(ny_d['Estimate']):.2f} ({ny_lo:.2f}–{ny_hi:.2f}), "
        f"n = {int(ny_d['n linkages'])} / {int(ny_d['n fires'])}.",
        f"  Peak: {float(ny_p['Estimate']):.2f} ({float(ny_p['CI low']):.2f}–{float(ny_p['CI high']):.2f}).",
        f"  After: {float(ny_a['Estimate']):.2f} ({float(ny_a['CI low']):.2f}–{float(ny_a['CI high']):.2f}), "
        f"n = {int(ny_a['n linkages'])} / {int(ny_a['n fires'])}.",
        f"  CI excludes 0: {ny_excludes0}.",
        "",
        "B. EVENT-DATE (sensitivity; event_id = cohort|name|year|date; Stage B dates)",
        "  Assumptions: one burn date is one event; cluster by event_id;",
        "  RX wind/downwind/season/area come from Stage B on the event date;",
        "  WF season/area come from Stage B; WF wind/downwind are copied from",
        "  the name-year modeling table because Stage B stores no WF wind.",
        "  Fuel is fire-year grain (see validation/event_date_covariate_provenance.txt).",
        f"  During: {float(ev_d['Estimate']):.2f} ({ev_lo:.2f}–{ev_hi:.2f}), "
        f"n = {int(ev_d['n linkages'])} / {int(ev_d['n events'])} / {int(ev_d['n fires'])}.",
        f"  Peak: {float(ev_p['Estimate']):.2f} ({float(ev_p['CI low']):.2f}–{float(ev_p['CI high']):.2f}).",
        f"  After: {float(ev_a['Estimate']):.2f} ({float(ev_a['CI low']):.2f}–{float(ev_a['CI high']):.2f}), "
        f"n = {int(ev_a['n linkages'])} / {int(ev_a['n events'])} / {int(ev_a['n fires'])}.",
        f"  Event-join extras: {extras}.",
        f"  CI excludes 0: {ev_excludes0}.",
        f"  During difference (event minus name-year): {diff:+.2f} µg m-3.",
        "",
        "CHL #12 2021 same-event audit (Stage B event+monitor)",
        f"  Dates: {', '.join(diag['chl_dates']) or 'NONE'}.",
        f"  April area ha: {_fmt_nums(diag['chl_april_area'])}; season: {diag['chl_april_season']}.",
        f"  October area ha: {_fmt_nums(diag['chl_october_area'])}; season: {diag['chl_october_season']}.",
        f"  Shared area values: {sorted(diag['chl_shared_area']) or 'none'}.",
        f"  Shared wind values: {sorted(diag['chl_shared_wind']) or 'none'}.",
        f"  Shared peak-daily values: {sorted(diag['chl_shared_peak']) or 'none'}.",
        "",
        "CHOICE OF PRIMARY MODEL",
        "  This comparison does not replace 4.55 with 4.45. CHL date splits,",
        "  extras = 0, and a CI that excludes 0 are not an automatic switch.",
        "  After wind and fuel provenance are verified, the event-date model",
        "  may be more defensible because size, season, and RX meteorology",
        "  belong to one Fire_Date. That is a possibility for methodological",
        "  review, not a recommendation produced by this script.",
        "",
        "The name-year 4.55 path remains in Table_S_event_id_sensitivity.csv",
        "and results/Table_adjusted_wildfire_vs_rx.csv.",
        "",
    ]
    path = RESULTS / "event_vs_name_year_comparison.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    ensure_results()
    table, _primary, event = event_id_sensitivity_table()
    table.to_csv(RESULTS / "Table_S_event_id_sensitivity.csv", index=False)
    diag = event_diagnostics(event)
    audit = diag["chl_audit"]
    if len(audit):
        audit.to_csv(RESULTS / "Table_S_CHL12_2021_same_event_audit.csv", index=False)
    write_event_vs_name_year_comparison(table, diag)
    ev = table[table["path"] == "B. event-date join (sensitivity)"]
    ny = table[table["path"] == "A. name-year join (primary)"]
    print(table.to_string(index=False))
    print()
    print(
        f"Event-join extras (full table / complete-case during): "
        f"{diag['event_join_extras']} / {diag['event_cc_extras']}"
    )
    print(f"CHL #12 2021 dates: {', '.join(diag['chl_dates']) or 'NONE'}")
    print(
        f"CHL shared area/wind/peak: "
        f"{diag['chl_shared_area'] or 'none'} / "
        f"{diag['chl_shared_wind'] or 'none'} / "
        f"{diag['chl_shared_peak'] or 'none'}"
    )
    fail = 0
    if diag["event_join_extras"] != EXPECTED_EVENT["join_extras"]:
        print("FAIL: extras after event join")
        fail += 1
    if EXPECTED_EVENT["chl_april"] not in diag["chl_dates"]:
        print("FAIL: CHL #12 2021-04-17 missing")
        fail += 1
    if EXPECTED_EVENT["chl_october"] not in diag["chl_dates"]:
        print("FAIL: CHL #12 2021-10-19 missing")
        fail += 1
    if diag["chl_shared_area"] or diag["chl_shared_wind"] or diag["chl_shared_peak"]:
        print("FAIL: CHL dates share area, wind, or peak")
        fail += 1

    ny_checks = [
        ("During minus Before", "during", "during_lo", "during_hi", "during_n", None, "during_fires"),
        ("Peak daily minus Before", "peak", "peak_lo", "peak_hi", None, None, None),
        ("After minus Before", "after", "after_lo", "after_hi", "after_n", None, "after_fires"),
    ]
    for outcome, e, lo, hi, n, evn, fr in ny_checks:
        row = ny[ny["Outcome"] == outcome].iloc[0]
        got_e, got_lo, got_hi = round(row["Estimate"], 2), round(row["CI low"], 2), round(row["CI high"], 2)
        ok = (
            abs(got_e - EXPECTED_NAMEYEAR[e]) <= 0.005
            and abs(got_lo - EXPECTED_NAMEYEAR[lo]) <= 0.01
            and abs(got_hi - EXPECTED_NAMEYEAR[hi]) <= 0.01
        )
        extra = ""
        if n is not None:
            ok_n = int(row["n linkages"]) == EXPECTED_NAMEYEAR[n] and int(row["n fires"]) == EXPECTED_NAMEYEAR[fr]
            ok = ok and ok_n
            extra = f" n={int(row['n linkages'])}/{int(row['n fires'])}"
        status = "PASS" if ok else "FAIL"
        if not ok:
            fail += 1
        print(f"{status}: name-year {outcome}: {got_e:.2f} ({got_lo:.2f}–{got_hi:.2f}){extra}")

    checks = [
        ("During minus Before", "during", "during_lo", "during_hi", "during_n", "during_events", "during_fires"),
        ("Peak daily minus Before", "peak", "peak_lo", "peak_hi", None, None, None),
        ("After minus Before", "after", "after_lo", "after_hi", "after_n", "after_events", "after_fires"),
    ]
    for outcome, e, lo, hi, n, evn, fr in checks:
        row = ev[ev["Outcome"] == outcome].iloc[0]
        got_e, got_lo, got_hi = round(row["Estimate"], 2), round(row["CI low"], 2), round(row["CI high"], 2)
        ok = (
            abs(got_e - EXPECTED_EVENT[e]) <= 0.005
            and abs(got_lo - EXPECTED_EVENT[lo]) <= 0.01
            and abs(got_hi - EXPECTED_EVENT[hi]) <= 0.01
        )
        extra = ""
        if n is not None:
            ok_n = (
                int(row["n linkages"]) == EXPECTED_EVENT[n]
                and int(row["n events"]) == EXPECTED_EVENT[evn]
                and int(row["n fires"]) == EXPECTED_EVENT[fr]
            )
            ok = ok and ok_n
            extra = f" n={int(row['n linkages'])}/{int(row['n events'])}/{int(row['n fires'])}"
        status = "PASS" if ok else "FAIL"
        if not ok:
            fail += 1
        print(
            f"{status}: event-ID {outcome}: {got_e:.2f} ({got_lo:.2f}–{got_hi:.2f}){extra}"
        )
    print(f"{'PASS' if fail == 0 else 'FAIL'}: event-ID table rebuilt from Stage B event+monitor.")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
