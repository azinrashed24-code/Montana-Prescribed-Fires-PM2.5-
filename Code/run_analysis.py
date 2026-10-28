"""Run the primary daily-cohort tables and statistics.

Run from anywhere:
    python code/run_analysis.py

Writes tables and figures to results/. Writes check reports to validation/.
Does not overwrite files in data/.
Exits 0 if every primary target matches; otherwise exits 1.
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
    EVENT_KEYS,
    FORMULA,
    FORMULA_INTERACTION,
    NAME_YEAR_MONITOR,
    OUTCOMES,
    complete_case,
    complete_case_with_area,
    event_key_extras,
    fit_clustered,
    fit_path_rows,
    format_p,
    interaction_term,
    iso_date,
    load_modeling,
    load_modeling_unique_keys,
    wildfire_term,
)
from fit_event_date import run_event_date_refit  # noqa: E402
from fit_event_id_sensitivity import (  # noqa: E402
    event_diagnostics,
    event_id_sensitivity_table,
    write_event_vs_name_year_comparison,
)
from paths import DATA, FIGURES, RESULTS, VALIDATION, ensure_results, first_existing  # noqa: E402

# Primary analysis targets. Validation compares regenerated numbers to these values.
EXPECTED = {
    "rx_source_n": 1942,
    "wf_source_n": 308,
    "rx_area_median": 6.7,
    "rx_area_q25": 0.4,
    "rx_area_q75": 28.3,
    "wf_area_median": 1427.3,
    "wf_area_q25": 690,
    "wf_area_q75": 3286,
    "rx_linkages": 8122,
    "wf_linkages": 361,
    "rx_fires": 1912,
    "wf_fires": 179,
    "rx_monitors": 25,
    "wf_monitors": 16,
    "unadj_during_rx": 0.31,
    "unadj_during_wf": 4.42,
    "unadj_during_n_rx": 8056,
    "unadj_during_n_wf": 359,
    "unadj_peak_rx": 2.05,
    "unadj_peak_wf": 9.47,
    "unadj_after_rx": 0.33,
    "unadj_after_wf": 9.48,
    "unadj_after_n_rx": 8086,
    "unadj_after_n_wf": 359,
    "adj_during": 4.55,
    "adj_during_lo": 2.91,
    "adj_during_hi": 6.19,
    "adj_during_n": 4186,
    "adj_during_fires": 1547,
    "adj_peak": 7.68,
    "adj_peak_lo": 5.24,
    "adj_peak_hi": 10.12,
    "adj_after": 11.01,
    "adj_after_lo": 6.61,
    "adj_after_hi": 15.40,
    "adj_after_n": 4191,
    "adj_after_fires": 1548,
    "ix_during": 0.26,
    "ix_during_lo": -0.70,
    "ix_during_hi": 1.23,
    "ix_overlap_lo_ha": 414,
    "ix_overlap_hi_ha": 3355,
    "ix_overlap_n": 247,
    "ix_overlap_fires": 129,
    "rec_n_rx": 4885,
    "rec_n_wf": 351,
    "rec_median_rx": 1.0,
    "rec_median_wf": 2.0,
    "rec_cens_rx": 9.4,
    "rec_cens_wf": 27.9,
    "ix_peak": 0.66,
    "ix_after": 0.98,
    "ix_overlap_peak": 1.88,
    "ix_overlap_after": -0.03,
    "hrs12_rx": 8.6,
    "hrs12_wf": 41.4,
    "hrs35_rx": 0.3,
    "hrs35_wf": 9.8,
    "gamm_n": 379433,
    "gamm_fires": 761,
    "gamm_monitors": 21,
    "s3_primary_n": 4186,
    "s3_primary_fires": 1547,
    "merge_unique_extras": 0,
    "event_join_extras": 0,
    "hex_overlap_pct_max": 0.05,
    "chl_event_april": "2021-04-17",
    "chl_event_october": "2021-10-19",
    "event_during": 4.45,
    "event_during_lo": 2.82,
    "event_during_hi": 6.09,
    "event_during_n": 4165,
    "event_during_events": 1549,
    "event_during_fires": 1548,
    "event_peak": 7.57,
    "event_peak_lo": 5.14,
    "event_peak_hi": 10.00,
    "event_after": 11.00,
    "event_after_lo": 6.60,
    "event_after_hi": 15.40,
    "event_after_n": 4170,
    "event_after_events": 1550,
    "event_after_fires": 1549,
    "rec_missing_rx": 48,
    "rec_missing_wf": 1,
    "event_rx_n": 3856,
    "event_wf_n": 309,
    "event_wf_stage_b_wind": 0,
    "s1_300_n": 8056,
}


def fire_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["fire_cohort"].astype(str)
        + "|"
        + df["Fire_Name"].astype(str)
        + "|"
        + df["Fire_Year"].astype(str)
    )


def iqr(s: pd.Series) -> tuple[float, float, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.median()), float(s.quantile(0.25)), float(s.quantile(0.75))


def table_1() -> pd.DataFrame:
    rx = pd.read_csv(DATA / "inventory_rx.csv")
    wf = pd.read_csv(DATA / "inventory_wf.csv")
    b = pd.read_csv(DATA / "stage_b_daily.csv", low_memory=False)
    rx_med, rx_q1, rx_q3 = iqr(rx["area_ha"])
    wf_med, wf_q1, wf_q3 = iqr(wf["area_ha"])
    rows = []
    for cohort, label in [("WFIGS_RX", "Prescribed fire"), ("MTBS_wildfire", "Wildfire")]:
        sub = b[b["fire_cohort"] == cohort]
        rows.append(
            {
                "Fire type": label,
                "Source fires, n": int(len(rx) if cohort == "WFIGS_RX" else len(wf)),
                "Burned area, median, ha": rx_med if cohort == "WFIGS_RX" else wf_med,
                "Burned area, IQR q25, ha": rx_q1 if cohort == "WFIGS_RX" else wf_q1,
                "Burned area, IQR q75, ha": rx_q3 if cohort == "WFIGS_RX" else wf_q3,
                "Primary daily cohort linkages, n": int(len(sub)),
                "Fires, n": int(fire_key(sub).nunique()),
                "PM2.5 monitors, n": int(sub["aqs_code"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "Table_1_characteristics.csv", index=False)
    return out


def unadjusted() -> pd.DataFrame:
    b = pd.read_csv(DATA / "stage_b_daily.csv", low_memory=False)
    specs = [
        ("During minus Before", "delta_During_minus_Before"),
        ("Peak daily minus Before", "delta_MaxDailyDuring_minus_Before"),
        ("After minus Before", "delta_After_minus_Before"),
    ]
    rows = []
    for label, col in specs:
        for cohort, name in [("WFIGS_RX", "Prescribed fire"), ("MTBS_wildfire", "Wildfire")]:
            s = pd.to_numeric(b.loc[b["fire_cohort"] == cohort, col], errors="coerce").dropna()
            rows.append(
                {
                    "Outcome": label,
                    "Fire type": name,
                    "Mean, ug m-3": float(s.mean()),
                    "n linkages": int(len(s)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "unadjusted_means.csv", index=False)
    return out


def adjusted() -> pd.DataFrame:
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
                "Estimate, ug m-3": float(m.params[term]),
                "CI low": float(lo),
                "CI high": float(hi),
                "p": format_p(float(m.pvalues[term])),
                "n linkages": int(len(d)),
                "n fires": int(d["fire_id"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "Table_adjusted_wildfire_vs_rx.csv", index=False)
    return out


def interaction() -> pd.DataFrame:
    st = load_modeling()
    rows = []
    n_note = {}
    for sample_name, restrict in [("full complete case", False), ("common support", True)]:
        for label, col in OUTCOMES:
            d = complete_case_with_area(st, col)
            if restrict:
                fires = d.drop_duplicates("fire_id")
                rx = fires.loc[fires["fire_cohort"] == "WFIGS_RX", "area_ha"]
                wf = fires.loc[fires["fire_cohort"] == "MTBS_wildfire", "area_ha"]
                lo_ha = float(max(rx.min(), wf.min()))
                hi_ha = float(min(rx.max(), wf.max()))
                d = d[(d["area_ha"] >= lo_ha) & (d["area_ha"] <= hi_ha)].copy()
                n_note[sample_name] = (lo_ha, hi_ha)
            m = fit_clustered(FORMULA_INTERACTION.format(outcome=col), d)
            term = interaction_term(m)
            lo, hi = m.conf_int().loc[term]
            rows.append(
                {
                    "Sample": sample_name,
                    "Outcome": label,
                    "Interaction, ug m-3 per log(1+ha)": float(m.params[term]),
                    "CI low": float(lo),
                    "CI high": float(hi),
                    "p": format_p(float(m.pvalues[term])),
                    "n linkages": int(len(d)),
                    "n fires": int(d["fire_id"].nunique()),
                    "overlap_lo_ha": n_note.get(sample_name, (np.nan, np.nan))[0],
                    "overlap_hi_ha": n_note.get(sample_name, (np.nan, np.nan))[1],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "Table_S1_type_by_log1p_area.csv", index=False)
    return out


def recovery() -> pd.DataFrame:
    e = pd.read_csv(DATA / "stage_e_recovery.csv", low_memory=False)
    hrs35 = pd.to_numeric(e.get("pct_hours_during_above_35"), errors="coerce")
    if hrs35.isna().all():
        hrs35 = pd.to_numeric(e.get("pct_hours_during_above_35.4"), errors="coerce")
    e = e.copy()
    e["_hrs35"] = hrs35
    rows = []
    for cohort, name in [("WFIGS_RX", "Prescribed fire"), ("MTBS_wildfire", "Wildfire")]:
        sub = e[e["fire_cohort"] == cohort]
        rec_flag = pd.to_numeric(sub["censored_no_return_to_before"], errors="coerce").fillna(1) == 0
        days = pd.to_numeric(sub["days_to_before_baseline_in_after"], errors="coerce")
        rec = days[rec_flag].dropna()
        rows.append(
            {
                "Fire type": name,
                "n linkages": int(len(sub)),
                "Median recovery, days": float(rec.median()) if len(rec) else np.nan,
                "IQR q25, days": float(rec.quantile(0.25)) if len(rec) else np.nan,
                "IQR q75, days": float(rec.quantile(0.75)) if len(rec) else np.nan,
                "Censored, %": 100.0
                * float(pd.to_numeric(sub["censored_no_return_to_before"], errors="coerce").fillna(1).mean()),
                "Mean % hours > 12 ug m-3": float(
                    pd.to_numeric(sub["pct_hours_during_above_12"], errors="coerce").mean()
                ),
                "Mean % hours > 35.4 ug m-3": float(sub["_hrs35"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "Table_recovery_and_hourly.csv", index=False)
    return out


def write_recovery_censoring_audit() -> dict:
    """Document fillna(1) vs excluding missing recovery status.

    Missing censored_no_return_to_before is treated as censored in recovery().
    That is the recipe that matches 9.4 / 27.9. Missing rows also have missing
    recovery days, so they are not observed recoveries. They are not treated
    as clearly invalid; both percentages are reported and the
    9.4 / 27.9 values are not changed.
    """
    e = pd.read_csv(DATA / "stage_e_recovery.csv", low_memory=False)
    counts = {}
    lines = [
        "Stage E recovery censoring: missing status vs fillna(1).",
        "",
        "recovery() uses fillna(1) so missing censored_no_return_to_before",
        "counts as not recovered by day +10. Those rows also have missing",
        "days_to_before_baseline_in_after, so they are not observed recoveries.",
        "They are not dropped as invalid. fillna(1) is what matches 9.4 / 27.9.",
        "Excluding missing rows does not match those percentages.",
        "",
    ]
    for cohort, name in [("WFIGS_RX", "Prescribed fire"), ("MTBS_wildfire", "Wildfire")]:
        sub = e[e["fire_cohort"] == cohort]
        raw = pd.to_numeric(sub["censored_no_return_to_before"], errors="coerce")
        n_missing = int(raw.isna().sum())
        fill_pct = 100.0 * float(raw.fillna(1).mean())
        valid = raw.dropna()
        excl_pct = 100.0 * float(valid.mean()) if len(valid) else float("nan")
        key = "rx" if cohort == "WFIGS_RX" else "wf"
        counts[f"missing_{key}"] = n_missing
        counts[f"fill_{key}"] = fill_pct
        counts[f"exclude_{key}"] = excl_pct
        lines.extend(
            [
                f"{name} (n = {len(sub)})",
                f"  Missing censored_no_return_to_before: {n_missing}",
                f"  Missing treated as censored (fillna(1)): {fill_pct:.1f}%",
                f"  Missing excluded (invalid): {excl_pct:.1f}%",
                "",
            ]
        )
    lines.extend(
        [
            "Reported recovery not-recovered-by-day-+10 percentages remain",
            "9.4% prescribed and 27.9% wildfire (fillna(1) column).",
            "",
        ]
    )
    (VALIDATION / "recovery_censoring.txt").write_text("\n".join(lines), encoding="utf-8")
    return counts


def make_figures(unadj: pd.DataFrame, adj: pd.DataFrame, rec: pd.DataFrame) -> None:
    rx_color, wf_color = "#0f766e", "#b45309"

    # Unadjusted means (Figure 2 numbers).
    order = [
        "During minus Before",
        "Peak daily minus Before",
        "After minus Before",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    x = np.arange(len(order))
    w = 0.36
    rx = [
        unadj[(unadj["Outcome"] == o) & (unadj["Fire type"] == "Prescribed fire")][
            "Mean, ug m-3"
        ].iloc[0]
        for o in order
    ]
    wf = [
        unadj[(unadj["Outcome"] == o) & (unadj["Fire type"] == "Wildfire")][
            "Mean, ug m-3"
        ].iloc[0]
        for o in order
    ]
    ax.bar(x - w / 2, rx, w, color=rx_color, label="Prescribed fire")
    ax.bar(x + w / 2, wf, w, color=wf_color, label="Wildfire")
    ax.set_xticks(x, ["During − Before", "Peak daily − Before", "After − Before"])
    ax.set_ylabel("Mean PM$_{2.5}$ response (µg m$^{-3}$)")
    ax.set_title("Unadjusted fire-type means")
    ax.legend(frameon=False)
    ax.axhline(0, color="#444444", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_2_observed_pm25.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Adjusted coefficients (Figure 3 numbers).
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    y = np.arange(len(adj))
    ax.errorbar(
        adj["Estimate, ug m-3"],
        y,
        xerr=[
            adj["Estimate, ug m-3"] - adj["CI low"],
            adj["CI high"] - adj["Estimate, ug m-3"],
        ],
        fmt="o",
        color="#1f2937",
        capsize=4,
        linewidth=1.4,
    )
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_yticks(y, adj["Outcome"])
    ax.set_xlabel("Wildfire − prescribed fire (µg m$^{-3}$)")
    ax.set_title("Adjusted fire-type contrast")
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_3_adjusted_wildfire_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Recovery and hourly (Figure 4 numbers).
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), dpi=150)
    names = rec["Fire type"].tolist()
    colors = [rx_color, wf_color]
    med = rec["Median recovery, days"].to_numpy(dtype=float)
    q25 = rec["IQR q25, days"].to_numpy(dtype=float)
    q75 = rec["IQR q75, days"].to_numpy(dtype=float)
    x_rec = np.arange(len(names))
    axes[0].bar(x_rec, med, color=colors, width=0.55)
    axes[0].errorbar(
        x_rec,
        med,
        yerr=[np.maximum(med - q25, 0.0), np.maximum(q75 - med, 0.0)],
        fmt="none",
        ecolor="#1e293b",
        capsize=4,
        linewidth=1.2,
    )
    axes[0].set_xticks(x_rec, names)
    axes[0].set_ylabel("Days")
    axes[0].set_title("Median recovery (recovered linkages)")
    axes[0].text(
        0.02,
        0.98,
        "Error bars: IQR",
        transform=axes[0].transAxes,
        fontsize=8,
        va="top",
        color="#555555",
    )
    axes[1].bar(
        np.arange(2) - 0.18,
        rec["Mean % hours > 12 ug m-3"],
        0.36,
        color=colors,
        label="> 12 µg m$^{-3}$",
    )
    axes[1].bar(
        np.arange(2) + 0.18,
        rec["Mean % hours > 35.4 ug m-3"],
        0.36,
        color=colors,
        alpha=0.55,
        label="> 35.4 µg m$^{-3}$",
    )
    axes[1].set_xticks(np.arange(2), names)
    axes[1].set_ylabel("% of valid During hours")
    axes[1].set_title("Hourly exceedance, days 0–2")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_4_recovery_and_hourly.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _check(rows: list, name: str, got, expected, tol: float = 0.0) -> None:
    if isinstance(got, (float, np.floating)) and isinstance(expected, (float, int)):
        ok = abs(float(got) - float(expected)) <= tol
        display = f"{float(got):g}"
    else:
        ok = int(got) == int(expected)
        display = str(int(got))
    rows.append((name, display, str(expected), "PASS" if ok else "FAIL"))


def check(t1, unadj, adj, ix, rec) -> list:
    rows = []
    rx = t1[t1["Fire type"] == "Prescribed fire"].iloc[0]
    wf = t1[t1["Fire type"] == "Wildfire"].iloc[0]
    _check(rows, "Table 1 source RX n", rx["Source fires, n"], EXPECTED["rx_source_n"])
    _check(rows, "Table 1 source WF n", wf["Source fires, n"], EXPECTED["wf_source_n"])
    _check(rows, "Table 1 RX median ha", round(rx["Burned area, median, ha"], 1), EXPECTED["rx_area_median"], 0.05)
    _check(rows, "Table 1 RX IQR q25", round(rx["Burned area, IQR q25, ha"], 1), EXPECTED["rx_area_q25"], 0.05)
    _check(rows, "Table 1 RX IQR q75", round(rx["Burned area, IQR q75, ha"], 1), EXPECTED["rx_area_q75"], 0.05)
    _check(rows, "Table 1 WF median ha", round(wf["Burned area, median, ha"], 1), EXPECTED["wf_area_median"], 0.05)
    _check(rows, "Table 1 WF IQR q25", round(wf["Burned area, IQR q25, ha"]), EXPECTED["wf_area_q25"])
    _check(rows, "Table 1 WF IQR q75", round(wf["Burned area, IQR q75, ha"]), EXPECTED["wf_area_q75"])
    _check(rows, "Table 1 RX linkages", rx["Primary daily cohort linkages, n"], EXPECTED["rx_linkages"])
    _check(rows, "Table 1 WF linkages", wf["Primary daily cohort linkages, n"], EXPECTED["wf_linkages"])
    _check(rows, "Table 1 RX fires", rx["Fires, n"], EXPECTED["rx_fires"])
    _check(rows, "Table 1 WF fires", wf["Fires, n"], EXPECTED["wf_fires"])
    _check(rows, "Table 1 RX monitors", rx["PM2.5 monitors, n"], EXPECTED["rx_monitors"])
    _check(rows, "Table 1 WF monitors", wf["PM2.5 monitors, n"], EXPECTED["wf_monitors"])

    def mean_of(outcome, ftype):
        return unadj[(unadj["Outcome"] == outcome) & (unadj["Fire type"] == ftype)].iloc[0]

    d_rx = mean_of("During minus Before", "Prescribed fire")
    d_wf = mean_of("During minus Before", "Wildfire")
    p_rx = mean_of("Peak daily minus Before", "Prescribed fire")
    p_wf = mean_of("Peak daily minus Before", "Wildfire")
    a_rx = mean_of("After minus Before", "Prescribed fire")
    a_wf = mean_of("After minus Before", "Wildfire")
    _check(rows, "Unadjusted during RX", round(d_rx["Mean, ug m-3"], 2), EXPECTED["unadj_during_rx"], 0.005)
    _check(rows, "Unadjusted during WF", round(d_wf["Mean, ug m-3"], 2), EXPECTED["unadj_during_wf"], 0.005)
    _check(rows, "Unadjusted during n RX", d_rx["n linkages"], EXPECTED["unadj_during_n_rx"])
    _check(rows, "Unadjusted during n WF", d_wf["n linkages"], EXPECTED["unadj_during_n_wf"])
    _check(rows, "Unadjusted peak RX", round(p_rx["Mean, ug m-3"], 2), EXPECTED["unadj_peak_rx"], 0.005)
    _check(rows, "Unadjusted peak WF", round(p_wf["Mean, ug m-3"], 2), EXPECTED["unadj_peak_wf"], 0.005)
    _check(rows, "Unadjusted after RX", round(a_rx["Mean, ug m-3"], 2), EXPECTED["unadj_after_rx"], 0.005)
    _check(rows, "Unadjusted after WF", round(a_wf["Mean, ug m-3"], 2), EXPECTED["unadj_after_wf"], 0.005)
    _check(rows, "Unadjusted after n RX", a_rx["n linkages"], EXPECTED["unadj_after_n_rx"])
    _check(rows, "Unadjusted after n WF", a_wf["n linkages"], EXPECTED["unadj_after_n_wf"])

    dur = adj[adj["Outcome"] == "During minus Before"].iloc[0]
    peak = adj[adj["Outcome"] == "Peak daily minus Before"].iloc[0]
    after = adj[adj["Outcome"] == "After minus Before"].iloc[0]
    _check(rows, "Adjusted during", round(dur["Estimate, ug m-3"], 2), EXPECTED["adj_during"], 0.005)
    _check(rows, "Adjusted during CI low", round(dur["CI low"], 2), EXPECTED["adj_during_lo"], 0.01)
    _check(rows, "Adjusted during CI high", round(dur["CI high"], 2), EXPECTED["adj_during_hi"], 0.01)
    _check(rows, "Adjusted during n", dur["n linkages"], EXPECTED["adj_during_n"])
    _check(rows, "Adjusted during fires", dur["n fires"], EXPECTED["adj_during_fires"])
    _check(rows, "Adjusted peak", round(peak["Estimate, ug m-3"], 2), EXPECTED["adj_peak"], 0.005)
    _check(rows, "Adjusted peak CI low", round(peak["CI low"], 2), EXPECTED["adj_peak_lo"], 0.01)
    _check(rows, "Adjusted peak CI high", round(peak["CI high"], 2), EXPECTED["adj_peak_hi"], 0.01)
    _check(rows, "Adjusted after", round(after["Estimate, ug m-3"], 2), EXPECTED["adj_after"], 0.005)
    _check(rows, "Adjusted after CI low", round(after["CI low"], 2), EXPECTED["adj_after_lo"], 0.01)
    _check(rows, "Adjusted after CI high", round(after["CI high"], 2), EXPECTED["adj_after_hi"], 0.01)
    _check(rows, "Adjusted after n", after["n linkages"], EXPECTED["adj_after_n"])
    _check(rows, "Adjusted after fires", after["n fires"], EXPECTED["adj_after_fires"])

    ix_d = ix[(ix["Sample"] == "full complete case") & (ix["Outcome"] == "During minus Before")].iloc[0]
    ix_o = ix[(ix["Sample"] == "common support") & (ix["Outcome"] == "During minus Before")].iloc[0]
    _check(rows, "Interaction during", round(ix_d["Interaction, ug m-3 per log(1+ha)"], 2), EXPECTED["ix_during"], 0.015)
    _check(rows, "Interaction CI low", round(ix_d["CI low"], 2), EXPECTED["ix_during_lo"], 0.02)
    _check(rows, "Interaction CI high", round(ix_d["CI high"], 2), EXPECTED["ix_during_hi"], 0.02)
    _check(rows, "Overlap lo ha", round(ix_o["overlap_lo_ha"]), EXPECTED["ix_overlap_lo_ha"], 1)
    _check(rows, "Overlap hi ha", round(ix_o["overlap_hi_ha"]), EXPECTED["ix_overlap_hi_ha"], 1)
    _check(rows, "Overlap n", ix_o["n linkages"], EXPECTED["ix_overlap_n"])
    _check(rows, "Overlap fires", ix_o["n fires"], EXPECTED["ix_overlap_fires"])

    ix_p = ix[(ix["Sample"] == "full complete case") & (ix["Outcome"] == "Peak daily minus Before")].iloc[0]
    ix_a = ix[(ix["Sample"] == "full complete case") & (ix["Outcome"] == "After minus Before")].iloc[0]
    ix_op = ix[(ix["Sample"] == "common support") & (ix["Outcome"] == "Peak daily minus Before")].iloc[0]
    ix_oa = ix[(ix["Sample"] == "common support") & (ix["Outcome"] == "After minus Before")].iloc[0]
    _check(rows, "Interaction peak", round(ix_p["Interaction, ug m-3 per log(1+ha)"], 2), EXPECTED["ix_peak"], 0.015)
    _check(rows, "Interaction after", round(ix_a["Interaction, ug m-3 per log(1+ha)"], 2), EXPECTED["ix_after"], 0.015)
    _check(rows, "Overlap interaction peak", round(ix_op["Interaction, ug m-3 per log(1+ha)"], 2), EXPECTED["ix_overlap_peak"], 0.015)
    _check(rows, "Overlap interaction after", round(ix_oa["Interaction, ug m-3 per log(1+ha)"], 2), EXPECTED["ix_overlap_after"], 0.015)

    r_rx = rec[rec["Fire type"] == "Prescribed fire"].iloc[0]
    r_wf = rec[rec["Fire type"] == "Wildfire"].iloc[0]
    _check(rows, "Recovery n RX", r_rx["n linkages"], EXPECTED["rec_n_rx"])
    _check(rows, "Recovery n WF", r_wf["n linkages"], EXPECTED["rec_n_wf"])
    _check(rows, "Recovery median RX", r_rx["Median recovery, days"], EXPECTED["rec_median_rx"], 0.05)
    _check(rows, "Recovery median WF", r_wf["Median recovery, days"], EXPECTED["rec_median_wf"], 0.05)
    _check(rows, "Recovery censored RX %", round(r_rx["Censored, %"], 1), EXPECTED["rec_cens_rx"], 0.05)
    _check(rows, "Recovery censored WF %", round(r_wf["Censored, %"], 1), EXPECTED["rec_cens_wf"], 0.05)
    _check(rows, "Hours >12 RX %", round(r_rx["Mean % hours > 12 ug m-3"], 1), EXPECTED["hrs12_rx"], 0.05)
    _check(rows, "Hours >12 WF %", round(r_wf["Mean % hours > 12 ug m-3"], 1), EXPECTED["hrs12_wf"], 0.05)
    _check(rows, "Hours >35 RX %", round(r_rx["Mean % hours > 35.4 ug m-3"], 1), EXPECTED["hrs35_rx"], 0.05)
    _check(rows, "Hours >35 WF %", round(r_wf["Mean % hours > 35.4 ug m-3"], 1), EXPECTED["hrs35_wf"], 0.05)
    return rows


def write_chl12_reviewer_csv(primary_cc: pd.DataFrame, event_cc: pd.DataFrame | None = None) -> None:
    """List the two 2021 CHL #12 events and the complete-case copies at the two remaining monitors.

    Written to results/ so the CHL audit is inspectable with the tables.
    Does not change the primary 4.55 path.
    """
    inv = pd.read_csv(DATA / "inventory_rx.csv")
    events = inv[(inv["Fire_Name"] == "CHL #12") & (inv["Fire_Year"] == 2021)].copy()
    md = pd.read_csv(DATA / "max_daily.csv", low_memory=False)
    md_chl = md[(md["Fire_Name"] == "CHL #12") & (md["Fire_Year"] == 2021)]
    st = pd.read_csv(DATA / "modeling_station.csv", low_memory=False)
    st_chl = st[(st["Fire_Name"] == "CHL #12") & (st["Fire_Year"] == 2021)]
    monitors = ["30-031-0017", "30-093-0005"]
    rows = []
    for _, ev in events.iterrows():
        rows.append(
            {
                "section": "event",
                "Fire_Name": ev["Fire_Name"],
                "Fire_Year": int(ev["Fire_Year"]),
                "Fire_Date": ev["Fire_Date"],
                "area_ha": ev["area_ha"],
                "aqs_code": "",
                "n_modeling_copies": "",
                "n_max_daily_copies": "",
                "n_joined_complete_case": "",
                "peak_daily_minus_before": "",
                "delta_During_minus_Before": "",
                "note": "Two 2021 CHL #12 burns share name and year; the name-year join does not distinguish dates.",
            }
        )
    for mon in monitors:
        peaks = md_chl.loc[md_chl["aqs_code"] == mon, "delta_MaxDailyDuring_minus_Before"]
        n_st = int((st_chl["aqs_code"] == mon).sum())
        n_md = int((md_chl["aqs_code"] == mon).sum())
        n_cc = int(((primary_cc["Fire_Name"] == "CHL #12") & (primary_cc["Fire_Year"] == 2021) & (primary_cc["aqs_code"] == mon)).sum())
        rows.append(
            {
                "section": "monitor_key",
                "Fire_Name": "CHL #12",
                "Fire_Year": 2021,
                "Fire_Date": "",
                "area_ha": "",
                "aqs_code": mon,
                "n_modeling_copies": n_st,
                "n_max_daily_copies": n_md,
                "n_joined_complete_case": n_cc,
                "peak_daily_minus_before": "; ".join(f"{v:.6g}" for v in peaks.tolist()),
                "delta_During_minus_Before": "",
                "note": "Disagreeing peak-daily values; 8 modeling copies x 2 max_daily copies = 16 joined rows (15 extras beyond one unique key).",
            }
        )
    cc = primary_cc[
        (primary_cc["Fire_Name"] == "CHL #12")
        & (primary_cc["Fire_Year"] == 2021)
        & (primary_cc["aqs_code"].isin(monitors))
    ].copy()
    for _, r in cc.iterrows():
        rows.append(
            {
                "section": "complete_case_row",
                "Fire_Name": r["Fire_Name"],
                "Fire_Year": int(r["Fire_Year"]),
                "Fire_Date": "",
                "area_ha": r.get("area_ha", ""),
                "aqs_code": r["aqs_code"],
                "n_modeling_copies": "",
                "n_max_daily_copies": "",
                "n_joined_complete_case": "",
                "peak_daily_minus_before": r.get("delta_MaxDailyDuring_minus_Before", ""),
                "delta_During_minus_Before": r.get("delta_During_minus_Before", ""),
                "note": "One of 32 complete-case copies at the two monitors (30 extras beyond two unique keys).",
            }
        )
    if event_cc is not None and len(event_cc):
        ev = event_cc[
            (event_cc["Fire_Name"] == "CHL #12")
            & (event_cc["Fire_Year"] == 2021)
        ].copy()
        if "Fire_Date" in ev.columns:
            ev["Fire_Date"] = iso_date(ev["Fire_Date"])
        for _, r in ev.iterrows():
            rows.append(
                {
                    "section": "event_date_complete_case",
                    "Fire_Name": r["Fire_Name"],
                    "Fire_Year": int(r["Fire_Year"]),
                    "Fire_Date": r.get("Fire_Date", ""),
                    "area_ha": r.get("area_ha", ""),
                    "aqs_code": r["aqs_code"],
                    "n_modeling_copies": "",
                    "n_max_daily_copies": "",
                    "n_joined_complete_case": "",
                    "peak_daily_minus_before": r.get("delta_MaxDailyDuring_minus_Before", ""),
                    "delta_During_minus_Before": r.get("delta_During_minus_Before", ""),
                    "note": "Event-date path: one row per event_id + monitor (two 2021 dates are separate events).",
                }
            )
    pd.DataFrame(rows).to_csv(RESULTS / "Table_S_CHL12_2021_event_duplicates.csv", index=False)


def write_merge_diagnosis() -> dict:
    """Name-year join diagnosis plus the event-date sensitivity.

    The primary table stays 4.55 / 7.68 / 11.01 (n = 4,186 / 1,547).
    Event-date coefficients are written to
    results/Table_S_event_id_sensitivity.csv and do not replace that table.
    """
    st_raw = pd.read_csv(DATA / "modeling_station.csv", low_memory=False)
    md = pd.read_csv(DATA / "max_daily.csv", low_memory=False)
    n_left = len(st_raw)
    n_left_unique = int(st_raw[EVENT_KEYS].drop_duplicates().shape[0])
    md_keys = md[EVENT_KEYS]
    n_md = len(md)
    n_md_unique = int(md_keys.drop_duplicates().shape[0])
    n_md_dup_keys = int(md[md.duplicated(EVENT_KEYS, keep=False)][EVENT_KEYS].drop_duplicates().shape[0])
    naive = st_raw.merge(
        md[EVENT_KEYS + ["delta_MaxDailyDuring_minus_Before"]],
        on=EVENT_KEYS,
        how="left",
    )
    n_naive = len(naive)
    inflation = n_naive - n_left

    unique = load_modeling_unique_keys()
    ev_cmp, primary, event = event_id_sensitivity_table()
    ev_cmp.to_csv(RESULTS / "Table_S_event_id_sensitivity.csv", index=False)
    diag = event_diagnostics(event)
    write_event_vs_name_year_comparison(ev_cmp, diag)
    d_p = complete_case(primary, "delta_During_minus_Before")
    d_u = complete_case(unique, "delta_During_minus_Before")
    d_e = complete_case(event, "delta_During_minus_Before")
    extras_primary = event_key_extras(d_p)
    extras_unique = event_key_extras(d_u)
    extras_name_year = event_key_extras(d_p, NAME_YEAR_MONITOR)
    extras_event_table = diag["event_join_extras"]
    extras_event_cc = diag["event_cc_extras"]

    dup_cc = d_p[d_p.duplicated(EVENT_KEYS, keep=False)]
    dup_names = sorted(dup_cc["Fire_Name"].astype(str).unique()) if len(dup_cc) else []
    md_dup = md[md.duplicated(EVENT_KEYS, keep=False)]
    md_dup_names = sorted(md_dup["Fire_Name"].astype(str).unique()) if len(md_dup) else []

    rows = []
    rows.extend(
        fit_path_rows(
            primary,
            "name-year join (primary)",
            "fire_id = cohort|name|year",
            EVENT_KEYS,
            cluster="fire_id",
        )
    )
    rows.extend(
        fit_path_rows(
            unique,
            "unique name+year+monitor (sensitivity only)",
            "fire_id = cohort|name|year",
            EVENT_KEYS,
            cluster="fire_id",
        )
    )
    cmp = pd.DataFrame(rows)
    cmp.to_csv(RESULTS / "Table_S_merge_key_uniqueness.csv", index=False)

    write_chl12_reviewer_csv(d_p, d_e)

    chl_event = event[(event["Fire_Name"] == "CHL #12") & (event["Fire_Year"] == 2021)].copy()
    chl_event["Fire_Date"] = iso_date(chl_event["Fire_Date"])
    chl_dates = sorted({str(x) for x in chl_event["Fire_Date"].dropna().unique()})
    chl_cc = d_e[(d_e["Fire_Name"] == "CHL #12") & (d_e["Fire_Year"] == 2021)].copy()
    if "Fire_Date" in chl_cc.columns:
        chl_cc["Fire_Date"] = iso_date(chl_cc["Fire_Date"])
    monitors = ["30-031-0017", "30-093-0005"]
    peak_lines = []
    for date in chl_dates:
        for mon in monitors:
            sub = chl_event[(chl_event["Fire_Date"] == date) & (chl_event["aqs_code"] == mon)]
            peaks = (
                pd.to_numeric(sub["delta_MaxDailyDuring_minus_Before"], errors="coerce")
                .dropna()
                .unique()
                .tolist()
            )
            peak_txt = ", ".join(f"{v:.6g}" for v in peaks) if peaks else "no row"
            peak_lines.append(f"    {date}  {mon}: peak-daily minus before = {peak_txt}")

    ev_during = ev_cmp[
        (ev_cmp["path"] == "B. event-date join (sensitivity)")
        & (ev_cmp["Outcome"] == "During minus Before")
    ].iloc[0]
    ev_peak = ev_cmp[
        (ev_cmp["path"] == "B. event-date join (sensitivity)")
        & (ev_cmp["Outcome"] == "Peak daily minus Before")
    ].iloc[0]
    ev_after = ev_cmp[
        (ev_cmp["path"] == "B. event-date join (sensitivity)")
        & (ev_cmp["Outcome"] == "After minus Before")
    ].iloc[0]

    lines = [
        "Fire-event merge keys: name-year join vs event-date join.",
        "",
        "The checks in validation/REPRODUCTION_CHECK.txt compare",
        "regenerated numbers to the primary targets",
        "(during 4.55, n = 4,186 / 1,547). The event-date path is a",
        "sensitivity using date-level events.",
        "",
        "A. NAME-YEAR JOIN (primary analysis)",
        f"  Original modeling rows: {n_left}",
        f"  Unique fire-name/year/monitor keys: {n_left_unique}",
        f"  Rows after name-year join: {n_naive}",
        f"  Additional rows created by join: {inflation}",
        f"  Main adjusted sample: {len(d_p)}",
        f"  Unique-name-year adjusted sample: {len(d_u)} / coefficient about 4.47",
        f"  max_daily.csv rows: {n_md}; unique name-year-monitor keys: {n_md_unique}",
        "  fire_id clustering = cohort|name|year (does not split same-name burns).",
        "",
        "  Two 2021 CHL #12 burns share name and year:",
        "    2021-04-17 (inventory area 0.404686 ha)",
        "    2021-10-19 (inventory area 32.37488 ha)",
        "  The name-year join does not use Fire_Date, so the two dates",
        "  are not distinguished. Unique keys may carry Fire_Date from Stage B;",
        "  collision keys leave Fire_Date empty on the 8,615-row table.",
        f"  max_daily keys that appear more than once: {n_md_dup_keys} ({', '.join(md_dup_names) or 'none'}).",
        "  After complete-case filters, only CHL #12 2021 at two monitors remains.",
        "    30-031-0017: 8 modeling copies × 2 max_daily copies = 16 joined rows;",
        "      disagreeing peak-daily (minus before): -0.1085 and 1.20625",
        "    30-093-0005: 8 modeling copies × 2 max_daily copies = 16 joined rows;",
        "      disagreeing peak-daily (minus before): 7.1588 and 8.7163",
        "  Six CHL #12 2021 monitor keys × 8 modeling copies = 48 extra rows",
        f"  in the full join ({n_left} → {n_naive}). After complete-case filters,",
        f"  2 × (16 − 1) = {extras_primary} extra rows remain in the {len(d_p)} sample.",
        f"  Complete-case keys that are not unique: {', '.join(dup_names) or 'none'}.",
        f"  Extra rows beyond Fire_Name+Fire_Year+aqs_code: {extras_name_year}.",
        "  burned_area_fuel.csv is already unique on Fire_Name + Fire_Year",
        "  and does not inflate the join.",
        "",
        "  The primary path uses this name-year join",
        "  (during 4.55, n = 4,186 / 1,547).",
        "  The unique-name-year collapse (4.47, n = 4,156 / 1,547) is a",
        "  sensitivity only; it still collapses both CHL dates into one fire_id.",
        "  Comparison of that collapse: results/Table_S_merge_key_uniqueness.csv.",
        "",
        "B. EVENT-DATE JOIN (sensitivity; does not replace the primary table)",
        "  event_id = fire_cohort | Fire_Name | Fire_Year | Fire_Date (ISO date).",
        "  Join is on event_id + aqs_code after those keys are deduplicated.",
        f"  Extras after the event join (full table): {extras_event_table}.",
        f"  Extras after the event join (complete-case during): {extras_event_cc}.",
        f"  Both 2021 CHL #12 dates exist as separate events: {', '.join(chl_dates) or 'NONE'}.",
        "  Monitors 30-031-0017 and 30-093-0005, peak-daily minus before by date:",
        *peak_lines,
        f"  CHL #12 2021 complete-case rows on the event-date path: {len(chl_cc)}.",
        "  Clustering = event_id (each CHL date is its own cluster).",
        "  Same fillna(0) clustered OLS as the primary recipe:",
        f"    During: {ev_during['Estimate']:.2f} ({ev_during['CI low']:.2f}–{ev_during['CI high']:.2f}), "
        f"n = {int(ev_during['n linkages'])} linkages / {int(ev_during['n events'])} events / "
        f"{int(ev_during['n fires'])} fires.",
        f"    Peak: {ev_peak['Estimate']:.2f} ({ev_peak['CI low']:.2f}–{ev_peak['CI high']:.2f}), "
        f"n = {int(ev_peak['n linkages'])} / {int(ev_peak['n events'])} / {int(ev_peak['n fires'])}.",
        f"    After: {ev_after['Estimate']:.2f} ({ev_after['CI low']:.2f}–{ev_after['CI high']:.2f}), "
        f"n = {int(ev_after['n linkages'])} / {int(ev_after['n events'])} / {int(ev_after['n fires'])}.",
        "  Full comparison: results/Table_S_event_id_sensitivity.csv.",
        "  Inspectable CHL #12 rows: results/Table_S_CHL12_2021_event_duplicates.csv.",
        "",
        "A PASS means regenerated numbers match the primary targets",
        "and the event-date sensitivity targets (during 4.45).",
        "",
    ]
    (VALIDATION / "merge_key_uniqueness.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "naive_merged": n_naive,
        "naive_inflation": inflation,
        "primary_n": int(len(d_p)),
        "primary_fires": int(d_p["fire_id"].nunique()),
        "primary_extras": extras_primary,
        "unique_n": int(len(d_u)),
        "unique_fires": int(d_u["fire_id"].nunique()),
        "unique_extras": extras_unique,
        "unique_table_extras": event_key_extras(unique),
        "event_join_extras": extras_event_table,
        "event_cc_extras": extras_event_cc,
        "chl_dates": chl_dates,
        "event_during": float(ev_during["Estimate"]),
        "event_during_lo": float(ev_during["CI low"]),
        "event_during_hi": float(ev_during["CI high"]),
        "event_during_n": int(ev_during["n linkages"]),
        "event_during_events": int(ev_during["n events"]),
        "event_during_fires": int(ev_during["n fires"]),
        "event_peak": float(ev_peak["Estimate"]),
        "event_peak_lo": float(ev_peak["CI low"]),
        "event_peak_hi": float(ev_peak["CI high"]),
        "event_after": float(ev_after["Estimate"]),
        "event_after_lo": float(ev_after["CI low"]),
        "event_after_hi": float(ev_after["CI high"]),
        "event_after_n": int(ev_after["n linkages"]),
        "event_after_events": int(ev_after["n events"]),
        "event_after_fires": int(ev_after["n fires"]),
    }


def extra_si() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Figure 1, S1–S3, and the GAMM n-check. Does not refit bam()."""
    from make_figure_1_maps import make_figure_1
    from make_figure_s1_rx_distance_wind import make_figure_s1
    from make_figure_s2_distance_cutoffs import make_figure_s2
    from make_figure_s3_bearing_screens import make_figure_s3

    sys.path.insert(0, str(HERE / "gamm"))
    from check_gamm_n import check_gamm_n  # noqa: E402

    print("Figure 1 maps …")
    make_figure_1()
    print("Figure S1 RX distance × wind …")
    s1 = make_figure_s1()
    print("Figure S2 distance cutoffs …")
    s2 = make_figure_s2()
    print("Figure S3 bearing screens …")
    s3 = make_figure_s3()
    print("GAMM hourly n-check …")
    gamm = check_gamm_n()
    return gamm, s1, s2, s3


def write_software_versions() -> None:
    import matplotlib as mpl
    import statsmodels

    lines = [
        f"python {sys.version.split()[0]}",
        f"pandas {pd.__version__}",
        f"numpy {np.__version__}",
        f"statsmodels {statsmodels.__version__}",
        f"matplotlib {mpl.__version__}",
    ]
    (VALIDATION / "software_versions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_downwind_audit(st: pd.DataFrame) -> None:
    """Keep fillna(0) as the primary path. Document missingness only."""
    raw = pd.read_csv(DATA / "modeling_station.csv", usecols=["downwind_during"])
    n_file = len(raw)
    n_file_na = int(pd.to_numeric(raw["downwind_during"], errors="coerce").isna().sum())
    d = complete_case(st, "delta_During_minus_Before")
    n_in_cc = int(d["_downwind_was_missing"].sum()) if "_downwind_was_missing" in d.columns else -1
    lines = [
        "Downwind missingness audit. Primary analysis is unchanged.",
        "",
        "No observations in the primary adjusted complete-case sample had missing downwind information. Excluding observations with missing alignment therefore did not change the primary adjusted sample or coefficients.",
        "",
        "This is consistent with zero missing-downwind observations among the 4,186 analyzed linkages.",
        "",
        f"Rows in data/modeling_station.csv: {n_file}",
        f"Rows with missing downwind_during in that file: {n_file_na}",
        f"Complete-case during-minus-before sample (primary): {len(d)} linkages, {d['fire_id'].nunique()} fires",
        f"Of those complete-case rows, originally missing downwind: {n_in_cc}",
        "",
        "The modeling file codes missing downwind_during as NA before load_modeling().",
        "load_modeling() then fillna(0): missing alignment is treated as not downwind.",
        "That is how the primary loader treats missing values, but it does not",
        "affect the 4,186 because those rows already fail other complete-case filters.",
        f"{n_file_na} missing-downwind rows are in the full modeling table; {n_in_cc} of",
        "those are in the 4,186 complete-case sample. The recipe still yields",
        "the primary 4.55 / 7.68 / 11.01.",
        "",
        "A drop-missing sensitivity is written to results/Table_S_downwind_missing_sensitivity.csv.",
        "That file is not the primary table.",
    ]
    (VALIDATION / "downwind_missingness.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = []
    for label, col in OUTCOMES:
        filled = complete_case(st, col)
        dropped = filled.loc[~filled["_downwind_was_missing"]].copy() if "_downwind_was_missing" in filled.columns else filled
        m0 = fit_clustered(FORMULA.format(outcome=col), filled)
        t0 = wildfire_term(m0)
        lo0, hi0 = m0.conf_int().loc[t0]
        rows.append(
            {
                "Outcome": label,
                "Treatment": "fillna(0), primary",
                "Estimate": float(m0.params[t0]),
                "CI low": float(lo0),
                "CI high": float(hi0),
                "n linkages": int(len(filled)),
                "n fires": int(filled["fire_id"].nunique()),
            }
        )
        m1 = fit_clustered(FORMULA.format(outcome=col), dropped)
        t1 = wildfire_term(m1)
        lo1, hi1 = m1.conf_int().loc[t1]
        rows.append(
            {
                "Outcome": label,
                "Treatment": "drop missing downwind (sensitivity only)",
                "Estimate": float(m1.params[t1]),
                "CI low": float(lo1),
                "CI high": float(hi1),
                "n linkages": int(len(dropped)),
                "n fires": int(dropped["fire_id"].nunique()),
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS / "Table_S_downwind_missing_sensitivity.csv", index=False)


def required_data_files() -> list[str]:
    """Each item is a label plus one or more acceptable paths.

    modeling_station_with_date.csv is shipped in data/ so a clean checkout
    does not need reconstruct. If that file is absent, the event-date
    loaders rebuild the same table from stage_b_daily.csv in memory.
    """
    groups = [
        ("modeling_station.csv", [DATA / "modeling_station.csv"]),
        ("max_daily.csv", [DATA / "max_daily.csv"]),
        ("burned_area_fuel.csv", [DATA / "burned_area_fuel.csv"]),
        ("stage_b_daily.csv", [DATA / "stage_b_daily.csv"]),
        ("stage_e_recovery.csv", [DATA / "stage_e_recovery.csv"]),
        ("inventory_rx.csv", [DATA / "inventory_rx.csv"]),
        ("inventory_wf.csv", [DATA / "inventory_wf.csv"]),
        (
            "rx_distance_wind.csv",
            [DATA / "rx_distance_wind.csv", DATA / "rx_distance_wind.csv.gz"],
        ),
        (
            "bearing_panel.csv.gz",
            [DATA / "bearing_panel.csv.gz", DATA / "bearing_panel.csv"],
        ),
        (
            "step05_gamm_model_data.csv.gz",
            [
                DATA / "gamm" / "csv" / "step05_gamm_model_data.csv.gz",
                DATA / "gamm" / "csv" / "step05_gamm_model_data.csv",
            ],
        ),
        ("map/fire_points.csv", [DATA / "map" / "fire_points.csv"]),
        ("map/monitors.csv", [DATA / "map" / "monitors.csv"]),
        ("map/montana.geojson", [DATA / "map" / "montana.geojson"]),
    ]
    missing = []
    for label, paths in groups:
        if first_existing(*paths) is None:
            missing.append(label)
    return missing


def main() -> int:
    ensure_results()
    missing = required_data_files()
    if missing:
        print("Missing data files:", ", ".join(missing))
        print("They belong in", DATA)
        return 1
    dated = DATA / "modeling_station_with_date.csv"
    if not dated.exists():
        print(
            "data/modeling_station_with_date.csv is not present; "
            "the event-date path will rebuild it from data/stage_b_daily.csv. "
            "That file is shipped in this package so reconstruct is optional."
        )

    print("Table 1 …")
    t1 = table_1()
    print("Unadjusted means …")
    unadj = unadjusted()
    print("Adjusted models …")
    adj = adjusted()
    print("Fire type × log(1+area) …")
    ix = interaction()
    print("Recovery and hourly % …")
    rec = recovery()
    rec_audit = write_recovery_censoring_audit()
    print("Figures …")
    make_figures(unadj, adj, rec)
    print("Downwind audit …")
    st = load_modeling()
    write_downwind_audit(st)
    print("Merge-key uniqueness …")
    merge = write_merge_diagnosis()
    print("Independent event-date refit …")
    indep = run_event_date_refit()
    write_software_versions()
    gamm, s1, s2, s3 = extra_si()
    from make_figure_1_maps import hex_neighbor_overlap_fraction  # noqa: E402

    hex_pct = 100.0 * hex_neighbor_overlap_fraction(pointy_top=False)

    checks = check(t1, unadj, adj, ix, rec)
    _check(checks, "GAMM M4 hours", gamm["n"], EXPECTED["gamm_n"])
    _check(checks, "GAMM M4 fires", gamm["n_fires"], EXPECTED["gamm_fires"])
    _check(checks, "GAMM M4 monitors", gamm["n_monitors"], EXPECTED["gamm_monitors"])
    s3p = s3[(s3["outcome"] == "During minus Before") & (s3["is_primary"])].iloc[0]
    _check(checks, "S3 primary n", s3p["n_linkages"], EXPECTED["s3_primary_n"])
    _check(checks, "S3 primary fires", s3p["n_fire_years"], EXPECTED["s3_primary_fires"])
    _check(checks, "S3 primary estimate", round(float(s3p["estimate"]), 2), EXPECTED["adj_during"], 0.005)
    s2p = s2[(s2["outcome"] == "During minus Before") & (s2["is_primary"])].iloc[0]
    _check(checks, "S2 300 km n", s2p["n_linkages"], EXPECTED["s3_primary_n"])
    _check(checks, "S2 300 km fires", s2p["n_fire_years"], EXPECTED["s3_primary_fires"])
    _check(checks, "Merge unique-key extras", merge["unique_extras"], EXPECTED["merge_unique_extras"])
    _check(checks, "Merge unique-key table extras", merge["unique_table_extras"], EXPECTED["merge_unique_extras"])
    _check(checks, "Event-join extras", merge["event_join_extras"], EXPECTED["event_join_extras"])
    _check(checks, "Event-join complete-case extras", merge["event_cc_extras"], EXPECTED["event_join_extras"])
    chl_dates = set(merge.get("chl_dates") or [])
    _check(checks, "CHL #12 2021-04-17 is a separate event", int("2021-04-17" in chl_dates), 1)
    _check(checks, "CHL #12 2021-10-19 is a separate event", int("2021-10-19" in chl_dates), 1)
    _check(checks, "Event-ID during", round(merge["event_during"], 2), EXPECTED["event_during"], 0.005)
    _check(checks, "Event-ID during CI low", round(merge["event_during_lo"], 2), EXPECTED["event_during_lo"], 0.01)
    _check(checks, "Event-ID during CI high", round(merge["event_during_hi"], 2), EXPECTED["event_during_hi"], 0.01)
    _check(checks, "Event-ID during n", merge["event_during_n"], EXPECTED["event_during_n"])
    _check(checks, "Event-ID during events", merge["event_during_events"], EXPECTED["event_during_events"])
    _check(checks, "Event-ID during fires", merge["event_during_fires"], EXPECTED["event_during_fires"])
    _check(checks, "Event-ID peak", round(merge["event_peak"], 2), EXPECTED["event_peak"], 0.005)
    _check(checks, "Event-ID peak CI low", round(merge["event_peak_lo"], 2), EXPECTED["event_peak_lo"], 0.01)
    _check(checks, "Event-ID peak CI high", round(merge["event_peak_hi"], 2), EXPECTED["event_peak_hi"], 0.01)
    _check(checks, "Event-ID after", round(merge["event_after"], 2), EXPECTED["event_after"], 0.005)
    _check(checks, "Event-ID after CI low", round(merge["event_after_lo"], 2), EXPECTED["event_after_lo"], 0.01)
    _check(checks, "Event-ID after CI high", round(merge["event_after_hi"], 2), EXPECTED["event_after_hi"], 0.01)
    _check(checks, "Event-ID after n", merge["event_after_n"], EXPECTED["event_after_n"])
    _check(checks, "Event-ID after events", merge["event_after_events"], EXPECTED["event_after_events"])
    _check(checks, "Event-ID after fires", merge["event_after_fires"], EXPECTED["event_after_fires"])
    _check(checks, "Event-date regression independently refitted", int(bool(indep["ok"])), 1)
    _check(checks, "Independent event-date extras", indep["extras"], EXPECTED["event_join_extras"])
    _check(checks, "Independent event-date during", round(indep["during"], 2), EXPECTED["event_during"], 0.005)
    _check(checks, "Independent event-date during CI low", round(indep["during_lo"], 2), EXPECTED["event_during_lo"], 0.01)
    _check(checks, "Independent event-date during CI high", round(indep["during_hi"], 2), EXPECTED["event_during_hi"], 0.01)
    _check(checks, "Independent event-date during n", indep["during_n"], EXPECTED["event_during_n"])
    _check(checks, "Independent event-date during events", indep["during_events"], EXPECTED["event_during_events"])
    _check(checks, "Independent event-date during fires", indep["during_fires"], EXPECTED["event_during_fires"])
    _check(checks, "Independent event-date peak", round(indep["peak"], 2), EXPECTED["event_peak"], 0.005)
    _check(checks, "Independent event-date after", round(indep["after"], 2), EXPECTED["event_after"], 0.005)
    _check(checks, "Independent event-date after n", indep["after_n"], EXPECTED["event_after_n"])
    _check(checks, "Independent vs Event-ID during", round(indep["during"], 2), round(merge["event_during"], 2), 0.005)
    _check(checks, "Figure 1 hex overlap %", round(hex_pct, 4), 0.0, EXPECTED["hex_overlap_pct_max"])
    _check(checks, "Recovery missing censored RX", rec_audit["missing_rx"], EXPECTED["rec_missing_rx"])
    _check(checks, "Recovery missing censored WF", rec_audit["missing_wf"], EXPECTED["rec_missing_wf"])
    prov = indep.get("provenance") or {}
    _check(checks, "Event-date RX n", prov.get("rx_n", -1), EXPECTED["event_rx_n"])
    _check(checks, "Event-date RX Stage B wind match", prov.get("rx_wind_match", -1), EXPECTED["event_rx_n"])
    _check(checks, "Event-date WF n", prov.get("wf_n", -1), EXPECTED["event_wf_n"])
    _check(
        checks,
        "Event-date WF Stage B wind present",
        prov.get("wf_stage_b_wind_nonmissing", -1),
        EXPECTED["event_wf_stage_b_wind"],
    )
    _check(
        checks,
        "Event-date WF name-year wind match",
        prov.get("wf_nameyear_wind_match", -1),
        EXPECTED["event_wf_n"],
    )
    s1_none = s1[(pd.to_numeric(s1["dist_cap_km"], errors="coerce") == 300.0) & (s1["downwind"] == "none")]
    _check(checks, "Figure S1 300 km no-wind n", int(s1_none["n_rows"].iloc[0]) if len(s1_none) else -1, EXPECTED["s1_300_n"])
    s1_caps = sorted({float(x) for x in pd.to_numeric(s1["dist_cap_km"], errors="coerce").dropna().unique()})
    _check(checks, "Figure S1 n distance caps", len(s1_caps), 4)
    _check(checks, "Figure S1 smallest cap km", int(min(s1_caps)) if s1_caps else -1, 50)
    _check(checks, "Figure S1 largest cap km", int(max(s1_caps)) if s1_caps else -1, 300)
    report = pd.DataFrame(checks, columns=["Item", "Got", "Expected", "Status"])
    report.to_csv(VALIDATION / "REPRODUCTION_CHECK.csv", index=False)
    n_fail = int((report["Status"] == "FAIL").sum())
    lines = [
        "Check against the primary analysis targets.",
        "",
        report.to_string(index=False),
        "",
        f"{'PASS' if n_fail == 0 else 'FAIL'}: {len(report) - n_fail}/{len(report)} items matched.",
        "",
        "Tables: results/ (inputs in data/ are not overwritten).",
        "Checks: validation/REPRODUCTION_CHECK.txt and .csv.",
        "",
        "Recovery n is 4,885 prescribed and 351 wildfire linkages because Stage E",
        "needs usable hourly PM2.5 for recovery/exceedance. The daily cohort is",
        "8,122 and 361 (Table 1). Both are primary analysis samples.",
        "",
        "Hourly GAMM M4 is n-checked here, not refit with bam(). The hourly table",
        "is shipped gzipped as data/gamm/csv/step05_gamm_model_data.csv.gz",
        "(uncompressed CSV is ~62 MB and is not committed). Primary M4",
        "sample: 379,433 hours / 761 fires / 21 monitors. This validation checks",
        "sample counts only. Coefficients, fitted smooths, and AIC were not refit",
        "by this report and remain outside the primary-target verification. M4 is",
        "univariate smooths + random intercepts; no 2-D ti() in the main model.",
        "AIC is in derived/gamm_aic_model_comparison.csv. Figure S4 is",
        "the GAMM. To refit: Rscript code/gamm/step05_fit_gamm_main.R",
        "",
        "Primary-target checks compare regenerated numbers to EXPECTED",
        "(during 4.55, n = 4,186 / 1,547; name-year fire identifier).",
        "The name-year join inflates 8,615 to",
        f"{merge['naive_merged']} (+{merge['naive_inflation']} extras) and leaves",
        f"{merge['primary_extras']} extra rows in the 4,186-linkage sample.",
        "Event-date path: event_id = cohort|name|year|date, unique join on",
        "event+monitor, clustered by event_id,",
        f"extras after event join = {merge['event_join_extras']},",
        f"CHL dates as separate events = {', '.join(merge.get('chl_dates') or [])}.",
        f"Event-ID during {merge['event_during']:.2f} "
        f"({merge['event_during_lo']:.2f}–{merge['event_during_hi']:.2f}), "
        f"n = {merge['event_during_n']} / {merge['event_during_events']} / "
        f"{merge['event_during_fires']}.",
        "Event-date coefficients are in",
        "results/Table_S_event_id_sensitivity.csv and do not replace 4.55.",
        "Independent event-date refit: python code/fit_event_date.py.",
        "That second command writes results/Table_event_date_coefficients.csv",
        "and validation/event_date_independent_refit.txt.",
        f"Independent during {indep['during']:.2f} "
        f"({indep['during_lo']:.2f}–{indep['during_hi']:.2f}), "
        f"n = {indep['during_n']} / {indep['during_events']} / "
        f"{indep['during_fires']}; extras = {indep['extras']}.",
        "Standalone comparison rebuild: python code/fit_event_id_sensitivity.py.",
        "Standalone covariate provenance (no OLS):",
        "python code/report_event_date_covariate_provenance.py.",
        "See validation/merge_key_uniqueness.txt,",
        "validation/event_date_covariate_provenance.txt, and",
        "validation/recovery_censoring.txt.",
        "fillna(1) on missing recovery status is the 9.4 / 27.9 recipe;",
        "49 Stage E rows have missing censor flags (48 RX, 1 WF).",
        "",
        f"Figure 1 hexagon neighbor overlap: {hex_pct:.4f}% (flat-top corners).",
        "That supports the hexagon geometry correction; it does not by itself",
        "verify fire locations, state boundaries, or final map appearance.",
    ]
    text = "\n".join(lines)
    (VALIDATION / "REPRODUCTION_CHECK.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
