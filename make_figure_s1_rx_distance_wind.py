"""Figure S1: RX-only descriptive distance × wind-alignment grid.

Not a 2-D GAMM and not the geographic 45–135° bearing screen.
Columns: extra distance caps ≤50 / ≤100 / ≤200 / ≤300 km.
Rows: meteorological alignment ±45 / ±60 / ±67.5° / none.
The Stage B cohort is already ≤300 km, so a 500 km or unlimited cap
would duplicate ≤300 km.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from paths import DATA, FIGURES, RESULTS, ensure_results, first_existing  # noqa: E402

DPI = 150
DIST_CAPS = [50.0, 100.0, 200.0, 300.0]
DIST_LABELS = ["≤50 km", "≤100 km", "≤200 km", "≤300 km"]
WIND_TOL = [45.0, 60.0, 67.5, None]
WIND_KEYS = ["45deg", "60deg", "67.5deg", "none"]
WIND_LABELS = {
    "45deg": "±45°",
    "60deg": "±60°",
    "67.5deg": "±67.5°",
    "none": "No wind-alignment\nfilter",
}


def build_grid(df: pd.DataFrame) -> pd.DataFrame:
    dist = pd.to_numeric(df["Distance_km"], errors="coerce")
    ang = pd.to_numeric(df["alignment_angle_deg"], errors="coerce")
    during = pd.to_numeric(df["delta_During_minus_Before"], errors="coerce")
    peak = pd.to_numeric(df["delta_MaxDailyDuring_minus_Before"], errors="coerce")
    rows = []
    for cap in DIST_CAPS:
        dmask = dist.notna() if cap is None else dist.le(cap)
        for tol, key in zip(WIND_TOL, WIND_KEYS):
            if tol is None:
                wmask = pd.Series(True, index=df.index)
            else:
                wmask = ang.notna() & ang.le(tol)
            keep = dmask & wmask & during.notna()
            sub_d = during[keep]
            sub_p = peak[keep]
            rows.append(
                {
                    "dist_cap_km": cap if cap is not None else np.nan,
                    "downwind": key,
                    "n_rows": int(keep.sum()),
                    "mean_During_minus_Before": float(sub_d.mean()) if keep.any() else np.nan,
                    "median_During_minus_Before": float(sub_d.median()) if keep.any() else np.nan,
                    "mean_MaxDailyDuring_minus_Before": float(sub_p.mean()) if keep.any() else np.nan,
                    "median_MaxDailyDuring_minus_Before": float(sub_p.median()) if keep.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_heatmap(grid: pd.DataFrame, out_png: Path) -> None:
    g = grid.copy()
    g["dist_key"] = g["dist_cap_km"].map(lambda x: "" if pd.isna(x) else f"{float(x):.1f}")
    piv = g.pivot(index="downwind", columns="dist_key", values="mean_During_minus_Before")
    col_keys = [f"{c:.1f}" for c in DIST_CAPS]
    piv = piv.reindex(index=WIND_KEYS, columns=[c for c in col_keys if c in piv.columns])
    npiv = g.pivot(index="downwind", columns="dist_key", values="n_rows")
    npiv = npiv.reindex(index=piv.index, columns=piv.columns)
    data = piv.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 0.5
    vmax = max(vmax, 0.4)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=DPI)
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([DIST_LABELS[col_keys.index(str(c))] for c in piv.columns], fontsize=9)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([WIND_LABELS[str(r)] for r in piv.index], fontsize=9)
    ax.set_xlabel("Maximum fire–monitor distance")
    ax.set_ylabel("Meteorological wind-alignment tolerance\n(not the geographic 45–135° east-sector screen)")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isfinite(val):
                continue
            n = int(npiv.iloc[i, j])
            txt_color = "white" if abs(val) > 0.55 * vmax else "#1e293b"
            ax.text(j, i, f"{val:.2f}\nn={n:,}", ha="center", va="center", fontsize=8, color=txt_color)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.90, bottom=0.28)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.055, pad=0.18, aspect=34)
    cbar.set_label("Mean During − Before PM$_{2.5}$ (µg m$^{-3}$; prescribed-fire linkages only)")
    fig.suptitle("Figure S1. Prescribed-fire distance × wind-alignment grid", fontsize=11, fontweight="600")
    fig.savefig(out_png, facecolor="white", bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def make_figure_s1() -> pd.DataFrame:
    ensure_results()
    src = first_existing(DATA / "rx_distance_wind.csv", DATA / "rx_distance_wind.csv.gz")
    if src is None:
        raise FileNotFoundError("Missing data/rx_distance_wind.csv or .csv.gz")
    df = pd.read_csv(src)
    grid = build_grid(df)
    grid.to_csv(RESULTS / "Figure_S1_rx_distance_wind_grid.csv", index=False)
    png = FIGURES / "Figure_S1_rx_distance_wind.png"
    plot_heatmap(grid, png)
    (FIGURES / "Figure_S1_caption.txt").write_text(
        "Figure S1. Sensitivity of the during-minus-before PM2.5 response among prescribed-fire "
        "linkages to extra distance caps (≤50, ≤100, ≤200, ≤300 km) and meteorological "
        "wind-alignment tolerances (±45, ±60, ±67.5°, or none). Cell values are descriptive "
        "means, not adjusted wildfire-versus-prescribed coefficients and not a 2-D GAMM. "
        "The underlying Stage B geographic cohort already uses the east-sector screen "
        "(bearing 45–135°, ≤300 km), so ≤300 km is the full RX during sample; a 500 km "
        "or unlimited cap would be identical to ≤300 km and is not shown. Alignment is "
        "the absolute angular difference between the fire-to-monitor bearing and the ISD "
        "smoke-transport direction.\n",
        encoding="utf-8",
    )
    print(grid.to_string(index=False))
    print("Wrote", png)
    return grid


def main() -> None:
    make_figure_s1()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
