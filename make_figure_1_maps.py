"""Figure 1: four-panel fire locations / screening map.

(a) WFIGS prescribed-fire points, MTBS wildfire centroids, PM2.5 monitors
(b) ~20 km-radius hexagons: prescribed only / wildfire only / both
(c–d) 2024 PLN Mosquito RX geographic screen (300 km, 45–135°)

Uses packaged coordinate tables in data/map/. Does not ship MTBS perimeter
polygons; wildfires are drawn at perimeter centroids.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from paths import DATA, FIGURES, VALIDATION, ensure_results, first_existing  # noqa: E402

MAP = DATA / "map"
DPI = 150
R_EARTH_KM = 6371.0
HEX_RADIUS_KM = 20.0
LINK_KM = 300.0
BEARING_MIN, BEARING_MAX = 45.0, 135.0
EXAMPLE_NAME = "PLN Mosquito RX"
EXAMPLE_YEAR = 2024
CITIES = [
    ("Missoula", -114.02, 46.88),
    ("Helena", -112.04, 46.59),
    ("Great Falls", -111.30, 47.50),
    ("Billings", -108.55, 45.78),
    ("Bozeman", -111.04, 45.68),
    ("Kalispell", -114.32, 48.20),
]


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R_EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    y = math.sin(dlmb) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlmb)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def dest_point(lat, lon, bearing, dist_km) -> tuple[float, float]:
    br = math.radians(bearing)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    ang = dist_km / R_EARTH_KM
    lat2 = math.asin(
        math.sin(lat1) * math.cos(ang) + math.cos(lat1) * math.sin(ang) * math.cos(br)
    )
    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(ang) * math.cos(lat1),
        math.cos(ang) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def load_montana_rings() -> list[np.ndarray]:
    geo = json.loads((MAP / "montana.geojson").read_text(encoding="utf-8"))
    geom = geo["features"][0]["geometry"]
    coords = geom["coordinates"]
    rings = []
    if geom["type"] == "Polygon":
        rings = [np.asarray(r) for r in coords]
    elif geom["type"] == "MultiPolygon":
        for poly in coords:
            rings.extend(np.asarray(r) for r in poly)
    return rings


def add_montana(ax, rings, facecolor="#f4efe6", edgecolor="#7c6a4a", lw=0.9, z=1) -> None:
    for ring in rings:
        ax.add_patch(
            MplPolygon(
                ring,
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=lw,
                zorder=z,
            )
        )


def lonlat_to_km(lon, lat, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = (lon - lon0) * 111.32 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110.57
    return x, y


def hex_index(x, y, radius_km: float) -> tuple[np.ndarray, np.ndarray]:
    """Axial indices for *flat-top* hexes with circumradius radius_km.

    Spacing matches hex_corners_xy: x = (3/2) q R, y = sqrt(3) (r + q/2) R.
    """
    q = (2.0 / 3.0 * x) / radius_km
    r = (-1.0 / 3.0 * x + math.sqrt(3.0) / 3.0 * y) / radius_km
    return cube_round(q, r)


def cube_round(q, r) -> tuple[np.ndarray, np.ndarray]:
    s = -q - r
    qn = np.round(q)
    rn = np.round(r)
    sn = np.round(s)
    dq = np.abs(qn - q)
    dr = np.abs(rn - r)
    ds = np.abs(sn - s)
    qn = np.where((dq > dr) & (dq > ds), -rn - sn, qn)
    rn = np.where(~((dq > dr) & (dq > ds)) & (dr > ds), -qn - sn, rn)
    return qn.astype(int), rn.astype(int)


def hex_corners_xy(q: float, r: float, radius_km: float, pointy_top: bool = False) -> np.ndarray:
    """Six vertices in the same km frame as hex_index (flat-top by default).

    Flat-top corners start at 0° (east). Pointy-top corners start at −30°.
    The grid spacing is flat-top; pointy-top corners on that grid overlap.
    """
    cx = radius_km * (3.0 / 2.0 * q)
    cy = radius_km * (math.sqrt(3.0) * (r + q / 2.0))
    offset = -30.0 if pointy_top else 0.0
    corners = []
    for k in range(6):
        ang = math.radians(60 * k + offset)
        corners.append((cx + radius_km * math.cos(ang), cy + radius_km * math.sin(ang)))
    return np.asarray(corners, dtype=float)


def hex_corners(q, r, radius_km: float, lon0: float, lat0: float) -> np.ndarray:
    xy = hex_corners_xy(q, r, radius_km, pointy_top=False)
    lon = lon0 + xy[:, 0] / (111.32 * math.cos(math.radians(lat0)))
    lat = lat0 + xy[:, 1] / 110.57
    return np.column_stack([lon, lat])


def _poly_area(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _inside_edge(p, a, b, eps: float = 1e-12) -> bool:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -eps


def _edge_intersect(p1, p2, a, b) -> np.ndarray:
    r = p2 - p1
    s = b - a
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-18:
        return np.asarray(p1, dtype=float)
    t = ((a[0] - p1[0]) * s[1] - (a[1] - p1[1]) * s[0]) / denom
    return p1 + t * r


def convex_polygon_intersection(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland–Hodgman clip of two convex polygons, CCW vertices."""
    output = np.asarray(subject, dtype=float)
    clip = np.asarray(clipper, dtype=float)
    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        inp = output
        if len(inp) == 0:
            return inp
        kept = []
        prev = inp[-1]
        for cur in inp:
            if _inside_edge(cur, a, b):
                if not _inside_edge(prev, a, b):
                    kept.append(_edge_intersect(prev, cur, a, b))
                kept.append(cur)
            elif _inside_edge(prev, a, b):
                kept.append(_edge_intersect(prev, cur, a, b))
            prev = cur
        output = np.asarray(kept, dtype=float) if kept else np.zeros((0, 2))
    return output


def hex_neighbor_overlap_fraction(radius_km: float = HEX_RADIUS_KM, pointy_top: bool = False) -> float:
    """Intersection / hex area for a cell and its six axial neighbors.

    Correct flat-top corners on the flat-top grid share edges (overlap ~0).
    Pointy-top corners on that same grid overlap by about 2.4%.
    """
    # Flat-top axial neighbors of (0, 0).
    neighbors = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    home = hex_corners_xy(0, 0, radius_km, pointy_top=pointy_top)
    area = _poly_area(home)
    if area <= 0:
        return 1.0
    fracs = []
    for q, r in neighbors:
        other = hex_corners_xy(q, r, radius_km, pointy_top=pointy_top)
        inter = convex_polygon_intersection(home, other)
        fracs.append(_poly_area(inter) / area)
    return float(max(fracs) if fracs else 0.0)


def write_hex_overlap_report() -> dict:
    ensure_results()
    wrong = hex_neighbor_overlap_fraction(pointy_top=True)
    right = hex_neighbor_overlap_fraction(pointy_top=False)
    rec = {
        "orientation": "flat-top",
        "overlap_fraction_flat_top": right,
        "overlap_pct_flat_top": 100.0 * right,
        "overlap_fraction_pointy_on_flat_grid": wrong,
        "overlap_pct_pointy_on_flat_grid": 100.0 * wrong,
    }
    lines = [
        "Figure 1 hexagon overlap check.",
        "",
        "The hex index and center spacing are flat-top (x = 3/2 q R).",
        "Corners must start at 0° (east). Pointy-top corners (−30°) on that",
        "grid overlap neighboring polygons instead of meeting along an edge.",
        "",
        f"Overlap with pointy-top corners (bug): {100.0 * wrong:.3f}%",
        f"Overlap with flat-top corners (current): {100.0 * right:.4f}%",
        "",
        "Neighboring hexes should meet along edges. Tolerance: < 0.05%.",
        f"Status: {'PASS' if right < 0.0005 else 'FAIL'}",
        "",
        "This supports the hexagon geometry correction; it does not by itself",
        "verify fire locations, state boundaries, or final map appearance.",
        "",
    ]
    (VALIDATION / "figure1_hex_overlap.txt").write_text("\n".join(lines), encoding="utf-8")
    return rec


def plot_hex_panel(ax, fires: pd.DataFrame, rings) -> None:
    add_montana(ax, rings)
    lon0, lat0 = -110.0, 47.0
    rx = fires[fires["fire_cohort"] == "WFIGS_RX"]
    wf = fires[fires["fire_cohort"] == "MTBS_wildfire"]
    rx_x, rx_y = lonlat_to_km(rx["Fire_Lon"], rx["Fire_Lat"], lon0, lat0)
    wf_x, wf_y = lonlat_to_km(wf["Fire_Lon"], wf["Fire_Lat"], lon0, lat0)
    rq, rr = hex_index(rx_x, rx_y, HEX_RADIUS_KM)
    wq, wr = hex_index(wf_x, wf_y, HEX_RADIUS_KM)
    rx_keys = set(zip(rq.tolist(), rr.tolist()))
    wf_keys = set(zip(wq.tolist(), wr.tolist()))
    colors = {
        "rx": "#0f766e",
        "wf": "#b45309",
        "both": "#6d28d9",
    }
    patches = {"rx": [], "wf": [], "both": []}
    for q, r in rx_keys | wf_keys:
        if (q, r) in rx_keys and (q, r) in wf_keys:
            key = "both"
        elif (q, r) in rx_keys:
            key = "rx"
        else:
            key = "wf"
        patches[key].append(MplPolygon(hex_corners(q, r, HEX_RADIUS_KM, lon0, lat0), closed=True))
    for key, color in colors.items():
        if patches[key]:
            coll = PatchCollection(
                patches[key],
                facecolor=color,
                edgecolor="white",
                linewidth=0.2,
                alpha=0.78,
                zorder=3,
            )
            ax.add_collection(coll)
    style_map_ax(ax, "b  Hexagon occurrence, 2014–2024")
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=colors["rx"], markersize=8, label="Prescribed only"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=colors["wf"], markersize=8, label="Wildfire only"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=colors["both"], markersize=8, label="Both (not simultaneous)"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=7)


def style_map_ax(ax, title: str) -> None:
    ax.set_xlim(-116.4, -103.7)
    ax.set_ylim(44.15, 49.35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, loc="left", fontsize=10, fontweight="600")
    ax.set_facecolor("#eef2f6")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")


def add_cities(ax, names=None) -> None:
    want = set(names) if names else None
    for name, lon, lat in CITIES:
        if want is not None and name not in want:
            continue
        ax.plot(lon, lat, ".", color="#1e293b", markersize=3, zorder=8)
        ax.text(lon + 0.12, lat + 0.05, name, fontsize=6.5, color="#334155", zorder=9)


def sector_polygon(lat, lon, bmin, bmax, dist_km, n=72) -> np.ndarray:
    pts = [(lon, lat)]
    for b in np.linspace(bmin, bmax, n):
        la, lo = dest_point(lat, lon, float(b), dist_km)
        pts.append((lo, la))
    pts.append((lon, lat))
    return np.asarray(pts)


def ring_polygon(lat, lon, dist_km, n=180) -> np.ndarray:
    pts = []
    for b in np.linspace(0, 360, n, endpoint=False):
        la, lo = dest_point(lat, lon, float(b), dist_km)
        pts.append((lo, la))
    return np.asarray(pts)


def classify_mosquito(fires: pd.DataFrame, monitors: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    fire = fires[
        (fires["Fire_Name"].astype(str) == EXAMPLE_NAME)
        & (pd.to_numeric(fires["Fire_Year"], errors="coerce") == EXAMPLE_YEAR)
    ].iloc[0]
    lat, lon = float(fire["Fire_Lat"]), float(fire["Fire_Lon"])
    mon = monitors.copy()
    mon["aqs_code"] = mon["aqs_code"].astype(str)
    mon = mon[mon["aqs_code"].str.match(r"^\d{2}-\d{3}-\d{4}$", na=False)].drop_duplicates("aqs_code")
    mon["Distance_km"] = [
        haversine_km(lat, lon, float(r.lat), float(r.lon)) for r in mon.itertuples(index=False)
    ]
    mon["Bearing_deg"] = [
        bearing_deg(lat, lon, float(r.lat), float(r.lon)) for r in mon.itertuples(index=False)
    ]
    stage_b = pd.read_csv(DATA / "stage_b_daily.csv", usecols=["Fire_Name", "Fire_Year", "aqs_code"])
    frames = [stage_b]
    panel_path = first_existing(DATA / "bearing_panel.csv.gz", DATA / "bearing_panel.csv")
    if panel_path is not None:
        frames.append(pd.read_csv(panel_path, usecols=["Fire_Name", "Fire_Year", "aqs_code"]))
    has_pm = pd.concat(frames, ignore_index=True)
    has_pm = has_pm[
        (has_pm["Fire_Name"].astype(str) == EXAMPLE_NAME)
        & (pd.to_numeric(has_pm["Fire_Year"], errors="coerce") == EXAMPLE_YEAR)
    ]
    pm_codes = set(has_pm["aqs_code"].astype(str))
    geo = (mon["Distance_km"] <= LINK_KM) & (mon["Bearing_deg"] >= BEARING_MIN) & (mon["Bearing_deg"] <= BEARING_MAX)
    in_dist = mon["Distance_km"] <= LINK_KM
    aqs = mon["aqs_code"].astype(str)
    mon["class"] = "outside_300km"
    mon.loc[in_dist & ~geo, "class"] = "in_distance_outside_bearing"
    mon.loc[geo & ~aqs.isin(pm_codes), "class"] = "geo_no_pm25"
    mon.loc[geo & aqs.isin(pm_codes), "class"] = "linked"
    return fire, mon


def plot_mosquito(ax, fire, mon, rings, zoom: bool) -> None:
    add_montana(ax, rings)
    lat, lon = float(fire["Fire_Lat"]), float(fire["Fire_Lon"])
    sector = sector_polygon(lat, lon, BEARING_MIN, BEARING_MAX, LINK_KM)
    ring = ring_polygon(lat, lon, LINK_KM)
    ax.add_patch(MplPolygon(sector, closed=True, facecolor="#fde68a", edgecolor="#d97706", alpha=0.35, lw=1.0, zorder=2))
    ax.plot(ring[:, 0], ring[:, 1], color="#7c3aed", lw=1.1, zorder=3, label="300 km")
    colors = {
        "linked": "#0f766e",
        "geo_no_pm25": "#64748b",
        "in_distance_outside_bearing": "#dc2626",
        "outside_300km": "#94a3b8",
    }
    labels = {
        "linked": "≤300 km, 45–135°, usable PM$_{2.5}$",
        "geo_no_pm25": "≤300 km, 45–135°, no usable PM$_{2.5}$",
        "in_distance_outside_bearing": "≤300 km, outside 45–135°",
        "outside_300km": "> 300 km",
    }
    for cls, color in colors.items():
        sub = mon[mon["class"] == cls]
        marker = "o" if cls != "outside_300km" else "."
        size = 28 if cls != "outside_300km" else 10
        ax.scatter(
            sub["lon"],
            sub["lat"],
            s=size,
            c=color,
            marker=marker,
            zorder=5,
            edgecolors="white",
            linewidths=0.4,
            label=labels[cls],
        )
        if zoom and cls != "outside_300km":
            for r in sub.itertuples(index=False):
                dx, dy = 0.10, 0.05
                if r.sitename == "Kalispell":
                    dx, dy = -0.55, 0.10
                elif r.sitename == "Whitefish":
                    dx, dy = 0.12, 0.10
                elif r.sitename == "Flathead Valley":
                    dx, dy = 0.12, -0.12
                elif r.sitename == "Cut Bank":
                    dx, dy = 0.12, 0.08
                ax.text(r.lon + dx, r.lat + dy, r.sitename, fontsize=6, color="#1f2937", zorder=6)
    ax.scatter([lon], [lat], s=70, marker="*", c="#b45309", edgecolors="#7c2d12", zorder=7, label="PLN Mosquito RX, 2024")
    if zoom:
        ax.set_xlim(-117.2, -110.6)
        ax.set_ylim(45.6, 49.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("d  Monitor classification (western Montana)", loc="left", fontsize=10, fontweight="600")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_facecolor("#eef2f6")
        ax.legend(loc="lower right", frameon=False, fontsize=6.2)
    else:
        style_map_ax(ax, "c  Geographic screen, PLN Mosquito RX 2024")
        add_cities(ax, {"Missoula", "Helena", "Great Falls"})
        ax.legend(loc="lower right", frameon=False, fontsize=6.2)


def make_figure_1() -> Path:
    ensure_results()
    overlap = write_hex_overlap_report()
    fires = pd.read_csv(MAP / "fire_points.csv")
    monitors = pd.read_csv(MAP / "monitors.csv")
    rings = load_montana_rings()
    fire, mon = classify_mosquito(fires, monitors)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 9.4), dpi=DPI)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    add_montana(ax_a, rings)
    rx = fires[fires["fire_cohort"] == "WFIGS_RX"]
    wf = fires[fires["fire_cohort"] == "MTBS_wildfire"]
    ax_a.scatter(rx["Fire_Lon"], rx["Fire_Lat"], s=6, c="#0f766e", alpha=0.55, linewidths=0, zorder=3, label="Prescribed fire (WFIGS)")
    ax_a.scatter(wf["Fire_Lon"], wf["Fire_Lat"], s=18, c="#b45309", marker="^", alpha=0.85, linewidths=0, zorder=4, label="Wildfire centroid (MTBS)")
    ax_a.scatter(monitors["lon"], monitors["lat"], s=18, c="#1d4ed8", marker="s", zorder=5, edgecolors="white", linewidths=0.3, label="PM$_{2.5}$ monitor")
    style_map_ax(ax_a, "a  Fires and monitors, 2014–2024")
    add_cities(ax_a)
    ax_a.legend(loc="lower left", frameon=False, fontsize=7)

    plot_hex_panel(ax_b, fires, rings)
    plot_mosquito(ax_c, fire, mon, rings, zoom=False)
    plot_mosquito(ax_d, fire, mon, rings, zoom=True)

    fig.suptitle(
        "Figure 1. Spatial distribution of fires and the fire–monitor screening procedure",
        fontsize=12,
        fontweight="600",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIGURES / "Figure_1_fire_locations_and_screening.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=DPI)
    plt.close(fig)
    (FIGURES / "Figure_1_caption.txt").write_text(
        "Figure 1. Spatial distribution of fires and fire–monitor linkage framework in Montana. "
        "(a) WFIGS prescribed fire locations, MTBS wildfire perimeter centroids, and PM2.5 "
        "monitoring sites. MTBS perimeter polygons are not shipped; wildfires are drawn at "
        "centroids. (b) Spatial occurrence of prescribed fire only, wildfire only, or both fire "
        "types within ~20-km-radius hexagons during 2014–2024; “both” indicates co-occurrence "
        "within a hexagon over the study period, not simultaneous burning. (c–d) Example "
        "fire–monitor screening for the 2024 PLN Mosquito RX using a 300-km distance limit and "
        "45–135° fire-to-monitor bearing criterion. The shaded sector is the geographic "
        "screening rule, not wind direction.\n",
        encoding="utf-8",
    )
    n_linked = int((mon["class"] == "linked").sum())
    n_geo = int((mon["class"] == "geo_no_pm25").sum())
    n_outb = int((mon["class"] == "in_distance_outside_bearing").sum())
    print(f"Mosquito linked={n_linked} geo_no_pm25={n_geo} in_dist_outside_bearing={n_outb}")
    print(
        f"Hex overlap flat-top={overlap['overlap_pct_flat_top']:.4f}% "
        f"(pointy-on-flat={overlap['overlap_pct_pointy_on_flat_grid']:.3f}%)"
    )
    print("Wrote", out)
    return out


def main() -> None:
    make_figure_1()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
