"""
compare_osm_hv_vs_buses.py

Validates the voltage-filtered direct-Overpass approach used in extract_osm_hv_grid_points.py
(Option 1 for the 5 missing countries) against the 33 countries where buses.csv already
provides a trusted dist_to_grid_m. This is the same before/after check compare_gridfinder_vs_buses.py
did for the raw (unfiltered) Gridfinder approach, which showed a ~84% systematic gap because
grid.gpkg has no voltage attribute to filter by. Here, each country's grid is queried directly
from OpenStreetMap via Overpass with a voltage tag required, then thresholded the same way as
extract_osm_hv_grid_points.py: keep everything >= 220 kV, falling back to a country's own top
tier if its grid never reaches 220 kV.

Nearest-neighbor distance is computed point-to-point (haversine), matching the quick check
already done for the 5 missing countries, since Overpass elements are converted to points
(substation centroids / line vertices) rather than kept as line geometries.

Run once (output is cached to disk, and Overpass queries are slow / rate-limited):
    python compare_osm_hv_vs_buses.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from extract_osm_hv_grid_points import (
    country_bbox, query_overpass, elements_to_points, HEADERS,
)

_REWIND_DIR = Path(__file__).resolve().parent / "REWIND" / "REWIND"
_DATA_DIR = _REWIND_DIR / "data"
_TURBINES_XLSX = _DATA_DIR / "EU_turbines_input_data.xlsx"
_BUSES_CSV = _DATA_DIR / "buses.csv"
_GEO_DIR = _DATA_DIR / "geo_precomputed"
_OUT_CSV = _DATA_DIR / "osm_hv_vs_buses_comparison.csv"

PAD_DEGREES = 1.0
MAX_PAD_DEGREES = 5.0
VOLTAGE_TARGET = 220_000.0


def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def fetch_hv_points(turbines: pd.DataFrame, iso: str):
    pad = PAD_DEGREES
    elements = []
    while pad <= MAX_PAD_DEGREES:
        bbox = country_bbox(turbines, iso, pad)
        elements = query_overpass(bbox)
        if len(elements) > 0:
            break
        pad *= 2

    if len(elements) == 0:
        return None, None

    points = elements_to_points(elements)
    voltages_found = sorted({v for _, _, v in points if v is not None}, reverse=True)
    if not voltages_found:
        return points, None

    threshold = min(VOLTAGE_TARGET, voltages_found[0])
    kept = [(lon, lat, v) for lon, lat, v in points if v is not None and v >= threshold]
    return kept, threshold


def main():
    turbines = pd.read_excel(_TURBINES_XLSX, sheet_name="EU_turbines_input_data")
    buses = pd.read_csv(_BUSES_CSV)
    covered_countries = sorted(set(turbines["ISO_code"].unique()) & set(buses["country"].unique()))
    print(f"Comparing {len(covered_countries)} already-covered countries: {covered_countries}")

    all_summaries = []

    for iso in covered_countries:
        geo_path = _GEO_DIR / f"{iso.lower()}_geo_precomputed.csv"
        if not geo_path.exists():
            print(f"  {iso}: skipped, no existing geo_precomputed file")
            continue
        existing = pd.read_csv(geo_path, index_col=0)

        sub = turbines[turbines["ISO_code"] == iso].copy()
        sub["Longitude"] = pd.to_numeric(sub["Longitude"], errors="coerce")
        sub["Latitude"] = pd.to_numeric(sub["Latitude"], errors="coerce")

        print(f"{iso}: querying Overpass...")
        try:
            kept, threshold = fetch_hv_points(turbines, iso)
        except RuntimeError as e:
            print(f"  {iso}: Overpass failed, skipped ({e})")
            continue

        if not kept:
            print(f"  [WARNING] {iso}: no voltage-tagged HV points found, skipped")
            continue

        cand = np.array(kept)
        cand_lon, cand_lat = cand[:, 0], cand[:, 1]

        osm_dist = np.array([
            haversine(lat, lon, cand_lat, cand_lon).min()
            for lat, lon in zip(sub["Latitude"], sub["Longitude"])
        ])
        existing_dist = existing.loc[sub.index, "dist_to_grid_m"].values

        abs_diff = np.abs(osm_dist - existing_dist)
        rel_diff_pct = np.where(existing_dist > 0, 100 * abs_diff / existing_dist, np.nan)

        summary = {
            "country": iso,
            "n_turbines": len(sub),
            "voltage_threshold_kv": threshold / 1000.0 if threshold else np.nan,
            "n_candidate_points": len(kept),
            "buses_mean_m": existing_dist.mean(),
            "osm_hv_mean_m": osm_dist.mean(),
            "abs_diff_mean_m": abs_diff.mean(),
            "abs_diff_median_m": np.median(abs_diff),
            "rel_diff_median_pct": np.nanmedian(rel_diff_pct),
        }
        all_summaries.append(summary)
        print(f"  {iso}: n={summary['n_turbines']:>6}  thr={summary['voltage_threshold_kv']:>5.0f}kV  "
              f"buses_mean={summary['buses_mean_m']:>10.0f}m  osm_hv_mean={summary['osm_hv_mean_m']:>10.0f}m  "
              f"rel_diff_median={summary['rel_diff_median_pct']:>6.1f}%")

    out = pd.DataFrame(all_summaries)
    out.to_csv(_OUT_CSV, index=False)
    print(f"\nWrote comparison summary for {len(out)} countries to {_OUT_CSV}")
    print("\nOverall (unweighted mean across countries):")
    print(out[["buses_mean_m", "osm_hv_mean_m", "abs_diff_mean_m", "abs_diff_median_m", "rel_diff_median_pct"]].mean())


if __name__ == "__main__":
    main()
