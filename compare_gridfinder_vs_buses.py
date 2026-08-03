"""
compare_gridfinder_vs_buses.py

Validation check: for the 33 fleet countries where buses.csv already provides a
dist_to_grid_m estimate (nearest-transmission-substation lookup, already stored in
REWIND/REWIND/data/geo_precomputed/<iso>_geo_precomputed.csv), independently compute
dist_to_grid using Gridfinder's real (openstreetmap-sourced) grid-line geometry instead,
and compare the two per turbine.

Result (see PLAN_lca_algebraic.md): the two approaches did NOT agree well (~84% systematic
gap, since grid.gpkg mixes in all-voltage lines while buses.csv is transmission-only), which
is why the 5 missing countries (BY, CY, FO, IS, XK) ended up using a direct voltage-filtered
Overpass query instead (extract_osm_hv_grid_points.py / extract_osm_hv_substations_all_countries.py),
not this Gridfinder-based approach. Kept here as the record of why that path was rejected.

Uses true nearest-point-on-line distance (not densified points) for the comparison itself,
computed in EPSG:3035 (ETRS89-LAEA Europe, an equal-area projection appropriate for this
data's extent) so distances come out directly in metres.

Run once (output is cached to disk):
    python compare_gridfinder_vs_buses.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

_REWIND_DIR = Path(__file__).resolve().parent / "REWIND" / "REWIND"
_DATA_DIR = _REWIND_DIR / "data"
_GRID_GPKG = _DATA_DIR / "grid.gpkg"
_TURBINES_XLSX = _DATA_DIR / "EU_turbines_input_data.xlsx"
_BUSES_CSV = _DATA_DIR / "buses.csv"
_GEO_DIR = _DATA_DIR / "geo_precomputed"
_OUT_CSV = _DATA_DIR / "gridfinder_vs_buses_comparison.csv"

PAD_DEGREES = 1.0
MAX_PAD_DEGREES = 5.0
METRIC_CRS = "EPSG:3035"  # ETRS89-LAEA Europe, equal-area, appropriate for this data's extent


def country_bbox(turbines: pd.DataFrame, iso: str, pad: float):
    sub = turbines[turbines["ISO_code"] == iso]
    lon = pd.to_numeric(sub["Longitude"], errors="coerce")
    lat = pd.to_numeric(sub["Latitude"], errors="coerce")
    return (lon.min() - pad, lat.min() - pad, lon.max() + pad, lat.max() + pad)


def fetch_real_lines(turbines: pd.DataFrame, iso: str):
    """Prefer real (openstreetmap) lines, widen the search box if none are found, fall back
    to gridfinder (predicted) only as a last resort."""
    pad = PAD_DEGREES
    while pad <= MAX_PAD_DEGREES:
        bbox = country_bbox(turbines, iso, pad)
        candidate = gpd.read_file(_GRID_GPKG, layer="grid", bbox=bbox)
        real = candidate[candidate["source"] == "openstreetmap"]
        if len(real) > 0:
            return real, pad, False
        pad *= 2
    bbox = country_bbox(turbines, iso, MAX_PAD_DEGREES)
    return gpd.read_file(_GRID_GPKG, layer="grid", bbox=bbox), MAX_PAD_DEGREES, True


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

        lines, pad_used, used_predicted = fetch_real_lines(turbines, iso)
        if len(lines) == 0:
            print(f"  {iso}: no grid lines found at all, skipped")
            continue

        turb_gdf = gpd.GeoDataFrame(
            sub, geometry=gpd.points_from_xy(sub["Longitude"], sub["Latitude"]), crs="EPSG:4326"
        ).to_crs(METRIC_CRS)
        lines_metric = lines.to_crs(METRIC_CRS)

        joined = gpd.sjoin_nearest(turb_gdf, lines_metric[["geometry"]], distance_col="gridfinder_dist_m")
        joined = joined[~joined.index.duplicated(keep="first")]  # sjoin_nearest can tie-multiply rows

        gf_dist = joined["gridfinder_dist_m"]
        existing_dist = existing.loc[sub.index, "dist_to_grid_m"]

        common_idx = gf_dist.index.intersection(existing_dist.index)
        gf = gf_dist.loc[common_idx].values
        ex = existing_dist.loc[common_idx].values
        abs_diff = np.abs(gf - ex)
        rel_diff_pct = np.where(ex > 0, 100 * abs_diff / ex, np.nan)

        summary = {
            "country": iso,
            "n_turbines": len(common_idx),
            "source": "predicted-fallback" if used_predicted else "openstreetmap",
            "bbox_pad_deg": pad_used,
            "buses_mean_m": ex.mean(),
            "gridfinder_mean_m": gf.mean(),
            "abs_diff_mean_m": abs_diff.mean(),
            "abs_diff_median_m": np.median(abs_diff),
            "rel_diff_median_pct": np.nanmedian(rel_diff_pct),
        }
        all_summaries.append(summary)
        print(f"  {iso}: n={summary['n_turbines']:>6}  buses_mean={summary['buses_mean_m']:>10.0f}m  "
              f"gridfinder_mean={summary['gridfinder_mean_m']:>10.0f}m  "
              f"abs_diff_median={summary['abs_diff_median_m']:>8.0f}m  "
              f"rel_diff_median={summary['rel_diff_median_pct']:>6.1f}%  ({summary['source']})")

    out = pd.DataFrame(all_summaries)
    out.to_csv(_OUT_CSV, index=False)
    print(f"\nWrote comparison summary for {len(out)} countries to {_OUT_CSV}")
    print("\nOverall (unweighted mean across countries):")
    print(out[["buses_mean_m", "gridfinder_mean_m", "abs_diff_mean_m", "abs_diff_median_m", "rel_diff_median_pct"]].mean())


if __name__ == "__main__":
    main()
