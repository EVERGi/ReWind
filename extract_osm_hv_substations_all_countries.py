"""
extract_osm_hv_substations_all_countries.py

Replaces buses.csv (PyPSA-Eur's transmission-bus extraction, which has zero coverage for
5 fleet countries -- see NEXT_STEPS_lca_algebraic.md) with one consistently-derived dataset:
real OpenStreetMap power=substation points, queried directly via Overpass, for every one of
the fleet's 38 countries -- not just the 5 missing ones. Using the same method everywhere
means dist_to_grid_m is comparable across the whole fleet rather than "PyPSA-Eur bus lookup
for 33 countries, something else for 5."

Validated in compare_osm_hv_vs_buses.py: substation-only OSM points (voltage >= 220 kV,
falling back to a country's own top tier if it never reaches that) reproduced buses.csv's
per-country mean distance to within a few percent for the large majority of the 30
countries checked (see PLAN_lca_algebraic.md, "Completed 30 Jul 2026").

Output schema matches buses.csv exactly (bus_id, voltage, dc, symbol, under_construction,
x, y, country, geometry, source) so it's a drop-in replacement in
precompute_geo_columns.py's vectorized_dist_to_grid: one pooled file, nearest-neighbor
search not restricted to a turbine's own country (same behaviour buses.csv always had --
a turbine near a border may genuinely connect to a substation just across it).

Large countries (DE, FR, GB, IT, GR, ...) return too many elements for one Overpass query
to finish before timing out (confirmed: single-bbox queries for these repeatedly hit 504s
in compare_osm_hv_vs_buses.py). This script retries each bbox a few times, then recursively
splits it into 4 quadrants and queries those instead, merging results and de-duplicating by
(type, id) since a substation exactly on a split line can appear in more than one quadrant's
result.

Run once (output is cached to disk; this queries Overpass ~38-150+ times depending on how
much splitting large countries need, so expect a long run):
    python extract_osm_hv_substations_all_countries.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from extract_osm_hv_grid_points import parse_voltage, elements_to_points, HEADERS, OVERPASS_URL

_REWIND_DIR = Path(__file__).resolve().parent / "REWIND" / "REWIND"
_DATA_DIR = _REWIND_DIR / "data"
_TURBINES_XLSX = _DATA_DIR / "EU_turbines_input_data.xlsx"
_OUT_CSV = _DATA_DIR / "osm_hv_substations_all_countries.csv"

PAD_DEGREES = 1.0
VOLTAGE_TARGET = 220_000.0
MAX_SPLIT_DEPTH = 3        # 4^3 = 64 leaf queries worst case for a single stubborn country
MIN_BBOX_DEG = 0.25        # stop splitting once a quadrant is this small either way
NEXT_BUS_ID_START = 20_000_000  # clear of buses.csv's own range (0-6050)


def country_bbox(turbines: pd.DataFrame, iso: str, pad: float):
    """Returns (south, west, north, east) -- Overpass's bbox order."""
    sub = turbines[turbines["ISO_code"] == iso]
    lon = pd.to_numeric(sub["Longitude"], errors="coerce")
    lat = pd.to_numeric(sub["Latitude"], errors="coerce")
    return (lat.min() - pad, lon.min() - pad, lat.max() + pad, lon.max() + pad)


def query_overpass_once(bbox, retries=3, timeout_s=120):
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:{timeout_s}];
    (
      node["power"="substation"]["voltage"]({south},{west},{north},{east});
      way["power"="substation"]["voltage"]({south},{west},{north},{east});
    );
    out geom;
    """
    for attempt in range(retries):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=timeout_s + 30, headers=HEADERS)
        except requests.exceptions.RequestException as e:
            print(f"      request error: {e}, retrying ({attempt+1}/{retries})...")
            time.sleep(10 * (attempt + 1))
            continue
        if resp.status_code == 200:
            try:
                body = resp.json()
            except ValueError:
                print(f"      200 response but invalid JSON, retrying ({attempt+1}/{retries})...")
                time.sleep(10 * (attempt + 1))
                continue
            # Overpass sometimes returns HTTP 200 with a "remark" describing an internal
            # timeout instead of an HTTP 504 -- confirmed for FR's first run, which silently
            # came back with 0 elements this way and got skipped instead of split. Treat that
            # the same as an HTTP failure so the caller retries/splits instead of trusting an
            # empty result from a query that never actually finished.
            if "remark" in body:
                print(f"      200 but Overpass remark: {body['remark']!r}, retrying ({attempt+1}/{retries})...")
                time.sleep(10 * (attempt + 1))
                continue
            return body["elements"]
        time.sleep(10 * (attempt + 1))
    return None  # signal: give up on this bbox as-is, caller decides whether to split


def query_overpass_recursive(bbox, depth=0):
    elements = query_overpass_once(bbox)
    if elements is not None:
        return elements

    south, west, north, east = bbox
    if depth >= MAX_SPLIT_DEPTH or (north - south) < MIN_BBOX_DEG or (east - west) < MIN_BBOX_DEG:
        print(f"      giving up on bbox {bbox} at split depth {depth} (still timing out)")
        return []

    mid_lat, mid_lon = (south + north) / 2, (west + east) / 2
    quadrants = [
        (south, west, mid_lat, mid_lon), (south, mid_lon, mid_lat, east),
        (mid_lat, west, north, mid_lon), (mid_lat, mid_lon, north, east),
    ]
    print(f"      splitting bbox at depth {depth} into 4 quadrants...")
    merged, seen = [], set()
    for q in quadrants:
        for el in query_overpass_recursive(q, depth + 1):
            key = (el["type"], el["id"])
            if key not in seen:
                seen.add(key)
                merged.append(el)
    return merged


def main():
    turbines = pd.read_excel(_TURBINES_XLSX, sheet_name="EU_turbines_input_data")
    countries = sorted(turbines["ISO_code"].unique())
    print(f"Extracting OSM HV substations for all {len(countries)} fleet countries: {countries}")

    all_rows = []
    next_bus_id = NEXT_BUS_ID_START

    for iso in countries:
        bbox = country_bbox(turbines, iso, PAD_DEGREES)
        print(f"{iso}: querying Overpass, bbox={bbox}")
        elements = query_overpass_recursive(bbox)

        if not elements:
            print(f"  [WARNING] {iso}: zero substations found -- skipped, needs manual follow-up.")
            continue

        points = elements_to_points(elements)
        voltages_found = sorted({v for _, _, v in points if v is not None}, reverse=True)
        if not voltages_found:
            print(f"  [WARNING] {iso}: {len(points)} substations found but none had a parseable "
                  f"voltage -- keeping all of them (can't apply a threshold).")
            kept, threshold = points, None
        else:
            threshold = min(VOLTAGE_TARGET, voltages_found[0])
            kept = [(lon, lat, v) for lon, lat, v in points if v is not None and v >= threshold]

        print(f"  {iso}: {len(points)} candidate substations, voltage tiers: {voltages_found}, "
              f"keeping {len(kept)} at >= {threshold}")

        for lon, lat, voltage in kept:
            all_rows.append({
                "bus_id": next_bus_id,
                "voltage": voltage / 1000.0 if voltage else np.nan,
                "dc": np.nan,
                "symbol": "OSM_HV_Substation",
                "under_construction": np.nan,
                "x": lon,
                "y": lat,
                "country": iso,
                "geometry": f"POINT ({lon} {lat})",
                "source": "openstreetmap_hv_substation",
            })
            next_bus_id += 1

    if not all_rows:
        print("\nNo substations extracted for any country.")
        return

    out = pd.DataFrame(all_rows)
    out.to_csv(_OUT_CSV, index=False)
    print(f"\nWrote {len(out)} substations across {out['country'].nunique()} countries to {_OUT_CSV}")
    print(out.groupby("country").size())
    missing = sorted(set(countries) - set(out["country"].unique()))
    if missing:
        print(f"\nCountries with NO substations found (need manual follow-up): {missing}")


if __name__ == "__main__":
    main()
