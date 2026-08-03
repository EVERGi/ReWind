"""
extract_osm_hv_grid_points.py

Option 1 fix for the buses.csv gap: instead of Gridfinder's undifferentiated grid.gpkg
(which mixes in low/medium-voltage distribution lines and systematically produces much
shorter "distance to grid" values than buses.csv's transmission-only substations — see
compare_gridfinder_vs_buses.py), query OpenStreetMap directly via the Overpass API for
power=substation features that actually carry a voltage tag, so the result is conceptually
comparable to buses.csv: distance to the nearest actual substation, not to the nearest
point along any nearby wire (power=line was tried first and included, but produced a
systematic ~36% undershoot vs buses.csv across 27 validation countries — a line passing
near a turbine is always closer than the nearest real substation; see
compare_osm_hv_vs_buses.py).

Two-pass design, since these are small/isolated grids that may not reach the same
absolute voltage tiers as interconnected mainland Europe (buses.csv is 220-440 kV):
  Pass 1: fetch every power=line/substation element with ANY voltage tag in a padded
          bbox around each missing country's turbines, and print the voltage distribution
          actually found there.
  Pass 2: keep only elements at or above a per-country threshold (defaults to the top
          voltage tier actually present for that country, so a small isolated grid whose
          highest tier is e.g. 66 kV isn't excluded just because it never reaches 220 kV).

Run once (output is cached to disk):
    python extract_osm_hv_grid_points.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_REWIND_DIR = Path(__file__).resolve().parent / "REWIND" / "REWIND"
_DATA_DIR = _REWIND_DIR / "data"
_TURBINES_XLSX = _DATA_DIR / "EU_turbines_input_data.xlsx"
_BUSES_CSV = _DATA_DIR / "buses.csv"
_OUT_CSV = _DATA_DIR / "osm_hv_supplement_precomputed.csv"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Overpass rejects requests with no/default User-Agent (406); identify the script clearly.
HEADERS = {"User-Agent": "ReWind-research-script/1.0 (masters thesis, contact: eliejoefarah@gmail.com)"}
MISSING_COUNTRIES = ["BY", "CY", "FO", "IS", "XK"]
PAD_DEGREES = 1.0
MAX_PAD_DEGREES = 5.0
NEXT_BUS_ID_START = 20_000_000  # clear of both buses.csv (0-6050) and the Gridfinder supplement (10M+)


def country_bbox(turbines: pd.DataFrame, iso: str, pad: float):
    """Returns (south, west, north, east) — Overpass's bbox order, not (minx,miny,maxx,maxy)."""
    sub = turbines[turbines["ISO_code"] == iso]
    lon = pd.to_numeric(sub["Longitude"], errors="coerce")
    lat = pd.to_numeric(sub["Latitude"], errors="coerce")
    return (lat.min() - pad, lon.min() - pad, lat.max() + pad, lon.max() + pad)


def parse_voltage(tag_value: str):
    """OSM voltage tags can be semicolon-separated for multi-circuit lines/substations
    (e.g. '110000;220000'); return the maximum numeric value found, or None if unparseable."""
    if not tag_value:
        return None
    values = []
    for part in str(tag_value).split(";"):
        try:
            values.append(float(part.strip()))
        except ValueError:
            continue
    return max(values) if values else None


def query_overpass(bbox, retries=5):
    """Substations only, not power=line: buses.csv (what this is being compared against/
    supplementing) represents actual substation locations, not arbitrary points along a
    transmission line's route. Including line vertices systematically understated distance
    (a line passing near a turbine is always closer than the nearest real substation) —
    see compare_osm_hv_vs_buses.py's 33-country check, which found a ~36% median undershoot
    with lines included."""
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:90];
    (
      node["power"="substation"]["voltage"]({south},{west},{north},{east});
      way["power"="substation"]["voltage"]({south},{west},{north},{east});
    );
    out geom;
    """
    for attempt in range(retries):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=150, headers=HEADERS)
        except requests.exceptions.RequestException as e:
            print(f"    Request error: {e}, retrying ({attempt+1}/{retries})...")
            time.sleep(10 * (attempt + 1))
            continue
        if resp.status_code == 200:
            return resp.json()["elements"]
        print(f"    Overpass returned {resp.status_code}, retrying ({attempt+1}/{retries})...")
        time.sleep(10 * (attempt + 1))  # back off more each retry — 504s are usually transient server load
    raise RuntimeError(f"Overpass query failed after {retries} attempts: {resp.status_code} {resp.text[:200]}")


def elements_to_points(elements):
    """Extract (lon, lat, voltage) for every substation element: nodes are direct points,
    ways (area outlines) use their centroid."""
    rows = []
    for el in elements:
        voltage = parse_voltage(el.get("tags", {}).get("voltage"))
        if el["type"] == "node":
            rows.append((el["lon"], el["lat"], voltage))
        elif el["type"] == "way" and "geometry" in el:
            coords = el["geometry"]
            lon = np.mean([c["lon"] for c in coords])
            lat = np.mean([c["lat"] for c in coords])
            rows.append((lon, lat, voltage))
    return rows


def main():
    turbines = pd.read_excel(_TURBINES_XLSX, sheet_name="EU_turbines_input_data")
    all_rows = []
    next_bus_id = NEXT_BUS_ID_START

    for iso in MISSING_COUNTRIES:
        pad = PAD_DEGREES
        elements = []
        while pad <= MAX_PAD_DEGREES:
            bbox = country_bbox(turbines, iso, pad)
            print(f"{iso}: querying Overpass, bbox pad={pad} deg, bbox={bbox}")
            elements = query_overpass(bbox)
            if len(elements) > 0:
                break
            pad *= 2
            print(f"  {iso}: 0 voltage-tagged elements found, widening to {pad} deg")

        if len(elements) == 0:
            print(f"  [WARNING] {iso}: no voltage-tagged power infrastructure found in OSM "
                  f"even at {MAX_PAD_DEGREES} deg padding — skipped, needs a different source.")
            continue

        points = elements_to_points(elements)
        voltages_found = sorted({v for _, _, v in points if v is not None}, reverse=True)
        print(f"  {iso}: {len(elements)} elements, {len(points)} candidate points, "
              f"voltage tiers found: {voltages_found}")

        if not voltages_found:
            print(f"  [WARNING] {iso}: elements found but none had a parseable voltage value — "
                  f"keeping all of them (can't apply a voltage threshold).")
            kept = points
            threshold = None
        else:
            # Cap at 220 kV — buses.csv itself only spans ~220-440 kV, so anything higher
            # (e.g. Belarus's 750 kV line, a rare international interconnector, not a normal
            # transmission-substation tier) shouldn't set the bar on its own. Countries whose
            # own grid never reaches 220 kV (Cyprus, the Faroe Islands) fall back to their own
            # actual maximum instead of being excluded entirely.
            threshold = min(220_000.0, voltages_found[0])
            kept = [(lon, lat, v) for lon, lat, v in points if v is not None and v >= threshold]
        print(f"  {iso}: keeping {len(kept)} points at voltage >= {threshold}")

        for lon, lat, voltage in kept:
            all_rows.append({
                "bus_id": next_bus_id,
                "voltage": voltage / 1000.0 if voltage else np.nan,  # OSM tags volts, buses.csv uses kV
                "dc": np.nan,
                "symbol": "OSM_HV_Point",
                "under_construction": np.nan,
                "x": lon,
                "y": lat,
                "country": iso,
                "geometry": f"POINT ({lon} {lat})",
                "source": "openstreetmap_hv",
            })
            next_bus_id += 1

    if not all_rows:
        print("\nNo points extracted for any missing country.")
        return

    out = pd.DataFrame(all_rows)
    out.to_csv(_OUT_CSV, index=False)
    print(f"\nWrote {len(out)} points across {out['country'].nunique()} countries to {_OUT_CSV}")
    print(out.groupby("country").size())


if __name__ == "__main__":
    main()
