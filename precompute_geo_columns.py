"""
precompute_geo_columns.py

Pre-computes the per-turbine location-dependent quantities that
fleet_evaluation_lca_algebraic.py previously approximated with fleet-average
constants (transport distances) or ignored (cable length):

  - dist_rotor_m, dist_nacelle_m, dist_tower_m, dist_found_m
        per-turbine truck-transport distances (manufacturer -> turbine site),
        vectorized re-implementation of scaling.calculate_minimum_aggregated_distances()
  - dist_to_grid_m
        per-turbine cable length (turbine -> nearest grid bus),
        vectorized re-implementation of prepare_inventories.calculate_closest_distance()
  - sea_depth_m
        sign-corrected per-turbine sea depth, sourced from Dominik's
        Shared_Rewind/wind_fleet_data_incl_sea_depths_corrected.csv (advisor-confirmed
        as the authoritative file). No GEBCO file needed: this CSV already covers the
        full EU fleet (onshore and offshore). Used by the offshore symbolic model
        (foundation-type bucketing) and by the baseline's offshore run.

Covers a full country's fleet (onshore + offshore together, one 'Offshore' column to
filter downstream) rather than onshore only, since the offshore extension needs the exact
same distances/cable/sea-depth machinery, just for a different turbine subset.

The two "vectorized_*" functions are validated (see validate_against_originals())
against the existing per-turbine functions calculate_minimum_aggregated_distances()
/ calculate_closest_distance() in scaling.py / prepare_inventories.py: same
manufacturer-site / bus tables, same nearest-neighbor logic, just computed for
the whole fleet at once via a haversine matrix instead of a Python loop that
re-reads its source file on every call. Measured max relative error vs. the
original geopy-geodesic (ellipsoidal) distances is ~0.4% (haversine is
spherical, not ellipsoidal), small next to the >1% error this replaces, but
not exactly zero; see PLAN_lca_algebraic.md for the net effect on GWP100 error.

Run once (output is cached to disk):
    python precompute_geo_columns.py
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

_REWIND_DIR = Path(__file__).resolve().parent / "REWIND" / "REWIND"
_DATA_DIR = _REWIND_DIR / "data"
_GEO_DIR = _DATA_DIR / "geo_precomputed"   # reorganized 28 Jul 2026, was flat inside _DATA_DIR
_GEO_DIR.mkdir(parents=True, exist_ok=True)
_SEA_DEPTH_CSV = Path(__file__).resolve().parent / "Shared_Rewind" / "wind_fleet_data_incl_sea_depths_corrected.csv"

R_EARTH = 6371008.8  # mean earth radius, meters (matches geopy's default WGS84 mean radius)


def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in meters. lat/lon in degrees, broadcastable arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def vectorized_dist_to_grid(lons, lats):
    """All-turbines-at-once equivalent of prepare_inventories.calculate_closest_distance(lon, lat)."""
    buses = pd.read_csv(_DATA_DIR / "buses.csv")
    bus_lon = buses['x'].values[None, :]
    bus_lat = buses['y'].values[None, :]
    t_lon = np.asarray(lons, dtype=float)[:, None]
    t_lat = np.asarray(lats, dtype=float)[:, None]
    d = haversine_m(t_lat, t_lon, bus_lat, bus_lon)
    return d.min(axis=1)


def vectorized_transport_distances(lons, lats):
    """All-turbines-at-once equivalent of scaling.calculate_minimum_aggregated_distances(lon, lat).

    Returns (dist_rotor_m, dist_nacelle_m, dist_tower_m, dist_found_m) arrays.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    n = len(lons)

    def sheet_component_mins(sheet):
        df = pd.read_excel(_DATA_DIR / "Manufacturing_location_20_largest_EU_manufacturer.xlsx", sheet_name=sheet)
        d = haversine_m(lats[:, None], lons[:, None], df['lat'].values[None, :], df['lon'].values[None, :])
        out = {}
        for subcomp in df['Sub-component'].unique():
            cols = np.where(df['Sub-component'].values == subcomp)[0]
            out[subcomp] = d[:, cols].min(axis=1)
        rotor = out.get('Blades', 0) + out.get('Hub', 0)
        nacelle = out.get('Generator', 0) + out.get('Nacelle', 0)
        tower = out['Tower']
        foundation = np.full(n, 50000.0)   # constant in the original function too
        total = rotor + nacelle + tower + foundation
        return rotor, nacelle, tower, foundation, total

    r_e, n_e, t_e, f_e, tot_e = sheet_component_mins('ENERCON')
    r_v, n_v, t_v, f_v, tot_v = sheet_component_mins('VESTAS')

    use_enercon = tot_e < tot_v   # same manufacturer-choice rule as the original, per turbine
    rotor = np.where(use_enercon, r_e, r_v)
    nacelle = np.where(use_enercon, n_e, n_v)
    tower = np.where(use_enercon, t_e, t_v)
    foundation = np.where(use_enercon, f_e, f_v)
    return rotor, nacelle, tower, foundation


def validate_against_originals(dk_onshore, n_sample=15, seed=42):
    """Cross-check the vectorized functions against the original per-turbine ones on a random sample."""
    sys.path.insert(0, str(_REWIND_DIR))
    from scaling import calculate_minimum_aggregated_distances
    from prepare_inventories import calculate_closest_distance

    sample = dk_onshore.sample(n_sample, random_state=seed)
    lons = sample['Longitude'].values
    lats = sample['Latitude'].values

    rotor_v, nacelle_v, tower_v, found_v = vectorized_transport_distances(lons, lats)
    grid_v = vectorized_dist_to_grid(lons, lats)

    max_rel_err = 0.0
    for i, (lo, la) in enumerate(zip(lons, lats)):
        orig = calculate_minimum_aggregated_distances(lo, la)
        r_o, n_o, t_o = orig.iloc[0, 1], orig.iloc[1, 1], orig.iloc[2, 1]
        g_o = calculate_closest_distance(lo, la)
        for orig_val, vec_val in [(r_o, rotor_v[i]), (n_o, nacelle_v[i]), (t_o, tower_v[i]), (g_o, grid_v[i])]:
            rel_err = abs(vec_val - orig_val) / orig_val
            max_rel_err = max(max_rel_err, rel_err)

    print(f"  Validation vs original per-turbine functions (N={n_sample}): max rel. error = {max_rel_err*100:.3f}%")
    assert max_rel_err < 0.01, "vectorized geo functions diverged >1% from the original per-turbine functions"


def compute_geo_columns(fleet: pd.DataFrame) -> pd.DataFrame:
    """Compute dist_rotor_m/dist_nacelle_m/dist_tower_m/dist_found_m/dist_to_grid_m/sea_depth_m
    for every row in `fleet` (must have Longitude/Latitude columns; index is preserved)."""
    lons = fleet['Longitude'].values
    lats = fleet['Latitude'].values

    print("  Computing per-turbine transport distances (rotor/nacelle/tower/foundation)...")
    t0 = time.perf_counter()
    dist_rotor, dist_nacelle, dist_tower, dist_found = vectorized_transport_distances(lons, lats)
    print(f"    done in {time.perf_counter()-t0:.2f}s")

    print("  Computing per-turbine cable length (distance to nearest grid bus)...")
    t0 = time.perf_counter()
    dist_to_grid = vectorized_dist_to_grid(lons, lats)
    print(f"    done in {time.perf_counter()-t0:.2f}s")

    print("  Merging pre-computed sea depth (advisor-confirmed 'corrected' file)...")
    sea_df = pd.read_csv(_SEA_DEPTH_CSV, index_col=0)
    # Row order/count is identical to EU_turbines_input_data.xlsx (verified: same 77 552 rows,
    # same column values, same order) so a direct positional join by original index is safe.
    sea_depth_raw = sea_df.loc[fleet.index, 'Sea_depth_corrected'].values.astype(float)
    # GEBCO convention (baked into this precomputed column already, no GEBCO file needed here):
    # negative = underwater (sea depth), positive = land elevation. scaling.py's offshore
    # foundation formulas expect a positive depth in meters, 0 onshore.
    sea_depth_m = np.where(sea_depth_raw < 0, -sea_depth_raw, 0.0)

    return pd.DataFrame({
        'Offshore': fleet['Offshore'].values,
        'dist_rotor_m': dist_rotor,
        'dist_nacelle_m': dist_nacelle,
        'dist_tower_m': dist_tower,
        'dist_found_m': dist_found,
        'dist_to_grid_m': dist_to_grid,
        'sea_depth_m': sea_depth_m,
    }, index=fleet.index)


def main(country='DK'):
    print("Loading turbine register...")
    data = pd.read_excel(_DATA_DIR / "EU_turbines_input_data.xlsx", sheet_name="EU_turbines_input_data")
    fleet = data[data['ISO_code'] == country].copy()
    onshore = fleet[fleet['Offshore'] == 0]
    offshore = fleet[fleet['Offshore'] == 1]
    print(f"  {len(fleet)} {country} turbines ({len(onshore)} onshore, {len(offshore)} offshore)")

    print("Validating vectorized geo functions against the originals...")
    _sample_pool = onshore if len(onshore) else fleet
    validate_against_originals(_sample_pool, n_sample=min(15, len(_sample_pool)))

    out = compute_geo_columns(fleet)

    n_offshore_nonzero = (out.loc[out['Offshore'] == 1, 'sea_depth_m'] > 0).sum()
    n_onshore_nonzero = (out.loc[out['Offshore'] == 0, 'sea_depth_m'] > 0).sum()
    print(f"  Offshore turbines with sea_depth_m > 0: {n_offshore_nonzero} of {len(offshore)}")
    print(f"  Onshore turbines with sea_depth_m > 0 (expected ~0, coastal GEBCO-grid edge "
          f"effects only): {n_onshore_nonzero} of {len(onshore)}")

    out_path = _GEO_DIR / f"{country.lower()}_geo_precomputed.csv"
    out.to_csv(out_path)
    print(f"\nSaved {len(out)} rows to {out_path}")
    print(out.groupby('Offshore').describe().T)


if __name__ == "__main__":
    main()
