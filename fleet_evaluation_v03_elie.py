import sys
from pathlib import Path

# The package modules use bare imports that only resolve from inside REWIND/REWIND/
_REWIND_DIR = Path(__file__).resolve().parent / "REWIND" / "REWIND"
sys.path.insert(0, str(_REWIND_DIR))

from built_inventory import create_dictionary_update
from calculations import lca_wimby_fleet_evaluation
from scaling import foundation_type
import time
import pandas as pd
import bw2data as bd

_DATA_DIR = _REWIND_DIR / "data"
_GEO_DIR = _DATA_DIR / "geo_precomputed"       # reorganized 28 Jul 2026, was flat inside _DATA_DIR
_BASELINE_DIR = _DATA_DIR / "baseline"          # this script's own output: the ground truth fleet_evaluation_lca_algebraic.py validates against
_BASELINE_DIR.mkdir(parents=True, exist_ok=True)

bd.projects.set_current('wimby')

eu_wind_turbines = pd.read_excel(_DATA_DIR / "EU_turbines_input_data.xlsx", sheet_name="EU_turbines_input_data")
print(f"Total turbines before filtering: {eu_wind_turbines.shape[0]}")

selected_countries = ['NO', 'DE', 'GB']

# Per-country sample sizes for the 21 Jul 2026 multi-country confirmation batch, chosen to
# total ~148 turbines across NO/DE/GB (covers onshore + all 3 offshore foundation buckets:
# Monopile, Semi-submersible, Spar buoy) before rolling lca_algebraic out to all remaining
# countries without a per-country baseline run. Falls back to the historical default
# (50 onshore / 20 per offshore bucket) for any country not listed here.
SAMPLE_CONFIG = {
    'NO': {'onshore_n': 30, 'offshore_per_bucket': 20},  # only source of Spar buoy (2) + 1 Semi-sub
    'DE': {'onshore_n': 30, 'offshore_per_bucket': 20},  # largest single fleet (36% of EU register)
    'GB': {'onshore_n': 15, 'offshore_per_bucket': 15},  # 2nd-largest offshore fleet, distinct background
}
DEFAULT_SAMPLE = {'onshore_n': 50, 'offshore_per_bucket': 20}

eu_wind_turbines = eu_wind_turbines[eu_wind_turbines['ISO_code'].isin(selected_countries)].copy()
print(f"Total turbines after filtering ({'/'.join(selected_countries)}, onshore+offshore): "
      f"{eu_wind_turbines.shape[0]}")

eu_wind_turbines['P_rated_kW'] = pd.to_numeric(eu_wind_turbines['P_rated_kW'], errors='coerce')
eu_wind_turbines['Longitude'] = pd.to_numeric(eu_wind_turbines['Longitude'], errors='coerce')
eu_wind_turbines['Latitude'] = pd.to_numeric(eu_wind_turbines['Latitude'], errors='coerce')
eu_wind_turbines['Hub_height_m'] = pd.to_numeric(eu_wind_turbines['Hub_height_m'], errors='coerce')
eu_wind_turbines['Diameter_m'] = pd.to_numeric(eu_wind_turbines['Diameter_m'], errors='coerce')
eu_wind_turbines['Lifetime_production_kWh'] = pd.to_numeric(eu_wind_turbines['Lifetime_production_kWh'], errors='coerce')
eu_wind_turbines['park_size'] = eu_wind_turbines['park_size'].astype(int)
eu_wind_turbines['Offshore'] = eu_wind_turbines['Offshore'].astype(int)
eu_wind_turbines['ISO_code'] = eu_wind_turbines['ISO_code'].astype('category')

lifecycle_stages_onshore = ['Input', 'Assembly', 'Transport', 'Maintenance', 'Disposal', 'Total']
lifecycle_stages_offshore = ['Input', 'Assembly', 'Transport', 'Disposal', 'Total']

for stage in lifecycle_stages_onshore:
    if stage not in eu_wind_turbines.columns:
        eu_wind_turbines[stage] = None

climate_change = ('EF v3.1', 'climate change', 'global warming potential (GWP100)')

# All EF v3.1 impact categories (same filter as brightway_example_for_elie.ipynb, cell 7,
# and fleet_evaluation_lca_algebraic.py's EF_METHODS), needed to compare per-stage,
# per-impact-category errors against the algebraic model, not just GWP100.
EF_METHODS = [m for m in bd.methods
              if 'EF v3.1' in str(m) and 'no LT' not in str(m) and 'EN1' not in str(m)]
print(f"Computing {len(EF_METHODS)} EF v3.1 impact categories per stage (in addition to the "
      f"single climate_change columns above, kept unchanged for backward compatibility)")

CHECKPOINT_EVERY = 5


def process_fleet(country_data, sea_depth_map, output_path, all_methods_path, label):
    """Run create_dictionary_update + lca_wimby_fleet_evaluation (climate_change + every
    EF_METHODS category) for every turbine in country_data, with checkpointing every
    CHECKPOINT_EVERY turbines.

    sea_depth_map: Series indexed like country_data, giving the sea_depth to pass per turbine
    (0 for onshore; the real per-turbine value, sourced from the GEBCO-derived
    Shared_Rewind/wind_fleet_data_incl_sea_depths_corrected.csv via precompute_geo_columns.py,
    no live GEBCO file needed, for offshore).

    This is the same per-turbine logic used for the onshore-only run in earlier versions of
    this script, extracted into a function so the onshore and offshore runs can never
    accidentally diverge from each other.
    """
    t_inv, t_lca, t_lca_all_methods = 0.0, 0.0, 0.0
    all_methods_rows = []

    for n_done, (idx, row) in enumerate(country_data.iterrows(), start=1):
        try:
            t0 = time.perf_counter()
            dct = create_dictionary_update(
                P=row['P_rated_kW'],
                lon=row['Longitude'],
                lat=row['Latitude'],
                h=row['Hub_height_m'],
                d=row['Diameter_m'],
                park_size=row['park_size'],
                sea_depth=sea_depth_map[idx],
                print_details=False
            )
            t_inv += time.perf_counter() - t0

            t1 = time.perf_counter()
            results = lca_wimby_fleet_evaluation(
                dict_activities=dct,
                impact_category=climate_change,
                aep=row['Lifetime_production_kWh']
            )
            t_lca += time.perf_counter() - t1

            if results is None:
                print(f"No results for turbine at index {idx} in {label}. Setting all stages to 0.")
                results = {stage: 0 for stage in lifecycle_stages_onshore}

            stages = lifecycle_stages_offshore if row['Offshore'] else lifecycle_stages_onshore
            for stage in stages:
                val = results.get(stage, 0)
                # lca_wimby_fleet_evaluation returns scalars or 1-row DataFrames
                if hasattr(val, 'iloc'):
                    val = float(val.iloc[0])
                country_data.loc[idx, stage] = val

            # All 25 EF v3.1 categories, same stage breakdown, same inventory dict (dct)
            # reused; only the LCA solve (lca_wimby_fleet_evaluation) is repeated per
            # method, exactly like the single climate_change call above.
            t2 = time.perf_counter()
            for method in EF_METHODS:
                method_df = lca_wimby_fleet_evaluation(
                    dict_activities=dct,
                    impact_category=method,
                    aep=row['Lifetime_production_kWh']
                )
                method_df.insert(0, 'turbine_idx', idx)
                # Identifying columns so any later comparison (e.g. against
                # fleet_evaluation_lca_algebraic.py's output) can hard-verify these rows
                # refer to the same physical turbines, not just a coincidentally-equal index.
                method_df.insert(1, 'P_rated_kW', row['P_rated_kW'])
                method_df.insert(2, 'Longitude', row['Longitude'])
                method_df.insert(3, 'Latitude', row['Latitude'])
                all_methods_rows.append(method_df)
            t_lca_all_methods += time.perf_counter() - t2

        except Exception as e:
            print(f"Error processing turbine at index {idx} in {label}: {e}")
            for stage in lifecycle_stages_onshore + lifecycle_stages_offshore:
                country_data.loc[idx, stage] = 0

        # Checkpoint: save progress every 5 turbines (and on the last one), so a crash or
        # interrupt partway through doesn't lose everything, and progress is visible on disk.
        if n_done % CHECKPOINT_EVERY == 0 or n_done == len(country_data):
            country_data.to_csv(output_path, index=False)
            if all_methods_rows:
                pd.concat(all_methods_rows, ignore_index=True).to_csv(all_methods_path, index=False)
            print(f"  [checkpoint] {n_done}/{len(country_data)} turbines processed, progress saved")

    all_methods_df = pd.concat(all_methods_rows, ignore_index=True) if all_methods_rows else pd.DataFrame()
    return country_data, all_methods_df, t_inv, t_lca, t_lca_all_methods


def print_summary(label, country_data, all_methods_df, output_path, all_methods_path,
                   elapsed, t_inv, t_lca, t_lca_all_methods):
    n = len(country_data)
    print(f"Finished {label}: {n} turbines in {elapsed:.1f}s ({elapsed/n:.1f}s/turbine). Saved to {output_path}")
    total = t_inv + t_lca + t_lca_all_methods
    print(f"  Inventory building         : {t_inv:.1f}s  ({100*t_inv/total:.0f}%)")
    print(f"  LCA solve (climate_change) : {t_lca:.1f}s  ({100*t_lca/total:.0f}%)")
    print(f"  LCA solve (all {len(EF_METHODS)} methods)   : {t_lca_all_methods:.1f}s  "
          f"({100*t_lca_all_methods/total:.0f}%)")
    print(f"  All-methods results ({len(all_methods_df)} rows = {n} turbines x {len(EF_METHODS)} methods) "
          f"saved to {all_methods_path}")


start_time_total = time.time()

for country in selected_countries:
    # ─────────────────────────────────────────────────────────────────────────
    # Onshore: unchanged scope/behavior from earlier versions of this script.
    # First 50 DK onshore turbines, sea_depth=0 (verified equivalent to GEBCO
    # auto-detection for confirmed-onshore turbines; see PLAN_lca_algebraic.md).
    # ─────────────────────────────────────────────────────────────────────────
    cfg = SAMPLE_CONFIG.get(country, DEFAULT_SAMPLE)
    print(f"Starting onshore simulation for {country} (sample of {cfg['onshore_n']})...")
    onshore_data = eu_wind_turbines[
        (eu_wind_turbines['ISO_code'] == country) & (eu_wind_turbines['Offshore'] == 0)
    ].head(cfg['onshore_n']).copy()
    sea_depth_onshore = pd.Series(0.0, index=onshore_data.index)

    output_path_onshore = _BASELINE_DIR / f"fleet_impacts_{country}.csv"
    all_methods_path_onshore = _BASELINE_DIR / f"fleet_impacts_{country}_baseline_all_methods.csv"

    start_time = time.time()
    onshore_data, all_methods_onshore, t_inv, t_lca, t_lca_all = process_fleet(
        onshore_data, sea_depth_onshore, output_path_onshore, all_methods_path_onshore,
        label=f"{country} onshore"
    )
    print_summary(f"{country} onshore", onshore_data, all_methods_onshore,
                  output_path_onshore, all_methods_path_onshore,
                  time.time() - start_time, t_inv, t_lca, t_lca_all)

    # ─────────────────────────────────────────────────────────────────────────
    # Offshore: real per-turbine sea_depth, sourced from precompute_geo_columns.py's
    # output (itself sourced from Shared_Rewind/wind_fleet_data_incl_sea_depths_corrected.csv,
    # the advisor-confirmed GEBCO-derived depth, no live GEBCO file needed).
    # lifecycle_stages_offshore (no Maintenance phase, matching built_inventory.py).
    #
    # Sampled PER FOUNDATION-TYPE BUCKET (Monopile/Semi-submersible/Spar buoy), not just the
    # first N rows: a plain .head(N) can badly under-sample a bucket that's rare in the row
    # order (e.g. BE: only 1 of its 131 Semi-submersible turbines fell in the first 20 offshore
    # rows). OFFSHORE_SAMPLE_PER_BUCKET turbines are taken from each bucket actually present.
    # ─────────────────────────────────────────────────────────────────────────
    geo_path = _GEO_DIR / f"{country.lower()}_geo_precomputed.csv"
    if not geo_path.exists():
        print(f"\n  ({geo_path} not found, run `python precompute_geo_columns.py` first "
              f"to enable the offshore run. Skipping offshore for {country}.)")
        continue

    OFFSHORE_SAMPLE_PER_BUCKET = cfg['offshore_per_bucket']
    geo = pd.read_csv(geo_path, index_col=0)
    offshore_all = eu_wind_turbines[
        (eu_wind_turbines['ISO_code'] == country) & (eu_wind_turbines['Offshore'] == 1)
    ].copy()
    offshore_all['bucket'] = geo.loc[offshore_all.index, 'sea_depth_m'].apply(
        lambda sd: foundation_type(True, sd))
    print(f"\n{country} offshore fleet: {len(offshore_all)} turbines, bucket distribution:")
    print(offshore_all['bucket'].value_counts())

    offshore_data = (
        offshore_all.groupby('bucket', group_keys=False)
        .apply(lambda g: g.head(OFFSHORE_SAMPLE_PER_BUCKET))
        .drop(columns='bucket')
    )
    print(f"\nStarting offshore simulation for {country} "
          f"(sample of {OFFSHORE_SAMPLE_PER_BUCKET} per bucket, {len(offshore_data)} total)...")

    sea_depth_offshore = geo.loc[offshore_data.index, 'sea_depth_m']

    output_path_offshore = _BASELINE_DIR / f"fleet_impacts_{country}_offshore.csv"
    all_methods_path_offshore = _BASELINE_DIR / f"fleet_impacts_{country}_offshore_baseline_all_methods.csv"

    start_time = time.time()
    offshore_data, all_methods_offshore, t_inv, t_lca, t_lca_all = process_fleet(
        offshore_data, sea_depth_offshore, output_path_offshore, all_methods_path_offshore,
        label=f"{country} offshore"
    )
    print_summary(f"{country} offshore", offshore_data, all_methods_offshore,
                  output_path_offshore, all_methods_path_offshore,
                  time.time() - start_time, t_inv, t_lca, t_lca_all)

print(f"\nTotal execution time: {time.time() - start_time_total:.1f}s")
