"""
fleet_evaluation_lca_algebraic.py: Option A (piecewise-exact + per-turbine geo)

The v1 of this file used linear approximations for material percentages
(frozen at P_REF=2000 kW) and polyfit for M_reinf / M_elec.

v2 uses exact SymPy Piecewise expressions that match the np.interp /
InterpolatedUnivariateSpline(k=1) calls in built_inventory.py. Each material
amount is expressed as:

    amount(P, d, h) = piecewise_interp(P, xs, perc_ys) × M_component_sym(d, h)

v3 (current) replaces the two remaining approximations (fleet-average
transport distances and a P-only cable proxy) with real per-turbine values,
pre-computed once by precompute_geo_columns.py and loaded as extra symbolic
parameters (dist_rotor, dist_nacelle, dist_tower, dist_to_grid). See
PLAN_lca_algebraic.md, "Completed 15 Jul 2026", for validation numbers.

Validation is against fleet_evaluation_v03_elie.py (the baseline, confirmed a
faithful, tiny-modification reconstruction of v02.py) via
fleet_impacts_DK_baseline_all_methods.csv, covering every lifecycle stage and
every EF v3.1 impact category, not just GWP100. fleet_evaluation_redo_lci.py
("Option B") is a separate speed-optimization exercise, not the reference;
it is no longer used here.
"""

import sys, os
from pathlib import Path

# REWIND/REWIND (the actual Python package: built_inventory.py, scaling.py, etc.) is not
# installed on sys.path by default when this script is run from the repo root, so add it
# explicitly before importing anything from it below.
_REWIND_DIR = Path(__file__).resolve().parent / "REWIND" / "REWIND"
sys.path.insert(0, str(_REWIND_DIR))
# percentage_inventory() writes pickle files to cwd; use REWIND/REWIND so they land there
os.chdir(str(_REWIND_DIR))

# Third-party numerical / LCA stack.
import numpy as np
import pandas as pd
from sympy import Piecewise             # symbolic piecewise-linear expressions (see sympy_interp/sympy_extrap below)
import bw2data as bd                     # Brightway2 project/database/activity access
import lca_algebraic as agb              # builds the symbolic foreground model and batch-evaluates it
from lca_algebraic.lca import method_name, method_unit   # human-readable method labels, used to name result columns
from lca_algebraic.cache import clear_caches
import time

# ReWind's own inventory-construction code (unchanged from the non-algebraic baseline),
# reused here purely as a source of formulas and one-off Activity lookups, never called
# per-turbine in a loop (that per-turbine loop is exactly what this script replaces).
from prepare_inventories import (ecoinvent_setup, transport_cement_elec, steel_dataset,
                                  percentage_inventory, inventory_not_kg,
                                  activities_and_uuids)
from power_transformer import transfo_10mva, transfo_500mva
from built_inventory import create_dictionary_update
from scaling import (func_tower_weight_d2h, func_nacelle_weight_power,
                     func_rotor_weight_rotor_diameter)

_DATA_DIR = _REWIND_DIR / "data"                                              # REWIND/REWIND/data: turbine register + reference files (EU_turbines_input_data.xlsx, buses.csv, datasets/)
_SHARED   = Path(__file__).resolve().parent / "Shared_Rewind" / "Fleet_results"  # Dominik's original per-country fleet result CSVs (not used for validation here, see module docstring)
CLIMATE_CHANGE = ('EF v3.1', 'climate change', 'global warming potential (GWP100)')  # the headline impact category, used for the printed sanity-check summaries below

# Output/input subfolders, reorganized 28 Jul 2026 (previously all flat inside _DATA_DIR).
_GEO_DIR              = _DATA_DIR / "geo_precomputed"      # <iso>_geo_precomputed.csv, written by precompute_geo_columns.py
_RESULTS_DIR          = _DATA_DIR / "results"               # fleet_impacts_<ISO>[_offshore_<bucket>]_lca_algebraic.csv: this script's headline output
_RESULTS_BY_STAGE_DIR = _RESULTS_DIR / "by_stage"           # same, broken down per lifecycle stage
_VALIDATION_DIR       = _DATA_DIR / "validation"            # algebraic-vs-baseline error summaries/detail, only for the 5 validated countries
_BASELINE_DIR         = _DATA_DIR / "baseline"              # ground-truth CSVs from fleet_evaluation_v03_elie.py: validate_against_baseline()'s reference
for _d in (_GEO_DIR, _RESULTS_DIR, _RESULTS_BY_STAGE_DIR, _VALIDATION_DIR, _BASELINE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 0. Country selection: change this to run a different country, or pass an ISO code as
#    argv[1] to drive it from a loop (see run_remaining_countries.py). Everything else
#    below (file names, foreground DB names, background activities, reference point)
#    is derived from this single variable. See NEXT_STEPS_lca_algebraic.md item 4.
# ─────────────────────────────────────────────────────────────────────────────
COUNTRY = sys.argv[1] if len(sys.argv) > 1 else 'GB'

# Representative reference point for this country's background activities
# (transport_cement_elec/steel_dataset lon/lat, and the reference inventory used to
# discover activity keys): the mean lon/lat of the country's own onshore turbines.
# For DK this works out close to the "central Jutland" point (9.5, 56.2) used originally.
_ref_data = pd.read_excel(_DATA_DIR / "EU_turbines_input_data.xlsx", sheet_name="EU_turbines_input_data")
_ref_onshore = _ref_data[(_ref_data['ISO_code'] == COUNTRY) & (_ref_data['Offshore'] == 0)]
if _ref_onshore.empty:
    _ref_onshore = _ref_data[_ref_data['ISO_code'] == COUNTRY]  # offshore-only country fallback
LON_REF = float(pd.to_numeric(_ref_onshore['Longitude'], errors='coerce').mean())
LAT_REF = float(pd.to_numeric(_ref_onshore['Latitude'], errors='coerce').mean())
print(f"Country: {COUNTRY}  (reference point lon={LON_REF:.3f}, lat={LAT_REF:.3f}, "
      f"from {len(_ref_onshore)} onshore turbines)")


def assert_same_turbines(index_values, df_a, df_b, id_cols=('P_rated_kW', 'Longitude', 'Latitude'), atol=1e-4):
    """Hard guarantee that `index_values` refer to the *same physical turbines* in both
    `df_a` and `df_b`, not just a coincidentally-equal pandas index label.

    All fleet_evaluation_*.py scripts load EU_turbines_input_data.xlsx fresh and never
    reset_index(), so index labels (e.g. 37093) are stable across scripts *by construction*
    today (verified: fleet_evaluation_v03_elie.py's .head(50), fleet_evaluation_redo_lci.py's
    .head(50), and fleet_evaluation_lca_algebraic.py's full dk_onshore all produce identical
    index values for the same turbines). But that's an accident of no script ever sorting or
    reindexing the source file; if EU_turbines_input_data.xlsx is ever regenerated/reordered
    upstream, or a comparison file predates this check, silently trusting the index label
    would silently compare the wrong turbines. This raises instead of warning.
    """
    missing = [c for c in id_cols if c not in df_a.columns or c not in df_b.columns]
    if missing:
        raise AssertionError(
            f"Cannot verify turbine identity: column(s) {missing} missing from one of the "
            f"two files being compared. Regenerate the older file with the current version "
            f"of its script (identifying columns were added specifically for this check)."
        )
    for col in id_cols:
        a = df_a.loc[index_values, col].to_numpy(dtype=float)
        b = df_b.loc[index_values, col].to_numpy(dtype=float)
        mismatch = ~np.isclose(a, b, atol=atol, equal_nan=False)
        if mismatch.any():
            bad_idx = np.asarray(index_values)[mismatch][:5]
            raise AssertionError(
                f"Index alignment broken: turbines at index {bad_idx.tolist()} have "
                f"different '{col}' values between the two files being compared "
                f"({a[mismatch][:5]} vs {b[mismatch][:5]}). They are not the same turbines; "
                f"do not trust this comparison."
            )


def _err_summary(df, group_col):
    """Collapse a long-format (turbine, stage, method, error) table into one summary row per
    value of `group_col` (e.g. one row per stage, or one row per impact category).

    `df` must already carry `rel_error_pct` ((algebraic - baseline) / baseline * 100) and
    `abs_error` (algebraic - baseline, unscaled) columns, computed once in
    validate_against_baseline() and reused across all three groupings (stage/method/turbine)
    so the error definition can't drift between them.

    No row filtering happens here or upstream: every (turbine, stage, method) row passed in is
    included. An earlier version of this function was only ever called on a pre-filtered subset
    (rows where baseline_value was >=0.1% of that turbine+method's Total, on the theory that
    relative error is meaningless near a zero denominator), that filter is gone (see
    NEXT_STEPS_lca_algebraic.md item 9): it silently dropped the *entire* Maintenance-stage row
    from 4 of 5 validated countries' summaries once every one of its rows fell under the
    threshold, which is a much bigger problem than the near-zero-denominator noise it was meant
    to solve. The only rows with an undefined `rel_error_pct` now are where baseline_value == 0
    (true division by zero, not a threshold judgment call); those are NaN and pandas' mean/
    median naturally skip them; `n` below reports how many rows had a defined percentage, and
    `mean_abs_error` (absolute units, never divides by anything) always covers every row.

    Returns a DataFrame indexed by `group_col` with:
      n                    : rows with a defined rel_error_pct (excludes baseline_value == 0)
      mean_error_pct       : signed mean (shows systematic over/under-estimation)
      median_error_pct     : signed median (robust to a handful of outlier rows)
      abs_mean_error_pct   : mean of |error|  (the headline "how far off, typically" number)
      abs_median_error_pct : median of |error| (robust alternative to abs_mean_error_pct)
      abs_max_error_pct    : max of |error|   (the worst single case in that group)
      mean_abs_error       : mean of |algebraic - baseline| in the metric's own units (not %),
                             computed over every row (including baseline_value == 0) since it
                             never divides by baseline_value
    """
    g = df.groupby(group_col)
    out_df = g['rel_error_pct'].agg(n='count',
                                     mean_error_pct='mean',
                                     median_error_pct='median',
                                     abs_mean_error_pct=lambda s: s.abs().mean(),
                                     abs_median_error_pct=lambda s: s.abs().median(),
                                     abs_max_error_pct=lambda s: s.abs().max())
    out_df['mean_abs_error'] = g['abs_error'].apply(lambda s: s.abs().mean())
    return out_df


def validate_against_baseline(label, baseline_all_path, turbines_index, out_df, by_stage_df,
                               method_cols, output_prefix):
    """Compare an algebraic model's results against the baseline's all-methods CSV, every
    (turbine, stage, impact category) triple, not just aggregate GWP100. Shared by the onshore
    and offshore models (see PLAN_lca_algebraic.md, "Completed 16 Jul 2026").

    out_df: total per-kWh results, one row per turbine (in the same order as turbines_index),
        must include identifying columns (P_rated_kW/Longitude/Latitude) for the hard guarantee.
    by_stage_df: by-stage per-kWh long format (turbine_idx, phase, <method columns>).
    method_cols: list of method column names shared by out_df and by_stage_df.
    output_prefix: filename prefix for this model's validation output CSVs.
    """
    if not baseline_all_path.exists():
        print(f"\n  (No baseline-all-methods CSV found for {label}, run fleet_evaluation_v03_elie.py first)")
        return

    baseline_all = pd.read_csv(baseline_all_path)
    baseline_all['turbine_idx'] = baseline_all['turbine_idx'].astype(int)

    common = pd.Index(turbines_index).intersection(baseline_all['turbine_idx'].unique())
    n_val = len(common)
    if n_val == 0:
        print(f"\n  (No common turbines between algebraic and baseline-all-methods results for {label})")
        return

    id_lookup = out_df.set_index('turbine_idx') if 'turbine_idx' in out_df.columns else out_df
    baseline_ids = baseline_all.drop_duplicates('turbine_idx').set_index('turbine_idx')
    assert_same_turbines(common, id_lookup, baseline_ids)  # hard guarantee: raises if misaligned

    STAGE_COLS = ['Input', 'Assembly', 'Transport', 'Maintenance', 'Disposal', 'Total']
    present_stage_cols = [c for c in STAGE_COLS if c in baseline_all.columns]
    baseline_long = baseline_all.melt(
        id_vars=['turbine_idx', 'Method'], value_vars=present_stage_cols,
        var_name='stage', value_name='baseline_value'
    )
    baseline_long['method_key'] = baseline_long['Method'].str.split(' - ', n=1).str[1]
    baseline_long = baseline_long[baseline_long['turbine_idx'].isin(common)]

    alg_total = out_df.copy()
    if 'turbine_idx' not in alg_total.columns:
        alg_total['turbine_idx'] = turbines_index
    alg_total_long = alg_total.melt(
        id_vars=['turbine_idx'], value_vars=method_cols, var_name='col', value_name='algebraic_value'
    )
    alg_total_long['stage'] = 'Total'

    alg_stage_long = by_stage_df.melt(
        id_vars=['turbine_idx', 'phase'], value_vars=method_cols, var_name='col', value_name='algebraic_value'
    ).rename(columns={'phase': 'stage'})

    alg_long = pd.concat([alg_total_long, alg_stage_long], ignore_index=True)
    alg_long['method_key'] = alg_long['col'].str.replace(r'\[.*\]$', '', regex=True)
    alg_long = alg_long[alg_long['turbine_idx'].isin(common)]

    merged = pd.merge(
        baseline_long[['turbine_idx', 'stage', 'method_key', 'baseline_value']],
        alg_long[['turbine_idx', 'stage', 'method_key', 'algebraic_value']],
        on=['turbine_idx', 'stage', 'method_key'], how='inner'
    )
    merged['abs_error'] = merged['algebraic_value'] - merged['baseline_value']
    merged['rel_error_pct'] = np.where(
        merged['baseline_value'] != 0,
        merged['abs_error'] / merged['baseline_value'] * 100,
        np.nan
    )

    # total_baseline_value is kept in the detail CSV purely as context (how big is this row
    # relative to that turbine+method's Total), it is no longer used to exclude rows from the
    # summaries below. An earlier version filtered out any row where baseline_value was <0.1%
    # of total_baseline_value on the theory that relative error is meaningless near a zero
    # denominator; that filter was removed (see NEXT_STEPS_lca_algebraic.md item 9) because it
    # silently deleted the entire Maintenance-stage row from 4 of 5 validated countries' summary
    # CSVs once every one of its rows fell under the threshold. The only rows genuinely excluded
    # from the %-based stats now are baseline_value == 0 (true division by zero, handled as NaN
    # above, not a judgment-call threshold); see _err_summary()'s docstring.
    total_baseline = (
        merged.loc[merged['stage'] == 'Total', ['turbine_idx', 'method_key', 'baseline_value']]
        .rename(columns={'baseline_value': 'total_baseline_value'})
    )
    merged = merged.merge(total_baseline, on=['turbine_idx', 'method_key'], how='left')

    detail_path = _VALIDATION_DIR / f"{output_prefix}_validation_errors.csv"
    merged.to_csv(detail_path, index=False)

    n_methods = merged['method_key'].nunique()
    n_stages = merged['stage'].nunique()
    n_undefined = merged['rel_error_pct'].isna().sum()
    print(f"\n  Full validation vs baseline for {label} (all stages, all impact categories): "
          f"{n_val} turbines x {n_methods} methods x {n_stages} stages "
          f"= {len(merged)} (turbine,stage,method) comparisons")
    print(f"  ({n_undefined} rows have baseline_value == 0 -> rel_error_pct is undefined (NaN) "
          f"and are skipped by the %-based stats below; abs_error/mean_abs_error still cover "
          f"every row. No other rows are excluded.)")

    by_stage_err = _err_summary(merged, 'stage')
    print("\n  By stage (all rows; 'n' = rows with a defined %, i.e. baseline_value != 0):")
    print(by_stage_err)

    by_method_err = _err_summary(merged, 'method_key')
    print("\n  By impact category (5 worst by abs mean error):")
    print(by_method_err.sort_values('abs_mean_error_pct', ascending=False).head())

    by_turbine_err = _err_summary(merged[merged['stage'] == 'Total'], 'turbine_idx')
    print(f"\n  Totals only, across all methods, per-turbine error "
          f"(N={len(by_turbine_err)} turbines):")
    print(f"    Mean error     : {by_turbine_err['mean_error_pct'].mean():.2f}%")
    print(f"    Abs mean error : {by_turbine_err['abs_mean_error_pct'].mean():.2f}%")
    print(f"    Abs max error  : {by_turbine_err['abs_max_error_pct'].max():.2f}%")

    by_stage_err.to_csv(_VALIDATION_DIR / f"{output_prefix}_validation_summary_by_stage.csv")
    by_method_err.to_csv(_VALIDATION_DIR / f"{output_prefix}_validation_summary_by_method.csv")
    by_turbine_err.to_csv(_VALIDATION_DIR / f"{output_prefix}_validation_summary_by_turbine.csv")

    print(f"\n  Detailed per-(turbine,stage,method) errors saved to {detail_path}")
    print(f"  Summaries saved to {output_prefix}_validation_summary_by_{{stage,method,turbine}}.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Project + lca_algebraic foreground DB
# ─────────────────────────────────────────────────────────────────────────────
bd.projects.set_current('wimby')
# Foreground DB name is per-country so that run_remaining_countries.sh (one subprocess per
# country) and re-runs of this same script never collide with or silently reuse another
# country's foreground activities.
FOREGROUND = f'rewind_foreground_{COUNTRY.lower()}'
agb.resetDb(FOREGROUND)      # wipe/recreate this country's foreground DB: safe to call every run
agb.setForeground(FOREGROUND)  # tell lca_algebraic which DB holds the parametric activities we're about to define

# lca_algebraic caches compiled expressions to disk, keyed only by db name, NOT by the
# methods/axis used to compute them. resetDb() does not clear this cache, so a previous run
# that used a different `axis`/methods combination can silently poison this one. Always clear
# it explicitly at the start of a run.
clear_caches()

# All EF v3.1 impact categories (same filter as brightway_example_for_elie.ipynb, cell 7),
# not just climate change. 'no LT' = no-long-term variants, 'EN1' = a duplicate/legacy set;
# both excluded there too.
EF_METHODS = [m for m in bd.methods
              if 'EF v3.1' in str(m) and 'no LT' not in str(m) and 'EN1' not in str(m)]
print(f"Computing {len(EF_METHODS)} EF v3.1 impact categories")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Symbolic parameters
# ─────────────────────────────────────────────────────────────────────────────
# The four foreground variables that vary per turbine in the original create_dictionary_update()
# call. Declaring them via agb.newFloatParam() (rather than plain Python floats) is what makes
# every formula built from them below a SymPy expression instead of a fixed number; that's the
# whole mechanism this script relies on: build the model once symbolically, then substitute a
# whole array of per-turbine values into it in one batched numpy evaluation (Section 16/17).
# min/max bound the fleet's actual range and are only used by lca_algebraic for validation/
# plotting, not to clip evaluated values.
P         = agb.newFloatParam('P',         default=2000, min=100,   max=10000)  # rated power, kW
h         = agb.newFloatParam('h',         default=100,  min=40,    max=200)    # hub height, m
d         = agb.newFloatParam('d',         default=80,   min=30,    max=160)    # rotor diameter, m
park_size = agb.newFloatParam('park_size', default=50,   min=1,     max=300)    # turbines sharing this wind park (land use / cabling shared per-turbine)

# Per-turbine location-dependent quantities, pre-computed once by
# precompute_geo_columns.py (see REWIND/REWIND/data/geo_precomputed/dk_geo_precomputed.csv).
# These replace the fleet-average transport distances and P-only cable proxy
# used in earlier versions; see PLAN_lca_algebraic.md for the error this fixes.
dist_rotor   = agb.newFloatParam('dist_rotor',   default=87545,  min=0, max=500000)  # m, manufacturer -> site
dist_nacelle = agb.newFloatParam('dist_nacelle', default=260911, min=0, max=500000)  # m
dist_tower   = agb.newFloatParam('dist_tower',   default=107267, min=0, max=500000)  # m
dist_to_grid = agb.newFloatParam('dist_to_grid', default=26000,  min=0, max=200000)  # m, cable length

# ─────────────────────────────────────────────────────────────────────────────
# 3. Inventory tables (loaded once)
#    df_perc:    percentage of each material within each component, indexed by P
#    df_inv_nkg: non-kg (m, km) quantities indexed by P
#    df_act:     dataset-name → ecoinvent activity UUID mapping
# ─────────────────────────────────────────────────────────────────────────────
print("Loading inventory tables...")
t0 = time.perf_counter()
df_perc   = percentage_inventory()
df_inv_nkg = inventory_not_kg()
df_act    = activities_and_uuids(lon=9.5, lat=56.2)   # central Jutland
print(f"  done in {time.perf_counter()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 4. sympy_interp: exact SymPy equivalent of np.interp
#    Converts (xs, ys) to a SymPy Piecewise that is linear between breakpoints
#    and clamped to ys[0] / ys[-1] outside the range.
# ─────────────────────────────────────────────────────────────────────────────
def sympy_interp(x_sym, xs, ys):
    """MAJOR BUG FIXED 16 Jul 2026: the last piece previously used cond=True (a catch-all)
    with the final segment's linear formula, which *extrapolates* for x beyond xs[-1] instead
    of clamping to ys[-1] like np.interp actually does. Found while debugging the offshore
    model (M_elec_sym evaluated to 7724.67 instead of the correct clamped 3946 for P=3600,
    since the M_electronics table's last breakpoint is 2000), but this affected EVERY
    percentage-split table in BOTH the onshore and offshore models for any turbine with P (or
    whatever the interpolation variable is) beyond that specific table's last breakpoint. Very
    likely the dominant cause of the ~6% abs mean error measured on the real 50-turbine
    baseline run before this fix. See PLAN_lca_algebraic.md, "Completed 16 Jul 2026".
    """
    xs = [float(v) for v in xs]
    ys = [float(v) for v in ys]
    pieces = [(ys[0], x_sym <= xs[0])]             # left clamp
    for i in range(len(xs) - 1):
        dx = xs[i+1] - xs[i]
        slope = (ys[i+1] - ys[i]) / dx if dx else 0.0
        seg   = ys[i] + slope * (x_sym - xs[i])
        pieces.append((seg, x_sym <= xs[i+1]))
    pieces.append((ys[-1], True))                  # right clamp (catch-all beyond last breakpoint)
    return Piecewise(*pieces)


def sympy_extrap(x_sym, xs, ys):
    """EXACT SymPy equivalent of scipy.interpolate.InterpolatedUnivariateSpline(xs, ys, k=1):
    piecewise-linear but EXTRAPOLATING beyond both endpoints (continues the boundary segment's
    slope), unlike sympy_interp/np.interp which clamp.

    Found 26 Jul 2026 while root-causing a "climate change: biogenic" Assembly-stage error (up
    to 213%) on large (P=5080kW) offshore turbines: built_inventory.py's non-kg Assembly loop
    (built_inventory.py:301-306) uses InterpolatedUnivariateSpline, not np.interp, for every
    non-Foundation activity; in practice only Tower's 'Galvanizing [m]' / 'Steel arc welding
    [m]' (confirmed the only two datasets on this code path; Foundation's own non-kg activities
    use np.interp and are unaffected). Their table's last breakpoint is 2000 kW; for a 5080kW
    turbine, spline(5080)=325.53 (extrapolated) vs np.interp(5080)=228.0 (clamped); confirmed
    against the live baseline via built_inventory.create_dictionary_update(). sympy_interp's
    clamping (itself a correct fix for the *other* np.interp-based tables, 16 Jul 2026) is wrong
    specifically for this one; this is the reverse of that bug. See PLAN_lca_algebraic.md,
    "Assembly non-kg extrapolation bug" for the full writeup.
    """
    xs = [float(v) for v in xs]
    ys = [float(v) for v in ys]
    if len(xs) == 1:
        return ys[0]  # single point: no slope to extrapolate, same as sympy_interp/np.interp
    slope_first = (ys[1] - ys[0]) / (xs[1] - xs[0])
    pieces = [(ys[0] + slope_first * (x_sym - xs[0]), x_sym <= xs[0])]   # left extrapolation
    for i in range(len(xs) - 1):
        dx = xs[i+1] - xs[i]
        slope = (ys[i+1] - ys[i]) / dx if dx else 0.0
        seg   = ys[i] + slope * (x_sym - xs[i])
        pieces.append((seg, x_sym <= xs[i+1]))
    slope_last = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    pieces.append((ys[-1] + slope_last * (x_sym - xs[-1]), True))       # right extrapolation
    return Piecewise(*pieces)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Symbolic component masses
#    Tower, Nacelle, Rotor are exact polynomial functions of (P, d, h).
#    M_reinf and M_elec used to be linear fits (polyfit), now exact Piecewise.
# ─────────────────────────────────────────────────────────────────────────────
# Reference turbine used to build the one-off reference inventory (Section 7) that discovers
# which ecoinvent activities exist and their keys, NOT a value any turbine is scaled relative
# to (every mass below is an exact closed-form function of the real per-turbine P/h/d, not a
# ratio against these constants). Only M_tower_ref/M_nacelle_ref/M_rotor_ref (further below) are
# still used as denominators, and only in fallback branches that should never actually fire.
P_REF, H_REF, D_REF, PS_REF = 2000.0, 100.0, 80.0, 50.0

# Material masses, all in kg, as exact SymPy expressions in the turbine's own P/h/d, same
# polynomial coefficients scaling.py fits from Dominik's original turbine-catalogue regression,
# just written out here directly and evaluated symbolically instead of as Python floats.
M_tower_sym   = (3.03584782e-04 * d**2 * h + 9.68652909e+00) * 1e3   # tower steel mass, kg, from func_tower_weight_d2h(d, h)
M_nacelle_sym = (1.66691134e-06 * P**2 + 3.20700974e-02 * P) * 1e3   # nacelle mass, kg, from func_nacelle_weight_power(P)
M_rotor_sym   = (0.00460956 * d**2 + 0.11199577 * d) * 1e3           # rotor (blades+hub) mass, kg, from func_rotor_weight_rotor_diameter(d)
M_found_sym   = 1696e3 * h / 80 * d**2 / 10000                        # onshore gravity-base foundation mass, kg (built_inventory.py's hardcoded formula)

# Exact piecewise (replaces np.polyfit from v1)
M_reinf_sym   = sympy_interp(P, [750, 2000, 4500], [10210, 27000, 51900])                       # reinforcement steel inside the foundation, kg
M_elec_sym    = sympy_interp(P, [30, 150, 600, 800, 2000], [150, 300, 862, 1112, 3946])          # nacelle electronics/electrical mass, kg
V_conc_sym    = (M_found_sym - M_reinf_sym) / 2200                                                # concrete volume, m3, remaining foundation mass after reinforcement, divided by concrete density (kg/m3)
M_all_sym     = M_tower_sym + M_nacelle_sym + M_rotor_sym + M_found_sym + M_elec_sym               # whole-turbine mass, kg, used to scale a handful of mass-proportional exchanges (e.g. end-of-life transport)

# Cable mass (onshore): exact formula from scaling.cable_requirements_Onshore_v01,
# using the real per-turbine dist_to_grid instead of the old P/P_REF proxy.
# cross_section1 = 300*P / df33.loc[300].P, where df33.loc[300].P = 33 * 652 = 21516 (nexans_cable_33kV table)
M_cable_sym = (300.0 * P / 21516.0) * 1e-6 * dist_to_grid * 8960.0 * (617.0 / 220.0) * 0.5

# Material split within the cable, from percentage_inventory(), Input/Power supply/Cable.
# These fractions are identical at every real data point (30/150/600/800 kW; the 2000kW row
# in df_perc is a flat-forward-fill artifact of pandas interpolation, not real data; see
# PLAN_lca_algebraic.md for why this is handled as a constant instead of via sympy_interp).
CABLE_PERC = {
    'Copper':                0.356564,
    'HDPE granules':         0.354943,
    'PP granules':           0.032415,
    'PVC impact resistant':  0.256078,
}

# Reference float values (for cable/power-supply fallback scaling only)
LT = 20  # default lifetime in built_inventory.py

M_tower_ref   = func_tower_weight_d2h(D_REF, H_REF, 3.03584782e-04, 9.68652909e+00)
M_nacelle_ref = func_nacelle_weight_power(P_REF, 1.66691134e-06, 3.20700974e-02)
M_rotor_ref   = func_rotor_weight_rotor_diameter(D_REF, 0.00460956, 0.11199577)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Country background activities (once, at the reference point computed in section 0)
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading {COUNTRY} background activities...")
t0 = time.perf_counter()
mybio, eidb = ecoinvent_setup(_DATA_DIR / "datasets")
truck_dk, ship_dk, cement_dk, elec_dk = transport_cement_elec(lon=LON_REF, lat=LAT_REF)
steel_dk   = steel_dataset(lon=LON_REF, lat=LAT_REF)
MV_transfo = transfo_10mva()
HV_transfo = transfo_500mva()
print(f"  done in {time.perf_counter()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Reference inventory (once): to get Activity objects for Foundation,
#    Power supply, and Assembly/Disposal activity keys
# ─────────────────────────────────────────────────────────────────────────────
print("Building reference inventory (once)...")
t0 = time.perf_counter()
ref = create_dictionary_update(
    P=P_REF, lon=LON_REF, lat=LAT_REF,
    h=H_REF, d=D_REF, park_size=int(PS_REF),
    sea_depth=0, print_details=False
)
print(f"  done in {time.perf_counter()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Material activity lookups (used to wire Assembly and Disposal)
# ─────────────────────────────────────────────────────────────────────────────
def _act(dataset, phase=None):
    """Look up a bw2 Activity by dataset name (and optionally phase) from df_act."""
    if phase:
        m = df_act[(df_act['Dataset'] == dataset) & (df_act['Phase'] == phase)]
    else:
        m = df_act[df_act['Dataset'] == dataset]
    if m.empty:
        return None
    uuid = m.iloc[0]['UUID']
    # df_act has two UUID formats: tuple (db, code) or bare code string
    if isinstance(uuid, tuple):
        return bd.get_activity(uuid)
    return bd.get_activity(code=uuid)

# One Activity object per raw material, looked up once by name from df_act (the dataset-name ->
# ecoinvent-UUID table). These are reused below wherever a material appears in more than one
# stage: e.g. copper_act is both an Input exchange (cable) and the target of Assembly's
# "wire drawing, copper" exchange (Section 10), and low_alloy_act is both Input (tower/rotor
# steel) and the Disposal-phase 'Steel, inert waste' target (Section 13).
copper_act    = _act('Copper',             'Input')
low_alloy_act = _act('Low-alloy steel',    'Input')
cast_iron_act = _act('Cast iron',          'Input')
aluminium_act = _act('Aluminium 0% recycled', 'Input')
chromium_act  = _act('Chromium steel',     'Input')
fiberglass_act= _act('Fiberglass',         'Input')
hdpe_act      = _act('HDPE granules',      'Input')
pp_act        = _act('PP granules',        'Input')
pvc_act       = _act('PVC impact resistant', 'Input')

# Foundation activities from reference dict: pulled from the one-off reference inventory
# (Section 7) rather than via _act() because built_inventory.py creates these two as bespoke,
# non-market activities (not a simple name lookup in df_act); the reference dict is the only
# place their real Activity object appears, so grab it once and reuse it symbolically below.
concrete_act  = list(ref['Input']['Foundation']['Concrete, 30MPa'].keys())[0]   # ecoinvent concrete activity: amount will be V_conc_sym (m3)
reinf_act     = list(ref['Input']['Foundation']['Reinforced concrete'].keys())[0]  # reinforcing steel activity: amount will be M_reinf_sym (kg)

# ─────────────────────────────────────────────────────────────────────────────
# 9. INPUT exchange dict: exact piecewise amounts
#
#    For Tower, Nacelle, Rotor, Electronics: each material amount is
#      amount(P, d, h) = piecewise_percentage(P) × M_component(d, h [or P])
#    This matches the np.interp call in built_inventory.py exactly.
# ─────────────────────────────────────────────────────────────────────────────
COMP_MASS = {
    'Tower':       M_tower_sym,
    'Nacelle':     M_nacelle_sym,
    'Rotor':       M_rotor_sym,
    'Electronics': M_elec_sym,
}

input_exc = {}   # dict: {ecoinvent Activity -> SymPy expression for its exchange amount}, built up incrementally below

phase = 'Input'
# df_perc is a MultiIndex-columned table (component -> sub_component -> dataset -> P breakpoint)
# read from the same "Wind turbines inventories_03.xlsx" percentage-split sheet
# built_inventory.py's own np.interp calls read from. Walking all three column levels here
# reproduces exactly the same (component, sub_comp, dataset) triples the baseline iterates over.
for component in df_perc[phase].columns.get_level_values(0).unique():
    if component in ('Foundation', 'Power supply'):
        continue   # handled separately below (Foundation: hardcoded formulas; Power supply: cable, further down)
    M_sym = COMP_MASS.get(component)
    if M_sym is None:
        continue   # a component not in COMP_MASS (Tower/Nacelle/Rotor/Electronics) has no mass formula here, skip it
    for sub_comp in df_perc[phase][component].columns.get_level_values(0).unique():
        for dataset in df_perc[phase][component][sub_comp].columns.get_level_values(0).unique():
            # series: this (component, sub_comp, dataset)'s percentage-of-mass at each P
            # breakpoint (e.g. Tower/Steel/'Low-alloy steel' = 92% at every breakpoint).
            # dropna() strips breakpoints that don't apply to this dataset.
            series = df_perc[phase][component][sub_comp][dataset].dropna()
            if series.empty:
                continue
            xs = series.index.astype(float).values   # P breakpoints (kW)
            ys = series.values.astype(float)          # mass percentage at each breakpoint
            perc_sym = sympy_interp(P, xs, ys)        # exact piecewise-linear percentage as a function of this turbine's P
            amt_sym  = perc_sym * M_sym                # amount, kg = percentage x total component mass (M_tower_sym etc.)
            act = _act(dataset, phase)
            if act is None:
                continue
            # += rather than = : two different (sub_comp, dataset) pairs can map to the same
            # ecoinvent activity (e.g. several steel grades all resolving to one market
            # activity), and their contributions must be summed, not overwritten.
            input_exc[act] = input_exc.get(act, 0) + amt_sym

# Foundation (hardcoded formulas, exact)
input_exc[concrete_act] = V_conc_sym
input_exc[reinf_act]    = M_reinf_sym

# Land use: built_inventory.py does NOT give this a static amount. It rewrites a shared,
# persistent "land use for onshore wind turbines" activity's own biosphere exchanges on every
# create_dictionary_update() call (built_inventory.py:190-257) via np.interp(P, ...) on each of
# the 6 underlying land-transformation/land-occupation datasets, then adds that activity to
# Input at a flat 1/park_size. That works for the baseline's sequential per-turbine loop (each
# turbine's own P overwrites the shared recipe immediately before its own LCA solve), but the
# recipe is NOT re-derived per turbine here: `ref` was built once at P_REF, so treating the
# frozen activity's amount as the only lever ignores that its *internal* percentages are also
# P-dependent (piecewise breakpoints only go up to 800 kW, so anything off that range gets
# silently clamped at whatever P built `ref`). Confirmed via the cached percentage tables: at
# P_REF=2000 (clamped to the 800kW row) 'Conversion to industrial area'=121 m2 and 'Use of
# industrial area'=4840 m2, vs the ~30x-smaller left-clamped values (4 m2, 160 m2) that apply
# below the first breakpoint (30kW), exactly the range of DK's smallest onshore turbines,
# which is where this showed up as a "land use" GWP-category error up to 5.9%. Fixed by
# expanding each land-use dataset directly as its own sympy_interp term (bypassing the frozen
# activity's amount entirely), matching built_inventory.py's per-turbine np.interp exactly. See
# PLAN_lca_algebraic.md, "land use bug" for the full writeup.
for _lu_dataset, _lu_series in df_inv_nkg['Input']['Foundation']['Land use'].items():
    _lu_series = _lu_series.dropna()
    if _lu_series.empty:
        continue
    _lu_xs = _lu_series.index.astype(float).values
    _lu_ys = _lu_series.values.astype(float)
    _lu_act = _act(_lu_dataset, 'Input')
    if _lu_act is None:
        continue
    input_exc[_lu_act] = input_exc.get(_lu_act, 0) + sympy_interp(P, _lu_xs, _lu_ys) / park_size

# Transformer: onshore only adds MV_transfo, with a hardcoded 19/35 (not lifetime/35).
# built_inventory.py:340-344 ("if offshore==False:") never adds HV_transfo at all, that's
# offshore-only (the export-cable step-up transformer, built_inventory.py:331-336, which does
# use lifetime/35). This was previously wrong here (both transfos added, using LT/35=20/35
# instead of the onshore-specific hardcoded 19/35); found while cross-checking the offshore
# branch for the Monopile model below; fixed 16 Jul 2026, see PLAN_lca_algebraic.md.
input_exc[MV_transfo] = P / 10e3 / 0.85 * 19 / 35

# Power supply cable: exact formula using the real per-turbine dist_to_grid
# (replaces the old P/P_REF proxy; see M_cable_sym / CABLE_PERC above)
for dataset, act in [('Copper', copper_act), ('HDPE granules', hdpe_act),
                     ('PP granules', pp_act), ('PVC impact resistant', pvc_act)]:
    if act is None:
        continue
    input_exc[act] = input_exc.get(act, 0) + CABLE_PERC[dataset] * M_cable_sym

# ─────────────────────────────────────────────────────────────────────────────
# 10. ASSEMBLY exchange dict
#
#     Rolling/drawing amounts = sum of corresponding INPUT activities (exact)
#     Non-kg activities (galvanizing, welding) = exact piecewise from df_inv_nkg
#     Road = 4 × P / park_size (exact: 2-point interp is perfectly linear)
#     Electricity = 0.5 × (M_nacelle + M_rotor + M_tower) (exact)
#     Explosives = 10 kg (constant)
# ─────────────────────────────────────────────────────────────────────────────
assembly_exc = {}

# Dispatch by activity *name* rather than by df_perc/df_inv_nkg table lookup: the reference
# inventory's Assembly dict (`ref['Assembly']`) already enumerates exactly the Activity objects
# that appear in this stage, so each is matched here by a name substring to decide which exact
# formula applies (rolling/drawing amounts mirror the Input-phase material sums, road/
# electricity/explosives are separate hardcoded formulas, and anything else falls through to
# the non-kg piecewise lookup, or, if that also fails to match, the fallback warning below).
for act, ref_amt in ref['Assembly'].items():
    name = act['name'].lower()

    if 'wire drawing, copper' in name:
        assembly_exc[act] = input_exc.get(copper_act, 0)

    elif 'sheet rolling, steel' in name and 'chromium' not in name:
        assembly_exc[act] = (input_exc.get(low_alloy_act, 0)
                             + input_exc.get(cast_iron_act, 0))

    elif 'sheet rolling, aluminium' in name:
        assembly_exc[act] = input_exc.get(aluminium_act, 0)

    elif 'sheet rolling, chromium' in name:
        assembly_exc[act] = input_exc.get(chromium_act, 0)

    elif 'explosive' in name:
        assembly_exc[act] = 10.0

    elif 'medium voltage' in name or 'electricity' in name:
        assembly_exc[act] = 0.5 * (M_nacelle_sym + M_rotor_sym + M_tower_sym)

    elif 'market for road' in name:
        # np.interp(P,[0,2000],[0,8000]) / park_size = 4*P / park_size (exactly linear)
        assembly_exc[act] = 4.0 * P / park_size

    else:
        # Non-kg assembly activities (galvanizing, arc welding):
        # look up the piecewise table in df_inv_nkg
        dataset_name = act['name']
        matched = False
        for comp in df_inv_nkg['Assembly'].columns.get_level_values(0).unique():
            for sc in df_inv_nkg['Assembly'][comp].columns.get_level_values(0).unique():
                for ds in df_inv_nkg['Assembly'][comp][sc].columns.get_level_values(0).unique():
                    series = df_inv_nkg['Assembly'][comp][sc][ds].dropna()
                    if series.empty:
                        continue
                    m = df_act[(df_act['Dataset'] == ds) & (df_act['Phase'] == 'Assembly')]
                    if m.empty:
                        continue
                    candidate = bd.get_activity(m.iloc[0]['UUID'])
                    if candidate.key == act.key:
                        xs = series.index.astype(float).values
                        ys = series.values.astype(float)
                        # sympy_extrap, not sympy_interp: this code path corresponds to
                        # built_inventory.py's InterpolatedUnivariateSpline (extrapolates),
                        # not np.interp (clamps); see sympy_extrap's docstring.
                        assembly_exc[act] = sympy_extrap(P, xs, ys)
                        matched = True
                        break
                if matched:
                    break
            if matched:
                break
        if not matched:
            print(f"    [WARNING] onshore Assembly M_all-scaled fallback hit for: {act['name']!r} "
                  f"(ref_amt={ref_amt}), confirmed dormant for all countries checked so far "
                  f"(26 Jul 2026); if this fires, that activity needs its own exact formula.")
            # Fallback: scale by M_all
            assembly_exc[act] = float(ref_amt) * (M_all_sym / M_all_ref)

# ─────────────────────────────────────────────────────────────────────────────
# 11. TRANSPORT exchange dict: exact per-component formulas
#
#     From scaling.transport_requirements():
#       truck_nacelle  = d_nacelle[m] / 1e6 * M_nacelle[kg]   (tkm)
#       truck_rotor    = d_rotor[m]   / 1e6 * M_rotor[kg]
#       truck_tower    = d_tower[m]   / 1e6 * M_tower[kg]
#       truck_found    = d_found[m]   / 1e6 * M_found[kg]
#       truck_eol      = 200          / 1e3  * M_all[kg]
#       truck_maint    = 2160/30/1000 * LT                     (constant)
#       ship_tower     = 8050         / 1e3  * M_tower[kg]
# ─────────────────────────────────────────────────────────────────────────────
# Real per-turbine distances (dist_rotor, dist_nacelle, dist_tower, dist_to_grid
# symbolic params, defined in Section 2) replace the old fleet-average constants.
# d_found is always 50 km: constant inside calculate_minimum_aggregated_distances
# itself, not an approximation.
d_found = 50000.0   # m

trsp_truck_sym = (
    dist_nacelle / 1e6 * M_nacelle_sym
  + dist_rotor   / 1e6 * M_rotor_sym
  + dist_tower   / 1e6 * M_tower_sym
  + d_found      / 1e6 * M_found_sym
  + 200.0        / 1e3 * M_all_sym          # end-of-life 200 km
  + 2160.0 / 30 / 1000 * LT                # maintenance constant (tkm)
)
trsp_ship_sym = 8050.0 / 1e3 * M_tower_sym   # 8050 km sea freight, tower only

# Assign to the two transport activities from the reference dict
transport_exc = {}
for act, ref_amt in ref['Transport'].items():
    name = act['name'].lower()
    if 'lorry' in name or 'truck' in name or 'freight, land' in name:
        transport_exc[act] = trsp_truck_sym
    elif 'inland' in name or 'ship' in name or 'sea' in name or 'water' in name:
        transport_exc[act] = trsp_ship_sym
    else:
        # Fallback (should not happen for onshore DK)
        transport_exc[act] = float(ref_amt) * (M_all_sym / M_all_ref)

# ─────────────────────────────────────────────────────────────────────────────
# 12. MAINTENANCE exchange dict: exact piecewise for car km
# ─────────────────────────────────────────────────────────────────────────────
maintenance_exc = {}

# Try exact piecewise from df_inv_nkg first: 'Car [km]' (technician site visits, distance
# driven per turbine over its lifetime) is the one Maintenance-phase quantity that has its own
# P-dependent breakpoint table in df_inv_nkg, same table shape as the Assembly non-kg lookup.
car_km_series = (df_inv_nkg['Maintenance']['Nacelle']['Transport by car']['Car [km]']
                 .dropna()
                 .pipe(lambda s: s.set_axis(s.index.astype(float))))
if not car_km_series.empty:
    car_act = _act('Car [km]', 'Maintenance')
    if car_act:
        xs = car_km_series.index.values
        ys = car_km_series.values.astype(float)
        maintenance_exc[car_act] = sympy_interp(P, xs, ys)

# Fallback: any remaining maintenance activities scale with P. Should be dormant in practice
# (the reference inventory's only Maintenance activity is expected to be the car-km one above),
# kept only as a safety net in case a future country's reference turbine pulls in something else.
for act, ref_amt in ref.get('Maintenance', {}).items():
    if act not in maintenance_exc:
        maintenance_exc[act] = float(ref_amt) * (P / P_REF)

# ─────────────────────────────────────────────────────────────────────────────
# 13. DISPOSAL exchange dict
#     Each disposal amount = its corresponding INPUT amount.
#     The mapping is identical to what built_inventory.py does:
#       'Fiberglass -waste' → Fiberglass input sum
#       'Steel, inert waste' → Low-alloy steel input sum
#       'Concrete, inert waste' → Concrete (m3) input
#       'Aluminium waste' → Aluminium input sum
#       'Chromium Steel waste' → Chromium steel input sum
#     All other disposal datasets keep their name → same-name input activity.
# ─────────────────────────────────────────────────────────────────────────────
DISPOSAL_MAP = {
    'Steel, inert waste':  'Low-alloy steel',
    'Concrete, inert waste': None,     # → concrete_act (m3 volume)
    'Aluminium waste':     'Aluminium 0% recycled',
    'Chromium Steel waste': 'Chromium steel',
}

disposal_exc = {}
for disp_ds_name in df_act[df_act['Phase'] == 'Disposal']['Dataset'].unique():
    disp_act = _act(disp_ds_name, 'Disposal')
    if disp_act is None:
        continue

    if disp_ds_name == 'Concrete, inert waste':
        # m3, not kg: use V_conc_sym directly
        disposal_exc[disp_act] = disposal_exc.get(disp_act, 0) + V_conc_sym
        continue

    # Map disposal dataset name → input dataset name
    raw = disp_ds_name.replace(' -waste', '')  # 'Fiberglass -waste' → 'Fiberglass'
    input_ds = DISPOSAL_MAP.get(disp_ds_name, raw)

    input_act = _act(input_ds, 'Input')
    if input_act is None:
        input_act = _act(input_ds)
    if input_act and input_act in input_exc:
        # NOT accumulated: deliberately replicating a baseline bug. built_inventory.py's
        # add_to_dict_2() (scaling.py:404-429) only nests by component/sub_comp for phase
        # =='Input'; every other phase does a flat `dict[phase][key] = value` with no +=.
        # 'Steel, inert waste' and 'Chromium Steel waste' are cloned to the same ecoinvent
        # activity (activities_and_uuids(), prepare_inventories.py:228-230) and 'Chromium
        # Steel waste' is processed last in built_inventory.py's Disposal loop, so baseline
        # silently keeps ONLY the chromium-steel mass and drops the (usually much larger)
        # low-alloy-steel mass for this exchange. Overwriting here (not summing) matches
        # that; see PLAN_lca_algebraic.md "Disposal bug root cause" for the full writeup
        # and NEXT_STEPS_lca_algebraic.md item 2b for the baseline bug report to file.
        disposal_exc[disp_act] = input_exc[input_act]

# ─────────────────────────────────────────────────────────────────────────────
# 14. Create foreground activities
# ─────────────────────────────────────────────────────────────────────────────
print("Building foreground model...")
t0 = time.perf_counter()

# One lca_algebraic Activity per lifecycle stage, each holding the exchange dict built up in
# Sections 9-13 above (Activity -> SymPy amount expression). newActivity() is what actually
# writes these into the foreground Brightway database and converts each SymPy expression into
# a stored exchange formula string.
input_act_fg       = agb.newActivity(FOREGROUND, 'wt_input_dk',       'unit', exchanges=input_exc)
assembly_act_fg    = agb.newActivity(FOREGROUND, 'wt_assembly_dk',    'unit', exchanges=assembly_exc)
transport_act_fg   = agb.newActivity(FOREGROUND, 'wt_transport_dk',   'unit', exchanges=transport_exc)
maintenance_act_fg = agb.newActivity(FOREGROUND, 'wt_maintenance_dk', 'unit', exchanges=maintenance_exc)
disposal_act_fg    = agb.newActivity(FOREGROUND, 'wt_disposal_dk',    'unit', exchanges=disposal_exc)

# Tag each stage with a "phase" attribute so compute_impacts(axis="phase") can
# ventilate results by lifecycle stage instead of returning one aggregate number.
input_act_fg.updateMeta(phase="Input")
assembly_act_fg.updateMeta(phase="Assembly")
transport_act_fg.updateMeta(phase="Transport")
maintenance_act_fg.updateMeta(phase="Maintenance")
disposal_act_fg.updateMeta(phase="Disposal")

# Top-level "one turbine" activity: links all 5 stage activities at amount 1 each, so its
# total impact for any method is simply the sum of the 5 stages' impacts for that method,
# this is what compute_impacts() is called on for the aggregate "Total" result (Section 16),
# while each stage activity is called separately to get the per-stage breakdown.
turbine_dk = agb.newActivity(
    FOREGROUND, 'wind_turbine_onshore_dk', 'unit',
    exchanges={
        input_act_fg:       1,
        assembly_act_fg:    1,
        transport_act_fg:   1,
        maintenance_act_fg: 1,
        disposal_act_fg:    1,
    }
)
print(f"  done in {time.perf_counter()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 15. Load fleet
# ─────────────────────────────────────────────────────────────────────────────
# Despite the "dk_" naming (left over from when this script was DK-only), this loads and
# filters the real fleet for whatever COUNTRY was set in Section 0.
dk_data    = pd.read_excel(_DATA_DIR / "EU_turbines_input_data.xlsx",
                            sheet_name="EU_turbines_input_data")
dk_onshore = dk_data[(dk_data['ISO_code'] == COUNTRY) & (dk_data['Offshore'] == 0)].copy()

# Force numeric dtype: the source spreadsheet stores some columns as text/mixed type, which
# would otherwise silently propagate into the param_df below and break lca_algebraic's
# array-valued parameter substitution.
for col in ['P_rated_kW', 'Hub_height_m', 'Diameter_m', 'Lifetime_production_kWh']:
    dk_onshore[col] = pd.to_numeric(dk_onshore[col], errors='coerce')
dk_onshore['park_size'] = pd.to_numeric(dk_onshore['park_size'], errors='coerce').fillna(50).astype(float)

# Per-turbine geo columns (transport distances, cable length, sea depth),
# built once by precompute_geo_columns.py (covers onshore+offshore together, see
# NEXT_STEPS_lca_algebraic.md), joined here by the original register index.
_geo_path = _GEO_DIR / f"{COUNTRY.lower()}_geo_precomputed.csv"
if not _geo_path.exists():
    raise FileNotFoundError(
        f"{_geo_path} not found, run `python precompute_geo_columns.py` first "
        "(see PLAN_lca_algebraic.md)."
    )
geo = pd.read_csv(_geo_path, index_col=0).drop(columns=['Offshore'])
# Join on the shared register index (index_col=0 in both files): this is the same "index
# label = same physical turbine" assumption assert_same_turbines() above exists to guard
# against elsewhere; here it's a plain left-join, so any misalignment would show up as NaNs,
# caught by the assert on the next line.
dk_onshore = dk_onshore.join(geo, how='left')
assert dk_onshore['dist_rotor_m'].notna().all(), f"geo precompute missing rows for some {COUNTRY} onshore turbines"

# One row per turbine, one column per symbolic parameter declared in Section 2: this is the
# array-valued "param_registry" that turns a single symbolic model into N evaluated turbines in
# one batched compute_impacts() call (Section 16).
param_df = pd.DataFrame({
    'P':            dk_onshore['P_rated_kW'].values.astype(float),
    'h':            dk_onshore['Hub_height_m'].values.astype(float),
    'd':            dk_onshore['Diameter_m'].values.astype(float),
    'park_size':    dk_onshore['park_size'].values.astype(float),
    'dist_rotor':   dk_onshore['dist_rotor_m'].values.astype(float),
    'dist_nacelle': dk_onshore['dist_nacelle_m'].values.astype(float),
    'dist_tower':   dk_onshore['dist_tower_m'].values.astype(float),
    'dist_to_grid': dk_onshore['dist_to_grid_m'].values.astype(float),
})

n = len(param_df)
print(f"\nRunning lca_algebraic for {n} {COUNTRY} onshore turbines...")

# ─────────────────────────────────────────────────────────────────────────────
# 16. Batch compute_impacts: all 25 EF v3.1 methods, total + per-stage breakdown
#
#     Note: lca_algebraic's axis="phase" ventilation cannot be combined with
#     array-valued (fleet-batch) parameters ("Multi params cannot be used together
#     with 'axis'": a hard constraint of the library). So the per-stage breakdown
#     is obtained instead by calling compute_impacts once per stage activity
#     (Input/Assembly/Transport/Maintenance/Disposal), each with the same batch
#     parameter arrays: still fully vectorized across turbines, just 6 calls
#     (1 total + 5 stages) instead of 1.
# ─────────────────────────────────────────────────────────────────────────────
# Keyword arguments for compute_impacts(): one numpy array per symbolic parameter, each of
# length n (one value per turbine). lca_algebraic matches these by parameter *name* (matching
# the newFloatParam() names from Section 2) against whichever parameters actually appear in the
# activity being evaluated: the same GEO_PARAMS dict is reused for all 6 calls below even
# though, e.g., the Maintenance activity's formulas only actually reference P.
GEO_PARAMS = dict(
    P=param_df['P'].values,
    h=param_df['h'].values,
    d=param_df['d'].values,
    park_size=param_df['park_size'].values,
    dist_rotor=param_df['dist_rotor'].values,
    dist_nacelle=param_df['dist_nacelle'].values,
    dist_tower=param_df['dist_tower'].values,
    dist_to_grid=param_df['dist_to_grid'].values,
)
# The 5 stage activities, keyed by the same phase names used throughout (out CSV columns,
# validation grouping, etc.), iterated below to get one compute_impacts() call per stage.
STAGE_ACTS = {
    'Input':       input_act_fg,
    'Assembly':    assembly_act_fg,
    'Transport':   transport_act_fg,
    'Maintenance': maintenance_act_fg,
    'Disposal':    disposal_act_fg,
}

t_start = time.perf_counter()
# reset_index(drop=True): lca_algebraic renames row 0 to the model's name when array-valued
# (ndarray) params are used instead of Python lists (a "single output" fallback path that
# doesn't actually apply here): each of the 6 calls below gets a *different* stray label for
# row 0 ('wind_turbine_onshore_dk', 'wt_input_dk', ...). Left alone, that breaks any
# index-aligned operation across these DataFrames (sum, concat) even though the underlying
# per-turbine values and their positional order are correct.
results_alg = agb.compute_impacts(turbine_dk, methods=EF_METHODS, **GEO_PARAMS).reset_index(drop=True)
stage_results = {phase: agb.compute_impacts(act, methods=EF_METHODS, **GEO_PARAMS).reset_index(drop=True)
                 for phase, act in STAGE_ACTS.items()}
t_algebraic = time.perf_counter() - t_start
print(f"  lca_algebraic : {t_algebraic:.2f}s for {n} turbines x {len(EF_METHODS)} methods x 5 phases "
      f"({1000*t_algebraic/n:.0f} ms/turbine)")

# GWP100 is one specific column among the 25: select it by name, not position
# (column order follows EF_METHODS order, e.g. 'acidification' comes first alphabetically).
GWP100_COL = method_name(CLIMATE_CHANGE) + "[%s]" % method_unit(CLIMATE_CHANGE)

# aep ("annual energy production", named after the source column but actually the turbine's
# whole-lifetime electricity output) is the functional-unit denominator: every impact score
# from compute_impacts() is a lifetime total, so dividing by it converts to the per-kWh basis
# the whole fleet is reported and compared in (matching the baseline's own normalization).
aep         = dk_onshore['Lifetime_production_kWh'].values
gwp_per_kwh = results_alg[GWP100_COL].values / aep

print(f"\n  GWP100 (gCO2eq/kWh):")
print(f"    mean = {gwp_per_kwh.mean():.4f}")
print(f"    min  = {gwp_per_kwh.min():.4f}")
print(f"    max  = {gwp_per_kwh.max():.4f}")

# Sanity check: the 5 stage totals must sum exactly to the turbine_dk total
# (turbine_dk is just {stage: 1 for stage in STAGE_ACTS}, so this should hold to float precision).
stage_sum = sum(stage_results.values())
max_stage_sum_err = (np.abs(stage_sum.values - results_alg[stage_sum.columns].values)).max()
print(f"  Sanity check: max |sum(stages) - total| across all methods/turbines: {max_stage_sum_err:.2e}")

# ─────────────────────────────────────────────────────────────────────────────
# 17. Save results
#
#     Two files:
#       fleet_impacts_DK_lca_algebraic.csv           : one row per turbine, all 25
#                                                       EF v3.1 categories (total
#                                                       across all 5 stages), per kWh
#       fleet_impacts_DK_lca_algebraic_by_stage.csv  : one row per (turbine, stage),
#                                                       same 25 categories, per kWh
# ─────────────────────────────────────────────────────────────────────────────
# --- Total (all 5 stages summed) file ---
per_kwh_total = results_alg.div(aep, axis=0)   # every method column, lifetime-total -> per-kWh
out = dk_onshore[['P_rated_kW', 'Hub_height_m', 'Diameter_m',
                   'Latitude', 'Longitude', 'Lifetime_production_kWh']].copy()
# reset_index(drop=True) on both sides before concat: dk_onshore keeps its original register
# index (e.g. 37093...), but results_alg/per_kwh_total were positionally aligned by
# compute_impacts() (see the reset_index(drop=True) note above): concatenating by position
# (not by index label) is what's actually correct here, then turbine_idx below restores the
# real register index as an explicit column for downstream joins/validation.
out = pd.concat([out.reset_index(drop=True), per_kwh_total.reset_index(drop=True)], axis=1)
out.insert(0, 'turbine_idx', dk_onshore.index.values)
out_path = _RESULTS_DIR / f"fleet_impacts_{COUNTRY}_lca_algebraic.csv"
out.to_csv(out_path, index=False)
print(f"\n  Results (total, {len(EF_METHODS)} categories) saved to {out_path}")

# --- Per-stage (long format: one row per turbine x stage) file ---
# concat with a dict of DataFrames + names=['phase','row'] stacks all 5 stages' per-kWh
# DataFrames on top of each other with a new outer 'phase' index level, which reset_index
# then turns into a plain column.
by_stage = pd.concat(
    {phase: df.div(aep, axis=0) for phase, df in stage_results.items()},
    names=['phase', 'row']
).reset_index(level='phase')
# np.tile repeats the turbine index once per stage, in the same stage order pd.concat above
# used (dict iteration order == STAGE_ACTS insertion order == Input/Assembly/Transport/
# Maintenance/Disposal): this must match, since it's assigned positionally, not via a join.
by_stage['turbine_idx'] = np.tile(dk_onshore.index.values, len(STAGE_ACTS))
by_stage = by_stage[['turbine_idx', 'phase'] + list(per_kwh_total.columns)]
by_stage_path = _RESULTS_BY_STAGE_DIR / f"fleet_impacts_{COUNTRY}_lca_algebraic_by_stage.csv"
by_stage.to_csv(by_stage_path, index=False)
print(f"  Results (by stage, {len(EF_METHODS)} categories) saved to {by_stage_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 18. Full validation vs the baseline: every turbine, every stage, every EF v3.1 category
#
#     fleet_impacts_DK_baseline_all_methods.csv is produced by fleet_evaluation_v03_elie.py
#     (the true baseline, a faithful, tiny-modification reconstruction of v02.py; see
#     PLAN_lca_algebraic.md). This is the sole validation reference for this script;
#     fleet_evaluation_redo_lci.py ("Option B") is a separate speed-optimization exercise, not
#     used here. Compares every lifecycle stage and every impact category, for every turbine
#     common to both files: the (turbine, stage, method) triple is the unit of comparison.
#
#     Baseline is long-format (turbine_idx, Method, stage columns); algebraic output (`out`,
#     `by_stage`, already computed above, already per-kWh) is the opposite shape (turbine/phase
#     rows, method columns). Both are melted to a common (turbine_idx, stage, method_key, value)
#     shape and merged on that key.
#
#     Method-name matching: calculations.py's `lca_wimby_fleet_evaluation` labels each method
#     "EF v3.1 - climate change - ..." (all 3 tuple parts); lca_algebraic's method_name() labels
#     it "climate change - ...[unit]" (2 parts + unit, no "EF v3.1" prefix). Both are normalized
#     to the same "climate change - ..." key (strip the "EF v3.1 - " prefix on one side, the
#     "[unit]" suffix on the other) rather than relying on matching column order between the
#     two independently-built EF_METHODS lists.
# ─────────────────────────────────────────────────────────────────────────────
validate_against_baseline(
    label=f'{COUNTRY} onshore',
    baseline_all_path=_BASELINE_DIR / f"fleet_impacts_{COUNTRY}_baseline_all_methods.csv",
    turbines_index=dk_onshore.index,
    out_df=out,
    by_stage_df=by_stage,
    method_cols=list(per_kwh_total.columns),
    output_prefix=f"fleet_impacts_{COUNTRY}",
)

# ═════════════════════════════════════════════════════════════════════════════
# 19. DK OFFSHORE: foundation-type bucket architecture
#
#     Offshore foundation is a discrete choice by sea_depth bucket
#     (scaling.foundation_type): Monopile <=30m, Semi-submersible 30-60m, Spar buoy >60m.
#     lca_algebraic's symbolic parameters can vary continuously but can't make an exchange
#     appear/disappear based on a parameter value, so each bucket needs its own foreground
#     database + model. All 604 DK offshore turbines fall in the Monopile bucket (sea_depth
#     1-30m, verified; see NEXT_STEPS_lca_algebraic.md item 3), so only that model is built
#     and run here. The Semi-submersible/Spar buoy mass formulas are derived and verified
#     below too (FOUNDATION_MASS_FORMULAS), so extending to a country with deeper offshore
#     turbines is a matter of calling build_offshore_model() with a different bucket, not new
#     engineering, but those two branches are NOT exercised or validated against real
#     turbines yet, since DK has none.
#
#     NOTE: real unit bug in the existing baseline, reproduced here as-is (not "fixed"):
#     scaling.spar_buoy_floating_foundation() returns m_spar_steel/m_spar_iron in TONNES
#     (its own docstring says so; base_steel_weight etc. use "t" with no *1e3), but
#     built_inventory.py uses both values directly as kg exchange amounts with no conversion.
#     Spar buoy foundations therefore get ~1000x too little material in the baseline itself.
#     Matched here for comparability; flag as a separate bug report once a country with
#     spar-buoy turbines is reached.
# ═════════════════════════════════════════════════════════════════════════════

def _foundation_mass_formulas(bucket, P_sym, sea_depth_sym):
    """Exact closed-form symbolic mass formula(s) for one offshore foundation-type bucket.
    Derived from scaling.py's grout_and_monopile_requirements / semi_sub_floating_foundation /
    spar_buoy_floating_foundation (all internally built from np.polyfit(deg=1) chains, so the
    composition is exactly linear in P and sea_depth, no interpolation/approximation).
    Verified numerically against the originals to <1e-3 relative error (floating point only);
    see PLAN_lca_algebraic.md, "Completed 16 Jul 2026: offshore Monopile model", for the
    verification table.
    """
    if bucket == 'Monopile':
        m_grout = 2.44094476537839 * P_sym + 2585.70316709778 * sea_depth_sym + 15406.5926965901
        m_monopile = 5.28641521017598 * P_sym + 5599.92210615505 * sea_depth_sym + 102643.279628469
        return {'grout': m_grout, 'material': m_monopile}
    elif bucket == 'Semi-submersible floating foundation':
        m_semi_sub = 289.473684210526 * P_sym + 17828.5714285714 * sea_depth_sym + 80000.0
        return {'material': m_semi_sub}
    elif bucket == 'Spar buoy foundation':
        # Reproduces the baseline's tonnes-used-as-kg bug (see note above): do not "fix"
        # without a separate decision, since it would break comparability with built_inventory.py.
        m_spar_steel = 0.383333333333333 * P_sym + 6.84 * sea_depth_sym + 300.0
        m_spar_iron = 0.833333333333333 * P_sym
        return {'steel': m_spar_steel, 'iron': m_spar_iron}
    else:
        raise ValueError(f"Unknown foundation bucket: {bucket}")


def build_offshore_model(bucket, ref_P, ref_h, ref_d, ref_sea_depth, ref_park_size=50):
    """Build a parametric lca_algebraic foreground model for one offshore foundation-type
    bucket. Mirrors the onshore model (COMP_MASS percentage-split loop, cable, transport,
    assembly, disposal construction) with the offshore-specific differences from
    built_inventory.py's `if offshore==True:` branch: foundation (bucket-specific, via
    _foundation_mass_formulas), cable (cable_requirements_v01, not _Onshore_v01, extra
    cross_section2 term), transformer (both MV+HV, lifetime/35, this IS correct for offshore,
    unlike the onshore branch's hardcoded 19/35), transport (extra trsp_ship_offshore term),
    assembly (extra scour + cable-laying-ship diesel, no "market for road").

    Returns (turbine_act, stage_acts_dict, sea_depth_param) for use with compute_impacts.
    """
    foreground = f'rewind_foreground_{COUNTRY.lower()}_offshore_{bucket.split()[0].lower()}'
    agb.resetDb(foreground)
    agb.setForeground(foreground)
    clear_caches()

    # Re-registering 'sea_depth' for each bucket (Monopile, Semi-submersible, ...) triggers a
    # benign "[ParamRegistry] Param sea_depth was already defined ... overriding" warning when
    # more than one bucket is built in the same run: harmless (each bucket's resetDb/
    # setForeground call properly isolates its own foreground DB; confirmed correct via full
    # validation on BE, which exercises both Monopile and Semi-submersible in one run).
    sea_depth_param = agb.newFloatParam('sea_depth', default=ref_sea_depth, min=0, max=200)

    # Reference inventory: real offshore turbine values, to discover this bucket's activities.
    ref_off = create_dictionary_update(
        P=ref_P, lon=LON_REF, lat=LAT_REF, h=ref_h, d=ref_d,
        park_size=int(ref_park_size), sea_depth=ref_sea_depth, print_details=False
    )

    # Extra background activities needed only offshore (scour excavation, cable-laying-ship
    # diesel): same direct-name lookup built_inventory.py itself uses (get_activities_from_names).
    digger_act = [act for act in eidb if act['name'] == 'market for excavation, hydraulic digger'][0]
    diesel_burned_act = [act for act in eidb if act['name'] == 'market for diesel, burned in building machine'][0]

    masses = _foundation_mass_formulas(bucket, P, sea_depth_param)

    # M_nacelle/M_rotor use OFFSHORE-SPECIFIC coefficients (built_inventory.py:73-77,
    # "if offshore==True:"): a real, separate distinction from the Monopile/Semi-sub/Spar
    # foundation choice. M_tower and M_foundation (default) do NOT have an offshore variant;
    # confirmed by direct exchange-by-exchange diff against create_dictionary_update: using the
    # onshore M_nacelle_sym/M_rotor_sym here made every nacelle/rotor-associated material
    # (Cast iron, Chromium steel, Aluminium, Fiberglass) wrong by 5-20%; switching to these
    # coefficients matched every one of them exactly. See PLAN_lca_algebraic.md.
    M_nacelle_off_sym = (2.15668283e-06 * P ** 2 + 3.24712680e-02 * P) * 1e3
    M_rotor_off_sym = (0.0088365 * d ** 2 + -0.16435292 * d) * 1e3
    COMP_MASS_OFF = {
        'Tower':       M_tower_sym,
        'Nacelle':     M_nacelle_off_sym,
        'Rotor':       M_rotor_off_sym,
        'Electronics': M_elec_sym,
    }

    # -- INPUT: Tower/Nacelle/Rotor/Electronics: identical percentage-split loop to onshore
    #    (built_inventory.py's loop over df_perc components isn't gated by offshore/onshore),
    #    but with the offshore-specific Nacelle/Rotor masses above.
    input_exc_off = {}
    for component in df_perc['Input'].columns.get_level_values(0).unique():
        if component in ('Foundation', 'Power supply'):
            continue
        M_sym = COMP_MASS_OFF.get(component)
        if M_sym is None:
            continue
        for sub_comp in df_perc['Input'][component].columns.get_level_values(0).unique():
            for dataset in df_perc['Input'][component][sub_comp].columns.get_level_values(0).unique():
                series = df_perc['Input'][component][sub_comp][dataset].dropna()
                if series.empty:
                    continue
                xs = series.index.astype(float).values
                ys = series.values.astype(float)
                amt_sym = sympy_interp(P, xs, ys) * M_sym
                act = _act(dataset, 'Input')
                if act is None:
                    continue
                input_exc_off[act] = input_exc_off.get(act, 0) + amt_sym

    # Foundation (bucket-specific)
    if bucket == 'Monopile':
        input_exc_off[cement_dk] = input_exc_off.get(cement_dk, 0) + masses['grout']
        M_found_off_sym = masses['material']
        found_perc = df_perc['Input']['Foundation']['Material']['Low-alloy steel'].dropna()
        xs, ys = found_perc.index.astype(float).values, found_perc.values.astype(float)
        input_exc_off[low_alloy_act] = input_exc_off.get(low_alloy_act, 0) + sympy_interp(P, xs, ys) * M_found_off_sym
    elif bucket == 'Semi-submersible floating foundation':
        M_found_off_sym = masses['material']
        steel_act_off = _act('Low-alloy steel', 'Input')  # Steel_dataset in built_inventory.py: same market
        input_exc_off[steel_act_off] = input_exc_off.get(steel_act_off, 0) + M_found_off_sym
    else:  # Spar buoy foundation
        M_found_off_sym = masses['steel'] + masses['iron']
        steel_act_off = _act('Low-alloy steel', 'Input')
        input_exc_off[steel_act_off] = input_exc_off.get(steel_act_off, 0) + masses['steel']
        iron_act_off = [act for act in eidb if act['name'] == 'market for iron ore, crude ore, 46% Fe'][0]
        input_exc_off[iron_act_off] = input_exc_off.get(iron_act_off, 0) + masses['iron']

    # NOTE: M_found_off_sym/the real offshore foundation mass is deliberately NOT used below,
    # for transport. built_inventory.py:87-88 computes M_foundation with the onshore
    # gravity-base formula (1696e3*h/80*d**2/10000) as an unconditional default whenever
    # M_foundation isn't explicitly passed in (which none of v02/v03/redo_lci/this script ever
    # do): BEFORE the offshore-specific grout+monopile mass is computed later in the function.
    # transport_requirements() (called with this M_foundation, before the offshore branch runs)
    # therefore uses the onshore-style mass for every turbine's transport calc regardless of
    # offshore/onshore: a real quirk/bug in the baseline, reproduced here to match it exactly
    # (verified: using M_found_sym/M_all_sym here instead of the true offshore mass brought
    # Transport-stage error down from -35% to ~1-2%, matching the residual haversine-vs-geodesic
    # tolerance already documented elsewhere). The Input-phase Foundation materials above
    # correctly use the real M_found_off_sym; only Transport is affected by this quirk.

    # Transformer: offshore adds BOTH MV and HV transfo, with lifetime/35 (built_inventory.py:
    # 331-336: this is the branch LT/35 was originally written for; correct here, unlike onshore).
    input_exc_off[MV_transfo] = P / 10e3 / 0.85 * LT / 35
    input_exc_off[HV_transfo] = P / 500e3 / 0.85 * LT / 35

    # Cable (offshore): exact formula from scaling.cable_requirements_v01, dist_transfo=1
    # (never passed a real value anywhere in this codebase; matches every other script).
    # Offshore cable has two segments, unlike onshore's single inter-array cable: a short
    # fixed-length per-turbine-to-transformer run (cross_section1_off, length DIST_TRANSFO),
    # and a shared park-to-shore export cable (cross_section2_off, length dist_to_grid, split
    # across park_size turbines). Both segments' cross-sections are themselves picked from a
    # cable-rating lookup table (indexed by total park power P*park_size for the export cable),
    # reproduced here as exact sympy_interp piecewise tables instead of built_inventory.py's
    # np.interp on the same nexans_cable_33kV/150kV data. E_CLS_sym (cable-laying-ship diesel)
    # scales with the same two cable lengths and is added to Assembly further down.
    DIST_TRANSFO = 1.0   # km, never given a real per-turbine value anywhere in this codebase, onshore included; kept as-is for comparability
    cross_section1_off = 300.0 * P / 21516.0  # same nexans_cable_33kV lookup as onshore
    m_copper_off = cross_section1_off * 1e-6 * (DIST_TRANSFO * 1e3) * 8960.0   # turbine-to-transformer segment, kg copper (8960 kg/m3 = copper density)
    E_CLS_sym = 450.0 * 39 / 15 * DIST_TRANSFO   # cable-laying-ship diesel for the turbine-to-transformer segment
    # Export-cable (park-to-shore) rating tables: 33kV cable up to 30 MW total park power,
    # 150kV cable above that: same two-table structure as the ecoinvent nexans datasheets.
    df33_P_pts   = [11616, 13167, 14718, 16566, 19173, 21516, 23958, 26763, 29832, 32769]
    df33_idx_pts = [95, 120, 150, 185, 240, 300, 400, 500, 630, 800]
    df150_P_pts   = [106500, 122250, 138750, 156750, 174000, 200250, 213750, 234000]
    df150_idx_pts = [400, 500, 630, 800, 1000, 1200, 1600, 2000]
    cross_section2_off = Piecewise(
        (sympy_interp(P * park_size, df33_P_pts, df33_idx_pts), P * park_size <= 30000),
        (sympy_interp(P * park_size, df150_P_pts, df150_idx_pts), True)
    )
    m_copper_off += cross_section2_off * 1e-6 * (dist_to_grid / park_size) * 8960.0   # export-cable segment, kg copper, shared across the park
    E_CLS_sym += 450.0 * 39 / 15 * (dist_to_grid / 1000.0) / park_size                 # + cable-laying-ship diesel for the export-cable segment
    M_cable_off_sym = (m_copper_off * 617.0 / 220.0) * 0.5   # total cable mass, kg: 617/220 converts copper-only mass to whole-cable mass (includes insulation/sheathing), the same ratio the onshore M_cable_sym formula uses
    E_CLS_sym = E_CLS_sym * 0.5

    for dataset, act in [('Copper', copper_act), ('HDPE granules', hdpe_act),
                         ('PP granules', pp_act), ('PVC impact resistant', pvc_act)]:
        if act is None:
            continue
        input_exc_off[act] = input_exc_off.get(act, 0) + CABLE_PERC[dataset] * M_cable_off_sym

    # ── ASSEMBLY: reuse onshore's non-kg (Tower galvanizing/welding) + rolling/drawing/
    #    electricity/explosives logic, add scour + cable-laying-ship diesel, no "road".
    assembly_exc_off = {}
    for act, ref_amt in ref_off['Assembly'].items():
        name = act['name'].lower()
        if 'wire drawing, copper' in name:
            assembly_exc_off[act] = input_exc_off.get(copper_act, 0)
        elif 'sheet rolling, steel' in name and 'chromium' not in name:
            assembly_exc_off[act] = (input_exc_off.get(low_alloy_act, 0) + input_exc_off.get(cast_iron_act, 0))
        elif 'sheet rolling, aluminium' in name:
            assembly_exc_off[act] = input_exc_off.get(aluminium_act, 0)
        elif 'sheet rolling, chromium' in name:
            assembly_exc_off[act] = input_exc_off.get(chromium_act, 0)
        elif 'explosive' in name:
            assembly_exc_off[act] = 10.0
        elif 'medium voltage' in name or 'electricity' in name:
            assembly_exc_off[act] = 0.5 * (M_nacelle_off_sym + M_rotor_off_sym + M_tower_sym)
        elif 'excavation' in name or 'digger' in name:
            assembly_exc_off[act] = 0.191722269 * P + 1643.34862  # scour_volume(), verified
        elif 'diesel, burned in building machine' in name:
            assembly_exc_off[act] = E_CLS_sym
        else:
            dataset_name = act['name']
            matched = False
            for comp in df_inv_nkg['Assembly'].columns.get_level_values(0).unique():
                for sc in df_inv_nkg['Assembly'][comp].columns.get_level_values(0).unique():
                    for ds in df_inv_nkg['Assembly'][comp][sc].columns.get_level_values(0).unique():
                        series = df_inv_nkg['Assembly'][comp][sc][ds].dropna()
                        if series.empty:
                            continue
                        m = df_act[(df_act['Dataset'] == ds) & (df_act['Phase'] == 'Assembly')]
                        if m.empty:
                            continue
                        candidate = bd.get_activity(m.iloc[0]['UUID'])
                        if candidate.key == act.key:
                            xs = series.index.astype(float).values
                            ys = series.values.astype(float)
                            # sympy_extrap, not sympy_interp; see the matching onshore comment
                            # and sympy_extrap's docstring.
                            assembly_exc_off[act] = sympy_extrap(P, xs, ys)
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    break
            if not matched:
                print(f"    [WARNING] offshore Assembly static fallback hit for: {act['name']!r} "
                      f"(ref_amt={ref_amt}), confirmed dormant for all countries checked so "
                      f"far (26 Jul 2026); if this fires, that activity needs its own exact "
                      f"formula instead of a value frozen at the reference turbine.")
                assembly_exc_off[act] = float(ref_amt)  # static fallback (should not occur)

    # ── TRANSPORT: transport_requirements(lon,lat,M_nacelle,M_tower,M_rotor,M_foundation,M_all,LT)
    #    is called with whatever M_nacelle/M_rotor/M_foundation the function was already holding
    #    at that point (built_inventory.py:106): for offshore that's the offshore-specific
    #    M_nacelle_off/M_rotor_off (real), but M_foundation is still the onshore-style default
    #    (see note above, computed before the offshore branch runs, never overwritten). M_all
    #    here is this same mixed combination, used for both the end-of-life truck term and the
    #    offshore-only trsp_ship_offshore term.
    # Same tkm = distance[m]/1e6 x mass[kg] pattern as onshore's trsp_truck_sym/trsp_ship_sym
    # (Section 11), but every mass term is offshore-specific where one exists (nacelle/rotor),
    # and there's an extra sea-freight term (600 km) on top of onshore's 8050 km tower shipment
    # representing the additional offshore installation-vessel leg from port to the turbine site.
    M_all_for_transport_off_sym = M_nacelle_off_sym + M_tower_sym + M_rotor_off_sym + M_found_sym + M_elec_sym
    trsp_truck_off_sym = (
        dist_nacelle / 1e6 * M_nacelle_off_sym
      + dist_rotor   / 1e6 * M_rotor_off_sym
      + dist_tower   / 1e6 * M_tower_sym
      + d_found      / 1e6 * M_found_sym
      + 200.0        / 1e3 * M_all_for_transport_off_sym   # end-of-life truck transport, 200 km
      + 2160.0 / 30 / 1000 * LT                              # maintenance truck transport (constant, same as onshore)
    )
    trsp_ship_off_sym = 8050.0 / 1e3 * M_tower_sym + 600.0 / 1e3 * M_all_for_transport_off_sym  # + trsp_ship_offshore

    transport_exc_off = {}
    for act, ref_amt in ref_off['Transport'].items():
        name = act['name'].lower()
        if 'lorry' in name or 'truck' in name or 'freight, land' in name:
            transport_exc_off[act] = trsp_truck_off_sym
        elif 'inland' in name or 'ship' in name or 'sea' in name or 'water' in name:
            transport_exc_off[act] = trsp_ship_off_sym
        else:
            transport_exc_off[act] = float(ref_amt)  # static fallback (should not occur)

    # -- DISPOSAL: identical construction to onshore: map each Disposal dataset to its Input
    #    equivalent (naturally handles the offshore materials, e.g. monopile's Low-alloy
    #    steel, with no special-casing needed).
    #    NOT accumulated when two datasets collide on the same ecoinvent activity; see the
    #    matching comment in the onshore disposal_exc block above for why (replicating a
    #    baseline bug in built_inventory.py's add_to_dict_2, so validation stays comparable).
    disposal_exc_off = {}
    for disp_ds_name in df_act[df_act['Phase'] == 'Disposal']['Dataset'].unique():
        disp_act = _act(disp_ds_name, 'Disposal')
        if disp_act is None:
            continue
        if disp_ds_name == 'Concrete, inert waste':
            continue  # no concrete in any offshore foundation type
        raw = disp_ds_name.replace(' -waste', '')
        input_ds = DISPOSAL_MAP.get(disp_ds_name, raw)
        input_act = _act(input_ds, 'Input')
        if input_act is None:
            input_act = _act(input_ds)
        if input_act and input_act in input_exc_off:
            disposal_exc_off[disp_act] = input_exc_off[input_act]

    # ── Foreground activities
    input_act_off       = agb.newActivity(foreground, 'wt_input_dk_off',       'unit', exchanges=input_exc_off)
    assembly_act_off     = agb.newActivity(foreground, 'wt_assembly_dk_off',    'unit', exchanges=assembly_exc_off)
    transport_act_off    = agb.newActivity(foreground, 'wt_transport_dk_off',   'unit', exchanges=transport_exc_off)
    disposal_act_off      = agb.newActivity(foreground, 'wt_disposal_dk_off',    'unit', exchanges=disposal_exc_off)
    input_act_off.updateMeta(phase="Input")
    assembly_act_off.updateMeta(phase="Assembly")
    transport_act_off.updateMeta(phase="Transport")
    disposal_act_off.updateMeta(phase="Disposal")

    turbine_off = agb.newActivity(
        foreground, 'wind_turbine_offshore_dk', 'unit',
        exchanges={input_act_off: 1, assembly_act_off: 1, transport_act_off: 1, disposal_act_off: 1}
    )
    stage_acts_off = {'Input': input_act_off, 'Assembly': assembly_act_off,
                      'Transport': transport_act_off, 'Disposal': disposal_act_off}
    return turbine_off, stage_acts_off, sea_depth_param


geo_path = _GEO_DIR / f"{COUNTRY.lower()}_geo_precomputed.csv"
if not geo_path.exists():
    print(f"\n  ({geo_path} not found, run `python precompute_geo_columns.py` first to enable "
          f"the offshore model. Skipping offshore.)")
else:
    geo_all = pd.read_csv(geo_path, index_col=0)
    dk_offshore = dk_data[(dk_data['ISO_code'] == COUNTRY) & (dk_data['Offshore'] == 1)].copy()
    for col in ['P_rated_kW', 'Hub_height_m', 'Diameter_m', 'Lifetime_production_kWh']:
        dk_offshore[col] = pd.to_numeric(dk_offshore[col], errors='coerce')
    dk_offshore['park_size'] = pd.to_numeric(dk_offshore['park_size'], errors='coerce').fillna(50).astype(float)
    dk_offshore = dk_offshore.join(geo_all[['dist_rotor_m', 'dist_nacelle_m', 'dist_tower_m',
                                             'dist_to_grid_m', 'sea_depth_m']], how='left')

    # Same sea_depth -> bucket rule built_inventory.py itself uses (scaling.foundation_type),
    # applied here up front so the fleet can be split and one dedicated model built per bucket.
    from scaling import foundation_type as _foundation_type
    dk_offshore['bucket'] = dk_offshore['sea_depth_m'].apply(lambda d: _foundation_type(True, d))
    print(f"\n{COUNTRY} offshore fleet: {len(dk_offshore)} turbines, bucket distribution:")
    print(dk_offshore['bucket'].value_counts())

    # One full build_offshore_model() + compute_impacts() cycle per bucket present in this
    # country's fleet (usually just Monopile, sometimes also Semi-submersible/Spar buoy; see
    # the module-level docstring above for which countries have which buckets).
    for bucket, bucket_data in dk_offshore.groupby('bucket'):
        print(f"\nBuilding {COUNTRY} offshore ({bucket}) model for {len(bucket_data)} turbines...")
        # The reference turbine used only to discover this bucket's ecoinvent activities
        # (build_offshore_model's ref_off): any real turbine in the bucket works, since the
        # actual per-turbine values come from off_params below, not from this reference row.
        ref_row = bucket_data.iloc[0]
        turbine_off, stage_acts_off, sea_depth_param = build_offshore_model(
            bucket,
            ref_P=float(ref_row['P_rated_kW']), ref_h=float(ref_row['Hub_height_m']),
            ref_d=float(ref_row['Diameter_m']), ref_sea_depth=float(ref_row['sea_depth_m']),
            ref_park_size=float(ref_row['park_size']),
        )

        # Same role as GEO_PARAMS in the onshore section (Section 16): one array per symbolic
        # parameter, this time including sea_depth (only meaningful offshore).
        off_params = dict(
            P=bucket_data['P_rated_kW'].values.astype(float),
            h=bucket_data['Hub_height_m'].values.astype(float),
            d=bucket_data['Diameter_m'].values.astype(float),
            park_size=bucket_data['park_size'].values.astype(float),
            dist_rotor=bucket_data['dist_rotor_m'].values.astype(float),
            dist_nacelle=bucket_data['dist_nacelle_m'].values.astype(float),
            dist_tower=bucket_data['dist_tower_m'].values.astype(float),
            dist_to_grid=bucket_data['dist_to_grid_m'].values.astype(float),
            sea_depth=bucket_data['sea_depth_m'].values.astype(float),
        )
        # Work around a lca_algebraic_bw25 bug: with exactly 1 sample, its enum-expansion
        # path (_expand_params in params.py) turns array-valued params into raw Python
        # lists via _listOfDictToDictOflist, but only re-wraps them as np.array when
        # param_length > 1, so a batch of 1 keeps plain lists, and `list ** 2` inside the
        # formula crashes. Passing plain floats instead (param_length still computed as 1,
        # but no list ever appears) sidesteps the bug entirely. Never triggered by DK/BE
        # (buckets there always had >=20 turbines): first hit on NO's 1-turbine
        # Semi-submersible bucket.
        if len(bucket_data) == 1:
            off_params = {k: float(v[0]) for k, v in off_params.items()}

        t0 = time.perf_counter()
        results_off = agb.compute_impacts(turbine_off, methods=EF_METHODS, **off_params).reset_index(drop=True)
        stage_results_off = {phase: agb.compute_impacts(act, methods=EF_METHODS, **off_params).reset_index(drop=True)
                              for phase, act in stage_acts_off.items()}
        print(f"  compute_impacts: {time.perf_counter()-t0:.2f}s for {len(bucket_data)} turbines "
              f"x {len(EF_METHODS)} methods x {len(stage_acts_off)} phases")

        aep_off = bucket_data['Lifetime_production_kWh'].values
        gwp_off_per_kwh = results_off[GWP100_COL].values / aep_off
        print(f"  GWP100 (gCO2eq/kWh): mean={gwp_off_per_kwh.mean():.4f} "
              f"min={gwp_off_per_kwh.min():.4f} max={gwp_off_per_kwh.max():.4f}")

        # Total and by-stage CSVs, one pair per bucket: same construction as the onshore
        # "Save results" section above (Section 17: div by aep for per-kWh, positional concat
        # + explicit turbine_idx column, np.tile to repeat the index once per stage), just
        # scoped to this bucket's turbines and named with the bucket slug so Monopile/
        # Semi-submersible/Spar buoy each get their own file per country.
        bucket_slug = bucket.split()[0].lower()
        per_kwh_total_off = results_off.div(aep_off, axis=0)
        out_off = bucket_data[['P_rated_kW', 'Hub_height_m', 'Diameter_m',
                                'Latitude', 'Longitude', 'Lifetime_production_kWh']].copy()
        out_off = pd.concat([out_off.reset_index(drop=True), per_kwh_total_off.reset_index(drop=True)], axis=1)
        out_off.insert(0, 'turbine_idx', bucket_data.index.values)
        out_off_path = _RESULTS_DIR / f"fleet_impacts_{COUNTRY}_offshore_{bucket_slug}_lca_algebraic.csv"
        out_off.to_csv(out_off_path, index=False)
        print(f"  Results (total) saved to {out_off_path}")

        by_stage_off = pd.concat(
            {phase: df.div(aep_off, axis=0) for phase, df in stage_results_off.items()},
            names=['phase', 'row']
        ).reset_index(level='phase')
        by_stage_off['turbine_idx'] = np.tile(bucket_data.index.values, len(stage_acts_off))
        by_stage_off = by_stage_off[['turbine_idx', 'phase'] + list(per_kwh_total_off.columns)]
        by_stage_off_path = _RESULTS_BY_STAGE_DIR / f"fleet_impacts_{COUNTRY}_offshore_{bucket_slug}_lca_algebraic_by_stage.csv"
        by_stage_off.to_csv(by_stage_off_path, index=False)
        print(f"  Results (by stage) saved to {by_stage_off_path}")

        # Validate this bucket against fleet_evaluation_v03_elie.py's offshore baseline run,
        # same mechanics as the onshore validate_against_baseline() call in Section 18.
        validate_against_baseline(
            label=f'{COUNTRY} offshore ({bucket})',
            baseline_all_path=_BASELINE_DIR / f"fleet_impacts_{COUNTRY}_offshore_baseline_all_methods.csv",
            turbines_index=bucket_data.index,
            out_df=out_off,
            by_stage_df=by_stage_off,
            method_cols=list(per_kwh_total_off.columns),
            output_prefix=f"fleet_impacts_{COUNTRY}_offshore_{bucket_slug}",
        )
