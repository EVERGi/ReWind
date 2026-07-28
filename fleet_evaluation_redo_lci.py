"""
fleet_evaluation_redo_lci.py: Option B

Exact results, zero approximation. Eliminates repeated matrix factorisation
by calling redo_lci() instead of a new bc.LCA() for every turbine.

How it compares to the two other approaches:

  Baseline (v03):
    N turbines × 5 stages × bc.LCA().lci().lcia()  =  5N full factorisations
    Each bc.LCA() call decomposes the full (~21k × ~21k) ecoinvent matrix.

  Option A (lca_algebraic):
    1 symbolic model  →  1 factorisation  →  N numpy batch evaluations
    Very fast, but uses approximate scaling for material amounts (~6% mean error).

  Option B (this file):
    Inventory: exact per-turbine (calls create_dictionary_update per turbine)
    LCA solve: 5 factorisations total (one per stage), then redo_lci() for the rest
    Result: zero error vs baseline, LCA-solve bottleneck eliminated.
    Remaining bottleneck: inventory building still O(N), dominates for large fleets.

Speedup profile (measured / extrapolated from 50-turbine benchmark):
  Baseline   50 turbines:  1337s  (26.7s/turbine)
  Option B   50 turbines:   ~820s (16.4s/turbine)   ≈ 1.6× speedup
  Option A   50 turbines:    ~11s  (0.22s/turbine)   ≈ 120× speedup

For a 4328-turbine full DK fleet:
  Baseline:  ~32 hours  (extrapolated)
  Option B:  ~19 hours  (LCA-solve is eliminated, inventory still ~19h)
  Option A:  ~10 seconds (batch numpy)
"""

import sys
from pathlib import Path
from collections import defaultdict

_REWIND_DIR = Path(__file__).resolve().parent / "REWIND" / "REWIND"
sys.path.insert(0, str(_REWIND_DIR))

import pandas as pd
import bw2data as bd
import bw2calc as bc
import time

from built_inventory import create_dictionary_update

_DATA_DIR = _REWIND_DIR / "data"
_BASELINE_DIR = _DATA_DIR / "baseline"   # reorganized 28 Jul 2026, was flat inside _DATA_DIR
_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
CLIMATE_CHANGE = ('EF v3.1', 'climate change', 'global warming potential (GWP100)')
STAGES = ['Input', 'Assembly', 'Maintenance', 'Transport', 'Disposal']

bd.projects.set_current('wimby')

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load turbine fleet
# ─────────────────────────────────────────────────────────────────────────────
dk_data    = pd.read_excel(_DATA_DIR / "EU_turbines_input_data.xlsx",
                            sheet_name="EU_turbines_input_data")
dk_onshore = dk_data[(dk_data['ISO_code'] == 'DK') & (dk_data['Offshore'] == 0)].head(50).copy()

for col in ['P_rated_kW', 'Hub_height_m', 'Diameter_m',
            'Longitude', 'Latitude', 'Lifetime_production_kWh']:
    dk_onshore[col] = pd.to_numeric(dk_onshore[col], errors='coerce')
dk_onshore['park_size'] = pd.to_numeric(dk_onshore['park_size'], errors='coerce').astype(int)

n = len(dk_onshore)
print(f"Fleet: {n} DK onshore turbines")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Helper: flatten one stage of a nested dict_activities into a flat demand
# ─────────────────────────────────────────────────────────────────────────────
def _demand_for_stage(d: dict, stage: str) -> dict:
    """Flatten the dict_activities for one lifecycle stage into {int_id: amount}.
    redo_lci() requires integer node IDs, not Activity objects."""
    stage_data = d[stage]
    if stage == 'Input':
        flat = defaultdict(float)
        for component, sub_dict in stage_data.items():
            for sub_comp, activities in sub_dict.items():
                for act, qty in activities.items():
                    flat[act.id] += qty
        return dict(flat)
    else:
        return {act.id: qty for act, qty in stage_data.items()}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Phase 1: build all inventories (exact, same as v03)
#    This is still O(N); it is the remaining bottleneck in Option B.
# ─────────────────────────────────────────────────────────────────────────────
print("\nPhase 1: building exact inventories...")
t_inv_start = time.perf_counter()
inventories = []   # list of (idx, aep, dict_activities)

for idx, row in dk_onshore.iterrows():
    try:
        d = create_dictionary_update(
            P=row['P_rated_kW'], lon=row['Longitude'], lat=row['Latitude'],
            h=row['Hub_height_m'], d=row['Diameter_m'],
            park_size=row['park_size'], sea_depth=0, print_details=False
        )
        inventories.append((idx, row['Lifetime_production_kWh'], d))
    except Exception as e:
        print(f"  Inventory failed for idx={idx}: {e}")

t_inv = time.perf_counter() - t_inv_start
print(f"  done: {len(inventories)} turbines in {t_inv:.1f}s  ({t_inv/len(inventories):.1f}s/turbine)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Phase 2: LCA with redo_lci (one factorisation per stage, reused for all)
#
#    bc.LCA() decomposes the full ecoinvent matrix (expensive, ~2s per call).
#    redo_lci() swaps only the demand vector and solves again (cheap, ~0.05s).
#
#    Here we call bc.LCA() exactly 5 times (once per stage), then redo_lci()
#    for the remaining N-1 turbines × 5 stages.
# ─────────────────────────────────────────────────────────────────────────────
print("\nPhase 2: LCA with redo_lci...")
t_lca_start = time.perf_counter()

results_by_idx = {idx: {} for idx, _, _ in inventories}
lca_per_stage  = {}   # cache the LCA objects (one per stage) for redo_lci

for stage in STAGES:
    for i, (idx, aep, d) in enumerate(inventories):
        demand = _demand_for_stage(d, stage)

        if i == 0:
            # First turbine: full factorisation
            lca = bc.LCA(demand, CLIMATE_CHANGE)
            lca.lci()
            lca.lcia()
            lca_per_stage[stage] = lca
        else:
            # All subsequent turbines: swap demand vector, reuse factors
            lca = lca_per_stage[stage]
            lca.redo_lci(demand)
            lca.lcia()

        results_by_idx[idx][stage] = lca.score

t_lca = time.perf_counter() - t_lca_start
print(f"  done in {t_lca:.1f}s  ({t_lca/len(inventories):.1f}s/turbine)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Assemble results
# ─────────────────────────────────────────────────────────────────────────────
t_total = t_inv + t_lca
print(f"\nSummary:")
print(f"  Inventory building : {t_inv:.1f}s  ({100*t_inv/t_total:.0f}%)")
print(f"  LCA solve (redo)   : {t_lca:.1f}s  ({100*t_lca/t_total:.0f}%)")
print(f"  Total              : {t_total:.1f}s  ({t_total/n:.1f}s/turbine)")

rows = []
for idx, aep, _ in inventories:
    r = results_by_idx[idx]
    total = sum(r.values())
    gwp   = total / aep
    # Identifying columns (P_rated_kW/Longitude/Latitude) are carried through so that any
    # script comparing against this CSV can hard-verify it's matching against the same
    # physical turbines, not just a coincidentally-equal index label; see
    # NEXT_STEPS_lca_algebraic.md / PLAN_lca_algebraic.md for why this matters.
    row = dk_onshore.loc[idx]
    rows.append({'index': idx,
                 'P_rated_kW': row['P_rated_kW'], 'Longitude': row['Longitude'], 'Latitude': row['Latitude'],
                 'GWP100_total_kgCO2eq': total,
                 'GWP100_gCO2eq_kWh': gwp,
                 **{s: v/aep for s, v in r.items()}})

out = pd.DataFrame(rows).set_index('index')
out_path = _BASELINE_DIR / "fleet_impacts_DK_redo_lci.csv"
out.to_csv(out_path)
print(f"\n  GWP100 (gCO2eq/kWh): mean={out['GWP100_gCO2eq_kWh'].mean():.4f}  "
      f"min={out['GWP100_gCO2eq_kWh'].min():.4f}  max={out['GWP100_gCO2eq_kWh'].max():.4f}")
print(f"  Saved to {out_path}")
