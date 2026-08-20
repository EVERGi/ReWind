#!/usr/bin/env bash
# regenerate_all.sh
#
# Runs the full REWIND analysis pipeline end-to-end and refreshes every derived result,
# figure, and markdown table referenced in the paper -- fleet-level impact results per
# country/siting type, permutation feature importance, material importance/contribution,
# Spearman correlation stats, and all paper figures.
#
# Intended use: run this before cutting a new Zenodo release, so every file in the
# release was regenerated from the same pipeline in one pass, rather than trusting a
# patchwork of outputs from separate runs at different times.
#
# Requires local access to licensed/external inputs that are NOT in this repo (see .gitignore):
#   REWIND/REWIND/data/EU_turbines_input_data.xlsx               (turbine register, licensed)
#   REWIND/REWIND/data/datasets/                                 (ecoinvent 3.9.1, licensed)
#   Shared_Rewind/wind_fleet_data_incl_sea_depths_corrected.csv  (only needed without --skip-lca)
#   Shared_Rewind/Fleet_results/NUTS_RG_20M_2024_4326.gpkg       (NUTS boundaries, always needed)
# GEBCO bathymetry is NOT needed at runtime -- sea depth is already baked into
# REWIND/REWIND/data/geo_precomputed/*.csv.
#
# A Python environment with this project's dependencies must already be active
# (see README.md "Installation") -- this script does not create or activate one.
#
# Usage:
#   ./regenerate_all.sh              # run everything
#   ./regenerate_all.sh --skip-lca   # skip the slow fleet_evaluation_lca_algebraic.py step
#                                     # and reuse existing REWIND/REWIND/data/results/ CSVs
#                                     # (useful when only re-plotting downstream figures)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SKIP_LCA=false
for arg in "$@"; do
    case "$arg" in
        --skip-lca) SKIP_LCA=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

log() { printf '\n=== %s ===\n' "$1"; }

# --- preflight -----------------------------------------------------------------------
log "Preflight checks"

python3 -c "import numpy, pandas, sklearn, matplotlib, scipy, adjustText" || {
    echo "Missing required Python packages -- activate the project's environment first" >&2
    exit 1
}

if [ ! -f "Shared_Rewind/Fleet_results/NUTS_RG_20M_2024_4326.gpkg" ]; then
    echo "Missing Shared_Rewind/Fleet_results/NUTS_RG_20M_2024_4326.gpkg (NUTS boundaries)." >&2
    echo "Needed by generate_paper_figures.py regardless of --skip-lca." >&2
    exit 1
fi

if [ "$SKIP_LCA" = false ]; then
    if [ ! -f "REWIND/REWIND/data/EU_turbines_input_data.xlsx" ]; then
        echo "Missing REWIND/REWIND/data/EU_turbines_input_data.xlsx (licensed turbine register)." >&2
        echo "This file is intentionally gitignored and cannot be redistributed -- place your" >&2
        echo "own licensed copy there, or pass --skip-lca to reuse existing results/ CSVs." >&2
        exit 1
    fi
    if [ ! -d "REWIND/REWIND/data/datasets" ] || [ -z "$(ls -A REWIND/REWIND/data/datasets 2>/dev/null)" ]; then
        echo "Missing/empty REWIND/REWIND/data/datasets/ (ecoinvent 3.9.1, licensed)." >&2
        echo "Place your own licensed ecoinvent export there, or pass --skip-lca." >&2
        exit 1
    fi
    if [ ! -f "Shared_Rewind/wind_fleet_data_incl_sea_depths_corrected.csv" ]; then
        echo "Missing Shared_Rewind/wind_fleet_data_incl_sea_depths_corrected.csv." >&2
        echo "Needed by precompute_geo_columns.py; pass --skip-lca to skip this step." >&2
        exit 1
    fi
fi

# --- pipeline --------------------------------------------------------------------------
if [ "$SKIP_LCA" = false ]; then
    log "Step 1/6: precompute_geo_columns.py (grid distance, transport distances, sea depth)"
    python3 precompute_geo_columns.py

    log "Step 2/6: fleet_evaluation_lca_algebraic.py (LCA for all 38 countries -- this is the slow step)"
    python3 fleet_evaluation_lca_algebraic.py
else
    log "Skipping geo-precompute and LCA evaluation (--skip-lca) -- reusing existing results/ CSVs"
fi

log "Step 3/6: generate_feature_importance.py (permutation importance, headline-category figure)"
python3 generate_feature_importance.py

log "Step 4/6: generate_material_importance.py + generate_material_contribution.py"
python3 generate_material_importance.py
python3 generate_material_contribution.py

log "Step 5/6: generate_paper_summary_stats.py (fleet stats, Spearman correlation vs. CC, validation)"
python3 generate_paper_summary_stats.py

log "Step 6/6: generate_paper_figures.py, generate_structural_figures.py, generate_driver_analysis_visuals.py"
python3 generate_paper_figures.py
python3 generate_structural_figures.py
python3 generate_driver_analysis_visuals.py

log "Done"
echo "All results, figures, and markdown tables have been refreshed."
echo "Review 'git status' before packaging a Zenodo release or committing."
