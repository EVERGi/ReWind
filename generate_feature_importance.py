"""
generate_feature_importance.py

Ranks which per-turbine characteristics actually drive impact variation across the fleet,
using a random forest (impurity-based importance) and permutation importance on a held-out
test set (less biased toward high-cardinality continuous features than impurity alone).

This is a descriptive tool, not a discovery one: fleet_evaluation_lca_algebraic.py's formulas
for how P/d/h/dist_to_grid/etc. map to material mass and impact are already known exactly
(symbolic, not a black box). The point here is to confirm and rank, in the actual fleet data,
which of those known drivers matters most, and to give the paper a figure for it, not to
reverse-engineer something unknown.

Run separately per turbine category (onshore, offshore monopile, offshore semi-submersible,
offshore spar) rather than mixing them with an "Offshore" flag: siting drives which features
even apply (sea depth is meaningless onshore), so pooling categories together and treating
Offshore as just another feature understates how differently the fleet behaves site-to-site.

Features used (all already computed per turbine by precompute_geo_columns.py, or derived here
from mass_formulas.py -- the same closed-form component-mass formulas
fleet_evaluation_lca_algebraic.py itself uses, not a re-fit):
    P_rated_kW, Hub_height_m, Diameter_m    turbine size
    dist_rotor_m, dist_nacelle_m, dist_tower_m   manufacturer -> site transport distances
    dist_to_grid_m                          cable length (turbine -> nearest substation)
    sea_depth_m                             offshore categories only (0 and constant onshore)
    turbine_age, park_size                  fleet/siting characteristics
    Blade_mass_kg, Nacelle_mass_kg, Tower_mass_kg, Foundation_mass_kg   component masses, kg,
        computed from P/d/h(/sea_depth) via mass_formulas.py -- deterministic (nonlinear)
        functions of features already in the list above, so expect these to cluster tightly
        with P_rated_kW/Hub_height_m/Diameter_m/sea_depth_m in the multicollinearity check
        (see importance_stats.py); read the cluster, not any one member, as the finding.

NOT used as predictors, despite being loaded and saved in the per-turbine CSVs:
Lifetime_production_kWh and Capacity_factors. Every target here is impact / lifetime
production (fleet_evaluation_lca_algebraic.py:981), so using that same denominator (or
capacity factor, which determines it almost exactly given rated power) as a predictor is
close to regressing X/Y on Y -- confirmed empirically, not just in principle: including them
pushed every category's held-out R^2 to 0.98-1.00 (vs. a believable 0.73-0.97 without them),
with one of the two topping almost every impact -- the signature of leakage.

Targets: all impact categories in the fleet_impacts_*.csv results (24-25 depending on
whether biogenic/fossil/land-use climate change sub-splits are counted separately from the
headline "climate change - GWP100").

Run once (output is cached to disk):
    python generate_feature_importance.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from importance_stats import (compute_clusters, grouped_permutation_importance,
                               permutation_significance, wrap_cluster_label)
from mass_formulas import (M_tower, M_nacelle_onshore, M_nacelle_offshore, M_rotor_onshore,
                            M_rotor_offshore, M_found_onshore, foundation_masses_offshore)

REPO = Path(__file__).resolve().parent
DATA = REPO / "REWIND" / "REWIND" / "data"
RESULTS_DIR = DATA / "results"
GEO_DIR = DATA / "geo_precomputed"
TURBINES_XLSX = DATA / "EU_turbines_input_data.xlsx"
OUT = REPO / "figures" / "feature_importance"
OUT.mkdir(parents=True, exist_ok=True)
MD_OUT = REPO / "FEATURE_IMPORTANCE.md"

# --- turbine categories -----------------------------------------------------------------
# "onshore" = fleet_impacts_<ISO>_lca_algebraic.csv (no "_offshore_" suffix).
# Offshore files are split by foundation type: fleet_impacts_<ISO>_offshore_<type>_...csv
CATEGORY_ORDER = ["onshore", "offshore_monopile", "offshore_semi-submersible", "offshore_spar"]
CATEGORY_TITLES = {
    "onshore": "Onshore",
    "offshore_monopile": "Offshore – monopile",
    "offshore_semi-submersible": "Offshore – semi-submersible",
    "offshore_spar": "Offshore – spar",
}
MIN_SAMPLES = 30  # below this, an 80/20 split is meaningless; skip modeling and say so

# --- features ----------------------------------------------------------------------------
# Lifetime_production_kWh and Capacity_factors are loaded (TURBINE_FEATURE_COLS below) and kept
# in the saved per-turbine CSVs, but deliberately EXCLUDED from the modeled feature list: every
# impact target here is impact / Lifetime_production_kWh (fleet_evaluation_lca_algebraic.py:981),
# so using that same denominator (or Capacity_factors, which determines it almost exactly given
# rated power and a near-fixed design life) as a predictor of a per-kWh target is close to
# regressing X/Y on Y -- confirmed empirically: including them pushed every category's held-out
# R^2 to 0.98-1.00 (was 0.73-0.97) with "Lifetime production"/"Capacity factor" topping almost
# every impact, the signature of leakage, not a real finding.
FEATURES_COMMON = ["P_rated_kW", "Hub_height_m", "Diameter_m", "dist_rotor_m", "dist_nacelle_m",
                    "dist_tower_m", "dist_to_grid_m", "turbine_age", "park_size",
                    "Blade_mass_kg", "Nacelle_mass_kg", "Tower_mass_kg", "Foundation_mass_kg"]
FEATURES_ONSHORE = FEATURES_COMMON
FEATURES_OFFSHORE = FEATURES_COMMON + ["sea_depth_m"]

FEATURE_LABELS = {
    "P_rated_kW": "Rated power",
    "Hub_height_m": "Hub height",
    "Diameter_m": "Rotor diameter",
    "dist_rotor_m": "Transport dist. (rotor)",
    "dist_nacelle_m": "Transport dist. (nacelle)",
    "dist_tower_m": "Transport dist. (tower)",
    "dist_to_grid_m": "Cable length (grid dist.)",
    "sea_depth_m": "Sea depth",
    "turbine_age": "Turbine age",
    "park_size": "Park size",
    "Lifetime_production_kWh": "Lifetime production",
    "Capacity_factors": "Capacity factor",
    "Blade_mass_kg": "Blade mass",
    "Nacelle_mass_kg": "Nacelle mass",
    "Tower_mass_kg": "Tower mass",
    "Foundation_mass_kg": "Foundation mass",
}

GEO_FEATURE_COLS = ["dist_rotor_m", "dist_nacelle_m", "dist_tower_m", "dist_to_grid_m", "sea_depth_m"]
TURBINE_FEATURE_COLS = ["turbine_age", "park_size", "P_rated_kW", "Hub_height_m", "Diameter_m",
                         "Lifetime_production_kWh", "Capacity_factors"]
# Blade_mass_kg/Nacelle_mass_kg/Tower_mass_kg/Foundation_mass_kg are computed per turbine, not
# loaded from a file -- see _add_mass_columns() below. Onshore/offshore use different
# nacelle/rotor/foundation coefficients (mass_formulas.py), so they're filled in per category
# inside load_feature_tables(), not read from TURBINES_XLSX or GEO_DIR.

# --- impact categories: (full column name in results csv, short label, filename slug) ----
IMPACTS = [
    ("acidification - accumulated exceedance (AE)[mol H+-Eq]",
     "Acidification", "acidification"),
    ("climate change - global warming potential (GWP100)[kg CO2-Eq]",
     "Climate change (total)", "climate_change_total"),
    ("climate change: biogenic - global warming potential (GWP100)[kg CO2-Eq]",
     "Climate change (biogenic)", "climate_change_biogenic"),
    ("climate change: fossil - global warming potential (GWP100)[kg CO2-Eq]",
     "Climate change (fossil)", "climate_change_fossil"),
    ("climate change: land use and land use change - global warming potential (GWP100)[kg CO2-Eq]",
     "Climate change (land use)", "climate_change_land_use"),
    ("ecotoxicity: freshwater - comparative toxic unit for ecosystems (CTUe)[CTUe]",
     "Ecotoxicity, freshwater", "ecotoxicity_freshwater"),
    ("ecotoxicity: freshwater, inorganics - comparative toxic unit for ecosystems (CTUe)[CTUe]",
     "Ecotoxicity, freshwater (inorg.)", "ecotoxicity_freshwater_inorganics"),
    ("ecotoxicity: freshwater, organics - comparative toxic unit for ecosystems (CTUe)[CTUe]",
     "Ecotoxicity, freshwater (org.)", "ecotoxicity_freshwater_organics"),
    ("energy resources: non-renewable - abiotic depletion potential (ADP): fossil fuels[MJ, net calorific value]",
     "Fossil resource use", "fossil_resource_use"),
    ("eutrophication: freshwater - fraction of nutrients reaching freshwater end compartment (P)[kg P-Eq]",
     "Eutrophication, freshwater", "eutrophication_freshwater"),
    ("eutrophication: marine - fraction of nutrients reaching marine end compartment (N)[kg N-Eq]",
     "Eutrophication, marine", "eutrophication_marine"),
    ("eutrophication: terrestrial - accumulated exceedance (AE)[mol N-Eq]",
     "Eutrophication, terrestrial", "eutrophication_terrestrial"),
    ("human toxicity: carcinogenic - comparative toxic unit for human (CTUh)[CTUh]",
     "Human toxicity, carcinogenic", "human_toxicity_carcinogenic"),
    ("human toxicity: carcinogenic, inorganics - comparative toxic unit for human (CTUh)[CTUh]",
     "Human toxicity, carc. (inorg.)", "human_toxicity_carcinogenic_inorganics"),
    ("human toxicity: carcinogenic, organics - comparative toxic unit for human (CTUh)[CTUh]",
     "Human toxicity, carc. (org.)", "human_toxicity_carcinogenic_organics"),
    ("human toxicity: non-carcinogenic - comparative toxic unit for human (CTUh)[CTUh]",
     "Human toxicity, non-carc.", "human_toxicity_non_carcinogenic"),
    ("human toxicity: non-carcinogenic, inorganics - comparative toxic unit for human (CTUh)[CTUh]",
     "Human toxicity, non-carc. (inorg.)", "human_toxicity_non_carcinogenic_inorganics"),
    ("human toxicity: non-carcinogenic, organics - comparative toxic unit for human (CTUh)[CTUh]",
     "Human toxicity, non-carc. (org.)", "human_toxicity_non_carcinogenic_organics"),
    ("ionising radiation: human health - human exposure efficiency relative to u235[kBq U235-Eq]",
     "Ionising radiation", "ionising_radiation"),
    ("land use - soil quality index[dimensionless]",
     "Land use", "land_use"),
    ("material resources: metals/minerals - abiotic depletion potential (ADP): elements (ultimate reserves)[kg Sb-Eq]",
     "Mineral resource use", "mineral_resource_use"),
    ("ozone depletion - ozone depletion potential (ODP)[kg CFC-11-Eq]",
     "Ozone depletion", "ozone_depletion"),
    ("particulate matter formation - impact on human health[disease incidence]",
     "Particulate matter", "particulate_matter"),
    ("photochemical oxidant formation: human health - tropospheric ozone concentration increase[kg NMVOC-Eq]",
     "Photochemical ozone formation", "photochemical_ozone_formation"),
    ("water use - user deprivation potential (deprivation-weighted water consumption)[m3 world eq. deprived]",
     "Water use", "water_use"),
]
IMPACT_COLUMNS = [c for c, _, _ in IMPACTS]
IMPACT_LABELS = {c: label for c, label, _ in IMPACTS}
IMPACT_SLUGS = {c: slug for c, _, slug in IMPACTS}
GWP_COL = "climate change - global warming potential (GWP100)[kg CO2-Eq]"

# Short (2-5 char) abbreviations for the point labels in fig_max_importance_by_siting.png --
# keyed by the same short label used as perm_matrix's column names.
IMPACT_ABBREV = {
    "Acidification": "AC",
    "Climate change (total)": "CC",
    "Climate change (biogenic)": "CCb",
    "Climate change (fossil)": "CCf",
    "Climate change (land use)": "CClu",
    "Ecotoxicity, freshwater": "ET",
    "Ecotoxicity, freshwater (inorg.)": "ETi",
    "Ecotoxicity, freshwater (org.)": "ETo",
    "Fossil resource use": "FRU",
    "Eutrophication, freshwater": "EPf",
    "Eutrophication, marine": "EPm",
    "Eutrophication, terrestrial": "EPt",
    "Human toxicity, carcinogenic": "HTC",
    "Human toxicity, carc. (inorg.)": "HTCi",
    "Human toxicity, carc. (org.)": "HTCo",
    "Human toxicity, non-carc.": "HTN",
    "Human toxicity, non-carc. (inorg.)": "HTNi",
    "Human toxicity, non-carc. (org.)": "HTNo",
    "Ionising radiation": "IR",
    "Land use": "LU",
    "Mineral resource use": "MRM",
    "Ozone depletion": "ODP",
    "Particulate matter": "PM",
    "Photochemical ozone formation": "POF",
    "Water use": "WU",
}

# Palette (from the dataviz skill's validated reference palette, matching generate_paper_figures.py).
# BLUE/ORANGE/AQUA are the palette's first three categorical slots -- the only ones that validate
# on the all-pairs pairlist (scatter/small-multiples), which is what the 3-turbine-category dot
# plot below needs.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, TEXT_SECONDARY = "#c7c6c0", "#52514e"
CATEGORY_COLORS = {"onshore": BLUE, "offshore_monopile": ORANGE, "offshore_semi-submersible": AQUA}

# "Headline" impact categories for fig_top_features_by_impact_grid.png -- a fixed spread across
# what papers in this space usually lead with (a climate metric, a resource-depletion metric, two
# human-health metrics, land, and water), not a data-driven selection.
HEADLINE_IMPACTS = ["Climate change (total)", "Mineral resource use", "Human toxicity, carcinogenic",
                     "Human toxicity, non-carc.", "Land use", "Ionising radiation"]

# Manual row groups for fig_top_features_by_impact_grid.png -- puts related features on the same
# row (turbine size/mass, then transport/logistics, then site/foundation) so a reader scanning
# the grid sees like features side by side instead of raw importance-rank order.
GRID_ROW_GROUPS = [
    ["Hub_height_m", "Diameter_m", "Blade_mass_kg"],
    ["dist_nacelle_m", "dist_tower_m", "dist_to_grid_m"],
    ["sea_depth_m", "Foundation_mass_kg", "park_size"],
]


def _order_features_by_group(features: list[str]) -> list[str]:
    """Reorders `features` into GRID_ROW_GROUPS's row-by-row layout. Any feature not covered by
    the groups is appended at the end, so this stays safe if the top-N feature set ever changes."""
    grouped = [f for row in GRID_ROW_GROUPS for f in row if f in features]
    leftover = [f for f in features if f not in grouped]
    return grouped + leftover


def classify_category(filename: str) -> str:
    m = re.search(r"_offshore_(monopile|semi-submersible|spar)_lca_algebraic\.csv$", filename)
    if m:
        return f"offshore_{m.group(1)}"
    return "onshore"


def _add_mass_columns(merged: pd.DataFrame, category: str) -> pd.DataFrame:
    """Blade_mass_kg, Nacelle_mass_kg, Tower_mass_kg, Foundation_mass_kg, kg -- computed from
    P_rated_kW/Diameter_m/Hub_height_m(/sea_depth_m) via mass_formulas.py, the same closed-form
    coefficients fleet_evaluation_lca_algebraic.py itself uses (not a re-fit or approximation).
    Onshore and offshore use different nacelle/rotor coefficients and a different foundation
    formula entirely (bucketed by foundation type for offshore), so this must run per category,
    not once globally."""
    P = merged["P_rated_kW"].astype(float)
    d = merged["Diameter_m"].astype(float)
    h = merged["Hub_height_m"].astype(float)

    merged["Tower_mass_kg"] = M_tower(d, h)
    if category == "onshore":
        merged["Nacelle_mass_kg"] = M_nacelle_onshore(P)
        merged["Blade_mass_kg"] = M_rotor_onshore(d)
        merged["Foundation_mass_kg"] = M_found_onshore(d, h)
    else:
        merged["Nacelle_mass_kg"] = M_nacelle_offshore(P)
        merged["Blade_mass_kg"] = M_rotor_offshore(d)
        sea_depth = merged["sea_depth_m"].astype(float)
        found = foundation_masses_offshore(category, P, sea_depth)
        merged["Foundation_mass_kg"] = sum(found.values())
    return merged


def load_feature_tables() -> dict[str, pd.DataFrame]:
    """category -> one row per turbine: all IMPACT_COLUMNS + all feature columns."""
    turbines = pd.read_excel(TURBINES_XLSX, sheet_name="EU_turbines_input_data")

    frames_by_category: dict[str, list[pd.DataFrame]] = {c: [] for c in CATEGORY_ORDER}
    result_files = sorted(RESULTS_DIR.glob("fleet_impacts_*_lca_algebraic.csv"))
    for f in result_files:
        category = classify_category(f.name)
        name = f.stem.replace("fleet_impacts_", "").replace("_lca_algebraic", "")
        iso = name.split("_offshore_")[0]
        geo_path = GEO_DIR / f"{iso.lower()}_geo_precomputed.csv"
        geo = pd.read_csv(geo_path, index_col=0)
        res = pd.read_csv(f, index_col=0)

        merged = res[IMPACT_COLUMNS].join(geo[GEO_FEATURE_COLS], how="inner")
        merged = merged.join(turbines[TURBINE_FEATURE_COLS], how="inner")
        merged = _add_mass_columns(merged, category)
        frames_by_category[category].append(merged)

    tables = {}
    for category, frames in frames_by_category.items():
        if not frames:
            tables[category] = pd.DataFrame(columns=IMPACT_COLUMNS + FEATURES_OFFSHORE)
            continue
        df = pd.concat(frames)
        features = FEATURES_ONSHORE if category == "onshore" else FEATURES_OFFSHORE
        df = df.dropna(subset=features + IMPACT_COLUMNS)
        tables[category] = df
        print(f"{category}: {len(df)} turbines with complete feature data")
    return tables


def train_and_rank(df: pd.DataFrame, features: list[str], target_col: str, clusters: dict):
    X = df[features].astype(float)
    y = df[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the target before fitting. Several EF3.0 impact columns have very small
    # absolute magnitudes (e.g. human toxicity, CTUh ~1e-10-1e-11; ozone depletion, kg CFC-11-Eq
    # ~1e-10) -- sklearn's tree-splitter silently fails to find ANY split at that magnitude
    # (every tree collapses to a single-leaf constant predictor, R^2 comes back ~0 from
    # floating-point noise alone, not a real null result). Verified directly: rescaling one such
    # column by a constant factor (which changes nothing statistically) flipped R^2 from -0.003
    # to 0.97. R^2 is exactly invariant under this affine transform (mean/std from TRAIN only,
    # to avoid test-set leakage), so this reports the same number a numerically well-behaved fit
    # would have given, not a different metric -- and is applied to every target uniformly, not
    # just the ones currently broken, since it's harmless at any scale.
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_s = (y_train - y_mean) / y_std
    y_test_s = (y_test - y_mean) / y_std

    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train_s)

    r2 = r2_score(y_test_s, model.predict(X_test))

    impurity = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    perm = permutation_importance(model, X_test, y_test_s, n_repeats=15, random_state=42, n_jobs=-1)
    permutation = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)

    # Significance: is each feature's importance consistently above zero across the 15 shuffle
    # repeats, or indistinguishable from noise? Uses sklearn's own raw per-repeat values
    # (perm.importances), already computed, previously discarded. See importance_stats.py.
    significance = permutation_significance(perm, features)

    # Grouped importance: mitigates permutation importance's known bias under correlated
    # features (shuffling one of several correlated columns barely hurts the model, since it
    # reads the same signal off the correlated partner) by shuffling each cluster of
    # near-collinear features together. See importance_stats.py and clusters computed once per
    # category in main().
    grouped = grouped_permutation_importance(model, X_test, y_test_s, clusters)

    return impurity, permutation, r2, significance, grouped


def plot_importances(impurity: pd.Series, permutation: pd.Series, r2: float, features: list[str],
                      impact_label: str, category_label: str, out_path: Path,
                      significance: pd.DataFrame = None):
    order = impurity.reindex(features).sort_values().index  # ascending, so barh reads largest-on-top
    labels = [FEATURE_LABELS[f] for f in order]
    # Non-significant features (one-sided t-test on the 15 shuffle repeats, p >= 0.05: can't
    # distinguish this feature's importance from zero) get a "n.s." suffix on their label.
    if significance is not None:
        labels = [lbl + ("" if significance.loc[f, "p_value"] < 0.05 else "  (n.s.)")
                  for f, lbl in zip(order, labels)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    fig.suptitle(f"What drives per-turbine {impact_label}: {category_label}",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")

    axes[0].barh(labels, impurity.loc[order].values, color=BLUE, height=0.62)
    axes[0].set_title("Impurity-based (random forest)", fontsize=10.5, color=TEXT_SECONDARY)
    axes[0].set_xlabel("Importance", fontsize=9.5)

    xerr = significance.loc[order, "std"].values if significance is not None else None
    axes[1].barh(labels, permutation.loc[order].values, xerr=xerr, color=ORANGE, height=0.62,
                 ecolor=TEXT_SECONDARY, capsize=2.5, error_kw={"linewidth": 1})
    axes[1].set_title("Permutation (held-out test set, error bars = std across 15 repeats)",
                       fontsize=10.5, color=TEXT_SECONDARY)
    axes[1].set_xlabel("Mean R^2 drop when shuffled", fontsize=9.5)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRAY)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9.5)
        ax.grid(axis="x", color=GRAY, linewidth=0.6, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    fig.text(0.99, 0.01, f"Random forest held-out R² = {r2:.3f}",
              ha="right", fontsize=9, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_known_formula_scatter(onshore_df: pd.DataFrame, offshore_dfs: list[pd.DataFrame]):
    """Two panels tying the top-ranked features back to their known symbolic formulas
    (M_cable_sym scales linearly with dist_to_grid; M_nacelle_sym is quadratic in P).
    Uses the headline GWP100 metric as the illustrative example."""
    offshore_df = pd.concat(offshore_dfs) if offshore_dfs else pd.DataFrame(columns=onshore_df.columns)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for df, color, label in [(onshore_df, BLUE, "Onshore"), (offshore_df, ORANGE, "Offshore")]:
        if df.empty:
            continue
        axes[0].scatter(df["dist_to_grid_m"] / 1000, df[GWP_COL], s=6, alpha=0.35,
                         color=color, label=label, edgecolors="none")
        axes[1].scatter(df["P_rated_kW"], df[GWP_COL], s=6, alpha=0.35,
                         color=color, label=label, edgecolors="none")

    axes[0].set_xlabel("Cable length, dist_to_grid (km)", fontsize=9.5)
    axes[0].set_title("GWP100 vs cable length", fontsize=10.5, color=TEXT_SECONDARY)
    axes[1].set_xlabel("Rated power, P (kW)", fontsize=9.5)
    axes[1].set_title("GWP100 vs rated power", fontsize=10.5, color=TEXT_SECONDARY)

    for ax in axes:
        ax.set_ylabel("GWP100 (kg CO2-Eq/kWh)", fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRAY)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9.5)
        ax.grid(color=GRAY, linewidth=0.6, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.suptitle("The two known formulas behind the top-ranked features (GWP100 example)",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = OUT / "fig_known_formula_scatter.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_heatmap(matrices: dict[str, pd.DataFrame], modeled_categories: list[str], title: str,
                  cbar_label: str, out_path: Path, cmap: str = "YlOrRd", vmin: float = 0,
                  vmax: float = 1, row_height: float = 3.1, ytick_fontsize: float = 8.5):
    """One figure, one subplot per modeled category: rows (already display-labeled by the
    caller) x the 25 impacts, color = value in `matrices[category]`."""
    n = len(modeled_categories)
    fig, axes = plt.subplots(n, 1, figsize=(13, row_height * n + 1.2))
    if n == 1:
        axes = [axes]

    impact_labels = [label for _, label, _ in IMPACTS]
    im = None
    for ax, category in zip(axes, modeled_categories):
        mat = matrices[category].reindex(columns=impact_labels)
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=ytick_fontsize)
        ax.set_xticks(range(len(impact_labels)))
        if ax is axes[-1]:
            ax.set_xticklabels(impact_labels, rotation=60, ha="right", fontsize=7.5)
        else:
            ax.set_xticklabels([])
        ax.set_title(CATEGORY_TITLES[category], fontsize=10.5, color=TEXT_SECONDARY, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.02, ha="left")
    # Reserve the panels' layout FIRST (right=0.84), then place the colorbar in an explicit
    # fixed-position axes to its right. fig.colorbar(im, ax=axes, ...) alone computes the
    # colorbar's position from the axes' bounding boxes at call time; a subsequent
    # subplots_adjust(right=...) then stretches the panels without moving the colorbar,
    # so the two land on top of each other. An explicit cax sidesteps that ordering bug.
    fig.subplots_adjust(left=0.28, right=0.84, top=0.93, bottom=0.22, hspace=0.55)
    pos_top, pos_bot = axes[0].get_position(), axes[-1].get_position()
    cbar_ax = fig.add_axes((0.87, pos_bot.y0, 0.02, pos_top.y1 - pos_bot.y0))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=8.5)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def _rank_features_by_max_importance(perm_matrices: dict[str, pd.DataFrame],
                                      modeled_categories: list[str]) -> list[str]:
    """Every feature that appears in any modeled category, ranked DESCENDING by the largest
    permutation importance it reaches for any impact category in any siting -- i.e. "how
    important can this feature be, at its single best" rather than an average. Shared by
    plot_max_importance_by_type (all features, ascending for barh) and the top-N feature grid
    below (descending, then sliced)."""
    all_features = sorted({f for cat in modeled_categories for f in perm_matrices[cat].index})
    row_max = {f: max(perm_matrices[cat].loc[f].max() for cat in modeled_categories
                       if f in perm_matrices[cat].index) for f in all_features}
    return sorted(all_features, key=lambda f: row_max[f], reverse=True)


def plot_max_importance_by_type(perm_matrices: dict[str, pd.DataFrame],
                                 modeled_categories: list[str], out_path: Path) -> Path:
    """One point per (feature, turbine category): that feature's LARGEST permutation importance
    across all 25 impact categories, labeled with which impact category produced it. Replaces
    the paper draft's placeholder "Example visualization (fake data)" figure with the real
    numbers from perm_matrices (raw, un-normalized permutation importance -- NOT the
    column-normalized `perm_shares` used by the heatmap, since "share of column" and "max
    R^2-drop across columns" answer different questions).

    Only categories in `modeled_categories` are plotted (offshore spar has 2 turbines
    fleet-wide, below MIN_SAMPLES, and was never modeled)."""
    # Ascending so the most-influential features read at the top (barh/scatter convention
    # already used elsewhere in this file).
    ordered_features = _rank_features_by_max_importance(perm_matrices, modeled_categories)[::-1]
    y_pos = {f: i for i, f in enumerate(ordered_features)}
    # Small per-category vertical dodge so near-equal values across categories start out
    # separated -- adjust_text (below) still does the real collision avoidance, but a good
    # starting layout means it has to move points less.
    dodge = {cat: (i - (len(modeled_categories) - 1) / 2) * 0.20
             for i, cat in enumerate(modeled_categories)}

    fig, ax = plt.subplots(figsize=(10, 0.62 * len(ordered_features) + 1.8))

    all_xs, all_ys, texts, used_abbrevs = [], [], [], set()
    for cat in modeled_categories:
        mat = perm_matrices[cat]
        xs, ys, labels = [], [], []
        for f in ordered_features:
            if f not in mat.index:
                continue
            best_impact = mat.loc[f].idxmax()
            xs.append(mat.loc[f, best_impact])
            ys.append(y_pos[f] + dodge[cat])
            abbrev = IMPACT_ABBREV.get(best_impact, best_impact[:3])
            labels.append(abbrev)
            used_abbrevs.add(abbrev)
        ax.scatter(xs, ys, s=75, color=CATEGORY_COLORS[cat], edgecolors="white",
                   linewidths=0.8, zorder=3, label=CATEGORY_TITLES[cat])
        all_xs.extend(xs)
        all_ys.extend(ys)
        for x, y, lbl in zip(xs, ys, labels):
            texts.append(ax.text(x, y, lbl, ha="center", va="center", fontsize=6.5,
                                  color=TEXT_SECONDARY, zorder=4))

    # Real collision avoidance (not just a fixed offset): repels labels from every OTHER label
    # AND from every marker (all_xs/all_ys, all categories at once, not just the label's own
    # point) until nothing overlaps, drawing a short connector line for any label that had to
    # move away from its point.
    adjust_text(texts, x=all_xs, y=all_ys, ax=ax,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6),
                expand=(1.3, 1.6), force_text=(0.3, 0.6), force_points=(0.3, 0.6))

    ax.set_yticks(range(len(ordered_features)))
    ax.set_yticklabels([FEATURE_LABELS[f] for f in ordered_features], fontsize=9.5)
    ax.set_xlabel("Maximum permutation importance across impact categories\n"
                   "(mean R² drop when shuffled, held-out test set)", fontsize=9.5)
    ax.set_title("What drives per-turbine impacts, by feature and siting",
                  fontsize=13, fontweight="bold", x=0.0, ha="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.grid(axis="x", color=GRAY, linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")

    # Abbreviation key: only the impacts that actually appear as a "winning" label somewhere in
    # this figure, not the full 25 -- a legend listing entries nobody can find on the plot is
    # worse than no legend.
    key_entries = sorted((abbrev, full) for full, abbrev in IMPACT_ABBREV.items()
                          if abbrev in used_abbrevs)
    n_cols = 3
    rows_per_col = -(-len(key_entries) // n_cols)  # ceil
    fig.tight_layout()  # auto margins for title/yticks/xlabel first
    fig.subplots_adjust(bottom=0.09 + 0.022 * rows_per_col)  # then carve out room for the key below
    for i, (abbrev, full) in enumerate(key_entries):
        col, row = divmod(i, rows_per_col)
        fig.text(0.02 + col * 0.33, 0.045 + 0.022 * (rows_per_col - 1 - row),
                  f"{abbrev} = {full}", fontsize=7, color=TEXT_SECONDARY, transform=fig.transFigure)

    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_top_features_by_impact_grid(perm_matrices: dict[str, pd.DataFrame],
                                      modeled_categories: list[str], features: list[str],
                                      impact_labels: list[str], out_path: Path) -> Path:
    """One subplot per feature (3x3 grid): x-axis = the given impact categories, y-axis =
    permutation importance, grouped bars colored by turbine siting. Companion to
    plot_max_importance_by_type -- that figure picks each feature's single best impact; this one
    fixes a small set of impacts and shows every feature's importance across all of them side by
    side, so a feature that's merely "pretty good" everywhere isn't hidden by one spike."""
    n = len(features)
    ncols = 3
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.5 * nrows), squeeze=False)
    axes = axes.flatten()

    x = np.arange(len(impact_labels))
    width = 0.8 / len(modeled_categories)
    xticklabels = [IMPACT_ABBREV.get(lbl, lbl[:3]) for lbl in impact_labels]

    # One shared y-axis ceiling across every panel -- with each subplot free-scaling on its own,
    # a feature that barely matters anywhere (max ~0.02) fills its panel just as much as one that
    # dominates (max ~1.3), so a reader's first glance overstates how much is going on. Pinning
    # all panels to the same ceiling means panel "fullness" is directly comparable at a glance.
    all_vals = [perm_matrices[cat].loc[f, impact_labels].values.astype(float)
                for cat in modeled_categories for f in features if f in perm_matrices[cat].index]
    ymax = float(np.concatenate(all_vals).max()) * 1.08 if all_vals else 1.0

    for ax, f in zip(axes, features):
        for i, cat in enumerate(modeled_categories):
            mat = perm_matrices[cat]
            if f not in mat.index:
                continue  # e.g. sea depth has no onshore bar -- onshore's slot just stays empty
            vals = mat.loc[f, impact_labels].values.astype(float)
            offset = (i - (len(modeled_categories) - 1) / 2) * width
            ax.bar(x + offset, vals, width=width * 0.9, color=CATEGORY_COLORS[cat],
                   label=CATEGORY_TITLES[cat], zorder=3)
        ax.set_title(FEATURE_LABELS[f], fontsize=12, color=TEXT_SECONDARY)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, fontsize=10.5)
        ax.set_ylim(0, ymax)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRAY)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
        ax.grid(axis="y", color=GRAY, linewidth=0.6, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    for ax in axes[n:]:
        ax.axis("off")
    for row in range(nrows):
        axes[row * ncols].set_ylabel("Permutation\nimportance", fontsize=10)

    # One legend for the whole grid, built from whichever subplots have categories present -- a
    # subplot missing a category (e.g. sea depth, onshore) would otherwise give an incomplete
    # legend if grabbed from the wrong axes.
    handles_by_label: dict[str, object] = {}
    for ax in axes[:n]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles_by_label[label] = handle
    fig.suptitle("Top features' importance across headline impact categories, by siting",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.legend(handles_by_label.values(), handles_by_label.keys(), loc="lower center",
               ncol=len(modeled_categories), frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.045))
    key_line = "    ".join(f"{IMPACT_ABBREV.get(lbl, lbl[:3])} = {lbl}" for lbl in impact_labels)
    fig.text(0.5, 0.01, key_line, ha="center", fontsize=8, color=TEXT_SECONDARY)

    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def write_markdown(tables: dict[str, pd.DataFrame], summary: dict[str, list[dict]],
                    r2_by_category: dict[str, list[float]], heatmap_path: Path,
                    grouped_heatmap_path: Path, cluster_members: dict[str, list[list[str]]],
                    modeled_categories: list[str], max_importance_path: Path = None,
                    top_features_grid_path: Path = None):
    lines = []
    lines.append("# What actually drives per-turbine impacts, by category and siting")
    lines.append("")
    lines.append(
        "Generated by `generate_feature_importance.py`, across all "
        f"{sum(len(df) for df in tables.values()):,} turbines in the fleet. This is a "
        "descriptive exercise, not a discovery one: `fleet_evaluation_lca_algebraic.py`'s "
        "formulas for how rated power, rotor diameter, hub height, cable length, etc. map to "
        "material mass and each of the 25 EF3.0 impact categories are already known exactly "
        "(the model is symbolic, not a black box). The point here is to confirm, in the actual "
        "fleet data, which of those known drivers matters most, for every impact category and "
        "not just the headline climate metric, and to check whether the answer changes with how "
        "a turbine is sited."
    )
    lines.append("")
    lines.append(
        "Onshore and offshore turbines are modeled **separately**, and offshore is further "
        "split by foundation type (monopile, semi-submersible, spar), rather than mixed "
        "together with an `Offshore` 0/1 feature as in the earlier single-metric version of "
        "this analysis. Sea depth is meaningless onshore (always 0) and foundation type changes "
        "which formulas apply, so pooling categories together was hiding that structure rather "
        "than revealing it."
    )
    lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append(
        "A random forest (300 trees, max depth 12) trained per (turbine category, impact "
        "category) pair, up to 4 x 25 = 100 models, on an 80/20 train/test split. Two rankings "
        "per model, computed two different ways so neither's blind spots go unchecked: the "
        "model's own impurity-based `feature_importances_`, and permutation importance (how "
        "much held-out R² drops when a single feature is shuffled, averaged over 15 repeats). "
        "Permutation is the more trustworthy of the two, since impurity-based importance is "
        "known to inflate high-cardinality continuous features."
    )
    lines.append("")
    onshore_n = len(FEATURES_ONSHORE)
    offshore_n = len(FEATURES_OFFSHORE)
    lines.append(
        f"Onshore uses {onshore_n} features (no sea depth, always 0). Offshore categories use "
        f"{offshore_n} features (adds sea depth). Categories with fewer than {MIN_SAMPLES} "
        "turbines are reported but not modeled: an 80/20 split on a handful of rows doesn't "
        "produce a meaningful ranking."
    )
    lines.append("")

    lines.append("## Multicollinearity: why individual rankings need a second opinion")
    lines.append("")
    lines.append(
        "Permutation importance is known to misattribute credit when predictors are correlated: "
        "shuffling one of two correlated features barely hurts the model, since it can still "
        "read the same signal off the other one, so importance gets split or hidden rather than "
        "measured (Strobl, Boulesteix, Kneib, Augustin & Zeileis, \"Conditional variable "
        "importance for random forests\", *BMC Bioinformatics* 2008). This fleet has real "
        "collinearity: rated power, rotor diameter, hub height, and turbine age correlate at "
        "Spearman |r| up to 0.93 (bigger, newer turbines are simultaneously taller, wider-rotor, "
        "and higher-rated: one underlying trend, four correlated measurements of it)."
    )
    lines.append("")
    lines.append(
        "Mitigation follows scikit-learn's own documented approach for this problem "
        "(*Permutation Importance with Multicollinear or Correlated Features*): cluster "
        "features by hierarchical clustering on Spearman-correlation distance (merge at "
        f"|r| ≥ 0.7, a standard high-correlation cutoff), then shuffle every feature in a "
        "cluster together. This credits the cluster as a whole instead of letting the credit "
        "land arbitrarily on whichever member wins tie-breaking. Both the individual ranking "
        "(heatmap and per-impact figures above) and this clustered ranking are reported; where "
        "they agree, the individual ranking is trustworthy. Where a feature's individual rank is "
        "high but it's clustered with others, read the *cluster*, not that one feature, as the "
        "real finding."
    )
    lines.append("")
    for category in modeled_categories:
        lines.append(f"**{CATEGORY_TITLES[category]} clusters**: " +
                     "; ".join(" + ".join(g) for g in cluster_members[category]) + ".")
        lines.append("")
    lines.append(f"![Clustered feature importance heatmap]({grouped_heatmap_path.relative_to(REPO)})")
    lines.append("")
    lines.append(
        "Each per-impact figure above also marks individually non-significant features "
        "\"(n.s.)\": a one-sided one-sample t-test on the 15 permutation-repeat values per "
        "feature (H1: mean importance > 0), using scikit-learn's own raw per-repeat output "
        "(normally discarded once `.importances_mean` is read). p < 0.05 is treated as "
        "significant; how many of the candidate features clear that bar for each impact is in "
        "the per-impact table below."
    )
    lines.append("")

    lines.append("## Summary: importance across all impact categories, by siting")
    lines.append("")
    lines.append(f"![Feature importance heatmap]({heatmap_path.relative_to(REPO)})")
    lines.append("")
    lines.append(
        "Each row of each panel is a feature, each column an impact category; color is that "
        "feature's permutation importance as a share of the total within that column (columns "
        "sum to 1). Reading across a panel shows whether one or two features dominate every "
        "impact (a mostly-uniform column pattern) or whether the ranking reshuffles "
        "impact-by-impact."
    )
    lines.append("")

    if max_importance_path is not None:
        lines.append("## Summary: each feature's single best impact category, by siting")
        lines.append("")
        lines.append(f"![Max permutation importance by feature and siting]({max_importance_path.relative_to(REPO)})")
        lines.append("")
        lines.append(
            "One point per (feature, turbine category): that feature's LARGEST permutation "
            "importance across all 25 impact categories (raw R² drop, not a column-normalized "
            "share as in the heatmap above), labeled with the impact category that produced it "
            "(abbreviations in the table below). Component masses (blade/nacelle/tower/"
            "foundation) are deterministic functions of rated power, hub height, rotor diameter "
            "and (offshore) sea depth already in the feature list -- see the Multicollinearity "
            "section: where a mass and its generating dimension both rank high, read that as one "
            "finding reported twice, not two independent drivers."
        )
        lines.append("")
        lines.append("| Abbrev. | Impact category | Abbrev. | Impact category |")
        lines.append("|---|---|---|---|")
        abbrev_items = list(IMPACT_ABBREV.items())
        for i in range(0, len(abbrev_items), 2):
            left = f"{abbrev_items[i][1]} | {abbrev_items[i][0]}"
            right = f"{abbrev_items[i+1][1]} | {abbrev_items[i+1][0]}" if i + 1 < len(abbrev_items) else " | "
            lines.append(f"| {left} | {right} |")
        lines.append("")

    if top_features_grid_path is not None:
        lines.append("## Summary: top 9 features across 6 headline impact categories, by siting")
        lines.append("")
        lines.append(f"![Top features across headline impact categories]({top_features_grid_path.relative_to(REPO)})")
        lines.append("")
        lines.append(
            "The 9 features ranked highest by the single-best-impact figure above, now shown "
            "across a fixed set of 6 \"headline\" impact categories (climate change, mineral "
            "resource use, both human toxicity splits, land use, water use) instead of each "
            "feature's one best impact -- so a feature that's consistently solid across several "
            "impacts isn't indistinguishable from one that only spikes on a single narrow metric. "
            "Bars are permutation importance (raw R² drop), grouped by turbine siting; a missing "
            "bar means that feature doesn't apply to that siting (e.g. sea depth has no onshore "
            "bar)."
        )
        lines.append("")

    for category in CATEGORY_ORDER:
        df = tables[category]
        n = len(df)
        lines.append(f"## {CATEGORY_TITLES[category]} (n={n:,})")
        lines.append("")
        if category not in modeled_categories:
            lines.append(
                f"Only {n} turbines in the fleet in this category, below the "
                f"{MIN_SAMPLES}-sample threshold for a meaningful train/test split. Excluded "
                "from modeling; not enough data to rank feature importance."
            )
            lines.append("")
            continue

        r2s = r2_by_category[category]
        lines.append(
            f"Held-out R² across the 25 impact categories ranges "
            f"{min(r2s):.3f}–{max(r2s):.3f} (mean {np.mean(r2s):.3f})."
        )
        lines.append("")
        lines.append("| Impact category | Held-out R² | Top feature (permutation) | Significant (p<0.05) |")
        lines.append("|---|---|---|---|")
        for row in summary[category]:
            lines.append(f"| {row['impact_label']} | {row['r2']:.3f} | {row['top_feature']} | "
                          f"{row['n_significant']}/{row['n_total']} |")
        lines.append("")

        for row in summary[category]:
            fig_rel = row["fig_path"].relative_to(REPO)
            lines.append(f"#### {row['impact_label']}")
            lines.append("")
            lines.append(f"![{row['impact_label']}, {CATEGORY_TITLES[category]}]({fig_rel})")
            lines.append("")

    lines.append("## Why: the known formulas behind rated power, hub height, and rotor diameter")
    lines.append("")
    lines.append(
        "These aren't a black box's guess, they're read straight from "
        "`fleet_evaluation_lca_algebraic.py`:"
    )
    lines.append("")
    lines.append("```")
    lines.append("M_tower_sym   = (3.03584782e-04 * d**2 * h + 9.68652909) * 1e3   # tower steel mass, kg")
    lines.append("M_nacelle_sym = (1.66691134e-06 * P**2 + 3.20700974e-02 * P) * 1e3   # nacelle mass, kg")
    lines.append("M_rotor_sym   = (0.00460956 * d**2 + 0.11199577 * d) * 1e3           # rotor (blades+hub) mass, kg")
    lines.append("```")
    lines.append("")
    lines.append(
        "Hub height (`h`) only ever appears in the tower formula, linearly. Rotor diameter "
        "(`d`) appears in both the tower formula (quadratic) and the rotor formula (quadratic). "
        "Rated power (`P`) appears in the nacelle formula (quadratic) and, separately, sets the "
        "whole turbine's scale almost everywhere else in the model: material percentage splits, "
        "transport formulas, and the cable mass term below all scale with `P` directly or "
        "indirectly. This shows up consistently across most impact categories and siting types "
        "in the tables above, which is exactly what the importance ranking finds independently, "
        "with no knowledge of the formulas going in."
    )
    lines.append("")
    lines.append("Cable length works differently, and is real but genuinely smaller:")
    lines.append("")
    lines.append("```")
    lines.append("M_cable_sym = (300.0 * P / 21516.0) * 1e-6 * dist_to_grid * 8960.0 * (617.0 / 220.0) * 0.5")
    lines.append("```")
    lines.append("")
    lines.append(
        "Cable mass is linear in `dist_to_grid`, but copper cable is a small fraction of a "
        "turbine's total mass next to steel tower, nacelle, and rotor."
    )
    lines.append("")

    lines.append("## The scatter, and why it looks noisy")
    lines.append("")
    lines.append("![GWP100 vs cable length and rated power](figures/feature_importance/fig_known_formula_scatter.png)")
    lines.append("")
    lines.append(
        "Neither panel shows a clean trend line, and that's expected given the importance "
        "ranking, not a contradiction of it: GWP100 per kWh is dominated by rated power and hub "
        "height, so plotting it against any *other* single feature (cable length) or even "
        "against rated power itself, without holding the other two fixed, mixes in that "
        "dominant variance as noise. This is the same relationship already quantified in "
        "`SUMMARY_STATISTICS.md` section 4 (GWP100 vs turbine age, Spearman r = -0.46), showing "
        "up here from a different angle."
    )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        f"This ranks {onshore_n}-{offshore_n} candidate features, not the full input space the "
        "algebraic model actually uses (component-level material percentage splits, park size "
        "interacting with maintenance/transport terms, per-country background electricity "
        "activities). A held-out R² in the ranges reported above means the ranking is "
        "trustworthy for these features on the impacts where R² is high; treat rankings for "
        "impacts with low R² more cautiously, since the model is explaining less of that "
        "impact's variance to begin with. The offshore spar category has only "
        f"{len(tables['offshore_spar'])} turbines fleet-wide and isn't modeled at all."
    )
    lines.append("")

    MD_OUT.write_text("\n".join(lines))
    print(f"\nSaved {MD_OUT}")


def main():
    tables = load_feature_tables()

    # Full per-turbine data: all characteristics + all 25 impacts, one row per turbine. Saved
    # for every category with data, including ones below MIN_SAMPLES (offshore spar) that
    # aren't modeled but were still used/loaded.
    for category in CATEGORY_ORDER:
        df = tables[category]
        if len(df) == 0:
            continue
        cat_dir = OUT / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cat_dir / "per_turbine_data.csv")

    modeled_categories = [c for c in CATEGORY_ORDER if len(tables[c]) >= MIN_SAMPLES]

    summary: dict[str, list[dict]] = {c: [] for c in CATEGORY_ORDER}
    r2_by_category: dict[str, list[float]] = {c: [] for c in modeled_categories}
    perm_shares: dict[str, pd.DataFrame] = {}
    perm_matrices_raw: dict[str, pd.DataFrame] = {}
    grouped_shares: dict[str, pd.DataFrame] = {}
    cluster_members: dict[str, list[list[str]]] = {}

    for category in modeled_categories:
        df = tables[category]
        features = FEATURES_ONSHORE if category == "onshore" else FEATURES_OFFSHORE
        cat_dir = OUT / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {CATEGORY_TITLES[category]} (n={len(df)}) ===")

        # Cluster near-collinear features (Spearman |r| >= 0.7) once per category -- see
        # importance_stats.py. Correlated features bias BOTH impurity and permutation
        # importance; shuffling a whole cluster together (grouped_permutation_importance) is
        # the documented mitigation, reported alongside (not instead of) the individual ranking.
        clusters = compute_clusters(df, features)  # {raw joined label: [raw members]}
        cluster_label_map = {raw: wrap_cluster_label([FEATURE_LABELS[c] for c in members])
                              for raw, members in clusters.items()}
        display_index = list(cluster_label_map.values())
        cluster_members[category] = [[FEATURE_LABELS[c] for c in members] for members in clusters.values()]
        pd.DataFrame({"cluster_members": [", ".join(g) for g in cluster_members[category]]}
                     ).to_csv(cat_dir / "feature_clusters.csv", index=False)

        perm_matrix = pd.DataFrame(index=features, dtype=float)
        grouped_matrix = pd.DataFrame(index=display_index, dtype=float)
        significance_matrix = pd.DataFrame(index=features, dtype=float)
        for full_col, short_label, slug in IMPACTS:
            impurity, permutation, r2, significance, grouped = train_and_rank(
                df, features, full_col, clusters)
            out_path = cat_dir / f"fig_feature_importance_{slug}.png"
            plot_importances(impurity, permutation, r2, features, short_label,
                              CATEGORY_TITLES[category], out_path, significance)

            perm_matrix[short_label] = permutation.reindex(features)
            grouped_matrix[short_label] = grouped.rename(index=cluster_label_map).reindex(display_index)
            significance_matrix[short_label] = significance["p_value"].reindex(features)
            r2_by_category[category].append(r2)
            top_feature = FEATURE_LABELS[permutation.idxmax()]
            n_sig = int((significance["p_value"] < 0.05).sum())
            summary[category].append({
                "impact_label": short_label, "r2": r2, "top_feature": top_feature,
                "fig_path": out_path, "n_significant": n_sig, "n_total": len(features),
            })
            print(f"  {short_label}: R^2={r2:.3f}, top={top_feature}, "
                  f"significant={n_sig}/{len(features)}")

        # normalize each column (impact) so permutation importances sum to 1 -> comparable share
        perm_shares[category] = perm_matrix.div(perm_matrix.sum(axis=0), axis=1)
        perm_matrices_raw[category] = perm_matrix.copy()  # un-normalized, for the max-across-impacts plot below
        grouped_shares[category] = grouped_matrix.div(grouped_matrix.sum(axis=0), axis=1)
        # Cache the raw matrices so a plot-only fix (e.g. colorbar layout) never needs a full
        # RF+permutation retrain again -- that's the expensive part, not the plotting.
        perm_matrix.to_csv(cat_dir / "permutation_importance_matrix.csv")
        grouped_matrix.to_csv(cat_dir / "grouped_permutation_importance_matrix.csv")
        significance_matrix.to_csv(cat_dir / "permutation_pvalue_matrix.csv")

    perm_shares_display = {c: mat.rename(index=FEATURE_LABELS) for c, mat in perm_shares.items()}
    heatmap_path = plot_heatmap(
        perm_shares_display, modeled_categories,
        "Feature importance share across all impact categories, by turbine siting",
        "Share of permutation importance\n(within each impact column)",
        OUT / "fig_importance_heatmap.png")

    grouped_heatmap_path = plot_heatmap(
        grouped_shares, modeled_categories,
        "Clustered importance share (correlated features shuffled together as one unit)",
        "Share of grouped permutation importance\n(within each impact column)",
        OUT / "fig_grouped_importance_heatmap.png", row_height=3.6, ytick_fontsize=7.5)

    offshore_dfs = [tables[c] for c in ["offshore_monopile", "offshore_semi-submersible", "offshore_spar"]
                    if len(tables[c]) > 0]
    plot_known_formula_scatter(tables["onshore"], offshore_dfs)

    max_importance_path = plot_max_importance_by_type(
        perm_matrices_raw, modeled_categories, OUT / "fig_max_importance_by_siting.png")

    top9_features = _order_features_by_group(
        _rank_features_by_max_importance(perm_matrices_raw, modeled_categories)[:9])
    top_features_grid_path = plot_top_features_by_impact_grid(
        perm_matrices_raw, modeled_categories, top9_features, HEADLINE_IMPACTS,
        OUT / "fig_top_features_by_impact_grid.png")

    write_markdown(tables, summary, r2_by_category, heatmap_path, grouped_heatmap_path,
                    cluster_members, modeled_categories, max_importance_path, top_features_grid_path)


if __name__ == "__main__":
    main()
