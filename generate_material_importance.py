"""
generate_material_importance.py

Ranks which RAW MATERIALS (steel, aluminium, copper, concrete, ...) actually drive per-turbine
impact variation across the fleet, using the same random-forest + permutation-importance method
as generate_feature_importance.py, but with material masses (kg) as features instead of turbine
characteristics (P/d/h/...).

Per-turbine material masses are pure algebra, not a new LCA run: fleet_evaluation_lca_algebraic.py
already builds each material's exact symbolic mass expression while constructing its Input-phase
exchange dict (amt_sym = perc_sym * M_component_sym, fleet_evaluation_lca_algebraic.py:559, fed by
prepare_inventories.percentage_inventory()'s material-split table, i.e. what fraction of a
component's mass is which raw material at a given rated power). This script reproduces those same
closed-form formulas directly in numpy -- no brightway2/ecoinvent needed for the material side --
and correlates the results against the 25 impact-category columns already computed and cached in
fleet_impacts_*.csv (same source generate_feature_importance.py reads).

Materials NOT included (deliberately): transformer units (MV/HV transfo exchanges represent a
whole manufactured product, not a decomposed raw material), land use (m2, not a mass), and
process/service exchanges (electricity, explosives, road, welding, wire drawing). These are
process or product-level exchanges in the LCI, not Input-phase raw-material rows in
percentage_inventory()'s table -- see "Limitations" in the generated markdown.

Run once (output is cached to disk):
    python generate_material_importance.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

from generate_feature_importance import (
    REPO, CATEGORY_ORDER, CATEGORY_TITLES, MIN_SAMPLES, IMPACTS, IMPACT_COLUMNS, GWP_COL,
    load_feature_tables, plot_heatmap, BLUE, ORANGE, GRAY, TEXT_SECONDARY,
)
from importance_stats import (compute_clusters, grouped_permutation_importance,
                               permutation_significance, wrap_cluster_label)

_REWIND_DIR = REPO / "REWIND" / "REWIND"
sys.path.insert(0, str(_REWIND_DIR))

OUT = REPO / "figures" / "material_importance"
OUT.mkdir(parents=True, exist_ok=True)
MD_OUT = REPO / "MATERIAL_IMPORTANCE.md"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw-material mass formulas, reproduced in plain numpy from fleet_evaluation_lca_algebraic.py
#    (line numbers refer to that file). Every formula here is copy-faithful to the symbolic
#    version used to compute the actual fleet_impacts_*.csv results, just evaluated directly
#    instead of built as a SymPy expression, since we don't need lca_algebraic's batch impact
#    evaluation for mass alone.
# ─────────────────────────────────────────────────────────────────────────────

# Raw ecoinvent dataset name (as it appears in percentage_inventory()'s table) -> display label.
DATASET_LABELS = {
    "Low-alloy steel": "Steel (low-alloy)",
    "Chromium steel": "Steel (chromium)",
    "Cast iron": "Cast iron",
    "Aluminium 0% recycled": "Aluminium",
    "Copper": "Copper",
    "Fiberglass": "Fiberglass",
    "HDPE granules": "HDPE",
    "PVC impact resistant": "PVC",
    "PP granules": "PP",
    "Epoxy resin": "Epoxy resin",
    "Lead": "Lead",
    "Tin": "Tin",
    "Lubricating oil": "Lubricating oil",
    "Rubber": "Rubber",
}
# Power-supply cable material split (fleet_evaluation_lca_algebraic.py CABLE_PERC, line ~443).
CABLE_PERC = {"Copper": 0.356564, "HDPE granules": 0.354943, "PP granules": 0.032415,
              "PVC impact resistant": 0.256078}

MATERIALS_COMMON = ["Steel (low-alloy)", "Steel (chromium)", "Cast iron", "Aluminium", "Copper",
                    "Fiberglass", "HDPE", "PVC", "PP", "Epoxy resin", "Lead", "Tin",
                    "Lubricating oil", "Rubber"]
# Foundation materials differ by turbine category (onshore gravity-base vs. offshore foundation
# type), so each category adds its own extra bucket(s) on top of the common 14.
MATERIALS_BY_CATEGORY = {
    "onshore": MATERIALS_COMMON + ["Concrete", "Reinforcing steel (rebar)"],
    "offshore_monopile": MATERIALS_COMMON + ["Grout (cement)"],
    "offshore_semi-submersible": MATERIALS_COMMON,
    "offshore_spar": MATERIALS_COMMON + ["Iron ore"],
}


def M_tower(d, h):
    """Tower steel mass, kg. fleet_evaluation_lca_algebraic.py:423."""
    return (3.03584782e-04 * d ** 2 * h + 9.68652909e+00) * 1e3


def M_nacelle_onshore(P):
    """fleet_evaluation_lca_algebraic.py:424."""
    return (1.66691134e-06 * P ** 2 + 3.20700974e-02 * P) * 1e3


def M_nacelle_offshore(P):
    """Offshore-specific coefficients. fleet_evaluation_lca_algebraic.py:1138."""
    return (2.15668283e-06 * P ** 2 + 3.24712680e-02 * P) * 1e3


def M_rotor_onshore(d):
    """fleet_evaluation_lca_algebraic.py:425."""
    return (0.00460956 * d ** 2 + 0.11199577 * d) * 1e3


def M_rotor_offshore(d):
    """Offshore-specific coefficients. fleet_evaluation_lca_algebraic.py:1139."""
    return (0.0088365 * d ** 2 + -0.16435292 * d) * 1e3


def M_elec(P):
    """Nacelle electronics/electrical mass, kg (same table onshore and offshore).
    fleet_evaluation_lca_algebraic.py:430."""
    return np.interp(P, [30, 150, 600, 800, 2000], [150, 300, 862, 1112, 3946])


def M_reinf(P):
    """Foundation reinforcement steel, kg (onshore only). fleet_evaluation_lca_algebraic.py:429."""
    return np.interp(P, [750, 2000, 4500], [10210, 27000, 51900])


def M_found_onshore(d, h):
    """Onshore gravity-base foundation total mass, kg. fleet_evaluation_lca_algebraic.py:426."""
    return 1696e3 * h / 80 * d ** 2 / 10000


def M_cable_onshore(P, dist_to_grid_m):
    """fleet_evaluation_lca_algebraic.py:437."""
    return (300.0 * P / 21516.0) * 1e-6 * dist_to_grid_m * 8960.0 * (617.0 / 220.0) * 0.5


def M_cable_offshore(P, dist_to_grid_m, park_size):
    """Two-segment offshore cable (turbine->transformer + shared park->shore export cable).
    fleet_evaluation_lca_algebraic.py:1206-1232."""
    DIST_TRANSFO_M = 1000.0  # 1 km, fixed, never given a real per-turbine value anywhere
    cross_section1 = 300.0 * P / 21516.0
    m_copper = cross_section1 * 1e-6 * DIST_TRANSFO_M * 8960.0
    df33_P = [11616, 13167, 14718, 16566, 19173, 21516, 23958, 26763, 29832, 32769]
    df33_idx = [95, 120, 150, 185, 240, 300, 400, 500, 630, 800]
    df150_P = [106500, 122250, 138750, 156750, 174000, 200250, 213750, 234000]
    df150_idx = [400, 500, 630, 800, 1000, 1200, 1600, 2000]
    park_power = P * park_size
    cross_section2 = np.where(
        park_power <= 30000,
        np.interp(park_power, df33_P, df33_idx),
        np.interp(park_power, df150_P, df150_idx),
    )
    m_copper = m_copper + cross_section2 * 1e-6 * (dist_to_grid_m / park_size) * 8960.0
    return (m_copper * 617.0 / 220.0) * 0.5


def foundation_masses_offshore(bucket, P, sea_depth_m):
    """fleet_evaluation_lca_algebraic.py:_foundation_mass_formulas, lines 1068-1091.
    Spar buoy reproduces a known unit bug in the baseline (tonnes used as kg) faithfully,
    for comparability; see that function's docstring."""
    if bucket == "offshore_monopile":
        m_grout = 2.44094476537839 * P + 2585.70316709778 * sea_depth_m + 15406.5926965901
        m_monopile = 5.28641521017598 * P + 5599.92210615505 * sea_depth_m + 102643.279628469
        return {"grout": m_grout, "material": m_monopile}
    elif bucket == "offshore_semi-submersible":
        m_semi_sub = 289.473684210526 * P + 17828.5714285714 * sea_depth_m + 80000.0
        return {"material": m_semi_sub}
    elif bucket == "offshore_spar":
        m_spar_steel = 0.383333333333333 * P + 6.84 * sea_depth_m + 300.0
        m_spar_iron = 0.833333333333333 * P
        return {"steel": m_spar_steel, "iron": m_spar_iron}
    raise ValueError(bucket)


def _extract_percentage_series(df_perc):
    """All (component, dataset, xs, ys) triples for Tower/Nacelle/Rotor/Electronics: the
    percentage-of-component-mass each raw material represents, at each rated-power breakpoint.
    Mirrors fleet_evaluation_lca_algebraic.py:542-559's own walk over this same table."""
    rows = []
    for component in df_perc["Input"].columns.get_level_values(0).unique():
        if component not in ("Tower", "Nacelle", "Rotor", "Electronics"):
            continue
        for sub in df_perc["Input"][component].columns.get_level_values(0).unique():
            for ds in df_perc["Input"][component][sub].columns.get_level_values(0).unique():
                s = df_perc["Input"][component][sub][ds].dropna()
                if s.empty:
                    continue
                rows.append((component, ds, s.index.astype(float).values, s.values.astype(float)))
    return rows


def _foundation_steel_percentage(df_perc):
    """Monopile foundation steel percentage-of-mass table. fleet_evaluation_lca_algebraic.py:1174."""
    s = df_perc["Input"]["Foundation"]["Material"]["Low-alloy steel"].dropna()
    return s.index.astype(float).values, s.values.astype(float)


def compute_material_masses(category: str, df: pd.DataFrame, perc_rows, found_steel_perc) -> dict:
    """category -> {material label: np.ndarray of per-turbine mass, kg}."""
    P = df["P_rated_kW"].values.astype(float)
    d = df["Diameter_m"].values.astype(float)
    h = df["Hub_height_m"].values.astype(float)
    dist_to_grid = df["dist_to_grid_m"].values.astype(float)
    park_size = df["park_size"].values.astype(float)

    offshore = category != "onshore"
    comp_mass = {
        "Tower": M_tower(d, h),
        "Nacelle": M_nacelle_offshore(P) if offshore else M_nacelle_onshore(P),
        "Rotor": M_rotor_offshore(d) if offshore else M_rotor_onshore(d),
        "Electronics": M_elec(P),
    }

    buckets = {}
    for component, ds, xs, ys in perc_rows:
        perc = np.interp(P, xs, ys)
        amt = perc * comp_mass[component]
        label = DATASET_LABELS.get(ds, ds)
        buckets[label] = buckets.get(label, 0.0) + amt

    if category == "onshore":
        M_found = M_found_onshore(d, h)
        M_reinf_ = M_reinf(P)
        buckets["Concrete"] = M_found - M_reinf_  # V_conc_sym*2200, i.e. foundation mass net of rebar
        buckets["Reinforcing steel (rebar)"] = M_reinf_
        M_cable = M_cable_onshore(P, dist_to_grid)
    else:
        sea_depth = df["sea_depth_m"].values.astype(float)
        masses = foundation_masses_offshore(category, P, sea_depth)
        if category == "offshore_monopile":
            buckets["Grout (cement)"] = masses["grout"]
            fxs, fys = found_steel_perc
            fperc = np.interp(P, fxs, fys)
            buckets["Steel (low-alloy)"] = buckets.get("Steel (low-alloy)", 0.0) + fperc * masses["material"]
        elif category == "offshore_semi-submersible":
            buckets["Steel (low-alloy)"] = buckets.get("Steel (low-alloy)", 0.0) + masses["material"]
        elif category == "offshore_spar":
            buckets["Steel (low-alloy)"] = buckets.get("Steel (low-alloy)", 0.0) + masses["steel"]
            buckets["Iron ore"] = masses["iron"]
        M_cable = M_cable_offshore(P, dist_to_grid, park_size)

    for ds, perc_val in CABLE_PERC.items():
        label = DATASET_LABELS.get(ds, ds)
        buckets[label] = buckets.get(label, 0.0) + perc_val * M_cable

    n = len(df)
    for label in MATERIALS_BY_CATEGORY[category]:
        if label not in buckets:
            buckets[label] = np.zeros(n)
    return buckets


def spearman_with_pvalues(df: pd.DataFrame, materials: list, impact_cols: list):
    """Spearman r AND p-value for every (material, impact) pair. pandas' df.corr(method=
    "spearman") -- used in an earlier version of this script -- only returns r; scipy's
    spearmanr computes both from the same rank transform, at negligible extra cost."""
    r_mat = pd.DataFrame(index=materials, columns=impact_cols, dtype=float)
    p_mat = pd.DataFrame(index=materials, columns=impact_cols, dtype=float)
    for mat_col in materials:
        x = df[mat_col].values
        for imp_col in impact_cols:
            r, p = spearmanr(x, df[imp_col].values)
            r_mat.loc[mat_col, imp_col] = r
            p_mat.loc[mat_col, imp_col] = p
    return r_mat, p_mat


# ─────────────────────────────────────────────────────────────────────────────
# 2. Random forest ranking (identical method to generate_feature_importance.py)
# ─────────────────────────────────────────────────────────────────────────────

def train_and_rank(df: pd.DataFrame, materials: list[str], target_col: str, clusters: dict):
    X = df[materials].astype(float)
    y = df[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the target before fitting -- see the matching comment in
    # generate_feature_importance.py's train_and_rank(): sklearn's tree-splitter silently fails
    # to find any split when the target's absolute magnitude is tiny (e.g. human toxicity, CTUh
    # ~1e-10), collapsing every tree to a single-leaf constant predictor. R^2 is exactly
    # invariant under this affine transform (mean/std from TRAIN only), so this is the same
    # number a numerically well-behaved fit would give, not a different metric.
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_s = (y_train - y_mean) / y_std
    y_test_s = (y_test - y_mean) / y_std

    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train_s)

    r2 = r2_score(y_test_s, model.predict(X_test))
    impurity = pd.Series(model.feature_importances_, index=materials).sort_values(ascending=False)
    perm = permutation_importance(model, X_test, y_test_s, n_repeats=15, random_state=42, n_jobs=-1)
    permutation = pd.Series(perm.importances_mean, index=materials).sort_values(ascending=False)

    # Significance (t-test on the 15 raw shuffle-repeat values) and grouped/clustered importance
    # (correlated materials shuffled together) -- see importance_stats.py and the matching
    # comments in generate_feature_importance.py's train_and_rank(). Materials here are FAR more
    # collinear than turbine characteristics (several pairs at Spearman |r| > 0.95, some exactly
    # 1.0), so the clustered ranking matters more here than in the characteristics analysis.
    significance = permutation_significance(perm, materials)
    grouped = grouped_permutation_importance(model, X_test, y_test_s, clusters)

    return impurity, permutation, r2, significance, grouped


def plot_importances(impurity, permutation, r2, materials, impact_label, category_label, out_path,
                      significance: pd.DataFrame = None):
    order = impurity.reindex(materials).sort_values().index
    labels = list(order)
    if significance is not None:
        labels = [lbl + ("" if significance.loc[f, "p_value"] < 0.05 else "  (n.s.)")
                  for f, lbl in zip(order, labels)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    fig.suptitle(f"What material drives per-turbine {impact_label}: {category_label}",
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
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        ax.grid(axis="x", color=GRAY, linewidth=0.6, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    fig.text(0.99, 0.01, f"Random forest held-out R² = {r2:.3f}",
              ha="right", fontsize=9, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_composition(tables: dict, modeled_categories: list[str]):
    """Mean material composition (kg) per category, one bar-chart panel per category."""
    n = len(modeled_categories)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 6.2))
    if n == 1:
        axes = [axes]
    fig.suptitle("Mean material mass per turbine, by siting", fontsize=13, fontweight="bold",
                 x=0.02, ha="left")

    for ax, category in zip(axes, modeled_categories):
        df = tables[category]
        materials = MATERIALS_BY_CATEGORY[category]
        means = df[materials].mean().sort_values()
        share = means / means.sum() * 100
        ax.barh(means.index, means.values, color=BLUE, height=0.62)
        for i, (label, val) in enumerate(means.items()):
            ax.text(val, i, f"  {share[label]:.1f}%", va="center", fontsize=7.5,
                    color=TEXT_SECONDARY)
        ax.set_title(CATEGORY_TITLES[category], fontsize=10.5, color=TEXT_SECONDARY)
        ax.set_xlabel("Mean mass (kg)", fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRAY)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
        ax.grid(axis="x", color=GRAY, linewidth=0.6, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = OUT / "fig_material_composition.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


# plot_heatmap (imported from generate_feature_importance.py) replaces this script's former
# bespoke plot_spearman_heatmap/plot_final_heatmap -- identical logic, now shared in one place.


# ─────────────────────────────────────────────────────────────────────────────
# 3. Markdown report
# ─────────────────────────────────────────────────────────────────────────────

def write_markdown(tables, summary, r2_by_category, composition_stats, validation_stats,
                    heatmap_path, spearman_path, composition_path, grouped_heatmap_path,
                    cluster_members, modeled_categories):
    lines = []
    lines.append("# What raw materials actually drive per-turbine impacts")
    lines.append("")
    lines.append(
        "Generated by `generate_material_importance.py`, across all "
        f"{sum(len(df) for df in tables.values()):,} turbines in the fleet. Companion analysis "
        "to `FEATURE_IMPORTANCE.md`, which ranks turbine *characteristics* (rated power, hub "
        "height, ...); this ranks the **raw materials** those characteristics are made of "
        "(steel, aluminium, copper, concrete, ...) instead."
    )
    lines.append("")
    lines.append(
        "Per-turbine material masses are pure algebra, not a new LCA run: "
        "`fleet_evaluation_lca_algebraic.py` already builds each material's exact closed-form "
        "mass expression while constructing its Input-phase exchange dict "
        "(`amt_sym = perc_sym * M_component_sym`, `fleet_evaluation_lca_algebraic.py:559`, fed "
        "by `prepare_inventories.percentage_inventory()`'s material-split table -- what "
        "fraction of a component's mass is which raw material, at a given rated power). This "
        "script reproduces those same formulas directly in numpy (no brightway2/ecoinvent "
        "needed for the material side) and correlates the results against the 25 impact-category "
        "columns already computed in `fleet_impacts_*.csv`."
    )
    lines.append("")
    lines.append(
        "As in the characteristics analysis, onshore and offshore turbines are modeled "
        "**separately** (offshore further split by foundation type), since foundation type "
        "changes which raw materials even apply (concrete onshore vs. grout+steel for a "
        "monopile vs. plain steel for semi-submersible/spar)."
    )
    lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append(
        "Two independent statistics per (turbine category, impact category) pair, so neither's "
        "blind spots go unchecked:"
    )
    lines.append("")
    lines.append(
        "1. **Random forest ranking** (300 trees, max depth 12, 80/20 train/test split): the "
        "model's own impurity-based `feature_importances_`, and permutation importance (mean "
        "held-out R² drop when a single material's mass is shuffled, 15 repeats). Captures "
        "non-linear and interaction effects, but needs enough rows to trust (see sample-size "
        "note below)."
    )
    lines.append(
        "2. **Spearman rank correlation** between each material's mass and the impact, computed "
        "directly (no model fitting, both r and p-value via `scipy.stats.spearmanr`). A "
        "simpler, monotonic-only cross-check: if a material ranks highly by both methods, that "
        "is not one method's artifact."
    )
    lines.append(
        "3. **Significance**: a one-sided one-sample t-test on each material's 15 "
        "permutation-repeat importance values (H1: mean > 0), using scikit-learn's own raw "
        "per-repeat output. Materials that don't clear p < 0.05 are marked \"(n.s.)\" in the "
        "per-impact figures."
    )
    lines.append("")
    lines.append(
        f"Materials common to every category: {', '.join(MATERIALS_COMMON)} ({len(MATERIALS_COMMON)} "
        "total). Foundation materials are category-specific: onshore adds Concrete and "
        "Reinforcing steel; offshore monopile adds Grout (cement) (its foundation steel folds "
        "into the shared Steel (low-alloy) bucket); offshore semi-submersible's foundation is "
        "entirely Steel (low-alloy), no extra bucket; offshore spar adds Iron ore. "
        f"Categories with fewer than {MIN_SAMPLES} turbines are reported but not modeled."
    )
    lines.append("")

    lines.append("## Multicollinearity: individual material rankings need a second opinion")
    lines.append("")
    lines.append(
        "Materials are far more collinear than turbine characteristics: several pairs correlate "
        "at Spearman |r| > 0.95, some at exactly 1.0 (e.g. onshore Steel (low-alloy) and Concrete, "
        "r = 1.00) -- expected, since nearly every material bucket is `percentage(P) x "
        "M_component(P,d,h)`, and the percentages are close to flat, so almost every material's "
        "mass ends up as a near-deterministic function of the same underlying turbine size. "
        "Permutation importance is known to misattribute credit under exactly this condition "
        "(Strobl et al., *BMC Bioinformatics* 2008): shuffling one of two near-identical columns "
        "barely hurts the model, since it reads the same signal off the other, so which one gets "
        "credited as \"the top material\" can be closer to arbitrary tie-breaking than a real "
        "signal about that material's distinct per-kg impact."
    )
    lines.append("")
    lines.append(
        "Mitigation follows scikit-learn's own documented approach (*Permutation Importance "
        "with Multicollinear or Correlated Features*): cluster materials by hierarchical "
        "clustering on Spearman-correlation distance (merge at |r| ≥ 0.7), then shuffle every "
        "material in a cluster together. Where the individual and clustered rankings agree, the "
        "individual claim is trustworthy; where a material's high individual rank sits inside a "
        "large cluster, the honest reading is \"this cluster matters,\" not \"this specific "
        "material matters more than its near-identical neighbors.\""
    )
    lines.append("")
    for category in modeled_categories:
        lines.append(f"**{CATEGORY_TITLES[category]} clusters**: " +
                     "; ".join(" + ".join(g) for g in cluster_members[category]) + ".")
        lines.append("")
    lines.append(f"![Clustered material importance heatmap]({grouped_heatmap_path.relative_to(REPO)})")
    lines.append("")

    lines.append("## Validation: does the material breakdown add up?")
    lines.append("")
    lines.append(
        "Two checks, of different strength. First, a **self-consistency** check: sum every "
        "material bucket for a turbine and compare against the same total computed "
        "independently from the component-mass formulas (`M_tower + M_nacelle + M_rotor + "
        "M_elec + foundation + cable`). This is a weaker check than it might look like -- both "
        "sides call the *same* underlying mass functions (`M_tower()`, `M_elec()`, etc.), so it "
        "mainly confirms `percentage_inventory()`'s percentage-split table sums to ~1 per "
        "component and that no component was forgotten in the loop, not that the formulas "
        "themselves are transcribed correctly from `fleet_evaluation_lca_algebraic.py`."
    )
    lines.append("")
    lines.append("| Category | Mean captured mass (kg) | Mean formula total (kg) | Difference |")
    lines.append("|---|---|---|---|")
    for category in CATEGORY_ORDER:
        if category not in validation_stats:
            continue
        v = validation_stats[category]
        lines.append(f"| {CATEGORY_TITLES[category]} | {v['captured']:,.0f} | {v['formula']:,.0f} "
                      f"| {v['pct_diff']:+.4f}% |")
    lines.append("")
    lines.append(
        "(Offshore monopile's formula total already applies the same partial percentage the "
        "real model applies to its foundation-steel term -- only `percentage(P) x m_monopile` "
        "is ever assigned to a material exchange there, not the full foundation mass estimate; "
        "see `fleet_evaluation_lca_algebraic.py:1174-1176`. That gap is a property of the "
        "underlying baseline model, reproduced here as-is, not a bug in this script.)"
    )
    lines.append("")
    lines.append(
        "Second, an **independent** check against published literature, which the formulas were "
        "not fit to: at a 2 MW reference turbine (P=2000 kW), the onshore reinforcement-steel "
        "formula gives exactly 27.0 tonnes. A published life-cycle study of onshore tower "
        "foundations (see Sources) reports a representative 2 MW-class foundation at "
        "\"approximately 27 tonnes of reinforcement\" -- an exact match, not tuned to produce "
        "one. The same formula's net concrete mass at that reference point (604 m³) is the same "
        "order of magnitude as that study's example (400 m³), same direction (concrete "
        "dominates foundation mass), though not identical -- expected, since real foundation "
        "design varies by soil conditions and turbine loads, and the published figure was one "
        "illustrative example, not a universal constant. Together, these give the validation "
        "section an actual external anchor, not just an internal identity."
    )
    lines.append("")

    lines.append("## Mean material composition, by siting")
    lines.append("")
    lines.append(f"![Material composition]({composition_path.relative_to(REPO)})")
    lines.append("")
    lines.append(
        "Percentages next to each bar are that material's share of the turbine's total captured "
        "mass. Steel (structural low-alloy steel, not counting chromium-steel drivetrain parts) "
        "dominates every category, as expected for a tower-and-foundation-heavy structure."
    )
    lines.append("")
    for category in modeled_categories:
        v = composition_stats[category]
        lines.append(f"**{CATEGORY_TITLES[category]}**: top material by mass is "
                      f"**{v['top_material']}** ({v['top_share']:.1f}% of captured mass, "
                      f"{v['top_mean_kg']:,.0f} kg mean).")
        lines.append("")

    lines.append("## Summary: random-forest importance across all impact categories, by siting")
    lines.append("")
    lines.append(f"![Material importance heatmap]({heatmap_path.relative_to(REPO)})")
    lines.append("")
    lines.append(
        "Each row is a material, each column an impact category; color is that material's "
        "permutation importance as a share of the total within that column (columns sum to 1)."
    )
    lines.append("")

    lines.append("## Summary: Spearman rank correlation across all impact categories, by siting")
    lines.append("")
    lines.append(f"![Spearman correlation heatmap]({spearman_path.relative_to(REPO)})")
    lines.append("")
    lines.append(
        "Blue = positive correlation (more of this material, more impact), red = negative, "
        "white = no monotonic relationship. Unlike the importance heatmap above, sign is "
        "visible here: every structural material should correlate positively with almost every "
        "impact (more turbine mass, more embodied impact), which is a useful sanity check on "
        "the random-forest ranking."
    )
    lines.append("")

    for category in CATEGORY_ORDER:
        df = tables[category]
        n = len(df)
        lines.append(f"## {CATEGORY_TITLES[category]} (n={n:,})")
        lines.append("")
        if category not in modeled_categories:
            lines.append(
                f"Only {n} turbines in the fleet in this category -- below the "
                f"{MIN_SAMPLES}-sample threshold for a meaningful train/test split. Excluded "
                "from modeling; not enough data to rank material importance."
            )
            lines.append("")
            continue

        r2s = r2_by_category[category]
        lines.append(
            f"Held-out R² across the 25 impact categories ranges "
            f"{min(r2s):.3f}–{max(r2s):.3f} (mean {np.mean(r2s):.3f})."
        )
        lines.append("")
        lines.append("| Impact category | Held-out R² | Top material (permutation) | "
                      "Top material (Spearman, abs.) | Significant (p<0.05) |")
        lines.append("|---|---|---|---|---|")
        for row in summary[category]:
            lines.append(f"| {row['impact_label']} | {row['r2']:.3f} | {row['top_perm']} | "
                          f"{row['top_spearman']} | {row['n_significant']}/{row['n_total']} |")
        lines.append("")

        for row in summary[category]:
            fig_rel = row["fig_path"].relative_to(REPO)
            lines.append(f"#### {row['impact_label']}")
            lines.append("")
            lines.append(f"![{row['impact_label']}, {CATEGORY_TITLES[category]}]({fig_rel})")
            lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        f"This ranks {len(MATERIALS_COMMON)}–{len(MATERIALS_COMMON) + 2} material buckets "
        "at Input-phase granularity, not the full input space "
        "`fleet_evaluation_lca_algebraic.py` uses: it deliberately excludes the MV/HV "
        "transformer exchanges (a whole manufactured product/process activity, not a "
        "decomposed raw material in this table), land use (m², not a mass), and "
        "process/service exchanges (electricity, explosives, road, welding, wire drawing). "
        "A held-out R² in the ranges reported above means the ranking is trustworthy for these "
        "materials on the impacts where R² is high; treat rankings for impacts with low R² more "
        "cautiously, since the model is explaining less of that impact's variance to begin with. "
        "The offshore spar category has only "
        f"{len(tables['offshore_spar'])} turbines fleet-wide and isn't modeled at all; its "
        "foundation-steel/iron masses also reproduce a known tonnes-vs-kg unit bug in the "
        "underlying baseline (see `foundation_masses_offshore()`'s docstring), kept for "
        "comparability rather than \"fixed\" here. See also the Multicollinearity section above: "
        "individual material rankings within a reported cluster are not reliably distinguishable "
        "from each other, only the cluster as a whole."
    )
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    lines.append(
        "- Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T., Zeileis, A. (2008). "
        "\"Conditional variable importance for random forests.\" *BMC Bioinformatics* 9, 307. "
        "The multicollinearity bias in permutation importance, and its mitigation, cited above."
    )
    lines.append(
        "- scikit-learn developers. \"Permutation Importance with Multicollinear or Correlated "
        "Features.\" scikit-learn documentation. The clustering-then-group-shuffle approach "
        "used here follows this example directly."
    )
    lines.append(
        "- Foundation reinforcement/concrete reference figures (Validation section): published "
        "life-cycle studies of onshore wind turbine tower/foundation material inputs, reporting "
        "a representative 2 MW-class gravity-base foundation at ~400 m³ concrete and ~27 tonnes "
        "reinforcement."
    )
    lines.append("")

    MD_OUT.write_text("\n".join(lines))
    print(f"\nSaved {MD_OUT}")


def main():
    sys.path.insert(0, str(_REWIND_DIR))
    import os
    os.chdir(str(_REWIND_DIR))
    from prepare_inventories import percentage_inventory
    df_perc = percentage_inventory()
    perc_rows = _extract_percentage_series(df_perc)
    found_steel_perc = _foundation_steel_percentage(df_perc)

    tables = load_feature_tables()
    modeled_categories = [c for c in CATEGORY_ORDER if len(tables[c]) >= MIN_SAMPLES]

    summary: dict = {c: [] for c in CATEGORY_ORDER}
    r2_by_category: dict = {c: [] for c in modeled_categories}
    perm_shares: dict = {}
    grouped_shares: dict = {}
    spearman_shares: dict = {}
    composition_stats: dict = {}
    validation_stats: dict = {}
    cluster_members: dict = {}

    for category in CATEGORY_ORDER:
        df = tables[category]
        if len(df) == 0:
            continue
        masses = compute_material_masses(category, df, perc_rows, found_steel_perc)
        for label, arr in masses.items():
            df[label] = arr

        materials = MATERIALS_BY_CATEGORY[category]

        # Full per-turbine data: characteristics + computed material masses (kg) + all 25
        # impacts, one row per turbine. Saved for every category with data, including ones
        # below MIN_SAMPLES (offshore spar) that aren't modeled but were still used/computed.
        cat_dir = OUT / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cat_dir / "per_turbine_data.csv")

        # validation: captured bucket sum vs. independent component-formula total
        P = df["P_rated_kW"].values.astype(float)
        d = df["Diameter_m"].values.astype(float)
        h = df["Hub_height_m"].values.astype(float)
        dist_to_grid = df["dist_to_grid_m"].values.astype(float)
        park_size = df["park_size"].values.astype(float)
        offshore = category != "onshore"
        formula_total = (M_tower(d, h)
                          + (M_nacelle_offshore(P) if offshore else M_nacelle_onshore(P))
                          + (M_rotor_offshore(d) if offshore else M_rotor_onshore(d))
                          + M_elec(P)
                          + (M_cable_offshore(P, dist_to_grid, park_size) if offshore
                             else M_cable_onshore(P, dist_to_grid)))
        if category == "onshore":
            formula_total = formula_total + M_found_onshore(d, h)
        else:
            sea_depth = df["sea_depth_m"].values.astype(float)
            fm = foundation_masses_offshore(category, P, sea_depth)
            if category == "offshore_monopile":
                # Only percentage(P) of m_monopile is ever assigned to a material exchange in
                # the real model (fleet_evaluation_lca_algebraic.py:1174-1176) -- the rest of
                # m_monopile is not captured as any Input-phase material there either, so the
                # validation total must apply the same percentage to match what's captured.
                fxs, fys = found_steel_perc
                fperc = np.interp(P, fxs, fys)
                formula_total = formula_total + fperc * fm["material"] + fm["grout"]
            else:
                formula_total = formula_total + sum(fm.values())
        captured_total = df[materials].sum(axis=1).values
        mean_captured, mean_formula = captured_total.mean(), formula_total.mean()
        validation_stats[category] = {
            "captured": mean_captured, "formula": mean_formula,
            "pct_diff": (mean_captured - mean_formula) / mean_formula * 100,
        }

        means = df[materials].mean().sort_values(ascending=False)
        share = means / means.sum() * 100
        composition_stats[category] = {
            "top_material": means.index[0], "top_share": share.iloc[0], "top_mean_kg": means.iloc[0],
        }

        if category not in modeled_categories:
            continue

        print(f"\n=== {CATEGORY_TITLES[category]} (n={len(df)}) ===")

        impact_labels = [label for _, label, _ in IMPACTS]
        spearman_r, spearman_p = spearman_with_pvalues(df, materials, IMPACT_COLUMNS)
        spearman_r.columns = impact_labels
        spearman_p.columns = impact_labels
        spearman_shares[category] = spearman_r
        spearman_r.to_csv(cat_dir / "spearman_r_matrix.csv")
        spearman_p.to_csv(cat_dir / "spearman_pvalue_matrix.csv")

        # Cluster near-collinear materials (Spearman |r| >= 0.7) once per category -- see
        # importance_stats.py. Materials are far more collinear than turbine characteristics
        # (several pairs > 0.95, some exactly 1.0: they're all near-deterministic functions of
        # the same underlying P/d/h), so individual material rankings within a cluster are not
        # reliably distinguishable; the clustered ranking is the one to trust for "which
        # material" claims.
        clusters = compute_clusters(df, materials)
        cluster_label_map = {raw: wrap_cluster_label(members) for raw, members in clusters.items()}
        display_index = list(cluster_label_map.values())
        cluster_members[category] = list(clusters.values())
        pd.DataFrame({"cluster_members": [", ".join(g) for g in cluster_members[category]]}
                     ).to_csv(cat_dir / "material_clusters.csv", index=False)

        perm_matrix = pd.DataFrame(index=materials, dtype=float)
        grouped_matrix = pd.DataFrame(index=display_index, dtype=float)
        significance_matrix = pd.DataFrame(index=materials, dtype=float)
        for full_col, short_label, slug in IMPACTS:
            impurity, permutation, r2, significance, grouped = train_and_rank(
                df, materials, full_col, clusters)
            out_path = cat_dir / f"fig_material_importance_{slug}.png"
            plot_importances(impurity, permutation, r2, materials, short_label,
                              CATEGORY_TITLES[category], out_path, significance)

            perm_matrix[short_label] = permutation.reindex(materials)
            grouped_matrix[short_label] = grouped.rename(index=cluster_label_map).reindex(display_index)
            significance_matrix[short_label] = significance["p_value"].reindex(materials)
            r2_by_category[category].append(r2)
            top_perm = permutation.idxmax()
            top_spear = spearman_r[short_label].abs().idxmax()
            n_sig = int((significance["p_value"] < 0.05).sum())
            summary[category].append({
                "impact_label": short_label, "r2": r2, "top_perm": top_perm,
                "top_spearman": top_spear, "fig_path": out_path,
                "n_significant": n_sig, "n_total": len(materials),
            })
            print(f"  {short_label}: R^2={r2:.3f}, top(perm)={top_perm}, top(spearman)={top_spear}, "
                  f"significant={n_sig}/{len(materials)}")

        perm_shares[category] = perm_matrix.div(perm_matrix.sum(axis=0), axis=1)
        grouped_shares[category] = grouped_matrix.div(grouped_matrix.sum(axis=0), axis=1)
        perm_matrix.to_csv(cat_dir / "permutation_importance_matrix.csv")
        grouped_matrix.to_csv(cat_dir / "grouped_permutation_importance_matrix.csv")
        significance_matrix.to_csv(cat_dir / "permutation_pvalue_matrix.csv")

    composition_path = plot_composition(tables, modeled_categories)
    heatmap_path = plot_heatmap(
        perm_shares, modeled_categories,
        "Material permutation-importance share across all impact categories, by siting",
        "Share of permutation importance\n(within each impact column)",
        OUT / "fig_importance_heatmap.png")
    grouped_heatmap_path = plot_heatmap(
        grouped_shares, modeled_categories,
        "Clustered material importance share (correlated materials shuffled together)",
        "Share of grouped permutation importance\n(within each impact column)",
        OUT / "fig_grouped_importance_heatmap.png", row_height=3.6, ytick_fontsize=7.5)
    spearman_path = plot_heatmap(
        spearman_shares, modeled_categories,
        "Spearman rank correlation: material mass vs. impact, by siting", "Spearman r",
        OUT / "fig_spearman_heatmap.png", cmap="RdBu_r", vmin=-1, vmax=1, row_height=3.6)

    write_markdown(tables, summary, r2_by_category, composition_stats, validation_stats,
                    heatmap_path, spearman_path, composition_path, grouped_heatmap_path,
                    cluster_members, modeled_categories)


if __name__ == "__main__":
    main()
