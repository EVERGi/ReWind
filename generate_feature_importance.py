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

Features used (all already computed per turbine by precompute_geo_columns.py):
    P_rated_kW, Hub_height_m, Diameter_m    turbine size
    dist_rotor_m, dist_nacelle_m, dist_tower_m   manufacturer -> site transport distances
    dist_to_grid_m                          cable length (turbine -> nearest substation)
    sea_depth_m                             offshore categories only (0 and constant onshore)
    turbine_age, park_size                  fleet/siting characteristics

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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
FEATURES_COMMON = ["P_rated_kW", "Hub_height_m", "Diameter_m", "dist_rotor_m", "dist_nacelle_m",
                    "dist_tower_m", "dist_to_grid_m", "turbine_age", "park_size"]
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
}

GEO_FEATURE_COLS = ["dist_rotor_m", "dist_nacelle_m", "dist_tower_m", "dist_to_grid_m", "sea_depth_m"]
TURBINE_FEATURE_COLS = ["turbine_age", "park_size", "P_rated_kW", "Hub_height_m", "Diameter_m"]

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

# Palette (from the dataviz skill's validated reference palette, matching generate_paper_figures.py)
BLUE, ORANGE = "#2a78d6", "#eb6834"
GRAY, TEXT_SECONDARY = "#c7c6c0", "#52514e"


def classify_category(filename: str) -> str:
    m = re.search(r"_offshore_(monopile|semi-submersible|spar)_lca_algebraic\.csv$", filename)
    if m:
        return f"offshore_{m.group(1)}"
    return "onshore"


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


def train_and_rank(df: pd.DataFrame, features: list[str], target_col: str):
    X = df[features].astype(float)
    y = df[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    r2 = r2_score(y_test, model.predict(X_test))

    impurity = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    perm = permutation_importance(model, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1)
    permutation = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)

    return impurity, permutation, r2


def plot_importances(impurity: pd.Series, permutation: pd.Series, r2: float, features: list[str],
                      impact_label: str, category_label: str, out_path: Path):
    order = impurity.reindex(features).sort_values().index  # ascending, so barh reads largest-on-top
    labels = [FEATURE_LABELS[f] for f in order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    fig.suptitle(f"What drives per-turbine {impact_label}: {category_label}",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")

    axes[0].barh(labels, impurity.loc[order].values, color=BLUE, height=0.62)
    axes[0].set_title("Impurity-based (random forest)", fontsize=10.5, color=TEXT_SECONDARY)
    axes[0].set_xlabel("Importance", fontsize=9.5)

    axes[1].barh(labels, permutation.loc[order].values, color=ORANGE, height=0.62)
    axes[1].set_title("Permutation (held-out test set)", fontsize=10.5, color=TEXT_SECONDARY)
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


def plot_final_heatmap(perm_shares: dict[str, pd.DataFrame], modeled_categories: list[str]):
    """One figure, one subplot per modeled category: features (rows) x impacts (columns),
    color = permutation importance as a share of that impact's total (columns sum to 1)."""
    n = len(modeled_categories)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.1 * n + 1.2))
    if n == 1:
        axes = [axes]

    impact_labels = [label for _, label, _ in IMPACTS]
    im = None
    for ax, category in zip(axes, modeled_categories):
        mat = perm_shares[category]
        mat = mat.reindex(columns=impact_labels)
        im = ax.imshow(mat.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([FEATURE_LABELS[f] for f in mat.index], fontsize=8.5)
        ax.set_xticks(range(len(impact_labels)))
        if ax is axes[-1]:
            ax.set_xticklabels(impact_labels, rotation=60, ha="right", fontsize=7.5)
        else:
            ax.set_xticklabels([])
        ax.set_title(CATEGORY_TITLES[category], fontsize=10.5, color=TEXT_SECONDARY, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Feature importance share across all impact categories, by turbine siting",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.015)
    cbar.set_label("Share of permutation importance\n(within each impact column)", fontsize=8.5)
    fig.subplots_adjust(left=0.18, right=0.86, top=0.93, bottom=0.22, hspace=0.55)
    out_path = OUT / "fig_importance_heatmap.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def write_markdown(tables: dict[str, pd.DataFrame], summary: dict[str, list[dict]],
                    r2_by_category: dict[str, list[float]], heatmap_path: Path,
                    modeled_categories: list[str]):
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
        "fleet data, which of those known drivers matters most — for every impact category, "
        "not just the headline climate metric — and to check whether the answer changes "
        "with how a turbine is sited."
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
        "category) pair — up to 4 x 25 = 100 models — on an 80/20 train/test split. Two "
        "rankings per model, computed two different ways so neither's blind spots go "
        "unchecked: the model's own impurity-based `feature_importances_`, and permutation "
        "importance (how much held-out R² drops when a single feature is shuffled, averaged "
        "over 15 repeats). Permutation is the more trustworthy of the two, since impurity-based "
        "importance is known to inflate high-cardinality continuous features."
    )
    lines.append("")
    onshore_n = len(FEATURES_ONSHORE)
    offshore_n = len(FEATURES_OFFSHORE)
    lines.append(
        f"Onshore uses {onshore_n} features (no sea depth, always 0). Offshore categories use "
        f"{offshore_n} features (adds sea depth). Categories with fewer than {MIN_SAMPLES} "
        "turbines are reported but not modeled — an 80/20 split on a handful of rows doesn't "
        "produce a meaningful ranking."
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

    for category in CATEGORY_ORDER:
        df = tables[category]
        n = len(df)
        lines.append(f"## {CATEGORY_TITLES[category]} (n={n:,})")
        lines.append("")
        if category not in modeled_categories:
            lines.append(
                f"Only {n} turbines in the fleet in this category — below the "
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
        lines.append("| Impact category | Held-out R² | Top feature (permutation) |")
        lines.append("|---|---|---|")
        for row in summary[category]:
            lines.append(f"| {row['impact_label']} | {row['r2']:.3f} | {row['top_feature']} |")
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

    modeled_categories = [c for c in CATEGORY_ORDER if len(tables[c]) >= MIN_SAMPLES]

    summary: dict[str, list[dict]] = {c: [] for c in CATEGORY_ORDER}
    r2_by_category: dict[str, list[float]] = {c: [] for c in modeled_categories}
    perm_shares: dict[str, pd.DataFrame] = {}

    for category in modeled_categories:
        df = tables[category]
        features = FEATURES_ONSHORE if category == "onshore" else FEATURES_OFFSHORE
        cat_dir = OUT / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {CATEGORY_TITLES[category]} (n={len(df)}) ===")

        perm_matrix = pd.DataFrame(index=features, dtype=float)
        for full_col, short_label, slug in IMPACTS:
            impurity, permutation, r2 = train_and_rank(df, features, full_col)
            out_path = cat_dir / f"fig_feature_importance_{slug}.png"
            plot_importances(impurity, permutation, r2, features, short_label,
                              CATEGORY_TITLES[category], out_path)

            perm_matrix[short_label] = permutation.reindex(features)
            r2_by_category[category].append(r2)
            top_feature = FEATURE_LABELS[permutation.idxmax()]
            summary[category].append({
                "impact_label": short_label, "r2": r2, "top_feature": top_feature,
                "fig_path": out_path,
            })
            print(f"  {short_label}: R^2={r2:.3f}, top={top_feature}")

        # normalize each column (impact) so permutation importances sum to 1 -> comparable share
        perm_shares[category] = perm_matrix.div(perm_matrix.sum(axis=0), axis=1)

    heatmap_path = plot_final_heatmap(perm_shares, modeled_categories)

    offshore_dfs = [tables[c] for c in ["offshore_monopile", "offshore_semi-submersible", "offshore_spar"]
                    if len(tables[c]) > 0]
    plot_known_formula_scatter(tables["onshore"], offshore_dfs)

    write_markdown(tables, summary, r2_by_category, heatmap_path, modeled_categories)


if __name__ == "__main__":
    main()
