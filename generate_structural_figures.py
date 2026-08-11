"""
generate_structural_figures.py

Three figures supporting STRUCTURAL_DEPENDENCIES.md's claims with pictures instead of only
tables of numbers:

1. Material x material and feature x feature Spearman correlation matrices, per turbine
   category -- the actual structure the clustering step (importance_stats.compute_clusters)
   acts on, visualized directly rather than described in a table.
2. M_tower(d,h) vs M_found_onshore(d,h): the near-perfect line behind "epoxy resin correlates
   with concrete at r=0.999" (STRUCTURAL_DEPENDENCIES.md section 2).
3. Semi-submersible economies of scale: rated power vs. mean foundation steel mass, and rated
   power vs. mean GWP100/kWh, side by side (two panels, not a dual-axis chart -- see the
   dataviz skill's "one axis" rule) -- STRUCTURAL_DEPENDENCIES.md section 7.

Run once (output is cached to disk):
    python generate_structural_figures.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import generate_feature_importance as gfi
import generate_material_importance as gmi
from generate_feature_importance import (
    REPO, CATEGORY_ORDER, CATEGORY_TITLES, FEATURES_ONSHORE, FEATURES_OFFSHORE, FEATURE_LABELS,
    GWP_COL, load_feature_tables, BLUE, ORANGE, GRAY, TEXT_SECONDARY,
)

OUT = REPO / "figures" / "structural_dependencies"
OUT.mkdir(parents=True, exist_ok=True)
MODELED_CATEGORIES = ["onshore", "offshore_monopile", "offshore_semi-submersible"]


def _style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(color=GRAY, linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)


def plot_correlation_matrix(matrices: dict, labels_by_category: dict, title: str, out_path):
    """One panel per category, a square Spearman correlation matrix (diverging: negative blue,
    positive red, 0 white -- the same RdBu_r convention already used by the Spearman-vs-impact
    heatmap elsewhere in this project, for visual consistency).

    Cells left NaN (a material/feature with zero variance in this category -- e.g. Lead/Tin,
    offshore semi-submersible, whose percentage-of-mass table is flat there -- correlation with
    anything, including itself, is mathematically undefined, 0/0) are rendered as a distinct
    flat gray via cmap.set_bad(), NOT filled with 0.0: "undefined because constant" and "computed
    as genuinely uncorrelated" are different claims and must not share a color."""
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#bbbbbb")
    n = len(MODELED_CATEGORIES)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6.2))
    im = None
    for ax, category in zip(axes, MODELED_CATEGORIES):
        mat = matrices[category]
        labels = labels_by_category[category]
        im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_title(CATEGORY_TITLES[category], fontsize=10.5, color=TEXT_SECONDARY)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.subplots_adjust(top=0.86, bottom=0.22, wspace=0.5, right=0.9)
    pos = axes[-1].get_position()
    cbar_ax = fig.add_axes((0.93, pos.y0, 0.015, pos.y1 - pos.y0))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Spearman r", fontsize=8.5)
    fig.text(0.02, 0.02, "Gray = undefined (zero variance in this category, e.g. a clamped "
              "percentage table), not zero correlation", fontsize=8, color=TEXT_SECONDARY)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_tower_vs_foundation(df: pd.DataFrame):
    """STRUCTURAL_DEPENDENCIES.md section 2: M_tower and M_found_onshore are both (near-)exact
    linear functions of d^2*h, so they're near-perfectly correlated with each other -- shown
    directly, not just asserted via a correlation coefficient."""
    d = df["Diameter_m"].values.astype(float)
    h = df["Hub_height_m"].values.astype(float)
    tower = gmi.M_tower(d, h)
    found = gmi.M_found_onshore(d, h)
    r = pd.Series(tower).corr(pd.Series(found), method="spearman")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(tower / 1000, found / 1000, s=5, alpha=0.15, color=BLUE, edgecolors="none")
    ax.set_xlabel("M_tower (tonnes)", fontsize=9.5)
    ax.set_ylabel("M_found_onshore (tonnes)", fontsize=9.5)
    ax.set_title("Tower mass vs. foundation mass, onshore\n"
                  "(both are linear in the same d²h -- Spearman r = %.3f)" % r,
                  fontsize=11, color=TEXT_SECONDARY)
    _style_axes(ax)
    fig.tight_layout()
    out_path = OUT / "fig_tower_vs_foundation.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_economies_of_scale(df: pd.DataFrame):
    """STRUCTURAL_DEPENDENCIES.md section 7, as a picture: grouped by rated power, foundation
    mass grows sub-linearly in P while GWP100/kWh falls -- two panels (never a dual-axis chart:
    the two y-quantities have different units and different scales)."""
    P = df["P_rated_kW"].values.astype(float)
    sea_depth = df["sea_depth_m"].values.astype(float)
    gwp = df[GWP_COL].values.astype(float)
    mass = 289.473684210526 * P + 17828.5714285714 * sea_depth + 80000.0

    g = pd.DataFrame({"P": P, "mass": mass, "gwp": gwp}).groupby("P").agg(
        mass_mean=("mass", "mean"), gwp_mean=("gwp", "mean"), n=("mass", "size"))
    g = g[g["n"] >= 5].sort_index()  # drop P values with too few turbines to trust the mean

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(g.index, g["mass_mean"] / 1e6, "o-", color=BLUE, markersize=5)
    axes[0].set_xlabel("Rated power (kW)", fontsize=9.5)
    axes[0].set_ylabel("Mean foundation steel mass (kt)", fontsize=9.5)
    axes[0].set_title("Mass grows sub-linearly in P", fontsize=10.5, color=TEXT_SECONDARY)

    axes[1].plot(g.index, g["gwp_mean"], "o-", color=ORANGE, markersize=5)
    axes[1].set_xlabel("Rated power (kW)", fontsize=9.5)
    axes[1].set_ylabel("Mean GWP100 (kg CO2-Eq/kWh)", fontsize=9.5)
    axes[1].set_title("...while GWP100/kWh falls", fontsize=10.5, color=TEXT_SECONDARY)

    for ax in axes:
        _style_axes(ax)
    fig.suptitle("Offshore semi-submersible: economies of scale", fontsize=13, fontweight="bold",
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = OUT / "fig_economies_of_scale.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    tables = load_feature_tables()

    # --- Figure 1: feature x feature correlation matrices ---
    feat_matrices, feat_labels = {}, {}
    for category in MODELED_CATEGORIES:
        df = tables[category]
        features = FEATURES_ONSHORE if category == "onshore" else FEATURES_OFFSHORE
        corr = df[features].corr(method="spearman")
        feat_matrices[category] = corr.values
        feat_labels[category] = [FEATURE_LABELS[f] for f in features]
    plot_correlation_matrix(feat_matrices, feat_labels,
                             "Feature x feature Spearman correlation, by siting",
                             OUT / "fig_feature_correlation_matrix.png")

    # --- Figure 2: material x material correlation matrices ---
    os.chdir(str(gmi._REWIND_DIR))
    from prepare_inventories import percentage_inventory
    df_perc = percentage_inventory()
    perc_rows = gmi._extract_percentage_series(df_perc)
    found_steel_perc = gmi._foundation_steel_percentage(df_perc)

    mat_matrices, mat_labels = {}, {}
    dfs_with_materials = {}
    for category in MODELED_CATEGORIES:
        df = tables[category].copy()
        masses = gmi.compute_material_masses(category, df, perc_rows, found_steel_perc)
        for label, arr in masses.items():
            df[label] = arr
        dfs_with_materials[category] = df
        materials = gmi.MATERIALS_BY_CATEGORY[category]
        corr = df[materials].corr(method="spearman")  # left un-filled: see plot_correlation_matrix
        mat_matrices[category] = corr.values
        mat_labels[category] = materials
    plot_correlation_matrix(mat_matrices, mat_labels,
                             "Material x material Spearman correlation, by siting",
                             OUT / "fig_material_correlation_matrix.png")

    # --- Figure 3: tower vs foundation scatter (onshore) ---
    plot_tower_vs_foundation(tables["onshore"])

    # --- Figure 4: semi-submersible economies of scale ---
    plot_economies_of_scale(dfs_with_materials["offshore_semi-submersible"])


if __name__ == "__main__":
    main()
