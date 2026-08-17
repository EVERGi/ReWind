"""
generate_driver_analysis_visuals.py

Two extra visuals for FEATURE_DRIVER_ANALYSIS.md, built directly from the
permutation_importance_matrix.csv files generate_feature_importance.py already writes per
siting (figures/feature_importance/<siting>/) -- no retraining, this only reads cached numbers.

1. fig_gwp100_importance_by_siting.png
   The existing fig_max_importance_by_siting.png plots each feature's SINGLE BEST impact
   category per siting -- but that category differs across siting for every feature except
   dist_to_grid_m (cable length, always Mineral resource use), so most rows on that figure
   aren't actually a fair cross-siting comparison: the three dots in a row can be three
   different targets. This figure instead fixes the category to the headline GWP100 metric
   ("Climate change (total)") for every feature, so all three siting bars in a row are always
   the same target -- a genuine apples-to-apples comparison.

2. fig_importance_spread_dumbbell.png
   Same underlying "max permutation importance across all 25 categories" data as
   fig_max_importance_by_siting.png, replotted as one row per feature with its siting values
   connected by a line, sorted by cross-siting spread (max/min ratio) descending and
   log-scaled on x -- so the features that flip hardest between siting types (foundation mass,
   sea depth, blade mass, rotor diameter) are immediately readable as the longest lines at the
   top, instead of requiring a reader to compare jittered x-coordinates row by row. Deliberately
   omits the per-point impact-category labels the original figure carries: this figure is about
   magnitude divergence, not which category produced it -- see fig_max_importance_by_siting.png
   for that.

3. fig_gwp100_mean_by_siting.png
   Not a feature-importance figure -- reads REWIND/REWIND/data/summary_stats/
   summary_gwp_distribution.csv (already computed by generate_paper_summary_stats.py) and plots
   mean GWP100/kWh by siting/foundation type. Exists to put a number behind FEATURE_DRIVER_ANALYSIS.md
   section 0's "this isn't just 'bigger turbine = more impact'" point: monopile (physically
   lighter than semi-submersible) has the LOWEST mean GWP100/kWh of any group, and semi-submersible
   (heaviest, most material) has the HIGHEST -- foundation type, not turbine size, is what the
   per-kWh metric actually tracks, consistent with fig_top_features_by_impact_grid.png's
   Foundation mass panel.

Run once (output cached to disk):
    python generate_driver_analysis_visuals.py
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from generate_feature_importance import (
    FEATURE_LABELS, CATEGORY_COLORS, CATEGORY_TITLES, GRAY, TEXT_SECONDARY, GWP_COL, IMPACT_LABELS,
    BLUE, ORANGE, AQUA,
)

REPO = Path(__file__).resolve().parent
OUT = REPO / "figures" / "feature_importance"
OUT.mkdir(parents=True, exist_ok=True)
GWP_DIST_CSV = REPO / "REWIND" / "REWIND" / "data" / "summary_stats" / "summary_gwp_distribution.csv"

# Slot 4 (yellow) of the same validated categorical palette BLUE/ORANGE/AQUA (slots 1-3) come
# from -- spar isn't one of the three modeled siting categories (n=2 fleet-wide, below
# MIN_SAMPLES), so it has no CATEGORY_COLORS entry, but it's a real group in the GWP
# distribution figure and needs a distinct, fixed-order color, not a reused one.
YELLOW = "#eda100"

# Offshore spar (n=2 fleet-wide) was never modeled by generate_feature_importance.py -- no
# matrix file exists for it, so it's excluded here too, same as the figures it builds on.
CATEGORIES = ["onshore", "offshore_monopile", "offshore_semi-submersible"]
GWP_LABEL = IMPACT_LABELS[GWP_COL]  # "Climate change (total)" -- the short label used as the
                                     # column name in permutation_importance_matrix.csv


def _load_matrices() -> dict[str, pd.DataFrame]:
    matrices = {}
    for cat in CATEGORIES:
        path = OUT / cat / "permutation_importance_matrix.csv"
        matrices[cat] = pd.read_csv(path, index_col=0)
    return matrices


def plot_gwp100_importance_by_siting(matrices: dict[str, pd.DataFrame], out_path: Path) -> Path:
    features = sorted(
        {f for m in matrices.values() for f in m.index},
        key=lambda f: max(matrices[c].loc[f, GWP_LABEL] for c in CATEGORIES if f in matrices[c].index),
    )
    y_pos = {f: i for i, f in enumerate(features)}
    bar_h = 0.24
    offsets = {cat: (i - 1) * bar_h for i, cat in enumerate(CATEGORIES)}

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(features) + 1.8))
    for cat in CATEGORIES:
        mat = matrices[cat]
        present = [f for f in features if f in mat.index]
        ys = [y_pos[f] + offsets[cat] for f in present]
        xs = [mat.loc[f, GWP_LABEL] for f in present]
        ax.barh(ys, xs, height=bar_h * 0.92, color=CATEGORY_COLORS[cat],
                label=CATEGORY_TITLES[cat], zorder=3)

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([FEATURE_LABELS[f] for f in features], fontsize=9.5)
    ax.set_xlabel("Permutation importance on Climate change (total) / GWP100\n"
                   "(mean R² drop when shuffled, held-out test set)", fontsize=9.5)
    ax.set_title("What drives GWP100 specifically, by feature and siting",
                  fontsize=13, fontweight="bold", x=0.0, ha="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.grid(axis="x", color=GRAY, linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_importance_spread_dumbbell(matrices: dict[str, pd.DataFrame], out_path: Path) -> Path:
    features = sorted({f for m in matrices.values() for f in m.index})
    rows = []
    for f in features:
        vals = {cat: matrices[cat].loc[f].max() for cat in CATEGORIES if f in matrices[cat].index}
        if len(vals) < 2:
            continue  # can't show a spread with only one siting's value
        spread = max(vals.values()) / min(vals.values())
        rows.append((f, vals, spread))
    rows.sort(key=lambda r: r[2])  # ascending: barh puts index 0 at the bottom, so the
                                     # largest spread ends up at the top of the plot

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 1.8))
    labeled_cats = set()
    for i, (f, vals, spread) in enumerate(rows):
        xs = list(vals.values())
        ax.plot([min(xs), max(xs)], [i, i], color=GRAY, linewidth=1.4, zorder=2)
        for cat in CATEGORIES:
            if cat not in vals:
                continue
            label = CATEGORY_TITLES[cat] if cat not in labeled_cats else None
            ax.scatter(vals[cat], i, s=65, color=CATEGORY_COLORS[cat],
                       edgecolors="white", linewidths=0.7, zorder=3, label=label)
            labeled_cats.add(cat)

    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([FEATURE_LABELS[f] for f, _, _ in rows], fontsize=9.5)
    ax.set_xlabel("Maximum permutation importance across impact categories, log scale\n"
                   "(mean R² drop when shuffled, held-out test set)", fontsize=9.5)
    ax.set_title("Which features flip most between siting types",
                  fontsize=13, fontweight="bold", x=0.0, ha="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.grid(axis="x", color=GRAY, linewidth=0.6, alpha=0.5, which="both", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_gwp100_mean_by_siting(out_path: Path) -> Path:
    dist = pd.read_csv(GWP_DIST_CSV, index_col="group")
    groups = ["Onshore", "Offshore - monopile", "Offshore - semi-submersible", "Offshore - spar"]
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    means = [dist.loc[g, "mean"] for g in groups]
    ns = [int(dist.loc[g, "n"]) for g in groups]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(range(len(groups)), means, color=colors, width=0.6, zorder=3)
    for i, (bar, mean, n) in enumerate(zip(bars, means, ns)):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.0006,
                f"{mean:.4f}\n(n={n:,})", ha="center", va="bottom",
                fontsize=8.5, color=TEXT_SECONDARY)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["Onshore", "Offshore\nmonopile", "Offshore\nsemi-sub.",
                         "Offshore\nspar (n=2)"], fontsize=9.5)
    ax.set_ylabel("Mean GWP100 (kg CO2-eq / kWh)", fontsize=9.5)
    ax.set_title("GWP100/kWh tracks foundation type, not turbine size",
                  fontsize=13, fontweight="bold", x=0.0, ha="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRAY, linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(top=max(means) * 1.22)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main():
    matrices = _load_matrices()
    plot_gwp100_importance_by_siting(matrices, OUT / "fig_gwp100_importance_by_siting.png")
    plot_importance_spread_dumbbell(matrices, OUT / "fig_importance_spread_dumbbell.png")
    plot_gwp100_mean_by_siting(OUT / "fig_gwp100_mean_by_siting.png")


if __name__ == "__main__":
    main()
