"""
Generate maps, fleet figures, and validation-error figures from the
lca_algebraic fleet results produced by fleet_evaluation_lca_algebraic.py.

Inputs  (REWIND/REWIND/data/, reorganized into subfolders 28 Jul 2026):
    results/fleet_impacts_<ISO>_lca_algebraic.csv                          onshore, all 38 countries
    results/fleet_impacts_<ISO>_offshore_<bucket>_lca_algebraic.csv         offshore, per foundation bucket
    validation/fleet_impacts_<ISO>_validation_summary_by_stage.csv         algebraic-vs-baseline error, per stage
    validation/fleet_impacts_<ISO>_validation_errors.csv                   algebraic-vs-baseline error, per turbine
    Shared_Rewind/Fleet_results/NUTS_RG_20M_2024_4326.gpkg          subnational (NUTS) boundaries

Outputs (figures/):
    fig1_country_gwp_capacity.png        headline country ranking (GWP100), adapted from Figure5
    fig2_nuts3_choropleth.png            headline subnational map (GWP100), adapted from Figure6
    fig3_turbine_scatter_map.png         headline turbine-level EU map (GWP100)
    fig4_offshore_foundation_map.png     offshore turbines by foundation type
    fig5_validation_error_by_stage.png   GWP100 validation error per lifecycle stage per country
    fig6_disposal_bug_scatter.png        baseline vs algebraic Disposal-stage values, post-fix
    fig7_error_boxplot_by_stage.png      GWP100 per-turbine relative error distribution by stage
    fig8_error_heatmap_stage_by_method.png    validation error, all 25 EF v3.1 methods x lifecycle stage
    fig9_error_heatmap_country_by_method.png  validation error (Total stage), all 25 methods x country
    by_metric/country_ranking/<slug>.png      country ranking, one per EF v3.1 method (25 files)
    by_metric/nuts3_choropleth/<slug>.png     NUTS3 choropleth, one per EF v3.1 method (25 files)
    by_metric/turbine_scatter/<slug>.png      turbine scatter map, one per EF v3.1 method (25 files)
"""
import re
import textwrap
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=FutureWarning)

REPO = Path(__file__).resolve().parent
DATA = REPO / "REWIND" / "REWIND" / "data"
RESULTS_DIR = DATA / "results"           # reorganized 28 Jul 2026, was flat inside DATA
VALIDATION_DIR = DATA / "validation"
NUTS_PATH = REPO / "Shared_Rewind" / "Fleet_results" / "NUTS_RG_20M_2024_4326.gpkg"
OUT = REPO / "figures"
OUT.mkdir(exist_ok=True)
BY_METRIC_DIRS = {name: OUT / "by_metric" / name
                  for name in ("country_ranking", "nuts3_choropleth", "turbine_scatter")}
for d in BY_METRIC_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

GWP_COL = "climate change - global warming potential (GWP100)[kg CO2-Eq]"
BASE_COLS = ["turbine_idx", "P_rated_kW", "Hub_height_m", "Diameter_m", "Latitude", "Longitude",
             "Lifetime_production_kWh"]


def parse_method(col: str):
    """'category - indicator[unit]' -> (category, indicator, unit, slug)."""
    m = re.match(r"^(.*?) - (.*?)\[(.*?)\]$", col)
    category, indicator, unit = m.groups()
    slug = re.sub(r"[^a-z0-9]+", "-", category.lower()).strip("-")
    return category, indicator, unit, slug


ALL_METHOD_COLS = None  # discovered from data at load time (see main())
COLS = None  # set in main() once ALL_METHOD_COLS is known

# ---------------------------------------------------------------------------
# Palette (from the dataviz skill's validated reference palette)
# ---------------------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, TEXT_SECONDARY = "#c7c6c0", "#52514e"
CRITICAL, WARNING, GOOD = "#d03b3b", "#fab219", "#0ca30c"
SEQ_BLUE_CMAP = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq_blue", ["#eaf2fc", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"]
)
SEQ_ORANGE_CMAP = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq_orange", ["#fdeee5", "#f5b48f", "#eb6834", "#b8481f", "#7a2f13"]
)

# Colors chosen to match the original Figure5/Figure6 (dark green onshore /
# cyan offshore, blue sequential) so the adapted figures stay visually
# comparable to Shared_Rewind/4_submission.
ONSHORE_GREEN = "#0b6e2f"
OFFSHORE_CYAN = "#7fd9e8"

FOUNDATION_COLORS = {"monopile": BLUE, "semi-submersible": ORANGE, "spar": AQUA}

COUNTRY_NAMES = {
    "AT": "Austria", "BA": "Bosnia and Herzegovina", "BE": "Belgium",
    "BG": "Bulgaria", "BY": "Belarus", "CH": "Switzerland", "CY": "Cyprus",
    "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
    "ES": "Spain", "FI": "Finland", "FO": "Faroe Islands", "FR": "France",
    "GB": "United Kingdom", "GR": "Greece", "HR": "Croatia", "HU": "Hungary",
    "IE": "Ireland", "IS": "Iceland", "IT": "Italy", "LT": "Lithuania",
    "LU": "Luxembourg", "LV": "Latvia", "ME": "Montenegro",
    "MK": "North Macedonia", "NL": "Netherlands", "NO": "Norway",
    "PL": "Poland", "PT": "Portugal", "RO": "Romania", "RS": "Serbia",
    "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia", "UA": "Ukraine",
    "XK": "Kosovo",
}
NUTS_CNTR_OVERRIDE = {"GB": "UK", "GR": "EL"}  # Eurostat NUTS country codes

VALIDATED_COUNTRIES = ["DK", "BE", "DE", "GB", "NO"]  # have baseline vs algebraic comparisons
STAGES = ["Input", "Assembly", "Transport", "Maintenance", "Disposal", "Total"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def discover_methods() -> list:
    sample = next(RESULTS_DIR.glob("fleet_impacts_*_lca_algebraic.csv"))
    cols = pd.read_csv(sample, nrows=0).columns.tolist()
    exclude = set(BASE_COLS)
    return [c for c in cols if c not in exclude]


def load_onshore() -> pd.DataFrame:
    frames = []
    for f in sorted(RESULTS_DIR.glob("fleet_impacts_*_lca_algebraic.csv")):
        m = re.match(r"fleet_impacts_([A-Z]{2})_lca_algebraic$", f.stem)
        if not m:
            continue
        iso = m.group(1)
        df = pd.read_csv(f, usecols=COLS)
        df["ISO"] = iso
        df["Offshore"] = 0
        df["foundation"] = "onshore"
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["gwp_g_per_kwh"] = out[GWP_COL] * 1000.0
    return out


def load_offshore() -> pd.DataFrame:
    frames = []
    for f in sorted(RESULTS_DIR.glob("fleet_impacts_*_offshore_*_lca_algebraic.csv")):
        m = re.match(r"fleet_impacts_([A-Z]{2})_offshore_([a-z-]+)_lca_algebraic$", f.stem)
        if not m:
            continue
        iso, bucket = m.groups()
        df = pd.read_csv(f, usecols=COLS)
        df["ISO"] = iso
        df["Offshore"] = 1
        df["foundation"] = bucket
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=COLS + ["ISO", "Offshore", "foundation", "gwp_g_per_kwh"])
    out = pd.concat(frames, ignore_index=True)
    out["gwp_g_per_kwh"] = out[GWP_COL] * 1000.0
    return out


def load_validation_summaries_by_stage() -> pd.DataFrame:
    frames = []
    for iso in VALIDATED_COUNTRIES:
        parts = [("onshore", VALIDATION_DIR / f"fleet_impacts_{iso}_validation_summary_by_stage.csv")]
        for f in sorted(VALIDATION_DIR.glob(f"fleet_impacts_{iso}_offshore_*_validation_summary_by_stage.csv")):
            bucket = re.match(rf"fleet_impacts_{iso}_offshore_([a-z-]+)_validation_summary_by_stage",
                               f.stem).group(1)
            parts.append((bucket, f))
        for part_name, path in parts:
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["ISO"] = iso
            df["part"] = part_name
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_validation_errors(method_prefix: str | None = None) -> pd.DataFrame:
    """Per-turbine baseline-vs-algebraic errors, all validated countries+parts.

    method_prefix: if given, keep only method_key values starting with this string
    (e.g. "climate change - global warming potential" for GWP100 only). None keeps all 25 methods.
    """
    frames = []
    for iso in VALIDATED_COUNTRIES:
        paths = [VALIDATION_DIR / f"fleet_impacts_{iso}_validation_errors.csv"]
        paths += sorted(VALIDATION_DIR.glob(f"fleet_impacts_{iso}_offshore_*_validation_errors.csv"))
        for path in paths:
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if method_prefix:
                df = df[df["method_key"].str.startswith(method_prefix)]
            df["ISO"] = iso
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_nuts3():
    gdf = gpd.read_file(NUTS_PATH)
    gdf = gdf[gdf["LEVL_CODE"] == 3].copy()
    # Crop to mainland-Europe extent (drop overseas territories), matching Figure6's framing.
    bounds = gdf.geometry.bounds
    keep = (bounds["minx"] > -25) & (bounds["maxx"] < 45) & (bounds["miny"] > 33) & (bounds["maxy"] < 72)
    return gdf[keep]


# ---------------------------------------------------------------------------
# Figure 1: country ranking (adapted from Figure5_CC_country_installed_capacity.png)
# ---------------------------------------------------------------------------
def country_ranking_chart(onshore, offshore, metric_col, value_label, unit, title, out_path,
                           scale=1.0):
    on_vals = onshore[metric_col] * scale
    off_vals = offshore[metric_col] * scale if len(offshore) else pd.Series(dtype=float)

    on_g = onshore.assign(_v=on_vals).groupby("ISO").agg(
        mean_v=("_v", "mean"), cap_gw=("P_rated_kW", lambda s: s.sum() / 1e6))
    if len(offshore):
        off_g = offshore.assign(_v=off_vals).groupby("ISO").agg(
            mean_v=("_v", "mean"), cap_gw=("P_rated_kW", lambda s: s.sum() / 1e6))
    else:
        off_g = pd.DataFrame(columns=["mean_v", "cap_gw"])

    isos = sorted(on_g.index, key=lambda i: on_g.loc[i, "mean_v"], reverse=True)
    eu_avg_on = on_vals.mean()
    eu_avg_off = off_vals.mean() if len(off_vals) else np.nan

    fig, ax = plt.subplots(figsize=(9, 0.32 * len(isos) + 1.5))
    ax_top = ax.twiny()

    for i, iso in enumerate(isos):
        y = len(isos) - i
        ax.barh(y, on_g.loc[iso, "mean_v"], height=0.38, color=ONSHORE_GREEN, align="center",
                label="Onshore" if i == 0 else None)
        if iso in off_g.index:
            ax.barh(y - 0.42, off_g.loc[iso, "mean_v"], height=0.38, color=OFFSHORE_CYAN,
                     align="center", label="Offshore" if i == 0 else None)
            ax_top.scatter(off_g.loc[iso, "cap_gw"], y - 0.42, marker="o", color="dimgray",
                            s=22, zorder=5, label="Offshore capacity" if i == 0 else None)
        ax_top.scatter(on_g.loc[iso, "cap_gw"], y, marker="^", color="black", s=26, zorder=5,
                        label="Onshore capacity" if i == 0 else None)

    ax.axvline(eu_avg_on, color="black", ls="--", lw=1)
    ax.text(eu_avg_on, len(isos) + 1.3, f"EU avg onshore = {eu_avg_on:.3g}", ha="center", fontsize=8)
    if not np.isnan(eu_avg_off):
        ax.axvline(eu_avg_off, color="black", ls=":", lw=1)
        ax.text(eu_avg_off, len(isos) + 0.5, f"EU avg offshore = {eu_avg_off:.3g}", ha="center", fontsize=8)

    ax.set_yticks(range(len(isos), 0, -1))
    ax.set_yticklabels([COUNTRY_NAMES.get(i, i) for i in isos], fontsize=8)
    ax.set_ylim(0.2, len(isos) + 2.0)
    ax.set_xlabel(f"Mean {value_label} impact of fleet ({unit}/kWh)")
    ax_top.set_xscale("log")
    ax_top.set_xlabel("Installed capacity [GW] (log scale)")

    handles = [Patch(color=ONSHORE_GREEN, label="Onshore"),
               Patch(color=OFFSHORE_CYAN, label="Offshore"),
               Line2D([0], [0], marker="^", color="black", lw=0, label="Onshore installed capacity"),
               Line2D([0], [0], marker="o", color="dimgray", lw=0, label="Offshore installed capacity")]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.9)
    ax.set_title(textwrap.fill(title, 62), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: NUTS3 subnational choropleth (adapted from Figure6_impact_incl_offshore.png)
# ---------------------------------------------------------------------------
def nuts3_choropleth(joined, offshore, nuts3, metric_col, value_label, unit, title, out_path,
                      scale=1.0, cmap=SEQ_BLUE_CMAP):
    region_mean = joined.groupby("NUTS_ID")[metric_col].mean() * scale
    plot_gdf = nuts3.merge(region_mean.rename("v"), left_on="NUTS_ID", right_index=True, how="left")

    vmin, vmax = np.nanpercentile(plot_gdf["v"].dropna(), [2, 98])

    fig, ax = plt.subplots(figsize=(11, 12))
    plot_gdf[plot_gdf["v"].isna()].plot(ax=ax, color=GRAY, edgecolor="white", linewidth=0.15)
    plot_gdf[plot_gdf["v"].notna()].plot(ax=ax, column="v", cmap=cmap, edgecolor="white",
                                          linewidth=0.15, vmin=vmin, vmax=vmax)

    if len(offshore):
        ax.scatter(offshore["Longitude"], offshore["Latitude"], c=offshore[metric_col] * scale,
                   cmap=cmap, vmin=vmin, vmax=vmax, s=10, edgecolor="black",
                   linewidth=0.3, zorder=5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(f"{value_label} impact ({unit}/kWh)")

    ax.set_xlim(-25, 35)
    ax.set_ylim(34, 71)
    ax.set_axis_off()
    ax.set_title(textwrap.fill(title, 78), fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: turbine-level scatter map
# ---------------------------------------------------------------------------
def turbine_scatter_map(onshore, offshore, nuts0, metric_col, value_label, unit, title, out_path,
                         scale=1.0, cmap=SEQ_BLUE_CMAP):
    fig, ax = plt.subplots(figsize=(11, 12))
    nuts0.plot(ax=ax, color="#f3f2ef", edgecolor="white", linewidth=0.3)

    on_v = onshore[metric_col] * scale
    off_v = offshore[metric_col] * scale if len(offshore) else pd.Series(dtype=float)
    vmin, vmax = np.nanpercentile(pd.concat([on_v, off_v]) if len(off_v) else on_v, [2, 98])
    ax.scatter(onshore["Longitude"], onshore["Latitude"], c=on_v, cmap=cmap,
               vmin=vmin, vmax=vmax, s=3, marker="o", label="Onshore", linewidths=0)
    if len(offshore):
        ax.scatter(offshore["Longitude"], offshore["Latitude"], c=off_v,
                   cmap=cmap, vmin=vmin, vmax=vmax, s=14, marker="^",
                   edgecolor="black", linewidth=0.3, label="Offshore")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(f"{value_label} impact ({unit}/kWh)")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="gray", lw=0, label="Onshore turbine"),
                        Line2D([0], [0], marker="^", color="gray", lw=0, label="Offshore turbine")],
              loc="lower left", fontsize=9)
    ax.set_xlim(-25, 35)
    ax.set_ylim(34, 71)
    ax.set_axis_off()
    ax.set_title(textwrap.fill(title, 78), fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: offshore foundation-type map
# ---------------------------------------------------------------------------
def fig4_offshore_foundation_map(offshore, nuts0):
    fig, ax = plt.subplots(figsize=(9, 10))
    nuts0.plot(ax=ax, color="#f3f2ef", edgecolor="white", linewidth=0.3)

    for bucket, color in FOUNDATION_COLORS.items():
        sub = offshore[offshore["foundation"] == bucket]
        if not len(sub):
            continue
        ax.scatter(sub["Longitude"], sub["Latitude"], color=color, s=22, edgecolor="black",
                   linewidth=0.3, label=f"{bucket.capitalize()} (n={len(sub)})", zorder=5)

    ax.set_xlim(-12, 32)
    ax.set_ylim(43, 65)
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_title("Offshore fleet by foundation type (sea-depth-driven bucket)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_offshore_foundation_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig4_offshore_foundation_map.png")


# ---------------------------------------------------------------------------
# Figure 5: validation error per lifecycle stage per country (GWP100, headline view)
# ---------------------------------------------------------------------------
def fig5_validation_error_by_stage(val_summary):
    onshore_summary = val_summary[val_summary["part"] == "onshore"]
    pivot = onshore_summary.pivot(index="ISO", columns="stage", values="abs_mean_error_pct")
    stage_order = [s for s in STAGES if s in pivot.columns]
    pivot = pivot[stage_order]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    n_stages = len(stage_order)
    n_countries = len(pivot)
    width = 0.8 / n_stages
    x = np.arange(n_countries)

    for j, stage in enumerate(stage_order):
        color = WARNING if stage == "Total" else (GOOD if stage == "Disposal" else BLUE)
        ax.bar(x + j * width, pivot[stage].values, width=width, color=color,
               label=stage, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x + width * (n_stages - 1) / 2)
    ax.set_xticklabels([COUNTRY_NAMES.get(i, i) for i in pivot.index], fontsize=9)
    ax.set_ylabel("Abs. mean error, algebraic vs. baseline (%)")
    ax.set_yscale("log")
    ax.set_ylim(top=10)
    ax.axhline(1, color="gray", ls="--", lw=0.8)
    ax.text(n_countries - 0.5, 1.4, "1% reference", fontsize=8, color="gray", ha="right")
    ax.set_title("GWP100 validation error by lifecycle stage: onshore, 5 countries\n"
                 "Disposal (green) was a 62-98% mean-error bug, fixed 26 Jul 2026 "
                 "(see PLAN_lca_algebraic.md)",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc="upper left", framealpha=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_validation_error_by_stage.png", dpi=200)
    plt.close(fig)
    print("wrote fig5_validation_error_by_stage.png")


# ---------------------------------------------------------------------------
# Figure 6: Disposal bug in detail, baseline vs algebraic, log-log (GWP100)
# ---------------------------------------------------------------------------
def fig6_disposal_bug_scatter(val_errors_gwp):
    disp = val_errors_gwp[val_errors_gwp["stage"] == "Disposal"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    ax = axes[0]
    colors = {iso: c for iso, c in zip(VALIDATED_COUNTRIES, [BLUE, ORANGE, AQUA, YELLOW, "#4a3aa7"])}
    for iso in VALIDATED_COUNTRIES:
        sub = disp[disp["ISO"] == iso]
        if not len(sub):
            continue
        ax.scatter(sub["baseline_value"].abs(), sub["algebraic_value"].abs(), s=10, alpha=0.6,
                   color=colors[iso], label=iso, linewidths=0)
    lims = [1e-7, disp[["baseline_value", "algebraic_value"]].abs().max().max() * 1.5]
    ax.plot(lims, lims, color="black", lw=1, ls="--", label="y = x (perfect match)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Baseline Disposal-stage GWP100 (kg CO$_2$-Eq, exact bc.LCA)")
    ax.set_ylabel("Algebraic Disposal-stage GWP100 (kg CO$_2$-Eq, lca_algebraic)")
    ax.set_title("Per-turbine Disposal-stage impact, post-fix:\nbaseline and algebraic model now track closely")
    ax.legend(fontsize=8)

    ax = axes[1]
    stage_share = (disp.set_index(["ISO", "turbine_idx"])["total_baseline_value"]
                   .rename("total") .to_frame()
                   .join(disp.set_index(["ISO", "turbine_idx"])["baseline_value"].rename("disposal")))
    stage_share["disposal_share_pct"] = 100 * stage_share["disposal"].abs() / stage_share["total"].abs()
    ax.hist(stage_share["disposal_share_pct"], bins=40, color=BLUE, edgecolor="white")
    ax.set_xlabel("Disposal stage share of Total GWP100 (%, baseline/exact)")
    ax.set_ylabel("Turbine count")
    ax.set_title("For context: Disposal is a small share of Total impact,\n"
                 "even the pre-fix bug never threatened the headline Total number")

    fig.suptitle("Disposal-stage bug, fixed 26 Jul 2026: was up to 98% mean relative error, "
                 "now <0.2% (see PLAN_lca_algebraic.md)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_disposal_bug_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig6_disposal_bug_scatter.png")


# ---------------------------------------------------------------------------
# Figure 7: error distribution by stage, box-whisker, GWP100 (style of FigureA12)
# ---------------------------------------------------------------------------
def fig7_error_boxplot_by_stage(val_errors_gwp):
    stage_order = [s for s in STAGES if s in val_errors_gwp["stage"].unique()]
    data = [val_errors_gwp.loc[val_errors_gwp["stage"] == s, "rel_error_pct"].clip(-500, 500).dropna()
            for s in stage_order]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bp = ax.boxplot(data, tick_labels=stage_order, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=2, alpha=0.3, color=TEXT_SECONDARY))
    for patch, stage in zip(bp["boxes"], stage_order):
        patch.set_facecolor(GOOD if stage == "Disposal" else BLUE)
        patch.set_alpha(0.75)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Relative error, algebraic vs. baseline (%), clipped to ±500%")
    ax.set_title("GWP100 per-turbine relative error distribution by lifecycle stage\n"
                 "(all validated countries pooled: DK, BE, DE, GB, NO, onshore + offshore; "
                 "Disposal fixed 26 Jul 2026)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_error_boxplot_by_stage.png", dpi=200)
    plt.close(fig)
    print("wrote fig7_error_boxplot_by_stage.png")


# ---------------------------------------------------------------------------
# Figure 8: validation error heatmap, all 25 methods x lifecycle stage (pooled countries)
# ---------------------------------------------------------------------------
def fig8_error_heatmap_stage_by_method(val_errors_all, method_categories):
    pooled = val_errors_all.copy()
    pooled["category"] = pooled["method_key"].str.split(" - ", n=1).str[0]
    grp = pooled.groupby(["category", "stage"])["rel_error_pct"].apply(lambda s: s.abs().mean())
    stage_order = [s for s in STAGES if s in pooled["stage"].unique()]
    categories = sorted(method_categories.keys())
    mat = pd.DataFrame(index=categories, columns=stage_order, dtype=float)
    for cat in categories:
        for stage in stage_order:
            mat.loc[cat, stage] = grp.get((cat, stage), np.nan)

    fig, ax = plt.subplots(figsize=(7, 10))
    data = np.log10(mat.values.astype(float) + 1e-6)
    im = ax.imshow(data, cmap=SEQ_ORANGE_CMAP, aspect="auto")
    ax.set_xticks(range(len(stage_order)))
    ax.set_xticklabels(stage_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=8)
    for i in range(len(categories)):
        for j in range(len(stage_order)):
            v = mat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2g}", ha="center", va="center", fontsize=6.5,
                        color="white" if data[i, j] > np.nanmedian(data) else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("log$_{10}$(abs. mean error %)")
    ax.set_title("Validation error by impact category × lifecycle stage\n"
                 "(all 25 EF v3.1 methods, pooled across DK/BE/DE/GB/NO, onshore + offshore)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_error_heatmap_stage_by_method.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig8_error_heatmap_stage_by_method.png")


# ---------------------------------------------------------------------------
# Figure 9: validation error heatmap, Total stage, all 25 methods x country
# ---------------------------------------------------------------------------
def fig9_error_heatmap_country_by_method(val_errors_all, method_categories):
    total = val_errors_all[val_errors_all["stage"] == "Total"].copy()
    total["category"] = total["method_key"].str.split(" - ", n=1).str[0]
    grp = total.groupby(["category", "ISO"])["rel_error_pct"].apply(lambda s: s.abs().mean())
    categories = sorted(method_categories.keys())
    mat = pd.DataFrame(index=categories, columns=VALIDATED_COUNTRIES, dtype=float)
    for cat in categories:
        for iso in VALIDATED_COUNTRIES:
            mat.loc[cat, iso] = grp.get((cat, iso), np.nan)

    fig, ax = plt.subplots(figsize=(6, 10))
    data = np.log10(mat.values.astype(float) + 1e-6)
    im = ax.imshow(data, cmap=SEQ_BLUE_CMAP, aspect="auto")
    ax.set_xticks(range(len(VALIDATED_COUNTRIES)))
    ax.set_xticklabels(VALIDATED_COUNTRIES, fontsize=9)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=8)
    for i in range(len(categories)):
        for j in range(len(VALIDATED_COUNTRIES)):
            v = mat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2g}", ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] > np.nanmedian(data) else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    cbar.set_label("log$_{10}$(abs. mean error %)")
    ax.set_title("Headline (Total-stage) validation error by impact category × country\n"
                 "(all 25 EF v3.1 methods)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig9_error_heatmap_country_by_method.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig9_error_heatmap_country_by_method.png")


# ---------------------------------------------------------------------------
def main():
    global ALL_METHOD_COLS, COLS
    print("Discovering impact methods...")
    ALL_METHOD_COLS = discover_methods()
    COLS = BASE_COLS + ALL_METHOD_COLS
    method_categories = {}  # category -> (full_col, indicator, unit, slug)
    for col in ALL_METHOD_COLS:
        category, indicator, unit, slug = parse_method(col)
        method_categories[category] = (col, indicator, unit, slug)
    print(f"  {len(ALL_METHOD_COLS)} impact methods found")

    print("Loading fleet results (all methods)...")
    onshore = load_onshore()
    offshore = load_offshore()
    print(f"  onshore: {len(onshore):,} turbines across {onshore['ISO'].nunique()} countries")
    print(f"  offshore: {len(offshore):,} turbines across {offshore['ISO'].nunique() if len(offshore) else 0} countries")

    print("Loading validation data (baseline vs algebraic)...")
    val_summary = load_validation_summaries_by_stage()
    val_errors_gwp = load_validation_errors("climate change - global warming potential")
    val_errors_all = load_validation_errors(None)

    print("Loading NUTS3 / country boundaries...")
    nuts3 = load_nuts3()
    nuts0_path = REPO / "REWIND" / "REWIND" / "data" / "ne_10m_admin_0_countries.shp"
    nuts0 = gpd.read_file(nuts0_path)
    bounds = nuts0.geometry.bounds
    nuts0 = nuts0[(bounds["minx"] > -30) & (bounds["maxx"] < 45)]

    print("Spatial-joining onshore turbines to NUTS3 regions (once, reused for all methods)...")
    pts = gpd.GeoDataFrame(onshore, geometry=gpd.points_from_xy(onshore.Longitude, onshore.Latitude),
                            crs="EPSG:4326")
    joined = gpd.sjoin(pts, nuts3[["NUTS_ID", "geometry"]], how="inner", predicate="within")

    # --- Headline GWP100 figures (fig1-fig7) -------------------------------
    country_ranking_chart(onshore, offshore, "gwp_g_per_kwh", "GWP100", "gCO$_2$eq",
                           "Fleet GWP100 impact by country: lca_algebraic full-EU results",
                           OUT / "fig1_country_gwp_capacity.png")
    print("wrote fig1_country_gwp_capacity.png")
    nuts3_choropleth(joined, offshore, nuts3, "gwp_g_per_kwh", "GWP100", "gCO$_2$eq",
                      "EU wind fleet GWP100 impact by NUTS3 region (onshore) and offshore turbine locations",
                      OUT / "fig2_nuts3_choropleth.png")
    print("wrote fig2_nuts3_choropleth.png")
    turbine_scatter_map(onshore, offshore, nuts0, "gwp_g_per_kwh", "GWP100", "gCO$_2$eq",
                         f"EU wind fleet, {len(onshore) + len(offshore):,} turbines, GWP100 per kWh",
                         OUT / "fig3_turbine_scatter_map.png")
    print("wrote fig3_turbine_scatter_map.png")
    fig4_offshore_foundation_map(offshore, nuts0)
    fig5_validation_error_by_stage(val_summary)
    fig6_disposal_bug_scatter(val_errors_gwp)
    fig7_error_boxplot_by_stage(val_errors_gwp)
    fig8_error_heatmap_stage_by_method(val_errors_all, method_categories)
    fig9_error_heatmap_country_by_method(val_errors_all, method_categories)

    # --- Every metric, every geographic figure type -------------------------
    print(f"\nGenerating by-metric figures for all {len(method_categories)} impact categories...")
    for i, (category, (col, indicator, unit, slug)) in enumerate(sorted(method_categories.items()), 1):
        print(f"  [{i}/{len(method_categories)}] {category}")
        title_suffix = f"{category}: {indicator}"
        country_ranking_chart(
            onshore, offshore, col, category, unit,
            f"Fleet impact by country: {title_suffix}",
            BY_METRIC_DIRS["country_ranking"] / f"{slug}.png")
        nuts3_choropleth(
            joined, offshore, nuts3, col, category, unit,
            f"EU wind fleet impact by NUTS3 region: {title_suffix}",
            BY_METRIC_DIRS["nuts3_choropleth"] / f"{slug}.png")
        turbine_scatter_map(
            onshore, offshore, nuts0, col, category, unit,
            f"EU wind fleet: {title_suffix}",
            BY_METRIC_DIRS["turbine_scatter"] / f"{slug}.png")

    print(f"\nAll figures written to {OUT}/")


if __name__ == "__main__":
    main()
