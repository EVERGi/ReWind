"""
generate_material_contribution.py

Answers a different question than generate_material_importance.py does. That script ranks
materials by STATISTICAL DRIVER: does a material's mass VARIATION across turbines help predict
impact VARIATION (random forest + permutation importance). This script computes the actual
CONTRIBUTION: how much of a turbine's real impact (kg CO2-Eq, etc.) is caused by producing each
material, via `contribution = mass(turbine, material) x impact_per_kg(material)`.

These can disagree. generate_material_importance.py found fiberglass/epoxy resin as the top
*driver* of offshore-monopile impacts despite being <1% of turbine mass, and speculated (without
verifying) that resins might carry disproportionately high impact-per-kg. This script computes
that per-kg factor directly against the real ecoinvent database and checks.

Unlike generate_material_importance.py, this DOES need brightway2/ecoinvent -- but only for a
small, fixed set of single-unit background LCA computations (17 materials x 25 methods = 425
queries against fixed activities), not the 77,552-turbine fleet computation. Verified first
(see IMPORTANCE_METHODOLOGY.md discussion) that none of these 17 raw-material activities vary by
country in this ecoinvent version (steel, cement, and everything else all resolve to the same
GLO/regional market activity regardless of turbine location) -- so one global factor per material
covers the whole fleet, no per-country loop needed.

Run once (output is cached to disk):
    python generate_material_contribution.py
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import generate_material_importance as gmi
from generate_feature_importance import (
    REPO, CATEGORY_ORDER, CATEGORY_TITLES, IMPACTS, IMPACT_COLUMNS, GWP_COL,
    load_feature_tables, BLUE, ORANGE, GRAY, TEXT_SECONDARY, plot_heatmap,
)
from generate_material_importance import MATERIALS_BY_CATEGORY, DATASET_LABELS

OUT = REPO / "figures" / "material_contribution"
OUT.mkdir(parents=True, exist_ok=True)
MD_OUT = REPO / "MATERIAL_CONTRIBUTION.md"
FACTORS_CSV = OUT / "impact_per_kg_factors.csv"
MODELED_CATEGORIES = ["onshore", "offshore_monopile", "offshore_semi-submersible"]

# Concrete's real ecoinvent exchange is in m3 (input_exc[concrete_act] = V_conc_sym,
# fleet_evaluation_lca_algebraic.py:569); our "Concrete" mass bucket is in kg
# (M_found_onshore - M_reinf, generate_material_importance.py:227). 2200 kg/m3 converts the
# per-m3 factor this script computes back to per-kg, matching the bucket's units.
CONCRETE_DENSITY_KG_PER_M3 = 2200.0

# Reference point for resolving material activities -- confirmed (see module docstring) that
# every one of these 17 materials resolves to the identical activity regardless of location, so
# any point works; reuses the same DK reference point fleet_evaluation_lca_algebraic.py's own
# validated run used.
LON_REF, LAT_REF = 9.5, 56.2


def _act(df_act, dataset, phase=None):
    """Same lookup fleet_evaluation_lca_algebraic.py's _act() does."""
    import bw2data as bd
    if phase:
        m = df_act[(df_act["Dataset"] == dataset) & (df_act["Phase"] == phase)]
    else:
        m = df_act[df_act["Dataset"] == dataset]
    if m.empty:
        return None
    uuid = m.iloc[0]["UUID"]
    if isinstance(uuid, tuple):
        return bd.get_activity(uuid)
    return bd.get_activity(code=uuid)


def resolve_material_activities():
    """{material display label: (Activity, unit_conversion_to_kg)}. unit_conversion is 1.0 for
    every kg-native material, 1/2200 for Concrete (m3 -> kg)."""
    _REWIND_DIR = REPO / "REWIND" / "REWIND"
    sys.path.insert(0, str(_REWIND_DIR))
    os.chdir(str(_REWIND_DIR))
    from prepare_inventories import ecoinvent_setup, activities_and_uuids
    from built_inventory import create_dictionary_update

    print("Setting up ecoinvent (once)...")
    ecoinvent_setup(_REWIND_DIR / "data" / "datasets")
    df_act = activities_and_uuids(lon=LON_REF, lat=LAT_REF)

    activities = {}
    for raw_name, label in DATASET_LABELS.items():
        act = _act(df_act, raw_name, "Input")
        if act is None:
            print(f"  [WARNING] could not resolve activity for {raw_name!r}, skipping")
            continue
        activities[label] = (act, 1.0)

    # Foundation-specific: Concrete and Reinforcing steel come from the bespoke reference-turbine
    # dict (built_inventory.py constructs them directly, not a simple name lookup), same as
    # fleet_evaluation_lca_algebraic.py:518-519.
    print("Building reference inventory for Concrete/Reinforcing steel/Iron ore...")
    ref = create_dictionary_update(P=2000, lon=LON_REF, lat=LAT_REF, h=100, d=80,
                                    park_size=50, sea_depth=0, print_details=False)
    concrete_act = list(ref["Input"]["Foundation"]["Concrete, 30MPa"].keys())[0]
    reinf_act = list(ref["Input"]["Foundation"]["Reinforced concrete"].keys())[0]
    activities["Concrete"] = (concrete_act, 1.0 / CONCRETE_DENSITY_KG_PER_M3)
    activities["Reinforcing steel (rebar)"] = (reinf_act, 1.0)

    # Monopile grout is the same Cement activity fleet_evaluation_lca_algebraic.py's cement_dk
    # uses (transport_cement_elec's Cement, market for cement Portland) -- confirmed
    # location-independent (resolves to "Europe without Switzerland" for every country tested).
    from prepare_inventories import transport_cement_elec
    _, _, cement_act, _ = transport_cement_elec(lon=LON_REF, lat=LAT_REF)
    activities["Grout (cement)"] = (cement_act, 1.0)

    # Spar-only, not modeled (n=2) but computed for completeness.
    import bw2data as bd
    iron_act = [a for a in bd.Database("ecoinvent-391-cutoff")
                if a["name"] == "market for iron ore, crude ore, 46% Fe"]
    if iron_act:
        activities["Iron ore"] = (iron_act[0], 1.0)

    return activities


def compute_factors(activities: dict) -> pd.DataFrame:
    """{material label: {impact column: impact per kg}}, via lca_algebraic.compute_impacts on
    each background activity directly (1 unit = 1 kg for everything except Concrete, handled by
    the unit_conversion factor from resolve_material_activities)."""
    import lca_algebraic as agb
    import bw2data as bd

    EF_METHODS = [m for m in bd.methods
                  if "EF v3.1" in str(m) and "no LT" not in str(m) and "EN1" not in str(m)]
    assert len(EF_METHODS) == 25, f"expected 25 EF v3.1 methods, got {len(EF_METHODS)}"

    rows = {}
    for label, (act, conv) in activities.items():
        print(f"  computing impacts for {label} ({act['name']}, {act['unit']})...")
        res = agb.compute_impacts(act, methods=EF_METHODS)
        row = res.iloc[0]
        row.index = [str(c) for c in row.index]  # already IMPACT_COLUMNS-format strings
        rows[label] = row * conv
    return pd.DataFrame(rows).T  # materials x impact columns, per-kg


def load_or_compute_factors() -> pd.DataFrame:
    if FACTORS_CSV.exists():
        print(f"Loading cached factors from {FACTORS_CSV}")
        return pd.read_csv(FACTORS_CSV, index_col=0)
    activities = resolve_material_activities()
    factors = compute_factors(activities)
    factors.to_csv(FACTORS_CSV)
    print(f"Saved {FACTORS_CSV}")
    return factors


def plot_factor_comparison(factors: pd.DataFrame, out_path: Path):
    """Impact-per-kg for GWP100, every material, log scale -- directly answers "do resins really
    carry more impact per kg than steel" (IMPORTANCE_METHODOLOGY.md section 5a/b's open question).

    Symlog, not plain log: "Reinforcing steel (rebar)" is negative (see the write-up's callout on
    why -- a pre-existing bug reproduced faithfully, not an error in this script). A plain log
    axis cannot represent a negative value at all -- matplotlib just silently drops that bar,
    which looks like missing data, not "known and negative." Symlog is linear through zero
    (linthresh) and log-scaled beyond it in both directions, so the negative bar renders
    correctly instead of vanishing."""
    vals = factors[GWP_COL].sort_values()
    colors = [ORANGE if v == vals.max() else BLUE for v in vals.values]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(vals.index, vals.values, color=colors, height=0.62)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("kg CO2-Eq per kg of material (symlog scale)", fontsize=9.5)
    ax.set_title("Impact intensity per kg, by material (GWP100)", fontsize=12,
                 fontweight="bold", color=TEXT_SECONDARY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
    ax.grid(axis="x", color=GRAY, linewidth=0.6, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved {out_path}")


def impact_unit(impact_column: str) -> str:
    """Extract the unit from an IMPACT_COLUMNS-style string, e.g.
    "acidification - accumulated exceedance (AE)[mol H+-Eq]" -> "mol H+-Eq". Every impact
    category has its own unit (not all "kg CO2-Eq" like the GWP100-only figures/tables use),
    so the per-impact table needs this to be honest about what each row's number means."""
    m = re.search(r"\[([^\[\]]+)\]$", impact_column)
    return m.group(1) if m else ""


def plot_impact_composition(means: pd.Series, category: str, impact_label: str, unit: str,
                             out_path: Path):
    """Single-panel version of plot_contribution_composition: one (category, impact) pair, mean
    contribution per material -- the per-impact analogue of MATERIAL_IMPORTANCE.md's 75
    individual per-impact figures, but showing actual contribution instead of predictive
    importance.

    Y-axis: each raw material. X-axis: that material's mean contribution to this impact category,
    per turbine, in the impact's own unit (shown in the axis label -- units differ category to
    category, e.g. CTUh for human toxicity vs. kg CO2-Eq for climate change, so this can't be
    assumed the way the GWP100-only charts could get away with). The %, printed at each bar's
    end, is that material's share of the total across all materials for this one impact column
    (columns are independent -- shares here do not compare across different impacts)."""
    order = means.sort_values()
    share = order / order.sum() * 100 if order.sum() != 0 else order * 0
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.barh(order.index, order.values, color=ORANGE, height=0.62)
    for i, (label, val) in enumerate(order.items()):
        ax.text(val, i, f"  {share[label]:.1f}%", va="center", fontsize=7.5, color=TEXT_SECONDARY)
    # Wrapped onto two lines: some impact category names are long enough (combined with the
    # category title) to run past the figure's right edge at fontsize 12 on one line -- e.g.
    # "What contributes to per-turbine Human toxicity, non-carcinogenic, inorganics: Offshore -
    # semi-submersible" was silently truncated mid-word before this fix.
    title = f"What contributes to per-turbine {impact_label}:\n{CATEGORY_TITLES[category]}"
    ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT_SECONDARY, loc="left")
    ax.set_xlabel(f"Mean contribution ({unit})" if unit else "Mean contribution", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
    ax.grid(axis="x", color=GRAY, linewidth=0.6, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_contribution_composition(contribution_means: dict, out_path: Path):
    """Mean GWP100 contribution (kg CO2-Eq) per material, one panel per category -- the
    contribution analogue of generate_material_importance.py's mass-composition chart."""
    n = len(MODELED_CATEGORIES)
    fig, axes = plt.subplots(1, n, figsize=(5.8 * n, 6.2))
    fig.suptitle("Mean GWP100 contribution per turbine, by material and siting", fontsize=13,
                 fontweight="bold", x=0.02, ha="left")
    for ax, category in zip(axes, MODELED_CATEGORIES):
        means = contribution_means[category].sort_values()
        share = means / means.sum() * 100
        ax.barh(means.index, means.values, color=ORANGE, height=0.62)
        for i, (label, val) in enumerate(means.items()):
            ax.text(val, i, f"  {share[label]:.1f}%", va="center", fontsize=7.5,
                    color=TEXT_SECONDARY)
        ax.set_title(CATEGORY_TITLES[category], fontsize=10.5, color=TEXT_SECONDARY)
        ax.set_xlabel("Mean GWP100 contribution (kg CO2-Eq)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRAY)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        ax.grid(axis="x", color=GRAY, linewidth=0.6, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved {out_path}")


def write_markdown(factors, contribution_means, contribution_shares, rank_comparison, summary_rows):
    lines = []
    lines.append("# What the utilized materials actually contribute to impact")
    lines.append("")
    lines.append(
        "Companion to `MATERIAL_IMPORTANCE.md`, which ranks materials by *statistical driver* "
        "(does a material's mass variation across turbines predict impact variation). This "
        "answers a different, more literal question: how much of a turbine's actual impact is "
        "caused by producing each material, in real units (kg CO2-Eq, etc.), not a predictive "
        "ranking."
    )
    lines.append("")
    lines.append(
        "`contribution = mass(turbine, material) x impact_per_kg(material)`. Material masses "
        "are the same ones `generate_material_importance.py` already computes (pure algebra, no "
        "new LCA run). The per-kg factors are new: one background LCA computation per material "
        "against the real ecoinvent database (`generate_material_contribution.py`), 17 "
        "materials x 25 impact methods, cached in `impact_per_kg_factors.csv`. Verified first "
        "that none of these 17 materials' background activities vary by country in this "
        "ecoinvent version (steel, cement, and everything else all resolve to the same "
        "GLO/regional market for every location tested) -- so one global factor per material "
        "covers the whole fleet correctly, no per-country computation needed."
    )
    lines.append("")

    lines.append("## Impact intensity per kg (GWP100)")
    lines.append("")
    lines.append(f"![Impact per kg by material](figures/material_contribution/fig_factor_comparison.png)")
    lines.append("")
    top3 = factors[GWP_COL].sort_values(ascending=False).head(3)
    bot3 = factors[GWP_COL].sort_values(ascending=True).head(3)
    lines.append(
        f"Highest impact-per-kg: {', '.join(f'**{m}** ({v:.2f} kg CO2-Eq/kg)' for m, v in top3.items())}. "
        f"Lowest: {', '.join(f'**{m}** ({v:.3f} kg CO2-Eq/kg)' for m, v in bot3.items())}."
    )
    lines.append("")

    lines.append("## Does this explain MATERIAL_IMPORTANCE.md's monopile finding?")
    lines.append("")
    lines.append(rank_comparison)
    lines.append("")

    lines.append("## Mean GWP100 contribution, by material and siting")
    lines.append("")
    lines.append("![Contribution composition](figures/material_contribution/fig_contribution_composition.png)")
    lines.append("")
    for category in MODELED_CATEGORIES:
        means = contribution_means[category].sort_values(ascending=False)
        share = means / means.sum() * 100
        lines.append(f"**{CATEGORY_TITLES[category]}**: top contributor is **{means.index[0]}** "
                      f"({share.iloc[0]:.1f}% of captured GWP100, {means.iloc[0]:,.2f} kg CO2-Eq mean).")
        lines.append("")

    lines.append("## Full contribution share, all 25 impact categories")
    lines.append("")
    lines.append("![Contribution share heatmap](figures/material_contribution/fig_contribution_heatmap.png)")
    lines.append("")
    lines.append(
        "Each column is one impact category, normalized to sum to 1 -- same convention as "
        "`MATERIAL_IMPORTANCE.md`'s heatmaps, but color here is actual share of impact, not "
        "predictive importance."
    )
    lines.append("")

    lines.append(
        "## Full breakdown, all 25 impact categories\n\n"
        "Every impact category has its own unit (not all kg CO2-Eq like the GWP100-only "
        "sections above) -- shown per row below rather than assumed."
    )
    lines.append("")
    for category in MODELED_CATEGORIES:
        rows = summary_rows[category]
        lines.append(f"### {CATEGORY_TITLES[category]}")
        lines.append("")
        lines.append("| Impact category | Unit | Top contributor | Value | Share of total |")
        lines.append("|---|---|---|---|---|")
        for row in rows:
            lines.append(f"| {row['impact_label']} | {row['unit']} | {row['top_material']} | "
                          f"{row['top_value']:,.4g} | {row['top_share']:.1f}% |")
        lines.append("")
        for row in rows:
            fig_rel = row["fig_path"].relative_to(REPO)
            lines.append(f"#### {row['impact_label']}")
            lines.append("")
            lines.append(f"![{row['impact_label']}, {CATEGORY_TITLES[category]}]({fig_rel})")
            lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "Impact-per-kg factors come from `lca_algebraic.compute_impacts()` on each material's "
        "background market activity directly -- the same ecoinvent-391-cutoff database "
        "`fleet_evaluation_lca_algebraic.py` uses, but computed once per material rather than "
        "read back from a fleet run. Covers the same 17 material buckets "
        "`MATERIAL_IMPORTANCE.md` does (same Input-phase-only scope: no transformer units, no "
        "land use, no process/service exchanges -- see that file's own Limitations). Offshore "
        "spar (n=2) is computed but not separately reported, same threshold as elsewhere in "
        "this project."
    )
    lines.append("")

    MD_OUT.write_text("\n".join(lines))
    print(f"\nSaved {MD_OUT}")


def main():
    factors = load_or_compute_factors()
    plot_factor_comparison(factors, OUT / "fig_factor_comparison.png")

    tables = load_feature_tables()
    os.chdir(str(REPO / "REWIND" / "REWIND"))
    from prepare_inventories import percentage_inventory
    df_perc = percentage_inventory()
    perc_rows = gmi._extract_percentage_series(df_perc)
    found_steel_perc = gmi._foundation_steel_percentage(df_perc)

    contribution_means = {}
    contribution_shares = {}  # category -> DataFrame materials x impacts, share of total per column
    all_impact_means = {}     # category -> {impact_label: Series(material -> mean contribution)}
    summary_rows = {}         # category -> list of per-impact summary dicts, for the full table
    for category in MODELED_CATEGORIES:
        df = tables[category]
        masses = gmi.compute_material_masses(category, df, perc_rows, found_steel_perc)
        materials = MATERIALS_BY_CATEGORY[category]
        cat_dir = OUT / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        contrib = pd.DataFrame(index=df.index)
        for material in materials:
            if material not in factors.index:
                continue
            for col in IMPACT_COLUMNS:
                contrib[(material, col)] = masses[material] * factors.loc[material, col]

        mean_by_material = pd.Series(
            {m: contrib[(m, GWP_COL)].mean() for m in materials if (m, GWP_COL) in contrib.columns})
        contribution_means[category] = mean_by_material

        share_mat = pd.DataFrame(index=materials, dtype=float)
        impact_means = {}
        rows = []
        for col, label, slug in IMPACTS:
            col_means = pd.Series({m: contrib[(m, col)].mean() for m in materials
                                    if (m, col) in contrib.columns})
            share_mat[label] = col_means / col_means.sum()
            impact_means[label] = col_means

            out_path = cat_dir / f"fig_contribution_{slug}.png"
            plot_impact_composition(col_means, category, label, impact_unit(col), out_path)

            top_material = col_means.abs().idxmax()  # abs(): some impacts have negative-signed
                                                       # totals (e.g. disposal credits dominate),
                                                       # where the "top" contributor by magnitude
                                                       # isn't the max signed value
            total = col_means.sum()
            top_share = (col_means[top_material] / total * 100) if total != 0 else float("nan")
            rows.append({
                "impact_label": label, "unit": impact_unit(col), "top_material": top_material,
                "top_value": col_means[top_material], "top_share": top_share,
                "fig_path": out_path,
            })
        contribution_shares[category] = share_mat
        all_impact_means[category] = impact_means
        summary_rows[category] = rows

        print(f"{CATEGORY_TITLES[category]}: top GWP100 contributor = "
              f"{mean_by_material.idxmax()} ({mean_by_material.max():,.2f} kg CO2-Eq mean)")

    plot_contribution_composition(contribution_means, OUT / "fig_contribution_composition.png")
    plot_heatmap(contribution_shares, MODELED_CATEGORIES,
                 "Material contribution share across all impact categories, by siting",
                 "Share of mean contribution\n(within each impact column)",
                 OUT / "fig_contribution_heatmap.png", row_height=3.6, ytick_fontsize=7.5)

    # Rank comparison for the monopile fiberglass/epoxy open question
    monopile_importance_order = ["Fiberglass", "Epoxy resin", "Steel (low-alloy)"]
    monopile_contrib = contribution_means["offshore_monopile"]
    monopile_contrib_rank = monopile_contrib.rank(ascending=False)
    lines = []
    for m in monopile_importance_order:
        if m in monopile_contrib.index:
            lines.append(f"- **{m}**: {factors.loc[m, GWP_COL]:.3f} kg CO2-Eq/kg, "
                          f"contributes {monopile_contrib[m]:,.2f} kg CO2-Eq mean "
                          f"(rank {int(monopile_contrib_rank[m])} of {len(monopile_contrib)} by contribution)")
    rank_comparison = "\n".join(lines)
    print("\n" + rank_comparison)

    write_markdown(factors, contribution_means, contribution_shares, rank_comparison, summary_rows)


if __name__ == "__main__":
    main()
