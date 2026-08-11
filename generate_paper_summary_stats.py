"""
generate_paper_summary_stats.py

Aggregates the per-turbine fleet_impacts_<ISO>[_offshore_<bucket>]_lca_algebraic[_by_stage].csv
results (38 countries, 25 EF v3.1 methods) into paper-ready summary tables:

  1. EU-wide GWP100 distribution, overall and split by onshore/offshore/foundation bucket
  2. Per-country GWP100 ranking, both fleet-size-weighted (true population mean) and
     unweighted (mean of country means, so Germany's 36%-of-fleet size doesn't dominate)
  3. Lifecycle-stage contribution (% of Total), EU-wide and per country
  4. GWP100 vs rated power / turbine age / park size, Pearson + Spearman correlation
  5. Whether climate change (GWP100) tracks the other 24 EF v3.1 methods: per-turbine
     correlation, and per-country rank correlation (does a country that ranks high/low on
     GWP100 rank the same way on other categories, or does it diverge)

Run once (output is cached to disk):
    python generate_paper_summary_stats.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

_REWIND_DIR = Path(__file__).resolve().parent / "REWIND" / "REWIND"
_DATA_DIR = _REWIND_DIR / "data"
_RESULTS_DIR = _DATA_DIR / "results"
_BY_STAGE_DIR = _RESULTS_DIR / "by_stage"
_TURBINES_XLSX = _DATA_DIR / "EU_turbines_input_data.xlsx"
_OUT_DIR = _DATA_DIR / "summary_stats"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

GWP = "climate change - global warming potential (GWP100)[kg CO2-Eq]"


def _bucket_from_filename(path: Path) -> str:
    name = path.stem
    if "_offshore_" not in name:
        return "onshore"
    return name.split("_offshore_")[1].replace("_lca_algebraic", "")


def load_fleet_results() -> pd.DataFrame:
    """One row per turbine across the whole fleet (38 countries, onshore + every offshore
    bucket), tagged with country/offshore/bucket, method columns unchanged."""
    frames = []
    for f in sorted(_RESULTS_DIR.glob("fleet_impacts_*_lca_algebraic.csv")):
        df = pd.read_csv(f, index_col=0)
        name = f.stem.replace("fleet_impacts_", "").replace("_lca_algebraic", "")
        iso = name.split("_offshore_")[0]
        df["ISO_code"] = iso
        df["Offshore"] = "_offshore_" in name
        df["bucket"] = _bucket_from_filename(f)
        frames.append(df)
    out = pd.concat(frames)
    print(f"Loaded {len(out)} turbines across {out['ISO_code'].nunique()} countries "
          f"({(~out['Offshore']).sum()} onshore, {out['Offshore'].sum()} offshore)")
    return out


def load_by_stage() -> pd.DataFrame:
    """One row per (turbine, phase) across the whole fleet."""
    frames = []
    for f in sorted(_BY_STAGE_DIR.glob("fleet_impacts_*_lca_algebraic_by_stage.csv")):
        df = pd.read_csv(f, index_col=0)
        name = f.stem.replace("fleet_impacts_", "").replace("_lca_algebraic_by_stage", "")
        iso = name.split("_offshore_")[0]
        df["ISO_code"] = iso
        frames.append(df)
    out = pd.concat(frames)
    print(f"Loaded {len(out)} (turbine, phase) rows for the stage breakdown")
    return out


def method_columns(df: pd.DataFrame) -> list:
    exclude = {"P_rated_kW", "Hub_height_m", "Diameter_m", "Latitude", "Longitude",
               "Lifetime_production_kWh", "ISO_code", "Offshore", "bucket", "phase"}
    return [c for c in df.columns if c not in exclude]


def section_1_gwp_distribution(fleet: pd.DataFrame):
    print("\n=== 1. EU-wide GWP100 distribution ===")
    rows = []

    def _describe(label, series):
        d = series.describe(percentiles=[0.25, 0.5, 0.75])
        rows.append({"group": label, "n": int(d["count"]), "mean": d["mean"], "std": d["std"],
                     "min": d["min"], "p25": d["25%"], "median": d["50%"], "p75": d["75%"],
                     "max": d["max"]})

    _describe("EU-wide (all turbines)", fleet[GWP])
    _describe("Onshore", fleet.loc[~fleet["Offshore"], GWP])
    _describe("Offshore (all buckets)", fleet.loc[fleet["Offshore"], GWP])
    for bucket in sorted(fleet.loc[fleet["Offshore"], "bucket"].unique()):
        _describe(f"Offshore - {bucket}", fleet.loc[fleet["bucket"] == bucket, GWP])

    out = pd.DataFrame(rows)
    out.to_csv(_OUT_DIR / "summary_gwp_distribution.csv", index=False)
    print(out.to_string(index=False))
    return out


def section_2_country_ranking(fleet: pd.DataFrame):
    print("\n=== 2. Per-country GWP100 ranking ===")
    by_country = fleet.groupby("ISO_code")[GWP]
    weighted = by_country.agg(n="count", mean_weighted="mean", median="median", std="std")
    unweighted_mean = weighted["mean_weighted"].mean()
    weighted_mean = fleet[GWP].mean()

    out = weighted.sort_values("mean_weighted").reset_index()
    out.to_csv(_OUT_DIR / "summary_country_ranking.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nUnweighted mean-of-country-means: {unweighted_mean:.5f} kg CO2-Eq/kWh")
    print(f"Fleet-size-weighted (true population) mean: {weighted_mean:.5f} kg CO2-Eq/kWh")
    print("(Germany is ~36% of the EU turbine count, so the weighted mean leans toward its "
          "own distribution; the unweighted mean treats every country equally regardless of "
          "fleet size.)")
    return out


def section_3_stage_contribution(by_stage: pd.DataFrame):
    print("\n=== 3. Lifecycle-stage contribution to GWP100 (% of Total) ===")
    total_by_turbine = by_stage.groupby(by_stage.index)[GWP].sum()

    eu_wide = by_stage.groupby("phase")[GWP].sum()
    eu_wide_pct = 100 * eu_wide / eu_wide.sum()
    print("EU-wide:")
    print(eu_wide_pct.sort_values(ascending=False).to_string())

    per_country = by_stage.groupby(["ISO_code", "phase"])[GWP].sum().unstack("phase")
    per_country_pct = per_country.div(per_country.sum(axis=1), axis=0) * 100

    out = per_country_pct.reset_index()
    out.to_csv(_OUT_DIR / "summary_stage_contribution.csv", index=False)
    eu_wide_pct.to_csv(_OUT_DIR / "summary_stage_contribution_eu_wide.csv")
    print("\nPer-country (first 10 rows):")
    print(out.head(10).to_string(index=False))
    return out


def section_4_correlations(fleet: pd.DataFrame):
    print("\n=== 4. GWP100 vs turbine characteristics ===")
    turbines = pd.read_excel(_TURBINES_XLSX, sheet_name="EU_turbines_input_data")
    merged = fleet.join(turbines[["turbine_age", "park_size"]], how="left")

    rows = []
    for var in ["P_rated_kW", "turbine_age", "park_size"]:
        sub = merged[[var, GWP]].dropna()
        pear_r, pear_p = pearsonr(sub[var], sub[GWP])
        spear_r, spear_p = spearmanr(sub[var], sub[GWP])
        rows.append({"variable": var, "n": len(sub), "pearson_r": pear_r, "pearson_p": pear_p,
                     "spearman_r": spear_r, "spearman_p": spear_p})
        print(f"  GWP100 vs {var}: pearson r={pear_r:.3f} (p={pear_p:.2e}), "
              f"spearman r={spear_r:.3f} (p={spear_p:.2e}), n={len(sub)}")

    out = pd.DataFrame(rows)
    out.to_csv(_OUT_DIR / "summary_correlations_turbine_characteristics.csv", index=False)
    return out


def section_5_method_consistency(fleet: pd.DataFrame):
    print("\n=== 5. Does GWP100 track the other 24 EF v3.1 methods? ===")
    methods = method_columns(fleet)
    other_methods = [m for m in methods if m != GWP]

    turbine_level, country_level = [], []
    country_means = fleet.groupby("ISO_code")[methods].mean()
    gwp_rank = country_means[GWP].rank()

    for m in other_methods:
        sub = fleet[[GWP, m]].replace([np.inf, -np.inf], np.nan).dropna()
        sub = sub[(sub[GWP] != 0) & (sub[m] != 0)]
        t_pear_r, t_pear_p = pearsonr(sub[GWP], sub[m])
        t_spear_r, t_spear_p = spearmanr(sub[GWP], sub[m])
        # NOTE: at n up to 77,552, turbine-level p-values are essentially always significant
        # (p < 1e-10) regardless of effect size -- included for completeness, but the r values
        # are what actually distinguishes a strong from a weak relationship here, not p.
        turbine_level.append({"method": m, "n": len(sub), "pearson_r": t_pear_r,
                               "pearson_p": t_pear_p, "spearman_r": t_spear_r,
                               "spearman_p": t_spear_p})

        m_rank = country_means[m].rank()
        c_spear_r, c_spear_p = spearmanr(gwp_rank, m_rank)
        country_level.append({"method": m, "country_rank_spearman_r": c_spear_r,
                               "country_rank_spearman_p": c_spear_p})

    turbine_df = pd.DataFrame(turbine_level).sort_values("spearman_r")
    country_df = pd.DataFrame(country_level).sort_values("country_rank_spearman_r")

    turbine_df.to_csv(_OUT_DIR / "summary_gwp_consistency_turbine_level.csv", index=False)
    country_df.to_csv(_OUT_DIR / "summary_gwp_consistency_country_level.csv", index=False)

    print("\nTurbine-level correlation with GWP100, 5 weakest (methods GWP100 tracks worst):")
    print(turbine_df.head(5).to_string(index=False))
    print(f"\nTurbine-level: {(turbine_df['spearman_r'] > 0.9).sum()} of {len(turbine_df)} "
          f"methods have spearman r > 0.9 with GWP100 (strong agreement).")

    print("\nCountry-ranking correlation with GWP100, 5 weakest (a country's GWP100 rank "
          "doesn't predict its rank in these):")
    print(country_df.head(5).to_string(index=False))
    print(f"\nCountry-level: {(country_df['country_rank_spearman_r'] > 0.9).sum()} of "
          f"{len(country_df)} methods rank countries the same way GWP100 does (r > 0.9).")
    return turbine_df, country_df


def main():
    fleet = load_fleet_results()
    by_stage = load_by_stage()

    section_1_gwp_distribution(fleet)
    section_2_country_ranking(fleet)
    section_3_stage_contribution(by_stage)
    section_4_correlations(fleet)
    section_5_method_consistency(fleet)

    print(f"\nAll summary CSVs written to {_OUT_DIR}")


if __name__ == "__main__":
    main()
