"""
generate_weighting_sensitivity.py

Reviewer 2, point 2: unweighted vs. weighted fleet averages, and a sensitivity test without
turbines of 50 kW and below, for all 25 EF v3.1 methods.

Each turbine's impact is per kWh of its lifetime production. The three averages differ only in
the weight passed to np.average:
  unweighted           plain mean, every turbine counts once (as in the manuscript)
  production_weighted  weight = lifetime kWh, i.e. total impact / total kWh. Same basis as
                       the ecoinvent markets, which are built from production shares.
  capacity_weighted    weight = rated power
Scenarios: "all" turbines and "excl_le50kW" (P_rated_kW <= 50 removed).

Outputs in REWIND/REWIND/data/summary_stats/weighting_sensitivity/ (impact per kWh, each
method's own unit):
  eu_means.csv                    EU-wide, onshore, offshore, per foundation type
  country_means.csv               per country and siting
  country_ranking_stability.csv   Spearman of the onshore country ranking vs the unweighted one
  small_turbines.csv              where the <= 50 kW turbines are, share of count and production
  ecoinvent_validation.csv        Sect. 3.4.2 metrics
  ecoinvent_country.csv           per-country model vs ecoinvent
  distributional_consistency.csv  Sect. 3.4.3 share within +-25 / +-50 % of the median, by
                                  turbine count (as in the manuscript) and by production
  nuts3_summary.csv               NUTS3 range (2016 boundaries, as in the manuscript maps)

Run generate_ecoinvent_markets.py first. The script stops if the unweighted results do not
reproduce manuscript 6 or if the weighted averages are not total impact / total kWh.
    python generate_weighting_sensitivity.py
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

from generate_paper_summary_stats import GWP, load_fleet_results, method_columns

REPO = Path(__file__).resolve().parent
DATA = REPO / "REWIND" / "REWIND" / "data"
OUT = DATA / "summary_stats" / "weighting_sensitivity"
ECOINVENT_CSV = DATA / "validation" / "ecoinvent_wind_market_by_country_all_methods.csv"
NUTS_PATH = REPO / "Shared_Rewind" / "Fleet_results" / "NUTS_RG_20M_2016_4326.gpkg"  # has UK

E = "Lifetime_production_kWh"
P = "P_rated_kW"
SMALL_KW = 50
BANDS = (25, 50)  # Sect. 3.4.3, % around the median
WEIGHTS = {"unweighted": None, "production_weighted": E, "capacity_weighted": P}

# the six categories discussed in the manuscript
ABBR = {
    GWP: "CC",
    "human toxicity: carcinogenic - comparative toxic unit for human (CTUh)[CTUh]": "HTC",
    "human toxicity: non-carcinogenic - comparative toxic unit for human (CTUh)[CTUh]": "HTN",
    "ionising radiation: human health - human exposure efficiency relative to u235[kBq U235-Eq]": "IR",
    "land use - soil quality index[dimensionless]": "LU",
    "material resources: metals/minerals - abiotic depletion potential (ADP): elements "
    "(ultimate reserves)[kg Sb-Eq]": "MRM",
}


def average(d, weight, cols):
    """Mean of each column in `cols` over the turbines in `d`, weighted by column `weight`
    (plain .mean() when weight is None)."""
    if weight is None:
        return d[cols].mean()
    return pd.Series(np.average(d[cols], weights=d[weight], axis=0), index=cols)


def grouped_average(df, by, weight, cols):
    if weight is None:
        return df.groupby(by)[cols].mean()
    return df.groupby(by)[cols + [weight]].apply(lambda d: average(d, weight, cols))


def scenarios(fleet):
    return {"all": fleet, f"excl_le{SMALL_KW}kW": fleet[fleet[P] > SMALL_KW]}


def tidy(wide, keys):
    """{weighting: DataFrame (rows = groups, columns = methods)} -> one row per group and
    method, one column per weighting, plus % differences vs unweighted."""
    out = pd.concat(wide, axis=1).stack(level=1, future_stack=True)
    out.index.names = keys + ["method"]
    out = out.reset_index()
    out.insert(len(keys) + 1, "abbr", out["method"].map(ABBR).fillna(""))
    for w in ("production_weighted", "capacity_weighted"):
        out[f"{w}_vs_unweighted_pct"] = 100 * (out[w] / out["unweighted"] - 1)
    return out


def eu_means(fleet, methods):
    parts = []
    for scen, f in scenarios(fleet).items():
        groups = {"EU-wide": f, "Onshore": f[~f["Offshore"]], "Offshore": f[f["Offshore"]],
                  **{f"Offshore - {b}": d for b, d in f[f["Offshore"]].groupby("bucket")}}
        wide = {w: pd.DataFrame({g: average(d, col, methods) for g, d in groups.items()}).T
                for w, col in WEIGHTS.items()}
        wide["median"] = pd.DataFrame({g: d[methods].median() for g, d in groups.items()}).T
        out = tidy(wide, ["group"])
        out.insert(0, "scenario", scen)
        out.insert(2, "n", out["group"].map({g: len(d) for g, d in groups.items()}))
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def country_means(fleet, methods):
    parts = []
    for scen, f in scenarios(fleet).items():
        f = f.assign(siting=np.where(f["Offshore"], "offshore", "onshore"))
        by = ["siting", "ISO_code"]
        out = tidy({w: grouped_average(f, by, col, methods) for w, col in WEIGHTS.items()}, by)
        out.insert(0, "scenario", scen)
        out.insert(3, "n", pd.MultiIndex.from_frame(out[by]).map(f.groupby(by).size()))
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def ranking_stability(cm):
    """Spearman of the onshore country ranking under each variant vs the unweighted,
    all-turbine ranking used in the manuscript."""
    on = cm[cm["siting"] == "onshore"].set_index(["method", "scenario", "ISO_code"]).sort_index()
    rows = []
    for m in on.index.unique("method"):
        base = on.loc[(m, "all"), "unweighted"]
        for scen in on.index.unique("scenario"):
            for w in WEIGHTS:
                v = on.loc[(m, scen), w].reindex(base.index)
                rows.append({"method": m, "abbr": ABBR.get(m, ""), "scenario": scen, "weighting": w,
                             "spearman_vs_unweighted_all": spearmanr(base, v, nan_policy="omit")[0]})
    return pd.DataFrame(rows)


def small_turbines(fleet):
    f = fleet.assign(small=fleet[P] <= SMALL_KW, small_kwh=fleet[E].where(fleet[P] <= SMALL_KW, 0))
    g = pd.concat([f, f.assign(ISO_code="EU")]).groupby("ISO_code")
    out = pd.DataFrame({"n_total": g.size(), "n_le50kW": g["small"].sum(),
                        "share_count_pct": 100 * g["small"].mean(),
                        "share_production_pct": 100 * g["small_kwh"].sum() / g[E].sum(),
                        "mean_gwp_le50kW_g": 1000 * f[f["small"]].groupby("ISO_code")[GWP].mean()})
    out.loc["EU", "mean_gwp_le50kW_g"] = 1000 * f.loc[f["small"], GWP].mean()
    return out[out["n_le50kW"] > 0].sort_values("n_le50kW", ascending=False)


def validation_metrics(y_true, y_pred):
    """Same formulas as the validation notebook (cell 184)."""
    err = y_pred - y_true
    return {"n_countries": len(y_true),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MPE_pct": 100 * np.mean(err / y_true),
            "MAPE_pct": 100 * np.mean(np.abs(err) / np.abs(y_true)),
            "R2": r2_score(y_true, y_pred),
            "spearman_rho": spearmanr(y_true, y_pred)[0]}


def ecoinvent_validation(fleet, methods):
    """Model country means (onshore + offshore pooled, as in the notebook) vs ecoinvent.
    The ecoinvent markets are the same in both scenarios: <= 50 kW turbines are < 0.13 % of
    production even in DK, so they do not move the production shares."""
    ei = pd.read_csv(ECOINVENT_CSV, index_col="ISO_code")
    metrics, per_country = [], []
    for scen, f in scenarios(fleet).items():
        for w, col in WEIGHTS.items():
            model = grouped_average(f, "ISO_code", col, methods)
            for m in methods:
                d = pd.DataFrame({"ecoinvent": ei[m], "model": model[m]}).dropna()
                d = d[d["ecoinvent"] != 0]
                key = {"method": m, "abbr": ABBR.get(m, ""), "scenario": scen, "model_weighting": w}
                metrics.append({**key, **validation_metrics(d["ecoinvent"], d["model"])})
                per_country.append(d.assign(**key, rel_diff_pct=100 * (d["model"] / d["ecoinvent"] - 1)))
    return pd.DataFrame(metrics), pd.concat(per_country).rename_axis("ISO_code").reset_index()


def weighted_median(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return x[o][np.searchsorted(cw, 0.5 * cw[-1])]


def distributional_consistency(fleet, methods):
    """by_turbine: share of turbines within +-x % of the median (manuscript).
    by_production: share of lifetime kWh within +-x % of the production-weighted median."""
    rows = []
    for scen, f in scenarios(fleet).items():
        w = f[E].to_numpy()
        for m in methods:
            v = f[m].to_numpy()
            med, wmed = np.median(v), weighted_median(v, w)
            row = {"method": m, "abbr": ABBR.get(m, ""), "scenario": scen}
            for b in BANDS:
                row[f"by_turbine_within_{b}pct"] = 100 * np.mean(np.abs(v - med) <= b / 100 * abs(med))
                row[f"by_production_within_{b}pct"] = \
                    100 * w[np.abs(v - wmed) <= b / 100 * abs(wmed)].sum() / w.sum()
            rows.append(row)
    return pd.DataFrame(rows)


def nuts3_summary(fleet, methods):
    nuts = gpd.read_file(NUTS_PATH).to_crs("EPSG:4326")
    nuts3 = nuts.loc[nuts["LEVL_CODE"] == 3, ["NUTS_ID", "geometry"]]
    rows = []
    for scen, f in scenarios(fleet).items():
        on = f[~f["Offshore"]]
        pts = gpd.GeoDataFrame(on, geometry=gpd.points_from_xy(on["Longitude"], on["Latitude"]),
                               crs="EPSG:4326")
        j = pd.DataFrame(gpd.sjoin(pts, nuts3, how="inner", predicate="within"))
        for w, col in WEIGHTS.items():
            s = grouped_average(j, "NUTS_ID", col, methods)
            rows += [{"method": m, "abbr": ABBR.get(m, ""), "scenario": scen, "weighting": w,
                      "n_regions": s[m].notna().sum(), "min": s[m].min(), "max": s[m].max(),
                      "lowest_region": s[m].idxmin(), "highest_region": s[m].idxmax()}
                     for m in methods]
    return pd.DataFrame(rows)


def check_weighting(fleet, methods):
    """Production-weighted average == total impact / total kWh, and the unweighted one is the
    plain mean, for every method, EU-wide and per country."""
    on = fleet[~fleet["Offshore"]]
    assert (average(on, None, methods) == on[methods].mean()).all()
    total = on[methods].mul(on[E], axis=0)
    assert np.allclose(average(on, E, methods), total.sum() / on[E].sum(), rtol=1e-12)
    per_country = total.groupby(on["ISO_code"]).sum().div(on.groupby("ISO_code")[E].sum(), axis=0)
    assert np.allclose(grouped_average(on, "ISO_code", E, methods), per_country, rtol=1e-12)


def check_manuscript6(ev, dc):
    """Unweighted, all turbines: must reproduce manuscript 6, Sect. 3.4.2 and 3.4.3."""
    v = ev[(ev["scenario"] == "all") & (ev["model_weighting"] == "unweighted")].set_index("abbr")
    assert round(v.loc["CC", "spearman_rho"], 3) == 0.323
    assert round(v.loc["LU", "MPE_pct"]) == round(v.loc["LU", "MAPE_pct"]) == 194
    assert round(v.loc["IR", "MAPE_pct"]) == 70
    assert all(33 <= round(v.loc[a, "MAPE_pct"]) <= 44 for a in ("HTC", "HTN", "MRM"))
    d = dc[(dc["scenario"] == "all") & (dc["abbr"] != "")].set_index("abbr")
    assert round(d.loc["CC", "by_turbine_within_25pct"]) == 61
    assert round(d.loc["CC", "by_turbine_within_50pct"]) == 81
    assert round(d.loc["MRM", "by_turbine_within_25pct"]) == round(d["by_turbine_within_25pct"].min()) == 53
    assert round(d.loc["LU", "by_turbine_within_25pct"]) == round(d["by_turbine_within_25pct"].max()) == 74
    assert 80 <= round(d["by_turbine_within_50pct"].min()) and round(d["by_turbine_within_50pct"].max()) <= 90


def main():
    fleet = load_fleet_results()
    methods = method_columns(fleet)
    check_weighting(fleet, methods)

    ev, ev_country = ecoinvent_validation(fleet, methods)
    dc = distributional_consistency(fleet, methods)
    check_manuscript6(ev, dc)

    cm = country_means(fleet, methods)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df, idx in [("eu_means", eu_means(fleet, methods), False),
                          ("country_means", cm, False),
                          ("country_ranking_stability", ranking_stability(cm), False),
                          ("small_turbines", small_turbines(fleet), True),
                          ("ecoinvent_validation", ev, False),
                          ("ecoinvent_country", ev_country, False),
                          ("distributional_consistency", dc, False),
                          ("nuts3_summary", nuts3_summary(fleet, methods), False)]:
        df.to_csv(OUT / f"{name}.csv", index=idx)
    print(f"Checks passed (weighting, manuscript 6). CSVs written to {OUT}")


if __name__ == "__main__":
    main()
