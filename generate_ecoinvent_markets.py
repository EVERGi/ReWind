"""
generate_ecoinvent_markets.py

Rebuilds the country-specific ecoinvent wind electricity markets (manuscript Sect. 3.4.2) for
all 25 EF v3.1 methods, with the rules of Fleet_basic_charts_validation.ipynb (cells 112-147):

  1. Size class per turbine: <1MW (P < 1000 kW), 1-3MW (<= 3000 kW), >3MW; onshore/offshore.
     Offshore <1MW and >3MW are mapped to 1-3MW offshore.
  2. share(country, class) = lifetime kWh of the class / lifetime kWh of the country.
  3. Each share links to "electricity production, wind, <class>" in that country. A missing
     onshore class falls back to another onshore class of the same country; classes without
     any dataset are dropped and shares are not renormalised.
  4. Market impact = sum(share x dataset impact), equal to the notebook's LCA of the assembled
     market since LCA is linear. Nothing is written to the Brightway project.

The climate-change result must match the notebook's 31 country values
(validation/ecoinvent_wind_market_cc_by_country.csv), otherwise the script stops.

Outputs (REWIND/REWIND/data/validation/):
  ecoinvent_wind_market_by_country_all_methods.csv  impact per kWh, one column per method
  ecoinvent_wind_market_shares.csv                  production shares and linked classes

Needs the 'wimby' Brightway project (ecoinvent 3.9.1 cutoff).
    python generate_ecoinvent_markets.py
"""
from pathlib import Path

import bw2calc as bc
import bw2data as bd
import numpy as np
import pandas as pd

from generate_paper_summary_stats import GWP, load_fleet_results, method_columns

VAL = Path(__file__).resolve().parent / "REWIND" / "REWIND" / "data" / "validation"
OUT_CSV = VAL / "ecoinvent_wind_market_by_country_all_methods.csv"
SHARES_CSV = VAL / "ecoinvent_wind_market_shares.csv"
CC_REF_CSV = VAL / "ecoinvent_wind_market_cc_by_country.csv"
E = "Lifetime_production_kWh"
PREFIX = "electricity production, wind, "

OFFSHORE_MAP = {">3MW turbine, offshore": "1-3MW turbine, offshore",
                "<1MW turbine, offshore": "1-3MW turbine, offshore"}
ONSHORE_FALLBACK = {"<1MW turbine, onshore": ["1-3MW turbine, onshore", ">3MW turbine, onshore"],
                    "1-3MW turbine, onshore": [">3MW turbine, onshore", "<1MW turbine, onshore"],
                    ">3MW turbine, onshore": ["1-3MW turbine, onshore", "<1MW turbine, onshore"]}


def production_shares(fleet):
    p = fleet["P_rated_kW"]
    size = pd.Series(np.where(p < 1000, "<1MW", np.where(p <= 3000, "1-3MW", ">3MW")), index=p.index)
    cls = size + " turbine, " + fleet["Offshore"].map({True: "offshore", False: "onshore"})
    kwh = fleet.groupby(["ISO_code", cls.replace(OFFSHORE_MAP).rename("mapped_class")])[E].sum()
    return (kwh / kwh.groupby(level="ISO_code").transform("sum")).rename("share").reset_index()


def main():
    bd.projects.set_current("wimby")
    fleet = load_fleet_results()
    cols = method_columns(fleet)
    by_label = {f"{m[1]} - {m[2]}[{bd.Method(m).metadata.get('unit', '')}]": m
                for m in bd.methods if m[0] == "EF v3.1"}
    wind = {(a["location"], a["name"][len(PREFIX):]): a
            for a in bd.Database("ecoinvent-391-cutoff") if a["name"].startswith(PREFIX)}

    shares = production_shares(fleet)
    shares["linked_class"] = [next((c for c in [cls] + ONSHORE_FALLBACK.get(cls, [])
                                    if (iso, c) in wind), None)
                              for iso, cls in zip(shares["ISO_code"], shares["mapped_class"])]
    shares.to_csv(SHARES_CSV, index=False)
    linked = shares.dropna(subset=["linked_class"])

    # impact per kWh of every linked ecoinvent dataset, all methods
    acts = {k: wind[k] for k in set(zip(linked["ISO_code"], linked["linked_class"]))}
    lca = bc.LCA({next(iter(acts.values())): 1}, by_label[GWP])
    lca.lci()
    scores = {}
    for k, act in acts.items():
        lca.redo_lci({act.id: 1})
        scores[k] = {}
        for c in cols:
            lca.switch_method(by_label[c])
            lca.lcia()
            scores[k][c] = lca.score

    contrib = pd.DataFrame([scores[k] for k in zip(linked["ISO_code"], linked["linked_class"])],
                           index=linked.index).mul(linked["share"], axis=0)
    market = contrib.groupby(linked["ISO_code"]).sum()
    market.to_csv(OUT_CSV)

    ref = pd.read_csv(CC_REF_CSV, index_col="ISO_code")["ecoinvent_cc_g_per_kWh"]
    diff = (1000 * market[GWP].reindex(ref.index) - ref).abs().max()
    assert diff < 1e-5, f"climate change differs from the notebook by {diff} g/kWh"
    print(f"{len(market)} country markets x {len(cols)} methods written to {OUT_CSV.name}; "
          f"climate change matches the notebook (max diff {diff:.1e} g/kWh). "
          f"No ecoinvent wind dataset: {sorted(set(fleet['ISO_code']) - set(market.index))}")


if __name__ == "__main__":
    main()
