# Notes for Dominik: weighting and small-turbine sensitivity

This covers Reviewer 2, point 2 (unweighted averages, and the check without turbines of 50 kW and below).

## What was done

Every fleet, country and NUTS3 average was recomputed three ways, for all 25 EF v3.1 categories:

- **unweighted**: the plain mean over turbines. This is what manuscript 6 uses.
- **production-weighted**: weighted by lifetime kWh, i.e. total impact / total kWh. Use this as the main result.
- **capacity-weighted**: weighted by rated power. This is only a robustness check.

Each of these is computed for all turbines, and again without turbines of **50 kW and below** (871 turbines).

The ecoinvent country markets were rebuilt for all 25 categories using the rules from your validation notebook.

**Check:** with no weighting, the scripts reproduce manuscript 6 exactly.

## How to run

```bash
conda activate rewind
python generate_ecoinvent_markets.py
python generate_weighting_sensitivity.py
```

Rerun both whenever the fleet results change, in particular after the foundation-type fix (Reviewer 2, point 1).

## Files for the figures

All in `REWIND/REWIND/data/summary_stats/weighting_sensitivity/`.

| Use | File | Filter |
|---|---|---|
| Fig. 6 heatmap | `ecoinvent_country.csv` | `scenario == "all"`, `model_weighting == "production_weighted"`, six `abbr`. Plot `rel_diff_pct`. |
| Fig. 7 (±25 / ±50 %) | `distributional_consistency.csv` | `by_turbine_*` = current figure; `by_production_*` = weighted version |
| Validation stats (all 25 categories) | `ecoinvent_validation.csv` | |
| Numbers in the text | `eu_means.csv`, `country_means.csv` | |

Filtering on `model_weighting == "unweighted"` gives back the current figures, so you can check them before switching.

## Main results

- **EU averages barely change.** CC onshore goes from 21.7 to 21.5 g/kWh and offshore from 17.3 to 17.9. Across all 25 categories the change is within about ±7 %.
- **Denmark onshore changes a lot.** It drops from 31.8 to 16.0 g/kWh, because 806 small Danish turbines pull up the unweighted mean. Sect. 3.4.1 (DK 31 g vs Sacchi) needs rewriting.
- **Agreement with ecoinvent improves.** With production weighting, MAPE goes down for 19 of 25 categories. For CC, MAPE goes from 21.9 to 18.2 % and ρ from 0.32 to 0.44. LU (169 %) and IR (65 %) still differ a lot.
- **Removing turbines of 50 kW and below changes almost nothing once results are weighted.** These turbines are 1.1 % of the fleet but only 0.007 % of production.

## Decisions taken

1. **50 kW cut-off:** turbines of 50 kW and below are excluded (≤ 50 kW, inclusive), matching the manuscript's "≤50 kW".
2. **The ecoinvent side is the same in both scenarios.** Removing turbines of 50 kW and below would only shift Denmark's <1 MW production share by under 0.13 %, so the markets were not rebuilt for that scenario.
3. **±25 / ±50 % by production** counts lifetime kWh rather than turbines, and is measured around the production-weighted median.

## Left as is (from the original setup)

- **Slovenia:** ecoinvent only has a >3 MW onshore dataset, so the whole fleet is mapped to it. This explains Slovenia's large gap.
- **Offshore classes:** offshore <1 MW and >3 MW turbines are mapped to the 1–3 MW offshore dataset.
- **Negative values:** 56 turbines have negative values for human toxicity (carcinogenic, organics). None of the six focus categories is affected.

## To check

- **NUTS3 boundaries:** the new summary uses the 2016 NUTS file (which includes the UK). `generate_paper_figures.py` still uses the 2024 file, which has no UK regions. (not important if we are not regenerating maps).

- **ecoinvent licence:** the new CSVs contain impact scores derived from ecoinvent, in case of any licensing stuff. 
