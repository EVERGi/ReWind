# What the utilized materials actually contribute to impact

Companion to `MATERIAL_IMPORTANCE.md`, which ranks materials by *statistical driver* (does a material's mass variation across turbines predict impact variation). This answers a different, more literal question: how much of a turbine's actual impact is caused by producing each material, in real units (kg CO2-Eq, etc.), not a predictive ranking.

`contribution = mass(turbine, material) x impact_per_kg(material)`. Material masses are the same ones `generate_material_importance.py` already computes (pure algebra, no new LCA run). The per-kg factors are new: one background LCA computation per material against the real ecoinvent database (`generate_material_contribution.py`), 17 materials x 25 impact methods, cached in `impact_per_kg_factors.csv`. Verified first that none of these 17 materials' background activities vary by country in this ecoinvent version (steel, cement, and everything else all resolve to the same GLO/regional market for every location tested) -- so one global factor per material covers the whole fleet correctly, no per-country computation needed.

## Impact intensity per kg (GWP100)

![Impact per kg by material](figures/material_contribution/fig_factor_comparison.png)

Highest impact-per-kg: **Aluminium** (13.83 kg CO2-Eq/kg), **Tin** (10.38 kg CO2-Eq/kg), **Fiberglass** (8.91 kg CO2-Eq/kg). Lowest: **Reinforcing steel (rebar)** (-0.014 kg CO2-Eq/kg), **Iron ore** (0.009 kg CO2-Eq/kg), **Concrete** (0.123 kg CO2-Eq/kg).

## Does this explain MATERIAL_IMPORTANCE.md's monopile finding?

- **Fiberglass**: 8.915 kg CO2-Eq/kg, contributes 787,340.72 kg CO2-Eq mean (rank 2 of 15 by contribution)
- **Epoxy resin**: 4.988 kg CO2-Eq/kg, contributes 10,168.19 kg CO2-Eq mean (rank 8 of 15 by contribution)
- **Steel (low-alloy)**: 1.979 kg CO2-Eq/kg, contributes 967,422.19 kg CO2-Eq mean (rank 1 of 15 by contribution)

## Mean GWP100 contribution, by material and siting

![Contribution composition](figures/material_contribution/fig_contribution_composition.png)

**Onshore**: top contributor is **Steel (low-alloy)** (37.4% of captured GWP100, 450,916.69 kg CO2-Eq mean).

**Offshore – monopile**: top contributor is **Steel (low-alloy)** (37.7% of captured GWP100, 967,422.19 kg CO2-Eq mean).

**Offshore – semi-submersible**: top contributor is **Steel (low-alloy)** (74.0% of captured GWP100, 6,124,026.08 kg CO2-Eq mean).

## Full contribution share, all 25 impact categories

![Contribution share heatmap](figures/material_contribution/fig_contribution_heatmap.png)

Each column is one impact category, normalized to sum to 1 -- same convention as `MATERIAL_IMPORTANCE.md`'s heatmaps, but color here is actual share of impact, not predictive importance.

## Full breakdown, all 25 impact categories

Every impact category has its own unit (not all kg CO2-Eq like the GWP100-only sections above) -- shown per row below rather than assumed.

### Onshore

| Impact category | Unit | Top contributor | Value | Share of total |
|---|---|---|---|---|
| Acidification | mol H+-Eq | Steel (low-alloy) | 2,037 | 29.2% |
| Climate change (total) | kg CO2-Eq | Steel (low-alloy) | 4.509e+05 | 37.4% |
| Climate change (biogenic) | kg CO2-Eq | Steel (low-alloy) | 393.5 | 30.9% |
| Climate change (fossil) | kg CO2-Eq | Steel (low-alloy) | 4.502e+05 | 37.4% |
| Climate change (land use) | kg CO2-Eq | Steel (low-alloy) | 299.2 | 41.1% |
| Ecotoxicity, freshwater | CTUe | Steel (low-alloy) | 2.468e+06 | 37.3% |
| Ecotoxicity, freshwater (inorg.) | CTUe | Steel (low-alloy) | 2.313e+06 | 37.1% |
| Ecotoxicity, freshwater (org.) | CTUe | Steel (low-alloy) | 1.551e+05 | 40.5% |
| Fossil resource use | MJ, net calorific value | Steel (low-alloy) | 4.77e+06 | 35.8% |
| Eutrophication, freshwater | kg P-Eq | Steel (low-alloy) | 219.3 | 43.9% |
| Eutrophication, marine | kg N-Eq | Steel (low-alloy) | 520 | 37.1% |
| Eutrophication, terrestrial | mol N-Eq | Steel (low-alloy) | 4,736 | 38.3% |
| Human toxicity, carcinogenic | CTUh | Steel (chromium) | 0.004893 | 51.3% |
| Human toxicity, carc. (inorg.) | CTUh | Steel (chromium) | 0.004703 | 61.3% |
| Human toxicity, carc. (org.) | CTUh | Steel (low-alloy) | 0.001499 | 80.0% |
| Human toxicity, non-carc. | CTUh | Copper | 0.02417 | 50.5% |
| Human toxicity, non-carc. (inorg.) | CTUh | Copper | 0.023 | 49.8% |
| Human toxicity, non-carc. (org.) | CTUh | Copper | 0.001165 | 69.8% |
| Ionising radiation | kBq U235-Eq | Steel (low-alloy) | 1.463e+04 | 29.9% |
| Land use | dimensionless | Steel (low-alloy) | 1.566e+06 | 38.5% |
| Mineral resource use | kg Sb-Eq | Copper | 23.48 | 69.7% |
| Ozone depletion | kg CFC-11-Eq | Steel (low-alloy) | 0.007865 | 52.0% |
| Particulate matter | disease incidence | Steel (low-alloy) | 0.03919 | 45.8% |
| Photochemical ozone formation | kg NMVOC-Eq | Steel (low-alloy) | 2,146 | 45.7% |
| Water use | m3 world eq. deprived | Fiberglass | 2.825e+05 | 51.3% |

#### Acidification

![Acidification, Onshore](figures/material_contribution/onshore/fig_contribution_acidification.png)

#### Climate change (total)

![Climate change (total), Onshore](figures/material_contribution/onshore/fig_contribution_climate_change_total.png)

#### Climate change (biogenic)

![Climate change (biogenic), Onshore](figures/material_contribution/onshore/fig_contribution_climate_change_biogenic.png)

#### Climate change (fossil)

![Climate change (fossil), Onshore](figures/material_contribution/onshore/fig_contribution_climate_change_fossil.png)

#### Climate change (land use)

![Climate change (land use), Onshore](figures/material_contribution/onshore/fig_contribution_climate_change_land_use.png)

#### Ecotoxicity, freshwater

![Ecotoxicity, freshwater, Onshore](figures/material_contribution/onshore/fig_contribution_ecotoxicity_freshwater.png)

#### Ecotoxicity, freshwater (inorg.)

![Ecotoxicity, freshwater (inorg.), Onshore](figures/material_contribution/onshore/fig_contribution_ecotoxicity_freshwater_inorganics.png)

#### Ecotoxicity, freshwater (org.)

![Ecotoxicity, freshwater (org.), Onshore](figures/material_contribution/onshore/fig_contribution_ecotoxicity_freshwater_organics.png)

#### Fossil resource use

![Fossil resource use, Onshore](figures/material_contribution/onshore/fig_contribution_fossil_resource_use.png)

#### Eutrophication, freshwater

![Eutrophication, freshwater, Onshore](figures/material_contribution/onshore/fig_contribution_eutrophication_freshwater.png)

#### Eutrophication, marine

![Eutrophication, marine, Onshore](figures/material_contribution/onshore/fig_contribution_eutrophication_marine.png)

#### Eutrophication, terrestrial

![Eutrophication, terrestrial, Onshore](figures/material_contribution/onshore/fig_contribution_eutrophication_terrestrial.png)

#### Human toxicity, carcinogenic

![Human toxicity, carcinogenic, Onshore](figures/material_contribution/onshore/fig_contribution_human_toxicity_carcinogenic.png)

#### Human toxicity, carc. (inorg.)

![Human toxicity, carc. (inorg.), Onshore](figures/material_contribution/onshore/fig_contribution_human_toxicity_carcinogenic_inorganics.png)

#### Human toxicity, carc. (org.)

![Human toxicity, carc. (org.), Onshore](figures/material_contribution/onshore/fig_contribution_human_toxicity_carcinogenic_organics.png)

#### Human toxicity, non-carc.

![Human toxicity, non-carc., Onshore](figures/material_contribution/onshore/fig_contribution_human_toxicity_non_carcinogenic.png)

#### Human toxicity, non-carc. (inorg.)

![Human toxicity, non-carc. (inorg.), Onshore](figures/material_contribution/onshore/fig_contribution_human_toxicity_non_carcinogenic_inorganics.png)

#### Human toxicity, non-carc. (org.)

![Human toxicity, non-carc. (org.), Onshore](figures/material_contribution/onshore/fig_contribution_human_toxicity_non_carcinogenic_organics.png)

#### Ionising radiation

![Ionising radiation, Onshore](figures/material_contribution/onshore/fig_contribution_ionising_radiation.png)

#### Land use

![Land use, Onshore](figures/material_contribution/onshore/fig_contribution_land_use.png)

#### Mineral resource use

![Mineral resource use, Onshore](figures/material_contribution/onshore/fig_contribution_mineral_resource_use.png)

#### Ozone depletion

![Ozone depletion, Onshore](figures/material_contribution/onshore/fig_contribution_ozone_depletion.png)

#### Particulate matter

![Particulate matter, Onshore](figures/material_contribution/onshore/fig_contribution_particulate_matter.png)

#### Photochemical ozone formation

![Photochemical ozone formation, Onshore](figures/material_contribution/onshore/fig_contribution_photochemical_ozone_formation.png)

#### Water use

![Water use, Onshore](figures/material_contribution/onshore/fig_contribution_water_use.png)

### Offshore – monopile

| Impact category | Unit | Top contributor | Value | Share of total |
|---|---|---|---|---|
| Acidification | mol H+-Eq | Steel (low-alloy) | 4,371 | 30.5% |
| Climate change (total) | kg CO2-Eq | Steel (low-alloy) | 9.674e+05 | 37.7% |
| Climate change (biogenic) | kg CO2-Eq | Fiberglass | 900.7 | 31.7% |
| Climate change (fossil) | kg CO2-Eq | Steel (low-alloy) | 9.659e+05 | 37.7% |
| Climate change (land use) | kg CO2-Eq | Steel (low-alloy) | 641.9 | 37.9% |
| Ecotoxicity, freshwater | CTUe | Steel (low-alloy) | 5.296e+06 | 40.6% |
| Ecotoxicity, freshwater (inorg.) | CTUe | Steel (low-alloy) | 4.963e+06 | 40.6% |
| Ecotoxicity, freshwater (org.) | CTUe | Steel (low-alloy) | 3.328e+05 | 40.6% |
| Fossil resource use | MJ, net calorific value | Fiberglass | 1.185e+07 | 38.2% |
| Eutrophication, freshwater | kg P-Eq | Steel (low-alloy) | 470.6 | 47.7% |
| Eutrophication, marine | kg N-Eq | Steel (low-alloy) | 1,116 | 35.9% |
| Eutrophication, terrestrial | mol N-Eq | Steel (low-alloy) | 1.016e+04 | 39.3% |
| Human toxicity, carcinogenic | CTUh | Steel (chromium) | 0.01312 | 56.8% |
| Human toxicity, carc. (inorg.) | CTUh | Steel (chromium) | 0.01261 | 66.7% |
| Human toxicity, carc. (org.) | CTUh | Steel (low-alloy) | 0.003216 | 77.0% |
| Human toxicity, non-carc. | CTUh | Copper | 0.0338 | 37.8% |
| Human toxicity, non-carc. (inorg.) | CTUh | Copper | 0.03217 | 37.1% |
| Human toxicity, non-carc. (org.) | CTUh | Copper | 0.001629 | 59.4% |
| Ionising radiation | kBq U235-Eq | Steel (chromium) | 3.564e+04 | 35.8% |
| Land use | dimensionless | Steel (low-alloy) | 3.36e+06 | 39.2% |
| Mineral resource use | kg Sb-Eq | Copper | 32.84 | 57.9% |
| Ozone depletion | kg CFC-11-Eq | Steel (low-alloy) | 0.01687 | 53.7% |
| Particulate matter | disease incidence | Steel (low-alloy) | 0.08408 | 43.0% |
| Photochemical ozone formation | kg NMVOC-Eq | Steel (low-alloy) | 4,605 | 45.9% |
| Water use | m3 world eq. deprived | Fiberglass | 7.864e+05 | 59.2% |

#### Acidification

![Acidification, Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_acidification.png)

#### Climate change (total)

![Climate change (total), Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_climate_change_total.png)

#### Climate change (biogenic)

![Climate change (biogenic), Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_climate_change_biogenic.png)

#### Climate change (fossil)

![Climate change (fossil), Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_climate_change_fossil.png)

#### Climate change (land use)

![Climate change (land use), Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_climate_change_land_use.png)

#### Ecotoxicity, freshwater

![Ecotoxicity, freshwater, Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_ecotoxicity_freshwater.png)

#### Ecotoxicity, freshwater (inorg.)

![Ecotoxicity, freshwater (inorg.), Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_ecotoxicity_freshwater_inorganics.png)

#### Ecotoxicity, freshwater (org.)

![Ecotoxicity, freshwater (org.), Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_ecotoxicity_freshwater_organics.png)

#### Fossil resource use

![Fossil resource use, Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_fossil_resource_use.png)

#### Eutrophication, freshwater

![Eutrophication, freshwater, Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_eutrophication_freshwater.png)

#### Eutrophication, marine

![Eutrophication, marine, Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_eutrophication_marine.png)

#### Eutrophication, terrestrial

![Eutrophication, terrestrial, Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_eutrophication_terrestrial.png)

#### Human toxicity, carcinogenic

![Human toxicity, carcinogenic, Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_human_toxicity_carcinogenic.png)

#### Human toxicity, carc. (inorg.)

![Human toxicity, carc. (inorg.), Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_human_toxicity_carcinogenic_inorganics.png)

#### Human toxicity, carc. (org.)

![Human toxicity, carc. (org.), Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_human_toxicity_carcinogenic_organics.png)

#### Human toxicity, non-carc.

![Human toxicity, non-carc., Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_human_toxicity_non_carcinogenic.png)

#### Human toxicity, non-carc. (inorg.)

![Human toxicity, non-carc. (inorg.), Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_human_toxicity_non_carcinogenic_inorganics.png)

#### Human toxicity, non-carc. (org.)

![Human toxicity, non-carc. (org.), Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_human_toxicity_non_carcinogenic_organics.png)

#### Ionising radiation

![Ionising radiation, Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_ionising_radiation.png)

#### Land use

![Land use, Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_land_use.png)

#### Mineral resource use

![Mineral resource use, Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_mineral_resource_use.png)

#### Ozone depletion

![Ozone depletion, Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_ozone_depletion.png)

#### Particulate matter

![Particulate matter, Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_particulate_matter.png)

#### Photochemical ozone formation

![Photochemical ozone formation, Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_photochemical_ozone_formation.png)

#### Water use

![Water use, Offshore – monopile](figures/material_contribution/offshore_monopile/fig_contribution_water_use.png)

### Offshore – semi-submersible

| Impact category | Unit | Top contributor | Value | Share of total |
|---|---|---|---|---|
| Acidification | mol H+-Eq | Steel (low-alloy) | 2.767e+04 | 68.1% |
| Climate change (total) | kg CO2-Eq | Steel (low-alloy) | 6.124e+06 | 74.0% |
| Climate change (biogenic) | kg CO2-Eq | Steel (low-alloy) | 5,345 | 66.0% |
| Climate change (fossil) | kg CO2-Eq | Steel (low-alloy) | 6.115e+06 | 74.0% |
| Climate change (land use) | kg CO2-Eq | Steel (low-alloy) | 4,063 | 73.7% |
| Ecotoxicity, freshwater | CTUe | Steel (low-alloy) | 3.352e+07 | 77.4% |
| Ecotoxicity, freshwater (inorg.) | CTUe | Steel (low-alloy) | 3.142e+07 | 77.5% |
| Ecotoxicity, freshwater (org.) | CTUe | Steel (low-alloy) | 2.107e+06 | 75.1% |
| Fossil resource use | MJ, net calorific value | Steel (low-alloy) | 6.478e+07 | 69.4% |
| Eutrophication, freshwater | kg P-Eq | Steel (low-alloy) | 2,979 | 82.1% |
| Eutrophication, marine | kg N-Eq | Steel (low-alloy) | 7,062 | 72.4% |
| Eutrophication, terrestrial | mol N-Eq | Steel (low-alloy) | 6.432e+04 | 75.5% |
| Human toxicity, carcinogenic | CTUh | Steel (low-alloy) | 0.05011 | 70.0% |
| Human toxicity, carc. (inorg.) | CTUh | Steel (low-alloy) | 0.02975 | 59.7% |
| Human toxicity, carc. (org.) | CTUh | Steel (low-alloy) | 0.02036 | 93.8% |
| Human toxicity, non-carc. | CTUh | Steel (low-alloy) | 0.1404 | 63.1% |
| Human toxicity, non-carc. (inorg.) | CTUh | Steel (low-alloy) | 0.1367 | 63.2% |
| Human toxicity, non-carc. (org.) | CTUh | Steel (low-alloy) | 0.003652 | 59.7% |
| Ionising radiation | kBq U235-Eq | Steel (low-alloy) | 1.987e+05 | 68.3% |
| Land use | dimensionless | Steel (low-alloy) | 2.127e+07 | 75.2% |
| Mineral resource use | kg Sb-Eq | Steel (low-alloy) | 44.4 | 43.2% |
| Ozone depletion | kg CFC-11-Eq | Steel (low-alloy) | 0.1068 | 84.9% |
| Particulate matter | disease incidence | Steel (low-alloy) | 0.5322 | 77.6% |
| Photochemical ozone formation | kg NMVOC-Eq | Steel (low-alloy) | 2.915e+04 | 80.0% |
| Water use | m3 world eq. deprived | Steel (low-alloy) | 2.047e+06 | 59.5% |

#### Acidification

![Acidification, Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_acidification.png)

#### Climate change (total)

![Climate change (total), Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_climate_change_total.png)

#### Climate change (biogenic)

![Climate change (biogenic), Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_climate_change_biogenic.png)

#### Climate change (fossil)

![Climate change (fossil), Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_climate_change_fossil.png)

#### Climate change (land use)

![Climate change (land use), Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_climate_change_land_use.png)

#### Ecotoxicity, freshwater

![Ecotoxicity, freshwater, Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_ecotoxicity_freshwater.png)

#### Ecotoxicity, freshwater (inorg.)

![Ecotoxicity, freshwater (inorg.), Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_ecotoxicity_freshwater_inorganics.png)

#### Ecotoxicity, freshwater (org.)

![Ecotoxicity, freshwater (org.), Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_ecotoxicity_freshwater_organics.png)

#### Fossil resource use

![Fossil resource use, Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_fossil_resource_use.png)

#### Eutrophication, freshwater

![Eutrophication, freshwater, Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_eutrophication_freshwater.png)

#### Eutrophication, marine

![Eutrophication, marine, Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_eutrophication_marine.png)

#### Eutrophication, terrestrial

![Eutrophication, terrestrial, Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_eutrophication_terrestrial.png)

#### Human toxicity, carcinogenic

![Human toxicity, carcinogenic, Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_human_toxicity_carcinogenic.png)

#### Human toxicity, carc. (inorg.)

![Human toxicity, carc. (inorg.), Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_human_toxicity_carcinogenic_inorganics.png)

#### Human toxicity, carc. (org.)

![Human toxicity, carc. (org.), Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_human_toxicity_carcinogenic_organics.png)

#### Human toxicity, non-carc.

![Human toxicity, non-carc., Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_human_toxicity_non_carcinogenic.png)

#### Human toxicity, non-carc. (inorg.)

![Human toxicity, non-carc. (inorg.), Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_human_toxicity_non_carcinogenic_inorganics.png)

#### Human toxicity, non-carc. (org.)

![Human toxicity, non-carc. (org.), Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_human_toxicity_non_carcinogenic_organics.png)

#### Ionising radiation

![Ionising radiation, Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_ionising_radiation.png)

#### Land use

![Land use, Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_land_use.png)

#### Mineral resource use

![Mineral resource use, Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_mineral_resource_use.png)

#### Ozone depletion

![Ozone depletion, Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_ozone_depletion.png)

#### Particulate matter

![Particulate matter, Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_particulate_matter.png)

#### Photochemical ozone formation

![Photochemical ozone formation, Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_photochemical_ozone_formation.png)

#### Water use

![Water use, Offshore – semi-submersible](figures/material_contribution/offshore_semi-submersible/fig_contribution_water_use.png)

## Limitations

Impact-per-kg factors come from `lca_algebraic.compute_impacts()` on each material's background market activity directly -- the same ecoinvent-391-cutoff database `fleet_evaluation_lca_algebraic.py` uses, but computed once per material rather than read back from a fleet run. Covers the same 17 material buckets `MATERIAL_IMPORTANCE.md` does (same Input-phase-only scope: no transformer units, no land use, no process/service exchanges -- see that file's own Limitations). Offshore spar (n=2) is computed but not separately reported, same threshold as elsewhere in this project.
