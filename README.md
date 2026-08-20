# ReWind

Regionalized, cradle-to-grave life cycle assessment (LCA) model for onshore and offshore
wind turbines in Europe. ReWind combines turbine-specific technical parameters with
geographically resolved inventory data (grid connection distance, transport distance,
sea depth, foundation type) to estimate environmental impacts for individual turbines
and for the full European fleet.

This repository supports the paper *Integrating geographic data into life cycle
assessment: a spatial analysis of European wind turbines* (Huber et al.).

## Requirements

- Python 3.10 or later
- GDAL and PROJ (system libraries required by geopandas)
- A Brightway2 project with the ecoinvent database imported (see "Data you need to
  provide" below)

## Installation

```bash
python3 -m venv rewind_env
source rewind_env/bin/activate
pip install -r requirements.txt
```

`geopandas` depends on GDAL and PROJ. If `pip install` fails to build them, install
`geopandas` and `pyogrio` through your system package manager or conda first
(`conda install geopandas`), then `pip install` the rest.

## Data you need to provide

None of the following are included in this repository. Two are licensed and cannot be
redistributed; the other two are free but too large to bundle.

| File | Why it's not here | Where to get it |
|---|---|---|
| `REWIND/REWIND/data/EU_turbines_input_data.xlsx` | Licensed turbine register | Contact the data provider |
| `REWIND/REWIND/data/datasets/` | ecoinvent 3.9.1, licensed | [ecoinvent.org](https://ecoinvent.org) |
| `Shared_Rewind/wind_fleet_data_incl_sea_depths_corrected.csv` | Large derived file | Provided with the Zenodo release |
| `Shared_Rewind/Fleet_results/NUTS_RG_20M_2024_4326.gpkg` | Large, freely available | [Eurostat GISCO](https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/territorial-units-statistics) |

GEBCO bathymetry is not needed to run the scripts: sea depth is already computed and
stored in `REWIND/REWIND/data/geo_precomputed/`.

## Running the pipeline

`regenerate_all.sh` runs the full analysis, from turbine-level LCA results through
every figure and summary table used in the paper. Run it from the repository root with
the environment above active:

```bash
./regenerate_all.sh
```

This takes the licensed turbine register and ecoinvent database as input and produces:
fleet-level impact results per country and siting type, permutation feature importance,
material importance and contribution analysis, summary statistics (variability,
extreme-turbine analysis, validation against ecoinvent), and every figure in the paper.

If you already have `REWIND/REWIND/data/results/` populated (for example, from the
Zenodo release) and only want to regenerate figures and statistics, skip the slow LCA
step:

```bash
./regenerate_all.sh --skip-lca
```

The script checks for required input files before running and stops with a clear
message if something is missing, rather than failing partway through.

## Repository layout

- `REWIND/REWIND/` — core LCA package: inventory construction, scaling formulas, and
  the local (gitignored) data directory
- `Shared_Rewind/` — external supporting data (gitignored, see table above)
- `figures/` — all generated figures, organized by analysis
- `*.md` — methodology notes for specific analyses (feature importance, material
  contribution, structural dependencies, summary statistics)
- `regenerate_all.sh` — runs the full pipeline end to end

## Reproducibility

Full reproduction of the European fleet assessment requires the licensed ecoinvent
database and turbine register, so it is not possible from this repository alone. Once
those inputs are in place, `regenerate_all.sh` reproduces every result and figure in
the paper. The Zenodo archive (below) provides the processed fleet-level results, so
published numbers can be checked without needing licensed access to the raw inputs.

## Data availability

Processed results and analysis outputs are archived on Zenodo:
DOI: [10.5281/zenodo.17857554](https://doi.org/10.5281/zenodo.17857554)

## Citation

```bibtex
@misc{ReWind2026,
  author       = {Huber, Dominik},
  title        = {Climate change impacts and annual electricity
                   production of all wind turbines installed in
                   Europe until 2020},
  month        = dec,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {0.2},
  doi          = {10.5281/zenodo.17857554},
  url          = {https://doi.org/10.5281/zenodo.17857554},
}
```

Associated publication: Huber et al. (2026). *Integrating geographic data into life
cycle assessment: a spatial analysis of European wind turbines*. International Journal
of Life Cycle Assessment.

## License

BSD 3-Clause License. See `LICENSE` for full terms.
