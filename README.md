# ReWind
Regionalized cradle-to-grave life cycle assessment (LCA) model for on- and offshore wind energy in Europe.

## Overview

`ReWind` is a Python package and set of scripts to perform regionalized cradle-to-grave life cycle assessments for onshore and offshore wind projects in Europe. The code assembles component inventories, applies region-specific scaling and calculation methods, and produces impact estimates suitable for comparative analysis and research.

Key features
- On- & offshore wind energy
- Introduction of novel offshore floating foundations
- Regionalized inventory preparation
- Support for onshore and offshore wind technologies
- Reproducible, scriptable workflows for batch processing

## Installation

Prerequisites
- Python 3.8+ (recommend 3.10 or later)
- System libraries for geospatial Python packages (GDAL, PROJ)

Typical installation (editable install for development):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

If the project provides a `requirements.txt` or uses `pyproject.toml`, install dependencies accordingly (for example `pip install -r requirements.txt` or `pip install .`). Installing `geopandas` and related packages may require system dependencies on Windows (GDAL/PROJ); consult your package manager or conda for an easier install (`conda install geopandas gdal rasterio`).

## How to Run

The repository includes `example.py` which demonstrates a minimal run. The simplest way to get started is:

```powershell
# Activate environment
.\.venv\Scripts\activate

# Run the example script
python example.py
```

For custom runs, inspect and adapt `example.py` or import the package in your own scripts. The main package code is in the `REWIND` package directory.

## Data Requirements

Place required data files under `REWIND/data/` or provide a path to your data when calling the scripts. Required items typically include:
- `buses.csv` (grid / region mapping)
- Shapefiles for country/region boundaries (all shapefile components: `.shp`, `.shx`, `.dbf`, etc.)
- Any inventory CSVs or lookup tables used by `prepare_inventories.py` and `built_inventory.py`

Notes
- Shapefiles must be complete (all component files present) and encoded in a common CRS (WGS84 recommended).
- Large datasets (GIS, country-level inventories) can be heavy—ensure sufficient disk space and memory.
- Check data licenses before redistribution; some sources in `REWIND/data/` may have restrictions.

## Not included in this repository

- ecoinvent database, cut-off version 3.9.1 (licensed)
- GEBCO bathymetry (can be downloaded separately via: **GEBCO global bathymetry dataset (2024)**  
  https://www.gebco.net/data_and_products/gridded_bathymetry_data/, File used: `GEBCO_2024_sub_ice_topo.nc`)

## Reproducibility

Due to licensing restrictions (e.g. ecoinvent) and the size of certain external datasets (e.g. bathymetry data), full reproduction of the European fleet assessment is not possible using this repository alone. 

However, once the required external inputs — namely the ecoinvent database and the GEBCO bathymetry dataset — are provided in the `/data` directory (or the corresponding input paths defined in the scripts), the model can be fully executed using the supplied scripts. The included example workflow enables users to run the model on a reduced dataset and verify the implementation and calculation logic. 

In addition, the Zenodo archive provides the processed fleet-level datasets used in this study, allowing validation of the reported results and facilitating direct comparison with published values.


## Example Workflow

1. Create and activate a Python virtual environment.
2. Install the package and dependencies (see Installation).
3. Place the required input data in `REWIND/data/` or update paths in `example.py`.
4. Run the example script:

```powershell
python example.py
```

5. Inspect outputs (console, CSVs or output folder used by the script). Adapt parameters and rerun for other regions or scenarios.

## Limitations (VERY IMPORTANT)

- Geographic scope: The model and bundled data are configured for Europe; applying them outside Europe may produce invalid results.
- Spatial resolution: Many regionalizations use coarse mappings and assumptions; results are intended for comparative research, not detailed site-level engineering.
- Inventory completeness: Some component inventories use proxies or literature averages where itemized, measured data are not available.
- Validation: The model has been validated on a country-level accross Europe. Results are provided in the paper.
- External dependencies: Geospatial packages (e.g. `geopandas`, `rasterio`) may require system-level libraries which are outside of Python's control.
- Data licensing: Some input datasets may be proprietary or have redistribution limits — verify each dataset's license before sharing derived outputs.

## Data Availability
The skript to analyze the data along with obtained results is provided via the following link on Zenodo:

	DOI: 10.5281/zenodo.17857554


## Citation

Please cite the project and any associated Zenodo record.

```bibtex
@misc{ReWind2026,
  author       = {Huber, Dominik},
  title        = {Climate change impacts and annual electricity
                   production of all wind turbines installed in
                   Europe until 2020
                  },
  month        = dec,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {0.2},
  doi          = {10.5281/zenodo.17857554},
  url          = {https://doi.org/10.5281/zenodo.17857554},
}
```
## Associated Publication

Huber et al. (2026). [Integrating geographic data into greenhouse gas emission footprinting: a spatial analysis of European wind turbines]. *International Journal of Life Cycle Assessment*.

## License

This project is distributed under the BSD 3-Clause License. See the `LICENSE` file for full terms.
