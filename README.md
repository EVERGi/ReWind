# ReWind
Regionalized cradle-to-grave life cycle assessment (LCA) model for on- and offshore wind energy in Europe.

## Overview

`ReWind` is a Python package and set of scripts to perform regionalized cradle-to-grave life cycle assessments for onshore and offshore wind projects in Europe. The code assembles component inventories, applies region-specific scaling and calculation methods, and produces impact estimates suitable for comparative analysis and research.

Key features
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

- ecoinvent datasets (licensed)
- large external datasets such as GEBCO bathymetry (can be downloaded separately via: **GEBCO global bathymetry dataset (2024)**  
  https://www.gebco.net/data_and_products/gridded_bathymetry_data/, File used: `GEBCO_2024_sub_ice_topo.nc`)

## Reproducibility

Due to licensing restrictions (e.g. ecoinvent) and the size of certain external datasets (e.g. bathymetry data), full reproduction of the European fleet assessment is not possible using this repository alone. 

However, the provided code and example workflow allow users to execute the model on a reduced dataset and verify the implementation and calculation logic. The Zenodo archive provides the processed fleet-level datasets used in this study, enabling validation and comparison of results.


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
- Temporal scope: The model does not (currently) include full temporal dynamics for supply-chain changes or equipment degradation over time.
- Uncertainty quantification: Uncertainties are not comprehensively propagated in all modules — users should treat point estimates with caution and run sensitivity analyses.
- Validation: The model has limited validation against ground-truth project-level LCAs; validate results against other studies before use in policy or investment decisions.
- External dependencies: Geospatial packages (e.g. `geopandas`, `rasterio`) may require system-level libraries which are outside of Python's control.
- Data licensing: Some input datasets may be proprietary or have redistribution limits — verify each dataset's license before sharing derived outputs.

## Data Availability (GitHub + Zenodo)

- GitHub: the most recent source code and (small) example data are available from this repository. Replace the placeholder below with your repository URL:

	https://github.com/<OWNER>/<REPOSITORY>

- Zenodo: If you have a DOI-archived snapshot, link it here (example placeholder):

	DOI: 10.5281/zenodo.YOUR_DOI_HERE

Include persistent links to the data snapshots you used for any published analyses.

## Citation

Please cite the project and any associated Zenodo record. Example BibTeX template (fill in authors, year, title, version, DOI):

```bibtex
@misc{ReWind2026,
	author = {Author, A. and Contributor, B.},
	title = {ReWind: Regionalized cradle-to-grave LCA model for wind energy},
	year = {2026},
	howpublished = {Zenodo},
	doi = {10.5281/zenodo.YOUR_DOI_HERE},
	url = {https://github.com/<OWNER>/<REPOSITORY>}
}
```
## Associated Publication

Huber et al. (2026). [Integrating geographic data into greenhouse gas emission footprinting: a spatial analysis of European wind turbines]. *International Journal of Life Cycle Assessment*.

## License

This project is distributed under the BSD 3-Clause License. See the `LICENSE` file for full terms.

--
If you'd like, I can (1) add real GitHub/Zenodo links, (2) add a short example that runs a specific function from the `REWIND` package, or (3) produce a short `requirements.txt` based on `pyproject.toml`.
