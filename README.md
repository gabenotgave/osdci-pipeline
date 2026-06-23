# OSDCI — Open Source Data Center Intelligence

> A geospatial ML pipeline for identifying optimal data center siting locations across the continental United States. Part of the OSDCI project (osdci-pipeline).

## Overview

OSDCI is an open research pipeline that scores every H3 hexagonal grid cell across the CONUS on infrastructure viability and environmental impact constraints. The model learns from known data center facility locations (IM3 Open Source Data Center Atlas) and produces ranked candidate sites for infrastructure planners evaluating cost, grid access, network connectivity, and sustainability trade-offs. The pipeline processes ~214,000 res-6 cells (~2.2 km² each), assembles 60+ geospatial features from 18 public datasets, trains a LightGBM classifier with spatial holdout validation, and exports frontend-ready artifacts for interactive exploration.

## Key findings

- AUC-ROC: 0.971 on spatial holdout (NJ, CO, WA)
- Top-100 precision: 0.160 (72× better than random)
- Top features: IXP distance, cloud region proximity, traffic proximity, population within 25 mi, electricity price
- Top candidate regions: Norfolk/Hampton Roads VA, Columbus OH, Des Moines IA, Atlanta GA, Kansas City MO, Dallas/Fort Worth TX, Sacramento CA

## Repository structure

```
osdci-pipeline/
├── osdci/
│   ├── __init__.py                      # Package marker
│   ├── grid.py                          # H3 grid generation from Natural Earth USA boundary
│   ├── features.py                      # Canonical feature column registry for training and assembly
│   └── cloud_regions.py                 # AWS/Azure/GCP region coordinates and city name filters
├── scripts/
│   ├── build_grid.py                    # Step 0: build CONUS H3 grid
│   ├── ingest_im3.py                    # Step 1: ingest IM3 data center labels
│   ├── aggregate_labels.py              # Step 2: aggregate facility labels to H3 cells
│   ├── compute_tx_distance.py             # Step 3: transmission line distance (HIFLD)
│   ├── compute_substation_distance.py   # Step 4: substation distance (HIFLD)
│   ├── compute_eia860_features.py       # Step 5: county/BA generation capacity and queue congestion
│   ├── compute_eia861_industrial_price.py  # Step 6: industrial electricity price by county
│   ├── compute_egrid_features.py        # Step 7: grid carbon intensity and renewable share
│   ├── compute_nearby_generation.py     # Step 8: nearby operating generation capacity
│   ├── compute_ixp_distance.py          # Step 9: Internet exchange proximity (PeeringDB)
│   ├── compute_proximity_features.py    # Step 10: metro and cloud region proximity
│   ├── compute_nlcd_features.py         # Step 11: NLCD 2024 land cover zonal statistics
│   ├── compute_impervious_features.py   # Step 12: NLCD impervious surface
│   ├── compute_slope_features.py        # Step 13: USGS 3DEP terrain slope (py3dep)
│   ├── compute_padus_features.py        # Step 14: PAD-US protected area exclusion
│   ├── compute_nri_features.py          # Step 15: FEMA National Risk Index hazards
│   ├── compute_wetland_features.py      # Step 16: NLCD wetland proxy flags
│   ├── compute_climate_features.py      # Step 17: PRISM temperature and cooling degree days
│   ├── compute_water_stress_features.py  # Step 18: WRI Aqueduct water stress
│   ├── compute_population_features.py   # Step 19: WorldPop population density and buffer
│   ├── compute_ejscreen_features.py     # Step 20: EPA EJScreen environmental justice
│   ├── compute_tribal_features.py       # Step 21: Census AIANNH tribal land exclusion
│   ├── compute_critical_habitat_features.py  # Step 22: USFWS ESA critical habitat exclusion
│   ├── assemble_features.py             # Step 23: join all features into master table
│   ├── train_baseline.py                # Step 24: train LightGBM baseline model
│   ├── compute_final_scores.py          # Step 25: apply impact penalties and export artifacts
│   ├── sanity_check.py                  # Optional: validate pipeline outputs before training
│   └── archive/                         # Deprecated scripts (e.g. scrape_dcmap.py)
├── data/
│   ├── raw/                             # Source files and download caches (gitignored)
│   └── processed/                       # Grid, labels, and feature parquets (gitignored)
├── models/                              # Model artifacts and frontend handoff files
├── requirements.txt
└── README.md
```

## Data sources

> **Note:** `data/raw/` and `data/processed/` are gitignored. After cloning this repo, create both directories and populate `data/raw/` using the tables below before running the pipeline. Automated downloads will populate additional files on first run.
>
> ```bash
> mkdir -p data/raw data/processed models
> ```

| Dataset | Source | License | Used for |
|---------|--------|---------|----------|
| IM3 Open Source Data Center Atlas | DOE/PNNL ([OSTI 3017294](https://www.osti.gov/biblio/3017294)) | ODbL | Training labels |
| USGS NLCD 2024 | [USGS EROS](https://www.mrlc.gov/) | Public domain | Land cover, impervious surface |
| USGS 3DEP (via py3dep) | [USGS](https://www.usgs.gov/3d-elevation-program) | Public domain | Terrain slope |
| USGS PAD-US 4.1 | [USGS Gap Analysis Project](https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-download) | Public domain | Protected areas |
| HIFLD Electric Transmission Lines | [DHS/HIFLD](https://hifld-geoplatform.opendata.arcgis.com/) | Public domain | Transmission distance |
| HIFLD Electric Substations | [DHS/HIFLD](https://hifld-geoplatform.opendata.arcgis.com/) | Public domain | Substation distance |
| EIA Form 860 (2024) | [U.S. Energy Information Administration](https://www.eia.gov/electricity/data/eia860/) | Public domain | Generation capacity, BA renewables |
| EIA Form 861 (2024) | [U.S. Energy Information Administration](https://www.eia.gov/electricity/data/eia861/) | Public domain | Industrial electricity price |
| EPA eGRID 2023 | [U.S. EPA](https://www.epa.gov/egrid) | Public domain | Grid carbon intensity |
| EPA EJScreen 2024 v2.31 | [U.S. EPA](https://www.epa.gov/ejscreen) (via [Zenodo archive](https://zenodo.org/)) | Public domain | Environmental justice |
| FEMA National Risk Index v1.20 | [FEMA](https://hazards.fema.gov/nri/) | Public domain | Flood, seismic, wildfire risk |
| WRI Aqueduct 4.0 | [World Resources Institute](https://www.wri.org/data/aqueduct-global-maps-40-data) | CC BY 4.0 | Water stress |
| WorldPop 2020 (UN-adjusted, 1 km) | [WorldPop / Univ. of Southampton](https://www.worldpop.org/) | CC BY 4.0 | Population density |
| PRISM 30-yr Climate Normals | [Oregon State University](https://prism.oregonstate.edu/) | CC BY 4.0 | Temperature, cooling degree days |
| PeeringDB IXP locations | [PeeringDB](https://www.peeringdb.com/) (API) | CC0 | Network connectivity |
| Census TIGER AIANNH 2024 | [U.S. Census Bureau](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) | Public domain | Tribal lands |
| USFWS Critical Habitat | [USFWS ECOS](https://ecos.fws.gov/) (ArcGIS REST API) | Public domain | ESA critical habitat |
| Census Places 2023 (via pygris) | [U.S. Census Bureau](https://www.census.gov/) | Public domain | Metro proximity |

## Reproducing the pipeline

### Prerequisites

- Python 3.11+
- `pip install -r requirements.txt`
- `pip install -e .`

> The second command installs the `osdci` package in editable mode, making it importable by all pipeline scripts without any PYTHONPATH configuration.

Dependencies include `pygris` (Census geometries), `rasterio`/`rasterstats` (NLCD, impervious, PAD-US), `scipy` (WorldPop buffer convolution), and `py3dep` (slope).

**NLCD (rasterio):** On Python versions without a pre-built `rasterio` wheel, pip builds from source and requires GDAL:

```bash
brew install gdal
export GDAL_CONFIG="$(brew --prefix gdal)/bin/gdal-config"
export GDAL_VERSION="$($GDAL_CONFIG --version)"
pip install rasterio rasterstats
```

If `pip install -r requirements.txt` fails on `rasterio` with "gdal-config not found", run the commands above, then re-run `pip install -r requirements.txt`.

### Manual downloads required

Download these files manually before running the pipeline. All paths are relative to the project root.

| Dataset | Exact filename(s) | Destination path | Where to get it | Notes |
|---------|-------------------|------------------|-----------------|-------|
| IM3 Open Source Data Center Atlas | `im3_open_source_data_center_atlas_v2026.02.09.csv` | `data/raw/` | [OSTI 3017294](https://www.osti.gov/biblio/3017294) — download the CSV from the dataset landing page | **Required** (step 1). Rename to match if the download uses a different version suffix. |
| NLCD 2024 land cover | `Annual_NLCD_LndCov_2024_CU_C1V1.tif` | `data/raw/` | [MRLC Data Download](https://www.mrlc.gov/data) → NLCD 2024 → Annual NLCD Land Cover (CONUS GeoTIFF) | **Required** (step 11). Any `.tif` with `NLCD` and `LndCov` in the name is auto-discovered. |
| NLCD 2024 impervious surface | `Annual_NLCD_FctImp_2024_CU_C1V1.tif` | `data/raw/` | [MRLC Data Download](https://www.mrlc.gov/data) → NLCD 2024 → Annual NLCD Impervious Surface (CONUS GeoTIFF) | **Required** (step 12). |
| PAD-US 4.1 | `PADUS4_1Geodatabase.gdb/` (full geodatabase folder) | `data/raw/PADUS4_1Geodatabase.gdb/` | [PAD-US 4.1 download](https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-download) → Geodatabase format | **Required** (step 14). Extract the entire `.gdb` directory. |
| WRI Aqueduct 4.0 baseline annual | `Aqueduct40_baseline_annual_y2023m07d05.csv` | `data/raw/` | [WRI Aqueduct 4.0](https://www.wri.org/data/aqueduct-global-maps-40-data) → Baseline Annual data → CSV download | **Required** (step 18). Catchment geometry is fetched automatically (see below). |
| FEMA National Risk Index v1.20 | `NRI_Table_CensusTracts.csv` | `data/raw/` | [FEMA NRI Data Resources](https://hazards.fema.gov/nri/data-resources) → Download → Census Tracts CSV | **Required** (step 15). Census tract boundaries are fetched automatically. |
| EPA EJScreen 2024 v2.31 | `EJSCREEN_2024_BG_with_AS_CNMI_GU_VI.csv` | `data/raw/` | [EPA EJScreen download](https://www.epa.gov/ejscreen/download-ejscreen-data) → Block Group CSV (includes AS, CNMI, GU, VI) | **Required** (step 20). Census block groups are fetched automatically. |
| PRISM 30-yr normals (1991–2020) | 13 GeoTIFFs — see list below | `data/raw/prism_normals/` | [PRISM Normals](https://prism.oregonstate.edu/normals/) → 1991–2020 → 800m resolution → download **annual** and **monthly** mean temperature (`tmean`) rasters for CONUS | **Required** (step 17). Scripts glob `**/prism_tmean*.tif`. |
| WorldPop 2020 (1 km, UN-adjusted) | `usa_population_2020_1km.tif` | `data/raw/` | [WorldPop USA 2020 1 km](https://www.worldpop.org/geodata/listing?id=77) | **Required** (step 19). A reprojected cache is built automatically on first run. |
| Census TIGER AIANNH 2024 | `tl_2024_us_aiannh.shp` + sidecars (`.dbf`, `.shx`, `.prj`, `.cpg`) | `data/raw/tl_2024_us_aiannh/` | [Census TIGER 2024 AIANNH](https://www2.census.gov/geo/tiger/TIGER2024/AIANNH/) → `tl_2024_us_aiannh.zip` | **Required** (step 21). Extract the full shapefile bundle. |
| EIA Form 860 (2024) | `2___Plant_Y2024.xlsx`, `3_1_Generator_Y2024.xlsx` | `data/raw/eia_860/` | [EIA-860 annual data](https://www.eia.gov/electricity/data/eia860/) → download the 2024 zip → extract the Plant and Operable/Proposed Generator workbooks | **Required** (steps 5 and 8). Both files needed. |
| EIA Form 861 (2024) | `Sales_Ult_Cust_2024.xlsx`, `Service_Territory_2024.xlsx` | `data/raw/eia_861/` | [EIA-861 annual data](https://www.eia.gov/electricity/data/eia861/) → download the 2024 zip → extract sales and service territory workbooks | **Required** (step 6). `Utility_Data_2024.xlsx` is present in the archive but **not used** by the pipeline. |
| HIFLD electric transmission lines | `Electric_Power_Transmission_Lines_A.shp` + sidecars (`.dbf`, `.shx`, `.prj`, `.cpg`) | `data/raw/US_Electric_Power_Transmission_Lines_*/` | [HIFLD Transmission Lines](https://hifld-geoplatform.opendata.arcgis.com/datasets/geoplatform::electric-power-transmission-lines) → Download → Shapefile | **Required** (step 3). Folder name varies; script auto-discovers `Electric_Power_Transmission_Lines_A.shp`. |
| HIFLD electric substations | `substations.csv` (or `Substations.csv`) | `data/raw/` | [HIFLD Substations](https://hifld-geoplatform.opendata.arcgis.com/datasets/geoplatform::electric-substations) → Download → CSV | **Required** (step 4). Script auto-discovers any `*ubstation*.csv` under `data/raw/`. |
| EPA eGRID 2023 | `egrid2023_data_rev2.xlsx` | `data/raw/` | [EPA eGRID download](https://www.epa.gov/egrid/download-data) → 2023 Excel workbook | **Required** (step 7). |
| EPA eGRID 2023 subregions | `eGRID2023_Subregions.shp` + sidecars (`.dbf`, `.shx`, `.prj`, `.cpg`, `.sbn`, `.sbx`) | `data/raw/egrid2023_subregions/` | Same eGRID 2023 download page → subregion shapefile zip | **Required** (step 7). Extract full shapefile bundle. County boundaries are fetched automatically. |

**PRISM normals — 13 required GeoTIFFs**

Place each file under `data/raw/prism_normals/` (subfolders are fine). Only the `.tif` in each directory is read; ancillary `.prj`, `.xml`, etc. may be present.

| Period | Exact filename |
|--------|----------------|
| Annual | `prism_tmean_us_25m_2020_avg_30y.tif` |
| January | `prism_tmean_us_25m_202001_avg_30y.tif` |
| February | `prism_tmean_us_25m_202002_avg_30y.tif` |
| March | `prism_tmean_us_25m_202003_avg_30y.tif` |
| April | `prism_tmean_us_25m_202004_avg_30y.tif` |
| May | `prism_tmean_us_25m_202005_avg_30y.tif` |
| June | `prism_tmean_us_25m_202006_avg_30y.tif` |
| July | `prism_tmean_us_25m_202007_avg_30y.tif` |
| August | `prism_tmean_us_25m_202008_avg_30y.tif` |
| September | `prism_tmean_us_25m_202009_avg_30y.tif` |
| October | `prism_tmean_us_25m_202010_avg_30y.tif` |
| November | `prism_tmean_us_25m_202011_avg_30y.tif` |
| December | `prism_tmean_us_25m_202012_avg_30y.tif` |

### Automated downloads and caches

These files are **not** required before the first pipeline run. Scripts create them on first execution and reuse them on subsequent runs.

| File cached to `data/raw/` | Fetched by | Source |
|----------------------------|------------|--------|
| `tl_2023_us_county.gpkg` | `compute_eia860_features.py`, `compute_eia861_industrial_price.py`, `compute_egrid_features.py` | [Census TIGER 2023 Counties](https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip) |
| `census_places_2023.parquet` | `compute_proximity_features.py` | pygris / Census TIGER Places (2023) |
| `cloud_regions.parquet` | `compute_proximity_features.py` | Built from `osdci/cloud_regions.py` (AWS/Azure/GCP coordinates) |
| `peeringdb/ixfac_us.json`, `peeringdb/fac_us.json` | `compute_ixp_distance.py` | [PeeringDB API](https://www.peeringdb.com/api) |
| `peeringdb/ixp_sites_us.parquet` | `compute_ixp_distance.py` | Derived from PeeringDB API responses |
| `census_tracts_2021_cb.parquet` | `compute_nri_features.py` | pygris / Census TIGER Tracts (2021 cartographic boundary) |
| `block_groups_2020_cb.parquet` | `compute_ejscreen_features.py` | pygris / Census TIGER Block Groups (2020 cartographic boundary) |
| `h3_bg_mapping_res6.parquet` | `compute_ejscreen_features.py` | Derived spatial join (H3 centroids → block groups) |
| `hydrobasins_na_lev06.parquet` | `compute_water_stress_features.py` | [HydroBASINS North America lev 06](https://data.hydrosheds.org/file/HydroBASINS/standard/hybas_na_lev06_v1c.zip) |
| `critical_habitat_usfws.parquet` | `compute_critical_habitat_features.py` | [USFWS Critical Habitat FeatureServer](https://services.arcgis.com/QVENGdaPbd4LUkLV/arcgis/rest/services/USFWS_Critical_Habitat/FeatureServer) |
| `padus_gap12_5070.parquet` | `compute_padus_features.py` | Derived from `PADUS4_1Geodatabase.gdb` (GAP Status 1/2 filter) |
| `worldpop_3857_1km.tif` | `compute_population_features.py` | Reprojected from `usa_population_2020_1km.tif` |
| `nlcd_impervious_stats_res6.pkl` | `compute_impervious_features.py` | Checkpoint (resume support; optional) |
| `slope_checkpoint.jsonl` | `compute_slope_features.py` | Checkpoint (resume support; optional) |

| File cached to `data/processed/` | Fetched by | Source |
|----------------------------------|------------|--------|
| `h3_tract_mapping_res6.parquet` | `compute_nri_features.py` | Derived spatial join (H3 centroids → census tracts) |

**Fetched on demand (not written to `data/raw/`):** USGS 3DEP slope tiles via `py3dep` in `compute_slope_features.py` (network requests per CONUS tile).

**Not used by the active pipeline:** `dcmap_checkpoint.jsonl` and other files under `scripts/archive/` are from deprecated scrapers only.

### Pipeline execution order

Run each step from the project root:

| Step | Command | Description |
|------|---------|-------------|
| 0 | `python scripts/build_grid.py` | Build CONUS H3 res-6 grid from Natural Earth USA boundary |
| 1 | `python scripts/ingest_im3.py` | Ingest IM3 data center labels and assign H3 indices |
| 2 | `python scripts/aggregate_labels.py` | Aggregate facility labels to one row per H3 cell |
| 3 | `python scripts/compute_tx_distance.py` | Distance to nearest ≥69 kV transmission line |
| 4 | `python scripts/compute_substation_distance.py` | Distance to nearest electrical substation |
| 5 | `python scripts/compute_eia860_features.py` | County/BA generation capacity and queue congestion |
| 6 | `python scripts/compute_eia861_industrial_price.py` | County-level industrial electricity price |
| 7 | `python scripts/compute_egrid_features.py` | Grid carbon intensity and renewable share |
| 8 | `python scripts/compute_nearby_generation.py` | Nearby operating generation capacity |
| 9 | `python scripts/compute_ixp_distance.py` | Distance to nearest IXP and peering density |
| 10 | `python scripts/compute_proximity_features.py` | Metro and cloud region proximity |
| 11 | `python scripts/compute_nlcd_features.py` | NLCD land cover zonal statistics (~20–40 min) |
| 12 | `python scripts/compute_impervious_features.py` | NLCD impervious surface (~20–40 min) |
| 13 | `python scripts/compute_slope_features.py` | USGS 3DEP terrain slope via py3dep (long-running) |
| 14 | `python scripts/compute_padus_features.py` | PAD-US GAP 1/2 protected area exclusion |
| 15 | `python scripts/compute_nri_features.py` | FEMA NRI flood, seismic, and wildfire risk |
| 16 | `python scripts/compute_wetland_features.py` | NLCD wetland proxy exclusion flags |
| 17 | `python scripts/compute_climate_features.py` | PRISM mean temperature and cooling degree days |
| 18 | `python scripts/compute_water_stress_features.py` | WRI Aqueduct baseline water stress |
| 19 | `python scripts/compute_population_features.py` | WorldPop density and 25-mile buffer population |
| 20 | `python scripts/compute_ejscreen_features.py` | EPA EJScreen environmental justice burden |
| 21 | `python scripts/compute_tribal_features.py` | Census AIANNH tribal land exclusion |
| 22 | `python scripts/compute_critical_habitat_features.py` | USFWS ESA critical habitat exclusion |
| 23 | `python scripts/assemble_features.py` | Join all feature parquets into master training table |
| 24 | `python scripts/train_baseline.py` | Train LightGBM baseline with spatial holdout |
| 25 | `python scripts/compute_final_scores.py` | Apply impact penalties and export frontend artifacts |

Optional validation before training:

```bash
python scripts/sanity_check.py
```

Several feature scripts support checkpoint/resume flags for long runs (see per-script `--help`). Step 11 (NLCD) and step 12 (impervious) require GDAL and rasterio.

## Model

### Architecture

LightGBM binary classifier with:

- Spatial train/test split (NJ + CO + WA holdout bounding boxes)
- Soft labels: k-ring-1 neighbors of known facilities (weight = 0.25)
- `is_unbalance=True` for class imbalance
- Early stopping on AUC-PR (`stopping_rounds=200`)

### Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.971 (target ≥ 0.85) |
| AUC-PR | 0.186 |
| Top-100 precision | 0.160 |
| Best iteration | 216 |

Full metrics are in `models/baseline_results.json`.

### Features

63 canonical features across 10 categories defined in `osdci/features.py`: power infrastructure, grid economics, network connectivity, market proximity, land cover, climate, water, risk, population, and environmental justice. The baseline model trains on 61 features present in the assembled master table. See `models/feature_importances.csv` for full rankings.

Hard exclusion flags (`in_protected_area`, `in_wetland`, `in_tribal_land`, `in_critical_habitat`) are kept in the master table but excluded from model input; they are applied as post-prediction masks and penalty components.

## Output artifacts

Regenerate handoff files with step 25 (`compute_final_scores.py`). Large artifacts are gitignored; small reference files are kept in `models/`.

| File | Description |
|------|-------------|
| `models/top_candidates.geojson` | 24,927 candidate cells with full feature properties and pre-normalized penalty components (`c_*`) for client-side slider recomputation |
| `models/basemap.csv` | 88,445 cells (viability ≥ 0.02) with H3 indexes and scores for the background map layer |
| `models/cell_explanations.json` | Human-readable explanation strings for all 214,145 cells |
| `models/baseline_results.json` | Full model metrics |
| `models/feature_importances.csv` | Feature importance rankings |
| `data/processed/scored_cells.parquet` | Complete scored research table (all CONUS cells) |
| `models/baseline_lgbm.pkl` | Trained LightGBM model (from step 24) |
| `models/spatial_residuals.csv` | Per-cell viability scores (from step 24) |

## Two-stage scoring

```
final_score = viability_score × (1 − impact_penalty)
```

**Viability score** — learned from known data center locations (LightGBM output on spatial holdout).

**Impact penalty** — weighted sum of environmental and community constraint features (`c_*` fields in GeoJSON properties). Penalty weights are defined in the OSDCI frontend and applied client-side; no backend is required.

Display categories: Existing Data Center, Hard Exclusion, High Community Impact, Candidate, Low Viability.

## Citation

If you use this work, please cite:

```
Arnold, G. (2025). OSDCI: Open Source Data Center Intelligence —
A Geospatial ML Framework for Infrastructure Siting Optimization.
GitHub. https://github.com/gabenotgave/osdci-pipeline
```

Data sources should be cited individually per their license requirements — see the Data sources table above.

## License

The OSDCI pipeline code is licensed under the
**Apache License 2.0**. See [LICENSE](LICENSE) for the full text.

### Data source licenses

Output artifacts produced by this pipeline are derived from
source datasets with their own license terms. Users of model
outputs and GeoJSON artifacts should comply with the following:

| Dataset | License | Requirement |
|---------|---------|-------------|
| IM3 Open Source Data Center Atlas | ODbL | Derivative databases must be ODbL |
| WRI Aqueduct 4.0 | CC BY 4.0 | Attribution required |
| WorldPop 2020 | CC BY 4.0 | Attribution required |
| PRISM Climate Normals | CC BY 4.0 | Attribution required |
| All U.S. federal datasets (EIA, EPA, USGS, FEMA, Census, USFWS, DHS/HIFLD) | Public domain | No restrictions |

See [NOTICE](NOTICE) for full attribution details.

## Contributing

`data/raw/` and `data/processed/` are gitignored. To contribute, reproduce the pipeline locally using the manual download instructions above. Large frontend handoff files (`top_candidates.geojson`, `basemap.csv`, `cell_explanations.json`) are also gitignored; regenerate them with `compute_final_scores.py` after training, or configure Git LFS if you prefer to track them in version control.
