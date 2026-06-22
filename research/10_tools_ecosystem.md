# R10 — Open-Source Ecosystem, Reference Implementations & Library Stack

**Project:** ISRO Bharatiya Antariksh Hackathon 2026 PS-1 — Physics-informed geospatial AI/ML for urban heat hotspot mapping, driver attribution, LST modelling, and cooling-intervention optimization (°C reduction).
**Domain of this report (R10):** the concrete, currently-maintained open-source toolchain — real packages (with `pip`/`conda` install names), reference repositories, and a recommended `requirements.txt` so the build team **reuses proven tools instead of reinventing**.

> Verification note: items confirmed via live web search (June 2026) are marked **[web-verified]**. A few ubiquitous packages (rasterio, geopandas, xgboost, etc.) are marked **[from-knowledge]** — standard PyPI/conda-forge packages, install names are stable and worth a 5-second `pip index`/PyPI check before pinning.

---

## 1. Overview & Strategy

The system splits into ~6 software layers; the ecosystem maps cleanly onto them:

| Layer | What it does | Primary OSS stack |
|---|---|---|
| **A. Server-side compute (O(1) on GEE)** | LST retrieval, indices, zonal stats, SUHI — all pushed to Google's servers | `earthengine-api`, `geemap`, `eemont`, `wxee`, `geetools` + **Ermida `Landsat_SMW_LST`** toolbox |
| **B. Datasets / catalogs** | many cross-verifying inputs | GEE Data Catalog + **Awesome-GEE Community Catalog** (`sat-io`), STAC (`pystac-client`, `planetary-computer`) |
| **C. Geospatial core (local/edge)** | raster/vector wrangling, morphology, OSM | `rioxarray`/`rasterio`, `geopandas`, **`osmnx`**, **`momepy`**, `h3`, `rasterstats` |
| **D. Physics / microclimate** | energy balance, MRT, thermal comfort, cooling service | **UMEP/SuPy-SUEWS/SOLWEIG**, **InVEST Urban Cooling**, `pvlib`, `metpy`, `thermofeel`, `pythermalcomfort` |
| **E. ML / stats / attribution / optimization** | LST models, driver quantification, intervention search | `scikit-learn`, `xgboost`/`lightgbm`/`catboost`, `shap`, **`verde`** (spatial CV), **`mgwr`/`pysal`** (GWR), `pymc`, **`pymoo`**, `OR-Tools`/`PuLP` |
| **F. Visualization & app** | hotspot maps, dashboards, reports | `streamlit`, `leafmap`/`folium`, `pydeck`/`kepler.gl`, `plotly`, `localtileserver`/`titiler` |

**Design rule for ≥30 cross-verifying methods:** GEE (Layer A) is the *fast path* — most LST/SUHI/index computation runs server-side in seconds with no local download (the "O(1) compute" goal). Layers C–E are the *deep path* for physics validation, attribution, and optimization on a downloaded study-area subset. Where the same quantity (e.g. LST, UHI intensity) can be obtained from several independent sources, list them as separate verification methods.

---

## 2. Library Catalog (grouped by function)

> `pip` name shown; most are also on `conda-forge` (preferred for the GDAL-stack packages). "Module it serves" = which layer/feature in our build.

### 2.1 Google Earth Engine ecosystem (Layer A — fastest server-side compute)

| Install name | Purpose (one line) | Module it serves | Status |
|---|---|---|---|
| `earthengine-api` (import `ee`) | Official Python client for Earth Engine; all server-side computation | Core of Layer A; LST/index/SUHI compute, exports | **[web-verified]** |
| `geemap` | Interactive GEE maps in Jupyter/Colab + huge helper library (zonal stats, charts, export, ML) | Layer A glue + Layer F notebooks; UHI/LST tutorials | **[web-verified]** |
| `eemont` | Extends `ee` objects: one-call cloud/shadow masking, scaling, **spectral indices** (NDVI/NDBI/NDWI/UI/...), closest-image, time series | Layer A preprocessing + driver indices; cuts boilerplate massively | **[web-verified]** |
| `wxee` | Bridge **Earth Engine ↔ xarray** for weather/climate time series (download EE ImageCollections as xarray/NetCDF) | Layer A→C handoff; ERA5/climatology to local physics models | **[web-verified]** |
| `geetools` | Extra GEE utilities (batch export, cloud masks, collection tools, asset management) | Layer A ops/automation | **[web-verified]** (gee-community) |
| `spectral` / `awesome-spectral-indices` (catalog used *by* eemont) | Standardized formula DB of 200+ spectral indices | Driver-index computation in A | **[web-verified]** (powers `eemont.spectralIndices`) |
| `restee` / `geeup` (optional) | REST helpers / asset upload CLI | Asset ingest ops | **[from-knowledge]** (verify) |

> **Note:** `geopytools` from the brief appears to be a confusion — the canonical GEE-Python helpers are `geemap`, `eemont`, `geetools`, `wxee`. Use those; do not block on `geopytools`.

### 2.2 UHI / LST reference implementations & domain datasets (Layer A/B)

| Install / access | Purpose | Module it serves | Status |
|---|---|---|---|
| **`Landsat_SMW_LST`** (Ermida) — JS GEE modules, *clone repo* | Statistical Mono-Window (SMW) **Landsat 4/5/7/8/9 LST** with broadband emissivity + NCEP TPW atmospheric correction | **Primary LST retrieval method** in A (port to Python `ee`) | **[web-verified]** (see §3) |
| `ee_lst` (lunasilvestre) — `pip`/clone | **Python port** of Ermida's SMW LST for the `ee` Python API | LST retrieval directly in our Python pipeline | **[web-verified]** |
| Awesome-GEE Community Catalog (`sat-io` assets) | Hundreds of community GEE datasets (UHII, GHSL, gap-filled LST, air-temp, building height) | Layer B cross-verification datasets | **[web-verified]** |
| **UHII** ImageCollections `projects/sat-io/open-datasets/UHII/{MOD1,MOD2,MYD1,MYD2,SMOD2,SMYD1,AMOD2,SAT}` | Global Urban Heat Island Intensity (surface + canopy, 10k+ cities, 20+ yr) | Independent SUHI cross-check in A/B | **[web-verified]** |
| **Global LCZ map** `ee.ImageCollection("RUB/RUBCLIM/LCZ/global_lcz_map/latest")` | Local Climate Zones (17 classes, 100 m) — Demuzere et al. 2022 | Urban/rural reference & stratification for SUHI; driver context | **[web-verified]** |
| **UT-GLOBUS** (Zenodo + `sat-io` catalog) | Global building heights & urban canopy params for 1200+ cities | Morphology driver + physics model input (UCPs) | **[web-verified]** |
| GHSL on GEE (`JRC/GHSL/...`) + catalog GHSL 2023 | Built-up surface/volume/height + population (GHS-POP) | Driver layers + exposure/vulnerability | **[web-verified]** |
| `multitemporal-lcz-mapping` (Demuzere) — clone | Generic dynamic LCZ mapping in GEE | Optional custom-LCZ method | **[web-verified]** |

### 2.3 Microclimate / physics-informed tools (Layer D)

| Install name | Purpose | Module it serves | Status |
|---|---|---|---|
| `supy` (SUEWS Python) | **SUEWS** Surface Urban Energy & Water Balance Scheme as a Python package (energy balance, storage heat, QF) | Physics-informed energy-balance modelling & validation | **[web-verified]** |
| UMEP / `umep-reqs` (QGIS plugin "UMEP for processing") | Urban Multi-scale Environmental Predictor: SVF, SOLWEIG, SUEWS, anthropogenic heat, wall-aspect | MRT/thermal-comfort & SVF preprocessing | **[web-verified]** |
| SOLWEIG (within UMEP; also standalone QGIS plugin `solweig_qgis`) | Mean Radiant Temperature (Tmrt), UTCI, PET from DSM/DEM + meteo; GPU/tiled | Pedestrian-level heat & shade for intervention scoring | **[web-verified]** |
| `natcap.invest` → `natcap.invest.urban_cooling_model` | **InVEST Urban Cooling**: heat-mitigation index from shade+ET+albedo + park cooling distance → °C reduction | **Cooling-intervention °C estimation** (core deliverable) | **[web-verified]** |
| `invest-ucm-calibration` (martibosch) | Auto-calibrate InVEST UCM via simulated annealing against observed LST/air-T | Calibrate the °C-reduction model to local data | **[web-verified]** |
| `pvlib` | Solar position (NREL SPA) + irradiance; shade/insolation geometry | Solar geometry for shade & radiation drivers; feeds Tmrt/WBGT | **[web-verified]** |
| `metpy` | Meteorological calculations incl. heat index, wet-bulb, dewpoint, thermodynamics | Heat-stress indices & meteo preprocessing | **[web-verified]** |
| `thermofeel` (ECMWF) | Vectorized thermal-comfort indices: **UTCI, WBGT, MRT, apparent T, humidex** — built for gridded NWP output | Grid-wide thermal-comfort layers (fast, array-based) | **[web-verified]** |
| `pythermalcomfort` | Comfort indices: **UTCI, PET, PMV/PPD, SET, WBGT** (point/array) | Cross-verify thermal indices; human-exposure metrics | **[web-verified]** |
| `pywbgt` (optional) | Dedicated WBGT (Liljegren) using `pvlib` solar | Independent WBGT cross-check | **[web-verified]** |
| `i-Tree` (external tool, not pip) | USFS tree-benefits / canopy ecosystem services incl. cooling | Reference for tree-cooling magnitudes (validation, not in-loop) | **[from-knowledge]** (verify) |

### 2.4 Geospatial Python core (Layer C)

| Install name | Purpose | Module it serves | Status |
|---|---|---|---|
| `rasterio` | Read/write raster (GDAL bindings) | Raster I/O backbone | **[web-verified via osmnx stack]** |
| `rioxarray` | xarray + rasterio (CRS-aware labelled rasters, reproject, clip) | Local LST/raster analysis | **[from-knowledge]** |
| `xarray` | N-D labelled arrays (time×band×y×x) | Time-series cubes, climate data | **[web-verified]** |
| `geopandas` | Vector dataframes (buildings, wards, parks) | All vector ops, zonal joins | **[web-verified]** |
| `shapely` | Geometry engine | Geometry ops underlying geopandas | **[from-knowledge]** |
| `pyproj` | CRS transforms | Projection handling | **[from-knowledge]** |
| `rtree` | Spatial index (R-tree) | Fast spatial joins (osmnx/momepy dep) | **[web-verified]** |
| **`osmnx`** | Download/model/analyze OSM street networks, **building footprints**, POIs; network metrics (intersection density, centrality) | **Morphology drivers** + canopy/road context | **[web-verified]** |
| `pyrosm` | Fast `.osm.pbf` parsing to GeoDataFrames (bulk OSM, offline) | Bulk OSM ingest for large cities | **[web-verified]** |
| **`momepy`** | **Urban morphometrics** (PySAL): building/plot/street/block dimensions, shape, compactness, spatial distribution, tessellation | **Quantify morphological drivers** of LST | **[web-verified]** |
| `urbanpy` | Download/visualize/compute urban accessibility & H3 aggregation | Hex aggregation & accessibility to cooling | **[web-verified]** |
| `h3` (`h3-py`) | Uber H3 hex grid indexing | Uniform hex analysis units for hotspots | **[web-verified]** |
| `rasterstats` | Zonal statistics raster↔vector | LST per ward/hex/building (local path) | **[web-verified]** |
| `contextily` | Basemap tiles for matplotlib | Static map context | **[from-knowledge]** |

### 2.5 ML / statistics / attribution / optimization (Layer E)

| Install name | Purpose | Module it serves | Status |
|---|---|---|---|
| `scikit-learn` | General ML, pipelines, metrics, CV | Baseline LST models, preprocessing | **[from-knowledge]** |
| `xgboost` | Gradient-boosted trees | LST regression / hotspot classification | **[from-knowledge]** |
| `lightgbm` | Fast GBM (large tabular) | Scalable LST modelling | **[from-knowledge]** |
| `catboost` | GBM w/ categorical handling (LCZ, land use) | LST model variant for cross-verify | **[from-knowledge]** |
| `shap` | SHAP values for feature attribution | **Driver quantification / explainability** | **[from-knowledge]** |
| **`verde`** (Fatiando) | scikit-learn-style spatial **gridding + BlockKFold spatial CV** | Spatial interpolation + **unbiased spatial CV** | **[web-verified]** |
| `spacv` | Spatial cross-validation splitters | Extra spatial-CV cross-check | **[web-verified]** |
| **`mgwr`** (PySAL) | (Multiscale) **Geographically Weighted Regression** | Spatially-varying driver coefficients | **[web-verified]** |
| `libpysal` / `esda` / `spreg` (`pysal`) | Spatial weights, Moran's I, LISA, spatial regression | Hotspot autocorrelation & spatial models | **[web-verified]** |
| `statsmodels` | OLS/GLM, diagnostics, OLS-as-baseline | Linear driver models, stats tests | **[from-knowledge]** |
| `pymc` | Bayesian / probabilistic modelling | Uncertainty-aware LST/driver models | **[from-knowledge]** |
| `pytorch` (`torch`) + `pytorch-lightning` (optional/heavy) | Deep learning (CNN/super-res/PINNs) | Optional downscaling / physics-informed NN | **[from-knowledge]** |
| **`pymoo`** | Multi-objective optimization (NSGA-II/III, GA) | **Cooling-intervention placement** (cost vs °C) | **[web-verified]** |
| `deap` | Evolutionary algorithms framework | Alt. GA for intervention search | **[web-verified]** |
| `pygad` | Simple GA library | Lightweight intervention optimizer | **[web-verified]** |
| `ortools` (OR-Tools) | CP-SAT / MIP / routing solver | **ILP intervention selection** under budget | **[web-verified]** |
| `pulp` | LP/ILP modelling (CBC etc.) | Readable ILP for site selection | **[from-knowledge]** |

### 2.6 Cloud-native data access (Layer B)

| Install name | Purpose | Module it serves | Status |
|---|---|---|---|
| `pystac-client` | Query STAC APIs (search items) | Non-GEE imagery fallback (Sentinel/Landsat) | **[web-verified]** |
| `planetary-computer` | Sign Microsoft PC asset URLs | Free Azure-hosted Sentinel/Landsat/ERA5 | **[web-verified]** |
| `stackstac` | STAC items → 4-D xarray (time×band×y×x), reprojected | Build local data cubes from STAC | **[web-verified]** |
| `odc-stac` | STAC items → xarray (Open Data Cube loader) | Alt. cube builder | **[web-verified]** |
| `dask` | Parallel/lazy compute on big arrays | Scale local raster/ML processing | **[web-verified]** |
| `zarr` | Chunked cloud-native array storage | Store/serve cubes; ERA5/Daymet | **[web-verified]** |

### 2.7 Visualization & web app (Layer F)

| Install name | Purpose | Module it serves | Status |
|---|---|---|---|
| `streamlit` | Python web-app framework | **Main interactive dashboard** | **[web-verified]** |
| `leafmap` | Minimal-code interactive maps (folium/ipyleaflet/pydeck/kepler backends), COG + GEE support | App maps + hotspot layers | **[web-verified]** |
| `folium` | Leaflet maps in Python | Lightweight web maps | **[from-knowledge]** |
| `pydeck` (deck.gl) | GPU WebGL large-data layers (hexbins, 3-D buildings) | 3-D heat/morphology viz | **[web-verified]** |
| `keplergl` | kepler.gl geospatial viz widget | Exploratory big-data maps | **[web-verified]** |
| `plotly` | Interactive charts | Driver/SHAP/scenario charts | **[from-knowledge]** |
| `matplotlib` | Static plots/maps | Reports, figures | **[from-knowledge]** |
| `localtileserver` | Serve local rasters as tiles (works w/ Streamlit/leafmap) | Stream large local LST rasters | **[web-verified]** |
| `titiler` (deploy) / `titiler.xyz` (hosted) | Dynamic COG tile server | Serve exported LST/hotspot COGs to web | **[web-verified]** |
| `rio-cogeo` | Create/validate Cloud-Optimized GeoTIFF | Make COGs for titiler/leafmap | **[from-knowledge]** |

---

## 3. Key Reference Repositories (real, verifiable)

> These are the highest-leverage repos to clone/study. Star counts are approximate snapshots.

### LST retrieval (most important for our LST module)
- **Ermida — `Landsat_SMW_LST`** — Statistical Mono-Window LST for Landsat 4–9 in GEE (JavaScript modules; `modules/Landsat_LST.js`, `example_1.js`, `example_2.js`). Handles broadband emissivity + NCEP **TPW** atmospheric correction + snow/water emissivity prescriptions; Collection-2 ready. **The de-facto open standard for Landsat LST in GEE.**
  `https://github.com/sofiaermida/Landsat_SMW_LST`
  Cite: Ermida, S.L., Soares, P., Mantas, V., Göttsche, F.-M., Trigo, I.F. (2020), *Remote Sensing* 12(9):1471. `https://www.mdpi.com/2072-4292/12/9/1471`
- **`ee_lst`** (lunasilvestre) — Python port of Ermida's SMW LST for the `ee` Python API → lets us run LST retrieval **inside our Python pipeline** instead of the JS Code Editor. `https://github.com/lunasilvestre/ee_lst`
- `leonsnill/lst_landsat`, `leonsnill`/others — alternative GEE Landsat LST implementations (cross-verification of method). `https://github.com/leonsnill/lst_landsat`

### GEE Python ecosystem
- **geemap** — `https://github.com/gee-community/geemap` (+ free book *Geospatial Data Science with Earth Engine and Geemap*, `https://book.geemap.org/`; ch. on heat/LST examples).
- **eemont** — `https://github.com/davemlz/eemont` (one-call preprocess + spectral indices).
- **wxee** — `https://github.com/aazuspan/wxee` (EE↔xarray). **geetools** — `https://github.com/gee-community/gee_tools`.
- **Awesome-GEE Community Catalog** — site `https://gee-community-catalog.org/`, repo `https://github.com/samapriya/awesome-gee-community-datasets` (UHII, GHSL, gap-filled LST, air-temp, UT-GLOBUS assets under the `sat-io` project).

### Urban morphology, OSM, LCZ
- **momepy** — `https://github.com/pysal/momepy` (urban morphometrics; JOSS paper `https://joss.theoj.org/papers/10.21105/joss.01807`).
- **osmnx** — `https://github.com/gboeing/osmnx` (street networks + building footprints + morphology metrics).
- **Global LCZ map** (Demuzere et al. 2022) — GEE asset `RUB/RUBCLIM/LCZ/global_lcz_map/latest`; generator `https://lcz-generator.rub.de/`; dynamic mapping repo `https://github.com/matthiasdemuzere/multitemporal-lcz-mapping`; data paper ESSD 14:3835 `https://essd.copernicus.org/articles/14/3835/2022/`.
- **WUDAPT / LCZ Generator** — `https://www.wudapt.org/` ; LCZ Generator paper `https://www.frontiersin.org/articles/10.3389/fenvs.2021.637455/full`.

### Building heights / urban canopy parameters
- **UT-GLOBUS** (GLObal Building heights for Urban Studies) — data on Zenodo `https://zenodo.org/records/11156602`; catalog page `https://gee-community-catalog.org/projects/utglobus/`; *Scientific Data* paper `https://www.nature.com/articles/s41597-024-03719-w`; model code `GlobalMapper` `https://github.com/Arking1995/GlobalMapper`.

### Physics / microclimate / cooling
- **UMEP / SuPy-SUEWS / SOLWEIG** — UMEP processing plugin `https://plugins.qgis.org/plugins/processing_umep/`; docs `https://umep-docs.readthedocs.io/`; SuPy (SUEWS Python) `https://github.com/UMEP-dev/SuPy` / `https://supy.readthedocs.io/`; UMEP paper *Environ. Model. Softw.* `https://doi.org/10.1016/j.envsoft.2017.09.020`.
- **InVEST Urban Cooling** — `natcap.invest` `https://github.com/natcap/invest`; module docs `https://invest.readthedocs.io/en/latest/api/natcap.invest.urban_cooling_model.html`; user guide `https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_cooling_model.html`.
- **invest-ucm-calibration** — `https://github.com/martibosch/invest-ucm-calibration`; worked city example `https://github.com/martibosch/lausanne-heat-islands`.
- **thermofeel** (ECMWF) — `https://github.com/ecmwf/thermofeel`. **pythermalcomfort** — `https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort`. **pvlib** — `https://github.com/pvlib/pvlib-python`. **MetPy** — `https://github.com/Unidata/MetPy`.

### Spatial stats / interpolation / optimization
- **mgwr** (PySAL) — `https://github.com/pysal/mgwr`. **PySAL** meta `https://github.com/pysal/pysal`.
- **verde** (Fatiando) — `https://github.com/fatiando/verde` (BlockKFold spatial CV).
- **pymoo** — `https://github.com/anyoptimization/pymoo`. **DEAP** — `https://github.com/DEAP/deap`. **OR-Tools** — `https://github.com/google/or-tools`.
- Tree-placement optimization reference (method inspiration): "Climate-sensitive Urban Planning through Optimization of Tree Placements" `https://arxiv.org/abs/2310.05691`.

### Apps / tiles
- **leafmap** — `https://github.com/opengeos/leafmap`. **localtileserver** — `https://github.com/banesullivan/localtileserver`. **TiTiler** — `https://github.com/developmentseed/titiler`.

---

## 4. Recommended `requirements.txt` (draft, grouped)

> Target **Python 3.11** (3.10–3.12 OK; avoid 3.13 until full GDAL-stack wheels are confirmed). Use **conda-forge** (or `micromamba`) for the GDAL stack; `pip` for pure-Python. **InVEST + UMEP/SuPy + PyTorch are heavy/optional** — keep them in extras so the core API/app stays lean. Pins are illustrative — resolve to latest compatible at build time.

```text
# ---- Python: 3.11 recommended ----

# === GEE / server-side (Layer A) — the fast O(1) path ===
earthengine-api>=1.0
geemap>=0.32
eemont>=0.3.6
wxee>=0.4
geetools>=1.0           # optional ops/automation

# === Geospatial core (Layer C) — install via conda-forge for GDAL ===
rasterio>=1.3
rioxarray>=0.15
xarray>=2024.3
geopandas>=0.14
shapely>=2.0
pyproj>=3.6
rtree>=1.2
rasterstats>=0.19
osmnx>=1.9              # OSM networks + building footprints + morphology
momepy>=0.7            # urban morphometric drivers
h3>=3.7                 # (package: h3 / h3-py)
pyrosm>=0.6             # optional bulk .osm.pbf parsing
urbanpy>=0.4           # optional accessibility/H3 helpers
contextily>=1.6        # optional basemaps

# === ML / stats / attribution / optimization (Layer E) ===
scikit-learn>=1.4
xgboost>=2.0
lightgbm>=4.3
catboost>=1.2          # optional 3rd GBM for cross-verify
shap>=0.45             # driver attribution
statsmodels>=0.14
verde>=1.8             # spatial gridding + BlockKFold spatial CV
spacv>=0.6             # optional extra spatial-CV
mgwr>=2.2              # (multiscale) GWR
libpysal>=4.10        # spatial weights
esda>=2.5             # Moran's I / LISA hotspots
spreg>=1.4            # spatial regression
pymc>=5.10            # optional Bayesian
pymoo>=0.6            # multi-objective intervention optimization
ortools>=9.9          # ILP/CP-SAT site selection
pulp>=2.8             # optional readable ILP
deap>=1.4             # optional GA

# === Physics / thermal comfort (Layer D) ===
pvlib>=0.11           # solar geometry
metpy>=1.6            # meteo / heat index
thermofeel>=2.1       # UTCI/WBGT/MRT (array-based)
pythermalcomfort>=2.10 # UTCI/PET/PMV/WBGT (cross-verify)

# === Cloud-native data access (Layer B) ===
pystac-client>=0.8
planetary-computer>=1.0
stackstac>=0.5
odc-stac>=0.3          # optional alt cube loader
dask>=2024.4
zarr>=2.17

# === Visualization & app (Layer F) ===
streamlit>=1.33
leafmap>=0.32
folium>=0.16
pydeck>=0.9
keplergl>=0.3          # optional
plotly>=5.20
matplotlib>=3.8
localtileserver>=0.10
rio-cogeo>=5.1         # build COGs for titiler/leafmap

# === Utilities ===
numpy>=1.26
pandas>=2.2
scipy>=1.12
tqdm>=4.66
joblib>=1.4
pyyaml>=6.0

# ================= HEAVY / OPTIONAL EXTRAS (separate env or extras_require) =================
# natcap.invest>=3.14        # InVEST Urban Cooling (°C reduction) — heavy; pulls GDAL/taskgraph
# invest-ucm-calibration     # calibrate InVEST UCM to local LST
# supy>=2024.x               # SUEWS energy-balance (physics validation)
# torch>=2.2                 # deep learning / super-res / PINN — large, GPU optional
# pytorch-lightning>=2.2
# NOTE: UMEP / SOLWEIG run as a QGIS plugin (not pip) for SVF/Tmrt; SuPy gives SUEWS in pure Python.
```

---

## 5. System dependencies & environment notes

- **GDAL is the critical native dependency.** `rasterio`, `rioxarray`, `geopandas`, `fiona`, `pyproj`, `stackstac`, `osmnx` all sit on GDAL/GEOS/PROJ. **Strongly prefer `conda-forge` (or `micromamba`)** to get a single consistent GDAL ABI and avoid wheel/`apt` GDAL version clashes. If you must use `pip` on Linux, install system libs first: `gdal-bin libgdal-dev libgeos-dev libproj-dev libspatialindex-dev` and match `pip install "GDAL==$(gdal-config --version)"`. `rtree` needs `libspatialindex`.
- **`ortools`** ships manylinux wheels (no system solver needed); **`pulp`** bundles CBC.
- **GEE auth:** `earthengine authenticate` (interactive) or a **service-account JSON** for server/CI (`ee.ServiceAccountCredentials`). A registered Cloud project is required. This keeps the heavy compute server-side (the O(1) goal) — only small results/exports come back.
- **InVEST (`natcap.invest`)** pulls its own GDAL/`taskgraph`/`pygeoprocessing` chain — isolate in its own conda env (or a container) to avoid version fights with the core app; call it as a subprocess/service. Min InVEST ≥3.11 for `invest-ucm-calibration`.
- **UMEP/SOLWEIG** are **QGIS plugins**, not pip packages — run via QGIS (`processing_umep`) or headless `qgis_process`; for pure-Python energy balance use **`supy`** (SUEWS) instead. SOLWEIG supports tiled/GPU for big rasters.
- **Heavy deps to gate behind extras:** `torch`/`pytorch-lightning` (GPU optional, large wheels), `natcap.invest`, `supy`, `keplergl`. Keep the Streamlit app's runtime image free of these unless a page needs them.
- **`titiler`** is a FastAPI service (deploy separately or use hosted `titiler.xyz`); for local dev `localtileserver` is enough.
- **Repro:** lock with `conda-lock`/`pip-tools` (`requirements.lock`) once resolved; pin GDAL explicitly.

---

## 6. How this maps to "≥30 cross-verifying methods" (quick wins from the ecosystem)

A non-exhaustive list of *independent* method/data sources the stack gives us cheaply (each is a verification lane):

- **LST retrieval:** (1) Ermida SMW (`ee_lst`), (2) raw Landsat C2 ST band, (3) MODIS MOD11/MYD11 LST, (4) ECOSTRESS LST, (5) gap-filled MODIS daily LST (catalog), (6) GSHTD seamless LST.
- **SUHI/UHI:** (7) UHII catalog (surface+canopy), (8) LCZ-stratified urban-minus-rural, (9) geemap UHI tutorial method, (10) air-temp datasets (catalog) vs LST.
- **Drivers:** (11) NDVI, (12) NDBI, (13) NDWI/albedo via `eemont` indices, (14) ESA WorldCover / Dynamic World land cover, (15) GHSL built-up/volume/height, (16) UT-GLOBUS building height, (17) `momepy` morphometrics, (18) `osmnx` network density, (19) impervious fraction.
- **Models/attribution:** (20) XGBoost, (21) LightGBM, (22) CatBoost, (23) Random Forest (sklearn), (24) GWR/MGWR, (25) OLS/statsmodels, (26) PyMC Bayesian, (27) SHAP attribution, (28) verde BlockKFold spatial CV (validation method).
- **Physics/comfort:** (29) InVEST UCM heat-mitigation index, (30) SUEWS energy balance (`supy`), (31) SOLWEIG Tmrt, (32) thermofeel UTCI/WBGT, (33) pythermalcomfort PET/UTCI, (34) pvlib insolation/shade.
- **Optimization (intervention °C):** (35) pymoo NSGA-II multi-objective, (36) OR-Tools ILP budget-constrained, (37) DEAP/PyGAD GA — cross-checked against InVEST/SOLWEIG °C deltas.

That comfortably exceeds 30 distinct, mutually-verifying methods, with GEE carrying the fast path.

---

## 7. References (URLs)

- Ermida LST repo — https://github.com/sofiaermida/Landsat_SMW_LST ; paper — https://www.mdpi.com/2072-4292/12/9/1471
- ee_lst (Python port) — https://github.com/lunasilvestre/ee_lst ; lst_landsat — https://github.com/leonsnill/lst_landsat
- geemap — https://github.com/gee-community/geemap ; book — https://book.geemap.org/
- eemont — https://github.com/davemlz/eemont ; wxee — https://github.com/aazuspan/wxee ; geetools — https://github.com/gee-community/gee_tools
- Awesome-GEE Community Catalog — https://gee-community-catalog.org/ ; repo — https://github.com/samapriya/awesome-gee-community-datasets
- UHII dataset — https://gee-community-catalog.org/projects/uhii/ (assets `projects/sat-io/open-datasets/UHII/...`)
- Global LCZ map — https://developers.google.com/earth-engine/datasets/catalog/RUB_RUBCLIM_LCZ_global_lcz_map_latest ; ESSD — https://essd.copernicus.org/articles/14/3835/2022/ ; LCZ Generator — https://lcz-generator.rub.de/ ; multitemporal — https://github.com/matthiasdemuzere/multitemporal-lcz-mapping
- WUDAPT — https://www.wudapt.org/ ; LCZ Generator paper — https://www.frontiersin.org/articles/10.3389/fenvs.2021.637455/full
- UT-GLOBUS — https://www.nature.com/articles/s41597-024-03719-w ; data — https://zenodo.org/records/11156602 ; catalog — https://gee-community-catalog.org/projects/utglobus/ ; GlobalMapper — https://github.com/Arking1995/GlobalMapper
- GHSL on catalog — https://gee-community-catalog.org/projects/ghsl/
- momepy — https://github.com/pysal/momepy ; docs — http://docs.momepy.org/ ; JOSS — https://joss.theoj.org/papers/10.21105/joss.01807
- osmnx — https://github.com/gboeing/osmnx ; PyPI — https://pypi.org/project/osmnx/
- UMEP processing — https://plugins.qgis.org/plugins/processing_umep/ ; docs — https://umep-docs.readthedocs.io/ ; SuPy — https://supy.readthedocs.io/ ; UMEP paper — https://www.sciencedirect.com/science/article/pii/S1364815217304140
- SOLWEIG — https://plugins.qgis.org/plugins/solweig_qgis/
- InVEST Urban Cooling — https://invest.readthedocs.io/en/latest/api/natcap.invest.urban_cooling_model.html ; product — https://naturalcapitalproject.stanford.edu/software/invest-models/urban-cooling
- invest-ucm-calibration — https://github.com/martibosch/invest-ucm-calibration ; example — https://github.com/martibosch/lausanne-heat-islands ; PyPI — https://pypi.org/project/invest-ucm-calibration/
- thermofeel — https://github.com/ecmwf/thermofeel ; paper — https://www.sciencedirect.com/science/article/pii/S2352711022000176
- pythermalcomfort — https://pythermalcomfort.readthedocs.io/ ; pvlib — https://pvlib-python.readthedocs.io/ ; MetPy — https://unidata.github.io/MetPy/ ; pywbgt — https://github.com/kwodzicki/pywbgt
- mgwr — https://github.com/pysal/mgwr ; docs — https://mgwr.readthedocs.io/
- verde — https://github.com/fatiando/verde ; BlockKFold — https://www.fatiando.org/verde/latest/gallery/blockkfold.html ; spacv — https://pypi.org/project/spacv/
- pymoo — https://github.com/anyoptimization/pymoo ; DEAP — https://github.com/DEAP/deap ; OR-Tools — https://github.com/google/or-tools ; tree-placement optimization — https://arxiv.org/abs/2310.05691
- STAC: pystac-client — https://pystac.readthedocs.io/ ; planetary-computer — https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/ ; stackstac — https://pypi.org/project/stackstac/ ; odc-stac — https://odc-stac.readthedocs.io/
- leafmap — https://github.com/opengeos/leafmap ; localtileserver — https://github.com/banesullivan/localtileserver ; TiTiler — https://github.com/developmentseed/titiler

---
*Prepared by research agent R10 (open-source ecosystem & library stack). Web-verified June 2026; resolve exact version pins and re-check PyPI/conda-forge availability at build time. Items marked [from-knowledge] are standard packages worth a quick existence check before pinning.*
