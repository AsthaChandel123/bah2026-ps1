# `urbanheat` — Physics-Informed AI/ML for Urban Heat Mitigation

**ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 1**
*Optimizing Urban Heat Mitigation and cooling strategies via Artificial Intelligence and Machine Learning, backed by physics-informed decision making.*

> A multi-satellite, physics-informed geospatial AI/ML system that **maps urban heat-stress hotspots**,
> **quantifies the drivers of urban heating**, **models the LST↔drivers relationship with physics-informed
> ML**, and **simulates + optimizes cooling interventions** — returning the intervention **type**, **spatial
> placement**, and **estimated temperature reduction (°C)**. India-focused, GEE-fast, and demonstrable with
> **zero credentials** via a built-in synthetic mode.

---

## The problem (PS-1)

Cities heat unevenly. Impervious surfaces, low albedo, sparse vegetation, deep street canyons and
anthropogenic heat redistribute the surface energy budget away from evaporative cooling and into stored
and sensible heat — producing dangerous, inequitably-distributed **urban heat-stress hotspots**. PS-1 asks
for a framework that identifies those hotspots from satellite + meteorological data, quantifies *why* they
are hot, models the physics with AI/ML, and proposes *optimal* cooling interventions with a quantified °C
benefit and a place to put them.

## What it does (the four PS-1 deliverables)

1. **Heat-stress maps identifying hotspots** — a layered, defensible 5-class composite: surface hotspots
   (LST percentile + UTFVI + SUHII) gated by **statistically-significant clustering** (Getis-Ord Gi* and
   local Moran's I), human heat-stress indices (wet-bulb, Heat Index, Humidex, WBGT, UTCI), and a
   **vulnerability-weighted** (Heat Vulnerability Index) priority layer.
2. **Quantitative driver assessment** — ranked, %-contribution attribution across the four PS-1 families
   (**land use/land cover, urban morphology, vegetation, atmosphere**) via mean|SHAP| + ALE + variance
   partitioning, with spatially-varying **GWR/MGWR** coefficient maps and a physics-sign audit.
3. **A validated AIML model** — a **physics-informed** hybrid (surface-energy-balance backbone +
   monotone-constrained gradient-boosting ensemble + MGWR + optional PINN with an SEB-closure loss),
   validated with **spatial cross-validation** and physics-consistency checks, with per-pixel uncertainty.
4. **Optimal cooling strategy** — counterfactual ΔLST/ΔT_air per intervention (greening, cool roofs,
   albedo, water bodies …), cross-checked by an **InVEST Urban Cooling** port and a **SOLWEIG** hook, then a
   **lazy-greedy submodular + ILP + NSGA-II** optimizer that returns a ranked portfolio (**type · placement ·
   °C ± σ**) under budget / area / equity constraints.

## Why it stands out

- **Fastest "O(1)" server-side compute.** Heavy raster work runs on **Google Earth Engine**; the client
  submits a recipe and gets back small reduced results — **client effort is ≈ constant regardless of AOI
  size**. A STAC / Microsoft Planetary Computer fallback provides portability and covers GEE catalogue gaps
  (e.g. ECOSTRESS over India).
- **Many satellites, no single source.** **≥30 cross-verifying methods/datasets** (5 LST sensors, 4 LULC
  products, 4 building-footprint sources, 3 reanalyses + ground stations, ~19 analytical methods) that
  **fill each other's gaps** and **cross-check** one another by design (orthogonal errors).
- **Physics-informed throughout.** The surface energy balance and the radiative law `L↑ = εσTs⁴` are baked
  into the model as monotonic sign constraints and an optional PINN loss, so cooling counterfactuals are
  **energy-conserving and correctly-signed**, not arbitrary extrapolations.
- **Demonstrable anywhere.** A **source-agnostic dual backend** means the *entire* pipeline runs offline on
  synthetic-but-physically-plausible data — no GEE account, no network — for instant demos and CI.
- **India-first.** City presets (Delhi, Mumbai, Hyderabad, Ahmedabad, Bengaluru), pre-monsoon worst-case
  window, soil-moisture-conditioned cooling, INSAT/Bhuvan/CPCB/IMD sources, IMD heat-wave criteria,
  Census-based HVI.

---

## Architecture at a glance

```
        DATA BACKENDS                FEATURE STACK                MODELLING & OPTIMIZATION
  ┌──────────────────────┐      (co-registered 2-D layers   ┌────────────────────────────────┐
  │ GEEDataSource  (O(1)) │────►  + transform + CRS + meta) ─│ indices: SUHII/UTFVI/Gi*/Moran │
  │   server-side Earth   │      ┌───────────────────────┐  │          + comfort + HVI       │
  │   Engine compute      │      │     FeatureStack       │  │ physics: SEB backbone + PINN   │
  ├──────────────────────┤ ───► │  lst, ndvi, albedo,    │─►│ models : monotone GBM + MGWR   │
  │ SyntheticDataSource  │      │  impervious_frac, svf, │  │          + SHAP/ALE attribution │
  │   offline demo/tests │      │  air_temp, ...         │  │          + spatial-CV validation│
  │   (no GEE / no net)  │      └───────────────────────┘  │ interventions: ΔLST + InVEST +  │
  └──────────────────────┘                                  │   SOLWEIG + greedy/ILP/NSGA-II │
            ▲                                                └────────────────┬───────────────┘
            │  one DataSource interface, two interchangeable backends         ▼
   STAC/Planetary-Computer fallback (portability + ECOSTRESS)        viz: maps + report
```

The full design and the **module interface contracts** (exact signatures for every module) are in
[`ARCHITECTURE.md`](ARCHITECTURE.md). The single source of truth for dataset IDs/scale-factors/formulas/
thresholds is [`urbanheat/constants.py`](urbanheat/constants.py); the verified source material is in
[`research/`](research/).

---

## Install

The GDAL stack (`rasterio`, `geopandas`, `rioxarray`, `osmnx`) is happiest from **conda-forge**.

**conda (recommended):**
```bash
conda env create -f environment.yml
conda activate urbanheat
pip install -e ".[dev]"
```

**pip (install system GDAL first on Linux):**
```bash
sudo apt-get install -y gdal-bin libgdal-dev libgeos-dev libproj-dev libspatialindex-dev
pip install -e ".[dev]"           # lean core (synthetic + GEE paths)
pip install -e ".[all]"           # everything (adds torch, mgwr, InVEST, comfort, app, ...)
```

**Earth Engine auth (only for `--mode gee`):**
```bash
earthengine authenticate                     # interactive OAuth
# or, headless: pass a service-account key by PATH and a GCP project (never commit the key)
```

Optional heavy extras are opt-in to keep the base install lean and credential/GPU-free:
`.[gee] .[stac] .[app] .[morphology] .[comfort] .[optimize] .[spatial] .[physics] .[dl]`.

---

## Quickstart

### Synthetic demo mode (no GEE, no network)
Runs the **entire** pipeline — feature engineering → hotspots → physics-informed ML → attribution →
intervention simulation → optimization → maps/report — on physically-plausible synthetic fields.
```bash
make demo                               # Delhi, pre-monsoon, synthetic
urbanheat run --mode synthetic --city Mumbai --output-dir outputs
make app                                # interactive Streamlit dashboard
```
```python
from urbanheat import Config, get_data_source
from urbanheat.cli import run_pipeline

cfg = Config()                          # zero-arg: Delhi, synthetic, 100 m grid
results = run_pipeline(cfg)             # FeatureStack -> ... -> report
print(results["metrics"], results["report_path"])
```

### GEE production mode (the "O(1)" path)
```bash
earthengine authenticate
urbanheat run --mode gee --city Hyderabad --gee-project <your-gcp-project> --resolution 30
```
```python
from urbanheat import Config
from urbanheat.cli import run_pipeline

cfg = Config.from_city("Ahmedabad", mode="gee", gee_project="my-gcp-project",
                       start_date="2024-03-01", end_date="2024-05-31")
results = run_pipeline(cfg)
```

The only code difference between the two modes is `mode="synthetic"` vs `mode="gee"` — everything
downstream is identical because both backends return the same `FeatureStack`.

---

## Repo layout

```
urbanheat/                       installable package (`urbanheat` console script)
├── config.py  constants.py  datamodel.py   foundation: Config, the catalog, FeatureStack + DataSource
├── cli.py                                   orchestration + `run_pipeline`
├── gee/                                     GEE backend (auth, collections, lst, lulc, meteo,
│                                            morphology, fusion, features, source)
├── synthetic/source.py                      offline synthetic backend (demo + tests)
├── indices/  (heat_indices, hotspots)       spectral/comfort indices + Gi*/Moran + 5-class composite + HVI
├── physics/  (energy_balance, pinn)         surface-energy-balance + physics-informed neural net
├── models/   (features, train,              physics-informed ML core + SHAP/ALE/GWR attribution +
│             attribution, validation)       spatial cross-validation
├── interventions/ (catalog, simulate,       cooling catalog + counterfactual ΔLST + InVEST port +
│                  invest_cooling, optimize) greedy/ILP/NSGA-II placement optimizer
├── fusion/robustness.py                     ensemble agreement, multi-sensor reconcile, MC uncertainty
└── viz/      (maps, report)                 interactive/static maps + the deliverable report
app/streamlit_app.py · notebooks/ · tests/ · data/ (gitignored) · outputs/ (gitignored)
research/01..10_*.md                         verified domain research (the source of every ID/formula)
ARCHITECTURE.md                              deep design + module interface contracts (the build contract)
```

---

## Datasets used (selected; full catalog in `urbanheat/constants.py`)

- **LST / thermal:** Landsat 8/9 C2-L2 (`ST_B10`, ×0.00341802 +149.0 → K), MODIS Terra+Aqua
  (`MOD11A1`/`MYD11A1` SW + `MOD21`/`MYD21` TES, ×0.02 → K), VIIRS (`VNP21`, TES, 750 m), ECOSTRESS
  (70 m TES, LP-DAAC for India), ASTER GED emissivity; INSAT-3D/3DR/3DS (MOSDAC, India geostationary).
- **LULC / vegetation / surface:** Dynamic World, ESA WorldCover, ESRI LULC; Sentinel-2 SR; MODIS VI /
  LAI / ET (MOD13/MOD15/MOD16/PML); Hansen & MOD44B tree cover; MCD43A3 albedo; SMAP soil moisture.
- **Urban morphology / terrain:** GHSL (built surface/height/volume/population), Google Open Buildings v3 +
  2.5D, **UT-GLOBUS** heights/UCPs, Copernicus GLO-30 DSM + FABDEM bare-earth, Global LCZ map.
- **Meteorology / atmosphere:** ERA5-Land + ERA5 (BLH), GLDAS-2.1 fluxes, MERRA-2; MAIAC AOD, Sentinel-5P
  NO₂; VIIRS Black Marble nightlights; CPCB + IMD ground networks; NASA POWER solar.

## Methods

LST retrieval (split-window / TES / single-channel) and multi-sensor **fusion + gap-filling** (thermal
sharpening with mass-conserving residual, ESTARFM/FSDAF, diurnal-cycle normalization, all-sky/ERA5 prior,
Triple Collocation, uncertainty-weighted ensemble) · spectral indices + surface-energy-balance physics ·
**hotspot statistics** (LST percentile/z-score, UTFVI/EEI, SUHII, Getis-Ord Gi*, local Moran's I) ·
**heat-stress indices** (wet-bulb/Stull, NWS Heat Index, Humidex, WBGT, UTCI/PET) · **HVI** (PCA over
exposure/sensitivity/adaptive-capacity) · **physics-informed ML** (monotone XGBoost/LightGBM/Extra-Trees +
MGWR + PINN) · **attribution** (SHAP, ALE, variance partitioning, GWR) · **validation** (spatial block CV,
buffered SLOO, metric panel, physics-consistency) · **intervention modelling** (InVEST Urban Cooling,
SOLWEIG Tmrt) · **optimization** (lazy-greedy submodular, ILP, NSGA-II) with equity/HVI weighting.

---

## Credits

Built for the **ISRO Bharatiya Antariksh Hackathon 2026 (Problem Statement 1)**.

Stands on the open-source geospatial + scientific Python ecosystem: Google Earth Engine (`earthengine-api`,
`geemap`, `eemont`, `wxee`), `rasterio`/`rioxarray`/`xarray`/`geopandas`, `scikit-learn` /
`xgboost`/`lightgbm` / `shap`, `verde` / PySAL (`esda`, `libpysal`, `mgwr`), `pymoo` / OR-Tools, `momepy` /
`osmnx`, `thermofeel` / `pythermalcomfort` / MetPy, the Natural Capital Project **InVEST Urban Cooling**
model, **SOLWEIG/UMEP**, **UT-GLOBUS**, the **GHSL** suite, ESA **WorldCover**, Google **Dynamic World**, the
global **LCZ** map, and the many satellite missions (Landsat, Sentinel, MODIS, VIIRS, ECOSTRESS, ERA5,
INSAT) whose data make this possible.

Method and dataset provenance for every number is documented inline in `urbanheat/constants.py` and in the
ten verified research notes under [`research/`](research/).

## License

MIT (see `pyproject.toml`). Note: FABDEM is CC-BY-NC-SA (research/hackathon use); flag for any commercial
productization.
