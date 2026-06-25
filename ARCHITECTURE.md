# ARCHITECTURE — `urbanheat`
### Physics-informed, multi-satellite geospatial AI/ML for urban heat
**ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 1**

> This is the master architecture document and the **build contract**. Sections 1–10 explain the
> system; **Section 11 (Module Interface Contracts)** specifies the exact public signature of every
> module so 8 builders can work in parallel without coordinating. The catalog of exact dataset IDs,
> scale-factors, formulas, thresholds and parameters lives in code at `urbanheat/constants.py`
> (the single source of truth) and in `research/01..10/*.md` (the verified source material).

---

## 1. Executive summary

`urbanheat` is a Python system that ingests **many cross-verifying satellite + meteorological
datasets**, fuses them into a gap-free Land Surface Temperature (LST) field and a co-registered
driver stack, learns the **LST↔drivers** relationship with a **physics-informed** ML model, and uses
that differentiable model to **simulate and optimize cooling interventions** with a per-intervention
temperature reduction (°C) and explicit spatial placement. It is **India-first** (city presets,
pre-monsoon worst-case window, ISRO/Indian data sources) and engineered for the **fastest possible
server-side compute** via Google Earth Engine (GEE), with a STAC/Planetary-Computer fallback.

It satisfies the four PS-1 objectives and the stated **Outcomes** directly:

| PS-1 objective | How `urbanheat` delivers it | Modules |
|---|---|---|
| **1. Identify urban heat hotspots** — generate heat-stress maps from satellite + met data | A layered 5-class composite: surface hotspots (LST percentile + UTFVI + SUHII) gated by statistically-significant clustering (**Getis-Ord Gi\***, **local Moran's I**), plus human heat-stress indices (wet-bulb, Heat Index, Humidex, WBGT, UTCI) and a vulnerability-weighted (**HVI**) priority layer | `indices.heat_indices`, `indices.hotspots` |
| **2. Analyze drivers of urban heating** — quantify LULC, morphology, vegetation, atmosphere | Ranked, %-contribution **driver attribution** (mean\|SHAP\| + ALE + variance partitioning) plus spatially-varying **GWR/MGWR** coefficient maps, with a physics-sign audit and ≥2-method agreement before any claim | `models.attribution` |
| **3. Model heat dynamics with AIML** — establish LST↔drivers with physics-informed ML | A hybrid stack: **physics SEB backbone** + **monotone-constrained gradient-boosting ensemble** (signs enforced) + **MGWR spatial layer** + optional **PINN reconciler** (heat-PDE + SEB-closure loss), validated with **spatial cross-validation** and physics-consistency checks | `physics.*`, `models.*` |
| **4. Generate & optimize cooling scenarios** — simulate interventions, evaluate °C reduction, give type + placement + ΔT | **Counterfactual ΔLST/ΔT_air** by perturbing drivers and re-predicting (monotonicity-guaranteed), cross-checked by an **InVEST Urban Cooling** port and **SOLWEIG** hook, then a **lazy-greedy submodular + ILP + NSGA-II** optimizer that returns a ranked portfolio under budget/area/equity constraints | `interventions.*` |

**Outcomes produced** (the PS-1 deliverables, all assembled by `viz.report`):
heat-stress maps identifying hotspots; a quantitative driver-attribution table + maps; a validated AIML
model with spatial-CV metrics + uncertainty; scenario-based ΔT evaluation of interventions; and an
**optimal intervention strategy** (type, placement geometry, estimated °C reduction per site and city-wide).

The headline differentiator for judging is **physics throughout**: the surface energy balance is baked
into the model (as monotonic sign constraints and an optional SEB-closure PINN loss), so a +0.1 NDVI or
+0.3 albedo perturbation produces an **energy-conserving, correctly-signed** ΔLST instead of an
arbitrary extrapolation.

---

## 2. Design principles

1. **Fastest / "O(1)" server-side compute (GEE-first).** All heavy raster work (LST retrieval, energy-
   balance band math, decadal composites, zonal statistics over thousands of wards) runs **server-side
   on Google's cluster**. The client only ever submits a *recipe* (a lazy computation graph) and receives
   back *small results* — a reduced table, a thumbnail, a few thousand numbers. **Client effort and data
   egress are ≈ constant regardless of AOI size** — that is the operational meaning of "O(1)" (the total
   FLOPs are not O(1); *who pays and what crosses the wire* is). [research/07]

2. **Multi-source robustness — no single source.** A deliberate **menu of ≥30 cross-verifying
   methods/datasets** (5 LST sensors, 4 LULC products, 4 footprint sources, 3 reanalyses + station
   networks, ~19 analytical methods) where each source both **fills others' gaps** and is **itself
   verified** by an independent source. Errors are *orthogonal* by design (split-window vs TES vs single-
   channel algorithms; LUT vs retrieved vs climatology emissivity; sun-sync-morning vs afternoon vs ISS-
   precessing vs geostationary orbits). The full 35-entry robustness matrix is in `research/09`. [research/01, 09]

3. **Physics-informed throughout.** The surface energy balance `Q* = Q_H + Q_E + ΔQ_S + Q_F` and the
   radiative law `L↑ = εσT_s⁴` are first-class: a physics backbone provides the trend and guarantees
   extrapolation; monotonic constraints enforce the driver-sign table; an optional PINN puts the SEB
   residual in the loss; every counterfactual is clipped to physically attainable cooling. [research/05]

4. **Source-agnostic dual backend.** One `DataSource` interface, two interchangeable backends:
   **`GEEDataSource`** (production, the O(1) path) and **`SyntheticDataSource`** (offline demo + tests,
   generates physically-plausible LST + drivers on a grid). Everything downstream consumes a common
   **`FeatureStack`** (co-registered 2-D layers + geo-reference + metadata). This makes the entire product
   demonstrable and unit-testable **without GEE credentials or network**.

5. **Reproducibility & honesty.** The recipe *is* the algorithm; runs are seeded; every deliverable map
   ships with a paired **uncertainty** map; validation uses **spatial** cross-validation (not leaky
   random CV) and reports a full metric panel; cooling ΔT is reported as **ΔT ± σ (°C)**; out-of-training-
   envelope counterfactuals are flagged. [research/09]

6. **India-first.** City presets (Delhi, Mumbai, Hyderabad, Ahmedabad, Bengaluru with real bboxes + UTM
   zones), pre-monsoon (Mar–May) worst-case window, soil-moisture-conditioned cooling (trees cool ~4 °C
   wet vs ~1 °C dry), Indian geostationary (INSAT-3D) + ISRO/NRSC (Bhuvan) + CPCB/IMD sources, IMD
   heat-wave criteria, and Census-2011-based HVI. [research/01–04, 08]

---

## 3. System architecture (layers)

```
                        ┌──────────────────────────────────────────────────────────────┐
                        │                      SERVING (Section 7)                       │
                        │   urbanheat.cli  ·  app/streamlit_app.py  ·  viz.maps/report   │
                        │   sliders: city · date · "+trees / cool-roof / water" scenario │
                        └───────────────▲───────────────────────────────▲────────────────┘
                                        │ small JSON / PNG tiles         │ GeoTIFF / report
   ┌────────────────────────────────────┴───────────────────────────────┴────────────────────────────┐
   │                                      PIPELINE  (urbanheat orchestration)                          │
   └───────────────────────────────────────────────────────────────────────────────────────────────────┘
        │                                                                                            ▲
   (1) DATA BACKENDS  ── DataSource.load(config) -> FeatureStack ───────────────────────────────────┘
        │   GEEDataSource  (server-side EE, O(1))      SyntheticDataSource (offline demo/tests)
        │     gee.{auth,collections,lst,lulc,             synthetic.source
        │          meteo,morphology,fusion,features}
        ▼
   (2) FEATURE ENGINEERING ── indices.heat_indices (spectral + comfort) · models.features (X,y) ──────►
        │   writes canonical FeatureStack layers (NDVI, albedo, SVF, impervious_frac, air_temp, ...)
        ▼
   (3) HEAT-STRESS & HOTSPOTS ── indices.heat_indices (SUHII/UTFVI/z/percentile) ──────────────────────►
        │   indices.hotspots (Gi*, Moran's I, 5-class composite, HVI weighting)
        ▼
   (4) PHYSICS-INFORMED ML & ATTRIBUTION
        │   physics.energy_balance (SEB backbone) + physics.pinn (PDE/SEB-closure reconciler, optional)
        │   models.train (monotone GBM ensemble + MGWR) -> differentiable LST=F(drivers)
        │   models.attribution (SHAP/ALE/variance-partition/GWR)  ·  models.validation (spatial CV)
        ▼
   (5) INTERVENTION SIMULATION & OPTIMIZATION
        │   interventions.catalog -> interventions.simulate (ΔLST counterfactual, physical clip)
        │   interventions.invest_cooling (independent ΔT cross-check)  ·  SOLWEIG hook (Tmrt tiles)
        │   interventions.optimize (lazy-greedy submodular + ILP + NSGA-II; equity/HVI weighting)
        ▼
   (6) VALIDATION / ROBUSTNESS ── fusion.robustness (ensemble agreement, multi-sensor reconcile, MC UQ) ►
        │   >=30-method cross-verification accounting; per-pixel uncertainty + agreement layers
        ▼
   (7) SERVING ── viz.maps (interactive + static) · viz.report (the 5 PS-1 deliverables)
```

The **FeatureStack** flows down the entire pipeline; modules read canonical layers and write derived
ones. Layers (4)–(6) train **once** (offline) and are **O(1) at inference**, so the interactive
intervention loop in (5) and the maps in (7) stay fast.

---

## 4. The "O(1) / fastest platform" compute model

(From `research/07`.) **Primary engine = Google Earth Engine.** Every `ee.*` object is a *handle to a
computation on Google's servers*, not data in local RAM; operations compose a lazy graph; nothing runs
until a terminal pulls an **already-reduced** result across the wire.

- **Why client effort is ≈ constant.** A `reduceRegion`/`reduceRegions` over a 10 km² or a 100,000 km²
  AOI returns the **same handful of numbers / one table keyed by ward** and is the **same few lines of
  code**. Doubling the area doubles *Google's* internal EECU-time, not your code, RAM, or download. The
  only things that cross the wire are (a) metadata, (b) rendered tiles/thumbnails, (c) reduced
  aggregates / sample tables. The physics (energy-balance, emissivity) executes as `ee.Image.expression`
  on the cluster at planetary scale.
- **Canonical idiom:** `ImageCollection.filterDate().filterBounds() → .map(scale+mask+bandmath) →
  median()/reduce() → reduceRegions(wards) / sampleRegions → tiny CSV → sklearn/torch`.
- **Quotas to respect (per Cloud project; see `research/07 §2.5`):** 40 concurrent interactive / 40 high-
  volume requests; ~100 req/s; `getDownloadURL`/`getThumbURL` ≤ 32 MB (use `Export` beyond); aggregation
  cache 100 MiB; payload ≤ 10 MB; batch concurrency ~2. **EECU-time is the real cost unit.**
- **High-volume endpoint** (`earthengine-highvolume.googleapis.com`) for many small parallel tile/chip
  pulls (`config.use_highvolume=True`); interactive endpoint for few heavy calls.
- **Anti-patterns forbidden:** `getInfo()` in a loop; pulling rasters client-side for ML (use
  `sampleRegions`); `reproject` for analysis (forces eager compute); Python `for` over images (use
  `.map()`). Use `tileScale`/`maxPixels` to defeat "User memory limit exceeded".
- **STAC/COG fallback** (Microsoft Planetary Computer + AWS Open Data via `pystac-client` +
  `stackstac`/`odc-stac` + Dask): vendor-independence **and** catalogue-gap coverage. The strongest
  concrete reason it exists: **ECOSTRESS 70 m LST in GEE currently holds only Los-Angeles tiles**, so
  full ECOSTRESS over Indian cities must come from NASA LP DAAC / a STAC path. The fallback obeys the
  same discipline: STAC search returns metadata only → `stackstac.stack` builds a lazy Dask graph →
  reduce first → `.compute()` the small result.

The **`SyntheticDataSource`** is the third compute mode: it bypasses both networks entirely and generates
the FeatureStack in-process, so demos/CI never depend on credentials or connectivity.

---

## 5. Data architecture

### 5.1 Consolidated dataset catalog (chosen sources)

Exact IDs/bands/scale-factors live in `urbanheat/constants.py::GEE_DATASETS` (44 entries) and
`BAND_SCALE_OVERRIDES`. The table below is the build-time selection; **P**=primary, **S**=secondary,
**F**=fusion, **R**=reference/prior. `K = DN·scale + offset`; `°C = K − 273.15` *after* scaling.

| Domain | Dataset | GEE ID | Key band(s) | scale·DN+offset | Role | Source |
|---|---|---|---|---|---|---|
| LST | Landsat 8 C2 L2 ST | `LANDSAT/LC08/C02/T1_L2` | `ST_B10` | ×0.00341802 +149.0 → K | **P** | R1 |
| LST | Landsat 9 C2 L2 ST | `LANDSAT/LC09/C02/T1_L2` | `ST_B10` | ×0.00341802 +149.0 → K | **P** | R1 |
| LST | Landsat 5/7 ST (history) | `LANDSAT/LT05\|LE07/C02/T1_L2` | `ST_B6` | ×0.00341802 +149.0 → K | S | R1 |
| LST | MODIS Terra SW daily | `MODIS/061/MOD11A1` | `LST_Day_1km`,`LST_Night_1km` | ×0.02 → K (view_time ×0.1) | **P** | R1 |
| LST | MODIS Aqua SW daily | `MODIS/061/MYD11A1` | `LST_Day_1km`,`LST_Night_1km` | ×0.02 → K | **P** | R1 |
| LST | MODIS Terra/Aqua TES | `MODIS/061/MOD21A1D`,`MYD21A1D` | `LST_1KM` | ×0.02 → K | **F** | R1 |
| LST | VIIRS SNPP TES day/night | `NASA/VIIRS/002/VNP21A1D`,`VNP21A1N` | `LST_1KM` | ×0.02 → K | S | R1 |
| LST | ECOSTRESS 70 m TES | `NASA/ECOSTRESS/L2T_LSTE/V2` ⚠LA-only | `LST` | already K | **R** | R1 |
| Emis | ASTER GED (static) | `NASA/ASTER_GED/AG100_003` | `emissivity_band10..14` | ×0.001 | **R** | R1/R2 |
| LULC | Dynamic World NRT | `GOOGLE/DYNAMICWORLD/V1` | `built`,`trees`,`water`,… (prob) | 0–1 | **P** | R2 |
| LULC | ESA WorldCover v200 | `ESA/WorldCover/v200` | `Map` (built=50) | class | S | R2 |
| LULC | ESRI annual LULC | `projects/sat-io/.../ESRI_Global-LULC_10m_TS` | `b1` | class | S | R2 |
| Veg | Sentinel-2 SR harmonized | `COPERNICUS/S2_SR_HARMONIZED` | `B2,B3,B4,B8,B11,B12` | ×0.0001 | **P** | R2 |
| Veg | MODIS VI (gap-fill) | `MODIS/061/MOD13Q1` | `NDVI`,`EVI` | ×0.0001 | **F** | R2 |
| Veg | MODIS LAI/FPAR | `MODIS/061/MOD15A2H` | `Lai_500m`,`Fpar_500m` | ×0.1 / ×0.01 | S | R2 |
| Veg | Hansen tree cover/loss | `UMD/hansen/global_forest_change_2024_v1_12` | `treecover2000`,`lossyear` | % / year | S | R2 |
| Veg | ET (gap-filled) | `MODIS/061/MOD16A2GF` | `ET`,`LE`,`PET` | ×0.1 | S | R2 |
| Veg | ET partitioned | `projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a` | `Ec`,`Es`,`Ei` | mm | S | R2 |
| Surf | MODIS albedo (BRDF) | `MODIS/061/MCD43A3` | `Albedo_BSA_shortwave`,`WSA` | ×0.001 | **R** | R2 |
| Surf | SMAP soil moisture | `NASA/SMAP/SPL4SMGP/008` | `sm_surface`,`sm_rootzone` | m³/m³ | S | R2 |
| Morph | GHSL built surface 10 m | `JRC/GHSL/P2023A/GHS_BUILT_S_10m` | `built_surface` | m²/cell → λ_P | **P** | R4 |
| Morph | GHSL building height | `JRC/GHSL/P2023A/GHS_BUILT_H` | `built_height` | m | S | R4 |
| Morph | GHSL built volume | `JRC/GHSL/P2023A/GHS_BUILT_V` | `built_volume_total` | m³/cell | S | R4 |
| Morph | GHSL population | `JRC/GHSL/P2023A/GHS_POP` | `population_count` | persons/cell | **P** | R3/R4 |
| Morph | Open Buildings v3 | `GOOGLE/Research/open-buildings/v3/polygons` | (vector, conf≥0.70) | — | S | R4 |
| Morph | Open Buildings 2.5D height | `GOOGLE/Research/open-buildings-temporal/v1` | `building_height` | m | S | R4 |
| Morph | UT-GLOBUS heights+UCPs | `projects/sat-io/open-datasets/UT-GLOBUS/<city>` | `height` (+λ_P,λ_F,h_a) | m | S | R4 |
| Terr | Copernicus GLO-30 DSM | `COPERNICUS/DEM/GLO30` | `DEM` | m | **P** | R4 |
| Terr | FABDEM bare-earth | `projects/sat-io/open-datasets/FABDEM` | `b1` | m | S | R4 |
| Class | Global LCZ map | `RUB/RUBCLIM/LCZ/global_lcz_map/latest` | `LCZ_Filter` | 1–17 | **P** | R4 |
| Met | ERA5-Land hourly | `ECMWF/ERA5_LAND/HOURLY` | `temperature_2m`,`dewpoint…`,`u/v_10m`,`ssrd`,`strd`,… | K / accum J/m² | **P** | R3 |
| Met | ERA5 hourly (BLH) | `ECMWF/ERA5/HOURLY` | `boundary_layer_height` | m | S | R3 |
| Met | GLDAS-2.1 NOAH fluxes | `NASA/GLDAS/V021/NOAH/G025/T3H` | `Qh_tavg`,`Qle_tavg`,`Qg_tavg` | W/m² (ready) | S | R3 |
| Met | MERRA-2 (independent) | `NASA/GSFC/MERRA/slv/2` | `T2M`,`QV2M`,`U10M`,`V10M` | K / kg·kg⁻¹ | **R** | R3 |
| Atm | MODIS MAIAC AOD | `MODIS/061/MCD19A2_GRANULES` | `Optical_Depth_055` | ×0.001 | S | R3 |
| Atm | Sentinel-5P NO₂ | `COPERNICUS/S5P/OFFL/L3_NO2` | `tropospheric_NO2_column…` | mol/m² | S | R3 |
| QF | VIIRS Black Marble | `NASA/VIIRS/002/VNP46A2` | `Gap_Filled_DNB_BRDF_Corrected_NTL` | nW/cm²/sr | **P** | R3/R4 |

**Not in GEE — ingest externally** (`constants.EXTERNAL_SOURCES`): ECOSTRESS-India (LP DAAC/AppEEARS),
Sentinel-3 SLSTR (Copernicus/Planetary Computer), **INSAT-3D/3DR/3DS LST + INSOLATION** (MOSDAC — India's
geostationary diurnal anchor), CPCB CAAQMS + IMD gridded (ground truth/bias-correction), NASA POWER
(solar), Bhuvan/NRSC LULC (ISRO-authoritative cross-check).

### 5.2 Multi-sensor LST fusion & gap-filling (R1 §3, R9 §3)

No single thermal sensor gives fine space + frequent time + cloud penetration + diurnal coverage.
The fusion is a **three-stage fuse + cross-validate**:

1. **Reconcile** sensors to a common reference (overpass-time normalization via per-pixel `view_time`/DTC
   model; view-angle/BRDF normalization; CDF/quantile matching) so they blend without seams.
2. **Spatial sharpening** (coarse → fine, same time): Random-Forest / GWATPRK sharpen MODIS/VIIRS 1 km →
   30/70 m using fine predictors {NDVI, NDBI, NDWI, albedo, impervious%, building density, DEM, night-
   lights, LULC}, with a **mass-conserving residual** added so the fine field aggregates back to the
   observed coarse pixel (energy conservation — DisTrad/TsHARP/ATPRK lineage).
3. **Spatiotemporal fusion** (fine texture × frequent time): ESTARFM/FSDAF blend rare-fine
   (Landsat/ECOSTRESS) with frequent-coarse (MODIS/VIIRS) → daily 30–100 m LST.
4. **Temporal gap-fill / diurnal reconstruction**: cloud-mask first (never interpolate over cloud);
   geostationary infill (INSAT-3D + FY-4 east) under broken cloud; fit a **diurnal temperature-cycle**
   model to normalize all polar LSTs to a common time; all-sky LST + **ERA5 skin-temperature physical
   prior** for cloudy pixels; harmonic/ATC + GP/kriging for residual gaps.
5. **Uncertainty-weighted ensemble fusion**: combine per-pixel by each product's error layer (Landsat
   `ST_QA`, MODIS/VIIRS `LST_err`, ECOSTRESS `LST_err`), **not** a naive mean; flag pixels where
   split-window vs TES vs single-channel disagree > ~2 K; **Triple Collocation** gives data-driven error
   variances + optimal weights without assuming any sensor is truth.

In synthetic mode the equivalent "fusion" is degenerate (one consistent field), but the same
`fusion.robustness` API still emits agreement/uncertainty layers so downstream code is identical.

### 5.3 ≥30-method robustness matrix (R9 §4)

The **full 35-entry matrix** (each row: source/method · role · gap it fills · what verifies it) lives in
`research/09_validation_attribution_fusion.md §4`. Headline accounting is surfaced in
`constants.ROBUSTNESS_SUMMARY`: **5 LST sensors + 4 LULC products + 4 footprint sources + 3 reanalyses
(+ CPCB/IMD/Netatmo stations) + ~19 analytical methods = 35 distinct, mutually-verifying entries ≥ the
≥30 target.** The principle of **orthogonal errors** (different algorithms, emissivity handling,
orbits/times, resolutions) makes agreement strong evidence and localizes disagreement (cloud edge,
emissivity, water vapour, view angle).

---

## 6. Feature engineering & the FeatureStack schema

The **FeatureStack** (`urbanheat/datamodel.py`) is a dict of co-registered `(H, W)` float32 numpy arrays
+ affine `transform` + `crs` + `bounds` + `meta`, keyed by **canonical variable-name constants**. Every
module reads/writes these exact names. Each physical band may carry companion `*_uncertainty` and
`*_agreement` layers (R2/R9 convention).

| Variable (constant) | Name | Units | Source / how derived | Physical meaning (SEB role) |
|---|---|---|---|---|
| `LST` | `lst` | °C | fused thermal (R1) | **target**; skin temperature, the radiometric heat field |
| `LST_DAY` / `LST_NIGHT` | `lst_day`/`lst_night` | °C | MODIS/VIIRS day/night | diurnal split (day=impervious-driven, night=storage/QF) |
| `LST_UNCERTAINTY` | `lst_uncertainty` | °C | per-pixel error / TC | fusion confidence |
| `EMISSIVITY` | `emissivity` | 0–1 | ASTER GED + NDVI-ε | controls `L↑=εσTs⁴` (longwave emission) |
| `NDVI`,`EVI`,`SAVI` | … | index | S2/Landsat (R2) | vegetation amount → `Q_E` (evaporative cooling), shade |
| `NDWI`,`MNDWI` | … | index | S2/Landsat | open water → strong `Q_E` + thermal inertia |
| `NDBI`,`NDBAI`,`UI` | … | index | S2/Landsat | built/bare → `↑ΔQ_S`, `↓Q_E`, `↑β` (hotter) |
| `LAI` | `lai` | m²/m² | MOD15A2H | canopy density → transpiration (`Q_E`) |
| `FVC` | `fvc` | 0–1 | from NDVI | fractional veg → ε + LE partition |
| `ET` | `et` | mm/period | MOD16/PML | latent-heat term `Q_E` made explicit |
| `ALBEDO` | `albedo` | 0–1 | MCD43A3 + S2/Landsat | sets absorbed `(1−α)K↓` (dominant daytime input) |
| `LULC` | `lulc` | class | Dynamic World / vote | land-cover class |
| `IMPERVIOUS_FRAC` | `impervious_frac` | 0–1 | GHSL/DW/GAIA fuse | λ_P; storage `G`, thermal admittance (↑↑ LST) |
| `GREEN_FRAC` | `green_frac` | 0–1 | DW/WorldCover | vegetation fraction (cooling source) |
| `WATER_FRAC` | `water_frac` | 0–1 | DW/MNDWI | water fraction (cooling source) |
| `TREE_FRAC` | `tree_frac` | 0–1 | Hansen/MOD44B | canopy fraction (strongest veg lever) |
| `LCZ` | `lcz` | 1–17 | Global LCZ map | morphology+thermal class / stratifier |
| `BUILDING_HEIGHT` | `building_height` | m | UT-GLOBUS/GHSL/DSM−DEM | → z0, zd, H/W |
| `BUILDING_VOLUME` | `building_volume` | m³/cell | GHSL BUILT_V | thermal mass → nocturnal UHI (`ΔQ_S`) |
| `SVF` | `svf` | 0–1 | DSM hillshade-sweep / LCZ | sky view factor; **master nighttime-UHI var** (longwave trapping) |
| `ASPECT_RATIO` | `aspect_ratio` | H/W | footprints + height | canyon trapping/ventilation |
| `PLAN_AREA_FRAC` | `plan_area_frac` | 0–1 | GHSL/footprints | λ_P (daytime trapping, storage) |
| `FRONTAL_AREA_INDEX` | `frontal_area_index` | 0–1 | footprints×height | λ_F (ventilation/drag → z0) |
| `ROUGHNESS_LENGTH` | `roughness_length` | m | Macdonald(λ_P,λ_F,H) | z0; turbulent `Q_H` export efficiency |
| `DISPLACEMENT_HEIGHT` | `displacement_height` | m | Macdonald | zd |
| `ELEVATION` | `elevation` | m | GLO30/NASADEM | topographic insolation + lapse-rate covariate |
| `SLOPE` | `slope` | ° | from DEM | insolation modifier |
| `AIR_TEMP` | `air_temp` | °C | ERA5-Land downscaled | drives `Q_H` (Ts−Ta); comfort-index input |
| `DEWPOINT` | `dewpoint` | °C | ERA5-Land | humidity → `Q_E`, comfort indices |
| `REL_HUMIDITY` | `rel_humidity` | % | from T/Td | comfort indices |
| `WIND_SPEED` | `wind_speed` | m/s | ERA5-Land u/v | ventilation (`Q_H` export), comfort |
| `PRESSURE` | `pressure` | kPa | ERA5-Land | wet-bulb pressure correction |
| `SOLAR_RADIATION` | `solar_radiation` | W/m² | ERA5/POWER/INSAT, AOD-attenuated | `K↓` (radiative driver) |
| `LONGWAVE_DOWN` | `longwave_down` | W/m² | ERA5-Land `strd` | `L↓` |
| `NET_RADIATION` | `net_radiation` | W/m² | ERA5/GLDAS / `(K↓−K↑)+(L↓−L↑)` | `Q*` (available energy) |
| `SOIL_MOISTURE` | `soil_moisture` | m³/m³ | SMAP L4 | **caps achievable `Q_E`** (monsoon-critical) |
| `AOD` | `aod` | — | MAIAC | `K↓` attenuation (hazy cities) |
| `PBL_HEIGHT` | `pbl_height` | m | ERA5 BLH | stability / turbulent regime |
| `POPULATION` | `population` | persons/cell | GHSL POP | `Q_F` + exposure weighting |
| `NIGHTLIGHTS` | `nightlights` | nW/cm²/sr | VIIRS VNP46A2 | `Q_F` proxy (energy use) |
| `ANTHRO_HEAT` | `anthro_heat` | W/m² | VIIRS+POP+BUILT_V model | `Q_F` term (esp. night) |
| `NO2` | `no2` | mol/m² | S5P | combustion/traffic `Q_F` proxy |
| `SUHII` | `suhii` | °C | LST − rural ref | surface UHI intensity |
| `UTFVI` | `utfvi` | — | (Ts−Tm)/Tm in K | thermal-field-variance / EEI class basis |
| `EEI` | `eei` | class | UTFVI reclass | ecological evaluation index |
| `LST_PERCENTILE` | `lst_percentile` | 0–100 | rank in AOI | distribution-free hotspot magnitude |
| `LST_ZSCORE` | `lst_zscore` | σ | (LST−μ)/σ | hotspot magnitude |
| `GISTAR_Z` | `gistar_z` | z | Getis-Ord Gi* | significant hot-cluster z-score |
| `MORAN_LOCAL` | `moran_local` | class | local Moran's I | HH/LL/HL/LH cluster category |
| `HOTSPOT_MASK` | `hotspot_mask` | bool | (LST≥P90)∧(Gi*≥1.96) | surface hotspot |
| `HEAT_INDEX`,`HUMIDEX`,`WET_BULB`,`WBGT`,`UTCI` | … | °C | comfort formulas (R8) | human heat-stress |
| `TMRT` | `tmrt` | °C | SOLWEIG (offline) | mean radiant temp (street-scale comfort) |
| `HVI` | `hvi` | 0–1 | Exposure+Sensitivity−AdaptiveCap | vulnerability priority |
| `PRIORITY_SCORE` | `priority_score` | 0–100 | 0.5·Hazard+0.5·HVI | final layered 5-class score |

`datamodel.DRIVER_FAMILIES` groups these into the four PS-1 families {lulc, morphology, vegetation,
atmosphere} for family-level attribution; `datamodel.DEFAULT_PREDICTORS` is the default ML feature set
(the subset that exists in synthetic mode and is physically primary).

---

## 7. Heat-stress & hotspot methodology (R8)

"Heat stress" is **three physically distinct families** — conflating them is the most common error:

1. **Surface thermal metrics** (LST only, fully O(1) in GEE): `LST`, `SUHII = LST_urban − LST_rural`
   (rural reference = LCZ-D / SMOD-rural ring, ≥2 definitions reported for sensitivity), `LST_ZSCORE`,
   `LST_PERCENTILE` (distribution-free, the India default), and **`UTFVI = (Ts−Tm)/Tm`** *in Kelvin* →
   6-class **EEI** (Liu & Zhang 2011) in `constants.UTFVI_CLASSES`.
2. **Human heat-stress indices** (need air-T + humidity, ± wind/radiation): wet-bulb (Stull closed-form),
   Heat Index (NWS Rothfusz, piecewise), Humidex, Apparent Temperature, Discomfort Index, **WBGT** (ABM-
   simplified for raster, full needs globe temp), and **UTCI/PET** (need `T_mrt` from SOLWEIG → 🔴 not
   native O(1)). Danger thresholds in `constants.HEAT_STRESS_THRESHOLDS`, `UTCI_CATEGORIES`.
3. **Vulnerability-weighted risk (HVI)**: `HVI = f(Exposure, Sensitivity, −Adaptive Capacity)` (IPCC
   three-domain, `constants.HVI_DOMAINS`), built by min–max/z-score normalization + **PCA** (and equal-
   weight cross-check), classified into quintiles.

**Spatial hotspot statistics** turn noisy percentile thresholds into statistically-defensible contiguous
hotspots: **Getis-Ord Gi\*** (z-score; +z = hot cluster; `constants.HOTSPOT_GISTAR_Z` {p90:1.65, p95:1.96,
p99:2.58}; FDR-corrected) as the primary delineator, **local Moran's I** (HH/LL/HL/LH) as cross-check +
outlier flag.

**The layered 5-class composite** (the deliverable, R8 §12):
- **Layer A — Surface Heat Hotspot** (O(1)): `SurfaceHotspot = (LST ≥ P90) AND (Gi* z ≥ 1.96)`, severity
  by P90/P95/P98, cross-checked by UTFVI ≥ "strong" and Moran HH.
- **Layer B — Human Heat-Stress Hotspot**: ensemble of ≥3-of-5 cheap indices (wet-bulb/HI/Humidex/DI/ABM-
  WBGT) in danger; optional Tier-B2 SOLWEIG UTCI/PET on priority wards.
- **Layer C — Vulnerability Priority**: `PriorityScore = 0.5·HazardScore + 0.5·HVI_norm`,
  `HazardScore = max(SurfaceScore, HumanStressScore)`.
- **Legend**: `constants.HOTSPOT_LEGEND` = Low/Moderate/High/Severe/Extreme with colour-blind-safe hex
  (RdYlBu reversed); pure-surface maps use the YlOrRd ramp `constants.LST_COLOR_RAMP`.

Minimal build = Layer A only (zero ancillary data); Standard = A+B1+C; Showcase adds B2.

---

## 8. Physics-informed ML design (R5)

**Surface energy balance** (the physics the model must respect):
```
Q*  =  Q_H + Q_E + ΔQ_S + Q_F                                    (1) balance
Q*  =  (1−α)K↓ + (L↓ − L↑)                                       (2) net radiation
L↑  =  εσT_s⁴ + (1−ε)L↓                                          (3) radiative law  →  LST = invert (3)
Q_H =  ρc_p(T_s − T_a)/r_a                                       (4) sensible
Q_E =  (ρc_p/γ)(e_s(T_s) − e_a)/(r_a + r_s)                      (5) latent (vegetation enters here)
β   =  Q_H/Q_E                                                   (6) Bowen ratio (compact "why hot")
ΔQ_S = Σ_i f_i (a₁ᵢQ* + a₂ᵢ∂Q*/∂t + a₃ᵢ)                        (7) OHM storage (impervious mass)
```
The **driver→SEB→∂LST sign table** (`research/05 §1.6`) is the bridge to ML and the set of monotonicity
constraints: α↑→cooler; NDVI/water↑→cooler; impervious/NDBI↑→hotter; building-height↑ / SVF↓→hotter
(esp. night); Q_F↑→hotter (esp. night); ε↑→cooler; wind/roughness↑→cooler.

**Recommended architecture** (hybrid, cross-verifying, O(1) at inference):
- **(A) Physics backbone** — NARP `Q*` + AnOHM `ΔQ_S` + LUMPS `Q_H,Q_E` → `LST_phys`; guarantees
  extrapolation. (`physics.energy_balance`)
- **(B) Constrained ML core** — ensemble of **monotone-XGBoost + monotone-LightGBM + Extra-Trees** that
  learn the **residual** `r = LST_obs − LST_phys`, with `monotone_constraints` set from the sign table so
  interventions never go the wrong way; three learners cross-check. (`models.train`)
- **(C) Spatial layer** — **MGWR** β(s) coefficient maps (each driver acts at its own scale) + spatial
  residual kriging for local heterogeneity / gap-fill. (`models.train` + `models.attribution`)
- **(D) PINN reconciler** (optional, the headline novelty) — NN with composite loss `L = L_data +
  λ_pde·L_pde + λ_seb·L_seb + λ_mono·L_mono`: heat-PDE residual (autodiff), **SEB-closure residual**
  (Eq. 1), and monotonicity penalties; produces the physically-consistent response surface.
  (`physics.pinn`)
- **(E) Attribution** — mean\|SHAP\| + ALE (physics-sign audit) + variance partitioning + GWR maps.
  (`models.attribution`)
- **(F) Counterfactual engine** — `ΔLST = F(X+ΔX) − F(X)`, monotonicity-guaranteed, SEB-closed,
  physically clipped. (`interventions.simulate`)
- **(G) Uncertainty** — GP/Bayesian / quantile / conformal per-pixel σ(LST) and σ(ΔLST).

**Minimal viable** (time-boxed): monotone-XGBoost (B) + SHAP (E) + MGWR (C) + a thin SEB-residual
penalty ≈ 80% of the value; add the PINN (D) as the differentiator.

**Validation** (R9): **spatial** cross-validation (BlockKFold + buffered SLOO, block size ≥ residual
variogram range) as the headline (random-CV reported only as a "leaky upper bound", ~28% optimism);
forward-chaining temporal CV for time series; the metric panel `constants.VALIDATION_METRICS` (RMSE, MAE,
bias, ubRMSE, R², NSE, CCC, KGE) per LCZ/LULC stratum; residual Moran's I; and **physics-consistency
checks** (SEB closure ≈ 0; every ALE/SHAP sign matches the table; mass-preservation in downscaling;
ML fluxes vs SUEWS). Anchors to beat in `constants.VALIDATION_ANCHORS` (Extra-Trees LST R²≈0.908 /
RMSE≈0.745 °C; XGBoost SUHII R²≈0.879).

---

## 9. Cooling intervention & optimization design (R6)

Every passive intervention acts on a SEB lever: ↑albedo (cut absorbed K↓); ↑Q_E (ET/evaporation:
trees, green roofs/walls, parks, water, misting, permeable pavement); ↑shade (block K↓: canopy);
↓SVF/geometry (Tmrt). **Two temperature targets, never conflated**: surface `T_s` (large, 10–45 °C roof
swings — what the model is trained on) vs air `T_air`/`T_mrt` (small, 0.3–5 °C — what people feel). The
report shows **both**.

**Intervention catalog** (`constants.INTERVENTION_PARAMS`, 9 types) gives each: the SEB mechanism, cited
surface/air/Tmrt **°C ranges**, the FeatureStack driver perturbations (`perturbs`), and the feasibility
class. Magnitudes are climate/time-dependent (ranges, not points) and **non-additive** (the optimizer
re-predicts combinations). Examples: urban trees (surface 2–12 °C, −0.3 °C air per +10% canopy); cool
roofs (+0.1 α → −0.2…−0.6 °C, cheapest city-wide lever); urban parks (PCI 0.5–4.6 °C, >2 ha exports
cooling); cool pavement carries the **pedestrian-Tmrt trade-off** warning.

**Independent biophysical cross-checks** (the "many methods cross-verify" mandate):
- **InVEST Urban Cooling port** (`interventions.invest_cooling`, GEE/numpy-portable raster algebra):
  `CC = 0.6·shade + 0.2·albedo + 0.2·ETI`; `CC_park = Σ g_j·CC_j·exp(−d/d_cool)`;
  `HM = CC` unless a >2 ha park's `CC_park` dominates; `T_air_nomix = T_ref + (1−HM)·UHI_max`; `T_air =
  GaussianBlur(T_air_nomix, r)`. Defaults in `constants.INVEST_UCM` (weights 0.6/0.2/0.2, d_cool 100 m,
  r 500 m, park 2 ha). Validation anchor: Lausanne R²=0.903, RMSE=1.144 °C.
- **SOLWEIG hook** (`interventions.simulate.solweig_tmrt`): microscale Tmrt/shade truth on hotspot tiles
  (bump CDSM where trees added → recompute SVF → re-run), for pedestrian comfort and to verify tree
  ΔTmrt; uses the standalone Rust `solweig` PyPI package on demand.

**Optimization** (`interventions.optimize`): discretize the city into candidate `(site, type)` pairs
behind a feasibility mask; each perturbs drivers → `ΔT` field (local, with InVEST-style `exp(−d/d_cool)`
decay) weighted by `w_p = POP_p · HVI_p` (equity). Objective `max Σ_p w_p·ΔT_p` s.t. budget + area +
one-intervention-per-site + feasibility. Three solvers:
- **Primary: lazy-greedy submodular** (CELF) — ΔT-coverage is monotone submodular (overlap → diminishing
  returns), giving a **(1−1/e)≈0.63** guarantee (`constants.SUBMODULAR_GREEDY_BOUND`), O(1) to add the
  next site, and a **ranked placement list** ("plant here first…") — ideal for the PS-1 *placement* output.
- **Cross-check / exact: ILP** (PuLP/OR-Tools CP-SAT) on the reduced candidate set — confirms greedy is
  near-optimal, handles hard logical constraints (mutual exclusivity, must-cover vulnerable tracts).
- **Decision support: NSGA-II** (pymoo) — Pareto front over {cooling, −cost, co-benefit, equity}.

**Output per selected site**: intervention **TYPE**, **PLACEMENT** (geometry/coords), **estimated ΔT
(°C ± σ)** with a literature-range sanity flag, cost; plus cumulative city-wide ΔT and population-heat-
exposure reduction — the PS-1 "optimal intervention strategy" deliverable.

---

## 10. Validation & robustness (R9)

Three principles operationalized: (1) no single source trusted alone — every source has an independent
verifier (the 35-entry matrix in `research/09`); (2) validation respects geography/time — **spatial CV**
headline, design-based probability-sample accuracy as the complementary map-accuracy estimate; (3)
attribution is quantitative, ranked, spatially-explicit, with ≥2 agreeing methods before any claim.

- **Cross-sensor LST validation** against withheld sensors with published agreement envelopes (ECOSTRESS
  bias≈−0.9 K/RMSE≈2.2 K; MOD11 bias<0.8 K/RMSE<2.8 K) — if the fused product sits inside these against
  an independent sensor, that is strong evidence of validity.
- **Triple Collocation** (e.g. ECOSTRESS × MODIS × ERA5-skin) for error variances + fusion weights
  *without* assuming truth.
- **Ground truth**: CPCB (in-canopy urban T/RH, dense in metros) + IMD AWS + Netatmo (CrowdQC+-filtered
  for solar bias) for air-temp/heat-stress; ECOSTRESS/ASTER spot scenes for LST.
- **Uncertainty end-to-end**: sensor error → fusion/kriging posterior variance → classification
  probabilities → epistemic (spatial-CV-fold + bootstrap) → **Monte-Carlo propagation** to every map and
  to cooling-ΔT, surfaced as **paired uncertainty maps** and **ΔT ± σ**.

`fusion.robustness` provides the agreement/disagreement maps and the cross-verification accounting; it
runs in both backends so the demo also reports honest confidence layers.

---

## 11. MODULE INTERFACE CONTRACTS

> **This is the build contract.** Every module below lists its public functions with **exact Python
> signatures** and a one-line behavior contract, plus the FeatureStack variable names it reads/writes.
> Builders implement strictly against these. Canonical layer names (e.g. `LST`, `NDVI`, `SVF`) refer to
> the constants in `urbanheat/datamodel.py`; `fs` denotes a `FeatureStack`; `cfg` a `Config`. Heavy/
> optional libs (`ee`, `torch`, `natcap.invest`, `mgwr`, `shap`, `solweig`) MUST be imported **lazily
> inside functions**, never at module top, so the synthetic path imports without them.
>
> Conventions: functions that enrich a stack **mutate `fs` in place and also return it** (chaining).
> "writes `X`" = adds canonical layer `X`. Type names: `np.ndarray` (2-D float32 unless noted),
> `pd.DataFrame`, `gpd.GeoDataFrame`, `dict`, `Any` (model objects / `ee.*` handles kept opaque).

### 11.0 Foundation (already implemented — do not change signatures)

**`urbanheat/config.py`**
- `Config(...)` dataclass — fields per §6 of this doc; runnable with zero args in synthetic mode.
- `Config.from_city(name: str, **overrides) -> Config` — preset city + field overrides.
- `Config.from_dict(d: dict) -> Config` ; `Config.to_dict() -> dict` ; `Config.is_dataset_enabled(key: str) -> bool`.

**`urbanheat/datamodel.py`**
- `FeatureStack` — `.get(name, default=None)`, `.has(name)`, `.names()`, `.add_layer(name, array, overwrite=True)`,
  `.select(names)`, `.validate()`, `.grid_coords() -> (xx, yy)`,
  `.sample_table(variables=None, dropna=True, max_samples=None, seed=0) -> pd.DataFrame`,
  `.to_geotiff(path, variables=None) -> str`, `.to_xarray()`. Constructors `.empty(...)`, `.from_arrays(...)`.
- `DataSource(ABC)` — `.load(cfg) -> FeatureStack` (abstract); `.available_layers() -> list[str]`.
- Canonical name constants + `DRIVER_FAMILIES`, `DEFAULT_PREDICTORS`, `ALL_VARIABLES`.

**`urbanheat/constants.py`** — pure data (no functions to implement). Import values; never hard-code.

**`urbanheat/__init__.py`** — `get_data_source(cfg) -> DataSource` factory (implemented).

---

### 11.1 `urbanheat/cli.py`  *(builder: CLI/orchestration)*
```python
def main(argv: list[str] | None = None) -> int:
    """Console entry point `urbanheat`. Subcommands: run, hotspots, train,
    optimize, report, info. Parses args -> Config -> calls run_pipeline; returns exit code."""

def build_config_from_args(args) -> Config:
    """Map parsed argparse Namespace (city/bbox/dates/mode/resolution/output-dir/...) to a Config."""

def run_pipeline(cfg: Config, steps: list[str] | None = None) -> dict[str, Any]:
    """Execute the end-to-end pipeline for `cfg`: get_data_source(cfg).load -> indices ->
    hotspots -> features -> train -> attribution -> validation -> interventions -> optimize ->
    robustness -> viz/report. `steps` optionally restricts stages. Returns a results dict
    {'fs': FeatureStack, 'model': ..., 'attribution': ..., 'metrics': ..., 'portfolio': ...,
     'report_path': str}. This is the single function the app and `make demo` call."""
```
Reads: nothing (constructs). Writes: orchestrates all modules; persists to `cfg.output_dir`.

---

### 11.2 Data backends

**`urbanheat/synthetic/source.py`**  *(builder: synthetic backend — unblocks everyone; build FIRST)*
```python
class SyntheticDataSource(DataSource):
    name = "synthetic"
    def load(self, config: Config) -> FeatureStack:
        """Generate a physically-plausible FeatureStack on a grid sized from config bbox &
        resolution_m (or config.grid_shape), in config.target_crs. MUST populate at minimum:
        LST, LST_DAY, LST_NIGHT, NDVI, EVI, ALBEDO, EMISSIVITY, IMPERVIOUS_FRAC, GREEN_FRAC,
        WATER_FRAC, TREE_FRAC, LCZ, BUILDING_HEIGHT, BUILDING_VOLUME, SVF, ELEVATION, AIR_TEMP,
        DEWPOINT, REL_HUMIDITY, WIND_SPEED, SOLAR_RADIATION, NET_RADIATION, SOIL_MOISTURE,
        POPULATION, NIGHTLIGHTS, ANTHRO_HEAT. Seeded by config.seed; reproducible. LST is built
        from the synthetic drivers via the SEB sign table so that ↑NDVI/↑albedo/↑water => cooler
        and ↑impervious/↓SVF => hotter (so attribution & counterfactuals behave correctly)."""

def make_synthetic_fields(shape: tuple[int, int], seed: int = 0) -> dict[str, np.ndarray]:
    """Pure helper: return a dict of canonical-name -> 2-D array of correlated, plausible driver
    fields (spatial-coherent via smoothed noise + an urban-core gradient). No geo-referencing."""

def synthesize_lst(drivers: dict[str, np.ndarray]) -> np.ndarray:
    """Compute a synthetic LST (degC) from driver arrays using the SEB sign table
    (constants + datamodel signs). Used by load() and by tests as ground-truth physics."""
```
Reads: `cfg`. Writes: the full driver stack (see list). No optional deps (numpy only).

**`urbanheat/gee/auth.py`**  *(builder: GEE backend)*
```python
def initialize(project: str | None = None, high_volume: bool = True,
               service_account: str | None = None, key_file: str | None = None) -> None:
    """Lazy-import ee; ee.Initialize with OAuth (interactive), a service-account key, or ADC.
    When high_volume, target https://earthengine-highvolume.googleapis.com. Idempotent.
    Raise a clear error if `ee` is not installed or auth fails. [R7 §2.1]"""

def is_initialized() -> bool:
    """True if Earth Engine has been initialized in this process."""

def ee_geometry(bbox: tuple[float, float, float, float]) -> Any:
    """Return an ee.Geometry.Rectangle for an EPSG:4326 bbox (xmin,ymin,xmax,ymax)."""
```

**`urbanheat/gee/collections.py`**  *(builder: GEE backend)*
```python
def get_collection(key: str) -> Any:
    """Return ee.ImageCollection/ee.Image for a constants.GEE_DATASETS key (raises on unknown)."""

def scaled(image: Any, key: str) -> Any:
    """Apply GEE_DATASETS[key] scale/offset (+ BAND_SCALE_OVERRIDES per-band) to an ee.Image,
    returning physical-unit bands. The single place scale/offset is applied for GEE images."""

def mask_clouds(image: Any, key: str) -> Any:
    """Apply the product's QA/cloud bitmask (Landsat QA_PIXEL bits 1/3/4; MODIS/VIIRS QC;
    S2 via S2_CLOUD_PROBABILITY). Returns the masked image. [R7 §2.7]"""

def composite(key: str, cfg: Config, reducer: str = "median") -> Any:
    """filterBounds(bbox).filterDate(start,end) -> map(scaled+mask) -> reduce -> clip. Returns a
    single ee.Image cloud-free composite at native scale for dataset `key`."""
```

**`urbanheat/gee/lst.py`**  *(builder: GEE backend)*
```python
def landsat_lst(cfg: Config) -> Any:
    """Server-side Landsat C2-L2 LST composite (degC) over the AOI. Uses ST_B10 directly and an
    explicit NDVI-emissivity Planck path (constants.PLANCK_*, SPECTRAL_INDEX_COEFFS) to expose the
    physics. Returns an ee.Image with bands incl. 'lst' (degC), 'ndvi', 'emissivity'. [R7 §2.6]"""

def modis_lst(cfg: Config, which: str = "MOD11A1", day: bool = True) -> Any:
    """MODIS/VIIRS LST composite (degC) for a given product key and day/night band. ee.Image."""

def diurnal_normalize(images: dict[str, Any], ref_hour: float = 13.5) -> Any:
    """Normalize multi-sensor LST to a common local time via a diurnal-temperature-cycle model
    using per-pixel view_time bands. Returns an ee.Image of normalized LST. [R1 §3.C]"""
```

**`urbanheat/gee/lulc.py`**  *(builder: GEE backend)*
```python
def fractional_cover(cfg: Config) -> Any:
    """Server-side ensemble fractional cover (impervious/green/water/tree) from Dynamic World
    seasonal-mean probabilities + Copernicus fractions + WorldCover, co-registered. Returns an
    ee.Image with bands impervious_frac, green_frac, water_frac, tree_frac. [R2 §6.1-6.2]"""

def spectral_indices(cfg: Config) -> Any:
    """Compute NDVI/EVI/SAVI/NDWI/MNDWI/NDBI/UI + broadband albedo + NDVI-emissivity from
    S2/Landsat SR (constants.SPECTRAL_INDEX_COEFFS) as an ee.Image. [R2 §5]"""

def lcz(cfg: Config) -> Any:
    """Clip the global LCZ map to the AOI -> ee.Image band 'lcz'. [R4 §5]"""
```

**`urbanheat/gee/meteo.py`**  *(builder: GEE backend)*
```python
def era5_drivers(cfg: Config) -> Any:
    """Time-reduced ERA5-Land driver image (air_temp degC, dewpoint, rel_humidity, wind_speed,
    solar_radiation W/m2 de-accumulated, longwave_down, net_radiation, pressure, soil_moisture).
    Handles the 00-UTC accumulation gotcha for flux/radiation bands. [R3 §2.1]"""

def anthropogenic_heat(cfg: Config) -> Any:
    """QF proxy image from VIIRS nightlights x GHSL population x built-volume (+ S5P NO2),
    returning band anthro_heat (W/m2) and nightlights. [R3 §5 / R4 §4.6]"""

def downscale_air_temp(coarse_t: Any, predictors: Any, cfg: Config) -> Any:
    """Anomaly + regression downscaling of ERA5 air_temp to grid resolution using LST/NDVI/
    impervious/DEM/SVF predictors (delta-downscaling preserves the coarse mean). ee.Image. [R3 §4]"""
```

**`urbanheat/gee/morphology.py`**  *(builder: GEE backend)*
```python
def building_height(cfg: Config) -> Any:
    """Per-cell building height (m) via fallback cascade UT-GLOBUS -> Open Buildings 2.5D ->
    (GLO30 DSM - FABDEM) -> GHSL BUILT_H, masked to built pixels. ee.Image band building_height. [R4 §6.2]"""

def sky_view_factor(cfg: Config) -> Any:
    """SVF (0-1) by hillshade-sweep over the DSM (N azimuths), or LCZ-class default fallback.
    ee.Image band svf. [R4 §4.3]"""

def morphometrics(cfg: Config) -> Any:
    """Derived morphology image: plan_area_frac (lambda_P), frontal_area_index (lambda_F),
    aspect_ratio (H/W), roughness_length (z0, Macdonald), displacement_height (zd),
    building_volume, elevation, slope. Uses constants.MACDONALD_CONSTANTS. [R4 §4]"""
```

**`urbanheat/gee/fusion.py`**  *(builder: GEE backend)*
```python
def sharpen_lst(coarse: Any, predictors: Any, cfg: Config, method: str = "rf") -> Any:
    """Thermal sharpening of coarse LST to grid resolution using fine predictors with a
    mass-conserving residual (DisTrad/TsHARP/RF-ATPRK). ee.Image. [R1 §3.A / R5 §3.2]"""

def fuse_lst(sensor_images: dict[str, Any], cfg: Config) -> Any:
    """Uncertainty-weighted multi-sensor LST fusion (per-pixel error layers); returns ee.Image
    bands lst (degC) and lst_uncertainty. [R1 §3.D / R9 §3.1]"""
```

**`urbanheat/gee/features.py`**  *(builder: GEE backend)*
```python
def build_feature_image(cfg: Config) -> Any:
    """Assemble ALL driver bands into one server-side ee.Image (lst + spectral + fractional +
    meteo + morphology + QF), band-named with canonical FeatureStack names. The server-side
    counterpart of a FeatureStack. [R7 §2]"""

def sample_to_featurestack(feature_image: Any, cfg: Config) -> FeatureStack:
    """Sample/export the ee.Image at cfg resolution over the AOI into a numpy-backed FeatureStack
    (the wire-crossing step: rasters stay on GEE; only the reduced grid comes back). Sets crs=
    cfg.target_crs, transform/bounds from the AOI grid. [R7 §2.3]"""
```

**`urbanheat/gee/source.py`**  *(builder: GEE backend)*
```python
class GEEDataSource(DataSource):
    name = "gee"
    def load(self, config: Config) -> FeatureStack:
        """Production O(1) path: auth.initialize -> features.build_feature_image ->
        fusion.fuse_lst/sharpen_lst -> features.sample_to_featurestack. Returns a FeatureStack
        with the same canonical layers as SyntheticDataSource so downstream code is identical."""
```
Reads: `cfg`. Writes: full driver stack (same layer set as synthetic). All `ee` lazy.

---

### 11.3 `urbanheat/indices/`  *(builder: indices/hotspots)*

**`urbanheat/indices/heat_indices.py`**
```python
def add_spectral_indices(fs: FeatureStack) -> FeatureStack:
    """If raw reflectance present, (re)compute NDVI/EVI/SAVI/NDWI/MNDWI/NDBI/UI/FVC/ALBEDO/
    EMISSIVITY into fs (constants.SPECTRAL_INDEX_COEFFS). Idempotent. Reads reflectance/NDVI;
    writes the index layers."""

def surface_uhi(fs: FeatureStack, rural_method: str = "lcz") -> FeatureStack:
    """Compute SUHII = LST - rural_reference (rural_method in {'lcz','smod_ring','percentile'};
    LCZ-D / constants.LCZ_RURAL_REFERENCE). Writes SUHII. Reads LST, LCZ."""

def utfvi(fs: FeatureStack) -> FeatureStack:
    """UTFVI = (Ts - Tm)/Tm in KELVIN; classify to EEI via constants.UTFVI_CLASSES.
    Writes UTFVI, EEI. Reads LST."""

def lst_statistics(fs: FeatureStack) -> FeatureStack:
    """Per-pixel LST_PERCENTILE (0-100) and LST_ZSCORE over the AOI. Writes both. Reads LST."""

def heat_index(fs: FeatureStack) -> FeatureStack:
    """NWS Rothfusz Heat Index (degC, piecewise + low/high-RH adjustments). Writes HEAT_INDEX.
    Reads AIR_TEMP, REL_HUMIDITY. [R8 §5.1]"""

def humidex(fs: FeatureStack) -> FeatureStack:
    """Humidex from AIR_TEMP + DEWPOINT. Writes HUMIDEX. [R8 §5.2]"""

def wet_bulb(fs: FeatureStack) -> FeatureStack:
    """Stull (2011) closed-form wet-bulb (degC), pressure-corrected via PRESSURE if present.
    Writes WET_BULB. Reads AIR_TEMP, REL_HUMIDITY. [R8 §4]"""

def wbgt(fs: FeatureStack, method: str = "abm") -> FeatureStack:
    """WBGT: 'abm' simplified (AIR_TEMP+RH) or 'full' (needs TMRT/globe). Writes WBGT. [R8 §6.1]"""

def utci(fs: FeatureStack) -> FeatureStack:
    """UTCI 6th-order polynomial from AIR_TEMP, TMRT, WIND_SPEED, humidity (needs TMRT).
    Writes UTCI; classify via constants.UTCI_CATEGORIES. [R8 §6.3]"""

def human_stress_ensemble(fs: FeatureStack, min_agree: int = 3) -> FeatureStack:
    """Flag 'stressed' where >= min_agree of {wet_bulb, heat_index, humidex, discomfort, abm-wbgt}
    cross their danger thresholds (constants.HEAT_STRESS_THRESHOLDS). Writes a human-stress score
    layer. [R8 §12.2]"""
```

**`urbanheat/indices/hotspots.py`**
```python
def getis_ord_gi_star(fs: FeatureStack, var: str = LST, k: int = 1) -> FeatureStack:
    """Getis-Ord Gi* z-score on `var` using a (2k+1) neighbourhood kernel (Σwx, Σw, Σw^2 via
    neighbourhood reducers; global mean/SD over AOI). Writes GISTAR_Z. [R8 §9.1]"""

def local_moran(fs: FeatureStack, var: str = LST) -> FeatureStack:
    """Local Moran's I cluster category (HH/LL/HL/LH) on `var`. Writes MORAN_LOCAL. [R8 §9.2]"""

def surface_hotspots(fs: FeatureStack, percentile: float = 90.0,
                     gi_z: float = 1.96) -> FeatureStack:
    """HOTSPOT_MASK = (LST_PERCENTILE >= percentile) AND (GISTAR_Z >= gi_z). Writes HOTSPOT_MASK.
    Reads LST_PERCENTILE, GISTAR_Z (computes them if absent). [R8 §10]"""

def heat_vulnerability_index(fs: FeatureStack, census: "pd.DataFrame | None" = None,
                             method: str = "pca") -> FeatureStack:
    """Build HVI from Exposure/Sensitivity/-AdaptiveCapacity (constants.HVI_DOMAINS) via 'pca' or
    'equal' weighting; normalize 0-1. Writes HVI. Uses LST/SUHII/IMPERVIOUS_FRAC/NDVI as exposure;
    `census` joins socio-economic indicators (synthetic proxy if None). [R8 §11]"""

def composite_priority(fs: FeatureStack) -> FeatureStack:
    """PRIORITY_SCORE (0-100) = 0.5*HazardScore + 0.5*HVI_norm,
    HazardScore = max(SurfaceScore, HumanStressScore); classify to the 5-class
    constants.HOTSPOT_LEGEND. Writes PRIORITY_SCORE. [R8 §12]"""
```

---

### 11.4 `urbanheat/physics/`  *(builder: physics)*

**`urbanheat/physics/energy_balance.py`**
```python
def net_radiation(fs: FeatureStack) -> FeatureStack:
    """Q* = (1-albedo)*K_down + (L_down - eps*sigma*Ts^4) [+(1-eps)L_down]. Writes NET_RADIATION.
    Reads ALBEDO, SOLAR_RADIATION, LONGWAVE_DOWN, EMISSIVITY, LST. [R5 Eq.2-3]"""

def longwave_up(lst_c: np.ndarray, emissivity: np.ndarray,
                longwave_down: np.ndarray | None = None) -> np.ndarray:
    """L_up = eps*sigma*Ts^4 (+ (1-eps)L_down). Pure array helper, W/m2. constants.SIGMA_SB. [R5 Eq.3]"""

def sensible_heat(lst_c, air_temp_c, wind_speed, roughness_length) -> np.ndarray:
    """Q_H = rho*cp*(Ts-Ta)/r_a with r_a from wind & z0. Array helper, W/m2. [R5 Eq.4]"""

def bowen_ratio(q_h: np.ndarray, q_e: np.ndarray) -> np.ndarray:
    """beta = Q_H/Q_E (the compact 'why is it hot' number). Array helper. [R5 Eq.6]"""

def storage_heat_ohm(net_rad: np.ndarray, fractions: dict[str, np.ndarray]) -> np.ndarray:
    """OHM storage dQ_S = Σ_i f_i (a1_i*Q* + a3_i) per surface-cover fraction. W/m2. [R5 Eq.7]"""

def physics_lst(fs: FeatureStack) -> np.ndarray:
    """Physics-only LST backbone (degC): invert the SEB/radiative balance from drivers to give
    LST_phys used as the hybrid-model trend. Reads NET_RADIATION/ALBEDO/EMISSIVITY/AIR_TEMP/
    SOIL_MOISTURE/GREEN_FRAC etc. [R5 §4 (A)]"""

def seb_residual(fs: FeatureStack) -> np.ndarray:
    """|Q* - (Q_H+Q_E+dQ_S+Q_F)| closure residual map (physics-consistency check; ~0 if balanced). [R9 §1.7]"""
```

**`urbanheat/physics/pinn.py`**  *(torch lazy; optional module)*
```python
class HeatPINN:
    def __init__(self, predictors: list[str] = list(DEFAULT_PREDICTORS),
                 lambda_pde: float = 1.0, lambda_seb: float = 1.0,
                 lambda_mono: float = 1.0, **kw): ...
    def fit(self, fs: FeatureStack, epochs: int = 2000) -> "HeatPINN":
        """Train with composite loss L = L_data + λ_pde*L_pde + λ_seb*L_seb + λ_mono*L_mono
        (heat-PDE residual via autodiff, SEB-closure Eq.1, monotonicity per the sign table).
        Lazy-imports torch. [R5 §3.1]"""
    def predict(self, fs: FeatureStack) -> np.ndarray:
        """Return the physically-consistent LST response surface (degC) on the grid."""
    def predict_delta(self, fs: FeatureStack, perturb: dict[str, float]) -> np.ndarray:
        """ΔLST for a driver perturbation, SEB-closed and correctly signed. [R5 §6]"""
```

---

### 11.5 `urbanheat/models/`  *(builder: ML core)*

**`urbanheat/models/features.py`**
```python
def build_xy(fs: FeatureStack, predictors: list[str] | None = None,
             target: str = LST, dropna: bool = True,
             max_samples: int | None = 50000, seed: int = 0
             ) -> tuple["pd.DataFrame", "pd.Series", "np.ndarray"]:
    """Assemble (X, y, coords) for ML: predictors default to DEFAULT_PREDICTORS (intersected with
    present layers), target default LST. coords = (N,2) [x,y] for spatial CV. Uses fs.sample_table."""

def monotone_constraints(predictors: list[str]) -> dict[str, int]:
    """Map each predictor -> {-1,0,+1} sign from the SEB driver-sign table (datamodel signs / R5
    §1.6): albedo/ndvi/water/green/tree/svf/emissivity/wind => -1; impervious/ndbi/building_*/
    anthro_heat => +1. Feeds xgboost/lightgbm monotone_constraints. [R5 §3]"""

def predictor_grid(fs: FeatureStack, predictors: list[str]) -> np.ndarray:
    """Return an (H*W, P) matrix of predictors for full-grid prediction (NaNs preserved)."""
```

**`urbanheat/models/train.py`**
```python
def train_model(fs: FeatureStack, predictors: list[str] | None = None,
                target: str = LST, use_physics_backbone: bool = True,
                use_mgwr: bool = False, use_pinn: bool = False, seed: int = 0) -> Any:
    """Fit the physics-informed model: optional physics backbone (residual target r=LST-LST_phys),
    a monotone-constrained GBM ensemble (XGBoost+LightGBM+ExtraTrees) with signs from
    monotone_constraints, optional MGWR spatial layer and PINN reconciler. Returns a fitted model
    object exposing .predict(X)->ndarray and .predict_grid(fs)->2-D ndarray. [R5 §4]"""

def predict_lst(model: Any, fs: FeatureStack, write: bool = False) -> np.ndarray:
    """Predict LST (degC) on the full grid; if write, add a 'lst_pred' layer to fs."""

def save_model(model: Any, path: str) -> str: ...
def load_model(path: str) -> Any: ...
```

**`urbanheat/models/attribution.py`**
```python
def shap_importance(model: Any, X: "pd.DataFrame", max_samples: int = 2000
                    ) -> "pd.DataFrame":
    """Global mean(|SHAP|) per predictor (TreeSHAP; shap lazy). Returns DataFrame
    [feature, mean_abs_shap, mean_shap_signed]. [R5 §5 / R9 §2.1]"""

def family_attribution(importance: "pd.DataFrame") -> "pd.DataFrame":
    """Aggregate per-feature importance into the four PS-1 families (datamodel.DRIVER_FAMILIES) ->
    DataFrame [family, pct_contribution]. Answers 'LULC vs morphology vs vegetation vs atmosphere'."""

def ale_curves(model: Any, X: "pd.DataFrame", features: list[str]) -> dict[str, "pd.DataFrame"]:
    """Accumulated Local Effects curve per feature (unbiased under correlation). [R9 §2.1]"""

def physics_sign_audit(importance: "pd.DataFrame") -> "pd.DataFrame":
    """Check each driver's signed effect matches the SEB sign table; flag violations
    (model fitting spurious correlation). Returns [feature, expected_sign, observed_sign, ok]. [R9 §1.7]"""

def variance_partition(fs: FeatureStack, predictors: list[str], target: str = LST
                       ) -> "pd.DataFrame":
    """LMG/Shapley-regression %-share of R^2 per predictor (clean additive decomposition,
    cross-checks SHAP). Returns [feature, r2_share]. [R9 §2.2]"""

def gwr_coefficients(fs: FeatureStack, predictors: list[str], target: str = LST
                     ) -> dict[str, np.ndarray]:
    """Per-pixel GWR/MGWR coefficient maps + dominant-driver map (mgwr lazy). Returns dict of
    coefficient grids keyed by predictor + 'dominant'. [R9 §2.3]"""
```

**`urbanheat/models/validation.py`**
```python
def spatial_cv(model_factory: "Callable[[], Any]", X: "pd.DataFrame", y: "pd.Series",
               coords: np.ndarray, n_splits: int = 5, block_size: float | None = None,
               seed: int = 0) -> "pd.DataFrame":
    """Spatial block k-fold CV (verde.BlockKFold; block_size >= residual variogram range).
    Returns per-fold metrics DataFrame (constants.VALIDATION_METRICS columns). The HEADLINE
    validation. [R9 §1.2-1.4]"""

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, bias, ubRMSE, R2, NSE, CCC, KGE (constants.VALIDATION_METRICS). [R9 §1.4]"""

def stratified_metrics(y_true, y_pred, strata: np.ndarray) -> "pd.DataFrame":
    """Metrics broken out per LCZ/LULC stratum (a model good on built but poor on vegetation is
    NOT validated for cooling claims). [R9 §1.5]"""

def residual_autocorrelation(residuals: np.ndarray, coords: np.ndarray) -> float:
    """Global Moran's I of residuals (structured residuals => missing covariate / need spatial
    term). [R9 §1.5]"""
```

---

### 11.6 `urbanheat/interventions/`  *(builder: interventions/optimization)*

**`urbanheat/interventions/catalog.py`**
```python
def list_interventions() -> list[str]:
    """Keys of constants.INTERVENTION_PARAMS (the 9 intervention types)."""

def get_intervention(name: str) -> dict:
    """Return the full param dict (mechanism, surface/air/tmrt dC ranges, perturbs, feasibility)
    for one intervention. [R6 §2]"""

def feasibility_mask(fs: FeatureStack, name: str) -> np.ndarray:
    """Boolean grid where intervention `name` is feasible (e.g. cool_roof on roof/built pixels;
    urban_trees on plantable low-NDVI pixels; urban_park on vacant >=2ha; water_body on open
    space). Reads IMPERVIOUS_FRAC, NDVI, GREEN_FRAC, BUILDING_HEIGHT, LULC. [R6 §7.1]"""
```

**`urbanheat/interventions/simulate.py`**
```python
def apply_perturbation(fs: FeatureStack, name: str, mask: np.ndarray | None = None
                       ) -> FeatureStack:
    """Return a COPY of fs with the intervention's `perturbs` applied to driver layers within
    `mask` (clipped to physical ranges, e.g. albedo<=0.9, NDVI<=1). Does not mutate input. [R6 §7B]"""

def delta_lst(model: Any, fs: FeatureStack, name: str, mask: np.ndarray | None = None,
              d_cool_m: float | None = None) -> np.ndarray:
    """Counterfactual ΔLST (degC, positive=cooling) = LST(base) - LST(perturbed) via the trained
    model, with InVEST-style exp(-d/d_cool) spatial decay around treated pixels and physical
    clipping. The core of cooling simulation. [R5 §6 / R6 §7B]"""

def scenario(model: Any, fs: FeatureStack, plan: dict[str, np.ndarray]) -> dict[str, Any]:
    """Evaluate a full scenario (dict intervention_name -> placement mask): returns
    {'delta_lst': 2-D ndarray, 'mean_dC': float, 'hotspot_dC': float, 'per_type': {...}}.
    Handles NON-additivity by re-predicting from the combined perturbed drivers. [R6 Stage B]"""

def solweig_tmrt(fs: FeatureStack, cfg: Config, perturb: dict | None = None) -> np.ndarray:
    """Optional microscale Tmrt (degC) via the standalone `solweig` package on a hotspot tile
    (DSM/CDSM + meteo); used to cross-verify tree ΔTmrt. solweig imported lazily. Writes/returns
    TMRT. [R6 §4]"""
```

**`urbanheat/interventions/invest_cooling.py`**  *(numpy port; natcap.invest optional)*
```python
def cooling_capacity(fs: FeatureStack, weights: tuple[float, float, float] | None = None
                     ) -> np.ndarray:
    """CC = 0.6*shade + 0.2*albedo + 0.2*ETI (constants.INVEST_UCM weights). shade from TREE_FRAC,
    ETI from ET normalized. Returns CC grid [0,1]. [R6 §5.1]"""

def heat_mitigation(fs: FeatureStack, cc: np.ndarray | None = None,
                    d_cool_m: float | None = None) -> np.ndarray:
    """HM index with the park-cooling (>2ha) override and exp(-d/d_cool) green-area term
    (constants.INVEST_UCM). Returns HM grid. [R6 §5.2-5.3]"""

def air_temperature(fs: FeatureStack, hm: np.ndarray, t_ref: float, uhi_max: float,
                    radius_m: float | None = None) -> np.ndarray:
    """T_air = GaussianBlur(T_ref + (1-HM)*UHI_max, r). Independent ΔT estimate to cross-verify the
    ML model. Returns T_air grid (degC). [R6 §5.4]"""

def run_invest_ucm(fs: FeatureStack, cfg: Config, t_ref: float, uhi_max: float) -> dict[str, np.ndarray]:
    """Full CC->HM->T_air pass (pure-numpy port; optionally delegate to natcap.invest if installed).
    Returns {'cc':..,'hm':..,'t_air':..}. [R6 §5]"""
```

**`urbanheat/interventions/optimize.py`**
```python
def generate_candidates(fs: FeatureStack, model: Any, top_k: int = 2000
                        ) -> "pd.DataFrame":
    """Enumerate feasible (site, intervention) candidates over hotspot pixels (feasibility_mask
    per type), each with precomputed marginal ΔT, cost, area. Capped to top_k by a coarse
    pre-filter. Returns a candidate DataFrame. [R6 Stage A]"""

def equity_weights(fs: FeatureStack) -> np.ndarray:
    """w_p = POP_p * HVI_p (population x vulnerability). Reads POPULATION, HVI. [R6 §8]"""

def optimize_greedy(candidates: "pd.DataFrame", weights: np.ndarray, budget: float,
                    max_area: float) -> "pd.DataFrame":
    """Lazy-greedy submodular maximization of Σ w_p*ΔT_p under budget+area knapsack; returns a
    RANKED portfolio (insertion order = priority). (1-1/e)=0.63 guarantee. [R6 §7.2-7.3]"""

def optimize_ilp(candidates: "pd.DataFrame", weights: np.ndarray, budget: float,
                 max_area: float) -> "pd.DataFrame":
    """Exact ILP (PuLP/OR-Tools) on the reduced candidate set with hard constraints; cross-checks
    greedy near-optimality. ortools/pulp lazy. [R6 §7.3]"""

def optimize_nsga2(candidates: "pd.DataFrame", fs: FeatureStack, n_gen: int = 100
                   ) -> "pd.DataFrame":
    """NSGA-II Pareto front over {cooling, -cost, co-benefit, equity}. pymoo lazy. Returns
    non-dominated portfolios. [R6 §7.3]"""

def optimize(fs: FeatureStack, model: Any, cfg: Config) -> dict[str, Any]:
    """Top-level optimizer dispatch on cfg.optimizer_method ('greedy'|'ilp'|'nsga2') with
    cfg.optimizer_budget / max_area_frac / equity_weighting. Returns
    {'portfolio': gpd.GeoDataFrame[type, geometry, delta_C, cost], 'city_dC': float,
     'exposure_reduction': float} — the PS-1 'optimal intervention strategy'. [R6 Stage C-D]"""
```

---

### 11.7 `urbanheat/fusion/`  *(builder: fusion/robustness)*

**`urbanheat/fusion/robustness.py`**
```python
def ensemble_agreement(estimates: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack candidate estimates of one quantity -> (mean, spread) maps; high spread = low
    confidence = field-check candidate. [R9 §3.6]"""

def reconcile_sensors(sensor_arrays: dict[str, np.ndarray],
                      ref: str | None = None) -> dict[str, np.ndarray]:
    """Bias-correct/CDF-match multiple LST sensor arrays to a common reference (overpass-time/
    view-angle aware) before fusion. [R9 §3.2]"""

def uncertainty_weighted_fuse(values: list[np.ndarray], errors: list[np.ndarray]
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-error-variance weighted fusion -> (fused_value, fused_uncertainty). [R1 §3.D]"""

def triple_collocation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> dict[str, float]:
    """Error-variance of three independent datasets without truth (+ optimal weights). [R9 §3.3]"""

def monte_carlo_uncertainty(fn: "Callable", inputs: dict[str, np.ndarray],
                            errors: dict[str, np.ndarray], n: int = 100, seed: int = 0
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Propagate input errors through `fn` (LST->hotspot->ΔT) -> (mean, std) per pixel. [R9 §3.6/§5]"""

def robustness_report(fs: FeatureStack) -> dict[str, Any]:
    """Summarize the >=30-method cross-verification accounting (constants.ROBUSTNESS_SUMMARY) +
    per-layer agreement/uncertainty present in fs. Returns a dict for the report. [R9 §4]"""
```

---

### 11.8 `urbanheat/viz/`  *(builder: viz/report)*

**`urbanheat/viz/maps.py`**
```python
def hotspot_map(fs: FeatureStack, interactive: bool = True) -> Any:
    """Render PRIORITY_SCORE (or HOTSPOT_MASK) with the 5-class constants.HOTSPOT_LEGEND.
    interactive -> leafmap/folium Map; else a matplotlib Figure. leafmap/folium lazy."""

def lst_map(fs: FeatureStack, interactive: bool = True) -> Any:
    """Render LST with the YlOrRd constants.LST_COLOR_RAMP."""

def driver_map(fs: FeatureStack, var: str, interactive: bool = True) -> Any:
    """Render any canonical driver/coefficient layer with a sensible ramp + legend."""

def intervention_map(fs: FeatureStack, portfolio: "gpd.GeoDataFrame",
                     interactive: bool = True) -> Any:
    """Overlay the optimized intervention portfolio (typed/coloured by intervention, sized by
    ΔT) on the hotspot basemap."""
```

**`urbanheat/viz/report.py`**
```python
def driver_table(importance: "pd.DataFrame", families: "pd.DataFrame") -> "pd.DataFrame":
    """Assemble the deliverable driver-attribution table (per-driver mean|SHAP| °C + ALE sign +
    variance %-share + family rollup). [R5 §5]"""

def build_report(results: dict[str, Any], cfg: Config, output_dir: str | None = None) -> str:
    """Assemble the 5 PS-1 deliverables (heat-stress maps, driver-attribution table+maps,
    validated-model metrics+uncertainty, scenario ΔT maps, optimal portfolio) into an HTML/PDF
    report written under output_dir; return its path. The terminal artifact of run_pipeline."""

def export_geotiffs(fs: FeatureStack, output_dir: str,
                    variables: list[str] | None = None) -> list[str]:
    """Write key layers (LST, PRIORITY_SCORE, HVI, ΔLST, uncertainty) as COG GeoTIFFs via
    fs.to_geotiff; return the written paths. [R7 §4]"""
```

---

## 12. Repo layout, serving, security, quickstart

### 12.1 Repo layout
```
bah2026-ps1/
├── ARCHITECTURE.md              ← this document (the build contract)
├── README.md                    ← project front door (judges read first)
├── requirements.txt  environment.yml  pyproject.toml  Makefile  .gitignore
├── idea.md                      ← PS-1 problem statement (do not edit)
├── research/01..10_*.md         ← verified source material (do not edit)
├── urbanheat/                   ← the installable package
│   ├── __init__.py  config.py  constants.py  datamodel.py  cli.py
│   ├── gee/        {__init__,auth,collections,lst,lulc,meteo,morphology,fusion,features,source}.py
│   ├── synthetic/  {__init__,source}.py
│   ├── indices/    {__init__,heat_indices,hotspots}.py
│   ├── physics/    {__init__,energy_balance,pinn}.py
│   ├── models/     {__init__,features,train,attribution,validation}.py
│   ├── interventions/ {__init__,catalog,simulate,invest_cooling,optimize}.py
│   ├── fusion/     {__init__,robustness}.py
│   └── viz/        {__init__,maps,report}.py
├── app/streamlit_app.py         ← interactive dashboard
├── notebooks/                   ← demo script(s)
├── tests/                       ← pytest (synthetic-mode end-to-end + unit)
├── data/                        ← (gitignored) inputs / cached assets
└── outputs/                     ← (gitignored) generated maps / GeoTIFFs / reports
```
Foundation files (`__init__`, `config`, `constants`, `datamodel`, and all subpackage `__init__`) are
implemented and import-clean. Every other listed file is a builder target whose contract is in §11.

### 12.2 Serving / deployment
- **Primary UI**: Streamlit + `geemap`/`leafmap` (`app/streamlit_app.py`, `make app`) — fastest demo;
  city/date pickers and "+trees / cool-roof / water" scenario sliders that call `run_pipeline` and the
  `interventions.*` loop.
- **Production path** (documented, optional): FastAPI + TiTiler (dynamic COG tiles) + MapLibre/deck.gl;
  GEE service-account init; `Export.image.toAsset` for server-side intermediates.
- **CLI**: `urbanheat run|hotspots|train|optimize|report|info` (`pyproject` console-script).

### 12.3 Security (no secrets in repo)
- **No credentials committed.** `.gitignore` blocks `*-key.json`, `service-account*.json`, `gee-key*.json`,
  `.config/earthengine/`, `.env`. GEE auth is via `earthengine authenticate` (interactive) or a
  service-account key passed **by path** through `Config.gee_project` / env — never inlined.
- `data/` and `outputs/` are gitignored (regenerable; never commit large rasters).
- The synthetic path needs **no credentials at all**, so CI and public demos run with zero secrets.

### 12.4 Build/run quickstart
```bash
# install (conda recommended for the GDAL stack)
conda env create -f environment.yml && conda activate urbanheat
pip install -e ".[dev]"            # or: make setup

# OFFLINE demo — runs the whole pipeline with no GEE / no network
make demo                          # == urbanheat run --mode synthetic --city Delhi
make app                           # Streamlit dashboard

# GEE (production O(1)) mode
earthengine authenticate
urbanheat run --mode gee --city Mumbai --gee-project <your-gcp-project>

make test    # pytest (synthetic end-to-end; GEE-marked tests skipped)
make lint    # ruff + black --check
```

---

## 13. Limitations, assumptions, roadmap

**Limitations / assumptions**
- **LST ≠ air temperature ≠ thermal comfort.** Each layer is labelled correctly; the surface (SUHI) layer
  is O(1)/GEE-native, the human-stress layer needs downscaled met (and Tmrt/SOLWEIG for UTCI/PET, which is
  not O(1)).
- **ECOSTRESS in GEE = Los-Angeles only** → Indian-city ECOSTRESS needs the LP DAAC/STAC fallback;
  **Sentinel-3 SLSTR, INSAT, GOES/Himawari/SEVIRI/FY are not in GEE** → external ingest (MOSDAC for INSAT).
- **Emissivity priors (ASTER GED, Landsat ST) are 2000–2008 climatologies** → stale over fast-urbanizing
  fringes; prefer TES products (MOD21/VNP21/ECOSTRESS) there.
- **Morphology**: UT-GLOBUS per-building RMSE ≈ 9 m (excellent for block/UCP aggregates, weaker for a
  single skyscraper); FABDEM is CC-BY-NC-SA (research/hackathon OK, flag for productization).
- **HVI** uses Census-2011-era socio-economics (update with projections/NFHS/SECC; MAUP applies).
- **Synthetic mode** is for demonstration/testing of the *pipeline*, not a substitute for real data; its
  ΔT magnitudes are illustrative.
- **PINN / InVEST / SUEWS / SOLWEIG** are heavy/optional; the minimal viable model (monotone-XGBoost +
  SHAP + MGWR + thin SEB penalty) delivers ~80% of the value without them.

**Roadmap**
- Forward-compatible **pluggable sensor adapters** for **TRISHNA** (ISRO+CNES, ~2026, 60 m, 3-day,
  day+night — the single most important future Indian thermal source), LSTM (2028), SBG-TIR (2029),
  Landsat Next (~2030); the LST data layer is designed so these drop in without redesign.
- Ingest **INSAT-3D/3DR/3DS** diurnal LST + INSOLATION from MOSDAC and **Bhuvan/NRSC** LULC as GEE assets
  for stronger Indian-source credibility and a true diurnal cycle.
- Add the **all-weather (passive-microwave merge)** LST path for cloud penetration in monsoon India.
- Calibrate the **InVEST UCM** to local LST (`invest-ucm-calibration`) and add the **SUEWS** energy-balance
  time series (`supy`) as an additional physics cross-check.
- City-specific **PET/UTCI tropical recalibration** and CPCB/IMD bias-correction layers.
```
