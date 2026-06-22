# R4 — Urban Morphology, Building Footprints, 3D City Form & Terrain

**ISRO Bharatiya Antariksh Hackathon 2026 — PS-1: Physics-Informed Geospatial AI/ML for Urban Heat Hotspot Mapping**

**Domain:** The *geometry* that controls the urban canopy energy balance — building footprints, 3D building height/volume, terrain (DEM/DSM), Local Climate Zones (LCZ), and the derived morphology parameters (Sky View Factor, canyon aspect ratio, frontal/plan area indices, roughness length, anthropogenic heat proxies). These are the physical drivers that turn a flat thermal map into a *physics-informed* one.

> **Verification status.** Every dataset ID, resolution and access path below was cross-checked against the Google Earth Engine (GEE) Data Catalog, the `awesome-gee-community-catalog`, peer‑reviewed papers (Nature Scientific Data, ESSD, ERL, BAMS) and provider pages (JRC/EC, DLR, Google Research, JAXA, University of Bristol) as of **June 2026**. Items I could not pin to a live page are tagged **(from-knowledge — verify)**. Stewart & Oke (2012) Table 3 numeric values: LCZ 1 and LCZ 2 web-confirmed; LCZ 3–10 are the canonical published values **(from-knowledge — verify against the original Table 3)**.

---

## 1. Overview — why morphology is the spine of urban-heat physics

Land Surface Temperature (LST) and the canopy-layer Urban Heat Island (UHI) are not explained by land cover alone. The *3D form* of the city governs the surface energy balance through four physical mechanisms, each tied to a measurable morphological parameter:

1. **Trapped longwave radiation (nighttime UHI).** A deep, narrow street canyon has a **low Sky View Factor (SVF)**. At night, surfaces emit longwave radiation upward; with a low SVF, much of that radiation is intercepted and re-emitted by opposing walls instead of escaping to the cold sky. The canyon therefore *cools more slowly*, producing the classic **nighttime UHI**. SVF and canyon **aspect ratio H/W** are the master variables — empirically, peak nocturnal UHI intensity scales roughly with `ΔT(max) ≈ 7.45 + 3.97·ln(H/W)` (Oke 1981), i.e. it rises with H/W and falls with SVF.

2. **Shadowing & daytime trapping (daytime LST).** Tall, dense buildings shade streets (lowering daytime wall/road insolation) but also trap shortwave through multiple reflections in the canyon. Net daytime effect depends on albedo and geometry; built surface fraction (λP) and building height drive it.

3. **Reduced ventilation (heat accumulation).** **Frontal Area Index (λF)** — the building wall area facing the wind per unit ground area — sets the aerodynamic drag. High λF means low wind speed in the canopy, so sensible heat and pollutants accumulate. λF feeds the **roughness length z0** and **displacement height zd** that any boundary-layer / WRF-Urban scheme needs.

4. **Anthropogenic heat (QF).** Heat released by AC, traffic, industry and metabolism. Proxied by population density (GHS-POP) and nighttime radiance (VIIRS Black Marble). Building **volume** (GHS-BUILT-V) and floor area are proxies for cooling demand.

**Strategy for PS-1.** No single dataset gives all of this for Indian cities. We **fuse**: settlement/built-up surface (GHSL, WSF) → footprints (Google Open Buildings v3 + Microsoft + OSM, conflated by VIDA) → heights (UT-GLOBUS, GHS-BUILT-H, WSF3D, Open Buildings 2.5D Temporal, or DSM−DEM) → terrain (Copernicus GLO-30, FABDEM) → LCZ (global LCZ map) → **derived parameters** (SVF, H/W, λP, λF, z0, zd) computed server-side in GEE. Cross-verification across ≥3 independent sources fills gaps and bounds uncertainty.

---

## 2. Master comparison table (build-ready: exact IDs, resolution, coverage, role)

| # | Dataset | Exact ID / access | Native res. | Coverage / years | Type | Physical role in urban heat |
|---|---------|-------------------|-------------|------------------|------|-----------------------------|
| **Built-up & settlement** |
| 1 | GHS-BUILT-S (built-up surface 10 m) | `JRC/GHSL/P2023A/GHS_BUILT_S_10m` (GEE Image) | **10 m** | Global, 2018 | Raster | Built surface fraction λP per cell; impervious mask |
| 2 | GHS-BUILT-S (built-up surface multitemporal) | `JRC/GHSL/P2023A/GHS_BUILT_S` (GEE ImageColl.) | **100 m** | Global, 1975–2030 (5-yr) | Raster | λP trajectory; urban growth driver of UHI |
| 3 | GHS-BUILT-H (avg building height, ANBH) | `JRC/GHSL/P2023A/GHS_BUILT_H` (GEE Image), band `built_height` | **100 m** | Global, 2018 | Raster | Mean building height → z0, zd, H/W, volume |
| 4 | GHS-BUILT-V (built-up volume) | `JRC/GHSL/P2023A/GHS_BUILT_V` (GEE ImageColl.) | **100 m** | Global, 1975–2030 | Raster | Floor-volume proxy for thermal mass & QF (cooling demand) |
| 5 | GHS-BUILT-C (settlement characteristics / morphological classes) | `JRC/GHSL/P2023A/GHS_BUILT_C` (GEE Image) | **10 m** | Global, 2018 | Raster | MSZ morphological zones; open/compact discrimination |
| 6 | GHS-SMOD (Degree of Urbanization) | `JRC/GHSL/P2023A/GHS_SMOD_V2-0` (GEE ImageColl.) | **1 km** (100 m available at JRC) | Global, 1975–2030 | Raster (8 classes) | Urban centre / cluster / rural stratification for sampling |
| 7 | GHS-POP (residential population) | `JRC/GHSL/P2023A/GHS_POP` (GEE ImageColl.) | **100 m** | Global, 1975–2030 | Raster | **QF anthropogenic-heat proxy**; exposure weighting |
| 8 | World Settlement Footprint 2019 (WSF2019) | DLR EOC Geoservice / community asset `projects/sat-io/open-datasets/WSF/WSF2019` *(from-knowledge — verify)* | **10 m** | Global, 2019 | Raster (binary) | High-precision settlement mask, complements GHSL |
| 9 | WSF3D (height/volume/area/fraction) | DLR EOC Geoservice (download); community GEE asset *(from-knowledge — verify)* | **90 m** | Global (~2012–2019, TanDEM-X era) | Raster (4 layers) | Mean building height, area, **volume, fraction** → λP, z0 |
| 10 | Global Urban Boundary (GUB) | Star Cloud / Tsinghua download (GAIA-derived) | **30 m** | Global, 1990–2018 (7 epochs) | Vector | Defines analysis extent / urban-rural boundary |
| **Building footprints (vector)** |
| 11 | Google Open Buildings v3 (polygons) | `GOOGLE/Research/open-buildings/v3/polygons` (GEE FeatureColl.) | source 50 cm | Africa, **South Asia**, SE Asia, LatAm, Caribbean (58 M km², 1.8 B bldgs) | Vector | **Footprints for India** → λP, building count, frontal area |
| 12 | Open Buildings 2.5D Temporal v1 | `GOOGLE/Research/open-buildings-temporal/v1` (GEE ImageColl.) | **4 m** | Same regions, **annual 2016–2023** | Raster (presence, count, **height**) | **Per-year building height for India** (no LiDAR needed) |
| 13 | Microsoft Global Building Footprints | Source Cooperative / PMTiles; bulk on `source.coop` | source ~0.3–0.6 m | Global (1.4 B+ bldgs) | Vector | Footprints in Northern Hemisphere & where OSM/Google sparse |
| 14 | VIDA Google+Microsoft(+OSM) combined | `source.coop/vida/google-microsoft-open-buildings` (GeoParquet/FlatGeobuf/PMTiles); also `map.vida.place` | source | **Global, ~2.58 B footprints**, 92% of L0 admin | Vector | **One conflated footprint layer**; each labeled by source |
| 15 | OpenStreetMap buildings/landuse/highways | Overpass API (live); Geofabrik `download.geofabrik.de/asia/india.html` (`india-latest.osm.pbf`); GEE OSM mirrors | vector | Global, continuous; India highly variable | Vector | Footprints + `building:levels` heights; roads → canyon axes |
| **3D / building height** |
| 16 | **UT-GLOBUS** (GLObal Building heights for Urban Studies) | Zenodo `10.5281/zenodo.11156602`; GEE `projects/sat-io/open-datasets/UT-GLOBUS/<city>` (attr. `height`, m AGL) | **building-level vector** + 1 km UCP grid | **>1200 cities, all habitable continents incl. India** | Vector | **Per-building heights + UCPs (λP, λF, hₐ, λB)** — flagship 3D input |
| 17 | EUBUCCO v0.1 | Zenodo `10.5281/zenodo.7225259` | building-level vector | EU-27 + Switzerland (~202 M bldgs) | Vector | European validation/benchmark (height 73% complete) |
| 18 | Microsoft building heights *(subset)* | with MS footprints where available | vector | partial | Vector | Supplementary height where present |
| 19 | Local LiDAR / DSM (city) | survey / municipal | 0.5–2 m | city-specific | Point/raster | Gold-standard height truth for calibration |
| **Terrain / elevation (DEM vs DSM)** |
| 20 | Copernicus GLO-30 DEM | `COPERNICUS/DEM/GLO30` (GEE ImageColl., band `DEM`) | **30 m** | Global (~2011–2015, TanDEM-X) | **DSM** (incl. buildings) | DSM term in `height = DSM − DEM`; canyon geometry |
| 21 | SRTM GL1 v3 | `USGS/SRTMGL1_003` (GEE Image, band `elevation`) | **30 m** | 60°N–56°S, 2000 | DEM/DSM (C-band, ~bare-ish) | Terrain baseline; slope/aspect; hillshade |
| 22 | NASADEM | `NASA/NASADEM_HGT/001` (GEE Image, band `elevation`) | **30 m** | Near-global, 2000 (reprocessed SRTM) | DEM | Improved SRTM; terrain detrending of LST |
| 23 | ALOS AW3D30 v4.1 | `JAXA/ALOS/AW3D30/V4_1` (GEE ImageColl., band `DSM`) | **30 m** | Global | **DSM** | Alt. DSM for height differencing; input to UT-GLOBUS |
| 24 | **FABDEM** (Forest And Buildings removed) | GEE `projects/sat-io/open-datasets/FABDEM` (ImageColl.) | **30 m** | Global, V1-2 | **bare-earth DEM** | **The DEM term**: clean ground for `DSM − FABDEM = object height` |
| 25 | TanDEM-X (12 m / 30 m / 90 m) | DLR (90 m free `DEM/TDM90`-style); 12 m commercial | 12/30/90 m | Global | DSM | Highest-res global DSM (12 m restricted) |
| **Local Climate Zones & derived** |
| 26 | Global LCZ map (Demuzere et al. 2022) | `RUB/RUBCLIM/LCZ/global_lcz_map/latest` (GEE ImageColl., band `LCZ_Filter`) | **100 m** | Global | Raster (17 classes) | **One-stop morphology+thermal class**; per-class SVF/H/W/λP priors |
| 27 | WUDAPT LCZ (city level) / LCZ Generator | `lcz-generator.rub.de` (submit ROI → LCZ map) | 100 m | per-city on demand | Raster | Custom Indian-city LCZ when global map is coarse |
| 28 | VIIRS Black Marble VNP46A2 | `NASA/VIIRS/002/VNP46A2` (GEE ImageColl., band `Gap_Filled_DNB_BRDF_Corrected_NTL`) | **500 m** | Global, daily 2012– | Raster | **QF / anthropogenic-heat & activity proxy** (nightlights) |

> **DEM vs DSM — the single most important distinction in this table.** A **DSM** (Copernicus GLO-30, ALOS AW3D30, TanDEM-X, WSF3D inputs) measures the *top of objects* (buildings, trees). A **DEM/bare-earth** (FABDEM, NASADEM, MERIT) measures the *ground*. **Building height ≈ DSM − bare-earth DEM**, computed per pixel and aggregated to footprints. Using SRTM (which is partway between) as the "DEM" underestimates heights; **FABDEM is purpose-built as the bare-earth term** because it explicitly removes buildings and forests from Copernicus GLO-30.

---

## 3. Per-dataset details

### 3.1 GHSL suite (JRC, European Commission) — release P2023A
The Global Human Settlement Layer is the backbone gridded product. P2023A (2023 release) is the current generation.

- **GHS-BUILT-S — built-up surface.** Fraction of each cell covered by building roofs. Two GEE assets: **10 m** single-epoch 2018 (`..._GHS_BUILT_S_10m`) and **100 m** multitemporal 1975–2030 (`..._GHS_BUILT_S`). Derived from a Sentinel-2 composite + Landsat. **Direct measurement of plan area fraction λP** (after normalizing roof area by cell area). Open & free with attribution.
- **GHS-BUILT-H — average building height (ANBH).** GEE `JRC/GHSL/P2023A/GHS_BUILT_H`, band `built_height`, **100 m, 2018**, units **metres**. It is the **Average Net Building Height** (height averaged over built-up area only). Derived from **ALOS World 3D (AW3D30) + SRTM + a 2017–2018 Sentinel-2 composite**. This is the cheapest global per-cell height and the natural input to z0, zd and volume. *Caveat:* 100 m smoothing means individual tall buildings are blurred — use UT-GLOBUS or Open Buildings 2.5D for building-level work.
- **GHS-BUILT-V — built-up volume.** `JRC/GHSL/P2023A/GHS_BUILT_V`, **100 m, 1975–2030**. Joint assessment of Sentinel-2/Landsat + global DEM. Volume = built surface × height; serves as **thermal-mass and cooling-demand (QF) proxy**.
- **GHS-BUILT-C — settlement characteristics / Morphological Settlement Zones (MSZ).** `..._GHS_BUILT_C`, **10 m, 2018**. Classifies built cells into morphological classes (e.g., open vs compact, vegetation in built area). Useful as an LCZ cross-check.
- **GHS-SMOD — Degree of Urbanization.** `JRC/GHSL/P2023A/GHS_SMOD_V2-0`. Eight DEGURBA classes (urban centre / dense & semi-dense urban cluster / suburban / rural classes / water) built from GHS-POP + GHS-BUILT-S decision rules. **1 km** in GEE (100 m at the JRC portal). Use it to stratify hotspot sampling and to define "urban core" for UHI baselines.
- **GHS-POP — residential population.** `JRC/GHSL/P2023A/GHS_POP`, **100 m, 1975–2030**, disaggregates census (GPWv4.11-derived) onto built-up surface. **Primary anthropogenic-heat (QF) and exposure proxy.**

### 3.2 World Settlement Footprint (DLR)
- **WSF2019** — **10 m** global binary human-settlement mask from 2019 Sentinel-1 + Sentinel-2, produced entirely in GEE. Higher precision than older masks; pairs with GHS-BUILT-S to bound λP.
- **WSF3D** — **90 m** global, four layers: **mean building height, total building area, total building volume, building fraction**. Generated from the WSF mask + **TanDEM-X** elevation/radar via edge-based height-difference analysis. It is a key independent height/volume source (used inside UT-GLOBUS) and yields λP directly (building fraction). Free download via DLR EOC Geoservice; GEE community mirrors exist.

### 3.3 Global Urban Boundary (GUB)
Vector urban boundaries at **30 m** for 1990–2018 (7 epochs), delineated on GEE from GAIA impervious data (Tsinghua, Li et al. 2020, ERL). Use as the **analysis mask / urban-rural divide** and to compute UHI as urban-minus-rural LST consistently.

### 3.4 Building footprints
- **Google Open Buildings v3** — GEE `GOOGLE/Research/open-buildings/v3/polygons`. **1.8 billion** footprints from **50 cm** imagery (inferred May 2023) over Africa, **South Asia (India)**, SE Asia, LatAm, Caribbean. Per-polygon attributes: `area_in_meters`, **`confidence` (0.65–1.0)**, `full_plus_code`, `longitude_latitude`. **Recommended: filter `confidence ≥ 0.70`** (or 0.75 for high precision) to suppress false positives; lower it in sparse rural areas to gain recall. CC-BY-4.0. **This is the single best footprint layer for Indian cities.**
- **Open Buildings 2.5D Temporal v1** — GEE `GOOGLE/Research/open-buildings-temporal/v1`. **4 m** raster, **annual 2016–2023**, three bands: building **presence/fractional count** and **building height (m)**, trained on Sentinel-2 with a deep model. Combine with v3 polygons (zonal-mean height per polygon) to get **per-building height for India without LiDAR**, *and* a time series of vertical growth.
- **Microsoft Global Building Footprints** — ~1.4 B+ footprints globally from Bing imagery; some regions carry height. Best in the Northern Hemisphere and where Google/OSM are thin.
- **VIDA Google+Microsoft(+OSM) combined** — `source.coop/vida/google-microsoft-open-buildings` (and the `-osm-` variant). **~2.58 B footprints**, 185 partitions, **92% of Level-0 admin areas**, each footprint **labeled by source**. Cloud-native **GeoParquet / FlatGeobuf / PMTiles**. *This is the recommended pre-conflated footprint layer* — it already solves most of the Google-vs-Microsoft union problem.
- **OpenStreetMap** — Overpass API (live, minute-updated) for targeted city pulls; **Geofabrik** `download.geofabrik.de/asia/india.html` (`india-latest.osm.pbf`, daily) for bulk; GEE OSM mirrors for raster rasterization. Tags of interest: `building`, `building:levels`/`height`, `landuse`, `highway`. **Completeness caveat (India):** OSM building completeness exceeds 80% for only ~16% of India's urban population and is <20% for ~48% — **never rely on OSM alone in India**; conflate with Google/Microsoft.

### 3.5 UT-GLOBUS — deep dive (the flagship 3D morphology dataset)
*Marshall Shepherd / Naik / Demuzere et al., "GLObal Building heights for Urban Studies (UT-GLOBUS)", **Nature Scientific Data 11:886 (15 Aug 2024)**, DOI `10.1038/s41597-024-03719-w`; preprint arXiv:2205.12224.*

**What it is.** The first open, globally-relevant, *city-centric* dataset of **building-level heights and Urban Canopy Parameters (UCPs)** for **>1200 cities/locales across all habitable continents** (Asia incl. **India**, Europe, Americas, Africa, Oceania). It is purpose-built so that mesoscale (WRF-Urban) and microscale (SOLWEIG) modelers can replace the crude table-based LCZ approach with *real* morphology.

**Products.**
- **Vector**: individual **building polygons** with a `height` attribute in **metres above ground level** ("Level-of-Detail-1" blocks), in UTM, QGIS/ArcGIS/Python-ready (GeoPackage/shapefile-style).
- **Gridded UCPs** at **1 km²** (300 m sliding kernel) in **WRF-preprocessing binary** format. Table-1 UCPs include: **plan area fraction λP, area-averaged building height hₐ, building surface-to-plan-area ratio λB, frontal area index, building-height histogram (5 m bins), mean height, std-dev of height**. SVF and roughness lengths are explicitly noted as *derivable* from these.

**Inputs (a fusion in itself).**
- Spaceborne altimetry: **ICESat-2 ATL08** + **GEDI** (RH98 top-of-canopy).
- Coarse 3D: **ALOS World 3D (AW3D30, 30 m DSM)** + **WSF3D (90 m heights)**.
- Footprints: **OSM, Google (Southern Hemisphere), Microsoft (Northern Hemisphere)**; missing footprints synthesized with generative methods (GlobalMapper using road networks + partial footprints).
- Population: **LandScan (~1 km, 2020)** for a linear population-correction factor; **ESA WorldCover** urban fractions for WRF prep.

**Method.** A **Random Forest regressor** (≈240 trees, max depth 50, bootstrap, √features at split) maps altimetry + coarse-DSM + footprint features to per-building height. Trained on **~268,000 buildings** from **6 US LiDAR cities** (incl. New York City, Philadelphia, Boston, Los Angeles, San Francisco); 80/20 split, 3-fold CV.

**Validation.**
- **Per-building height:** testing **RMSE ≈ 9.1 m** (MBE ≈ 0.1 m) on **6 independent US cities** (**Atlanta, Austin, Chicago, Houston, Pittsburgh, San Antonio**, ~123,020 buildings); internal validation RMSE ≈ 5.4 m.
- **1 km² mean height:** **RMSE ≈ 7.8 m** across the 6 US cities **+ Hamburg (Germany) + Sydney (Australia)**.
- **Model impact:** in **WRF-Urban (Houston)**, 2 m air-temperature RMSE improved from **1.21 K → 0.53 K (~55%)** vs the table-based LCZ approach; street-scale SOLWEIG (Baltimore) Tmrt RMSE ≈ 2.85 °C.

**Access.**
- **Zenodo:** `https://doi.org/10.5281/zenodo.11156602` (per-region downloads; a `coverage_<region>.gpkg`, e.g. `coverage_asia.gpkg`, lists all cities + extents → use it to confirm which Indian cities are present).
- **GEE (community, sat-io):** per-city FeatureCollections at **`projects/sat-io/open-datasets/UT-GLOBUS/<city>`** (e.g. `.../peoria`), attribute `height` (m AGL). **1088 of 1200** cities are ingested into GEE (112 failed ingestion — for those, use Zenodo). Interactive viewer: `https://sat-io.earthengine.app/view/ut-globus`. License **CC-BY-4.0**.
- Code: `github.com/.../UT-GLOBUS` and GlobalMapper (`github.com/Arking1995/GlobalMapper`).

**Why it matters for PS-1 (India).** UT-GLOBUS is the *only* open product giving **per-building heights + ready-made UCPs (λP, λF, hₐ, λB)** for Indian cities, exactly the inputs a physics-informed LST/UHI model needs. **Action item:** open `coverage_asia.gpkg` and enumerate the covered Indian cities (Mumbai/Delhi/Bengaluru/Kolkata/Chennai/Hyderabad/Ahmedabad/Pune are expected given the >1200-city, all-continents claim; **verify the exact list before relying on it**).

**Known uncertainties (per the paper).** Errors propagate from ALOS/WSF3D resolution & bias, altimeter sparsity/urban coverage, **missing footprints**, and the linear population-correction assumption. RMSE ~9 m per building means UT-GLOBUS is excellent for **block/neighborhood aggregates and UCPs**, less so for any *single* skyscraper — reconcile with DSM-derived heights (§6).

### 3.6 EUBUCCO v0.1
~**202 million** individual building footprints for **EU-27 + Switzerland**; attributes height/age/type complete for 73%/24%/46% respectively; harmonized from 50 government datasets + OSM; mostly ODbL. Zenodo `10.5281/zenodo.7225259`. **Role for PS-1:** a clean European *benchmark* to validate fusion/UCP code before applying to India (not Indian coverage).

### 3.7 Terrain / elevation
- **Copernicus GLO-30** (`COPERNICUS/DEM/GLO30`, 30 m, **DSM**, TanDEM-X) — best global DSM in GEE; **the DSM term** for height differencing. ImageCollection (mosaic before use).
- **FABDEM** (`projects/sat-io/open-datasets/FABDEM`, 30 m, **bare-earth**) — Copernicus GLO-30 with **buildings and forests removed** via random forest; built-up MAE improved 1.61 m → **1.12 m** vs raw GLO-30. **The DEM term**: `object/building height = GLO-30 (DSM) − FABDEM (DEM)`. *License note:* **CC-BY-NC-SA 4.0 (non-commercial)** — fine for a hackathon/research deliverable; flag for any commercial productization.
- **SRTM GL1 v3** (`USGS/SRTMGL1_003`, 30 m) — terrain baseline, slope/aspect, hillshade for shadowing; 2000-era C-band.
- **NASADEM** (`NASA/NASADEM_HGT/001`, 30 m) — improved/reprocessed SRTM; good bare-earth-ish DEM for LST terrain-detrending.
- **ALOS AW3D30 v4.1** (`JAXA/ALOS/AW3D30/V4_1`, 30 m, **DSM**) — independent optical-stereo DSM; alternative numerator for height; also a UT-GLOBUS input.
- **TanDEM-X** — 12 m (commercial), 30 m, and free **90 m** (`DLR/TDM/90`-class) — highest-resolution global DSM; 12 m is the aspiration where licensing allows.

### 3.8 VIIRS Black Marble (anthropogenic heat / activity)
`NASA/VIIRS/002/VNP46A2`, **500 m, daily since 2012**, moonlight/atmosphere/terrain-corrected nighttime radiance (band `Gap_Filled_DNB_BRDF_Corrected_NTL`). **Primary QF proxy** alongside GHS-POP: nightlight radiance correlates with energy use, traffic and industrial activity → anthropogenic heat. Aggregate to monthly/annual to denoise.

---

## 4. Derived morphology parameters & formulas (the physics core)

All of the following are computed **server-side in GEE** from §2 inputs and become predictor features for the LST/UHI model. Notation: `λ` = areal index, ground cell area `AT`, building `i` with footprint area `Ap,i`, height `Hi`, frontal (wall) area `Af,i`.

### 4.1 Plan Area Fraction λP (building/plan density)
`λP = (Σ Ap,i) / AT` — fraction of ground covered by roofs.
- **Sources:** GHS-BUILT-S (10 m, direct), WSF3D building fraction, or rasterized footprints (Google/MS/VIDA).
- **Role:** controls daytime trapping and the impervious heat-storage term; high λP → more stored heat → stronger nighttime UHI. Enters z0/zd.

### 4.2 Frontal Area Index λF (per wind direction θ)
`λF(θ) = (Σ Af,i(θ)) / AT`, where `Af,i = Hi × Wi(θ)` (wall width projected normal to wind).
- **Sources:** footprints (width) × heights (UT-GLOBUS / GHS-BUILT-H / DSM−DEM). Compute for the prevailing wind direction(s).
- **Role:** **the ventilation/drag variable.** High λF → low canyon wind → poor heat dissipation → heat accumulation. Primary control on z0.

### 4.3 Canyon Aspect Ratio H/W and Sky View Factor (SVF)
**Aspect ratio** `H/W = building height / street width` (street width from road centerline buffers / footprint gaps).
**SVF** = fraction of sky hemisphere visible from a ground point (0 = fully enclosed, 1 = open field). For an infinite symmetric canyon:
`SVF_canyon = cos(arctan(2H/W)) = 1 / sqrt(1 + (2H/W)^2)` (i.e. `SVF = cos β`, with `tan β = 2H/W` for the wall-half geometry; the floor-center form is `SVF = cos(arctan(H/(W/2)))`).
- **Compute robustly from raster DSM** (preferred for irregular cities): for each ground pixel, sample max elevation angle to obstructions in N azimuth directions (e.g. 16–36) within a search radius, then
`SVF ≈ 1 − (1/N) Σ_n sin²(γ_n)`, where `γ_n` is the max obstruction elevation angle in azimuth n. Implementations: SAGA `SkyViewFactor`, GRASS `r.skyview`, UMEP/SOLWEIG, or a GEE hillshade-sweep over GLO-30/UT-GLOBUS-rasterized DSM.
- **Role — the master nighttime-UHI variable.** Low SVF / high H/W → trapped longwave → slow nocturnal cooling → strong UHI (Oke: `ΔTmax ≈ 7.45 + 3.97·ln(H/W)`). Also reduces daytime road insolation (shading) — geometry has opposite-sign day vs night effects, which is exactly why a **physics-informed** model must carry SVF/H/W explicitly rather than learn LST from land cover alone.

### 4.4 Building surface fraction λB and area-averaged height hₐ
`λB = (Σ (Ap,i + Af,i)) / AT` (roof + wall area per ground area); `hₐ = Σ(Hi·Ap,i)/Σ Ap,i`.
- **Sources:** UT-GLOBUS provides both directly (λB, hₐ).
- **Role:** total active surface for radiative/convective exchange; thermal admittance scaling.

### 4.5 Roughness length z0 and displacement height zd (Macdonald 1998)
Morphometric (no wind data needed):
```
zd/H = 1 + A^(-λP) · (λP − 1)                              [A ≈ 4.43]
z0/H = (1 − zd/H) · exp{ −[ 0.5·β·(Cd/κ²)·(1 − zd/H)·λF ]^(-0.5) }
```
with `κ = 0.40` (von Kármán), drag `Cd ≈ 1.2`, `β ≈ 1.0`, `H` = mean building height.
- **Sources:** λP (GHS-BUILT-S), λF (footprints×heights), H (GHS-BUILT-H / UT-GLOBUS hₐ).
- **Role:** the **aerodynamic inputs** any surface-energy-balance / WRF-Urban / boundary-layer scheme requires; they set turbulent transport of sensible heat out of the canopy. z0 rises with λF then falls at very high density (sheltering) — capture this non-monotonicity.
- *Alternative:* Kanda (2013) or Kent (2017) give updated z0/zd fits; LCZ-table z0 (terrain roughness class) is the fallback when geometry is unavailable.

### 4.6 Anthropogenic heat flux QF (proxies)
No global per-cell QF exists at city scale → build a proxy:
`QF_proxy ≈ a·(VIIRS NTL) + b·(GHS-POP density) + c·(GHS-BUILT-V)` (calibrate a,b,c regionally; LUCY/Dong et al. coefficients as priors).
- **Sources:** VIIRS VNP46A2 (activity/energy), GHS-POP (metabolic + residential), GHS-BUILT-V (cooling demand/thermal mass).
- **Role:** the explicit human-heat source term in the canopy energy balance, especially important in dense Indian commercial cores at night.

> **One-line physics summary.** *SVF/H/W govern radiative trapping (nighttime UHI); λP/λB govern heat storage and active surface; λF governs ventilation (z0, zd, heat removal); GHS-POP/VIIRS/GHS-BUILT-V govern anthropogenic heat. Together these make the LST model physics-informed rather than purely statistical.*

---

## 5. Local Climate Zones (LCZ) — the 17-class thermal table

LCZ (Stewart & Oke 2012, BAMS) classify the landscape into **17 standard classes**: **10 built types (1–10)** + **7 land-cover types (A–G)**, each with characteristic geometry and thermal behavior. The **global LCZ map** (`RUB/RUBCLIM/LCZ/global_lcz_map/latest`, **100 m**, Demuzere et al. 2022 ESSD) gives every Indian city a ready morphological+thermal class layer — and each class carries **default SVF, H/W, λP, z0** that can seed the §4 parameters where building data is missing.

**Stewart & Oke (2012) Table 3 — geometric/cover properties & thermal relevance.** (LCZ 1, 2 web-confirmed; LCZ 3–10 are canonical published values — **verify against original Table 3**.)

| LCZ | Name | SVF | Aspect ratio H/W | Bldg surface frac. % | Impervious % | Height of roughness elements (m) | Terrain roughness class | Thermal behavior (UHI relevance) |
|-----|------|-----|------------------|----------------------|--------------|----------------------------------|--------------------------|----------------------------------|
| **1** | Compact high-rise | **0.2–0.4** | **> 2** | 40–60 | 40–60 | **> 25** | 8 | **Strongest nocturnal UHI**: deep canyons trap longwave; large QF; little sky for cooling |
| **2** | Compact midrise | **0.3–0.6** | **0.75–2** | 40–70 | 30–50 | 10–25 | 6–7 | Strong UHI; dense storage + low ventilation |
| **3** | Compact low-rise | 0.2–0.6 | 0.75–1.5 | 40–70 | 20–50 | 3–10 | 6 | High UHI in dense low-rise (typical of older Indian city cores) |
| **4** | Open high-rise | 0.5–0.7 | 0.75–1.25 | 20–40 | 30–40 | > 25 | 7–8 | Moderate UHI; towers give roughness but more sky/ventilation |
| **5** | Open midrise | 0.5–0.8 | 0.3–0.75 | 20–40 | 30–50 | 10–25 | 5–6 | Moderate UHI; better cooling than compact |
| **6** | Open low-rise | 0.6–0.9 | 0.3–0.75 | 20–40 | 20–50 | 3–10 | 5–6 | Lower UHI; suburban; higher SVF → faster cooling |
| **7** | Lightweight low-rise | 0.2–0.5 | 1–2 | 60–90 | <20 | 2–4 | 4–5 | Dense informal settlements/slums; low albedo, low thermal mass → **hot days, can cool faster at night** |
| **8** | Large low-rise | > 0.7 | 0.1–0.3 | 30–50 | 40–50 | 3–10 | 5 | Warehouses/malls; vast hot impervious roofs → strong daytime SUHI |
| **9** | Sparsely built | > 0.8 | 0.1–0.25 | 10–20 | <20 | 3–10 | 5–6 | Weak UHI; mostly natural cover between buildings |
| **10** | Heavy industry | 0.6–0.9 | 0.2–0.5 | 20–30 | 20–40 | 5–15 | 5–6 | **Large QF (industrial heat)**; metal surfaces; localized hotspots |
| **A** | Dense trees | < 0.4 | — | <10 | <10 | 3–30 | 8 | **Cooling**: shade + evapotranspiration; UHI mitigation target |
| **B** | Scattered trees | 0.5–0.8 | — | <10 | <10 | 3–15 | 5–6 | Moderate cooling |
| **C** | Bush, scrub | 0.7–0.9 | — | <10 | <10 | <2 | 4–5 | Slight cooling |
| **D** | Low plants | > 0.9 | — | <10 | <10 | <1 | 3–4 | **Rural reference** for UHI baseline; grass/crops |
| **E** | Bare rock / paved | > 0.9 | — | <10 | > 90 | <0.25 | 1–2 | **Hot**: high storage, no ET — strong daytime SUHI |
| **F** | Bare soil / sand | > 0.9 | — | <10 | <10 | <0.25 | 1–2 | Variable; dry soil heats strongly by day |
| **G** | Water | > 0.9 | — | <10 | <10 | — | 1 | **Cool sink**; high heat capacity; moderates surroundings |

**How to use LCZ in PS-1.**
1. Pull `RUB/RUBCLIM/LCZ/global_lcz_map/latest` for the city → instant morphology stratification.
2. Where building footprints/heights are missing, **seed λP, SVF, H/W, z0 from the class defaults above** (table-based prior), then **refine with UT-GLOBUS / DSM-derived parameters** where available — this is the gap-fill bridge between coarse and fine.
3. For finer/custom maps, run the **LCZ Generator** (`lcz-generator.rub.de`) on the city ROI.
4. Use **LCZ D (low plants)** or rural pixels as the UHI reference for `ΔT = T(urban LCZ) − T(LCZ D)`.

> **Caveat:** UT-GLOBUS itself showed that *real* per-building UCPs beat LCZ table values by ~55% RMSE in WRF — so treat LCZ defaults as a **fallback/prior**, not the final answer, wherever building-level data exists.

---

## 6. Multi-source footprint & height fusion / gap-filling

No single source is complete over India. The fusion pipeline (server-side GEE + cloud-native parquet):

### 6.1 Footprint fusion (geometry)
**Priority union, deduplicated by spatial overlap:**
1. **OSM** (highest where present — human-verified, carries `building:levels`).
2. **Google Open Buildings v3** (`confidence ≥ 0.70`) — best AI layer for India.
3. **Microsoft Global Building Footprints** — fills Google gaps.
4. **GHS-BUILT-S 10 m** — raster backstop where *no* vector footprint exists (gives λP even without polygons).

**Shortcut:** the **VIDA Google+Microsoft+OSM** product already performs steps 1–3 (~2.58 B footprints, source-labeled). Start from VIDA, then overlay GHS-BUILT-S to confirm/fill. Deduplicate by IoU > 0.5 between sources; keep the higher-priority geometry.

### 6.2 Height fusion (the 3rd dimension)
Assign a height to every footprint by a **fallback cascade** (use the first available, record provenance + uncertainty):
1. **UT-GLOBUS** per-building `height` (best where the city is covered).
2. **OSM** `height` or `building:levels × 3.0 m` (where tagged).
3. **Open Buildings 2.5D Temporal** zonal-mean height over the polygon (India-wide, 4 m, annual).
4. **DSM − DEM** = **Copernicus GLO-30 (DSM) − FABDEM (bare earth)**, zonal-max/mean per footprint (independent physical estimate everywhere).
5. **GHS-BUILT-H** (100 m ANBH) or **WSF3D** (90 m) cell mean — coarse backstop.
6. **LCZ class default** height (last resort, §5 table).

### 6.3 DSM−DEM building-height recipe (GEE)
```
DSM   = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic()
DEM   = ee.ImageCollection('projects/sat-io/open-datasets/FABDEM').mosaic()
objH  = DSM.subtract(DEM).max(0)                  // object height (bldg+veg)
bldgH = objH.updateMask(builtMask)                // mask to GHS-BUILT-S / footprints to drop trees
// then reduceRegions over footprint FeatureCollection -> per-building height
```
Mask out vegetation using GHS-BUILT-S / ESA WorldCover so `objH` over built pixels ≈ building height. Note GLO-30 (30 m) blurs narrow buildings — best for blocks, not single villas.

### 6.4 Reconciling UT-GLOBUS with DSM-derived heights
- **Agreement check:** regress UT-GLOBUS height vs (GLO-30 − FABDEM) per building. Expect scatter ~ the UT-GLOBUS RMSE (~9 m).
- **Trust rules:** for **aggregates/UCPs (λP, λF, hₐ at block/1 km)** → prefer **UT-GLOBUS** (validated at these scales, RMSE 7.8 m gridded). For an **individual tall landmark** → cross-check against the DSM-max; if DSM and UT-GLOBUS disagree by > 2×RMSE, flag and prefer DSM-max for that footprint.
- **Bias correction:** if a local LiDAR tile or OSM-tagged heights exist for the city, fit a linear correction `H_corr = α·H_source + β` per source and apply citywide.

### 6.5 Uncertainty propagation
Carry a **per-footprint height uncertainty σ_H** by source: UT-GLOBUS ≈ 9 m; Open Buildings 2.5D ≈ provider RMSE; DSM−DEM ≈ √(σ_DSM² + σ_FABDEM²) (FABDEM built MAE ≈ 1.12 m + GLO-30 ~2–4 m); GHS-BUILT-H ≈ 100 m-smoothing bias; LCZ default = class range width. Propagate into SVF/H/W/λF (Monte-Carlo or first-order) so the final LST/UHI prediction carries a credible error band — required for a *robust* PS-1 submission.

---

## 7. Cross-validation role (how this domain checks the rest of the system)

- **Footprint agreement (3-way):** OSM ∩ Google ∩ Microsoft IoU per cell → a **completeness/confidence raster**; low agreement flags where downstream LST drivers are weakly constrained.
- **Height triangulation:** UT-GLOBUS vs Open Buildings 2.5D vs (GLO-30 − FABDEM) vs GHS-BUILT-H — four independent estimates; their spread *is* the height uncertainty map.
- **Built-up consensus:** GHS-BUILT-S (10 m) vs WSF2019 (10 m) vs rasterized footprints → robust impervious/λP mask; disagreement marks edge/peri-urban zones.
- **LCZ vs measured morphology:** compare derived λP/SVF/H/W against the LCZ-class defaults; large deviations identify mis-classified or rapidly-changing neighborhoods.
- **Morphology vs thermal (physics closure):** the ultimate cross-check — bin LST by SVF/H/W/λP/λF and confirm the *expected* signs (nighttime LST ↑ as SVF ↓ and H/W ↑; daytime SUHI ↑ as λP and impervious ↑). If the data violate the physics, suspect a data error, not new physics.
- **DEM/DSM sanity:** SRTM vs NASADEM vs GLO-30 over flat bare ground should agree to a few metres; FABDEM should sit at/below GLO-30 in built/forest areas.

---

## 8. Recommended stack for Indian cities (build-ready)

**Tier 1 — must-use (server-side GEE, free, India-covering):**
- Built-up/λP: **GHS-BUILT-S 10 m** (`JRC/GHSL/P2023A/GHS_BUILT_S_10m`) + **WSF2019**.
- Footprints: **Google Open Buildings v3** (`GOOGLE/Research/open-buildings/v3/polygons`, conf ≥ 0.70) → augment with **VIDA** (Google+MS+OSM) for completeness.
- Heights: **UT-GLOBUS** (`projects/sat-io/open-datasets/UT-GLOBUS/<city>`) where covered; else **Open Buildings 2.5D Temporal** (`GOOGLE/Research/open-buildings-temporal/v1`) + **GLO-30 − FABDEM**.
- Per-cell height/volume backstop: **GHS-BUILT-H** (`JRC/GHSL/P2023A/GHS_BUILT_H`) + **GHS-BUILT-V**.
- Terrain: **Copernicus GLO-30** (`COPERNICUS/DEM/GLO30`, DSM) + **FABDEM** (bare earth) + **NASADEM** (slope/aspect).
- Climate class: **Global LCZ map** (`RUB/RUBCLIM/LCZ/global_lcz_map/latest`).
- QF proxies: **GHS-POP** (`JRC/GHSL/P2023A/GHS_POP`) + **VIIRS Black Marble** (`NASA/VIIRS/002/VNP46A2`).
- Urban extent: **GUB** or **GHS-SMOD** for masking.

**Tier 2 — refine/validate:**
- **WSF3D** (90 m height/volume/fraction) as an independent height/λP check.
- **OSM** (`building:levels`) for tagged Indian neighborhoods; **LCZ Generator** for custom city LCZ.
- **EUBUCCO** to validate the fusion/UCP code on clean European data before India.
- **Local municipal LiDAR/DSM** (if obtainable for the target city) for final bias correction.

**Derived layers to produce (the deliverable features):** λP, λF, SVF, H/W, λB, hₐ, z0, zd, QF_proxy — each as a 30–100 m raster with a companion uncertainty raster, feeding the physics-informed LST/UHI model and the cooling-intervention (°C-reduction) optimizer.

**Workflow note:** keep everything as **GEE server-side** reducers (`reduceRegions`, `reduceResolution`, hillshade-sweep for SVF) for O(1)-style scalability; export only the final morphology feature stack. For footprint-heavy ops, prefer the **VIDA GeoParquet** in a cloud-native engine (DuckDB/Spark) and bring summarized rasters back into GEE.

---

## 9. References (with URLs)

**UT-GLOBUS (flagship 3D)**
- Nature Scientific Data 11:886 (2024): https://www.nature.com/articles/s41597-024-03719-w — DOI 10.1038/s41597-024-03719-w
- Open access (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC11327349/
- Preprint: https://arxiv.org/abs/2205.12224 ; HTML https://arxiv.org/html/2205.12224
- Data (Zenodo): https://doi.org/10.5281/zenodo.11156602 ; record https://zenodo.org/records/11156602
- GEE community catalog: https://gee-community-catalog.org/projects/utglobus/ — asset `projects/sat-io/open-datasets/UT-GLOBUS/<city>`; app https://sat-io.earthengine.app/view/ut-globus

**GHSL (JRC)**
- GHS-BUILT-S 10 m: https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_S_10m
- GHS-BUILT-S 100 m: https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_S
- GHS-BUILT-H: https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_H
- GHS-BUILT-V: https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_V
- GHS-BUILT-C: https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_C
- GHS-SMOD: https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_SMOD_V2-0
- JRC GHSL hub: https://human-settlement.emergency.copernicus.eu/datasets.php ; community: https://gee-community-catalog.org/projects/ghsl/

**World Settlement Footprint (DLR)**
- WSF3D dataset: https://geoservice.dlr.de/web/datasets/wsf_3d ; map https://geoservice.dlr.de/web/maps/eoc:wsf3d
- WSF2019: https://download.geoservice.dlr.de/WSF2019/ ; map https://geoservice.dlr.de/web/maps/eoc:wsf2019

**Building footprints**
- Open Buildings v3: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings_v3_polygons ; project https://sites.research.google/gr/open-buildings/
- Open Buildings 2.5D Temporal: https://sites.research.google/gr/open-buildings/temporal/ ; GEE tag https://developers.google.com/earth-engine/datasets/tags/open-buildings ; height tutorial https://spatialthoughts.com/2025/03/29/building_height_gee/
- VIDA Google+Microsoft(+OSM): https://source.coop/vida/google-microsoft-open-buildings ; https://source.coop/vida/google-microsoft-osm-open-buildings ; community https://gee-community-catalog.org/projects/global_buildings/
- Microsoft Global ML Building Footprints (via VIDA/source.coop, as above).
- OSM India: Geofabrik https://download.geofabrik.de/asia/india.html ; Overpass https://www.geofabrik.de/data/overpass-api.html ; OSM completeness study https://pmc.ncbi.nlm.nih.gov/articles/PMC10326063/

**EUBUCCO**
- Paper (Sci Data): https://www.nature.com/articles/s41597-023-02040-2 ; PMC https://pmc.ncbi.nlm.nih.gov/articles/PMC10027854/ ; data https://zenodo.org/records/7225259

**Terrain / DEM**
- Copernicus GLO-30: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_DEM_GLO30 ; community https://gee-community-catalog.org/projects/glo30/
- SRTM GL1 v3: https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003
- NASADEM: https://developers.google.com/earth-engine/datasets/catalog/NASA_NASADEM_HGT_001
- ALOS AW3D30 v4.1: https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_AW3D30_V4_1 ; JAXA https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/
- FABDEM: paper (ERL) https://iopscience.iop.org/article/10.1088/1748-9326/ac4d4f ; data https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn ; community (`projects/sat-io/open-datasets/FABDEM`) https://gee-community-catalog.org/projects/fabdem/

**LCZ**
- Global LCZ map (GEE): https://developers.google.com/earth-engine/datasets/catalog/RUB_RUBCLIM_LCZ_global_lcz_map_latest
- Demuzere et al. 2022 (ESSD): https://essd.copernicus.org/articles/14/3835/2022/ ; data https://zenodo.org/records/6364594
- Stewart & Oke 2012 (BAMS, LCZ + Table 3 parameters): https://journals.ametsoc.org/view/journals/bams/93/12/bams-d-11-00019.1.xml
- WUDAPT / LCZ framework: https://www.wudapt.org/lcz/ ; LCZ Generator https://lcz-generator.rub.de/

**Anthropogenic heat / morphometric methods**
- VIIRS Black Marble VNP46A2: https://developers.google.com/earth-engine/datasets/catalog/NASA_VIIRS_002_VNP46A2 ; product https://blackmarble.gsfc.nasa.gov/VNP46A2.html
- Global Urban Boundary (GAIA): https://iopscience.iop.org/article/10.1088/1748-9326/ab9be3
- Macdonald et al. 1998 (z0/zd morphometric), Oke 1981/1987 (SVF–UHI), Kanda et al. 2013, Grimmond & Oke 1999 — standard urban-canopy references **(from-knowledge — verify editions)**.

---

*Prepared by Research Agent R4 (urban morphology / 3D form / terrain). All GEE IDs and resolutions verified against the live GEE Data Catalog and provider pages, June 2026. Stewart & Oke Table 3 values: LCZ 1–2 web-confirmed, LCZ 3–10 canonical-from-knowledge (verify). UT-GLOBUS investigated in depth per task; confirm the exact Indian-city list via `coverage_asia.gpkg` on Zenodo before build.*
