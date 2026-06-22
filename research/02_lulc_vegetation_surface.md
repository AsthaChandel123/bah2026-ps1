# R2 — LULC, Vegetation & Surface Biophysical Properties Catalog (GEE Build-Ready)

**Project:** ISRO Bharatiya Antariksh Hackathon 2026 — PS-1: Physics-informed geospatial AI/ML to map urban heat hotspots, quantify heating drivers, model LST vs. drivers, and optimize cooling interventions (°C reduction).

**This document (R2 domain):** Land Use/Land Cover (LULC), vegetation indices/products, and **surface radiative/biophysical properties** that physically drive Land Surface Temperature (LST). Every entry is a Google Earth Engine (GEE) asset with exact collection ID, resolution, revisit, the **physical driver** it represents, and **how it links to LST**. Closes with a multi-product **fusion/gap-filling** strategy and a **cross-validation** role matrix — the team's cross-verification theme.

**Verification status legend:**
- ✅ **verified-2026** — confirmed against GEE Data Catalog / authoritative source via web during this research (June 2026).
- 🟡 **from-knowledge (verify)** — expert knowledge, not re-confirmed live; flagged for a quick catalog check before build.

**Compute philosophy (shared team principle):** All assets below are server-side in GEE. Prefer (a) `ImageCollection.filterDate().filterBounds()` then a single `.reduce()`/`.median()`/`.mosaic()` to collapse time → **O(1)-feeling** server-side reductions; (b) precompute static surface-property stacks as `ee.Image` and `clip` per AOI; (c) use `reduceRegion`/`reduceRegions` with `bestEffort:true` + tileScale for hotspot extraction. Avoid client-side loops.

---

## 1. Overview & how this fits the physics

LST at a pixel is set by the **surface energy balance (SEB)**:

```
Rn = H + LE + G                                   (net radiation partitioning)
Rn = (1 − α)·S↓ + ε·L↓ − ε·σ·T_s^4                (net radiation)
```

where `α` = broadband albedo, `S↓` = incoming shortwave, `ε` = broadband emissivity, `L↓` = incoming longwave, `σ` = Stefan–Boltzmann, `T_s` = surface temperature (LST), `H` = sensible heat, `LE` = latent heat (evapotranspiration), `G` = ground/storage heat flux. Urban heat is fundamentally a **redistribution of Rn away from LE (vegetation/water/ET) and into H and G (impervious mass)**.

Each surface property in this catalog maps onto a term in that balance:

| SEB term | Controlled by surface property | Catalog products (R2) |
|---|---|---|
| `(1−α)·S↓` net shortwave gain | **Albedo (α)** | MCD43A3, Sentinel-2/Landsat-derived albedo |
| `ε·σ·T_s^4` longwave loss + emission | **Emissivity (ε)** | ASTER GED, MOD11 bands 31/32, NDVI-threshold ε |
| `LE` latent-heat (cooling) | **Vegetation fraction, ET, soil moisture** | NDVI/EVI, LAI/FPAR, FVC, MOD16/PML/OpenET/ECOSTRESS, SMAP |
| `G` storage / thermal admittance | **Imperviousness, built fraction, material** | GHSL BUILT, GISA/GAIA imperviousness, NDBI, LULC built class |
| roughness → `H` efficiency | **Surface roughness (from canopy/built height)** | derived from GHSL height, tree canopy, LAI |

The system needs **many, cross-verifying products per term** so gaps in one (cloud, deprecation, coarse resolution, classification error) are filled by others. That is the design rationale for the breadth below.

---

## 2. Big comparison table — all R2 GEE assets

> Resolutions are native; "revisit" is effective observation cadence. IDs are exact GEE paths. ES = Earth System.

### 2A. LULC products

| # | Product | GEE Collection ID | Native res | Temporal coverage / cadence | Classes | Primary LST driver use | Status |
|---|---|---|---|---|---|---|---|
| L1 | **ESA WorldCover v200 (2021)** | `ESA/WorldCover/v200` | 10 m | 2021 (single epoch) | 11 | Built/veg/water masks; FVC priors | ✅ |
| L2 | **ESA WorldCover v100 (2020)** | `ESA/WorldCover/v100` | 10 m | 2020 (single epoch) | 11 | Change baseline vs v200 | ✅ |
| L3 | **Google Dynamic World V1 (NRT)** | `GOOGLE/DYNAMICWORLD/V1` | 10 m | 2015-06-27 → present, 2–5 day | 9 (+ per-class prob.) | NRT built/veg; sub-pixel probability for fractions | ✅ |
| L4 | **ESRI/Impact Observatory Annual LULC** | `projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS` | 10 m | 2017 → 2024 (annual) | 9 | Multi-year built expansion, agreement voting | ✅ |
| L5 | **MODIS MCD12Q1 Land Cover** | `MODIS/061/MCD12Q1` | 500 m | 2001 → present, yearly | 17 IGBP (+ UMD/LAI/PFT) | Long climatology, urban (class 13) | ✅ |
| L6 | **Copernicus Global Land Cover (CGLS-LC100 C3)** | `COPERNICUS/Landcover/100m/Proba-V-C3/Global` | 100 m | 2015 → 2019, yearly | 23 (+ cover fractions) | Cover-fraction layers (tree/grass/built %) | ✅ |
| L7 | **C3S Land Cover (ESA CCI successor)** | `projects/sat-io/open-datasets/ESA/cci-lc` 🟡 / native `ESA CCI 1992–2015` legacy | 300 m | 1992 → 2022, yearly | 22 (LCCS) | Multi-decadal urbanization trajectory | 🟡 |
| L8 | **GHSL Built-up Surface (multitemporal)** | `JRC/GHSL/P2023A/GHS_BUILT_S` | 100 m | 1975 → 2030, 5-yr | built m²/cell | Built **fraction** → thermal admittance | ✅ |
| L9 | **GHSL Built-up Surface 10 m (2018)** | `JRC/GHSL/P2023A/GHS_BUILT_S_10m` | 10 m | 2018 | built m²/cell | High-res impervious fraction | ✅ |
| L10 | **GHSL Settlement Model (SMOD)** | `JRC/GHSL/P2023A/GHS_SMOD_V2-0` 🟡 | 1 km | 1975 → 2030, 5-yr | Degree-of-Urbanisation | Urban/rural/city stratification of analysis | 🟡 |
| L11 | **GHSL Built-up Volume** | `JRC/GHSL/P2023A/GHS_BUILT_V` | 100 m | 1975 → 2030 | m³/cell | Heat-storage mass proxy (G flux) | ✅ |
| L12 | **GHSL Settlement Characteristics (built type)** | `JRC/GHSL/P2023A/GHS_BUILT_C` | 10 m | 2018 | res/non-res morph. | Material/morphology proxy | ✅ |
| L13 | **Tsinghua FROM-GLC GAIA (impervious year)** | `Tsinghua/FROM-GLC/GAIA/v10` | 30 m | 1985 → 2018 (year-of-change) | impervious year | Impervious extent + age | ✅ |
| L14 | **GISA global impervious (year-of-first)** | `projects/sat-io/open-datasets/GISA_1972_2021` | 30 m | 1972 → 2021 (year-of-first) | impervious year | Independent impervious cross-check | ✅ |
| L15 | **Bhuvan / NRSC India LULC (50K, 250K)** | *Not native in GEE — ingest from NRSC Bhuvan* 🟡 | 56 m (AWiFS 250K) / 23.5 m (LISS-III 50K) | 250K: annual since 2004-05; 50K: 2005-06/2011-12/2015-16 | ~19–54 classes | India-authoritative cross-check, crop-cycle context | 🟡 |

### 2B. Vegetation indices & products

| # | Product | GEE Collection ID | Native res | Cadence | Driver represented | Status |
|---|---|---|---|---|---|---|
| V1 | **Sentinel-2 SR Harmonized** (compute NDVI/EVI/NDWI/NDBI…) | `COPERNICUS/S2_SR_HARMONIZED` | 10–20 m | ~5 day (2-sat) | All optical indices, fine urban texture | ✅ |
| V2 | **Sentinel-2 Cloud Probability** (masking) | `COPERNICUS/S2_CLOUD_PROBABILITY` | 10 m | per-scene | Cloud mask for V1 | ✅ |
| V3 | **Landsat 8/9 L2 SR** (compute indices + LST) | `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2` | 30 m (TIR 100 m) | 8 day combined | Indices + ST_B10 LST (shared w/ R1) | ✅ |
| V4 | **Landsat 4–7 L2 SR** (historical) | `LANDSAT/LT05/C02/T1_L2`, `LANDSAT/LE07/C02/T1_L2` | 30 m | 16 day | Historical NDVI/LST baselines | ✅ |
| V5 | **MODIS MOD13Q1 (Terra VI)** | `MODIS/061/MOD13Q1` | 250 m | 16 day | NDVI/EVI climatology, gap-fill | ✅ |
| V6 | **MODIS MYD13Q1 (Aqua VI)** | `MODIS/061/MYD13Q1` | 250 m | 16 day (offset) | Doubles VI cadence with MOD13Q1 | ✅ |
| V7 | **MODIS MOD13A1 (500 m VI)** | `MODIS/061/MOD13A1` | 500 m | 16 day | Coarse VI for fusion w/ albedo grid | ✅ |
| V8 | **MODIS MOD15A2H LAI/FPAR (Terra)** | `MODIS/061/MOD15A2H` | 500 m | 8 day | LAI → canopy transpiration / shading | ✅ |
| V9 | **MODIS MCD15A3H LAI/FPAR (combined)** | `MODIS/061/MCD15A3H` | 500 m | 4 day | Higher-cadence LAI/FPAR | ✅ |
| V10 | **Hansen Global Forest Change 2024 v1.12** | `UMD/hansen/global_forest_change_2024_v1_12` | 30 m | 2000 → 2024 (annual loss) | Tree canopy %2000, loss/gain → cooling loss | ✅ |
| V11 | **MODIS MOD44B VCF (tree/non-tree %)** | `MODIS/061/MOD44B` | 250 m | yearly | Continuous tree-cover fraction | ✅ |
| V12 | **MODIS MOD16A2 ET (Terra)** | `MODIS/061/MOD16A2` | 500 m | 8 day | Latent-heat cooling (LE) | ✅ |
| V13 | **MODIS MOD16A2GF ET (gap-filled)** | `MODIS/061/MOD16A2GF` | 500 m | 8 day | Continuous ET (pre-2021 + gap-filled) | ✅ |
| V14 | **PML_V2.2a ET + GPP (coupled)** | `projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a` | 500 m | 8 day | ET partitioned (Ec/Es/Ei) → cooling source | ✅ |
| V15 | **OpenET Ensemble (field-scale ET)** | `OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0` (CONUS only) | 30 m | monthly | High-res ET benchmark (method transfer) | ✅ |
| V16 | **ECOSTRESS L2T LST&E V2** | `NASA/ECOSTRESS/L2T_LSTE/...` 🟡 (ingest/LP DAAC) | 70 m | sub-daily, variable | Fine ET/LST at building scale | 🟡 |

### 2C. Surface radiative & physical properties (physics-critical)

| # | Product | GEE Collection ID | Native res | Cadence | SEB term | Status |
|---|---|---|---|---|---|---|
| S1 | **MODIS MCD43A3 Albedo (BSA/WSA)** | `MODIS/061/MCD43A3` | 500 m | daily (16-day window) | net shortwave `(1−α)S↓` | ✅ |
| S2 | **MODIS MCD43A4 NBAR** (compute albedo/indices) | `MODIS/061/MCD43A4` | 500 m | daily | BRDF-corrected reflectance → albedo | ✅ |
| S3 | **Sentinel-2 derived broadband albedo** (Bonafoni/Liang) | from `COPERNICUS/S2_SR_HARMONIZED` (formula §5) | 10–20 m | ~5 day | high-res α for interventions | ✅ |
| S4 | **Landsat derived albedo** (Liang/Smith) | from `LANDSAT/LC08/C02/T1_L2` (formula §5) | 30 m | 8–16 day | α with thermal co-registration | ✅ |
| S5 | **ASTER GED emissivity AG100 v3** | `NASA/ASTER_GED/AG100_003` | 100 m | static (2000–2008) | longwave `ε`; 5 TIR bands + mean LST | ✅ |
| S6 | **MODIS MOD11A1 LST & emissivity (daily)** | `MODIS/061/MOD11A1` | 1 km | daily (Terra 10:30/22:30) | LST + ε bands 31/32 | ✅ |
| S7 | **MODIS MOD11A2 LST & emissivity (8-day)** | `MODIS/061/MOD11A2` | 1 km | 8 day | ε (bands 31/32) + LST climatology | ✅ |
| S8 | **MYD11A1 / MYD11A2 (Aqua LST/ε)** | `MODIS/061/MYD11A1`, `MODIS/061/MYD11A2` | 1 km | daily / 8-day (13:30 overpass) | Afternoon-peak LST + ε | ✅ |
| S9 | **GISA / GAIA imperviousness** | (see L13, L14) | 30 m | annual/epoch | storage `G`, thermal admittance | ✅ |
| S10 | **SMAP L4 soil moisture (surface + root-zone)** | `NASA/SMAP/SPL4SMGP/008` | 9 km (≈11 km posting) | 3-hourly | water availability → LE cap, dry-soil heating | ✅ |
| S11 | **SMAP/Sentinel-1 L2 9 km/3 km** (if needed) | `NASA/SMAP/SPL2SMAP_S/...` 🟡 | 1–3 km | ~12 day | finer soil moisture | 🟡 |
| S12 | **Surface roughness (z0)** — derived | from GHSL `GHS_BUILT_V`/height + LAI (formula §5) | 100 m | derived | aerodynamic resistance → `H` | 🟡 |
| S13 | **GLO-30 DEM (Copernicus)** — context | `COPERNICUS/DEM/GLO30` | 30 m | static | elevation/slope/aspect → LST modifier | ✅ |

---

## 3. Per-product details (driver + physical LST linkage)

### 3.1 LULC products

**L1/L2 — ESA WorldCover 10 m (`ESA/WorldCover/v200`, `ESA/WorldCover/v100`)** ✅
Sentinel-1 + Sentinel-2 derived, 11 classes (tree, shrub, grass, crop, built-up=50, bare/sparse, snow/ice, permanent water, herbaceous wetland, mangrove, moss/lichen), CC-BY-4.0. v100=2020, v200=2021. Single `ImageCollection` with one image — use `.first()`. **Physical role:** the cleanest 10 m **built-up mask** and **vegetation/water mask** for the AOI; built-up (50) marks high-`G`/low-`α`/low-`LE` surfaces; tree/grass mark evaporative-cooling sources. Used as a **fractional-cover prior** (aggregate 10 m classes to the 30/100 m grid → impervious/veg fraction). Two epochs give a built-up **change** layer.

**L3 — Google Dynamic World V1 (`GOOGLE/DYNAMICWORLD/V1`)** ✅
Per-Sentinel-2-L1C-scene, 9 classes, **plus a continuous per-class probability band for each class** (`water, trees, grass, flooded_vegetation, crops, shrub_and_scrub, built, bare, snow_and_ice`) + `label`. 2015-06-27→present, 2–5 day revisit. **Physical role:** the **near-real-time** LULC and, crucially, the probability bands act as **sub-pixel fractional cover** — e.g. mean(`built` prob) over a season ≈ impervious fraction; mean(`trees`+`grass`+`shrub`) ≈ green fraction. Temporal compositing (median of probabilities over the LST analysis window) yields a stable, gap-filled fractional surface that aligns with the same dates as your Landsat/Sentinel LST. This is the single most useful LULC product for **continuous driver fractions**.

**L4 — ESRI / Impact Observatory Annual LULC (`projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS`)** ✅
Sentinel-2 derived, 9 classes, **annual 2017–2024**, deep-learned (Nat-Geo labels), ~75%+ accuracy. **Physical role:** independent 10 m annual built/veg map for **multi-product majority voting** (with WorldCover + Dynamic World) and for **time-consistent built-up expansion** trajectories that explain LST trend hotspots.

**L5 — MODIS MCD12Q1 (`MODIS/061/MCD12Q1`)** ✅
17 IGBP classes (urban/built = 13), 500 m, yearly 2001→present; also UMD, LAI, PFT schemes (`LC_Type1..5`). **Physical role:** long, consistent **climatological** LULC to define stable urban core vs. fringe, and to provide PFT/biome priors for emissivity and roughness lookups.

**L6 — Copernicus Global Land Cover 100 m (`COPERNICUS/Landcover/100m/Proba-V-C3/Global`)** ✅
23 LCCS classes **plus continuous cover-fraction bands** (tree/shrub/grass/crop/built/bare/water %, 0–100). 2015–2019 yearly. **Physical role:** ready-made **fractional cover** at 100 m — directly usable as driver fractions co-registered to GHSL 100 m and MODIS 500 m grids without re-deriving from class labels.

**L7 — C3S / ESA CCI Land Cover (300 m)** 🟡
22 LCCS classes, 1992→2022 yearly (C3S continuation of ESA CCI-LC). **Physical role:** multi-decadal **urbanization trajectory** for attributing long-term LST warming. Native ESA CCI legacy (1992–2015) and the C3S extension are typically accessed via the community catalog (`gee-community-catalog.org/projects/c3slc`); confirm exact asset path at build.

**L8–L12 — GHSL suite (JRC P2023A)** ✅ (SMOD 🟡)
- `GHS_BUILT_S` (100 m, built **surface m²/cell**, 1975–2030 5-yr) and `GHS_BUILT_S_10m` (10 m, 2018): convert to **built fraction** = built_m² / cell_area. This is the physically-grounded **impervious fraction** driving heat storage.
- `GHS_BUILT_V` (100 m, built **volume m³/cell**): proxy for **thermal mass / heat-storage capacity** (G flux, nocturnal UHI). High volume → large heat-release lag → warm nights.
- `GHS_BUILT_C` (10 m, 2018): residential vs. non-residential **morphology** → material/albedo priors.
- `GHS_SMOD` (1 km, Degree of Urbanisation): stratify the city into urban-centre / dense-urban / semi-dense / rural for **stratified hotspot statistics** and fair °C-reduction baselines. (Confirm exact ID `JRC/GHSL/P2023A/GHS_SMOD_V2-0`.)

**L13/L14 — GAIA & GISA imperviousness (`Tsinghua/FROM-GLC/GAIA/v10`, `projects/sat-io/open-datasets/GISA_1972_2021`)** ✅
30 m, value = **year of first impervious detection** (GAIA 1985–2018; GISA 1972–2021, year encoded 1972=1, 1978=2, 1985=3 …, no-data=0). **Physical role:** two **independent** impervious masks (different teams/methods) → cross-verify the built-up footprint and provide **impervious age** (older impervious ↔ denser, hotter cores). Threshold `>0` for a binary impervious mask; agreement of GAIA∩GISA∩GHSL gives a high-confidence impervious layer.

**L15 — Bhuvan / NRSC India LULC** 🟡 (**not native in GEE — must ingest**)
India-authoritative: LULC-250K (AWiFS/Resourcesat, ~56 m, **annual crop-year since 2004-05**, e.g. "2022–23 1:250,000") and LULC-50K (LISS-III, ~23.5 m, epochs 2005-06/2011-12/2015-16), hosted on Bhuvan thematic services. **Action for build:** download from Bhuvan (NRSC) and **ingest as a GEE asset** (`projects/<your-cloud-project>/assets/bhuvan_lulc_...`). **Physical role:** ISRO/Indian-context **ground-truth-grade cross-check** for the global 10 m products (critical for hackathon credibility), plus rich agricultural/cropping-pattern classes that explain seasonal LST swings around Indian cities. *No confirmed public GEE mirror as of June 2026 — verify Bhuvan API/WMS terms.*

### 3.2 Vegetation indices & products

**V1–V4 — Sentinel-2 & Landsat surface reflectance** ✅
The optical engines. From `COPERNICUS/S2_SR_HARMONIZED` (10–20 m, ~5 day) and `LANDSAT/LC0{8,9}/C02/T1_L2` (30 m optical, 100 m thermal resampled to 30 m, 8-day combined) you compute the **entire spectral-index catalog** (§5). Landsat additionally carries the **thermal band → LST** (R1's domain) so indices and LST are **intrinsically co-registered** — essential for clean LST-vs-driver regressions. Mask S2 with `S2_CLOUD_PROBABILITY` / `s2cloudless`; mask Landsat with the `QA_PIXEL` bitmask.
**Physical role:** NDVI/EVI → vegetation amount → **latent-heat cooling + shading**; NDWI/MNDWI → water → strong evaporative cooling + high thermal inertia; NDBI/NDBaI/UI/IBI → built/bare → heat storage; all feed the driver stack at the finest resolution available.

**V5–V7 — MODIS Vegetation Indices (`MODIS/061/MOD13Q1`, `MYD13Q1`, `MOD13A1`)** ✅
Atmospherically corrected, BRDF-aware **NDVI & EVI** at 250 m/500 m, 16-day (Terra+Aqua interleaved → ~8-day effective). Built-in QA. **Physical role:** the **gap-filling backbone** for vegetation cooling — cloud-robust, long (2000→present), and a temporal-smoothness reference to in-fill Sentinel/Landsat NDVI gaps and to build NDVI **climatologies/anomalies** (heat hotspots often coincide with negative NDVI anomalies).

**V8/V9 — MODIS LAI/FPAR (`MODIS/061/MOD15A2H`, `MCD15A3H`)** ✅
LAI (one-sided green leaf area per ground area) and FPAR (fraction of absorbed PAR), 500 m, 8-day/4-day. **Physical role:** LAI is the **physically correct canopy-density variable** for transpiration and radiation interception — it drives the `LE` term far better than NDVI alone (NDVI saturates at high biomass). FPAR feeds light-use-efficiency ET/GPP models. Use LAI to estimate **canopy roughness** and shading for the SEB.

**V10/V11 — Tree canopy (`UMD/hansen/global_forest_change_2024_v1_12`, `MODIS/061/MOD44B`)** ✅
Hansen: `treecover2000` (% canopy >5 m within 30 m pixel), `loss`/`lossyear`/`gain`, 2000→2024. MOD44B VCF: continuous % tree / non-tree-veg / non-veg, 250 m yearly. **Physical role:** **tree canopy fraction** is the highest-leverage cooling intervention variable (shade + transpiration). `lossyear` pinpoints **where canopy loss preceded LST increase** — direct driver attribution. VCF gives a continuous, independent canopy fraction for cross-check.

**V12–V14 — Evapotranspiration (`MODIS/061/MOD16A2`, `MOD16A2GF`, PML_V2.2a)** ✅
- **MOD16A2 / MOD16A2GF** (500 m, 8-day): Penman–Monteith ET, PET, latent-heat flux (`ET`, `LE`, `PET`, `PLE`). GF = year-end gap-filled (recommended pre-2021 & for continuity).
- **PML_V2.2a** (`projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a`, 500 m, 8-day, 2000-03→2024-12, scale 0.01): **coupled ET + GPP** with ET **partitioned** into `Ec` (transpiration), `Es` (soil evaporation), `Ei` (interception), `Ew` (water/ice), plus `PET`. **Physical role:** ET **is the `LE` cooling term made explicit** — high ET = strong evaporative cooling = lower LST. Partitioned ET (PML) tells you *how* a surface cools (canopy transpiration vs. soil/water evaporation), which determines which intervention (trees vs. water vs. permeable soil) yields the most °C reduction.

**V15 — OpenET Ensemble (`OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0`)** ✅ *(CONUS-only)*
30 m field-scale ET, ensemble mean of 6 models (ALEXI/DisALEXI, eeMETRIC, PT-JPL, geeSEBAL, SIMS, SSEBop). **Not over India**, but invaluable as a **methodological template**: the same SEBAL/PT-JPL logic (geeSEBAL is open GEE code) can be **ported to Indian Landsat/ECOSTRESS** to get 30–70 m ET where MODIS is too coarse for intra-urban hotspots.

**V16 — ECOSTRESS LST&E / ET (V2, `NASA/ECOSTRESS/L2T_LSTE` family)** 🟡
70 m TIR from ISS, sub-daily at **varying times of day** (captures diurnal LST, including afternoon peaks MODIS/Landsat miss). V1 products deprecated Sept 2025 → use **V2** (`ECO_L2T_LSTE`, `ECO_L3T_JET`). **Physical role:** **building-block-scale** LST and ET — the closest thing to intra-neighborhood thermal truth for validating modeled hotspot °C and for evaluating cooling interventions at park/street scale. Availability over a given Indian city is sparse/irregular; confirm GEE asset path (may require LP DAAC ingest).

### 3.3 Surface radiative & physical properties

**S1/S2 — MODIS Albedo (`MODIS/061/MCD43A3`, `MCD43A4`)** ✅
MCD43A3: BRDF-corrected **black-sky (BSA)** and **white-sky (WSA)** broadband albedo. Exact bands: `Albedo_BSA_vis/nir/shortwave`, `Albedo_WSA_vis/nir/shortwave` (+ per-band QA via MCD43A2). **Scale = 0.001**, valid 0–32766, fill 32767; 500 m, daily (16-day retrieval window), 2000-02-24→present. Actual (blue-sky) albedo ≈ skydiffuse-weighted blend of BSA & WSA (≈WSA under overcast, ≈BSA under clear). MCD43A4 = NBAR reflectance to derive custom albedo/indices. **Physical role:** α directly sets **absorbed shortwave `(1−α)S↓`** — the dominant daytime heating input. Low-α asphalt/roofs (α≈0.05–0.15) absorb far more than high-α surfaces; **raising α (cool roofs/pavements) is the most quantifiable °C-reduction lever**, and MCD43A3 is the physically rigorous albedo reference.

**S3/S4 — Sentinel-2 & Landsat derived broadband albedo** ✅
Computed from SR bands via narrow-to-broadband (NTB) coefficients (Liang 2001; **Bonafoni & Şekertekin 2020** new S2 coefficients — see §5). **Physical role:** α at **10–30 m** so individual roofs, roads, and parks get their own albedo — essential because interventions are applied at building/street scale, far below MODIS 500 m. Validate S2/Landsat albedo against MCD43A3 aggregated to 500 m (cross-check, §7).

**S5 — ASTER GED emissivity (`NASA/ASTER_GED/AG100_003`)** ✅
Mean **emissivity for 5 ASTER TIR bands** + std-dev + **mean LST** + NDVI, 100 m, static (clear-sky 2000–2008), TES algorithm. **Physical role:** ε governs both **longwave emission `ε·σ·T_s^4`** (how efficiently a surface sheds heat at night) and the **emitted-radiance → LST** conversion. Urban materials (concrete ε≈0.92, metal roofs ε≈0.85–0.90) differ from vegetation (ε≈0.98); using a spatially-varying ε instead of a constant removes a systematic LST bias of up to several K. ASTER GED is the **highest-resolution emissivity** available for the SEB.

**S6–S8 — MODIS LST & Emissivity (`MOD11A1/A2`, `MYD11A1/A2`)** ✅
1 km LST plus **emissivity bands 31 & 32** (`Emis_31`, `Emis_32`), daily/8-day, Terra (10:30/22:30) + Aqua (13:30/01:30). **Physical role:** (i) provides ε at the MODIS grid as an **independent emissivity cross-check** to ASTER GED; (ii) Terra+Aqua give **4 daily LST samples** → diurnal temperature range (DTR) and **nocturnal UHI** (driven by `G`/built volume); (iii) the LST climatology anchors hotspot definitions at city scale (R1 owns LST modeling; R2 supplies the ε needed to compute it correctly).

**S9 — Imperviousness (GISA/GAIA)** ✅ — see L13/L14. The physical bridge from LULC to the **`G` (storage) term**: impervious fraction ↑ → thermal admittance ↑ → daytime heat stored and **released at night** → elevated nocturnal LST. The single most important non-vegetation driver of urban heat.

**S10/S11 — SMAP soil moisture (`NASA/SMAP/SPL4SMGP/008`)** ✅
L4 assimilation: **surface (0–5 cm)** and **root-zone (0–100 cm)** soil moisture (`sm_surface`, `sm_rootzone`), 9 km, 3-hourly, 2015→present (also surface temp, ET, Rn research bands). **Physical role:** soil moisture **caps the achievable `LE`** — dry soil cannot evaporatively cool, so the same vegetation/LULC produces much higher LST when SMAP shows dry conditions. It explains *why* identical green cover yields different °C in pre-monsoon vs. monsoon India, and identifies where **irrigation/permeable-surface interventions** would actually lower LST. SPL2SMAP_S (SMAP/Sentinel-1) offers finer (1–3 km) where available.

**S12 — Surface roughness `z0`** 🟡 (derived)
Estimate aerodynamic roughness length from **building height** (GHSL `GHS_BUILT_V` volume ÷ footprint → mean height) and **canopy** (LAI/tree height), via rule-of-thumb `z0 ≈ 0.1·h` or Raupach/Macdonald morphometric methods. **Physical role:** roughness controls **aerodynamic resistance** and hence how efficiently `H` (sensible heat) is convected away. Smooth, dense urban canyons trap heat (low ventilation) → higher LST; this term matters for explaining why some equally-impervious areas are hotter. Derived, not a turnkey GEE asset — flag for build.

**S13 — Copernicus GLO-30 DEM (`COPERNICUS/DEM/GLO30`)** ✅
30 m global DEM. **Physical role:** elevation/slope/aspect modulate incoming shortwave and air temperature (lapse rate); used as a **covariate/normalizer** so topographic LST variation isn't mis-attributed to land cover (important for hill-adjacent Indian cities like Pune, Shimla, Guwahati).

---

## 4. Physical-linkage-to-LST table (driver → mechanism → sign)

| Surface property (product) | SEB term affected | Physical mechanism | Effect on LST | Intervention lever (°C) |
|---|---|---|---|---|
| **NDVI / EVI ↑** (V1,V3,V5) | `LE` ↑, shading | More photosynthetic canopy → transpiration + shade | **↓ LST** (cooling) | Urban greening, parks |
| **LAI / FPAR ↑** (V8,V9) | `LE` ↑, `Rn` interception | Denser canopy → more transpiration, less ground insolation | **↓ LST** | Dense tree planting |
| **Tree canopy % ↑** (V10,V11) | `LE` ↑, shading, `α` slight ↑ | Shade blocks shortwave to ground + transpiration | **↓↓ LST** (strongest veg lever) | Street trees, canopy targets |
| **ET ↑** (V12–V15) | `LE` ↑ | Direct latent-heat removal | **↓ LST** | Anything boosting ET (veg+water+moisture) |
| **NDWI / MNDWI ↑ (water)** (§5) | `LE` ↑, `G` ↑ (inertia) | Open water: evaporation + huge heat capacity | **↓↓ LST daytime** | Water bodies, fountains, wetlands |
| **Albedo α ↑** (S1,S3,S4) | `(1−α)S↓` ↓ | Less absorbed shortwave | **↓↓ LST** (most quantifiable) | Cool/white roofs & pavements |
| **Emissivity ε ↑** (S5,S6) | `ε·σT^4` ↑ emission | More efficient longwave cooling | **↓ LST** (esp. night) | High-ε coatings |
| **Imperviousness ↑ / Built fraction ↑** (L8,L13,L14) | `G` ↑, `LE` ↓ | Heat stored in mass, no evaporation | **↑↑ LST** (day store → night release) | De-paving, permeable surfaces |
| **Built volume / thermal mass ↑** (L11) | `G` ↑ | Large heat-storage & release lag | **↑ nocturnal LST** | Material choice, urban form |
| **NDBI / NDBaI / UI / IBI ↑** (§5) | `G` ↑, `α` ↓, `LE` ↓ | Built/bare signature | **↑ LST** | Identify retrofit targets |
| **Soil moisture ↑** (S10) | `LE` ceiling ↑ | Enables evaporative cooling | **↓ LST** (conditional on veg/soil) | Irrigation, permeable+green |
| **Roughness z0 ↑** (S12) | `H` ventilation ↑ | Turbulent removal of sensible heat | **↓ LST** (better ventilation) | Urban form / corridors |
| **Elevation / slope** (S13) | `S↓`, air T | Topographic insolation + lapse rate | modifier (±) | (covariate, not lever) |

---

## 5. Spectral-index & derivation formula table

Band notation: ρ = surface reflectance. Sentinel-2: B2=Blue(490), B3=Green(560), B4=Red(665), B5/B6/B7=RedEdge, B8=NIR(842), B8A=NIR-narrow(865), B11=SWIR1(1610), B12=SWIR2(2190). Landsat-8/9 OLI: B2 Blue, B3 Green, B4 Red, B5 NIR, B6 SWIR1, B7 SWIR2.

| Index / variable | Formula | What it measures | LST linkage |
|---|---|---|---|
| **NDVI** | (NIR − Red)/(NIR + Red) | Green vegetation amount | ↑NDVI → ↑LE/shade → ↓LST |
| **EVI** | 2.5·(NIR − Red)/(NIR + 6·Red − 7.5·Blue + 1) | Veg, de-saturated, soil/aerosol-corrected | better high-biomass cooling proxy |
| **SAVI** | (1.5)·(NIR − Red)/(NIR + Red + 0.5) | Veg with soil-brightness correction | sparse-veg cooling (arid Indian cities) |
| **FVC** (fractional veg cover) | ((NDVI − NDVI_soil)/(NDVI_veg − NDVI_soil))² | Sub-pixel green fraction | drives ε and LE partition |
| **NDWI (McFeeters)** | (Green − NIR)/(Green + NIR) | Open water | ↑ → evaporative cooling + inertia |
| **MNDWI** | (Green − SWIR1)/(Green + SWIR1) | Water (built-up-robust) | water-body cooling mask |
| **NDMI / NDWI(veg)** | (NIR − SWIR1)/(NIR + SWIR1) | Vegetation/canopy water content | moisture stress → ↓LE → ↑LST |
| **NDBI** | (SWIR1 − NIR)/(SWIR1 + NIR) | Built-up | ↑ → impervious → ↑LST |
| **NDBaI** | (SWIR1 − TIR)/(SWIR1 + TIR) | Bare soil | bare/dry heating |
| **UI (Urban Index)** | (SWIR2 − NIR)/(SWIR2 + NIR) | Urban built-up | impervious heating |
| **IBI (Index-Based Built-up)** | (NDBI − (SAVI + MNDWI)/2)/(NDBI + (SAVI + MNDWI)/2) | Built-up (veg+water suppressed) | cleaner built signal |
| **BU (Built-Up)** | NDBI − NDVI | Built minus veg | net heating tendency |
| **BSI (Bare Soil Index)** | ((SWIR1+Red) − (NIR+Blue))/((SWIR1+Red) + (NIR+Blue)) | Bare soil/impervious | exposed-surface heating |
| **Albedo — S2 (Bonafoni & Şekertekin 2020)** | α ≈ 0.2266·B2 + 0.1236·B3 + 0.1573·B4 + 0.3417·B8 + 0.1170·B11 + 0.0338·B12 (coeffs from paper; **verify exact set at build**) 🟡 | Broadband shortwave albedo @10–20 m | sets `(1−α)S↓` |
| **Albedo — Landsat (Liang 2001)** | α = 0.356·B2 + 0.130·B4 + 0.373·B5 + 0.085·B6 + 0.072·B7 − 0.0018 (OLI form of Liang SWB) | Broadband shortwave albedo @30 m | sets `(1−α)S↓` |
| **Blue-sky albedo (MODIS)** | α = (1 − S)·α_BSA + S·α_WSA, S = diffuse fraction | Actual albedo under real sky | physically-correct absorbed SW |
| **Emissivity from NDVI (NDVI-threshold, Sobrino)** | ε = ε_v·FVC + ε_s·(1 − FVC) + dε (ε_v≈0.985, ε_s≈0.96–0.97) | Broadband ε where ASTER absent | longwave term `ε·σT^4` |
| **Land Surface Temperature (mono-window/SC)** | T_s = BT/(1 + (λ·BT/ρ)·ln ε) (Landsat ST_B10 already provides LST) | Surface temperature | **target variable** |
| **Roughness length z0 (rule-of-thumb)** | z0 ≈ 0.1·h (h = mean obstacle height from GHSL/canopy) | Aerodynamic roughness | controls `H` ventilation |
| **UTFVI (Urban Thermal Field Variance Index)** | (LST − LST_mean)/LST_mean | Heat-island ecological evaluation | hotspot severity classification |

*(Index references: Tucker 1979 NDVI; Huete 2002 EVI; McFeeters 1996 NDWI; Xu 2006 MNDWI/IBI; Zha 2003 NDBI; Sobrino 2004/2008 NDVI-threshold emissivity & LST; Liang 2001 albedo; Bonafoni & Şekertekin 2020 S2 albedo.)*

---

## 6. Multi-product fusion & gap-filling strategy (cross-verification core)

**Goal:** build one robust, gap-free **surface-property stack** (an `ee.Image` with bands: `impervious_frac, green_frac, water_frac, tree_frac, ndvi, lai, et, albedo, emissivity, soil_moisture, built_volume, …`) co-registered to a chosen analysis grid (recommend **30 m Landsat grid** for intra-urban hotspots, with 100 m/500 m fallbacks), where **every band is cross-verified by ≥2 independent sources**. The rule: *no single source is trusted; agreement raises confidence, disagreement raises an uncertainty flag that triggers fallback.*

### 6.1 LULC fusion — majority voting + agreement + uncertainty

1. **Harmonize legends** of WorldCover(L1), Dynamic World(L3), ESRI(L4), MCD12Q1(L5) to a common minimal scheme: `{built, tree, grass/shrub, crop, water, bare, wetland}`.
2. **Reproject** all to the target grid (10 m products → keep 10 m, then `reduceResolution` to 30 m for fractions).
3. **Per-pixel majority vote** across the harmonized maps (`ee.Reducer.mode()` on a multi-band image of the relabeled products). Tie-break by Dynamic World probability (continuous).
4. **Agreement map** = count of products agreeing with the majority (0–4). Store as `lulc_agreement` band → directly an **uncertainty/confidence layer**.
5. **Fractional cover** (the physically useful product, not the hard class): from Dynamic World seasonal **mean probabilities** + Copernicus 100 m **cover-fraction** bands + aggregated 10 m WorldCover counts → ensemble-mean fraction per class. Disagreement (variance across the three) = fraction uncertainty.
6. **Impervious fraction** is special-cased (§6.2) because it most strongly drives heat.

### 6.2 Impervious / built fraction fusion

Combine **5 independent estimators**: GHSL `GHS_BUILT_S` fraction (L8/L9), GAIA(L13)>0, GISA(L14)>0, Dynamic World `built` mean prob (L3), WorldCover built==50 aggregated (L1).
- **Consensus impervious** = pixels where ≥3 of 5 indicate built → high-confidence mask.
- **Continuous impervious fraction** = mean of (GHSL fraction, DW built prob, NDBI-scaled) with GAIA/GISA as binary priors.
- **Uncertainty** = std-dev across estimators; high where products disagree (typical at the urban fringe and over bright bare soil — exactly where NDBI false-positives, so use the LULC vote to suppress).

### 6.3 Vegetation / NDVI gap-filling

- **Temporal fill:** for the LST analysis window, build a per-pixel NDVI from Sentinel-2(V1) median; where clouds leave gaps, fill from Landsat(V3) median, then from MODIS MOD13Q1/MYD13Q1(V5/V6) (resampled), preserving the MODIS **temporal climatology** shape.
- **Harmonic / Savitzky–Golay** smoothing on the MODIS series gives a gap-free NDVI climatology; compute **anomaly = observed − climatology** to flag stressed vegetation coincident with hotspots.
- **Cross-scale check:** S2-NDVI aggregated to 500 m vs. MODIS-NDVI should correlate r>0.9; large residuals flag BRDF/cloud contamination.

### 6.4 Albedo fusion

- **Reference:** MCD43A3 blue-sky albedo (S1) at 500 m (physically rigorous, BRDF-corrected).
- **High-res:** S2-derived (S3, 10–20 m) and Landsat-derived (S4, 30 m).
- **Fusion (downscaling):** fit a local linear bias-correction so that S2/Landsat albedo aggregated to 500 m matches MCD43A3, then keep the high-res spatial detail (a simple histogram/mean match per land-cover stratum). Result: 10–30 m albedo that is **anchored to the MODIS physical standard**. Disagreement after correction = albedo uncertainty.

### 6.5 Emissivity fusion

- **Primary:** ASTER GED (S5) 100 m broadband ε (convert 5 TIR bands → broadband via Ogawa/weighted combination).
- **Secondary:** MODIS MOD11/MYD11 (S6–S8) `Emis_31/32` at 1 km (independent).
- **Dynamic:** NDVI-threshold ε (§5) computed per-scene from current FVC — captures **seasonal/leaf-on-leaf-off** changes ASTER GED (static 2000–08) misses.
- **Fusion:** use ASTER GED as the spatial baseline, **adjust by ΔFVC** between the ASTER-era NDVI and the current NDVI (so newly-built or newly-green pixels get corrected ε). Cross-check against MODIS ε; flag where they differ >0.02.

### 6.6 ET fusion (latent-heat term)

- Stack MOD16A2GF(V13), PML_V2.2a(V14), and (where applicable) a ported SEBAL/PT-JPL Landsat ET (template from OpenET/geeSEBAL, V15) and ECOSTRESS(V16).
- **Ensemble ET** = median across available products (robust to single-model bias, mirroring OpenET's own median-absolute-deviation outlier removal).
- **Partition** from PML (Ec/Es/Ei) tells the intervention story; ensemble magnitude gives the cooling quantity.
- **Constraint:** ET is bounded by SMAP soil-moisture availability (S10) — flag pixels where a product reports high ET but SMAP shows very dry soil (physical inconsistency).

### 6.7 Output stack & uncertainty bands

For every physical band, emit **(value, n_sources, std/disagreement)**. The `*_uncertainty` and `*_agreement` bands feed (a) the LST-vs-driver model as **weights** (down-weight uncertain pixels) and (b) the final hotspot/intervention maps as **confidence overlays** — so recommendations carry honest error bars (important for a physics-informed, defensible hackathon submission).

---

## 7. Cross-validation role matrix (who checks whom)

| Quantity | Primary source | Independent cross-checkers | Disagreement → action |
|---|---|---|---|
| **Built-up extent** | WorldCover(L1) | Dynamic World(L3), ESRI(L4), GHSL(L8), GAIA(L13), GISA(L14) | majority vote; fringe → manual/flag |
| **Impervious fraction** | GHSL BUILT_S(L8) | DW built-prob(L3), GAIA/GISA(L13/14), NDBI(§5) | mean + std uncertainty band |
| **Vegetation fraction** | Dynamic World(L3) | Copernicus fractions(L6), WorldCover-agg(L1), NDVI(V1) | ensemble mean; variance = uncertainty |
| **NDVI / greenness** | Sentinel-2(V1) | Landsat(V3), MOD13Q1/MYD13Q1(V5/V6) | cross-scale r>0.9 else flag BRDF/cloud |
| **Tree canopy %** | Hansen treecover(V10) | MOD44B VCF(V11), WorldCover tree(L1) | reconcile; loss-year for change |
| **LAI / canopy density** | MOD15A2H(V8) | MCD15A3H(V9), NDVI→LAI empirical | temporal consistency check |
| **Albedo** | MCD43A3(S1) | S2-derived(S3), Landsat-derived(S4) | bias-correct hi-res to MODIS @500 m |
| **Emissivity** | ASTER GED(S5) | MOD11/MYD11 ε(S6–S8), NDVI-threshold ε | ΔFVC adjust; flag |Δε|>0.02 |
| **Evapotranspiration** | PML_V2.2a(V14) | MOD16A2GF(V13), Landsat-SEBAL(V15-template), ECOSTRESS(V16) | ensemble median; SMAP consistency |
| **Soil moisture** | SMAP L4(S10) | SMAP/S1(S11), ERA5-Land (team R-met) | assimilation vs. obs sanity |
| **LST (R1 owns)** | Landsat ST(V3) | MODIS MOD11/MYD11(S6–S8), ECOSTRESS(V16), ASTER GED mean LST(S5) | multi-sensor diurnal reconciliation |
| **India ground-truth** | Bhuvan/NRSC LULC(L15) | all global 10 m products | accuracy report per class (credibility) |

This matrix is the explicit ≥30-method cross-verification fabric for the R2 domain: **15 LULC/built methods + 16 vegetation/ET methods + 13 radiative/physical methods ≈ 44 independent products**, each cross-checked by ≥2 others.

---

## 8. Recommended stack for the Indian urban context

**Design constraints for India:** strong monsoon seasonality (leaf-on/leaf-off + soil-moisture swings dominate LST), heavy pre-monsoon cloud, dense informal/mixed urban morphology (10 m resolution essential), and the credibility need to reconcile with **ISRO/NRSC (Bhuvan)** products.

**Tier 1 — core build stack (use these):**
1. **Dynamic World V1** (`GOOGLE/DYNAMICWORLD/V1`) — NRT 10 m fractional cover (built/green/water) aligned to your LST dates. *Backbone.*
2. **ESA WorldCover v200 + v100** (`ESA/WorldCover/v200`,`v100`) — clean 10 m masks + change.
3. **ESRI Annual LULC** (`projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS`) — third 10 m vote + annual trend.
4. **GHSL BUILT_S 10 m + BUILT_S + BUILT_V** (`JRC/GHSL/P2023A/GHS_BUILT_S_10m`, `GHS_BUILT_S`, `GHS_BUILT_V`) — physical impervious fraction + thermal-mass.
5. **GISA + GAIA** (`projects/sat-io/open-datasets/GISA_1972_2021`, `Tsinghua/FROM-GLC/GAIA/v10`) — independent impervious + age.
6. **Sentinel-2 SR Harmonized** (`COPERNICUS/S2_SR_HARMONIZED`) + cloud-prob — all indices + S2 albedo @10 m.
7. **Landsat 8/9 L2** (`LANDSAT/LC08|LC09/C02/T1_L2`) — indices + albedo + co-registered LST (with R1).
8. **MCD43A3 albedo** (`MODIS/061/MCD43A3`) — physical albedo reference to anchor S2/Landsat α.
9. **ASTER GED** (`NASA/ASTER_GED/AG100_003`) + **MOD11/MYD11 ε** — emissivity (static + dynamic NDVI-ε).
10. **PML_V2.2a + MOD16A2GF** ET — latent-heat cooling, partitioned.
11. **SMAP L4** (`NASA/SMAP/SPL4SMGP/008`) — soil-moisture LE ceiling (monsoon-critical).
12. **MOD13Q1/MYD13Q1** + **MOD15A2H** — cloud-robust NDVI/LAI gap-fill backbone.
13. **Hansen GFC 2024 v1.12** + **MOD44B** — tree canopy fraction + loss (intervention target).
14. **Copernicus GLO-30 DEM** — topographic covariate (hill-adjacent cities).

**Tier 2 — credibility & fine-scale (ingest/port):**
15. **Bhuvan/NRSC LULC** (ingest as GEE asset) — ISRO-authoritative cross-validation & cropping context.
16. **ECOSTRESS V2 LST&E / ET** (where available) — 70 m diurnal validation of hotspots/interventions.
17. **Landsat SEBAL/PT-JPL ET** (port geeSEBAL/OpenET method) — 30 m ET where MODIS too coarse.

**Seasonality handling:** build **two stacks** (pre-monsoon: Mar–May, the hottest/driest = worst-case hotspots; and post-monsoon/winter for contrast), each with its own NDVI/LAI/ET/soil-moisture so the LST-vs-driver model learns the **moisture-dependent** cooling efficiency. Always carry the SMAP soil-moisture state as a model input — in India it is the difference between "trees cool 4 °C" and "trees cool 1 °C."

**Resolution policy:** model and report hotspots/interventions at **30 m** (Landsat-grid, where LST is real and 10 m drivers aggregate cleanly); use 10 m fractions as inputs and 100 m/500 m (GHSL/MODIS) only as priors/fallbacks. Keep an uncertainty band everywhere.

---

## 9. Pitfalls & build notes

- **Deprecations:** old `CAS/IGSNRR/PML/V2_v018` is **retired** → use `projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a`. ECOSTRESS V1 retired (2025) → V2. SMAP → v008. Hansen → 2024 v1.12. Always check the GEE **Data Catalog release notes** at build.
- **Scale factors:** MCD43A3 albedo ×0.001; MODIS LST ×0.02 (Kelvin); MOD13 NDVI ×0.0001; PML ×0.01; MOD16 ET ×0.1 (kg/m²/8day). Apply before physics.
- **NDBI false-positives** over bright bare/dry soil (common around Indian cities pre-monsoon) — always gate NDBI with the LULC vote and MNDWI.
- **MODIS 500 m–1 km is too coarse for intra-urban hotspots** — use it only as climatology/anchor/gap-fill, never as the hotspot resolution.
- **Albedo from S2/Landsat** is top-of-canopy directional, not BRDF-corrected — bias-correct to MCD43A3 (§6.4) before using in SEB.
- **Emissivity static bias:** ASTER GED is 2000–08; adjust by ΔFVC for new development/greening.
- **Bhuvan licensing/ingest:** verify NRSC data-use terms before redistributing/ingesting; no confirmed public GEE mirror (June 2026).
- **CONUS-only:** OpenET Ensemble does **not** cover India — use as method template, not data source.
- **Cloud masking is mandatory** for all optical/index work over monsoon India (s2cloudless + QA_PIXEL).

---

## 10. References (with URLs)

**LULC**
- ESA WorldCover v200 (GEE): https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200 ; v100: https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100 ; portal: https://esa-worldcover.org/en/data-access
- Google Dynamic World V1 (GEE): https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1 ; paper (Nature Sci Data 2022): https://www.nature.com/articles/s41597-022-01307-4 ; site: https://dynamicworld.app/about/
- ESRI/Impact Observatory 10 m Annual LULC (community catalog): https://gee-community-catalog.org/projects/S2TSLULC/ ; AWS: https://registry.opendata.aws/io-lulc/
- MODIS MCD12Q1 (GEE): https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1
- Copernicus Global Land Cover CGLS-LC100 C3 (GEE): https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_Landcover_100m_Proba-V-C3_Global
- C3S / ESA CCI Land Cover (community catalog): https://gee-community-catalog.org/projects/c3slc/
- GHSL BUILT_S 10 m (GEE): https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_S_10m ; BUILT_S: https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_S ; BUILT_V: https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_V ; BUILT_C: https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_C ; GHSL community: https://gee-community-catalog.org/projects/ghsl/ ; SMOD (JRC): https://data.jrc.ec.europa.eu/dataset/a0df7a6f-49de-46ea-9bde-563437a6e2ba
- GAIA Tsinghua FROM-GLC (GEE): https://developers.google.com/earth-engine/datasets/catalog/Tsinghua_FROM-GLC_GAIA_v10
- GISA (community catalog): https://gee-community-catalog.org/projects/gisa/ ; GISA method (ISPRS 2024): https://www.sciencedirect.com/science/article/abs/pii/S0924271624004945 ; GISD30 (ESSD 2022): https://essd.copernicus.org/articles/14/1831/2022/
- Bhuvan/NRSC India LULC: https://www.nrsc.gov.in/nrscnew/Apps_LULC.php ; thematic: https://www.nrsc.gov.in/nrscnew/Dataproducts_Thematic_overview.php

**Vegetation & ET**
- Sentinel-2 SR Harmonized (GEE): https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
- Landsat C2 L2 (GEE LC08): https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2
- MODIS MOD13Q1 (GEE): https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1 ; MYD13Q1: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MYD13Q1
- MODIS MOD15A2H LAI/FPAR (GEE): https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD15A2H
- Hansen Global Forest Change 2024 v1.12: https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/download.html ; 2023 v1.11: https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/download.html
- MODIS MOD16A2 ET (GEE): https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD16A2 ; MOD16A2GF: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD16A2GF
- PML_V2.2a ET+GPP (GEE): https://developers.google.com/earth-engine/datasets/catalog/projects_pml_evapotranspiration_PML_OUTPUT_PML_V22a ; data (Zenodo): https://zenodo.org/records/10647618
- OpenET Ensemble (GEE): https://developers.google.com/earth-engine/datasets/catalog/OpenET_ENSEMBLE_CONUS_GRIDMET_MONTHLY_v2_0 ; methods: https://etdata.org/methods/ ; accuracy (Nature Water 2023): https://www.nature.com/articles/s44221-023-00181-7
- ECOSTRESS L2 LST&E V1 (deprecated, see V2): https://www.earthdata.nasa.gov/data/catalog/lpcloud-eco2lste-001 ; ECO_L3T_JET V2: https://www.earthdata.nasa.gov/data/catalog/lpcloud-eco-l3t-jet-002

**Surface radiative / physical**
- MODIS MCD43A3 Albedo (GEE): https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD43A3
- ASTER GED AG100 v3 (GEE): https://developers.google.com/earth-engine/datasets/catalog/NASA_ASTER_GED_AG100_003 ; JPL: https://masterprojects.jpl.nasa.gov/emissivity/aster-ged
- MODIS MOD11A2 LST/ε (GEE): https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A2 ; MOD11A1: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A1
- SMAP L4 SPL4SMGP v008 (GEE): https://developers.google.com/earth-engine/datasets/catalog/NASA_SMAP_SPL4SMGP_008 ; tutorial: https://developers.google.com/earth-engine/tutorials/community/smap-soil-moisture
- Copernicus GLO-30 DEM (GEE): https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_DEM_GLO30

**Methods / formulas**
- Liang (2001) Narrowband-to-broadband albedo: https://terpconnect.umd.edu/~sliang/papers/RSE.N2B.2.pdf ; algorithms (ScienceDirect): https://www.sciencedirect.com/science/article/abs/pii/S0034425700002054
- Bonafoni & Şekertekin (2020) Sentinel-2 albedo coefficients (IEEE GRSL): https://ieeexplore.ieee.org/document/8974188/
- Sobrino et al. NDVI-threshold emissivity & LST (single-channel/split-window) — standard references for §5 emissivity & LST.

**General**
- GEE Data Catalog release notes (check deprecations at build): https://developers.google.com/earth-engine/docs/data-catalog/release-notes
- Awesome GEE Community Catalog: https://gee-community-catalog.org/

---

*Prepared by research agent R2 (LULC / vegetation / surface biophysical properties). Items marked 🟡 from-knowledge (verify) should be confirmed against the live GEE Data Catalog before the build commits to an exact asset path. All ✅ items were verified against authoritative sources during this research pass (June 2026).*
