# R6 — Cooling Interventions, SOLWEIG & InVEST, and Spatial Optimization

**Research agent:** R6
**Scope:** Cooling-intervention catalog (quantitative °C effects), the two named tools **SOLWEIG** and **InVEST Urban Cooling Model**, and **spatial optimization** for choosing *which* interventions to place *where* to maximize cooling under budget/area/equity constraints.
**Project:** ISRO Bharatiya Antariksh Hackathon 2026, PS-1 — physics-informed geospatial AI/ML to map urban heat hotspots, quantify drivers, model LST, and **simulate + optimize cooling interventions**, outputting intervention **TYPE**, spatial **PLACEMENT**, and estimated **temperature reduction (°C)**.

> **How this fits the pipeline.** Upstream agents produce (a) an ML/physics LST model `LST = f(drivers)` and (b) per-pixel driver maps (NDVI, albedo, impervious fraction, building/canopy geometry, ET). R6 supplies: (1) **literature-anchored ΔT priors** per intervention to sanity-check the ML deltas; (2) two **independent biophysical estimators** (SOLWEIG for radiant/shade physics, InVEST for a fast O(1)-style city-wide CC/HM raster) that *cross-verify* the ML model — satisfying the "many methods that cross-verify" priority; and (3) an **optimizer** that consumes per-pixel ΔLST predictions and returns an intervention portfolio.

---

## 1. Overview & design philosophy

Every passive cooling intervention acts on one or more terms of the **surface energy balance**:

```
Q*  =  Q_H  +  Q_E  +  Q_G            (net radiation = sensible + latent + storage)
Q*  =  (1 - α)·K↓  +  ε·L↓  -  ε·σ·T_s^4
```

| Lever | Energy-balance mechanism | Interventions that use it |
|---|---|---|
| ↑ **albedo α** | reflect more shortwave `K↓` → less absorbed | cool/high-albedo roofs, cool pavements, light surfaces |
| ↑ **latent flux Q_E** (evapotranspiration / evaporation) | convert absorbed energy to latent heat instead of sensible → lowers `T_s` and `T_air` | trees, green roofs, green walls, parks, water, misting, permeable pavement |
| ↑ **shading** (block `K↓` reaching the surface) | canopy/structure intercepts direct beam → surface never heats | tree canopy, pergolas, building shade |
| ↓ **L↓ / Tmrt** (radiant exposure to humans) | reduce sky/surface longwave seen by a body → comfort | shade, geometry, high-SVF control |
| **storage Q_G / geometry** | aspect ratio H/W, thermal mass, ventilation corridors change trapping & flushing | canyon geometry, materials, wind corridors |

**Two distinct temperature targets** (do not conflate):
- **LST / surface temperature `T_s`** — what satellites (Landsat/ECOSTRESS/MODIS) measure; large swings (surface ΔT of 10–45 °C are common for roofs/pavements).
- **Air temperature `T_air`** (2 m) and **mean radiant temperature `Tmrt`** — what people feel; much smaller (typically 0.3–5 °C for air; Tmrt can drop >15–30 °C in shade).

The build should report **both**, because PS-1 asks for "temperature reduction (°C)" and the ML model is trained on LST, while human-relevant benefit is air/Tmrt.

---

## 2. Intervention effectiveness table (°C, cited)

Ranges are from peer-reviewed literature; "surface" = LST/skin temperature, "air" = 2 m air temp, "Tmrt" = mean radiant temp, "PET/feel" = perceived. Where a single representative value is useful for a prior, it is **bolded**.

| Intervention | Mechanism (energy-balance) | Surface ΔT (°C, cooling) | Air ΔT (°C) | Tmrt / feel ΔT (°C) | Cost / feasibility | Best where |
|---|---|---|---|---|---|---|
| **Urban trees / canopy** | shade (block K↓) + evapotranspiration (↑Q_E) | **2–12** under canopy; up to **5.1** at H/W=1, **8.2** at H/W=2 (LAI≈3.5) [1][5] | global midday LST −**1.5**; air −**0.3 per +10% canopy** [3][4] | PET/feel **2–8**; shaded sidewalk up to −8 [4] | Low–med capital, high maintenance (water, space, time to mature). Needs ~30–40% cover for population benefit [4] | Wide canyons, parking lots, pedestrian corridors, schools/clinics; daytime |
| **Green roofs** (extensive/intensive) | ET (↑Q_E) + insulation (↓Q_G into building) | **15–45** roof skin; substrate up to −24 to −33 vs bare [6] | near-surface −**2–5** locally; daytime surface-UHI −~4, night −~1 [6] | improves rooftop/indoor comfort | High capital + structural load + irrigation | Dense cores with flat roofs, low canopy potential; building-energy co-benefit |
| **Cool / high-albedo roofs** | ↑α → ↓absorbed K↓ | roof skin **10–30+** | per **+0.1 α → −0.2 to −0.6** near-surface (clear afternoon); −0.3 city-avg per +0.1 α; peak −0.6 to −2.3 [2] | indoor max −1.2 to −3.3 (non-AC) [2] | **Low cost**, fast, retrofittable (coating/membrane) | Large flat/low-slope roof stock; most cost-effective city-wide roof option [2] |
| **Cool / reflective pavement** | ↑α → ↓absorbed K↓ | pavement skin reduced (albedo dominates Tmax) [9] | modest; can *raise* daytime Tmrt for pedestrians via reflected K↓ (trade-off) | mixed (reflection ↑ on body) | Med cost; durability/glare concerns | Large paved areas away from pedestrians; parking, service roads |
| **Permeable / evaporative pavement** | evaporation of stored water (↑Q_E) + convection | **15–35** when wet [9]; 2–10 vs impervious [9] | small, transient (needs moisture) | local | Med cost; needs water/recharge | Hot-dry climates after rain/irrigation; stormwater co-benefit |
| **Water bodies / rivers / ponds** | evaporation (↑Q_E) + thermal mass | surface low | static water max −~1 [B]; up to −2–3 nearby | local, leeward | Site-dependent; land-intensive | Existing blue assets; downwind neighborhoods |
| **Fountains** | evaporation (↑Q_E, atomized → large surface area) | — | −**0.7–3** (very local, leeward) [B] | Low–med | Plazas, pedestrian nodes; hot-dry best |
| **Misting** | fine droplet evaporation (↑Q_E) | — | up to −**17.5** canyon max (hot-dry, Phoenix); ~½ benefit in humid; acts in <10 min [A] | strong local Tmrt drop | Low capital, ongoing water/energy | Transit stops, event spaces, hot-dry cities |
| **Urban parks / green space (PCI)** | aggregate shade+ET of a large patch | — | **Park Cool Island ~0.5–3.7** typical; large parks −3.28; up to ~−4.65 [7][8] | comfort gain inside & nearby | Land acquisition cost high | Patches >~2 ha cool surroundings (mixing) [InVEST] |
| **Green walls / vertical greening** | ET + shading of façade | wall skin **2–13.7** (avg ~7.5) [10] | near-façade (15 cm) up to −3.3; outdoor at street −1.2 to −3.0 [10] | local pedestrian comfort | High capital + maintenance/irrigation | Dense cores with no ground space; west/south façades, hot-arid |
| **↑ Vegetation fraction / NDVI** | ET + shade (continuous proxy) | strong negative LST–NDVI slope (city-specific; often −1 to −3 °C per 0.1 NDVI in summer) [from-knowledge (verify)] | scales with cover | — | Programmatic | Anywhere impervious-dominated |
| **Geometry / aspect ratio H/W** | controls shade & sky-view (Tmrt) and heat trapping/ventilation | — | ventilation corridors flush hot air; deep canyons shade day but trap night | Tmrt sensitive to H/W (deeper → cooler day Tmrt) [5] | Design-stage only (hard to retrofit) | New developments, master planning |
| **Materials / emissivity** | ↑ε (or engineered radiative coolers) → ↑L↑ emission | surface lower | small at city scale | — | Med (specialty coatings) | Roofs/surfaces; pairs with high-α |

**Caveats for the build:**
- Magnitudes are **strongly climate- and time-dependent**. Reflective surfaces shine in hot-dry/clear conditions; evaporative measures lose ~half their punch in humid air. Quote ranges, not point values, in outputs.
- **Pedestrian trade-off:** cool *pavement* lowers LST but can *raise* daytime Tmrt by reflecting shortwave onto bodies. For human-comfort objectives, prefer **shade (trees)** over high-albedo ground in pedestrian zones — SOLWEIG captures this; a pure-LST optimizer will not.
- Effects are **non-additive**: a tree over a cool roof does not give Δ_tree+Δ_roof. Optimizer must use the ML/biophysical model to evaluate combinations, not sum priors.

---

## 3. Per-intervention mechanism notes (concise)

**Urban trees / canopy.** Two coupled effects: (a) **shading** removes direct-beam `K↓` from the surface so it never stores heat (dominant for LST); (b) **transpiration** routes absorbed energy into latent `Q_E`, cooling air. High-LAI species cool more. Daytime-dominant; minimal night effect (can even slightly warm by reducing sky-view longwave loss). Global anchor: **−1.5 °C midday LST**, **−0.3 °C air per +10% canopy** [3][4]; local under-canopy −2 to −12 °C [1][5]. Implement as: ↑ shade factor, ↑ Kc/ET, ↑ NDVI, ↑ CDSM in the relevant tools.

**Green roofs.** Substrate + vegetation give **evaporative cooling** + **insulation** (cuts `Q_G` into building → energy savings). Drops roof skin 15–45 °C; near-surface air −2–5 °C locally; daytime surface-UHI −~4 °C [6]. Best in dense cores with flat roofs and little ground-canopy potential; main cost is structural load + irrigation.

**Cool / high-albedo roofs.** Pure **albedo** play: ↑α → ↓(1−α)K↓ absorbed. Cheapest, fastest, retrofittable (white membrane/coating). **+0.1 α ⇒ −0.2 to −0.6 °C** near-surface air; city-average −0.3 °C per +0.1 α; peak −0.6 to −2.3 °C [2]. London mesoscale study found cool roofs the **single most effective** roof-level measure for outdoor temperature [2]. Implement as ↑ albedo in biophysical table.

**Cool / permeable pavements.** Reflective variant = ↑α (albedo controls Tmax). Permeable/evaporative variant stores water and cools by **evaporation** when wet (15–35 °C skin reduction transiently) [9]. Watch the **pedestrian Tmrt trade-off** for reflective ground.

**Water / blue infrastructure / fountains / misting.** All evaporative (↑Q_E). Static water modest (−~1 °C) but stable; **atomization** (fountains/misters) multiplies surface area → far stronger but very local and leeward. Misting up to −17.5 °C canyon max in hot-dry Phoenix, halved in humid climates, effective in <10 min [A][B]. Cheap capital, ongoing water/energy cost; ideal at transit stops/plazas.

**Parks / green space (Park Cool Island).** A large green patch is cooler than its surroundings and **exports cooling** to a downwind/adjacent buffer that decays with distance. PCI magnitude commonly **0.5–3.7 °C**, occasionally up to ~4.6 °C; cooling distance ~100–300 m typical, up to km for very large parks [7][8]. InVEST encodes this via the `d_cool` exponential-decay term and a >2 ha threshold.

**Green walls / vertical greening.** ET + façade shading. Wall-skin −2 to −13.7 °C (avg ~7.5) [10]; near-façade air −3.3 °C at 15 cm, street-level −1.2 to −3.0 °C [10]. Niche where there is no ground space; high maintenance.

**Vegetation fraction / NDVI.** Continuous proxy combining shade+ET; the ML LST model already exposes NDVI as a driver, so "add vegetation" = perturb NDVI upward and re-predict. City-specific LST–NDVI slope (often steeply negative in summer).

**Materials & geometry.** ↑ε increases longwave emission (modest); engineered passive radiative coolers push further. **Aspect ratio H/W** and **sky-view factor** strongly control daytime shade and Tmrt and nighttime trapping; **ventilation corridors** flush accumulated heat. These are design-stage levers — most relevant to SOLWEIG (Tmrt-sensitive) and to new-development scenarios.

---

## 4. SOLWEIG deep-dive (UMEP/QGIS + Python)

**What it is.** SOLWEIG (SOlar and LongWave Environmental Irradiance Geometry) is a **microscale radiation model** that computes the spatial distribution of **3-D shortwave + longwave radiation fluxes** and, from them, **mean radiant temperature `Tmrt`**, **shadow patterns**, and **thermal-comfort indices** (PET, UTCI) at fine resolution (~1 m typical, demonstrated at 10 m). It is the radiation/shade workhorse of **UMEP** (Urban Multi-scale Environmental Predictor), a QGIS plugin. [SOLWEIG manual, UMEP-dev]

**What it computes / outputs (per timestep grids):**
1. **Tmrt** (mean radiant temperature, °C) — primary output.
2. Shadow pattern (binary/fractional shade).
3. Incoming/outgoing **shortwave** (`Kdown`, `Kup`) and **longwave** (`Ldown`, `Lup`).
4. At points-of-interest: directional fluxes (E/S/W/N + up/down), surface temp `Tg`, sky emissivity, clearness index, SVF, **PET**, **UTCI**.

**Mean radiant temperature physics.** SOLWEIG integrates the shortwave and longwave radiation a body receives from the **6 cardinal directions** (the Höppe 1992 approach), weights them by angular/projected-area factors for a standing/sitting person, then inverts Stefan–Boltzmann:

```
Tmrt = [ ( S_str / (ε_p · σ) ) ]^(1/4)  - 273.15           (°C)

S_str = α_k · Σ_i (W_i · K_i)  +  ε_p · Σ_i (W_i · L_i)
```
where `S_str` = total absorbed radiation (W m⁻²), `K_i`/`L_i` = short/longwave from direction *i*, `W_i` = angular weighting factors, `α_k` = body shortwave absorption (**default 0.70**), `ε_p` = body longwave emissivity/absorption (**default 0.95**), `σ` = Stefan–Boltzmann. (Equation form standard in the SOLWEIG literature; the UMEP manual states the 6-direction Höppe method but does not print the closed form.)

**Inputs required:**

| Input | Description |
|---|---|
| **Building+ground DSM** | Digital Surface Model (m a.s.l.) — buildings & terrain top |
| **DEM / DTM** (or land-cover to mask buildings) | ground elevation to separate buildings from terrain |
| **CDSM** (Canopy DSM) | top of vegetation (m above ground) — **trees** |
| **TDSM** (Trunk-zone DSM) | bottom of canopy / trunk height — lets light under trees |
| **Wall height + wall aspect** rasters | from UMEP pre-processor; needed for longwave from walls |
| **Sky View Factor (SVF)** grids | from UMEP "Sky View Factor Calculator" using DSM **and** CDSM |
| **Land-cover grid** | UMEP standard classes; sets ground α/ε and surface-temp scheme |
| **Meteorology** | global shortwave `K↓` (split into **direct** + **diffuse**), air temp `Ta`, relative humidity `RH`, (wind speed for PET/UTCI), pressure |
| **Params** | ground/wall albedo & emissivity; body absorption (α_k=0.70, ε_p=0.95) |

**How it evaluates shading / tree interventions.** Vegetation is represented **geometrically** by the CDSM (canopy top) and TDSM (trunk base): trees are solid shadow-casters with a transmissive trunk zone, so adding/removing trees = editing CDSM/TDSM and re-running. To test a tree-planting scenario: bump CDSM where trees are added, recompute SVF (DSM+CDSM), re-run SOLWEIG, and read the **ΔTmrt** map. This is the gold-standard for **pedestrian shade comfort** and is the right tool when the objective is Tmrt/PET rather than satellite LST. (Vegetation modeled as discrete obstacles, not a transmissivity coefficient, per UMEP manual; some SOLWEIG versions add a canopy transmissivity for sparse foliage.)

**Python / programmatic access — two routes:**
1. **UMEP for processing** — UMEP exposes its tools as **QGIS Processing algorithms** (`umep:Outdoor Thermal Comfort SOLWEIG`), callable headless via `processing.run(...)` from `qgis.core` / PyQGIS, or via the `umep-reqs` / `supy` ecosystem. Good for reproducing the full UMEP pre-processing chain (SVF, wall height/aspect).
2. **Standalone `solweig` PyPI package** — a **Rust-reimplemented** SOLWEIG (`pip install solweig`, Python 3.11–3.13, prebuilt wheels Linux/macOS/Win) for **high performance**. Minimal API:

```python
import numpy as np, solweig
from datetime import datetime

dsm = np.full((200, 200), 2.0, dtype=np.float32)
dsm[80:120, 80:120] = 15.0                      # a building block

surface  = solweig.SurfaceData.prepare(dsm=dsm, pixel_size=1.0)   # +optional cdsm, dem, landcover
location = solweig.Location(latitude=48.8, longitude=2.3, utc_offset=1)
weather  = solweig.Weather(datetime=datetime(2025,7,15,14,0),
                           ta=32.0, rh=40.0, global_rad=850.0)

summary = solweig.calculate(surface, weather=[weather],
                            location=location, output_dir="output/")
# outputs: Tmrt, shadow, UTCI/PET, Kdown/Kup, Ldown/Lup grids + multi-step stats
```
Key classes: `SurfaceData` (DSM/CDSM/DEM/land cover + caching), `Location` (lat/lon/UTC), `Weather` (Ta, RH, global_rad), `HumanParams` (posture, absorption).

**Role in our system.** SOLWEIG is the **microscale, physics-truth checker** for shade interventions and the source of **Tmrt/PET** human-comfort outputs. It is *not* O(1) city-wide (it is geometry-heavy), so use it on **hotspot tiles** flagged by the ML model to validate that predicted ΔLST/ΔTmrt for tree scenarios is physically consistent. The standalone Rust package makes tile-scale runs fast enough for an interactive optimizer "verify" step.

---

## 5. InVEST Urban Cooling Model deep-dive (formulas + Python)

**What it is.** A **fast, raster-algebra** ecosystem-services model (Stanford Natural Capital Project) that maps a **Cooling Capacity (CC)** index and a **Heat Mitigation (HM)** index city-wide from LULC + a biophysical table, then estimates **air-temperature reduction** and **valuation** (energy, productivity). It is essentially **map-algebra + two convolutions**, so it is **O(pixels)** and well-suited to the "fastest server-side compute" priority (it can be reproduced exactly in **Google Earth Engine** — see §5.5). Reference implementation: Bosch et al., *GMD* 14:3521, 2021 [InVEST-GMD].

### 5.1 Cooling Capacity index (daytime)

```
CC_i  =  0.6 · shade_i  +  0.2 · albedo_i  +  0.2 · ETI_i
```
Weights **(0.6, 0.2, 0.2)** are the empirical default (shade dominates). Each component ∈ [0,1].

- **shade** — proportion of tree canopy (≥2 m tall) for that LULC class (biophysical table).
- **albedo** — surface reflectivity (0–1) for that LULC class.
- **ETI** — Evapotranspiration Index:
  ```
  ETI_i = (K_c,i · ET0_i) / ET0_max
  ```
  `K_c` = crop/vegetation coefficient (table), `ET0` = reference evapotranspiration raster (mm, user-supplied, e.g. from CGIAR/GEE), `ET0_max` = max of ET0 over the AOI (normalizer).

**Nighttime alternative** (`cc_method="intensity"`):
```
CC_i = 1 - building_intensity_i      (building_intensity = floor-area ratio, 0–1)
```

### 5.2 Green-area cooling (Park Cool Island term)

For pixels within search radius `d_cool` of green space, compute a distance-decayed park contribution:

```
GA_i        = cell_area · Σ_{j ∈ d_cool}  g_j                 (green area nearby, j within d_cool)

CC_park_i   = Σ_{j ∈ d_cool}  g_j · CC_j · exp( − d(i,j) / d_cool )
```
`g_j` = 1 if pixel *j* is green (green_area flag), else 0; `d(i,j)` = distance; `d_cool` = max cooling distance (**default 100 m in GMD; UI default 450 m** — calibrate, Lausanne fit gave 89 m).

### 5.3 Heat Mitigation index

```
HM_i = CC_i           if  CC_i ≥ CC_park_i   OR   GA_i < 2 ha
HM_i = CC_park_i       otherwise
```
i.e. large parks (>2 ha) override local CC in their cooling buffer.

### 5.4 Air-temperature model

```
T_air_nomix,i = T_ref + (1 − HM_i) · UHI_max
T_air,i       = GaussianBlur( T_air_nomix , kernel_radius = r )       # spatial mixing
```
`T_ref` = rural reference air temp (°C), `UHI_max` = city UHI magnitude (°C, max urban–rural). Mixing radius **`r` default 500 m** (Gaussian convolution; Lausanne calibrated 236 m). Higher HM ⇒ lower local temperature, proportional to UHI_max.

**Validation anchor (Lausanne, GMD 2021):** calibrated InVEST reached **R² = 0.903, MAE = 0.955 °C, RMSE = 1.144 °C** against reference air temperature, **beating** a satellite-based spatial regression (R² 0.832). This is the cross-check accuracy to expect.

### 5.5 Inputs / outputs / valuation

**Inputs**
- `lulc_raster` — integer LULC; every code must appear in the biophysical table.
- `ref_eto_raster` — reference ET0 (mm).
- `aoi_vector` — analysis polygon.
- **Biophysical table (CSV)** columns: `lucode`, `kc`, `green_area` (0/1), `shade` (0–1), `albedo` (0–1), `building_intensity` (0–1, night method only).
- Scalars: `t_ref` (°C), `uhi_max` (°C), `green_area_cooling_distance` = `d_cool` (m), `t_air_average_radius` = `r` (m), `cc_method`.
- Optional valuation: `building_vector` (with `type`), `energy_consumption_table`, `avg_rel_humidity`.

**Outputs**
- Rasters: `cc_[suffix].tif` (Cooling Capacity), `hm_[suffix].tif` (Heat Mitigation), `T_air_nomix_[suffix].tif`, `T_air_[suffix].tif` (estimated air temperature).
- `uhi_results_[suffix].shp` (AOI stats: avg_cc, avg air temp, avg anomaly, avoided energy, WBGT, % work loss).
- `buildings_with_stats_[suffix].shp` (per-building energy savings, mean temp).

**Valuation methods**
- **Energy savings (per building b):**
  ```
  Energy.savings(b) = consumption.increase(b) · (T_air,MAX − mean(T_air over b))   [· cost(b)]
  T_air,MAX = T_ref + UHI_max ;  consumption.increase in kWh/(m²·°C)·footprint ; cost in currency/kWh
  ```
- **Work productivity** via **Wet-Bulb Globe Temperature**:
  ```
  WBGT_i = 0.567·T_air,i + 0.393·e_i + 3.94
  e_i    = (RH/100)·6.105·exp(17.27·T_air,i/(237.7+T_air,i))     (vapor pressure, hPa)
  ```
  Work-loss % thresholds — **light work:** 0% if WBGT<31.5; 25% [31.5–32.0); 50% [32.0–32.5); 75% ≥32.5 °C. **Heavy work:** 0% if WBGT<27.5; 25% [27.5–29.5); 50% [29.5–31.5); 75% ≥31.5 °C.
- **Mortality** is **not built in** (geographically variable); the manual points to external concentration–response functions (McMichael et al. 2003; Gasparrini et al. 2014) to attribute heat mortality to the ΔT the model predicts.

### 5.6 Python access

```python
# pip install natcap.invest
from natcap.invest.urban_cooling_model import urban_cooling_model

args = {
  "workspace_dir": "out/",
  "lulc_raster_path": "lulc.tif",
  "ref_eto_raster_path": "et0.tif",
  "aoi_vector_path": "aoi.gpkg",
  "biophysical_table_path": "biophys.csv",   # lucode,kc,green_area,shade,albedo,building_intensity
  "t_ref": 21.5,
  "uhi_max": 3.5,
  "t_air_average_radius": 500,               # r (m)
  "green_area_cooling_distance": 100,        # d_cool (m)
  "cc_method": "factors",                     # or "intensity" (night)
  "cc_weight_shade": 0.6, "cc_weight_albedo": 0.2, "cc_weight_eti": 0.2,
  "do_energy_valuation": True,
  "do_productivity_valuation": True,
  "building_vector_path": "buildings.gpkg",
  "energy_consumption_table_path": "energy.csv",
  "avg_rel_humidity": 45.0,
}
urban_cooling_model.execute(args)
```

**How to value interventions with it.** An intervention = **edit the biophysical attributes of affected pixels** (↑shade for trees, ↑albedo for cool roofs, ↑Kc/green_area for vegetation/parks) → re-run → read **ΔT_air** and **Δ(energy/WBGT)**. Because it is raster algebra, the whole CC→HM→T_air chain is **trivially portable to Google Earth Engine** (reduceNeighborhood for the `d_cool` exp-decay and the Gaussian mixing), giving the desired **server-side O(1)-style** city-wide compute and a second independent ΔT estimate to cross-verify the ML model.

---

## 6. Other models (note only)

| Model | Type / scale | Use here |
|---|---|---|
| **SUEWS** | Surface Urban Energy & Water Balance Scheme (neighborhood energy/water fluxes, hourly); in UMEP; Python `supy` | Energy-balance time series / ET partitioning; complements InVEST's static map |
| **ENVI-met** | CFD microclimate (m-scale, very detailed, very slow) | Reference/validation only — too slow for city-wide optimization |
| **i-Tree (Eco / Cool Air)** | Tree ecosystem services + 30 m cooling | Tree-specific cooling, energy, pollution co-benefits; calibratable at 30 m |
| **TARGET** | Tile-based fast urban temperature scheme | Fast neighborhood air-temp alternative to InVEST |
| **GREEN-CC** | Green cooling-capacity indicator | Lightweight greening-scenario screening |

**Recommendation:** **InVEST = fast city-wide screening (GEE-portable, O(pixels))**, **SOLWEIG = microscale shade/Tmrt truth on hotspot tiles**, ML model = data-driven LST. Triangulating these three gives the "many methods that cross-verify" robustness PS-1 asks for.

---

## 7. Optimization: choosing WHERE and WHICH interventions

### 7.1 Problem formulation

**Decision variables.** Discretize the city into candidate **sites** (pixels, parcels, roof polygons, street segments). For each site *s* and feasible intervention type *t*, a binary `x_{s,t} ∈ {0,1}` ("place type *t* at site *s*"). Feasibility mask `F_{s,t}` (e.g. trees only on plantable ground; cool roofs only on roofs; misting only at nodes).

**Per-decision cooling.** From the ML LST model (or InVEST/SOLWEIG), each `(s,t)` perturbs drivers → predict a **ΔLST (or ΔT_air) field** `Δ_{s,t}(p)` over pixels *p* (often local: the site + a decay buffer like InVEST's `d_cool`). Benefit at pixel *p* weighted by population and vulnerability `w_p` (see §8).

**Objective (single, scalarized):**
```
maximize   Σ_p  w_p · ΔT_p(x)                    # total (weighted) cooling delivered
subject to Σ_{s,t} c_{s,t} · x_{s,t} ≤ B         # budget
           Σ_{s,t} a_{s,t} · x_{s,t} ≤ A         # land/area (e.g. canopy %)
           Σ_t x_{s,t} ≤ 1   ∀ s                  # one intervention per site
           x_{s,t} ≤ F_{s,t}                      # feasibility
           x_{s,t} ∈ {0,1}
```
`c` = cost, `a` = area/resource use, `B`,`A` = budgets. ΔT_p(x) is **not** simply Σ Δ_{s,t} because effects interact and saturate — handle by (a) evaluating combinations through the model, or (b) using **submodular set-coverage** semantics where overlapping interventions on the same pixel yield **diminishing returns** (max/serial-absorption rather than sum).

**Multi-objective form** (preferred for decision support): vector objective
```
max [ cooling ,  −cost ,  co-benefits(stormwater/carbon/air) ,  equity ] ,  s.t. feasibility
```
→ produce a **Pareto front** of portfolios; let stakeholders pick the trade-off.

### 7.2 Algorithm toolbox (with guarantees and Python)

| Method | When / why | Guarantee | Python |
|---|---|---|---|
| **Greedy marginal-gain** | default; ΔT-coverage is **monotone submodular** (overlap → diminishing returns) | **(1 − 1/e) ≈ 0.63** of optimum for cardinality/budget constraints | hand-rolled; `apricot`, `submodlib` |
| **Lazy greedy (CELF) / Stochastic-greedy** | scale to millions of pixels | same bound, ~orders faster (lazy) / (1−1/e−ε) linear-time (stochastic) | `apricot` |
| **ILP / MILP (knapsack, facility-location)** | exact optimum, hard budget/area + logical constraints | **optimal** (or gap-bounded) | **PuLP** (CBC/HiGHS), **OR-Tools** CP-SAT |
| **NSGA-II / NSGA-III** | true multi-objective → Pareto front (cooling vs cost vs equity vs co-benefit) | heuristic; good spread via crowding distance | **pymoo**, DEAP |
| **Simulated annealing** | rugged landscape, custom non-linear objective, single-objective | heuristic; escapes local optima | `scipy.optimize.dual_annealing`, `simanneal` |
| **Spatial prioritization (Marxan-style)** | conservation-planning analogue; minimize cost s.t. cooling/coverage targets; emphasizes spatial compactness | SA-based heuristic | `prioritizr` (R), Marxan; or replicate via ILP |
| **Genetic algorithm (single-obj)** | flexible encoding, mixed intervention catalog | heuristic | `pymoo`, `DEAP` |

**Why submodular greedy is the recommended default (§7.3).** Heat-coverage with diminishing returns is exactly the setting where greedy carries a **provable 63% guarantee**, is **O(1) to add the next site** with lazy evaluation, handles the "stop when budget/area exhausted" rule natively, and produces a **ranked list** ("plant here first, then here…") that is directly actionable and explainable — ideal for a hackathon demo and for the PS-1 requirement to output *placement*.

### 7.3 Recommended approach (concrete, build-ready)

**Stage A — Candidate generation.** From hotspot map (ML LST) + feasibility layers, enumerate candidate `(site, type)` pairs only where feasible (e.g. trees on plantable pixels with low NDVI; cool roofs on roof polygons with low albedo; parks on vacant ≥2 ha). Cap with a coarse pre-filter (e.g. top-K hotspots) to bound size.

**Stage B — Marginal-ΔT oracle.** Wrap the ML model (and/or InVEST raster pass) as a function `eval(portfolio) → Σ_p w_p·ΔT_p`. For trees, apply ΔT with InVEST-style exponential distance decay (`exp(−d/d_cool)`); enforce **non-additivity** by taking, per pixel, `ΔT_p(portfolio) = T_baseline,p − model(perturbed drivers from all selected interventions)` (re-predict), or a `max`/serial-absorption combiner if re-prediction is too slow.

**Stage C — Optimize.**
1. **Primary: lazy-greedy submodular maximization** under the budget (and area) knapsack — fast, 0.63-guaranteed, yields a ranked placement list. Pseudocode:
   ```
   S = ∅; gains = priority_queue(all (s,t) by upper-bound marginal gain)
   while budget remaining and queue nonempty:
       pop (s,t) with largest stale gain; recompute g = eval(S ∪ {(s,t)}) − eval(S)
       if g still ≥ next item's stale gain (lazy check) and cost fits budget:
           S += (s,t); update spent; (mark site s used)
       else: reinsert (s,t) with updated g
   return S            # ordered by insertion = priority ranking
   ```
2. **Cross-check / exact: ILP in PuLP/OR-Tools** on the (reduced) candidate set with the same objective + constraints — confirms greedy is near-optimal and handles hard logical constraints (mutual exclusivity, must-cover vulnerable tracts). Use a **linearized** objective (precomputed `Δ_{s,t}` with an overlap penalty) for tractability.
3. **Decision support: NSGA-II (pymoo)** to expose the **cooling ↔ cost ↔ equity ↔ co-benefit** Pareto front when stakeholders want trade-offs rather than one answer.

**Stage D — Output.** For the chosen portfolio emit, per selected site: **intervention TYPE**, **PLACEMENT** (geometry/coords), **estimated ΔT (°C)** (with literature-range sanity flag from §2), cost, and cumulative city-wide ΔT and population-heat-exposure reduction.

**Coupling with the ML LST model (the core loop):**
```
for each candidate intervention:
    perturb driver rasters locally  (↑NDVI/shade/albedo/ET, ↑CDSM, ↓impervious)
    ΔLST_pixel = LST_model(baseline_drivers) − LST_model(perturbed_drivers)
    convert ΔLST → ΔT_air if needed; apply distance decay; weight by w_p (pop × vuln)
optimizer (greedy/ILP/NSGA-II) selects the portfolio maximizing Σ w_p·ΔT s.t. constraints
(optional) SOLWEIG/InVEST re-evaluate the chosen tiles to cross-verify ΔT physically
```

### 7.4 Empirical note
Spatial optimization beats random/uniform placement most when the **coverage target is mid-range (~30–70%)** — at very low or very high targets there is little freedom to exploit [opt-roof]. NSGA-II is the established choice for cooling-vs-cost-vs-connectivity green-infrastructure trade-offs [opt-multi][pymoo].

---

## 8. Equity / vulnerability weighting

Replace the uniform benefit with a **vulnerability-weighted** one so cooling is steered to those who need it most:

```
w_p = POP_p · HVI_p          (or  POP_p · g(HVI_p)  for a nonlinear emphasis)
```

**Heat Vulnerability Index (HVI).** Standard practice composes three pillars (IPCC framing): **Exposure** (LST/Tmrt, % impervious, low canopy), **Sensitivity** (% elderly ≥65, children, low income, chronic illness, isolation), **Adaptive capacity** (AC ownership, green access, income — enters with negative sign). Combine indicators via **min–max normalization + PCA** (or AHP weights) into a 0–1 tract/pixel index [equity-PNAS][HVI-Santiago].
- Variant **H3I** = hourly thermal-exposure (UTCI/SOLWEIG) × PCA social-vulnerability index — directly couples R6's Tmrt outputs to equity [equity-review].
- Data sources: census/socio-economic (e.g. India census wards, SECC), plus our exposure layers (LST, NDVI, impervious).

**Where it plugs in.** `w_p` enters the optimizer objective `Σ_p w_p·ΔT_p`. Options: (a) **weighting** (above), (b) **hard equity constraints** (e.g. ≥X% of budget, or guaranteed min ΔT, in top-quartile-HVI tracts), or (c) an **explicit equity objective** in NSGA-II (e.g. maximize ΔT delivered to the most-vulnerable decile, or minimize the ΔT gap between vulnerable and non-vulnerable populations). Studies show city-scale mitigation can be made equitable by explicitly accounting for demographic composition [equity-PNAS][equity-review]. AHP/weighted-overlay prioritization (Olgun et al. 2024) is a lighter-weight alternative for ranking priority zones [equity-review].

---

## 9. References (with URLs)

**Interventions — trees & canopy**
- [1] "Evaluating the effectiveness of tree canopy and building shade … solar radiation transmittance," ScienceDirect — under-canopy −5.1 °C (H/W=1), −8.2 °C (H/W=2). https://www.sciencedirect.com/science/article/abs/pii/S221209552500238X
- [3] WRI, "Cooling Potential of Urban Trees" — 806-city study, −1.5 °C midday LST; −0.3 °C air per +10% canopy; PET 2–8 °C. https://www.wri.org/insights/urban-trees-cooling-potential
- [4] Same WRI source + meta-analysis figures (30% canopy target; European mortality). https://www.wri.org/insights/urban-trees-cooling-potential
- [5] "Scale-dependent interactions between tree canopy cover and impervious surfaces …," PNAS. https://www.pnas.org/doi/10.1073/pnas.1817561116
- Review (115 studies 2018–2024), Arboriculture & Urban Forestry. https://auf.isa-arbor.com/content/early/2025/06/12/jauf.2025.023
- Frontiers — Bayesian spatially-varying canopy cooling. https://www.frontiersin.org/journals/forests-and-global-change/articles/10.3389/ffgc.2025.1644486/full
- Springer (Theoretical & Applied Climatology) — cooling-effect review/strategies. https://link.springer.com/article/10.1007/s00704-025-05904-2

**Green roofs**
- [6] "Assessing the impact of evapotranspiration from green roofs on reducing surface temperatures," ScienceDirect — substrate −24 to −33 °C. https://www.sciencedirect.com/science/article/abs/pii/S2352710224016632
- "Effectiveness of cool and green roofs as UHI mitigation," IOP — daytime surface-UHI ~−4, night ~−1 °C. https://iopscience.iop.org/article/10.1088/1748-9326/9/5/055002
- "Green roof effects on building surface processes and energy budgets," ScienceDirect. https://www.sciencedirect.com/science/article/abs/pii/S0196890423004466

**Cool roofs / albedo**
- [2] Brousse et al. 2024, *GRL*, "Cool Roofs … Most Effective … London." https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024GL109634
- [2b] "Cool roofs: climate change mitigation/adaptation for residential," ScienceDirect — indoor −1.2 to −3.3 °C. https://www.sciencedirect.com/science/article/pii/S0360132323002986
- EPA, "Using Cool Roofs to Reduce Heat Islands." https://www.epa.gov/heatislands/using-cool-roofs-reduce-heat-islands
- Nature Cities — temperature/mortality impact of cool roofs & rooftop PV, London. https://www.nature.com/articles/s44284-024-00138-1

**Cool / permeable pavements**
- [9] "Cool Pavements: State of the Art and New Technologies," MDPI Sustainability. https://www.mdpi.com/2071-1050/14/9/5159
- "Reflective and permeable pavements for heat-island & stormwater," IOP. https://iopscience.iop.org/article/10.1088/1748-9326/8/1/015023
- "Permeable pavements … evaporative cooling … field experiments," ScienceDirect. https://www.sciencedirect.com/science/article/abs/pii/S0360132325000071
- "Reducing Urban Heat Islands with Cool Pavements," MDPI Buildings. https://www.mdpi.com/2075-5309/15/3/504

**Water / blue infrastructure / misting**
- [A] "Urban heat mitigation through misting … blue infrastructure portfolios," ScienceDirect — canyon max −17.5 °C (Phoenix). https://www.sciencedirect.com/science/article/abs/pii/S0169204624002895
- [B] "Reducing heat with water," Urban Green-Blue Grids — static water −~1 °C, fountains −0.7–3 °C. https://urbangreenbluegrids.com/thema/heat/reducing-heat-with-water/
- "Vegetation + fountain cooling, horizontal & vertical," ScienceDirect. https://www.sciencedirect.com/science/article/abs/pii/S0360132324010345
- "Blue-Green Systems for urban heat mitigation," IWA. https://iwaponline.com/bgs/article/4/2/348/92495/

**Parks (PCI)**
- [7] "Graduated urban park size on park cooling island and distance vs LST," ScienceDirect — large park −3.28 °C. https://www.sciencedirect.com/science/article/abs/pii/S2212095522001730
- [8] "Cooling island effect … internal park landscape," Nature HSSC — Beijing mean 0.68 °C, Changzhou 3.65 °C. https://www.nature.com/articles/s41599-023-02209-5
- "Park cool island … tropical urban park (radiative cooling)," Sci Reports — day 2.21, night 1.69 °C. https://www.nature.com/articles/s41598-025-00207-y
- "Size threshold for cooling effect … accessibility & equity," PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC11246519/

**Green walls / vertical greening**
- [10] "Assessing green walls' effects on outdoor human thermal exposure," Sci Reports — outdoor −3.0 °C (wall), −1.2 °C (façade). https://www.nature.com/articles/s41598-025-26214-7
- "Green façades to control wall surface temperature," ScienceDirect — wall skin −2 to −12 °C. https://www.sciencedirect.com/science/article/abs/pii/S0360132317305607
- "Impact of vertical green façades on UHI & air quality," ScienceDirect. https://www.sciencedirect.com/science/article/pii/S0378778825015142

**SOLWEIG / UMEP / Tmrt**
- SOLWEIG Manual (UMEP). https://umep-docs.readthedocs.io/en/latest/OtherManuals/SOLWEIG.html
- UMEP-dev `solweig` README. https://github.com/UMEP-dev/solweig/blob/main/README.md
- `solweig` on PyPI (Rust reimplementation, Python API). https://pypi.org/project/solweig/
- UMEP Tutorial — Introduction to SOLWEIG. https://umep-docs.readthedocs.io/projects/tutorial/en/latest/Tutorials/IntroductionToSolweig.html

**InVEST Urban Cooling**
- InVEST Urban Cooling User Guide (formulas CC/HM/T_air, valuation). https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_cooling_model.html
- [InVEST-GMD] Bosch et al. 2021, *GMD* 14:3521 — spatially explicit InVEST; calibration R²=0.903, RMSE=1.144 °C; r/d_cool defaults. https://gmd.copernicus.org/articles/14/3521/2021/
- `natcap.invest.urban_cooling_model` API. https://invest.readthedocs.io/en/latest/api/natcap.invest.urban_cooling_model.html
- "Heat mitigation capacity of urban greenspaces with InVEST, verified vs daytime LST," ScienceDirect. https://www.sciencedirect.com/science/article/abs/pii/S0169204621001262

**Other models**
- i-Tree Cool Air + Copernicus land cover (30 m heat mitigation). https://www.tandfonline.com/doi/full/10.1080/22797254.2022.2125833
- "Evaluating urban greening scenarios … spatially-explicit" (InVEST-type), PMC. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8652265/

**Optimization**
- [opt-roof] "Optimization of Roof Greening Spatial Planning to Cool Down the Summer of the City," ScienceDirect — opt > random at 30–70% coverage. https://www.sciencedirect.com/science/article/abs/pii/S2210670721004996
- [opt-multi] "Optimizing urban green space spatial patterns … multi-objective (urban renewal)," ScienceDirect. https://www.sciencedirect.com/science/article/abs/pii/S0198971525000730
- "Modelling and optimization of urban green-blue infrastructure design for city cooling," ScienceDirect. https://www.sciencedirect.com/science/article/abs/pii/S0360132325005773
- "Optimal planning of urban greening … genetic algorithm (Tianjin)," ScienceDirect. https://www.sciencedirect.com/science/article/abs/pii/S2210670722005492
- [pymoo] NSGA-II in pymoo (docs). https://pymoo.org/algorithms/moo/nsga2.html
- PuLP (COIN-OR) — Python MILP modeler. https://github.com/coin-or/pulp ; docs https://coin-or.github.io/pulp/
- Google OR-Tools (CP-SAT / MILP). https://developers.google.com/optimization
- Submodular greedy (1−1/e): "Lazier Than Lazy Greedy" (Mirzasoleiman et al., AAAI). https://cdn.aaai.org/ojs/9486/9486-13-13014-1-2-20201228.pdf
- `apricot` submodular optimization (Python). https://github.com/jmschrei/apricot

**Equity / Heat Vulnerability**
- [equity-PNAS] "Prioritizing social vulnerability in urban heat mitigation," *PNAS Nexus* 2024. https://academic.oup.com/pnasnexus/article/3/9/pgae360/7745573
- [equity-review] "Urban heat, vulnerability, and mitigation in U.S. cities: a systematic review," ScienceDirect (H3I = exposure×SoVI; AHP prioritization). https://www.sciencedirect.com/science/article/pii/S3051052X26000550
- [HVI-Santiago] "A Heat Vulnerability Index … Santiago de Chile" (exposure/sensitivity/adaptive capacity, PCA). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5015864/
- Marxan-style spatial prioritization: `prioritizr` (R). https://prioritizr.net/

**Provenance flags.** All quantitative figures above are from the cited sources except: the **SOLWEIG Tmrt closed-form equation** (standard form assembled from the literature; UMEP manual states the 6-direction Höppe method but does not print it) and the **LST–NDVI slope (~1–3 °C per 0.1 NDVI)** — both marked *from-knowledge (verify)* in-text. Component formulas for InVEST (CC, ETI, CC_park, HM, T_air, WBGT, energy/productivity) are quoted from the InVEST User Guide and GMD 2021 paper.
