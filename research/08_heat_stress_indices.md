# 08 — Heat Stress Indices, Hotspot Statistics & Heat Vulnerability (R8)

**Project:** ISRO Bharatiya Antariksh Hackathon 2026 PS-1 — Physics-informed geospatial AI/ML for urban heat-stress hotspots, driver quantification, LST modelling, cooling optimization. India-focused.

**Scope of this document:** How to *define* and *compute* "heat stress" / "hotspots" rigorously. Full formulas, inputs, output units, thresholds/categories, pros/cons, satellite-scale suitability, spatial hotspot statistics (Getis-Ord Gi*, Moran's I), Heat Vulnerability Index (HVI) construction for India, and a recommended **layered composite hotspot definition** the build team can implement directly in Google Earth Engine (GEE) + downscaled meteorology.

> **Notation convention used throughout.** `T` = dry-bulb air temperature (°C unless a formula is explicitly in °F), `Ta` = 2 m air temperature, `Ts` / `LST` = land surface temperature (skin), `RH` = relative humidity (%), `Td` = dew-point temperature (°C), `e` = water-vapour pressure (hPa), `ws` = 10 m wind speed (m/s), `Tw` = wet-bulb temperature (°C), `Tg` = globe temperature (°C), `Tmrt` = mean radiant temperature (°C), `Q` = net absorbed shortwave radiation (W/m²).

> **Provenance tags.** Formulas/thresholds are tagged `[verified]` (confirmed against an authoritative source this session, see References) or `[from-knowledge (verify)]` (standard literature value, recommend a final cross-check before publication). The hackathon's "many methods cross-verifying" priority is served by computing several indices and checking agreement.

---

## 1. Overview & the core conceptual split

"Heat stress" is **not one number**. There are three physically distinct families, and conflating them is the most common error:

1. **Surface thermal metrics** — derived from **satellite LST only**. They describe the *surface* (skin) thermal field: where impervious surfaces are radiometrically hot. Cheap, global, **fully O(1) server-side in GEE**. Examples: LST, SUHII, UTFVI, LST z-score / percentile, Getis-Ord Gi* on LST. **These do NOT measure what a human feels** — LST ≠ air temperature, and the urban canopy (2 m air) heat island is weaker and often peaks at night while the *surface* island peaks at midday.
2. **Human heat-stress / thermal-comfort indices** — describe physiological strain on a person. Require **air temperature + humidity** (and for the advanced ones, **wind + radiation**). Need downscaled ERA5 / station data, *not* raw LST. Examples: Heat Index, Humidex, Apparent Temperature, WBGT, Wet-Bulb Temperature, UTCI, PET, Discomfort Index, NET.
3. **Vulnerability-weighted heat risk** — combines *hazard/exposure* (1 and/or 2) with *population sensitivity* (age, health, poverty) and *adaptive capacity* (AC, green space, water). This is the **Heat Vulnerability Index (HVI)** and is what turns a physics map into an *intervention-priority* map.

The recommended deliverable (Section 12) is a **layered composite**: Surface hotspots (LST, O(1) GEE) ⊕ Human heat-stress hotspots (index-based) ⊕ Vulnerability-weighted priority hotspots (HVI), each as its own defensible layer plus a combined category legend.

**Key physics caveats the build team must respect:**
- LST is a *radiometric skin* temperature (emissivity-corrected brightness temperature), retrieved at satellite overpass (e.g., MODIS Terra ~10:30, Aqua ~13:30; Landsat ~10:30 local; ECOSTRESS at varying times). It is **instantaneous**, **clear-sky only**, and **not air temperature**. Typical midday LST−Ta gaps over dry impervious surfaces are 5–15 °C.
- The *Surface* UHI (SUHI, from LST) and the *Canopy* UHI (from 2 m air temperature) differ in magnitude, spatial pattern, and diurnal phase. Do not claim one is the other.
- Comfort indices (UTCI/PET) need **Tmrt**, which needs a radiation/geometry model (**SOLWEIG**) — heavy, not native O(1) GEE.

---

## 2. Master indices table

GEE-feasibility key: ✅ = native, O(1) per-pixel server-side on satellite LST; 🟡 = feasible in GEE but needs downscaled ERA5/station met layers ingested as images; 🔴 = needs an external radiation/geometry model (SOLWEIG/RayMan) or iterative solver, not native O(1).

| # | Index | Family | Core formula (compact) | Inputs | Output | Thresholds / categories | Data needs | GEE-feasible? |
|---|-------|--------|------------------------|--------|--------|--------------------------|------------|---------------|
| 1 | **LST** | Surface | radiative-transfer / split-window / single-channel from TIR + emissivity | TIR band(s), ε | °C / K | relative; use percentiles | Satellite TIR | ✅ |
| 2 | **SUHII** | Surface | `LST_urban − LST_rural` | LST + urban/rural mask | °C | >0 = warmer than rural | LST + LULC | ✅ |
| 3 | **LST z-score** | Surface | `(LST − μ)/σ` over AOI | LST | σ units | ≥+2σ hotspot | LST | ✅ |
| 4 | **LST percentile** | Surface | rank within AOI | LST | percentile | ≥90th / ≥95th hot | LST | ✅ |
| 5 | **Getis-Ord Gi\*** | Surface (spatial) | see §9 | LST + weights | z-score | \|z\|>1.96 (95%) | LST | ✅ |
| 6 | **Local Moran's I** | Surface (spatial) | see §9 | LST + weights | I_i, z | HH clusters | LST | ✅ |
| 7 | **UTFVI** | Surface | `(Ts − Tm)/Tm` | LST, mean LST | dimensionless | 6 EEI classes (§3) | LST | ✅ |
| 8 | **Ecological Eval. Index (EEI)** | Surface | categorical map of UTFVI | UTFVI | class | excellent→worst | LST | ✅ |
| 9 | **Heat Index (Rothfusz/NWS)** | Air comfort | 9-term regression (§5) | T(°F), RH | °F→°C | Caution→Extreme Danger | Ta, RH | 🟡 |
| 10 | **Humidex** | Air comfort | `T + 0.5555(e−10)` | T, e(Td) | °C-like | <30…>54 (§5) | Ta, Td | 🟡 |
| 11 | **Apparent Temp. (Steadman AT)** | Air comfort | `Ta + 0.33e − 0.70ws − 4.00` | Ta, e, ws | °C | feels-like | Ta, RH, ws | 🟡 |
| 12 | **WBGT (full)** | Air+rad comfort | `0.7Tnw + 0.2Tg + 0.1Ta` | Tnw, Tg, Ta | °C | ISO 7243 work limits | Tnw, Tg, Ta | 🔴 (needs Tg) |
| 13 | **WBGT (ABM simplified)** | Air comfort | `0.567Ta + 0.393e + 3.94` | Ta, e | °C | <28 / 28–32 / >32 | Ta, RH | 🟡 |
| 14 | **Wet-bulb Temp (Stull)** | Air | arctan expansion (§5) | T, RH | °C | 35 °C survivability | Ta, RH | 🟡 |
| 15 | **UTCI** | Air+rad comfort | 6th-order 210-coeff polynomial | Ta, Td, ws, Tmrt | °C | 10 stress classes (§6) | Ta, RH, ws, Tmrt | 🔴 (Tmrt) |
| 16 | **PET** | Air+rad comfort | MEMI energy balance (RayMan/SOLWEIG) | Ta, e, ws, Tmrt | °C | 9 perception classes (§6) | Ta, RH, ws, Tmrt | 🔴 |
| 17 | **Discomfort Index (Thom)** | Air comfort | `T − (0.55 − 0.0055·RH)(T − 14.5)` | T, RH | °C | <21…>32 (§5) | Ta, RH | 🟡 |
| 18 | **Net Effective Temp (NET)** | Air comfort | Hentschel/Missenard form (§5) | T, RH, ws | °C | feels-like | Ta, RH, ws | 🟡 |
| 19 | **Mean Radiant Temp (Tmrt)** | Radiation | from 6-directional fluxes / SOLWEIG | radiation, geometry | °C | input to UTCI/PET | DSM, DEM, met | 🔴 |
| 20 | **HVI** | Vulnerability | PCA / weighted z-scores (§11) | census + heat layers | index 0–1 | quintiles | census + LST/heat | 🟡 (precompute) |

---

## 3. Surface thermal metrics (LST-only — the O(1) GEE core)

### 3.1 LST (Land Surface Temperature)

**What it is.** Emissivity-corrected radiometric skin temperature of the ground. The single most important satellite input. Retrieval methods (covered in detail by the LST-retrieval research note; summarized here):
- **Split-Window (SW)** for sensors with two adjacent TIR bands (MODIS bands 31/32; Landsat 8/9 TIRS bands 10/11): `LST = Tb10 + c1(Tb10 − Tb11) + c2(Tb10 − Tb11)² + c0 + (c3 + c4·w)(1 − ε) + (c5 + c6·w)Δε`, where `w` = column water vapour, `ε` = mean emissivity, `Δε` = emissivity difference.
- **Single-Channel / Mono-window** (Jiménez-Muñoz & Sobrino; Qin) for one TIR band (Landsat 5/7): `LST = γ[ε⁻¹(ψ1·Lsen + ψ2) + ψ3] + δ`.
- **Mod. emissivity:** `ε = εv·Pv + εs(1 − Pv) + dε`, with `Pv = ((NDVI − NDVI_s)/(NDVI_v − NDVI_s))²` (fractional vegetation).

**Output:** K or °C. **GEE-feasible: ✅** — MODIS MOD11A1/MYD11A2 LST products are pre-computed; Landsat C2 L2 ST band is pre-computed; both are loaded directly. Custom SW/SC retrieval is a per-pixel band-math expression (still O(1)).

**Pros:** dense spatial coverage, long archive (MODIS since 2000, Landsat since 1982), physics-based. **Cons:** clear-sky only (cloud gaps), skin not air temperature, overpass-time snapshot, mixed-pixel/emissivity error, coarse (MODIS 1 km) vs fine (Landsat 100 m native ST, ECOSTRESS ~70 m).

### 3.2 Surface Urban Heat Island Intensity (SUHII)

**Definition.** `SUHII = LST_urban − LST_rural` `[verified]` — the difference in LST between the urban area and its adjacent rural surroundings, measured with satellite LST.

**Critical design choice = the rural reference.** Results swing with how "rural" is delineated. Recommended robust definitions (compute ≥2 and report sensitivity):
1. **Urban-mask vs ring buffer.** Urban = impervious/built (e.g., ESA WorldCover "built-up", Dynamic World "built", or GHSL). Rural = a buffer ring around the urban polygon (e.g., 1× to 2× the equal-area radius), **excluding** water bodies, impervious pixels, and pixels whose elevation differs strongly (>±50–100 m) from the city mean. `[verified]` (standard practice: within the buffer, remove water + impervious + large-elevation-difference pixels; remainder = rural).
2. **LCZ-based.** Urban = LCZ 1–10 (built types); Rural = LCZ A–G (natural) within the same scene (cleanest, recommended where an LCZ map exists).
3. **Percentile / SUHI-surface.** `SUHII_pixel = LST_pixel − median(LST_rural_reference)`; produces a continuous SUHI *surface* rather than one scalar.

**Output:** °C. **GEE-feasible: ✅** (reduceRegion for the rural median, then per-pixel subtraction).

**Pros:** intuitive, comparable across cities, directly maps the heat-island. **Cons:** rural-reference sensitivity (can change SUHII by several °C); seasonal/diurnal dependence; dry rural cropland can be *hotter* than irrigated urban parks (sign flips) — common in arid India (e.g., pre-monsoon NW India). **Mitigation:** restrict rural reference to vegetated/cropland with similar elevation, and report the choice.

### 3.3 LST z-score and percentile hotspots

**z-score:** `z(LST) = (LST_pixel − μ_AOI) / σ_AOI`, μ and σ over the analysis area (city or city+buffer). **Hotspot rule:** `z ≥ +2` (≈ top 2.3% if normal). Variants: `≥+1.5σ` (warm), `≥+2σ` (hot), `≥+2.5σ` or `≥+3σ` (extreme).

**Percentile:** rank LST within the AOI; **hotspot = LST ≥ P90** (hot), `≥ P95` (very hot), `≥ P98` (extreme). Percentiles are **distribution-free** (robust to skew/outliers) and are the recommended default for India where LST distributions are non-Gaussian.

**Output:** σ units / percentile rank. **GEE-feasible: ✅** (ee.Reducer.mean/stdDev or ee.Reducer.percentile). **Pros:** simple, scene-relative, no ancillary data. **Cons:** purely *statistical* (no spatial structure → salt-and-pepper noise; fix with Gi*/Moran's I §9); relative to scene so not absolute heat.

### 3.4 Urban Thermal Field Variance Index (UTFVI) & Ecological Evaluation Index (EEI)

**Formula** `[verified]`:
```
UTFVI = (Ts − Tm) / Tm
```
where `Ts` = pixel LST, `Tm` = mean LST of the study area. **Units of Tm/Ts:** use Kelvin (most published implementations use K so the ratio is well-behaved; using °C changes magnitudes and breaks the standard thresholds — **use K**). Dimensionless output.

**Standard 6-class threshold table (Liu & Zhang 2011; widely used)** `[verified]`:

| UTFVI range | UHI phenomenon | Ecological Evaluation Index (EEI) |
|-------------|----------------|-----------------------------------|
| < 0 | None | **Excellent** |
| 0 – 0.005 | Weak | Good |
| 0.005 – 0.010 | Moderate | Normal |
| 0.010 – 0.015 | Strong | Bad |
| 0.015 – 0.020 | Stronger | Worse |
| ≥ 0.020 | Strongest | **Worst** |

**EEI** = the categorical (excellent→worst) reclassification of UTFVI; the two are the same map under different labels. **Output:** class. **GEE-feasible: ✅** (band math + `.where()` reclass). **Pros:** standard in UHI literature, normalizes LST into transferable ecological-comfort classes, single-input. **Cons:** thresholds are empirical/scene-relative (Tm-dependent), not physiological; sensitive to AOI extent (changes Tm). Best as a *comparative* surface-comfort proxy, not human stress.

---

## 4. Wet-bulb temperature (the physiological survivability limit)

Wet-bulb temperature `Tw` is foundational: it is the lowest temperature reachable by evaporative cooling at ambient humidity, hence the **survivability threshold** (sustained `Tw ≈ 35 °C` ⇒ the body cannot shed metabolic heat by sweating — theoretically lethal; recent empirical work puts the practical limit lower, ~31–32 °C for many).

**Stull (2011) empirical approximation** `[verified]` — direct, non-iterative, from `T`(°C) and `RH`(%), at sea-level pressure (1013.25 hPa):
```
Tw = T·atan[0.151977·(RH + 8.313659)^0.5]
   + atan(T + RH) − atan(RH − 1.676331)
   + 0.00391838·(RH)^1.5·atan(0.023101·RH)
   − 4.686035
```
(arctan in radians, output °C.) **Valid:** −20 °C ≤ T ≤ 50 °C, 5% ≤ RH ≤ 99%; MAE < 0.3 °C, errors −1.0 to +0.65 °C; accuracy degrades at low RH if pressure ≠ 1013 hPa. **GEE-feasible: 🟡** (pure band math on Ta + RH layers).

**Pros:** closed-form (no psychrometric iteration), perfect for raster math; the cleanest single humid-heat danger metric. **Cons:** assumes sea-level pressure (apply a pressure correction for the Indian plateau/Deccan); RH (not LST) required.

---

## 5. Air-temperature comfort/stress indices (need air temp + humidity)

### 5.1 Heat Index — Rothfusz / NWS regression `[verified]`

The US operational "feels-like". **Primary regression (apply when the result ≳ 80 °F):** with `T` in **°F** and `RH` in **%**:
```
HI = −42.379 + 2.04901523·T + 10.14333127·RH
     − 0.22475541·T·RH − 0.00683783·T·T
     − 0.05481717·RH·RH + 0.00122874·T·T·RH
     + 0.00085282·T·RH·RH − 0.00000199·T·T·RH·RH
```
**Adjustments:**
- **Low-humidity** (RH < 13% AND 80 °F ≤ T ≤ 112 °F): `subtract  ADJ = [(13 − RH)/4]·sqrt{[17 − |T − 95|]/17}`.
- **High-humidity** (RH > 85% AND 80 °F ≤ T ≤ 87 °F): `add  ADJ = [(RH − 85)/10]·[(87 − T)/5]`.
- **Low-HI fallback** (when the simple value < 80 °F): use the **Steadman simple form** `HI = 0.5·{T + 61.0 + [(T − 68)·1.2] + (RH·0.094)}`, then average with `T`; if that average ≥ 80 °F switch to the full regression. `[verified]`

Convert to °C for mapping: `°C = (°F − 32)·5/9`.

**NWS categories** `[verified]` (on HI in °F): **Caution 80–90**, **Extreme Caution 91–103**, **Danger 103–124**, **Extreme Danger ≥ 125**. **GEE-feasible: 🟡.** **Pros:** operational, well-validated, widely recognized for public messaging. **Cons:** assumes shade + light wind (1.34 m/s) + a specific body model; ignores radiation and wind; only valid for warm/humid range; piecewise (must branch).

### 5.2 Humidex (Canada) `[verified]`
```
Humidex = T + 0.5555·(e − 10)
e = 6.11·exp[5417.7530·(1/273.16 − 1/(273.16 + Td))]
```
`T`, `Td` in °C; `e` in hPa; output °C-like (no unit per se). (`e` from dew point; equivalently from RH via the Tetens form `e = (RH/100)·6.105·exp[17.27T/(237.7+T)]`.) **Categories** `[verified]`: **< 30** little discomfort; **30–39** some discomfort; **40–45** great discomfort, avoid exertion; **46–53** dangerous (heat-stroke risk); **≥ 54** heat stroke imminent. **GEE-feasible: 🟡.** **Pros:** simple, dew-point-based (captures absolute moisture). **Cons:** humidity-only correction (no wind/radiation); undefined/clipped below ~20 °C; tends to overstate at very high humidity.

### 5.3 Apparent Temperature — Steadman (Australian BOM) `[verified]`

Non-radiation (shade) form:
```
AT = Ta + 0.33·e − 0.70·ws − 4.00
```
With-radiation form:
```
AT = Ta + 0.348·e − 0.70·ws + 0.70·Q/(ws + 10) − 4.25
```
`Ta` °C; `e = (RH/100)·6.105·exp[17.27·Ta/(237.7+Ta)]` hPa; `ws` = 10 m wind (m/s); `Q` = net radiation (W/m²); output °C. **GEE-feasible: 🟡** (non-radiation form); the radiation form needs `Q` (derivable from GEE solar + albedo, still 🟡). **Pros:** includes **wind** (cooling) and optionally radiation — more physical than HI/Humidex; the BOM operational standard. **Cons:** needs wind (and radiation) layers; "shade, walking adult" assumption.

### 5.4 Discomfort Index (Thom / THI) `[verified]`
```
DI = T − (0.55 − 0.0055·RH)·(T − 14.5)
```
`T` °C, `RH` %, output °C. **Categories** (Thom/Indian-applicable bands) `[from-knowledge (verify)]`: **< 21** no discomfort; **21–24** < 50% population uncomfortable; **24–27** > 50% uncomfortable; **27–29** most uncomfortable; **29–32** strong discomfort, health alert; **≥ 32** medical emergency / heat stress. **GEE-feasible: 🟡.** **Pros:** extremely simple, derivable directly from satellite-LST-proxied T in screening studies, classic in Indian thermal-remote-sensing work. **Cons:** crude, no wind/radiation; bands vary by author (calibrate for India).

### 5.5 Net Effective Temperature (NET) `[from-knowledge (verify)]`

Hentschel/Missenard "feels-like" adding the wind chill of evaporation:
```
NET = 37 − (37 − T)/[0.68 − 0.0014·RH + 1/(1.76 + 1.4·ws^0.75)] − 0.29·T·(1 − 0.01·RH)
```
`T` °C, `RH` %, `ws` m/s; output °C. **GEE-feasible: 🟡.** **Pros:** combines T, RH, wind in one closed form. **Cons:** several formula variants exist; less standardized than UTCI; verify coefficients before use.

---

## 6. Advanced energy-balance indices (need radiation → Tmrt)

### 6.1 WBGT (Wet-Bulb Globe Temperature)

The international occupational heat-stress standard (ISO 7243).

**Full outdoor (in sun)** `[verified]`: `WBGT = 0.7·Tnw + 0.2·Tg + 0.1·Ta`, where `Tnw` = natural (un-aspirated) wet-bulb, `Tg` = black-globe temperature (captures radiation), `Ta` = dry-bulb. **Indoor / no solar load:** `WBGT = 0.7·Tnw + 0.3·Tg`. Output °C.

**ABM (Australian BoM) simplified WBGT** `[verified]` — when only T + humidity are available:
```
e = (RH/100)·6.105·exp[17.27·Ta/(237.7 + Ta)]
WBGT = 0.567·Ta + 0.393·e + 3.94
```
`Ta` °C, `e` hPa, output °C. **Note:** this approximation **omits the solar/globe term**, so it under-reads in strong sun — treat as a *shade/indoor proxy* lower bound.

**Thresholds (ISO 7243 / common occupational, acclimatized, continuous work)** `[from-knowledge (verify)]`: **< 28** low risk; **28–30** moderate (reduce heavy work); **30–32** high (work/rest cycling, hydration); **> 32** very high/extreme (curtail strenuous work). Sport bodies use city-specific cutoffs. **GEE-feasible:** full = 🔴 (needs `Tg`, i.e., a radiation/globe model ~ Tmrt); ABM-simplified = 🟡.

**Pros:** the legal/operational heat-work standard, radiation-aware (full form). **Cons:** full form needs globe temp (radiation modelling); simplified form drops radiation; humid-weighted (0.7 on wet-bulb).

### 6.2 Mean Radiant Temperature (Tmrt) — the gateway to UTCI/PET

**Definition.** The uniform temperature of an imaginary black enclosure that would exchange the same net radiation with a person as the actual (non-uniform) environment. It is the dominant driver of *outdoor daytime* thermal stress (often more than air temperature).

**From measured 6-directional radiation (globe-thermometer or integral method):**
```
Tmrt = [ (1/σ)·Σ_i Wi·(ai_k·Ki + ai_l·Li) ]^0.25 − 273.15
```
σ = Stefan-Boltzmann (5.67×10⁻⁸ W m⁻² K⁻⁴); `Ki`,`Li` = short/long-wave fluxes from 6 directions; `Wi` angular/projection factors; `ai` absorption coefficients (≈0.7 SW, 0.97 LW for a human). From a **globe thermometer**: `Tmrt = [(Tg+273.15)⁴ + (1.10×10⁸·ws^0.6)/(ε·D^0.4)·(Tg − Ta)]^0.25 − 273.15` (D = globe diameter m, ε ≈ 0.95). **Output:** °C.

**At satellite/city scale → SOLWEIG.** Tmrt over a city cannot be measured per pixel; it is **modelled** by **SOLWEIG** (Solar and LongWave Environmental Irradiance Geometry; part of UMEP/QGIS) or **RayMan**. SOLWEIG ingests a **Digital Surface Model (building+tree heights), DEM, land cover, sky-view factor, and meteorology (Ta, RH, ws, global radiation)** and computes spatially-distributed Tmrt (and then PET/UTCI) accounting for shadows, walls, and sky-view. **GEE-feasible: 🔴** (run SOLWEIG offline on a DSM, then ingest the Tmrt/PET raster into GEE as an asset for fusion). This is the bridge from "surface" to "human" stress at street scale and is the recommended path for the *human-stress* layer in high-priority neighbourhoods.

### 6.3 UTCI (Universal Thermal Climate Index)

The reference outdoor human-stress index. Defined as the air temperature (in a reference condition: 50% RH capped at 20 hPa, 0.5 m/s wind at 10 m, Tmrt = Ta) producing the same physiological strain as the actual environment, via a multi-node (Fiala) thermoregulation + adaptive-clothing model.

**Computation:** a **6th-order polynomial with 210 coefficients** `[verified]` in four inputs — **Ta, (Ta−Tmrt), ws (10 m), and a humidity term (vapour pressure / Td)** — i.e. `UTCI ≈ Ta + offset(Ta, Tmrt−Ta, ws, e)`. (Use the published BioKlima/ utci_approx Fortran/Python coefficient set; do not retype by hand.) **Output:** °C (equivalent temperature).

**10-category thermal-stress scale** `[verified]`:

| UTCI (°C) | Stress category |
|-----------|-----------------|
| > 46 | **Extreme heat stress** |
| 38 – 46 | Very strong heat stress |
| 32 – 38 | Strong heat stress |
| 26 – 32 | Moderate heat stress |
| 9 – 26 | No thermal stress |
| 0 – 9 | Slight cold stress |
| −13 – 0 | Moderate cold stress |
| −27 – −13 | Strong cold stress |
| −40 – −27 | Very strong cold stress |
| < −40 | Extreme cold stress |

**GEE-feasible: 🔴** (needs Tmrt → SOLWEIG; the polynomial itself is band-math 🟡 once Tmrt exists). **Pros:** most comprehensive, validated, all four drivers, fine category resolution. **Cons:** needs Tmrt + wind; polynomial approximation RMSE ~1.1 °C vs lookup; heavy.

### 6.4 PET (Physiological Equivalent Temperature)

Air temperature of a *reference indoor setting* (Ta = Tmrt, ws = 0.1 m/s, e = 12 hPa, light activity, 0.9 clo) at which the human heat balance (Munich Energy-balance Model, MEMI) yields the same core/skin temperature as the actual outdoor condition. Computed by **RayMan / SOLWEIG**. **Inputs:** Ta, e, ws, Tmrt, plus person parameters. **Output:** °C.

**9-class thermal-perception / stress scale (Matzarakis, central-European calibration)** `[verified for the heat bands]`:

| PET (°C) | Perception | Stress |
|----------|-----------|--------|
| < 4 | Very cold | Extreme cold |
| 4 – 8 | Cold | Strong cold |
| 8 – 13 | Cool | Moderate cold |
| 13 – 18 | Slightly cool | Slight cold |
| 18 – 23 | Comfortable | No stress |
| 23 – 29 | Slightly warm | Slight heat |
| 29 – 35 | Warm | **Moderate heat** |
| 35 – 41 | Hot | **Strong heat** |
| > 41 | Very hot | **Extreme heat** |

**India note:** the bands above are temperate-calibrated; tropical-India studies show acclimatized comfort extends higher (neutral often 26–30 °C; recalibrate per local field surveys). **GEE-feasible: 🔴** (Tmrt/RayMan). **Pros:** intuitive °C scale, the standard for urban micro-climate / planning, directly couples to SOLWEIG. **Cons:** Tmrt-dependent; default thresholds need tropical recalibration for India.

---

## 7. Heat-wave definitions (temporal hazard) — IMD India + percentile

A *hotspot* is spatial; a *heatwave* is temporal — both feed exposure. Use the official IMD criteria for India.

**IMD Heat-Wave criteria** `[verified]`:
- **Pre-conditions:** declared only where the station's daily **Tmax ≥ 40 °C (plains)**, **≥ 37 °C (coastal)**, or **≥ 30 °C (hilly)**.
- **Based on departure from normal (when normal Tmax ≤ 40 °C):** *Heat Wave* = departure **4.5 to 6.4 °C**; *Severe Heat Wave* = departure **> 6.4 °C**.
- **Based on actual Tmax (when normal Tmax > 40 °C):** *Heat Wave* = departure **4.5 to 6.4 °C** (i.e., **≥ 45 °C** triggers); *Severe* = departure **> 6.4 °C**. **Override:** if **actual Tmax ≥ 45 °C** (Heat Wave) / **≥ 47 °C** (Severe), declare irrespective of normal.
- **Duration / spatial:** the criterion must be met at **≥ 2 stations in a meteorological subdivision for ≥ 2 consecutive days** (declared on day 2).

**Percentile / climatological definitions (research, gridded):** a heatwave = Tmax (or Tmin, or a humid-heat index) exceeding a **local percentile (commonly the 90th, 95th, or 99th of the daily climatology)** for **≥ 3 consecutive days** (definitions of duration vary: ≥2, ≥3, or ≥6 days). Percentile/relative thresholds travel across climates better than fixed °C and pair naturally with the percentile hotspot logic in §3.3. **Excess Heat Factor (EHF)** is a recommended duration×intensity metric for severity.

**GEE-feasible: ✅/🟡** — percentile heatwave detection on ERA5-Land or gridded IMD Tmax is a temporal reducer in GEE; IMD official declaration uses station normals (ingest as a table).

---

## 8. What needs what data (mapping at scale — the decision rule)

| You want… | Minimum data | Engine | Cost |
|-----------|--------------|--------|------|
| Surface hotspots (LST, SUHII, UTFVI, z/percentile, Gi*, Moran's I) | **Satellite LST only** | **GEE native** | **O(1)/pixel** ✅ |
| Humid-heat danger (Wet-bulb, Heat Index, Humidex, Discomfort Index, ABM-WBGT, Apparent Temp) | air **T + RH** (+ wind for AT/NET) — downscaled **ERA5-Land** or station-interpolated | GEE band math on ingested met rasters | 🟡 cheap |
| Full human comfort (UTCI, PET, full WBGT) | T + RH + **wind + Tmrt** | **SOLWEIG/RayMan offline** → ingest Tmrt/UTCI/PET raster → fuse in GEE | 🔴 heavy, neighbourhood-scale |
| Heatwave (temporal) | gridded daily Tmax (ERA5-Land / IMD) | GEE temporal reducer | ✅ |
| Vulnerability priority (HVI) | census + heat layer | precompute (PCA), join to grid/ward | 🟡 |

**Downscaling note.** ERA5-Land (~9 km) is too coarse for intra-city air temperature. Recommended O(1)-friendly downscaling: **statistically relate ERA5 Ta to satellite LST + NDVI + impervious fraction + elevation** (a regression/RF "LST→Ta" model) to produce ~100 m air-temperature surfaces, then feed humid-heat indices. This keeps the human-stress layer in GEE without SOLWEIG, at the cost of physical fidelity (no Tmrt/shadows).

---

## 9. Spatial hotspot statistics — robust clustering (Gi*, Moran's I)

Raw percentile/z thresholding gives noisy, speckled "hotspots". **Spatial statistics test whether high values *cluster* beyond chance**, giving statistically defensible, contiguous hotspots — directly aligned with the project's "robust" priority.

### 9.1 Getis-Ord Gi* (hot-spot analysis) `[verified]`

For each feature *i*, Gi* sums the values of *i and its neighbours* (weighted) and compares to the global sum; output is a **z-score**:
```
        Σⱼ wᵢⱼ·xⱼ − X̄·Σⱼ wᵢⱼ
Gi* = ───────────────────────────────────────────────
        S · sqrt{ [ n·Σⱼ wᵢⱼ² − (Σⱼ wᵢⱼ)² ] / (n − 1) }
```
where `xⱼ` = attribute value (e.g., LST) at feature *j*; `wᵢⱼ` = spatial weight between *i* and *j* (Gi* includes *i* itself, i.e., `wᵢᵢ`≠0); `n` = total features; `X̄ = (Σⱼ xⱼ)/n`; `S = sqrt{ (Σⱼ xⱼ²)/n − X̄² }`.

**Interpretation:** the output **is a z-score** (and p-value). **High +z = hot spot** (high LST clustered with high-LST neighbours); **high −z = cold spot**. Confidence bins (two-tailed): **|z| ≥ 1.65 → 90%**, **≥ 1.96 → 95%**, **≥ 2.58 → 99%**. (Apply an **FDR / Bonferroni** correction for multiple testing across many pixels.) **GEE-feasible: ✅** (neighbourhood reducers with a kernel give Σ wx, Σ w, Σ w²; global mean/SD via reduceRegion).

### 9.2 Local Moran's I (LISA) `[verified]`

Detects clusters **and spatial outliers**:
```
I_i = [ (xᵢ − X̄) / m₂ ] · Σⱼ wᵢⱼ·(xⱼ − X̄) ,   m₂ = Σ(xⱼ − X̄)²/n
```
Categorizes each location into **High-High** (hot cluster), **Low-Low** (cool cluster), **High-Low** / **Low-High** (outliers), each with a pseudo-p-value (permutation). **Global Moran's I** `I = (n/W)·[ΣᵢΣⱼ wᵢⱼ(xᵢ−X̄)(xⱼ−X̄)] / Σᵢ(xᵢ−X̄)²` (W = Σ all weights) gives one number for whether the LST field is clustered overall (I>0), random (≈0), or dispersed (<0).

**Gi* vs Moran's I:** Gi* = "where are the *hot* clusters" (signed by value, ideal for **heat hotspots**). Local Moran's I = "where are clusters **and** anomalies/edges". **Recommendation:** use **Gi\* on LST** as the primary hotspot delineator (clean hot/cold, statistically significant), and **Local Moran's I** as a cross-check + to flag isolated hot outliers (e.g., a single industrial roof). Running both satisfies cross-verification.

---

## 10. Recommended surface-hotspot pipeline (all O(1) in GEE)

1. Build a clean **multi-temporal LST composite** (e.g., warm-season median of Landsat C2 ST + MODIS for gap-fill), emissivity-corrected.
2. Compute, per pixel: **LST percentile**, **LST z-score**, **UTFVI/EEI class**, **SUHII** (≥2 rural references).
3. Run **Getis-Ord Gi\*** on the LST composite → significant hot clusters (95/99%). Cross-check with **Local Moran's I** (HH clusters).
4. **Surface-hotspot mask = (LST ≥ P90) AND (Gi\* z ≥ 1.96)** — combining magnitude (percentile) with significant clustering (Gi*) removes speckle and gives defensible polygons.
5. Grade severity by percentile bands (P90/P95/P98) and/or UTFVI class.

---

## 11. Heat Vulnerability Index (HVI) construction for India

HVI turns the physical hazard into an **intervention-priority** map by combining three IPCC-style domains:

**HVI = f(Exposure, Sensitivity, − Adaptive Capacity)** `[verified — IPCC three-domain framework used in Indian HVI studies]`.

### 11.1 Indicators (India-appropriate; map to Census 2011 ward/district + satellite layers)

**Exposure (the heat hazard itself):**
- Mean/max warm-season **LST** (or **SUHII**, **UTFVI**) — satellite.
- **Heatwave frequency/duration** (IMD/ERA5) and humid-heat (Wet-bulb/Heat Index) days.
- **Impervious surface fraction** / built-up density (GHSL, Dynamic World).
- **Lack of green space**: low **NDVI** / low tree-canopy fraction.

**Sensitivity (who is harmed more):**
- **% population aged ≥ 65** and **% aged ≤ 5** (Census age tables).
- **Population density** (Census; or WorldPop).
- **% with chronic illness / disability** (Census disability tables; NFHS health proxies).
- **% SC/ST**, **% below poverty / low income**, **% illiterate** (Census socio-economic).
- **% outdoor/manual workers** (Census economic activity; informal-sector proxy).

**Adaptive Capacity (who can cope — *inverted* in the index):**
- **Air-conditioner / electric-fan ownership**, **electricity access** (Census household amenities / assets).
- **Access to safe drinking water**, **good housing quality** (pucca vs kutcha roof/wall — Census housing; tin/asbestos roofs amplify indoor heat).
- **Green/blue space access**, **tree canopy** (satellite NDVI / parks).
- **Health-facility access**, **literacy**, **household income** (Census/NSSO).

### 11.2 Build method

1. **Aggregate** every indicator to a common unit (Census ward / enumeration block, or a 100–1000 m grid). Resolve LST/NDVI/impervious by zonal stats; join census tables by ward code.
2. **Normalize** each indicator to comparable scale: **z-score** `(x−μ)/σ`, or **min–max** `(x−min)/(max−min)` to [0,1]. **Orient** all so higher = more vulnerable (invert adaptive-capacity indicators, e.g., use `1 − AC_ownership`).
3. **Weight / combine** — two standard routes (run both; they cross-verify):
   - **PCA (recommended, data-driven):** run PCA per domain (or on all indicators), retain components with eigenvalue > 1 (Kaiser) / scree, weight each component by its **% variance explained**, sum the (sign-corrected) component scores → domain score; then combine domains. Two-step PCA (per-domain then overall) is the published Indian-city approach (e.g., 11–21 indicators). Removes subjectivity and collinearity.
   - **Equal / expert weights:** `HVI = wE·Exposure + wS·Sensitivity + wA·(1 − AdaptiveCapacity)` with `wE=wS=wA=1/3` as a transparent default, or AHP-derived weights. Evidence from Indian cities shows **adaptive capacity dominates** household vulnerability — consider up-weighting it or reporting its sensitivity.
4. **Classify** the final HVI into **quintiles** (Very Low → Very High) for mapping.
5. **Validate** against health outcomes where available (heat ED visits, all-cause summer mortality, ambulance calls) to check the index actually tracks harm.

**Output:** dimensionless HVI (0–1 or quintile). **GEE-feasible: 🟡** — the heat layers are GEE-native; PCA/weighting on census is best precomputed (Python/sklearn) and the result ingested as an asset for fusion. **Pros:** actionable, equity-aware, the standard for heat-action-plan targeting. **Cons:** census currency (2011 → use projections/NFHS/SECC updates), modifiable-areal-unit effects, indicator availability varies by city.

---

## 12. Recommended LAYERED composite "HEAT STRESS HOTSPOT" definition (deliverable)

A single number is indefensible; deliver **three transparent layers** plus a **combined priority legend**. Each layer is independently computable and citable.

### Layer A — **Surface Heat Hotspot** (LST-only; fully O(1) in GEE)
Per pixel, build a 0–100 **Surface Heat Score** from normalized LST percentile, with a clustering gate:
```
SurfaceScore = 100 × percentile_rank(LST_warmseason_composite)
SurfaceHotspot = (LST ≥ P90 of AOI) AND (Getis-Ord Gi* z ≥ 1.96)
```
Severity sub-bands by LST percentile: P90–P95 (High), P95–P98 (Severe), ≥P98 (Extreme). Cross-checked by UTFVI class (≥ "Strong") and Local Moran's I (HH). **Data:** satellite LST only. ✅

### Layer B — **Human Heat-Stress Hotspot** (index-based; needs met)
Compute a small ensemble and require **majority agreement** (cross-verification):
- **Tier B1 (cheap, T+RH, 🟡):** Wet-bulb Temperature (Stull), Heat Index (NWS), Humidex, Discomfort Index, ABM-WBGT. Flag pixel "stressed" when an index crosses its danger band (e.g., **WBT ≥ 28 °C** approaching danger / **≥ 31 °C** extreme; **Heat Index ≥ "Danger" (≈103 °F / 39.4 °C)**; **Humidex ≥ 40**; **DI ≥ 29**). **HumanStress = ≥ 3 of 5 indices in danger.**
- **Tier B2 (street-scale, 🔴):** where a DSM exists for priority wards, run **SOLWEIG → Tmrt → UTCI / PET**; flag **UTCI ≥ 32 °C (Strong heat stress)** / **≥ 38 (Very strong)**; **PET ≥ 35 °C** (recalibrated for India).

### Layer C — **Vulnerability-Weighted Priority Hotspot** (HVI)
```
PriorityScore = 0.5 × HazardScore + 0.5 × HVI_norm
HazardScore   = max(normalized Surface Heat Score [A], normalized Human Stress Score [B])
```
(Weights tunable; 50/50 is a defensible default. Use `max` so a pixel that is *either* a surface or a human-stress hotspot keeps full hazard weight.)

### 12.1 Recommended 5-class categorical legend (apply to PriorityScore 0–100)

| Category | Score | Meaning | Suggested colour (hex) |
|----------|-------|---------|------------------------|
| **Low** | 0 – 20 | No significant heat concern | `#2c7bb6` (blue) |
| **Moderate** | 20 – 40 | Elevated surface heat, low vulnerability | `#abd9e9` (light blue) |
| **High** | 40 – 60 | Significant heat hotspot **or** vulnerable area | `#ffffbf` (pale yellow) |
| **Severe** | 60 – 80 | Hot + clustered (Gi* sig.) **and** sensitive population | `#fdae61` (orange) |
| **Extreme** | 80 – 100 | Hot + humid-heat danger + high HVI — **act first** | `#d7191c` (red) |

(Diverging blue→red ColorBrewer `RdYlBu` reversed; colour-blind-safe.) For pure surface maps, an LST-style **YlOrRd** ramp (`#ffffb2`→`#bd0026`) is conventional and recommended.

**Why this is defensible:** (a) every layer cites established, peer-reviewed formulas and official thresholds (IMD, NWS, ISO 7243, UTCI, UTFVI/EEI); (b) the surface layer is reproducible and O(1) in GEE; (c) hotspots are statistically significant (Gi*/Moran's I), not arbitrary cutoffs; (d) multiple indices must *agree* (robustness); (e) vulnerability makes it equity-aware and actionable for an Indian Heat Action Plan.

### 12.2 Minimal vs full build

- **Minimal (hackathon-fast, all ✅):** Layer A only — LST composite → percentile + UTFVI + SUHII + Gi* → 5-class legend. Defensible "surface heat-stress hotspot" map with zero ancillary data.
- **Standard:** A + B-Tier1 (ingest downscaled ERA5/IMD T+RH → Wet-bulb/Heat-Index/Humidex ensemble) + C (HVI from Census 2011).
- **Full / showcase:** add B-Tier2 (SOLWEIG UTCI/PET for selected wards) for street-level human-stress validation.

---

## 13. Pitfalls & cross-verification checklist

- **LST ≠ air temperature ≠ comfort.** Label each layer correctly; never call a Surface-UHI map a "feels-like" map.
- **SUHII rural reference** dominates results → report ≥2 definitions; beware sign flips over dry Indian cropland.
- **UTFVI in Kelvin**, not °C, to keep standard thresholds valid.
- **Heat Index / Humidex are piecewise / range-limited** — branch correctly; convert °F↔°C consistently.
- **Wet-bulb (Stull) assumes 1013 hPa** — apply pressure correction for elevated Indian terrain.
- **UTCI/PET thresholds are temperate-calibrated** — recalibrate for tropical-India acclimatization before quoting.
- **Multiple-testing** in Gi*/Moran's I → apply FDR.
- **Census 2011 currency** → update with projections/NFHS; flag MAUP.
- **Cross-verify**: agreement among (percentile, UTFVI, Gi*) for surface, and among (WBT, HI, Humidex, DI) for human stress, is the robustness story for the jury.

---

## References (URLs)

**Surface metrics / UTFVI / SUHII**
- NOAA/Wikipedia — Urban Heat Island & Surface UHI definition: https://en.wikipedia.org/wiki/Urban_heat_island
- UTFVI formula + EEI threshold table (Liu & Zhang 2011 scheme): https://rashidfaridi.com/2026/03/08/urban-thermal-field-variance-index-utfvi-its-relation-to-uhi-and-lulc/
- UTFVI/EEI threshold table (ResearchGate fig.): https://www.researchgate.net/figure/Threshold-values-of-Urban-Thermal-Field-Variance-Index-UTFVI-and-ecological-evaluation_tbl1_342459246
- SUHII rural-reference sensitivity (urban-extent discrepancy, 892 cities): https://www.osti.gov/pages/biblio/2246610
- SUHII vs reference rural land cover: https://www.sciencedirect.com/science/article/abs/pii/S2212095521003047

**Spatial hotspot statistics**
- ArcGIS Pro — How Hot Spot Analysis (Getis-Ord Gi*) works (exact z-score formula): https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/h-how-hot-spot-analysis-getis-ord-gi-spatial-stati.html
- Getis–Ord statistics (Wikipedia): https://en.wikipedia.org/wiki/Getis%E2%80%93Ord_statistics
- Spatial autocorrelation / Local Moran's I (CASA0005): https://andrewmaclachlan.github.io/CASA0005repo/spatial-autocorrelation.html

**Air-temperature comfort/stress indices**
- NWS Heat Index — full Rothfusz regression + adjustments + simple form: https://www.wpc.ncep.noaa.gov/heat_index/details_hi.html
- NWS Heat Index equation body: https://www.wpc.ncep.noaa.gov/html/heatindex_equationbody.html
- NWS Technical Attachment SR 90-23 (Rothfusz original): https://www.weather.gov/media/ffc/ta_htindx.PDF
- NWS Heat Index categories chart: https://www.noaa.gov/sites/default/files/2022-05/heatindex_chart_rh.pdf
- Humidex — CCOHS rating & work (formula + categories): https://www.ccohs.ca/oshanswers/phys_agents/humidex.html
- Humidex (Wikipedia, formula + vapour pressure): https://en.wikipedia.org/wiki/Humidex
- Apparent Temperature — BoM Steadman model & formulas: https://www.bom.gov.au/info/thermal_stress/
- Australian Apparent Temperature (vCalc, both forms): https://www.vcalc.com/wiki/australian-apparent-temperature
- Apparent temperature (Wikipedia): https://en.wikipedia.org/wiki/Apparent_temperature
- Thom Discomfort Index (UrbanSIS): https://urbansis.eu/thom-discomfort-index/
- Thermal remote sensing of Thom's DI (Comparison w/ in-situ): https://www.researchgate.net/publication/260262908_Thermal_remote_sensing_of_Thom's_Discomfort_Index_DI_Comparison_with_in_situ_measurements

**Wet-bulb temperature**
- Stull (2011) — Wet-Bulb Temperature from RH and air temperature (JAMC): https://journals.ametsoc.org/view/journals/apme/50/11/jamc-d-11-0143.1.xml
- NCL `wetbulb_stull` (implementation): https://www.ncl.ucar.edu/Document/Functions/Contributed/wetbulb_stull.shtml

**WBGT**
- ABM simplified WBGT model (validation vs station data): https://pubmed.ncbi.nlm.nih.gov/30280211/
- ABM WBGT in outdoor workplaces (applicability): https://www.sciencedirect.com/science/article/abs/pii/S2212095519302469

**UTCI / PET / Tmrt / SOLWEIG**
- UTCI categories (Climate-ADAPT, EEA): https://climate-adapt.eea.europa.eu/en/metadata/indicators/thermal-comfort-indices-universal-thermal-climate-index-1979-2019
- UTCI equivalent-temperature stress categories (table): https://www.researchgate.net/figure/UTCI-equivalent-temperature-categorized-in-terms-of-thermal-stress_tbl1_239525964
- Introduction to UTCI (210-coeff polynomial, inputs): https://www.researchgate.net/publication/239525964_An_introduction_to_the_Universal_Thermal_Climate_Index_UTCI
- PET thermal-perception ranges (Matzarakis): https://www.researchgate.net/figure/Ranges-of-Physiologically-Equivalent-Temperature-PET-in-C-for-different-grades-of_tbl1_46010215
- PET calibration for tropical climate city: https://www.sciencedirect.com/science/article/abs/pii/S2212095522001146
- (SOLWEIG/RayMan are in UMEP/QGIS; Tmrt drives UTCI & PET.)

**Heat-wave definitions (India)**
- IMD FAQ on Heat Wave (criteria): https://internal.imd.gov.in/section/nhac/dynamic/FAQ_heat_wave.pdf
- IMD Cold & Heat Wave Indices and Methodology (Chapter-2): https://mausam.imd.gov.in/responsive/pdf_viewer_css/met2/Chapter%20-2/Chapter%20-2.pdf
- NDMA — Heat Wave: https://ndma.gov.in/Natural-Hazards/Heat-Wave
- IASPOINT — IMD heatwave criteria summary: https://iaspoint.com/about-heatwaves-imd-criteria-and-impacts/

**Heat Vulnerability Index (India)**
- HVI for four Indian cities — exposure/sensitivity/adaptive capacity, PCA (MDPI IJERPH, PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC8750942/
- HVI for Jodhpur City, India (two-step PCA, 11 indicators): https://www.sciencedirect.com/science/article/pii/S2667278225001130
- Heat Wave Vulnerability Mapping for India: https://www.researchgate.net/publication/315931202_Heat_Wave_Vulnerability_Mapping_for_India

---
*Prepared by research agent R8. Formulas tagged `[verified]` were confirmed against the cited authoritative sources this session; `[from-knowledge (verify)]` denotes standard literature values recommended for a final cross-check (notably Discomfort-Index bands, NET coefficients, WBGT/PET threshold bands which vary by author/region).*
