# 01 — Thermal Infrared / Land Surface Temperature (LST) Datasets & Sensors Catalog

> Research agent **R1** deliverable for ISRO Bharatiya Antariksh Hackathon 2026 PS‑1.
> Domain: Thermal‑infrared (TIR) remote sensing of **Land Surface Temperature** across *every* relevant satellite/sensor on Earth, plus the **fusion / gap‑filling** machinery that reconciles coarse‑frequent and fine‑rare sources into a cloud‑free, diurnal LST product.
> Status convention: items confirmed against a current web source are marked **[verified]**; items relying on expert knowledge (web not consulted or not authoritative for the exact figure) are marked **[from‑knowledge (verify)]**. Last verified: **2026‑06‑22**.

---

## 0. How to read this catalog (orientation for downstream build agents)

- **Server‑side / O(1) priority.** Wherever a product has a Google Earth Engine (GEE) `ImageCollection` ID, *prefer GEE* — the heavy compute (mosaic, reduce, regression, sharpening) runs on Google's servers and returns only the answer. Collection IDs are given verbatim; copy them exactly.
- **No single source.** This catalog is deliberately a *menu of ≥30 thermal products/methods* that **cross‑verify** and **fill each other's gaps**. The fusion section explains the blending logic.
- **Physics‑informed.** For each sensor we record the **LST retrieval algorithm** (split‑window SW / Temperature‑Emissivity‑Separation TES / single‑channel SC) and **emissivity handling**, because the downstream physics‑informed ML must respect the radiative‑transfer assumptions baked into each product.
- **Units gotcha.** Almost every gridded LST product is stored as **scaled integers in Kelvin**. The scale factors below are load‑bearing — getting them wrong silently corrupts every downstream °C number. Convert to °C as `K − 273.15` *after* applying scale/offset.
- **Indian‑city focus.** A dedicated stack recommendation at the end targets cities like Delhi, Mumbai, Ahmedabad, Hyderabad — high water‑vapour, high‑aerosol, monsoon cloud, dense built‑up.

### Three families of thermal sensors (the core trade‑off)

| Family | Examples | Spatial | Revisit | Why it matters for UHI |
|---|---|---|---|---|
| **Fine‑resolution polar (rare)** | Landsat TIRS (100 m→30 m), ASTER (90 m), ECOSTRESS (~70 m), TRISHNA/LSTM/SBG (future, 50–60 m) | 30–100 m | 3–16 d (ECOSTRESS irregular ISS) | Resolves *intra‑urban* hotspots (rooftops, parks, water bodies). Too infrequent alone. |
| **Coarse polar (daily, 2× / day)** | MODIS Terra+Aqua (1 km), VIIRS SNPP/NOAA‑20/21 (~750 m) | 750 m–1 km | 1–2 d, **day + night** | Captures *diurnal range* and frequent clear‑sky looks. Too coarse for street scale. |
| **Geostationary (sub‑hourly)** | INSAT‑3D/3DR/3DS (4 km), GOES‑R ABI (2 km), Himawari AHI (2 km), Meteosat SEVIRI (3 km), FY‑4A/4B AGRI (4 km) | 2–4 km | **10–15 min** | Full diurnal cycle, peak‑heat timing, gap‑filling under broken cloud. Very coarse spatially. |

The whole "≥30 methods that cross‑verify and fill gaps" thesis is: **use the geostationary fleet for *time*, the coarse polar pair for *day/night + frequency*, and the fine polar fleet for *space* — then statistically/physically fuse them.**

---

## 1. Master comparison table (all thermal/LST sources)

> GEE ID column: `—` means *not natively in GEE* (access via Earthdata/Copernicus/AWS instead). Resolutions are native at nadir. "Overpass" = approx. local solar time. All marked **[verified]** unless noted.

| # | Sensor / Product | Official product ID | GEE collection ID | Native res | Thermal bands (µm) | Revisit | Overpass (day/night) | Swath | LST algorithm | Emissivity | Access |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Landsat 8 TIRS** Surface Temp | `Landsat 8‑9 C2 L2 ST` (`ST_B10`) | `LANDSAT/LC08/C02/T1_L2` | 100 m TIR → resampled/served at 30 m | B10 ~10.9; B11 ~12.0 (B11 *not* for SW) | 16 d (8 d w/ L9) | ~10:30 desc / night via T2 rare | 185 km | **Single‑Channel (RIT/JPL v1.3)** on B10 | ASTER GED + NDVI‑scaled | GEE / USGS EE / AWS |
| 2 | **Landsat 9 TIRS‑2** Surface Temp | `Landsat 9 C2 L2 ST` (`ST_B10`) | `LANDSAT/LC09/C02/T1_L2` | 100 m → 30 m | B10 ~10.9; B11 ~12.0 | 16 d (8 d combined w/ L8) | ~10:30 | 185 km | Single‑Channel (B10) | ASTER GED + NDVI | GEE / USGS / AWS |
| 3 | **Landsat 7 ETM+** ST | `Landsat 7 C2 L2 ST` (`ST_B6`) | `LANDSAT/LE07/C02/T1_L2` | 60 m TIR → 30 m | B6 ~11.45 (single) | 16 d | ~10:00 (1999–2022; SLC‑off post‑2003) | 185 km | Single‑Channel (B6) | ASTER GED + NDVI | GEE / USGS |
| 4 | **Landsat 5 TM** ST | `Landsat 5 C2 L2 ST` (`ST_B6`) | `LANDSAT/LT05/C02/T1_L2` | 120 m TIR → 30 m | B6 ~11.45 (single) | 16 d | ~09:45 (1984–2013) | 185 km | Single‑Channel (B6) | ASTER GED + NDVI | GEE / USGS |
| 5 | **Landsat 4 TM** ST | `Landsat 4 C2 L2 ST` | `LANDSAT/LT04/C02/T1_L2` | 120 m → 30 m | B6 ~11.45 | 16 d | ~09:45 (1982–1993) | 185 km | Single‑Channel | ASTER GED + NDVI | GEE / USGS |
| 6 | **ECOSTRESS** LST&E (tiled) | `ECO_L2T_LSTE` v002 | `NASA/ECOSTRESS/L2T_LSTE/V2` ⚠ LA‑metro only in GEE | ~70 m | 5 TIR bands 8–12.5 | irregular (ISS, ~1–5 d) | **all hours incl. night** (ISS precessing) | ~384–402 km | **TES** (5‑band) + WVS atm corr | Retrieved (TES) | Earthdata LP DAAC / AppEEARS / GEE(LA) |
| 7 | **ECOSTRESS** LST&E (gridded) | `ECO_L2G_LSTE` v002/v003 | — | ~70 m | 5 TIR 8–12.5 | irregular ISS | all hours | swath‑derived | TES | Retrieved | LP DAAC / AppEEARS |
| 8 | **MODIS Terra** LST (SW) | `MOD11A1.061` daily | `MODIS/061/MOD11A1` | 1 km | B31 ~11.0; B32 ~12.0 | daily | ~10:30 / ~22:30 | 2330 km | **Generalized split‑window** | Classification‑based (land‑cover LUT) | GEE / LP DAAC |
| 9 | **MODIS Aqua** LST (SW) | `MYD11A1.061` daily | `MODIS/061/MYD11A1` | 1 km | B31; B32 | daily | ~13:30 / ~01:30 | 2330 km | Generalized split‑window | Classification LUT | GEE / LP DAAC |
| 10 | **MODIS Terra** LST 8‑day | `MOD11A2.061` | `MODIS/061/MOD11A2` | 1 km | B31; B32 | 8‑day composite | 10:30/22:30 | 2330 km | Split‑window | Classification LUT | GEE / LP DAAC |
| 11 | **MODIS Aqua** LST 8‑day | `MYD11A2.061` | `MODIS/061/MYD11A2` | 1 km | B31; B32 | 8‑day | 13:30/01:30 | 2330 km | Split‑window | Classification LUT | GEE / LP DAAC |
| 12 | **MODIS Terra** LST&E day (TES) | `MOD21A1D.061` | `MODIS/061/MOD21A1D` | 1 km | B29 ~8.5; B31; B32 | daily day | ~10:30 | 2330 km | **TES** + WVS | Retrieved (3‑band) | GEE / LP DAAC |
| 13 | **MODIS Terra** LST&E night (TES) | `MOD21A1N.061` | `MODIS/061/MOD21A1N` | 1 km | B29;B31;B32 | daily night | ~22:30 | 2330 km | TES | Retrieved | GEE / LP DAAC |
| 14 | **MODIS Aqua** LST&E day/night (TES) | `MYD21A1D/N.061` | `MODIS/061/MYD21A1D`, `MODIS/061/MYD21A1N` | 1 km | B29;B31;B32 | daily | 13:30 / 01:30 | 2330 km | TES + WVS | Retrieved | GEE / LP DAAC |
| 15 | **MODIS** LST&E 8‑day TES | `MOD21A2/MYD21A2.061` | `MODIS/061/MOD21A2`, `MODIS/061/MYD21A2` | 1 km | B29;B31;B32 | 8‑day | as above | 2330 km | TES | Retrieved | GEE / LP DAAC |
| 16 | **VIIRS SNPP** LST&E day | `VNP21A1D.002` | `NASA/VIIRS/002/VNP21A1D` | 750 m (served 1 km SIN) | M14 ~8.55; M15 ~10.76; M16 ~12.01 | daily day | ~13:30 | 3060 km | **TES** + WVS | Retrieved (3‑band) | GEE / LP DAAC |
| 17 | **VIIRS SNPP** LST&E night | `VNP21A1N.002` | `NASA/VIIRS/002/VNP21A1N` | 750 m→1 km | M14;M15;M16 | daily night | ~01:30 | 3060 km | TES | Retrieved | GEE / LP DAAC |
| 18 | **VIIRS SNPP** LST&E 8‑day | `VNP21A2.001/002` | `NASA/VIIRS/002/VNP21A2` (check) | 1 km | M14;M15;M16 | 8‑day | 13:30/01:30 | 3060 km | TES | Retrieved | GEE / LP DAAC |
| 19 | **VIIRS SNPP** LST (SW heritage) | `VNP21`/`VJ121` & `VNP15`/legacy `MOD11‑style` | `NOAA/VIIRS/001/VNP21A1D`? (use NASA/VIIRS) | 750 m | M15;M16 | daily | 13:30/01:30 | 3060 km | Split‑window (heritage) | LUT | LP DAAC |
| 20 | **VIIRS NOAA‑20 (JPSS‑1)** LST&E | `VJ121A1D/N` | — (Earthdata) | 750 m | M14;M15;M16 | daily | ~13:25 / ~01:25 | 3060 km | TES | Retrieved | LP DAAC |
| 21 | **VIIRS NOAA‑21 (JPSS‑2)** LST&E | `VJ221A1D/N` | — | 750 m | M14;M15;M16 | daily | ~13:25/01:25 | 3060 km | TES | Retrieved | LP DAAC |
| 22 | **Sentinel‑3A/B SLSTR** LST | `SL_2_LST___` | — (NOT in GEE) | 1 km TIR | S7 3.74; S8 10.85; S9 12.0 | ~1 d (2 sats) | ~10:00 desc / night | 1420 km (dual‑view 750 km) | **Split‑window** (biome+WVC+veg‑frac coeffs) | Biome/FVC‑classified | Copernicus DataSpace / EUMETSAT / MS Planetary Computer |
| 23 | **ASTER** Surface Kinetic Temp | `AST_08` v003 | — (use ASTER GED for emis) | 90 m | 5 TIR 8.125–11.65 | on‑demand (~16 d) | ~10:30 (Terra) | 60 km | **TES** + NEM | Retrieved (5‑band) | LP DAAC / Earthdata |
| 24 | **ASTER GED** emissivity+LST climatology | `AG100` v003 | `NASA/ASTER_GED/AG100_003` | 100 m | 5 TIR | static (2000–2008 mean) | n/a | n/a | TES + WVS | **Retrieved 5‑band** | GEE |
| 25 | **ECOSTRESS GEDv2** emissivity | `ECO_L3T_SEEBOP`/GED | — | ~70 m | 5 TIR | climatology | n/a | n/a | TES | Retrieved | LP DAAC |
| 26 | **GOES‑16/18/19 ABI** LST | `ABI‑L2‑LSTC/F/M` | — (AWS S3) | 2 km | B14 11.2; B15 12.3 | 60 min (FD); 5 min (meso) | continuous (all hours) | full disk | **Split‑window** (B14/B15) | Emissivity LUT (land‑cover) | AWS `noaa-goes16/18/19` / NOAA CLASS |
| 27 | **Himawari‑8/9 AHI** LST | JAXA/NOAA Himawari LST | — (JAXA P‑Tree, AWS) | 2 km | B13 10.4; B14 11.2; B15 12.4 | 10 min FD | continuous | full disk | **Nonlinear 3‑band / SW** | LUT | JAXA P‑Tree / AWS `noaa-himawari` |
| 28 | **Meteosat MSG SEVIRI** LST | LSA‑SAF `MLST` (LSA‑001/LSA‑004) | — (LSA SAF / EUMETSAT) | 3 km | IR10.8; IR12.0 (+ IR8.7) | 15 min | continuous | full disk (Africa/Europe; limited E. Asia) | **Generalized split‑window** | Veg‑fraction emissivity model | LSA SAF / EUMETCast |
| 29 | **Meteosat MSG SEVIRI** LST all‑sky | LSA‑SAF `MLST‑AS` (LSA‑005) | — | 3 km / 5 km | IR10.8;IR12.0 | 30 min | continuous | full disk | SW + energy‑balance (cloud) | model | LSA SAF |
| 30 | **Metop AVHRR** EPS daily LST | LSA‑SAF `EDLST` (LSA‑002) | — | 1.1 km (served 0.01°) | Ch4 10.8; Ch5 12.0 | daily | ~09:30 / ~21:30 | 2400 km | **Split‑window** (generalized) | Veg‑fraction model | LSA SAF / EUMETSAT |
| 31 | **INSAT‑3D Imager** LST | `3DIMG_*_L2B_LST` | — (MOSDAC) | **4 km** | TIR1 10.3–11.3; TIR2 11.5–12.5 | 30 min (≤15 min rapid) | continuous (Indian Ocean disk) | full disk | **Split‑window** (Singh et al.) | Emissivity from land‑cover | MOSDAC (registration) |
| 32 | **INSAT‑3DR Imager** LST | `3RIMG_*_L2B_LST` | — (MOSDAC) | 4 km | TIR1 10.3–11.3; TIR2 11.5–12.5 | 30 min | continuous | full disk | Split‑window | land‑cover emis | MOSDAC |
| 33 | **INSAT‑3DS Imager** LST | `3SIMG_*_L2B_LST` | — (MOSDAC) | 4 km | TIR1 10.3–11.3; TIR2 11.5–12.5 | 30 min | continuous | full disk | Split‑window | land‑cover emis | MOSDAC (2024+) |
| 34 | **FY‑4A AGRI** LST | NSMC `FY4A‑AGRI‑LST` | — (NSMC) | 4 km | IR 10.8; 12.0 (+3.7,6.2,7.1,8.5,13.5) | 15 min–1 h | continuous (E. Asia disk, covers India W. edge) | full disk | **SW (Ulivieri)** | LUT | NSMC Satellite Centre |
| 35 | **FY‑4B AGRI** LST | NSMC `FY4B‑AGRI‑LST` | — (NSMC) | 4 km | 4 TIR channels | 15 min FD | continuous | full disk | **Nonlinear SW** / TES research | LUT/retrieved | NSMC |
| 36 | **FY‑3D/3E MERSI‑II/LL** LST | NSMC MERSI LST | — | 1 km | TIR ~10.8;12.0 | daily (3E dawn‑dusk) | varies | 2900 km | Split‑window | LUT | NSMC |
| 37 | **TRISHNA** (ISRO+CNES) — *future ~2026* | TRISHNA L2 LST | — (planned) | **~57–60 m** | 4 TIR bands (8.6–11.6) | **~3 days** | day + night (SSO) | ~900 km | TES (planned) | Retrieved | ISRO/CNES (future) |
| 38 | **LSTM / Sentinel‑8** (Copernicus) — *future ~2028* | LSTM L2 LST | — (planned GEE/Copernicus) | **50 m** | 5 TIR 8.5–12.2 | **1–2 days** (2 sats) | ~13:00 + night | ~ wide | Split‑window/TES | Retrieved | Copernicus (future) |
| 39 | **SBG‑TIR** (NASA) — *future ~2029* | SBG‑TIR L2 LST | — (planned) | **60 m** | 5–7 bands 4–12 | **3 days** | ~12:30 + night | wide | TES | Retrieved | NASA/ASI (future) |
| 40 | **Landsat Next** — *future ~2030/31* | LС Next ST | — (planned GEE) | **60 m TIR** (10–20 m VSWIR) | **5 TIR bands** | **6 days** (3‑sat) | ~10:30 | 185 km | TES (5‑band) | Retrieved | USGS/NASA (future) |

> **Bold caveats already surfaced:** ECOSTRESS in GEE currently covers **only the Los Angeles metro** (rest of world via LP DAAC/AppEEARS) ⚠ critical for India. **Sentinel‑3 SLSTR LST is *not* in the GEE catalog** — pull from Copernicus Data Space, EUMETSAT, or Microsoft Planetary Computer STAC. Landsat **Band 11 is officially *not recommended* for split‑window** (stray‑light calibration uncertainty), which is *why* USGS Collection‑2 ST uses a **single‑channel** algorithm on Band 10 only.

---

## 2. Per‑sensor detail subsections

### 2.1 Landsat 8/9 TIRS — USGS Collection 2 Level‑2 Surface Temperature (PRIMARY fine‑res workhorse) **[verified]**

- **GEE IDs:** `LANDSAT/LC08/C02/T1_L2` (Landsat 8), `LANDSAT/LC09/C02/T1_L2` (Landsat 9). Tier‑2 variants `..._T2_L2` exist for lower geometric quality; Tier‑1 is the analysis‑ready set.
- **Surface Temperature band:** `ST_B10`. **Units = Kelvin** after applying **scale 0.00341802 and offset 149.0** → `K = DN*0.00341802 + 149.0`; then `°C = K − 273.15`. Stored as `uint16`, valid 1–65535. **This scale/offset is load‑bearing** — verified against USGS C2 L2 spec and GEE catalog.
- **Native resolution:** TIRS thermal optics are **100 m**, *resampled to 30 m* in the L2 product to match the OLI grid (cubic convolution). Treat the *effective* thermal resolution as ~100 m even though the raster is 30 m.
- **Thermal bands:** Band 10 ≈ 10.9 µm, Band 11 ≈ 12.0 µm. **USGS does NOT recommend Band 11 for split‑window** because of large stray‑light calibration uncertainty (post‑2017 correction reduced B10 error 2.1 K→0.3 K, B11 4.4 K→0.19 K mean but residual variability/uncertainty in B11 remained too high for SW). Hence the operational product is **single‑channel on Band 10**.
- **LST algorithm:** **Single‑Channel (SC)**, "Landsat Surface Temperature algorithm" v1.3.0 (RIT + NASA JPL). Inputs: L1 TIR (B10), TOA reflectance, TOA brightness temperature, **ASTER GED** emissivity, **ASTER NDVI**, and atmospheric profiles (geopotential height, specific humidity, air temperature) for radiative‑transfer atmospheric correction.
- **Emissivity handling:** From **ASTER GED**, dynamically **adjusted by current‑scene NDVI** (vegetation fraction). ⚠ **Known anomaly:** where NDVI changed substantially between the ASTER era (2000–2008) and the Landsat scene, emissivity adjustment produces erroneous ST — relevant for rapidly urbanizing Indian cities (cropland→built‑up).
- **Ancillary bands for QA:** `ST_QA` (ST uncertainty, K, scale 0.01), `ST_CDIST` (cloud distance), `QA_PIXEL` (CFMask cloud/shadow/snow bitmask — use for masking), `ST_EMIS`, `ST_ATRAN`, `ST_DRAD`, `ST_URAD`, `ST_TRAD`.
- **Overpass:** ~10:30 local solar time, descending (daytime). Nighttime Landsat thermal exists only rarely.
- **Strengths:** Best *spatial* detail for urban hotspots; 40+ yr archive (L4/5/7/8/9) → trend analysis; free, CARD‑ARD compliant; in GEE = O(1)-style server‑side.
- **Weaknesses:** 16‑day revisit (8‑day with L8+L9 combined) → sparse clear‑sky looks in monsoon India; single daytime overpass (no diurnal); single‑channel ⇒ more atmospheric sensitivity than SW; 100 m effective thermal optics.

### 2.2 Landsat 5/7 (and 4) historical ST **[verified]**

- **GEE IDs:** `LANDSAT/LT05/C02/T1_L2`, `LANDSAT/LE07/C02/T1_L2`, `LANDSAT/LT04/C02/T1_L2`. ST band = **`ST_B6`** (single thermal band), same **0.00341802 / 149.0** scale/offset to Kelvin.
- **Resolution:** TM B6 native **120 m** (served 30 m); ETM+ B6 native **60 m** (served 30 m).
- **Algorithm:** Single‑channel on Band 6, same v1.3.0 framework, ASTER GED + NDVI emissivity.
- **Coverage windows:** L5 1984–2013, L7 1999–present (SLC‑off striping after 2003‑05‑31 — gaps ~22% per scene off‑nadir), L4 1982–1993.
- **Use:** Long‑baseline UHI trend reconstruction back to the 1980s — invaluable to quantify *historical* urban‑heating drivers. L7 SLC‑off gaps must be gap‑filled (see fusion section).

### 2.3 ECOSTRESS (ECO_L2T_LSTE / ECO2LSTE) — diurnal high‑res TES gem **[verified]**

- **Product:** `ECO_L2T_LSTE` v002 (tiled, MGRS 109.8 km tiles, COG, one band per COG). Predecessor `ECO2LSTE` v001 (daily). Gridded `ECO_L2G_LSTE` v002/v003; swath `ECO_L2_LSTE` v002.
- **GEE ID:** `NASA/ECOSTRESS/L2T_LSTE/V2`. **Bands:** `LST` (Kelvin, direct), `LST_err` (K), `QC`, `EmisWB` (wideband emissivity), `cloud`, `height` (m), `water`, `view_zenith`. GEE temporal coverage seen ~2018‑07‑09 → 2026‑06‑20. **⚠ MAJOR CAVEAT: only tiles over the Los Angeles metro are ingested into GEE.** For Indian cities you MUST use **LP DAAC / AppEEARS / Earthdata** direct download (full global archive there).
- **Resolution:** **~70 m** — the *finest* frequent‑ish thermal globally available now.
- **Bands/spectral:** 5 TIR bands across 8–12.5 µm.
- **Algorithm:** **Physics‑based TES** (5‑band) with Water‑Vapor‑Scaling atmospheric correction → retrieves LST *and* emissivity simultaneously (no land‑cover LUT assumption).
- **Overpass:** Flown on **ISS** → *precessing, non‑sun‑synchronous* orbit. This is the **killer feature**: it samples **all local times including night** over weeks, enabling reconstruction of the **diurnal temperature cycle at 70 m** — exactly what UHI peak‑heat and nighttime‑heat‑retention analysis needs.
- **Strengths:** 70 m + diurnal sampling + true TES emissivity; best for fine‑scale ET and heat‑stress; pairs with Landsat for fusion.
- **Weaknesses:** *Irregular* revisit (depends on ISS overpass + duty cycle; gaps of days–weeks); historically some scenes had geolocation/duty‑cycle issues (v002 improved); not globally in GEE (LA‑only). Use as *opportunistic fine‑res truth*, not a regular time series.

### 2.4 MODIS Terra + Aqua — the diurnal coarse backbone **[verified]**

Two algorithm families coexist; **use both and cross‑check**:

**(a) MxD11 — Generalized Split‑Window (heritage, robust)**
- **GEE IDs:** `MODIS/061/MOD11A1` (Terra daily), `MODIS/061/MYD11A1` (Aqua daily), `MODIS/061/MOD11A2` / `MODIS/061/MYD11A2` (8‑day).
- **Key bands:** `LST_Day_1km`, `LST_Night_1km` — **scale 0.02**, Kelvin → `K = DN*0.02`; `°C = DN*0.02 − 273.15`. Valid DN 7500–65535. Also `QC_Day`/`QC_Night` (bitmask — filter to good quality), `Day_view_time`/`Night_view_time` (decimal hours, scale 0.1 — *use this to know the actual overpass time per pixel*), `Day_view_angl`/`Night_view_angl`, `Emis_31`, `Emis_32`.
- **Algorithm:** Wan & Dozier generalized **split‑window** (bands 31≈11 µm, 32≈12 µm). **Emissivity assigned by land‑cover classification LUT** (not retrieved) — fast and stable but wrong where land‑cover map is wrong (e.g., misclassified urban).
- **Overpass:** Terra ~10:30/22:30, Aqua ~13:30/01:30 → **four looks per day** (≈ four points on the diurnal curve). This is the workhorse for diurnal modeling.

**(b) MxD21 — TES (8.5/11/12 µm) (physically retrieves emissivity)**
- **GEE IDs:** `MODIS/061/MOD21A1D`, `MODIS/061/MOD21A1N`, `MODIS/061/MYD21A1D`, `MODIS/061/MYD21A1N`, plus `..._MOD21A2`/`MYD21A2` (8‑day).
- **Bands:** `LST_1KM` (Kelvin, scale 0.02), `Emis_29`, `Emis_31`, `Emis_32`, `View_Angle`, `View_Time`, `QC`.
- **Algorithm:** ASTER‑style **TES** + improved **Water‑Vapor‑Scaling**; retrieves LST *and* 3‑band emissivity → better over **bare/arid/heterogeneous** surfaces (much of India pre‑monsoon) where the land‑cover‑LUT emissivity of MxD11 is poor. ⚠ Caveat: MxD21 v6/6.1 shows **more high‑temperature outliers** than MxD11 — filter aggressively with QC.
- **Cross‑verify rule:** Where MxD11 and MxD21 agree within ~1 K → high confidence; large disagreement flags emissivity/atmospheric problems (often arid or humid extremes).

**Strengths (MODIS overall):** 1 km, twice‑daily ×2 satellites = best diurnal+frequency sampling among *free global* sensors; 2000–present (Terra), 2002–present (Aqua); native in GEE. **Weaknesses:** 1 km too coarse for intra‑urban; cloud gaps; Terra MODIS degrading with age (mind end‑of‑life — VIIRS is the continuity).

### 2.5 VIIRS (SNPP, NOAA‑20, NOAA‑21) — MODIS continuity at ~750 m **[verified]**

- **GEE IDs (SNPP):** `NASA/VIIRS/002/VNP21A1D` (day), `NASA/VIIRS/002/VNP21A1N` (night), `NASA/VIIRS/002/VNP21A2` (8‑day; verify exact GEE path). NOAA‑20 `VJ121A1D/N` and NOAA‑21 `VJ221A1D/N` via **Earthdata LP DAAC** (not confirmed in GEE).
- **Bands:** `LST_1KM` (Kelvin; product served on 1 km SIN grid, **native 750 m** resampled), `LST_err`, `Emis_14/15/16`, `QC`, `View_Angle`, `View_Time`. (SNPP GEE coverage observed 2012‑01‑19 → 2025‑10‑15, updated daily.)
- **Algorithm:** **TES** on M14 (8.55), M15 (10.76), M16 (12.01 µm) + improved WVS — same physical lineage as MOD21. There is also a heritage split‑window VIIRS LST (VNP21 vs older), but the **VNP21 TES** is the current standard.
- **Overpass:** SNPP/NOAA‑20/21 ~13:30/01:30 (afternoon constellation). **Three satellites** now → denser afternoon+night sampling and **MODIS‑gap insurance** as Terra/Aqua age.
- **Strengths:** 750 m (finer than MODIS), TES emissivity, guaranteed multi‑decade continuity (JPSS through 2030s), wide swath (no orbital gaps at equator). **Weaknesses:** afternoon‑only overpass cluster (no mid‑morning like Terra); 750 m still coarse for streets; bow‑tie/pixel growth off‑nadir.

### 2.6 Sentinel‑3 SLSTR — dual‑view split‑window, NOT in GEE **[verified]**

- **Product:** `SL_2_LST___` (Level‑2 LST). Files: `LST_in.nc` (the LST), `LST_ancillary.nc`. Two satellites **Sentinel‑3A & 3B**.
- **Access:** **Copernicus Data Space Ecosystem**, **EUMETSAT Data Store**, or **Microsoft Planetary Computer** (`sentinel-3-slstr-lst-l2-netcdf` STAC). **Not in the GEE catalog** — for a GEE‑centric pipeline you must ingest these externally or query Planetary Computer's STAC/COG.
- **Resolution:** TIR gridded to **1 km** (nadir+oblique dual view; VIS/SWIR 500 m). Dual‑view (nadir + ~55° backward) improves atmospheric correction.
- **Algorithm:** Operational **split‑window** with coefficients varying by **biome, water‑vapour content, fractional vegetation cover** (S8 10.85, S9 12.0 µm). S7 (3.74 µm) used for night.
- **Emissivity:** Biome/FVC‑classified (vegetation‑fraction model).
- **Overpass:** ~10:00 descending (mid‑morning) — complements MODIS Terra; combined 3A+3B ≈ daily.
- **Strengths:** Dual‑view atmospheric correction; operational, well‑validated (sub‑°C bias in validations); mid‑morning slot. **Weaknesses:** not in GEE (integration cost); 1 km; LST product mainly land — needs external pipeline.

### 2.7 ASTER (AST_08) + ASTER GED — the emissivity reference **[verified]**

- **AST_08 Surface Kinetic Temperature:** v003/v004, **90 m**, 5 TIR bands (8.125–11.65 µm), **TES + NEM** (Normalized Emissivity Method, Kirchhoff's law iteration). Terra ~10:30. **On‑demand acquisition** (ASTER is pointable; not continuous) → sparse but very high quality. Access via **LP DAAC / Earthdata**. Swath only **60 km**.
- **ASTER GED (`AG100` v003):** **GEE ID `NASA/ASTER_GED/AG100_003`.** A **static 100 m global emissivity + LST climatology** built from clear‑sky ASTER 2000–2008 via TES + WVS. Bands include `emissivity_band10..14`, `temperature` (mean LST), `ndvi`, `elevation`. **This is the emissivity backbone** that USGS Landsat C2 ST, and many downscaling schemes, build on. Compare with ECOSTRESS GED for change.
- **Role:** ASTER GED is the *go‑to prior* for per‑band emissivity in physics‑informed retrieval/sharpening; AST_08 scenes serve as fine‑res spot validation.
- **Weakness:** GED is a 2000–2008 climatology — stale over rapidly changing Indian urban fringes (same NDVI‑era anomaly as Landsat ST).

### 2.8 GOES‑R ABI LST (GOES‑16/18/19) — geostationary, Americas **[verified]**

- **Product:** `ABI‑L2‑LSTC` (CONUS), `ABI‑L2‑LSTF` (Full Disk), `ABI‑L2‑LSTM` (Mesoscale). **2 km** at nadir.
- **Access:** **AWS S3** open buckets `noaa-goes16`, `noaa-goes18`, `noaa-goes19` (us‑east‑1); also NOAA CLASS. **Not in GEE.**
- **Algorithm:** **Split‑window**, ABI **Band 14 (11.2 µm)** + **Band 15 (12.3 µm)**, emissivity from land‑cover LUT.
- **Cadence:** Full Disk hourly (LST product); ABI imaging 10‑min FD / 5‑min CONUS / 30‑s–1‑min meso. **Continuous diurnal**.
- **Relevance to India:** **Does NOT cover India** (Americas/Pacific). Listed for completeness and as the *template* for how to consume a geostationary LST stack from AWS — the Indian analog is **INSAT‑3D/3DR/3DS**.

### 2.9 Himawari‑8/9 AHI LST — geostationary, East Asia/Pacific **[verified]**

- **Access:** **JAXA P‑Tree** (Level‑3 LST), AWS `noaa-himawari8`/`noaa-himawari9`. **Not in GEE.** 2 km.
- **Algorithm:** **Nonlinear three‑band / split‑window** using AHI thermal bands B13 (10.4), B14 (11.2), B15 (12.4 µm); emissivity LUT.
- **Cadence:** **10‑min full disk** → excellent diurnal sampling.
- **Status note:** Himawari‑9 image‑observation function failed 2025‑10‑11; ops temporarily reverted to Himawari‑8 (2025‑10‑12). Operations evolving — verify current satellite before use.
- **Relevance to India:** Disk centered ~140°E → **covers only the far‑eastern edge of India / Bay of Bengal margins**, generally *not* mainland Indian cities. Useful only for NE‑India edge; INSAT is the primary geostationary source for India.

### 2.10 Meteosat SEVIRI (MSG) LST — geostationary, Africa/Europe (+ IODC over Indian Ocean) **[verified]**

- **Products (LSA SAF):** `MLST` = **LSA‑001** (15‑min clear‑sky LST); upgraded **LSA‑004**; all‑sky **MLST‑AS = LSA‑005** (30‑min, fills cloud via energy balance); daily‑max **MDLST**. **3 km** at nadir.
- **Algorithm:** **Generalized split‑window** (IR10.8, IR12.0; IR8.7 aux), **vegetation‑fraction emissivity model**.
- **Cadence:** **15 min**. Validations show excellent accuracy (~0.1 °C bias vs in‑situ in EU/Africa).
- **Access:** **LSA SAF** portal, **EUMETCast**.
- **Relevance to India:** The **0°‑disk** covers Africa/Europe (India near/over the eastern limb, high view‑zenith → degraded). **BUT** the **Meteosat IODC (Indian Ocean Data Coverage)** service (MSG positioned ~45.5°E, and MTG‑I) gives **much better geometry over India** — LSA SAF produces an **IODC LST**. ⚠ Use the **IODC** product, not the 0° product, for Indian cities (better view angle). Verify current IODC satellite (MSG → MTG transition).

### 2.11 Metop AVHRR (EPS) LST — polar, global daily **[verified]**

- **Product:** LSA‑SAF **EDLST = LSA‑002**, global daily day+night, **0.01°** sinusoidal grid (AVHRR native 1.1 km). Metop‑B/‑C (Metop‑A retired). Algorithm **split‑window** (Ch4 10.8, Ch5 12.0 µm), vegetation‑fraction emissivity.
- **Overpass:** ~09:30/21:30 (mid‑morning + evening) — a *different* time slot than MODIS/VIIRS → adds diurnal coverage.
- **Access:** LSA SAF / EUMETSAT. Validations ≈ −0.3 °C bias. **Continuity:** Metop‑SG with **METimage** (VII) extends this into the 2030s.
- **Role:** Independent polar cross‑check on MODIS/VIIRS at a complementary overpass time.

### 2.12 INSAT‑3D / 3DR / 3DS Imager LST — **India's geostationary thermal backbone** **[verified]**

- **Payload:** 6‑channel Imager incl. **TIR‑1 (10.2–11.2 µm)** and **TIR‑2 (11.5–12.5 µm)**, **4 km × 4 km** at sub‑satellite point. 3DR and 3DS imagers are replicas of 3D.
- **LST product:** `3DIMG_*_L2B_LST` (and `3RIMG_`, `3SIMG_` analogs). **Split‑window** retrieval (Singh et al., *JGR* 2016): reported std dev **1.78 K day / 1.41 K night**. Emissivity from land‑cover.
- **Cadence:** Full‑disk every ~30 min (rapid‑scan sectors faster) → **diurnal cycle over India**.
- **Access:** **MOSDAC** (`mosdac.gov.in`, registration) and IMD. **Not in GEE** — ingest from MOSDAC.
- **Why central:** This is the **only geostationary source with good viewing geometry directly over India** giving sub‑hourly diurnal LST. Coarse (4 km) but *temporally dense* — it is the time‑axis anchor for the Indian fusion stack, gap‑filling polar sensors under broken cloud and capturing peak‑heat timing (~14:00–15:00 LST) that twice‑daily polar passes miss.
- **Weakness:** 4 km spatial; split‑window emissivity assumptions over heterogeneous urban; product latency/availability varies (NRT via MOSDAC).

### 2.13 FengYun (FY‑4A/4B AGRI, FY‑3 MERSI) LST — China, partial India coverage **[verified]**

- **FY‑4A AGRI:** 14‑band geostationary, **4 km** TIR; official LST uses **Ulivieri (1985) split‑window**; hourly. FY‑4B AGRI: 4 TIR channels, **nonlinear split‑window**, 15‑min FD; research work also does simultaneous LST+emissivity (TES‑like) retrieval. **FY‑3D/3E MERSI** polar LST at ~1 km (FY‑3E is the world's first dawn‑dusk‑orbit civil met sat → unusual ~early‑morning overpass = extra diurnal sample).
- **Access:** **NSMC** (National Satellite Meteorological Centre, `nsmc.org.cn`). Not in GEE.
- **Relevance to India:** FY‑4 disk (~105°E) covers **eastern/most of India** at moderate view angle → a *useful independent geostationary cross‑check* alongside INSAT, especially for eastern India. FY‑3E dawn‑dusk adds a rare ~06:00/18:00 diurnal point.

### 2.14 Future / emerging fine‑res thermal missions (design for forward‑compatibility) **[verified]**

| Mission | Agency | Res (TIR) | Bands | Revisit | Launch | Algorithm | Note |
|---|---|---|---|---|---|---|---|
| **TRISHNA** | ISRO + CNES | **~57–60 m** | 4 TIR (8.6–11.6 µm) | **~3 d** day+night | ~2026 (PSLV) | TES | *India‑led*; 0.5 K instrument accuracy, ~1 K LST precision; **the single most important future source for Indian UHI** — design schemas to slot it in. |
| **LSTM (Sentinel‑8A/B)** | ESA/Copernicus | **50 m** | 5 TIR (8.5–12.2 µm) | **1–2 d** (2 sats) | A: 2028, B: 2030 | SW/TES | Field‑scale ET; ~13:00 + night; likely → GEE/Copernicus. |
| **SBG‑TIR** | NASA (+ASI) | **60 m** | 5–7 (4–12 µm) | **3 d** | ~Sep 2029 | TES | 0.2 K NEdT; 12:30 LST; ECOSTRESS successor. |
| **Landsat Next** | USGS/NASA | **60 m** TIR (10–20 m VSWIR) | **5 TIR** | **6 d** (3‑sat) | ~2030/31 | TES (5‑band) | 26 bands total; replaces L8/9; ST product continuity + better emissivity. |

> Build implication: TRISHNA (≈3‑day, 60 m, day+night) + LSTM (50 m) + SBG (60 m) will, by ~2029, *transform* the fine‑res revisit problem. Architect the LST data layer with a **pluggable sensor adapter** so these collections drop in without redesign.

---

## 3. LST Fusion & Gap‑Filling Strategy (the core of the cross‑verification thesis)

The fundamental problem: **no single thermal sensor gives fine space + frequent time + cloud penetration + diurnal coverage.** Solution = a **three‑stage fuse**: (A) **spatial downscaling/sharpening** of coarse LST to fine grid using fine‑res predictors; (B) **spatiotemporal fusion** to inject fine‑res *texture* into frequent coarse *time series*; (C) **temporal gap‑filling / diurnal reconstruction** using geostationary + harmonic/physical models. Then **cross‑validate** across sensors.

### 3.A Spatial downscaling / thermal sharpening (coarse → fine, same time)

Goal: take MODIS/VIIRS/geostationary 1–4 km LST and sharpen to ~30–100 m using *fine‑resolution explanatory variables* observed by other satellites (NDVI, NDBI, albedo, impervious fraction, NDWI, DEM, etc.).

| Method | Core idea | Predictors | Strength | Weakness |
|---|---|---|---|---|
| **DisTrad** (Kustas 2003) | Linear LST↔NDVI within coarse box; apply at fine scale; add residual | NDVI | Simple, fast, physical (veg cools) | Breaks over bare/urban/water where LST⊥NDVI |
| **TsHARP** (Agam 2007) | DisTrad using **fractional vegetation cover** `fc=(NDVI−NDVImin)/(NDVImax−NDVImin)`, often LST↔fc^0.625 | fc/NDVI | Standard baseline; works in vegetated areas | Same NDVI‑only limitation; poor in arid/urban |
| **Random‑Forest / ML sharpening** | Nonlinear regression LST=f(many fine predictors) at coarse scale, predict fine, add ATPRK/kriged residual | NDVI, NDBI, NDWI, albedo, impervious %, DEM, slope, LULC, night‑lights | **+~19% accuracy vs TsHARP**; handles urban heterogeneity & nonlinearity | Needs many predictors; can overfit; residual handling matters |
| **ATPRK** (Area‑To‑Point Regression Kriging) | Regression part + **geostatistical residual** downscaled by kriging so coarse pixel mean is preserved (mass‑conserving) | NDVI/multi | Preserves coarse aggregate exactly; reduces artifacts | Assumes stationarity |
| **GWR / GWRK / GWATPRK** | **Geographically‑weighted** regression (local coefficients) + ATP kriging residual | multi + space | Handles **non‑stationarity** (city ≠ countryside); best over heterogeneous terrain | Heavier compute |
| **Geographically‑Neural‑Network‑Weighted RK** | GNN learns local weights, esp. **nighttime LST** | multi | State of art for night downscaling | Complex |
| **Attention/Deep super‑resolution** | CNN/attention nets learn LST↑ texture from paired coarse–fine | imagery | Temporally dense 100 m LST demonstrated | Data‑hungry, less interpretable |

**Physics‑informed sharpening rule:** always (i) regress at the *coarse* scale, (ii) apply the model at the *fine* scale, (iii) **add the kriged/interpolated residual** so the fine prediction *aggregates back to the original coarse LST* (energy/mass conservation). This residual step is what separates principled sharpening (ATPRK/TsHARP‑with‑residual) from naive regression.

### 3.B Spatiotemporal fusion (fine texture × frequent time)

Goal: produce a **synthetic daily 30–100 m LST** by blending the *rare fine* (Landsat/ECOSTRESS) with the *frequent coarse* (MODIS/VIIRS) time series.

| Method | Idea | Use |
|---|---|---|
| **STARFM** | Predicts fine‑res image at date *t* from one/few fine–coarse pairs + coarse image at *t*, weighting spectrally/temporally/spatially similar neighbors | Baseline; good in homogeneous areas |
| **ESTARFM** | Enhanced STARFM; two pairs + conversion coefficient; better in **heterogeneous** landscapes (cities) | Preferred for urban |
| **FSDAF / Flexible STDF** | Adds unmixing + TPS interpolation; captures land‑cover change & gradual+abrupt change | Urbanizing fringes |
| **RFCDF / hybrid frameworks** | **Combine downscaling (TsHARP) + fusion (STARFM)** to get daily high‑res LST from ASTER+MODIS+Landsat | The "robust framework" pattern to emulate |
| **Deep STF (e.g., GAN/transformer)** | Learn the fusion mapping | Emerging, highest accuracy, less interpretable |

### 3.C Temporal gap‑filling & diurnal reconstruction (cloud + missing‑time)

1. **Cloud masking first:** every product has a QA/cloud bitmask (`QA_PIXEL`/CFMask for Landsat, `QC`/`cloud` for MODIS/VIIRS/ECOSTRESS, ABI/SEVIRI cloud masks). Mask, never interpolate over, unflagged cloud.
2. **Geostationary infill:** under broken/transient cloud, **INSAT‑3D (+FY‑4 for E‑India)** 15–30‑min frames catch clear moments a once/twice‑daily polar pass misses. Use them to fill polar gaps *and* to model the **diurnal temperature cycle (DTC)**.
3. **DTC model (physics‑informed):** fit a **diurnal temperature‑cycle model** (e.g., Göttsche–Olesen harmonic‑plus‑attenuated‑cosine, or Inamdar models) to the geostationary clear‑sky LST series → predict LST at any hour, normalize all polar observations to a **common reference time** (e.g., 13:30) so multi‑sensor LSTs are comparable. This is essential because Terra/Aqua/VIIRS/SLSTR/Metop each sample *different* local times.
4. **All‑sky LST:** for truly cloudy pixels, use **energy‑balance / surface‑energy‑budget all‑sky LST** (cf. SEVIRI MLST‑AS LSA‑005) or reanalysis‑assisted reconstruction (ERA5 skin temperature as a physical prior — see R‑meteo agent).
5. **Spatiotemporal interpolation:** for residual gaps, ML/geostatistical reconstruction (e.g., the "worldwide continuous gap‑filled MODIS LST" approach; spatio‑temporal attention networks demonstrated on FY‑4A) yields a **gap‑free, cloud‑free** product.
6. **Harmonic/seasonal model:** fit annual+diurnal harmonics per pixel for climatology‑based filling of long gaps and anomaly detection.

### 3.D Recommended end‑to‑end blended product (build recipe)

```
[Fine texture]      Landsat 8/9 ST (30 m) + ECOSTRESS 70 m (LP DAAC, India)
        │  thermal sharpening target grid = 30 m (or 70 m)
[Coarse frequent]   MODIS MxD11 + MxD21 (1 km, 4×/day) + VIIRS VNP21 (750 m, day+night)
        │  RF/GWATPRK sharpening with fine predictors (NDVI/NDBI/NDWI/albedo/imperv%/DEM)
        │  + residual kriging (mass‑conserving)
[Diurnal/time]      INSAT‑3D/3DR/3DS (4 km, 15–30 min)  (+ FY‑4 east, Metop & S3 extra slots)
        │  fit DTC model → normalize all to common overpass time; fill cloud gaps
[Fusion]            STARFM/ESTARFM (or RFCDF) blend fine+coarse → daily 30–100 m LST
[All‑sky/gap‑fill]  energy‑balance all‑sky + ERA5 skin‑T prior + ST attention reconstruction
        ▼
  GAP‑FREE, CLOUD‑FREE, DIURNAL 30–100 m LST cube  ← feed physics‑informed LST‑vs‑drivers ML
```

All sharpening/fusion regressions and reductions run **server‑side in GEE** wherever the inputs are GEE‑native (Landsat, MODIS, VIIRS, ASTER GED). For non‑GEE inputs (ECOSTRESS‑India, SLSTR, INSAT, GOES/Himawari/SEVIRI/FY) ingest via Earthdata/Copernicus/AWS/MOSDAC/NSMC and either upload as GEE assets or process in a parallel pipeline (xarray/dask on COG/NetCDF), then join.

---

## 4. Cross‑validation / gap‑filling role matrix (how each source verifies & fills the others)

| Source | Fills WHAT gap | Cross‑checks WHOM | Independence axis |
|---|---|---|---|
| **Landsat 8/9 ST (30 m, SC)** | *Spatial* detail (intra‑urban hotspots) | Validates sharpened MODIS/VIIRS; truth grid for STARFM | Single‑channel algo (independent of SW errors) |
| **ECOSTRESS (70 m, TES)** | Fine‑res **night + off‑hours** (diurnal at 70 m) | Validates Landsat ST & sharpened products; emissivity check vs ASTER GED | ISS precessing orbit → unique times; TES emissivity |
| **MODIS MxD11 (SW)** | Frequency + diurnal (4×/day) | Baseline vs MxD21 (algo cross‑check) & VIIRS | Split‑window + LUT emissivity |
| **MODIS MxD21 (TES)** | Better emissivity over arid/urban | Cross‑checks MxD11 (agreement→confidence) | TES emissivity (independent of LUT) |
| **VIIRS VNP21 (TES, 750 m)** | MODIS **continuity** + finer + afternoon/night | Cross‑checks MODIS Aqua (same ~13:30 slot) | Separate platform/sensor; TES |
| **Sentinel‑3 SLSTR** | **Mid‑morning (~10:00)** slot; dual‑view atm corr | Independent SW check on MODIS/VIIRS | Dual‑view geometry; biome‑coeff SW |
| **Metop AVHRR (EPS)** | **~09:30/21:30** slot | Independent SW check; different overpass | Different platform & time |
| **ASTER GED / AST_08** | **Emissivity reference** (90–100 m) + spot LST truth | Anchors emissivity for Landsat ST & sharpening | TES/NEM emissivity climatology |
| **INSAT‑3D/3DR/3DS (4 km, 15–30 min)** | **Diurnal cycle + cloud‑gap infill over India** | Provides DTC to normalize all polar sensors to common time | Geostationary (orthogonal sampling) |
| **GOES/Himawari/SEVIRI‑IODC/FY‑4** | Geostationary diurnal where they view; SEVIRI‑IODC & FY‑4 reach India | Multi‑geostationary cross‑check; SEVIRI all‑sky LST as cloud filler | Independent geostationary platforms/algos |
| **ERA5 skin temp (reanalysis)** | Physical all‑sky prior (cloudy pixels) | Sanity bound; bias‑correct extremes | Model‑based (truly independent of optical retrieval) |

**Principle of orthogonal errors:** sources differ along *algorithm* (SW vs TES vs SC), *emissivity* (LUT vs retrieved vs climatology), *orbit/time* (sun‑sync morning vs afternoon vs ISS‑precessing vs geostationary), and *resolution*. Because their error sources are largely **independent**, agreement among them is strong evidence; disagreement localizes the problem (cloud edge, emissivity, water vapour, view angle). Use an **ensemble/Bayesian fusion** that *weights each source by its per‑pixel QA + known uncertainty* (Landsat `ST_QA`, MODIS/VIIRS `LST_err`, ECOSTRESS `LST_err`, SLSTR uncertainty layers) rather than a naive mean.

---

## 5. Recommended primary + secondary stack for an Indian‑city urban‑heat build

> Optimized for: fast server‑side compute (GEE‑first), no single source, physics‑informed, robust, India geometry, day **and** night, intra‑urban detail.

### PRIMARY (do these first — all give strong India coverage)
1. **Landsat 8 + 9 C2 L2 ST** — `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2`, band `ST_B10` (×0.00341802 +149.0 → K). *Fine‑res spatial backbone, 8‑day combined, full archive for trends.*
2. **MODIS MxD11 + MxD21 (Terra & Aqua)** — `MODIS/061/MOD11A1`, `/MYD11A1`, `/MOD21A1D`, `/MOD21A1N`, `/MYD21A1D`, `/MYD21A1N` (LST ×0.02 → K). *Diurnal 4×/day, dual algorithm cross‑check, native GEE.*
3. **VIIRS VNP21 (SNPP)** — `NASA/VIIRS/002/VNP21A1D`, `/VNP21A1N` (750 m, TES, day+night). *Finer than MODIS + continuity + afternoon/night.*
4. **INSAT‑3D / 3DR / 3DS LST** — MOSDAC `3DIMG/3RIMG/3SIMG_*_L2B_LST` (4 km, 15–30 min). *India's geostationary diurnal anchor; DTC modeling & cloud‑gap infill.*
5. **ASTER GED** — `NASA/ASTER_GED/AG100_003`. *Emissivity prior for sharpening & retrieval QA.*

### SECONDARY (robustness, cross‑validation, fusion inputs)
6. **ECOSTRESS `ECO_L2T_LSTE` v002** via **LP DAAC/AppEEARS** (NOT GEE for India — GEE has LA only). *70 m diurnal validation & fine ET.*
7. **Sentinel‑3A/B SLSTR `SL_2_LST___`** via Copernicus Data Space / MS Planetary Computer. *Mid‑morning, dual‑view, independent SW check.*
8. **Metop AVHRR EPS LST (LSA‑002)** via LSA SAF. *09:30/21:30 slot.*
9. **Meteosat SEVIRI IODC LST (LSA‑SAF)** — use the **IODC** product (good India geometry), incl. **MLST‑AS (LSA‑005)** all‑sky for cloud filling.
10. **FY‑4A/4B AGRI LST** via NSMC. *Eastern‑India geostationary cross‑check; FY‑3E dawn‑dusk extra diurnal point.*
11. **ERA5 / ERA5‑Land skin temperature** (GEE: `ECMWF/ERA5_LAND/HOURLY`) — physical all‑sky prior & bias bound (coordinate with the meteorology agent).
12. **Future‑ready adapters:** TRISHNA (~2026, 60 m, 3‑day, day+night) → top priority when launched; LSTM (2028), SBG‑TIR (2029), Landsat Next (~2030).

### Fusion configuration for the Indian build
- **Sharpening:** Random‑Forest (or GWATPRK) sharpen MODIS/VIIRS 1 km → 30/70 m using fine predictors {NDVI, NDBI, NDWI, MNDWI, albedo, impervious %, building density, DEM, slope, VIIRS night‑lights, LULC}; **mass‑conserving residual** added.
- **Fusion:** ESTARFM/RFCDF to make **daily 30 m** LST from Landsat/ECOSTRESS × MODIS/VIIRS.
- **Diurnal/gap‑fill:** fit **DTC harmonic model** to INSAT (+FY‑4/SEVIRI‑IODC) → normalize all polar LSTs to common time; **SEVIRI MLST‑AS + ERA5** for cloudy all‑sky.
- **Ensemble:** per‑pixel **uncertainty‑weighted** fusion using each product's error layer; flag pixels where SW vs TES vs SC disagree > ~2 K.
- **Validation:** ECOSTRESS 70 m & ASTER 90 m spot scenes + (if available) in‑situ/AWS air‑temp proxies + cross‑sensor agreement statistics.

---

## 6. Quick reference — GEE collection IDs & scale factors (copy‑paste safe)

```text
# Landsat Collection 2 Level-2 Surface Temperature (band ST_B10 or ST_B6); K = DN*0.00341802 + 149.0
LANDSAT/LC09/C02/T1_L2          # Landsat 9 OLI-2/TIRS-2  (ST_B10)
LANDSAT/LC08/C02/T1_L2          # Landsat 8 OLI/TIRS      (ST_B10)
LANDSAT/LE07/C02/T1_L2          # Landsat 7 ETM+          (ST_B6)
LANDSAT/LT05/C02/T1_L2          # Landsat 5 TM            (ST_B6)
LANDSAT/LT04/C02/T1_L2          # Landsat 4 TM            (ST_B6)
#   QA: ST_QA (×0.01 K), QA_PIXEL (CFMask bitmask), ST_CDIST, ST_EMIS

# MODIS (LST_* bands ×0.02 -> Kelvin); view_time ×0.1 (hours)
MODIS/061/MOD11A1   MODIS/061/MYD11A1     # SW daily  (LST_Day_1km, LST_Night_1km, QC_Day/Night)
MODIS/061/MOD11A2   MODIS/061/MYD11A2     # SW 8-day
MODIS/061/MOD21A1D  MODIS/061/MOD21A1N    # TES Terra day/night (LST_1KM, Emis_29/31/32)
MODIS/061/MYD21A1D  MODIS/061/MYD21A1N    # TES Aqua  day/night
MODIS/061/MOD21A2   MODIS/061/MYD21A2     # TES 8-day

# VIIRS SNPP (LST_1KM Kelvin; native 750 m -> 1 km SIN); TES
NASA/VIIRS/002/VNP21A1D   NASA/VIIRS/002/VNP21A1N    # day / night
NASA/VIIRS/002/VNP21A2                                # 8-day (verify path)

# ECOSTRESS (LST band already Kelvin)  -- !! GEE = Los Angeles tiles ONLY !!
NASA/ECOSTRESS/L2T_LSTE/V2    # for India use LP DAAC / AppEEARS instead

# ASTER GED emissivity + LST climatology (static)
NASA/ASTER_GED/AG100_003

# Reanalysis skin temperature (physical all-sky prior)
ECMWF/ERA5_LAND/HOURLY        # skin_temperature band (Kelvin)

# NOT in GEE (ingest externally):
#   Sentinel-3 SLSTR  SL_2_LST___   -> Copernicus Data Space / EUMETSAT / MS Planetary Computer (sentinel-3-slstr-lst-l2-netcdf)
#   INSAT-3D/3DR/3DS  3DIMG/3RIMG/3SIMG_*_L2B_LST -> MOSDAC (mosdac.gov.in)
#   GOES-R ABI        ABI-L2-LSTC/F/M -> AWS s3 noaa-goes16/18/19   (Americas only)
#   Himawari AHI      LST            -> JAXA P-Tree / AWS noaa-himawari8/9 (E.Asia)
#   Meteosat SEVIRI   MLST/MLST-AS (LSA-001/004/005), use IODC for India -> LSA SAF / EUMETCast
#   Metop AVHRR       EDLST (LSA-002) -> LSA SAF
#   FengYun FY-4A/4B  AGRI LST       -> NSMC (nsmc.org.cn)
```

---

## 7. References (URLs actually retrieved during this research, 2026‑06‑22)

**Landsat Collection 2 Surface Temperature**
- USGS Landsat Collection 2: https://www.usgs.gov/landsat-missions/landsat-collection-2
- USGS Landsat C2 Surface Temperature: https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature
- USGS Landsat C2 Level‑2 Science Products: https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products
- Digital Earth Africa — Landsat C2 ST specs (ST_B10 scaling 0.00341802 + 149.0): https://docs.digitalearthafrica.org/en/latest/data_specs/Landsat_C2_ST_specs.html
- GEE catalog LC08 C02 T1_L2: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2
- TIRS stray light / Band 11 not recommended: https://landsat.gsfc.nasa.gov/article/landsat-8-thermal-data-ghost-free-after-stray-light-exorcism/
- TIRS LST before/after stray‑light correction (SURFRAD): https://www.mdpi.com/2072-4292/12/6/1023
- USGS Landsat 9: https://www.usgs.gov/landsat-missions/landsat-9
- TIRS split‑window uncertainty workflow (2025): https://arxiv.org/pdf/2511.12729

**ECOSTRESS**
- GEE ECO_L2T_LSTE V2 (LA‑only caveat, band list): https://developers.google.com/earth-engine/datasets/catalog/NASA_ECOSTRESS_L2T_LSTE_V2
- Earthdata ECO_L2T_LSTE v002: https://www.earthdata.nasa.gov/data/catalog/lpcloud-eco-l2t-lste-002
- LP DAAC ECO_L2_LSTE v002: https://lpdaac.usgs.gov/products/eco_l2_lstev002/
- ECO_L2G_LSTE v003: https://www.earthdata.nasa.gov/data/catalog/lpcloud-eco-l2g-lste-003

**MODIS**
- GEE MOD11A1.061: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A1
- GEE MOD11A2.061: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A2
- GEE MOD21A1N.061: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD21A1N
- LP DAAC MYD21A1D v061 (TES vs MxD11 split‑window): https://lpdaac.usgs.gov/products/myd21a1dv061/
- MODIS products in GEE guide: https://gee.geojamal.com/2025/04/exploring-modis-products-in-gee.html

**VIIRS**
- GEE VNP21A1D.002: https://developers.google.com/earth-engine/datasets/catalog/NASA_VIIRS_002_VNP21A1D
- Earthdata VNP21A1N v002: https://www.earthdata.nasa.gov/data/catalog/lpcloud-vnp21a1n-002
- LP DAAC VIIRS V2 LST&E release (TES, M14/M15/M16, 750 m): https://lpdaac.usgs.gov/news/lp-daac-releases-viirs-version-2-land-surface-temperature-and-3-band-emissivity-data-products/
- LP DAAC VNP21A2: https://lpdaac.usgs.gov/products/vnp21a2v001/
- NASA VIIRS LST&E products page: https://viirsland.gsfc.nasa.gov/Products/NASA/LSTESDR.html

**Sentinel‑3 SLSTR**
- SentiWiki SLSTR products: https://sentiwiki.copernicus.eu/web/slstr-products
- SentiWiki SLSTR processing: https://sentiwiki.copernicus.eu/web/slstr-processing
- SLSTR L2 LST ATBD (split‑window, biome/WVC/FVC): https://sentinels.copernicus.eu/documents/247904/349589/SLSTR_Level-2_LST_ATBD.pdf
- MS Planetary Computer Sentinel‑3 SLSTR LST: https://planetarycomputer.microsoft.com/dataset/sentinel-3-slstr-lst-l2-netcdf
- EUMETSAT SLSTR L1 data guide: https://user.eumetsat.int/resources/user-guides/sentinel-3-slstr-level-1-data-guide

**ASTER / emissivity**
- GEE ASTER GED AG100 v003: https://developers.google.com/earth-engine/datasets/catalog/NASA_ASTER_GED_AG100_003
- Earthdata ASTER L2 Surface Temperature (AST_08) v003: https://www.earthdata.nasa.gov/data/catalog/lpcloud-ast-08-003
- ASTER GED paper (Hulley 2015): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015GL065564
- ASTER vs ECOSTRESS GED comparison: https://www.sciencedirect.com/science/article/pii/S1569843223000493

**GOES‑R ABI**
- NCEI GOES‑R ABI L2 LST: https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01521
- NOAA STAR GOES‑R LST (split‑window B14/B15): https://www.star.nesdis.noaa.gov/goesr/product_land_lst.php
- AWS open‑data GOES‑16 README: https://github.com/awslabs/open-data-docs/blob/main/docs/noaa/noaa-goes16/README.md
- GOES‑R ABI instrument: https://www.goes-r.gov/spacesegment/abi.html

**Himawari AHI**
- NOAA STAR Himawari LST: https://www.star.nesdis.noaa.gov/smcd/emb/land/index.php?sat=HIMAWARI8&product=LST
- JAXA Himawari Monitor P‑Tree user guide: https://www.eorc.jaxa.jp/ptree/userguide.html
- Copernicus Global LST Himawari‑9→8 switch (2025): https://land.copernicus.eu/en/production-updates/global-land-surface-temperature-switch-from-himawari-9-to-himawari-8
- Himawari‑8 AHI 3‑band LST algorithm: https://www.researchgate.net/publication/321276588_An_Algorithm_for_Land_Surface_Temperature_Retrieval_Using_Three_Thermal_Infrared_Bands_of_Himawari-8

**Meteosat SEVIRI / Metop AVHRR (LSA SAF)**
- LSA SAF LST & Emissivity products (MLST LSA‑001/004, MLST‑AS LSA‑005, EDLST LSA‑002): http://lsa-saf.eumetsat.int/en/data/products/land-surface-temperature-and-emissivity/
- EUMETSAT upgraded MSG MLST: https://www.eumetsat.int/upgraded-lsa-saf-msg-land-surface-temperature-product-eumetcast-soon
- CEDA MSG SEVIRI LST (LSASAF v3.0): https://catalogue.ceda.ac.uk/uuid/29e8c659fdec4217b47399bc5c19dd54
- SEVIRI/MSG vs AVHRR/Metop validation (0.13°C / −0.32°C): https://www.sciencedirect.com/science/article/abs/pii/S0924271621000848

**INSAT‑3D / 3DR / 3DS (India)**
- MOSDAC INSAT‑3D payloads (TIR1 10.2–11.2, TIR2 11.5–12.5, 4 km): https://www.mosdac.gov.in/insat-3d-payloads
- MOSDAC INSAT‑3DR payloads: https://www.mosdac.gov.in/insat-3dr-payloads
- MOSDAC INSAT‑3DS payloads: https://www.mosdac.gov.in/insat-3s-payloads
- INSAT‑3D LST retrieval & assimilation (Singh 2016, JGR): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016JD024752
- eoPortal INSAT‑3DR: https://www.eoportal.org/satellite-missions/insat-3dr
- eoPortal INSAT‑3D/3DS: https://www.eoportal.org/satellite-missions/insat-3d

**FengYun**
- FY‑4A official LST inversion & validation: https://www.mdpi.com/2072-4292/15/9/2437
- FY‑4B nonlinear split‑window LST: https://www.researchgate.net/publication/377072053_A_Nonlinear_Split-Window_Algorithm_for_Retrieving_Land_Surface_Temperatures_from_Fengyun-4B_Thermal_Infrared_Data
- FY‑4B simultaneous LST+emissivity (TES‑like): https://ieeexplore.ieee.org/document/10282032
- FY‑4A LST reconstruction (spatio‑temporal attention): https://www.sciencedirect.com/science/article/pii/S156984322500127X

**Future missions**
- TRISHNA (CNES): https://cnes.fr/en/projects/trishna
- TRISHNA (eoPortal): https://www.eoportal.org/satellite-missions/trishna
- TRISHNA (ISRO): https://www.isro.gov.in/TRISHNA_Mission.html
- TRISHNA TIR instrument status (SPIE): https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13699/1369903/TRISHNA-TIR-instrument-development-status/10.1117/12.3071581.full
- LSTM (Sentinel Online): https://sentinels.copernicus.eu/web/sentinel/copernicus/lstm
- LSTM (eoPortal): https://www.eoportal.org/satellite-missions/lstm
- LSTM design/technology/status: https://www.researchgate.net/publication/374861762_The_Copernicus_land_surface_temperature_monitoring_LSTM_mission_design_technology_and_status
- SBG (NASA GSFC): https://science.gsfc.nasa.gov/solarsystem/projects/621/
- SBG‑TIR free‑flyer concept (60 m, 3‑day, 0.2 K, 12:30): https://ntrs.nasa.gov/citations/20230007027
- SBG (NASA Science): https://science.nasa.gov/earth-science/decadal-surveys/decadal-sbg/
- Landsat Next defined (NASA): https://science.nasa.gov/missions/landsat/ls-landsat-next-defined/
- Landsat Next (eoPortal): https://www.eoportal.org/satellite-missions/landsat-next
- Landsat Next technical specs (5 TIR bands, 60 m, 6‑day): https://landsatnext.com/technical-specs.html

**Fusion / downscaling / gap‑filling**
- TsHARP thermal sharpening assessment (VENµS/Sentinel‑2 NDVI): https://www.mdpi.com/2072-4292/13/6/1155
- Downscaling LST with random‑forest regression: https://www.sciencedirect.com/science/article/abs/pii/S0034425716300992
- RF downscaling considering spatial features: https://www.mdpi.com/2072-4292/13/18/3645
- Two‑step RF MODIS LST downscaling: https://www.mdpi.com/2073-4433/16/4/424
- Robust framework: downscaling + spatiotemporal fusion (TsHARP+STARFM, ASTER/MODIS/Landsat): https://www.researchgate.net/publication/371545683_A_Robust_Framework_for_Resolution_Enhancement_of_Land_Surface_Temperature_by_Combining_Spatial_Downscaling_and_Spatiotemporal_Fusion_Methods
- Geographically‑weighted ATP regression kriging (GWATPRK): https://www.mdpi.com/2072-4292/10/4/579/htm
- GWR kriging downscaling of ASTER thermal: https://www.mdpi.com/2072-4292/10/4/633
- Geographically‑neural‑network‑weighted RK (nighttime LST): https://www.mdpi.com/2072-4292/16/14/2542
- Attention‑based super‑resolution 100 m LST: https://www.sciencedirect.com/science/article/pii/S2666017225001415
- Worldwide continuous gap‑filled MODIS LST: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7933132/
- Multi‑model downscaling across agroclimatic zones: https://www.nature.com/articles/s41598-025-92135-0
- Five LST downscaling methods compared (plateau/mountain): https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2024.1488711/full
- High‑res global LST via coupled mechanism‑ML (physics‑informed): https://arxiv.org/pdf/2509.04991
- Gapless high spatio‑temporal LST by fusing satellite+model data: https://arxiv.org/pdf/2111.15636

---

### Appendix A — Algorithm primer (physics‑informed context)

- **Split‑Window (SW):** `LST = T_i + c1(T_i − T_j) + c2(T_i − T_j)^2 + c0 + (offset terms in ε, WVC)`, exploiting **differential water‑vapour absorption** between two adjacent ~11 & ~12 µm bands. Needs **two thermal bands** + emissivity. Used by MODIS MxD11, SLSTR, SEVIRI, AVHRR, GOES/Himawari/FY‑4, INSAT (TIR1/TIR2).
- **Temperature‑Emissivity Separation (TES):** uses **≥3 TIR bands** to *simultaneously* solve LST and per‑band emissivity (NEM → ratio → MMD empirical relation), with WVS atmospheric correction. More accurate emissivity over **bare/heterogeneous/urban**; used by ASTER, ECOSTRESS, MODIS MxD21, VIIRS VNP21, (future TRISHNA/SBG/Landsat Next).
- **Single‑Channel (SC):** one thermal band + **externally supplied emissivity** (ASTER GED) + **atmospheric profile** radiative‑transfer correction. Used by **Landsat C2 ST** (because Band 11 is unreliable for SW). Most sensitive to atmospheric‑profile error.
- **Emissivity matters because** `L = ε·B(LST) + (1−ε)·L_atm↓ + L_path`; a 0.01 emissivity error ≈ ~0.5–0.7 K LST error in the TIR window — so emissivity provenance (LUT vs retrieved vs climatology) is a first‑order cross‑validation axis.

### Appendix B — Known pitfalls checklist for the build
- Apply **scale/offset before** any °C conversion (Landsat 0.00341802/+149.0; MODIS/VIIRS ×0.02; ECOSTRESS already K).
- **Mask clouds** with each product's QA bitmask *before* compositing; cloud‑contaminated LST is cold‑biased.
- **Normalize overpass time** (use `*_view_time` / DTC model) before comparing Terra vs Aqua vs VIIRS vs SLSTR.
- **ECOSTRESS in GEE = LA only** → use LP DAAC/AppEEARS for India.
- **Sentinel‑3 SLSTR, INSAT, GOES, Himawari, SEVIRI, FY are NOT in GEE** → external ingest.
- For India geostationary, prefer **INSAT‑3D/3DR/3DS** and **Meteosat IODC** (not the 0° MSG) and **FY‑4** (east) — mind **high view‑zenith** degradation at disk edges.
- Landsat/ASTER **emissivity is a 2000–2008 climatology** → stale over fast‑urbanizing fringes (NDVI‑era anomaly); prefer TES products (MxD21/VNP21/ECOSTRESS) there.
- **Band 11 of Landsat: do not use for split‑window.**
- Watch **MxD21 high‑temp outliers** — filter with QC.
