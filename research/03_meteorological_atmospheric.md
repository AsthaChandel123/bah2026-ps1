# 03 — Meteorological, Atmospheric & Air-Quality Data Catalog
**ISRO Bharatiya Antariksh Hackathon 2026 — PS-1 (Urban Heat Hotspots, LST modelling, physics-informed cooling)**
**Research agent: R3 | Domain: Meteorology / Atmosphere / Air-Quality | Date: 2026-06-22**

> Scope: the *atmospheric forcing* layer — the air-temperature, humidity, wind, radiation, boundary-layer and aerosol/air-quality variables that drive **heat-stress indices** (e.g., Heat Index, UTCI, WBGT, Humidex) and the **surface-energy-balance (SEB)** physics used to model Land Surface Temperature (LST), urban heat hotspots and °C cooling potential. This catalog is *build-ready*: every entry gives exact dataset/asset IDs, exact band/variable names, spatial & temporal resolution, latency, access route, and the **physics term** it feeds.
>
> Companion files (other R-agents): `01` LST/thermal (MODIS/Landsat/ECOSTRESS), `02` land cover / built-up / NDVI / impervious, `04` elevation/terrain, `05` population/socio-economic. This file deliberately avoids re-deriving those and instead shows how they *couple* to the atmospheric layer (downscaling, QF estimation).

---

## 0. How this fits the PS-1 physics

The instantaneous **surface energy balance** at an urban facet/pixel is:

```
Q*  +  QF  =  QH  +  QE  +  ΔQS  (+ ΔQA)
```

| Term | Name | Sign convention | Primary atmospheric inputs from THIS catalog |
|------|------|-----------------|-----------------------------------------------|
| **Q\*** | Net all-wave radiation | input | K↓ (shortwave down), K↑ (=albedo·K↓), L↓ (longwave down), L↑ (=εσT_s⁴, from LST file 01) |
| **QF** | Anthropogenic heat flux | input | Population (GHSL), nighttime lights (VIIRS Black Marble), energy/traffic proxies |
| **QH** | Turbulent **sensible** heat | output | T_air (2 m), wind (u,v / speed), surface pressure, stability (PBL height), (T_s − T_air) |
| **QE** | Turbulent **latent** heat | output | Dewpoint/specific/relative humidity, vapour-pressure deficit, wind, available water/ET |
| **ΔQS** | Net storage heat flux | output | Net radiation + surface fabric (OHM coefficients from land cover, file 02) |
| **ΔQA** | Net advected heat | output | Wind speed/direction × horizontal T & humidity gradients |

**Heat-stress indices** additionally need only `{T_air, RH or dewpoint, wind, K↓ / mean-radiant-temperature}`. Mean Radiant Temperature (T_mrt), the dominant driver of outdoor thermal comfort, is reconstructed from K↓, L↓, L↑ + sky-view factor (SVF from DSM, file 04) → this is exactly why the **radiation bands below are first-class**, not optional.

**Design philosophy for PS-1 (fast, many sources, ≥30 cross-checks):** prefer **Google Earth Engine (GEE)** server-side O(1)-per-request reductions for everything that is a raster (reanalysis, satellite). Use **station APIs (CPCB/IMD/NASA POWER/Open-Meteo)** for ground truth, bias-correction and gap-filling. Never trust a single source; the *physics-term mapping* and *cross-correction* sections below make the redundancy explicit.

---

## 1. Master comparison table

Latencies and "data thru" dates are as observed in the GEE catalog on **2026-06-22** (ranges quoted from the live asset metadata where fetched).

| # | Dataset | GEE Asset ID / Access | Native res. | Temporal | Latency | Period (thru) | Role |
|---|---------|----------------------|-------------|----------|---------|----------------|------|
| **REANALYSIS / LDAS (rasters in GEE — primary forcing)** |
| 1 | **ERA5-Land Hourly** | `ECMWF/ERA5_LAND/HOURLY` | 0.1° (~11,132 m) | Hourly | ~2–3 months (CDS) / ~5 days (ERA5-Land-T preview) | 2026-06-16 | **Primary** T/Td/wind/radiation/flux forcing |
| 2 | ERA5-Land Daily Agg. | `ECMWF/ERA5_LAND/DAILY_AGGR` | 0.1° (~11.1 km) | Daily | ~months | recent | Daily means/min/max, fast climatology |
| 3 | ERA5-Land Monthly Agg. | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | 0.1° | Monthly | ~months | recent | Long-term normals, anomalies |
| 4 | ERA5 Hourly (atmos) | `ECMWF/ERA5/HOURLY` | 0.25° (~27.8 km) | Hourly | ~months | recent | Adds **boundary-layer height**, MSLP, TCWV, cloud, full-atmos vars not in ERA5-Land |
| 5 | ERA5 Daily Agg. | `ECMWF/ERA5/DAILY` (a.k.a. DAILY aggregates) | 0.25° | Daily | ~months | recent | Daily T/Td/wind/MSLP/precip; mean/min/max T2m |
| 6 | ERA5 Monthly Agg. | `ECMWF/ERA5/MONTHLY` | 0.25° | Monthly | ~months | recent | Climate normals |
| 7 | **GLDAS-2.1 NOAH 3-hourly** | `NASA/GLDAS/V021/NOAH/G025/T3H` | 0.25° (~27,830 m) | 3-hourly | ~1–1.5 months | 2026-06-17 | **Land-surface fluxes** (QH,QE,QG, net SW/LW) ready-made |
| 8 | GLDAS-2.2 CLSM (DA) | `NASA/GLDAS/V022/CLSM/G025/DA1D` | 0.25° | Daily | ~weeks | recent | Catchment LSM + GRACE DA (water-limited QE) |
| 9 | FLDAS (FEWS NET) | `NASA/FLDAS/NOAH01/C/GL/M/V001` | 0.1° | Monthly | ~1 month | recent | Africa/Asia famine-grade LSM forcing |
| 10 | MERRA-2 single-level | `NASA/GSFC/MERRA/slv/2` (M2T1NXSLV) | 0.5°×0.625° (~50 km) | Hourly | ~3 weeks | recent | T2M/T10M, winds, MSLP, TQV; aerosol companion `.../aer/2` |
| **GROUND STATIONS / GRIDDED-FROM-STATION (APIs — truth & bias-correction)** |
| 11 | **CPCB CAAQMS** (India) | `data.gov.in` OGD API + CPCB CCR portal | point (~500+ stns) | 15-min / hourly | near-real-time | live | **Ground met + pollutants** (T,RH,WS,WD,rain + PM/NO2/O3/CO/SO2) |
| 12 | **IMD 0.25° rainfall** | IMD Pune CMPG / `IMDLIB` | 0.25° | Daily | ~1 day–annual | 1901–present | Rain truth; QE/water-availability constraint |
| 13 | IMD 1.0° temperature | IMD Pune CMPG / `IMDLIB` | 1.0° (Tmax/Tmin) | Daily | annual | 1951–present | Independent station-gridded T (bias-check ERA5) |
| 14 | IMD AWS/ARG network | IMD / RAPID portals | point | hourly | NRT | live | Dense AWS T/RH/wind/rain (where accessible) |
| 15 | NOAA **ISD** / **GHCN-D/H** | NOAA NCEI; GHCN-D in GEE `NOAA/GHCND` | point (global) | sub-daily/daily | ~1 day | live | Airport/synoptic stations, global QC'd |
| 16 | **NASA POWER** | REST API `power.larc.nasa.gov` (NOT in GEE) | 0.5°×0.625° met / 1° solar | hourly/daily/monthly | ~2–3 days (solar), prior-yr+ (met) | live | Solar K↓ + T2M/RH/WS bundled, no-auth |
| 17 | **Open-Meteo** | REST API `open-meteo.com` (free, CC-BY) | ERA5 0.25° / ERA5-Land 0.1° + HRES | hourly | ~1–5 days | 1940–present | Fast gap-free reanalysis+forecast, no key |
| 18 | OpenWeatherMap | REST API (key, freemium) | model+station blend | hourly/current | NRT | live | Convenience current/forecast, urban points |
| **SATELLITE ATMOSPHERIC / AIR-QUALITY (rasters in GEE)** |
| 19 | **MODIS MAIAC AOD** | `MODIS/061/MCD19A2_GRANULES` | 1 km | Daily (Terra+Aqua) | ~1–7 days | live | Aerosol load → K↓ attenuation, PM proxy |
| 20 | **Sentinel-5P NO₂** | `COPERNICUS/S5P/NRTI/L3_NO2` / `.../OFFL/...` | ~1,113 m grid (L2 ~3.5×5.5 km) | ~daily | NRTI ~3 h / OFFL ~days | live | Combustion/traffic → QF & pollution co-mapping |
| 21 | S5P Aerosol Index | `COPERNICUS/S5P/NRTI/L3_AER_AI` | ~1 km grid | ~daily | NRTI ~3 h | live | UV-absorbing aerosol (dust/smoke) |
| 22 | S5P CO | `COPERNICUS/S5P/NRTI/L3_CO` | ~1 km grid | ~daily | ~3 h | live | Combustion tracer (QF proxy) |
| 23 | S5P SO₂ / HCHO / O₃ / CH₄ / AER_LH / CLOUD | `COPERNICUS/S5P/{NRTI,OFFL}/L3_{SO2,HCHO,O3,CH4,AER_LH,CLOUD}` | ~1 km grid | ~daily | ~3 h–days | live | Pollution drivers, cloud for K↓ |
| 24 | **NASA POWER solar** | API (see #16) | 1° solar | daily | ~2–3 days | live | ALLSKY/CLRSKY SW & LW down |
| 25 | **CAMS solar radiation** | Atmosphere Data Store (ADS) API | site time-series (≤4–5 km MSG) | 1-min–hourly | T-2 days (Europe/Africa/Asia in MSG disk) | 2004–present | High-quality K↓ GHI/DNI/DHI (clear+all sky) |
| 26 | SARAH-3 (CM SAF) | EUMETSAT CM SAF order/THREDDS | 0.05° (MSG disk) | 30-min/daily | ~days–months | 1983–present | Satellite SIS/SID/DNI radiation climate record |
| 27 | INSAT-3D/3DR sounder & imager | MOSDAC (ISRO) | ~4–10 km (IMG), sounder profiles | 30-min (full disk) / 15-min sector | NRT | live | **Indian geostationary** T/humidity profiles, OLR, INSOLATION product |
| 28 | AIRS (Aqua) | `NASA/AIRS/...` / GES DISC | ~45 km | 2×/day | ~days | live | Atmospheric T/humidity profiles, PBL proxy |
| 29 | CALIPSO (CALIOP) | ASDC/Earthdata (not GEE native) | ~333 m along-track | 16-day repeat | ~days | 2006–2023+ | Aerosol vertical profile / layer height |
| **ANTHROPOGENIC-HEAT PROXIES (rasters in GEE — feed QF)** |
| 30 | **VIIRS Black Marble** | `NASA/VIIRS/002/VNP46A2` | 500 m | Daily | ~10 days | 2026-06-09 | Nightlights → energy use → **QF** |
| 31 | VIIRS Black Marble monthly/annual | `NASA/VIIRS/002/VNP46A3` (monthly), `.../VNP46A4` (annual) | 500 m | Monthly/Annual | ~weeks | recent | Stable composites for QF baselines |
| 32 | **GHSL population** | `JRC/GHSL/P2023A/GHS_POP` | 100 m (also 1 km) | 5-yr epochs 1975–2030 | static | to 2030 proj. | Population density → **QF** (metabolic+building) |
| 33 | GHSL built-up surface/volume | `JRC/GHSL/P2023A/GHS_BUILT_S` (+ `_BUILT_V`) | 100 m | 5-yr epochs | static | to 2030 | Built volume → QF building-energy weighting |

> Notes: (a) **NASA POWER, Open-Meteo, CAMS, SARAH, INSAT/MOSDAC, CALIPSO are NOT in the GEE catalog** — they are pulled via their own REST/order APIs and (if needed) uploaded as assets or used station-wise. Treat them as the "out-of-GEE" tier. (b) S5P "grid" resolution in GEE is the harpconvert L3 grid (~0.01°≈1.11 km); the *true* sensor footprint is ~3.5×5.5 km at nadir (≥2019; pre-2019 ~7×3.5 km). (c) GLDAS native is 0.25°; GEE reports ~27,830 m. ERA5-Land native 0.1°; GEE reports 11,132 m.

---

## 2. Per-source details with EXACT band names

### 2.1 ERA5-Land Hourly — `ECMWF/ERA5_LAND/HOURLY`  ★PRIMARY
- **Res:** 0.1° (GEE: 11,132 m). **Temporal:** hourly. **Coverage:** 1950-01-01 → ~2026-06-16 (live metadata). **Latency:** CDS-quality ~2–3 months; ERA5-Land "T" preview ~5 days.
- **Exact bands (verified against GEE catalog page):**

| Physics need | Band ID | Units |
|--------------|---------|-------|
| 2 m air temperature | `temperature_2m` | K |
| 2 m dewpoint temperature | `dewpoint_temperature_2m` | K |
| 10 m U wind | `u_component_of_wind_10m` | m s⁻¹ |
| 10 m V wind | `v_component_of_wind_10m` | m s⁻¹ |
| Surface **net** shortwave | `surface_net_solar_radiation` | J m⁻² (accum.) |
| Surface **net** longwave | `surface_net_thermal_radiation` | J m⁻² |
| Surface shortwave **down** (K↓) | `surface_solar_radiation_downwards` | J m⁻² |
| Surface longwave **down** (L↓) | `surface_thermal_radiation_downwards` | J m⁻² |
| Surface pressure | `surface_pressure` | Pa |
| Total precipitation | `total_precipitation` | m |
| Sensible heat flux | `surface_sensible_heat_flux` | J m⁻² |
| Latent heat flux | `surface_latent_heat_flux` | J m⁻² |
| Skin temperature | `skin_temperature` | K |
| Soil temp level 1 | `soil_temperature_level_1` | K |
| Volumetric soil water L1 | `volumetric_soil_water_layer_1` | m³ m⁻³ |

- **Critical gotcha — accumulation:** radiation & flux bands are **accumulated from 00 UTC**. For an instantaneous hourly flux (W m⁻²) take the **difference between successive hours** and divide by 3600 s. ERA5-Land also exposes pre-disaggregated `*_hourly` variants in some products — verify per band. ERA5 sign: downward fluxes positive; net radiation positive downward.
- **Derived quantities (server-side in GEE):**
  - Wind speed `= sqrt(u² + v²)`; direction `= atan2(-u,-v)·180/π`.
  - RH from T & Td (Magnus): `e = 6.112·exp(17.67·(Td−273.15)/(Td−29.65))`, `es` same with T; `RH = 100·e/es`. VPD `= es − e`.
  - **Net radiation Q\*** ≈ `surface_net_solar_radiation + surface_net_thermal_radiation` (per-hour difference → W m⁻²).
- **Feeds:** Q\* (K↓,L↓, net SW/LW), QH (T_air, wind, pressure), QE (Td/RH, latent flux), advection (wind × gradients).

### 2.2 ERA5 Hourly (atmosphere) — `ECMWF/ERA5/HOURLY`
- **Res:** 0.25°. **Why also use it:** ERA5-Land has **no boundary-layer height**; ERA5 does. Key extra bands: `boundary_layer_height` (m) — turbulence/stability for QH and pollutant dispersion; `mean_sea_level_pressure`; `total_column_water_vapour`; `total_cloud_cover`; plus the same `*_2m`, `*_wind_10m`, radiation, `surface_sensible/latent_heat_flux`. Use ERA5 for BLH and full-atmosphere context, ERA5-Land for the finer surface state.

### 2.3 ERA5 Daily / Monthly aggregates
- `ECMWF/ERA5/DAILY` — 7 params daily: `mean_2m_air_temperature`, `minimum_2m_air_temperature`, `maximum_2m_air_temperature`, `dewpoint_2m_temperature`, `mean_sea_level_pressure`, `surface_pressure`, `u_component_of_wind_10m`, `v_component_of_wind_10m`, `total_precipitation`. Fast for heatwave/percentile climatology (e.g., Tmax 90th-pct hot days).
- `ECMWF/ERA5_LAND/DAILY_AGGR` & `..._MONTHLY_AGGR` — full 50-variable daily/monthly aggregation; flow bands (precip/flux) summed, state bands averaged. **`ECMWF/ERA5_LAND/DAILY_RAW` is deprecated** — use `DAILY_AGGR`.

### 2.4 GLDAS-2.1 NOAH 3-hourly — `NASA/GLDAS/V021/NOAH/G025/T3H`  ★FLUX-READY
- **Res:** 0.25° (GEE 27,830 m). **Temporal:** 3-hourly. **Coverage:** 2000-01-01 → ~2026-06-17. **Latency:** ~1–1.5 months.
- **Naming convention:** `_inst` = instantaneous; `_tavg` = mean over previous 3 h; `_f` = forcing input. **Big advantage vs ERA5: fluxes are already in W m⁻², no de-accumulation.**
- **Exact bands (verified):**

| Physics need | Band ID | Units |
|--------------|---------|-------|
| Air temperature | `Tair_f_inst` | K |
| Specific humidity | `Qair_f_inst` | kg kg⁻¹ (mass fraction) |
| Wind speed | `Wind_f_inst` | m s⁻¹ |
| Shortwave down (K↓) | `SWdown_f_tavg` | W m⁻² |
| Longwave down (L↓) | `LWdown_f_tavg` | W m⁻² |
| Surface pressure | `Psurf_f_inst` | Pa |
| **Net** shortwave | `Swnet_tavg` | W m⁻² |
| **Net** longwave | `Lwnet_tavg` | W m⁻² |
| **Sensible** heat flux QH | `Qh_tavg` | W m⁻² |
| **Latent** heat flux QE | `Qle_tavg` | W m⁻² |
| **Ground/storage** heat flux QG | `Qg_tavg` | W m⁻² |
| Surface skin temperature | `AvgSurfT_inst` | K |
| Rainfall rate | `Rainf_f_tavg` | kg m⁻² s⁻¹ |
| Total ET | `Evap_tavg` | kg m⁻² s⁻¹ |
| Net radiation (derive) | `Swnet_tavg + Lwnet_tavg` | W m⁻² |

- **Why it matters for PS-1:** GLDAS gives **observation-consistent partitioning of Q\*** into QH/QE/QG already. Use it to (a) sanity-check your own SEB closure, (b) provide Bowen-ratio (`Qh/Qle`) priors per land-cover class, and (c) bias-correct ERA5-derived fluxes. Specific humidity → RH via `RH = (q·P)/((0.378·q+0.622)·es)`.

### 2.5 GLDAS-2.2 CLSM DA — `NASA/GLDAS/V022/CLSM/G025/DA1D`
- Daily, 0.25°. Catchment LSM with **GRACE terrain-water assimilation** → better water-limited QE in monsoon-dry transitions. Bands include `Evap_tavg`, `Qsm_acc`, `TWS_inst`, `GWS_inst`, soil-moisture columns. Use as a second-opinion ET/soil-water source.

### 2.6 FLDAS — `NASA/FLDAS/NOAH01/C/GL/M/V001`
- Monthly, 0.1°, Noah LSM tuned for food-security regions incl. South Asia. Bands mirror GLDAS (`Tair_f_tavg`, `Qair_f_tavg`, `SWdown_f_tavg`, `Qh_tavg`, `Qle_tavg`, `Evap_tavg`, `SoilMoi*`). Good monthly cross-check at finer 0.1° than GLDAS.

### 2.7 MERRA-2 single-level — `NASA/GSFC/MERRA/slv/2` (M2T1NXSLV)
- **Res:** 0.5°×0.625°. **Temporal:** hourly. Bands: `T2M`, `T10M`, `QV2M`/`QV10M` (specific humidity), `U2M`,`V2M`,`U10M`,`V10M`,`U50M`,`V50M`, `SLP`, `PS`, `TQV` (total precipitable water). For **PBL height** use the companion **MERRA-2 flux collection** (`M2T1NXFLX`, band `PBLH`) and for **aerosols** the MERRA-2 aerosol collection (`NASA/GSFC/MERRA/aer/2`: `DUSMASS`, `BCSMASS`, `OCSMASS`, `SO4SMASS`, `SSSMASS`, `TOTEXTTAU` AOD). MERRA-2 AOD is a *gap-free* aerosol alternative to MAIAC for K↓ correction.
- **Feeds:** independent reanalysis #2 for T/humidity/wind (verify ERA5), aerosol-extinction for radiation, PBLH for stability.

### 2.8 CPCB CAAQMS (India ground network) — `data.gov.in` OGD + CPCB CCR  ★GROUND TRUTH
- **What:** Continuous Ambient Air Quality Monitoring Stations under NAMP; **500+ real-time CAAQMS** (within ~800+ NAMP/manual stations) across 340+ cities. Each CAAQMS transmits **15-min averaged** data.
- **Met variables exposed per station:** Temperature (Temp/AT), Relative Humidity (RH), Wind Speed (WS), Wind Direction (WD), Solar Radiation (SR, at many sites), Barometric Pressure (BP), Rainfall (RF). **Pollutants:** PM2.5, PM10, NO₂, NOx, SO₂, O₃, CO, NH₃, Benzene/Toluene, occasionally O₃-precursors.
- **Why gold for PS-1:** these are **in-canopy urban met** measurements (≈2–10 m, inside the city) — exactly the urban-scale T_air ground truth that coarse reanalysis lacks. Dense in metros (Delhi ~40, Mumbai/Bengaluru/Chennai/Kolkata/Hyderabad/Pune tens each).
- **Access — two routes (see §6 for how-to):** (1) **OGD `data.gov.in`** "Real-time Air Quality Index" resource → register, get API key, `GET` JSON/XML of latest station readings (pollutants + AQI; met where published). (2) **CPCB CCR** (`app.cpcbccr.com` / `airquality.cpcb.gov.in`) → station-wise historical met+pollutant download (CAPTCHA/portal; some endpoints reverse-engineered by community libs).
- **Feeds:** ground T_air/RH/WS/WD for **bias-correcting reanalysis**, validating downscaled LST→T_air, AND pollutant fields for co-mapping heat × air-quality vulnerability.

### 2.9 IMD gridded & AWS (India) — IMD Pune CMPG / `IMDLIB`
- **IMD 0.25°×0.25° daily rainfall** (Pai et al. 2014; 6,955+ stations; **1901–present**, NetCDF/binary). **IMD 1.0°×1.0° daily Tmax/Tmin** (1951–present). Download via IMD Pune *Climate Prediction Group* gridded-data page or the **`IMDLIB`** Python library (programmatic retrieval + xarray export, clip to AOI/admin unit).
- **IMD AWS/ARG network:** thousands of Automatic Weather/Rain-gauge Stations (hourly T/RH/wind/pressure/rain) — accessible via IMD/RAPID/MAUSAM portals (access tiered).
- **Feeds:** independent **station-gridded** rainfall (QE/water-availability gate) and temperature (bias-check vs ERA5/GLDAS at 0.25°/1°). IMD rainfall is the authoritative monsoon constraint for India.

### 2.10 NOAA ISD / GHCN — `NOAA/GHCND` (daily) + NCEI ISD (sub-daily)
- **GHCN-Daily** in GEE: `NOAA/GHCND` (point FeatureCollection; `TMAX`,`TMIN`,`TAVG`,`PRCP` in tenths). **ISD** (Integrated Surface Database): global hourly/3-hourly synoptic+METAR (temp, dewpoint, wind, pressure, visibility) via NCEI. Use airport stations for QC'd long records and to anchor reanalysis bias.

### 2.11 NASA POWER — REST API (`power.larc.nasa.gov`)  ★SOLAR+MET BUNDLE (not in GEE)
- **Res:** met 0.5°×0.625° (MERRA-2), **solar 1°×1°** (CERES SYN1deg / SRB / FLASHFlux). **Temporal:** hourly/daily/monthly/climatology. **Latency:** solar ~2–3 days behind real-time; met up to prior year for some params. **No authentication.**
- **Exact parameter names (API `parameters=`):**

| Need | POWER parameter | Units |
|------|-----------------|-------|
| 2 m air temp | `T2M` (+ `T2M_MAX`,`T2M_MIN`) | °C |
| 2 m dewpoint | `T2MDEW` | °C |
| 2 m wet-bulb | `T2MWET` | °C |
| 2 m relative humidity | `RH2M` | % |
| 2 m specific humidity | `QV2M` | g kg⁻¹ |
| Wind speed 2 m / 10 m | `WS2M` / `WS10M` (+ `WD10M`) | m s⁻¹ / ° |
| **All-sky SW down (K↓ GHI)** | `ALLSKY_SFC_SW_DWN` | kWh m⁻² day⁻¹ (or W m⁻² hourly) |
| Clear-sky SW down | `CLRSKY_SFC_SW_DWN` | kWh m⁻² day⁻¹ |
| Direct normal / diffuse | `ALLSKY_SFC_SW_DNI` / `ALLSKY_SFC_SW_DIFF` | — |
| **Longwave down (L↓)** | `ALLSKY_SFC_LW_DWN` | W m⁻² |
| Surface pressure | `PS` | kPa |
| Precipitation | `PRECTOTCORR` | mm day⁻¹ |
| UV index / erythemal | `ALLSKY_SFC_UV_INDEX` | — |

- **Example call:** `https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,RH2M,WS2M,ALLSKY_SFC_SW_DWN,ALLSKY_SFC_LW_DWN&community=RE&longitude=77.21&latitude=28.61&start=20250101&end=20251231&format=JSON`
- **Feeds:** Q\* (K↓ and L↓ in one place), plus a fully independent T/RH/wind triad — ideal *third* reanalysis cross-check and the simplest radiation source if CDS/GLDAS unavailable.

### 2.12 Open-Meteo — REST API (`open-meteo.com`, free CC-BY, no key)  ★FAST GAP-FREE
- Serves **ERA5 (0.25°) + ERA5-Land (0.1°)** historical reanalysis (1940–present) AND HRES forecast/historical-forecast, hourly. Variables: `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `apparent_temperature`, `wind_speed_10m`, `wind_direction_10m`, `surface_pressure`, `shortwave_radiation`, `direct_radiation`, `diffuse_radiation`, `terrestrial_radiation`, `precipitation`, `cloud_cover`, `boundary_layer_height`, and ready-made indices (`apparent_temperature`). **JSON over HTTP GET, no auth, gap-free.**
- **Why for PS-1:** fastest way to pull a *consistent hourly time series at an exact lat/lon* for validation points without GEE export; great for rapid prototyping of heat-index time series and for filling station gaps. (Note: it *is* ERA5 underneath, so not independent of ERA5 for bias purposes — use POWER/MERRA-2/CPCB for true independence.)

### 2.13 MODIS MAIAC AOD — `MODIS/061/MCD19A2_GRANULES`
- **Res:** **1 km** (rare for AOD). **Temporal:** daily (Terra+Aqua combined). **Exact bands:** `Optical_Depth_047` (AOD @ 0.47 µm), `Optical_Depth_055` (AOD @ 0.55 µm), `AOD_Uncertainty`, `FineModeFraction`, `Column_WV` (column water vapour, cm), `Injection_Height` (smoke, m), `AOD_QA`, `cosSZA`,`cosVZA`,`RelAZ`,`Scattering_Angle`,`Glint_Angle`. Apply scale factor 0.001 to AOD bands; filter with `AOD_QA` bits.
- **Feeds:** (i) **K↓ attenuation** — aerosols reduce surface shortwave (cuts daytime Q\*, an oft-missed physics term in hazy Indian cities); (ii) **PM2.5 surrogate** for the air-quality × heat vulnerability overlay (AOD–PM regression with CPCB ground PM).

### 2.14 Sentinel-5P TROPOMI — `COPERNICUS/S5P/{NRTI,OFFL}/L3_*`
- **Grid in GEE:** ~0.01° (~1,113 m) harpconvert L3; true footprint ~3.5×5.5 km (≥Aug-2019). **Versions:** **NRTI** (~3 h latency, smaller swath) vs **OFFL** (days latency, fuller coverage). **CH₄ has no NRTI.**
- **Key product → band:**

| Product | Asset suffix | Primary band | Units |
|---------|--------------|--------------|-------|
| NO₂ | `L3_NO2` | `tropospheric_NO2_column_number_density` (also `NO2_column_number_density`) | mol m⁻² |
| Aerosol Index | `L3_AER_AI` | `absorbing_aerosol_index` | dimensionless |
| CO | `L3_CO` | `CO_column_number_density` | mol m⁻² |
| SO₂ | `L3_SO2` | `SO2_column_number_density` | mol m⁻² |
| HCHO | `L3_HCHO` | `tropospheric_HCHO_column_number_density` | mol m⁻² |
| O₃ | `L3_O3` | `O3_column_number_density` | mol m⁻² |
| CH₄ | `OFFL/L3_CH4` | `CH4_column_volume_mixing_ratio_dry_air` | ppb |
| Aerosol layer height | `L3_AER_LH` | `aerosol_height` | m |
| Cloud | `L3_CLOUD` | `cloud_fraction`, `cloud_top_pressure` | — |

- **Feeds:** NO₂/CO = **combustion intensity → QF proxy & traffic-corridor heat** (validate VIIRS nightlights). AER_AI/AER_LH = aerosol context for K↓. Cloud fraction = K↓ gating.

### 2.15 CAMS solar radiation — Atmosphere Data Store (ADS) API
- **CAMS Radiation Time-Series** (`cams-solar-radiation-timeseries`): site-specific **GHI/DNI/DHI**, clear-sky & all-sky, **1-min to hourly**, **2004 → T-2 days**, derived from MSG/MFG geostationary (covers Europe, Africa, **Middle East/most of Asia within the Meteosat disk; India is within coverage**). Resolution effectively ~3–5 km satellite. Requires free ADS account + API key (`ads.atmosphere.copernicus.eu`).
- **Feeds:** the **most accurate operational K↓** for SEB/T_mrt where in the MSG disk; superior to 1° NASA POWER for urban radiation.

### 2.16 SARAH-3 (EUMETSAT CM SAF)
- Climate Data Record of surface solar radiation from MFG/MSG: **SIS** (surface incoming shortwave), **SID** (direct), **DNI**, **SAL** (albedo), **0.05°**, 30-min/daily/monthly, **1983–present**. Order via CM SAF Web User Interface / THREDDS. Use as a *long-term radiation climatology* and gap-filler for CAMS.

### 2.17 INSAT-3D / 3DR (ISRO geostationary) — MOSDAC
- **Indian** geostationary sounder + imager. Products via **MOSDAC** (`mosdac.gov.in`): **INSOLATION** (surface incoming solar, ~hourly, ~4–8 km), **OLR** (outgoing longwave), **TPW** (total precipitable water), **sounder vertical T & humidity profiles**, LST, cloud. 30-min full-disk / faster sector scans.
- **Feeds:** indigenous K↓ (INSOLATION) and L-out context at high temporal cadence over India — fills the daytime diurnal cycle between MODIS overpasses; strong "Indian-source" credibility for an ISRO PS.

### 2.18 AIRS (Aqua) — GES DISC / `NASA/AIRS/*`
- ~45 km, 2×/day. Vertical **air-temperature & humidity profiles**, surface T, OLR; PBL proxies derivable. Use for free-atmosphere lapse-rate context and as an independent humidity profile source.

### 2.19 CALIPSO (CALIOP) — Earthdata ASDC
- Lidar **aerosol & cloud vertical profiles**, ~333 m along-track, 16-day repeat (2006–2023+). Not gridded/GEE-native. Use sparsely for **aerosol layer-height validation** of S5P AER_LH / MERRA-2.

### 2.20 VIIRS Black Marble — `NASA/VIIRS/002/VNP46A2`  ★QF DRIVER
- **Res:** 500 m. **Temporal:** daily. **Coverage:** 2012-01-19 → ~2026-06-09. **Exact bands:** `Gap_Filled_DNB_BRDF_Corrected_NTL` (preferred, moonlight+atmos+gap-filled), `DNB_BRDF_Corrected_NTL`, `Mandatory_Quality_Flag`, `Latest_High_Quality_Retrieval`, `QF_Cloud_Mask`, `Snow_Flag`, `DNB_Lunar_Irradiance` (radiance nW cm⁻² sr⁻¹, scale 0.1). Monthly/annual stable composites: `NASA/VIIRS/002/VNP46A3` / `VNP46A4`.
- **Feeds QF:** nightlight radiance is a robust **proxy for energy consumption** → anthropogenic heat. See §5.

### 2.21 GHSL population & built-up — `JRC/GHSL/P2023A/GHS_POP`, `GHS_BUILT_S/V`
- **GHS_POP** 100 m (and 1 km), residents per cell, epochs 1975–2030 (5-yr). **GHS_BUILT_S** built-up surface fraction, **GHS_BUILT_V** built volume. Static (per epoch). **Feeds QF** (population metabolic + building-energy weighting) and the storage-flux/morphology coupling (file 02/04).

---

## 3. Physics-term mapping table (which band → which energy-balance term)

| Physics term | Definition | Best source(s) & EXACT bands | How combined |
|--------------|-----------|------------------------------|--------------|
| **K↓** (SW down) | Incoming shortwave | ERA5-Land `surface_solar_radiation_downwards` (de-accum→W m⁻²); GLDAS `SWdown_f_tavg`; CAMS GHI; NASA POWER `ALLSKY_SFC_SW_DWN`; INSAT INSOLATION; SARAH `SIS` | Pick best-available; AOD-attenuate with MAIAC `Optical_Depth_055`; cloud-gate with S5P/ERA5 `total_cloud_cover` |
| **K↑** (SW up) | Reflected shortwave | `K↓ × albedo` (albedo from MODIS `MCD43A3` / Sentinel-2, file 01/02) | Per-pixel albedo map |
| **L↓** (LW down) | Atmospheric longwave | ERA5-Land `surface_thermal_radiation_downwards`; GLDAS `LWdown_f_tavg`; NASA POWER `ALLSKY_SFC_LW_DWN` | Use directly; or parametrize from T_air, e, clouds (Prata/Brutsaert) |
| **L↑** (LW up) | Surface emission | `ε σ T_s⁴` with T_s = **LST** (file 01) + emissivity (MOD11/ASTER GED) | Couples LST file to atmosphere |
| **Q\*** | Net all-wave | ERA5-Land `surface_net_solar_radiation + surface_net_thermal_radiation`; GLDAS `Swnet_tavg + Lwnet_tavg`; or `(K↓−K↑)+(L↓−L↑)` | Two independent routes → cross-check |
| **QH** | Sensible heat | Drivers: ERA5-Land `temperature_2m`, `u/v_component_of_wind_10m`, `surface_pressure`; GLDAS ready: `Qh_tavg`; stability: ERA5 `boundary_layer_height` | Aerodynamic/resistance: `QH=ρcp(T_s−T_air)/r_ah`; benchmark vs GLDAS `Qh_tavg` |
| **QE** | Latent heat | Drivers: ERA5-Land `dewpoint_temperature_2m`→RH/VPD, wind; water: GLDAS `Evap_tavg`, soil moisture, IMD rain; GLDAS ready: `Qle_tavg` | Penman-Monteith / Bowen; cap by water availability |
| **QG / ΔQS** | Ground/storage | GLDAS `Qg_tavg`; or OHM `ΔQS=Σ(a1,a2,a3)·Q*` with land-cover coefficients (file 02) | Net radiation × fabric coeffs |
| **QF** | Anthropogenic | VIIRS `Gap_Filled_DNB_BRDF_Corrected_NTL` + GHSL `GHS_POP`/`GHS_BUILT_V`; tracers S5P `NO2`,`CO` | LST-DCM / LUCY / population-energy model (§5) |
| **ΔQA** | Advection | ERA5-Land wind (`u/v_10m`) × ∇(T_air, q) | Finite-difference horizontal gradients × wind |
| **Stability / mixing** | Turbulent regime | ERA5 `boundary_layer_height`; MERRA-2 `M2T1NXFLX:PBLH`; AIRS profiles | Modulates r_ah, pollutant & heat dispersion |
| **Humidity (comfort)** | For HI/UTCI/WBGT | ERA5-Land `dewpoint_temperature_2m`; CPCB RH; POWER `RH2M`,`T2MWET`; MERRA-2 `QV2M` | Compute Heat Index, Humidex, WBGT, UTCI |
| **Aerosol radiative** | K↓ reduction | MAIAC `Optical_Depth_055`; S5P `absorbing_aerosol_index`; MERRA-2 `TOTEXTTAU` | Beer-Lambert attenuation of clear-sky K↓ |

---

## 4. Reanalysis → urban-scale downscaling (the ~9–25 km → ~100 m problem)

Reanalysis air temperature is **smooth at 11–28 km** and *misses the urban heat island entirely* (a single ERA5-Land pixel can blanket a whole metropolis). PS-1 needs **neighbourhood-scale (~30–100 m) air temperature**. Three complementary, cross-verifying methods:

### 4.1 Statistical downscaling — "anomaly + regression" (LST as the spatial template)
1. Take coarse ERA5-Land `temperature_2m` (T_coarse, 11 km) as the **regional background**.
2. Build a **high-resolution predictor stack** at 30–100 m: LST (file 01, MODIS 1 km / Landsat 100 m / ECOSTRESS 70 m), NDVI, NDBI/impervious fraction (file 02), albedo, elevation (SRTM/Copernicus DEM, file 04), sky-view factor, distance-to-water, building density/height.
3. **Regression / ML** (random forest, gradient boosting, or physics-guided GAM) mapping predictors → *observed* T_air at CPCB/IMD/AWS stations: `T_air = f(LST, NDVI, impervious, elev, SVF, hour, …)`.
4. **Anomaly preservation:** predict the **fine-scale anomaly** `ΔT = T_air − T_coarse` from predictors, then `T_air(fine) = T_coarse(bilinear) + ΔT(fine)`. This guarantees the downscaled field collapses back to the trusted reanalysis mean while injecting urban structure — the standard "delta/anomaly downscaling" used for urban climate.

### 4.2 LST→T_air transfer ("Statistical/TsHARP-style" + lapse correction)
- Day-time **T_air ≈ a + b·LST** is biased (LST ≫ T_air over dry impervious surfaces), so condition on land cover and add **elevation lapse-rate** correction (`−6.5 °C km⁻¹` standard; use local from station network). Night-time LST tracks T_air more closely — split day/night models. This is the urban-air-temperature-mapping literature (e.g., satellite-derived screen-level temperature).

### 4.3 Land-Use Regression (LUR) for the air-quality/heat overlay
- Classic for pollutants: regress CPCB station PM/NO₂ on **buffered land-use predictors** (road density, traffic, built fraction, population, NDVI, distance-to-source) → continuous concentration surface. The *same LUR machinery* applies to **T_air and humidity**, giving an independent estimate to cross-check the LST-regression downscaling. S5P NO₂ and VIIRS lights become predictors here.

### 4.4 Physics-informed / dynamical refinement (optional, heavier)
- Where compute allows, an **Urban Energy Balance** (SUEWS/TEB-style) or LES/microscale model takes the downscaled forcing + morphology to *physically* resolve QF, ΔQS and intra-canyon T. For PS-1's "fastest O(1)" priority, keep this as a **per-AOI offline calibration** that yields transfer coefficients applied server-side in GEE, not a per-request solve.

**Validation of any downscaling:** hold out CPCB/IMD stations (spatial k-fold), report RMSE/MAE/bias of downscaled T_air vs station; expect 1–2.5 °C RMSE for good urban models.

---

## 5. Anthropogenic heat flux (QF) from population & nightlights

QF is the term that makes **urban** energy balance differ from rural — essential for "quantify drivers" and "cooling" objectives. Three increasingly physical tiers:

1. **Top-down per-capita scaling (fast baseline):** `QF = AHE_percapita · PopDensity / Area`. Use **GHSL `GHS_POP`** (100 m). National/city annual energy + traffic + metabolism budget ÷ population → per-capita W; distribute by population grid. Add a **metabolic floor** (~75–100 W person⁻¹ awake).
2. **Nightlight-weighted disaggregation (spatial detail):** redistribute the city energy/QF total by **VIIRS `Gap_Filled_DNB_BRDF_Corrected_NTL`** radiance (proxy for commercial/industrial/residential energy use), optionally weighted by **`GHS_BUILT_V`** (built volume) and S5P `NO2`/`CO` (combustion/traffic). Global QF datasets (e.g., Dong/Chen nightlight-based, or LUCY/AH4GUC inventories) are built exactly this way and can be ingested as priors.
3. **Diurnal & sectoral profiles:** apply weekday/weekend × hourly load curves (separating building cooling load — itself temperature-dependent, a **heat-amplifying feedback** in summer — traffic peaks, and industry). Tie building-cooling QF to the very T_air being modelled to capture the AC-driven positive feedback explicitly.

**QF magnitudes (sanity):** rural ~0–5, suburban ~10–40, dense Indian metro cores can reach **50–200+ W m⁻²** in summer afternoons — comparable to a meaningful fraction of Q\*, hence non-negligible for °C-reduction accounting.

**Feeds:** QF enters the LHS of the SEB; its spatial pattern (lights + pop + traffic NO₂) is a primary **"driver"** layer to attribute hotspot intensity, and reducing it (efficiency, traffic, waste-heat capture) is one cooling lever alongside albedo/greening.

---

## 6. Station ↔ reanalysis cross-correction & gap-filling (robustness theme)

This is the heart of PS-1's "≥30 cross-verifying methods, fill each other's gaps." Two directions:

### 6.1 Stations correct reanalysis (bias-correction / debiasing)
- **Problem:** ERA5/GLDAS/MERRA-2 have systematic biases over India (warm/cool, dry, urban-blind).
- **Method (per variable, per station-neighbourhood):**
  1. Co-locate reanalysis pixel value with CPCB/IMD/ISD station obs over an overlapping period.
  2. Compute bias / **quantile mapping (QM)** or **CDF-matching** transfer functions (handles distribution, not just mean — important for Tmax/heatwave tails).
  3. Apply the transfer function to the full reanalysis field (interpolating correction coefficients spatially, e.g., by kriging/IDW or as another RF predictor).
  4. Result: a **bias-corrected reanalysis** that respects station truth where available and degrades gracefully elsewhere.
- **Multi-source agreement = confidence:** stack ERA5-Land, GLDAS, MERRA-2, NASA POWER, IMD at each point; their **spread is an uncertainty map**. Where they agree → high confidence; where they diverge → flag and down-weight. (Open-Meteo is *not* independent — it's ERA5 — so exclude it from the independence count.)

### 6.2 Reanalysis gap-fills stations (sparse-network completion)
- **Problem:** CPCB/IMD stations are uneven (dense in metros, sparse in peri-urban/rural); records have outages.
- **Method:**
  1. For a station gap (time or space), use the **bias-corrected reanalysis** as the background.
  2. **Kriging-with-external-drift / regression-kriging:** interpolate station residuals (obs − reanalysis) over the domain, add to reanalysis → spatially complete, station-anchored field.
  3. For temporal gaps: fill from co-located reanalysis (or Open-Meteo time series) preserving the station's mean/variance via QM.
  4. **ML imputation:** RF/XGBoost or graph-based virtual sensing using neighbouring stations + reanalysis + LST/landcover predictors.
- **Outcome:** every grid cell has a **best-estimate T_air/RH/wind/radiation** with an associated uncertainty, derived from the *consensus* of station + multiple reanalyses + satellite, exactly the gap-filling robustness PS-1 rewards.

### 6.3 Cross-verification matrix (who checks whom)
| Variable | Source A | Source B | Source C (independent) | Station truth |
|----------|----------|----------|------------------------|---------------|
| T_air 2 m | ERA5-Land | GLDAS | MERRA-2 / NASA POWER | CPCB, IMD 1°, ISD/GHCN |
| Humidity | ERA5-Land Td | GLDAS q | NASA POWER RH2M | CPCB RH |
| Wind | ERA5-Land u/v | GLDAS Wind_f | MERRA-2 10 m | CPCB WS/WD, ISD |
| K↓ (SW) | ERA5-Land | GLDAS SWdown | NASA POWER / CAMS / INSAT / SARAH | CPCB SR (where present) |
| Rainfall | ERA5-Land tp | GLDAS Rainf | IMD 0.25° (authoritative) | CPCB RF gauges |
| Aerosol | MAIAC AOD | MERRA-2 TOTEXTTAU | S5P AER_AI | CPCB PM (AOD↔PM) |
| PBL height | ERA5 BLH | MERRA-2 PBLH | AIRS profile | radiosonde (IMD) |

---

## 7. CPCB & IMD access — practical how-to

### 7.1 CPCB real-time air quality + met
**Route A — OGD `data.gov.in` (recommended for automation):**
1. Create account at https://www.data.gov.in and generate an **API key** (profile → "My Account" → API key).
2. Find the resource **"Real time Air Quality Index from various monitoring stations"** (CPCB). It exposes the latest station readings (station, city, state, lat/lon, pollutant, value, last-update) as JSON/XML/CSV.
3. Call: `https://api.data.gov.in/resource/<RESOURCE_ID>?api-key=<KEY>&format=json&limit=10000` → parse per-station pollutants/AQI. (Met fields appear where the station publishes them; pollutant coverage is universal.)
4. Poll hourly; persist to your own store to build the historical series you cross-correct against.

**Route B — CPCB CCR portal (historical + full met):**
- https://app.cpcbccr.com / https://airquality.cpcb.gov.in → station-wise download of **met (Temp, RH, WS, WD, SR, BP, RF) + pollutants**, 15-min/hourly, with date-range selection (portal UI; CAPTCHA-gated). Community Python clients exist that wrap the underlying endpoints for batch pulls — verify ToS.
- **Station list & metadata** (names, coordinates, city) are published on the CCR/CPCB site; use to map each CAAQMS to your grid for bias-correction.

### 7.2 IMD gridded + AWS
**Gridded (rainfall 0.25°, temperature 1°):**
- **`IMDLIB`** (Python): `pip install imdlib`; `imdlib.get_data('rain', start_yr, end_yr, fn_format='yearwise')` (also `'tmax'`,`'tmin'`); then `.open_data()` → xarray; export NetCDF and clip to AOI. Cleanest programmatic route.
- **Manual:** IMD Pune Climate Prediction (CMPG) gridded-data pages — download yearly NetCDF/binary for **0.25° rainfall** (1901–present) and **1° Tmax/Tmin** (1951–present).
**AWS / station / current weather:**
- IMD MAUSAM / RAPID / Synergie portals provide AWS & synoptic data (tiered access; some require institutional request). For automated current obs prefer NOAA ISD (Indian airport WMO stations) which is open.

### 7.3 Quick GEE skeleton (server-side, O(1)-style reduction)
```javascript
// Mean midday summer 2 m air temp over an AOI from ERA5-Land, °C
var aoi = ee.Geometry.Rectangle([77.0,28.4,77.4,28.8]); // e.g., Delhi box
var t2m = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
  .filterDate('2025-04-01','2025-07-01')
  .filter(ee.Filter.calendarRange(8,11,'hour'))      // ~midday IST window
  .select('temperature_2m')
  .mean().subtract(273.15);
// Net radiation from de-accumulated hourly net SW+LW (illustrative)
var era = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
  .filterDate('2025-05-01','2025-05-02')
  .select(['surface_net_solar_radiation','surface_net_thermal_radiation']);
// GLDAS ready-made fluxes (already W/m^2)
var qh = ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H')
  .filterDate('2025-05-01','2025-06-01').select('Qh_tavg').mean();
```

---

## 8. Recommended build stack for PS-1 (atmospheric layer)

- **Core forcing (GEE, server-side):** `ECMWF/ERA5_LAND/HOURLY` (state + radiation), `NASA/GLDAS/V021/NOAH/G025/T3H` (ready fluxes + Bowen priors), `ECMWF/ERA5/HOURLY` (BLH/stability).
- **Radiation best-available cascade:** CAMS/INSAT → NASA POWER → GLDAS `SWdown_f_tavg` → ERA5-Land, AOD-attenuated by MAIAC.
- **Ground truth & bias-correction:** CPCB CAAQMS (urban in-canopy T/RH/wind) + IMD 0.25° rain + ISD/GHCN; quantile-map reanalysis to these.
- **Downscaling to ~100 m:** anomaly + RF regression on LST/NDVI/impervious/DEM/SVF (files 01/02/04); validate on held-out stations.
- **QF:** VIIRS Black Marble (`VNP46A2`) × GHSL `GHS_POP`/`GHS_BUILT_V`, traffic via S5P NO₂/CO.
- **Air-quality × heat overlay:** MAIAC AOD + S5P NO₂ + CPCB PM (LUR).
- **Independence for cross-checks (≥3 truly independent reanalyses):** ERA5-family · GLDAS · MERRA-2 · NASA POWER · IMD/CPCB stations. (Open-Meteo = ERA5; use for speed, not independence.)

---

## 9. References (with URLs)

**Reanalysis / LDAS (GEE catalog)**
- ERA5-Land Hourly — https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY
- ERA5-Land Daily Aggregated — https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR
- ERA5-Land Monthly Aggregated — https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR
- ERA5 Hourly — https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_HOURLY
- ERA5 Daily Aggregates — https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_DAILY
- ERA5 Monthly — https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_MONTHLY
- GLDAS-2.1 NOAH 3H — https://developers.google.com/earth-engine/datasets/catalog/NASA_GLDAS_V021_NOAH_G025_T3H
- GLDAS-2.2 CLSM — https://developers.google.com/earth-engine/datasets/catalog/NASA_GLDAS_V022_CLSM_G025_DA1D
- FLDAS — https://developers.google.com/earth-engine/datasets/catalog/NASA_FLDAS_NOAH01_C_GL_M_V001
- MERRA-2 M2T1NXSLV — https://developers.google.com/earth-engine/datasets/catalog/NASA_GSFC_MERRA_slv_2
- ECMWF "ERA5 in Earth Engine" — https://www.ecmwf.int/en/newsletter/162/news/era5-reanalysis-data-available-earth-engine

**Satellite atmospheric / air-quality (GEE catalog)**
- MODIS MAIAC MCD19A2 (1 km AOD) — https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD19A2_GRANULES
- Sentinel-5P NRTI NO₂ — https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_NO2
- Sentinel-5P OFFL NO₂ — https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2
- Sentinel-5P NRTI AER_AI — https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_AER_AI
- Sentinel-5P NRTI CO — https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_CO
- Sentinel-5P NRTI HCHO — https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_HCHO
- Sentinel-5P NRTI AER_LH — https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_AER_LH
- VIIRS Black Marble VNP46A2 — https://developers.google.com/earth-engine/datasets/catalog/NASA_VIIRS_002_VNP46A2
- GHSL Population P2023A — https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_POP

**Ground networks / station-gridded / APIs**
- CPCB real-time air quality — https://cpcb.nic.in/real-time-air-qulity-data/
- CPCB CCR portal — https://airquality.cpcb.gov.in/ ; https://app.cpcbccr.com/
- CPCB CAAQM data-transmission protocol (PDF) — https://airquality.cpcb.gov.in/ccr_docs/Protocol_CAAQM.pdf
- OGD data.gov.in (CPCB AQI resource) — https://www.data.gov.in/catalog/real-time-air-quality-index ; ministry page — https://www.data.gov.in/ministrydepartment/Central%20Pollution%20Control%20Board
- IMD Pune gridded-data download — https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html ; https://www.imdpune.gov.in/Clim_Pred_LRF_New/Grided_Data_Download.html
- IMDLIB docs — https://imdlib.readthedocs.io/en/latest/Usage.html
- Pai et al. 2014 (0.25° rainfall) — https://www.researchgate.net/publication/287868289
- NOAA GHCN-Daily in GEE — https://developers.google.com/earth-engine/datasets/catalog/NOAA_GHCND
- NASA POWER API (parameters / hourly / daily) — https://power.larc.nasa.gov/docs/services/api/ ; https://power.larc.nasa.gov/docs/services/api/temporal/hourly/ ; https://power.larc.nasa.gov/docs/services/api/temporal/daily/
- NASA POWER methodology / data sources — https://power.larc.nasa.gov/docs/methodology/data/sources/
- Open-Meteo Historical Weather API — https://open-meteo.com/en/docs/historical-weather-api ; Historical Forecast — https://open-meteo.com/en/docs/historical-forecast-api

**Solar radiation (out-of-GEE)**
- CAMS solar radiation service — https://atmosphere.copernicus.eu/solar-radiation-services ; ADS dataset — https://ads.atmosphere.copernicus.eu/datasets/cams-solar-radiation-timeseries ; docs — https://confluence.ecmwf.int/display/CKB/CAMS+solar+radiation+time-series:+data+documentation
- SARAH-3 / CM SAF — https://www.cmsaf.eu/ (Surface Solar Radiation Data Set – Heliosat)
- ISRO MOSDAC (INSAT-3D/3DR products) — https://www.mosdac.gov.in/

**Methods / context**
- BlackMarble VNP46A2 product page — https://blackmarble.gsfc.nasa.gov/VNP46A2.html
- GHSL GHS-POP (EC) — https://human-settlement.emergency.copernicus.eu/ghs_pop.php
- Global anthropogenic-heat from nightlights — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5667572/
- PBL height from AIRS/MERRA-2 — https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021EA001859
- IMDLIB paper (Env. Modelling & Software 2024) — https://www.sciencedirect.com/science/article/abs/pii/S1364815223002554

---

### Verification status
- **Web-verified (GEE catalog fetched 2026-06-22):** ERA5-Land Hourly band IDs/units/dates; GLDAS-2.1 band IDs/units/dates; MCD19A2 ID & AOD bands; S5P NO2/AER_AI/CO/HCHO/AER_LH IDs & key bands; VNP46A2 ID/bands/500 m/dates; GHS_POP ID/100 m; NASA POWER parameter names & resolutions; Open-Meteo ERA5/ERA5-Land coverage; CPCB CAAQMS met+pollutant variables & data.gov.in route; IMD 0.25° rain / 1° temp specs & IMDLIB.
- **From-knowledge (verify on integration):** exact MERRA-2 `M2T1NXFLX:PBLH` and `NASA/GSFC/MERRA/aer/2` band spellings; CAMS/SARAH/INSAT exact endpoints & India-disk coverage edges; S5P true footprint dates (~Aug-2019 pixel-size change); GLDAS `Qair` unit label (mass fraction = kg/kg); precise per-band accumulation flags for ERA5-Land `*_hourly` variants; CPCB CCR reverse-engineered endpoints (subject to ToS/change). Treat all latency figures as nominal — confirm against live asset metadata at build time.
