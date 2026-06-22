"""urbanheat.constants — THE SINGLE SOURCE OF TRUTH catalog.

Every dataset ID, band name, scale/offset, physical constant, spectral-index
coefficient, classification threshold and intervention parameter used anywhere
in the system is defined here, exactly once, with a citation to the research
file it was extracted from (``research/01..10``). Builders MUST import these
values rather than hard-coding magic numbers.

Design rules
------------
* Pure data only. No heavy imports (no ``ee``, ``torch``, ``numpy`` required at
  import time) so every module — including the offline synthetic path — can read
  the catalog with zero optional dependencies.
* ``K = DN * scale + offset`` is the convention for scaled satellite products;
  convert to Celsius with ``C = K - 273.15`` *after* applying scale/offset.
* Provenance tags in comments: [R1]..[R10] = the research file the value came
  from; "(verify)" = flagged in research as from-knowledge / to re-confirm.

References: research/01_lst_thermal_datasets.md ... research/10_tools_ecosystem.md
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 0. Physical constants  [R5 §1.2, R2 §1, R8 §6.2]
# ---------------------------------------------------------------------------
PHYSICAL_CONSTANTS: dict[str, float] = {
    "STEFAN_BOLTZMANN": 5.670374419e-8,  # sigma, W m^-2 K^-4  [R5 Eq.3]
    "VON_KARMAN": 0.40,                   # kappa, Macdonald z0/zd  [R4 §4.5]
    "KELVIN_OFFSET": 273.15,              # K -> degC
    "CP_AIR": 1004.0,                     # specific heat of air, J kg^-1 K^-1  [R5 Eq.4]
    "RHO_AIR": 1.2,                       # air density, kg m^-3 (sea level, ~20C)
    "PSYCHROMETRIC_GAMMA": 0.066,         # kPa K^-1  [R5 Eq.5]
    "LAPSE_RATE_C_PER_M": -0.0065,        # standard environmental lapse  [R3 §4.2]
    "PLANCK_LAMBDA_B10": 10.895e-6,       # Landsat B10 wavelength, m  [R7 §2.6]
    "PLANCK_RHO": 1.438e-2,               # h*c/k_B, m*K  [R7 §2.6]
    "SOLAR_CONST": 1361.0,                # W m^-2, TOA solar irradiance
    "EMISSIVITY_BODY_SW": 0.70,           # SOLWEIG body shortwave absorption  [R6 §4]
    "EMISSIVITY_BODY_LW": 0.95,           # SOLWEIG body longwave emissivity   [R6 §4]
}

# Convenience top-level aliases (used throughout the physics modules).
SIGMA_SB = PHYSICAL_CONSTANTS["STEFAN_BOLTZMANN"]
KELVIN = PHYSICAL_CONSTANTS["KELVIN_OFFSET"]
VON_KARMAN = PHYSICAL_CONSTANTS["VON_KARMAN"]


# ---------------------------------------------------------------------------
# 1. GEE dataset catalog  [R1, R2, R3, R4]
# ---------------------------------------------------------------------------
# Schema per entry:
#   id      : exact GEE ImageCollection / Image asset id ('' => not in GEE)
#   bands   : list of band names we read
#   scale   : multiplicative scale factor to apply to the primary band(s)
#   offset  : additive offset (after scale)  =>  physical = DN*scale + offset
#   units   : physical units after scale/offset
#   role    : 'primary' | 'secondary' | 'fusion' | 'reference' | 'prior'
#   note    : provenance / caveats
#
# IMPORTANT: scale/offset apply to the value/measurement band named in 'bands'
# unless 'note' says otherwise. Where products bundle bands with *different*
# scales (e.g. MODIS view_time x0.1 vs LST x0.02) the extra scale lives in
# 'note' / BAND_SCALE_OVERRIDES below.

GEE_DATASETS: dict[str, dict] = {
    # ----- LST / thermal (PRIMARY fine-res + diurnal backbone) [R1] -----
    "LANDSAT8_L2": {
        "id": "LANDSAT/LC08/C02/T1_L2",
        "bands": ["ST_B10", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
                  "ST_QA", "QA_PIXEL"],
        "scale": 0.00341802, "offset": 149.0, "units": "K",
        "role": "primary",
        "note": "Surface Temp ST_B10: K=DN*0.00341802+149.0. SR bands x2.75e-05-0.2. "
                "ST_QA x0.01 K. Effective TIR res ~100 m served at 30 m. [R1 2.1]",
    },
    "LANDSAT9_L2": {
        "id": "LANDSAT/LC09/C02/T1_L2",
        "bands": ["ST_B10", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
                  "ST_QA", "QA_PIXEL"],
        "scale": 0.00341802, "offset": 149.0, "units": "K",
        "role": "primary",
        "note": "L9 pairs with L8 -> 8-day combined revisit. [R1 2.1]",
    },
    "LANDSAT7_L2": {
        "id": "LANDSAT/LE07/C02/T1_L2",
        "bands": ["ST_B6", "SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7",
                  "QA_PIXEL"],
        "scale": 0.00341802, "offset": 149.0, "units": "K",
        "role": "secondary",
        "note": "Historical ST_B6, same scale/offset. SLC-off gaps after 2003. [R1 2.2]",
    },
    "LANDSAT5_L2": {
        "id": "LANDSAT/LT05/C02/T1_L2",
        "bands": ["ST_B6", "SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7",
                  "QA_PIXEL"],
        "scale": 0.00341802, "offset": 149.0, "units": "K",
        "role": "secondary",
        "note": "Long-baseline UHI trend back to 1984. [R1 2.2]",
    },
    "MODIS_MOD11A1": {
        "id": "MODIS/061/MOD11A1",
        "bands": ["LST_Day_1km", "LST_Night_1km", "QC_Day", "QC_Night",
                  "Day_view_time", "Night_view_time", "Emis_31", "Emis_32"],
        "scale": 0.02, "offset": 0.0, "units": "K",
        "role": "primary",
        "note": "Terra SW daily. LST x0.02 -> K. view_time x0.1 (hours). "
                "Emis x0.002+0.49. Terra ~10:30/22:30. [R1 2.4]",
    },
    "MODIS_MYD11A1": {
        "id": "MODIS/061/MYD11A1",
        "bands": ["LST_Day_1km", "LST_Night_1km", "QC_Day", "QC_Night",
                  "Day_view_time", "Night_view_time", "Emis_31", "Emis_32"],
        "scale": 0.02, "offset": 0.0, "units": "K",
        "role": "primary",
        "note": "Aqua SW daily. Aqua ~13:30/01:30 (afternoon peak). [R1 2.4]",
    },
    "MODIS_MOD21A1D": {
        "id": "MODIS/061/MOD21A1D",
        "bands": ["LST_1KM", "Emis_29", "Emis_31", "Emis_32", "View_Time", "QC"],
        "scale": 0.02, "offset": 0.0, "units": "K",
        "role": "fusion",
        "note": "Terra TES day. Better emissivity over arid/urban; cross-check vs "
                "MOD11. Filter high-temp outliers via QC. [R1 2.4b]",
    },
    "MODIS_MYD21A1D": {
        "id": "MODIS/061/MYD21A1D",
        "bands": ["LST_1KM", "Emis_29", "Emis_31", "Emis_32", "View_Time", "QC"],
        "scale": 0.02, "offset": 0.0, "units": "K",
        "role": "fusion",
        "note": "Aqua TES day. [R1 2.4b]",
    },
    "VIIRS_VNP21A1D": {
        "id": "NASA/VIIRS/002/VNP21A1D",
        "bands": ["LST_1KM", "Emis_14", "Emis_15", "Emis_16", "View_Time", "QC"],
        "scale": 0.02, "offset": 0.0, "units": "K",
        "role": "secondary",
        "note": "SNPP TES day, native 750 m -> 1 km SIN. ~13:30. MODIS continuity. [R1 2.5]",
    },
    "VIIRS_VNP21A1N": {
        "id": "NASA/VIIRS/002/VNP21A1N",
        "bands": ["LST_1KM", "Emis_14", "Emis_15", "Emis_16", "View_Time", "QC"],
        "scale": 0.02, "offset": 0.0, "units": "K",
        "role": "secondary",
        "note": "SNPP TES night ~01:30. [R1 2.5]",
    },
    "ECOSTRESS_L2T_LSTE": {
        "id": "NASA/ECOSTRESS/L2T_LSTE/V2",
        "bands": ["LST", "LST_err", "EmisWB", "QC", "cloud"],
        "scale": 1.0, "offset": 0.0, "units": "K",
        "role": "reference",
        "note": "~70 m TES, ISS precessing (all hours incl. night). LST band already "
                "Kelvin. WARNING: GEE holds LA-metro tiles ONLY -> for India use "
                "LP DAAC/AppEEARS. [R1 2.3]",
    },
    # ----- Emissivity reference [R1, R2] -----
    "ASTER_GED": {
        "id": "NASA/ASTER_GED/AG100_003",
        "bands": ["emissivity_band10", "emissivity_band11", "emissivity_band12",
                  "emissivity_band13", "emissivity_band14", "temperature", "ndvi"],
        "scale": 0.001, "offset": 0.0, "units": "emissivity",
        "role": "prior",
        "note": "Static 100 m emissivity climatology (2000-2008). emissivity x0.001; "
                "temperature x0.01 K; ndvi x0.01. Stale over fast-urbanizing fringe. [R1 2.7]",
    },
    # ----- LULC / vegetation / surface (drivers) [R2] -----
    "DYNAMIC_WORLD": {
        "id": "GOOGLE/DYNAMICWORLD/V1",
        "bands": ["built", "trees", "grass", "water", "crops", "shrub_and_scrub",
                  "bare", "flooded_vegetation", "snow_and_ice", "label"],
        "scale": 1.0, "offset": 0.0, "units": "probability",
        "role": "primary",
        "note": "NRT 10 m, per-class probability bands -> sub-pixel fractional cover. "
                "mean(built) ~ impervious frac; mean(trees+grass+shrub) ~ green frac. [R2 L3]",
    },
    "ESA_WORLDCOVER_V200": {
        "id": "ESA/WorldCover/v200",
        "bands": ["Map"],
        "scale": 1.0, "offset": 0.0, "units": "class",
        "role": "secondary",
        "note": "10 m, 11 classes (built-up=50), 2021. Use .first(). [R2 L1]",
    },
    "ESRI_LULC_TS": {
        "id": "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS",
        "bands": ["b1"],
        "scale": 1.0, "offset": 0.0, "units": "class",
        "role": "secondary",
        "note": "10 m annual 2017-2024, 3rd LULC vote. [R2 L4]",
    },
    "S2_SR_HARMONIZED": {
        "id": "COPERNICUS/S2_SR_HARMONIZED",
        "bands": ["B2", "B3", "B4", "B8", "B8A", "B11", "B12"],
        "scale": 0.0001, "offset": 0.0, "units": "reflectance",
        "role": "primary",
        "note": "10-20 m, ~5 day. Source of all spectral indices + S2 albedo. "
                "Mask with S2_CLOUD_PROBABILITY. SR x0.0001. [R2 V1]",
    },
    "S2_CLOUD_PROBABILITY": {
        "id": "COPERNICUS/S2_CLOUD_PROBABILITY",
        "bands": ["probability"],
        "scale": 1.0, "offset": 0.0, "units": "percent",
        "role": "reference",
        "note": "Per-scene cloud mask for S2. [R2 V2]",
    },
    "MODIS_MOD13Q1": {
        "id": "MODIS/061/MOD13Q1",
        "bands": ["NDVI", "EVI"],
        "scale": 0.0001, "offset": 0.0, "units": "index",
        "role": "fusion",
        "note": "250 m 16-day NDVI/EVI, cloud-robust gap-fill backbone. x0.0001. [R2 V5]",
    },
    "MODIS_MOD15A2H": {
        "id": "MODIS/061/MOD15A2H",
        "bands": ["Lai_500m", "Fpar_500m"],
        "scale": 0.1, "offset": 0.0, "units": "m2/m2",
        "role": "secondary",
        "note": "LAI x0.1, FPAR x0.01. Canopy transpiration / shading driver. [R2 V8]",
    },
    "HANSEN_GFC": {
        "id": "UMD/hansen/global_forest_change_2024_v1_12",
        "bands": ["treecover2000", "loss", "lossyear", "gain"],
        "scale": 1.0, "offset": 0.0, "units": "percent",
        "role": "secondary",
        "note": "Tree canopy %2000 + loss-year -> canopy-loss attribution. [R2 V10]",
    },
    "MODIS_MOD44B": {
        "id": "MODIS/061/MOD44B",
        "bands": ["Percent_Tree_Cover"],
        "scale": 1.0, "offset": 0.0, "units": "percent",
        "role": "secondary",
        "note": "Continuous tree-cover fraction, independent of Hansen. [R2 V11]",
    },
    "MODIS_MOD16A2GF": {
        "id": "MODIS/061/MOD16A2GF",
        "bands": ["ET", "PET", "LE", "PLE"],
        "scale": 0.1, "offset": 0.0, "units": "kg/m2/8day",
        "role": "secondary",
        "note": "Penman-Monteith ET (gap-filled). ET/PET x0.1. Latent-heat cooling. [R2 V13]",
    },
    "PML_V2": {
        "id": "projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a",
        "bands": ["Ec", "Es", "Ei", "ET_water", "GPP"],
        "scale": 1.0, "offset": 0.0, "units": "mm/8day",
        "role": "secondary",
        "note": "Partitioned ET (transpiration/soil/interception) -> intervention story. [R2 V14]",
    },
    # ----- Surface radiative properties [R2] -----
    "MODIS_MCD43A3": {
        "id": "MODIS/061/MCD43A3",
        "bands": ["Albedo_BSA_shortwave", "Albedo_WSA_shortwave",
                  "Albedo_BSA_vis", "Albedo_WSA_vis"],
        "scale": 0.001, "offset": 0.0, "units": "albedo",
        "role": "reference",
        "note": "BRDF-corrected broadband albedo, 500 m. x0.001 (valid 0-32766). "
                "Physical albedo reference to anchor S2/Landsat albedo. [R2 S1]",
    },
    "SMAP_L4": {
        "id": "NASA/SMAP/SPL4SMGP/008",
        "bands": ["sm_surface", "sm_rootzone"],
        "scale": 1.0, "offset": 0.0, "units": "m3/m3",
        "role": "secondary",
        "note": "9 km 3-hourly soil moisture -> LE ceiling (monsoon-critical: trees "
                "cool 4C wet vs 1C dry). [R2 S10]",
    },
    # ----- Urban morphology / 3D form [R4] -----
    "GHSL_BUILT_S_10M": {
        "id": "JRC/GHSL/P2023A/GHS_BUILT_S_10m",
        "bands": ["built_surface"],
        "scale": 1.0, "offset": 0.0, "units": "m2/cell",
        "role": "primary",
        "note": "Built surface m2/cell, 10 m, 2018. built_frac = built_m2 / cell_area "
                "= plan area fraction lambda_P / impervious. [R4 #1]",
    },
    "GHSL_BUILT_H": {
        "id": "JRC/GHSL/P2023A/GHS_BUILT_H",
        "bands": ["built_height"],
        "scale": 1.0, "offset": 0.0, "units": "m",
        "role": "secondary",
        "note": "Average Net Building Height (ANBH), 100 m, 2018, metres. -> z0,zd,H/W. "
                "Coarse backstop for building height. [R4 #3]",
    },
    "GHSL_BUILT_V": {
        "id": "JRC/GHSL/P2023A/GHS_BUILT_V",
        "bands": ["built_volume_total"],
        "scale": 1.0, "offset": 0.0, "units": "m3/cell",
        "role": "secondary",
        "note": "Built volume m3/cell, 100 m -> thermal-mass / nocturnal-UHI proxy (G). [R4 #4]",
    },
    "GHSL_POP": {
        "id": "JRC/GHSL/P2023A/GHS_POP",
        "bands": ["population_count"],
        "scale": 1.0, "offset": 0.0, "units": "persons/cell",
        "role": "primary",
        "note": "Residential population, 100 m -> anthropogenic-heat QF + exposure. [R4 #7]",
    },
    "GHSL_SMOD": {
        "id": "JRC/GHSL/P2023A/GHS_SMOD_V2-0",
        "bands": ["smod_code"],
        "scale": 1.0, "offset": 0.0, "units": "class",
        "role": "reference",
        "note": "Degree of Urbanisation (urban centre/cluster/rural), 1 km. Stratify "
                "hotspot stats + rural reference for SUHII. [R4 #6] (verify id)",
    },
    "OPEN_BUILDINGS_V3": {
        "id": "GOOGLE/Research/open-buildings/v3/polygons",
        "bands": [],
        "scale": 1.0, "offset": 0.0, "units": "vector",
        "role": "secondary",
        "note": "1.8B footprints (India covered), attrs area_in_meters, confidence "
                "(filter >=0.70). FeatureCollection. [R4 #11]",
    },
    "OPEN_BUILDINGS_TEMPORAL": {
        "id": "GOOGLE/Research/open-buildings-temporal/v1",
        "bands": ["building_presence", "building_height", "building_fractional_count"],
        "scale": 1.0, "offset": 0.0, "units": "mixed",
        "role": "secondary",
        "note": "4 m annual 2016-2023, per-year building HEIGHT (m) for India w/o LiDAR. [R4 #12]",
    },
    "UT_GLOBUS": {
        "id": "projects/sat-io/open-datasets/UT-GLOBUS",  # append /<city>
        "bands": [],
        "scale": 1.0, "offset": 0.0, "units": "vector",
        "role": "secondary",
        "note": "Per-building height (m AGL, attr 'height') + UCPs (lambda_P,lambda_F,h_a,"
                "lambda_B) for >1200 cities. Per-city FC: .../UT-GLOBUS/<city>. RMSE~9 m. [R4 #16]",
    },
    "COPERNICUS_DEM_GLO30": {
        "id": "COPERNICUS/DEM/GLO30",
        "bands": ["DEM"],
        "scale": 1.0, "offset": 0.0, "units": "m",
        "role": "primary",
        "note": "30 m DSM (incl. buildings). DSM term in height = DSM - FABDEM. Mosaic "
                "before use. [R4 #20]",
    },
    "FABDEM": {
        "id": "projects/sat-io/open-datasets/FABDEM",
        "bands": ["b1"],
        "scale": 1.0, "offset": 0.0, "units": "m",
        "role": "secondary",
        "note": "Bare-earth DEM (buildings+forest removed). object_height = GLO30 - "
                "FABDEM. CC-BY-NC-SA (research only). [R4 #24]",
    },
    "NASADEM": {
        "id": "NASA/NASADEM_HGT/001",
        "bands": ["elevation"],
        "scale": 1.0, "offset": 0.0, "units": "m",
        "role": "reference",
        "note": "30 m DEM for slope/aspect + LST terrain-detrending. [R4 #22]",
    },
    "GLOBAL_LCZ": {
        "id": "RUB/RUBCLIM/LCZ/global_lcz_map/latest",
        "bands": ["LCZ_Filter"],
        "scale": 1.0, "offset": 0.0, "units": "class",
        "role": "primary",
        "note": "Local Climate Zones (17 classes, 100 m). Morphology+thermal class; "
                "seeds SVF/H-W/lambda_P/z0 priors (see LCZ_TABLE). [R4 #26]",
    },
    # ----- Meteorology / atmosphere [R3] -----
    "ERA5_LAND_HOURLY": {
        "id": "ECMWF/ERA5_LAND/HOURLY",
        "bands": ["temperature_2m", "dewpoint_temperature_2m",
                  "u_component_of_wind_10m", "v_component_of_wind_10m",
                  "surface_solar_radiation_downwards", "surface_thermal_radiation_downwards",
                  "surface_net_solar_radiation", "surface_net_thermal_radiation",
                  "surface_pressure", "total_precipitation",
                  "surface_sensible_heat_flux", "surface_latent_heat_flux",
                  "skin_temperature"],
        "scale": 1.0, "offset": 0.0, "units": "mixed",
        "role": "primary",
        "note": "0.1 deg hourly PRIMARY atmospheric forcing. T/Td/skin_T in K. Radiation "
                "& flux bands ACCUMULATED from 00 UTC -> diff successive hours /3600 for "
                "W/m2. [R3 2.1]",
    },
    "ERA5_HOURLY": {
        "id": "ECMWF/ERA5/HOURLY",
        "bands": ["boundary_layer_height", "mean_sea_level_pressure",
                  "total_column_water_vapour", "total_cloud_cover"],
        "scale": 1.0, "offset": 0.0, "units": "mixed",
        "role": "secondary",
        "note": "0.25 deg. Adds boundary_layer_height (stability) ERA5-Land lacks. [R3 2.2]",
    },
    "GLDAS_NOAH": {
        "id": "NASA/GLDAS/V021/NOAH/G025/T3H",
        "bands": ["Tair_f_inst", "Qair_f_inst", "Wind_f_inst",
                  "SWdown_f_tavg", "LWdown_f_tavg", "Psurf_f_inst",
                  "Swnet_tavg", "Lwnet_tavg", "Qh_tavg", "Qle_tavg", "Qg_tavg",
                  "AvgSurfT_inst", "Evap_tavg"],
        "scale": 1.0, "offset": 0.0, "units": "mixed",
        "role": "secondary",
        "note": "0.25 deg 3-hourly. FLUXES ALREADY in W/m2 (Qh/Qle/Qg) -> SEB closure "
                "check + Bowen-ratio priors. _inst/_tavg/_f naming. [R3 2.4]",
    },
    "MERRA2_SLV": {
        "id": "NASA/GSFC/MERRA/slv/2",
        "bands": ["T2M", "T10M", "QV2M", "U10M", "V10M", "SLP", "PS", "TQV"],
        "scale": 1.0, "offset": 0.0, "units": "mixed",
        "role": "reference",
        "note": "0.5x0.625 deg. Independent reanalysis #2 for cross-check (T/humidity/wind). [R3 2.7]",
    },
    "MODIS_MAIAC_AOD": {
        "id": "MODIS/061/MCD19A2_GRANULES",
        "bands": ["Optical_Depth_047", "Optical_Depth_055", "AOD_Uncertainty",
                  "Column_WV", "AOD_QA"],
        "scale": 0.001, "offset": 0.0, "units": "AOD",
        "role": "secondary",
        "note": "1 km daily AOD x0.001 -> K_down attenuation (hazy Indian cities) + PM "
                "proxy. Filter AOD_QA. [R3 2.13]",
    },
    "S5P_NO2": {
        "id": "COPERNICUS/S5P/OFFL/L3_NO2",
        "bands": ["tropospheric_NO2_column_number_density", "NO2_column_number_density"],
        "scale": 1.0, "offset": 0.0, "units": "mol/m2",
        "role": "secondary",
        "note": "~1 km grid. Combustion/traffic -> QF proxy + heat x air-quality overlay. [R3 2.14]",
    },
    "VIIRS_BLACK_MARBLE": {
        "id": "NASA/VIIRS/002/VNP46A2",
        "bands": ["Gap_Filled_DNB_BRDF_Corrected_NTL", "DNB_BRDF_Corrected_NTL",
                  "Mandatory_Quality_Flag", "QF_Cloud_Mask"],
        "scale": 1.0, "offset": 0.0, "units": "nW/cm2/sr",
        "role": "primary",
        "note": "500 m daily nightlights -> QF anthropogenic-heat proxy. Prefer "
                "Gap_Filled_DNB_BRDF_Corrected_NTL. [R3 2.20 / R4 #28]",
    },
}

# Bands inside a product that carry a DIFFERENT scale than the product default.
# Reader code must consult this before applying GEE_DATASETS[*]['scale'].
BAND_SCALE_OVERRIDES: dict[str, dict[str, tuple[float, float]]] = {
    # product_key: {band: (scale, offset)}
    "MODIS_MOD11A1": {
        "Day_view_time": (0.1, 0.0), "Night_view_time": (0.1, 0.0),
        "Emis_31": (0.002, 0.49), "Emis_32": (0.002, 0.49),
    },
    "MODIS_MYD11A1": {
        "Day_view_time": (0.1, 0.0), "Night_view_time": (0.1, 0.0),
        "Emis_31": (0.002, 0.49), "Emis_32": (0.002, 0.49),
    },
    "ASTER_GED": {
        "temperature": (0.01, 0.0), "ndvi": (0.01, 0.0),
    },
    "MODIS_MOD15A2H": {"Fpar_500m": (0.01, 0.0)},
    "LANDSAT8_L2": {
        "SR_B2": (2.75e-05, -0.2), "SR_B3": (2.75e-05, -0.2),
        "SR_B4": (2.75e-05, -0.2), "SR_B5": (2.75e-05, -0.2),
        "SR_B6": (2.75e-05, -0.2), "SR_B7": (2.75e-05, -0.2),
        "ST_QA": (0.01, 0.0),
    },
    "LANDSAT9_L2": {
        "SR_B2": (2.75e-05, -0.2), "SR_B3": (2.75e-05, -0.2),
        "SR_B4": (2.75e-05, -0.2), "SR_B5": (2.75e-05, -0.2),
        "SR_B6": (2.75e-05, -0.2), "SR_B7": (2.75e-05, -0.2),
        "ST_QA": (0.01, 0.0),
    },
}

# Non-GEE sources to ingest externally (Earthdata / Copernicus / MOSDAC / AWS). [R1, R3]
EXTERNAL_SOURCES: dict[str, str] = {
    "ECOSTRESS_INDIA": "LP DAAC / AppEEARS (ECO_L2T_LSTE v002) -- GEE = LA only",
    "SENTINEL3_SLSTR": "Copernicus Data Space / MS Planetary Computer (SL_2_LST___)",
    "INSAT_3D_LST": "MOSDAC 3DIMG/3RIMG/3SIMG_*_L2B_LST (4 km, 15-30 min, India geostationary)",
    "INSAT_INSOLATION": "MOSDAC INSAT-3D INSOLATION (indigenous K_down over India)",
    "CPCB_CAAQMS": "data.gov.in OGD API + CPCB CCR (in-canopy urban T/RH/WS/WD + PM)",
    "IMD_GRIDDED": "IMD Pune CMPG / IMDLIB (0.25 deg rain, 1 deg Tmax/Tmin)",
    "NASA_POWER": "power.larc.nasa.gov REST (solar K_down + L_down + T/RH/WS, no auth)",
    "BHUVAN_NRSC": "NRSC Bhuvan LULC (ISRO-authoritative, ingest as GEE asset)",
}


# ---------------------------------------------------------------------------
# 2. Spectral index coefficients  [R2 §5]
# ---------------------------------------------------------------------------
# Generic index formulas (band roles, not raw band names) live in the indices
# module; here we keep the *coefficient sets* that are data, not logic.
SPECTRAL_INDEX_COEFFS: dict[str, dict] = {
    # Broadband albedo from Landsat-8 OLI SR (Liang 2001 SWB form). [R2 §5]
    "ALBEDO_LANDSAT_LIANG": {
        "B2": 0.356, "B4": 0.130, "B5": 0.373, "B6": 0.085, "B7": 0.072,
        "const": -0.0018,
    },
    # Broadband albedo from Sentinel-2 (Bonafoni & Sekertekin 2020). [R2 §5] (verify)
    "ALBEDO_S2_BONAFONI": {
        "B2": 0.2266, "B3": 0.1236, "B4": 0.1573, "B8": 0.3417,
        "B11": 0.1170, "B12": 0.0338, "const": 0.0,
    },
    # NDVI-threshold emissivity (Sobrino). [R2 §5 / R7 2.6]
    "EMISSIVITY_NDVI": {
        "eps_veg": 0.985, "eps_soil": 0.96, "d_eps": 0.005,
        "ndvi_soil": 0.2, "ndvi_veg": 0.5,  # fractional-veg-cover thresholds
    },
    # EVI constants. [R2 §5]
    "EVI": {"G": 2.5, "C1": 6.0, "C2": 7.5, "L": 1.0},
    "SAVI": {"L": 0.5},
}


# ---------------------------------------------------------------------------
# 3. Heat-stress / hotspot classification  [R8]
# ---------------------------------------------------------------------------
# LST percentile hotspot bands (distribution-free; recommended for India). [R8 §3.3]
LST_PERCENTILE_THRESHOLDS: dict[str, float] = {
    "high": 90.0, "very_hot": 95.0, "extreme": 98.0,
}
LST_ZSCORE_THRESHOLDS: dict[str, float] = {
    "warm": 1.5, "hot": 2.0, "extreme": 2.5,
}

# UTFVI -> Ecological Evaluation Index (Liu & Zhang 2011). UTFVI=(Ts-Tm)/Tm in KELVIN. [R8 §3.4]
UTFVI_CLASSES: list[dict] = [
    {"max": 0.000, "uhi": "none", "eei": "excellent"},
    {"max": 0.005, "uhi": "weak", "eei": "good"},
    {"max": 0.010, "uhi": "moderate", "eei": "normal"},
    {"max": 0.015, "uhi": "strong", "eei": "bad"},
    {"max": 0.020, "uhi": "stronger", "eei": "worse"},
    {"max": float("inf"), "uhi": "strongest", "eei": "worst"},
]

# Getis-Ord Gi* / Moran significance (two-tailed z). [R8 §9]
HOTSPOT_GISTAR_Z: dict[str, float] = {"p90": 1.65, "p95": 1.96, "p99": 2.58}

# Final 5-class priority legend + hex colours (RdYlBu reversed, colour-blind safe). [R8 §12.1]
HOTSPOT_LEGEND: list[dict] = [
    {"name": "Low",      "min": 0,  "max": 20,  "hex": "#2c7bb6"},
    {"name": "Moderate", "min": 20, "max": 40,  "hex": "#abd9e9"},
    {"name": "High",     "min": 40, "max": 60,  "hex": "#ffffbf"},
    {"name": "Severe",   "min": 60, "max": 80,  "hex": "#fdae61"},
    {"name": "Extreme",  "min": 80, "max": 100, "hex": "#d7191c"},
]
# Conventional LST ramp for pure-surface maps (YlOrRd). [R8 §12.1]
LST_COLOR_RAMP: list[str] = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]

# Human heat-stress danger thresholds (Tier B1, degC unless noted). [R8 §12.2, §5]
HEAT_STRESS_THRESHOLDS: dict[str, dict] = {
    "wet_bulb": {"danger": 28.0, "extreme": 31.0, "lethal": 35.0},
    "heat_index_f": {"caution": 80, "extreme_caution": 91, "danger": 103, "extreme_danger": 125},
    "humidex": {"discomfort": 30, "great_discomfort": 40, "dangerous": 46, "stroke": 54},
    "discomfort_index": {"half_uncomf": 24, "most_uncomf": 27, "strong": 29, "emergency": 32},
    "wbgt": {"low": 28, "moderate": 30, "high": 32},
    "utci": {"moderate": 26, "strong": 32, "very_strong": 38, "extreme": 46},
    "pet": {"slight": 23, "moderate": 29, "strong": 35, "extreme": 41},
}

# UTCI 10-category stress scale (lower bound degC -> label). [R8 §6.3]
UTCI_CATEGORIES: list[dict] = [
    {"min": 46, "label": "extreme heat stress"},
    {"min": 38, "label": "very strong heat stress"},
    {"min": 32, "label": "strong heat stress"},
    {"min": 26, "label": "moderate heat stress"},
    {"min": 9, "label": "no thermal stress"},
    {"min": 0, "label": "slight cold stress"},
    {"min": -13, "label": "moderate cold stress"},
    {"min": -27, "label": "strong cold stress"},
    {"min": -40, "label": "very strong cold stress"},
    {"min": float("-inf"), "label": "extreme cold stress"},
]

# IMD heat-wave criteria (India). [R8 §7]
IMD_HEATWAVE: dict[str, float] = {
    "tmax_plains_min": 40.0, "tmax_coastal_min": 37.0, "tmax_hilly_min": 30.0,
    "departure_hw_low": 4.5, "departure_hw_high": 6.4,
    "absolute_hw": 45.0, "absolute_severe": 47.0,
    "min_days": 2,
}


# ---------------------------------------------------------------------------
# 4. Local Climate Zone table (Stewart & Oke 2012, Table 3)  [R4 §5]
# ---------------------------------------------------------------------------
# Per class: midpoints/representative values used to SEED morphology priors where
# building data is missing. svf, hw (H/W aspect ratio), bsf (building surface
# fraction %), isf (impervious %), height (roughness-element m), z0_class
# (terrain roughness class), is_built. Values: LCZ 1-2 web-confirmed, 3-G
# canonical-from-knowledge (verify). [R4 §5]
LCZ_TABLE: dict[int, dict] = {
    1:  {"name": "Compact high-rise",   "svf": 0.30, "hw": 2.5,  "bsf": 50, "isf": 50, "height": 30, "z0_class": 8, "is_built": True},
    2:  {"name": "Compact midrise",     "svf": 0.45, "hw": 1.25, "bsf": 55, "isf": 40, "height": 17, "z0_class": 6, "is_built": True},
    3:  {"name": "Compact low-rise",    "svf": 0.40, "hw": 1.0,  "bsf": 55, "isf": 35, "height": 6,  "z0_class": 6, "is_built": True},
    4:  {"name": "Open high-rise",      "svf": 0.60, "hw": 1.0,  "bsf": 30, "isf": 35, "height": 30, "z0_class": 7, "is_built": True},
    5:  {"name": "Open midrise",        "svf": 0.65, "hw": 0.5,  "bsf": 30, "isf": 40, "height": 17, "z0_class": 5, "is_built": True},
    6:  {"name": "Open low-rise",       "svf": 0.75, "hw": 0.5,  "bsf": 30, "isf": 35, "height": 6,  "z0_class": 5, "is_built": True},
    7:  {"name": "Lightweight low-rise","svf": 0.35, "hw": 1.5,  "bsf": 75, "isf": 15, "height": 3,  "z0_class": 4, "is_built": True},
    8:  {"name": "Large low-rise",      "svf": 0.80, "hw": 0.2,  "bsf": 40, "isf": 45, "height": 6,  "z0_class": 5, "is_built": True},
    9:  {"name": "Sparsely built",      "svf": 0.85, "hw": 0.15, "bsf": 15, "isf": 15, "height": 6,  "z0_class": 5, "is_built": True},
    10: {"name": "Heavy industry",      "svf": 0.75, "hw": 0.35, "bsf": 25, "isf": 30, "height": 10, "z0_class": 5, "is_built": True},
    11: {"name": "Dense trees",         "svf": 0.35, "hw": 0.0,  "bsf": 5,  "isf": 5,  "height": 15, "z0_class": 8, "is_built": False},  # LCZ A
    12: {"name": "Scattered trees",     "svf": 0.65, "hw": 0.0,  "bsf": 5,  "isf": 5,  "height": 9,  "z0_class": 6, "is_built": False},  # LCZ B
    13: {"name": "Bush, scrub",         "svf": 0.80, "hw": 0.0,  "bsf": 5,  "isf": 5,  "height": 1,  "z0_class": 4, "is_built": False},  # LCZ C
    14: {"name": "Low plants",          "svf": 0.95, "hw": 0.0,  "bsf": 5,  "isf": 5,  "height": 0.5, "z0_class": 3, "is_built": False}, # LCZ D (rural ref)
    15: {"name": "Bare rock / paved",   "svf": 0.95, "hw": 0.0,  "bsf": 5,  "isf": 95, "height": 0.25, "z0_class": 1, "is_built": False},# LCZ E
    16: {"name": "Bare soil / sand",    "svf": 0.95, "hw": 0.0,  "bsf": 5,  "isf": 5,  "height": 0.25, "z0_class": 1, "is_built": False},# LCZ F
    17: {"name": "Water",               "svf": 0.95, "hw": 0.0,  "bsf": 5,  "isf": 5,  "height": 0.0, "z0_class": 1, "is_built": False}, # LCZ G
}
# LCZ class used as the rural/baseline reference for SUHI (ΔT = T - T[LCZ D]). [R4 §5]
LCZ_RURAL_REFERENCE = 14  # "Low plants" (LCZ D)

# Macdonald (1998) morphometric roughness constants. [R4 §4.5]
MACDONALD_CONSTANTS: dict[str, float] = {"A": 4.43, "Cd": 1.2, "beta": 1.0}


# ---------------------------------------------------------------------------
# 5. Intervention parameters (cooling levers)  [R6 §2]
# ---------------------------------------------------------------------------
# Per intervention: surface/air/Tmrt ΔT cooling RANGES (degC, positive=cooling),
# the SEB mechanism, the FeatureStack driver variables it perturbs, and the
# feasibility class. Magnitudes are climate/time-dependent -> always ranges, and
# they are NON-ADDITIVE (the optimizer must re-predict combinations). [R6 §2,§3]
INTERVENTION_PARAMS: dict[str, dict] = {
    "urban_trees": {
        "mechanism": "shade (block K_down) + evapotranspiration (raise Q_E)",
        "surface_dC": (2.0, 12.0), "air_dC": (0.3, 2.0), "tmrt_dC": (2.0, 8.0),
        "perturbs": {"ndvi": +0.20, "tree_frac": +0.20, "shade": +0.30, "albedo": +0.02},
        "feasibility": "plantable_ground", "cost": "low-med",
        "note": "global anchor -1.5C midday LST, -0.3C air per +10% canopy [R6 ref3,4]; "
                "under-canopy -5.1C at H/W=1, -8.2C at H/W=2 [R6 ref1,5].",
    },
    "green_roof": {
        "mechanism": "ET + insulation (cut Q_G into building)",
        "surface_dC": (15.0, 45.0), "air_dC": (2.0, 5.0), "tmrt_dC": (1.0, 3.0),
        "perturbs": {"ndvi": +0.15, "albedo": +0.05, "green_frac": +0.10},
        "feasibility": "flat_roof", "cost": "high",
        "note": "roof skin 15-45C; near-surface air -2..-5C; day SUHI ~-4C [R6 ref6].",
    },
    "cool_roof": {
        "mechanism": "raise albedo -> cut absorbed (1-alpha)K_down",
        "surface_dC": (10.0, 30.0), "air_dC": (0.2, 2.3), "tmrt_dC": (0.0, 1.0),
        "perturbs": {"albedo": +0.30},
        "feasibility": "roof", "cost": "low",
        "note": "+0.1 albedo -> -0.2..-0.6C near-surface; cheapest/fastest, most "
                "cost-effective city-wide roof lever [R6 ref2].",
    },
    "cool_pavement": {
        "mechanism": "raise albedo on paved surfaces",
        "surface_dC": (5.0, 20.0), "air_dC": (0.0, 1.0), "tmrt_dC": (-3.0, 0.0),
        "perturbs": {"albedo": +0.25},
        "feasibility": "paved_non_pedestrian", "cost": "med",
        "note": "WARNING: can RAISE daytime Tmrt for pedestrians via reflected K_down "
                "-> prefer trees in pedestrian zones [R6 §2 caveat].",
    },
    "permeable_pavement": {
        "mechanism": "evaporation of stored water (raise Q_E) when wet",
        "surface_dC": (2.0, 35.0), "air_dC": (0.0, 1.0), "tmrt_dC": (0.0, 1.0),
        "perturbs": {"albedo": +0.05, "soil_moisture": +0.10},
        "feasibility": "paved", "cost": "med",
        "note": "15-35C skin reduction transiently when wet; needs moisture/recharge [R6 ref9].",
    },
    "water_body": {
        "mechanism": "evaporation (raise Q_E) + thermal mass",
        "surface_dC": (1.0, 3.0), "air_dC": (0.5, 3.0), "tmrt_dC": (0.0, 2.0),
        "perturbs": {"water_frac": +0.50, "ndwi": +0.30, "soil_moisture": +0.20},
        "feasibility": "open_space", "cost": "high",
        "note": "static water max ~-1C local, fountains -0.7..-3C, leeward decay [R6 refA,B].",
    },
    "urban_park": {
        "mechanism": "aggregate shade+ET of a large green patch (Park Cool Island)",
        "surface_dC": (1.0, 4.6), "air_dC": (0.5, 3.7), "tmrt_dC": (1.0, 5.0),
        "perturbs": {"ndvi": +0.30, "green_frac": +0.40, "shade": +0.20, "tree_frac": +0.20},
        "feasibility": "vacant_2ha", "cost": "high",
        "note": "PCI 0.5-3.7C typical; >2ha cools surroundings, exp decay ~100-300 m "
                "(InVEST d_cool) [R6 ref7,8].",
    },
    "green_wall": {
        "mechanism": "ET + facade shading",
        "surface_dC": (2.0, 13.7), "air_dC": (1.2, 3.3), "tmrt_dC": (1.0, 3.0),
        "perturbs": {"ndvi": +0.10, "albedo": +0.03},
        "feasibility": "building_facade", "cost": "high",
        "note": "wall skin -2..-13.7C (avg ~7.5); street-level -1.2..-3.0C [R6 ref10].",
    },
    "increase_albedo": {
        "mechanism": "generic albedo raise (continuous lever)",
        "surface_dC": (5.0, 25.0), "air_dC": (0.2, 2.0), "tmrt_dC": (-2.0, 0.5),
        "perturbs": {"albedo": +0.20},
        "feasibility": "any_surface", "cost": "low-med",
        "note": "city-avg -0.3C per +0.1 albedo [R6 ref2].",
    },
}

# InVEST Urban Cooling Model defaults (Bosch et al. GMD 2021). [R6 §5]
INVEST_UCM: dict[str, float] = {
    "cc_weight_shade": 0.6, "cc_weight_albedo": 0.2, "cc_weight_eti": 0.2,
    "green_area_cooling_distance_m": 100.0,   # d_cool (GMD default; UI=450)
    "t_air_average_radius_m": 500.0,          # mixing kernel r
    "park_area_threshold_ha": 2.0,
    # WBGT work-productivity (degC): light & heavy work loss thresholds. [R6 §5.5]
    "wbgt_light_25pct": 31.5, "wbgt_light_50pct": 32.0, "wbgt_light_75pct": 32.5,
    "wbgt_heavy_25pct": 27.5, "wbgt_heavy_50pct": 29.5, "wbgt_heavy_75pct": 31.5,
}

# Optimizer guarantee: lazy-greedy submodular bound. [R6 §7.2]
SUBMODULAR_GREEDY_BOUND = 0.632  # (1 - 1/e)


# ---------------------------------------------------------------------------
# 6. Heat Vulnerability Index domains  [R8 §11 / R6 §8]
# ---------------------------------------------------------------------------
# HVI = f(Exposure, Sensitivity, -Adaptive Capacity), IPCC three-domain. [R8 §11]
HVI_DOMAINS: dict[str, dict] = {
    "exposure": {
        "sign": +1,
        "indicators": ["lst_warmseason", "suhii", "utfvi", "impervious_frac", "low_ndvi"],
    },
    "sensitivity": {
        "sign": +1,
        "indicators": ["pct_elderly_65", "pct_children_5", "population_density",
                       "pct_chronic_illness", "pct_low_income", "pct_outdoor_workers"],
    },
    "adaptive_capacity": {
        "sign": -1,  # inverted: more capacity -> less vulnerable
        "indicators": ["ac_ownership", "electricity_access", "water_access",
                       "green_access", "health_facility_access", "literacy"],
    },
}
HVI_DOMAIN_WEIGHTS: dict[str, float] = {
    "exposure": 1 / 3, "sensitivity": 1 / 3, "adaptive_capacity": 1 / 3,
}


# ---------------------------------------------------------------------------
# 7. Cross-verification / robustness summary  [R9]
# ---------------------------------------------------------------------------
# The full 35-entry robustness matrix lives in research/09. We surface the
# headline counts here so the report/CLI can cite ">=30 cross-verifying methods".
ROBUSTNESS_SUMMARY: dict[str, int] = {
    "lst_sensors": 5,        # Landsat, ECOSTRESS, MODIS, VIIRS, Sentinel-3
    "lulc_products": 4,      # ESA WorldCover, ESRI, Dynamic World, GHSL/local
    "footprint_sources": 4,  # OSM, GHSL, Google, Microsoft/VIDA, UT-GLOBUS
    "met_sources": 3,        # ERA5-family, GLDAS, MERRA-2 (+ CPCB/IMD stations)
    "analytical_methods": 19,
    "total_matrix_entries": 35,
}

# Validation metric panel (computed on spatial-CV folds). [R9 §1.4]
VALIDATION_METRICS: tuple[str, ...] = (
    "rmse", "mae", "bias", "ubrmse", "r2", "nse", "ccc", "kge",
)
# Literature anchors to beat/compare. [R5 §7, R9 §1.6]
VALIDATION_ANCHORS: dict[str, float] = {
    "extra_trees_lst_r2": 0.908, "extra_trees_lst_rmse_C": 0.745,
    "xgboost_suhii_r2": 0.879,
    "ecostress_bias_K": -0.9, "ecostress_rmse_K": 2.2,
    "mod11_bias_K": 0.8, "mod11_rmse_K": 2.8,
    "spatial_cv_optimism_pct": 28.0,  # random-CV over-optimism to disclose
}

__all__ = [
    "PHYSICAL_CONSTANTS", "SIGMA_SB", "KELVIN", "VON_KARMAN",
    "GEE_DATASETS", "BAND_SCALE_OVERRIDES", "EXTERNAL_SOURCES",
    "SPECTRAL_INDEX_COEFFS",
    "LST_PERCENTILE_THRESHOLDS", "LST_ZSCORE_THRESHOLDS", "UTFVI_CLASSES",
    "HOTSPOT_GISTAR_Z", "HOTSPOT_LEGEND", "LST_COLOR_RAMP",
    "HEAT_STRESS_THRESHOLDS", "UTCI_CATEGORIES", "IMD_HEATWAVE",
    "LCZ_TABLE", "LCZ_RURAL_REFERENCE", "MACDONALD_CONSTANTS",
    "INTERVENTION_PARAMS", "INVEST_UCM", "SUBMODULAR_GREEDY_BOUND",
    "HVI_DOMAINS", "HVI_DOMAIN_WEIGHTS",
    "ROBUSTNESS_SUMMARY", "VALIDATION_METRICS", "VALIDATION_ANCHORS",
]
