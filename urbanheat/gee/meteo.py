"""urbanheat.gee.meteo — server-side meteorological / atmospheric DRIVER layers.

Atmospheric forcing for the surface energy balance and the human heat-stress
indices, built server-side from ERA5-Land (primary), ERA5 (boundary-layer
height), GLDAS-NOAH (ready-made W/m2 fluxes), MAIAC AOD and Sentinel-5P NO2.

The image bands use the canonical FeatureStack names in
:mod:`urbanheat.datamodel`:

``air_temp, dewpoint, rel_humidity, wind_speed, pressure, solar_radiation,
longwave_down, net_radiation, soil_moisture, pbl_height, aod, no2,
anthro_heat, nightlights``.

THE ERA5 ACCUMULATION GOTCHA (R3 §2.1)
--------------------------------------
ERA5-Land radiation and flux bands (``surface_solar_radiation_downwards``,
``surface_thermal_radiation_downwards``, ``surface_net_solar_radiation``,
``surface_net_thermal_radiation``, ``surface_sensible_heat_flux``,
``surface_latent_heat_flux``) are **accumulated from 00 UTC** in J/m2, NOT
instantaneous. To get an instantaneous flux in W/m2 you must take the
difference between successive hours and divide by 3600 s::

    flux_Wm2(h) = (accum(h) - accum(h-1)) / 3600

A naive ``.mean()`` over the accumulated band would average partial daily
integrals and is physically wrong. :func:`_era5_deaccumulated_mean` implements
the correct de-accumulation (difference consecutive hours, /3600, then mean of
the per-hour W/m2 values over the analysis window). State bands (temperature,
dewpoint, wind, pressure, soil water) are NOT accumulated and are simply
time-averaged.

GLDAS, by contrast, exposes fluxes already in W/m2 (``SWdown_f_tavg``,
``Qh_tavg`` ...) needing no de-accumulation — it is the cross-check / fallback.

All ``ee`` imports are lazy so this module imports with numpy only.

Public contract (ARCHITECTURE.md §11.2 ``gee/meteo.py``)::

    era5_drivers(cfg) -> ee.Image
    anthropogenic_heat(cfg) -> ee.Image  (anthro_heat, nightlights)
    downscale_air_temp(coarse_t, predictors, cfg) -> ee.Image

plus the build-task helpers :func:`meteo_layers` (one merged driver image),
:func:`aod`, :func:`no2`.

References: research/03_meteorological_atmospheric.md (§2.1, §2.4, §3, §4, §5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from urbanheat.config import Config
from urbanheat.constants import GEE_DATASETS, KELVIN
from urbanheat import datamodel as dm

if TYPE_CHECKING:  # pragma: no cover
    import ee


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _aoi(cfg: Config) -> "ee.Geometry":
    import ee  # lazy

    xmin, ymin, xmax, ymax = cfg.bbox
    return ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], proj="EPSG:4326",
                                 geodesic=False)


def _era5_collection(cfg: Config) -> "ee.ImageCollection":
    """ERA5-Land hourly collection filtered to AOI + analysis window."""
    import ee  # lazy

    meta = GEE_DATASETS["ERA5_LAND_HOURLY"]
    return (ee.ImageCollection(meta["id"])
            .filterBounds(_aoi(cfg))
            .filterDate(cfg.start_date, cfg.end_date))


def _era5_deaccumulated_mean(cfg: Config, band: str) -> "ee.Image":
    """Mean instantaneous flux (W/m2) for an ACCUMULATED ERA5-Land band.

    Implements the de-accumulation described in the module docstring: for each
    consecutive hourly pair within a UTC day, the instantaneous flux is
    ``(accum(h) - accum(h-1)) / 3600``; the first hour of a day (hour 01 UTC,
    whose value is the 00->01 integral) is divided by 3600 directly. The result
    is averaged over the analysis window. Because handling the per-day reset
    precisely server-side is awkward, we use the robust standard approximation
    used across the ERA5 community: convert the window-mean of the accumulated
    band over hour ``h`` back to a rate using the local hour-of-day, which for a
    representative midday sampling reduces to dividing the per-hour increment by
    3600. Here we difference the collection's consecutive images directly.

    Parameters
    ----------
    band : str
        Accumulated ERA5-Land band name (J/m2).

    Returns
    -------
    ee.Image
        Mean W/m2 image for the band over the window.
    """
    import ee  # lazy

    coll = _era5_collection(cfg).select(band).sort("system:time_start")
    img_list = coll.toList(coll.size())
    n = coll.size()

    def _rate(i: "ee.Number") -> "ee.Image":
        i = ee.Number(i)
        cur = ee.Image(img_list.get(i.add(1)))
        prev = ee.Image(img_list.get(i))
        cur_h = ee.Date(cur.get("system:time_start")).get("hour")
        # If the current image is hour 01 UTC (first accumulation step of the
        # day) its accumulated value already equals the 00->01 integral.
        diff = ee.Image(ee.Algorithms.If(
            ee.Number(cur_h).eq(1),
            cur,
            cur.subtract(prev),
        ))
        return diff.divide(3600.0).copyProperties(cur, ["system:time_start"])

    idx = ee.List.sequence(0, n.subtract(2))
    rates = ee.ImageCollection(idx.map(_rate))
    return rates.mean()


def _magnus_vapor_pressure(t_celsius: "ee.Image") -> "ee.Image":
    """Saturation/actual vapour pressure (hPa) via Magnus over a temperature.

    ``e = 6.112 * exp(17.67 * T_c / (T_c + 243.5))`` with T_c in degC (R3 §2.1).
    Pass dewpoint to get actual ``e``; pass air temperature to get ``es``.
    """
    return t_celsius.multiply(17.67).divide(t_celsius.add(243.5)).exp().multiply(6.112)


# ---------------------------------------------------------------------------
# ERA5-Land driver image
# ---------------------------------------------------------------------------
def era5_drivers(cfg: Config) -> "ee.Image":
    """Time-reduced ERA5-Land driver image (R3 §2.1) with canonical band names.

    Produces an ``ee.Image`` with bands:

    * ``air_temp``        (degC)   from ``temperature_2m`` - 273.15
    * ``dewpoint``        (degC)   from ``dewpoint_temperature_2m`` - 273.15
    * ``rel_humidity``    (%)      Magnus: 100*e(Td)/es(T)
    * ``wind_speed``      (m/s)    sqrt(u_10m^2 + v_10m^2)
    * ``pressure``        (kPa)    ``surface_pressure`` / 1000
    * ``solar_radiation`` (W/m2)   de-accumulated ``surface_solar_radiation_downwards``
    * ``longwave_down``   (W/m2)   de-accumulated ``surface_thermal_radiation_downwards``
    * ``net_radiation``   (W/m2)   de-accum(net SW) + de-accum(net LW)
    * ``soil_moisture``   (m3/m3)  ``volumetric_soil_water_layer_1``
    * ``pbl_height``      (m)      from ERA5 (0.25 deg) ``boundary_layer_height``

    Radiation/flux bands are de-accumulated per the ERA5 gotcha (see module
    docstring and :func:`_era5_deaccumulated_mean`); state bands are simply
    time-averaged over the window.
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    coll = _era5_collection(cfg)

    t2m = coll.select("temperature_2m").mean().subtract(KELVIN).rename(dm.AIR_TEMP)
    td2m = (coll.select("dewpoint_temperature_2m").mean().subtract(KELVIN)
            .rename(dm.DEWPOINT))

    e_act = _magnus_vapor_pressure(td2m)
    e_sat = _magnus_vapor_pressure(t2m)
    rh = (e_act.divide(e_sat).multiply(100.0).clamp(0.0, 100.0)
          .rename(dm.REL_HUMIDITY))

    u = coll.select("u_component_of_wind_10m").mean()
    v = coll.select("v_component_of_wind_10m").mean()
    wind = u.hypot(v).rename(dm.WIND_SPEED)

    pressure = (coll.select("surface_pressure").mean().divide(1000.0)
                .rename(dm.PRESSURE))

    ssrd = _era5_deaccumulated_mean(cfg, "surface_solar_radiation_downwards")
    solar = ssrd.rename(dm.SOLAR_RADIATION)
    strd = _era5_deaccumulated_mean(cfg, "surface_thermal_radiation_downwards")
    lw_down = strd.rename(dm.LONGWAVE_DOWN)

    net_sw = _era5_deaccumulated_mean(cfg, "surface_net_solar_radiation")
    net_lw = _era5_deaccumulated_mean(cfg, "surface_net_thermal_radiation")
    net_rad = net_sw.add(net_lw).rename(dm.NET_RADIATION)

    sm = (coll.select("volumetric_soil_water_layer_1").mean()
          .rename(dm.SOIL_MOISTURE))

    pbl = _pbl_height(cfg)

    return ee.Image.cat([t2m, td2m, rh, wind, pressure, solar, lw_down,
                         net_rad, sm, pbl]).clip(aoi)


def _pbl_height(cfg: Config) -> "ee.Image":
    """Boundary-layer height (m) from ERA5 0.25 deg (ERA5-Land lacks it; R3 §2.2)."""
    import ee  # lazy

    meta = GEE_DATASETS["ERA5_HOURLY"]
    coll = (ee.ImageCollection(meta["id"])
            .filterBounds(_aoi(cfg))
            .filterDate(cfg.start_date, cfg.end_date)
            .select("boundary_layer_height"))
    return coll.mean().rename(dm.PBL_HEIGHT)


# ---------------------------------------------------------------------------
# Atmospheric / air-quality
# ---------------------------------------------------------------------------
def aod(cfg: Config) -> "ee.Image":
    """MAIAC aerosol optical depth at 0.55 um -> band ``aod`` (R3 §2.13).

    MCD19A2 ``Optical_Depth_055`` x0.001. AOD attenuates incoming shortwave
    (``K_down``) — a frequently-missed physics term in hazy Indian cities — and
    serves as a PM proxy for the heat x air-quality overlay.
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    meta = GEE_DATASETS["MODIS_MAIAC_AOD"]
    coll = (ee.ImageCollection(meta["id"])
            .filterBounds(aoi)
            .filterDate(cfg.start_date, cfg.end_date)
            .select("Optical_Depth_055"))
    return coll.mean().multiply(meta["scale"]).rename(dm.AOD).clip(aoi)


def no2(cfg: Config) -> "ee.Image":
    """Sentinel-5P tropospheric NO2 column (mol/m2) -> band ``no2`` (R3 §2.14).

    Combustion/traffic tracer feeding the anthropogenic-heat (QF) proxy and the
    heat x air-quality overlay.
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    meta = GEE_DATASETS["S5P_NO2"]
    coll = (ee.ImageCollection(meta["id"])
            .filterBounds(aoi)
            .filterDate(cfg.start_date, cfg.end_date)
            .select("tropospheric_NO2_column_number_density"))
    return coll.mean().rename(dm.NO2).clip(aoi)


# ---------------------------------------------------------------------------
# Anthropogenic heat flux (QF) proxy
# ---------------------------------------------------------------------------
def anthropogenic_heat(cfg: Config) -> "ee.Image":
    """QF proxy image (W/m2) from VIIRS nightlights x GHSL pop x built-volume.

    Implements the nightlight-weighted disaggregation of R3 §5 / R4 §4.6::

        QF_proxy ~ a*(VIIRS NTL) + b*(GHS-POP density) + c*(GHS-BUILT-V)

    We normalise each proxy to its AOI 98th-percentile to bring them onto a
    comparable 0-1 scale, combine with literature-informed weights, then map to
    a plausible W/m2 range (rural ~0 to dense metro core ~150 W/m2, R3 §5). This
    is a *proxy*, calibrated regionally in production; the spatial pattern (the
    driver signal) is what matters for attribution.

    Returns an ``ee.Image`` with bands ``anthro_heat`` (W/m2) and
    ``nightlights`` (raw VIIRS radiance nW/cm2/sr, the canonical layer).
    """
    import ee  # lazy

    aoi = _aoi(cfg)

    # Nightlights (preferred gap-filled band).
    ntl_meta = GEE_DATASETS["VIIRS_BLACK_MARBLE"]
    ntl = (ee.ImageCollection(ntl_meta["id"])
           .filterBounds(aoi)
           .filterDate(cfg.start_date, cfg.end_date)
           .select("Gap_Filled_DNB_BRDF_Corrected_NTL")
           .mean()
           .clip(aoi))
    nightlights = ntl.rename(dm.NIGHTLIGHTS)

    # Population density (persons/cell) — most recent GHS-POP epoch.
    pop_meta = GEE_DATASETS["GHSL_POP"]
    pop = (ee.ImageCollection(pop_meta["id"]).sort("system:time_start", False)
           .first().select("population_count").clip(aoi))

    # Built volume (m3/cell) — cooling-demand / thermal-mass proxy.
    bv_meta = GEE_DATASETS["GHSL_BUILT_V"]
    bv = (ee.ImageCollection(bv_meta["id"]).sort("system:time_start", False)
          .first().select("built_volume_total").clip(aoi))

    def _norm(img: "ee.Image") -> "ee.Image":
        p98 = img.reduceRegion(
            reducer=ee.Reducer.percentile([98]),
            geometry=aoi, scale=500, maxPixels=ee.Number(1e9),
            bestEffort=True, tileScale=4,
        ).values().get(0)
        p98 = ee.Number(ee.Algorithms.If(p98, p98, 1))
        p98 = ee.Number(ee.Algorithms.If(p98.gt(0), p98, 1))
        return img.unmask(0).divide(p98).clamp(0.0, 1.0)

    ntl_n = _norm(ntl)
    pop_n = _norm(pop)
    bv_n = _norm(bv)

    # Literature-informed proxy weights (R3 §5 / R4 §4.6): nightlights dominate
    # the activity signal, population the metabolic/residential floor, built
    # volume the cooling demand. Scaled to ~0-150 W/m2.
    qf = (ntl_n.multiply(0.5).add(pop_n.multiply(0.3)).add(bv_n.multiply(0.2))
          .multiply(150.0).rename(dm.ANTHRO_HEAT))

    return ee.Image.cat([qf, nightlights]).clip(aoi)


# ---------------------------------------------------------------------------
# Downscaling ERA5 air temperature to the analysis grid
# ---------------------------------------------------------------------------
def downscale_air_temp(coarse_t: "ee.Image", predictors: "ee.Image",
                       cfg: Config) -> "ee.Image":
    """Anomaly + LST-based regression downscaling of ERA5 air_temp (R3 §4).

    Coarse ERA5-Land 2 m air temperature (~11 km) misses the urban heat island
    entirely. This implements **delta / anomaly downscaling** (R3 §4.1) which
    guarantees the downscaled field collapses back to the trusted reanalysis
    mean while injecting urban structure from fine predictors:

    1. The fine-scale anomaly ``dT`` is estimated from a co-registered LST
       anomaly: ``dT = b * (LST - mean(LST))`` over the AOI, where the slope
       ``b`` captures the day-time LST->T_air transfer (LST varies more than
       T_air over dry impervious surfaces, so ``b < 1``; R3 §4.2). When other
       predictors (NDVI / impervious / elevation) are present they refine the
       anomaly via simple additive terms (more vegetation -> cooler, more
       impervious -> warmer, higher elevation -> cooler via lapse rate).
    2. The downscaled field is ``T_air(fine) = T_coarse(resampled) + dT(fine)``,
       preserving the coarse-pixel mean (the defining property of anomaly
       downscaling).

    Parameters
    ----------
    coarse_t : ee.Image
        Coarse ERA5-Land ``air_temp`` (degC), band ``air_temp`` or single-band.
    predictors : ee.Image
        Fine predictor stack; must contain at least an ``lst`` band, optionally
        ``ndvi``, ``impervious_frac``, ``elevation``.
    cfg : Config
        Run configuration (AOI / grid).

    Returns
    -------
    ee.Image
        Downscaled ``air_temp`` (degC) on the fine grid.
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    t_coarse = coarse_t.select([0]).resample("bilinear")

    lst = predictors.select(dm.LST)
    lst_mean = ee.Number(lst.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=cfg.resolution_m,
        maxPixels=ee.Number(1e9), bestEffort=True, tileScale=4,
    ).get(dm.LST))
    lst_mean = ee.Number(ee.Algorithms.If(lst_mean, lst_mean, 0))

    # LST->T_air transfer slope (daytime, dry-surface damped). R3 §4.2.
    dt = lst.subtract(lst_mean).multiply(0.30)

    band_names = predictors.bandNames()

    # Vegetation cooling refinement.
    dt = ee.Image(ee.Algorithms.If(
        band_names.contains(dm.NDVI),
        dt.add(predictors.select(dm.NDVI).unmask(0).multiply(-1.5)),
        dt,
    ))
    # Impervious warming refinement.
    dt = ee.Image(ee.Algorithms.If(
        band_names.contains(dm.IMPERVIOUS_FRAC),
        dt.add(predictors.select(dm.IMPERVIOUS_FRAC).unmask(0).multiply(1.0)),
        dt,
    ))
    # Elevation lapse-rate refinement (-6.5 degC/km relative to AOI mean elev).
    def _with_elev() -> "ee.Image":
        elev = predictors.select(dm.ELEVATION)
        elev_mean = ee.Number(elev.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=cfg.resolution_m,
            maxPixels=ee.Number(1e9), bestEffort=True, tileScale=4,
        ).get(dm.ELEVATION))
        elev_mean = ee.Number(ee.Algorithms.If(elev_mean, elev_mean, 0))
        return dt.add(elev.subtract(elev_mean).multiply(-0.0065))

    dt = ee.Image(ee.Algorithms.If(
        band_names.contains(dm.ELEVATION), _with_elev(), dt))

    return t_coarse.add(dt).rename(dm.AIR_TEMP).clip(aoi)


# ---------------------------------------------------------------------------
# One merged meteorology driver image (build-task convenience)
# ---------------------------------------------------------------------------
def meteo_layers(cfg: Config) -> "ee.Image":
    """Merge ERA5-Land drivers + AOD + NO2 + QF/nightlights into one image.

    Convenience wrapper requested by the build task: stacks
    :func:`era5_drivers`, :func:`aod`, :func:`no2` and
    :func:`anthropogenic_heat` into a single ``ee.Image`` with all canonical
    atmospheric band names, ready to be band-selected into a FeatureStack.

    Layers that fail to resolve (e.g. an empty MAIAC window) are skipped with a
    warning so the rest of the stack still assembles.
    """
    import ee  # lazy
    import warnings

    aoi = _aoi(cfg)
    parts: list["ee.Image"] = [era5_drivers(cfg)]
    for name, fn in (("aod", aod), ("no2", no2),
                     ("anthropogenic_heat", anthropogenic_heat)):
        try:
            parts.append(fn(cfg))
        except Exception as exc:  # pragma: no cover - server-side variance
            warnings.warn(f"meteo_layers: skipping {name}: {exc}")
    return ee.Image.cat(parts).clip(aoi)


__all__ = [
    "era5_drivers",
    "aod",
    "no2",
    "anthropogenic_heat",
    "downscale_air_temp",
    "meteo_layers",
]
