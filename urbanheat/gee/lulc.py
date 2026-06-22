"""urbanheat.gee.lulc — server-side LULC / vegetation / surface-radiative DRIVER layers.

This module is the Google Earth Engine **driver factory** for the *surface*
energy-balance terms that set Land Surface Temperature (LST):

* **Vegetation / spectral indices** — NDVI, EVI, SAVI, NDWI, MNDWI, NDBI, NDBaI,
  UI, FVC, LAI from Sentinel-2 SR (R2 §5 formulas, coefficients from
  :data:`urbanheat.constants.SPECTRAL_INDEX_COEFFS`). These drive the latent-heat
  cooling term ``LE`` (vegetation/water) and the storage term ``G`` (built/bare).
* **Broadband albedo** — Bonafoni & Sekertekin (2020) narrow-to-broadband from
  Sentinel-2 and Liang (2001) from Landsat-8 OLI, plus the BRDF-corrected
  MCD43A3 reference. Albedo sets the absorbed shortwave ``(1-alpha)K_down`` — the
  dominant daytime heating input and the most quantifiable cooling lever.
* **Emissivity** — NDVI-threshold (Sobrino) + the static ASTER GED climatology.
  Emissivity controls longwave emission ``eps*sigma*Ts^4`` (nocturnal cooling).
* **LULC** — Dynamic World probabilities + ESA WorldCover class map, and the
  derived continuous fractions ``impervious_frac/green_frac/water_frac/tree_frac``.
* **Evapotranspiration** — MOD16A2GF / PML_V2 (latent-heat cooling made explicit).

All band names match the canonical FeatureStack variable names in
:mod:`urbanheat.datamodel`, so the assembled :func:`spectral_indices` /
:func:`fractional_cover` images can be band-selected straight into a FeatureStack.

Compute philosophy (R2 §6, R7): every ``ee.*`` object is a *handle to a server-side
computation*; we ``filterBounds().filterDate() -> map(scale+mask) -> reduce`` and
only the reduced result ever crosses the wire. Heavy ``ee`` / ``geemap`` imports
are **lazy** (inside functions) so this module imports with numpy only — the
offline synthetic path never needs Earth Engine installed.

Public contract (ARCHITECTURE.md §11.2 ``gee/lulc.py``)::

    fractional_cover(cfg) -> ee.Image  (impervious_frac, green_frac, water_frac, tree_frac)
    spectral_indices(cfg) -> ee.Image  (ndvi, evi, savi, ndwi, mndwi, ndbi, ndbai, ui, fvc, lai, albedo, emissivity)
    lcz(cfg)             -> ee.Image   (lcz)

plus the explicit physics helpers requested by the build task: :func:`albedo`,
:func:`albedo_mcd43a3`, :func:`emissivity`, :func:`landcover`, :func:`et`.

References: research/02_lulc_vegetation_surface.md (§5 formulas, §6 fusion).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from urbanheat.config import Config
from urbanheat.constants import (
    GEE_DATASETS,
    SPECTRAL_INDEX_COEFFS,
)
from urbanheat import datamodel as dm

if TYPE_CHECKING:  # pragma: no cover - hints only, never imported at runtime
    import ee


# ---------------------------------------------------------------------------
# Internal helpers (all lazy-import ``ee`` themselves or receive ee objects)
# ---------------------------------------------------------------------------
def _aoi(cfg: Config) -> "ee.Geometry":
    """Return an ``ee.Geometry.Rectangle`` for the config bbox (EPSG:4326)."""
    import ee  # lazy

    xmin, ymin, xmax, ymax = cfg.bbox
    return ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], proj="EPSG:4326",
                                 geodesic=False)


def _s2_surface_reflectance(cfg: Config) -> "ee.Image":
    """Cloud-masked, scaled Sentinel-2 SR median composite over the AOI.

    Returns an ``ee.Image`` whose B2/B3/B4/B8/B8A/B11/B12 bands are physical
    reflectance (0-1), i.e. the raw DN already multiplied by the
    ``S2_SR_HARMONIZED`` scale (0.0001). Cloud pixels are masked using
    ``S2_CLOUD_PROBABILITY`` (R2 §3.2, V2). This is the optical engine behind
    all spectral indices and the S2 broadband albedo.
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    s2_id = GEE_DATASETS["S2_SR_HARMONIZED"]["id"]
    cloud_id = GEE_DATASETS["S2_CLOUD_PROBABILITY"]["id"]
    s2_scale = GEE_DATASETS["S2_SR_HARMONIZED"]["scale"]
    bands = GEE_DATASETS["S2_SR_HARMONIZED"]["bands"]

    s2 = (ee.ImageCollection(s2_id)
          .filterBounds(aoi)
          .filterDate(cfg.start_date, cfg.end_date)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))
    clouds = (ee.ImageCollection(cloud_id)
              .filterBounds(aoi)
              .filterDate(cfg.start_date, cfg.end_date))

    joined = ee.Join.saveFirst("cloud").apply(
        primary=s2,
        secondary=clouds,
        condition=ee.Filter.equals(leftField="system:index",
                                   rightField="system:index"),
    )

    def _mask(img: "ee.Image") -> "ee.Image":
        img = ee.Image(img)
        prob = ee.Image(img.get("cloud")).select("probability")
        cloud_mask = prob.lt(50)
        return (img.select(bands).multiply(s2_scale)
                .updateMask(cloud_mask)
                .copyProperties(img, ["system:time_start"]))

    masked = ee.ImageCollection(joined).map(_mask)
    return masked.median().clip(aoi)


def _safe_div(num: "ee.Image", den: "ee.Image") -> "ee.Image":
    """Normalized-difference style safe division (den==0 -> masked)."""
    return num.divide(den)


# ---------------------------------------------------------------------------
# Spectral indices (NDVI/EVI/SAVI/NDWI/MNDWI/NDBI/NDBaI/UI/FVC/LAI)
# ---------------------------------------------------------------------------
def spectral_indices(cfg: Config) -> "ee.Image":
    """Compute the full spectral-index + albedo + emissivity driver image (R2 §5).

    Builds, server-side from a Sentinel-2 SR composite, an ``ee.Image`` whose
    bands use the canonical FeatureStack names:

    ``ndvi, evi, savi, ndwi, mndwi, ndbi, ndbai, ui, fvc, lai, albedo,
    emissivity``.

    Formulas (rho = surface reflectance; S2 B2=Blue, B3=Green, B4=Red, B8=NIR,
    B11=SWIR1, B12=SWIR2):

    * NDVI  = (NIR - Red)/(NIR + Red)
    * EVI   = G*(NIR - Red)/(NIR + C1*Red - C2*Blue + L)  [G,C1,C2,L from coeffs]
    * SAVI  = (1+L)*(NIR - Red)/(NIR + Red + L)            [L from coeffs]
    * NDWI  = (Green - NIR)/(Green + NIR)                  (McFeeters open water)
    * MNDWI = (Green - SWIR1)/(Green + SWIR1)              (built-up robust water)
    * NDBI  = (SWIR1 - NIR)/(SWIR1 + NIR)                  (built-up)
    * NDBaI = (SWIR1 - SWIR2)/(SWIR1 + SWIR2)              (bare soil; SWIR2 proxy
              for the TIR term so this stays an optical, co-registered band)
    * UI    = (SWIR2 - NIR)/(SWIR2 + NIR)                  (urban index)
    * FVC   = clamp(((NDVI - NDVI_soil)/(NDVI_veg - NDVI_soil))^2, 0, 1)
    * LAI   ~ from EVI (Boegh 2002 linear form, 3.618*EVI - 0.118), clamped >=0

    Parameters
    ----------
    cfg : Config
        Run configuration (AOI bbox, date window). Honours ``cfg.bbox`` and the
        analysis window for the S2 composite.

    Returns
    -------
    ee.Image
        Multi-band index image clipped to the AOI. Combine with :func:`albedo`
        and :func:`emissivity` (already merged here) before sampling.
    """
    import ee  # lazy

    sr = _s2_surface_reflectance(cfg)
    blue = sr.select("B2")
    green = sr.select("B3")
    red = sr.select("B4")
    nir = sr.select("B8")
    swir1 = sr.select("B11")
    swir2 = sr.select("B12")

    ndvi = _safe_div(nir.subtract(red), nir.add(red)).rename(dm.NDVI)

    evi_c = SPECTRAL_INDEX_COEFFS["EVI"]
    evi = nir.subtract(red).multiply(evi_c["G"]).divide(
        nir.add(red.multiply(evi_c["C1"]))
        .subtract(blue.multiply(evi_c["C2"]))
        .add(evi_c["L"])
    ).rename(dm.EVI)

    savi_l = SPECTRAL_INDEX_COEFFS["SAVI"]["L"]
    savi = nir.subtract(red).multiply(1.0 + savi_l).divide(
        nir.add(red).add(savi_l)).rename(dm.SAVI)

    ndwi = _safe_div(green.subtract(nir), green.add(nir)).rename(dm.NDWI)
    mndwi = _safe_div(green.subtract(swir1), green.add(swir1)).rename(dm.MNDWI)
    ndbi = _safe_div(swir1.subtract(nir), swir1.add(nir)).rename(dm.NDBI)
    ndbai = _safe_div(swir1.subtract(swir2), swir1.add(swir2)).rename(dm.NDBAI)
    ui = _safe_div(swir2.subtract(nir), swir2.add(nir)).rename(dm.UI)

    # Fractional vegetation cover from NDVI thresholds (Sobrino).
    em = SPECTRAL_INDEX_COEFFS["EMISSIVITY_NDVI"]
    ndvi_soil = em["ndvi_soil"]
    ndvi_veg = em["ndvi_veg"]
    fvc = (ndvi.subtract(ndvi_soil).divide(ndvi_veg - ndvi_soil)
           .clamp(0.0, 1.0).pow(2).rename(dm.FVC))

    # LAI from EVI (Boegh et al. 2002 empirical), clamped non-negative.
    lai = evi.multiply(3.618).subtract(0.118).max(0.0).rename(dm.LAI)

    alb = albedo(cfg, sensor="s2", _sr=sr)
    emis = emissivity(cfg, _ndvi=ndvi)

    return ee.Image.cat([ndvi, evi, savi, ndwi, mndwi, ndbi, ndbai, ui,
                         fvc, lai, alb, emis]).clip(_aoi(cfg))


# ---------------------------------------------------------------------------
# Broadband albedo
# ---------------------------------------------------------------------------
def albedo(cfg: Config, sensor: str = "s2",
           _sr: "ee.Image | None" = None) -> "ee.Image":
    """Broadband shortwave albedo (0-1) from S2 or Landsat narrow-to-broadband.

    Two narrow-to-broadband (NTB) coefficient sets from
    :data:`urbanheat.constants.SPECTRAL_INDEX_COEFFS` (R2 §5):

    * ``sensor='s2'`` -> Bonafoni & Sekertekin (2020) Sentinel-2 form:
      ``alpha = 0.2266*B2 + 0.1236*B3 + 0.1573*B4 + 0.3417*B8 + 0.1170*B11 + 0.0338*B12``
    * ``sensor='landsat'`` -> Liang (2001) OLI shortwave form:
      ``alpha = 0.356*B2 + 0.130*B4 + 0.373*B5 + 0.085*B6 + 0.072*B7 - 0.0018``

    These are top-of-canopy/directional; for SEB use they should be
    bias-corrected to MCD43A3 (see :func:`albedo_mcd43a3`, R2 §6.4) — that
    anchoring is performed in the fusion layer, not here.

    Returns an ``ee.Image`` with a single band named ``albedo`` (canonical),
    clamped to [0, 1].
    """
    import ee  # lazy

    if sensor == "s2":
        sr = _sr if _sr is not None else _s2_surface_reflectance(cfg)
        c = SPECTRAL_INDEX_COEFFS["ALBEDO_S2_BONAFONI"]
        a = (sr.select("B2").multiply(c["B2"])
             .add(sr.select("B3").multiply(c["B3"]))
             .add(sr.select("B4").multiply(c["B4"]))
             .add(sr.select("B8").multiply(c["B8"]))
             .add(sr.select("B11").multiply(c["B11"]))
             .add(sr.select("B12").multiply(c["B12"]))
             .add(c["const"]))
    elif sensor == "landsat":
        sr = _landsat_sr(cfg)
        c = SPECTRAL_INDEX_COEFFS["ALBEDO_LANDSAT_LIANG"]
        a = (sr.select("SR_B2").multiply(c["B2"])
             .add(sr.select("SR_B4").multiply(c["B4"]))
             .add(sr.select("SR_B5").multiply(c["B5"]))
             .add(sr.select("SR_B6").multiply(c["B6"]))
             .add(sr.select("SR_B7").multiply(c["B7"]))
             .add(c["const"]))
    else:
        raise ValueError(f"sensor must be 's2' or 'landsat', got {sensor!r}")

    return a.clamp(0.0, 1.0).rename(dm.ALBEDO).clip(_aoi(cfg))


def _landsat_sr(cfg: Config) -> "ee.Image":
    """Scaled, cloud-masked Landsat-8/9 SR median composite (for Liang albedo)."""
    import ee  # lazy

    aoi = _aoi(cfg)
    out = []
    for key in ("LANDSAT8_L2", "LANDSAT9_L2"):
        meta = GEE_DATASETS[key]
        coll = (ee.ImageCollection(meta["id"])
                .filterBounds(aoi)
                .filterDate(cfg.start_date, cfg.end_date))

        def _scale(img: "ee.Image") -> "ee.Image":
            img = ee.Image(img)
            # SR optical bands: DN*2.75e-05 - 0.2 (BAND_SCALE_OVERRIDES).
            opt = (img.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6",
                               "SR_B7"])
                   .multiply(2.75e-05).add(-0.2))
            # QA_PIXEL cloud/shadow/cirrus bits (1,3,4) per R7 §2.7.
            qa = img.select("QA_PIXEL")
            clear = (qa.bitwiseAnd(1 << 1).eq(0)
                     .And(qa.bitwiseAnd(1 << 3).eq(0))
                     .And(qa.bitwiseAnd(1 << 4).eq(0)))
            return opt.updateMask(clear).copyProperties(img,
                                                        ["system:time_start"])

        out.append(coll.map(_scale))
    merged = ee.ImageCollection(out[0].merge(out[1]))
    return merged.median().clip(aoi)


def albedo_mcd43a3(cfg: Config, which: str = "shortwave") -> "ee.Image":
    """BRDF-corrected MCD43A3 broadband albedo reference (500 m), 0-1.

    Returns the blue-sky-style mean of black-sky (BSA) and white-sky (WSA)
    shortwave albedo (a robust clear-sky surrogate), scaled by the catalog
    factor (x0.001) and clamped to [0, 1]. This is the physically rigorous
    reference used to anchor the finer S2/Landsat albedo (R2 §3.3 S1, §6.4).

    Parameters
    ----------
    which : {'shortwave', 'vis'}
        Broadband window. Defaults to shortwave.
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    meta = GEE_DATASETS["MODIS_MCD43A3"]
    scale = meta["scale"]
    if which == "shortwave":
        bsa, wsa = "Albedo_BSA_shortwave", "Albedo_WSA_shortwave"
    elif which == "vis":
        bsa, wsa = "Albedo_BSA_vis", "Albedo_WSA_vis"
    else:
        raise ValueError(f"which must be 'shortwave' or 'vis', got {which!r}")

    coll = (ee.ImageCollection(meta["id"])
            .filterBounds(aoi)
            .filterDate(cfg.start_date, cfg.end_date)
            .select([bsa, wsa]))
    mean = coll.mean().multiply(scale)
    blue_sky = mean.select(bsa).add(mean.select(wsa)).multiply(0.5)
    return blue_sky.clamp(0.0, 1.0).rename(dm.ALBEDO).clip(aoi)


# ---------------------------------------------------------------------------
# Emissivity
# ---------------------------------------------------------------------------
def emissivity(cfg: Config, _ndvi: "ee.Image | None" = None) -> "ee.Image":
    """Broadband surface emissivity (0-1) via NDVI-threshold + ASTER GED blend.

    Primary (dynamic, per-scene): NDVI-threshold method (Sobrino 2004/2008,
    coefficients from ``SPECTRAL_INDEX_COEFFS['EMISSIVITY_NDVI']``)::

        FVC = clamp(((NDVI - NDVI_soil)/(NDVI_veg - NDVI_soil))^2, 0, 1)
        eps = eps_veg*FVC + eps_soil*(1 - FVC) + d_eps

    where the NDVI-threshold piecewise is captured by clamping FVC: bare-soil
    (NDVI<=NDVI_soil) -> eps_soil, full-veg (NDVI>=NDVI_veg) -> eps_veg, and the
    mixed band interpolates with the cavity term ``d_eps``.

    Secondary (static prior, gap-fill): ASTER GED broadband emissivity. ASTER
    GED stores per-TIR-band emissivity (x0.001); we approximate the broadband
    from bands 13 & 14 (the 10.6-11.3 um window dominating LST) and use it where
    the NDVI path is masked (e.g. cloud-masked optical). This captures the
    static climatology baseline while the NDVI term injects seasonal change
    (R2 §6.5).

    Returns an ``ee.Image`` with a single band ``emissivity`` (canonical).
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    em = SPECTRAL_INDEX_COEFFS["EMISSIVITY_NDVI"]

    ndvi = _ndvi
    if ndvi is None:
        sr = _s2_surface_reflectance(cfg)
        ndvi = _safe_div(sr.select("B8").subtract(sr.select("B4")),
                         sr.select("B8").add(sr.select("B4")))

    fvc = (ndvi.subtract(em["ndvi_soil"])
           .divide(em["ndvi_veg"] - em["ndvi_soil"]).clamp(0.0, 1.0))
    eps_ndvi = (fvc.multiply(em["eps_veg"])
                .add(fvc.multiply(-1).add(1).multiply(em["eps_soil"]))
                .add(em["d_eps"]))

    # ASTER GED broadband prior from bands 13 & 14 (window channels).
    aster_meta = GEE_DATASETS["ASTER_GED"]
    aster = ee.Image(aster_meta["id"]).clip(aoi)
    aster_scale = aster_meta["scale"]
    eps_aster = (aster.select("emissivity_band13").multiply(aster_scale)
                 .add(aster.select("emissivity_band14").multiply(aster_scale))
                 .multiply(0.5))

    eps = eps_ndvi.unmask(eps_aster).clamp(0.9, 1.0).rename(dm.EMISSIVITY)
    return eps.clip(aoi)


# ---------------------------------------------------------------------------
# LULC: class maps + continuous fractions
# ---------------------------------------------------------------------------
def _dynamic_world_mean(cfg: Config) -> "ee.Image":
    """Seasonal-mean Dynamic World per-class probability image over the AOI.

    Mean probability of each class over the analysis window approximates a
    stable sub-pixel fractional cover (R2 §3.1 L3): ``mean(built)`` ~ impervious
    fraction, ``mean(trees+grass+shrub)`` ~ green fraction, etc.
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    meta = GEE_DATASETS["DYNAMIC_WORLD"]
    prob_bands = ["built", "trees", "grass", "water", "crops",
                  "shrub_and_scrub", "bare", "flooded_vegetation",
                  "snow_and_ice"]
    coll = (ee.ImageCollection(meta["id"])
            .filterBounds(aoi)
            .filterDate(cfg.start_date, cfg.end_date)
            .select(prob_bands))
    return coll.mean().clip(aoi)


def landcover(cfg: Config, product: str = "dynamic_world") -> "ee.Image":
    """Discrete LULC class image over the AOI -> band ``lulc`` (R2 §3.1).

    Parameters
    ----------
    product : {'dynamic_world', 'worldcover'}
        * ``'dynamic_world'`` -> argmax of the seasonal-mean Dynamic World class
          probabilities (NRT 10 m, 9 classes 0-8 matching the ``label`` band).
        * ``'worldcover'`` -> ESA WorldCover v200 ``Map`` (11 classes, built=50).

    Returns an ``ee.Image`` band ``lulc`` (integer class codes).
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    if product == "dynamic_world":
        prob = _dynamic_world_mean(cfg)
        # argmax over the 9 probability bands -> hard class (DW label scheme).
        label = prob.toArray().arrayArgmax().arrayGet([0])
        return label.rename(dm.LULC).clip(aoi)
    if product == "worldcover":
        meta = GEE_DATASETS["ESA_WORLDCOVER_V200"]
        wc = ee.ImageCollection(meta["id"]).first().select("Map")
        return wc.rename(dm.LULC).clip(aoi)
    raise ValueError(
        f"product must be 'dynamic_world' or 'worldcover', got {product!r}")


def fractional_cover(cfg: Config) -> "ee.Image":
    """Continuous impervious/green/water/tree fractions (0-1) over the AOI.

    Ensemble fractional cover (R2 §6.1-6.2) co-registered to the AOI, primarily
    from Dynamic World seasonal-mean probabilities, blended with ESA WorldCover
    aggregated fractions and the GHSL built-surface fraction (the physically
    grounded impervious term). Returns an ``ee.Image`` with the canonical bands:

    * ``impervious_frac`` = mean(GHSL built fraction, DW built prob)  [lambda_P]
    * ``green_frac``      = DW (trees + grass + shrub_and_scrub + crops)
    * ``water_frac``      = DW water (+ flooded_vegetation)
    * ``tree_frac``       = DW trees

    All bands are clamped to [0, 1].
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    dw = _dynamic_world_mean(cfg)

    dw_built = dw.select("built")
    green = (dw.select("trees").add(dw.select("grass"))
             .add(dw.select("shrub_and_scrub")).add(dw.select("crops"))
             .clamp(0.0, 1.0).rename(dm.GREEN_FRAC))
    water = (dw.select("water").add(dw.select("flooded_vegetation"))
             .clamp(0.0, 1.0).rename(dm.WATER_FRAC))
    tree = dw.select("trees").clamp(0.0, 1.0).rename(dm.TREE_FRAC)

    # GHSL built surface fraction = built_m2 / cell_area (10 m -> area 100 m^2).
    ghsl_meta = GEE_DATASETS["GHSL_BUILT_S_10M"]
    ghsl = ee.Image(ghsl_meta["id"]).select("built_surface").clip(aoi)
    cell_area = ee.Image.pixelArea()
    ghsl_frac = ghsl.divide(cell_area).clamp(0.0, 1.0)

    impervious = (ghsl_frac.add(dw_built).multiply(0.5)
                  .clamp(0.0, 1.0).rename(dm.IMPERVIOUS_FRAC))

    return ee.Image.cat([impervious, green, water, tree]).clip(aoi)


# ---------------------------------------------------------------------------
# Evapotranspiration (latent-heat cooling term)
# ---------------------------------------------------------------------------
def et(cfg: Config, product: str = "MODIS_MOD16A2GF") -> "ee.Image":
    """Evapotranspiration driver image (mm/period) -> band ``et`` (R2 §6.6).

    Parameters
    ----------
    product : {'MODIS_MOD16A2GF', 'PML_V2'}
        * ``'MODIS_MOD16A2GF'`` -> Penman-Monteith gap-filled ET band (x0.1 ->
          kg/m2/8day).
        * ``'PML_V2'`` -> partitioned transpiration+soil+interception
          (Ec+Es+Ei), mm/8day.

    Returns an ``ee.Image`` band ``et``. ET makes the latent-heat cooling term
    ``LE`` explicit: high ET == strong evaporative cooling == lower LST.
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    meta = GEE_DATASETS[product]
    coll = (ee.ImageCollection(meta["id"])
            .filterBounds(aoi)
            .filterDate(cfg.start_date, cfg.end_date))
    if product == "MODIS_MOD16A2GF":
        out = coll.select("ET").mean().multiply(meta["scale"])
    elif product == "PML_V2":
        out = (coll.select(["Ec", "Es", "Ei"]).mean()
               .reduce(ee.Reducer.sum()).multiply(meta["scale"]))
    else:
        raise ValueError(
            f"product must be 'MODIS_MOD16A2GF' or 'PML_V2', got {product!r}")
    return out.rename(dm.ET).clip(aoi)


# ---------------------------------------------------------------------------
# LCZ (Local Climate Zones)
# ---------------------------------------------------------------------------
def lcz(cfg: Config) -> "ee.Image":
    """Clip the global LCZ map to the AOI -> ``ee.Image`` band ``lcz`` (R4 §5).

    Uses the RUB/RUBCLIM global Local Climate Zone map (``LCZ_Filter`` band, 17
    classes, 100 m). The class codes seed morphology priors (SVF, H/W, lambda_P,
    z0) via :data:`urbanheat.constants.LCZ_TABLE` in the morphology module.
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    meta = GEE_DATASETS["GLOBAL_LCZ"]
    coll = ee.ImageCollection(meta["id"]).filterBounds(aoi)
    img = ee.Image(coll.mosaic()).select("LCZ_Filter")
    return img.rename(dm.LCZ).clip(aoi)


__all__ = [
    "spectral_indices",
    "albedo",
    "albedo_mcd43a3",
    "emissivity",
    "landcover",
    "fractional_cover",
    "et",
    "lcz",
]
