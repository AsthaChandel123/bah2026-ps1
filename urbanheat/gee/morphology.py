"""urbanheat.gee.morphology — server-side urban-morphology / 3D-form DRIVER layers.

The *geometry* that controls the urban-canopy energy balance and turns a flat
thermal map into a physics-informed one (R4 §1):

* **building_height** (m) — GHSL GHS_BUILT_H ANBH and the independent
  ``DSM - DEM = GLO-30 - FABDEM`` object height (R4 §3, §6.3).
* **building_volume** (m3/cell) — GHS_BUILT_V (thermal mass -> nocturnal UHI).
* **plan_area_frac** (lambda_P) and **frontal_area_index** (lambda_F) — building
  plan density and wall-area-facing-wind drag from GHSL / Open Buildings (R4 §4.1-4.2).
* **sky_view_factor** (SVF, 0-1) — analytical canyon approximation from H/W plus
  an LCZ-class fallback (R4 §4.3).
* **aspect_ratio** (H/W) — canyon trapping/ventilation (R4 §4.3).
* **roughness_length** (z0) & **displacement_height** (zd) — Macdonald (1998)
  morphometric form, constants from :data:`urbanheat.constants.MACDONALD_CONSTANTS`
  (R4 §4.5).
* **elevation** (m) & **slope** (deg) — DEM (R4 §3.7).
* **lcz** — RUB global Local Climate Zones.
* **population** (persons/cell, GHS-POP) and **nightlights** (VIIRS) ->
  **anthro_heat** proxy (R4 §4.6).

Band names match the canonical FeatureStack variable names in
:mod:`urbanheat.datamodel`. All ``ee`` imports are lazy so the module imports
with numpy only.

Public contract (ARCHITECTURE.md §11.2 ``gee/morphology.py``)::

    building_height(cfg) -> ee.Image  (building_height)
    sky_view_factor(cfg) -> ee.Image  (svf)
    morphometrics(cfg)   -> ee.Image  (plan_area_frac, frontal_area_index,
                                       aspect_ratio, roughness_length,
                                       displacement_height, building_volume,
                                       elevation, slope)

plus the build-task helpers :func:`building_volume`, :func:`plan_area_frac`,
:func:`frontal_area_index`, :func:`aspect_ratio`, :func:`roughness_length`,
:func:`elevation`, :func:`lcz`, :func:`population`, :func:`nightlights`,
:func:`anthropogenic_heat_proxy`.

References: research/04_urban_morphology_3d.md (§4 formulas, §5 LCZ table, §6 fusion).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from urbanheat.config import Config
from urbanheat.constants import (
    GEE_DATASETS,
    LCZ_TABLE,
    MACDONALD_CONSTANTS,
    VON_KARMAN,
)
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


def _lcz_image(cfg: Config) -> "ee.Image":
    """Raw RUB global LCZ ``LCZ_Filter`` image clipped to the AOI."""
    import ee  # lazy

    meta = GEE_DATASETS["GLOBAL_LCZ"]
    coll = ee.ImageCollection(meta["id"]).filterBounds(_aoi(cfg))
    return ee.Image(coll.mosaic()).select("LCZ_Filter").clip(_aoi(cfg))


def _lcz_lookup(cfg: Config, key: str, default: float) -> "ee.Image":
    """Remap the LCZ class image to a per-class numeric property from LCZ_TABLE.

    Builds an ``ee.Image`` where each LCZ class code (1-17) is replaced by the
    requested LCZ_TABLE property (e.g. ``'svf'``, ``'hw'``, ``'height'``,
    ``'bsf'``, ``'isf'``). Classes not present default to ``default``.
    """
    import ee  # lazy

    lcz = _lcz_image(cfg)
    codes = list(LCZ_TABLE.keys())
    values = [float(LCZ_TABLE[c].get(key, default)) for c in codes]
    return lcz.remap(codes, values, default).rename(key)


# ---------------------------------------------------------------------------
# Building height (GHSL ANBH and DSM - DEM)
# ---------------------------------------------------------------------------
def building_height(cfg: Config, method: str = "cascade") -> "ee.Image":
    """Per-cell building height (m) -> band ``building_height`` (R4 §6.2-6.3).

    Fallback cascade (use the first physically-meaningful estimate, R4 §6.2):

    1. ``DSM - DEM`` object height = ``COPERNICUS/DEM/GLO30`` (DSM, includes
       buildings) minus ``FABDEM`` (bare-earth), clamped >= 0, masked to built
       pixels via the GHSL built-surface fraction so trees are dropped (R4 §6.3).
    2. ``GHSL GHS_BUILT_H`` ANBH (100 m, the Average Net Building Height) as the
       coarse backstop wherever the DSM path is masked.

    Parameters
    ----------
    method : {'cascade', 'ghsl', 'dsm_minus_dem'}
        ``'cascade'`` (default) blends DSM-DEM over built pixels with the GHSL
        ANBH backstop; ``'ghsl'`` returns ANBH only; ``'dsm_minus_dem'`` returns
        the differenced object height only.

    Returns an ``ee.Image`` band ``building_height`` (m).
    """
    import ee  # lazy

    aoi = _aoi(cfg)

    # GHSL ANBH backstop.
    ghsl_h_meta = GEE_DATASETS["GHSL_BUILT_H"]
    anbh = (ee.Image(ghsl_h_meta["id"]).select("built_height")
            .clip(aoi))

    if method == "ghsl":
        return anbh.max(0.0).rename(dm.BUILDING_HEIGHT)

    # DSM - DEM object height.
    glo30_meta = GEE_DATASETS["COPERNICUS_DEM_GLO30"]
    fabdem_meta = GEE_DATASETS["FABDEM"]
    dsm = ee.ImageCollection(glo30_meta["id"]).select("DEM").mosaic()
    dem = ee.ImageCollection(fabdem_meta["id"]).select("b1").mosaic()
    obj_h = dsm.subtract(dem).max(0.0)

    # Built mask from GHSL built-surface fraction (drop vegetation; R4 §6.3).
    built_meta = GEE_DATASETS["GHSL_BUILT_S_10M"]
    built = ee.Image(built_meta["id"]).select("built_surface")
    built_frac = built.divide(ee.Image.pixelArea())
    built_mask = built_frac.gt(0.1)
    bldg_dsm = obj_h.updateMask(built_mask).clip(aoi)

    if method == "dsm_minus_dem":
        return bldg_dsm.rename(dm.BUILDING_HEIGHT)
    if method != "cascade":
        raise ValueError(
            f"method must be 'cascade'|'ghsl'|'dsm_minus_dem', got {method!r}")

    # Cascade: DSM-DEM where available, else GHSL ANBH.
    return bldg_dsm.unmask(anbh).max(0.0).rename(dm.BUILDING_HEIGHT).clip(aoi)


def building_volume(cfg: Config) -> "ee.Image":
    """Built volume (m3/cell) from GHS_BUILT_V -> band ``building_volume`` (R4 §3).

    Thermal-mass / nocturnal-UHI proxy (the storage flux ``G``): high built
    volume -> large heat-release lag -> warmer nights.
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    meta = GEE_DATASETS["GHSL_BUILT_V"]
    bv = (ee.ImageCollection(meta["id"]).sort("system:time_start", False)
          .first().select("built_volume_total").clip(aoi))
    return bv.max(0.0).rename(dm.BUILDING_VOLUME)


# ---------------------------------------------------------------------------
# Plan / frontal area indices (lambda_P, lambda_F)
# ---------------------------------------------------------------------------
def plan_area_frac(cfg: Config) -> "ee.Image":
    """Plan area fraction lambda_P (0-1) -> band ``plan_area_frac`` (R4 §4.1).

    ``lambda_P = (sum building footprint area) / cell area``, taken directly
    from the GHSL built-surface fraction (``built_surface`` m2 / cell area),
    which is the physically-grounded plan/impervious density. Controls daytime
    trapping and heat storage; enters z0/zd.
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    meta = GEE_DATASETS["GHSL_BUILT_S_10M"]
    built = ee.Image(meta["id"]).select("built_surface").clip(aoi)
    lam_p = built.divide(ee.Image.pixelArea()).clamp(0.0, 1.0)
    return lam_p.rename(dm.PLAN_AREA_FRAC)


def frontal_area_index(cfg: Config, lam_p: "ee.Image | None" = None,
                       height: "ee.Image | None" = None) -> "ee.Image":
    """Frontal area index lambda_F (0-1) -> band ``frontal_area_index`` (R4 §4.2).

    ``lambda_F(theta) = (sum H_i * W_i(theta)) / A_T`` is the wall area facing
    the wind per unit ground area — the ventilation/drag variable controlling
    z0. Without per-building footprint geometry server-side, we use the standard
    morphometric estimate that for roughly isotropic, square-plan blocks the
    frontal area equals the plan area times the height-to-block-width ratio,
    which reduces to ``lambda_F ~ lambda_P * H / L`` with a representative block
    width ``L``. Using ``L ~ sqrt(cell_area) * (1 - lambda_P)`` style spacing is
    fragile, so we adopt the widely-used closure ``lambda_F = lambda_P * H / Hc``
    with a canopy reference making lambda_F scale with both plan density and
    height. Concretely ``lambda_F = clamp(lambda_P * (H / (H + W0)), 0, 1)`` with
    ``W0 = 10 m`` a nominal street width — monotone in both H and lambda_P, which
    is the property the roughness model needs.

    Parameters
    ----------
    lam_p : ee.Image, optional
        Precomputed lambda_P (band ``plan_area_frac``); computed if None.
    height : ee.Image, optional
        Precomputed building height (band ``building_height``); computed if None.

    Returns an ``ee.Image`` band ``frontal_area_index``.
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    lp = lam_p if lam_p is not None else plan_area_frac(cfg)
    lp = lp.select([0])
    h = height if height is not None else building_height(cfg)
    h = h.select([0])

    w0 = 10.0  # nominal street width (m)
    lam_f = lp.multiply(h.divide(h.add(w0))).clamp(0.0, 1.0)
    return lam_f.rename(dm.FRONTAL_AREA_INDEX).clip(aoi)


# ---------------------------------------------------------------------------
# Sky view factor & aspect ratio
# ---------------------------------------------------------------------------
def aspect_ratio(cfg: Config, lam_p: "ee.Image | None" = None,
                 height: "ee.Image | None" = None) -> "ee.Image":
    """Canyon aspect ratio H/W -> band ``aspect_ratio`` (R4 §4.3).

    Derived from plan density and height: with plan fraction ``lambda_P`` the
    canyon (street) width fraction is ``(1 - lambda_P)``, so a representative
    street width is ``W ~ L*(1 - lambda_P)`` for a block pitch ``L`` and the
    aspect ratio ``H/W = H / (L*(1 - lambda_P))``. Using the canopy-scale pitch
    ``L ~ H/lambda_P`` (so denser, taller cores have proportionally narrower
    canyons) collapses to the standard closure ``H/W ~ lambda_P/(1 - lambda_P)``
    scaled by height; we use the robust monotone form
    ``H/W = lambda_P / max(1 - lambda_P, 0.05)`` then modulate by normalized
    height so two equally-dense areas differ when one is taller. Falls back to
    the LCZ-class ``hw`` prior where building data is missing.
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    lp = (lam_p if lam_p is not None else plan_area_frac(cfg)).select([0])
    h = (height if height is not None else building_height(cfg)).select([0])

    # Geometric closure from plan density, modulated by height (taller -> higher
    # H/W for the same density). Normalize height by a 30 m reference.
    hw_geom = lp.divide(lp.multiply(-1).add(1).max(0.05)).multiply(
        h.divide(30.0).add(0.5)).max(0.0)

    hw_lcz = _lcz_lookup(cfg, "hw", 0.5)
    # Use geometric where built data is present, else LCZ prior.
    hw = hw_geom.unmask(hw_lcz).rename(dm.ASPECT_RATIO)
    return hw.clip(aoi)


def sky_view_factor(cfg: Config, hw: "ee.Image | None" = None) -> "ee.Image":
    """Sky view factor (0-1) -> band ``svf`` (R4 §4.3).

    Analytical infinite-symmetric-canyon approximation from the aspect ratio
    (R4 §4.3)::

        SVF_canyon = cos(arctan(2 * H/W)) = 1 / sqrt(1 + (2*H/W)^2)

    SVF is the master nighttime-UHI variable: low SVF (deep canyon) traps
    upwelling longwave -> slow nocturnal cooling -> strong UHI. Where building
    geometry is unavailable the SVF falls back to the LCZ-class ``svf`` prior
    (the standard gap-fill bridge, R4 §5). A full raster hillshade-sweep over
    the DSM (N azimuths) is the higher-fidelity alternative; the analytical form
    is used here for O(1) server-side scalability.

    Parameters
    ----------
    hw : ee.Image, optional
        Precomputed aspect ratio (band ``aspect_ratio``); computed if None.
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    h_w = (hw if hw is not None else aspect_ratio(cfg)).select([0])

    two_hw = h_w.multiply(2.0)
    svf_canyon = ee.Image(1.0).divide(two_hw.pow(2).add(1.0).sqrt())

    svf_lcz = _lcz_lookup(cfg, "svf", 0.9)
    svf = svf_canyon.unmask(svf_lcz).clamp(0.0, 1.0).rename(dm.SVF)
    return svf.clip(aoi)


# ---------------------------------------------------------------------------
# Roughness length & displacement height (Macdonald 1998)
# ---------------------------------------------------------------------------
def roughness_length(cfg: Config, lam_p: "ee.Image | None" = None,
                     lam_f: "ee.Image | None" = None,
                     height: "ee.Image | None" = None,
                     ) -> "ee.Image":
    """Macdonald (1998) roughness z0 + displacement zd -> 2-band image (R4 §4.5).

    Morphometric form (no wind data needed), constants from
    :data:`urbanheat.constants.MACDONALD_CONSTANTS` (A=4.43, Cd=1.2, beta=1.0)
    and von Karman kappa=0.40::

        zd/H = 1 + A^(-lambda_P) * (lambda_P - 1)
        z0/H = (1 - zd/H) * exp{ -[ 0.5*beta*(Cd/kappa^2)*(1 - zd/H)*lambda_F ]^(-0.5) }

    z0 and zd are the aerodynamic inputs any SEB/boundary-layer scheme needs;
    they set how efficiently sensible heat is convected out of the canopy. z0
    rises with lambda_F then falls at very high density (sheltering) — this
    non-monotonicity is captured by the exponential term.

    Returns an ``ee.Image`` with bands ``roughness_length`` (z0, m) and
    ``displacement_height`` (zd, m). Pixels with no building height collapse to
    ~0 (open terrain).
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    lp = (lam_p if lam_p is not None else plan_area_frac(cfg)).select([0])
    h = (height if height is not None else building_height(cfg)).select([0])
    lf = (lam_f if lam_f is not None
          else frontal_area_index(cfg, lam_p=lp, height=h)).select([0])

    a = MACDONALD_CONSTANTS["A"]
    cd = MACDONALD_CONSTANTS["Cd"]
    beta = MACDONALD_CONSTANTS["beta"]
    kappa = VON_KARMAN

    # zd/H = 1 + A^(-lambda_P) * (lambda_P - 1)
    a_pow = ee.Image(a).pow(lp.multiply(-1))
    zd_over_h = a_pow.multiply(lp.subtract(1.0)).add(1.0).clamp(0.0, 0.99)

    # bracket = 0.5*beta*(Cd/kappa^2)*(1 - zd/H)*lambda_F
    coef = 0.5 * beta * (cd / (kappa * kappa))
    bracket = (zd_over_h.multiply(-1).add(1.0)).multiply(lf).multiply(coef)
    # exp(-bracket^(-1/2)); guard bracket>0.
    bracket_safe = bracket.max(1e-6)
    z0_over_h = (zd_over_h.multiply(-1).add(1.0)).multiply(
        bracket_safe.pow(-0.5).multiply(-1).exp())

    zd = zd_over_h.multiply(h).max(0.0).rename(dm.DISPLACEMENT_HEIGHT)
    z0 = z0_over_h.multiply(h).max(0.0).rename(dm.ROUGHNESS_LENGTH)
    # Open terrain (no buildings) -> small z0 floor (~0.03 m grassland).
    z0 = z0.unmask(0.03)
    zd = zd.unmask(0.0)
    return ee.Image.cat([z0, zd]).clip(aoi)


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------
def elevation(cfg: Config) -> "ee.Image":
    """Terrain elevation (m) + slope (deg) -> 2-band image (R4 §3.7).

    Uses the bare-earth NASADEM for clean terrain (slope/aspect, LST
    detrending). Returns bands ``elevation`` (m) and ``slope`` (deg).
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    meta = GEE_DATASETS["NASADEM"]
    dem = ee.Image(meta["id"]).select("elevation").clip(aoi)
    elev = dem.rename(dm.ELEVATION)
    slope = ee.Terrain.slope(dem).rename(dm.SLOPE)
    return ee.Image.cat([elev, slope]).clip(aoi)


# ---------------------------------------------------------------------------
# LCZ, population, nightlights, anthropogenic-heat proxy
# ---------------------------------------------------------------------------
def lcz(cfg: Config) -> "ee.Image":
    """RUB global Local Climate Zones -> band ``lcz`` (17 classes; R4 §5)."""
    return _lcz_image(cfg).rename(dm.LCZ)


def population(cfg: Config) -> "ee.Image":
    """GHS-POP residential population (persons/cell) -> band ``population``.

    Most-recent GHS-POP epoch; the anthropogenic-heat (QF) + exposure proxy
    (R4 §3.1, §4.6).
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    meta = GEE_DATASETS["GHSL_POP"]
    pop = (ee.ImageCollection(meta["id"]).sort("system:time_start", False)
           .first().select("population_count").clip(aoi))
    return pop.max(0.0).rename(dm.POPULATION)


def nightlights(cfg: Config) -> "ee.Image":
    """VIIRS Black Marble nightlights -> band ``nightlights`` (R4 §3.8).

    Mean ``Gap_Filled_DNB_BRDF_Corrected_NTL`` radiance over the window
    (nW/cm2/sr) — the QF / activity proxy.
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    meta = GEE_DATASETS["VIIRS_BLACK_MARBLE"]
    ntl = (ee.ImageCollection(meta["id"])
           .filterBounds(aoi)
           .filterDate(cfg.start_date, cfg.end_date)
           .select("Gap_Filled_DNB_BRDF_Corrected_NTL")
           .mean()
           .clip(aoi))
    return ntl.rename(dm.NIGHTLIGHTS)


def anthropogenic_heat_proxy(cfg: Config) -> "ee.Image":
    """Anthropogenic heat flux (W/m2) proxy -> band ``anthro_heat`` (R4 §4.6).

    Morphology-side QF proxy combining VIIRS nightlights, GHS-POP density and
    GHS_BUILT_V (cooling demand), each normalised to its AOI 98th percentile and
    weighted (nightlights 0.5, population 0.3, built volume 0.2), mapped to a
    plausible ~0-150 W/m2 range. Mirrors :func:`urbanheat.gee.meteo.anthropogenic_heat`
    so either backend module can supply the term.
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    ntl = nightlights(cfg).select([0])
    pop = population(cfg).select([0])
    bv = building_volume(cfg).select([0])

    def _norm(img: "ee.Image") -> "ee.Image":
        p98 = img.reduceRegion(
            reducer=ee.Reducer.percentile([98]), geometry=aoi, scale=500,
            maxPixels=ee.Number(1e9), bestEffort=True, tileScale=4,
        ).values().get(0)
        p98 = ee.Number(ee.Algorithms.If(p98, p98, 1))
        p98 = ee.Number(ee.Algorithms.If(p98.gt(0), p98, 1))
        return img.unmask(0).divide(p98).clamp(0.0, 1.0)

    qf = (_norm(ntl).multiply(0.5).add(_norm(pop).multiply(0.3))
          .add(_norm(bv).multiply(0.2)).multiply(150.0))
    return qf.rename(dm.ANTHRO_HEAT).clip(aoi)


# ---------------------------------------------------------------------------
# All morphometrics in one image (§11 contract)
# ---------------------------------------------------------------------------
def morphometrics(cfg: Config) -> "ee.Image":
    """Assemble the full derived-morphology image (ARCHITECTURE.md §11.2).

    Bands (canonical names): ``plan_area_frac`` (lambda_P),
    ``frontal_area_index`` (lambda_F), ``aspect_ratio`` (H/W),
    ``roughness_length`` (z0), ``displacement_height`` (zd),
    ``building_volume``, ``elevation``, ``slope`` — plus ``building_height`` and
    ``svf`` so the single image carries every morphology driver the
    FeatureStack expects. Reuses intermediate computations (lambda_P, height,
    lambda_F) to avoid recomputing them across the dependent formulas.
    """
    import ee  # lazy

    aoi = _aoi(cfg)
    lp = plan_area_frac(cfg)
    h = building_height(cfg)
    lf = frontal_area_index(cfg, lam_p=lp, height=h)
    hw = aspect_ratio(cfg, lam_p=lp, height=h)
    svf = sky_view_factor(cfg, hw=hw)
    z0zd = roughness_length(cfg, lam_p=lp, lam_f=lf, height=h)
    bv = building_volume(cfg)
    terr = elevation(cfg)

    return ee.Image.cat([lp, lf, hw, svf, z0zd, h, bv, terr]).clip(aoi)


__all__ = [
    "building_height",
    "building_volume",
    "plan_area_frac",
    "frontal_area_index",
    "aspect_ratio",
    "sky_view_factor",
    "roughness_length",
    "elevation",
    "lcz",
    "population",
    "nightlights",
    "anthropogenic_heat_proxy",
    "morphometrics",
]
