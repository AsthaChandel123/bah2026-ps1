"""urbanheat.gee.fusion — server-side multi-sensor LST harmonisation & fusion.

Implements the cross-verification thesis of ``research/01`` §3-§4: bring every
thermal sensor to a **common °C scale**, build **cloud-free composites**
(median / quality-mosaic), combine them with an **uncertainty-weighted ensemble**
(weight each sensor by its known per-pixel error so disagreement is downweighted,
not averaged in), and **temporally gap-fill** residual holes. Everything is an
``ee.Image`` graph — no ``getInfo`` loops, no rasters crossing the wire.

A small pure-numpy :func:`weighted_ensemble` mirrors the server-side ensemble for
the **offline robustness layer** (synthetic path / unit tests); ``numpy`` is the
only top-level import.

Public production entry point
-----------------------------
:func:`fuse_lst_server` ``-> ee.Image`` with bands ``lst`` (fused °C) and
``lst_uncertainty`` (1-σ °C). Also provided: :func:`fuse_lst` (the
ARCHITECTURE.md §11 dict-input signature) and :func:`sharpen_lst` (thermal
sharpening of coarse LST with fine predictors, mass-conserving residual).

All ``ee`` imports are lazy; importing this module needs only ``numpy`` + the
constants/datamodel catalogs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from urbanheat.constants import VALIDATION_ANCHORS
from urbanheat.datamodel import LST, LST_UNCERTAINTY
from urbanheat.gee.auth import ee_geometry

if TYPE_CHECKING:  # hints only
    from urbanheat.config import Config

# Default per-sensor 1-σ LST uncertainty (°C ≈ K) when no per-pixel error band is
# available. Anchored to the literature RMSEs in
# ``constants.VALIDATION_ANCHORS`` where present; others are typical figures from
# research/01 §5 (lower = trust more in the inverse-variance weighting). [R9 §3.1]
DEFAULT_SENSOR_SIGMA: dict[str, float] = {
    "landsat": 1.0,                                            # 30 m SC, ST_QA ~1 K
    "ecostress": VALIDATION_ANCHORS.get("ecostress_rmse_K", 2.2),
    "modis": VALIDATION_ANCHORS.get("mod11_rmse_K", 2.8),
    "modis_tes": 2.5,                                          # MxD21 TES
    "viirs": 2.5,
    "sentinel3": 2.0,
    "default": 2.5,
}


def _import_ee() -> Any:
    from urbanheat.gee.auth import _import_ee as _imp  # noqa: PLC0415

    return _imp()


def _sensor_sigma(name: str) -> float:
    """Resolve a default 1-σ (°C) for a sensor name (substring match)."""
    n = name.lower()
    for key, sig in DEFAULT_SENSOR_SIGMA.items():
        if key != "default" and key in n:
            return float(sig)
    return float(DEFAULT_SENSOR_SIGMA["default"])


# ===========================================================================
# Server-side composites
# ===========================================================================
def cloud_free_composite(collection: Any, reducer: str = "median",
                         clip_to: Any = None) -> Any:
    """Reduce a (pre-masked) ``ee.ImageCollection`` to a cloud-free ``ee.Image``.

    ``reducer`` in {``median``, ``mean``, ``mosaic``/``quality``}. ``median`` is
    the robust default (outlier-resistant temporal composite); ``mosaic`` takes
    the most-recent unmasked pixel. Optionally ``clip_to`` a geometry. [R1 §3.C]
    """
    ee = _import_ee()
    r = reducer.lower()
    if r == "median":
        out = collection.median()
    elif r == "mean":
        out = collection.mean()
    elif r in ("mosaic", "quality", "qualitymosaic"):
        out = collection.mosaic()
    else:
        raise ValueError(f"unknown reducer {reducer!r}")
    if clip_to is not None:
        out = out.clip(clip_to)
    return ee.Image(out)


def harmonize_to_celsius(images: dict[str, Any], band: str | None = None) -> dict[str, Any]:
    """Rename each sensor's LST band to the canonical :data:`datamodel.LST`.

    Inputs are assumed already in °C (the :mod:`urbanheat.gee.lst` helpers return
    °C). This just normalises band names so the ensemble can stack them. If
    ``band`` is given, that band is selected from each image; otherwise the first
    band is used. Returns a new ``{name: ee.Image[lst]}`` dict.
    """
    _import_ee()
    out: dict[str, Any] = {}
    for name, img in images.items():
        sel = img.select(band) if band is not None else img.select(0)
        out[name] = sel.rename(LST)
    return out


# ===========================================================================
# Uncertainty-weighted ensemble (server-side)
# ===========================================================================
def uncertainty_weighted_fusion(
    images: dict[str, Any],
    sigmas: dict[str, Any] | None = None,
) -> Any:
    """Inverse-variance (uncertainty-weighted) ensemble of sensor LSTs.

    Fuses ``{name: ee.Image[°C]}`` as ``Σ(w_i x_i)/Σ w_i`` with
    ``w_i = 1/σ_i²`` (per-pixel where an error image is supplied in ``sigmas``,
    else the sensor default from :data:`DEFAULT_SENSOR_SIGMA`). The fused 1-σ is
    ``sqrt(1/Σ w_i)``. Where only some sensors see a pixel, masking makes the sum
    skip the missing ones automatically — so coverage gaps are handled for free.
    [R1 §4 "ensemble/Bayesian fusion"; R9 §3.1]

    Returns an ``ee.Image`` with bands :data:`datamodel.LST` (fused °C) and
    :data:`datamodel.LST_UNCERTAINTY` (1-σ °C).
    """
    ee = _import_ee()
    if not images:
        raise ValueError("uncertainty_weighted_fusion: `images` must be non-empty")
    sigmas = sigmas or {}

    sum_wx = ee.Image.constant(0.0)
    sum_w = ee.Image.constant(0.0)

    for name, img in images.items():
        x = ee.Image(img).select(0)
        if name in sigmas and sigmas[name] is not None:
            sig = ee.Image(sigmas[name])
        else:
            sig = ee.Image.constant(_sensor_sigma(name))
        # Weight only where the sensor has data; mask weight to the obs footprint.
        w = sig.pow(2).pow(-1).updateMask(x.mask())
        sum_wx = sum_wx.add(x.unmask(0).multiply(w.unmask(0)))
        sum_w = sum_w.add(w.unmask(0))

    valid = sum_w.gt(0)
    fused = sum_wx.divide(sum_w).updateMask(valid).rename(LST)
    unc = sum_w.pow(-1).sqrt().updateMask(valid).rename(LST_UNCERTAINTY)
    return fused.addBands(unc)


# ===========================================================================
# Temporal gap-filling
# ===========================================================================
def temporal_gap_fill(
    primary: Any,
    fallbacks: Sequence[Any] = (),
    smooth_radius_m: float = 0.0,
) -> Any:
    """Fill masked pixels in ``primary`` from ``fallbacks`` (in priority order).

    Server-side cascade: keep ``primary`` where valid, else take the first valid
    ``fallback`` pixel (e.g. a coarser/longer-window composite, or a reanalysis
    skin-temperature prior). Optionally a focal-mean ``smooth_radius_m`` fills the
    last residual holes from neighbours (set 0 to disable). Implements the
    "cloud + missing-time" gap-fill of R1 §3.C without any client round-trips.

    Returns an ``ee.Image`` (single band, same name as ``primary``).
    """
    ee = _import_ee()
    out = ee.Image(primary)
    for fb in fallbacks:
        out = out.unmask(ee.Image(fb), sameFootprint=False)
    if smooth_radius_m and smooth_radius_m > 0:
        filled = out.focalMean(radius=smooth_radius_m, units="meters")
        out = out.unmask(filled, sameFootprint=False)
    return out


# ===========================================================================
# Thermal sharpening (coarse -> fine, mass-conserving residual)
# ===========================================================================
def sharpen_lst(coarse: Any, predictors: Any, cfg: "Config",
                method: str = "tsharp") -> Any:
    """Thermal sharpening of a coarse LST to the grid resolution.

    Regress LST on fine predictors at the **coarse** scale, apply at the **fine**
    scale, and add the **mass-conserving residual** (coarse minus aggregated
    prediction) so the sharpened image re-aggregates to the original coarse LST —
    the principled DisTrad/TsHARP-with-residual / RF-ATPRK pattern. [R1 §3.A]

    ``method='tsharp'`` uses an ``ee.Reducer.linearFit`` of LST on NDVI/predictor
    (closed-form, fully server-side). Returns an ``ee.Image`` band
    :data:`datamodel.LST` at ``cfg.resolution_m``.
    """
    ee = _import_ee()
    geom = ee_geometry(cfg.bbox)
    fine_scale = float(cfg.resolution_m)
    coarse_img = ee.Image(coarse).select(0).rename("lst_coarse")
    pred = ee.Image(predictors).select(0).rename("pred")

    # Coarse-scale projection to regress at (use the coarse image's projection).
    coarse_proj = coarse_img.projection()

    pair = pred.addBands(coarse_img)
    fit = pair.reduceRegion(
        reducer=ee.Reducer.linearFit(),
        geometry=geom, scale=coarse_proj.nominalScale(), maxPixels=1e13,
        bestEffort=True,
    )
    scale = ee.Number(fit.get("scale"))
    offset = ee.Number(fit.get("offset"))

    # Predict at fine scale, then mass-conserving residual added back.
    pred_fine = pred.multiply(ee.Image.constant(scale)).add(
        ee.Image.constant(offset)).rename("pred_fine")
    pred_coarse = pred_fine.reproject(coarse_proj).reduceResolution(
        ee.Reducer.mean(), maxPixels=1024)
    residual = coarse_img.subtract(pred_coarse)
    sharp = pred_fine.add(residual).reproject(
        crs=cfg.target_crs, scale=fine_scale).rename(LST)
    return sharp.clip(geom).set("sharpen_method", method)


# ===========================================================================
# Top-level production entry points
# ===========================================================================
def fuse_lst_server(
    aoi: Any,
    start: str,
    end: str,
    sensors: Sequence[str] = ("landsat", "modis", "viirs"),
    cloud_pct: float = 60.0,
    ref_hour: float = 13.5,
) -> Any:
    """Production O(1) path: fetch -> harmonise -> diurnal-normalise -> fuse.

    For each requested ``sensors`` name, build a cloud-free °C LST composite with
    the :mod:`urbanheat.gee.lst` helpers, diurnally normalise the multi-sensor set
    to ``ref_hour`` (so Terra/Aqua/VIIRS/Landsat overpass differences are
    reconciled), then combine with the uncertainty-weighted ensemble. The whole
    chain is a server-side ``ee.Image`` graph. [R1 §3-§4]

    Supported ``sensors`` (case-insensitive substring): ``landsat``,
    ``modis``/``modis_terra`` (MOD11A1), ``modis_aqua`` (MYD11A1),
    ``modis_tes`` (MOD21A1D), ``viirs`` (VNP21A1D). (ECOSTRESS / Sentinel-3 are
    not GEE-native for India — they enter via the offline robustness layer; see
    :func:`weighted_ensemble`.)

    Returns an ``ee.Image`` with bands :data:`datamodel.LST` (fused °C) and
    :data:`datamodel.LST_UNCERTAINTY` (1-σ °C).
    """
    ee = _import_ee()
    # Lazy import here keeps fusion importable without lst's deps resolved early.
    from urbanheat.gee import lst as lstmod  # noqa: PLC0415
    from urbanheat.gee.lst import diurnal_normalize  # noqa: PLC0415

    geom = ee_geometry(aoi) if isinstance(aoi, (tuple, list)) else aoi

    images: dict[str, Any] = {}
    for raw in sensors:
        s = raw.lower()
        if "landsat" in s:
            img = lstmod.landsat_lst(geom, start, end, cloud_pct=cloud_pct)
            images["landsat"] = img.select(LST)
        elif "aqua" in s or s in ("myd11a1", "modis_myd11a1"):
            images["MYD11A1"] = lstmod.modis_lst(geom, start, end,
                                                 which="MYD11A1", day=True)
        elif "tes" in s or "mod21" in s:
            images["MOD21A1D"] = lstmod.modis_lst(geom, start, end,
                                                  which="MOD21A1D", day=True)
        elif "viirs" in s or "vnp21" in s:
            images["VNP21A1D"] = lstmod.viirs_lst(geom, start, end, day=True)
        elif "modis" in s or "terra" in s or s == "mod11a1":
            images["MOD11A1"] = lstmod.modis_lst(geom, start, end,
                                                 which="MOD11A1", day=True)
        else:
            raise ValueError(
                f"unknown/non-GEE sensor {raw!r}; supported: landsat, modis, "
                "modis_aqua, modis_tes, viirs")

    if not images:
        raise ValueError("fuse_lst_server: no recognised sensors requested")

    harmonised = harmonize_to_celsius(images)
    # Diurnal normalisation reconciles overpass-time differences before fusion.
    # When >1 sensor, fold the time-normalised consensus in as an extra member so
    # the ensemble benefits from overpass reconciliation while still propagating
    # each sensor's per-pixel uncertainty.
    if len(harmonised) > 1:
        harmonised = dict(harmonised)
        harmonised["dtc_consensus"] = diurnal_normalize(
            harmonised, ref_hour=ref_hour)
    fused = uncertainty_weighted_fusion(harmonised)
    return ee.Image(fused).clip(geom)


def fuse_lst(sensor_images: dict[str, Any], cfg: "Config") -> Any:
    """ARCHITECTURE.md §11 signature: uncertainty-weighted fusion of given images.

    Thin adapter over :func:`uncertainty_weighted_fusion` for callers that have
    already built per-sensor LST images (e.g. ``features.build_feature_image``).
    Returns an ``ee.Image`` with bands :data:`datamodel.LST` and
    :data:`datamodel.LST_UNCERTAINTY`, clipped to ``cfg``'s AOI.
    """
    ee = _import_ee()
    harmonised = harmonize_to_celsius(sensor_images)
    fused = uncertainty_weighted_fusion(harmonised)
    return ee.Image(fused).clip(ee_geometry(cfg.bbox))


# ===========================================================================
# Pure-numpy ensemble (offline robustness layer; numpy-only)
# ===========================================================================
def weighted_ensemble(
    arrays: Sequence[np.ndarray],
    weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Inverse-variance / weighted mean of co-registered LST arrays (NaN-aware).

    The offline mirror of :func:`uncertainty_weighted_fusion`, reused by the
    synthetic robustness layer and unit tests. Computes ``Σ w_i x_i / Σ w_i`` per
    pixel, ignoring NaNs (a sensor that did not see a pixel simply drops out of
    that pixel's weighted sum). ``weights`` may be:

    * ``None`` -> equal weights;
    * a 1-D sequence of length ``len(arrays)`` -> one scalar weight per array;
    * a sequence of per-pixel weight arrays (same shape as each input).

    Parameters
    ----------
    arrays : sequence of np.ndarray
        Same-shaped LST arrays (°C). May contain NaNs for missing pixels.
    weights : sequence/np.ndarray | None
        Per-array scalars or per-pixel weight arrays.

    Returns
    -------
    np.ndarray
        The fused array (float64); pixels with zero total weight -> NaN.
    """
    if len(arrays) == 0:
        raise ValueError("weighted_ensemble: `arrays` must be non-empty")
    stack = np.stack([np.asarray(a, dtype=np.float64) for a in arrays], axis=0)
    n = stack.shape[0]

    if weights is None:
        w = np.ones((n,), dtype=np.float64)
        w_full = w.reshape((n,) + (1,) * (stack.ndim - 1))
        w_full = np.broadcast_to(w_full, stack.shape).copy()
    else:
        w_arr = np.asarray(weights, dtype=np.float64)
        if w_arr.ndim == 1 and w_arr.shape[0] == n:
            w_full = w_arr.reshape((n,) + (1,) * (stack.ndim - 1))
            w_full = np.broadcast_to(w_full, stack.shape).copy()
        else:
            # Per-pixel weights: accept (n, ...) or a stackable sequence.
            w_full = np.stack(
                [np.asarray(wi, dtype=np.float64) for wi in weights], axis=0)
            if w_full.shape != stack.shape:
                raise ValueError(
                    f"per-pixel weights shape {w_full.shape} != arrays shape "
                    f"{stack.shape}")

    valid = np.isfinite(stack)
    w_full = np.where(valid, w_full, 0.0)
    x = np.where(valid, stack, 0.0)

    sum_w = w_full.sum(axis=0)
    sum_wx = (w_full * x).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sum_wx / sum_w
    out = np.where(sum_w > 0, out, np.nan)
    return out


def ensemble_uncertainty(
    sigmas: Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Fused 1-σ from per-source 1-σ values via ``sqrt(1/Σ(1/σ²))`` (numpy).

    Companion to :func:`weighted_ensemble` for the offline path; accepts scalars
    or per-pixel σ arrays.
    """
    arr = [np.asarray(s, dtype=np.float64) for s in sigmas]
    inv_var = np.sum([1.0 / (s ** 2) for s in arr], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sqrt(1.0 / inv_var)
    return out


__all__ = [
    "cloud_free_composite",
    "harmonize_to_celsius",
    "uncertainty_weighted_fusion",
    "temporal_gap_fill",
    "sharpen_lst",
    "fuse_lst_server",
    "fuse_lst",
    "weighted_ensemble",
    "ensemble_uncertainty",
    "DEFAULT_SENSOR_SIGMA",
]
