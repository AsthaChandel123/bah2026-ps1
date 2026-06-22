"""urbanheat.gee.features — assemble the server-side driver image and bridge to FeatureStack.

This module is the **wire-crossing seam** of the GEE backend. It does two things:

1. :func:`build_feature_image` — orchestrates every driver factory
   (:mod:`urbanheat.gee.lst`/``fusion`` for LST, :mod:`~urbanheat.gee.lulc` for
   spectral indices / fractions / albedo / emissivity / LCZ, :mod:`~urbanheat.gee.meteo`
   for ERA5-Land / AOD / NO2 / anthropogenic heat, :mod:`~urbanheat.gee.morphology`
   for building height / volume / SVF / lambda_P / lambda_F / z0 / zd / terrain /
   population / nightlights) into ONE multi-band server-side ``ee.Image`` whose
   bands carry the canonical FeatureStack names (R7 §2). The rasters never leave
   Google — only the recipe is submitted.

2. :func:`sample_to_featurestack` — the only step that crosses the wire: it
   reprojects the feature image to ``cfg.target_crs`` at ``cfg.resolution_m`` and
   samples it onto the analysis grid, returning a numpy-backed
   :class:`~urbanheat.datamodel.FeatureStack`. The grid (shape / affine transform
   / bounds) is computed with the **same** routine the synthetic backend uses
   (:func:`urbanheat.synthetic.source._grid_geometry`) so a GEE FeatureStack and a
   SyntheticDataSource FeatureStack are pixel-for-pixel interchangeable and the
   whole downstream pipeline is identical regardless of backend.

Robustness: missing layers (an empty MAIAC window, an un-ingested LCZ tile, ...)
are handled gracefully — each driver group is wrapped so a server-side failure
warns and is skipped, and any canonical layer absent after sampling is filled
with NaN so the FeatureStack always has a consistent schema.

All ``ee`` imports are lazy (inside functions); importing this module needs only
numpy + the constants/datamodel catalogs.

Public contract (ARCHITECTURE.md §11.2 ``gee/features.py``)::

    build_feature_image(cfg) -> ee.Image
    sample_to_featurestack(feature_image, cfg) -> FeatureStack

plus the build-task entry point :func:`build_feature_stack` ``(cfg) -> FeatureStack``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from urbanheat.config import Config
from urbanheat import datamodel as dm
from urbanheat.datamodel import FeatureStack

if TYPE_CHECKING:  # pragma: no cover
    import ee


# Canonical layers we attempt to populate from GEE (the full driver stack, the
# same set the SyntheticDataSource produces so the two backends are swappable).
_FEATURE_LAYERS: tuple[str, ...] = (
    # thermal / target
    dm.LST, dm.LST_DAY, dm.LST_NIGHT, dm.LST_UNCERTAINTY, dm.EMISSIVITY,
    # vegetation / spectral
    dm.NDVI, dm.EVI, dm.SAVI, dm.NDWI, dm.MNDWI, dm.NDBI, dm.NDBAI, dm.UI,
    dm.LAI, dm.FVC, dm.ET, dm.ALBEDO,
    # land cover / fractions
    dm.LULC, dm.IMPERVIOUS_FRAC, dm.GREEN_FRAC, dm.WATER_FRAC, dm.TREE_FRAC,
    dm.LCZ,
    # morphology / 3D form
    dm.BUILDING_HEIGHT, dm.BUILDING_VOLUME, dm.SVF, dm.ASPECT_RATIO,
    dm.PLAN_AREA_FRAC, dm.FRONTAL_AREA_INDEX, dm.ROUGHNESS_LENGTH,
    dm.DISPLACEMENT_HEIGHT, dm.ELEVATION, dm.SLOPE,
    # meteorology / atmosphere
    dm.AIR_TEMP, dm.DEWPOINT, dm.REL_HUMIDITY, dm.WIND_SPEED, dm.PRESSURE,
    dm.SOLAR_RADIATION, dm.LONGWAVE_DOWN, dm.NET_RADIATION, dm.SOIL_MOISTURE,
    dm.AOD, dm.PBL_HEIGHT,
    # anthropogenic
    dm.POPULATION, dm.NIGHTLIGHTS, dm.ANTHRO_HEAT, dm.NO2,
)


def _import_ee() -> Any:
    """Lazy-import Earth Engine via the auth module's actionable-error helper."""
    from urbanheat.gee.auth import _import_ee as _imp  # noqa: PLC0415

    return _imp()


def _aoi(cfg: Config) -> "ee.Geometry":
    from urbanheat.gee.auth import ee_geometry  # noqa: PLC0415

    return ee_geometry(cfg.bbox)


def _try_group(name: str, fn: Callable[[], "ee.Image"],
               parts: list["ee.Image"]) -> None:
    """Append ``fn()`` to ``parts``; on failure warn and skip (graceful degrade)."""
    try:
        parts.append(fn())
    except Exception as exc:  # noqa: BLE001 - server-side variance, never fatal
        warnings.warn(f"build_feature_image: skipping driver group {name!r}: "
                      f"{type(exc).__name__}: {exc}")


# ===========================================================================
# 1. Server-side feature image
# ===========================================================================
def build_feature_image(cfg: Config) -> "ee.Image":
    """Assemble ALL driver bands into one server-side ``ee.Image`` (ARCHITECTURE.md §11.2).

    Orchestrates the driver factories and band-names everything with the
    canonical FeatureStack names. The LST backbone comes from the fused
    multi-sensor product (:func:`urbanheat.gee.fusion.fuse_lst_server`) with the
    Landsat day/night MODIS split added; the surface drivers from
    :mod:`urbanheat.gee.lulc`; meteorology from :mod:`urbanheat.gee.meteo`; and
    morphology from :mod:`urbanheat.gee.morphology`. Each group is optional —
    a group that fails server-side is warned about and skipped so the rest of
    the stack still assembles.

    Returns
    -------
    ee.Image
        Multi-band image (canonical band names) clipped to the AOI, in EPSG:4326
        at native resolutions. Reproject/sample happens in
        :func:`sample_to_featurestack`.
    """
    ee = _import_ee()
    aoi = _aoi(cfg)

    from urbanheat.gee import lulc as lulcmod  # noqa: PLC0415
    from urbanheat.gee import meteo as meteomod  # noqa: PLC0415
    from urbanheat.gee import morphology as morphmod  # noqa: PLC0415

    parts: list["ee.Image"] = []

    # --- LST backbone (fused multi-sensor + diurnal split) ---------------
    def _lst() -> "ee.Image":
        from urbanheat.gee import fusion as fusionmod  # noqa: PLC0415
        from urbanheat.gee import lst as lstmod  # noqa: PLC0415

        bands: list["ee.Image"] = []
        # Fused day LST + uncertainty (the model target).
        fused = fusionmod.fuse_lst_server(
            cfg.bbox, cfg.start_date, cfg.end_date,
            sensors=("landsat", "modis", "viirs"))
        bands.append(ee.Image(fused).select([dm.LST, dm.LST_UNCERTAINTY]))
        # Emissivity exposed by the Landsat physics path.
        try:
            ls = lstmod.landsat_lst(cfg.bbox, cfg.start_date, cfg.end_date)
            bands.append(ee.Image(ls).select([dm.EMISSIVITY]))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"features: landsat emissivity skipped: {exc}")
        # Day / night MODIS LST for the diurnal split.
        try:
            day = lstmod.modis_lst(cfg.bbox, cfg.start_date, cfg.end_date,
                                   which="MOD11A1", day=True)
            bands.append(ee.Image(day).select([dm.LST_DAY]))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"features: MODIS day LST skipped: {exc}")
        try:
            night = lstmod.modis_lst(cfg.bbox, cfg.start_date, cfg.end_date,
                                     which="MOD11A1", day=False)
            bands.append(ee.Image(night).select([dm.LST_NIGHT]))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"features: MODIS night LST skipped: {exc}")
        return ee.Image.cat(bands)

    _try_group("lst", _lst, parts)

    # --- surface: spectral indices + albedo + emissivity -----------------
    _try_group("spectral_indices", lambda: lulcmod.spectral_indices(cfg), parts)
    # --- surface: continuous fractions -----------------------------------
    _try_group("fractional_cover", lambda: lulcmod.fractional_cover(cfg), parts)
    # --- surface: LULC class + ET ----------------------------------------
    _try_group("landcover", lambda: lulcmod.landcover(cfg), parts)
    _try_group("et", lambda: lulcmod.et(cfg), parts)

    # --- morphology (one merged image: lambda_P/lambda_F/H-W/svf/z0/zd/
    #     building_height/building_volume/elevation/slope) -----------------
    _try_group("morphometrics", lambda: morphmod.morphometrics(cfg), parts)
    # --- morphology: LCZ + population + nightlights ----------------------
    _try_group("lcz", lambda: morphmod.lcz(cfg), parts)
    _try_group("population", lambda: morphmod.population(cfg), parts)

    # --- meteorology + atmosphere + anthropogenic heat -------------------
    _try_group("era5_drivers", lambda: meteomod.era5_drivers(cfg), parts)
    _try_group("aod", lambda: meteomod.aod(cfg), parts)
    _try_group("no2", lambda: meteomod.no2(cfg), parts)
    _try_group("anthropogenic_heat",
               lambda: meteomod.anthropogenic_heat(cfg), parts)

    if not parts:
        raise RuntimeError(
            "build_feature_image: every driver group failed; cannot assemble a "
            "feature image. Check Earth Engine auth/quota or use mode='synthetic'.")

    image = ee.Image.cat(parts)

    # Keep only canonical bands that are actually present (dedupe in case two
    # groups expose the same name, e.g. nightlights via meteo + morphology).
    present = image.bandNames()
    wanted = ee.List([b for b in _FEATURE_LAYERS])
    keep = wanted.filter(ee.Filter.inList("item", present))
    return image.select(keep).clip(aoi)


# ===========================================================================
# 2. Sample the feature image onto the FeatureStack grid (the wire crossing)
# ===========================================================================
def sample_to_featurestack(feature_image: "ee.Image", cfg: Config) -> FeatureStack:
    """Sample/export the feature image at the config grid into a FeatureStack.

    The wire-crossing step (R7 §2.3): the rasters stay on Earth Engine; only the
    reduced grid of values comes back. The analysis grid (shape / affine
    transform / bounds in ``cfg.target_crs``) is computed with the SAME routine
    the synthetic backend uses, so the two backends produce identical grids.

    Mechanism: the feature image is reprojected to ``cfg.target_crs`` at
    ``cfg.resolution_m`` and pulled band-by-band with ``sampleRectangle`` over the
    AOI (a regular dense grid -> 2-D numpy arrays). ``sampleRectangle`` returns
    the native pixel block, which is then resampled (nearest, NaN-aware) onto the
    exact ``(rows, cols)`` FeatureStack grid so every layer is co-registered.
    Layers that fail to sample are filled with NaN, and any canonical layer not
    present in the image is added as an all-NaN layer, so the returned stack has
    a stable schema regardless of which GEE products were available.

    Parameters
    ----------
    feature_image : ee.Image
        The multi-band image from :func:`build_feature_image` (canonical bands).
    cfg : Config
        Run configuration (AOI / CRS / resolution / grid).

    Returns
    -------
    FeatureStack
        Numpy-backed stack with ``crs == cfg.target_crs`` and the grid derived
        from the AOI; provenance ``meta['mode'] == 'gee'``.
    """
    ee = _import_ee()
    from urbanheat.synthetic.source import _grid_geometry  # noqa: PLC0415

    shape, transform, bounds = _grid_geometry(cfg)
    rows, cols = shape
    aoi = _aoi(cfg)

    # Reproject to the analysis CRS/resolution once (server-side); sampling then
    # returns values already on the metric grid.
    img = ee.Image(feature_image).reproject(crs=cfg.target_crs,
                                             scale=float(cfg.resolution_m))

    band_names = _get_band_names(img)

    layers: dict[str, np.ndarray] = {}
    sampled_meta: dict[str, str] = {}
    for band in band_names:
        arr = _sample_band(img, band, aoi, cfg)
        if arr is None:
            continue
        arr = _resample_to_grid(arr, rows, cols)
        layers[band] = arr.astype(np.float32)
        sampled_meta[band] = "gee"

    # Ensure a stable schema: every canonical feature layer exists (NaN if absent).
    nan_layers: list[str] = []
    for name in _FEATURE_LAYERS:
        if name not in layers:
            layers[name] = np.full((rows, cols), np.nan, dtype=np.float32)
            nan_layers.append(name)

    if dm.LST not in sampled_meta:
        warnings.warn(
            "sample_to_featurestack: no 'lst' layer was sampled from GEE — the "
            "FeatureStack target is all-NaN. Check the LST fusion step / auth.")

    meta: dict[str, Any] = {
        "mode": "gee",
        "source": "GEEDataSource",
        "city": getattr(cfg, "city", None),
        "resolution_m": float(cfg.resolution_m),
        "bbox_lonlat": tuple(cfg.bbox),
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "provenance": {name: "gee" for name in layers},
        "missing_layers": nan_layers,
        "description": (
            "GEE-derived driver stack sampled onto the analysis grid "
            f"({rows}x{cols} @ {cfg.resolution_m} m, {cfg.target_crs})."
        ),
    }

    stack = FeatureStack(layers={k: np.asarray(v, dtype=np.float32)
                                 for k, v in layers.items()},
                         transform=transform, crs=cfg.target_crs,
                         bounds=bounds, shape=(rows, cols), meta=meta)
    stack.validate()
    return stack


def _get_band_names(img: "ee.Image") -> list[str]:
    """Best-effort client-side list of band names (``getInfo`` once)."""
    try:
        names = img.bandNames().getInfo()
        return list(names) if names else []
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"sample_to_featurestack: could not list band names: {exc}")
        return []


def _sample_band(img: "ee.Image", band: str, aoi: "ee.Geometry",
                 cfg: Config) -> "np.ndarray | None":
    """Pull a single band over the AOI as a 2-D numpy array via ``sampleRectangle``.

    ``sampleRectangle`` returns the pixels inside the AOI as a nested list which
    numpy turns into a 2-D array. ``defaultValue`` keeps masked pixels finite
    (NaN) so the array shape is rectangular. Returns ``None`` (with a warning) if
    the band cannot be retrieved (e.g. exceeds the sampleRectangle pixel cap),
    so the caller fills NaN.
    """
    ee = _import_ee()
    try:
        rect = ee.Image(img).select([band]).sampleRectangle(
            region=aoi, defaultValue=float("nan"))
        data = rect.get(band).getInfo()
        arr = np.array(data, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            warnings.warn(
                f"sample_to_featurestack: band {band!r} returned non-2D/empty; "
                "filling NaN.")
            return None
        return arr
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"sample_to_featurestack: band {band!r} sample failed "
            f"({type(exc).__name__}: {exc}); filling NaN.")
        return None


def _resample_to_grid(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Nearest-neighbour resample a 2-D array onto an exact (rows, cols) grid.

    ``sampleRectangle`` yields the native pixel block over the AOI which may
    differ from the target ``(rows, cols)`` by a pixel or two (edge rounding).
    A NaN-preserving nearest-neighbour index map snaps it to the FeatureStack
    grid without pulling in scipy/skimage — index arithmetic only. If the block
    already matches the target shape it is returned unchanged.
    """
    arr = np.asarray(arr, dtype=np.float64)
    src_r, src_c = arr.shape
    if (src_r, src_c) == (rows, cols):
        return arr
    if src_r == 0 or src_c == 0:
        return np.full((rows, cols), np.nan, dtype=np.float64)
    # nearest-neighbour index maps (image rows are north->south, same as ours).
    ri = np.clip((np.arange(rows) * src_r / rows).astype(int), 0, src_r - 1)
    ci = np.clip((np.arange(cols) * src_c / cols).astype(int), 0, src_c - 1)
    return arr[np.ix_(ri, ci)]


# ===========================================================================
# 3. End-to-end convenience: build + sample (build-task entry point)
# ===========================================================================
def build_feature_stack(cfg: Config) -> FeatureStack:
    """Build the server-side feature image and sample it into a FeatureStack.

    The single call the :class:`~urbanheat.gee.source.GEEDataSource` makes after
    initialising Earth Engine: :func:`build_feature_image` (assemble the
    server-side recipe) then :func:`sample_to_featurestack` (cross the wire onto
    the analysis grid). Returns a :class:`~urbanheat.datamodel.FeatureStack` with
    the same canonical layers as the synthetic backend (``provenance='gee'``),
    so downstream code is backend-agnostic.

    Layers that no available GEE product could supply are present but all-NaN
    (recorded in ``meta['missing_layers']``).
    """
    feature_image = build_feature_image(cfg)
    return sample_to_featurestack(feature_image, cfg)


__all__ = [
    "build_feature_image",
    "sample_to_featurestack",
    "build_feature_stack",
]
