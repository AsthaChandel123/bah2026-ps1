"""urbanheat.gee.collections — preprocessed Earth Engine collection helpers.

This module is the single place where raw Earth Engine ``ImageCollection`` /
``Image`` assets become **analysis-ready**: cloud/shadow-masked and converted to
physical units (°C, reflectance, …) using the scale/offset catalog in
:mod:`urbanheat.constants` (``GEE_DATASETS`` + ``BAND_SCALE_OVERRIDES``). Nothing
is hard-coded here — every id, band list and scale/offset is imported from the
constants catalog so the "single source of truth" rule holds. [R1 §6, R7 §2.7]

Cloud/shadow masking
--------------------
* **Landsat C2-L2** — CFMask ``QA_PIXEL`` bitmask: bit 1 = dilated cloud, bit 3 =
  cloud, bit 4 = cloud shadow (also bit 5 = snow). [R7 §2.7]
* **MODIS / VIIRS** — ``QC``/``QC_Day``/``QC_Night`` mandatory-QA bits (00 = good).
* **Sentinel-2** — joined ``S2_CLOUD_PROBABILITY`` thresholded.

All ``ee`` imports are lazy (inside functions); importing this module needs no
optional dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from urbanheat.constants import BAND_SCALE_OVERRIDES, GEE_DATASETS, KELVIN

if TYPE_CHECKING:  # hints only
    from urbanheat.config import Config

from urbanheat.gee.auth import ee_geometry

# ---------------------------------------------------------------------------
# QA_PIXEL bit positions (Landsat Collection-2 CFMask). [R7 §2.7]
# ---------------------------------------------------------------------------
QA_DILATED_CLOUD_BIT = 1
QA_CLOUD_BIT = 3
QA_CLOUD_SHADOW_BIT = 4
QA_SNOW_BIT = 5

# Default per-dataset cloud-cover scene-metadata property + filter ceiling.
_CLOUD_PROPERTY = {
    "LANDSAT": "CLOUD_COVER",
    "S2": "CLOUDY_PIXEL_PERCENTAGE",
}


def _import_ee() -> Any:
    """Lazy-import ``ee`` (delegates to the auth module's clear error)."""
    from urbanheat.gee.auth import _import_ee as _imp  # noqa: PLC0415

    return _imp()


def _bbox_from(aoi: Any) -> Any:
    """Coerce ``aoi`` to an ``ee.Geometry``.

    Accepts an existing ``ee.Geometry``/``ee.Feature`` (returned as-is or via
    ``.geometry()``) or an EPSG:4326 ``(xmin, ymin, xmax, ymax)`` tuple/list.
    """
    if isinstance(aoi, (tuple, list)) and len(aoi) == 4:
        return ee_geometry(tuple(aoi))  # type: ignore[arg-type]
    geom = getattr(aoi, "geometry", None)
    if callable(geom):
        return geom()
    return aoi


# ===========================================================================
# Scaling
# ===========================================================================
def scaled(image: Any, key: str) -> Any:
    """Apply ``GEE_DATASETS[key]`` scale/offset (+ per-band overrides) to ``image``.

    The single place GEE scale/offset is applied. The product-default
    ``scale``/``offset`` is applied to every band in ``GEE_DATASETS[key]['bands']``
    that is present, **except** bands listed in
    ``BAND_SCALE_OVERRIDES[key]`` which get their own ``(scale, offset)``.
    Bands not in the catalog (e.g. ``QA_PIXEL``) are passed through untouched.

    Returns an ``ee.Image`` with the same band names but physical-unit values.
    """
    if key not in GEE_DATASETS:
        raise KeyError(f"unknown dataset key {key!r}; see constants.GEE_DATASETS")
    _import_ee()  # validate ee present (band math below is server-side)
    spec = GEE_DATASETS[key]
    scale = float(spec["scale"])
    offset = float(spec["offset"])
    overrides = BAND_SCALE_OVERRIDES.get(key, {})

    band_names = image.bandNames()
    out = image

    for band in spec.get("bands", []):
        if band in overrides:
            bscale, boffset = overrides[band]
        elif band in ("QA_PIXEL", "QC", "QC_Day", "QC_Night", "label", "Map",
                      "QF_Cloud_Mask", "Mandatory_Quality_Flag", "AOD_QA"):
            # Bitmask / class bands carry no physical scale — never rescale.
            continue
        else:
            bscale, boffset = scale, offset
        if bscale == 1.0 and boffset == 0.0:
            continue
        scaled_band = (image.select(band).multiply(bscale).add(boffset)
                       .rename(band))
        # Guard: only add the band if the source actually contains it.
        out = ee_image_replace_band(out, band, scaled_band, band_names)
    return out


def ee_image_replace_band(
    image: Any, band: str, new_band: Any, present_names: Any
) -> Any:
    """Server-side: replace ``band`` with ``new_band`` iff ``band`` is present.

    Uses ``ee.Algorithms.If`` against the (server-side) ``present_names`` list so
    no ``getInfo`` round-trip is needed.
    """
    ee = _import_ee()
    has = present_names.contains(band)
    replaced = image.addBands(new_band, [band], True)
    return ee.Image(ee.Algorithms.If(has, replaced, image))


def lst_to_celsius(image: Any, band: str, new_name: str | None = None) -> Any:
    """Convert a Kelvin band to °C (subtract :data:`constants.KELVIN`)."""
    _import_ee()
    name = new_name or band
    return image.select(band).subtract(KELVIN).rename(name)


# ===========================================================================
# Cloud / shadow masking
# ===========================================================================
def mask_landsat_qa(image: Any, include_snow: bool = True) -> Any:
    """Mask Landsat C2 cloud (bit 3), shadow (bit 4), dilated cloud (bit 1).

    [R7 §2.7] ``QA_PIXEL`` CFMask bits. Optionally also masks snow (bit 5).
    Returns ``image`` with the cloudy/shadow pixels masked out.
    """
    ee = _import_ee()
    qa = image.select("QA_PIXEL")
    cloud = qa.bitwiseAnd(1 << QA_CLOUD_BIT).neq(0)
    shadow = qa.bitwiseAnd(1 << QA_CLOUD_SHADOW_BIT).neq(0)
    dilate = qa.bitwiseAnd(1 << QA_DILATED_CLOUD_BIT).neq(0)
    bad = cloud.Or(shadow).Or(dilate)
    if include_snow:
        bad = bad.Or(qa.bitwiseAnd(1 << QA_SNOW_BIT).neq(0))
    clear = bad.Not()
    return image.updateMask(clear)


def mask_modis_qc(image: Any, qc_band: str = "QC_Day") -> Any:
    """Mask MODIS/VIIRS LST by the mandatory-QA bits of ``qc_band``.

    Keeps pixels whose bits 0-1 == 0 ("LST produced, good quality"). The same
    encoding is used by MxD11/MxD21/VNP21 ``QC``/``QC_Day``/``QC_Night``.
    """
    _import_ee()
    qc = image.select(qc_band)
    good = qc.bitwiseAnd(0b11).eq(0)
    return image.updateMask(good)


def mask_clouds(image: Any, key: str) -> Any:
    """Apply the product-appropriate cloud/QA mask for dataset ``key``.

    Dispatches on the dataset family: Landsat -> ``QA_PIXEL`` bits; MODIS/VIIRS
    LST -> ``QC`` bits; everything else is returned unchanged (S2 is masked via
    the dedicated :func:`sentinel2_collection` join). [R7 §2.7]
    """
    if key not in GEE_DATASETS:
        raise KeyError(f"unknown dataset key {key!r}; see constants.GEE_DATASETS")
    if key.startswith("LANDSAT"):
        return mask_landsat_qa(image)
    if key.startswith("MODIS_MOD11") or key.startswith("MODIS_MYD11"):
        return mask_modis_qc(image, "QC_Day")
    if key.startswith("MODIS_MOD21") or key.startswith("MODIS_MYD21") \
            or key.startswith("VIIRS_VNP21"):
        return mask_modis_qc(image, "QC")
    return image


# ===========================================================================
# Collection builders (filter -> mask -> scale), all server-side
# ===========================================================================
def get_collection(key: str) -> Any:
    """Return the raw ``ee.ImageCollection``/``ee.Image`` for a catalog ``key``.

    Vector datasets (``units == 'vector'``) are returned as
    ``ee.FeatureCollection``. Raises ``KeyError`` on an unknown key or one with
    no GEE id (``id == ''``).
    """
    if key not in GEE_DATASETS:
        raise KeyError(f"unknown dataset key {key!r}; see constants.GEE_DATASETS")
    spec = GEE_DATASETS[key]
    asset_id = spec["id"]
    if not asset_id:
        raise KeyError(f"dataset {key!r} has no GEE id (external source)")
    ee = _import_ee()
    if spec.get("units") == "vector":
        return ee.FeatureCollection(asset_id)
    # ESA WorldCover etc. are single mosaics exposed as a 1-image collection.
    return ee.ImageCollection(asset_id)


def _filtered(
    key: str, aoi: Any, start: str, end: str, cloud_pct: float | None,
) -> Any:
    """``filterBounds -> filterDate -> (optional cloud-cover filter)`` for ``key``."""
    geom = _bbox_from(aoi)
    col = get_collection(key).filterBounds(geom).filterDate(start, end)
    if cloud_pct is not None:
        family = "LANDSAT" if key.startswith("LANDSAT") else (
            "S2" if key.startswith("S2") else None)
        prop = _CLOUD_PROPERTY.get(family) if family else None
        if prop is not None:
            from urbanheat.gee.auth import _import_ee as _imp  # noqa: PLC0415
            ee = _imp()
            col = col.filter(ee.Filter.lt(prop, float(cloud_pct)))
    return col


def landsat_c2_collection(
    aoi: Any,
    start: str,
    end: str,
    cloud_pct: float = 60.0,
    keys: tuple[str, ...] = ("LANDSAT8_L2", "LANDSAT9_L2"),
) -> Any:
    """Analysis-ready Landsat C2-L2 collection over the AOI.

    Filters by bounds/date/scene-cloud-cover, masks cloud+shadow via
    ``QA_PIXEL``, applies SR (``×2.75e-05 − 0.2``) and ST_B10
    (``×0.00341802 + 149`` K) scaling, then merges the requested platforms
    (L8+L9 by default = ~8-day combined revisit). Reflectance bands are in
    0-1 reflectance; ``ST_B10`` is in **Kelvin** (convert with
    :func:`lst_to_celsius`). [R1 §2.1, R7 §2.7]

    Returns an ``ee.ImageCollection`` (merged, masked, scaled).
    """
    ee = _import_ee()
    merged = None
    for key in keys:
        if key not in GEE_DATASETS:
            raise KeyError(f"unknown Landsat key {key!r}")
        col = _filtered(key, aoi, start, end, cloud_pct)
        col = col.map(lambda img, k=key: scaled(mask_landsat_qa(img), k))
        merged = col if merged is None else merged.merge(col)
    if merged is None:  # pragma: no cover - keys is always non-empty
        return ee.ImageCollection([])
    return ee.ImageCollection(merged)


def modis_lst_collection(
    aoi: Any,
    start: str,
    end: str,
    key: str = "MODIS_MOD11A1",
    cloud_pct: float | None = None,
) -> Any:
    """Analysis-ready MODIS/VIIRS LST collection (QC-masked, scaled to Kelvin).

    Works for any 1 km LST product key in the catalog (``MODIS_MOD11A1``,
    ``MODIS_MYD11A1``, ``MODIS_MOD21A1D``, ``VIIRS_VNP21A1D``, …). LST bands are
    scaled ``×0.02`` to Kelvin and QC-masked to good-quality pixels. [R1 §2.4-2.5]

    Returns an ``ee.ImageCollection``.
    """
    _import_ee()
    if key not in GEE_DATASETS:
        raise KeyError(f"unknown MODIS/VIIRS LST key {key!r}")
    col = _filtered(key, aoi, start, end, cloud_pct)
    return col.map(lambda img, k=key: scaled(mask_clouds(img, k), k))


def viirs_lst_collection(
    aoi: Any,
    start: str,
    end: str,
    key: str = "VIIRS_VNP21A1D",
    cloud_pct: float | None = None,
) -> Any:
    """Analysis-ready VIIRS VNP21 LST collection (thin wrapper over MODIS path)."""
    return modis_lst_collection(aoi, start, end, key=key, cloud_pct=cloud_pct)


def sentinel2_collection(
    aoi: Any,
    start: str,
    end: str,
    cloud_pct: float = 60.0,
    mask_probability: int = 50,
) -> Any:
    """Analysis-ready Sentinel-2 SR collection with S2_CLOUD_PROBABILITY masking.

    Joins ``COPERNICUS/S2_SR_HARMONIZED`` to ``COPERNICUS/S2_CLOUD_PROBABILITY``
    on ``system:index`` and masks pixels whose cloud probability exceeds
    ``mask_probability`` (%). SR bands are scaled ``×0.0001`` to reflectance.
    [R2 V1-V2]

    Returns an ``ee.ImageCollection`` of masked, scaled S2 reflectance.
    """
    ee = _import_ee()
    geom = _bbox_from(aoi)
    s2 = ("S2_SR_HARMONIZED", "S2_CLOUD_PROBABILITY")
    sr = (get_collection(s2[0]).filterBounds(geom).filterDate(start, end)
          .filter(ee.Filter.lt(_CLOUD_PROPERTY["S2"], float(cloud_pct))))
    clouds = (get_collection(s2[1]).filterBounds(geom).filterDate(start, end))

    join = ee.Join.saveFirst("cloud_mask")
    cond = ee.Filter.equals(leftField="system:index", rightField="system:index")
    joined = ee.ImageCollection(join.apply(sr, clouds, cond))

    def _mask(img: Any) -> Any:
        prob = ee.Image(img.get("cloud_mask")).select("probability")
        clear = prob.lt(mask_probability)
        return scaled(img.updateMask(clear), "S2_SR_HARMONIZED")

    return joined.map(_mask)


def composite(key: str, cfg: "Config", reducer: str = "median") -> Any:
    """``filter -> mask+scale -> reduce -> clip`` -> single cloud-free ``ee.Image``.

    Builds a cloud-free composite for dataset ``key`` over ``cfg``'s AOI/date
    window at native scale, using the named ``reducer`` (``'median'``,
    ``'mean'``, ``'min'``, ``'max'``). For Landsat/MODIS/VIIRS/S2 the appropriate
    masked+scaled collection helper is used; other datasets fall back to a plain
    filtered+scaled collection. [R7 §2.7]
    """
    ee = _import_ee()
    geom = ee_geometry(cfg.bbox)
    start, end = cfg.start_date, cfg.end_date

    if key.startswith("LANDSAT"):
        col = landsat_c2_collection(geom, start, end, keys=(key,))
    elif key.startswith("MODIS_MOD11") or key.startswith("MODIS_MYD11") \
            or key.startswith("MODIS_MOD21") or key.startswith("MODIS_MYD21") \
            or key.startswith("VIIRS_VNP21"):
        col = modis_lst_collection(geom, start, end, key=key)
    elif key.startswith("S2"):
        col = sentinel2_collection(geom, start, end)
    else:
        col = _filtered(key, geom, start, end, None).map(
            lambda img, k=key: scaled(img, k))

    reduced = _reduce(col, reducer)
    return reduced.clip(geom)


def _reduce(col: Any, reducer: str) -> Any:
    """Apply a named temporal reducer to an ``ee.ImageCollection``."""
    _import_ee()
    r = reducer.lower()
    if r == "median":
        return col.median()
    if r == "mean":
        return col.mean()
    if r == "min":
        return col.min()
    if r == "max":
        return col.max()
    if r in ("mosaic", "quality", "qualitymosaic"):
        return col.mosaic()
    raise ValueError(
        f"unknown reducer {reducer!r}; use median/mean/min/max/mosaic")


__all__ = [
    "scaled",
    "lst_to_celsius",
    "mask_landsat_qa",
    "mask_modis_qc",
    "mask_clouds",
    "get_collection",
    "landsat_c2_collection",
    "modis_lst_collection",
    "viirs_lst_collection",
    "sentinel2_collection",
    "composite",
    "QA_DILATED_CLOUD_BIT",
    "QA_CLOUD_BIT",
    "QA_CLOUD_SHADOW_BIT",
    "QA_SNOW_BIT",
]
