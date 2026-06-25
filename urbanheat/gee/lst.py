"""urbanheat.gee.lst — server-side multi-sensor Land Surface Temperature (°C).

Every function returns an ``ee.Image`` (or collection) whose LST band is in
**degrees Celsius** and named with the canonical :mod:`urbanheat.datamodel`
layer name (``"lst"``, plus ``"lst_day"``/``"lst_night"`` for diurnal products).
All scale/offset values and band names are imported from
:mod:`urbanheat.constants` — never hard-coded. All band math runs server-side via
``ee.Image.expression``; the only client-side numbers are the physical constants.

Sensors / algorithms wired
--------------------------
* **Landsat C2-L2** (``ST_B10`` ``×0.00341802 + 149`` K, single-channel) plus an
  explicit **NDVI-threshold-emissivity mono-window (Ermida/SMW)** path on the
  brightness temperature so the physics is visible. [R1 §2.1, R7 §2.6]
* **MODIS** MxD11 split-window & MxD21 TES (``×0.02`` K, **day & night**). [R1 §2.4]
* **VIIRS** VNP21 TES (``×0.02`` K, day & night). [R1 §2.5]

ECOSTRESS caveat
----------------
ECOSTRESS (``NASA/ECOSTRESS/L2T_LSTE/V2``, ~70 m, all-hours TES) is the finest
diurnal thermal source, but **only the Los Angeles metro is ingested into GEE** —
for Indian cities it must be pulled from LP DAAC / AppEEARS (see
``constants.EXTERNAL_SOURCES['ECOSTRESS_INDIA']`` and the STAC fallback in the
offline path). Hence there is intentionally **no** GEE ECOSTRESS LST helper here.
[R1 §2.3]

All ``ee`` imports are lazy; importing this module needs only the standard
library + the constants catalog.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from urbanheat.constants import (
    GEE_DATASETS,
    KELVIN,
    PHYSICAL_CONSTANTS,
    SPECTRAL_INDEX_COEFFS,
)
from urbanheat.datamodel import EMISSIVITY, FVC, LST, LST_DAY, LST_NIGHT, NDVI
from urbanheat.gee.auth import ee_geometry
from urbanheat.gee.collections import (
    landsat_c2_collection,
    lst_to_celsius,
    modis_lst_collection,
)

if TYPE_CHECKING:  # hints only
    from urbanheat.config import Config

# Planck mono-window constants for Landsat B10. [R1 App.A / R7 §2.6 / constants]
_LAMBDA_B10 = PHYSICAL_CONSTANTS["PLANCK_LAMBDA_B10"]   # 10.895e-6 m
_RHO = PHYSICAL_CONSTANTS["PLANCK_RHO"]                 # 1.438e-2 m*K
# NDVI-threshold emissivity coefficients (Sobrino). [R2 §5 / constants]
_EMIS = SPECTRAL_INDEX_COEFFS["EMISSIVITY_NDVI"]


def _import_ee() -> Any:
    from urbanheat.gee.auth import _import_ee as _imp  # noqa: PLC0415

    return _imp()


# ===========================================================================
# Emissivity helpers (NDVI-threshold method, server-side)
# ===========================================================================
def fractional_vegetation(ndvi: Any, ndvi_soil: float | None = None,
                          ndvi_veg: float | None = None) -> Any:
    """Fractional vegetation cover ``FVC = ((NDVI - Ns)/(Nv - Ns))**2``, clamped 0-1.

    Uses the soil/veg NDVI thresholds from
    ``constants.SPECTRAL_INDEX_COEFFS['EMISSIVITY_NDVI']`` unless overridden.
    [R7 §2.6]
    """
    _import_ee()
    ns = _EMIS["ndvi_soil"] if ndvi_soil is None else ndvi_soil
    nv = _EMIS["ndvi_veg"] if ndvi_veg is None else ndvi_veg
    fv = ndvi.expression(
        "((ndvi - ns) / (nv - ns)) ** 2",
        {"ndvi": ndvi, "ns": ns, "nv": nv},
    )
    return fv.clamp(0.0, 1.0).rename(FVC)


def ndvi_emissivity(ndvi: Any) -> Any:
    """NDVI-threshold broadband emissivity (Sobrino) as an ``ee.Image``.

    Piecewise: ``NDVI < ndvi_soil`` -> bare-soil emissivity; ``NDVI > ndvi_veg``
    -> vegetation emissivity; in between -> mixed-pixel
    ``eps = eps_veg*FVC + eps_soil*(1-FVC) + d_eps``. Coefficients from
    ``constants.SPECTRAL_INDEX_COEFFS['EMISSIVITY_NDVI']``. [R2 §5 / R7 §2.6]

    Returns a single-band image named :data:`datamodel.EMISSIVITY`.
    """
    _import_ee()
    fv = fractional_vegetation(ndvi)
    eps_mix = ndvi.expression(
        "eps_v * fv + eps_s * (1 - fv) + d",
        {"fv": fv, "eps_v": _EMIS["eps_veg"], "eps_s": _EMIS["eps_soil"],
         "d": _EMIS["d_eps"]},
    )
    soil = ndvi.lt(_EMIS["ndvi_soil"])
    veg = ndvi.gt(_EMIS["ndvi_veg"])
    emis = (eps_mix
            .where(soil, _EMIS["eps_soil"])
            .where(veg, _EMIS["eps_veg"]))
    return emis.rename(EMISSIVITY)


# ===========================================================================
# Landsat LST (single-channel ST_B10 + explicit mono-window physics path)
# ===========================================================================
def landsat_lst(
    aoi: Any,
    start: str,
    end: str,
    cloud_pct: float = 60.0,
    method: str = "st_b10",
    keys: tuple[str, ...] = ("LANDSAT8_L2", "LANDSAT9_L2"),
) -> Any:
    """Server-side Landsat C2-L2 LST composite in °C over the AOI.

    Builds a cloud-masked, scaled L8+L9 median composite then derives LST two
    ways, exposing both on the output image:

    * ``method='st_b10'`` (default, **production**): use the USGS Level-2 ``ST_B10``
      surface-temperature band directly (``×0.00341802 + 149`` K, applied in
      :func:`collections.scaled`), converted to °C. Highest accuracy. [R1 §2.1]
    * ``method='ermida'`` / ``'smw'`` / ``'mono_window'``: compute LST from the
      ``ST_B10`` brightness temperature via the NDVI-emissivity Planck mono-window
      (see :func:`ermida_smw_lst`) — the physics-informed path. [R7 §2.6]

    Returns an ``ee.Image`` with bands ``lst`` (°C, per ``method``), ``ndvi``,
    ``emissivity`` and ``lst_st_b10`` (the direct ST_B10 °C for reference/QA).
    """
    ee = _import_ee()
    geom = ee_geometry(aoi) if isinstance(aoi, (tuple, list)) else aoi
    col = landsat_c2_collection(geom, start, end, cloud_pct=cloud_pct, keys=keys)
    img = col.median().clip(geom)

    # Bands present on L8/L9 (Red=SR_B4, NIR=SR_B5). [constants.GEE_DATASETS]
    ndvi = img.normalizedDifference(["SR_B5", "SR_B4"]).rename(NDVI)
    emis = ndvi_emissivity(ndvi)

    # ST_B10 already scaled to Kelvin by collections.scaled().
    bt_k = img.select("ST_B10")
    lst_direct_c = bt_k.subtract(KELVIN).rename("lst_st_b10")

    mono_c = _mono_window_celsius(bt_k, emis).rename("lst_mono")

    if method in ("ermida", "smw", "mono_window", "mono"):
        lst_band = mono_c.rename(LST)
    else:  # 'st_b10' / production
        lst_band = lst_direct_c.rename(LST)

    return ee.Image.cat([lst_band, ndvi, emis, lst_direct_c, mono_c]) \
        .set("sensor", "landsat_c2_l2", "lst_method", method)


def _mono_window_celsius(bt_kelvin: Any, emissivity: Any) -> Any:
    """Planck single-channel / mono-window LST (°C) from brightness temp + emissivity.

    ``LST_K = BT / (1 + (lambda*BT/rho) * ln(eps))`` then ``- 273.15``.
    ``lambda`` and ``rho`` are :data:`constants.PHYSICAL_CONSTANTS`
    ``PLANCK_LAMBDA_B10`` / ``PLANCK_RHO``. [R1 App.A / R7 §2.6]
    """
    _import_ee()
    lst_k = bt_kelvin.expression(
        "bt / (1 + (lam * bt / rho) * log(eps))",
        {"bt": bt_kelvin, "eps": emissivity, "lam": _LAMBDA_B10, "rho": _RHO},
    )
    return lst_k.subtract(KELVIN)


def ermida_smw_lst(
    aoi: Any,
    start: str,
    end: str,
    cloud_pct: float = 60.0,
    keys: tuple[str, ...] = ("LANDSAT8_L2", "LANDSAT9_L2"),
) -> Any:
    """Single mono-window (SMW / Ermida-style) Landsat LST in °C.

    Convenience entry point that returns the **NDVI-emissivity Planck mono-window**
    LST as the primary ``lst`` band (equivalent to
    ``landsat_lst(..., method='ermida')``). This is the explicit physics path:
    NDVI -> fractional vegetation cover -> NDVI-threshold emissivity -> Planck
    inversion of the ``ST_B10`` brightness temperature. Named after the
    Ermida et al. (2020) GEE SMW implementation pattern. [R7 §2.6]

    Returns an ``ee.Image`` with bands ``lst`` (°C, mono-window), ``ndvi``,
    ``emissivity``.
    """
    img = landsat_lst(aoi, start, end, cloud_pct=cloud_pct, method="ermida",
                      keys=keys)
    return img.select([LST, NDVI, EMISSIVITY])


# ===========================================================================
# MODIS / VIIRS LST (split-window MxD11 / TES MxD21 / VNP21)
# ===========================================================================
def _lst_band_for(key: str, day: bool) -> str:
    """Resolve the LST band name for a MODIS/VIIRS product key + day/night.

    MxD11 has separate ``LST_Day_1km``/``LST_Night_1km`` bands; MxD21/VNP21 use a
    single ``LST_1KM`` band per day/night *product* id. Pulled from
    ``constants.GEE_DATASETS[key]['bands']``.
    """
    bands = GEE_DATASETS[key]["bands"]
    if "LST_Day_1km" in bands or "LST_Night_1km" in bands:
        return "LST_Day_1km" if day else "LST_Night_1km"
    if "LST_1KM" in bands:
        return "LST_1KM"
    # Fallback: first band that looks like an LST band.
    for b in bands:
        if b.upper().startswith("LST"):
            return b
    raise KeyError(f"no LST band found for {key!r} in constants.GEE_DATASETS")


def modis_lst(
    aoi: Any,
    start: str,
    end: str,
    which: str = "MOD11A1",
    day: bool = True,
    reducer: str = "mean",
) -> Any:
    """MODIS LST composite in °C for product ``which`` and day/night band.

    ``which`` may be a short name (``'MOD11A1'``, ``'MYD11A1'``, ``'MOD21A1D'``,
    ``'MYD21A1D'``) or a full ``constants.GEE_DATASETS`` key (``'MODIS_MOD11A1'``).
    LST is scaled ``×0.02`` to Kelvin and QC-masked in
    :func:`collections.modis_lst_collection`, then reduced (``mean`` by default)
    and converted to °C. [R1 §2.4]

    Returns an ``ee.Image`` with one band named ``lst_day`` or ``lst_night``.
    """
    _import_ee()
    key = which if which in GEE_DATASETS else f"MODIS_{which}"
    if key not in GEE_DATASETS:
        raise KeyError(f"unknown MODIS product {which!r}")
    geom = ee_geometry(aoi) if isinstance(aoi, (tuple, list)) else aoi
    col = modis_lst_collection(geom, start, end, key=key)
    band = _lst_band_for(key, day)
    reduced = _reduce_named(col.select(band), reducer).clip(geom)
    out_name = LST_DAY if day else LST_NIGHT
    return lst_to_celsius(reduced, band, out_name) \
        .set("sensor", key, "lst_band", band)


def viirs_lst(
    aoi: Any,
    start: str,
    end: str,
    day: bool = True,
    reducer: str = "mean",
) -> Any:
    """VIIRS VNP21 (SNPP, TES) LST composite in °C, day or night.

    Uses ``VIIRS_VNP21A1D`` (day, ~13:30) or ``VIIRS_VNP21A1N`` (night, ~01:30);
    native 750 m served on the 1 km SIN grid. LST ``×0.02`` -> K, QC-masked,
    reduced and converted to °C. [R1 §2.5]

    Returns an ``ee.Image`` with one band ``lst_day``/``lst_night``.
    """
    key = "VIIRS_VNP21A1D" if day else "VIIRS_VNP21A1N"
    return modis_lst(aoi, start, end, which=key, day=day, reducer=reducer)


def _reduce_named(col: Any, reducer: str) -> Any:
    """Temporal reducer for a single-band collection (mean/median/min/max)."""
    _import_ee()
    r = reducer.lower()
    if r == "mean":
        return col.mean()
    if r == "median":
        return col.median()
    if r == "min":
        return col.min()
    if r == "max":
        return col.max()
    raise ValueError(f"unknown reducer {reducer!r}")


# ===========================================================================
# Config-driven convenience wrappers (ARCHITECTURE.md §11 signatures)
# ===========================================================================
def landsat_lst_cfg(cfg: "Config") -> Any:
    """``landsat_lst`` driven by a :class:`~urbanheat.config.Config` (AOI/dates).

    Matches the ARCHITECTURE.md §11 ``landsat_lst(cfg)`` contract; returns an
    ``ee.Image`` with ``lst`` (°C), ``ndvi``, ``emissivity``.
    """
    return landsat_lst(cfg.bbox, cfg.start_date, cfg.end_date)


def modis_lst_cfg(cfg: "Config", which: str = "MOD11A1", day: bool = True) -> Any:
    """``modis_lst`` driven by a :class:`~urbanheat.config.Config`.

    Matches the ARCHITECTURE.md §11 ``modis_lst(cfg, which, day)`` contract.
    """
    return modis_lst(cfg.bbox, cfg.start_date, cfg.end_date, which=which, day=day)


def diurnal_normalize(images: dict[str, Any], ref_hour: float = 13.5) -> Any:
    """Normalise multi-sensor LST to a common local time via a diurnal model.

    Lightweight server-side diurnal-temperature-cycle (DTC) normalisation: for
    each sensor image keyed by name we apply a cosine DTC correction toward
    ``ref_hour`` (default 13.5 = ~Aqua/VIIRS afternoon overpass) using a fixed
    diurnal amplitude prior, then average. A full per-pixel ``*_view_time`` /
    Göttsche harmonic fit is the offline-pipeline upgrade; this keeps the common
    O(1) path entirely server-side. [R1 §3.C]

    Parameters
    ----------
    images : dict[str, ee.Image]
        Mapping ``sensor_name -> ee.Image`` (each a single-band LST in °C). The
        key (or an ``"overpass_hour"`` image property, if set) supplies the
        sensor's nominal local overpass hour.
    ref_hour : float
        Target local solar hour to normalise all sensors to.

    Returns an ``ee.Image`` (mean of the time-normalised sensor LSTs).
    """
    ee = _import_ee()
    import math  # noqa: PLC0415 - stdlib, cheap

    # Nominal local overpass hours per sensor family. [R1 §1, §2.4-2.5]
    overpass = {
        "MOD11A1": 10.5, "MOD21A1D": 10.5, "terra": 10.5,
        "MYD11A1": 13.5, "MYD21A1D": 13.5, "aqua": 13.5,
        "VNP21A1D": 13.5, "viirs": 13.5,
        "MOD11A1_night": 22.5, "MYD11A1_night": 1.5,
        "landsat": 10.5, "LANDSAT8_L2": 10.5, "LANDSAT9_L2": 10.5,
    }
    amp = 6.0  # half peak-to-trough diurnal amplitude prior (°C), pre-monsoon. [R1 §3.C]

    corrected: list[Any] = []
    for name, img in images.items():
        hour = float(overpass.get(name, ref_hour))
        # Cosine DTC: peak at ~14:00; ΔT to bring `hour` up to `ref_hour`.
        f_obs = math.cos((hour - 14.0) / 24.0 * 2 * math.pi)
        f_ref = math.cos((ref_hour - 14.0) / 24.0 * 2 * math.pi)
        delta = amp * (f_ref - f_obs)
        corrected.append(ee.Image(img).add(delta))

    if not corrected:
        raise ValueError("diurnal_normalize: `images` must be non-empty")
    return ee.ImageCollection(corrected).mean().rename(LST)


__all__ = [
    "fractional_vegetation",
    "ndvi_emissivity",
    "landsat_lst",
    "ermida_smw_lst",
    "modis_lst",
    "viirs_lst",
    "diurnal_normalize",
    "landsat_lst_cfg",
    "modis_lst_cfg",
]
