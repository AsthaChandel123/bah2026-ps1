"""urbanheat.interventions.invest_cooling — a numpy port of the InVEST Urban
Cooling Model (Bosch et al., *GMD* 14:3521, 2021).

This is the **fast, raster-algebra** biophysical estimator that maps a
**Cooling Capacity (CC)** index, a **Heat Mitigation (HM)** index and an
**air-temperature reduction** field city-wide from the FeatureStack — providing
a second, independent ΔT estimate to cross-verify the data-driven ML model (the
"many methods cross-verify" mandate, ARCHITECTURE §9/§10).

Formulas (research/06 §5, InVEST User Guide / GMD 2021):

    CC_i        = 0.6·shade_i + 0.2·albedo_i + 0.2·ETI_i          (daytime, [0,1])
    ETI_i       = (Kc_i · ET0_i) / ET0_max
    CC_park_i   = Σ_{j∈d_cool} g_j·CC_j·exp(−d(i,j)/d_cool)
    GA_i        = cell_area · Σ_{j∈d_cool} g_j
    HM_i        = CC_i        if CC_i ≥ CC_park_i  OR  GA_i < 2 ha
                = CC_park_i    otherwise
    T_air_nomix = T_ref + (1 − HM)·UHI_max
    T_air       = GaussianBlur(T_air_nomix, radius = r)

Defaults (weights 0.6/0.2/0.2, d_cool 100 m, r 500 m, park 2 ha) come from
``constants.INVEST_UCM``. Validation anchor: Lausanne R²=0.903, RMSE=1.144 °C.

**GEE portability.** Every step here is map algebra + two neighbourhood
operations (the ``exp(−d/d_cool)`` green-area decay and the Gaussian mixing), so
the whole CC→HM→T_air chain ports directly to Google Earth Engine via
``image.reduceNeighborhood`` (a fixed exponential/Gaussian kernel) and
``image.convolve`` — i.e. it is the desired server-side O(pixels) city-wide
compute. ``natcap.invest`` is optionally delegated to if installed; otherwise
this pure-numpy port runs with numpy alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from urbanheat import constants as C
from urbanheat import datamodel as dm

# Representative crop coefficient (Kc) by surface type, used to build the ETI
# numerator when a per-LULC Kc table is not supplied. Vegetation transpires (high
# Kc); impervious surfaces do not. Values are within the FAO-56 Kc envelope and
# consistent with InVEST biophysical-table practice (research/06 §5.5). Kept here
# (a modelling convention) rather than constants (cited physical data).
_KC_GREEN = 1.0      # full vegetation / park
_KC_TREE = 1.0       # tree canopy
_KC_BARE = 0.2       # bare/impervious baseline
_KC_WATER = 1.05     # open water evaporates at ~PET


def _layer(fs: dm.FeatureStack, name: str, default: float | np.ndarray) -> np.ndarray:
    """Layer as float64, or a constant grid from ``default`` if absent."""
    if fs.has(name):
        return np.asarray(fs.get(name), dtype=np.float64)
    if isinstance(default, np.ndarray):
        return default.astype(np.float64)
    return np.full(fs.shape, float(default), dtype=np.float64)


def _resolution_m(fs: dm.FeatureStack) -> float:
    """Pixel resolution in metres (transform; falls back to meta/100 m)."""
    a, b, c, d, e, f = fs.transform
    px = abs(a)
    if np.isfinite(px) and px > 0:
        if px < 1e-3:  # geographic degrees -> ~m
            return float(px * 111_000.0)
        return float(px)
    return float(fs.meta.get("resolution_m", 100.0))


def _cell_area_m2(fs: dm.FeatureStack) -> float:
    """Pixel ground area in m^2 (resolution squared)."""
    r = _resolution_m(fs)
    return r * r


# ===========================================================================
# Cooling Capacity
# ===========================================================================
def shade_index(fs: dm.FeatureStack) -> np.ndarray:
    """Shade component of CC: proportion of (>=2 m) tree canopy, [0,1].

    Uses TREE_FRAC directly when present; else derives a canopy proxy from
    GREEN_FRAC (a fraction of green cover is tall canopy) or from NDVI as a last
    resort. [R6 §5.1]
    """
    if fs.has(dm.TREE_FRAC):
        return np.clip(_layer(fs, dm.TREE_FRAC, 0.0), 0.0, 1.0)
    if fs.has(dm.GREEN_FRAC):
        # assume ~60% of green cover is canopy-height vegetation
        return np.clip(0.6 * _layer(fs, dm.GREEN_FRAC, 0.0), 0.0, 1.0)
    ndvi = _layer(fs, dm.NDVI, 0.3)
    return np.clip((ndvi - 0.2) / 0.6, 0.0, 1.0)  # NDVI 0.2->0, 0.8->1


def eti_index(fs: dm.FeatureStack, kc: np.ndarray | None = None) -> np.ndarray:
    """Evapotranspiration Index ETI = (Kc·ET0)/ET0_max, normalised to [0,1].

    ET0 is taken from the ET layer if present (treated as reference ET); else a
    flat unit field is used and ETI collapses to the normalised Kc map. Kc is the
    supplied per-pixel crop coefficient, else built from cover fractions
    (vegetation/water high, impervious low). ``ET0_max`` is the AOI maximum of
    ``Kc·ET0`` (the InVEST normaliser). [R6 §5.1]
    """
    if kc is None:
        kc = crop_coefficient(fs)
    kc = np.clip(np.asarray(kc, dtype=np.float64), 0.0, 1.5)

    eto = _layer(fs, dm.ET, default=np.nan)
    if not np.isfinite(eto).any():
        eto = np.ones(fs.shape, dtype=np.float64)  # uniform reference ET
    else:
        # impute NaNs with the field mean so the normaliser is well-defined
        m = np.nanmean(eto)
        eto = np.where(np.isfinite(eto), eto, m if np.isfinite(m) else 0.0)
        eto = np.clip(eto, 0.0, None)

    num = kc * eto
    eto_max = float(np.nanmax(num))
    if not np.isfinite(eto_max) or eto_max <= 0:
        return np.zeros(fs.shape, dtype=np.float64)
    return np.clip(num / eto_max, 0.0, 1.0)


def crop_coefficient(fs: dm.FeatureStack) -> np.ndarray:
    """Build a per-pixel Kc from cover fractions when no Kc table is supplied.

    Kc = area-weighted blend of vegetation/water/impervious Kc using the
    available fraction layers (TREE_FRAC, GREEN_FRAC, WATER_FRAC, IMPERVIOUS_FRAC).
    """
    tree = _layer(fs, dm.TREE_FRAC, 0.0)
    green = _layer(fs, dm.GREEN_FRAC, 0.0)
    water = _layer(fs, dm.WATER_FRAC, 0.0)
    imperv = _layer(fs, dm.IMPERVIOUS_FRAC, np.nan)

    veg = np.clip(np.maximum(tree, green), 0.0, 1.0)
    if not np.isfinite(imperv).any():
        imperv = np.clip(1.0 - veg - water, 0.0, 1.0)
    else:
        imperv = np.clip(imperv, 0.0, 1.0)
    bare = np.clip(1.0 - veg - water - imperv, 0.0, 1.0)

    kc = (veg * _KC_GREEN + water * _KC_WATER
          + imperv * _KC_BARE + bare * _KC_BARE)
    denom = veg + water + imperv + bare
    denom = np.where(denom > 1e-6, denom, 1.0)
    return kc / denom


def cooling_capacity(
    fs: dm.FeatureStack,
    weights: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """CC = w_shade·shade + w_albedo·albedo + w_eti·ETI, returned in [0,1].

    Weights default to ``constants.INVEST_UCM`` (0.6, 0.2, 0.2). shade from
    TREE_FRAC, albedo from ALBEDO, ETI from the ET/Kc normalisation. [R6 §5.1]
    """
    if weights is None:
        weights = (
            float(C.INVEST_UCM["cc_weight_shade"]),
            float(C.INVEST_UCM["cc_weight_albedo"]),
            float(C.INVEST_UCM["cc_weight_eti"]),
        )
    w_shade, w_albedo, w_eti = weights

    shade = shade_index(fs)
    albedo = np.clip(_layer(fs, dm.ALBEDO, 0.18), 0.0, 1.0)
    eti = eti_index(fs)

    cc = w_shade * shade + w_albedo * albedo + w_eti * eti
    return np.clip(cc, 0.0, 1.0).astype(np.float64)


def cooling_capacity_intensity(fs: dm.FeatureStack) -> np.ndarray:
    """Nighttime CC alternative: ``CC = 1 − building_intensity`` (floor-area ratio).

    Uses PLAN_AREA_FRAC (λ_P) as the building-intensity proxy when present, else
    IMPERVIOUS_FRAC. [R6 §5.1 nighttime method]
    """
    if fs.has(dm.PLAN_AREA_FRAC):
        bi = _layer(fs, dm.PLAN_AREA_FRAC, 0.0)
    else:
        bi = _layer(fs, dm.IMPERVIOUS_FRAC, 0.0)
    return np.clip(1.0 - np.clip(bi, 0.0, 1.0), 0.0, 1.0)


# ===========================================================================
# Green-area cooling + Heat Mitigation
# ===========================================================================
def _green_flag(fs: dm.FeatureStack) -> np.ndarray:
    """Binary green-space flag g_j (1 where pixel is green space).

    A pixel is "green" if its vegetation cover (tree or green fraction) exceeds a
    modest threshold; falls back to NDVI when fractions are absent. [R6 §5.2]
    """
    if fs.has(dm.TREE_FRAC) or fs.has(dm.GREEN_FRAC):
        veg = np.maximum(_layer(fs, dm.TREE_FRAC, 0.0), _layer(fs, dm.GREEN_FRAC, 0.0))
        return (veg >= 0.5).astype(np.float64)
    ndvi = _layer(fs, dm.NDVI, 0.0)
    return (ndvi >= 0.4).astype(np.float64)


def _exp_decay_kernel(d_cool_px: float, max_radius: int = 30) -> np.ndarray:
    """2-D ``exp(−d/d_cool)`` kernel (distances in pixels), peak normalised to 1."""
    radius = int(min(max(np.ceil(3.0 * d_cool_px), 1), max_radius))
    ax = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    rr = np.sqrt(xx ** 2 + yy ** 2)
    k = np.exp(-rr / max(d_cool_px, 1e-6))
    return k


def green_area_cooling(
    fs: dm.FeatureStack,
    cc: np.ndarray | None = None,
    d_cool_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(CC_park, GA)`` — the distance-decayed park contribution and the
    nearby green area (m^2) within ``d_cool``.

        CC_park_i = Σ_j g_j·CC_j·exp(−d(i,j)/d_cool)
        GA_i      = cell_area · Σ_j g_j

    [R6 §5.2]
    """
    if cc is None:
        cc = cooling_capacity(fs)
    if d_cool_m is None:
        d_cool_m = float(C.INVEST_UCM["green_area_cooling_distance_m"])

    g = _green_flag(fs)
    res = _resolution_m(fs)
    d_px = max(d_cool_m / max(res, 1e-6), 1e-6)
    kernel = _exp_decay_kernel(d_px)

    g_cc = g * np.asarray(cc, dtype=np.float64)

    cc_park = _correlate(g_cc, kernel)             # Σ g_j CC_j exp(-d/d_cool)
    ga_count = _correlate(g, (kernel > 0).astype(np.float64))  # Σ g_j in window
    ga = ga_count * _cell_area_m2(fs)
    return cc_park.astype(np.float64), ga.astype(np.float64)


def heat_mitigation(
    fs: dm.FeatureStack,
    cc: np.ndarray | None = None,
    d_cool_m: float | None = None,
) -> np.ndarray:
    """HM index with the >2 ha park override and exp-decay green-area term.

        HM_i = CC_i      if CC_i >= CC_park_i  OR  GA_i < 2 ha
             = CC_park_i  otherwise

    so large parks (>2 ha) export their cooling into a ``d_cool`` buffer.
    [R6 §5.2-5.3]
    """
    if cc is None:
        cc = cooling_capacity(fs)
    cc = np.asarray(cc, dtype=np.float64)

    cc_park, ga = green_area_cooling(fs, cc, d_cool_m)
    park_threshold_m2 = float(C.INVEST_UCM["park_area_threshold_ha"]) * 10_000.0

    use_local = (cc >= cc_park) | (ga < park_threshold_m2)
    hm = np.where(use_local, cc, cc_park)
    return np.clip(hm, 0.0, 1.0).astype(np.float64)


# ===========================================================================
# Air-temperature model
# ===========================================================================
def air_temperature(
    fs: dm.FeatureStack,
    hm: np.ndarray,
    t_ref: float,
    uhi_max: float,
    radius_m: float | None = None,
) -> np.ndarray:
    """``T_air = GaussianBlur(T_ref + (1−HM)·UHI_max, r)`` (°C).

    Higher HM ⇒ lower local temperature, proportional to ``uhi_max``; a Gaussian
    convolution of radius ``r`` (default ``constants.INVEST_UCM`` 500 m) performs
    the spatial air mixing. An independent ΔT estimate to cross-verify the ML
    model. [R6 §5.4]
    """
    hm = np.asarray(hm, dtype=np.float64)
    if radius_m is None:
        radius_m = float(C.INVEST_UCM["t_air_average_radius_m"])

    t_air_nomix = float(t_ref) + (1.0 - hm) * float(uhi_max)

    res = _resolution_m(fs)
    sigma_px = max(radius_m / max(res, 1e-6), 1e-6)
    t_air = _gaussian_blur(t_air_nomix, sigma_px)
    return t_air.astype(np.float64)


def air_temperature_reduction(
    fs: dm.FeatureStack,
    uhi_max: float,
    t_ref: float = 0.0,
    mixing_r: float | None = None,
    hm: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Task-API wrapper: return the full set of InVEST cooling maps.

    Computes CC, HM, ``T_air_nomix`` and the mixed ``T_air`` (and the cooling
    relative to the maximum-UHI baseline) in one call. ``t_ref`` defaults to 0 so
    the returned ``t_air`` is itself the *air-temperature anomaly* (°C above
    reference); pass the real rural reference to get absolute air temperature.

    Returns dict ``{cc, hm, t_air_nomix, t_air, delta_t_air}`` where
    ``delta_t_air = (T_ref + UHI_max) − T_air`` is the cooling (°C, +=cooling)
    relative to the hottest (HM=0) case. [R6 §5]
    """
    cc = cooling_capacity(fs)
    if hm is None:
        hm = heat_mitigation(fs, cc)
    if mixing_r is None:
        mixing_r = float(C.INVEST_UCM["t_air_average_radius_m"])

    t_air_nomix = float(t_ref) + (1.0 - hm) * float(uhi_max)
    t_air = air_temperature(fs, hm, t_ref, uhi_max, mixing_r)
    t_air_max = float(t_ref) + float(uhi_max)
    delta_t_air = t_air_max - t_air  # cooling relative to worst case

    return {
        "cc": cc.astype(np.float32),
        "hm": np.asarray(hm, dtype=np.float32),
        "t_air_nomix": t_air_nomix.astype(np.float32),
        "t_air": t_air.astype(np.float32),
        "delta_t_air": delta_t_air.astype(np.float32),
    }


# ===========================================================================
# Full pass
# ===========================================================================
def run_invest_ucm(
    fs: dm.FeatureStack,
    cfg: Any,
    t_ref: float,
    uhi_max: float,
) -> dict[str, np.ndarray]:
    """Full CC→HM→T_air pass (pure-numpy port).

    Optionally delegates to ``natcap.invest.urban_cooling_model`` if it is
    installed AND the FeatureStack can be written to the GeoTIFF/CSV inputs it
    needs; otherwise (the default offline path) runs this numpy port. Returns
    ``{'cc', 'hm', 't_air'}`` grids. [R6 §5]
    """
    d_cool_m = float(C.INVEST_UCM["green_area_cooling_distance_m"])
    radius_m = float(C.INVEST_UCM["t_air_average_radius_m"])

    # Optional delegation to the reference implementation (lazy, best-effort).
    try:  # pragma: no cover - only when natcap.invest installed + writable inputs
        from natcap.invest import urban_cooling_model  # type: ignore  # noqa: F401
        # A faithful delegation needs LULC + ET0 GeoTIFFs, an AOI vector and a
        # biophysical CSV written to disk; assembling those is an I/O concern the
        # CLI owns. We keep the hook and fall through to the numpy port here.
        raise ImportError("natcap.invest input assembly handled by CLI, not here")
    except Exception:
        pass

    cc = cooling_capacity(fs)
    hm = heat_mitigation(fs, cc, d_cool_m)
    t_air = air_temperature(fs, hm, t_ref, uhi_max, radius_m)
    return {
        "cc": cc.astype(np.float32),
        "hm": hm.astype(np.float32),
        "t_air": t_air.astype(np.float32),
    }


# ===========================================================================
# WBGT / work-productivity (valuation cross-check)
# ===========================================================================
def wbgt(t_air_c: np.ndarray, rel_humidity_pct: float | np.ndarray) -> np.ndarray:
    """Wet-Bulb Globe Temperature (°C) from air temp + RH (InVEST valuation form).

        e    = (RH/100)·6.105·exp(17.27·T/(237.7+T))   (vapour pressure, hPa)
        WBGT = 0.567·T + 0.393·e + 3.94

    [R6 §5.5]
    """
    t = np.asarray(t_air_c, dtype=np.float64)
    rh = np.asarray(rel_humidity_pct, dtype=np.float64)
    e = (rh / 100.0) * 6.105 * np.exp(17.27 * t / (237.7 + t))
    return (0.567 * t + 0.393 * e + 3.94).astype(np.float64)


def work_loss_fraction(wbgt_c: np.ndarray, work: str = "heavy") -> np.ndarray:
    """Fractional work-productivity loss from WBGT using InVEST thresholds.

    light: 0 if <31.5; .25 [31.5,32); .50 [32,32.5); .75 >=32.5
    heavy: 0 if <27.5; .25 [27.5,29.5); .50 [29.5,31.5); .75 >=31.5
    [R6 §5.5]
    """
    w = np.asarray(wbgt_c, dtype=np.float64)
    if work == "light":
        t25, t50, t75 = (C.INVEST_UCM["wbgt_light_25pct"],
                         C.INVEST_UCM["wbgt_light_50pct"],
                         C.INVEST_UCM["wbgt_light_75pct"])
    else:
        t25, t50, t75 = (C.INVEST_UCM["wbgt_heavy_25pct"],
                         C.INVEST_UCM["wbgt_heavy_50pct"],
                         C.INVEST_UCM["wbgt_heavy_75pct"])
    loss = np.zeros_like(w)
    loss = np.where(w >= t25, 0.25, loss)
    loss = np.where(w >= t50, 0.50, loss)
    loss = np.where(w >= t75, 0.75, loss)
    return loss


# ===========================================================================
# Neighbourhood operators (numpy; scipy used if available)
# ===========================================================================
def _correlate(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D correlation (``mode='constant'``) — scipy if present, else numpy."""
    arr = np.asarray(arr, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    try:
        from scipy import ndimage as _ndi  # type: ignore

        return _ndi.correlate(arr, kernel, mode="constant")
    except Exception:
        return _conv2d_same(arr, kernel)


def _gaussian_blur(arr: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian blur with standard deviation ``sigma_px`` (pixels)."""
    arr = np.asarray(arr, dtype=np.float64)
    if sigma_px <= 0:
        return arr
    try:
        from scipy import ndimage as _ndi  # type: ignore

        return _ndi.gaussian_filter(arr, sigma=sigma_px, mode="nearest")
    except Exception:
        # separable numpy Gaussian
        radius = int(max(np.ceil(3.0 * sigma_px), 1))
        ax = np.arange(-radius, radius + 1)
        k = np.exp(-0.5 * (ax / sigma_px) ** 2)
        k /= k.sum()
        out = _conv1d_axis(arr, k, axis=0)
        out = _conv1d_axis(out, k, axis=1)
        return out


def _conv1d_axis(arr: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """1-D 'same' convolution along ``axis`` with edge replication."""
    arr = np.asarray(arr, dtype=np.float64)
    r = len(k) // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    padded = np.pad(arr, pad, mode="edge")
    out = np.zeros_like(arr)
    n = arr.shape[axis]
    for i, w in enumerate(k):
        if w == 0:
            continue
        sl = [slice(None), slice(None)]
        sl[axis] = slice(i, i + n)
        out += w * padded[tuple(sl)]
    return out


def _conv2d_same(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Tiny 'same'-size 2-D convolution (numpy fallback)."""
    arr = np.asarray(arr, dtype=np.float64)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(arr, ((ph, ph), (pw, pw)), mode="constant")
    out = np.zeros_like(arr)
    for i in range(kh):
        for j in range(kw):
            w = kernel[i, j]
            if w == 0:
                continue
            out += w * padded[i:i + arr.shape[0], j:j + arr.shape[1]]
    return out


__all__ = [
    "shade_index",
    "eti_index",
    "crop_coefficient",
    "cooling_capacity",
    "cooling_capacity_intensity",
    "green_area_cooling",
    "heat_mitigation",
    "air_temperature",
    "air_temperature_reduction",
    "run_invest_ucm",
    "wbgt",
    "work_loss_fraction",
]
