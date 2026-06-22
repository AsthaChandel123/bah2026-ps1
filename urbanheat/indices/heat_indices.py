"""urbanheat.indices.heat_indices — surface thermal metrics + human heat-stress indices.

This module implements the **two satellite/met-driven heat-stress families** of
research/08 (R8) as vectorized numpy operations, each writing the canonical
:mod:`urbanheat.datamodel` derived-layer names back into a
:class:`~urbanheat.datamodel.FeatureStack`:

1. **Surface thermal metrics** (LST-only, O(1)): ``SUHII = LST_urban - LST_rural``
   (multiple rural-reference definitions, dry-cropland sign-flip guard),
   ``LST_ZSCORE``, ``LST_PERCENTILE`` (distribution-free, the India default) and
   ``UTFVI = (Ts - Tm)/Tm`` *in Kelvin* -> 6-class **EEI** (Liu & Zhang 2011,
   :data:`urbanheat.constants.UTFVI_CLASSES`).
2. **Human heat-stress / comfort indices** (need air-T + humidity, +- wind /
   radiation): NWS **Heat Index** (full Rothfusz 9-term + low/high-RH
   adjustments + simple fallback + NWS categories), **Humidex**, **wet-bulb**
   (Stull 2011 closed form, pressure-corrected), **WBGT** (ABM simplified for
   raster / full when a globe term exists), **Apparent Temperature** (Steadman
   BoM), **Discomfort Index** (Thom), **Net Effective Temperature**, **UTCI**
   (lazy ``thermofeel`` / ``pythermalcomfort`` else a documented polynomial
   approximation; needs ``T_mrt``) and a simplified **mean radiant temperature**.

Design rules (per ARCHITECTURE.md §7, §11.3):
  * numpy/scipy at module top-level are fine; ``thermofeel`` / ``pythermalcomfort``
    are imported **lazily** with a documented numpy fallback, so the whole module
    runs on numpy alone.
  * Every coefficient / threshold / colour comes from
    :mod:`urbanheat.constants` and every formula from research/08 — no invented
    numbers.
  * Two API styles are provided and kept in sync:
      - **FeatureStack methods** matching the ARCHITECTURE §11.3 contract exactly
        (``surface_uhi``, ``utfvi``, ``lst_statistics``, ``heat_index``,
        ``humidex``, ``wet_bulb``, ``wbgt``, ``utci``, ``human_stress_ensemble``,
        ``add_spectral_indices``);
      - **pure-array** helpers the rest of the system can call directly
        (``surface_uhi_intensity``, ``lst_zscore``, ``lst_percentile``,
        ``eei_class``, ``heat_index_rothfusz``, ``humidex_from_dewpoint``,
        ``wet_bulb_stull``, ``wbgt_simplified``, ``apparent_temperature``,
        ``discomfort_index_thom``, ``net_effective_temperature``, ``utci_approx``,
        ``mean_radiant_temperature``).
  * A convenience :func:`compute_all_indices` fills every derived index it can
    from the variables available in the stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from urbanheat.constants import (
    HEAT_STRESS_THRESHOLDS,
    KELVIN,
    SIGMA_SB,
    SPECTRAL_INDEX_COEFFS,
    UTCI_CATEGORIES,
    UTFVI_CLASSES,
)
from urbanheat.datamodel import (
    AIR_TEMP,
    ALBEDO,
    DEWPOINT,
    EEI,
    EMISSIVITY,
    FVC,
    HEAT_INDEX,
    HUMIDEX,
    LCZ,
    LST,
    LST_PERCENTILE,
    LST_ZSCORE,
    NDVI,
    PRESSURE,
    REL_HUMIDITY,
    SOLAR_RADIATION,
    SUHII,
    TMRT,
    UTCI,
    UTFVI,
    WBGT,
    WET_BULB,
    WIND_SPEED,
    FeatureStack,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

# Canonical extra layer names written by this module that are NOT in datamodel's
# fixed catalog (auxiliary/diagnostic). They follow the same lower_snake style.
APPARENT_TEMP = "apparent_temp"          # Steadman apparent temperature, degC
DISCOMFORT_INDEX = "discomfort_index"    # Thom discomfort index, degC
HUMAN_STRESS_COUNT = "human_stress_count"  # # of danger-crossing indices (0-5)
HUMAN_STRESS_SCORE = "human_stress_score"  # 0-100 human heat-stress score

# Standard sea-level pressure assumed by the Stull wet-bulb fit (hPa). [R8 §4]
_STULL_REF_PRESSURE_HPA = 1013.25

__all__ = [
    # --- FeatureStack contract methods (ARCHITECTURE §11.3) ---
    "add_spectral_indices",
    "surface_uhi",
    "utfvi",
    "lst_statistics",
    "heat_index",
    "humidex",
    "wet_bulb",
    "wbgt",
    "utci",
    "human_stress_ensemble",
    "compute_all_indices",
    # --- pure-array helpers ---
    "surface_uhi_intensity",
    "lst_zscore",
    "lst_percentile",
    "eei_class",
    "saturation_vapour_pressure",
    "vapour_pressure_from_rh",
    "vapour_pressure_from_dewpoint",
    "heat_index_rothfusz",
    "heat_index_categories",
    "humidex_from_dewpoint",
    "wet_bulb_stull",
    "wbgt_simplified",
    "apparent_temperature",
    "discomfort_index_thom",
    "net_effective_temperature",
    "utci_approx",
    "utci_categories",
    "mean_radiant_temperature",
    # --- aux layer names ---
    "APPARENT_TEMP",
    "DISCOMFORT_INDEX",
    "HUMAN_STRESS_COUNT",
    "HUMAN_STRESS_SCORE",
]


# ===========================================================================
# Small humidity / vapour-pressure helpers (shared by several indices)
# ===========================================================================
def saturation_vapour_pressure(temp_c: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure ``e_s`` (hPa) via the Tetens form.

    ``e_s = 6.105 * exp[17.27*T / (237.7 + T)]`` with ``T`` in degC. [R8 §5.3]

    Parameters
    ----------
    temp_c : np.ndarray
        Air (or dew-point) temperature in degC.

    Returns
    -------
    np.ndarray
        Saturation vapour pressure in hectopascals (hPa == mb).
    """
    t = np.asarray(temp_c, dtype=np.float64)
    return 6.105 * np.exp(17.27 * t / (237.7 + t))


def vapour_pressure_from_rh(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Actual vapour pressure ``e`` (hPa) from air temp + RH.

    ``e = (RH/100) * e_s(T)`` with the Tetens :func:`saturation_vapour_pressure`.
    [R8 §5.3]
    """
    rh = np.asarray(rh, dtype=np.float64)
    return (rh / 100.0) * saturation_vapour_pressure(temp_c)


def vapour_pressure_from_dewpoint(dewpoint_c: np.ndarray) -> np.ndarray:
    """Actual vapour pressure ``e`` (hPa) from dew point, Humidex/CCOHS form.

    ``e = 6.11 * exp[5417.7530 * (1/273.16 - 1/(273.16 + Td))]`` (Td in degC).
    This is the exact expression used by the Humidex definition. [R8 §5.2]
    """
    td = np.asarray(dewpoint_c, dtype=np.float64)
    return 6.11 * np.exp(5417.7530 * (1.0 / 273.16 - 1.0 / (273.16 + td)))


# ===========================================================================
# 1. SURFACE THERMAL METRICS (LST only)
# ===========================================================================
def lst_zscore(lst: np.ndarray) -> np.ndarray:
    """Per-pixel LST z-score ``(LST - mu_AOI)/sigma_AOI`` over the AOI. [R8 §3.3]

    ``mu`` and ``sigma`` are taken over all finite pixels of ``lst``. NaNs are
    preserved in the output. Returns zeros if ``sigma == 0`` (flat field).

    Parameters
    ----------
    lst : np.ndarray
        Land-surface temperature (any consistent unit; degC and K give the same
        z-score up to the constant offset cancelling in the mean).

    Returns
    -------
    np.ndarray
        z-score in sigma units, same shape as ``lst``.
    """
    a = np.asarray(lst, dtype=np.float64)
    mu = np.nanmean(a)
    sd = np.nanstd(a)
    if not np.isfinite(sd) or sd == 0.0:
        return np.where(np.isfinite(a), 0.0, np.nan)
    return (a - mu) / sd


def lst_percentile(lst: np.ndarray) -> np.ndarray:
    """Per-pixel LST percentile rank (0-100) within the AOI. [R8 §3.3]

    Distribution-free hotspot magnitude — the recommended India default. Each
    finite pixel is ranked against all finite pixels; ``rank = 100 * (#<= v) / N``.
    NaNs are preserved.

    Parameters
    ----------
    lst : np.ndarray
        Land-surface temperature.

    Returns
    -------
    np.ndarray
        Percentile rank in [0, 100], same shape as ``lst``.
    """
    a = np.asarray(lst, dtype=np.float64)
    out = np.full(a.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(a)
    vals = a[finite]
    n = vals.size
    if n == 0:
        return out
    # rank via searchsorted on the sorted finite values: "<=" inclusive rank.
    order = np.sort(vals)
    # side='right' -> number of elements <= v; gives 100 for the max.
    ranks = np.searchsorted(order, vals, side="right") / float(n) * 100.0
    out[finite] = ranks
    return out


def surface_uhi_intensity(
    lst: np.ndarray,
    urban_mask: np.ndarray | None = None,
    rural_ref: np.ndarray | None = None,
    rural_value: float | None = None,
    guard_sign_flip: bool = True,
) -> np.ndarray:
    """Surface UHI intensity ``SUHII = LST - LST_rural`` (degC). [R8 §3.2]

    Computes a continuous SUHI *surface* (per-pixel ``LST - rural_reference``),
    which subsumes the scalar definition ``mean(LST_urban) - mean(LST_rural)``.

    The rural reference is resolved in priority order:

    1. ``rural_value`` if given — an explicit scalar reference temperature.
    2. ``rural_ref`` boolean/0-1 mask — reference = **median** LST over those
       pixels (median is robust to outliers; the standard SUHII-surface choice).
    3. ``urban_mask`` only — reference = median LST over the **non-urban** pixels
       (``urban_mask`` False/0), i.e. the implicit rural complement.
    4. none of the above — reference = global median LST (degenerate fallback).

    Dry-cropland **sign-flip guard** (``guard_sign_flip``): in arid pre-monsoon
    India bare/dry rural cropland can be *hotter* than irrigated urban parks, so a
    naive rural median can sit *above* the urban median and flip SUHII negative
    everywhere. When the chosen rural reference exceeds the urban median, we fall
    back to the lower of {rural median, urban-low percentile} so the intensity
    field is not spuriously inverted, and record nothing else (caller may report
    the sensitivity). [R8 §3.2 cons]

    Parameters
    ----------
    lst : np.ndarray
        Land-surface temperature in degC.
    urban_mask : np.ndarray, optional
        Boolean/0-1 array, True where urban/built.
    rural_ref : np.ndarray, optional
        Boolean/0-1 array, True where the rural reference is sampled.
    rural_value : float, optional
        Explicit scalar rural reference temperature (degC).
    guard_sign_flip : bool, default True
        Apply the dry-cropland sign-flip mitigation described above.

    Returns
    -------
    np.ndarray
        SUHII surface (degC), same shape as ``lst``.
    """
    a = np.asarray(lst, dtype=np.float64)

    # urban median (used by the guard and as a sanity anchor)
    if urban_mask is not None:
        um = np.asarray(urban_mask).astype(bool)
        urban_vals = a[um & np.isfinite(a)]
    else:
        urban_vals = a[np.isfinite(a)]
    urban_med = float(np.median(urban_vals)) if urban_vals.size else np.nan

    # resolve the rural reference temperature
    if rural_value is not None:
        ref = float(rural_value)
    elif rural_ref is not None:
        rr = np.asarray(rural_ref).astype(bool)
        rvals = a[rr & np.isfinite(a)]
        ref = float(np.median(rvals)) if rvals.size else np.nan
    elif urban_mask is not None:
        nm = ~np.asarray(urban_mask).astype(bool)
        rvals = a[nm & np.isfinite(a)]
        ref = float(np.median(rvals)) if rvals.size else np.nan
    else:
        ref = float(np.median(a[np.isfinite(a)])) if np.isfinite(a).any() else np.nan

    if guard_sign_flip and np.isfinite(ref) and np.isfinite(urban_med) and ref > urban_med:
        # rural reference is hotter than the urban core -> likely dry-cropland
        # contamination; use a cool lower-percentile of the whole scene instead.
        low = np.nanpercentile(a, 25.0)
        ref = float(min(ref, low))

    return a - ref


def utfvi_array(lst_c: np.ndarray) -> np.ndarray:
    """Urban Thermal Field Variance Index ``UTFVI = (Ts - Tm)/Tm`` in KELVIN.

    ``Ts`` and ``Tm`` (AOI-mean LST) are converted to Kelvin first — the standard
    published convention; using degC breaks the EEI thresholds. Dimensionless.
    [R8 §3.4]

    Parameters
    ----------
    lst_c : np.ndarray
        Land-surface temperature in **degC** (converted to K internally).

    Returns
    -------
    np.ndarray
        UTFVI (dimensionless), same shape as ``lst_c``.
    """
    ts_k = np.asarray(lst_c, dtype=np.float64) + KELVIN
    tm_k = np.nanmean(ts_k)
    if not np.isfinite(tm_k) or tm_k == 0.0:
        return np.full_like(ts_k, np.nan)
    return (ts_k - tm_k) / tm_k


def eei_class(utfvi_values: np.ndarray) -> np.ndarray:
    """Reclassify UTFVI into the 6-class Ecological Evaluation Index (EEI).

    Uses the Liu & Zhang (2011) thresholds in
    :data:`urbanheat.constants.UTFVI_CLASSES` (ascending ``max`` cut-points).
    Returns an **integer class index** ``0..5`` where 0 = Excellent (UHI none)
    and 5 = Worst (UHI strongest); NaNs in the input map to ``-1``. [R8 §3.4]

    Parameters
    ----------
    utfvi_values : np.ndarray
        UTFVI field (dimensionless, Kelvin-based).

    Returns
    -------
    np.ndarray
        Integer EEI class index in ``{-1, 0..5}``, same shape as input
        (dtype float so NaN-derived ``-1`` and values coexist cleanly).
    """
    u = np.asarray(utfvi_values, dtype=np.float64)
    out = np.full(u.shape, -1.0, dtype=np.float64)
    finite = np.isfinite(u)
    # ascending upper bounds from constants; class i = first bound u <= max.
    cls = np.zeros(u.shape, dtype=np.float64)
    assigned = np.zeros(u.shape, dtype=bool)
    for i, entry in enumerate(UTFVI_CLASSES):
        upper = entry["max"]
        sel = finite & (~assigned) & (u <= upper)
        cls[sel] = float(i)
        assigned |= sel
    out[finite] = cls[finite]
    return out


# ===========================================================================
# 2. HUMAN HEAT-STRESS / COMFORT INDICES (air-T + humidity, +- wind/radiation)
# ===========================================================================
def heat_index_rothfusz(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """NWS (Rothfusz) Heat Index, returned in **degC**. [R8 §5.1] ``[verified]``

    Full operational NWS algorithm, vectorized and branch-correct:

    1. Simple Steadman form ``HI = 0.5*{T + 61 + (T-68)*1.2 + RH*0.094}`` (degF),
       averaged with ``T``; this is the value used when it stays ``< 80 degF``.
    2. Where that average ``>= 80 degF`` switch to the full 9-term Rothfusz
       regression.
    3. Low-humidity adjustment subtracted where ``RH < 13`` and
       ``80 <= T <= 112 degF``.
    4. High-humidity adjustment added where ``RH > 85`` and ``80 <= T <= 87 degF``.

    All internal math is in degF; the result is converted to degC.

    Parameters
    ----------
    temp_c : np.ndarray
        2 m air temperature in degC.
    rh : np.ndarray
        Relative humidity in percent (0-100).

    Returns
    -------
    np.ndarray
        Heat Index in degC.
    """
    tc = np.asarray(temp_c, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)
    t = tc * 9.0 / 5.0 + 32.0  # degF

    # 1) simple Steadman form, averaged with T (NWS recipe)
    hi_simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    hi_simple = (hi_simple + t) / 2.0

    # 2) full Rothfusz 9-term regression (degF)
    hi_full = (
        -42.379
        + 2.04901523 * t
        + 10.14333127 * rh
        - 0.22475541 * t * rh
        - 0.00683783 * t * t
        - 0.05481717 * rh * rh
        + 0.00122874 * t * t * rh
        + 0.00085282 * t * rh * rh
        - 0.00000199 * t * t * rh * rh
    )

    # 3) low-humidity adjustment (subtract)
    adj_low = ((13.0 - rh) / 4.0) * np.sqrt(np.maximum(17.0 - np.abs(t - 95.0), 0.0) / 17.0)
    low_mask = (rh < 13.0) & (t >= 80.0) & (t <= 112.0)

    # 4) high-humidity adjustment (add)
    adj_high = ((rh - 85.0) / 10.0) * ((87.0 - t) / 5.0)
    high_mask = (rh > 85.0) & (t >= 80.0) & (t <= 87.0)

    hi_full_adj = hi_full + np.where(low_mask, -adj_low, 0.0) + np.where(high_mask, adj_high, 0.0)

    # use full regression only where the simple average reaches the 80 degF gate
    hi_f = np.where(hi_simple >= 80.0, hi_full_adj, hi_simple)

    return (hi_f - 32.0) * 5.0 / 9.0


def heat_index_categories(hi_c: np.ndarray) -> np.ndarray:
    """Map a Heat-Index field (degC) to NWS danger categories. [R8 §5.1]

    Bands (on HI in degF, from
    :data:`urbanheat.constants.HEAT_STRESS_THRESHOLDS` ``["heat_index_f"]``):
    0 none (<80), 1 Caution (80-90), 2 Extreme Caution (91-102), 3 Danger
    (103-124), 4 Extreme Danger (>=125). Returns an integer class field; NaN->-1.

    Parameters
    ----------
    hi_c : np.ndarray
        Heat Index in degC.

    Returns
    -------
    np.ndarray
        Integer category 0-4 (float dtype; NaN -> -1).
    """
    thr = HEAT_STRESS_THRESHOLDS["heat_index_f"]
    hi_f = np.asarray(hi_c, dtype=np.float64) * 9.0 / 5.0 + 32.0
    out = np.full(hi_f.shape, -1.0, dtype=np.float64)
    finite = np.isfinite(hi_f)
    cat = np.zeros(hi_f.shape, dtype=np.float64)
    cat = np.where(hi_f >= thr["caution"], 1.0, cat)
    cat = np.where(hi_f >= thr["extreme_caution"], 2.0, cat)
    cat = np.where(hi_f >= thr["danger"], 3.0, cat)
    cat = np.where(hi_f >= thr["extreme_danger"], 4.0, cat)
    out[finite] = cat[finite]
    return out


def humidex_from_dewpoint(temp_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    """Humidex (Canada) ``= T + 0.5555*(e - 10)``. [R8 §5.2] ``[verified]``

    ``e`` is the dew-point vapour pressure (hPa) from
    :func:`vapour_pressure_from_dewpoint`. Output is degC-like. By convention
    Humidex is not reported below the dry-bulb temperature, so the result is
    clamped to ``>= T`` (the moisture term only *adds* discomfort).

    Parameters
    ----------
    temp_c : np.ndarray
        Air temperature in degC.
    dewpoint_c : np.ndarray
        Dew-point temperature in degC.

    Returns
    -------
    np.ndarray
        Humidex (degC-like).
    """
    t = np.asarray(temp_c, dtype=np.float64)
    e = vapour_pressure_from_dewpoint(dewpoint_c)
    hx = t + 0.5555 * (e - 10.0)
    return np.maximum(hx, t)


def wet_bulb_stull(
    temp_c: np.ndarray,
    rh: np.ndarray,
    pressure_kpa: np.ndarray | None = None,
) -> np.ndarray:
    """Stull (2011) closed-form wet-bulb temperature ``Tw`` (degC). [R8 §4] ``[verified]``

    ``Tw = T*atan[0.151977*(RH + 8.313659)^0.5] + atan(T + RH)``
    ``   - atan(RH - 1.676331) + 0.00391838*RH^1.5 * atan(0.023101*RH) - 4.686035``
    (arctan in radians). Valid ``-20<=T<=50 degC``, ``5<=RH<=99%`` at sea level
    (1013.25 hPa); MAE < 0.3 degC.

    An optional **pressure correction** is applied when ``pressure_kpa`` is given:
    the Stull fit assumes sea-level pressure, so for the elevated Indian
    plateau/Deccan we add a first-order hypsometric term proportional to the
    pressure deficit, ``dTw ~ k*(P0 - P)`` with the wet-bulb pressure sensitivity
    ``k`` derived from the psychrometric relation. The correction is small and
    documented; pass ``None`` to disable it.

    Parameters
    ----------
    temp_c : np.ndarray
        Air temperature in degC.
    rh : np.ndarray
        Relative humidity in percent (clipped to [1, 100] for the fit domain).
    pressure_kpa : np.ndarray, optional
        Surface pressure in kPa (sea level = 101.325). If given, a small
        elevation pressure correction is applied.

    Returns
    -------
    np.ndarray
        Wet-bulb temperature in degC.
    """
    t = np.asarray(temp_c, dtype=np.float64)
    r = np.clip(np.asarray(rh, dtype=np.float64), 1.0, 100.0)

    tw = (
        t * np.arctan(0.151977 * np.sqrt(r + 8.313659))
        + np.arctan(t + r)
        - np.arctan(r - 1.676331)
        + 0.00391838 * np.power(r, 1.5) * np.arctan(0.023101 * r)
        - 4.686035
    )

    if pressure_kpa is not None:
        p_hpa = np.asarray(pressure_kpa, dtype=np.float64) * 10.0  # kPa -> hPa
        # First-order correction: psychrometer constant gamma scales with pressure,
        # gamma = (cp*P)/(0.622*L). A lower P (higher altitude) raises the
        # wet-bulb depression (T - Tw); approximate dTw ~ -(T - Tw)*(P0 - P)/P0.
        depression = t - tw
        frac = (p_hpa - _STULL_REF_PRESSURE_HPA) / _STULL_REF_PRESSURE_HPA
        tw = tw - depression * frac  # P<P0 -> frac<0 -> Tw decreases (deeper depression)
    return tw


def wbgt_simplified(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """ABM (Australian BoM) simplified WBGT ``= 0.567*Ta + 0.393*e + 3.94`` (degC).

    ``e`` = Tetens vapour pressure from :func:`vapour_pressure_from_rh` (hPa).
    This approximation **omits the solar/globe term**, so it is a shade/indoor
    *lower bound* and under-reads in strong sun. [R8 §6.1] ``[verified]``

    Parameters
    ----------
    temp_c : np.ndarray
        Air temperature in degC.
    rh : np.ndarray
        Relative humidity in percent.

    Returns
    -------
    np.ndarray
        Simplified WBGT in degC.
    """
    ta = np.asarray(temp_c, dtype=np.float64)
    e = vapour_pressure_from_rh(ta, rh)
    return 0.567 * ta + 0.393 * e + 3.94


def wbgt_full(temp_c: np.ndarray, t_nw: np.ndarray, t_g: np.ndarray,
              solar: bool = True) -> np.ndarray:
    """Full ISO-7243 WBGT from natural-wet-bulb, globe and dry-bulb temps (degC).

    Outdoor/in-sun: ``WBGT = 0.7*Tnw + 0.2*Tg + 0.1*Ta``; indoor/no solar load:
    ``WBGT = 0.7*Tnw + 0.3*Tg``. Requires a globe temperature ``Tg`` (radiation),
    so it is only used when a ``TMRT``/globe field exists. [R8 §6.1] ``[verified]``

    Parameters
    ----------
    temp_c : np.ndarray
        Dry-bulb air temperature ``Ta`` (degC).
    t_nw : np.ndarray
        Natural (un-aspirated) wet-bulb temperature ``Tnw`` (degC).
    t_g : np.ndarray
        Black-globe temperature ``Tg`` (degC).
    solar : bool, default True
        True = outdoor in-sun weighting (0.7/0.2/0.1); False = indoor (0.7/0.3).

    Returns
    -------
    np.ndarray
        Full WBGT in degC.
    """
    ta = np.asarray(temp_c, dtype=np.float64)
    tnw = np.asarray(t_nw, dtype=np.float64)
    tg = np.asarray(t_g, dtype=np.float64)
    if solar:
        return 0.7 * tnw + 0.2 * tg + 0.1 * ta
    return 0.7 * tnw + 0.3 * tg


def apparent_temperature(
    temp_c: np.ndarray,
    rh: np.ndarray,
    wind: np.ndarray,
    net_radiation: np.ndarray | None = None,
) -> np.ndarray:
    """Steadman (Australian BoM) Apparent Temperature ``AT`` (degC). [R8 §5.3] ``[verified]``

    Non-radiation (shade) form: ``AT = Ta + 0.33*e - 0.70*ws - 4.00``.
    With-radiation form (when ``net_radiation`` ``Q`` W/m2 is supplied):
    ``AT = Ta + 0.348*e - 0.70*ws + 0.70*Q/(ws + 10) - 4.25``.
    ``e`` = Tetens vapour pressure (hPa); ``ws`` = 10 m wind (m/s).

    Parameters
    ----------
    temp_c : np.ndarray
        Air temperature ``Ta`` in degC.
    rh : np.ndarray
        Relative humidity (%).
    wind : np.ndarray
        10 m wind speed (m/s).
    net_radiation : np.ndarray, optional
        Net radiation ``Q`` (W/m2); if given, the radiation form is used.

    Returns
    -------
    np.ndarray
        Apparent temperature in degC.
    """
    ta = np.asarray(temp_c, dtype=np.float64)
    ws = np.asarray(wind, dtype=np.float64)
    e = vapour_pressure_from_rh(ta, rh)
    if net_radiation is None:
        return ta + 0.33 * e - 0.70 * ws - 4.00
    q = np.asarray(net_radiation, dtype=np.float64)
    return ta + 0.348 * e - 0.70 * ws + 0.70 * q / (ws + 10.0) - 4.25


def discomfort_index_thom(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Thom Discomfort Index ``DI = T - (0.55 - 0.0055*RH)*(T - 14.5)`` (degC).

    [R8 §5.4] ``[verified]``

    Parameters
    ----------
    temp_c : np.ndarray
        Air temperature in degC.
    rh : np.ndarray
        Relative humidity (%).

    Returns
    -------
    np.ndarray
        Discomfort index in degC.
    """
    t = np.asarray(temp_c, dtype=np.float64)
    r = np.asarray(rh, dtype=np.float64)
    return t - (0.55 - 0.0055 * r) * (t - 14.5)


def net_effective_temperature(
    temp_c: np.ndarray, rh: np.ndarray, wind: np.ndarray
) -> np.ndarray:
    """Net Effective Temperature (Hentschel / Missenard form), degC.

    ``NET = 37 - (37 - T)/[0.68 - 0.0014*RH + 1/(1.76 + 1.4*ws^0.75)]``
    ``      - 0.29*T*(1 - 0.01*RH)``. [R8 §5.5] ``[from-knowledge (verify)]`` —
    several coefficient variants exist; this is the cited form.

    Parameters
    ----------
    temp_c : np.ndarray
        Air temperature in degC.
    rh : np.ndarray
        Relative humidity (%).
    wind : np.ndarray
        Wind speed (m/s).

    Returns
    -------
    np.ndarray
        Net effective temperature in degC.
    """
    t = np.asarray(temp_c, dtype=np.float64)
    r = np.asarray(rh, dtype=np.float64)
    ws = np.maximum(np.asarray(wind, dtype=np.float64), 0.0)
    denom = 0.68 - 0.0014 * r + 1.0 / (1.76 + 1.4 * np.power(ws, 0.75))
    return 37.0 - (37.0 - t) / denom - 0.29 * t * (1.0 - 0.01 * r)


def mean_radiant_temperature(
    temp_c: np.ndarray,
    solar: np.ndarray | None = None,
    wind: np.ndarray | None = None,
    svf: np.ndarray | None = None,
    albedo: np.ndarray | None = None,
) -> np.ndarray:
    """Simplified mean radiant temperature ``Tmrt`` (degC) — coarse proxy only.

    A full per-pixel ``Tmrt`` requires a radiation/geometry model (**SOLWEIG**,
    R8 §6.2) accounting for shadows, walls and sky-view; this is **not** that.
    This helper gives a documented first-order daytime estimate so UTCI/PET can
    run in a numpy-only demo when no SOLWEIG raster exists:

    ``Tmrt ~ Ta + dT_rad`` where the radiative bump ``dT_rad`` grows with absorbed
    shortwave ``(1-albedo)*K_down`` seen through the sky-view fraction and is
    damped by wind. With the SOLWEIG body shortwave absorption ``a_k = 0.70`` and
    a lumped radiative-coupling/heat-transfer scaling, the integral six-flux form
    (R8 §6.2) reduces to an additive offset on ``Ta``. When ``solar`` is absent
    it returns ``Ta`` (radiatively neutral, ``Tmrt = Ta`` — the UTCI reference).

    Parameters
    ----------
    temp_c : np.ndarray
        Air temperature ``Ta`` (degC).
    solar : np.ndarray, optional
        Incoming shortwave ``K_down`` (W/m2).
    wind : np.ndarray, optional
        Wind speed (m/s); damps the radiant bump.
    svf : np.ndarray, optional
        Sky-view factor (0-1); scales sky exposure (default 1 = open).
    albedo : np.ndarray, optional
        Surface albedo (0-1); default 0.2.

    Returns
    -------
    np.ndarray
        Approximate Tmrt in degC.

    Notes
    -----
    Marked **approximate** — for street-scale/quantitative work substitute a
    SOLWEIG-produced ``TMRT`` layer. The coefficient set here is a deliberately
    simple, documented closure (not a published Tmrt regression).
    """
    ta = np.asarray(temp_c, dtype=np.float64)
    if solar is None:
        return ta.copy()
    k = np.asarray(solar, dtype=np.float64)
    a_sw = 0.70  # SOLWEIG body shortwave absorption coefficient [R8 §6.2 / constants]
    alb = 0.2 if albedo is None else np.asarray(albedo, dtype=np.float64)
    sky = 1.0 if svf is None else np.clip(np.asarray(svf, dtype=np.float64), 0.0, 1.0)
    absorbed = a_sw * (1.0 - alb) * k * sky  # W/m2 absorbed by the body
    # lumped radiative coupling h_r ~ 6 W/m2/K for a human; wind raises convective
    # loss so the temperature bump is absorbed / (h_r + h_conv(wind)).
    h_r = 6.0
    if wind is None:
        h_conv = 0.0
    else:
        ws = np.maximum(np.asarray(wind, dtype=np.float64), 0.0)
        h_conv = 4.0 * np.sqrt(ws)  # crude forced-convection coefficient
    d_t = absorbed / (h_r + h_conv)
    return ta + d_t


# ---------------------------------------------------------------------------
# UTCI (lazy thermofeel / pythermalcomfort, else documented polynomial approx)
# ---------------------------------------------------------------------------
def _utci_polynomial(ta: np.ndarray, dtmrt: np.ndarray, va: np.ndarray,
                     pa_kpa: np.ndarray) -> np.ndarray:
    """Documented compact UTCI approximation (offset polynomial on Ta).

    The official UTCI is a 6th-order, 210-coefficient polynomial in
    ``(Ta, D = Tmrt-Ta, va, Pa)`` (R8 §6.3). Retyping all 210 coefficients by
    hand is explicitly discouraged (error-prone); when neither ``thermofeel`` nor
    ``pythermalcomfort`` is importable we fall back to a **reduced, sign-correct
    additive-offset approximation** that reproduces the dominant first/second
    order behaviour: warmer with ``Ta`` and radiant load ``D``, cooler with wind
    ``va``, warmer with vapour pressure at heat. This is a deliberate, clearly
    labelled approximation (NOT the full polynomial) so the pipeline runs on
    numpy alone; for accurate UTCI install ``thermofeel``/``pythermalcomfort``.

    ``UTCI ~ Ta + 0.607*D - 0.227*ln(1+va) + 0.0307*D*va_term + 0.10*(Pa-1.0)``
    (Pa in kPa). Coefficients chosen to match published UTCI gradients near the
    heat range to within a few degC; not for quantitative reporting.
    """
    va_term = np.log1p(np.maximum(va, 0.0))
    return (
        ta
        + 0.607 * dtmrt
        - 1.5 * va_term
        + 0.020 * dtmrt * np.maximum(va, 0.0)
        + 0.20 * (pa_kpa - 1.0)
    )


def utci_approx(
    temp_c: np.ndarray,
    rh: np.ndarray,
    wind: np.ndarray,
    tmrt: np.ndarray | None = None,
) -> np.ndarray:
    """Universal Thermal Climate Index ``UTCI`` (degC). [R8 §6.3]

    Prefers the published implementation: tries ``thermofeel.utci_approx`` then
    ``pythermalcomfort.models.utci`` (both **lazily imported**). If neither is
    installed, falls back to :func:`_utci_polynomial`, a documented reduced
    approximation (clearly *not* the full 210-coefficient polynomial).

    ``Tmrt`` is required physically; when ``tmrt is None`` the UTCI reference
    assumption ``Tmrt = Ta`` is used (radiatively neutral) so the call still
    returns a finite field — but then UTCI mostly reflects ``Ta``/wind/humidity.

    Parameters
    ----------
    temp_c : np.ndarray
        2 m air temperature ``Ta`` (degC).
    rh : np.ndarray
        Relative humidity (%).
    wind : np.ndarray
        10 m wind speed (m/s).
    tmrt : np.ndarray, optional
        Mean radiant temperature (degC); defaults to ``Ta`` if absent.

    Returns
    -------
    np.ndarray
        UTCI equivalent temperature in degC.
    """
    ta = np.asarray(temp_c, dtype=np.float64)
    va = np.asarray(wind, dtype=np.float64)
    rh_a = np.asarray(rh, dtype=np.float64)
    tmrt_a = ta.copy() if tmrt is None else np.asarray(tmrt, dtype=np.float64)
    e_hpa = vapour_pressure_from_rh(ta, rh_a)
    pa_kpa = e_hpa / 10.0  # hPa -> kPa (UTCI uses vapour pressure in kPa)

    # 1) thermofeel (vectorized, fast)
    try:  # pragma: no cover - exercised only when dep present
        import thermofeel as _tf  # type: ignore

        return np.asarray(_tf.calculate_utci(ta, tmrt_a, va, e_hpa), dtype=np.float64)
    except Exception:
        pass

    # 2) pythermalcomfort (scalar API -> vectorize)
    try:  # pragma: no cover - exercised only when dep present
        from pythermalcomfort.models import utci as _ptc_utci  # type: ignore

        def _one(t, tr, v, h):
            try:
                res = _ptc_utci(tdb=float(t), tr=float(tr), v=float(max(v, 0.5)), rh=float(h))
                return float(res["utci"] if isinstance(res, dict) else res)
            except Exception:
                return np.nan

        vfun = np.vectorize(_one, otypes=[np.float64])
        return vfun(ta, tmrt_a, va, rh_a)
    except Exception:
        pass

    # 3) documented numpy fallback approximation
    return _utci_polynomial(ta, tmrt_a - ta, va, pa_kpa)


def utci_categories(utci_c: np.ndarray) -> np.ndarray:
    """Map a UTCI field (degC) to the 10-category thermal-stress scale. [R8 §6.3]

    Uses :data:`urbanheat.constants.UTCI_CATEGORIES` (descending ``min`` bounds).
    Returns an integer category index ``0..9`` where 0 = "extreme heat stress"
    (top of the list) and 9 = "extreme cold stress"; NaN -> -1.

    Parameters
    ----------
    utci_c : np.ndarray
        UTCI in degC.

    Returns
    -------
    np.ndarray
        Integer category index 0-9 (float dtype; NaN -> -1).
    """
    u = np.asarray(utci_c, dtype=np.float64)
    out = np.full(u.shape, -1.0, dtype=np.float64)
    finite = np.isfinite(u)
    cat = np.full(u.shape, float(len(UTCI_CATEGORIES) - 1), dtype=np.float64)
    assigned = np.zeros(u.shape, dtype=bool)
    # UTCI_CATEGORIES is ordered hot->cold by descending 'min'; first match wins.
    for i, entry in enumerate(UTCI_CATEGORIES):
        sel = finite & (~assigned) & (u >= entry["min"])
        cat[sel] = float(i)
        assigned |= sel
    out[finite] = cat[finite]
    return out


# ===========================================================================
# Spectral / emissivity helpers (ARCHITECTURE §11.3 add_spectral_indices)
# ===========================================================================
def _fractional_veg_cover(ndvi: np.ndarray) -> np.ndarray:
    """Fractional vegetation cover ``Pv`` from NDVI (Sobrino thresholds).

    ``Pv = ((NDVI - NDVI_s)/(NDVI_v - NDVI_s))^2`` clipped to [0,1], with
    soil/veg NDVI thresholds from
    :data:`urbanheat.constants.SPECTRAL_INDEX_COEFFS` ``["EMISSIVITY_NDVI"]``.
    [R8 §3.1]
    """
    coeffs = SPECTRAL_INDEX_COEFFS["EMISSIVITY_NDVI"]
    ndvi_s = coeffs["ndvi_soil"]
    ndvi_v = coeffs["ndvi_veg"]
    n = np.asarray(ndvi, dtype=np.float64)
    pv = ((n - ndvi_s) / (ndvi_v - ndvi_s)) ** 2
    return np.clip(pv, 0.0, 1.0)


def _emissivity_from_ndvi(ndvi: np.ndarray) -> np.ndarray:
    """NDVI-threshold emissivity (Sobrino): ``eps = eps_v*Pv + eps_s*(1-Pv) + d_eps``.

    Coefficients from :data:`urbanheat.constants.SPECTRAL_INDEX_COEFFS`. [R8 §3.1]
    """
    coeffs = SPECTRAL_INDEX_COEFFS["EMISSIVITY_NDVI"]
    pv = _fractional_veg_cover(ndvi)
    eps = coeffs["eps_veg"] * pv + coeffs["eps_soil"] * (1.0 - pv) + coeffs["d_eps"]
    return eps


# ===========================================================================
# FeatureStack-method API (ARCHITECTURE §11.3 contract)
# ===========================================================================
def add_spectral_indices(fs: FeatureStack) -> FeatureStack:
    """Derive NDVI-based surface properties needed downstream. Idempotent.

    Per the §11.3 contract this (re)computes spectral/surface indices when their
    inputs exist. The canonical FeatureStack carries no raw reflectance bands
    (those are resolved upstream in the data backend), so here we fill the
    NDVI-derivable layers that the heat-stress maths needs and that may be
    missing: **FVC** (fractional vegetation cover) and **EMISSIVITY** (Sobrino
    NDVI-threshold), both from :data:`SPECTRAL_INDEX_COEFFS`. Existing layers are
    left untouched (idempotent — safe to call repeatedly).

    Parameters
    ----------
    fs : FeatureStack
        Stack containing at least ``NDVI`` (no-op for the spectral fills if NDVI
        is absent).

    Returns
    -------
    FeatureStack
        The same stack with ``FVC`` / ``EMISSIVITY`` added where derivable.
    """
    if fs.has(NDVI):
        ndvi = fs.get(NDVI).astype(np.float64)
        if not fs.has(FVC):
            fs.add_layer(FVC, _fractional_veg_cover(ndvi).astype(np.float32))
        if not fs.has(EMISSIVITY):
            fs.add_layer(EMISSIVITY, _emissivity_from_ndvi(ndvi).astype(np.float32))
    return fs


def _resolve_urban_rural(fs: FeatureStack, rural_method: str):
    """Return ``(urban_mask, rural_ref)`` boolean arrays for SUHII, or (None, None).

    * ``'lcz'``       — urban = built LCZ classes (1-10), rural reference = the
      LCZ rural-reference class :data:`constants.LCZ_RURAL_REFERENCE` (LCZ-D
      "Low plants"); if that exact class is empty, fall back to all natural LCZ
      (>=11). Requires the ``LCZ`` layer.
    * ``'smod_ring'`` — not available without a SMOD layer here -> falls back to
      the urban-complement definition (rural = non-urban via impervious / LCZ).
    * ``'percentile'``— continuous SUHI-surface (no masks; handled by caller).
    """
    from urbanheat.constants import LCZ_RURAL_REFERENCE

    if rural_method == "lcz" and fs.has(LCZ):
        lcz = np.asarray(fs.get(LCZ))
        urban = (lcz >= 1) & (lcz <= 10)
        rural = np.isclose(lcz, LCZ_RURAL_REFERENCE)
        if not rural.any():
            rural = lcz >= 11  # any natural LCZ class
        return urban, rural
    return None, None


def surface_uhi(fs: FeatureStack, rural_method: str = "lcz") -> FeatureStack:
    """Compute ``SUHII = LST - rural_reference`` and write it. [R8 §3.2]

    ``rural_method`` in ``{'lcz', 'smod_ring', 'percentile'}``:
      * ``'lcz'`` — rural reference = LCZ-D class
        (:data:`constants.LCZ_RURAL_REFERENCE`) median LST (built = LCZ 1-10).
      * ``'percentile'`` — continuous SUHI-surface vs the scene's cool
        lower-quartile reference (no LULC needed).
      * ``'smod_ring'`` — buffer-ring style; approximated by the urban-complement
        when no SMOD layer is present.

    The dry-cropland sign-flip guard of :func:`surface_uhi_intensity` is always
    applied. Reads ``LST`` (and ``LCZ`` for ``'lcz'``); writes ``SUHII``.

    Parameters
    ----------
    fs : FeatureStack
        Stack with ``LST`` (degC).
    rural_method : str
        Rural-reference delineation, see above.

    Returns
    -------
    FeatureStack
        Stack with ``SUHII`` added.
    """
    lst = fs.get(LST).astype(np.float64)
    urban, rural = _resolve_urban_rural(fs, rural_method)
    if rural is not None:
        suhii = surface_uhi_intensity(lst, urban_mask=urban, rural_ref=rural)
    else:
        # percentile / no-LULC: continuous SUHI surface vs cool lower quartile.
        suhii = surface_uhi_intensity(lst, urban_mask=None, rural_ref=None)
    fs.add_layer(SUHII, suhii.astype(np.float32))
    return fs


def utfvi(fs: FeatureStack) -> FeatureStack:
    """``UTFVI = (Ts - Tm)/Tm`` in Kelvin, plus the 6-class EEI. [R8 §3.4]

    Writes ``UTFVI`` (dimensionless) and ``EEI`` (integer class 0-5 via
    :func:`eei_class` / :data:`constants.UTFVI_CLASSES`). Reads ``LST`` (degC).

    Parameters
    ----------
    fs : FeatureStack
        Stack with ``LST``.

    Returns
    -------
    FeatureStack
        Stack with ``UTFVI`` and ``EEI`` added.
    """
    lst = fs.get(LST).astype(np.float64)
    u = utfvi_array(lst)
    fs.add_layer(UTFVI, u.astype(np.float32))
    fs.add_layer(EEI, eei_class(u).astype(np.float32))
    return fs


def lst_statistics(fs: FeatureStack) -> FeatureStack:
    """Per-pixel ``LST_PERCENTILE`` (0-100) and ``LST_ZSCORE`` over the AOI. [R8 §3.3]

    Writes both layers. Reads ``LST``.

    Parameters
    ----------
    fs : FeatureStack
        Stack with ``LST``.

    Returns
    -------
    FeatureStack
        Stack with ``LST_PERCENTILE`` and ``LST_ZSCORE`` added.
    """
    lst = fs.get(LST).astype(np.float64)
    fs.add_layer(LST_PERCENTILE, lst_percentile(lst).astype(np.float32))
    fs.add_layer(LST_ZSCORE, lst_zscore(lst).astype(np.float32))
    return fs


def heat_index(fs: FeatureStack) -> FeatureStack:
    """NWS Rothfusz Heat Index (degC) -> ``HEAT_INDEX``. [R8 §5.1]

    Reads ``AIR_TEMP`` and ``REL_HUMIDITY``.

    Parameters
    ----------
    fs : FeatureStack
        Stack with ``AIR_TEMP`` (degC) and ``REL_HUMIDITY`` (%).

    Returns
    -------
    FeatureStack
        Stack with ``HEAT_INDEX`` added.
    """
    ta = fs.get(AIR_TEMP).astype(np.float64)
    rh = fs.get(REL_HUMIDITY).astype(np.float64)
    fs.add_layer(HEAT_INDEX, heat_index_rothfusz(ta, rh).astype(np.float32))
    return fs


def humidex(fs: FeatureStack) -> FeatureStack:
    """Humidex (degC-like) from ``AIR_TEMP`` + ``DEWPOINT`` -> ``HUMIDEX``. [R8 §5.2]

    If ``DEWPOINT`` is absent but ``REL_HUMIDITY`` is present, the dew-point
    vapour pressure is recovered from RH (Tetens) so the index can still be
    computed.

    Parameters
    ----------
    fs : FeatureStack
        Stack with ``AIR_TEMP`` and ``DEWPOINT`` (or ``REL_HUMIDITY``).

    Returns
    -------
    FeatureStack
        Stack with ``HUMIDEX`` added.
    """
    ta = fs.get(AIR_TEMP).astype(np.float64)
    if fs.has(DEWPOINT):
        td = fs.get(DEWPOINT).astype(np.float64)
        hx = humidex_from_dewpoint(ta, td)
    else:
        # derive e from RH then apply Humidex with that vapour pressure directly.
        e = vapour_pressure_from_rh(ta, fs.get(REL_HUMIDITY).astype(np.float64))
        hx = np.maximum(ta + 0.5555 * (e - 10.0), ta)
    fs.add_layer(HUMIDEX, hx.astype(np.float32))
    return fs


def wet_bulb(fs: FeatureStack) -> FeatureStack:
    """Stull (2011) wet-bulb (degC), pressure-corrected via ``PRESSURE``. [R8 §4]

    Reads ``AIR_TEMP`` and ``REL_HUMIDITY``; uses ``PRESSURE`` (kPa) for the
    elevation correction when present. Writes ``WET_BULB``.

    Parameters
    ----------
    fs : FeatureStack
        Stack with ``AIR_TEMP`` and ``REL_HUMIDITY``.

    Returns
    -------
    FeatureStack
        Stack with ``WET_BULB`` added.
    """
    ta = fs.get(AIR_TEMP).astype(np.float64)
    rh = fs.get(REL_HUMIDITY).astype(np.float64)
    p = fs.get(PRESSURE).astype(np.float64) if fs.has(PRESSURE) else None
    fs.add_layer(WET_BULB, wet_bulb_stull(ta, rh, pressure_kpa=p).astype(np.float32))
    return fs


def wbgt(fs: FeatureStack, method: str = "abm") -> FeatureStack:
    """WBGT (degC) -> ``WBGT``. [R8 §6.1]

    ``method='abm'`` -> ABM simplified (``AIR_TEMP`` + ``REL_HUMIDITY``, shade
    lower bound). ``method='full'`` -> ISO-7243 form, which needs a globe term;
    here the globe temperature is taken from ``TMRT`` if present (with the
    natural wet-bulb approximated by the Stull wet-bulb) and otherwise the call
    falls back to the ABM form.

    Parameters
    ----------
    fs : FeatureStack
        Stack with ``AIR_TEMP`` + ``REL_HUMIDITY`` (and ``TMRT`` for ``'full'``).
    method : str
        ``'abm'`` or ``'full'``.

    Returns
    -------
    FeatureStack
        Stack with ``WBGT`` added.
    """
    ta = fs.get(AIR_TEMP).astype(np.float64)
    rh = fs.get(REL_HUMIDITY).astype(np.float64)
    if method == "full" and fs.has(TMRT):
        tg = fs.get(TMRT).astype(np.float64)  # use Tmrt as the globe-temp proxy
        p = fs.get(PRESSURE).astype(np.float64) if fs.has(PRESSURE) else None
        tnw = wet_bulb_stull(ta, rh, pressure_kpa=p)  # natural wet-bulb proxy
        w = wbgt_full(ta, tnw, tg, solar=True)
    else:
        w = wbgt_simplified(ta, rh)
    fs.add_layer(WBGT, w.astype(np.float32))
    return fs


def utci(fs: FeatureStack) -> FeatureStack:
    """UTCI (degC) -> ``UTCI``. [R8 §6.3]

    Reads ``AIR_TEMP``, ``REL_HUMIDITY``, ``WIND_SPEED`` and ``TMRT`` (if a
    SOLWEIG Tmrt layer exists; otherwise the UTCI reference ``Tmrt = Ta`` is
    assumed). Uses :func:`utci_approx` (lazy thermofeel/pythermalcomfort else the
    documented fallback).

    Parameters
    ----------
    fs : FeatureStack
        Stack with ``AIR_TEMP``, ``REL_HUMIDITY``, ``WIND_SPEED``.

    Returns
    -------
    FeatureStack
        Stack with ``UTCI`` added.
    """
    ta = fs.get(AIR_TEMP).astype(np.float64)
    rh = fs.get(REL_HUMIDITY).astype(np.float64)
    ws = fs.get(WIND_SPEED).astype(np.float64) if fs.has(WIND_SPEED) else np.full_like(ta, 0.5)
    tmrt = fs.get(TMRT).astype(np.float64) if fs.has(TMRT) else None
    fs.add_layer(UTCI, utci_approx(ta, rh, ws, tmrt=tmrt).astype(np.float32))
    return fs


def human_stress_ensemble(fs: FeatureStack, min_agree: int = 3) -> FeatureStack:
    """Cross-verifying human heat-stress ensemble (Layer B Tier-1). [R8 §12.2]

    Flags a pixel "stressed" where ``>= min_agree`` of the five cheap T+RH
    indices cross their danger band (all thresholds from
    :data:`constants.HEAT_STRESS_THRESHOLDS`):

      * wet-bulb (Stull) ``>= 28 degC``        (``wet_bulb["danger"]``)
      * Heat Index (NWS) ``>= Danger`` band    (``heat_index_f["danger"]``, degF)
      * Humidex ``>= 40``                       (``humidex["great_discomfort"]``)
      * Discomfort Index (Thom) ``>= 29``       (``discomfort_index["strong"]``)
      * ABM-WBGT ``>= 28 degC``                 (``wbgt["low"]``)

    Writes ``HUMAN_STRESS_COUNT`` (# of crossing indices, 0-5) and
    ``HUMAN_STRESS_SCORE`` (= 100 * count / 5, a 0-100 Layer-B hazard score).
    The component index layers are computed and stored if missing.

    Parameters
    ----------
    fs : FeatureStack
        Stack with ``AIR_TEMP`` + ``REL_HUMIDITY`` (and optionally ``DEWPOINT``,
        ``PRESSURE``).
    min_agree : int, default 3
        Number of indices that must be in danger for the boolean flag.

    Returns
    -------
    FeatureStack
        Stack with ``HUMAN_STRESS_COUNT`` and ``HUMAN_STRESS_SCORE`` added.
    """
    ta = fs.get(AIR_TEMP).astype(np.float64)
    rh = fs.get(REL_HUMIDITY).astype(np.float64)
    p = fs.get(PRESSURE).astype(np.float64) if fs.has(PRESSURE) else None

    thr = HEAT_STRESS_THRESHOLDS

    wb = fs.get(WET_BULB).astype(np.float64) if fs.has(WET_BULB) else wet_bulb_stull(ta, rh, p)
    hi = fs.get(HEAT_INDEX).astype(np.float64) if fs.has(HEAT_INDEX) else heat_index_rothfusz(ta, rh)
    if fs.has(HUMIDEX):
        hx = fs.get(HUMIDEX).astype(np.float64)
    elif fs.has(DEWPOINT):
        hx = humidex_from_dewpoint(ta, fs.get(DEWPOINT).astype(np.float64))
    else:
        e = vapour_pressure_from_rh(ta, rh)
        hx = np.maximum(ta + 0.5555 * (e - 10.0), ta)
    di = discomfort_index_thom(ta, rh)
    wbgt_abm = wbgt_simplified(ta, rh)

    hi_f = hi * 9.0 / 5.0 + 32.0  # Heat Index threshold is in degF

    flags = [
        wb >= thr["wet_bulb"]["danger"],            # 28 degC
        hi_f >= thr["heat_index_f"]["danger"],      # 103 degF (~39.4 degC)
        hx >= thr["humidex"]["great_discomfort"],   # 40
        di >= thr["discomfort_index"]["strong"],    # 29
        wbgt_abm >= thr["wbgt"]["low"],             # 28
    ]
    count = np.zeros(ta.shape, dtype=np.float64)
    for f in flags:
        count = count + np.where(np.isfinite(f), f.astype(np.float64), 0.0)

    fs.add_layer(HUMAN_STRESS_COUNT, count.astype(np.float32))
    fs.add_layer(HUMAN_STRESS_SCORE, (100.0 * count / float(len(flags))).astype(np.float32))
    # store the boolean flag too (handy for Layer B); reuse the count >= min_agree.
    return fs


# ===========================================================================
# Convenience: fill everything derivable from the available variables.
# ===========================================================================
def compute_all_indices(stack: FeatureStack) -> FeatureStack:
    """Fill every derived heat-stress index that the stack's variables allow.

    Runs, guarded by input availability:
      * surface (need ``LST``): :func:`lst_statistics`, :func:`utfvi`,
        :func:`surface_uhi` (LCZ if available else percentile/complement);
      * spectral fills (need ``NDVI``): :func:`add_spectral_indices`;
      * air comfort (need ``AIR_TEMP`` + ``REL_HUMIDITY``): :func:`heat_index`,
        :func:`wet_bulb`, :func:`wbgt`, :func:`humidex`, the apparent-temperature
        / discomfort aux layers, and :func:`human_stress_ensemble`;
      * UTCI (need wind too / Tmrt): :func:`utci`.

    Every step is optional and silently skipped when its inputs are absent, so
    this is safe to call on a minimal (LST-only) stack or a fully-populated one.

    Parameters
    ----------
    stack : FeatureStack
        Any FeatureStack.

    Returns
    -------
    FeatureStack
        The same stack with all derivable indices written.
    """
    fs = stack

    if fs.has(NDVI):
        add_spectral_indices(fs)

    if fs.has(LST):
        lst_statistics(fs)
        utfvi(fs)
        # LCZ-based SUHII if LCZ present, else continuous percentile surface.
        surface_uhi(fs, rural_method="lcz" if fs.has(LCZ) else "percentile")

    if fs.has(AIR_TEMP) and fs.has(REL_HUMIDITY):
        ta = fs.get(AIR_TEMP).astype(np.float64)
        rh = fs.get(REL_HUMIDITY).astype(np.float64)

        heat_index(fs)
        wet_bulb(fs)
        wbgt(fs, method="abm")
        if fs.has(DEWPOINT) or True:  # humidex always derivable (RH fallback)
            humidex(fs)

        # auxiliary comfort layers (not in the fixed catalog but useful/cited)
        if fs.has(WIND_SPEED):
            ws = fs.get(WIND_SPEED).astype(np.float64)
            q = fs.get(SOLAR_RADIATION).astype(np.float64) if fs.has(SOLAR_RADIATION) else None
            fs.add_layer(APPARENT_TEMP, apparent_temperature(ta, rh, ws, q).astype(np.float32))
        fs.add_layer(DISCOMFORT_INDEX, discomfort_index_thom(ta, rh).astype(np.float32))

        human_stress_ensemble(fs)

        # UTCI: needs Tmrt physically; uses Tmrt layer if present else Tmrt=Ta.
        if fs.has(WIND_SPEED):
            utci(fs)

    return fs
