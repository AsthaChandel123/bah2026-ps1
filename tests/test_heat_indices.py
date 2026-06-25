"""Tests for :mod:`urbanheat.indices.heat_indices` — surface + comfort indices.

Guards (ARCHITECTURE §7, §11.3): SUHII sign, UTFVI definition, LST statistics,
and the human heat-stress indices (Heat Index, Humidex, wet-bulb/Stull, WBGT).
Where a closed form is known we assert known-value / range / monotonicity
properties rather than re-implementing the formula bit-for-bit.

The module is built in parallel; tests skip cleanly until it is importable.
"""

from __future__ import annotations

import numpy as np
import pytest

from urbanheat import datamodel as dm
from urbanheat.datamodel import FeatureStack

hi = pytest.importorskip("urbanheat.indices.heat_indices")


# ---------------------------------------------------------------------------
# Helpers to build minimal stacks with prescribed met fields
# ---------------------------------------------------------------------------
def _met_stack(air_temp_c: float, rel_humidity: float,
               dewpoint_c: float | None = None,
               pressure_kpa: float = 100.0,
               shape=(4, 4)) -> FeatureStack:
    """A constant-field FeatureStack carrying the met inputs comfort indices need."""
    ones = np.ones(shape, dtype=np.float32)
    if dewpoint_c is None:
        # Magnus inverse: Td from T and RH (approx, deg C).
        a, b = 17.625, 243.04
        rh = max(min(rel_humidity, 100.0), 1.0) / 100.0
        gamma = np.log(rh) + a * air_temp_c / (b + air_temp_c)
        dewpoint_c = float(b * gamma / (a - gamma))
    layers = {
        dm.AIR_TEMP: ones * air_temp_c,
        dm.REL_HUMIDITY: ones * rel_humidity,
        dm.DEWPOINT: ones * dewpoint_c,
        dm.PRESSURE: ones * pressure_kpa,
    }
    transform = (100.0, 0.0, 0.0, 0.0, -100.0, 0.0)
    bounds = (0.0, -shape[0] * 100.0, shape[1] * 100.0, 0.0)
    return FeatureStack.from_arrays(layers, transform=transform,
                                    crs="EPSG:32643", bounds=bounds)


def _const(fs: FeatureStack, name: str) -> float:
    """Return the (constant) value of a written layer."""
    return float(np.nanmean(fs.get(name)))


# ---------------------------------------------------------------------------
# Heat Index (NWS Rothfusz)
# ---------------------------------------------------------------------------
def test_heat_index_equals_temp_in_mild_conditions() -> None:
    """At low temperature the Heat Index ~ the air temperature (no amplification)."""
    fs = hi.heat_index(_met_stack(20.0, 40.0))
    assert fs.has(dm.HEAT_INDEX)
    assert _const(fs, dm.HEAT_INDEX) == pytest.approx(20.0, abs=2.0)


def test_heat_index_amplifies_in_hot_humid() -> None:
    """In hot, humid air the Heat Index exceeds the dry-bulb temperature."""
    t = 35.0
    fs = hi.heat_index(_met_stack(t, 70.0))
    feels = _const(fs, dm.HEAT_INDEX)
    assert feels > t + 2.0, "HI should feel hotter than 35C at 70% RH"
    # NWS: ~35C/70%RH lands around the low-50s C; keep a generous envelope.
    assert 44.0 <= feels <= 62.0


def test_heat_index_monotone_in_humidity() -> None:
    """At fixed hot temperature, Heat Index increases with relative humidity."""
    t = 33.0
    lo = _const(hi.heat_index(_met_stack(t, 40.0)), dm.HEAT_INDEX)
    himid = _const(hi.heat_index(_met_stack(t, 80.0)), dm.HEAT_INDEX)
    assert himid > lo


# ---------------------------------------------------------------------------
# Humidex
# ---------------------------------------------------------------------------
def test_humidex_known_value() -> None:
    """Humidex at 30C with dewpoint 25C is ~42 (Environment Canada worked example).

    Humidex = T + 0.5555*(6.11*exp(5417.7530*(1/273.16 - 1/(273.15+Td))) - 10).
    """
    fs = hi.humidex(_met_stack(30.0, rel_humidity=74.0, dewpoint_c=25.0))
    assert fs.has(dm.HUMIDEX)
    assert _const(fs, dm.HUMIDEX) == pytest.approx(42.0, abs=2.0)


def test_humidex_at_least_air_temp() -> None:
    """Humidex is never below the dry-bulb temperature."""
    fs = hi.humidex(_met_stack(28.0, rel_humidity=50.0, dewpoint_c=17.0))
    assert _const(fs, dm.HUMIDEX) >= 28.0 - 0.5


# ---------------------------------------------------------------------------
# Wet-bulb (Stull 2011)
# ---------------------------------------------------------------------------
def test_wet_bulb_at_saturation_equals_air_temp() -> None:
    """At 100% RH the wet-bulb temperature equals the dry-bulb temperature."""
    fs = hi.wet_bulb(_met_stack(25.0, 100.0))
    assert fs.has(dm.WET_BULB)
    assert _const(fs, dm.WET_BULB) == pytest.approx(25.0, abs=1.0)


def test_wet_bulb_below_air_temp_when_dry() -> None:
    """Below saturation the wet-bulb is strictly cooler than the dry-bulb."""
    t = 30.0
    fs = hi.wet_bulb(_met_stack(t, 40.0))
    assert _const(fs, dm.WET_BULB) < t


def test_wet_bulb_stull_known_value() -> None:
    """Stull's own worked example: T=20C, RH=50% -> Tw ~ 13.7C."""
    fs = hi.wet_bulb(_met_stack(20.0, 50.0))
    assert _const(fs, dm.WET_BULB) == pytest.approx(13.7, abs=1.0)


def test_wet_bulb_monotone_increasing_in_humidity() -> None:
    """Wet-bulb rises monotonically toward the dry-bulb as RH increases."""
    t = 32.0
    vals = [_const(hi.wet_bulb(_met_stack(t, rh)), dm.WET_BULB)
            for rh in (20.0, 40.0, 60.0, 80.0, 100.0)]
    assert all(b >= a - 1e-6 for a, b in zip(vals, vals[1:])), \
        f"wet-bulb not monotone in RH: {vals}"
    assert vals[-1] == pytest.approx(t, abs=1.0)


# ---------------------------------------------------------------------------
# WBGT (ABM-simplified)
# ---------------------------------------------------------------------------
def test_wbgt_runs_and_is_in_physical_range() -> None:
    """Simplified outdoor WBGT writes a layer in a physically plausible range.

    The ABM-simplified outdoor WBGT (Australian BoM, ~0.567*Ta + 0.393*e + 3.94
    with e in hPa) can sit *above* the dry-bulb temperature in humid air because
    of the strong vapour-pressure term — so we only bound it to a sane window
    rather than below Ta.
    """
    t = 35.0
    fs = hi.wbgt(_met_stack(t, 60.0), method="abm")
    assert fs.has(dm.WBGT)
    wbgt = _const(fs, dm.WBGT)
    assert 18.0 <= wbgt <= t + 6.0


def test_wbgt_increases_with_humidity() -> None:
    """WBGT rises with humidity (vapour-pressure term)."""
    t = 34.0
    lo = _const(hi.wbgt(_met_stack(t, 30.0)), dm.WBGT)
    high = _const(hi.wbgt(_met_stack(t, 80.0)), dm.WBGT)
    assert high > lo


# ---------------------------------------------------------------------------
# Surface metrics: SUHII / UTFVI / LST statistics
# ---------------------------------------------------------------------------
def test_surface_uhi_sign(synthetic_stack: FeatureStack) -> None:
    """SUHII = LST - rural_ref: hot urban core is positive, on average ~0-centred."""
    fs = hi.surface_uhi(synthetic_stack)
    assert fs.has(dm.SUHII)
    suhii = fs.get(dm.SUHII)
    lst = fs.get(dm.LST)
    # The hottest pixels must have positive SUHII (hotter than the rural ref).
    hot = lst >= np.nanpercentile(lst, 90)
    assert np.nanmean(suhii[hot]) > 0.0


def test_utfvi_definition_and_class(synthetic_stack: FeatureStack) -> None:
    """UTFVI = (Ts - Tm)/Tm in Kelvin; sign tracks (LST - mean LST)."""
    fs = hi.utfvi(synthetic_stack)
    assert fs.has(dm.UTFVI)
    utfvi = fs.get(dm.UTFVI)
    lst = fs.get(dm.LST)
    # Pixels above the mean LST get positive UTFVI; below, negative.
    above = lst > np.nanmean(lst)
    assert np.nanmean(utfvi[above]) > 0.0
    assert np.nanmean(utfvi[~above]) < 0.0


def test_lst_statistics_percentile_and_zscore(synthetic_stack: FeatureStack) -> None:
    """LST_PERCENTILE in [0,100]; LST_ZSCORE ~ standardized LST."""
    fs = hi.lst_statistics(synthetic_stack)
    assert fs.has(dm.LST_PERCENTILE) and fs.has(dm.LST_ZSCORE)
    pct = fs.get(dm.LST_PERCENTILE)
    z = fs.get(dm.LST_ZSCORE)
    assert np.nanmin(pct) >= 0.0 and np.nanmax(pct) <= 100.0
    assert abs(float(np.nanmean(z))) < 0.5      # roughly zero-mean
    assert float(np.nanstd(z)) == pytest.approx(1.0, abs=0.4)
    # Percentile and z-score rank LST the same way.
    assert np.corrcoef(pct.ravel(), fs.get(dm.LST).ravel())[0, 1] > 0.9
