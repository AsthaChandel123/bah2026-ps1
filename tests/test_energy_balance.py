"""Tests for :mod:`urbanheat.physics.energy_balance` — SEB terms.

Guards (ARCHITECTURE §8, §11.4):
  * ``longwave_up`` numeric check against L_up = eps*sigma*Ts^4 (constants.SIGMA_SB).
  * ``net_radiation`` writes NET_RADIATION with the right qualitative response
    (rises with K_down, falls with albedo).
  * ``sensible_heat`` / ``bowen_ratio`` / ``storage_heat_ohm`` array helpers behave.
  * the driver->LST sign table is exposed for the ML monotonicity layer
    (``expected_lst_gradient_signs`` if present, else ``physics_lst`` response).

The module is built in parallel; tests skip cleanly until it is importable.
"""

from __future__ import annotations

import numpy as np
import pytest

from urbanheat import datamodel as dm
from urbanheat.constants import SIGMA_SB
from urbanheat.datamodel import FeatureStack

eb = pytest.importorskip("urbanheat.physics.energy_balance")

KELVIN = 273.15


# ---------------------------------------------------------------------------
# longwave_up  (exact numeric check)
# ---------------------------------------------------------------------------
def test_longwave_up_blackbody_numeric() -> None:
    """L_up = eps*sigma*Ts^4 for a unit-emissivity surface, evaluated in Kelvin."""
    ts_c = np.array([[25.0, 30.0], [0.0, 45.0]], dtype=np.float32)
    eps = np.ones_like(ts_c)
    lup = eb.longwave_up(ts_c, eps)
    expected = SIGMA_SB * (ts_c.astype(np.float64) + KELVIN) ** 4
    np.testing.assert_allclose(np.asarray(lup, dtype=np.float64), expected,
                               rtol=1e-3)


def test_longwave_up_scales_with_emissivity() -> None:
    """Halving emissivity (ignoring reflected term) roughly halves emitted L_up."""
    ts_c = np.full((3, 3), 30.0, dtype=np.float32)
    full = np.asarray(eb.longwave_up(ts_c, np.ones_like(ts_c)), dtype=np.float64)
    half = np.asarray(eb.longwave_up(ts_c, 0.5 * np.ones_like(ts_c),
                                     longwave_down=np.zeros_like(ts_c)),
                      dtype=np.float64)
    np.testing.assert_allclose(half, 0.5 * full, rtol=1e-3)


def test_longwave_up_monotone_in_temperature() -> None:
    """Emitted longwave increases monotonically with surface temperature."""
    eps = np.ones((1, 3), dtype=np.float32)
    lup = np.asarray(eb.longwave_up(np.array([[10.0, 30.0, 50.0]], np.float32), eps),
                     dtype=np.float64).ravel()
    assert lup[0] < lup[1] < lup[2]


# ---------------------------------------------------------------------------
# net_radiation
# ---------------------------------------------------------------------------
def _rad_stack(albedo=0.2, k_down=800.0, l_down=380.0, lst=35.0, eps=0.96,
               shape=(4, 4)) -> FeatureStack:
    """A minimal stack carrying the radiation inputs for Q*."""
    o = np.ones(shape, dtype=np.float32)
    layers = {
        dm.ALBEDO: o * albedo,
        dm.SOLAR_RADIATION: o * k_down,
        dm.LONGWAVE_DOWN: o * l_down,
        dm.LST: o * lst,
        dm.EMISSIVITY: o * eps,
    }
    transform = (100.0, 0.0, 0.0, 0.0, -100.0, 0.0)
    bounds = (0.0, -shape[0] * 100.0, shape[1] * 100.0, 0.0)
    return FeatureStack.from_arrays(layers, transform=transform,
                                    crs="EPSG:32643", bounds=bounds)


def test_net_radiation_written_and_positive_daytime() -> None:
    """Q* is written and positive under strong daytime insolation."""
    fs = eb.net_radiation(_rad_stack())
    assert fs.has(dm.NET_RADIATION)
    q = fs.get(dm.NET_RADIATION)
    assert q.shape == fs.shape
    assert float(np.nanmean(q)) > 0.0


def test_net_radiation_decreases_with_albedo() -> None:
    """Higher albedo reflects more K_down, lowering Q* (the cool-roof mechanism)."""
    low = float(np.nanmean(eb.net_radiation(_rad_stack(albedo=0.1)).get(dm.NET_RADIATION)))
    high = float(np.nanmean(eb.net_radiation(_rad_stack(albedo=0.5)).get(dm.NET_RADIATION)))
    assert high < low


def test_net_radiation_increases_with_solar() -> None:
    """More incoming shortwave raises net radiation."""
    dim = float(np.nanmean(eb.net_radiation(_rad_stack(k_down=400.0)).get(dm.NET_RADIATION)))
    bright = float(np.nanmean(eb.net_radiation(_rad_stack(k_down=1000.0)).get(dm.NET_RADIATION)))
    assert bright > dim


# ---------------------------------------------------------------------------
# sensible heat / bowen ratio / storage
# ---------------------------------------------------------------------------
def test_sensible_heat_sign_follows_surface_air_gradient() -> None:
    """Q_H > 0 when the surface is warmer than the air, < 0 when cooler."""
    if not hasattr(eb, "sensible_heat"):
        pytest.skip("sensible_heat not implemented")
    shape = (3, 3)
    wind = np.full(shape, 3.0, dtype=np.float32)
    z0 = np.full(shape, 0.5, dtype=np.float32)
    warm = eb.sensible_heat(np.full(shape, 35.0, np.float32),
                            np.full(shape, 30.0, np.float32), wind, z0)
    cool = eb.sensible_heat(np.full(shape, 25.0, np.float32),
                            np.full(shape, 30.0, np.float32), wind, z0)
    assert float(np.nanmean(warm)) > 0.0
    assert float(np.nanmean(cool)) < 0.0


def test_bowen_ratio_is_quotient() -> None:
    """beta = Q_H / Q_E element-wise."""
    if not hasattr(eb, "bowen_ratio"):
        pytest.skip("bowen_ratio not implemented")
    qh = np.array([[100.0, 200.0]], dtype=np.float32)
    qe = np.array([[50.0, 100.0]], dtype=np.float32)
    beta = np.asarray(eb.bowen_ratio(qh, qe), dtype=np.float64)
    np.testing.assert_allclose(beta, [[2.0, 2.0]], rtol=1e-5)


def test_storage_heat_ohm_runs() -> None:
    """OHM storage returns a finite W/m2 grid from Q* and cover fractions."""
    if not hasattr(eb, "storage_heat_ohm"):
        pytest.skip("storage_heat_ohm not implemented")
    shape = (4, 4)
    qstar = np.full(shape, 500.0, dtype=np.float32)
    fractions = {
        dm.IMPERVIOUS_FRAC: np.full(shape, 0.6, np.float32),
        dm.GREEN_FRAC: np.full(shape, 0.3, np.float32),
        dm.WATER_FRAC: np.full(shape, 0.1, np.float32),
    }
    dqs = np.asarray(eb.storage_heat_ohm(qstar, fractions), dtype=np.float64)
    assert dqs.shape == shape
    assert np.isfinite(dqs).all()


# ---------------------------------------------------------------------------
# physics LST backbone + sign table
# ---------------------------------------------------------------------------
def test_physics_lst_backbone_runs(synthetic_stack: FeatureStack) -> None:
    """physics_lst returns a 2-D degC field on the grid."""
    if not hasattr(eb, "physics_lst"):
        pytest.skip("physics_lst not implemented")
    out = np.asarray(eb.physics_lst(synthetic_stack), dtype=np.float64)
    assert out.shape == synthetic_stack.shape
    # Plausible surface-temperature range for a pre-monsoon Indian city (degC).
    assert np.nanmin(out) > -10.0 and np.nanmax(out) < 80.0


def test_seb_residual_runs(synthetic_stack: FeatureStack) -> None:
    """The SEB-closure residual map is finite and non-negative (|.|)."""
    if not hasattr(eb, "seb_residual"):
        pytest.skip("seb_residual not implemented")
    res = np.asarray(eb.seb_residual(synthetic_stack), dtype=np.float64)
    assert res.shape == synthetic_stack.shape
    assert np.nanmin(res) >= -1e-6


def test_expected_lst_gradient_signs_if_present() -> None:
    """If a driver->dLST sign table helper exists, it must match the SEB table.

    The exact name/signature is not pinned in §11.4 (the task references
    ``expected_lst_gradient_signs``); this guards it loosely so the integrator
    can reconcile. Cooling drivers => -1, warming drivers => +1.
    """
    fn = getattr(eb, "expected_lst_gradient_signs", None)
    if fn is None:
        pytest.skip("expected_lst_gradient_signs not implemented")
    signs = fn()
    assert isinstance(signs, dict)
    # Cooling levers.
    for cooler in (dm.ALBEDO, dm.NDVI, dm.WATER_FRAC, dm.GREEN_FRAC):
        if cooler in signs:
            assert signs[cooler] < 0, f"{cooler} should cool (sign -1)"
    # Warming levers.
    for warmer in (dm.IMPERVIOUS_FRAC, dm.ANTHRO_HEAT):
        if warmer in signs:
            assert signs[warmer] > 0, f"{warmer} should warm (sign +1)"
