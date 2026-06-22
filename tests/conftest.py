"""Shared pytest fixtures for the ``urbanheat`` test suite.

Everything here is import-clean on the **minimal stack** (numpy / scipy /
scikit-learn / matplotlib only). Heavy/optional dependencies (``ee``, ``torch``,
``xgboost``, ``lightgbm``, ``shap``, ``pulp``, ``pymoo``, ``streamlit``,
``geopandas`` ...) are never imported at collection time; tests that need them
guard with :func:`pytest.importorskip`.

The two workhorse fixtures are:

* :func:`tiny_config` — a 24x24, seed-fixed :class:`~urbanheat.config.Config`
  small enough that the whole pipeline runs in well under a second.
* :func:`synthetic_stack` — a small, physically-plausible
  :class:`~urbanheat.datamodel.FeatureStack` produced either by the real
  :class:`~urbanheat.synthetic.source.SyntheticDataSource` (when that sibling
  module is present) or, as a fallback so the deterministic-core tests can run
  before that module lands, by a self-contained generator in this file that
  obeys the same SEB sign conventions (cooler over vegetation/water/albedo,
  hotter over impervious/low-SVF).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from urbanheat import datamodel as dm
from urbanheat.config import Config
from urbanheat.datamodel import FeatureStack

if TYPE_CHECKING:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Grid / config fixtures
# ---------------------------------------------------------------------------
TINY_SHAPE: tuple[int, int] = (24, 24)
TINY_SEED: int = 1234


@pytest.fixture()
def tiny_shape() -> tuple[int, int]:
    """A small ``(rows, cols)`` grid shape used across the suite."""
    return TINY_SHAPE


@pytest.fixture()
def tiny_config(tiny_shape: tuple[int, int]) -> Config:
    """A minimal, fully-deterministic synthetic-mode :class:`Config`.

    Uses an explicit ``grid_shape`` so backends do not have to derive a huge
    grid from the city bbox + 100 m resolution (which would be ~500x500).
    """
    return Config(
        city="Delhi",
        mode="synthetic",
        resolution_m=100.0,
        grid_shape=tiny_shape,
        seed=TINY_SEED,
        output_dir="outputs",
        optimizer_budget=1.0e6,
        optimizer_max_area_frac=0.30,
    )


# ---------------------------------------------------------------------------
# Synthetic FeatureStack
# ---------------------------------------------------------------------------
def _fallback_fields(shape: tuple[int, int], seed: int) -> dict[str, np.ndarray]:
    """Self-contained, dependency-free driver fields obeying the SEB sign table.

    This mirrors what ``urbanheat.synthetic.source.make_synthetic_fields`` is
    contracted to produce (canonical-name -> 2-D float32 arrays, spatially
    coherent, with an urban-core gradient) so the deterministic-core tests have
    a valid :class:`FeatureStack` to exercise even before the synthetic backend
    is implemented. The real backend supersedes this whenever it is importable.
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape

    def smooth(a: np.ndarray, passes: int = 2) -> np.ndarray:
        """Cheap separable box-blur for spatial coherence (no scipy needed)."""
        out = a.astype(np.float64)
        for _ in range(passes):
            out = (
                out
                + np.roll(out, 1, 0) + np.roll(out, -1, 0)
                + np.roll(out, 1, 1) + np.roll(out, -1, 1)
            ) / 5.0
        return out

    def norm01(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        lo, hi = float(a.min()), float(a.max())
        if hi - lo < 1e-12:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    # Radial urban-core gradient: 1 at centre -> 0 at edge (the "city").
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    cy, cx = (rows - 1) / 2.0, (cols - 1) / 2.0
    r = np.sqrt(((yy - cy) / max(cy, 1)) ** 2 + ((xx - cx) / max(cx, 1)) ** 2)
    core = norm01(1.0 - np.clip(r, 0, 1))  # 1 in centre

    # Impervious fraction: high in the core + coherent noise.
    impervious = norm01(0.7 * core + 0.3 * smooth(rng.standard_normal(shape)))
    green = norm01(1.0 - impervious + 0.15 * smooth(rng.standard_normal(shape)))
    tree = norm01(0.6 * green + 0.2 * smooth(rng.standard_normal(shape)))
    water = np.clip(norm01(smooth(rng.standard_normal(shape))) - 0.6, 0, None)
    water = norm01(water) * 0.4

    ndvi = (green * 0.8 - impervious * 0.2)  # -ish 0..0.8, anti-correlated w/ impervious
    ndvi = np.clip(ndvi, -0.1, 0.9)
    evi = np.clip(0.8 * ndvi, -0.1, 0.8)
    albedo = np.clip(0.12 + 0.10 * green + 0.05 * smooth(rng.standard_normal(shape)) * 0.1, 0.05, 0.5)
    emissivity = np.clip(0.95 + 0.02 * ndvi, 0.9, 0.99)

    svf = np.clip(1.0 - 0.6 * impervious - 0.1 * norm01(smooth(rng.standard_normal(shape))), 0.2, 1.0)
    bheight = np.clip(30.0 * impervious + 3.0 * norm01(smooth(rng.standard_normal(shape))), 0.0, 60.0)
    bvolume = bheight * impervious * 100.0
    elevation = 200.0 + 20.0 * norm01(smooth(rng.standard_normal(shape)))

    air_temp = 30.0 + 6.0 * core + 1.0 * smooth(rng.standard_normal(shape))
    dewpoint = air_temp - (8.0 + 4.0 * (1.0 - green))
    rel_humidity = np.clip(70.0 - 30.0 * core + 5.0 * smooth(rng.standard_normal(shape)), 5, 100)
    wind = np.clip(3.0 - 1.0 * impervious + 0.3 * smooth(rng.standard_normal(shape)), 0.2, 8.0)
    pressure = np.full(shape, 100.0)  # kPa
    solar = 850.0 + 50.0 * smooth(rng.standard_normal(shape))
    longwave_down = np.full(shape, 380.0) + 10.0 * core
    soil_moisture = np.clip(0.30 * green + 0.05 * smooth(rng.standard_normal(shape)) * 0.1, 0.02, 0.45)
    anthro = np.clip(60.0 * impervious + 5.0 * norm01(smooth(rng.standard_normal(shape))), 0.0, 120.0)
    population = np.clip(5000.0 * core + 200.0 * norm01(smooth(rng.standard_normal(shape))), 0, None)
    nightlights = np.clip(40.0 * impervious + 2.0 * norm01(smooth(rng.standard_normal(shape))), 0, None)
    lulc = np.where(water > 0.2, 17.0, np.where(impervious > 0.5, 1.0, np.where(green > 0.5, 14.0, 3.0)))
    lcz = lulc.copy()

    # Synthetic LST via the SEB sign table: hotter w/ impervious & anthro & low
    # SVF; cooler w/ NDVI, water, albedo, green. This is the ground-truth physics
    # the attribution / counterfactual tests rely on.
    lst = (
        air_temp
        + 12.0 * impervious
        - 8.0 * ndvi
        - 6.0 * water
        - 10.0 * (albedo - 0.2)
        + 4.0 * (1.0 - svf)
        + 0.05 * anthro
        - 5.0 * soil_moisture
    )
    lst = lst + 0.2 * smooth(rng.standard_normal(shape))  # tiny noise
    lst_day = lst + 2.0
    lst_night = lst - 6.0 + 3.0 * (1.0 - svf)
    net_rad = (1.0 - albedo) * solar + (longwave_down - emissivity * 5.670374419e-8 * (lst + 273.15) ** 4)

    fields: dict[str, np.ndarray] = {
        dm.LST: lst,
        dm.LST_DAY: lst_day,
        dm.LST_NIGHT: lst_night,
        dm.NDVI: ndvi,
        dm.EVI: evi,
        dm.ALBEDO: albedo,
        dm.EMISSIVITY: emissivity,
        dm.IMPERVIOUS_FRAC: impervious,
        dm.GREEN_FRAC: green,
        dm.WATER_FRAC: water,
        dm.TREE_FRAC: tree,
        dm.LULC: lulc,
        dm.LCZ: lcz,
        dm.BUILDING_HEIGHT: bheight,
        dm.BUILDING_VOLUME: bvolume,
        dm.SVF: svf,
        dm.ELEVATION: elevation,
        dm.AIR_TEMP: air_temp,
        dm.DEWPOINT: dewpoint,
        dm.REL_HUMIDITY: rel_humidity,
        dm.WIND_SPEED: wind,
        dm.PRESSURE: pressure,
        dm.SOLAR_RADIATION: solar,
        dm.LONGWAVE_DOWN: longwave_down,
        dm.NET_RADIATION: net_rad,
        dm.SOIL_MOISTURE: soil_moisture,
        dm.ANTHRO_HEAT: anthro,
        dm.POPULATION: population,
        dm.NIGHTLIGHTS: nightlights,
    }
    return {k: np.asarray(v, dtype=np.float32) for k, v in fields.items()}


def make_fallback_stack(shape: tuple[int, int], seed: int,
                        crs: str = "EPSG:32643") -> FeatureStack:
    """Build a geo-referenced :class:`FeatureStack` from the fallback fields."""
    layers = _fallback_fields(shape, seed)
    rows, cols = shape
    res = 100.0
    # rasterio-style affine: (a, b, c, d, e, f) -> x = a*col + c, y = e*row + f
    transform = (res, 0.0, 500000.0, 0.0, -res, 3000000.0)
    bounds = (500000.0, 3000000.0 - rows * res, 500000.0 + cols * res, 3000000.0)
    return FeatureStack.from_arrays(
        layers, transform=transform, crs=crs, bounds=bounds,
        meta={"source": "test-fallback", "seed": seed},
    )


@pytest.fixture()
def fallback_stack(tiny_shape: tuple[int, int]) -> FeatureStack:
    """A FeatureStack from the dependency-free fallback generator (always available)."""
    return make_fallback_stack(tiny_shape, TINY_SEED)


@pytest.fixture()
def synthetic_stack(tiny_config: Config, tiny_shape: tuple[int, int]) -> FeatureStack:
    """A small synthetic FeatureStack.

    Prefers the real :class:`SyntheticDataSource` (so the fixture exercises the
    production synthetic path once that sibling module exists); otherwise falls
    back to the self-contained generator so the deterministic-core tests still
    have a valid stack to run against.
    """
    try:
        from urbanheat.synthetic.source import SyntheticDataSource
    except Exception:
        return make_fallback_stack(tiny_shape, TINY_SEED)
    fs = SyntheticDataSource().load(tiny_config)
    fs.validate()
    return fs
