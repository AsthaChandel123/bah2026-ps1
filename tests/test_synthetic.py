"""Tests for :mod:`urbanheat.synthetic.source` — the offline data backend.

Guards (ARCHITECTURE §11.2):
  * ``SyntheticDataSource().load(cfg)`` returns a valid :class:`FeatureStack`
    whose grid matches ``cfg.grid_shape`` and whose CRS == ``cfg.target_crs``,
    populated with the full contracted driver list.
  * the synthetic LST respects the SEB sign table — corr(LST, impervious) > 0
    and corr(LST, NDVI) < 0 — so attribution / counterfactuals behave.
  * ``make_synthetic_fields`` / ``synthesize_lst`` pure helpers behave.
  * generation is reproducible for a fixed seed.

The module is built in parallel; tests skip cleanly until it is importable.
"""

from __future__ import annotations

import numpy as np
import pytest

from urbanheat import datamodel as dm
from urbanheat.config import Config

# Skip the whole module until the synthetic backend lands.
source = pytest.importorskip("urbanheat.synthetic.source")

# The minimal driver set load() is contracted to populate (§11.2).
REQUIRED_LAYERS = [
    dm.LST, dm.LST_DAY, dm.LST_NIGHT, dm.NDVI, dm.EVI, dm.ALBEDO, dm.EMISSIVITY,
    dm.IMPERVIOUS_FRAC, dm.GREEN_FRAC, dm.WATER_FRAC, dm.TREE_FRAC, dm.LCZ,
    dm.BUILDING_HEIGHT, dm.BUILDING_VOLUME, dm.SVF, dm.ELEVATION, dm.AIR_TEMP,
    dm.DEWPOINT, dm.REL_HUMIDITY, dm.WIND_SPEED, dm.SOLAR_RADIATION,
    dm.NET_RADIATION, dm.SOIL_MOISTURE, dm.POPULATION, dm.NIGHTLIGHTS,
    dm.ANTHRO_HEAT,
]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over finite, co-located pixels."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def test_load_returns_valid_featurestack(tiny_config: Config) -> None:
    """load() returns a validated FeatureStack on the configured grid + CRS."""
    fs = source.SyntheticDataSource().load(tiny_config)
    fs.validate()
    assert fs.shape == tiny_config.grid_shape
    assert fs.crs == tiny_config.target_crs
    # Geo-reference is set.
    assert len(fs.transform) == 6
    assert len(fs.bounds) == 4


def test_load_populates_all_required_drivers(tiny_config: Config) -> None:
    """Every contracted driver layer is present (and 2-D, finite-ish)."""
    fs = source.SyntheticDataSource().load(tiny_config)
    for name in REQUIRED_LAYERS:
        assert fs.has(name), f"synthetic stack missing required layer {name!r}"
        arr = fs.get(name)
        assert arr.shape == fs.shape
        assert np.isfinite(arr).mean() > 0.5, f"{name} is mostly non-finite"


def test_load_covers_all_four_driver_families(tiny_config: Config) -> None:
    """At least one driver from each PS-1 family is present (for attribution)."""
    fs = source.SyntheticDataSource().load(tiny_config)
    for family, members in dm.DRIVER_FAMILIES.items():
        assert any(fs.has(m) for m in members), \
            f"no {family} driver present in synthetic stack"


def test_synthetic_lst_sign_physics(tiny_config: Config) -> None:
    """corr(LST, impervious) > 0 and corr(LST, NDVI) < 0 (SEB sign table)."""
    fs = source.SyntheticDataSource().load(tiny_config)
    lst = fs.get(dm.LST)
    assert _corr(lst, fs.get(dm.IMPERVIOUS_FRAC)) > 0.0, \
        "impervious surfaces must be hotter"
    assert _corr(lst, fs.get(dm.NDVI)) < 0.0, \
        "vegetation must be cooler"


def test_synthetic_water_cools(tiny_config: Config) -> None:
    """Water fraction is non-positively correlated with LST (cooling source)."""
    fs = source.SyntheticDataSource().load(tiny_config)
    # Only assert if there is meaningful water variability in this small grid.
    water = fs.get(dm.WATER_FRAC)
    if np.nanstd(water) > 1e-3:
        assert _corr(fs.get(dm.LST), water) <= 0.2


def test_load_is_reproducible(tiny_config: Config) -> None:
    """Two loads with the same seed produce identical LST fields."""
    a = source.SyntheticDataSource().load(tiny_config)
    b = source.SyntheticDataSource().load(tiny_config)
    np.testing.assert_allclose(a.get(dm.LST), b.get(dm.LST))


def test_different_seed_changes_field(tiny_shape) -> None:
    """A different seed yields a different LST field (RNG actually wired)."""
    c1 = Config(mode="synthetic", grid_shape=tiny_shape, seed=1)
    c2 = Config(mode="synthetic", grid_shape=tiny_shape, seed=2)
    a = source.SyntheticDataSource().load(c1).get(dm.LST)
    b = source.SyntheticDataSource().load(c2).get(dm.LST)
    assert not np.allclose(a, b), "LST identical across seeds -> seed not used"


def test_name_attribute() -> None:
    """The backend advertises its short name."""
    assert source.SyntheticDataSource().name == "synthetic"


def test_get_data_source_factory_returns_synthetic(tiny_config: Config) -> None:
    """The package factory returns the synthetic backend for synthetic mode."""
    from urbanheat import get_data_source
    ds = get_data_source(tiny_config)
    assert ds.name == "synthetic"
    fs = ds.load(tiny_config)
    assert fs.has(dm.LST)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not hasattr(source, "make_synthetic_fields"),
                    reason="make_synthetic_fields helper not implemented")
def test_make_synthetic_fields_helper(tiny_shape) -> None:
    """make_synthetic_fields returns canonical-name -> 2-D driver arrays of the shape.

    Per §11.2 this pure helper returns the *driver* fields (LST is derived
    separately by ``synthesize_lst``), so we assert on canonical drivers and on
    every array's shape, not on LST membership.
    """
    fields = source.make_synthetic_fields(tiny_shape, seed=0)
    assert isinstance(fields, dict) and fields
    assert dm.IMPERVIOUS_FRAC in fields and dm.NDVI in fields
    for name, arr in fields.items():
        assert name in dm.ALL_VARIABLES, f"{name!r} is not a canonical name"
        assert np.asarray(arr).shape == tiny_shape, f"{name} wrong shape"


@pytest.mark.skipif(not hasattr(source, "synthesize_lst"),
                    reason="synthesize_lst helper not implemented")
def test_synthesize_lst_helper(tiny_shape) -> None:
    """synthesize_lst maps driver arrays to a 2-D LST that respects SEB signs."""
    if not hasattr(source, "make_synthetic_fields"):
        pytest.skip("needs make_synthetic_fields to provide drivers")
    drivers = source.make_synthetic_fields(tiny_shape, seed=0)
    lst = source.synthesize_lst(drivers)
    assert np.asarray(lst).shape == tiny_shape
    if dm.IMPERVIOUS_FRAC in drivers:
        assert _corr(lst, drivers[dm.IMPERVIOUS_FRAC]) > 0.0
    if dm.NDVI in drivers:
        assert _corr(lst, drivers[dm.NDVI]) < 0.0
