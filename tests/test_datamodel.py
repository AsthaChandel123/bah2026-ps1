"""Tests for :mod:`urbanheat.datamodel` — FeatureStack + canonical names.

Guards (ARCHITECTURE §6, §11.0):
  * canonical variable-name constants exist and are the documented strings.
  * ``FeatureStack`` construct / validate / access / round-trip
    (``from_arrays`` -> ``select`` -> ``add_layer`` -> ``grid_coords``).
  * shape invariants are enforced by ``validate`` / ``add_layer``.
  * ``DRIVER_FAMILIES`` / ``DEFAULT_PREDICTORS`` reference real canonical names.
  * the ``DataSource`` ABC cannot be instantiated.
"""

from __future__ import annotations

import numpy as np
import pytest

from urbanheat import datamodel as dm
from urbanheat.datamodel import DataSource, FeatureStack


# ---------------------------------------------------------------------------
# Canonical names
# ---------------------------------------------------------------------------
def test_core_canonical_names_exist_with_expected_values() -> None:
    """The headline canonical layer-name constants have their documented values."""
    assert dm.LST == "lst"
    assert dm.NDVI == "ndvi"
    assert dm.ALBEDO == "albedo"
    assert dm.IMPERVIOUS_FRAC == "impervious_frac"
    assert dm.GREEN_FRAC == "green_frac"
    assert dm.WATER_FRAC == "water_frac"
    assert dm.TREE_FRAC == "tree_frac"
    assert dm.SVF == "svf"
    assert dm.BUILDING_HEIGHT == "building_height"
    assert dm.AIR_TEMP == "air_temp"
    assert dm.EMISSIVITY == "emissivity"


def test_derived_layer_names_exist() -> None:
    """Hotspot/heat-stress derived-layer constants exist for downstream modules."""
    for name in ("SUHII", "UTFVI", "GISTAR_Z", "HOTSPOT_MASK", "HEAT_INDEX",
                 "HUMIDEX", "WET_BULB", "WBGT", "UTCI", "HVI", "PRIORITY_SCORE",
                 "LST_PERCENTILE", "LST_ZSCORE", "NET_RADIATION"):
        assert hasattr(dm, name), f"missing canonical name constant {name}"
        assert isinstance(getattr(dm, name), str)


def test_all_variables_is_unique_and_covers_core() -> None:
    """ALL_VARIABLES is a unique tuple of strings covering the core layers."""
    av = dm.ALL_VARIABLES
    assert isinstance(av, tuple)
    assert len(av) == len(set(av)), "ALL_VARIABLES has duplicates"
    for core in (dm.LST, dm.NDVI, dm.ALBEDO, dm.SVF, dm.IMPERVIOUS_FRAC):
        assert core in av


def test_uncertainty_and_agreement_suffixes() -> None:
    """The companion-layer suffix convention is defined and excluded from ALL_VARIABLES."""
    assert dm.UNCERTAINTY_SUFFIX == "_uncertainty"
    assert dm.AGREEMENT_SUFFIX == "_agreement"
    assert dm.UNCERTAINTY_SUFFIX not in dm.ALL_VARIABLES


# ---------------------------------------------------------------------------
# Driver families / default predictors
# ---------------------------------------------------------------------------
def test_driver_families_are_the_four_ps1_groups() -> None:
    """The four PS-1 driver families exist and reference canonical names."""
    fams = dm.DRIVER_FAMILIES
    assert set(fams) == {"lulc", "morphology", "vegetation", "atmosphere"}
    for family, members in fams.items():
        assert members, f"{family} family is empty"
        for var in members:
            assert var in dm.ALL_VARIABLES, f"{family}: {var!r} not canonical"


def test_default_predictors_subset_of_all_variables() -> None:
    """The default ML predictor set is canonical and includes primary drivers."""
    dp = dm.DEFAULT_PREDICTORS
    assert set(dp) <= set(dm.ALL_VARIABLES)
    for primary in (dm.NDVI, dm.ALBEDO, dm.IMPERVIOUS_FRAC):
        assert primary in dp


# ---------------------------------------------------------------------------
# FeatureStack construction + validation
# ---------------------------------------------------------------------------
def _grid(shape=(6, 8)):
    """A small layers dict + geo-reference for ad-hoc stacks."""
    rng = np.random.default_rng(0)
    layers = {
        dm.LST: rng.normal(35, 3, shape).astype(np.float32),
        dm.NDVI: rng.uniform(0, 0.8, shape).astype(np.float32),
    }
    transform = (100.0, 0.0, 500000.0, 0.0, -100.0, 3000000.0)
    bounds = (500000.0, 3000000.0 - shape[0] * 100.0,
              500000.0 + shape[1] * 100.0, 3000000.0)
    return layers, transform, bounds


def test_from_arrays_builds_and_validates() -> None:
    """from_arrays builds a stack, infers shape, and stores float32 arrays."""
    layers, transform, bounds = _grid((6, 8))
    fs = FeatureStack.from_arrays(layers, transform=transform,
                                  crs="EPSG:32643", bounds=bounds)
    assert fs.shape == (6, 8)
    assert fs.crs == "EPSG:32643"
    assert fs.has(dm.LST) and fs.has(dm.NDVI)
    assert sorted(fs.names()) == [dm.LST, dm.NDVI]
    assert fs.get(dm.LST).dtype == np.float32
    fs.validate()  # should not raise


def test_empty_then_add_layer() -> None:
    """empty() + add_layer() builds a stack incrementally with shape checks."""
    fs = FeatureStack.empty(shape=(4, 5),
                            transform=(100, 0, 0, 0, -100, 0),
                            crs="EPSG:4326", bounds=(0, 0, 500, 400))
    assert fs.names() == []
    arr = np.ones((4, 5), dtype=np.float32)
    fs.add_layer(dm.ALBEDO, arr)
    assert fs.has(dm.ALBEDO)
    # Chaining returns self.
    assert fs.add_layer(dm.SVF, arr) is fs


def test_get_missing_raises_and_default_works() -> None:
    """get() raises KeyError for an absent layer unless a default is given."""
    layers, transform, bounds = _grid()
    fs = FeatureStack.from_arrays(layers, transform=transform,
                                  crs="EPSG:32643", bounds=bounds)
    with pytest.raises(KeyError):
        fs.get(dm.WATER_FRAC)
    default = np.zeros(fs.shape, dtype=np.float32)
    assert fs.get(dm.WATER_FRAC, default) is default


def test_add_layer_shape_mismatch_raises() -> None:
    """Adding a wrong-shaped layer is rejected."""
    layers, transform, bounds = _grid((6, 8))
    fs = FeatureStack.from_arrays(layers, transform=transform,
                                  crs="EPSG:32643", bounds=bounds)
    with pytest.raises(ValueError):
        fs.add_layer(dm.SVF, np.ones((3, 3), dtype=np.float32))


def test_add_layer_overwrite_guard() -> None:
    """overwrite=False refuses to clobber an existing layer."""
    layers, transform, bounds = _grid((5, 5))
    fs = FeatureStack.from_arrays(layers, transform=transform,
                                  crs="EPSG:32643", bounds=bounds)
    with pytest.raises(ValueError):
        fs.add_layer(dm.LST, np.zeros((5, 5), dtype=np.float32), overwrite=False)


def test_validate_rejects_non_2d_layer() -> None:
    """validate() rejects a non-2-D layer."""
    fs = FeatureStack.empty(shape=(4, 4), transform=(1, 0, 0, 0, -1, 0),
                            crs="EPSG:4326", bounds=(0, 0, 4, 4))
    # Bypass add_layer's checks to inject a bad layer, then validate.
    fs.layers["bad"] = np.ones((4, 4, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        fs.validate()


def test_validate_rejects_bad_transform_and_bounds() -> None:
    """validate() enforces a length-6 transform and length-4 bounds."""
    fs = FeatureStack.empty(shape=(2, 2), transform=(1, 0, 0, 0, -1, 0),
                            crs="EPSG:4326", bounds=(0, 0, 2, 2))
    fs.transform = (1, 0, 0)  # type: ignore[assignment]
    with pytest.raises(ValueError):
        fs.validate()


# ---------------------------------------------------------------------------
# Access helpers: select / grid_coords / sample_table
# ---------------------------------------------------------------------------
def test_select_returns_subset_same_grid() -> None:
    """select() returns a new stack with only requested layers, same geo-ref."""
    layers, transform, bounds = _grid((6, 8))
    fs = FeatureStack.from_arrays(layers, transform=transform,
                                  crs="EPSG:32643", bounds=bounds)
    sub = fs.select([dm.LST])
    assert sub.names() == [dm.LST]
    assert sub.shape == fs.shape
    assert sub.crs == fs.crs
    assert sub.transform == fs.transform


def test_grid_coords_shape_and_orientation() -> None:
    """grid_coords returns pixel-centre x/y arrays of the grid shape.

    With a north-up affine (negative e), y decreases down the rows, so the top
    row holds the largest y. x increases left-to-right.
    """
    layers, transform, bounds = _grid((6, 8))
    fs = FeatureStack.from_arrays(layers, transform=transform,
                                  crs="EPSG:32643", bounds=bounds)
    xx, yy = fs.grid_coords()
    assert xx.shape == fs.shape and yy.shape == fs.shape
    assert xx[0, -1] > xx[0, 0]          # x increases with column
    assert yy[0, 0] > yy[-1, 0]          # y decreases with row (north-up)


def test_sample_table_roundtrip() -> None:
    """sample_table flattens to a tidy DataFrame with x/y + requested columns.

    Uses pandas, which the foundation imports lazily; skip if absent.
    """
    pytest.importorskip("pandas")
    layers, transform, bounds = _grid((6, 8))
    fs = FeatureStack.from_arrays(layers, transform=transform,
                                  crs="EPSG:32643", bounds=bounds)
    df = fs.sample_table([dm.LST, dm.NDVI])
    assert list(df.columns)[:2] == ["x", "y"]
    assert dm.LST in df.columns and dm.NDVI in df.columns
    assert len(df) == 6 * 8  # no NaNs in synthetic grid -> nothing dropped


def test_sample_table_max_samples_is_seeded() -> None:
    """max_samples subsamples reproducibly for a fixed seed."""
    pytest.importorskip("pandas")
    layers, transform, bounds = _grid((10, 10))
    fs = FeatureStack.from_arrays(layers, transform=transform,
                                  crs="EPSG:32643", bounds=bounds)
    a = fs.sample_table([dm.LST], max_samples=20, seed=42)
    b = fs.sample_table([dm.LST], max_samples=20, seed=42)
    assert len(a) == 20
    assert a[dm.LST].to_numpy().tolist() == b[dm.LST].to_numpy().tolist()


# ---------------------------------------------------------------------------
# DataSource ABC
# ---------------------------------------------------------------------------
def test_datasource_is_abstract() -> None:
    """DataSource cannot be instantiated (abstract load())."""
    with pytest.raises(TypeError):
        DataSource()  # type: ignore[abstract]


def test_datasource_subclass_must_implement_load() -> None:
    """A subclass that implements load() can be instantiated and used."""
    class Dummy(DataSource):
        name = "dummy"

        def load(self, config):  # noqa: D401, ANN001
            layers, transform, bounds = _grid((3, 3))
            return FeatureStack.from_arrays(layers, transform=transform,
                                            crs="EPSG:32643", bounds=bounds)

    d = Dummy()
    assert isinstance(d.available_layers(), list)
    fs = d.load(None)
    assert fs.has(dm.LST)
