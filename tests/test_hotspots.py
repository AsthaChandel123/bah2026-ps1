"""Tests for :mod:`urbanheat.indices.hotspots` — spatial clustering + 5-class map.

Guards (ARCHITECTURE §7, §11.3):
  * ``getis_ord_gi_star`` flags an injected hot cluster with a high +z score.
  * ``surface_hotspots`` -> HOTSPOT_MASK over the grid.
  * ``composite_priority`` -> PRIORITY_SCORE covering the grid, classifiable
    into the 5-class ``constants.HOTSPOT_LEGEND``.
  * ``heat_vulnerability_index`` -> HVI in [0, 1].

The module is built in parallel; tests skip cleanly until it is importable.
"""

from __future__ import annotations

import numpy as np
import pytest

from urbanheat import constants as C
from urbanheat import datamodel as dm
from urbanheat.datamodel import FeatureStack

hs = pytest.importorskip("urbanheat.indices.hotspots")


def _stack_with_hot_cluster(shape=(20, 20)) -> tuple[FeatureStack, tuple[slice, slice]]:
    """A cool LST field with a hot square injected in the upper-left quadrant.

    Returns the stack and the (row, col) slice of the hot block so tests can
    check that the cluster statistic lights up there.
    """
    rng = np.random.default_rng(7)
    lst = (30.0 + 0.5 * rng.standard_normal(shape)).astype(np.float32)
    block = (slice(3, 8), slice(3, 8))
    lst[block] += 12.0  # a clearly hot contiguous cluster
    span = float(np.ptp(lst)) + 1e-6  # numpy 2.x: ndarray.ptp() was removed
    # Provide a couple of exposure drivers HVI/priority may read.
    layers = {
        dm.LST: lst,
        dm.IMPERVIOUS_FRAC: np.clip(0.2 + (lst - lst.min()) / span,
                                    0, 1).astype(np.float32),
        dm.NDVI: np.clip(0.6 - (lst - lst.min()) / span,
                         -0.1, 0.9).astype(np.float32),
        dm.POPULATION: (1000.0 * np.ones(shape)).astype(np.float32),
    }
    transform = (100.0, 0.0, 0.0, 0.0, -100.0, 0.0)
    bounds = (0.0, -shape[0] * 100.0, shape[1] * 100.0, 0.0)
    fs = FeatureStack.from_arrays(layers, transform=transform,
                                  crs="EPSG:32643", bounds=bounds)
    return fs, block


def test_getis_ord_flags_hot_cluster() -> None:
    """Gi* z is significantly positive inside an injected hot cluster."""
    fs, block = _stack_with_hot_cluster()
    fs = hs.getis_ord_gi_star(fs, var=dm.LST)
    assert fs.has(dm.GISTAR_Z)
    z = fs.get(dm.GISTAR_Z)
    assert z.shape == fs.shape
    # The cluster interior should clear at least the p90 (1.65) hot-spot level.
    assert np.nanmean(z[block]) >= C.HOTSPOT_GISTAR_Z["p90"]
    # And the cluster should be among the hottest Gi* in the scene.
    assert np.nanmean(z[block]) > float(np.nanmean(z)) + 1.0


def test_getis_ord_cold_region_is_not_flagged() -> None:
    """A cool corner far from the cluster has a non-significant / negative z."""
    fs, block = _stack_with_hot_cluster()
    fs = hs.getis_ord_gi_star(fs, var=dm.LST)
    z = fs.get(dm.GISTAR_Z)
    cold = z[-4:, -4:]   # bottom-right corner, away from the hot block
    assert np.nanmean(cold) < C.HOTSPOT_GISTAR_Z["p95"]


def test_local_moran_writes_category() -> None:
    """local_moran writes a MORAN_LOCAL cluster-category grid."""
    fs, _ = _stack_with_hot_cluster()
    fs = hs.local_moran(fs, var=dm.LST)
    assert fs.has(dm.MORAN_LOCAL)
    assert fs.get(dm.MORAN_LOCAL).shape == fs.shape


def test_surface_hotspots_mask_lights_up_cluster() -> None:
    """HOTSPOT_MASK = (LST>=P90) AND (Gi*>=z) marks the injected cluster."""
    fs, block = _stack_with_hot_cluster()
    fs = hs.surface_hotspots(fs, percentile=90.0, gi_z=1.96)
    assert fs.has(dm.HOTSPOT_MASK)
    mask = np.asarray(fs.get(dm.HOTSPOT_MASK))
    assert mask.shape == fs.shape
    # Mask is boolean-ish (0/1 or bool).
    uniq = set(np.unique(mask[np.isfinite(mask)]).tolist())
    assert uniq <= {0.0, 1.0, True, False}
    # Most of the injected hot block is flagged as a surface hotspot.
    assert np.asarray(mask[block]).astype(bool).mean() >= 0.5


def test_composite_priority_is_five_class_over_grid(synthetic_stack: FeatureStack) -> None:
    """PRIORITY_SCORE covers the grid in [0,100] and bins to 5 legend classes."""
    fs = synthetic_stack
    # Provide HVI if the implementation expects it pre-computed.
    if hasattr(hs, "heat_vulnerability_index") and not fs.has(dm.HVI):
        try:
            fs = hs.heat_vulnerability_index(fs)
        except Exception:
            pass
    fs = hs.composite_priority(fs)
    assert fs.has(dm.PRIORITY_SCORE)
    ps = fs.get(dm.PRIORITY_SCORE)
    assert ps.shape == fs.shape
    finite = ps[np.isfinite(ps)]
    assert finite.size > 0
    assert finite.min() >= 0.0 - 1e-6 and finite.max() <= 100.0 + 1e-6

    # Bin into the 5 legend classes; every pixel must fall in exactly one class.
    edges = [C.HOTSPOT_LEGEND[0]["min"]] + [c["max"] for c in C.HOTSPOT_LEGEND]
    binned = np.digitize(finite, edges[1:-1])  # 0..4
    assert binned.min() >= 0 and binned.max() <= 4


def test_heat_vulnerability_index_in_unit_range(synthetic_stack: FeatureStack) -> None:
    """HVI is normalized to [0, 1]."""
    if not hasattr(hs, "heat_vulnerability_index"):
        pytest.skip("heat_vulnerability_index not implemented")
    fs = hs.heat_vulnerability_index(synthetic_stack, method="equal")
    assert fs.has(dm.HVI)
    hvi = fs.get(dm.HVI)
    finite = hvi[np.isfinite(hvi)]
    assert finite.min() >= -1e-6 and finite.max() <= 1.0 + 1e-6
