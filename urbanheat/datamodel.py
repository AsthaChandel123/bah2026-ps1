"""urbanheat.datamodel — the FeatureStack container and DataSource ABC.

This module is the **spine** of the system. Everything downstream of the data
backends consumes a single in-memory representation — the :class:`FeatureStack`
— a bundle of co-registered 2-D layers (numpy arrays on one grid) plus the
geo-referencing (affine transform, CRS, bounds) and metadata.

Two interchangeable backends (``GEEDataSource`` in :mod:`urbanheat.gee.source`
and ``SyntheticDataSource`` in :mod:`urbanheat.synthetic.source`) both subclass
:class:`DataSource` and return a :class:`FeatureStack`, so the entire pipeline
(indices -> hotspots -> ML -> attribution -> intervention sim -> optimization ->
maps/report) is identical regardless of whether data came from Earth Engine or
from a synthetic generator. This is what makes the product demonstrable and
unit-testable with zero GEE credentials / network.

Canonical variable names
-------------------------
Every 2-D layer is referenced by a canonical string name, defined ONCE as a
module-level constant (e.g. ``LST = "lst"``). All modules — and the Module
Interface Contracts in ARCHITECTURE.md — use these exact names. Builders MUST
NOT invent ad-hoc layer names; add new ones here if needed.

Heavy/optional deps (xarray, rasterio) are imported lazily inside the methods
that need them so importing this module — and running the synthetic path — needs
only numpy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

if TYPE_CHECKING:  # only for type hints; never imported at runtime here
    from urbanheat.config import Config

# ===========================================================================
# CANONICAL FEATURESTACK VARIABLE NAMES  (single source of truth for layer ids)
# ===========================================================================
# --- Thermal / target ---
LST = "lst"                       # land surface temperature, degC (the model target)
LST_DAY = "lst_day"               # daytime LST, degC
LST_NIGHT = "lst_night"           # nighttime LST, degC
LST_UNCERTAINTY = "lst_uncertainty"  # per-pixel LST 1-sigma, degC
EMISSIVITY = "emissivity"         # broadband surface emissivity, 0-1

# --- Vegetation / spectral indices ---
NDVI = "ndvi"                     # normalized difference vegetation index, -1..1
EVI = "evi"                       # enhanced vegetation index
SAVI = "savi"                     # soil-adjusted vegetation index
NDWI = "ndwi"                     # water index (McFeeters), -1..1
MNDWI = "mndwi"                   # modified NDWI (built-up robust water)
NDBI = "ndbi"                     # normalized difference built-up index
NDBAI = "ndbai"                   # bare soil index
UI = "ui"                         # urban index
LAI = "lai"                       # leaf area index, m2/m2
FVC = "fvc"                       # fractional vegetation cover, 0-1
ET = "et"                         # evapotranspiration, mm or kg/m2/period (latent cooling)

# --- Surface radiative properties ---
ALBEDO = "albedo"                 # broadband shortwave albedo, 0-1

# --- Land cover / fractions ---
LULC = "lulc"                     # land-use/land-cover class code (integer)
IMPERVIOUS_FRAC = "impervious_frac"  # impervious / built fraction, 0-1 (lambda_P)
GREEN_FRAC = "green_frac"         # vegetation fraction, 0-1
WATER_FRAC = "water_frac"         # water fraction, 0-1
TREE_FRAC = "tree_frac"           # tree-canopy fraction, 0-1
LCZ = "lcz"                       # Local Climate Zone class (1-17)

# --- Urban morphology / 3D form ---
BUILDING_HEIGHT = "building_height"  # mean building height, m
BUILDING_VOLUME = "building_volume"  # built volume, m3/cell (thermal mass)
SVF = "svf"                       # sky view factor, 0-1
ASPECT_RATIO = "aspect_ratio"     # canyon H/W
PLAN_AREA_FRAC = "plan_area_frac"  # lambda_P building/plan density, 0-1
FRONTAL_AREA_INDEX = "frontal_area_index"  # lambda_F, ventilation/drag
ROUGHNESS_LENGTH = "roughness_length"  # z0, m
DISPLACEMENT_HEIGHT = "displacement_height"  # zd, m
ELEVATION = "elevation"           # terrain DEM elevation, m
SLOPE = "slope"                   # terrain slope, degrees

# --- Meteorology / atmosphere (downscaled to grid) ---
AIR_TEMP = "air_temp"             # 2 m air temperature, degC
DEWPOINT = "dewpoint"             # 2 m dewpoint temperature, degC
REL_HUMIDITY = "rel_humidity"     # relative humidity, %
WIND_SPEED = "wind_speed"         # 10 m wind speed, m/s
PRESSURE = "pressure"             # surface pressure, kPa
SOLAR_RADIATION = "solar_radiation"  # incoming shortwave K_down, W/m2
LONGWAVE_DOWN = "longwave_down"   # incoming longwave L_down, W/m2
NET_RADIATION = "net_radiation"   # net all-wave Q*, W/m2
SOIL_MOISTURE = "soil_moisture"   # surface soil moisture, m3/m3 (LE ceiling)
AOD = "aod"                       # aerosol optical depth (K_down attenuation)
PBL_HEIGHT = "pbl_height"         # boundary-layer height, m

# --- Anthropogenic / socioeconomic ---
POPULATION = "population"         # population density, persons/cell
NIGHTLIGHTS = "nightlights"       # VIIRS night-time radiance (QF proxy)
ANTHRO_HEAT = "anthro_heat"       # anthropogenic heat flux QF, W/m2
NO2 = "no2"                       # tropospheric NO2 column (combustion/QF proxy)

# --- Heat-stress / hotspot derived layers (written by indices module) ---
SUHII = "suhii"                   # surface UHI intensity, degC
UTFVI = "utfvi"                   # urban thermal field variance index (dimensionless)
EEI = "eei"                       # ecological evaluation index class
LST_PERCENTILE = "lst_percentile"  # per-pixel LST percentile rank, 0-100
LST_ZSCORE = "lst_zscore"         # per-pixel LST z-score
GISTAR_Z = "gistar_z"             # Getis-Ord Gi* z-score
MORAN_LOCAL = "moran_local"       # local Moran's I cluster category
HOTSPOT_MASK = "hotspot_mask"     # boolean surface-hotspot mask
HEAT_INDEX = "heat_index"         # NWS heat index, degC
HUMIDEX = "humidex"               # Humidex, degC
WET_BULB = "wet_bulb"             # wet-bulb temperature (Stull), degC
WBGT = "wbgt"                     # wet-bulb globe temperature, degC
UTCI = "utci"                     # universal thermal climate index, degC
HVI = "hvi"                       # heat vulnerability index, 0-1
PRIORITY_SCORE = "priority_score"  # final 0-100 layered priority score
TMRT = "tmrt"                     # mean radiant temperature (SOLWEIG), degC

# --- Per-layer uncertainty / agreement convention ---
UNCERTAINTY_SUFFIX = "_uncertainty"  # e.g. f"{ALBEDO}{UNCERTAINTY_SUFFIX}"
AGREEMENT_SUFFIX = "_agreement"      # n-source agreement count, per R2/R9

# Grouping of canonical names into the 4 PS-1 driver families (for attribution).
DRIVER_FAMILIES: dict[str, tuple[str, ...]] = {
    "lulc": (LULC, IMPERVIOUS_FRAC, GREEN_FRAC, WATER_FRAC, TREE_FRAC, LCZ),
    "morphology": (BUILDING_HEIGHT, BUILDING_VOLUME, SVF, ASPECT_RATIO,
                   PLAN_AREA_FRAC, FRONTAL_AREA_INDEX, ROUGHNESS_LENGTH,
                   ELEVATION, SLOPE),
    "vegetation": (NDVI, EVI, SAVI, LAI, FVC, ET, ALBEDO, EMISSIVITY),
    "atmosphere": (AIR_TEMP, DEWPOINT, REL_HUMIDITY, WIND_SPEED, SOLAR_RADIATION,
                   NET_RADIATION, SOIL_MOISTURE, AOD, ANTHRO_HEAT),
}

# The canonical predictor set the ML model uses by default (subset that exist
# in synthetic mode and are physically primary). Builders may extend.
DEFAULT_PREDICTORS: tuple[str, ...] = (
    NDVI, ALBEDO, IMPERVIOUS_FRAC, GREEN_FRAC, WATER_FRAC, TREE_FRAC,
    BUILDING_HEIGHT, SVF, ELEVATION, AIR_TEMP, SOLAR_RADIATION,
    SOIL_MOISTURE, ANTHRO_HEAT, EMISSIVITY,
)

# Convenience: every canonical layer name (for validation / introspection).
ALL_VARIABLES: tuple[str, ...] = tuple(
    v for k, v in globals().items()
    if k.isupper() and isinstance(v, str) and not k.endswith("SUFFIX")
    and k not in {"LST_COLOR_RAMP"}
)


# ===========================================================================
# FeatureStack
# ===========================================================================
@dataclass
class FeatureStack:
    """Co-registered 2-D layers on a single grid + geo-reference + metadata.

    A FeatureStack is the universal currency exchanged between modules. It wraps
    a dict of equal-shaped ``(H, W)`` float32 numpy arrays keyed by the canonical
    variable names above, together with the affine ``transform`` (rasterio-style
    6-tuple ``(a, b, c, d, e, f)`` mapping pixel->CRS coords), ``crs`` (e.g.
    ``"EPSG:32643"``), ``bounds`` (``(xmin, ymin, xmax, ymax)`` in CRS units) and
    free-form ``meta``.

    Invariants (enforced by :meth:`validate`):
      * every layer is a 2-D array of identical shape == :attr:`shape`;
      * ``transform`` is a length-6 affine tuple;
      * ``crs`` and ``bounds`` are set.

    Builders interact via :meth:`get`, :meth:`add_layer`, :meth:`has`,
    :meth:`grid_coords`, :meth:`sample_table` and :meth:`to_geotiff`.
    """

    layers: dict[str, np.ndarray] = field(default_factory=dict)
    transform: tuple[float, float, float, float, float, float] = (
        1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    crs: str = "EPSG:4326"
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    shape: tuple[int, int] = (0, 0)
    meta: dict[str, Any] = field(default_factory=dict)

    # ----- construction --------------------------------------------------
    def __post_init__(self) -> None:
        if self.shape == (0, 0) and self.layers:
            first = next(iter(self.layers.values()))
            self.shape = tuple(first.shape)  # type: ignore[assignment]

    @classmethod
    def empty(
        cls,
        shape: tuple[int, int],
        transform: tuple[float, float, float, float, float, float],
        crs: str,
        bounds: tuple[float, float, float, float],
        meta: dict[str, Any] | None = None,
    ) -> "FeatureStack":
        """Create an empty stack with a defined grid and no layers yet."""
        return cls(layers={}, transform=transform, crs=crs, bounds=bounds,
                   shape=shape, meta=meta or {})

    @classmethod
    def from_arrays(
        cls,
        layers: dict[str, np.ndarray],
        transform: tuple[float, float, float, float, float, float],
        crs: str,
        bounds: tuple[float, float, float, float],
        meta: dict[str, Any] | None = None,
    ) -> "FeatureStack":
        """Build directly from a dict of equal-shaped arrays."""
        stack = cls(layers={k: np.asarray(v, dtype=np.float32) for k, v in layers.items()},
                    transform=transform, crs=crs, bounds=bounds,
                    meta=meta or {})
        stack.validate()
        return stack

    # ----- access --------------------------------------------------------
    def get(self, name: str, default: np.ndarray | None = None) -> np.ndarray:
        """Return layer ``name``; raise KeyError if absent and no default given."""
        if name in self.layers:
            return self.layers[name]
        if default is not None:
            return default
        raise KeyError(f"FeatureStack has no layer {name!r}. Present: {self.names()}")

    def has(self, name: str) -> bool:
        """True if layer ``name`` is present."""
        return name in self.layers

    def names(self) -> list[str]:
        """Sorted list of present layer names."""
        return sorted(self.layers)

    def add_layer(self, name: str, array: np.ndarray, overwrite: bool = True) -> "FeatureStack":
        """Add/replace a layer (shape-checked); returns self for chaining."""
        arr = np.asarray(array, dtype=np.float32)
        if self.shape != (0, 0) and arr.shape != self.shape:
            raise ValueError(
                f"layer {name!r} shape {arr.shape} != stack shape {self.shape}")
        if name in self.layers and not overwrite:
            raise ValueError(f"layer {name!r} exists and overwrite=False")
        self.layers[name] = arr
        if self.shape == (0, 0):
            self.shape = tuple(arr.shape)  # type: ignore[assignment]
        return self

    def select(self, names: Iterable[str]) -> "FeatureStack":
        """Return a new stack containing only the requested layers (same grid)."""
        sub = {n: self.layers[n] for n in names if n in self.layers}
        return FeatureStack(layers=sub, transform=self.transform, crs=self.crs,
                            bounds=self.bounds, shape=self.shape, meta=dict(self.meta))

    # ----- validation ----------------------------------------------------
    def validate(self) -> None:
        """Assert the invariants; raise ValueError on the first violation."""
        if len(self.transform) != 6:
            raise ValueError("transform must be a length-6 affine tuple")
        if not self.crs:
            raise ValueError("crs must be set")
        if len(self.bounds) != 4:
            raise ValueError("bounds must be (xmin, ymin, xmax, ymax)")
        for name, arr in self.layers.items():
            if arr.ndim != 2:
                raise ValueError(f"layer {name!r} must be 2-D, got {arr.ndim}-D")
            if tuple(arr.shape) != self.shape:
                raise ValueError(
                    f"layer {name!r} shape {arr.shape} != stack shape {self.shape}")

    # ----- geo helpers ---------------------------------------------------
    def grid_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(xx, yy)`` arrays of pixel-centre CRS coordinates, shape == :attr:`shape`."""
        a, b, c, d, e, f = self.transform
        rows, cols = self.shape
        cc, rr = np.meshgrid(np.arange(cols) + 0.5, np.arange(rows) + 0.5)
        xx = a * cc + b * rr + c
        yy = d * cc + e * rr + f
        return xx.astype(np.float64), yy.astype(np.float64)

    def sample_table(
        self,
        variables: Iterable[str] | None = None,
        dropna: bool = True,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> "Any":
        """Flatten selected layers into a tidy table for ML.

        Returns a ``pandas.DataFrame`` with one row per pixel and columns
        ``[x, y, <variables...>]`` (``pandas`` imported lazily). When ``dropna``,
        rows with any NaN in the selected variables are removed. ``max_samples``
        randomly subsamples (seeded) for large grids — the "sample, don't haul"
        rule from R7.
        """
        import pandas as pd  # lazy

        variables = list(variables) if variables is not None else self.names()
        xx, yy = self.grid_coords()
        cols: dict[str, np.ndarray] = {"x": xx.ravel(), "y": yy.ravel()}
        for v in variables:
            cols[v] = self.get(v).ravel()
        df = pd.DataFrame(cols)
        if dropna:
            df = df.dropna(subset=[v for v in variables if v in df]).reset_index(drop=True)
        if max_samples is not None and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
        return df

    def to_geotiff(self, path: str, variables: Iterable[str] | None = None) -> str:
        """Write selected layers as a multi-band GeoTIFF (rasterio imported lazily).

        Returns the written ``path``. Band descriptions are set to the variable
        names. Used by the viz/report modules to persist outputs to ``outputs/``.
        """
        import rasterio  # lazy
        from rasterio.transform import Affine

        variables = list(variables) if variables is not None else self.names()
        a, b, c, d, e, f = self.transform
        rows, cols = self.shape
        profile = {
            "driver": "GTiff", "height": rows, "width": cols,
            "count": len(variables), "dtype": "float32",
            "crs": self.crs, "transform": Affine(a, b, c, d, e, f),
            "compress": "deflate", "nodata": float("nan"),
        }
        with rasterio.open(path, "w", **profile) as dst:
            for i, v in enumerate(variables, start=1):
                dst.write(self.get(v).astype(np.float32), i)
                dst.set_band_description(i, v)
        return path

    # ----- xarray bridge (optional) -------------------------------------
    def to_xarray(self) -> "Any":
        """Return an ``xarray.Dataset`` view of the layers (xarray imported lazily)."""
        import xarray as xr  # lazy

        xx, yy = self.grid_coords()
        x1d = xx[0, :]
        y1d = yy[:, 0]
        data_vars = {n: (("y", "x"), arr) for n, arr in self.layers.items()}
        ds = xr.Dataset(data_vars=data_vars, coords={"x": x1d, "y": y1d},
                        attrs={"crs": self.crs, "bounds": self.bounds, **self.meta})
        return ds

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"FeatureStack(shape={self.shape}, crs={self.crs!r}, "
                f"n_layers={len(self.layers)}, layers={self.names()})")


# ===========================================================================
# DataSource abstract base class
# ===========================================================================
class DataSource(ABC):
    """Abstract backend that produces a :class:`FeatureStack` from a Config.

    Two concrete implementations exist behind this one interface:

    * :class:`urbanheat.gee.source.GEEDataSource` — server-side Earth Engine
      compute (the "O(1)" production path); samples/exports small results into a
      FeatureStack.
    * :class:`urbanheat.synthetic.source.SyntheticDataSource` — generates
      physically-plausible synthetic LST + driver fields on a grid so the whole
      pipeline runs offline, with no credentials/network, for demos and tests.

    The rest of the system depends ONLY on this interface, never on a concrete
    backend, which is what makes the design source-agnostic.
    """

    #: short identifier, e.g. "gee" or "synthetic"
    name: str = "abstract"

    @abstractmethod
    def load(self, config: "Config") -> FeatureStack:
        """Build and return a fully-populated :class:`FeatureStack` for the AOI.

        Implementations honour ``config`` (AOI bbox, date range, target CRS, grid
        resolution, dataset toggles) and must return a stack whose ``crs`` ==
        ``config.target_crs`` and grid matches ``config`` resolution/extent, with
        at minimum the layer :data:`LST` populated.
        """
        raise NotImplementedError

    def available_layers(self) -> list[str]:
        """Layer names this backend can in principle produce (default: all)."""
        return list(ALL_VARIABLES)


__all__ = [
    "FeatureStack", "DataSource",
    # canonical names
    "LST", "LST_DAY", "LST_NIGHT", "LST_UNCERTAINTY", "EMISSIVITY",
    "NDVI", "EVI", "SAVI", "NDWI", "MNDWI", "NDBI", "NDBAI", "UI",
    "LAI", "FVC", "ET", "ALBEDO",
    "LULC", "IMPERVIOUS_FRAC", "GREEN_FRAC", "WATER_FRAC", "TREE_FRAC", "LCZ",
    "BUILDING_HEIGHT", "BUILDING_VOLUME", "SVF", "ASPECT_RATIO",
    "PLAN_AREA_FRAC", "FRONTAL_AREA_INDEX", "ROUGHNESS_LENGTH",
    "DISPLACEMENT_HEIGHT", "ELEVATION", "SLOPE",
    "AIR_TEMP", "DEWPOINT", "REL_HUMIDITY", "WIND_SPEED", "PRESSURE",
    "SOLAR_RADIATION", "LONGWAVE_DOWN", "NET_RADIATION", "SOIL_MOISTURE",
    "AOD", "PBL_HEIGHT",
    "POPULATION", "NIGHTLIGHTS", "ANTHRO_HEAT", "NO2",
    "SUHII", "UTFVI", "EEI", "LST_PERCENTILE", "LST_ZSCORE", "GISTAR_Z",
    "MORAN_LOCAL", "HOTSPOT_MASK", "HEAT_INDEX", "HUMIDEX", "WET_BULB",
    "WBGT", "UTCI", "HVI", "PRIORITY_SCORE", "TMRT",
    "UNCERTAINTY_SUFFIX", "AGREEMENT_SUFFIX",
    "DRIVER_FAMILIES", "DEFAULT_PREDICTORS", "ALL_VARIABLES",
]
