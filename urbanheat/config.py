"""urbanheat.config — typed run configuration + Indian city presets.

A single :class:`Config` dataclass parameterises an entire run: the area of
interest (bbox), CRS/resolution of the analysis grid, the date window, the data
backend (``'synthetic'`` or ``'gee'``), dataset toggles, and optimizer defaults.

Sensible defaults make it runnable with **zero arguments in synthetic mode**::

    from urbanheat.config import Config
    cfg = Config()                 # Delhi, pre-monsoon 2024, synthetic, 100 m
    cfg = Config.from_city("Mumbai")

No heavy imports here — pure dataclass + constants — so it loads instantly in any
environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Mode = Literal["synthetic", "gee"]

# ---------------------------------------------------------------------------
# Indian city presets: approximate analysis bbox (xmin/W, ymin/S, xmax/E, ymax/N
# in EPSG:4326) + the local UTM EPSG (metric CRS for area/morphology work).
# bboxes are intentionally city-core windows (~0.3-0.5 deg) suitable for an
# intra-urban heat study. UTM zones: 43N=EPSG:32643, 44N=32644.
# ---------------------------------------------------------------------------
CITY_PRESETS: dict[str, dict[str, Any]] = {
    "Delhi": {
        "bbox": (76.84, 28.40, 77.35, 28.88),
        "utm_epsg": "EPSG:32643",  # UTM 43N
        "name": "Delhi NCR",
    },
    "Mumbai": {
        "bbox": (72.77, 18.89, 72.99, 19.27),
        "utm_epsg": "EPSG:32643",  # UTM 43N
        "name": "Greater Mumbai",
    },
    "Hyderabad": {
        "bbox": (78.30, 17.20, 78.65, 17.55),
        "utm_epsg": "EPSG:32644",  # UTM 44N
        "name": "Hyderabad",
    },
    "Ahmedabad": {
        "bbox": (72.45, 22.92, 72.72, 23.15),
        "utm_epsg": "EPSG:32643",  # UTM 43N
        "name": "Ahmedabad",
    },
    "Bengaluru": {
        "bbox": (77.46, 12.83, 77.78, 13.14),
        "utm_epsg": "EPSG:32643",  # UTM 43N
        "name": "Bengaluru",
    },
}

DEFAULT_CITY = "Delhi"
# Pre-monsoon (Mar-May) is the hottest/driest worst-case window for India. [R2 §8]
DEFAULT_START = "2024-03-01"
DEFAULT_END = "2024-05-31"


@dataclass
class Config:
    """All parameters for one urban-heat analysis run.

    Attributes
    ----------
    city : str
        Preset key (see :data:`CITY_PRESETS`); informational once bbox is set.
    bbox : tuple[float, float, float, float]
        AOI as ``(xmin, ymin, xmax, ymax)`` in EPSG:4326 (lon/lat).
    target_crs : str
        Metric CRS of the analysis grid (city UTM zone), e.g. ``"EPSG:32643"``.
    resolution_m : float
        Grid resolution in metres (default 100 m; 30 m optional for fine work). [R2 §8]
    start_date, end_date : str
        ISO ``YYYY-MM-DD`` analysis window.
    mode : {"synthetic", "gee"}
        Which :class:`~urbanheat.datamodel.DataSource` backend to use.
    gee_project : str | None
        Google Cloud project for ``ee.Initialize`` (gee mode only).
    use_highvolume : bool
        Target the Earth Engine high-volume endpoint for many small requests. [R7 §2.1]
    datasets : dict[str, bool]
        Per-source toggles (keys are :data:`urbanheat.constants.GEE_DATASETS`
        keys); absent key => enabled.
    output_dir : str
        Directory for exported GeoTIFFs/maps/reports.
    seed : int
        Master RNG seed (synthetic generation + subsampling reproducibility).
    grid_shape : tuple[int, int] | None
        Optional explicit ``(rows, cols)``; if None it is derived from bbox &
        resolution by the data source.
    optimizer_budget : float
        Total intervention budget (currency units) for the optimizer.
    optimizer_max_area_frac : float
        Max fraction of AOI area that may be treated (0-1).
    optimizer_method : str
        ``"greedy"`` (lazy submodular, default), ``"ilp"`` or ``"nsga2"``. [R6 §7]
    equity_weighting : bool
        Weight cooling benefit by population x HVI in the optimizer. [R6 §8]
    """

    # --- area of interest ---
    city: str = DEFAULT_CITY
    bbox: tuple[float, float, float, float] = CITY_PRESETS[DEFAULT_CITY]["bbox"]
    target_crs: str = CITY_PRESETS[DEFAULT_CITY]["utm_epsg"]
    resolution_m: float = 100.0
    grid_shape: tuple[int, int] | None = None

    # --- time window ---
    start_date: str = DEFAULT_START
    end_date: str = DEFAULT_END

    # --- backend ---
    mode: Mode = "synthetic"
    gee_project: str | None = None
    use_highvolume: bool = True

    # --- dataset toggles (key -> enabled); empty => all GEE_DATASETS enabled ---
    datasets: dict[str, bool] = field(default_factory=dict)

    # --- I/O & reproducibility ---
    output_dir: str = "outputs"
    seed: int = 0

    # --- optimizer / intervention defaults [R6] ---
    optimizer_budget: float = 1.0e7
    optimizer_max_area_frac: float = 0.30   # opt gains best at 30-70% coverage [R6 §7.4]
    optimizer_method: str = "greedy"
    equity_weighting: bool = True

    # ----- constructors --------------------------------------------------
    @classmethod
    def from_city(cls, name: str, **overrides: Any) -> "Config":
        """Build a Config for a preset city; ``overrides`` set any other field.

        Raises ``KeyError`` with the list of valid cities if ``name`` is unknown
        (case-insensitive match).
        """
        key = _match_city(name)
        preset = CITY_PRESETS[key]
        base = dict(city=key, bbox=preset["bbox"], target_crs=preset["utm_epsg"])
        base.update(overrides)
        return cls(**base)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        """Build a Config from a plain dict (e.g. parsed YAML/JSON)."""
        bbox = d.get("bbox")
        if bbox is not None:
            d = {**d, "bbox": tuple(bbox)}
        gs = d.get("grid_shape")
        if gs is not None:
            d = {**d, "grid_shape": tuple(gs)}
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    # ----- helpers -------------------------------------------------------
    def is_dataset_enabled(self, key: str) -> bool:
        """True unless explicitly toggled off in :attr:`datasets`."""
        return self.datasets.get(key, True)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (for logging / report provenance)."""
        return asdict(self)

    def __post_init__(self) -> None:
        self.bbox = tuple(float(x) for x in self.bbox)  # type: ignore[assignment]
        if self.mode not in ("synthetic", "gee"):
            raise ValueError(f"mode must be 'synthetic' or 'gee', got {self.mode!r}")
        if not (self.bbox[0] < self.bbox[2] and self.bbox[1] < self.bbox[3]):
            raise ValueError(f"invalid bbox (need xmin<xmax, ymin<ymax): {self.bbox}")


def _match_city(name: str) -> str:
    """Case-insensitive resolve of a city preset key."""
    for key in CITY_PRESETS:
        if key.lower() == name.strip().lower():
            return key
    raise KeyError(
        f"unknown city {name!r}; available: {sorted(CITY_PRESETS)}")


__all__ = ["Config", "CITY_PRESETS", "DEFAULT_CITY", "DEFAULT_START", "DEFAULT_END"]
