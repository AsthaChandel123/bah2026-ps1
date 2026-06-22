"""urbanheat.interventions.catalog — the cooling-intervention type registry.

This module turns the pure-data ``constants.INTERVENTION_PARAMS`` table (cited
°C ranges, SEB mechanism, FeatureStack driver perturbations, feasibility class —
all from ``research/06``) into a small typed registry plus the two helpers the
rest of the optimization stack consumes:

* :class:`Intervention` — a dataclass view of one cooling lever (applicability,
  driver perturbations, per-unit cost, cited expected surface/air/Tmrt °C range).
* :data:`INTERVENTIONS` — ``{name -> Intervention}`` built once from the catalog.
* :func:`list_interventions` / :func:`get_intervention` — the §11.6 lookups.
* :func:`feasibility_mask` — a boolean grid telling the optimizer *which pixels*
  a given intervention may be placed on (cool roofs on built pixels, trees on
  plantable low-NDVI ground, parks on large vacant patches, water on open space).

Design rules honoured (see ARCHITECTURE §2, §9, §11.6):

* **No magic numbers** — every °C range / perturbation / mechanism string comes
  from :data:`urbanheat.constants.INTERVENTION_PARAMS` and
  :data:`urbanheat.constants.INVEST_UCM`; nothing is invented here.
* **numpy only** — no heavy/optional dependency is imported at module top, so the
  synthetic offline path imports this module with numpy alone.
* Canonical FeatureStack layer names are referenced via
  :mod:`urbanheat.datamodel` constants — never ad-hoc strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from urbanheat import constants as C
from urbanheat import datamodel as dm

# ---------------------------------------------------------------------------
# Per-unit (per-pixel-area) relative cost weights.
# ---------------------------------------------------------------------------
# constants.INTERVENTION_PARAMS stores cost as a qualitative band ("low",
# "low-med", "med", "high"). The optimizer needs a numeric per-area cost; we map
# the qualitative bands to a relative cost multiplier (currency units per m^2 of
# treated surface). These multipliers are ORDINAL placeholders consistent with
# the research cost ranking (cool roofs cheapest; parks/green roofs/water most
# expensive) — Config.optimizer_budget is in the same (relative) currency units.
# They are intentionally kept here (not in constants) because they are a
# build-time costing convention, not a cited physical constant.
COST_BAND_PER_M2: dict[str, float] = {
    "low": 5.0,
    "low-med": 12.0,
    "med": 25.0,
    "med-high": 60.0,
    "high": 120.0,
}
# Fallback if a band string is unrecognised.
_DEFAULT_COST_PER_M2 = 25.0


@dataclass(frozen=True)
class Intervention:
    """A single passive cooling lever, derived from ``INTERVENTION_PARAMS``.

    Attributes
    ----------
    name : str
        Catalog key (e.g. ``"urban_trees"``, ``"cool_roof"``).
    mechanism : str
        One-line surface-energy-balance mechanism (cited).
    perturbs : dict[str, float]
        Driver perturbations applied to a :class:`~urbanheat.datamodel.FeatureStack`
        when this intervention is placed — keyed by canonical layer name, value =
        signed delta (e.g. ``{"albedo": +0.30}`` for a cool roof). Names that are
        not canonical FeatureStack layers (e.g. ``"shade"``) are *conceptual*
        levers consumed by the InVEST/SOLWEIG estimators, not LST predictors;
        :attr:`driver_perturbs` exposes only the canonical-layer subset.
    surface_dC, air_dC, tmrt_dC : tuple[float, float]
        Cited cooling °C ranges (positive = cooling) for land-surface temperature,
        2 m air temperature and mean-radiant temperature respectively. [R6 §2]
    feasibility : str
        Feasibility class string from the catalog (``"roof"``, ``"plantable_ground"``,
        ``"vacant_2ha"`` ...); :func:`feasibility_mask` maps it to a pixel mask.
    cost_band : str
        Qualitative cost band from the catalog.
    cost_per_m2 : float
        Numeric per-area cost (currency/m^2) used by the optimizer (from
        :data:`COST_BAND_PER_M2`).
    note : str
        Provenance / caveat string from the catalog (carries the literature refs).
    """

    name: str
    mechanism: str
    perturbs: dict[str, float]
    surface_dC: tuple[float, float]
    air_dC: tuple[float, float]
    tmrt_dC: tuple[float, float]
    feasibility: str
    cost_band: str
    cost_per_m2: float
    note: str = ""

    # ----- derived views -------------------------------------------------
    @property
    def driver_perturbs(self) -> dict[str, float]:
        """Subset of :attr:`perturbs` whose keys are canonical FeatureStack layers.

        ``apply_perturbation`` only edits layers that actually exist as canonical
        driver variables; conceptual keys such as ``"shade"`` (an InVEST CC input,
        not a model predictor) are filtered out here so the ML counterfactual is
        applied to real predictors only.
        """
        return {k: v for k, v in self.perturbs.items() if k in dm.ALL_VARIABLES}

    @property
    def expected_surface_midpoint(self) -> float:
        """Midpoint of the cited surface °C range (a single prior for sanity flags)."""
        lo, hi = self.surface_dC
        return 0.5 * (lo + hi)

    @property
    def expected_air_midpoint(self) -> float:
        """Midpoint of the cited air °C range."""
        lo, hi = self.air_dC
        return 0.5 * (lo + hi)

    def in_surface_range(self, delta_c: float, slack: float = 1.5) -> bool:
        """True if a predicted surface ΔT (°C) sits within the cited range (× slack).

        Used by the optimizer to attach a *literature-range sanity flag* to each
        predicted cooling value (ARCHITECTURE §9 'Output per selected site').
        ``slack`` widens the band because magnitudes are climate/time-dependent.
        """
        lo, hi = self.surface_dC
        return (-0.25 * hi) <= delta_c <= (hi * slack)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view (for report/CLI/app serialisation)."""
        return {
            "name": self.name,
            "mechanism": self.mechanism,
            "perturbs": dict(self.perturbs),
            "driver_perturbs": self.driver_perturbs,
            "surface_dC": list(self.surface_dC),
            "air_dC": list(self.air_dC),
            "tmrt_dC": list(self.tmrt_dC),
            "feasibility": self.feasibility,
            "cost_band": self.cost_band,
            "cost_per_m2": self.cost_per_m2,
            "note": self.note,
        }


def _build_registry() -> dict[str, Intervention]:
    """Build the ``{name -> Intervention}`` registry from ``INTERVENTION_PARAMS``."""
    reg: dict[str, Intervention] = {}
    for name, p in C.INTERVENTION_PARAMS.items():
        cost_band = str(p.get("cost", "med"))
        reg[name] = Intervention(
            name=name,
            mechanism=str(p.get("mechanism", "")),
            perturbs=dict(p.get("perturbs", {})),
            surface_dC=tuple(p.get("surface_dC", (0.0, 0.0))),  # type: ignore[arg-type]
            air_dC=tuple(p.get("air_dC", (0.0, 0.0))),          # type: ignore[arg-type]
            tmrt_dC=tuple(p.get("tmrt_dC", (0.0, 0.0))),        # type: ignore[arg-type]
            feasibility=str(p.get("feasibility", "any_surface")),
            cost_band=cost_band,
            cost_per_m2=COST_BAND_PER_M2.get(cost_band, _DEFAULT_COST_PER_M2),
            note=str(p.get("note", "")),
        )
    return reg


#: The intervention registry, built once at import from the cited catalog.
INTERVENTIONS: dict[str, Intervention] = _build_registry()


# ===========================================================================
# §11.6 public lookups
# ===========================================================================
def list_interventions() -> list[str]:
    """Keys of ``constants.INTERVENTION_PARAMS`` (the intervention types)."""
    return list(C.INTERVENTION_PARAMS.keys())


def get_intervention(name: str) -> dict:
    """Return the full param dict for one intervention (mechanism, °C ranges,
    perturbs, feasibility). Raises ``KeyError`` listing valid names if unknown.

    [R6 §2]
    """
    if name not in C.INTERVENTION_PARAMS:
        raise KeyError(
            f"unknown intervention {name!r}; available: {list_interventions()}")
    return C.INTERVENTION_PARAMS[name]


def get_intervention_obj(name: str) -> Intervention:
    """Typed :class:`Intervention` accessor (convenience over the raw dict)."""
    if name not in INTERVENTIONS:
        raise KeyError(
            f"unknown intervention {name!r}; available: {list_interventions()}")
    return INTERVENTIONS[name]


# ===========================================================================
# Feasibility masks  (which pixels an intervention may be placed on)
# ===========================================================================
# LULC integer codes are backend-specific; we therefore derive feasibility from
# the *continuous* fraction layers (impervious/green/water/tree) and morphology
# wherever possible, which exist in both the GEE and synthetic backends, falling
# back to neutral all-True only when a needed layer is entirely absent.

# Default thresholds (fractions / NDVI) for feasibility logic. Conservative,
# physically-motivated; chosen to be backend-agnostic. [R6 §7.1 / §7.3]
FEASIBILITY_THRESHOLDS: dict[str, float] = {
    "built_min_impervious": 0.30,    # a "roof/built" pixel
    "plantable_max_impervious": 0.70,  # trees need some open/soft ground
    "plantable_max_ndvi": 0.55,      # only worth planting where canopy is low
    "open_max_impervious": 0.40,     # "open space" for water/parks
    "vacant_max_impervious": 0.50,   # vacant-ish ground for a park patch
    "vacant_min_patch_ha": C.INVEST_UCM["park_area_threshold_ha"],  # 2 ha rule
    "facade_min_building_h": 6.0,    # green walls need a real facade (m)
    "paved_min_impervious": 0.55,    # cool/permeable pavement target
}


def _get_or(fs: dm.FeatureStack, name: str, fill: float) -> np.ndarray:
    """Return layer ``name`` as float64, or a constant ``fill`` grid if absent."""
    if fs.has(name):
        return np.asarray(fs.get(name), dtype=np.float64)
    return np.full(fs.shape, float(fill), dtype=np.float64)


def _pixel_area_m2(fs: dm.FeatureStack) -> float:
    """Approximate pixel ground area (m^2) from the affine transform.

    Uses |a*e| (product of pixel width and height in CRS units). For a metric
    (UTM) CRS this is m^2 directly; for a geographic CRS it is an approximation.
    Falls back to ``meta['resolution_m']**2`` then 100 m pixels.
    """
    a, b, c, d, e, f = fs.transform
    px = abs(a) * abs(e)
    if np.isfinite(px) and px > 0:
        # Heuristic: geographic CRS pixels are << 1 (degrees); scale to ~m^2.
        if px < 1e-3:
            # degrees^2 -> m^2 at ~ mid-latitude (111 km/deg). Rough but only
            # used for the >=2 ha park rule, which is itself coarse.
            return float(px * (111_000.0 ** 2))
        return float(px)
    res = float(fs.meta.get("resolution_m", 100.0))
    return res * res


def _largest_true_patches(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Keep only 4-connected True components with >= ``min_pixels`` cells.

    Pure-numpy connected-components via iterative label propagation (no scipy
    dependency) so the synthetic path needs numpy only. For the small grids used
    here this is fast enough; large grids would use scipy.ndimage.label (lazy).
    """
    mask = np.asarray(mask, dtype=bool)
    if min_pixels <= 1 or not mask.any():
        return mask
    # Try scipy if available (fast); fall back to a numpy flood-fill.
    try:  # pragma: no cover - exercised only when scipy present
        from scipy import ndimage as _ndi  # type: ignore

        labels, n = _ndi.label(mask)
        if n == 0:
            return np.zeros_like(mask)
        counts = np.bincount(labels.ravel())
        keep_ids = {i for i in range(1, n + 1) if counts[i] >= min_pixels}
        return np.isin(labels, list(keep_ids))
    except Exception:
        pass

    # numpy fallback: label by iterative max-propagation on a unique-id seed.
    h, w = mask.shape
    ids = np.where(mask, np.arange(h * w).reshape(h, w) + 1, 0)
    changed = True
    while changed:
        changed = False
        prev = ids.copy()
        # propagate the max id to 4-neighbours within the mask
        up = np.zeros_like(ids); up[1:, :] = prev[:-1, :]
        dn = np.zeros_like(ids); dn[:-1, :] = prev[1:, :]
        lf = np.zeros_like(ids); lf[:, 1:] = prev[:, :-1]
        rt = np.zeros_like(ids); rt[:, :-1] = prev[:, 1:]
        nb = np.maximum.reduce([prev, up, dn, lf, rt])
        ids = np.where(mask, nb, 0)
        if not np.array_equal(ids, prev):
            changed = True
    # component sizes
    flat = ids.ravel()
    counts = np.bincount(flat)
    keep = np.zeros_like(mask)
    for cid in np.unique(flat):
        if cid == 0:
            continue
        if counts[cid] >= min_pixels:
            keep |= (ids == cid)
    return keep


def feasibility_mask(fs: dm.FeatureStack, name: str) -> np.ndarray:
    """Boolean grid where intervention ``name`` may be placed.

    Encodes the §11.6 feasibility rules from the catalog's ``feasibility`` class:

    * roof / cool_roof / green_roof  -> built/impervious pixels (a roof exists);
    * urban_trees / canopy           -> plantable ground (not fully impervious)
                                        with low existing NDVI (room to add canopy);
    * cool_pavement / permeable      -> paved pixels (high impervious, low veg);
    * water_body                     -> open space (low impervious, not already water);
    * urban_park                     -> vacant-ish patch >= 2 ha (connected pixels);
    * green_wall                     -> pixels with a real building facade;
    * increase_albedo / generic      -> any surface.

    Reads IMPERVIOUS_FRAC, NDVI, GREEN_FRAC, WATER_FRAC, TREE_FRAC,
    BUILDING_HEIGHT (each optional; neutral defaults when absent). [R6 §7.1]
    """
    obj = get_intervention_obj(name)
    feas = obj.feasibility
    T = FEASIBILITY_THRESHOLDS

    imperv = _get_or(fs, dm.IMPERVIOUS_FRAC, 0.5)
    ndvi = _get_or(fs, dm.NDVI, 0.3)
    water = _get_or(fs, dm.WATER_FRAC, 0.0)
    bh = _get_or(fs, dm.BUILDING_HEIGHT, 0.0)
    green = _get_or(fs, dm.GREEN_FRAC, np.nan)
    tree = _get_or(fs, dm.TREE_FRAC, np.nan)

    valid = np.ones(fs.shape, dtype=bool)  # placeholder for shape

    if feas in ("roof", "flat_roof"):
        # Any built pixel has roof area; green roofs additionally prefer flat
        # (low-rise) roofs but we do not have slope, so use built-ness.
        mask = imperv >= T["built_min_impervious"]

    elif feas in ("plantable_ground",):
        # Not (near-)fully impervious AND existing canopy is low (room to plant).
        room = imperv <= T["plantable_max_impervious"]
        lowcanopy = ndvi <= T["plantable_max_ndvi"]
        not_water = water < 0.5
        mask = room & lowcanopy & not_water

    elif feas in ("paved", "paved_non_pedestrian"):
        veg_ok = ndvi <= T["plantable_max_ndvi"]
        mask = (imperv >= T["paved_min_impervious"]) & veg_ok

    elif feas in ("open_space",):
        # Open, soft ground that is not already substantially water.
        mask = (imperv <= T["open_max_impervious"]) & (water < 0.5)

    elif feas in ("vacant_2ha",):
        # Vacant-ish ground, then keep only connected patches >= 2 ha.
        vacant = (imperv <= T["vacant_max_impervious"]) & (water < 0.5)
        # only meaningful where there is room for new green (low canopy)
        vacant &= ndvi <= T["plantable_max_ndvi"] + 0.15
        area = _pixel_area_m2(fs)
        ha_per_px = area / 10_000.0
        if ha_per_px <= 0:
            ha_per_px = 1.0
        min_px = max(1, int(np.ceil(T["vacant_min_patch_ha"] / ha_per_px)))
        mask = _largest_true_patches(vacant, min_px)

    elif feas in ("building_facade",):
        mask = bh >= T["facade_min_building_h"]

    elif feas in ("any_surface",):
        mask = valid.copy()

    else:
        # Unknown class: be permissive but never place on open water.
        mask = water < 0.5

    return np.asarray(mask, dtype=bool)


__all__ = [
    "Intervention",
    "INTERVENTIONS",
    "COST_BAND_PER_M2",
    "FEASIBILITY_THRESHOLDS",
    "list_interventions",
    "get_intervention",
    "get_intervention_obj",
    "feasibility_mask",
]
