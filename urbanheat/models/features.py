"""urbanheat.models.features — assemble the predictor matrix X and target y.

This module turns a :class:`~urbanheat.datamodel.FeatureStack` (a bundle of
co-registered 2-D driver layers) into the tidy ``(X, y, coords)`` arrays the ML
core trains on, and supplies the two pieces of *physics-informed* metadata the
rest of the pipeline needs:

* :func:`monotone_constraints` — the per-predictor ``{-1, 0, +1}`` sign vector
  derived from the surface-energy-balance driver-sign table
  (``research/05 §1.6`` / :func:`urbanheat.physics.energy_balance.expected_lst_gradient_signs`).
  These signs are handed verbatim to ``HistGradientBoostingRegressor``'s
  ``monotonic_cst`` and to XGBoost / LightGBM ``monotone_constraints`` so a
  trained model can *never* predict that adding vegetation warms a pixel.
* :func:`predictor_grid` — the full ``(H*W, P)`` matrix for whole-grid
  prediction (NaNs preserved so masked pixels stay masked).

Design rules honoured here
--------------------------
* ``numpy`` / ``scipy`` are top-level; ``pandas`` is imported lazily and only
  used when a DataFrame return is explicitly requested — the numpy path works
  with zero optional dependencies.
* All layer references use the canonical names from
  :mod:`urbanheat.datamodel`; no ad-hoc names are invented here.

Public API
----------
``build_xy`` and friends match the Module Interface Contract (ARCHITECTURE.md
§11.5); :func:`build_feature_table` / :func:`train_test_split_spatial` are the
ergonomic array-first wrappers used by the training / validation code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from urbanheat.datamodel import (
    DEFAULT_PREDICTORS,
    LST,
    FeatureStack,
)

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import pandas as pd


__all__ = [
    "build_feature_table",
    "build_xy",
    "monotone_constraints",
    "predictor_grid",
    "train_test_split_spatial",
    "resolve_predictors",
]


# ---------------------------------------------------------------------------
# Physics-informed monotonicity signs (the bridge between SEB physics and ML)
# ---------------------------------------------------------------------------
# Fallback driver -> expected sign of dLST/ddriver, transcribed verbatim from the
# surface-energy-balance sign table (research/05 §1.6). The authoritative source
# at runtime is ``physics.energy_balance.expected_lst_gradient_signs()``; this
# dict is used only when that optional module is unavailable, so the ML core
# stays decoupled from (and never blocked by) the physics builder.
#   -1 => driver up  =>  LST down (cooler)
#   +1 => driver up  =>  LST up   (hotter)
#    0 => no monotone constraint imposed
_FALLBACK_GRADIENT_SIGNS: dict[str, int] = {
    # cooling drivers (-1)
    "albedo": -1,            # (1-a)K_down down -> Q* down
    "ndvi": -1,              # Q_E up (lower Bowen ratio), slight emissivity up
    "evi": -1,
    "savi": -1,
    "lai": -1,
    "fvc": -1,
    "green_frac": -1,
    "tree_frac": -1,
    "water_frac": -1,        # Q_E up, heat capacity up
    "ndwi": -1,
    "mndwi": -1,
    "et": -1,                # latent-heat cooling
    "emissivity": -1,        # radiates outgoing longwave more efficiently
    "soil_moisture": -1,     # raises the Q_E ceiling
    "svf": -1,               # open sky -> more nocturnal longwave loss -> cooler
    "wind_speed": -1,        # lower r_a -> more sensible-heat export
    "roughness_length": -1,  # rougher -> better ventilation
    # warming drivers (+1)
    "impervious_frac": +1,   # Q_E down, storage dQ_S up, Bowen ratio up
    "ndbi": +1,
    "ndbai": +1,
    "ui": +1,
    "building_height": +1,   # radiation trapping + wall storage (esp. night)
    "building_volume": +1,   # thermal mass -> nocturnal heat release
    "aspect_ratio": +1,      # deep canyons trap longwave
    "plan_area_frac": +1,
    "anthro_heat": +1,       # direct sensible/latent source (esp. night)
    "nightlights": +1,       # anthropogenic-heat proxy
    "no2": +1,               # combustion / traffic proxy
    "solar_radiation": +1,   # more incoming K_down -> warmer surface
    "net_radiation": +1,     # more available energy
    "air_temp": +1,          # warmer air -> warmer surface (coupled)
    "lst_day": +1,
    "lst_night": +1,
}


def _expected_signs() -> dict[str, int]:
    """Return the canonical driver -> expected dLST sign map.

    Prefers the authoritative
    :func:`urbanheat.physics.energy_balance.expected_lst_gradient_signs` (imported
    lazily so this module never hard-depends on the physics builder) and falls
    back to the in-module transcription of ``research/05 §1.6`` when that module
    or function is not yet available.
    """
    try:  # optional dependency on the physics builder's deliverable
        from urbanheat.physics import energy_balance as _eb  # type: ignore

        fn = getattr(_eb, "expected_lst_gradient_signs", None)
        if callable(fn):
            signs = fn()
            if isinstance(signs, dict) and signs:
                # normalise values to ints in {-1, 0, +1}
                return {str(k): int(np.sign(v)) for k, v in signs.items()}
    except Exception:  # pragma: no cover - physics module optional/in-progress
        pass
    return dict(_FALLBACK_GRADIENT_SIGNS)


# ---------------------------------------------------------------------------
# Predictor resolution
# ---------------------------------------------------------------------------
def resolve_predictors(
    fs: FeatureStack,
    predictors: Sequence[str] | None = None,
    target: str = LST,
) -> list[str]:
    """Intersect the requested predictor list with the layers actually present.

    Parameters
    ----------
    fs:
        The source :class:`FeatureStack`.
    predictors:
        Requested predictor names; ``None`` -> :data:`DEFAULT_PREDICTORS`.
    target:
        Target layer name to exclude from the predictor set (a predictor must
        never be the target).

    Returns
    -------
    list[str]
        Present predictor names, order-preserved, target removed, de-duplicated.
    """
    requested = list(predictors) if predictors is not None else list(DEFAULT_PREDICTORS)
    seen: set[str] = set()
    out: list[str] = []
    for name in requested:
        if name == target or name in seen:
            continue
        if fs.has(name):
            out.append(name)
            seen.add(name)
    return out


# ---------------------------------------------------------------------------
# Monotonicity constraints (§11.5 contract)
# ---------------------------------------------------------------------------
def monotone_constraints(predictors: Sequence[str]) -> dict[str, int]:
    """Map each predictor to its expected ``{-1, 0, +1}`` ``dLST`` sign.

    The signs come from the SEB driver-sign table (``research/05 §1.6``):
    ``albedo / ndvi / water / green / tree / svf / emissivity / wind`` => ``-1``
    (cooling); ``impervious / ndbi / building_* / anthro_heat`` => ``+1``
    (warming). Unknown predictors get ``0`` (unconstrained). This dict feeds
    ``HistGradientBoostingRegressor(monotonic_cst=...)`` and the
    ``monotone_constraints`` of XGBoost / LightGBM. [R5 §3]

    Parameters
    ----------
    predictors:
        Ordered predictor names.

    Returns
    -------
    dict[str, int]
        ``{predictor: sign}`` for every predictor (0 when no physics sign known).
    """
    signs = _expected_signs()
    return {name: int(signs.get(name, 0)) for name in predictors}


def monotone_constraints_vector(predictors: Sequence[str]) -> np.ndarray:
    """Return the monotone-constraint signs as an ``(P,)`` int array, predictor-ordered.

    This is the form consumed directly by ``sklearn``'s ``monotonic_cst`` and by
    the XGBoost/LightGBM tuple-style ``monotone_constraints``.
    """
    cst = monotone_constraints(predictors)
    return np.asarray([cst[name] for name in predictors], dtype=int)


# ---------------------------------------------------------------------------
# Core flatten: FeatureStack -> arrays
# ---------------------------------------------------------------------------
def _flatten_stack(
    fs: FeatureStack,
    predictors: Sequence[str],
    target: str | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Flatten the chosen layers to ``(X_raw, y_raw, coords_raw)`` (NaNs kept).

    ``X_raw`` is ``(H*W, P)`` float64, ``y_raw`` is ``(H*W,)`` or ``None`` when
    ``target`` is ``None``/absent, ``coords_raw`` is ``(H*W, 2)`` ``[x, y]`` in
    CRS units.
    """
    xx, yy = fs.grid_coords()
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    if predictors:
        cols = [fs.get(name).ravel().astype(np.float64) for name in predictors]
        x_raw = np.column_stack(cols)
    else:
        x_raw = np.empty((coords.shape[0], 0), dtype=np.float64)
    y_raw: np.ndarray | None = None
    if target is not None and fs.has(target):
        y_raw = fs.get(target).ravel().astype(np.float64)
    return x_raw, y_raw, coords


def _impute_columns(x: np.ndarray, strategy: str = "median") -> np.ndarray:
    """Replace NaNs in each column with that column's median (or mean).

    A fully-NaN column is filled with 0.0. Returns a new array.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    if x.size == 0:
        return x
    for j in range(x.shape[1]):
        col = x[:, j]
        mask = np.isnan(col)
        if not mask.any():
            continue
        good = col[~mask]
        if good.size == 0:
            fill = 0.0
        elif strategy == "mean":
            fill = float(np.mean(good))
        else:
            fill = float(np.median(good))
        col[mask] = fill
        x[:, j] = col
    return x


def build_feature_table(
    stack: FeatureStack,
    predictors: Sequence[str] = DEFAULT_PREDICTORS,
    target: str | None = "lst",
    *,
    dropna: bool = True,
    impute: str | None = None,
    standardize: bool = False,
    max_samples: int | None = 50_000,
    seed: int = 0,
    return_dataframe: bool = False,
) -> tuple[Any, Any, np.ndarray, list[str]]:
    """Flatten a :class:`FeatureStack` to a sample table for ML.

    This is the array-first builder used across the ML core. It selects the
    requested predictors (intersected with present layers), flattens every
    layer to one row per pixel, handles missing values, optionally
    standardizes, subsamples large grids, and returns plain numpy arrays (plus
    the resolved feature-name list). With ``return_dataframe=True`` it instead
    returns a lazily-imported ``pandas.DataFrame`` for ``X``/``y``.

    Parameters
    ----------
    stack:
        Source :class:`FeatureStack`.
    predictors:
        Requested predictor names (default :data:`DEFAULT_PREDICTORS`); only
        those present in ``stack`` are kept, target excluded.
    target:
        Target layer name (default ``"lst"``). ``None`` -> no ``y`` is built
        (``y`` returned as ``None``), e.g. for pure inference tables.
    dropna:
        Drop any row with a NaN in a predictor or the target. Applied before
        ``impute`` (so ``dropna`` then ``impute`` is a no-op on rows kept).
    impute:
        ``None`` (default), ``"median"`` or ``"mean"`` — column-wise fill of
        residual NaNs (useful when ``dropna=False`` to keep every pixel).
    standardize:
        Z-score each predictor column (mean 0, unit variance). The fitted
        ``(mean, scale)`` are returned via the DataFrame's ``attrs`` and are
        otherwise discarded — training code that needs them should standardize
        inside its own pipeline. Trees do not require it.
    max_samples:
        Random (seeded) subsample cap for large grids (the "sample, don't haul"
        rule). ``None`` keeps all rows.
    seed:
        RNG seed for subsampling reproducibility.
    return_dataframe:
        Return ``X``/``y`` as a ``pandas.DataFrame``/``Series`` instead of
        numpy arrays (pandas imported lazily).

    Returns
    -------
    tuple
        ``(X, y, coords, feature_names)`` where ``X`` is ``(N, P)``, ``y`` is
        ``(N,)`` or ``None``, ``coords`` is ``(N, 2)`` ``[x, y]`` (CRS units),
        and ``feature_names`` is the resolved predictor list.
    """
    feature_names = resolve_predictors(stack, predictors, target or "")
    x_raw, y_raw, coords = _flatten_stack(stack, feature_names, target)

    n = coords.shape[0]
    keep = np.ones(n, dtype=bool)
    if dropna:
        if x_raw.shape[1] > 0:
            keep &= ~np.isnan(x_raw).any(axis=1)
        if y_raw is not None:
            keep &= ~np.isnan(y_raw)
    x = x_raw[keep]
    coords = coords[keep]
    y = y_raw[keep] if y_raw is not None else None

    if impute is not None and x.shape[1] > 0:
        x = _impute_columns(x, strategy=impute)

    # subsample (seeded) for large grids
    if max_samples is not None and x.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.shape[0], size=max_samples, replace=False)
        idx.sort()
        x = x[idx]
        coords = coords[idx]
        if y is not None:
            y = y[idx]

    mean = scale = None
    if standardize and x.shape[1] > 0:
        mean = np.nanmean(x, axis=0)
        scale = np.nanstd(x, axis=0)
        scale = np.where(scale == 0, 1.0, scale)
        x = (x - mean) / scale

    if return_dataframe:
        import pandas as pd  # lazy

        x_df = pd.DataFrame(x, columns=list(feature_names))
        if mean is not None:
            x_df.attrs["standardize_mean"] = mean
            x_df.attrs["standardize_scale"] = scale
        x_df.attrs["coords"] = coords
        y_ser = pd.Series(y, name=target) if y is not None else None
        return x_df, y_ser, coords, list(feature_names)

    return x, y, coords, list(feature_names)


def build_xy(
    fs: FeatureStack,
    predictors: Sequence[str] | None = None,
    target: str = LST,
    dropna: bool = True,
    max_samples: int | None = 50_000,
    seed: int = 0,
) -> tuple["pd.DataFrame", "pd.Series", np.ndarray]:
    """Assemble ``(X, y, coords)`` for ML as a pandas DataFrame/Series + coords.

    This is the Module Interface Contract entry-point (ARCHITECTURE.md §11.5):
    predictors default to :data:`DEFAULT_PREDICTORS` (intersected with present
    layers), target defaults to :data:`LST`, and ``coords`` is the ``(N, 2)``
    ``[x, y]`` array used by the spatial-CV code. Thin wrapper over
    :func:`build_feature_table` with ``return_dataframe=True``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series, numpy.ndarray]
        ``X`` ``(N, P)``, ``y`` ``(N,)``, ``coords`` ``(N, 2)``.
    """
    x_df, y_ser, coords, _ = build_feature_table(
        fs,
        predictors if predictors is not None else DEFAULT_PREDICTORS,
        target,
        dropna=dropna,
        max_samples=max_samples,
        seed=seed,
        return_dataframe=True,
    )
    return x_df, y_ser, coords


def predictor_grid(fs: FeatureStack, predictors: Sequence[str]) -> np.ndarray:
    """Return an ``(H*W, P)`` predictor matrix for full-grid prediction.

    Row order is C-order (row-major) over the grid so the result can be
    reshaped back to ``fs.shape`` after prediction. NaNs are **preserved** so a
    caller can mask invalid pixels in the prediction. Predictors not present in
    the stack are emitted as all-NaN columns (keeps the column layout stable
    against a fixed predictor list, e.g. a model's ``feature_names_``).

    Parameters
    ----------
    fs:
        Source :class:`FeatureStack`.
    predictors:
        Predictor names, in the exact order expected by the model.

    Returns
    -------
    numpy.ndarray
        ``(H*W, P)`` float64 matrix (NaNs preserved).
    """
    rows, cols = fs.shape
    n = rows * cols
    out = np.full((n, len(predictors)), np.nan, dtype=np.float64)
    for j, name in enumerate(predictors):
        if fs.has(name):
            out[:, j] = fs.get(name).ravel().astype(np.float64)
    return out


# ---------------------------------------------------------------------------
# Spatial train/test split helper
# ---------------------------------------------------------------------------
def train_test_split_spatial(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    test_size: float = 0.25,
    block_size: float | None = None,
    n_blocks: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Spatially-blocked single train/test split (no spatial leakage).

    The AOI is tiled into square blocks of side ``block_size`` (CRS units);
    whole blocks are assigned to test/train so neighbouring pixels never
    straddle the split. This is the single-split sibling of
    :func:`urbanheat.models.validation.spatial_block_cv` and is provided as a
    convenience hook for quick experiments.

    Parameters
    ----------
    X, y, coords:
        Feature matrix ``(N, P)``, target ``(N,)``, coordinates ``(N, 2)``.
    test_size:
        Approximate fraction of **blocks** assigned to the test set.
    block_size:
        Block side length in CRS units. If ``None`` it is derived to give
        roughly ``n_blocks`` blocks per axis (default ~10x10 grid of blocks).
    n_blocks:
        Target number of blocks per axis when ``block_size`` is ``None``.
    seed:
        RNG seed for the block assignment.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test, coords_train, coords_test)``.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    block_ids = _block_labels(coords, block_size=block_size, n_blocks=n_blocks)

    rng = np.random.default_rng(seed)
    uniq = np.unique(block_ids)
    rng.shuffle(uniq)
    n_test = max(1, int(round(test_size * uniq.size)))
    test_blocks = set(uniq[:n_test].tolist())
    test_mask = np.array([b in test_blocks for b in block_ids], dtype=bool)
    train_mask = ~test_mask

    return (
        X[train_mask], X[test_mask],
        y[train_mask], y[test_mask],
        coords[train_mask], coords[test_mask],
    )


def _block_labels(
    coords: np.ndarray,
    block_size: float | None = None,
    n_blocks: int | None = None,
) -> np.ndarray:
    """Assign each coordinate to an integer spatial-block id (square tiling).

    Returns an ``(N,)`` int array of block ids. Used by both the spatial split
    here and the spatial-block CV in :mod:`urbanheat.models.validation`.
    """
    coords = np.asarray(coords, dtype=np.float64)
    x = coords[:, 0]
    yv = coords[:, 1]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(yv)), float(np.max(yv))
    span_x = max(xmax - xmin, 1e-9)
    span_y = max(ymax - ymin, 1e-9)
    if block_size is None:
        nb = n_blocks if n_blocks is not None else 10
        nb = max(int(nb), 1)
        bs_x = span_x / nb
        bs_y = span_y / nb
    else:
        bs_x = bs_y = float(block_size)
    bs_x = max(bs_x, 1e-9)
    bs_y = max(bs_y, 1e-9)
    ix = np.floor((x - xmin) / bs_x).astype(np.int64)
    iy = np.floor((yv - ymin) / bs_y).astype(np.int64)
    ncols = int(np.max(ix)) + 1
    return iy * ncols + ix
