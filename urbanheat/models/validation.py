"""urbanheat.models.validation — spatial CV, the metric panel, physics checks.

Honest model skill for an LST surface requires **spatial** cross-validation: a
random split leaks because neighbouring pixels are autocorrelated, inflating R^2
by ~28 % (``constants.VALIDATION_ANCHORS['spatial_cv_optimism_pct']``). This
module makes spatial block CV the headline, reports the full metric panel
(``constants.VALIDATION_METRICS``: RMSE, MAE, bias, ubRMSE, R^2, NSE, CCC, KGE)
per fold and per LCZ/LULC stratum, measures residual spatial autocorrelation
(Moran's I — structured residuals signal a missing covariate), and runs the
physics-consistency report (learned driver-effect signs vs the SEB table).

Dependency policy: ``numpy`` / ``scipy`` top-level; ``verde`` (BlockKFold) and
``pandas`` imported lazily with numpy fallbacks, so the whole validation suite
runs on numpy alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from urbanheat.constants import VALIDATION_METRICS
from urbanheat.datamodel import FeatureStack
from urbanheat.models.features import _block_labels

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import pandas as pd

__all__ = [
    "metrics",
    "compute_metrics",
    "spatial_block_cv",
    "spatial_cv",
    "stratified_errors",
    "stratified_metrics",
    "residual_autocorrelation",
    "physics_consistency_report",
]


# ===========================================================================
# Metric panel
# ===========================================================================
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the full LST validation metric panel.

    Returns a dict keyed by :data:`urbanheat.constants.VALIDATION_METRICS`
    (``rmse, mae, bias, ubrmse, r2, nse, ccc, kge``). Definitions follow
    ``research/09 §1.4``:

    * ``rmse``   = sqrt(mean((pred-true)^2))
    * ``mae``    = mean(|pred-true|)
    * ``bias``   = mean(pred-true)
    * ``ubrmse`` = sqrt(rmse^2 - bias^2)  (random error after de-biasing)
    * ``r2``     = 1 - SS_res / SS_tot
    * ``nse``    = Nash-Sutcliffe = 1 - Σ(pred-true)^2 / Σ(true-mean(true))^2
    * ``ccc``    = Lin's concordance correlation coefficient (1:1-line agreement)
    * ``kge``    = Kling-Gupta = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)

    NaNs in either input are dropped pairwise. With < 2 valid points the
    correlation-based metrics return ``nan`` (RMSE/MAE/bias still computed).
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    m = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[m], yp[m]
    n = yt.size
    out: dict[str, float] = {k: float("nan") for k in VALIDATION_METRICS}
    if n == 0:
        return out
    err = yp - yt
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    out["rmse"] = rmse
    out["mae"] = float(np.mean(np.abs(err)))
    out["bias"] = bias
    ub2 = rmse * rmse - bias * bias
    out["ubrmse"] = float(np.sqrt(ub2)) if ub2 > 0 else 0.0
    if n < 2:
        return out
    mean_t = float(np.mean(yt))
    ss_tot = float(np.sum((yt - mean_t) ** 2))
    ss_res = float(np.sum(err ** 2))
    out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    out["nse"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # Lin's CCC
    var_t = float(np.var(yt))
    var_p = float(np.var(yp))
    cov = float(np.mean((yt - mean_t) * (yp - np.mean(yp))))
    denom = var_t + var_p + (np.mean(yp) - mean_t) ** 2
    out["ccc"] = (2.0 * cov / denom) if denom > 0 else float("nan")
    # KGE
    std_t = float(np.std(yt))
    std_p = float(np.std(yp))
    if std_t > 0 and std_p > 0:
        r = float(np.corrcoef(yt, yp)[0, 1])
        alpha = std_p / std_t            # variability ratio
        beta = float(np.mean(yp)) / mean_t if mean_t != 0 else float("nan")
        if np.isnan(beta):
            out["kge"] = float("nan")
        else:
            out["kge"] = 1.0 - float(
                np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
            )
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Contract alias (§11.5) for :func:`metrics`."""
    return metrics(y_true, y_pred)


# ===========================================================================
# Spatial block cross-validation
# ===========================================================================
def spatial_block_cv(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    *,
    model_factory: Callable[[], Any] | None = None,
    n_splits: int = 5,
    block_size: float | None = None,
    seed: int = 0,
    return_dataframe: bool = False,
) -> Any:
    """Spatial block k-fold CV — the headline (leakage-free) validation.

    The AOI is tiled into square blocks (side ``block_size`` in CRS units, or
    ~derived to give a sensible block grid); whole blocks are assigned to folds
    so a fold's train and test never share a block. Uses
    ``verde.BlockKFold`` when ``verde`` is installed, otherwise an equivalent
    numpy block-grouping. For each fold a fresh model from ``model_factory`` is
    fit on train and scored on test with the full :func:`metrics` panel.

    Parameters
    ----------
    X, y, coords:
        Feature matrix ``(N, P)``, target ``(N,)``, coordinates ``(N, 2)``.
    model_factory:
        Zero-arg callable returning a fresh estimator with ``.fit(X, y)`` and
        ``.predict(X)``. If ``None``, a default :class:`LSTModel` factory is used
        (imported lazily to avoid a circular import).
    n_splits:
        Number of folds.
    block_size:
        Block side length (CRS units); ``None`` -> derived from the coordinate
        span and ``n_splits``.
    seed:
        RNG seed for block-to-fold assignment.
    return_dataframe:
        Return a ``pandas.DataFrame`` of per-fold metrics (lazy) instead of a
        list of dicts.

    Returns
    -------
    list[dict] | pandas.DataFrame
        One record per fold with the metric panel plus ``fold``, ``n_train``,
        ``n_test``; a final ``"mean"`` record aggregates across folds.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    if model_factory is None:
        model_factory = _default_model_factory(X.shape[1])

    folds = _spatial_fold_indices(coords, n_splits, block_size, seed)
    records: list[dict[str, Any]] = []
    for f, test_idx in enumerate(folds):
        test_mask = np.zeros(X.shape[0], dtype=bool)
        test_mask[test_idx] = True
        train_idx = np.where(~test_mask)[0]
        if train_idx.size < 3 or test_idx.size < 1:
            continue
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        pred = np.asarray(model.predict(X[test_idx]), dtype=np.float64)
        rec = metrics(y[test_idx], pred)
        rec.update(
            {"fold": f, "n_train": int(train_idx.size), "n_test": int(test_idx.size)}
        )
        records.append(rec)

    if records:
        agg: dict[str, Any] = {"fold": "mean"}
        for k in VALIDATION_METRICS:
            vals = [r[k] for r in records if not np.isnan(r[k])]
            agg[k] = float(np.mean(vals)) if vals else float("nan")
        agg["n_train"] = int(np.mean([r["n_train"] for r in records]))
        agg["n_test"] = int(np.mean([r["n_test"] for r in records]))
        records.append(agg)

    if return_dataframe:
        import pandas as pd  # lazy

        return pd.DataFrame.from_records(records)
    return records


def _spatial_fold_indices(
    coords: np.ndarray,
    n_splits: int,
    block_size: float | None,
    seed: int,
) -> list[np.ndarray]:
    """Assign sample indices to ``n_splits`` spatial folds (whole blocks per fold)."""
    # Try verde for a properly spaced block k-fold.
    try:
        import verde as vd  # lazy

        spacing = block_size
        if spacing is None:
            span_x = float(np.max(coords[:, 0]) - np.min(coords[:, 0])) or 1.0
            span_y = float(np.max(coords[:, 1]) - np.min(coords[:, 1])) or 1.0
            spacing = max(span_x, span_y) / max(n_splits * 2, 2)
        bkf = vd.BlockKFold(spacing=spacing, n_splits=n_splits, shuffle=True,
                            random_state=seed)
        coord_t = (coords[:, 0], coords[:, 1])
        return [np.asarray(test, dtype=np.int64)
                for _train, test in bkf.split(coord_t)]
    except Exception:
        pass

    # numpy fallback: square blocks -> assign blocks round-robin to folds.
    # block grid is sized so there are comfortably more blocks than folds.
    n_blocks_axis = max(n_splits * 3, 6)
    labels = _block_labels(coords, block_size=block_size, n_blocks=n_blocks_axis)
    uniq = np.unique(labels)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    block_to_fold = {b: i % n_splits for i, b in enumerate(uniq)}
    fold_of = np.array([block_to_fold[b] for b in labels], dtype=np.int64)
    return [np.where(fold_of == f)[0] for f in range(n_splits)]


def _default_model_factory(n_features: int) -> Callable[[], Any]:
    """Return a zero-arg factory building a default :class:`LSTModel` (lazy import)."""

    def factory() -> Any:
        from urbanheat.models.train import LSTModel  # lazy (avoid circular import)

        return LSTModel(
            predictors=[f"f{i}" for i in range(n_features)],
            n_estimators=120,
        )

    return factory


def spatial_cv(
    model_factory: Callable[[], Any],
    X: Any,
    y: Any,
    coords: np.ndarray,
    n_splits: int = 5,
    block_size: float | None = None,
    seed: int = 0,
) -> "pd.DataFrame":
    """Contract entry-point (§11.5): spatial block k-fold CV -> per-fold DataFrame.

    Wraps :func:`spatial_block_cv` with the contract argument order
    (model factory first; ``X``/``y`` may be pandas) and always returns a
    ``pandas.DataFrame`` with the :data:`VALIDATION_METRICS` columns.
    """
    Xv = np.asarray(getattr(X, "values", X), dtype=np.float64)
    yv = np.asarray(getattr(y, "values", y), dtype=np.float64)
    return spatial_block_cv(
        Xv, yv, coords,
        model_factory=model_factory,
        n_splits=n_splits,
        block_size=block_size,
        seed=seed,
        return_dataframe=True,
    )


# ===========================================================================
# Stratified metrics (per LCZ / LULC)
# ===========================================================================
def stratified_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strata: np.ndarray,
    *,
    return_dataframe: bool = False,
) -> Any:
    """Metric panel broken out per stratum (LCZ / LULC class).

    A model that is accurate over built-up pixels but poor over vegetation is
    **not** validated for cooling claims, so per-stratum metrics are mandatory
    (``research/09 §1.5``). ``strata`` is an integer/category label per sample.

    Returns records (or DataFrame), one per stratum, with the full metric panel,
    the stratum label and its sample count, sorted by stratum.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    strata = np.asarray(strata).ravel()
    records: list[dict[str, Any]] = []
    for s in _unique_sorted(strata):
        sel = strata == s
        if not np.any(sel):
            continue
        rec = metrics(yt[sel], yp[sel])
        rec["stratum"] = _scalar(s)
        rec["n"] = int(np.count_nonzero(sel))
        records.append(rec)
    if return_dataframe:
        import pandas as pd  # lazy

        return pd.DataFrame.from_records(records)
    return records


def stratified_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, strata: np.ndarray
) -> "pd.DataFrame":
    """Contract alias (§11.5) for :func:`stratified_errors` returning a DataFrame."""
    return stratified_errors(y_true, y_pred, strata, return_dataframe=True)


def _unique_sorted(a: np.ndarray) -> list[Any]:
    try:
        u = np.unique(a[~_isnan_safe(a)])
    except Exception:
        u = np.unique(a)
    return list(u)


def _isnan_safe(a: np.ndarray) -> np.ndarray:
    if np.issubdtype(a.dtype, np.floating):
        return np.isnan(a)
    return np.zeros(a.shape, dtype=bool)


def _scalar(v: Any) -> Any:
    try:
        return v.item()
    except Exception:
        return v


# ===========================================================================
# Residual spatial autocorrelation (Moran's I)
# ===========================================================================
def residual_autocorrelation(
    residuals: np.ndarray,
    coords: np.ndarray,
    *,
    k: int = 8,
    max_points: int = 4000,
    seed: int = 0,
) -> float:
    """Global Moran's I of model residuals over a k-nearest-neighbour graph.

    Structured (spatially clustered) residuals => a missing covariate or the need
    for a spatial term (``research/09 §1.5``). Moran's I near 0 means residuals
    are spatially random (good). Uses a row-standardised kNN spatial weights
    matrix; subsamples to ``max_points`` for tractability.

    Returns
    -------
    float
        Moran's I in roughly ``[-1, 1]`` (``nan`` if < 3 points).
    """
    r = np.asarray(residuals, dtype=np.float64).ravel()
    xy = np.asarray(coords, dtype=np.float64)
    m = ~np.isnan(r)
    r, xy = r[m], xy[m]
    n = r.size
    if n < 3:
        return float("nan")
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        r, xy = r[idx], xy[idx]
        n = r.size
    z = r - np.mean(r)
    denom = float(np.sum(z * z))
    if denom == 0:
        return 0.0

    kk = min(k, n - 1)
    try:
        from scipy.spatial import cKDTree  # lazy (scipy top-level)

        tree = cKDTree(xy)
        _d, nbr = tree.query(xy, k=kk + 1)  # includes self at col 0
        nbr = nbr[:, 1:]
    except Exception:
        nbr = np.empty((n, kk), dtype=np.int64)
        for i in range(n):
            d2 = np.sum((xy - xy[i]) ** 2, axis=1)
            nbr[i] = np.argsort(d2)[1:kk + 1]

    # row-standardised weights => sum of all weights W = n
    num = 0.0
    for i in range(n):
        num += float(np.sum(z[nbr[i]])) * z[i] / kk
    W = float(n)
    return (n / W) * (num / denom)


# ===========================================================================
# Physics-consistency report
# ===========================================================================
def physics_consistency_report(
    model: Any,
    X: np.ndarray | None = None,
    *,
    fs: FeatureStack | None = None,
    n_probe: int = 256,
    seed: int = 0,
    return_dataframe: bool = False,
) -> Any:
    """Audit that the model's learned driver-effect signs match the SEB table.

    Delegates to :meth:`urbanheat.models.train.LSTModel.physics_consistency` when
    ``model`` is an :class:`LSTModel`; otherwise estimates each driver's effect
    sign from the ALE slope (via :mod:`urbanheat.models.attribution`). Every
    constrained driver is reported as ``ok`` or a violation — a violation means
    the model fit a correlation against physics and its counterfactuals are not
    trustworthy (``research/09 §1.7``).

    Parameters
    ----------
    model:
        Fitted model (ideally an :class:`LSTModel`).
    X:
        Optional sample matrix to probe at (else random probes are used).
    fs:
        Optional stack to derive a probe matrix from when ``X`` is ``None``.
    n_probe:
        Number of probe points.
    seed:
        RNG seed.
    return_dataframe:
        Return a ``pandas.DataFrame`` instead of records.

    Returns
    -------
    list[dict] | pandas.DataFrame
        Records ``[feature, expected_sign, observed_sign, ok, mean_delta?]`` with
        a summary record (``feature == "__summary__"``) carrying the violation
        count and overall ``ok`` flag.
    """
    if X is None and fs is not None:
        from urbanheat.models.features import predictor_grid

        names = getattr(model, "feature_names_", None)
        if names is not None:
            Xg = predictor_grid(fs, names)
            X = Xg[~np.isnan(Xg).any(axis=1)]

    records: list[dict[str, Any]]
    report = getattr(model, "physics_consistency", None)
    if callable(report):
        res = model.physics_consistency(X=X, n_probe=n_probe, seed=seed)
        records = [
            {
                "feature": feat,
                "expected_sign": int(d["expected"]),
                "observed_sign": int(d["observed"]),
                "mean_delta": float(d["mean_delta"]),
                "ok": bool(d["ok"]),
            }
            for feat, d in res.items()
        ]
    else:
        # generic path: ALE-slope sign audit
        from urbanheat.models.attribution import physics_sign_audit

        recs = physics_sign_audit(None, model=model, X=X, as_dataframe=False)
        records = list(recs)

    n_viol = sum(1 for r in records if not r["ok"])
    records.append(
        {
            "feature": "__summary__",
            "expected_sign": 0,
            "observed_sign": 0,
            "ok": bool(n_viol == 0),
            "n_violations": int(n_viol),
            "n_checked": int(len(records)),
        }
    )
    if return_dataframe:
        import pandas as pd  # lazy

        return pd.DataFrame.from_records(records)
    return records
