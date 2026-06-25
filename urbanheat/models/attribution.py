"""urbanheat.models.attribution — quantitative driver attribution for LST.

Answers PS-1 objective 2 — *which* drivers (and how much) make a place hot — and
audits that the model learned them with the physically-correct sign. Everything
degrades gracefully: SHAP is used when available, otherwise model-agnostic
permutation importance; GWR/MGWR coefficient maps are produced when ``mgwr`` is
present, otherwise a windowed local-OLS fallback.

The headline products (``research/09 §2``):

* **Global ranked importance** — mean(|SHAP|) (or permutation Δskill) per driver.
* **Driver-family aggregation** — collapse per-feature importance into the four
  PS-1 families (:data:`urbanheat.datamodel.DRIVER_FAMILIES`): *LULC vs
  morphology vs vegetation vs atmosphere*.
* **Effect shape + physics-sign audit** — ALE / PDP curves whose slope sign must
  match the SEB table (a physics-consistency check of the ML).
* **2-method agreement rule** — an attribution claim is trusted only when two
  independent paradigms (e.g. SHAP and permutation, or SHAP and variance
  partitioning) agree on the top drivers; disagreement is flagged, not hidden.

Dependency policy: ``numpy`` / ``scipy`` top-level; ``shap`` / ``mgwr`` / ``pandas``
imported lazily inside the functions that need them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from urbanheat.datamodel import DRIVER_FAMILIES, LST, FeatureStack
from urbanheat.models.features import (
    build_feature_table,
    monotone_constraints,
    resolve_predictors,
)

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import pandas as pd

__all__ = [
    "shap_attribution",
    "shap_importance",
    "permutation_importance_attr",
    "permutation_importance",
    "partial_dependence_attr",
    "ale_curves",
    "aggregate_by_driver_family",
    "family_attribution",
    "physics_sign_audit",
    "variance_partition",
    "gwr_coefficients",
    "attribution_agreement",
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _as_matrix(X: Any) -> tuple[np.ndarray, list[str]]:
    """Coerce ``X`` (DataFrame or ndarray) to ``(ndarray, feature_names)``."""
    cols = getattr(X, "columns", None)
    if cols is not None:  # pandas DataFrame
        names = [str(c) for c in cols]
        return np.asarray(X.values, dtype=np.float64), names
    arr = np.asarray(X, dtype=np.float64)
    names = [f"f{i}" for i in range(arr.shape[1])]
    return arr, names


def _feature_names(model: Any, X: Any, fallback: Sequence[str] | None) -> list[str]:
    """Best-effort resolve of the predictor names for ``model``/``X``."""
    if fallback is not None:
        return [str(c) for c in fallback]
    names = getattr(model, "feature_names_", None)
    if names is not None:
        return [str(c) for c in names]
    cols = getattr(X, "columns", None)
    if cols is not None:
        return [str(c) for c in cols]
    arr = np.asarray(X)
    return [f"f{i}" for i in range(arr.shape[1])]


def _predict(model: Any, X: np.ndarray) -> np.ndarray:
    """Call ``model.predict`` and return a 1-D float array."""
    return np.asarray(model.predict(X), dtype=np.float64).ravel()


def _ranked_records(
    names: Sequence[str],
    mean_abs: np.ndarray,
    mean_signed: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Build a descending-by-importance list of dict records (pandas-free)."""
    order = np.argsort(-np.asarray(mean_abs, dtype=np.float64))
    total = float(np.sum(mean_abs)) or 1.0
    recs: list[dict[str, Any]] = []
    for i in order:
        rec: dict[str, Any] = {
            "feature": str(names[i]),
            "importance": float(mean_abs[i]),
            "pct": 100.0 * float(mean_abs[i]) / total,
        }
        if mean_signed is not None:
            rec["mean_signed"] = float(mean_signed[i])
        recs.append(rec)
    return recs


def _maybe_dataframe(records: list[dict[str, Any]], as_dataframe: bool) -> Any:
    """Return ``records`` as a pandas DataFrame if requested (lazy), else as-is."""
    if not as_dataframe:
        return records
    import pandas as pd  # lazy

    return pd.DataFrame.from_records(records)


def _pandas_available() -> bool:
    """True if pandas can be imported (controls DataFrame-by-default behaviour)."""
    import importlib.util

    return importlib.util.find_spec("pandas") is not None


# ---------------------------------------------------------------------------
# SHAP / permutation global importance
# ---------------------------------------------------------------------------
def shap_attribution(
    model: Any,
    X: Any,
    feature_names: Sequence[str] | None = None,
    *,
    max_samples: int = 2000,
    as_dataframe: bool = False,
    seed: int = 0,
) -> Any:
    """Global mean(|SHAP|) per feature, with a permutation fallback.

    Uses TreeSHAP via the lazily-imported ``shap`` package when available
    (``shap.TreeExplainer`` for tree ensembles, ``shap.Explainer`` otherwise).
    If ``shap`` is not installed or fails on the backend, transparently falls
    back to :func:`permutation_importance_attr` so the attribution product is
    always produced.

    Parameters
    ----------
    model:
        A fitted estimator exposing ``.predict`` (and ideally tree internals).
    X:
        Feature matrix (``pandas.DataFrame`` or ``(N, P)`` ndarray).
    feature_names:
        Optional explicit names (else inferred from ``model``/``X``).
    max_samples:
        Cap on rows used for the SHAP computation (subsampled, seeded).
    as_dataframe:
        Return a ``pandas.DataFrame`` ``[feature, importance, pct, mean_signed]``
        instead of a list of dicts.
    seed:
        Subsampling seed.

    Returns
    -------
    list[dict] | pandas.DataFrame
        Records sorted by descending ``importance`` (mean |SHAP|), each with
        ``feature``, ``importance``, ``pct`` and the signed mean SHAP
        (``mean_signed``); plus a ``method`` key on each record.
    """
    arr, inferred = _as_matrix(X)
    names = _feature_names(model, X, feature_names)
    arr = arr[~np.isnan(arr).any(axis=1)]
    if arr.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        arr = arr[rng.choice(arr.shape[0], size=max_samples, replace=False)]

    try:
        import shap  # lazy

        inner = getattr(model, "_model", model)  # unwrap LSTModel
        try:
            explainer = shap.TreeExplainer(inner)
            sv = explainer.shap_values(arr)
        except Exception:
            explainer = shap.Explainer(lambda d: _predict(model, d), arr)
            sv = explainer(arr).values
        sv = np.asarray(sv, dtype=np.float64)
        if sv.ndim == 3:  # (n, p, outputs) -> take first output
            sv = sv[..., 0]
        mean_abs = np.mean(np.abs(sv), axis=0)
        mean_signed = np.mean(sv, axis=0)
        recs = _ranked_records(names, mean_abs, mean_signed)
        for r in recs:
            r["method"] = "shap"
        return _maybe_dataframe(recs, as_dataframe)
    except Exception:
        # graceful fallback
        recs = permutation_importance_attr(
            model, X, feature_names=names, n_repeats=10, seed=seed,
            as_dataframe=False,
        )
        for r in recs:
            r.setdefault("method", "permutation_fallback")
        return _maybe_dataframe(recs, as_dataframe)


def shap_importance(model: Any, X: Any, max_samples: int = 2000) -> "pd.DataFrame":
    """Contract alias (§11.5): mean(|SHAP|) per predictor as a DataFrame.

    Returns a ``pandas.DataFrame`` with columns
    ``[feature, mean_abs_shap, mean_shap_signed]`` (descending importance).
    """
    import pandas as pd  # lazy

    recs = shap_attribution(model, X, max_samples=max_samples, as_dataframe=False)
    df = pd.DataFrame.from_records(recs)
    out = pd.DataFrame(
        {
            "feature": df["feature"],
            "mean_abs_shap": df["importance"],
            "mean_shap_signed": df.get("mean_signed", df["importance"] * 0.0),
        }
    )
    return out


def permutation_importance_attr(
    model: Any,
    X: Any,
    y: np.ndarray | None = None,
    feature_names: Sequence[str] | None = None,
    *,
    n_repeats: int = 10,
    seed: int = 0,
    as_dataframe: bool = False,
) -> Any:
    """Model-agnostic permutation importance (Δskill when a feature is shuffled).

    Uses ``sklearn.inspection.permutation_importance`` when sklearn and ``y`` are
    available; otherwise computes a self-contained permutation importance against
    the model's own predictions (variance-of-prediction sensitivity), which needs
    no labels and no sklearn. Importance = mean increase in error (or prediction
    change) over ``n_repeats`` shuffles.

    Returns records (or DataFrame) sorted by descending importance.
    """
    arr, _ = _as_matrix(X)
    names = _feature_names(model, X, feature_names)
    mask = ~np.isnan(arr).any(axis=1)
    arr = arr[mask]
    rng = np.random.default_rng(seed)

    # Path A: true permutation importance against labels via sklearn
    try:
        if y is not None:
            from sklearn.inspection import permutation_importance  # lazy

            yv = np.asarray(y, dtype=np.float64)
            yv = yv[mask] if yv.shape[0] == mask.shape[0] else yv
            inner = getattr(model, "_model", model)
            r = permutation_importance(
                inner, arr, yv, n_repeats=n_repeats, random_state=seed,
            )
            mean_abs = np.asarray(r.importances_mean, dtype=np.float64)
            mean_abs = np.clip(mean_abs, 0.0, None)
            recs = _ranked_records(names, mean_abs)
            for rec in recs:
                rec["method"] = "permutation"
            return _maybe_dataframe(recs, as_dataframe)
    except Exception:
        pass

    # Path B: label-free prediction-sensitivity permutation (numpy only)
    base = _predict(model, arr)
    p = arr.shape[1]
    scores = np.zeros(p, dtype=np.float64)
    if y is not None:
        yv = np.asarray(y, dtype=np.float64)
        yv = yv[mask] if yv.shape[0] == mask.shape[0] else None
    else:
        yv = None
    base_err = float(np.mean((base - yv) ** 2)) if yv is not None else 0.0
    for j in range(p):
        accum = 0.0
        for _ in range(n_repeats):
            Xp = arr.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            pred = _predict(model, Xp)
            if yv is not None:
                accum += float(np.mean((pred - yv) ** 2)) - base_err
            else:
                accum += float(np.mean(np.abs(pred - base)))
        scores[j] = max(accum / n_repeats, 0.0)
    recs = _ranked_records(names, scores)
    for rec in recs:
        rec["method"] = "permutation"
    return _maybe_dataframe(recs, as_dataframe)


def permutation_importance(
    model: Any,
    X: Any,
    y: np.ndarray | None = None,
    *,
    n_repeats: int = 10,
    seed: int = 0,
) -> "pd.DataFrame":
    """Contract alias (§11.5 / §9.2): permutation importance as a DataFrame.

    Returns a ``pandas.DataFrame`` ``[feature, importance, pct]`` (descending),
    wrapping :func:`permutation_importance_attr`. ``y`` enables true label-based
    Δskill importance; without it a label-free prediction-sensitivity importance
    is used.
    """
    import pandas as pd  # lazy

    recs = permutation_importance_attr(
        model, X, y=y, n_repeats=n_repeats, seed=seed, as_dataframe=False,
    )
    return pd.DataFrame.from_records(recs)


# ---------------------------------------------------------------------------
# Effect-shape curves: ALE (primary) and PDP
# ---------------------------------------------------------------------------
def partial_dependence_attr(
    model: Any,
    X: Any,
    features: Sequence[str] | None = None,
    feature_names: Sequence[str] | None = None,
    *,
    grid_resolution: int = 20,
    kind: str = "ale",
) -> dict[str, dict[str, np.ndarray]]:
    """Per-feature effect curve (``kind='ale'`` default, or ``'pdp'``).

    ALE (Accumulated Local Effects) is preferred because it is unbiased under
    correlated drivers; PDP is offered for intuition. Both are computed in pure
    numpy. Each curve is returned as ``{"x": grid, "y": effect}`` and the slope
    sign of the curve is what :func:`physics_sign_audit` checks against the SEB
    table.

    Returns
    -------
    dict[str, dict[str, numpy.ndarray]]
        ``{feature: {"x": grid (K,), "y": effect (K,)}}``.
    """
    arr, _ = _as_matrix(X)
    names = _feature_names(model, X, feature_names)
    arr = arr[~np.isnan(arr).any(axis=1)]
    feats = list(features) if features is not None else list(names)
    name_to_idx = {n: i for i, n in enumerate(names)}
    out: dict[str, dict[str, np.ndarray]] = {}
    for f in feats:
        if f not in name_to_idx:
            continue
        j = name_to_idx[f]
        col = arr[:, j]
        if kind == "pdp":
            grid = np.linspace(np.min(col), np.max(col), grid_resolution)
            ys = np.empty_like(grid)
            for k, g in enumerate(grid):
                Xp = arr.copy()
                Xp[:, j] = g
                ys[k] = float(np.mean(_predict(model, Xp)))
            out[f] = {"x": grid, "y": ys - float(np.mean(ys))}
        else:  # ALE
            out[f] = _ale_1d(model, arr, j, grid_resolution)
    return out


def _ale_1d(
    model: Any, arr: np.ndarray, j: int, n_bins: int
) -> dict[str, np.ndarray]:
    """1-D Accumulated Local Effects for feature column ``j`` (numpy)."""
    col = arr[:, j]
    edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        return {"x": np.array([float(col[0])]), "y": np.array([0.0])}
    bin_idx = np.clip(np.digitize(col, edges[1:-1], right=False), 0, edges.size - 2)
    local = np.zeros(edges.size - 1, dtype=np.float64)
    counts = np.zeros(edges.size - 1, dtype=np.float64)
    for b in range(edges.size - 1):
        sel = bin_idx == b
        if not np.any(sel):
            continue
        Xlo = arr[sel].copy()
        Xhi = arr[sel].copy()
        Xlo[:, j] = edges[b]
        Xhi[:, j] = edges[b + 1]
        diff = _predict(model, Xhi) - _predict(model, Xlo)
        local[b] = float(np.mean(diff))
        counts[b] = float(np.count_nonzero(sel))
    acc = np.concatenate([[0.0], np.cumsum(local)])
    # centre by the count-weighted mean
    centres = 0.5 * (acc[:-1] + acc[1:])
    if counts.sum() > 0:
        acc = acc - float(np.sum(centres * counts) / counts.sum())
    return {"x": edges, "y": acc}


def ale_curves(
    model: Any, X: Any, features: Sequence[str]
) -> dict[str, "pd.DataFrame"]:
    """Contract alias (§11.5): ALE curve per feature as DataFrames.

    Returns ``{feature: DataFrame[x, ale]}`` (pandas lazy).
    """
    import pandas as pd  # lazy

    curves = partial_dependence_attr(model, X, features=features, kind="ale")
    return {
        f: pd.DataFrame({"x": c["x"], "ale": c["y"]}) for f, c in curves.items()
    }


# ---------------------------------------------------------------------------
# Driver-family aggregation (LULC / morphology / vegetation / atmosphere)
# ---------------------------------------------------------------------------
def aggregate_by_driver_family(
    importances: Any,
    *,
    as_dataframe: bool = False,
) -> Any:
    """Collapse per-feature importance into the four PS-1 driver families.

    Sums each feature's importance into its family bucket
    (:data:`urbanheat.datamodel.DRIVER_FAMILIES`) and returns a ranked
    contribution of *LULC vs morphology vs vegetation vs atmosphere*.

    Parameters
    ----------
    importances:
        Either a list of ``{"feature", "importance"}`` records (as returned by
        :func:`shap_attribution` / :func:`permutation_importance_attr`), a
        ``{feature: importance}`` mapping, or a ``pandas.DataFrame`` with
        ``feature`` and an importance column (``importance`` or ``mean_abs_shap``).
    as_dataframe:
        Return a ``pandas.DataFrame`` instead of a list of records.

    Returns
    -------
    list[dict] | pandas.DataFrame
        Records ``[family, importance, pct_contribution]`` ranked descending,
        covering all four families plus an ``"other"`` bucket for unmapped
        features (only present if non-empty).
    """
    weights = _importance_mapping(importances)
    # feature -> family lookup
    feat_to_family: dict[str, str] = {}
    for fam, members in DRIVER_FAMILIES.items():
        for m in members:
            feat_to_family[m] = fam
    buckets: dict[str, float] = {fam: 0.0 for fam in DRIVER_FAMILIES}
    other = 0.0
    for feat, w in weights.items():
        fam = feat_to_family.get(feat)
        if fam is None:
            other += float(w)
        else:
            buckets[fam] += float(w)
    if other > 0:
        buckets["other"] = other
    total = float(sum(buckets.values())) or 1.0
    records = [
        {
            "family": fam,
            "importance": float(val),
            "pct_contribution": 100.0 * float(val) / total,
        }
        for fam, val in buckets.items()
    ]
    records.sort(key=lambda r: -r["importance"])
    return _maybe_dataframe(records, as_dataframe)


def _importance_mapping(importances: Any) -> dict[str, float]:
    """Normalise the several accepted importance shapes to ``{feature: weight}``."""
    # pandas DataFrame
    cols = getattr(importances, "columns", None)
    if cols is not None:
        col_names = [str(c) for c in cols]
        val_col = (
            "importance" if "importance" in col_names
            else "mean_abs_shap" if "mean_abs_shap" in col_names
            else col_names[1] if len(col_names) > 1 else col_names[0]
        )
        return {
            str(f): float(v)
            for f, v in zip(importances["feature"], importances[val_col])
        }
    # mapping
    if isinstance(importances, dict):
        return {str(k): float(v) for k, v in importances.items()}
    # list of records
    out: dict[str, float] = {}
    for rec in importances:
        feat = str(rec["feature"])
        val = rec.get("importance", rec.get("mean_abs_shap", 0.0))
        out[feat] = float(val)
    return out


def family_attribution(importance: Any) -> "pd.DataFrame":
    """Contract alias (§11.5): family-level ``[family, pct_contribution]`` DataFrame."""
    import pandas as pd  # lazy

    recs = aggregate_by_driver_family(importance, as_dataframe=False)
    return pd.DataFrame(
        {
            "family": [r["family"] for r in recs],
            "pct_contribution": [r["pct_contribution"] for r in recs],
        }
    )


# ---------------------------------------------------------------------------
# Physics sign audit
# ---------------------------------------------------------------------------
def physics_sign_audit(
    importance: Any,
    model: Any | None = None,
    X: Any | None = None,
    *,
    as_dataframe: bool | None = None,
) -> Any:
    """Check every driver's signed effect matches the SEB sign table.

    The observed sign is taken from the signed importance when available
    (``mean_signed`` / ``mean_shap_signed`` column), else estimated from the
    slope of the model's ALE curve (requires ``model`` and ``X``). Each driver
    with a non-zero expected sign is reported as ``ok`` or a violation
    (a violation means the model fit a spurious correlation against physics).

    Per the §11.5 contract this returns a ``pandas.DataFrame``
    ``[feature, expected_sign, observed_sign, ok]`` by default (when pandas is
    importable); pass ``as_dataframe=False`` to force a list of records, or
    ``True`` to require the DataFrame.
    """
    if as_dataframe is None:
        as_dataframe = _pandas_available()
    expected = monotone_constraints(_all_feature_names(importance, model, X))
    observed = _observed_signs(importance, model, X)
    records: list[dict[str, Any]] = []
    for feat, exp in expected.items():
        if exp == 0:
            continue
        obs = int(observed.get(feat, 0))
        ok = (obs == exp) or (obs == 0)
        records.append(
            {
                "feature": feat,
                "expected_sign": int(exp),
                "observed_sign": obs,
                "ok": bool(ok),
            }
        )
    return _maybe_dataframe(records, as_dataframe)


def _all_feature_names(
    importance: Any, model: Any | None, X: Any | None
) -> list[str]:
    mapping = _importance_mapping(importance) if importance is not None else {}
    if mapping:
        return list(mapping.keys())
    return _feature_names(model, X, None)


def _observed_signs(
    importance: Any, model: Any | None, X: Any | None
) -> dict[str, int]:
    """Resolve observed effect signs from signed importance or ALE slopes."""
    signs: dict[str, int] = {}
    # signed importance column / records
    cols = getattr(importance, "columns", None)
    if cols is not None:
        col_names = [str(c) for c in cols]
        signed_col = (
            "mean_signed" if "mean_signed" in col_names
            else "mean_shap_signed" if "mean_shap_signed" in col_names
            else None
        )
        if signed_col is not None:
            for f, v in zip(importance["feature"], importance[signed_col]):
                signs[str(f)] = int(np.sign(v))
            return signs
    elif isinstance(importance, (list, tuple)):
        for rec in importance:
            if "mean_signed" in rec:
                signs[str(rec["feature"])] = int(np.sign(rec["mean_signed"]))
        if signs:
            return signs
    # fall back to ALE slope
    if model is not None and X is not None:
        names = _feature_names(model, X, None)
        curves = partial_dependence_attr(model, X, features=names, kind="ale")
        for f, c in curves.items():
            ys = c["y"]
            if ys.size >= 2:
                signs[f] = int(np.sign(ys[-1] - ys[0]))
    return signs


# ---------------------------------------------------------------------------
# Variance partitioning (LMG / Shapley regression, R^2-share)
# ---------------------------------------------------------------------------
def variance_partition(
    fs: FeatureStack,
    predictors: Sequence[str],
    target: str = LST,
    *,
    max_samples: int = 5000,
    max_orderings: int = 64,
    seed: int = 0,
    as_dataframe: bool = False,
) -> Any:
    """LMG / Shapley-regression %-share of linear-model R^2 per predictor.

    Averages each predictor's marginal R^2 contribution over random orderings of
    feature entry into an OLS model (a Monte-Carlo LMG estimate; exact when the
    predictor count is small). Produces a clean additive decomposition that sums
    to the full-model R^2 — an independent cross-check of the SHAP ranking and
    one half of the :func:`attribution_agreement` rule.

    Returns records (or DataFrame) ``[feature, r2_share, pct]`` ranked descending.
    """
    feats = resolve_predictors(fs, predictors, target)
    X, y, _coords, names = build_feature_table(
        fs, feats, target, dropna=True, max_samples=max_samples, seed=seed,
    )
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = X.shape[1]
    if p == 0 or X.shape[0] < 3:
        return _maybe_dataframe([], as_dataframe)

    # standardise for numerical stability
    Xs = (X - X.mean(0)) / np.where(X.std(0) == 0, 1.0, X.std(0))
    yc = y - y.mean()

    def r2_of(cols: list[int]) -> float:
        if not cols:
            return 0.0
        A = Xs[:, cols]
        coef, *_ = np.linalg.lstsq(A, yc, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((yc - pred) ** 2))
        ss_tot = float(np.sum(yc ** 2)) or 1.0
        return max(0.0, 1.0 - ss_res / ss_tot)

    rng = np.random.default_rng(seed)
    import math

    n_perm_full = math.factorial(p)
    use_exact = n_perm_full <= max_orderings and p <= 8
    contrib = np.zeros(p, dtype=np.float64)
    if use_exact:
        from itertools import permutations

        orders = list(permutations(range(p)))
    else:
        orders = [tuple(rng.permutation(p)) for _ in range(max_orderings)]
    for order in orders:
        used: list[int] = []
        prev = 0.0
        for feat in order:
            used.append(feat)
            cur = r2_of(used)
            contrib[feat] += cur - prev
            prev = cur
    contrib /= len(orders)
    contrib = np.clip(contrib, 0.0, None)
    total = float(np.sum(contrib)) or 1.0
    records = [
        {
            "feature": names[i],
            "r2_share": float(contrib[i]),
            "pct": 100.0 * float(contrib[i]) / total,
        }
        for i in np.argsort(-contrib)
    ]
    return _maybe_dataframe(records, as_dataframe)


# ---------------------------------------------------------------------------
# GWR / MGWR coefficient maps (spatial attribution)
# ---------------------------------------------------------------------------
def gwr_coefficients(
    fs: FeatureStack,
    predictors: Sequence[str],
    target: str = LST,
    *,
    max_samples: int = 3000,
    bandwidth: float | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Per-pixel GWR/MGWR coefficient maps + a dominant-driver map.

    Uses ``mgwr`` (PySAL) when installed for a proper multiscale fit; otherwise
    falls back to a distance-weighted local-OLS at each sampled point (a compact
    GWR built on numpy/scipy). Coefficients are interpolated back to the full
    grid via nearest-neighbour so every pixel has a value.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``{predictor: coefficient_grid (H, W)}`` plus ``"dominant"`` — a grid of
        the index (into ``predictors``) of the largest-|coefficient| driver per
        pixel, and ``"local_r2"`` where the backend provides it.
    """
    feats = resolve_predictors(fs, predictors, target)
    X, y, coords, names = build_feature_table(
        fs, feats, target, dropna=True, max_samples=max_samples, seed=seed,
    )
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    rows, cols = fs.shape
    p = X.shape[1]
    if p == 0 or X.shape[0] < 3:
        return {}

    coefs: np.ndarray  # (N_sample, p)
    local_r2: np.ndarray | None = None
    try:
        from mgwr.gwr import GWR  # lazy
        from mgwr.sel_bw import Sel_BW  # lazy

        Xs = (X - X.mean(0)) / np.where(X.std(0) == 0, 1.0, X.std(0))
        yy = y.reshape(-1, 1)
        bw = bandwidth
        if bw is None:
            bw = Sel_BW(coords, yy, Xs).search()
        gwr_model = GWR(coords, yy, Xs, bw)
        results = gwr_model.fit()
        # params columns: [intercept, x1..xp]
        coefs = np.asarray(results.params, dtype=np.float64)[:, 1:]
        local_r2 = np.asarray(results.localR2, dtype=np.float64).ravel()
    except Exception:
        coefs, local_r2 = _local_ols(X, y, coords, bandwidth)

    # interpolate sampled coefficients to the full grid via nearest neighbour
    out: dict[str, np.ndarray] = {}
    grid_xy = _grid_coords_flat(fs)
    nn = _nearest_index(grid_xy, coords)
    for j, name in enumerate(names):
        out[name] = coefs[nn, j].reshape(rows, cols)
    dom = np.argmax(np.abs(coefs), axis=1)
    out["dominant"] = dom[nn].reshape(rows, cols).astype(np.float64)
    if local_r2 is not None:
        out["local_r2"] = local_r2[nn].reshape(rows, cols)
    return out


def _local_ols(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    bandwidth: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance-weighted local OLS (compact GWR fallback). Returns (coefs, local_r2)."""
    n, p = X.shape
    Xs = (X - X.mean(0)) / np.where(X.std(0) == 0, 1.0, X.std(0))
    yc = y - y.mean()
    # adaptive Gaussian bandwidth from median pairwise spacing if not given
    if bandwidth is None:
        span_x = float(np.max(coords[:, 0]) - np.min(coords[:, 0])) or 1.0
        span_y = float(np.max(coords[:, 1]) - np.min(coords[:, 1])) or 1.0
        bandwidth = 0.25 * float(np.hypot(span_x, span_y))
    bw2 = max(bandwidth * bandwidth, 1e-12)
    coefs = np.zeros((n, p), dtype=np.float64)
    local_r2 = np.zeros(n, dtype=np.float64)
    A = Xs
    for i in range(n):
        d2 = np.sum((coords - coords[i]) ** 2, axis=1)
        w = np.exp(-0.5 * d2 / bw2)
        sw = np.sqrt(w)
        Aw = A * sw[:, None]
        yw = yc * sw
        coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        coefs[i] = coef
        pred = A @ coef
        ss_res = float(np.sum(w * (yc - pred) ** 2))
        ss_tot = float(np.sum(w * yc ** 2)) or 1.0
        local_r2[i] = max(0.0, 1.0 - ss_res / ss_tot)
    return coefs, local_r2


def _grid_coords_flat(fs: FeatureStack) -> np.ndarray:
    xx, yy = fs.grid_coords()
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)


def _nearest_index(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """For each row of ``query`` return the index of the nearest ``reference`` row."""
    try:
        from scipy.spatial import cKDTree  # lazy (scipy is a top-level dep)

        tree = cKDTree(reference)
        _d, idx = tree.query(query, k=1)
        return np.asarray(idx, dtype=np.int64).ravel()
    except Exception:
        idx = np.empty(query.shape[0], dtype=np.int64)
        for i in range(query.shape[0]):
            idx[i] = int(np.argmin(np.sum((reference - query[i]) ** 2, axis=1)))
        return idx


# ---------------------------------------------------------------------------
# The 2-method-agreement rule
# ---------------------------------------------------------------------------
def attribution_agreement(
    method_a: Any,
    method_b: Any,
    *,
    top_k: int = 5,
    label_a: str = "A",
    label_b: str = "B",
    as_dataframe: bool = False,
) -> dict[str, Any]:
    """Apply the 2-method-agreement rule to two attribution rankings.

    An attribution claim is trusted only when two *independent* paradigms agree
    on the leading drivers (``research/09 §2``: SHAP vs permutation, or SHAP vs
    variance partitioning). This compares the top-``k`` features of the two
    rankings, reports the overlap (Jaccard) and the rank correlation, and marks
    each feature ``agree`` when it is in both top-``k`` sets.

    Parameters
    ----------
    method_a, method_b:
        Two attribution outputs (records / DataFrame / mapping) to compare.
    top_k:
        Size of the top set used for the overlap test.
    label_a, label_b:
        Names for the two methods (for the per-feature table).
    as_dataframe:
        Return the per-feature comparison as a ``pandas.DataFrame`` under
        ``"per_feature"``.

    Returns
    -------
    dict
        ``{top_k, jaccard, rank_spearman, agreed_features, per_feature}`` where
        ``agreed_features`` are the features in both top-``k`` sets.
    """
    ra = _importance_mapping(method_a)
    rb = _importance_mapping(method_b)
    rank_a = _rank_map(ra)
    rank_b = _rank_map(rb)
    top_a = set(_top_features(ra, top_k))
    top_b = set(_top_features(rb, top_k))
    inter = top_a & top_b
    union = top_a | top_b
    jaccard = len(inter) / len(union) if union else 0.0
    spearman = _spearman_on_common(rank_a, rank_b)

    all_feats = sorted(set(ra) | set(rb))
    per_feature: list[dict[str, Any]] = []
    for f in all_feats:
        in_a = f in top_a
        in_b = f in top_b
        per_feature.append(
            {
                "feature": f,
                f"rank_{label_a}": rank_a.get(f, None),
                f"rank_{label_b}": rank_b.get(f, None),
                "agree": bool(in_a and in_b),
            }
        )
    pf: Any = per_feature
    if as_dataframe:
        import pandas as pd  # lazy

        pf = pd.DataFrame.from_records(per_feature)
    return {
        "top_k": int(top_k),
        "jaccard": float(jaccard),
        "rank_spearman": float(spearman),
        "agreed_features": sorted(inter),
        "per_feature": pf,
    }


def _rank_map(mapping: dict[str, float]) -> dict[str, int]:
    order = sorted(mapping, key=lambda k: -mapping[k])
    return {f: i + 1 for i, f in enumerate(order)}


def _top_features(mapping: dict[str, float], k: int) -> list[str]:
    return sorted(mapping, key=lambda f: -mapping[f])[:k]


def _spearman_on_common(rank_a: dict[str, int], rank_b: dict[str, int]) -> float:
    common = sorted(set(rank_a) & set(rank_b))
    if len(common) < 2:
        return float("nan")
    a = np.array([rank_a[f] for f in common], dtype=np.float64)
    b = np.array([rank_b[f] for f in common], dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom == 0:
        return float("nan")
    return float(np.sum(a * b) / denom)
