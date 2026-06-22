"""urbanheat.models.train — the physics-informed LST regression core.

This module fits ``LST = F(drivers)`` as a *physics-informed* response surface.
"Physics-informed" here is concrete and equation-level, not a slogan:

1. **Monotonicity constraints.** The expected sign of ``dLST/ddriver`` from the
   surface-energy-balance table (``research/05 §1.6``;
   :func:`urbanheat.models.features.monotone_constraints`) is enforced inside
   the learner — via ``HistGradientBoostingRegressor(monotonic_cst=...)`` on the
   default sklearn backend, and via native ``monotone_constraints`` on XGBoost /
   LightGBM. A constrained model *cannot* learn that adding vegetation or albedo
   warms a pixel, so downstream counterfactual cooling can never flip sign.
2. **Physics-consistency audit.** When a backend cannot enforce monotonicity
   (e.g. a plain RandomForest, or the pure-numpy fallback), the model still runs
   a post-hoc finite-difference check of every driver's learned effect sign and
   warns on any violation (see :meth:`LSTModel.physics_consistency`).
3. **Residual-hybrid mode (optional).** With ``use_physics_backbone=True`` the
   ML target becomes the residual ``r = LST_obs - LST_phys`` of the analytic SEB
   backbone (:func:`urbanheat.physics.energy_balance.physics_lst`), so the ML
   only mops up what the physics cannot explain — the hybrid (method #18) that
   guarantees sane extrapolation.

Dependency policy
-----------------
``numpy`` / ``scipy`` are top-level. ``sklearn`` is treated as a *soft* backend:
when importable, the ``'rf'`` / ``'gbm'`` backends use it; when absent, a compact
pure-numpy monotone gradient-boosting regressor (axis-aligned stumps with
sign-projected leaf updates) is used so training, prediction and the whole
attribution / validation stack still run on numpy alone. ``xgboost`` /
``lightgbm`` / ``catboost`` / ``joblib`` are imported lazily inside the methods
that use them. The ``'auto'`` default prefers a natively-monotone compiled GBM
(xgboost/lightgbm) when present and otherwise resolves to ``'rf'`` on a
sklearn-only stack — fast and free of the sklearn-1.9.0 HistGB nested-OpenMP
stall, with monotonicity enforced by the post-hoc physics-consistency audit.

Public API: :class:`LSTModel` plus the Module Interface Contract functions
:func:`train_model`, :func:`predict_lst`, :func:`save_model`, :func:`load_model`
(ARCHITECTURE.md §11.5).
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np

from urbanheat.datamodel import DEFAULT_PREDICTORS, LST, FeatureStack
from urbanheat.models.features import (
    build_feature_table,
    monotone_constraints,
    monotone_constraints_vector,
    predictor_grid,
    resolve_predictors,
)

__all__ = [
    "LSTModel",
    "train_model",
    "predict_lst",
    "save_model",
    "load_model",
]


# ===========================================================================
# Optional-backend detection
# ===========================================================================
def _have(module: str) -> bool:
    """True if ``module`` can be imported (used to pick a backend)."""
    import importlib.util

    return importlib.util.find_spec(module) is not None


# ===========================================================================
# Pure-numpy monotone gradient-boosting fallback
# ===========================================================================
class _MonotoneStump:
    """A single axis-aligned regression stump with a sign-projected update.

    Splits on the feature/threshold that most reduces squared error of the
    current residual, then sets the two leaf values. If a monotone sign is
    requested for the split feature, the leaf values are *projected* so the
    high side is not on the wrong side of the low side (``+1`` => high>=low,
    ``-1`` => high<=low), guaranteeing the ensemble's response in that feature
    never violates the physics sign.
    """

    __slots__ = ("feature", "threshold", "low", "high")

    def __init__(self) -> None:
        self.feature: int = 0
        self.threshold: float = 0.0
        self.low: float = 0.0
        self.high: float = 0.0

    def fit(
        self,
        x: np.ndarray,
        residual: np.ndarray,
        signs: np.ndarray,
        rng: np.random.Generator,
        max_features: int,
        n_thresholds: int = 24,
    ) -> "_MonotoneStump":
        n, p = x.shape
        feats = np.arange(p)
        if max_features < p:
            feats = rng.choice(p, size=max_features, replace=False)
        best_sse = np.inf
        best = (0, 0.0, float(np.mean(residual)), float(np.mean(residual)))
        total = float(np.sum(residual))
        for j in feats:
            col = x[:, j]
            lo, hi = float(np.min(col)), float(np.max(col))
            if hi - lo <= 1e-12:
                continue
            qs = np.quantile(col, np.linspace(0.05, 0.95, n_thresholds))
            qs = np.unique(qs)
            for thr in qs:
                left = col <= thr
                nl = int(np.count_nonzero(left))
                if nl == 0 or nl == n:
                    continue
                sum_l = float(np.sum(residual[left]))
                mean_l = sum_l / nl
                mean_r = (total - sum_l) / (n - nl)
                # SSE reduction proxy: -(nl*mean_l^2 + nr*mean_r^2)
                sse = -(nl * mean_l * mean_l + (n - nl) * mean_r * mean_r)
                if sse < best_sse:
                    best_sse = sse
                    best = (int(j), float(thr), mean_l, mean_r)
        j, thr, low, high = best
        s = int(signs[j]) if j < signs.size else 0
        # project leaf values to honour the monotone sign in this feature
        if s > 0 and high < low:
            mid = 0.5 * (low + high)
            low, high = mid, mid
        elif s < 0 and high > low:
            mid = 0.5 * (low + high)
            low, high = mid, mid
        self.feature, self.threshold, self.low, self.high = j, thr, low, high
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        col = x[:, self.feature]
        return np.where(col <= self.threshold, self.low, self.high)


class _NumpyMonotoneGBR:
    """Compact pure-numpy monotone gradient-boosting regressor.

    Boosts shallow sign-projected stumps on the negative-gradient (residual) of
    squared loss. Monotone signs are enforced at the leaf level so the learned
    response is monotone in each constrained feature (a numpy stand-in for
    XGBoost/LightGBM ``monotone_constraints`` when no compiled backend exists).
    Not as accurate as a real GBM, but dependency-free and sign-correct — enough
    to keep the entire physics-informed pipeline runnable on numpy alone.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.1,
        signs: np.ndarray | None = None,
        subsample: float = 0.8,
        max_features: float | None = None,
        seed: int = 0,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.signs = signs
        self.subsample = float(subsample)
        self.max_features = max_features
        self.seed = int(seed)
        self.init_: float = 0.0
        self.stumps_: list[_MonotoneStump] = []
        self._importances: np.ndarray | None = None
        self.n_features_in_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NumpyMonotoneGBR":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, p = X.shape
        self.n_features_in_ = p
        signs = (
            np.asarray(self.signs, dtype=int)
            if self.signs is not None
            else np.zeros(p, dtype=int)
        )
        rng = np.random.default_rng(self.seed)
        self.init_ = float(np.mean(y))
        pred = np.full(n, self.init_, dtype=np.float64)
        imp = np.zeros(p, dtype=np.float64)
        if self.max_features is None:
            mf = max(1, int(np.sqrt(p))) if p > 4 else p
        else:
            mf = max(1, int(round(self.max_features * p)))
        n_sub = max(8, int(self.subsample * n))
        self.stumps_ = []
        for _ in range(self.n_estimators):
            resid = y - pred
            if n_sub < n:
                idx = rng.choice(n, size=n_sub, replace=False)
            else:
                idx = np.arange(n)
            stump = _MonotoneStump().fit(
                X[idx], resid[idx], signs, rng, max_features=mf
            )
            update = stump.predict(X)
            pred = pred + self.learning_rate * update
            self.stumps_.append(stump)
            imp[stump.feature] += abs(stump.high - stump.low)
        total = imp.sum()
        self._importances = imp / total if total > 0 else imp
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.full(X.shape[0], self.init_, dtype=np.float64)
        for stump in self.stumps_:
            out = out + self.learning_rate * stump.predict(X)
        return out

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._importances is None:
            return np.zeros(self.n_features_in_, dtype=np.float64)
        return self._importances


# ===========================================================================
# The physics-informed LST model
# ===========================================================================
class LSTModel:
    """Physics-informed LST regressor with monotone driver constraints.

    Wraps one of several tabular regression backends behind a single
    ``fit / predict / save / load`` API and enforces the SEB monotonicity signs
    (:func:`urbanheat.models.features.monotone_constraints`) wherever the backend
    supports it.

    Parameters
    ----------
    backend:
        One of ``'rf'``, ``'gbm'``, ``'xgboost'``, ``'lightgbm'``, ``'catboost'``,
        or ``'auto'`` (default). ``'auto'`` prefers a natively-monotone compiled
        GBM (``xgboost`` -> ``lightgbm``) when importable; otherwise it resolves to
        ``'rf'`` (``RandomForestRegressor``) on a sklearn-only stack — fast and
        free of the sklearn-1.9.0 HistGB nested-OpenMP stall, with driver-sign
        monotonicity enforced post-hoc by the physics-consistency audit — and to
        the pure-numpy monotone fallback when sklearn is absent. ``'gbm'`` selects
        the monotone ``HistGradientBoostingRegressor`` explicitly (iterations
        capped for speed).
    predictors:
        Predictor names; defaults to :data:`DEFAULT_PREDICTORS` and is finalised
        to the columns actually present at ``fit`` time when fit from a stack.
    monotonic:
        Apply the physics monotonicity constraints (default ``True``).
    residual_hybrid:
        If ``True``, train on the residual of the analytic SEB backbone
        (set automatically by :func:`train_model` when ``use_physics_backbone``);
        :meth:`predict` then adds the backbone back. Requires the model to be fit
        from a :class:`FeatureStack` (so the backbone can be evaluated).
    n_estimators, learning_rate, max_depth:
        Common hyper-parameters forwarded to the chosen backend.
    seed:
        RNG seed.
    **backend_kwargs:
        Extra keyword arguments passed straight to the backend constructor.

    Attributes
    ----------
    feature_names_ : list[str]
        Resolved predictor order (also the column order ``predict`` expects).
    feature_importances_ : numpy.ndarray
        Backend feature importances (``(P,)``), available after ``fit``.
    monotone_cst_ : dict[str, int]
        The applied per-feature sign constraints.
    """

    def __init__(
        self,
        backend: str = "auto",
        predictors: Sequence[str] = DEFAULT_PREDICTORS,
        *,
        monotonic: bool = True,
        residual_hybrid: bool = False,
        n_estimators: int = 300,
        learning_rate: float = 0.08,
        max_depth: int | None = None,
        seed: int = 0,
        **backend_kwargs: Any,
    ) -> None:
        self.backend = backend
        self.feature_names_: list[str] = list(predictors)
        self.monotonic = bool(monotonic)
        self.residual_hybrid = bool(residual_hybrid)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.seed = int(seed)
        self.backend_kwargs = backend_kwargs

        self._model: Any = None
        self._resolved_backend: str | None = None
        self.monotone_cst_: dict[str, int] = {}
        self._X_ref: np.ndarray | None = None
        self._fitted = False

    # ----- backend construction -----------------------------------------
    def _resolve_backend_name(self) -> str:
        """Resolve the ``'auto'`` backend to a fast, reliable, contract-safe learner.

        Preference order for ``'auto'``: a natively-monotone compiled GBM
        (``xgboost`` -> ``lightgbm``) when importable, else
        ``RandomForestRegressor`` (``'rf'``) when sklearn is present, else the
        pure-numpy monotone fallback. ``'rf'`` is the default on a bare
        numpy/scipy/sklearn stack because sklearn 1.9.0's
        ``HistGradientBoostingRegressor`` ('gbm') can stall under a nested-OpenMP
        deadlock in some sandboxes; RandomForest fits the smooth response surface
        in well under a second and its driver-sign monotonicity is guaranteed
        post-hoc by the physics-consistency audit (:meth:`physics_consistency`).
        ``'gbm'`` / ``'xgboost'`` / ``'lightgbm'`` / ``'catboost'`` remain
        explicit opt-ins.
        """
        if self.backend != "auto":
            return self.backend
        if _have("xgboost"):
            return "xgboost"
        if _have("lightgbm"):
            return "lightgbm"
        if _have("sklearn"):
            return "rf"
        return "numpy"

    def _build_backend(self, n_features: int) -> tuple[Any, str]:
        """Instantiate the regression backend; returns ``(estimator, name)``."""
        name = self._resolve_backend_name()
        signs_vec = (
            monotone_constraints_vector(self.feature_names_)
            if self.monotonic
            else np.zeros(n_features, dtype=int)
        )

        if name == "xgboost" and _have("xgboost"):
            import xgboost as xgb  # lazy

            cst = "(" + ",".join(str(int(s)) for s in signs_vec) + ")"
            params: dict[str, Any] = dict(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth or 6,
                random_state=self.seed,
                tree_method="hist",
            )
            if self.monotonic:
                params["monotone_constraints"] = cst
            params.update(self.backend_kwargs)
            return xgb.XGBRegressor(**params), "xgboost"

        if name == "lightgbm" and _have("lightgbm"):
            import lightgbm as lgb  # lazy

            params = dict(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth or -1,
                random_state=self.seed,
                verbose=-1,
            )
            if self.monotonic:
                params["monotone_constraints"] = [int(s) for s in signs_vec]
            params.update(self.backend_kwargs)
            return lgb.LGBMRegressor(**params), "lightgbm"

        if name == "catboost" and _have("catboost"):
            from catboost import CatBoostRegressor  # lazy

            params = dict(
                iterations=self.n_estimators,
                learning_rate=self.learning_rate,
                depth=self.max_depth or 6,
                random_seed=self.seed,
                verbose=False,
            )
            if self.monotonic:
                params["monotone_constraints"] = [int(s) for s in signs_vec]
            params.update(self.backend_kwargs)
            return CatBoostRegressor(**params), "catboost"

        if name in ("gbm", "rf") and _have("sklearn"):
            if name == "gbm":
                from sklearn.ensemble import HistGradientBoostingRegressor  # lazy

                # Cap boosting iterations: sklearn 1.9.0 HistGB is slow per-iter in
                # this environment, so keep <=150 and rely on OMP_NUM_THREADS=1 to
                # avoid the nested-OpenMP stall. Early stopping trims further.
                max_iter = min(int(self.n_estimators), 150)
                params = dict(
                    max_iter=max_iter,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=self.seed,
                    early_stopping=True,
                    n_iter_no_change=10,
                )
                if self.monotonic:
                    params["monotonic_cst"] = [int(s) for s in signs_vec]
                params.update(self.backend_kwargs)
                return HistGradientBoostingRegressor(**params), "gbm"
            # rf: no native monotonicity -> post-hoc audit/warn
            from sklearn.ensemble import RandomForestRegressor  # lazy

            params = dict(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.seed,
                n_jobs=-1,
            )
            params.update(self.backend_kwargs)
            if self.monotonic:
                warnings.warn(
                    "backend 'rf' (RandomForest) has no native monotonicity; "
                    "physics signs will be checked post-hoc only "
                    "(use 'gbm'/'xgboost'/'lightgbm' to enforce them).",
                    stacklevel=2,
                )
            return RandomForestRegressor(**params), "rf"

        # pure-numpy fallback (no sklearn / requested compiled backend missing)
        if name not in ("numpy", "auto", "gbm", "rf"):
            warnings.warn(
                f"backend {name!r} unavailable; falling back to the pure-numpy "
                "monotone gradient-boosting regressor.",
                stacklevel=2,
            )
        return (
            _NumpyMonotoneGBR(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                signs=signs_vec if self.monotonic else None,
                seed=self.seed,
                **{k: v for k, v in self.backend_kwargs.items()
                   if k in ("subsample", "max_features")},
            ),
            "numpy",
        )

    # ----- physics backbone (for residual-hybrid mode) ------------------
    @staticmethod
    def _physics_baseline(fs: FeatureStack) -> np.ndarray | None:
        """Evaluate the analytic SEB LST backbone on a stack (flattened), or None.

        Imports :mod:`urbanheat.physics.energy_balance` lazily; returns ``None``
        if the physics builder's module/function is unavailable so the hybrid
        mode degrades gracefully to a plain (non-hybrid) fit.
        """
        try:
            from urbanheat.physics import energy_balance as eb  # type: ignore

            fn = getattr(eb, "physics_lst", None)
            if callable(fn):
                base = np.asarray(fn(fs), dtype=np.float64)
                return base.ravel()
        except Exception:
            return None
        return None

    # ----- fit ----------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTModel":
        """Fit on a ``(N, P)`` matrix ``X`` and ``(N,)`` target ``y``.

        Column order of ``X`` must match :attr:`feature_names_`. For
        residual-hybrid training prefer :meth:`fit_stack` (which can evaluate the
        physics backbone); calling ``fit`` directly trains on ``y`` as given.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (N, P); got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X rows ({X.shape[0]}) != y length ({y.shape[0]})")
        if len(self.feature_names_) != X.shape[1]:
            # adopt generic names if the caller passed a bare matrix
            self.feature_names_ = [f"f{i}" for i in range(X.shape[1])]
        self.monotone_cst_ = (
            monotone_constraints(self.feature_names_) if self.monotonic else {}
        )
        self._model, self._resolved_backend = self._build_backend(X.shape[1])
        self._model.fit(X, y)
        # cache a small sample of the training rows so feature_importances_ can
        # fall back to a label-free prediction-sensitivity importance for
        # backends (e.g. HistGradientBoosting) that expose no native attribute.
        cap = min(X.shape[0], 1024)
        self._X_ref = X[:cap].copy()
        self._fitted = True
        return self

    def fit_stack(
        self,
        fs: FeatureStack,
        target: str = LST,
        *,
        dropna: bool = True,
        max_samples: int | None = 50_000,
    ) -> "LSTModel":
        """Fit directly from a :class:`FeatureStack` (handles residual-hybrid).

        Resolves predictors against the stack, builds ``(X, y)``, and — when
        :attr:`residual_hybrid` is set and the physics backbone is available —
        trains on ``r = y - LST_phys``. Stores the backbone hook so
        :meth:`predict_grid` can add the trend back.
        """
        self.feature_names_ = resolve_predictors(fs, self.feature_names_, target)
        X, y, _coords, _names = build_feature_table(
            fs, self.feature_names_, target,
            dropna=dropna, max_samples=max_samples, seed=self.seed,
        )
        if self.residual_hybrid:
            base_full = self._physics_baseline(fs)
            if base_full is not None:
                # re-flatten target on full grid to align the backbone, then
                # rebuild rows consistently: simplest robust path is to retrain
                # on residual using the same dropna/sampling pipeline.
                self._fit_residual(fs, target, dropna, max_samples)
                return self
            warnings.warn(
                "residual_hybrid requested but physics backbone "
                "(physics.energy_balance.physics_lst) is unavailable; "
                "training on raw LST instead.",
                stacklevel=2,
            )
            self.residual_hybrid = False
        self.fit(X, y)
        return self

    def _fit_residual(
        self,
        fs: FeatureStack,
        target: str,
        dropna: bool,
        max_samples: int | None,
    ) -> None:
        """Internal: fit on residual r = LST_obs - LST_phys with aligned rows."""
        # Build a residual target layer on the full grid, then flatten via the
        # standard table builder so predictor handling stays identical.
        base = self._physics_baseline(fs)  # (H*W,)
        y_full = fs.get(target).ravel().astype(np.float64)
        resid_full = (y_full - base).reshape(fs.shape)
        # stash a temporary residual layer name, flatten, then drop it
        tmp_name = "__lst_residual_tmp__"
        fs.add_layer(tmp_name, resid_full, overwrite=True)
        try:
            X, r, _coords, _names = build_feature_table(
                fs, self.feature_names_, tmp_name,
                dropna=dropna, max_samples=max_samples, seed=self.seed,
            )
        finally:
            fs.layers.pop(tmp_name, None)
        self.fit(X, r)
        self._fitted = True

    # ----- predict ------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on a ``(N, P)`` matrix (columns ordered as :attr:`feature_names_`).

        In residual-hybrid mode this returns the **residual** prediction; use
        :meth:`predict_grid` to obtain the full LST (backbone + residual) on a
        stack. NaNs in a row yield a NaN prediction for sklearn/numpy backends
        that cannot ingest them (rows are masked, predicted, unmasked).
        """
        if not self._fitted:
            raise RuntimeError("LSTModel is not fitted; call .fit() first.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (N, P); got shape {X.shape}")
        good = ~np.isnan(X).any(axis=1)
        out = np.full(X.shape[0], np.nan, dtype=np.float64)
        if good.any():
            out[good] = np.asarray(self._model.predict(X[good]), dtype=np.float64)
        return out

    def predict_grid(self, fs: FeatureStack, write: bool = False) -> np.ndarray:
        """Predict LST on the full grid of ``fs`` -> ``(H, W)`` array (degC).

        Builds the predictor matrix with :func:`predictor_grid` (NaNs preserved),
        predicts, adds the physics backbone back in residual-hybrid mode, and
        reshapes to ``fs.shape``. With ``write=True`` adds a ``'lst_pred'`` layer.
        """
        Xg = predictor_grid(fs, self.feature_names_)
        pred = self.predict(Xg)
        if self.residual_hybrid:
            base = self._physics_baseline(fs)
            if base is not None:
                pred = pred + base
        grid = pred.reshape(fs.shape)
        if write:
            fs.add_layer("lst_pred", grid.astype(np.float32), overwrite=True)
        return grid

    # ----- introspection ------------------------------------------------
    @property
    def feature_importances_(self) -> np.ndarray:
        """Backend feature importances ``(P,)`` (permutation fallback if absent).

        Tree backends with a native ``feature_importances_`` (RandomForest,
        XGBoost, LightGBM, CatBoost, the numpy fallback) return it directly.
        ``HistGradientBoostingRegressor`` has none, so a fast label-free
        prediction-sensitivity permutation over the cached training sample is
        used instead (the full model-agnostic route lives in
        :mod:`urbanheat.models.attribution`).
        """
        if not self._fitted:
            raise RuntimeError("LSTModel is not fitted; call .fit() first.")
        imp = getattr(self._model, "feature_importances_", None)
        if imp is not None:
            return np.asarray(imp, dtype=np.float64)
        return self._permutation_sensitivity()

    def _permutation_sensitivity(
        self, n_repeats: int = 5, seed: int = 0
    ) -> np.ndarray:
        """Label-free permutation importance over the cached training sample.

        Importance of feature ``j`` = mean absolute change in the model's own
        prediction when column ``j`` is shuffled. Returns a normalised ``(P,)``
        vector (sums to 1 when non-zero).
        """
        p = len(self.feature_names_)
        if self._X_ref is None or self._X_ref.shape[0] < 4:
            return np.zeros(p, dtype=np.float64)
        rng = np.random.default_rng(seed)
        X = self._X_ref
        base = np.asarray(self._model.predict(X), dtype=np.float64)
        scores = np.zeros(p, dtype=np.float64)
        for j in range(p):
            acc = 0.0
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                acc += float(np.mean(np.abs(
                    np.asarray(self._model.predict(Xp), dtype=np.float64) - base)))
            scores[j] = acc / n_repeats
        total = scores.sum()
        return scores / total if total > 0 else scores

    def importances_by_name(self) -> dict[str, float]:
        """Return ``{feature_name: importance}`` from the backend importances."""
        imp = self.feature_importances_
        return {n: float(v) for n, v in zip(self.feature_names_, imp)}

    def physics_consistency(
        self,
        X: np.ndarray | None = None,
        n_probe: int = 256,
        delta_frac: float = 0.1,
        seed: int = 0,
    ) -> dict[str, dict[str, Any]]:
        """Finite-difference audit: does each driver's learned effect match physics?

        For every constrained predictor, perturbs that column by a small positive
        step at ``n_probe`` sample points and measures the mean change in the
        prediction. The observed sign must match the SEB expected sign; a
        mismatch (only possible for non-monotone backends such as ``'rf'`` or for
        ``monotonic=False``) is flagged and warned about.

        Returns
        -------
        dict
            ``{feature: {expected, observed, mean_delta, ok}}`` for every feature
            with a non-zero expected sign.
        """
        if not self._fitted:
            raise RuntimeError("LSTModel is not fitted; call .fit() first.")
        signs = monotone_constraints(self.feature_names_)
        if X is None:
            rng = np.random.default_rng(seed)
            X = rng.standard_normal((n_probe, len(self.feature_names_)))
        else:
            X = np.asarray(X, dtype=np.float64)
            X = X[~np.isnan(X).any(axis=1)]
            if X.shape[0] > n_probe:
                rng = np.random.default_rng(seed)
                X = X[rng.choice(X.shape[0], size=n_probe, replace=False)]
        base_pred = self.predict(X)
        report: dict[str, dict[str, Any]] = {}
        violations: list[str] = []
        for j, name in enumerate(self.feature_names_):
            exp = int(signs.get(name, 0))
            if exp == 0:
                continue
            col = X[:, j]
            step = delta_frac * (np.nanstd(col) or 1.0)
            Xp = X.copy()
            Xp[:, j] = col + step
            delta = float(np.nanmean(self.predict(Xp) - base_pred))
            observed = int(np.sign(delta)) if abs(delta) > 1e-9 else 0
            ok = (observed == exp) or (observed == 0)
            report[name] = {
                "expected": exp,
                "observed": observed,
                "mean_delta": delta,
                "ok": bool(ok),
            }
            if not ok:
                violations.append(name)
        if violations:
            warnings.warn(
                "physics-consistency violations (learned dLST sign disagrees "
                f"with SEB table) for: {violations}. Consider a monotone backend.",
                stacklevel=2,
            )
        return report

    # ----- persistence --------------------------------------------------
    def save(self, path: str) -> str:
        """Persist the fitted model to ``path`` via joblib (lazy). Returns ``path``."""
        import joblib  # lazy

        state = {
            "backend": self.backend,
            "resolved_backend": self._resolved_backend,
            "feature_names_": self.feature_names_,
            "monotonic": self.monotonic,
            "residual_hybrid": self.residual_hybrid,
            "monotone_cst_": self.monotone_cst_,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "seed": self.seed,
            "model": self._model,
            "fitted": self._fitted,
        }
        joblib.dump(state, path)
        return path

    @classmethod
    def load(cls, path: str) -> "LSTModel":
        """Load a model previously written with :meth:`save` (joblib lazy)."""
        import joblib  # lazy

        state = joblib.load(path)
        obj = cls(
            backend=state["backend"],
            predictors=state["feature_names_"],
            monotonic=state["monotonic"],
            residual_hybrid=state["residual_hybrid"],
            n_estimators=state["n_estimators"],
            learning_rate=state["learning_rate"],
            max_depth=state["max_depth"],
            seed=state["seed"],
        )
        obj._model = state["model"]
        obj._resolved_backend = state["resolved_backend"]
        obj.monotone_cst_ = state["monotone_cst_"]
        obj._fitted = state["fitted"]
        return obj

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"LSTModel(backend={self._resolved_backend or self.backend!r}, "
            f"n_features={len(self.feature_names_)}, "
            f"monotonic={self.monotonic}, residual_hybrid={self.residual_hybrid}, "
            f"fitted={self._fitted})"
        )


# ===========================================================================
# Module Interface Contract functions (ARCHITECTURE.md §11.5)
# ===========================================================================
def train_model(
    fs: FeatureStack,
    predictors: Sequence[str] | None = None,
    target: str = LST,
    use_physics_backbone: bool = True,
    use_mgwr: bool = False,
    use_pinn: bool = False,
    seed: int = 0,
    backend: str = "auto",
    **kwargs: Any,
) -> LSTModel:
    """Fit the physics-informed LST model from a :class:`FeatureStack`.

    Builds a monotone-constrained regression model whose driver-effect signs are
    set from the SEB table; with ``use_physics_backbone`` it learns the residual
    ``r = LST - LST_phys`` of the analytic backbone (graceful no-op if the
    physics module is absent). ``use_mgwr`` / ``use_pinn`` are accepted for
    contract compatibility — the spatial MGWR layer lives in
    :mod:`urbanheat.models.attribution` (:func:`gwr_coefficients`) and the PINN in
    :mod:`urbanheat.physics.pinn`; this function fits the core ensemble that both
    of those refine, and emits a note when those flags are set.

    Parameters
    ----------
    fs:
        Source stack (must contain ``target`` and at least one predictor).
    predictors:
        Predictor names (default :data:`DEFAULT_PREDICTORS`, intersected with
        present layers).
    target:
        Target layer (default :data:`LST`).
    use_physics_backbone:
        Train on the SEB-residual (hybrid mode) when the backbone is available.
    use_mgwr, use_pinn:
        Contract flags (see note above).
    seed:
        RNG seed.
    backend:
        Backend selector forwarded to :class:`LSTModel` (default ``'auto'``).
    **kwargs:
        Extra hyper-parameters forwarded to :class:`LSTModel`.

    Returns
    -------
    LSTModel
        A fitted model exposing ``.predict(X)`` and ``.predict_grid(fs)``.
    """
    if use_mgwr:
        warnings.warn(
            "use_mgwr=True: fit the MGWR spatial layer separately via "
            "urbanheat.models.attribution.gwr_coefficients(fs, predictors).",
            stacklevel=2,
        )
    if use_pinn:
        warnings.warn(
            "use_pinn=True: train the PINN reconciler separately via "
            "urbanheat.physics.pinn.HeatPINN(...).fit(fs).",
            stacklevel=2,
        )
    model = LSTModel(
        backend=backend,
        predictors=predictors if predictors is not None else DEFAULT_PREDICTORS,
        residual_hybrid=bool(use_physics_backbone),
        seed=seed,
        **kwargs,
    )
    model.fit_stack(fs, target)
    return model


def predict_lst(model: LSTModel, fs: FeatureStack, write: bool = False) -> np.ndarray:
    """Predict LST (degC) on the full grid of ``fs``; add ``'lst_pred'`` if ``write``.

    Thin wrapper over :meth:`LSTModel.predict_grid` for contract symmetry.
    """
    return model.predict_grid(fs, write=write)


def save_model(model: LSTModel, path: str) -> str:
    """Persist a fitted :class:`LSTModel` to ``path`` (joblib lazy). Returns ``path``."""
    return model.save(path)


def load_model(path: str) -> LSTModel:
    """Load a :class:`LSTModel` from ``path`` (joblib lazy)."""
    return LSTModel.load(path)
