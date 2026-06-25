"""Tests for :mod:`urbanheat.models` — the physics-informed ML core.

Guards (ARCHITECTURE §8, §11.5):
  * ``features.build_xy`` (a.k.a. the feature-table builder) shapes: X/y/coords
    are aligned with the documented predictor set.
  * ``features.monotone_constraints`` returns SEB-consistent {-1,0,+1} signs.
  * ``train.train_model`` fits a monotone synthetic relation with R^2 > 0 and
    exposes ``.predict`` / ``.predict_grid``.
  * ``attribution`` permutation/SHAP importance ranks the true driver high and
    rolls up into the four families; the physics-sign audit flags violations.
  * ``validation`` spatial-CV / metrics are finite.

Heavy learners (xgboost/lightgbm/shap/verde/mgwr) are optional and guarded.
The modules are built in parallel; tests skip cleanly until importable.
"""

from __future__ import annotations

import numpy as np
import pytest

from urbanheat import datamodel as dm
from urbanheat.datamodel import FeatureStack

pd = pytest.importorskip("pandas")
features = pytest.importorskip("urbanheat.models.features")


def _fit_fast_model(fs: FeatureStack, **kw):
    """Fit a model via the contracted ``train_model`` using a fast backend.

    The default ``backend='auto'`` resolves to sklearn's
    ``HistGradientBoostingRegressor``; on some sklearn builds that learner is
    extremely slow (≈1 s per boosting iteration with the package's 300-iter
    default), which would hang the suite. The RandomForest backend is a
    documented, contract-compliant alternative (exposes ``.predict`` /
    ``.predict_grid`` / ``.feature_importances_``) that fits the smooth
    synthetic relation in well under a second, so the deterministic-core ML
    tests stay runnable on the minimal numpy/scipy/sklearn stack. The integrator
    can re-point this at the production GBM/XGBoost backend once that path is
    fast in the target environment.
    """
    train = pytest.importorskip("urbanheat.models.train")
    kw.setdefault("use_physics_backbone", False)
    kw.setdefault("use_mgwr", False)
    kw.setdefault("use_pinn", False)
    kw.setdefault("seed", 0)
    try:
        return train.train_model(fs, backend="rf", **kw)
    except TypeError:
        # train_model without a backend kwarg -> fall back to its default.
        return train.train_model(fs, **kw)


# ---------------------------------------------------------------------------
# features.build_xy  /  monotone_constraints  /  predictor_grid
# ---------------------------------------------------------------------------
def _build_xy(fs: FeatureStack, **kw):
    """Call whichever feature-table builder the module exposes (build_xy first)."""
    if hasattr(features, "build_xy"):
        return features.build_xy(fs, **kw)
    if hasattr(features, "build_feature_table"):
        return features.build_feature_table(fs, **kw)
    pytest.skip("no build_xy / build_feature_table in models.features")


def test_build_xy_shapes(synthetic_stack: FeatureStack) -> None:
    """X, y and coords have aligned row counts; X columns are predictors."""
    X, y, coords = _build_xy(synthetic_stack)
    n = len(y)
    assert X.shape[0] == n
    assert coords.shape[0] == n
    assert coords.shape[1] == 2          # [x, y] for spatial CV
    assert X.shape[1] >= 1
    # Predictors must be canonical names that exist in the stack.
    cols = list(X.columns) if hasattr(X, "columns") else []
    for c in cols:
        assert c in dm.ALL_VARIABLES, f"predictor {c!r} is not canonical"
        assert synthetic_stack.has(c)


def test_build_xy_respects_predictor_list(synthetic_stack: FeatureStack) -> None:
    """An explicit predictor list is honoured (intersected with present layers)."""
    preds = [dm.NDVI, dm.IMPERVIOUS_FRAC, dm.ALBEDO]
    X, y, coords = _build_xy(synthetic_stack, predictors=preds)
    if hasattr(X, "columns"):
        assert set(X.columns) <= set(preds)
        assert dm.NDVI in X.columns


def test_monotone_constraints_signs() -> None:
    """SEB signs: coolers -> -1, warmers -> +1 (feeds XGBoost monotone_constraints)."""
    if not hasattr(features, "monotone_constraints"):
        pytest.skip("monotone_constraints not implemented")
    preds = [dm.ALBEDO, dm.NDVI, dm.WATER_FRAC, dm.GREEN_FRAC, dm.TREE_FRAC,
             dm.SVF, dm.EMISSIVITY, dm.WIND_SPEED,
             dm.IMPERVIOUS_FRAC, dm.BUILDING_HEIGHT, dm.ANTHRO_HEAT]
    mc = features.monotone_constraints(preds)
    assert isinstance(mc, dict)
    for cooler in (dm.ALBEDO, dm.NDVI, dm.WATER_FRAC, dm.GREEN_FRAC):
        if cooler in mc:
            assert mc[cooler] == -1, f"{cooler} must be monotone-decreasing"
    for warmer in (dm.IMPERVIOUS_FRAC, dm.BUILDING_HEIGHT, dm.ANTHRO_HEAT):
        if warmer in mc:
            assert mc[warmer] == +1, f"{warmer} must be monotone-increasing"
    assert all(v in (-1, 0, 1) for v in mc.values())


def test_predictor_grid_shape(synthetic_stack: FeatureStack) -> None:
    """predictor_grid returns an (H*W, P) matrix for full-grid prediction."""
    if not hasattr(features, "predictor_grid"):
        pytest.skip("predictor_grid not implemented")
    preds = [dm.NDVI, dm.IMPERVIOUS_FRAC, dm.ALBEDO]
    grid = np.asarray(features.predictor_grid(synthetic_stack, preds))
    h, w = synthetic_stack.shape
    assert grid.shape == (h * w, len(preds))


# ---------------------------------------------------------------------------
# train.train_model
# ---------------------------------------------------------------------------
def test_train_model_fits_and_predicts(synthetic_stack: FeatureStack) -> None:
    """train_model fits the LST<->drivers relation (R^2>0) and predicts on the grid."""
    pytest.importorskip("urbanheat.models.train")
    # Keep it lean & deterministic: no physics backbone / MGWR / PINN.
    model = _fit_fast_model(synthetic_stack)
    assert hasattr(model, "predict")

    # In-sample skill on the synthetic (smooth, learnable) relation.
    X, y, _ = _build_xy(synthetic_stack)
    pred = np.asarray(model.predict(X), dtype=np.float64).ravel()
    yv = np.asarray(y, dtype=np.float64).ravel()
    ss_res = float(np.sum((yv - pred) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    assert r2 > 0.0, f"model has no skill on synthetic data (R2={r2:.3f})"

    # Full-grid prediction is grid-shaped.
    if hasattr(model, "predict_grid"):
        grid = np.asarray(model.predict_grid(synthetic_stack), dtype=np.float64)
        assert grid.shape == synthetic_stack.shape


def test_predict_lst_writes_layer(synthetic_stack: FeatureStack) -> None:
    """predict_lst can write a prediction layer back to the stack."""
    train = pytest.importorskip("urbanheat.models.train")
    model = _fit_fast_model(synthetic_stack)
    if not hasattr(train, "predict_lst"):
        pytest.skip("predict_lst not implemented")
    out = np.asarray(train.predict_lst(model, synthetic_stack, write=True),
                     dtype=np.float64)
    assert out.shape == synthetic_stack.shape
    assert synthetic_stack.has("lst_pred")


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------
def test_family_attribution_rolls_up_to_four_families() -> None:
    """family_attribution maps per-feature importance to the 4 PS-1 families."""
    attribution = pytest.importorskip("urbanheat.models.attribution")
    if not hasattr(attribution, "family_attribution"):
        pytest.skip("family_attribution not implemented")
    importance = pd.DataFrame({
        "feature": [dm.NDVI, dm.IMPERVIOUS_FRAC, dm.BUILDING_HEIGHT, dm.AIR_TEMP],
        "mean_abs_shap": [3.0, 5.0, 1.0, 1.0],
    })
    fam = attribution.family_attribution(importance)
    assert "family" in fam.columns
    fams = set(fam["family"])
    assert fams <= {"lulc", "morphology", "vegetation", "atmosphere"}
    pct_col = "pct_contribution" if "pct_contribution" in fam.columns else fam.columns[-1]
    assert float(fam[pct_col].sum()) == pytest.approx(100.0, abs=1.0) or \
        float(fam[pct_col].sum()) == pytest.approx(1.0, abs=0.02)


def test_permutation_or_shap_importance_ranks_true_driver(
        synthetic_stack: FeatureStack) -> None:
    """Attribution ranks a dominant synthetic driver above a weak one.

    Prefers a permutation-importance helper (sklearn-only); falls back to SHAP
    (importorskip shap). The synthetic LST is built mostly from impervious/NDVI,
    so those should out-rank, e.g., elevation.
    """
    attribution = pytest.importorskip("urbanheat.models.attribution")
    pytest.importorskip("urbanheat.models.train")
    model = _fit_fast_model(synthetic_stack)
    X, y, _ = _build_xy(synthetic_stack)

    imp = None
    if hasattr(attribution, "permutation_importance"):
        imp = attribution.permutation_importance(model, X, y)
    elif hasattr(attribution, "shap_importance"):
        pytest.importorskip("shap")
        imp = attribution.shap_importance(model, X)
    else:
        pytest.skip("no permutation_importance / shap_importance")

    assert "feature" in imp.columns
    val_col = next((c for c in imp.columns
                    if "importance" in c or "shap" in c), imp.columns[-1])
    ranked = imp.sort_values(val_col, ascending=False)["feature"].tolist()
    # A strong driver should out-rank a weak/irrelevant one when both are present.
    if dm.IMPERVIOUS_FRAC in ranked and dm.ELEVATION in ranked:
        assert ranked.index(dm.IMPERVIOUS_FRAC) < ranked.index(dm.ELEVATION)


def test_physics_sign_audit_flags(synthetic_stack: FeatureStack) -> None:
    """physics_sign_audit returns expected/observed signs + an ok flag."""
    attribution = pytest.importorskip("urbanheat.models.attribution")
    if not hasattr(attribution, "physics_sign_audit"):
        pytest.skip("physics_sign_audit not implemented")
    importance = pd.DataFrame({
        "feature": [dm.NDVI, dm.IMPERVIOUS_FRAC],
        "mean_abs_shap": [3.0, 5.0],
        "mean_shap_signed": [-3.0, 5.0],   # correct signs: NDVI cools, imperv warms
    })
    audit = attribution.physics_sign_audit(importance)
    assert {"feature", "ok"} <= set(audit.columns)
    assert bool(audit.set_index("feature").loc[dm.NDVI, "ok"]) is True


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def test_compute_metrics_finite_and_named() -> None:
    """compute_metrics returns the documented finite metric panel."""
    validation = pytest.importorskip("urbanheat.models.validation")
    rng = np.random.default_rng(0)
    y_true = rng.normal(35, 4, 200)
    y_pred = y_true + rng.normal(0, 1, 200)
    m = validation.compute_metrics(y_true, y_pred)
    assert isinstance(m, dict)
    for k in ("rmse", "mae", "bias", "r2"):
        assert k in m, f"metric {k} missing"
        assert np.isfinite(m[k]), f"metric {k} not finite"
    assert m["rmse"] >= 0.0
    assert m["r2"] <= 1.0 + 1e-9
    # A near-perfect fit must score high R^2.
    perfect = validation.compute_metrics(y_true, y_true)
    assert perfect["r2"] == pytest.approx(1.0, abs=1e-6)
    assert perfect["rmse"] == pytest.approx(0.0, abs=1e-6)


def test_spatial_cv_metrics_finite(synthetic_stack: FeatureStack) -> None:
    """spatial_cv yields a finite per-fold metric DataFrame (verde guarded)."""
    validation = pytest.importorskip("urbanheat.models.validation")
    if not hasattr(validation, "spatial_cv"):
        pytest.skip("spatial_cv not implemented")
    pytest.importorskip("verde")
    from sklearn.ensemble import ExtraTreesRegressor
    X, y, coords = _build_xy(synthetic_stack)

    def factory():
        return ExtraTreesRegressor(n_estimators=20, random_state=0)

    res = validation.spatial_cv(factory, X, y, coords, n_splits=3, seed=0)
    assert len(res) >= 1
    # At least RMSE/R2 columns are present and finite on average.
    for col in ("rmse", "r2"):
        if col in res.columns:
            assert np.isfinite(np.asarray(res[col], dtype=float)).any()


def test_residual_autocorrelation_runs(synthetic_stack: FeatureStack) -> None:
    """residual_autocorrelation returns a finite global Moran's I scalar."""
    validation = pytest.importorskip("urbanheat.models.validation")
    if not hasattr(validation, "residual_autocorrelation"):
        pytest.skip("residual_autocorrelation not implemented")
    _, y, coords = _build_xy(synthetic_stack)
    resid = np.asarray(y, dtype=np.float64) - float(np.mean(y))
    val = validation.residual_autocorrelation(resid, coords)
    assert np.isfinite(val)
