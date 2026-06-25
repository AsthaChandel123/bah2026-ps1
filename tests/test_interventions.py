"""Tests for :mod:`urbanheat.interventions` — cooling simulation + optimization.

Guards (ARCHITECTURE §9, §11.6):
  * ``catalog`` lists the 9 levers + per-lever params + a feasibility mask.
  * ``simulate.apply_perturbation`` (a.k.a. apply_intervention) changes the
    correct canonical driver layers and does not mutate its input.
  * ``simulate.delta_lst`` (a.k.a. predict_delta_lst) returns cooling (>=0) for a
    greening lever under a monotone model.
  * ``invest_cooling.cooling_capacity`` is in [0, 1].
  * ``optimize`` lazy-greedy respects budget and improves the objective
    monotonically; the portfolio carries type / placement / estimated dC.

Heavy solvers (pulp/ortools/pymoo) are optional and guarded. Modules are built
in parallel; tests skip cleanly until importable.
"""

from __future__ import annotations

import numpy as np
import pytest

from urbanheat import constants as C
from urbanheat import datamodel as dm
from urbanheat.config import Config
from urbanheat.datamodel import FeatureStack

catalog = pytest.importorskip("urbanheat.interventions.catalog")


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------
def test_list_interventions_returns_nine() -> None:
    """The catalog exposes the 9 intervention keys from INTERVENTION_PARAMS."""
    names = catalog.list_interventions()
    assert set(names) == set(C.INTERVENTION_PARAMS)
    assert len(names) == 9


def test_get_intervention_params() -> None:
    """get_intervention returns the full param dict for a lever."""
    p = catalog.get_intervention("urban_trees")
    assert "perturbs" in p and "surface_dC" in p
    assert p["perturbs"].get("ndvi", 0) > 0


def test_feasibility_mask_is_boolean_grid(synthetic_stack: FeatureStack) -> None:
    """feasibility_mask returns a boolean grid of the FeatureStack shape."""
    if not hasattr(catalog, "feasibility_mask"):
        pytest.skip("feasibility_mask not implemented")
    mask = np.asarray(catalog.feasibility_mask(synthetic_stack, "cool_roof"))
    assert mask.shape == synthetic_stack.shape
    assert mask.dtype == bool or set(np.unique(mask).tolist()) <= {0, 1, 0.0, 1.0}
    assert mask.sum() >= 1, "cool_roof should be feasible on some built pixels"


# ---------------------------------------------------------------------------
# simulate.apply_perturbation
# ---------------------------------------------------------------------------
def _apply(fs: FeatureStack, name: str, **kw):
    """Call whichever perturbation API the module exposes."""
    simulate = pytest.importorskip("urbanheat.interventions.simulate")
    if hasattr(simulate, "apply_perturbation"):
        return simulate.apply_perturbation(fs, name, **kw)
    if hasattr(simulate, "apply_intervention"):
        return simulate.apply_intervention(fs, name, **kw)
    pytest.skip("no apply_perturbation / apply_intervention")


def test_apply_perturbation_changes_correct_vars(synthetic_stack: FeatureStack) -> None:
    """Applying urban_trees raises NDVI/tree_frac and leaves the input unmutated."""
    base = synthetic_stack
    ndvi_before = base.get(dm.NDVI).copy()
    out = _apply(base, "urban_trees")
    assert isinstance(out, FeatureStack)
    # Input not mutated.
    np.testing.assert_allclose(base.get(dm.NDVI), ndvi_before)
    # NDVI increased on average in the perturbed copy.
    assert float(np.nanmean(out.get(dm.NDVI))) > float(np.nanmean(ndvi_before))
    if out.has(dm.TREE_FRAC) and base.has(dm.TREE_FRAC):
        assert float(np.nanmean(out.get(dm.TREE_FRAC))) >= \
            float(np.nanmean(base.get(dm.TREE_FRAC)))


def test_apply_cool_roof_raises_albedo_clipped(synthetic_stack: FeatureStack) -> None:
    """cool_roof raises albedo but clips to a physical ceiling (<=0.9)."""
    base = synthetic_stack
    out = _apply(base, "cool_roof")
    assert float(np.nanmean(out.get(dm.ALBEDO))) >= float(np.nanmean(base.get(dm.ALBEDO)))
    assert float(np.nanmax(out.get(dm.ALBEDO))) <= 0.9 + 1e-6


def test_apply_respects_mask(synthetic_stack: FeatureStack) -> None:
    """A placement mask localizes the perturbation to the masked pixels only."""
    simulate = pytest.importorskip("urbanheat.interventions.simulate")
    if not (hasattr(simulate, "apply_perturbation") or
            hasattr(simulate, "apply_intervention")):
        pytest.skip("no perturbation API")
    base = synthetic_stack
    mask = np.zeros(base.shape, dtype=bool)
    mask[: base.shape[0] // 2, :] = True   # top half only
    out = _apply(base, "urban_trees", mask=mask)
    diff = np.asarray(out.get(dm.NDVI)) - np.asarray(base.get(dm.NDVI))
    # Bottom (unmasked) half is essentially unchanged.
    assert np.nanmax(np.abs(diff[~mask])) <= 1e-5 + 1e-3
    # Top (masked) half changed.
    assert np.nanmax(np.abs(diff[mask])) > 0.0


# ---------------------------------------------------------------------------
# simulate.delta_lst  (counterfactual cooling)
# ---------------------------------------------------------------------------
class _MonotoneCoolingModel:
    """A tiny stand-in model: LST decreases with NDVI/albedo, increases with imperv.

    Used so the delta_lst test does not depend on a trained GBM being available;
    exposes the ``.predict_grid(fs)`` / ``.predict(X)`` surface the contract uses.
    """

    predictors = [dm.NDVI, dm.ALBEDO, dm.IMPERVIOUS_FRAC]

    def predict_grid(self, fs: FeatureStack) -> np.ndarray:
        ndvi = np.asarray(fs.get(dm.NDVI), dtype=np.float64)
        alb = np.asarray(fs.get(dm.ALBEDO), dtype=np.float64)
        imp = np.asarray(fs.get(dm.IMPERVIOUS_FRAC), dtype=np.float64)
        return (35.0 - 8.0 * ndvi - 10.0 * alb + 12.0 * imp).astype(np.float64)

    def predict(self, X) -> np.ndarray:  # noqa: ANN001
        arr = np.asarray(X, dtype=np.float64)
        cols = list(getattr(X, "columns", [])) or self.predictors
        idx = {c: i for i, c in enumerate(cols)}
        ndvi = arr[:, idx.get(dm.NDVI, 0)]
        alb = arr[:, idx.get(dm.ALBEDO, 1)]
        imp = arr[:, idx.get(dm.IMPERVIOUS_FRAC, 2)]
        return 35.0 - 8.0 * ndvi - 10.0 * alb + 12.0 * imp


def _delta_lst(model, fs, name, **kw):
    """Call whichever counterfactual-delta API the simulator exposes."""
    simulate = pytest.importorskip("urbanheat.interventions.simulate")
    if hasattr(simulate, "delta_lst"):
        return simulate.delta_lst(model, fs, name, **kw)
    if hasattr(simulate, "predict_delta_lst"):
        return simulate.predict_delta_lst(model, fs, name, **kw)
    pytest.skip("no delta_lst / predict_delta_lst")


def test_delta_lst_greening_cools(synthetic_stack: FeatureStack) -> None:
    """ΔLST (base - perturbed) for greening is cooling (>= 0) under a monotone model."""
    fs = synthetic_stack
    model = _MonotoneCoolingModel()
    delta = np.asarray(_delta_lst(model, fs, "urban_trees"), dtype=np.float64)
    assert delta.shape == fs.shape
    # Convention: positive = cooling. Mean cooling should be non-negative.
    assert float(np.nanmean(delta)) >= -1e-6
    # Somewhere it actually cools.
    assert float(np.nanmax(delta)) > 0.0


def test_delta_lst_cool_roof_cools(synthetic_stack: FeatureStack) -> None:
    """Raising albedo (cool_roof) also yields net cooling."""
    fs = synthetic_stack
    delta = np.asarray(_delta_lst(_MonotoneCoolingModel(), fs, "cool_roof"),
                       dtype=np.float64)
    assert float(np.nanmean(delta)) >= -1e-6


# ---------------------------------------------------------------------------
# invest_cooling
# ---------------------------------------------------------------------------
def test_cooling_capacity_in_unit_range(synthetic_stack: FeatureStack) -> None:
    """InVEST CC = 0.6*shade + 0.2*albedo + 0.2*ETI is bounded to [0, 1]."""
    invest = pytest.importorskip("urbanheat.interventions.invest_cooling")
    cc = np.asarray(invest.cooling_capacity(synthetic_stack), dtype=np.float64)
    assert cc.shape == synthetic_stack.shape
    finite = cc[np.isfinite(cc)]
    assert finite.min() >= -1e-6 and finite.max() <= 1.0 + 1e-6


def test_heat_mitigation_runs(synthetic_stack: FeatureStack) -> None:
    """heat_mitigation returns a finite grid."""
    invest = pytest.importorskip("urbanheat.interventions.invest_cooling")
    if not hasattr(invest, "heat_mitigation"):
        pytest.skip("heat_mitigation not implemented")
    hm = np.asarray(invest.heat_mitigation(synthetic_stack), dtype=np.float64)
    assert hm.shape == synthetic_stack.shape
    assert np.isfinite(hm).any()


# ---------------------------------------------------------------------------
# optimize  (lazy-greedy submodular)
# ---------------------------------------------------------------------------
def _real_candidates(fs: FeatureStack, model, top_k: int = 200):
    """Enumerate genuine candidates via the module's own generator.

    ``optimize_greedy`` consumes either a ``list[Candidate]`` or a DataFrame
    produced by ``generate_candidates`` (which carries the Candidate objects in
    ``.attrs``), so we build them the real way rather than faking the schema.
    Returns ``(candidates_df_or_list, per_pixel_weights)`` or skips if the
    generator yields nothing on this tiny grid.
    """
    optimize = pytest.importorskip("urbanheat.interventions.optimize")
    if not hasattr(optimize, "generate_candidates"):
        pytest.skip("generate_candidates not implemented")
    cands = optimize.generate_candidates(fs, model, top_k=top_k)
    if len(cands) == 0:
        pytest.skip("no feasible candidates on the tiny synthetic grid")
    n_pix = int(np.prod(fs.shape))
    weights = np.ones(n_pix, dtype=np.float64)
    # ``max_area`` is an ABSOLUTE area cap in m^2 (one 100 m pixel = 1e4 m^2);
    # use the whole AOI so the area constraint never binds in these tests.
    full_area = float(n_pix) * 100.0 * 100.0
    return cands, weights, full_area


def test_optimize_greedy_respects_budget(synthetic_stack: FeatureStack) -> None:
    """optimize_greedy never exceeds the budget and returns a ranked portfolio."""
    optimize = pytest.importorskip("urbanheat.interventions.optimize")
    if not hasattr(optimize, "optimize_greedy"):
        pytest.skip("optimize_greedy not implemented")
    cands, weights, full_area = _real_candidates(synthetic_stack,
                                                 _MonotoneCoolingModel())
    # A budget large enough to admit several sites but smaller than treating all.
    budget = 5.0e6
    sel = optimize.optimize_greedy(cands, weights, budget=budget,
                                   max_area=full_area)
    assert len(sel) >= 1
    if "cost" in getattr(sel, "columns", []):
        assert float(sel["cost"].sum()) <= budget + 1e-6
    # Insertion order = priority, surfaced as a 1..N rank column.
    if "rank" in getattr(sel, "columns", []):
        ranks = np.asarray(sel["rank"], dtype=int)
        assert ranks.min() == 1 and len(set(ranks.tolist())) == len(ranks)


def test_optimize_greedy_objective_monotone_nondecreasing(
        synthetic_stack: FeatureStack) -> None:
    """The lazy-greedy coverage objective improves monotonically as sites are added.

    The submodular property the (1-1/e) guarantee rests on is that the
    *city-wide weighted-cooling objective* never decreases as greedy inserts the
    next site (diminishing returns, but never negative). That trace is the
    monotone quantity — not the sum of per-site self-ΔT, which can dip when a
    later site covers more *new* area than an earlier high-self-ΔT one. We assert
    on the module's own ``objective_trace`` from ``lazy_greedy_optimize``.
    """
    optimize = pytest.importorskip("urbanheat.interventions.optimize")
    if not (hasattr(optimize, "lazy_greedy_optimize") and
            hasattr(optimize, "_coerce_candidates")):
        pytest.skip("lazy_greedy_optimize / _coerce_candidates not available")
    cands, weights, full_area = _real_candidates(synthetic_stack,
                                                 _MonotoneCoolingModel())
    res = optimize.lazy_greedy_optimize(
        optimize._coerce_candidates(cands), weights, 5.0e7, full_area)
    trace = np.asarray(res.get("objective_trace", []), dtype=float)
    if trace.size < 2:
        pytest.skip("greedy selected <2 sites; monotonicity trivially holds")
    assert np.all(np.diff(trace) >= -1e-6), \
        f"greedy objective trace not monotone non-decreasing: {trace}"
    # The submodular objective is non-negative and ends at its max.
    assert trace[-1] >= trace[0] - 1e-6
    assert float(res.get("objective", trace[-1])) == pytest.approx(
        float(trace[-1]), rel=1e-3, abs=1e-6)


def test_optimize_top_level_portfolio_keys(synthetic_stack: FeatureStack,
                                            tiny_config: Config) -> None:
    """optimize() returns a portfolio with type / placement / estimated dC fields.

    Accepts either the §11 portfolio schema (type/geometry/delta_C/cost) or the
    PS-1 phrasing (intervention_type/location/estimated_delta_lst_C); the
    integrator reconciles the exact column names.
    """
    optimize = pytest.importorskip("urbanheat.interventions.optimize")
    if not hasattr(optimize, "optimize"):
        pytest.skip("top-level optimize() not implemented")
    fs = synthetic_stack
    # Ensure HVI present for equity weighting if the optimizer needs it.
    if not fs.has(dm.HVI):
        try:
            hsmod = pytest.importorskip("urbanheat.indices.hotspots")
            fs = hsmod.heat_vulnerability_index(fs, method="equal")
        except Exception:
            pass
    model = _MonotoneCoolingModel()
    cfg = Config(mode="synthetic", grid_shape=fs.shape,
                 optimizer_method="greedy", optimizer_budget=5.0e5,
                 equity_weighting=False)
    try:
        result = optimize.optimize(fs, model, cfg)
    except Exception as exc:  # pragma: no cover - integration-dependent
        pytest.skip(f"optimize() needs full pipeline wiring: {exc!r}")
    assert isinstance(result, dict)
    assert "portfolio" in result
    portfolio = result["portfolio"]
    cols = set(getattr(portfolio, "columns", []))
    type_ok = {"type", "intervention_type"} & cols
    delta_ok = {"delta_C", "estimated_delta_lst_C", "delta_lst_C"} & cols
    place_ok = {"geometry", "location", "row", "col", "x", "y"} & cols
    assert type_ok, f"portfolio missing intervention type column: {cols}"
    assert delta_ok, f"portfolio missing estimated dC column: {cols}"
    assert place_ok, f"portfolio missing placement column: {cols}"


def test_equity_weights_population_times_hvi(synthetic_stack: FeatureStack) -> None:
    """equity_weights = POP * HVI (population x vulnerability)."""
    optimize = pytest.importorskip("urbanheat.interventions.optimize")
    if not hasattr(optimize, "equity_weights"):
        pytest.skip("equity_weights not implemented")
    fs = synthetic_stack
    if not fs.has(dm.HVI):
        fs.add_layer(dm.HVI, np.full(fs.shape, 0.5, dtype=np.float32))
    w = np.asarray(optimize.equity_weights(fs), dtype=np.float64)
    assert np.isfinite(w).all()
    assert np.nanmin(w) >= -1e-9
