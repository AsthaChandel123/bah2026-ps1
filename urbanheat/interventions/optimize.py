"""urbanheat.interventions.optimize — the spatial intervention OPTIMIZER.

This is the PS-1 core deliverable: given the per-pixel driver
:class:`~urbanheat.datamodel.FeatureStack`, the trained LST model and a
:class:`~urbanheat.config.Config`, decide **which** cooling intervention to place
**where** to maximise (equity-weighted) cooling under a budget + treatable-area
constraint, and report the **estimated °C reduction** of the chosen portfolio.

Pipeline (research/06 §7, ARCHITECTURE §9/§11.6):

  Stage A  candidate generation — feasible ``(pixel, intervention)`` pairs behind
           the per-type feasibility mask, pre-filtered to the hottest / highest-
           priority sites and capped to ``top_k`` to bound the problem.
  Stage B  marginal-ΔT oracle — each candidate perturbs drivers →
           :func:`urbanheat.interventions.simulate.delta_lst` (counterfactual,
           with ``exp(−d/d_cool)`` decay), benefit per pixel weighted by
           ``w_p = POP_p·HVI_p`` (equity). Effects are NON-additive: overlapping
           cooling on a pixel saturates (serial absorption), which makes the
           coverage objective **monotone submodular**.
  Stage C  optimise — three interchangeable solvers:
             * :func:`optimize_greedy` — lazy-greedy (CELF) submodular max,
               ``(1−1/e)≈0.63`` guarantee, ranked placement list (PRIMARY);
             * :func:`optimize_ilp`    — exact ILP cross-check (PuLP/OR-Tools,
               lazy; numpy-greedy fallback);
             * :func:`optimize_nsga2`  — NSGA-II Pareto front over
               {cooling, −cost, equity} (pymoo, lazy; optional).
  Stage D  output — per selected site: intervention TYPE, PLACEMENT (row/col +
           lon/lat), area, cost, estimated ΔLST (°C) + estimated Δair (°C), with a
           literature-range sanity flag; plus city-wide totals and a placement
           raster — the "optimal intervention strategy".

All heavy/optional libs (pulp, ortools, pymoo, pandas, geopandas) are imported
**lazily inside functions**, each with a pure-numpy fallback, so the optimizer
ALWAYS runs (greedy) on numpy alone. The candidate table is a list of light
dataclasses; ``pandas``/``geopandas`` are only touched when the caller asks for a
DataFrame / GeoDataFrame.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from urbanheat import constants as C
from urbanheat import datamodel as dm
from urbanheat.interventions import catalog as cat
from urbanheat.interventions import invest_cooling as invest
from urbanheat.interventions import simulate as sim


# ===========================================================================
# Candidate representation
# ===========================================================================
@dataclass
class Candidate:
    """One feasible ``(pixel, intervention)`` decision and its cooling footprint.

    Attributes
    ----------
    site_id : int
        Flattened pixel index ``row*W + col`` of the treated cell.
    row, col : int
        Pixel coordinates of the treated cell.
    intervention : str
        Intervention type name (catalog key).
    cost : float
        Cost to deploy (currency) = ``cost_per_m2 · treated_area_m2``.
    area_m2 : float
        Treated ground area (one pixel).
    self_delta_c : float
        Counterfactual surface ΔLST (°C) at the treated cell (positive=cooling).
    foot_idx : np.ndarray
        Flattened pixel indices reached by this candidate's cooling footprint
        (treated cell + exp-decay buffer).
    foot_delta : np.ndarray
        Per-pixel surface ΔLST (°C) delivered to ``foot_idx`` (decay-weighted).
    """

    site_id: int
    row: int
    col: int
    intervention: str
    cost: float
    area_m2: float
    self_delta_c: float
    foot_idx: np.ndarray
    foot_delta: np.ndarray


def equity_weights(fs: dm.FeatureStack) -> np.ndarray:
    """``w_p = POP_p · HVI_p`` (population × heat vulnerability), flattened.

    Reads POPULATION and HVI; missing layers default to uniform (population 1,
    HVI 1) so the optimizer still runs and degrades to pure-cooling weighting.
    Normalised to mean 1 so budgets/objective magnitudes are interpretable.
    [R6 §8]
    """
    pop = _flat(fs, dm.POPULATION, default=1.0)
    hvi = _flat(fs, dm.HVI, default=1.0)
    pop = np.clip(pop, 0.0, None)
    hvi = np.clip(hvi, 0.0, None)
    w = pop * hvi
    mean = float(np.nanmean(w))
    if mean > 0 and np.isfinite(mean):
        w = w / mean
    else:
        w = np.ones_like(w)
    return np.where(np.isfinite(w), w, 0.0)


def priority_score(fs: dm.FeatureStack) -> np.ndarray:
    """Coarse per-pixel priority (flattened) used to pre-rank candidate sites.

    Combines heat (baseline LST percentile / hotspot mask) with equity weight so
    the candidate pre-filter keeps the hottest, most-vulnerable pixels. Uses the
    PRIORITY_SCORE layer if the indices module already wrote one. [R6 §7.3 / §8]
    """
    if fs.has(dm.PRIORITY_SCORE):
        return _flat(fs, dm.PRIORITY_SCORE, 0.0)

    w = equity_weights(fs)
    if fs.has(dm.LST):
        lst = _flat(fs, dm.LST, np.nan)
        finite = np.isfinite(lst)
        heat = np.zeros_like(lst)
        if finite.any():
            lo = np.nanpercentile(lst[finite], 5)
            hi = np.nanpercentile(lst[finite], 95)
            if hi > lo:
                heat = np.clip((lst - lo) / (hi - lo), 0.0, 1.0)
    else:
        heat = np.ones_like(w)
    return heat * w


# ===========================================================================
# Stage A + B — candidate generation with cooling footprints
# ===========================================================================
def build_candidates(
    fs: dm.FeatureStack,
    model: Any,
    interventions: list[str] | None = None,
    top_k: int = 2000,
    d_cool_m: float | None = None,
    foot_threshold: float = 0.05,
) -> list[Candidate]:
    """Enumerate feasible ``(pixel, intervention)`` candidates with ΔLST footprints.

    For each intervention type:
      1. compute its feasibility mask (catalog) and its full-grid decayed ΔLST
         field once via :func:`simulate.delta_lst` (Stage B oracle);
      2. for every feasible pixel, build a :class:`Candidate` whose footprint is
         the local exp-decay buffer of cooling around that pixel.

    Candidates are pre-ranked by ``priority_score × self_delta`` and capped to
    ``top_k`` to bound the optimisation (Stage A pre-filter). [R6 Stage A-B]
    """
    if interventions is None:
        interventions = cat.list_interventions()
    if d_cool_m is None:
        d_cool_m = float(C.INVEST_UCM["green_area_cooling_distance_m"])

    H, W = fs.shape
    res_m = _resolution_m(fs)
    area_m2 = res_m * res_m
    prio = priority_score(fs)  # flattened (H*W,)

    # precompute the exp-decay footprint kernel (offsets + weights) once
    d_px = max(d_cool_m / max(res_m, 1e-6), 1e-6)
    offsets, kweights = _decay_offsets(d_px, foot_threshold)

    cands: list[Candidate] = []
    for name in interventions:
        obj = cat.get_intervention_obj(name)
        mask = cat.feasibility_mask(fs, name)
        if not mask.any():
            continue

        # Stage B oracle: per-type counterfactual ΔLST (treated-cell magnitudes).
        # We perturb on the *full* feasibility mask once and read each treated
        # pixel's own ΔLST; the per-candidate footprint is then synthesised from
        # the decay kernel so we do not re-predict per pixel (O(types) predicts).
        perturbed = sim.apply_perturbation(fs, name, mask)
        raw = sim.predict_delta_lst(model, fs, perturbed, clip=True)  # 2-D °C
        raw = np.asarray(raw, dtype=np.float64)
        self_delta = np.where(mask, raw, 0.0)

        cost_per_unit = obj.cost_per_m2 * area_m2

        rows, cols = np.nonzero(mask)
        for r, c in zip(rows.tolist(), cols.tolist()):
            sd = float(self_delta[r, c])
            if sd <= 1e-3:
                continue  # no useful cooling here
            foot_idx, foot_delta = _footprint(r, c, sd, offsets, kweights, H, W)
            cands.append(Candidate(
                site_id=r * W + c, row=r, col=c, intervention=name,
                cost=cost_per_unit, area_m2=area_m2, self_delta_c=sd,
                foot_idx=foot_idx, foot_delta=foot_delta,
            ))

    # Stage A pre-filter: rank by priority(site) × self cooling, keep top_k.
    if len(cands) > top_k:
        keys = np.array([prio[cd.site_id] * cd.self_delta_c for cd in cands])
        keep = np.argsort(-keys)[:top_k]
        cands = [cands[i] for i in keep.tolist()]
    return cands


def generate_candidates(
    fs: dm.FeatureStack,
    model: Any,
    top_k: int = 2000,
):
    """§11.6 candidate enumeration, returned as a ``pandas.DataFrame``.

    Thin wrapper over :func:`build_candidates`; builds the DataFrame lazily so the
    numpy-only path never imports pandas. Columns: ``site_id, row, col,
    intervention, cost, area_m2, self_delta_c``. [R6 Stage A]
    """
    cands = build_candidates(fs, model, top_k=top_k)
    return _candidates_to_df(cands)


# ===========================================================================
# Stage C.1 — lazy-greedy (CELF) submodular maximisation  [PRIMARY]
# ===========================================================================
def lazy_greedy_optimize(
    candidates: list[Candidate],
    weights: np.ndarray,
    budget: float,
    area_cap: float,
    cap_c: float = 8.0,
) -> dict[str, Any]:
    """CELF lazy-greedy submodular maximisation of ``Σ_p w_p·ΔT_p(S)``.

    The portfolio's per-pixel cooling combines by **serial absorption**
    ``ΔT_p(S) = cap·(1 − Π_{cand∈S, p∈cand}(1 − δ_{cand,p}/cap))`` so overlapping
    interventions on the same pixel give diminishing returns — making the
    objective monotone submodular and the greedy add carry the
    ``(1−1/e)≈0.632`` guarantee (``constants.SUBMODULAR_GREEDY_BOUND``).

    CELF: keep a max-heap of stale marginal gains; pop the top, recompute its gain
    against the current absorbed field, and accept it if it is still the best
    (lazy evaluation) and fits budget + area + one-intervention-per-site. This is
    O(1) amortised to add the next site and yields a RANKED list (insertion order
    = priority).

    Parameters
    ----------
    cap_c : float
        Per-pixel cooling saturation cap (°C); the serial-absorption ceiling that
        encodes diminishing returns.

    Returns
    -------
    dict with keys ``selected`` (list[Candidate], ranked), ``objective`` (float,
    Σ w·ΔT), ``cost`` (float), ``area`` (float), ``absorbed`` (np.ndarray, flat
    per-pixel cooling of the final portfolio), ``objective_trace`` (monotone
    non-decreasing list of the running objective). [R6 §7.2-7.3]
    """
    weights = np.asarray(weights, dtype=np.float64)
    n_pix = weights.size
    cap_c = float(cap_c)

    # "remaining headroom" per pixel for serial absorption: prod(1 - d/cap).
    # absorbed_T_p = cap*(1 - residual_p); start residual=1 (no cooling).
    residual = np.ones(n_pix, dtype=np.float64)

    def cand_gain(cd: Candidate) -> float:
        # marginal Σ w·ΔΔT if this candidate is added given current residual
        idx = cd.foot_idx
        d = cd.foot_delta
        # new residual at those pixels
        factor = np.clip(1.0 - d / cap_c, 0.0, 1.0)
        new_res = residual[idx] * factor
        # ΔΔT = cap*( (1-new_res) - (1-old_res) ) = cap*(old_res - new_res)
        d_cool = cap_c * (residual[idx] - new_res)
        return float(np.dot(weights[idx], d_cool))

    # initialise the lazy heap with each candidate's standalone gain
    heap: list[tuple[float, int, int]] = []  # (-gain, staleness_iter, cand_index)
    for i, cd in enumerate(candidates):
        g = cand_gain(cd)
        if g > 0:
            heapq.heappush(heap, (-g, 0, i))

    selected: list[Candidate] = []
    used_sites: set[int] = set()
    spent = 0.0
    area_used = 0.0
    objective = 0.0
    trace: list[float] = []
    cur_iter = 0

    while heap:
        neg_g, stale_iter, i = heapq.heappop(heap)
        cd = candidates[i]
        # skip if this site already has an intervention (one-per-site)
        if cd.site_id in used_sites:
            continue
        # budget / area feasibility
        if spent + cd.cost > budget + 1e-9:
            continue
        if area_used + cd.area_m2 > area_cap + 1e-9:
            continue

        if stale_iter == cur_iter:
            # gain is fresh -> accept this candidate
            idx = cd.foot_idx
            factor = np.clip(1.0 - cd.foot_delta / cap_c, 0.0, 1.0)
            residual[idx] = residual[idx] * factor
            selected.append(cd)
            used_sites.add(cd.site_id)
            spent += cd.cost
            area_used += cd.area_m2
            objective += -neg_g
            trace.append(objective)
            cur_iter += 1
        else:
            # stale -> recompute against current residual and reinsert
            g = cand_gain(cd)
            if g > 0:
                heapq.heappush(heap, (-g, cur_iter, i))

    absorbed = cap_c * (1.0 - residual)
    return {
        "selected": selected,
        "objective": objective,
        "cost": spent,
        "area": area_used,
        "absorbed": absorbed,
        "objective_trace": trace,
        "cap_c": cap_c,
    }


def optimize_greedy(
    candidates,
    weights: np.ndarray,
    budget: float,
    max_area: float,
):
    """§11.6 lazy-greedy entry returning a ranked-portfolio ``pandas.DataFrame``.

    Accepts a candidate list or DataFrame; returns a ranked portfolio DataFrame
    (insertion order = priority). [R6 §7.2-7.3]
    """
    cands = _coerce_candidates(candidates)
    res = lazy_greedy_optimize(cands, weights, budget, max_area)
    return _selection_to_df(res["selected"], res["absorbed"], weights, res["cap_c"])


# ===========================================================================
# Stage C.2 — ILP exact cross-check (PuLP / OR-Tools lazy; numpy fallback)
# ===========================================================================
def ilp_optimize(
    candidates: list[Candidate],
    weights: np.ndarray,
    budget: float,
    area_cap: float,
    cap_c: float = 8.0,
    time_limit_s: float = 20.0,
) -> dict[str, Any]:
    """Exact-ish ILP cross-check of the greedy portfolio.

    Uses a **linearised** objective: each candidate's standalone weighted cooling
    ``v_i = Σ_p w_p·min(δ_{i,p}, cap)`` as its value, subject to budget, area and
    one-intervention-per-site constraints (a multi-dimensional knapsack /
    facility-location). Solved with PuLP (CBC) or OR-Tools CP-SAT if importable,
    else a deterministic **numpy greedy-by-density** fallback so this ALWAYS
    returns. The non-additive overlap is handled afterwards by re-absorbing the
    selected set, so the reported objective matches the greedy combiner.

    Returns the same dict shape as :func:`lazy_greedy_optimize`. [R6 §7.3]
    """
    weights = np.asarray(weights, dtype=np.float64)
    cap_c = float(cap_c)
    n = len(candidates)
    if n == 0:
        return _empty_result(weights.size, cap_c)

    values = np.array([_standalone_value(cd, weights, cap_c) for cd in candidates])
    costs = np.array([cd.cost for cd in candidates])
    areas = np.array([cd.area_m2 for cd in candidates])
    sites = np.array([cd.site_id for cd in candidates])

    chosen: list[int] | None = None

    # ----- try PuLP -----
    try:
        chosen = _ilp_pulp(values, costs, areas, sites, budget, area_cap, time_limit_s)
    except Exception:
        chosen = None

    # ----- try OR-Tools CP-SAT -----
    if chosen is None:
        try:
            chosen = _ilp_ortools(values, costs, areas, sites, budget, area_cap, time_limit_s)
        except Exception:
            chosen = None

    # ----- numpy greedy-by-density fallback -----
    if chosen is None:
        chosen = _knapsack_greedy_fallback(values, costs, areas, sites, budget, area_cap)

    selected = [candidates[i] for i in chosen]
    absorbed, objective = _absorb(selected, weights, cap_c)
    return {
        "selected": selected,
        "objective": objective,
        "cost": float(sum(c.cost for c in selected)),
        "area": float(sum(c.area_m2 for c in selected)),
        "absorbed": absorbed,
        "objective_trace": [objective],
        "cap_c": cap_c,
        "linear_value": float(values[chosen].sum()) if chosen else 0.0,
    }


def optimize_ilp(candidates, weights: np.ndarray, budget: float, max_area: float):
    """§11.6 ILP entry returning a portfolio ``pandas.DataFrame``."""
    cands = _coerce_candidates(candidates)
    res = ilp_optimize(cands, weights, budget, max_area)
    return _selection_to_df(res["selected"], res["absorbed"], weights, res["cap_c"])


def _ilp_pulp(values, costs, areas, sites, budget, area_cap, time_limit_s):  # pragma: no cover
    import pulp  # lazy

    prob = pulp.LpProblem("intervention_placement", pulp.LpMaximize)
    n = len(values)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    prob += pulp.lpSum(values[i] * x[i] for i in range(n))
    prob += pulp.lpSum(costs[i] * x[i] for i in range(n)) <= budget
    prob += pulp.lpSum(areas[i] * x[i] for i in range(n)) <= area_cap
    # one intervention per site
    site_to_vars: dict[int, list[int]] = {}
    for i, s in enumerate(sites.tolist()):
        site_to_vars.setdefault(s, []).append(i)
    for s, vs in site_to_vars.items():
        if len(vs) > 1:
            prob += pulp.lpSum(x[i] for i in vs) <= 1
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=int(max(time_limit_s, 1)))
    prob.solve(solver)
    return [i for i in range(n) if x[i].value() is not None and x[i].value() > 0.5]


def _ilp_ortools(values, costs, areas, sites, budget, area_cap, time_limit_s):  # pragma: no cover
    from ortools.sat.python import cp_model  # lazy

    model = cp_model.CpModel()
    n = len(values)
    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
    scale = 1000.0  # CP-SAT needs integer coefficients
    model.Add(sum(int(costs[i] * scale) * x[i] for i in range(n)) <= int(budget * scale))
    model.Add(sum(int(areas[i] * scale) * x[i] for i in range(n)) <= int(area_cap * scale))
    site_to_vars: dict[int, list[int]] = {}
    for i, s in enumerate(sites.tolist()):
        site_to_vars.setdefault(s, []).append(i)
    for s, vs in site_to_vars.items():
        if len(vs) > 1:
            model.Add(sum(x[i] for i in vs) <= 1)
    model.Maximize(sum(int(values[i] * scale) * x[i] for i in range(n)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.Solve(model)
    return [i for i in range(n) if solver.Value(x[i]) > 0.5]


def _knapsack_greedy_fallback(values, costs, areas, sites, budget, area_cap):
    """Deterministic value-density knapsack greedy (numpy-only ILP fallback)."""
    density = values / np.maximum(costs, 1e-9)
    order = np.argsort(-density)
    chosen: list[int] = []
    used_sites: set[int] = set()
    spent = 0.0
    area = 0.0
    for i in order.tolist():
        if sites[i] in used_sites:
            continue
        if values[i] <= 0:
            continue
        if spent + costs[i] > budget + 1e-9 or area + areas[i] > area_cap + 1e-9:
            continue
        chosen.append(i)
        used_sites.add(int(sites[i]))
        spent += costs[i]
        area += areas[i]
    return chosen


# ===========================================================================
# Stage C.3 — NSGA-II Pareto front (pymoo lazy; optional)
# ===========================================================================
def nsga2_optimize(
    candidates: list[Candidate],
    weights: np.ndarray,
    budget: float,
    area_cap: float,
    n_gen: int = 60,
    pop_size: int = 60,
    cap_c: float = 8.0,
    seed: int = 0,
) -> dict[str, Any]:
    """NSGA-II Pareto front over ``{maximise cooling, minimise cost, maximise equity}``.

    Each genome is a binary mask over candidates (one-intervention-per-site and
    budget/area enforced by repair + penalty). Objectives:
      f1 = −Σ_p ΔT_p            (total cooling; minimise negative)
      f2 = Σ cost               (cost; minimise)
      f3 = −Σ_p w_p·ΔT_p        (equity-weighted cooling; minimise negative)
    Uses ``pymoo`` if importable, else returns a small front built from the
    greedy solution at several budget fractions (numpy fallback) so a Pareto-style
    result is always available. [R6 §7.3]
    """
    weights = np.asarray(weights, dtype=np.float64)
    cap_c = float(cap_c)
    if len(candidates) == 0:
        return {"front": [], "method": "empty"}

    try:  # pragma: no cover - only when pymoo installed
        front = _nsga2_pymoo(candidates, weights, budget, area_cap, n_gen, pop_size, cap_c, seed)
        return {"front": front, "method": "pymoo"}
    except Exception:
        # numpy fallback: sweep budget fractions through greedy -> a cost/cooling
        # trade-off curve (a practical Pareto approximation).
        front = []
        for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
            res = lazy_greedy_optimize(candidates, weights, budget * frac, area_cap, cap_c)
            absorbed = res["absorbed"]
            front.append({
                "budget_frac": frac,
                "selected": res["selected"],
                "cost": res["cost"],
                "cooling_sum": float(np.sum(absorbed)),
                "equity_objective": res["objective"],
                "n_sites": len(res["selected"]),
            })
        return {"front": front, "method": "greedy-sweep"}


def _nsga2_pymoo(candidates, weights, budget, area_cap, n_gen, pop_size, cap_c, seed):  # pragma: no cover
    import numpy as _np
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation
    from pymoo.operators.sampling.rnd import BinaryRandomSampling
    from pymoo.optimize import minimize

    costs = _np.array([c.cost for c in candidates])
    areas = _np.array([c.area_m2 for c in candidates])
    sites = _np.array([c.site_id for c in candidates])
    n = len(candidates)

    def evaluate_mask(mask):
        sel = [candidates[i] for i in range(n) if mask[i]]
        absorbed, obj = _absorb(sel, weights, cap_c)
        cost = float(sum(c.cost for c in sel))
        cooling = float(_np.sum(absorbed))
        pen = 0.0
        if cost > budget:
            pen += (cost - budget)
        a = float(sum(c.area_m2 for c in sel))
        if a > area_cap:
            pen += (a - area_cap) * 1e3
        return -cooling + pen, cost + pen, -obj + pen

    class _P(Problem):
        def __init__(self):
            super().__init__(n_var=n, n_obj=3, n_constr=0, xl=0, xu=1, vtype=bool)

        def _evaluate(self, X, out, *a, **k):
            F = _np.zeros((X.shape[0], 3))
            for r in range(X.shape[0]):
                F[r] = evaluate_mask(X[r].astype(bool))
            out["F"] = F

    algo = NSGA2(pop_size=pop_size, sampling=BinaryRandomSampling(),
                 crossover=TwoPointCrossover(), mutation=BitflipMutation())
    res = minimize(_P(), algo, ("n_gen", n_gen), seed=seed, verbose=False)
    front = []
    X = _np.atleast_2d(res.X)
    F = _np.atleast_2d(res.F)
    for r in range(X.shape[0]):
        mask = X[r].astype(bool)
        sel = [candidates[i] for i in range(n) if mask[i]]
        front.append({
            "selected": sel,
            "cost": float(-0 + F[r, 1]),
            "cooling_sum": float(-F[r, 0]),
            "equity_objective": float(-F[r, 2]),
            "n_sites": len(sel),
        })
    return front


# ===========================================================================
# Stage D — high-level entry points
# ===========================================================================
def optimize_interventions(
    stack: dm.FeatureStack,
    model: Any,
    config: Any,
) -> dict[str, Any]:
    """High-level optimizer → the structured PS-1 "optimal intervention strategy".

    Generates candidates, computes equity weights (if ``config.equity_weighting``),
    dispatches on ``config.optimizer_method`` (``'greedy'`` | ``'ilp'`` | ``'nsga2'``),
    then assembles the per-site placement records + city-wide totals + a placement
    raster.

    Returns a dict (JSON/GeoJSON-ready) with keys:

    ``method`` : str — solver actually used.
    ``placements`` : list[dict] — one per selected site, each:
        ``{intervention_type, row, col, lon, lat, area_m2, cost,
           estimated_delta_lst_C, estimated_delta_air_C, within_literature_range}``.
        Ranked by insertion priority (greedy) or value (ilp).
    ``totals`` : dict —
        ``{n_sites, budget, budget_used, area_treated_m2, area_cap_m2,
           total_cost, mean_delta_lst_C, max_delta_lst_C, hotspot_delta_lst_C,
           population_weighted_exposure_reduction, objective,
           submodular_bound}``.
    ``placement_raster`` : np.ndarray (H, W) — delivered surface ΔLST (°C) of the
        whole portfolio (the placement map for viz/report/app).
    ``delta_air_raster`` : np.ndarray (H, W) — InVEST air-temperature reduction
        of the treated scenario (the human-relevant ΔT cross-check).
    ``candidates_considered`` : int.
    ``per_type_counts`` : dict[str, int].
    [R6 Stage C-D / ARCHITECTURE §11.6]
    """
    H, W = stack.shape
    n_pix = H * W

    # --- config knobs (defensive defaults if a bare object is passed) ---
    method = str(getattr(config, "optimizer_method", "greedy"))
    budget = float(getattr(config, "optimizer_budget", 1.0e7))
    max_area_frac = float(getattr(config, "optimizer_max_area_frac", 0.30))
    equity_on = bool(getattr(config, "equity_weighting", True))
    res_m = _resolution_m(stack)
    aoi_area_m2 = n_pix * res_m * res_m
    area_cap = max_area_frac * aoi_area_m2

    # --- weights ---
    if equity_on:
        weights = equity_weights(stack)
    else:
        weights = np.ones(n_pix, dtype=np.float64)

    # --- candidates (Stage A+B) ---
    cands = build_candidates(stack, model)
    per_type_counts: dict[str, int] = {}
    for cd in cands:
        per_type_counts[cd.intervention] = per_type_counts.get(cd.intervention, 0) + 1

    if not cands:
        return _empty_strategy(stack, budget, area_cap, method)

    # --- Stage C: dispatch ---
    if method == "ilp":
        res = ilp_optimize(cands, weights, budget, area_cap)
        used = "ilp"
    elif method == "nsga2":
        nsga = nsga2_optimize(cands, weights, budget, area_cap)
        # pick the knee/most-equitable feasible portfolio from the front
        res = _pick_from_front(nsga["front"], weights, cands)
        res["nsga_front"] = nsga["front"]
        res["nsga_method"] = nsga["method"]
        used = f"nsga2 ({nsga['method']})"
    else:
        res = lazy_greedy_optimize(cands, weights, budget, area_cap)
        used = "greedy"

    selected: list[Candidate] = res["selected"]
    absorbed = res["absorbed"]  # flat per-pixel delivered ΔLST (°C)
    cap_c = res.get("cap_c", 8.0)

    # --- Stage D: placement records ---
    xx, yy = stack.grid_coords()  # CRS coords per pixel
    placements = _build_placement_records(stack, selected, xx, yy, cap_c)

    # --- city-wide totals ---
    placement_raster = absorbed.reshape(H, W).astype(np.float32)
    pop = _flat(stack, dm.POPULATION, default=1.0)
    exposure_reduction = float(np.dot(pop, absorbed))  # person-°C avoided

    mean_dlst = float(np.mean(absorbed)) if absorbed.size else 0.0
    max_dlst = float(np.max(absorbed)) if absorbed.size else 0.0
    hotspot_dlst = _hotspot_mean(stack, absorbed)

    # --- InVEST air-temperature cross-check on the treated scenario ---
    # mean_dair_grid is the whole-AOI mean (small: InVEST mixes air at ~500 m);
    # mean_dair_treated concentrates on the treated footprint (the actionable
    # number). A placement-derived air estimate (per-site cited surface->air
    # mapping) is the robust secondary figure the report/app shows.
    delta_air_raster, mean_dair_grid, mean_dair_treated = _air_delta_for_plan(stack, selected)
    mean_dair_placements = (
        float(np.mean([p["estimated_delta_air_C"] for p in placements]))
        if placements else 0.0)

    total_cost = float(sum(c.cost for c in selected))
    area_treated = float(sum(c.area_m2 for c in selected))

    totals = {
        "n_sites": len(selected),
        "budget": budget,
        "budget_used": total_cost,
        "budget_fraction_used": (total_cost / budget) if budget > 0 else 0.0,
        "area_treated_m2": area_treated,
        "area_cap_m2": area_cap,
        "area_fraction_used": (area_treated / area_cap) if area_cap > 0 else 0.0,
        "total_cost": total_cost,
        "mean_delta_lst_C": mean_dlst,
        "max_delta_lst_C": max_dlst,
        "hotspot_delta_lst_C": hotspot_dlst,
        "mean_delta_air_C": mean_dair_grid,
        "mean_delta_air_treated_C": mean_dair_treated,
        "mean_delta_air_placements_C": mean_dair_placements,
        "population_weighted_exposure_reduction": exposure_reduction,
        "objective": float(res.get("objective", 0.0)),
        "submodular_bound": float(C.SUBMODULAR_GREEDY_BOUND),
        "equity_weighting": equity_on,
    }

    return {
        "method": used,
        "placements": placements,
        "totals": totals,
        "placement_raster": placement_raster,
        "delta_air_raster": delta_air_raster,
        "candidates_considered": len(cands),
        "per_type_counts": per_type_counts,
        "objective_trace": res.get("objective_trace", []),
    }


def optimize(fs: dm.FeatureStack, model: Any, cfg: Any) -> dict[str, Any]:
    """§11.6 top-level dispatch.

    Returns ``{'portfolio': <GeoDataFrame|DataFrame|list>, 'city_dC': float,
    'exposure_reduction': float}`` (the PS-1 'optimal intervention strategy'),
    wrapping :func:`optimize_interventions`. The ``portfolio`` is a
    ``geopandas.GeoDataFrame`` if geopandas is importable, else a
    ``pandas.DataFrame``, else the raw list of placement dicts. Also surfaces the
    full structured result under ``'result'`` for the CLI/report/app.
    [R6 Stage C-D]
    """
    result = optimize_interventions(fs, model, cfg)
    placements = result["placements"]
    totals = result["totals"]

    portfolio = _placements_to_geo(placements, fs.crs)
    return {
        "portfolio": portfolio,
        "city_dC": totals["mean_delta_lst_C"],
        "exposure_reduction": totals["population_weighted_exposure_reduction"],
        "result": result,
    }


# ===========================================================================
# Internal helpers — geometry, footprints, combiners, serialisation
# ===========================================================================
def _resolution_m(fs: dm.FeatureStack) -> float:
    a, b, c, d, e, f = fs.transform
    px = abs(a)
    if np.isfinite(px) and px > 0:
        if px < 1e-3:
            return float(px * 111_000.0)
        return float(px)
    return float(fs.meta.get("resolution_m", 100.0))


def _flat(fs: dm.FeatureStack, name: str, default: float) -> np.ndarray:
    if fs.has(name):
        return np.asarray(fs.get(name), dtype=np.float64).reshape(-1)
    return np.full(fs.shape[0] * fs.shape[1], float(default), dtype=np.float64)


def _decay_offsets(d_px: float, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """(row/col offsets, weights) of the exp(-r/d) footprint above ``threshold``."""
    radius = int(min(max(np.ceil(d_px * math.log(1.0 / max(threshold, 1e-6))), 0), 20))
    if radius < 1:
        return np.array([[0, 0]]), np.array([1.0])
    offs = []
    wts = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r = math.hypot(dr, dc)
            w = math.exp(-r / max(d_px, 1e-6))
            if w >= threshold:
                offs.append((dr, dc))
                wts.append(w)
    return np.array(offs, dtype=int), np.array(wts, dtype=np.float64)


def _footprint(r, c, self_delta, offsets, kweights, H, W):
    """Footprint (flat indices, delivered ΔLST) of a candidate at (r,c)."""
    rr = r + offsets[:, 0]
    cc = c + offsets[:, 1]
    inb = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    rr = rr[inb]; cc = cc[inb]
    w = kweights[inb]
    idx = rr * W + cc
    delivered = self_delta * w  # peak (centre, w=1) = self_delta
    return idx.astype(np.int64), delivered.astype(np.float64)


def _standalone_value(cd: Candidate, weights: np.ndarray, cap_c: float) -> float:
    d = np.minimum(cd.foot_delta, cap_c)
    return float(np.dot(weights[cd.foot_idx], d))


def _absorb(selected: list[Candidate], weights: np.ndarray, cap_c: float):
    """Serial-absorption combine of a selection -> (absorbed flat ΔLST, objective)."""
    n_pix = weights.size
    residual = np.ones(n_pix, dtype=np.float64)
    for cd in selected:
        idx = cd.foot_idx
        factor = np.clip(1.0 - cd.foot_delta / cap_c, 0.0, 1.0)
        residual[idx] = residual[idx] * factor
    absorbed = cap_c * (1.0 - residual)
    objective = float(np.dot(weights, absorbed))
    return absorbed, objective


def _empty_result(n_pix: int, cap_c: float) -> dict[str, Any]:
    return {
        "selected": [], "objective": 0.0, "cost": 0.0, "area": 0.0,
        "absorbed": np.zeros(n_pix, dtype=np.float64),
        "objective_trace": [], "cap_c": cap_c,
    }


def _hotspot_mean(fs: dm.FeatureStack, absorbed: np.ndarray) -> float:
    if not fs.has(dm.LST):
        return float(np.mean(absorbed)) if absorbed.size else 0.0
    lst = _flat(fs, dm.LST, np.nan)
    finite = np.isfinite(lst)
    if not finite.any():
        return float(np.mean(absorbed))
    thr = np.nanpercentile(lst[finite], 90.0)
    hot = finite & (lst >= thr)
    if not hot.any():
        return float(np.mean(absorbed))
    return float(np.mean(absorbed[hot]))


def _air_delta_for_plan(fs: dm.FeatureStack, selected: list[Candidate]):
    """InVEST air-temperature reduction (°C) of the treated scenario.

    Builds the combined perturbed stack from the selected sites, runs the InVEST
    CC→HM→T_air pass on baseline and treated stacks, and returns the per-pixel air
    cooling (baseline T_air − treated T_air) plus its mean. Uses a city UHI_max
    from baseline LST spread when none is configured.
    """
    H, W = fs.shape
    if not selected:
        return np.zeros((H, W), dtype=np.float32), 0.0, 0.0

    # UHI_max proxy: 95th-5th percentile of baseline LST (°C). Bounded.
    uhi_max = 3.5
    if fs.has(dm.LST):
        lst = _flat(fs, dm.LST, np.nan)
        finite = np.isfinite(lst)
        if finite.any():
            uhi_max = float(np.clip(
                np.nanpercentile(lst[finite], 95) - np.nanpercentile(lst[finite], 5),
                0.5, 12.0))
    t_ref = 0.0  # report air anomaly; cooling is difference so t_ref cancels

    plan: dict[str, list[Candidate]] = {}
    for cd in selected:
        plan.setdefault(cd.intervention, []).append(cd)
    # build combined mask per intervention type
    mask_plan: dict[str, np.ndarray] = {}
    for name, cds in plan.items():
        m = np.zeros((H, W), dtype=bool)
        for cd in cds:
            m[cd.row, cd.col] = True
        mask_plan[name] = m

    treated = sim.apply_plan(fs, mask_plan)

    base_air = invest.run_invest_ucm(fs, None, t_ref, uhi_max)["t_air"]
    treat_air = invest.run_invest_ucm(treated, None, t_ref, uhi_max)["t_air"]
    d_air = (np.asarray(base_air, dtype=np.float64)
             - np.asarray(treat_air, dtype=np.float64))
    d_air = np.clip(d_air, 0.0, None)  # report cooling only
    treated_union = np.zeros((H, W), dtype=bool)
    for m in mask_plan.values():
        treated_union |= m
    mean_treated = float(np.mean(d_air[treated_union])) if treated_union.any() else 0.0
    return d_air.astype(np.float32), float(np.mean(d_air)), mean_treated


def _build_placement_records(fs, selected, xx, yy, cap_c):
    """Per-site placement dicts (Stage D output)."""
    recs = []
    for rank, cd in enumerate(selected):
        obj = cat.get_intervention_obj(cd.intervention)
        lon = float(xx[cd.row, cd.col])
        lat = float(yy[cd.row, cd.col])
        # estimated air ΔT for this type = scale self surface ΔT into the cited
        # air range (air is much smaller than surface); use the ratio of cited
        # midpoints, bounded by the cited air range.
        d_air = _surface_to_air(cd.self_delta_c, obj)
        recs.append({
            "rank": rank + 1,
            "intervention_type": cd.intervention,
            "row": cd.row,
            "col": cd.col,
            "lon": lon,
            "lat": lat,
            "area_m2": cd.area_m2,
            "cost": cd.cost,
            "estimated_delta_lst_C": round(float(cd.self_delta_c), 3),
            "estimated_delta_air_C": round(float(d_air), 3),
            "within_literature_range": bool(obj.in_surface_range(cd.self_delta_c)),
            "mechanism": obj.mechanism,
        })
    return recs


def _surface_to_air(surface_dc: float, obj: cat.Intervention) -> float:
    """Map a predicted surface ΔLST to an estimated air ΔT within cited bounds."""
    s_mid = obj.expected_surface_midpoint
    a_mid = obj.expected_air_midpoint
    if s_mid > 1e-6:
        est = surface_dc * (a_mid / s_mid)
    else:
        est = a_mid
    lo, hi = obj.air_dC
    return float(np.clip(est, 0.0, hi))


def _pick_from_front(front, weights, candidates):
    """Pick the highest equity-objective feasible portfolio from a Pareto front."""
    if not front:
        return _empty_result(weights.size, 8.0)
    best = max(front, key=lambda d: d.get("equity_objective", 0.0))
    selected = best.get("selected", [])
    absorbed, obj = _absorb(selected, np.asarray(weights, dtype=np.float64), 8.0)
    return {
        "selected": selected, "objective": obj,
        "cost": float(sum(c.cost for c in selected)),
        "area": float(sum(c.area_m2 for c in selected)),
        "absorbed": absorbed, "objective_trace": [obj], "cap_c": 8.0,
    }


def _empty_strategy(fs, budget, area_cap, method):
    H, W = fs.shape
    return {
        "method": method,
        "placements": [],
        "totals": {
            "n_sites": 0, "budget": budget, "budget_used": 0.0,
            "budget_fraction_used": 0.0, "area_treated_m2": 0.0,
            "area_cap_m2": area_cap, "area_fraction_used": 0.0, "total_cost": 0.0,
            "mean_delta_lst_C": 0.0, "max_delta_lst_C": 0.0,
            "hotspot_delta_lst_C": 0.0, "mean_delta_air_C": 0.0,
            "population_weighted_exposure_reduction": 0.0, "objective": 0.0,
            "submodular_bound": float(C.SUBMODULAR_GREEDY_BOUND),
        },
        "placement_raster": np.zeros((H, W), dtype=np.float32),
        "delta_air_raster": np.zeros((H, W), dtype=np.float32),
        "candidates_considered": 0,
        "per_type_counts": {},
        "objective_trace": [],
    }


# ----- candidate <-> DataFrame coercion (pandas lazy) ----------------------
def _candidates_to_df(cands: list[Candidate]):
    try:
        import pandas as pd  # lazy

        rows = [{
            "site_id": cd.site_id, "row": cd.row, "col": cd.col,
            "intervention": cd.intervention, "cost": cd.cost,
            "area_m2": cd.area_m2, "self_delta_c": cd.self_delta_c,
        } for cd in cands]
        df = pd.DataFrame(rows)
        df.attrs["_candidates"] = cands  # carry objects for downstream solvers
        return df
    except Exception:
        return cands  # numpy-only path: return the list


def _coerce_candidates(candidates) -> list[Candidate]:
    """Accept a Candidate list or a DataFrame produced by generate_candidates."""
    if isinstance(candidates, list):
        return candidates
    # pandas DataFrame carrying the objects in .attrs
    attrs = getattr(candidates, "attrs", {})
    if "_candidates" in attrs:
        return attrs["_candidates"]
    raise TypeError(
        "candidates must be a list[Candidate] or a DataFrame from "
        "generate_candidates(); got %r" % type(candidates))


def _selection_to_df(selected: list[Candidate], absorbed, weights, cap_c):
    try:
        import pandas as pd  # lazy

        rows = []
        for rank, cd in enumerate(selected):
            obj = cat.get_intervention_obj(cd.intervention)
            rows.append({
                "rank": rank + 1, "intervention": cd.intervention,
                "row": cd.row, "col": cd.col, "cost": cd.cost,
                "area_m2": cd.area_m2,
                "delta_C": round(float(cd.self_delta_c), 3),
                "within_range": bool(obj.in_surface_range(cd.self_delta_c)),
            })
        return pd.DataFrame(rows)
    except Exception:
        return selected


def _placements_to_geo(placements: list[dict], crs: str):
    """Build a GeoDataFrame (lazy geopandas) else a DataFrame else the list."""
    if not placements:
        try:
            import pandas as pd  # lazy
            return pd.DataFrame(placements)
        except Exception:
            return placements
    try:
        import geopandas as gpd  # lazy
        from shapely.geometry import Point  # lazy

        geom = [Point(p["lon"], p["lat"]) for p in placements]
        gdf = gpd.GeoDataFrame(placements, geometry=geom, crs=crs)
        return gdf
    except Exception:
        try:
            import pandas as pd  # lazy
            return pd.DataFrame(placements)
        except Exception:
            return placements


__all__ = [
    "Candidate",
    "equity_weights",
    "priority_score",
    "build_candidates",
    "generate_candidates",
    "lazy_greedy_optimize",
    "optimize_greedy",
    "ilp_optimize",
    "optimize_ilp",
    "nsga2_optimize",
    "optimize_nsga2",
    "optimize_interventions",
    "optimize",
]


# §11.6 alias: optimize_nsga2(candidates, fs, n_gen=100) -> DataFrame of front.
def optimize_nsga2(candidates, fs: dm.FeatureStack, n_gen: int = 100):
    """§11.6 NSGA-II entry: Pareto front as a ``pandas.DataFrame`` (pymoo lazy).

    Builds equity weights from ``fs`` and a default budget/area from the AOI, runs
    :func:`nsga2_optimize`, and returns the non-dominated portfolios as a
    DataFrame (cost, cooling_sum, equity_objective, n_sites). [R6 §7.3]
    """
    cands = _coerce_candidates(candidates)
    weights = equity_weights(fs)
    res_m = _resolution_m(fs)
    aoi_area = fs.shape[0] * fs.shape[1] * res_m * res_m
    budget = float(getattr(fs, "meta", {}).get("optimizer_budget", 1.0e7))
    area_cap = 0.30 * aoi_area
    out = nsga2_optimize(cands, weights, budget, area_cap, n_gen=n_gen)
    front = out["front"]
    try:
        import pandas as pd  # lazy
        rows = [{
            "cost": f.get("cost", 0.0),
            "cooling_sum": f.get("cooling_sum", 0.0),
            "equity_objective": f.get("equity_objective", 0.0),
            "n_sites": f.get("n_sites", 0),
            "method": out["method"],
        } for f in front]
        return pd.DataFrame(rows)
    except Exception:
        return front
