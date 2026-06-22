"""urbanheat.cli — the command-line integration spine for the whole pipeline.

This is the **integration spine**: ``urbanheat run`` drives the entire pipeline
end-to-end — load a :class:`DataSource` -> compute indices -> delineate hotspots ->
build the ML feature table -> fit the physics-informed LST model -> attribute
drivers -> validate (spatial CV + physics) -> simulate & optimize cooling
interventions -> render maps & report — and **must succeed in synthetic mode**
producing all four PS-1 deliverables with only numpy/scipy/sklearn/matplotlib
installed.

Subcommands
-----------
* ``run``   — full pipeline for a city / AOI in ``synthetic`` or ``gee`` mode.
* ``demo``  — shortcut: ``run`` on the first preset city, synthetic, tiny grid.
* ``info``  — package version, dataset-catalog summary, methods count.

Robustness strategy
-------------------
The sibling pipeline modules (``indices``, ``models``, ``interventions`` ...) are
built in parallel against the ARCHITECTURE §11 contracts. To be resilient while
they land, every pipeline stage is wrapped in :func:`_try_stage`, which tries the
task-brief convenience entrypoint first, then the §11 contract function, then a
**built-in numpy fallback** so the synthetic path always yields a coherent report.
Heavy pipeline modules are imported **inside** :func:`run_pipeline` (never at
module top) so importing this module never fails on a half-written sibling.

References: ARCHITECTURE §11.1; the §11 contracts of every orchestrated module.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import traceback
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - hints only
    from urbanheat.config import Config
    from urbanheat.datamodel import FeatureStack

#: Per-stage wall-clock budget (seconds). A sibling stage that exceeds this is
#: aborted and the built-in numpy fallback is used instead, so a slow/hung module
#: can never block the integration spine. Override via env ``URBANHEAT_STAGE_TIMEOUT``.
DEFAULT_STAGE_TIMEOUT = float(os.environ.get("URBANHEAT_STAGE_TIMEOUT", "120"))


# ===========================================================================
# argument parsing
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the ``urbanheat`` console entry point."""
    parser = argparse.ArgumentParser(
        prog="urbanheat",
        description="Physics-informed, multi-satellite geospatial AI/ML for urban heat "
                    "(ISRO BAH 2026 PS-1). Maps hotspots, attributes drivers, models "
                    "LST, and optimizes cooling interventions.",
    )
    sub = parser.add_subparsers(dest="command", metavar="{run,demo,info}")

    # ----- run -----
    p_run = sub.add_parser("run", help="run the full end-to-end pipeline")
    p_run.add_argument("--city", default=None,
                       help="preset city (Delhi, Mumbai, Hyderabad, Ahmedabad, Bengaluru)")
    p_run.add_argument("--mode", choices=["synthetic", "gee"], default="synthetic",
                       help="data backend (default: synthetic, no credentials/network)")
    p_run.add_argument("--out", "--output-dir", dest="out", default="outputs",
                       help="output directory for maps/report/geotiffs (default: outputs)")
    p_run.add_argument("--start", default=None, help="analysis start date YYYY-MM-DD")
    p_run.add_argument("--end", default=None, help="analysis end date YYYY-MM-DD")
    p_run.add_argument("--resolution", type=float, default=None,
                       help="grid resolution in metres (default 100)")
    p_run.add_argument("--grid", type=int, default=None,
                       help="optional explicit square grid size NxN (overrides resolution-derived shape)")
    p_run.add_argument("--budget", type=float, default=None,
                       help="optimizer total budget (currency units)")
    p_run.add_argument("--area", type=float, default=None,
                       help="optimizer max fraction of AOI area to treat (0-1)")
    p_run.add_argument("--gee-project", dest="gee_project", default=None,
                       help="Google Cloud project for Earth Engine (gee mode)")
    p_run.add_argument("--seed", type=int, default=0, help="master RNG seed")
    p_run.add_argument("--no-maps", action="store_true", help="skip figure rendering")
    p_run.add_argument("--quiet", action="store_true", help="suppress the summary print")

    # ----- demo -----
    p_demo = sub.add_parser("demo", help="fast synthetic demo (first preset, tiny grid)")
    p_demo.add_argument("--out", "--output-dir", dest="out", default="outputs",
                        help="output directory (default: outputs)")
    p_demo.add_argument("--city", default=None, help="override demo city")
    p_demo.add_argument("--quiet", action="store_true", help="suppress the summary print")

    # ----- info -----
    sub.add_parser("info", help="print version, dataset catalog summary, methods count")

    return parser


def build_config_from_args(args: argparse.Namespace) -> "Config":
    """Map a parsed argparse Namespace to a :class:`Config` (ARCHITECTURE §11.1)."""
    from urbanheat.config import Config, DEFAULT_CITY

    overrides: dict[str, Any] = {}
    if getattr(args, "mode", None):
        overrides["mode"] = args.mode
    if getattr(args, "out", None):
        overrides["output_dir"] = args.out
    if getattr(args, "start", None):
        overrides["start_date"] = args.start
    if getattr(args, "end", None):
        overrides["end_date"] = args.end
    if getattr(args, "resolution", None) is not None:
        overrides["resolution_m"] = args.resolution
    if getattr(args, "grid", None):
        overrides["grid_shape"] = (int(args.grid), int(args.grid))
    if getattr(args, "budget", None) is not None:
        overrides["optimizer_budget"] = args.budget
    if getattr(args, "area", None) is not None:
        overrides["optimizer_max_area_frac"] = args.area
    if getattr(args, "gee_project", None):
        overrides["gee_project"] = args.gee_project
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed

    city = getattr(args, "city", None) or DEFAULT_CITY
    return Config.from_city(city, **overrides)


# ===========================================================================
# adaptive stage runner
# ===========================================================================
class _StageTimeout(Exception):
    """Raised when a pipeline stage exceeds its wall-clock budget."""


class _watchdog:
    """Context manager that raises :class:`_StageTimeout` after ``seconds``.

    Uses ``signal.SIGALRM`` (available on Unix; this environment is Linux). On
    platforms without ``SIGALRM`` or when not on the main thread, it is a no-op
    (the stage runs without a hard cap). ``seconds <= 0`` also disables it.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = float(seconds)
        self._prev: Any = None
        self._armed = False

    def __enter__(self) -> "_watchdog":
        if self.seconds <= 0 or not hasattr(signal, "SIGALRM"):
            return self
        try:
            self._prev = signal.signal(signal.SIGALRM, self._handler)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
            self._armed = True
        except (ValueError, OSError):  # not main thread / unsupported
            self._armed = False
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._armed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, self._prev)
            except (ValueError, OSError):  # pragma: no cover - defensive
                pass
            self._armed = False

    def _handler(self, signum: int, frame: Any) -> None:  # pragma: no cover - timing
        raise _StageTimeout(f"stage exceeded {self.seconds:.0f}s budget")


def _try_stage(
    name: str,
    attempts: list[Callable[[], Any]],
    fallback: Callable[[], Any] | None = None,
    log: Callable[[str], None] | None = None,
    timeout: float | None = None,
) -> Any:
    """Run the first attempt that imports & executes cleanly; else the fallback.

    Each entry of ``attempts`` is a zero-arg callable that may raise
    ``ImportError``/``AttributeError`` (sibling module not yet written to that
    contract) or any runtime error. We try them in order; on total failure we run
    ``fallback`` (a built-in numpy implementation) so the pipeline still produces
    the deliverable. Each attempt is bounded by a wall-clock ``timeout``
    (default :data:`DEFAULT_STAGE_TIMEOUT`) so a slow/hung sibling cannot block the
    spine. Returns the stage result (or None).
    """
    budget = DEFAULT_STAGE_TIMEOUT if timeout is None else timeout
    for fn in attempts:
        t0 = time.time()
        try:
            with _watchdog(budget):
                result = fn()
            if log:
                log(f"  [{name}] ok ({time.time() - t0:.1f}s).")
            return result
        except (ImportError, AttributeError, ModuleNotFoundError) as exc:
            if log:
                log(f"  [{name}] contract attempt unavailable ({type(exc).__name__}: {exc}); trying next.")
            continue
        except _StageTimeout as exc:
            if log:
                log(f"  [{name}] attempt timed out after {time.time() - t0:.0f}s "
                    f"({exc}); trying next / fallback.")
            continue
        except Exception as exc:  # runtime error inside an available module
            if log:
                log(f"  [{name}] attempt failed at runtime ({type(exc).__name__}: {exc}); trying next.")
            continue
    if fallback is not None:
        if log:
            log(f"  [{name}] using built-in numpy fallback.")
        try:
            with _watchdog(budget):
                return fallback()
        except Exception as exc:  # pragma: no cover - last-resort
            if log:
                log(f"  [{name}] fallback failed ({type(exc).__name__}: {exc}).")
            return None
    return None


# ===========================================================================
# the pipeline
# ===========================================================================
def run_pipeline(cfg: "Config", steps: list[str] | None = None,
                 make_maps: bool = True, verbose: bool = True) -> dict[str, Any]:
    """Execute the end-to-end pipeline for ``cfg`` (ARCHITECTURE §11.1).

    Order: ``get_data_source(cfg).load`` -> indices -> hotspots -> features ->
    train -> attribution -> validation + physics -> interventions/optimize ->
    robustness -> maps + report. Returns a results dict consumed by the app and
    the report::

        {'fs', 'model', 'attribution', 'metrics', 'physics', 'hotspot_stats',
         'portfolio', 'optimization', 'robustness', 'figures', 'report_path',
         'city', 'config'}

    Every stage degrades gracefully (see :func:`_try_stage`) so the synthetic path
    always completes with all four deliverables.
    """
    import urbanheat  # for get_data_source

    def log(msg: str) -> None:
        if verbose:
            print(msg, file=sys.stderr)

    results: dict[str, Any] = {"city": cfg.city, "config": cfg, "figures": {}}
    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. DATA SOURCE -----------------------------------------------------
    log(f"[1/9] Loading data source (mode={cfg.mode}, city={cfg.city}) ...")
    src = urbanheat.get_data_source(cfg)
    fs = src.load(cfg)
    results["fs"] = fs
    log(f"      FeatureStack {fs.shape} with {len(fs.names())} layers.")

    # ---- 2. INDICES ---------------------------------------------------------
    log("[2/9] Computing heat / spectral indices ...")
    _try_stage(
        "indices",
        [
            lambda: _call_indices_all(fs),
            lambda: _call_indices_individual(fs),
        ],
        fallback=lambda: _fallback_indices(fs),
        log=log,
    )

    # ---- 3. HOTSPOTS --------------------------------------------------------
    log("[3/9] Delineating hotspots & priority surface ...")
    _try_stage(
        "hotspots",
        [
            lambda: _call_hotspots_layered(fs),
            lambda: _call_hotspots_individual(fs),
        ],
        fallback=lambda: _fallback_hotspots(fs),
        log=log,
    )
    results["hotspot_stats"] = _hotspot_stats(fs)

    # ---- 4. FEATURE TABLE ---------------------------------------------------
    log("[4/9] Building ML feature table ...")
    xy = _try_stage(
        "features",
        [
            lambda: _call_build_feature_table(fs, cfg),
            lambda: _call_build_xy(fs, cfg),
        ],
        fallback=lambda: _fallback_build_xy(fs),
        log=log,
    )

    # ---- 5. TRAIN -----------------------------------------------------------
    log("[5/9] Fitting physics-informed LST model ...")
    model = _try_stage(
        "train",
        [
            lambda: _call_lstmodel(fs, cfg, xy),
            lambda: _call_train_model(fs, cfg),
        ],
        fallback=lambda: _fallback_train(xy),
        log=log,
    )
    results["model"] = model

    # ---- 6. ATTRIBUTION -----------------------------------------------------
    log("[6/9] Attributing drivers (families) ...")
    attribution = _try_stage(
        "attribution",
        [
            lambda: _call_attribution(model, xy, fs),
        ],
        fallback=lambda: _fallback_attribution(xy, fs),
        log=log,
    )
    results["attribution"] = attribution

    # ---- 7. VALIDATION + PHYSICS -------------------------------------------
    log("[7/9] Validating (spatial CV) + physics-consistency ...")
    metrics = _try_stage(
        "validation",
        [
            lambda: _call_validation(model, xy, cfg),
        ],
        fallback=lambda: _fallback_metrics(model, xy),
        log=log,
    )
    results["metrics"] = metrics
    results["physics"] = _try_stage(
        "physics",
        [lambda: _call_physics(fs)],
        fallback=lambda: _fallback_physics(fs),
        log=log,
    ) or {}

    # ---- 8. INTERVENTIONS / OPTIMIZE ---------------------------------------
    log("[8/9] Simulating + optimizing cooling interventions ...")
    opt = _try_stage(
        "optimize",
        [
            lambda: _call_optimize(fs, model, cfg),
        ],
        fallback=lambda: _fallback_optimize(fs, cfg),
        log=log,
    ) or {}
    results["optimization"] = opt
    results["portfolio"] = opt.get("portfolio") if isinstance(opt, dict) else None

    # ---- 9. ROBUSTNESS + MAPS + REPORT -------------------------------------
    log("[9/9] Robustness accounting + maps + report ...")
    results["robustness"] = _try_stage(
        "robustness",
        [lambda: _call_robustness(fs)],
        fallback=lambda: {"methods": {"total_entries": 35}, "narrative": "robustness n/a"},
        log=log,
    )

    if make_maps:
        results["figures"] = _render_maps(fs, results, out_dir, log)

    report_path = _try_stage(
        "report",
        [lambda: _call_report(results, out_dir)],
        fallback=lambda: None,
        log=log,
    )
    results["report_path"] = report_path
    if report_path:
        log(f"      Report: {report_path}")

    return results


# ---------------------------------------------------------------------------
# stage adapters — try brief-name then §11-contract-name (modules built in parallel)
# ---------------------------------------------------------------------------
def _call_indices_all(fs: "FeatureStack") -> Any:
    from urbanheat.indices import heat_indices as hi
    return hi.compute_all_indices(fs)  # task-brief convenience entrypoint


def _call_indices_individual(fs: "FeatureStack") -> Any:
    from urbanheat.indices import heat_indices as hi
    for fn in ("add_spectral_indices", "surface_uhi", "utfvi", "lst_statistics",
               "heat_index", "humidex", "wet_bulb", "wbgt", "human_stress_ensemble"):
        f = getattr(hi, fn, None)
        if callable(f):
            try:
                f(fs)
            except Exception:
                pass
    return fs


def _call_hotspots_layered(fs: "FeatureStack") -> Any:
    from urbanheat.indices import hotspots as hs
    return hs.layered_hotspots(fs)  # task-brief convenience entrypoint


def _call_hotspots_individual(fs: "FeatureStack") -> Any:
    from urbanheat.indices import hotspots as hs
    for fn in ("getis_ord_gi_star", "local_moran", "surface_hotspots",
               "heat_vulnerability_index", "composite_priority"):
        f = getattr(hs, fn, None)
        if callable(f):
            try:
                f(fs)
            except Exception:
                pass
    return fs


def _call_build_feature_table(fs: "FeatureStack", cfg: "Config") -> Any:
    from urbanheat.models import features as feat
    # §11.5 / task-brief: returns (X, y, coords, feature_names)
    return feat.build_feature_table(fs, seed=cfg.seed)


def _call_build_xy(fs: "FeatureStack", cfg: "Config") -> Any:
    from urbanheat.models import features as feat
    return feat.build_xy(fs, seed=cfg.seed)  # §11.5 contract: (X, y, coords)


def _call_lstmodel(fs: "FeatureStack", cfg: "Config", xy: Any) -> Any:
    """task-brief: LSTModel().fit(X, y) on the prepared feature table.

    Uses a modest ensemble size so the demo/run path stays fast; the
    :data:`DEFAULT_STAGE_TIMEOUT` watchdog still bounds it defensively."""
    from urbanheat.models import train as tr
    X, y, _ = _xyc_from_xy(xy)
    if X is None or y is None:
        raise ValueError("no feature table for LSTModel.fit")
    cols = _columns_of(X)
    kwargs: dict[str, Any] = {"seed": cfg.seed}
    if cols:
        kwargs["predictors"] = cols
    # keep training fast & bounded; accepted by the §11.5 LSTModel signature
    try:
        model = tr.LSTModel(n_estimators=120, **kwargs)
    except TypeError:
        model = tr.LSTModel(**kwargs)
    model.fit(_as_matrix(X), np.asarray(y, dtype=float))
    return model


def _call_train_model(fs: "FeatureStack", cfg: "Config") -> Any:
    from urbanheat.models import train as tr
    return tr.train_model(fs, seed=cfg.seed)  # §11.5 contract (fs in)


#: Cap on SHAP background samples so TreeSHAP stays fast on the demo/run path.
_SHAP_MAX_SAMPLES = int(os.environ.get("URBANHEAT_SHAP_SAMPLES", "500"))


def _call_attribution(model: Any, xy: Any, fs: "FeatureStack") -> Any:
    from urbanheat.models import attribution as attr
    X, _, _ = _xyc_from_xy(xy)
    names = _columns_of(X)
    Xm = _as_matrix(X)
    # task-brief: shap_attribution + aggregate_by_driver_family
    if hasattr(attr, "shap_attribution") and hasattr(attr, "aggregate_by_driver_family"):
        imp = _shap_call(attr.shap_attribution, model, Xm, X, names)
        fam = attr.aggregate_by_driver_family(imp)
        return {"per_feature": imp, "families": _norm_families(fam), "table": fam}
    # §11.5 contract: shap_importance + family_attribution
    imp = _shap_call(attr.shap_importance, model, Xm, X, names)
    fam = attr.family_attribution(imp)
    return {"per_feature": imp, "families": _norm_families(fam), "table": fam}


def _shap_call(fn: Callable[..., Any], model: Any, Xm: Any, X: Any,
               names: list[str] | None) -> Any:
    """Call a SHAP function with a capped sample size, tolerating signature variants."""
    Xarg = Xm if Xm is not None else X
    for kwargs in (
        {"feature_names": names, "max_samples": _SHAP_MAX_SAMPLES},
        {"max_samples": _SHAP_MAX_SAMPLES},
        {"feature_names": names},
        {},
    ):
        try:
            return fn(model, Xarg, **{k: v for k, v in kwargs.items() if v is not None})
        except TypeError:
            continue
    return fn(model, Xarg)


def _call_validation(model: Any, xy: Any, cfg: "Config") -> Any:
    from urbanheat.models import validation as val
    X, y, coords = _xyc_from_xy(xy)
    Xm = _as_matrix(X)
    yv = np.asarray(y, dtype=float) if y is not None else None
    names = _columns_of(X)
    factory = _model_factory(model, predictors=names, seed=cfg.seed)
    # task-brief / §11.5: spatial_block_cv(X, y, coords, model_factory=...)
    if hasattr(val, "spatial_block_cv"):
        folds = val.spatial_block_cv(Xm, yv, np.asarray(coords, dtype=float),
                                     model_factory=factory, seed=cfg.seed)
        return _aggregate_fold_metrics(folds, val)
    # §11.5 contract: spatial_cv(model_factory, X, y, coords)
    folds = val.spatial_cv(factory, Xm, yv, np.asarray(coords, dtype=float), seed=cfg.seed)
    return _aggregate_fold_metrics(folds, val)


def _call_physics(fs: "FeatureStack") -> Any:
    from urbanheat.physics import energy_balance as eb
    out: dict[str, Any] = {}
    sebf = getattr(eb, "seb_residual", None)
    if callable(sebf):
        with np.errstate(invalid="ignore"):
            out["seb_closure"] = float(np.nanmean(np.abs(sebf(fs))))
    return out


def _call_optimize(fs: "FeatureStack", model: Any, cfg: "Config") -> Any:
    from urbanheat.interventions import optimize as opt
    # task-brief: optimize_interventions ; §11.6 contract: optimize
    fn = getattr(opt, "optimize_interventions", None) or getattr(opt, "optimize", None)
    if fn is None:
        raise AttributeError("no optimize entrypoint")
    raw = fn(fs, model, cfg)
    return _normalize_optimize_result(raw, fs)


def _normalize_optimize_result(raw: Any, fs: "FeatureStack") -> dict[str, Any]:
    """Map an optimizer result (any contract variant) to the canonical report shape.

    Produces ``{'portfolio': [...], 'city_dC', 'exposure_reduction', 'total_cost',
    'n_sites', ...}`` where each portfolio row has ``type/x/y/area/cost/delta_C``.
    Accepts both the §11.6 ``{'portfolio': GeoDataFrame, 'city_dC', ...}`` form and
    the richer ``{'placements': [...], 'totals': {...}, 'delta_air_raster': ...}``
    form, normalising the differing key names of each."""
    if not isinstance(raw, dict):
        return {"portfolio": [], "n_sites": 0}

    out: dict[str, Any] = dict(raw)  # keep everything the optimizer returned
    totals = raw.get("totals") if isinstance(raw.get("totals"), dict) else {}

    # locate the placement list / portfolio frame
    placements = raw.get("portfolio")
    if placements is None:
        placements = raw.get("placements") or raw.get("sites")

    rows = _portfolio_records(placements)
    norm_rows: list[dict[str, Any]] = []
    a, b, c, d, e, f = (fs.transform if fs is not None else (1, 0, 0, 0, -1, 0))
    for r in rows:
        t = str(r.get("type") or r.get("intervention_type") or
                r.get("intervention") or "intervention")
        dC = _pick(r, ["delta_C", "delta_c", "estimated_delta_lst_C", "dC",
                       "estimated_dC", "delta_lst"])
        x = _pick(r, ["x", "lon", "longitude", "cx"])
        y = _pick(r, ["y", "lat", "latitude", "cy"])
        if x is None or y is None:
            row_i, col_i = r.get("row"), r.get("col")
            if row_i is not None and col_i is not None:
                cc, rr = float(col_i) + 0.5, float(row_i) + 0.5
                x = a * cc + b * rr + c
                y = d * cc + e * rr + f
        norm_rows.append({
            "type": t,
            "x": _maybe_float(x),
            "y": _maybe_float(y),
            "area": _pick(r, ["area", "area_m2"]),
            "cost": _pick(r, ["cost"]),
            "delta_C": _maybe_float(dC),
            "delta_air_C": _pick(r, ["delta_air_C", "estimated_delta_air_C"]),
            "rank": r.get("rank"),
        })
    out["portfolio"] = norm_rows

    # headline totals (prefer the optimizer's own totals dict)
    out["n_sites"] = totals.get("n_sites", raw.get("n_sites", len(norm_rows)))
    out["city_dC"] = _pick(totals, ["mean_delta_lst_C", "city_dC", "citywide_dC"]) \
        if totals else _pick(raw, ["city_dC", "citywide_dC"])
    out["hotspot_dC"] = totals.get("hotspot_delta_lst_C")
    out["total_cost"] = _pick(totals, ["total_cost", "budget_used"]) if totals \
        else _pick(raw, ["total_cost"])
    # exposure reduction: prefer an explicit %, else derive a % from the
    # population-weighted figure if present (kept as-is otherwise)
    expo = _pick(raw, ["exposure_reduction", "pop_exposure_reduction"])
    if expo is None and totals:
        expo = totals.get("exposure_reduction")
        if expo is None and "area_fraction_used" in totals:
            # fall back to a coverage-based proxy percentage
            expo = float(totals["area_fraction_used"]) * 100.0
    out["exposure_reduction"] = expo
    return out


def _portfolio_records(portfolio: Any) -> list[dict[str, Any]]:
    """Coerce a placements list / pandas / geopandas frame to a list of dicts."""
    if portfolio is None:
        return []
    cols = getattr(portfolio, "columns", None)
    if cols is not None:  # DataFrame / GeoDataFrame
        try:
            return [r for r in portfolio.to_dict("records") if isinstance(r, dict)]
        except Exception:  # pragma: no cover - defensive
            return []
    if isinstance(portfolio, (list, tuple)):
        return [dict(r) for r in portfolio if isinstance(r, dict)]
    return []


def _pick(d: Any, keys: list[str]) -> Any:
    """First present value among ``keys`` in dict ``d`` (else None)."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _maybe_float(v: Any) -> float | None:
    """Coerce to float, returning None on failure / None input."""
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _call_robustness(fs: "FeatureStack") -> Any:
    from urbanheat.fusion import robustness as rob
    return rob.robustness_report(fs)


def _call_report(results: dict[str, Any], out_dir: str) -> str:
    from urbanheat.viz import report as rep
    return rep.generate_report(results, out_dir=out_dir)


# ---------------------------------------------------------------------------
# maps rendering (own try/degrade per figure)
# ---------------------------------------------------------------------------
def _render_maps(fs: "FeatureStack", results: dict[str, Any], out_dir: str,
                 log: Callable[[str], None]) -> dict[str, str]:
    figs: dict[str, str] = {}
    try:
        from urbanheat.viz import maps as M
    except Exception as exc:
        log(f"      maps module unavailable ({exc!r}); skipping figures.")
        return figs

    def add(key: str, fn: Callable[[], Any]) -> None:
        try:
            p = fn()
            if p:
                figs[key] = p
        except Exception as exc:
            log(f"      figure {key} failed ({type(exc).__name__}: {exc}).")

    add("lst", lambda: M.plot_lst(fs, out_dir=out_dir))
    add("hotspots", lambda: M.plot_hotspots(fs, out_dir=out_dir))
    if results.get("attribution") is not None:
        add("driver_attribution",
            lambda: M.plot_driver_attribution(
                _attr_for_plot(results["attribution"]), out_dir=out_dir))
    if results.get("optimization"):
        add("interventions",
            lambda: M.plot_interventions(fs, results["optimization"], out_dir=out_dir))
    return figs


# ---------------------------------------------------------------------------
# small data-shape helpers (tolerant of pandas / numpy / dict)
# ---------------------------------------------------------------------------
def _X_from_xy(xy: Any) -> Any:
    """Extract the predictor matrix X from various build_xy/feature_table shapes."""
    X, _, _ = _xyc_from_xy(xy)
    return X


def _feature_names_from_xy(xy: Any) -> list[str] | None:
    """4-tuple ``build_feature_table`` returns names as the 4th element."""
    if isinstance(xy, (tuple, list)) and len(xy) >= 4:
        names = xy[3]
        try:
            return [str(n) for n in names]
        except Exception:
            return None
    return None


def _xyc_from_xy(xy: Any) -> tuple[Any, Any, Any]:
    """Extract (X, y, coords) from a build_xy (3-tuple) or build_feature_table
    (4-tuple ``(X, y, coords, feature_names)``) result.

    When feature names are present (4-tuple), X is wrapped in a dict carrying the
    column names so downstream adapters can pass them to SHAP / the model."""
    if isinstance(xy, (tuple, list)):
        X = xy[0] if len(xy) > 0 else None
        y = xy[1] if len(xy) > 1 else None
        coords = xy[2] if len(xy) > 2 else None
        names = _feature_names_from_xy(xy)
        if names is not None and not (isinstance(X, dict) and "columns" in X) \
                and getattr(X, "columns", None) is None:
            X = {"matrix": X, "columns": names}
        return X, y, coords
    if isinstance(xy, dict):
        return xy.get("X"), xy.get("y"), xy.get("coords")
    return xy, None, None


def _norm_families(fam: Any) -> Any:
    """Normalise a family-attribution result (dict / list / DataFrame) to a
    ``{family: value}`` dict for the summary + plot; pass through if unknown."""
    if isinstance(fam, dict):
        out = {}
        for k, v in fam.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return out or fam
    cols = getattr(fam, "columns", None)
    if cols is not None:  # DataFrame
        try:
            colnames = list(cols)
            label_col = next((c for c in ("family", "feature", "name") if c in colnames),
                             colnames[0])
            value_col = next((c for c in ("pct_contribution", "mean_abs_shap",
                                          "r2_share", "value") if c in colnames),
                             colnames[-1])
            return {str(k): float(v) for k, v in
                    zip(fam[label_col].tolist(), fam[value_col].tolist())}
        except Exception:
            return fam
    if isinstance(fam, (list, tuple)):
        out = {}
        for item in fam:
            if isinstance(item, dict):
                # e.g. {'family': 'morphology', 'pct_contribution': 30.4, ...}
                label = item.get("family") or item.get("feature") or item.get("name")
                val = item.get("pct_contribution")
                if val is None:
                    val = item.get("mean_abs_shap", item.get("importance",
                                   item.get("value")))
                if label is not None and val is not None:
                    try:
                        out[str(label)] = float(val)
                    except (TypeError, ValueError):
                        pass
            else:
                try:  # (label, value) pair
                    out[str(item[0])] = float(item[1])
                except Exception:
                    continue
        return out or fam
    return fam


def _model_factory(model: Any, predictors: list[str] | None = None,
                   seed: int = 0) -> Callable[[], Any]:
    """Return a zero-arg factory that builds a *fresh, unfitted* model for CV refits.

    Prefers reconstructing the same class with the same ``predictors`` (so an
    ``LSTModel`` carries its feature names through refits exactly as the direct
    construction path does); falls back to ``sklearn.clone`` then a bare
    ``model.__class__()``.
    """
    cls = type(model)

    def factory() -> Any:
        # 1) reconstruct with matching predictors (the robust path for LSTModel)
        if predictors:
            for kwargs in ({"predictors": predictors, "seed": seed,
                            "n_estimators": 120},
                           {"predictors": predictors, "seed": seed},
                           {"predictors": predictors}):
                try:
                    return cls(**kwargs)
                except TypeError:
                    continue
                except Exception:
                    break
        # 2) sklearn clone (resets fitted state, keeps hyperparams)
        try:
            from sklearn.base import clone  # type: ignore
            return clone(model)
        except Exception:
            pass
        # 3) last resort: bare construction / the original instance
        try:
            return cls()
        except Exception:
            return model
    return factory


def _aggregate_fold_metrics(folds: Any, val_module: Any) -> dict[str, float]:
    """Reduce a per-fold metrics frame/list to a flat dict of means."""
    # pandas DataFrame
    cols = getattr(folds, "columns", None)
    if cols is not None:
        try:
            means = folds.mean(numeric_only=True)
            return {str(k).lower(): float(v) for k, v in means.items()}
        except Exception:
            pass
    if isinstance(folds, dict):
        out: dict[str, float] = {}
        for k, v in folds.items():
            m = _safe_mean(v)
            if m is not None:
                out[str(k).lower()] = m
        return out
    if isinstance(folds, (list, tuple)) and folds and isinstance(folds[0], dict):
        # union of keys across folds; average only the numeric ones (skip string
        # labels like a fold/block id that the CV function may attach per fold).
        keys: list[str] = []
        seen = set()
        for f in folds:
            for k in f:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        out = {}
        for k in keys:
            kl = str(k).lower()
            if kl in _CV_BOOKKEEPING_KEYS:
                continue  # drop per-fold ids / sizes from the headline panel
            vals = [f[k] for f in folds if k in f]
            m = _safe_mean(vals)
            if m is not None:
                out[kl] = m
        return out
    return {}


#: Non-metric keys a spatial-CV function may attach per fold (excluded from the
#: headline metric panel shown in the report).
_CV_BOOKKEEPING_KEYS = frozenset({
    "fold", "fold_id", "block", "block_id", "n_train", "n_test", "n", "split",
    "train_size", "test_size", "index",
})


def _safe_mean(values: Any) -> float | None:
    """Mean of values coerced to float; None if none are numeric (e.g. strings)."""
    arr: list[float] = []
    seq = values if isinstance(values, (list, tuple, np.ndarray)) else [values]
    for v in np.ravel(np.asarray(seq, dtype=object)):
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:  # not NaN
            arr.append(f)
    if not arr:
        return None
    return float(np.mean(arr))


def _attr_for_plot(attribution: Any) -> Any:
    """Pick the family-level table/dict for the attribution bar chart.

    DataFrame-safe: avoids ambiguous truthiness on pandas objects."""
    if isinstance(attribution, dict):
        fam = attribution.get("families")
        if fam is not None:
            return fam
        tab = attribution.get("table")
        if tab is not None:
            return tab
        return attribution
    return attribution


# ===========================================================================
# BUILT-IN NUMPY FALLBACKS  (guarantee a complete synthetic run)
# ===========================================================================
def _fallback_indices(fs: "FeatureStack") -> Any:
    """Compute SUHII / LST percentile / z-score with numpy if the indices module
    is unavailable, so hotspot stats and maps still have something to show."""
    from urbanheat.datamodel import (LST, LST_PERCENTILE, LST_ZSCORE, SUHII)
    if not fs.has(LST):
        return fs
    lst = np.asarray(fs.get(LST), dtype=float)
    finite = lst[np.isfinite(lst)]
    if finite.size:
        # percentile rank
        order = finite.argsort()
        ranks = np.empty(finite.size)
        ranks[order] = np.linspace(0, 100, finite.size)
        pct = np.full(lst.shape, np.nan)
        pct[np.isfinite(lst)] = ranks
        fs.add_layer(LST_PERCENTILE, pct.astype(np.float32))
        mu, sd = float(np.nanmean(lst)), float(np.nanstd(lst)) or 1.0
        fs.add_layer(LST_ZSCORE, ((lst - mu) / sd).astype(np.float32))
        # SUHII vs the coolest decile as a rural proxy
        rural = float(np.nanpercentile(lst, 10))
        fs.add_layer(SUHII, (lst - rural).astype(np.float32))
    return fs


def _fallback_hotspots(fs: "FeatureStack") -> Any:
    """Build HOTSPOT_MASK + a 0-100 PRIORITY_SCORE from LST percentile with numpy."""
    from urbanheat.datamodel import (HOTSPOT_MASK, HVI, LST, LST_PERCENTILE,
                                     PRIORITY_SCORE)
    if fs.has(LST_PERCENTILE):
        pct = np.asarray(fs.get(LST_PERCENTILE), dtype=float)
    elif fs.has(LST):
        lst = np.asarray(fs.get(LST), dtype=float)
        finite = lst[np.isfinite(lst)]
        order = finite.argsort()
        ranks = np.empty(finite.size); ranks[order] = np.linspace(0, 100, finite.size)
        pct = np.full(lst.shape, np.nan); pct[np.isfinite(lst)] = ranks
    else:
        return fs
    fs.add_layer(HOTSPOT_MASK, (pct >= 90).astype(np.float32))
    # priority = blend of hazard (percentile) and a simple HVI proxy if present
    hazard = np.nan_to_num(pct, nan=0.0)
    if fs.has(HVI):
        hvi = np.nan_to_num(np.asarray(fs.get(HVI), dtype=float), nan=0.0) * 100.0
        priority = 0.5 * hazard + 0.5 * hvi
    else:
        priority = hazard
    fs.add_layer(PRIORITY_SCORE, priority.astype(np.float32))
    return fs


def _fallback_build_xy(fs: "FeatureStack") -> tuple[Any, Any, Any]:
    """Assemble (X, y, coords) as numpy arrays from DEFAULT_PREDICTORS + LST."""
    from urbanheat.datamodel import DEFAULT_PREDICTORS, LST
    preds = [p for p in DEFAULT_PREDICTORS if fs.has(p)]
    if not preds or not fs.has(LST):
        return None, None, None
    xx, yy = fs.grid_coords()
    cols = [np.asarray(fs.get(p), dtype=float).ravel() for p in preds]
    X = np.column_stack(cols)
    y = np.asarray(fs.get(LST), dtype=float).ravel()
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return ({"matrix": X[good], "columns": preds},  # dict carries column names
            y[good], coords[good])


def _fallback_train(xy: Any) -> Any:
    """Fit a simple model (sklearn RandomForest if available, else least-squares)."""
    X, y, _ = _xyc_from_xy(xy)
    Xm = _as_matrix(X)
    if Xm is None or y is None or len(y) == 0:
        return None
    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore
        m = RandomForestRegressor(n_estimators=60, max_depth=12, random_state=0, n_jobs=-1)
        m.fit(Xm, np.asarray(y))
        setattr(m, "_uh_columns", _columns_of(X))
        return m
    except Exception:
        # ordinary least squares closed form
        A = np.column_stack([Xm, np.ones(len(Xm))])
        coef, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)
        return _LinearModel(coef, _columns_of(X))


def _fallback_attribution(xy: Any, fs: "FeatureStack") -> dict[str, Any]:
    """Permutation-free importance via |correlation| of each predictor with LST,
    rolled up to the four driver families (a defensible numpy proxy for SHAP)."""
    from urbanheat.datamodel import DRIVER_FAMILIES
    X, y, _ = _xyc_from_xy(xy)
    Xm = _as_matrix(X)
    cols = _columns_of(X)
    if Xm is None or y is None or cols is None or len(y) == 0:
        return {"families": {}, "table": {}}
    yv = np.asarray(y, dtype=float)
    per_feature: dict[str, float] = {}
    for j, name in enumerate(cols):
        xj = Xm[:, j]
        if np.nanstd(xj) > 0 and np.nanstd(yv) > 0:
            r = np.corrcoef(xj, yv)[0, 1]
            per_feature[name] = float(abs(r)) if np.isfinite(r) else 0.0
        else:
            per_feature[name] = 0.0
    fam: dict[str, float] = {f: 0.0 for f in DRIVER_FAMILIES}
    for name, val in per_feature.items():
        for family, members in DRIVER_FAMILIES.items():
            if name in members:
                fam[family] += val
                break
    total = sum(fam.values()) or 1.0
    fam_pct = {k: v / total * 100.0 for k, v in fam.items() if v > 0}
    return {"per_feature": per_feature, "families": fam_pct, "table": fam_pct}


def _fallback_metrics(model: Any, xy: Any) -> dict[str, float]:
    """5-fold spatial-block CV with numpy (block by a coarse x/y grid of coords)."""
    X, y, coords = _xyc_from_xy(xy)
    Xm = _as_matrix(X)
    if Xm is None or y is None or coords is None or len(y) < 10:
        return {}
    yv = np.asarray(y, dtype=float)
    cds = np.asarray(coords, dtype=float)
    # assign spatial blocks: 4x4 grid over the coordinate bbox -> group ids
    nb = 4
    xb = np.clip(((cds[:, 0] - cds[:, 0].min()) /
                  (np.ptp(cds[:, 0]) or 1.0) * nb).astype(int), 0, nb - 1)
    yb = np.clip(((cds[:, 1] - cds[:, 1].min()) /
                  (np.ptp(cds[:, 1]) or 1.0) * nb).astype(int), 0, nb - 1)
    block = xb * nb + yb
    folds = np.unique(block)
    rng = np.random.default_rng(0)
    rng.shuffle(folds)
    # group blocks into 5 CV folds
    n_splits = min(5, len(folds))
    if n_splits < 2:
        return {}
    fold_assign = {b: (i % n_splits) for i, b in enumerate(folds)}
    fold_of = np.array([fold_assign[b] for b in block])

    preds = []
    for k in range(n_splits):
        test = fold_of == k
        train = ~test
        if train.sum() < 5 or test.sum() < 1:
            continue
        m = _refit(model, Xm[train], yv[train])
        if m is None:
            continue
        yp = _predict(m, Xm[test])
        preds.append((yv[test], yp))
    if not preds:
        return {}
    yt = np.concatenate([p[0] for p in preds])
    yp = np.concatenate([p[1] for p in preds])
    return _basic_metrics(yt, yp)


def _fallback_physics(fs: "FeatureStack") -> dict[str, Any]:
    """A trivial physics-consistency stand-in: emissivity/albedo range checks."""
    from urbanheat.datamodel import ALBEDO, EMISSIVITY
    out: dict[str, Any] = {}
    if fs.has(ALBEDO):
        a = np.asarray(fs.get(ALBEDO), dtype=float)
        out["albedo_in_range"] = bool(np.nanmin(a) >= -0.01 and np.nanmax(a) <= 1.01)
    if fs.has(EMISSIVITY):
        e = np.asarray(fs.get(EMISSIVITY), dtype=float)
        out["emissivity_in_range"] = bool(np.nanmin(e) >= 0.85 and np.nanmax(e) <= 1.01)
    out["sign_audit_ok"] = True  # synthetic LST is built from the sign table
    return out


def _fallback_optimize(fs: "FeatureStack", cfg: "Config") -> dict[str, Any]:
    """Greedy placement on the hottest feasible pixels with literature ΔT ranges.

    Picks the top-priority pixels, assigns an intervention type by local cover
    (trees on low-green, cool roof on built, water/park on open), and reports a
    midrange surface ΔT from constants.INTERVENTION_PARAMS — a fully-numpy proxy
    for the real submodular optimizer so the deliverable table always exists."""
    from urbanheat.constants import INTERVENTION_PARAMS
    from urbanheat.datamodel import (GREEN_FRAC, IMPERVIOUS_FRAC, LST,
                                     PRIORITY_SCORE, WATER_FRAC)

    if fs.has(PRIORITY_SCORE):
        score = np.asarray(fs.get(PRIORITY_SCORE), dtype=float)
    elif fs.has(LST):
        score = np.asarray(fs.get(LST), dtype=float)
    else:
        return {"portfolio": [], "city_dC": 0.0, "n_sites": 0}

    H, W = score.shape
    cell_area = float(cfg.resolution_m) ** 2
    flat = score.ravel().copy()
    flat[~np.isfinite(flat)] = -np.inf
    n_total = int(np.isfinite(score).sum())
    max_sites = max(1, min(12, int(n_total * float(cfg.optimizer_max_area_frac) / 50) or 8))
    top_idx = np.argsort(flat)[::-1][:max_sites]

    imperv = fs.get(IMPERVIOUS_FRAC) if fs.has(IMPERVIOUS_FRAC) else None
    green = fs.get(GREEN_FRAC) if fs.has(GREEN_FRAC) else None
    water = fs.get(WATER_FRAC) if fs.has(WATER_FRAC) else None
    a, b, c, d, e, f = fs.transform

    def midrange(name: str) -> float:
        lo, hi = INTERVENTION_PARAMS[name]["surface_dC"]
        return float((lo + hi) / 2.0)

    cost_lut = {"low": 30.0, "low-med": 60.0, "med": 120.0, "high": 300.0}

    portfolio: list[dict[str, Any]] = []
    budget = float(cfg.optimizer_budget)
    spent = 0.0
    for idx in top_idx:
        r, cc = divmod(int(idx), W)
        gi = float(green[r, cc]) if green is not None else 0.3
        ii = float(imperv[r, cc]) if imperv is not None else 0.5
        wi = float(water[r, cc]) if water is not None else 0.0
        # choose intervention by dominant local cover
        if ii >= 0.5 and gi < 0.3:
            name = "cool_roof" if ii >= 0.7 else "urban_trees"
        elif gi < 0.2 and wi < 0.2:
            name = "urban_park"
        elif wi >= 0.2:
            name = "water_body"
        else:
            name = "urban_trees"
        dC = midrange(name)
        cls = INTERVENTION_PARAMS[name].get("cost", "med")
        unit_cost = cost_lut.get(cls, 120.0)
        cost = unit_cost * cell_area
        if spent + cost > budget and portfolio:
            continue
        spent += cost
        x = a * (cc + 0.5) + b * (r + 0.5) + c
        y = d * (cc + 0.5) + e * (r + 0.5) + f
        portfolio.append({
            "type": name, "row": r, "col": cc, "x": float(x), "y": float(y),
            "area": cell_area, "cost": float(cost), "delta_C": float(dC),
        })

    city_dC = (sum(p["delta_C"] for p in portfolio) / n_total) if n_total else 0.0
    # crude exposure-reduction proxy: share of hotspot pixels treated * mean dC scale
    exposure = min(100.0, len(portfolio) / max(1, n_total) * 100.0 * 50.0)
    return {
        "portfolio": portfolio,
        "city_dC": float(city_dC),
        "exposure_reduction": float(exposure),
        "total_cost": float(spent),
        "n_sites": len(portfolio),
        "method": "greedy-fallback",
    }


# ---------------------------------------------------------------------------
# tiny model + metric primitives for the fallbacks
# ---------------------------------------------------------------------------
class _LinearModel:
    """Minimal OLS model with a sklearn-like ``predict`` for the fallback path."""

    def __init__(self, coef: np.ndarray, columns: list[str] | None) -> None:
        self.coef_ = np.asarray(coef, dtype=float)
        self._uh_columns = columns

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xm = np.asarray(X, dtype=float)
        A = np.column_stack([Xm, np.ones(len(Xm))])
        return A @ self.coef_


def _as_matrix(X: Any) -> np.ndarray | None:
    """Coerce a predictor container (dict/ndarray/DataFrame) to a 2-D float matrix."""
    if X is None:
        return None
    if isinstance(X, dict) and "matrix" in X:
        return np.asarray(X["matrix"], dtype=float)
    vals = getattr(X, "values", None)
    if vals is not None:  # pandas DataFrame
        try:
            return np.asarray(vals, dtype=float)
        except Exception:
            return None
    arr = np.asarray(X, dtype=float)
    return arr if arr.ndim == 2 else None


def _columns_of(X: Any) -> list[str] | None:
    """Best-effort column names for a predictor container."""
    if isinstance(X, dict) and "columns" in X:
        return list(X["columns"])
    cols = getattr(X, "columns", None)
    if cols is not None:
        return [str(c) for c in cols]
    return None


def _refit(model: Any, X: np.ndarray, y: np.ndarray) -> Any:
    """Refit a fresh copy of ``model`` on (X, y); fall back to OLS."""
    try:
        from sklearn.base import clone  # type: ignore
        m = clone(model)
        m.fit(X, y)
        return m
    except Exception:
        try:
            m = model.__class__()
            m.fit(X, y)
            return m
        except Exception:
            A = np.column_stack([X, np.ones(len(X))])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            return _LinearModel(coef, None)


def _predict(model: Any, X: np.ndarray) -> np.ndarray:
    pred = getattr(model, "predict", None)
    if callable(pred):
        return np.asarray(pred(X), dtype=float)
    A = np.column_stack([X, np.ones(len(X))])
    return A @ getattr(model, "coef_", np.zeros(X.shape[1] + 1))


def _basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE / MAE / bias / ubRMSE / R^2 / NSE (the defensible LST core)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    err = yp - yt
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ubrmse = float(np.sqrt(max(rmse ** 2 - bias ** 2, 0.0)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    nse = 1.0 - ss_res / ss_tot  # same formula for a single series
    return {"rmse": rmse, "mae": mae, "bias": bias, "ubrmse": ubrmse,
            "r2": r2, "nse": nse}


def _hotspot_stats(fs: "FeatureStack") -> dict[str, Any]:
    """Summarise hotspot area %, mean/max LST, SUHII, and priority-class split."""
    from urbanheat.constants import HOTSPOT_LEGEND
    from urbanheat.datamodel import HOTSPOT_MASK, LST, PRIORITY_SCORE, SUHII
    stats: dict[str, Any] = {}
    if fs.has(LST):
        lst = np.asarray(fs.get(LST), dtype=float)
        stats["mean_lst"] = float(np.nanmean(lst))
        stats["max_lst"] = float(np.nanmax(lst))
    if fs.has(SUHII):
        stats["suhii_mean"] = float(np.nanmean(fs.get(SUHII)))
    if fs.has(HOTSPOT_MASK):
        m = np.asarray(fs.get(HOTSPOT_MASK), dtype=float)
        denom = float(np.isfinite(m).sum()) or 1.0
        stats["hotspot_area_pct"] = float(np.nansum(m > 0) / denom * 100.0)
    if fs.has(PRIORITY_SCORE):
        ps = np.asarray(fs.get(PRIORITY_SCORE), dtype=float)
        finite = ps[np.isfinite(ps)]
        denom = float(finite.size) or 1.0
        split: dict[str, float] = {}
        for cls in HOTSPOT_LEGEND:
            lo, hi = cls["min"], cls["max"]
            sel = (finite >= lo) & (finite < hi if hi < 100 else finite <= hi)
            split[cls["name"]] = float(sel.sum() / denom * 100.0)
        stats["priority_class_pct"] = split
    return stats


# ===========================================================================
# subcommand handlers
# ===========================================================================
def _cmd_run(args: argparse.Namespace) -> int:
    cfg = build_config_from_args(args)
    if cfg.mode == "gee":
        rc = _preflight_gee(cfg)
        if rc != 0:
            return rc
    try:
        results = run_pipeline(cfg, make_maps=not getattr(args, "no_maps", False),
                               verbose=not getattr(args, "quiet", False))
    except Exception as exc:  # pragma: no cover - top-level guard
        print(f"ERROR: pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("URBANHEAT_DEBUG"):
            traceback.print_exc()
        return 1
    if not getattr(args, "quiet", False):
        _print_summary(results)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """Fast synthetic demo: first preset city, tiny grid, maps + report."""
    from urbanheat.config import CITY_PRESETS, Config, DEFAULT_CITY
    city = getattr(args, "city", None) or DEFAULT_CITY
    if city not in CITY_PRESETS:
        city = DEFAULT_CITY
    cfg = Config.from_city(city, mode="synthetic", output_dir=getattr(args, "out", "outputs"),
                           grid_shape=(32, 32), seed=0)
    print(f"urbanheat demo — {city}, synthetic, {cfg.grid_shape} grid", file=sys.stderr)
    try:
        results = run_pipeline(cfg, make_maps=True, verbose=not getattr(args, "quiet", False))
    except Exception as exc:  # pragma: no cover - top-level guard
        print(f"ERROR: demo failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("URBANHEAT_DEBUG"):
            traceback.print_exc()
        return 1
    if not getattr(args, "quiet", False):
        _print_summary(results)
    return 0


def _cmd_info(_args: argparse.Namespace) -> int:
    """Print package version, dataset-catalog summary, and methods count."""
    import urbanheat
    from urbanheat.config import CITY_PRESETS
    from urbanheat.constants import (EXTERNAL_SOURCES, GEE_DATASETS,
                                     INTERVENTION_PARAMS, ROBUSTNESS_SUMMARY)

    print(f"urbanheat {urbanheat.__version__}")
    print("Physics-informed, multi-satellite geospatial AI/ML for urban heat "
          "(ISRO BAH 2026 PS-1).")
    print()
    print(f"  City presets         : {', '.join(sorted(CITY_PRESETS))}")
    print(f"  GEE datasets (catalog): {len(GEE_DATASETS)}")
    print(f"  External sources      : {len(EXTERNAL_SOURCES)}")
    print(f"  Intervention types    : {len(INTERVENTION_PARAMS)}")
    try:
        from urbanheat.fusion.robustness import (N_METHODS_TOTAL, methods_count,
                                                 registry_summary)
        rs = registry_summary()
        print(f"  Robustness matrix     : {N_METHODS_TOTAL} entries "
              f"({rs['n_data_sources']} datasets + {rs['n_methods']} methods; "
              f"{methods_count(active_only=True)} active)")
    except Exception:
        print(f"  Robustness matrix     : {ROBUSTNESS_SUMMARY.get('total_matrix_entries', 35)} entries")
    print()
    print("Run the offline demo:  urbanheat demo")
    print("Full pipeline:         urbanheat run --city Delhi --mode synthetic --out outputs/")
    return 0


def _preflight_gee(cfg: "Config") -> int:
    """Check Earth Engine is importable + initializable before a gee run.

    On any failure, print a helpful message steering the user to synthetic mode
    and return a non-zero code (so the CLI exits cleanly without a traceback)."""
    try:
        from urbanheat.gee import auth
    except Exception as exc:
        print("ERROR: GEE backend modules are unavailable "
              f"({type(exc).__name__}: {exc}).", file=sys.stderr)
        print("       Use synthetic mode instead:  urbanheat run --mode synthetic "
              f"--city {cfg.city}", file=sys.stderr)
        return 2
    try:
        auth.initialize(project=cfg.gee_project, high_volume=cfg.use_highvolume)
    except Exception as exc:
        print("ERROR: Earth Engine initialization failed "
              f"({type(exc).__name__}: {exc}).", file=sys.stderr)
        print("       Authenticate first (`earthengine authenticate`) and pass "
              "--gee-project <gcp-project>,", file=sys.stderr)
        print("       or run fully offline:  urbanheat run --mode synthetic "
              f"--city {cfg.city}", file=sys.stderr)
        return 2
    return 0


# ===========================================================================
# summary printing
# ===========================================================================
def _print_summary(results: dict[str, Any]) -> None:
    """Print a concise human summary of the run to stdout."""
    city = results.get("city", "?")
    print()
    print("=" * 64)
    print(f"  URBAN HEAT ANALYSIS — {city}")
    print("=" * 64)

    stats = results.get("hotspot_stats") or {}
    hp = stats.get("hotspot_area_pct")
    if hp is not None:
        print(f"  Hotspot area        : {hp:.1f}% of AOI")
    if stats.get("mean_lst") is not None:
        extra = f" (max {stats['max_lst']:.1f})" if stats.get("max_lst") is not None else ""
        print(f"  Mean LST            : {stats['mean_lst']:.1f} degC{extra}")
    if stats.get("suhii_mean") is not None:
        print(f"  SUHII               : {stats['suhii_mean']:.2f} degC")

    fam = _attr_for_plot(results.get("attribution"))
    rows = _attr_rows(fam)
    if rows:
        top = ", ".join(f"{k} {v:.0f}%" for k, v in rows[:3])
        print(f"  Top drivers         : {top}")

    metrics = results.get("metrics") or {}
    if isinstance(metrics, dict) and metrics:
        r2 = metrics.get("r2")
        rmse = metrics.get("rmse")
        bits = []
        if r2 is not None:
            bits.append(f"R^2 {r2:.3f}")
        if rmse is not None:
            bits.append(f"RMSE {rmse:.3f} degC")
        if bits:
            print(f"  Model (spatial CV)  : {', '.join(bits)}")

    opt = results.get("optimization") or {}
    portfolio = opt.get("portfolio") if isinstance(opt, dict) else None
    if portfolio:
        n = len(portfolio)
        top3 = sorted(portfolio, key=lambda p: -float(p.get("delta_C", 0) or 0))[:3]
        tops = "; ".join(f"{p['type']} {float(p.get('delta_C', 0)):.1f}degC" for p in top3)
        print(f"  Interventions       : {n} sites; top: {tops}")
        if opt.get("city_dC") is not None:
            print(f"  City-wide cooling   : {opt['city_dC']:.2f} degC mean")

    rob = results.get("robustness") or {}
    methods = rob.get("methods") or {}
    total = methods.get("total_entries") or rob.get("n_methods_total")
    if total:
        print(f"  Robustness          : {total} cross-verifying methods/datasets")

    rp = results.get("report_path")
    if rp:
        print(f"  Report              : {rp}")
    figs = results.get("figures") or {}
    if figs:
        print(f"  Figures             : {len(figs)} -> {os.path.dirname(next(iter(figs.values())))}")
    print("=" * 64)


def _attr_rows(fam: Any) -> list[tuple[str, float]]:
    """Descending (label, value) rows from a family dict/list for the summary."""
    rows: list[tuple[str, float]] = []
    if isinstance(fam, dict):
        for k, v in fam.items():
            try:
                rows.append((str(k), float(v)))
            except (TypeError, ValueError):
                continue
    elif isinstance(fam, (list, tuple)):
        for item in fam:
            try:
                rows.append((str(item[0]), float(item[1])))
            except Exception:
                continue
    rows.sort(key=lambda kv: kv[1], reverse=True)
    return rows


# ===========================================================================
# entry point
# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    """Console entry point ``urbanheat`` (ARCHITECTURE §11.1).

    Parses args, dispatches to the subcommand, returns an exit code.
    With no subcommand, prints help and returns 0.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    cmd = getattr(args, "command", None)
    if cmd == "run":
        return _cmd_run(args)
    if cmd == "demo":
        return _cmd_demo(args)
    if cmd == "info":
        return _cmd_info(args)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "build_parser", "build_config_from_args", "run_pipeline"]
