#!/usr/bin/env python3
"""End-to-end ``urbanheat`` demo — the four PS-1 deliverables, offline.

Run it directly (no Jupyter, no GEE, no network)::

    python notebooks/demo.py
    python notebooks/demo.py --city Mumbai --grid 64 --output-dir outputs

This is a scripted, heavily-commented walkthrough that mirrors what
``urbanheat run --mode synthetic`` (and ``make demo``) does: it builds a
synthetic, physically-plausible :class:`~urbanheat.datamodel.FeatureStack`, runs
the whole pipeline via :func:`urbanheat.cli.run_pipeline`, prints a summary of
the **four PS-1 deliverables**, and saves the deliverable figures to
``outputs/``.

Deliverables printed/saved
--------------------------
1. **Heat-stress hotspot map** — the layered 5-class priority composite
   (surface hotspots gated by Getis-Ord Gi*/Moran's I + human heat-stress
   indices + a vulnerability-weighted priority layer).      -> ``hotspots.png``
2. **Driver attribution** — ranked %-contribution across the four families
   (LULC / morphology / vegetation / atmosphere).   -> ``driver_attribution.png``
3. **Validated AI/ML model** — the spatial-CV metric panel (RMSE/MAE/R²/…).
4. **Optimal cooling strategy** — selected interventions (type · placement ·
   estimated °C) and a placement map.               -> ``interventions.png``
   plus a written report (``urbanheat_report.md`` / ``.html``).

A pure-Python script (not a ``.ipynb``) on purpose: it is diff-able, runs in CI,
and needs only the lean stack (numpy/scipy/scikit-learn/matplotlib).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

# Make the script runnable from a fresh clone without ``pip install`` — put the
# repo root (the parent of this ``notebooks/`` dir) on ``sys.path`` so
# ``import urbanheat`` resolves even when the package is not installed.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Keep each pipeline stage snappy: ``run_pipeline`` wraps every stage in a
# SIGALRM watchdog (env-tunable) and falls back to a fast numpy implementation if
# a sibling stage is slow/unavailable — so the demo always finishes quickly and
# never hangs. A modest budget means the whole synthetic run completes in well
# under a minute even when an optional accelerated learner (e.g. a slow sklearn
# HistGradientBoosting build) is the bottleneck. Set this *before* importing the
# CLI so it is honoured; override with ``URBANHEAT_STAGE_TIMEOUT`` if desired.
os.environ.setdefault("URBANHEAT_STAGE_TIMEOUT", "12")

import numpy as np  # noqa: E402

from urbanheat import __version__ as URBANHEAT_VERSION  # noqa: E402
from urbanheat import datamodel as dm  # noqa: E402
from urbanheat.config import CITY_PRESETS, DEFAULT_CITY, Config  # noqa: E402


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------
def _rule(title: str) -> None:
    """Print a titled section rule."""
    line = "=" * 74
    print(f"\n{line}\n{title}\n{line}")


def _kv(label: str, value: Any, width: int = 34) -> None:
    """Print an aligned ``label: value`` line."""
    print(f"  {label:<{width}} {value}")


def _metrics_to_dict(metrics: Any) -> dict[str, float]:
    """Flatten a metrics object (dict or per-fold frame) into ``{name: value}``."""
    if metrics is None:
        return {}
    if isinstance(metrics, dict):
        out: dict[str, float] = {}
        for k, v in metrics.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return out
    try:  # pandas per-fold frame -> mean over folds
        import pandas as pd
        if isinstance(metrics, pd.DataFrame):
            num = metrics.select_dtypes("number")
            return {str(c): float(num[c].mean()) for c in num.columns}
    except Exception:  # noqa: BLE001
        pass
    return {}


def _portfolio_records(portfolio: Any) -> list[dict[str, Any]]:
    """Normalise a portfolio (frame / list-of-dicts / geo-frame) to plain dicts."""
    if portfolio is None:
        return []
    try:
        import pandas as pd
        if isinstance(portfolio, pd.DataFrame):
            return portfolio.to_dict("records")
    except Exception:  # noqa: BLE001
        pass
    if isinstance(portfolio, list):
        return [r for r in portfolio if isinstance(r, dict)]
    return []


def _first(d: dict[str, Any], *keys: str) -> Any:
    """First present, non-None value among ``keys``."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


# ---------------------------------------------------------------------------
# Deliverable reporters
# ---------------------------------------------------------------------------
def _report_hotspots(results: dict[str, Any], fs: Any) -> None:
    """Deliverable 1: heat-stress hotspots."""
    _rule("DELIVERABLE 1 — Heat-stress hotspot map (layered 5-class composite)")
    lst = np.asarray(fs.get(dm.LST), dtype=float)
    _kv("LST range (°C)", f"{np.nanmin(lst):.1f} … {np.nanmax(lst):.1f} "
        f"(mean {np.nanmean(lst):.1f})")
    for layer, label in ((dm.PRIORITY_SCORE, "priority score (0-100)"),
                         (dm.HOTSPOT_MASK, "surface-hotspot mask"),
                         (dm.GISTAR_Z, "Getis-Ord Gi* z"),
                         (dm.SUHII, "surface UHI intensity (°C)"),
                         (dm.HVI, "heat-vulnerability index")):
        if fs.has(layer):
            arr = np.asarray(fs.get(layer), dtype=float)
            if layer == dm.HOTSPOT_MASK:
                frac = float(np.nanmean(arr > 0)) * 100.0
                _kv(label, f"{frac:.1f}% of cells flagged")
            else:
                _kv(label, f"min {np.nanmin(arr):.2f}, max {np.nanmax(arr):.2f}")
    stats = results.get("hotspot_stats")
    if isinstance(stats, dict):
        for k in ("n_hotspot_pixels", "hotspot_area_frac", "max_priority"):
            if k in stats:
                _kv(k, stats[k])


def _report_attribution(results: dict[str, Any]) -> None:
    """Deliverable 2: driver attribution across the four families."""
    _rule("DELIVERABLE 2 — Driver attribution (LULC / morphology / veg / atmos)")
    attribution = results.get("attribution")
    families = None
    if isinstance(attribution, dict):
        families = _first(attribution, "families", "family", "table")
    rows: list[tuple[str, float]] = []
    try:
        import pandas as pd
        if isinstance(families, pd.DataFrame):
            label_col = next((c for c in ("family", "feature", "item")
                              if c in families.columns), families.columns[0])
            val_col = next((c for c in ("pct_contribution", "pct", "value",
                                        "mean_abs_shap", "r2_share")
                            if c in families.columns), families.columns[-1])
            rows = [(str(r[label_col]), float(r[val_col]))
                    for _, r in families.iterrows()]
    except Exception:  # noqa: BLE001
        pass
    if not rows and isinstance(families, dict):
        rows = [(str(k), float(v)) for k, v in families.items()]
    if rows:
        total = sum(abs(v) for _, v in rows) or 1.0
        for name, val in sorted(rows, key=lambda t: -abs(t[1])):
            pct = 100.0 * abs(val) / total if total > 1.5 else val
            bar = "█" * int(round(min(pct, 100.0) / 3.0))
            _kv(name, f"{pct:6.1f}%  {bar}")
    else:
        print("  (attribution table unavailable; see the report figure.)")


def _report_model(results: dict[str, Any]) -> None:
    """Deliverable 3: validated AI/ML model metrics."""
    _rule("DELIVERABLE 3 — Validated AI/ML model (spatial-CV metric panel)")
    metrics = _metrics_to_dict(results.get("metrics"))
    if metrics:
        for k in ("rmse", "mae", "bias", "ubrmse", "r2", "nse", "ccc", "kge"):
            if k in metrics:
                _kv(k.upper(), f"{metrics[k]:.4f}")
    else:
        print("  (model metrics unavailable for this run.)")
    physics = results.get("physics")
    if isinstance(physics, dict) and physics:
        print("  physics-consistency:")
        for k, v in list(physics.items())[:6]:
            _kv(f"  {k}", v, width=30)


def _report_interventions(results: dict[str, Any]) -> None:
    """Deliverable 4: optimal cooling intervention strategy."""
    _rule("DELIVERABLE 4 — Optimal cooling strategy (type · placement · °C)")
    opt = results.get("optimization")
    totals = opt.get("totals") if isinstance(opt, dict) else None
    if isinstance(opt, dict):
        city_dc = _first(opt, "city_dC", "city_delta_C")
        if city_dc is None and isinstance(totals, dict):
            city_dc = _first(totals, "mean_delta_lst_C", "max_delta_lst_C")
        if city_dc is not None:
            _kv("city-wide mean ΔLST (°C)", f"{float(city_dc):.2f}")
        exp = _first(opt, "exposure_reduction",
                     "population_weighted_exposure_reduction")
        if exp is None and isinstance(totals, dict):
            exp = totals.get("population_weighted_exposure_reduction")
        if exp is not None:
            _kv("population-heat exposure avoided", f"{float(exp):,.0f}")

    records = _portfolio_records(results.get("portfolio"))
    _kv("interventions selected", len(records))
    if records:
        print("\n  rank  type                location              ΔLST(°C)  cost")
        print("  " + "-" * 66)
        for i, rec in enumerate(records[:12], start=1):
            typ = _first(rec, "intervention_type", "intervention", "type") or "?"
            row, col = rec.get("row"), rec.get("col")
            lat, lon = rec.get("lat"), rec.get("lon")
            x, y = rec.get("x"), rec.get("y")
            if lat is not None and lon is not None:
                loc = f"{float(lat):.3f},{float(lon):.3f}"
            elif x is not None and y is not None:
                loc = f"{float(x):.0f},{float(y):.0f}"  # projected CRS coords
            elif row is not None and col is not None:
                loc = f"r{int(row)},c{int(col)}"
            else:
                loc = "-"
            dlst = _first(rec, "estimated_delta_lst_C", "delta_C", "delta_lst_C")
            cost = rec.get("cost")
            dlst_s = f"{float(dlst):8.2f}" if dlst is not None else "      - "
            cost_s = f"{float(cost):,.0f}" if cost is not None else "-"
            print(f"  {i:>4}  {str(typ):<18}  {loc:<18}  {dlst_s}  {cost_s}")
        if len(records) > 12:
            print(f"   … and {len(records) - 12} more.")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _save_figures(results: dict[str, Any], fs: Any, out_dir: str) -> list[str]:
    """Render/collect the deliverable figures into ``out_dir``; return paths.

    The pipeline already renders figures (``results['figures']``); we top up any
    that are missing by calling the ``urbanheat.viz.maps`` helpers directly. Each
    returns ``None`` if matplotlib is absent, which we report rather than crash.
    """
    figs: dict[str, str] = {}
    pipeline_figs = results.get("figures")
    if isinstance(pipeline_figs, dict):
        figs.update({k: v for k, v in pipeline_figs.items()
                     if isinstance(v, str) and os.path.exists(v)})

    try:
        from urbanheat.viz import maps as vmaps
    except Exception as exc:  # noqa: BLE001
        print(f"  (viz unavailable: {exc!r})")
        return list(figs.values())

    def _ensure(name: str, fn) -> None:
        if name in figs:
            return
        try:
            path = fn()
            if path:
                figs[name] = path
        except Exception as exc:  # noqa: BLE001
            print(f"  (could not render {name}: {exc!r})")

    _ensure("lst", lambda: vmaps.plot_lst(fs, out_dir=out_dir))
    _ensure("hotspots", lambda: vmaps.plot_hotspots(fs, out_dir=out_dir))
    attribution = results.get("attribution")
    if attribution is not None:
        ranked = attribution.get("families") if isinstance(attribution, dict) \
            else attribution
        _ensure("driver_attribution",
                lambda: vmaps.plot_driver_attribution(ranked, out_dir=out_dir))
    opt = results.get("optimization")
    if opt is not None:
        _ensure("interventions",
                lambda: vmaps.plot_interventions(fs, opt, out_dir=out_dir))
    return list(figs.values())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI options mirroring the relevant ``urbanheat run`` flags."""
    p = argparse.ArgumentParser(
        description="urbanheat synthetic end-to-end demo (the 4 PS-1 deliverables).")
    p.add_argument("--city", default=DEFAULT_CITY, choices=sorted(CITY_PRESETS),
                   help="Indian city preset (default: %(default)s).")
    p.add_argument("--grid", type=int, default=48,
                   help="Synthetic grid size N (N×N). Larger = slower (default 48).")
    p.add_argument("--resolution", type=float, default=100.0,
                   help="Grid resolution in metres (default 100).")
    p.add_argument("--budget", type=float, default=1.0e7,
                   help="Optimizer budget in currency units (default 1e7).")
    p.add_argument("--max-area-frac", type=float, default=0.30,
                   help="Max fraction of the AOI that may be treated (default 0.30).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default 0).")
    p.add_argument("--output-dir", default="outputs",
                   help="Where to write figures + report (default: outputs/).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the synthetic pipeline and print + save the four deliverables."""
    args = parse_args(argv)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    _rule("urbanheat — ISRO BAH 2026 PS-1 — synthetic end-to-end demo")
    _kv("version", URBANHEAT_VERSION)
    _kv("city", args.city)
    _kv("backend", "synthetic (offline; no GEE / no network)")
    _kv("grid", f"{args.grid}×{args.grid} @ {args.resolution:.0f} m")
    _kv("optimizer budget", f"{args.budget:,.0f}")
    _kv("output dir", out_dir)

    # Build the run configuration (synthetic mode, explicit small grid).
    cfg = Config.from_city(
        args.city,
        mode="synthetic",
        resolution_m=args.resolution,
        grid_shape=(args.grid, args.grid),
        seed=args.seed,
        output_dir=out_dir,
        optimizer_budget=args.budget,
        optimizer_max_area_frac=args.max_area_frac,
    )

    # The single function the CLI and the Streamlit app also call. It runs the
    # whole spine: data -> indices -> hotspots -> features -> physics-informed
    # ML -> attribution -> spatial-CV validation -> intervention simulation ->
    # optimization -> robustness -> maps -> report. Run on the MAIN thread so the
    # per-stage SIGALRM watchdog (fast numpy fallbacks) is active.
    from urbanheat.cli import run_pipeline

    _rule("Running the pipeline …")
    try:
        results = run_pipeline(cfg, make_maps=True, verbose=True)
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: the pipeline raised {type(exc).__name__}: {exc}",
              file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    fs = results.get("fs")
    if fs is None or not fs.has(dm.LST):
        print("ERROR: pipeline did not return a valid FeatureStack.", file=sys.stderr)
        return 1

    # --- print the four deliverables ---
    _report_hotspots(results, fs)
    _report_attribution(results)
    _report_model(results)
    _report_interventions(results)

    # --- figures + report ---
    _rule("Artifacts written")
    figure_paths = _save_figures(results, fs, out_dir)
    for path in figure_paths:
        _kv("figure", os.path.relpath(path, os.getcwd())
            if path.startswith(os.getcwd()) else path)
    report_path = results.get("report_path")
    if isinstance(report_path, str) and os.path.exists(report_path):
        _kv("report (markdown)", report_path)
        html = os.path.splitext(report_path)[0] + ".html"
        if os.path.exists(html):
            _kv("report (html)", html)

    _rule("Done.")
    print("  Re-run interactively with:  streamlit run app/streamlit_app.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
