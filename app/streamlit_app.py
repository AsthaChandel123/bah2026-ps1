"""Interactive Streamlit dashboard for ``urbanheat`` (ISRO BAH 2026, PS-1).

Run with::

    streamlit run app/streamlit_app.py
    #  or:  make app

The dashboard drives the **same** :func:`urbanheat.cli.run_pipeline` the CLI and
``make demo`` use, in **synthetic mode by default** (no Earth Engine credentials
or network required), and surfaces the four PS-1 deliverables:

1. a **heat-stress hotspot map** (the layered 5-class priority composite),
2. a **driver-attribution** chart across the four families
   (LULC / morphology / vegetation / atmosphere),
3. the **validated-model metrics** (spatial-CV panel), and
4. an **intervention optimizer** panel — selected interventions with their
   type, location and estimated °C reduction, plus a placement map — and a
   **download button** for the generated report.

Design notes
------------
* ``streamlit`` is imported at module top (this is an app entry script, not a
  library), but :func:`main` guards with a clear message if it is missing so the
  file still **byte-compiles and imports** without streamlit installed.
* Everything is wrapped in try/except and defaults to synthetic mode, so a
  missing optional dependency degrades gracefully rather than crashing the UI.
* The heavy lifting (data -> indices -> hotspots -> ML -> attribution ->
  validation -> interventions -> optimization -> report) lives in the package;
  this file is a thin, robust presentation layer.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any

# ``streamlit`` may be absent in the minimal/test environment. Import it at the
# top (idiomatic for an app entry point) but tolerate ImportError so the module
# still byte-compiles; :func:`main` re-checks and prints actionable guidance.
try:  # pragma: no cover - exercised only when streamlit is installed
    import streamlit as st
except Exception:  # noqa: BLE001
    st = None  # type: ignore[assignment]

# Package imports are light (numpy-only foundation); safe at module top.
from urbanheat import __version__ as URBANHEAT_VERSION
from urbanheat.config import CITY_PRESETS, DEFAULT_CITY, Config
from urbanheat import datamodel as dm


# ---------------------------------------------------------------------------
# Problem-statement header text
# ---------------------------------------------------------------------------
_PS1_INTRO = """
**Problem Statement 1 — Optimizing Urban Heat Mitigation & cooling strategies
via AI/ML, backed by physics-informed decision making.**

Cities heat unevenly: impervious surfaces, low albedo, sparse vegetation, deep
street canyons and anthropogenic heat push the surface energy budget away from
evaporative cooling and into stored/sensible heat, producing dangerous,
inequitably-distributed **urban heat-stress hotspots**. This tool:

1. **identifies hotspots** from satellite + meteorological data (a layered
   5-class composite gated by statistically-significant clustering);
2. **quantifies the drivers** of urban heating across **land use/land cover,
   urban morphology, vegetation and atmosphere**;
3. **models** the LST↔drivers relationship with a **physics-informed** ML model
   (surface-energy-balance signs enforced), validated with spatial CV; and
4. **simulates & optimizes cooling interventions** — returning the intervention
   **type**, **placement** and **estimated °C reduction**.

It runs end-to-end on **synthetic, physically-plausible data** with no
credentials, and on **Google Earth Engine** for production (the only change is
the data backend).
"""


# ---------------------------------------------------------------------------
# Pipeline driver (cached)
# ---------------------------------------------------------------------------
def _build_config(
    city: str,
    mode: str,
    resolution_m: float,
    grid_n: int,
    budget: float,
    max_area_frac: float,
    seed: int,
    gee_project: str | None,
) -> Config:
    """Assemble a :class:`Config` from the sidebar selections."""
    overrides: dict[str, Any] = dict(
        mode=mode,
        resolution_m=float(resolution_m),
        seed=int(seed),
        optimizer_budget=float(budget),
        optimizer_max_area_frac=float(max_area_frac),
        # A square grid keeps the synthetic demo fast and bounded regardless of
        # the city bbox; the GEE backend ignores grid_shape and uses resolution.
        grid_shape=(int(grid_n), int(grid_n)) if mode == "synthetic" else None,
    )
    if gee_project:
        overrides["gee_project"] = gee_project
    return Config.from_city(city, **overrides)


def _run_pipeline_cached(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Run the full pipeline for a serialised config; returns the results dict.

    Defined to be wrapped by ``st.cache_data`` (keyed on the plain-dict config)
    so re-rendering the page does not recompute the pipeline. Imported lazily so
    importing this module never pulls the heavy pipeline.
    """
    from urbanheat.cli import run_pipeline

    cfg = Config.from_dict(cfg_dict)
    os.makedirs(cfg.output_dir, exist_ok=True)
    return run_pipeline(cfg, make_maps=True, verbose=False)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _show_image_or_note(path: Any, caption: str) -> None:
    """Display a saved figure if it exists, else a friendly note."""
    if isinstance(path, str) and os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"{caption}: figure unavailable (matplotlib missing or layer absent).")


def _figures(results: dict[str, Any]) -> dict[str, str]:
    """The ``name -> path`` figure map produced by the pipeline (maybe empty)."""
    figs = results.get("figures")
    return figs if isinstance(figs, dict) else {}


def _render_hotspot_panel(results: dict[str, Any], fs: Any, out_dir: str) -> None:
    """Deliverable 1: the 5-class heat-stress hotspot map (+ LST basemap)."""
    st.subheader("1 · Heat-stress hotspot map")
    st.caption(
        "Layered 5-class priority composite (Low→Extreme): surface hotspots "
        "(LST percentile + UTFVI + SUHII) gated by Getis-Ord Gi* / local "
        "Moran's I clustering, fused with human heat-stress indices and a "
        "vulnerability-weighted (HVI) priority layer.")
    figs = _figures(results)

    col1, col2 = st.columns(2)
    with col1:
        path = figs.get("hotspots")
        if not path:
            try:
                from urbanheat.viz.maps import plot_hotspots
                path = plot_hotspots(fs, out_dir=out_dir)
            except Exception:  # noqa: BLE001
                path = None
        _show_image_or_note(path, "Hotspot priority (5-class)")
    with col2:
        path = figs.get("lst")
        if not path:
            try:
                from urbanheat.viz.maps import plot_lst
                path = plot_lst(fs, out_dir=out_dir)
            except Exception:  # noqa: BLE001
                path = None
        _show_image_or_note(path, "Land surface temperature (°C)")

    stats = results.get("hotspot_stats")
    if isinstance(stats, dict) and stats:
        with st.expander("Hotspot statistics"):
            st.json(stats)


def _render_driver_panel(results: dict[str, Any], fs: Any, out_dir: str) -> None:
    """Deliverable 2: ranked driver attribution across the four families."""
    st.subheader("2 · Driver attribution")
    st.caption(
        "Ranked %-contribution of the four PS-1 driver families — "
        "**LULC, morphology, vegetation, atmosphere** — to the modelled heat "
        "field (physics-sign-audited).")
    figs = _figures(results)
    path = figs.get("driver_attribution") or figs.get("attribution")
    attribution = results.get("attribution")
    if not path and attribution is not None:
        try:
            from urbanheat.viz.maps import plot_driver_attribution
            ranked = _attribution_for_plot(attribution)
            path = plot_driver_attribution(ranked, out_dir=out_dir)
        except Exception:  # noqa: BLE001
            path = None
    _show_image_or_note(path, "Driver attribution (ranked)")

    # Tabular family rollup, if available.
    table = _attribution_table(attribution)
    if table is not None:
        st.dataframe(table, use_container_width=True)


def _render_model_panel(results: dict[str, Any]) -> None:
    """Deliverable 3: the validated-model metric panel (spatial-CV)."""
    st.subheader("3 · Validated AI/ML model")
    st.caption(
        "Physics-informed model (SEB backbone + monotone-constrained ensemble + "
        "MGWR), validated with **spatial** cross-validation and physics-"
        "consistency checks.")
    metrics = _metrics_dict(results.get("metrics"))
    if metrics:
        # Headline metrics as cards, full panel as a table.
        keys = [k for k in ("r2", "rmse", "mae", "bias") if k in metrics]
        if keys:
            cols = st.columns(len(keys))
            labels = {"r2": "R²", "rmse": "RMSE (°C)", "mae": "MAE (°C)",
                      "bias": "Bias (°C)"}
            for c, k in zip(cols, keys):
                c.metric(labels.get(k, k.upper()), f"{metrics[k]:.3f}")
        st.dataframe(
            {"metric": list(metrics.keys()),
             "value": [round(float(v), 4) for v in metrics.values()]},
            use_container_width=True)
    else:
        st.info("Model metrics unavailable for this run.")

    physics = results.get("physics")
    if isinstance(physics, dict) and physics:
        with st.expander("Physics-consistency checks"):
            st.json(physics)


def _render_optimizer_panel(results: dict[str, Any], fs: Any, out_dir: str) -> None:
    """Deliverable 4: optimized interventions (type/location/°C) + placement map."""
    st.subheader("4 · Optimal cooling intervention strategy")
    st.caption(
        "Lazy-greedy submodular optimizer over feasible (site, intervention) "
        "candidates under budget / area / equity constraints — each selection "
        "carries its **type**, **placement** and **estimated °C reduction**.")

    figs = _figures(results)
    opt = results.get("optimization")
    portfolio = results.get("portfolio")

    col1, col2 = st.columns([3, 2])
    with col1:
        path = figs.get("interventions")
        if not path and opt is not None:
            try:
                from urbanheat.viz.maps import plot_interventions
                path = plot_interventions(fs, opt, out_dir=out_dir)
            except Exception:  # noqa: BLE001
                path = None
        _show_image_or_note(path, "Optimized intervention placement")
    with col2:
        totals = opt.get("totals") if isinstance(opt, dict) else None
        city_dc = _first_present(
            opt if isinstance(opt, dict) else {},
            ("city_dC", "city_delta_C", "mean_delta_lst_C"))
        if city_dc is None and isinstance(totals, dict):
            city_dc = _first_present(
                totals, ("mean_delta_lst_C", "city_dC", "max_delta_lst_C"))
        exposure = _first_present(
            opt if isinstance(opt, dict) else {},
            ("exposure_reduction", "population_weighted_exposure_reduction"))
        if exposure is None and isinstance(totals, dict):
            exposure = totals.get("population_weighted_exposure_reduction")
        if city_dc is not None:
            st.metric("City-wide mean ΔLST (°C)", f"{float(city_dc):.2f}")
        if exposure is not None:
            st.metric("Population-heat exposure avoided", f"{float(exposure):,.0f}")
        n_sites = _portfolio_len(portfolio)
        if n_sites is not None:
            st.metric("Sites selected", f"{n_sites}")

    # Per-site placement table: type · location · estimated °C.
    rows = _portfolio_rows(portfolio)
    if rows:
        st.markdown("**Selected interventions (ranked)**")
        st.dataframe(rows, use_container_width=True)
        # A point map of the placements, if we have geographic coords.
        latlon = _portfolio_latlon(rows)
        if latlon is not None and len(latlon):
            try:
                st.map(latlon)
            except Exception:  # noqa: BLE001
                pass
    else:
        st.info("No interventions were selected (try raising the budget/area).")


def _render_report_download(results: dict[str, Any]) -> None:
    """Offer the generated report (HTML preferred, else Markdown) for download."""
    st.subheader("Report")
    report_path = results.get("report_path")
    if not (isinstance(report_path, str) and os.path.exists(report_path)):
        st.info("Report not generated for this run.")
        return
    # Prefer the HTML sibling if present.
    html_path = os.path.splitext(report_path)[0] + ".html"
    chosen = html_path if os.path.exists(html_path) else report_path
    try:
        with open(chosen, "rb") as fh:
            data = fh.read()
    except Exception:  # noqa: BLE001
        st.warning(f"Report at {chosen} could not be read.")
        return
    mime = "text/html" if chosen.endswith(".html") else "text/markdown"
    st.download_button(
        label=f"⬇ Download report ({os.path.basename(chosen)})",
        data=data,
        file_name=os.path.basename(chosen),
        mime=mime,
    )
    st.caption(f"Report written to `{chosen}`.")


# ---------------------------------------------------------------------------
# Small data-coercion helpers (defensive: shapes vary across modules)
# ---------------------------------------------------------------------------
def _metrics_dict(metrics: Any) -> dict[str, float]:
    """Coerce a metrics object (dict / per-fold frame) into a flat dict."""
    if metrics is None:
        return {}
    if isinstance(metrics, dict):
        out = {}
        for k, v in metrics.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return out
    # pandas-like: average numeric columns over folds.
    try:
        import pandas as pd  # lazy
        if isinstance(metrics, pd.DataFrame):
            num = metrics.select_dtypes("number")
            return {str(c): float(num[c].mean()) for c in num.columns}
    except Exception:  # noqa: BLE001
        pass
    return {}


def _attribution_for_plot(attribution: Any) -> Any:
    """Pick the best representation of attribution for the bar-chart helper."""
    if isinstance(attribution, dict):
        for key in ("families", "family", "family_attribution", "ranked",
                    "importance"):
            if key in attribution and attribution[key] is not None:
                return attribution[key]
    return attribution


def _attribution_table(attribution: Any) -> Any:
    """Return a small family/feature contribution table if obtainable."""
    obj = _attribution_for_plot(attribution)
    try:
        import pandas as pd  # lazy
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, dict):
            return pd.DataFrame({"item": list(obj.keys()),
                                 "value": list(obj.values())})
    except Exception:  # noqa: BLE001
        pass
    if isinstance(obj, dict):
        return {"item": list(obj.keys()), "value": list(obj.values())}
    return None


def _first_present(d: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    """First finite numeric value among ``keys`` in ``d``."""
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return None


def _portfolio_rows(portfolio: Any) -> list[dict[str, Any]]:
    """Normalise a portfolio (frame / list of dicts) into display rows.

    Surfaces, per site: intervention **type**, **location** (row/col or lat/lon)
    and **estimated °C** reduction, tolerating the different column names the
    optimizer / geopandas paths use.
    """
    records: list[dict[str, Any]]
    if portfolio is None:
        return []
    try:
        import pandas as pd  # lazy
        if isinstance(portfolio, pd.DataFrame):
            records = portfolio.to_dict("records")
        elif isinstance(portfolio, list):
            records = list(portfolio)
        else:  # geopandas GeoDataFrame is a DataFrame subclass; handled above
            records = list(getattr(portfolio, "__iter__", lambda: [])())
    except Exception:  # noqa: BLE001
        records = portfolio if isinstance(portfolio, list) else []

    rows: list[dict[str, Any]] = []
    for i, rec in enumerate(records, start=1):
        if not isinstance(rec, dict):
            continue
        rows.append({
            "rank": rec.get("rank", i),
            "type": rec.get("intervention_type") or rec.get("intervention")
            or rec.get("type"),
            # Location: pixel (row/col), geographic (lat/lon) or projected (x/y).
            "row": rec.get("row"),
            "col": rec.get("col"),
            "lat": rec.get("lat"),
            "lon": rec.get("lon"),
            "x": _round(rec.get("x")),
            "y": _round(rec.get("y")),
            "ΔLST (°C)": _round(rec.get("estimated_delta_lst_C")
                                or rec.get("delta_C")
                                or rec.get("delta_lst_C")),
            "ΔT_air (°C)": _round(rec.get("estimated_delta_air_C")
                                  or rec.get("delta_air_C")),
            "cost": _round(rec.get("cost")),
        })
    # Drop all-empty columns so the table stays tidy across schema variants.
    keys = list(rows[0].keys()) if rows else []
    drop = {k for k in keys
            if all(r.get(k) in (None, "") for r in rows)}
    return [{k: v for k, v in r.items() if k not in drop} for r in rows]


def _portfolio_len(portfolio: Any) -> int | None:
    """Number of placements in the portfolio, if knowable."""
    if portfolio is None:
        return None
    try:
        return int(len(portfolio))
    except Exception:  # noqa: BLE001
        return None


def _portfolio_latlon(rows: list[dict[str, Any]]) -> Any:
    """Build a lat/lon DataFrame for ``st.map`` from portfolio rows (or None)."""
    try:
        import pandas as pd  # lazy
        pts = [(r["lat"], r["lon"]) for r in rows
               if r.get("lat") is not None and r.get("lon") is not None]
        if not pts:
            return None
        return pd.DataFrame(pts, columns=["lat", "lon"])
    except Exception:  # noqa: BLE001
        return None


def _round(v: Any, ndigits: int = 3) -> Any:
    """Round numbers for display; pass through non-numbers."""
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return v


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
def main() -> None:
    """Streamlit entry point. Renders the sidebar, runs the pipeline, shows results."""
    if st is None:
        msg = (
            "Streamlit is not installed. Install the app extra and run:\n\n"
            "    pip install -e \".[app]\"   # or: pip install streamlit\n"
            "    streamlit run app/streamlit_app.py\n")
        print(msg, file=sys.stderr)
        return

    st.set_page_config(page_title="urbanheat · ISRO BAH 2026 PS-1",
                       page_icon="🌡️", layout="wide")
    st.title("🌡️ urbanheat — Urban Heat Mitigation (ISRO BAH 2026, PS-1)")
    st.caption(f"urbanheat v{URBANHEAT_VERSION} · physics-informed, multi-satellite "
               "geospatial AI/ML")
    with st.expander("About this problem (PS-1)", expanded=False):
        st.markdown(_PS1_INTRO)

    # ----- sidebar controls -----
    with st.sidebar:
        st.header("Run configuration")
        city = st.selectbox("City preset", sorted(CITY_PRESETS),
                            index=sorted(CITY_PRESETS).index(DEFAULT_CITY))
        mode = st.radio("Data backend", ["synthetic", "gee"], index=0,
                        help="Synthetic runs offline with no credentials. "
                             "GEE is the production Earth-Engine path.")
        gee_project = None
        if mode == "gee":
            gee_project = st.text_input(
                "GEE Cloud project", value="",
                help="Google Cloud project for ee.Initialize (gee mode only).")
            st.warning("GEE mode needs `earthengine-api` + authentication. "
                       "If it fails, switch back to synthetic.")

        resolution_m = st.select_slider(
            "Resolution (m)", options=[30.0, 50.0, 100.0, 200.0], value=100.0)
        grid_n = st.slider(
            "Synthetic grid (N×N)", min_value=16, max_value=96, value=48, step=8,
            help="Grid size for the synthetic demo (larger = slower).")
        st.subheader("Optimizer")
        budget = st.number_input(
            "Budget (currency units)", min_value=0.0, value=1.0e7,
            step=1.0e6, format="%.0f")
        max_area_frac = st.slider(
            "Max treated area fraction", min_value=0.05, max_value=1.0,
            value=0.30, step=0.05)
        seed = st.number_input("Random seed", min_value=0, value=0, step=1)
        run = st.button("▶ Run analysis", type="primary", use_container_width=True)

    # Cache the pipeline run keyed on the plain-dict config.
    cached = st.cache_data(show_spinner=False)(_run_pipeline_cached)

    # Run on click, or on first load (synthetic default) for an instant demo.
    do_run = run or ("results" not in st.session_state)
    if do_run:
        try:
            cfg = _build_config(city, mode, resolution_m, grid_n, budget,
                                max_area_frac, seed, gee_project)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Invalid configuration: {exc}")
            return
        with st.spinner(f"Running the {mode} pipeline for {city}…"):
            try:
                results = cached(cfg.to_dict())
                st.session_state["results"] = results
                st.session_state["cfg_dict"] = cfg.to_dict()
            except Exception as exc:  # noqa: BLE001
                # Robust fallback: if GEE (or anything) failed, retry synthetic.
                st.warning(f"Pipeline failed ({exc!r}). Falling back to synthetic.")
                st.text(traceback.format_exc())
                try:
                    fallback = _build_config(city, "synthetic", resolution_m,
                                             grid_n, budget, max_area_frac, seed,
                                             None)
                    results = cached(fallback.to_dict())
                    st.session_state["results"] = results
                    st.session_state["cfg_dict"] = fallback.to_dict()
                except Exception as exc2:  # noqa: BLE001
                    st.error(f"Synthetic fallback also failed: {exc2}")
                    st.text(traceback.format_exc())
                    return

    results = st.session_state.get("results")
    if not results:
        st.info("Use the sidebar to configure a run, then press **Run analysis**.")
        return

    cfg_dict = st.session_state.get("cfg_dict", {})
    out_dir = cfg_dict.get("output_dir", "outputs")
    fs = results.get("fs")
    if fs is None or not getattr(fs, "has", lambda *_: False)(dm.LST):
        st.error("The pipeline did not return a valid FeatureStack.")
        return

    st.success(
        f"Completed: **{cfg_dict.get('city', city)}** · mode "
        f"**{cfg_dict.get('mode', mode)}** · grid {tuple(getattr(fs, 'shape', ()))}.")

    # ----- the four PS-1 deliverables -----
    _render_hotspot_panel(results, fs, out_dir)
    st.divider()
    _render_driver_panel(results, fs, out_dir)
    st.divider()
    _render_model_panel(results)
    st.divider()
    _render_optimizer_panel(results, fs, out_dir)
    st.divider()
    _render_report_download(results)

    with st.expander("Run configuration (provenance)"):
        st.json(cfg_dict)


if __name__ == "__main__":
    main()
