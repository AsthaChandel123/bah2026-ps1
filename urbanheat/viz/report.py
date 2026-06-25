"""urbanheat.viz.report — assemble the 4 PS-1 deliverables into a report.

:func:`generate_report` takes the ``results`` dict produced by the CLI pipeline
(``cli.run_pipeline``) and writes a **professional Markdown report** plus a simple,
**self-contained HTML** version under the output directory, embedding the figures
rendered by :mod:`urbanheat.viz.maps`.

The report covers the four PS-1 deliverables explicitly:

1. **Heat-stress hotspot map + statistics** (hotspot area %, priority-class split).
2. **Ranked driver attribution** across the four families
   (LULC / morphology / vegetation / atmosphere).
3. **Validated model metrics** — spatial-CV RMSE/R² (+ the full panel) and
   physics-consistency.
4. **Optimized interventions table** — type, location, area, cost, estimated °C
   reduction — plus city-wide totals.

A final **methods & datasets / robustness** section cites the ≥30 cross-verifying
entries from :mod:`urbanheat.fusion.robustness`.

Pure **stdlib** only (``os``, ``html``, ``datetime``) — no pandas/matplotlib here;
figures are produced upstream and merely linked. Numbers are read defensively so a
partial pipeline still yields a coherent (if abbreviated) report.

References: ARCHITECTURE §11.8 ``build_report``, §1 deliverables table.
"""

from __future__ import annotations

import html
import os
from datetime import datetime, timezone
from typing import Any, Sequence

from urbanheat.constants import (
    HOTSPOT_LEGEND,
    ROBUSTNESS_SUMMARY,
    VALIDATION_ANCHORS,
)


# ===========================================================================
# public entry point
# ===========================================================================
def generate_report(
    results: dict[str, Any],
    out_dir: str = "outputs",
    basename: str = "urbanheat_report",
) -> str:
    """Assemble and write the PS-1 report; return the Markdown path.

    Parameters
    ----------
    results
        The pipeline results dict. Recognised (all optional) keys:
        ``city``, ``config`` (a ``Config`` or its ``to_dict()``),
        ``hotspot_stats`` (dict), ``attribution`` (dict/list/frame of family or
        per-driver contributions), ``metrics`` (dict of metric->value or a
        per-fold frame), ``physics`` (dict of consistency checks),
        ``portfolio`` (list/frame of placements), ``optimization`` (dict with
        ``city_dC``/``exposure_reduction``/totals), ``robustness`` (dict from
        :func:`fusion.robustness.robustness_report`), and ``figures`` (dict of
        ``name -> path`` for embedding).
    out_dir
        Directory to write into (created if missing).
    basename
        Filename stem; writes ``<basename>.md`` and ``<basename>.html``.

    Returns
    -------
    str
        Absolute path to the written Markdown report. The HTML sits beside it.
    """
    os.makedirs(out_dir, exist_ok=True)

    city = results.get("city") or _city_from_config(results.get("config")) or "study area"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sections: list[str] = []
    sections.append(_md_header(city, generated, results))
    sections.append(_md_deliverable1_hotspots(results))
    sections.append(_md_deliverable2_attribution(results))
    sections.append(_md_deliverable3_metrics(results))
    sections.append(_md_deliverable4_interventions(results))
    sections.append(_md_robustness(results))
    sections.append(_md_footer())
    markdown = "\n\n".join(s for s in sections if s)

    md_path = os.path.join(out_dir, f"{basename}.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(markdown)

    html_path = os.path.join(out_dir, f"{basename}.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(_html_document(city, generated, markdown, results, out_dir))

    return os.path.abspath(md_path)


# ===========================================================================
# Markdown section builders
# ===========================================================================
def _md_header(city: str, generated: str, results: dict[str, Any]) -> str:
    cfg = _config_dict(results.get("config"))
    mode = cfg.get("mode", results.get("mode", "synthetic"))
    start = cfg.get("start_date", "")
    end = cfg.get("end_date", "")
    res = cfg.get("resolution_m", "")
    window = f"{start} -> {end}" if start and end else "n/a"
    return (
        f"# Urban Heat Analysis Report — {city}\n\n"
        f"*ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 1*  \n"
        f"**Generated:** {generated}  \n"
        f"**Mode:** `{mode}`  |  **Window:** {window}  |  "
        f"**Resolution:** {res} m\n\n"
        "This report assembles the four PS-1 deliverables: (1) heat-stress hotspot "
        "map + statistics, (2) ranked driver attribution, (3) a spatially-validated "
        "AI/ML model with physics-consistency, and (4) an optimized cooling-"
        "intervention strategy with per-site estimated temperature reduction.\n\n"
        "---"
    )


def _md_deliverable1_hotspots(results: dict[str, Any]) -> str:
    stats = results.get("hotspot_stats") or {}
    figs = results.get("figures") or {}
    lines = ["## 1. Heat-Stress Hotspots\n"]

    hp = _first_num(stats, ["hotspot_area_pct", "hotspot_pct", "hotspot_fraction_pct"])
    if hp is None:
        frac = _first_num(stats, ["hotspot_fraction", "hotspot_frac"])
        hp = frac * 100.0 if frac is not None else None
    if hp is not None:
        lines.append(f"- **Hotspot area:** {hp:.1f}% of the AOI flagged as a surface "
                     "heat hotspot (LST >= P90 AND Getis-Ord Gi* z >= 1.96).")
    mean_lst = _first_num(stats, ["mean_lst", "lst_mean"])
    max_lst = _first_num(stats, ["max_lst", "lst_max"])
    if mean_lst is not None:
        extra = f" (max {max_lst:.1f} degC)" if max_lst is not None else ""
        lines.append(f"- **Mean LST:** {mean_lst:.1f} degC{extra}.")
    suhii = _first_num(stats, ["suhii_mean", "mean_suhii", "suhii"])
    if suhii is not None:
        lines.append(f"- **Surface UHI intensity (SUHII):** {suhii:.2f} degC vs the "
                     "LCZ-D rural reference.")

    # priority-class breakdown table (if provided)
    class_split = stats.get("priority_class_pct") or stats.get("class_pct")
    if isinstance(class_split, dict) and class_split:
        lines.append("\n**Priority-class distribution (5-class legend):**\n")
        lines.append("| Class | Range | Share |")
        lines.append("|---|---|---|")
        for cls in HOTSPOT_LEGEND:
            name = cls["name"]
            pct = class_split.get(name, class_split.get(name.lower()))
            pct_s = f"{float(pct):.1f}%" if pct is not None else "-"
            lines.append(f"| {name} | {cls['min']}-{cls['max']} | {pct_s} |")
    else:
        lines.append("\nThe priority surface fuses the surface-heat hazard with the "
                     "Heat Vulnerability Index (HVI) into the 5-class "
                     "Low/Moderate/High/Severe/Extreme legend.")

    fig = figs.get("hotspots") or figs.get("priority")
    if fig:
        lines.append(f"\n![Heat-stress hotspot map]({_rel(fig)})")
    return "\n".join(lines)


def _md_deliverable2_attribution(results: dict[str, Any]) -> str:
    attribution = results.get("attribution")
    figs = results.get("figures") or {}
    lines = ["## 2. Driver Attribution (LULC / morphology / vegetation / atmosphere)\n"]
    lines.append("Ranked %-contribution of the four PS-1 driver families to LST, from "
                 "mean(|SHAP|) on the trained model (cross-checked by variance "
                 "partitioning where available). No driver claim is made without "
                 ">=2 agreeing methods.\n")

    rows = _attribution_rows(attribution)
    if rows:
        lines.append("| Rank | Driver family | Contribution |")
        lines.append("|---|---|---|")
        total = sum(v for _, v in rows) or 1.0
        for i, (label, val) in enumerate(rows, start=1):
            # render as a percentage if values look like shares; else raw
            pct = val / total * 100.0
            lines.append(f"| {i} | {label} | {pct:.1f}% |")
        top_label = rows[0][0]
        lines.append(f"\n**Leading driver family:** **{top_label}**.")
    else:
        lines.append("_Attribution table not available in this run._")

    fig = figs.get("driver_attribution") or figs.get("attribution")
    if fig:
        lines.append(f"\n![Ranked driver attribution]({_rel(fig)})")
    return "\n".join(lines)


def _md_deliverable3_metrics(results: dict[str, Any]) -> str:
    metrics = _metrics_dict(results.get("metrics"))
    physics = results.get("physics") or {}
    lines = ["## 3. Validated AI/ML Model\n"]
    lines.append("Headline validation uses **spatial block cross-validation** (not "
                 "leaky random CV); the full metric panel is reported as mean across "
                 "folds.\n")

    if metrics:
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        order = ["rmse", "mae", "bias", "ubrmse", "r2", "nse", "ccc", "kge"]
        labels = {"rmse": "RMSE (degC)", "mae": "MAE (degC)", "bias": "Bias (degC)",
                  "ubrmse": "ubRMSE (degC)", "r2": "R^2", "nse": "NSE",
                  "ccc": "Lin's CCC", "kge": "KGE"}
        shown = set()
        for k in order:
            if k in metrics:
                lines.append(f"| {labels[k]} | {_fmt(metrics[k])} |")
                shown.add(k)
        for k, v in metrics.items():
            if k not in shown and isinstance(v, (int, float)):
                lines.append(f"| {k} | {_fmt(v)} |")
        # contextualise vs literature anchors
        r2 = metrics.get("r2")
        if isinstance(r2, (int, float)):
            anc = VALIDATION_ANCHORS.get("extra_trees_lst_r2")
            lines.append(f"\nLiterature anchor (Extra-Trees LST): R^2 ~= {anc}; "
                         f"RMSE ~= {VALIDATION_ANCHORS.get('extra_trees_lst_rmse_C')} degC.")
    else:
        lines.append("_Spatial-CV metrics not available in this run._")

    if physics:
        lines.append("\n**Physics-consistency checks:**\n")
        seb = _first_num(physics, ["seb_closure", "seb_residual", "seb_closure_wm2"])
        if seb is not None:
            lines.append(
                f"- SEB closure residual ~ {_fmt(seb)} W/m^2 (diagnostic of "
                "temperature-dependent flux closure under default exchange "
                "coefficients; not driven to zero on the synthetic stack).")
        signs = physics.get("sign_audit_ok")
        if signs is not None:
            lines.append(f"- Driver-sign audit (SHAP/ALE vs SEB table): "
                         f"{'all signs consistent' if signs else 'violations flagged'}.")
        resid_moran = _first_num(physics, ["residual_moran", "residual_autocorrelation"])
        if resid_moran is not None:
            lines.append(f"- Residual spatial autocorrelation (Moran's I): "
                         f"{_fmt(resid_moran)} (near 0 = well-specified).")
    return "\n".join(lines)


def _md_deliverable4_interventions(results: dict[str, Any]) -> str:
    portfolio_src = results.get("portfolio")
    if portfolio_src is None:
        portfolio_src = _nested(results, "optimization", "portfolio")
    if portfolio_src is None:
        portfolio_src = _nested(results, "optimization", "placements")
    portfolio = _portfolio_rows(portfolio_src)
    opt = results.get("optimization") or {}
    figs = results.get("figures") or {}
    lines = ["## 4. Optimized Cooling-Intervention Strategy\n"]
    lines.append("A lazy-greedy submodular optimizer ((1 - 1/e) ~= 0.63 guarantee) "
                 "selects a ranked portfolio of typed, placed interventions under "
                 "budget / area / equity (population x HVI) constraints. ΔT is the "
                 "estimated **surface** temperature reduction per site.\n")

    if portfolio:
        lines.append("| # | Type | Location (x, y) | Area (m^2) | Cost | Delta T (degC) |")
        lines.append("|---|---|---|---|---|---|")
        for i, p in enumerate(portfolio[:25], start=1):
            loc = (f"{p['x']:.0f}, {p['y']:.0f}" if p.get("x") is not None
                   and p.get("y") is not None else "-")
            area = _fmt(p.get("area"))
            cost = _fmt(p.get("cost"))
            dC = _fmt(p.get("delta_C"))
            lines.append(f"| {i} | {p['type']} | {loc} | {area} | {cost} | {dC} |")
        if len(portfolio) > 25:
            lines.append(f"\n_({len(portfolio) - 25} further sites omitted for brevity.)_")
    else:
        lines.append("_Optimized portfolio not available in this run._")

    # city-wide totals
    city_dC = _first_num(opt, ["city_dC", "city_dc", "total_dC", "citywide_dC"])
    expo = _first_num(opt, ["exposure_reduction", "pop_exposure_reduction"])
    tot_cost = _first_num(opt, ["total_cost", "cost_total"])
    n_sites = opt.get("n_sites", len(portfolio) if portfolio else None)
    totals = []
    if n_sites is not None:
        totals.append(f"**{n_sites}** sites selected")
    if city_dC is not None:
        totals.append(f"city-wide mean cooling **{city_dC:.2f} degC**")
    if expo is not None:
        totals.append(f"population-heat-exposure reduction **{expo:.1f}%**")
    if tot_cost is not None:
        totals.append(f"total cost **{tot_cost:,.0f}**")
    if totals:
        lines.append("\n**Portfolio totals:** " + "; ".join(totals) + ".")

    fig = figs.get("interventions")
    if fig:
        lines.append(f"\n![Optimized intervention placements]({_rel(fig)})")
    return "\n".join(lines)


def _md_robustness(results: dict[str, Any]) -> str:
    rob = results.get("robustness") or {}
    methods = rob.get("methods") or {}
    total = methods.get("total_entries") or rob.get("n_methods_total") or \
        ROBUSTNESS_SUMMARY.get("total_matrix_entries", 35)
    active = methods.get("n_active") or rob.get("n_methods_active")
    n_data = methods.get("n_data_sources")
    n_meth = methods.get("n_methods")

    lines = ["## 5. Methods, Datasets & Robustness\n"]
    head = (f"The pipeline is backed by a **{total}-entry cross-verification matrix** "
            "(research/09) — every source both fills others' gaps and is itself "
            "verified by an independent source.")
    lines.append(head)
    bullets = []
    if n_data is not None and n_meth is not None:
        bullets.append(f"{n_data} datasets + {n_meth} analytical methods")
    if active is not None:
        bullets.append(f"{active} active in this run")
    bullets.append(f"{ROBUSTNESS_SUMMARY.get('lst_sensors', 5)} LST sensors, "
                   f"{ROBUSTNESS_SUMMARY.get('lulc_products', 4)} LULC products, "
                   f"{ROBUSTNESS_SUMMARY.get('footprint_sources', 4)} footprint sources, "
                   f"{ROBUSTNESS_SUMMARY.get('met_sources', 3)} reanalyses")
    lines.append("- " + "\n- ".join(bullets))

    unc = rob.get("uncertainty_layers") or []
    agr = rob.get("agreement_layers") or []
    if unc or agr:
        lines.append(f"\nHonest-confidence layers present: "
                     f"{len(unc)} uncertainty + {len(agr)} agreement layers "
                     "(every deliverable map ships with a paired uncertainty map).")
    lst_unc = rob.get("lst_uncertainty_mean")
    if lst_unc is not None:
        lines.append(f"Mean LST 1-sigma fusion uncertainty: {_fmt(lst_unc)} degC.")

    narrative = rob.get("narrative")
    if narrative:
        lines.append(f"\n> {narrative}")
    return "\n".join(lines)


def _md_footer() -> str:
    return ("---\n\n"
            "*Generated by `urbanheat` (physics-informed, multi-satellite "
            "geospatial AI/ML for urban heat). Synthetic-mode figures are "
            "illustrative of the pipeline, not a substitute for calibrated data.*")


# ===========================================================================
# HTML rendering (tiny, self-contained markdown -> HTML)
# ===========================================================================
def _html_document(city: str, generated: str, markdown: str,
                   results: dict[str, Any], out_dir: str) -> str:
    body = _markdown_to_html(markdown, out_dir)
    title = html.escape(f"Urban Heat Report — {city}")
    style = (
        "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
        "max-width:920px;margin:2rem auto;padding:0 1rem;color:#1a1a1a;line-height:1.55}"
        "h1{border-bottom:3px solid #d7191c;padding-bottom:.3rem}"
        "h2{margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.2rem;color:#b3001b}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0}"
        "th,td{border:1px solid #ccc;padding:.4rem .6rem;text-align:left;font-size:.92rem}"
        "th{background:#f5f5f5}"
        "img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:4px;margin:.5rem 0}"
        "blockquote{border-left:4px solid #fdae61;margin:1rem 0;padding:.3rem 1rem;"
        "background:#fff8ef;color:#444}"
        "code{background:#f0f0f0;padding:.1rem .3rem;border-radius:3px}"
    )
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{title}</title>\n<style>{style}</style>\n</head>\n<body>\n"
        f"{body}\n</body>\n</html>\n"
    )


def _markdown_to_html(markdown: str, out_dir: str) -> str:
    """A minimal, dependency-free Markdown -> HTML converter for this report.

    Supports the subset the report uses: ATX headers, pipe tables, images,
    bold/inline-code, blockquotes, ``-`` lists, ``---`` rules and paragraphs.
    """
    out: list[str] = []
    lines = markdown.split("\n")
    i = 0
    n = len(lines)
    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # table block
        if stripped.startswith("|") and i + 1 < n and set(lines[i + 1].strip()) <= set("|-: "):
            close_list()
            header = [c.strip() for c in stripped.strip("|").split("|")]
            out.append("<table>\n<thead><tr>"
                       + "".join(f"<th>{_inline(c)}</th>" for c in header)
                       + "</tr></thead>\n<tbody>")
            i += 2
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</tbody></table>")
            continue

        if not stripped:
            close_list()
            i += 1
            continue
        if stripped == "---":
            close_list()
            out.append("<hr>")
            i += 1
            continue
        if stripped.startswith("### "):
            close_list(); out.append(f"<h3>{_inline(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            close_list(); out.append(f"<h2>{_inline(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            close_list(); out.append(f"<h1>{_inline(stripped[2:])}</h1>")
        elif stripped.startswith("> "):
            close_list(); out.append(f"<blockquote>{_inline(stripped[2:])}</blockquote>")
        elif stripped.startswith("!["):
            close_list(); out.append(_image_html(stripped, out_dir))
        elif stripped.startswith("- "):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{_inline(stripped[2:])}</li>")
        else:
            close_list(); out.append(f"<p>{_inline(stripped)}</p>")
        i += 1
    close_list()
    return "\n".join(out)


def _inline(text: str) -> str:
    """Escape + apply inline bold/code; embedded images handled separately."""
    if text.startswith("![") and "](" in text:
        return text  # left for _image_html via caller; shouldn't reach here normally
    esc = html.escape(text)
    # bold **x**
    import re
    esc = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc)
    esc = re.sub(r"`(.+?)`", r"<code>\1</code>", esc)
    return esc


def _image_html(md: str, out_dir: str) -> str:
    """Convert a Markdown image ``![alt](path)`` to an <img>, path relative to HTML."""
    try:
        alt = md[md.index("[") + 1: md.index("]")]
        path = md[md.index("(") + 1: md.rindex(")")]
        return f'<img src="{html.escape(path)}" alt="{html.escape(alt)}">'
    except Exception:  # pragma: no cover - defensive
        return f"<p>{html.escape(md)}</p>"


# ===========================================================================
# coercion / formatting helpers (defensive — accept dict / frame / list)
# ===========================================================================
def _rel(path: str) -> str:
    """Relative basename for embedding (figures sit alongside the report)."""
    return os.path.basename(path) if isinstance(path, str) else str(path)


def _fmt(v: Any) -> str:
    """Format a number compactly; pass through non-numbers / None as '-'."""
    if v is None:
        return "-"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return f"{v:,}"
    if isinstance(v, float):
        if v != v:  # NaN
            return "-"
        if v == int(v) and abs(v) < 1e12:  # whole number -> grouped integer
            return f"{int(v):,}"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if v != 0 and abs(v) < 0.01:
            return f"{v:.3g}"
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def _first_num(d: dict[str, Any], keys: Sequence[str]) -> float | None:
    """Return the first present numeric value among ``keys`` in dict ``d``."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d:
            v = d[k]
            try:
                f = float(v)
                if f == f:  # not NaN
                    return f
            except (TypeError, ValueError):
                continue
    return None


def _nested(d: dict[str, Any], *keys: str) -> Any:
    """Safely walk nested dicts; return None if any level missing."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _config_dict(cfg: Any) -> dict[str, Any]:
    """Normalise a Config / dict / None into a plain dict."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    to_dict = getattr(cfg, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:  # pragma: no cover - defensive
            return {}
    return {}


def _city_from_config(cfg: Any) -> str | None:
    return _config_dict(cfg).get("city")


def _metrics_dict(metrics: Any) -> dict[str, float]:
    """Coerce metrics (dict, or a per-fold frame) to a flat dict of means."""
    if metrics is None:
        return {}
    if isinstance(metrics, dict):
        # flatten any nested {'mean':..} forms
        flat: dict[str, float] = {}
        for k, v in metrics.items():
            if isinstance(v, dict) and "mean" in v:
                v = v["mean"]
            try:
                flat[str(k).lower()] = float(v)
            except (TypeError, ValueError):
                continue
        return flat
    # pandas DataFrame -> column means
    mean_fn = getattr(metrics, "mean", None)
    cols = getattr(metrics, "columns", None)
    if callable(mean_fn) and cols is not None:
        try:
            means = metrics.mean(numeric_only=True)
            return {str(k).lower(): float(v) for k, v in means.items()}
        except Exception:  # pragma: no cover - defensive
            return {}
    return {}


def _attribution_rows(attribution: Any) -> list[tuple[str, float]]:
    """Normalise attribution into a descending list of (family/driver, value)."""
    rows: list[tuple[str, float]] = []
    if attribution is None:
        return rows
    if isinstance(attribution, dict):
        # could be {'families': {...}} / {'families': [..]} / a flat {label: value}
        src = attribution.get("families") if "families" in attribution else attribution
        if isinstance(src, dict):
            for k, v in src.items():
                try:
                    rows.append((str(k), float(v)))
                except (TypeError, ValueError):
                    continue
        elif isinstance(src, (list, tuple)):
            rows.extend(_rows_from_dicts(src))
    elif isinstance(attribution, (list, tuple)) and attribution and \
            isinstance(attribution[0], dict):
        rows.extend(_rows_from_dicts(attribution))
    else:
        cols = getattr(attribution, "columns", None)
        if cols is not None:  # DataFrame
            try:
                colnames = list(cols)
                label_col = next((c for c in ("family", "feature", "driver", "name")
                                  if c in colnames), colnames[0])
                value_col = next((c for c in ("pct_contribution", "mean_abs_shap",
                                              "r2_share", "importance", "value")
                                  if c in colnames), colnames[-1])
                for lab, val in zip(attribution[label_col].tolist(),
                                    attribution[value_col].tolist()):
                    rows.append((str(lab), float(val)))
            except Exception:  # pragma: no cover - defensive
                return []
        elif isinstance(attribution, (list, tuple)):
            for item in attribution:
                try:
                    rows.append((str(item[0]), float(item[1])))
                except Exception:
                    continue
    rows.sort(key=lambda kv: kv[1], reverse=True)
    return rows


def _rows_from_dicts(items: Sequence[Any]) -> list[tuple[str, float]]:
    """Rows from a list of attribution dicts like
    ``[{'family':..,'pct_contribution':..}, ...]`` or ``(label, value)`` pairs."""
    out: list[tuple[str, float]] = []
    for item in items:
        if isinstance(item, dict):
            label = item.get("family") or item.get("feature") or item.get("name")
            val = item.get("pct_contribution")
            if val is None:
                val = item.get("mean_abs_shap", item.get("importance", item.get("value")))
            if label is not None and val is not None:
                try:
                    out.append((str(label), float(val)))
                except (TypeError, ValueError):
                    continue
        else:
            try:
                out.append((str(item[0]), float(item[1])))
            except Exception:
                continue
    return out


def _portfolio_rows(portfolio: Any) -> list[dict[str, Any]]:
    """Normalise an optimizer portfolio into a list of plain dicts."""
    if portfolio is None:
        return []
    records: list[Any]
    cols = getattr(portfolio, "columns", None)
    if cols is not None:  # pandas / geopandas
        try:
            records = portfolio.to_dict("records")
        except Exception:  # pragma: no cover - defensive
            return []
    elif isinstance(portfolio, (list, tuple)):
        records = list(portfolio)
    else:
        return []

    out: list[dict[str, Any]] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        x = r.get("x", r.get("lon", r.get("longitude")))
        y = r.get("y", r.get("lat", r.get("latitude")))
        if (x is None or y is None):
            geom = r.get("geometry")
            x = getattr(geom, "x", x)
            y = getattr(geom, "y", y)
        out.append({
            "type": str(r.get("type", r.get("intervention_type",
                        r.get("intervention", "intervention")))),
            "x": _maybe_float(x),
            "y": _maybe_float(y),
            "area": r.get("area", r.get("area_m2")),
            "cost": r.get("cost"),
            "delta_C": r.get("delta_C", r.get("delta_c", r.get("dC",
                       r.get("estimated_dC", r.get("estimated_delta_lst_C"))))),
        })
    return out


def _maybe_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


__all__ = ["generate_report"]
