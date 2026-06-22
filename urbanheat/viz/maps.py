"""urbanheat.viz.maps — static (and optional interactive) maps of the deliverables.

Renders the PS-1 map products from a :class:`FeatureStack` and the optimizer
result, saving PNGs under ``outputs/``. Uses the **colour-blind-safe** legends and
ramps defined ONCE in :mod:`urbanheat.constants` (``HOTSPOT_LEGEND``,
``LST_COLOR_RAMP``) so every map is consistent with the report.

Heavy/optional libs are **lazy**: ``matplotlib`` is imported inside each plotting
function and, if it is missing, the function **warns and returns ``None``** rather
than raising — the synthetic pipeline still completes (just without figures).
``folium`` / ``geemap`` are only touched by :func:`interactive_map`.

Two naming sets are exposed for the same renderers so callers can use either the
task-brief names (``plot_lst``/``plot_hotspots``/``plot_driver_attribution``/
``plot_interventions``/``plot_layer``) or the ARCHITECTURE §11.8 contract names
(``lst_map``/``hotspot_map``/``driver_map``/``intervention_map``).

References: ARCHITECTURE §11.8; ``constants.HOTSPOT_LEGEND``, ``LST_COLOR_RAMP``.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from urbanheat.constants import HOTSPOT_LEGEND, LST_COLOR_RAMP
from urbanheat.datamodel import (
    HOTSPOT_MASK,
    HVI,
    LST,
    PRIORITY_SCORE,
    SUHII,
)

if TYPE_CHECKING:  # pragma: no cover - hints only
    from urbanheat.datamodel import FeatureStack


# ---------------------------------------------------------------------------
# lazy matplotlib + helpers
# ---------------------------------------------------------------------------
def _import_mpl() -> Any:
    """Import matplotlib (Agg backend) lazily; return the ``pyplot`` module or None.

    On ImportError, warns and returns ``None`` so callers degrade gracefully.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)  # headless: no display needed
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:  # pragma: no cover - environment dependent
        warnings.warn(
            f"matplotlib unavailable ({exc!r}); skipping static map. "
            "Install matplotlib to render figures.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None


def _ensure_dir(out_dir: str) -> str:
    """Create ``out_dir`` if needed; return it."""
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _listed_cmap(hex_colors: Sequence[str]) -> Any:
    """Build a matplotlib ListedColormap from hex strings (mpl already imported)."""
    from matplotlib.colors import ListedColormap
    return ListedColormap(list(hex_colors))


def _extent(fs: "FeatureStack") -> list[float] | None:
    """imshow extent ``[xmin, xmax, ymin, ymax]`` from the stack bounds, or None."""
    try:
        xmin, ymin, xmax, ymax = fs.bounds
        return [float(xmin), float(xmax), float(ymin), float(ymax)]
    except Exception:  # pragma: no cover - defensive
        return None


def _finite_or_none(arr: np.ndarray) -> bool:
    """True if the array has at least one finite value."""
    return bool(np.isfinite(np.asarray(arr, dtype=float)).any())


# ---------------------------------------------------------------------------
# 1. LST map (continuous, YlOrRd ramp)
# ---------------------------------------------------------------------------
def plot_lst(
    fs: "FeatureStack",
    out_dir: str = "outputs",
    var: str = LST,
    title: str | None = None,
    filename: str = "lst.png",
) -> str | None:
    """Render the LST (or any continuous thermal layer) with ``LST_COLOR_RAMP``.

    Saves ``<out_dir>/<filename>`` and returns its path (or ``None`` if matplotlib
    is missing or the layer is absent/empty). [ARCHITECTURE §11.8 ``lst_map``]
    """
    plt = _import_mpl()
    if plt is None:
        return None
    if not fs.has(var):
        warnings.warn(f"plot_lst: layer {var!r} absent; skipping.", RuntimeWarning, stacklevel=2)
        return None
    data = np.asarray(fs.get(var), dtype=float)
    if not _finite_or_none(data):
        warnings.warn(f"plot_lst: layer {var!r} all-NaN; skipping.", RuntimeWarning, stacklevel=2)
        return None

    _ensure_dir(out_dir)
    cmap = _listed_cmap(LST_COLOR_RAMP)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, cmap=cmap, extent=_extent(fs), origin="upper", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Land Surface Temperature (degC)")
    ax.set_title(title or f"Land Surface Temperature ({var})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. Hotspot / priority map (5-class legend from constants)
# ---------------------------------------------------------------------------
def plot_hotspots(
    fs: "FeatureStack",
    out_dir: str = "outputs",
    title: str | None = None,
    filename: str = "hotspots.png",
) -> str | None:
    """Render ``PRIORITY_SCORE`` (fallback ``HVI``/``SUHII``/``HOTSPOT_MASK``) with the
    5-class ``constants.HOTSPOT_LEGEND`` (Low..Extreme, RdYlBu-reversed, CB-safe).

    Saves ``<out_dir>/<filename>``; returns its path or ``None``. [§11.8 ``hotspot_map``]
    """
    plt = _import_mpl()
    if plt is None:
        return None

    # pick the best available hotspot layer
    var = None
    for candidate in (PRIORITY_SCORE, HVI, SUHII, HOTSPOT_MASK):
        if fs.has(candidate) and _finite_or_none(fs.get(candidate)):
            var = candidate
            break
    if var is None:
        warnings.warn("plot_hotspots: no priority/HVI/SUHII/mask layer; skipping.",
                      RuntimeWarning, stacklevel=2)
        return None

    data = np.asarray(fs.get(var), dtype=float)
    # normalise to the 0-100 priority scale for the legend bins
    if var == HOTSPOT_MASK:
        scaled = np.where(data > 0, 90.0, 10.0)
    elif var in (HVI,):  # HVI is 0-1
        scaled = data * 100.0
    elif var == SUHII:  # degC anomaly -> rescale to 0-100 over its own range
        lo, hi = np.nanmin(data), np.nanmax(data)
        rng = (hi - lo) or 1.0
        scaled = (data - lo) / rng * 100.0
    else:
        scaled = data

    from matplotlib.colors import BoundaryNorm
    bounds = [HOTSPOT_LEGEND[0]["min"]] + [c["max"] for c in HOTSPOT_LEGEND]
    colors = [c["hex"] for c in HOTSPOT_LEGEND]
    cmap = _listed_cmap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(scaled, cmap=cmap, norm=norm, extent=_extent(fs),
                   origin="upper", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, ticks=[(c["min"] + c["max"]) / 2 for c in HOTSPOT_LEGEND],
                        fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([c["name"] for c in HOTSPOT_LEGEND])
    cbar.set_label("Heat-stress priority class")
    ax.set_title(title or f"Heat-Stress Hotspots / Priority ({var})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Driver attribution bar chart
# ---------------------------------------------------------------------------
def plot_driver_attribution(
    ranked: Any,
    out_dir: str = "outputs",
    title: str = "Driver Attribution (ranked)",
    filename: str = "driver_attribution.png",
) -> str | None:
    """Horizontal bar chart of ranked driver contributions.

    ``ranked`` may be:
      * a ``pandas.DataFrame`` with a label column (``family``/``feature``) and a
        value column (``pct_contribution``/``mean_abs_shap``/``r2_share``), or
      * a ``dict`` ``{label: value}``, or
      * a sequence of ``(label, value)`` pairs.

    Saves ``<out_dir>/<filename>``; returns its path or ``None``. [§11.8 driver view]
    """
    plt = _import_mpl()
    if plt is None:
        return None

    labels, values = _coerce_ranked(ranked)
    if not labels:
        warnings.warn("plot_driver_attribution: empty/unrecognised input; skipping.",
                      RuntimeWarning, stacklevel=2)
        return None

    # sort ascending so the largest bar is on top of a horizontal chart
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(7.5, max(2.5, 0.5 * len(labels) + 1.5)))
    ax.barh(range(len(labels)), values, color="#fdae61", edgecolor="#d7191c")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Contribution")
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.3g}", va="center", ha="left", fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _coerce_ranked(ranked: Any) -> tuple[list[str], list[float]]:
    """Normalise a ranked-attribution object into (labels, values) lists."""
    # pandas DataFrame?
    cols = getattr(ranked, "columns", None)
    if cols is not None:
        try:
            colnames = list(cols)
            label_col = next((c for c in ("family", "feature", "driver", "name")
                              if c in colnames), colnames[0])
            value_col = next((c for c in ("pct_contribution", "mean_abs_shap",
                                          "r2_share", "importance", "value")
                              if c in colnames), colnames[-1])
            labels = [str(x) for x in ranked[label_col].tolist()]
            values = [float(x) for x in ranked[value_col].tolist()]
            return labels, values
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(ranked, dict):
        labels = [str(k) for k in ranked]
        values = [float(v) for v in ranked.values()]
        return labels, values
    try:  # sequence of (label, value)
        labels, values = [], []
        for item in ranked:
            labels.append(str(item[0]))
            values.append(float(item[1]))
        return labels, values
    except Exception:  # pragma: no cover - defensive
        return [], []


# ---------------------------------------------------------------------------
# 4. Intervention placement map (over the hotspot basemap)
# ---------------------------------------------------------------------------
def plot_interventions(
    fs: "FeatureStack",
    opt_result: Any,
    out_dir: str = "outputs",
    title: str = "Optimized Cooling Interventions",
    filename: str = "interventions.png",
) -> str | None:
    """Overlay the optimized intervention portfolio on the LST basemap.

    ``opt_result`` is the dict returned by ``interventions.optimize.optimize`` /
    ``optimize_interventions``; the portfolio is read from
    ``opt_result['portfolio']`` (a ``pandas``/``geopandas`` frame or a list of
    dicts with ``type``, location and ``delta_C`` fields). Points are coloured by
    intervention type and sized by estimated ΔT (degC). A ``delta_lst`` raster, if
    present, is drawn as the basemap instead of LST.

    Saves ``<out_dir>/<filename>``; returns its path or ``None``. [§11.8 ``intervention_map``]
    """
    plt = _import_mpl()
    if plt is None:
        return None

    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    extent = _extent(fs)

    # basemap: prefer a ΔLST field, else LST, else blank
    base = None
    if isinstance(opt_result, dict) and isinstance(opt_result.get("delta_lst"), np.ndarray):
        base = np.asarray(opt_result["delta_lst"], dtype=float)
        base_label = "Delta T (degC, +=cooling)"
        base_cmap = "YlGnBu"
    elif fs.has(LST):
        base = np.asarray(fs.get(LST), dtype=float)
        base_label = "LST (degC)"
        base_cmap = _listed_cmap(LST_COLOR_RAMP)
    if base is not None and _finite_or_none(base):
        im = ax.imshow(base, cmap=base_cmap, extent=extent, origin="upper",
                       aspect="auto", alpha=0.85)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(base_label)

    placements = _coerce_portfolio(opt_result, fs)
    if placements:
        types = sorted({p["type"] for p in placements})
        palette = _type_palette(types)
        max_dC = max((abs(p["delta_C"]) for p in placements), default=1.0) or 1.0
        for t in types:
            pts = [p for p in placements if p["type"] == t]
            xs = [p["x"] for p in pts]
            ys = [p["y"] for p in pts]
            sizes = [30 + 220 * (abs(p["delta_C"]) / max_dC) for p in pts]
            ax.scatter(xs, ys, s=sizes, c=palette[t], edgecolors="black",
                       linewidths=0.5, label=t, alpha=0.9, zorder=5)
        ax.legend(title="Intervention (size ~ Delta T)", fontsize=8, loc="best",
                  framealpha=0.9)
    else:
        warnings.warn("plot_interventions: no placements found in opt_result.",
                      RuntimeWarning, stacklevel=2)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _type_palette(types: Sequence[str]) -> dict[str, str]:
    """Stable categorical colours for intervention types."""
    base = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
            "#e6ab02", "#a6761d", "#666666", "#1f78b4"]
    return {t: base[i % len(base)] for i, t in enumerate(types)}


def _coerce_portfolio(opt_result: Any, fs: "FeatureStack") -> list[dict[str, Any]]:
    """Extract a list of ``{type, x, y, delta_C}`` placements from an optimizer result.

    Handles a dict-with-'portfolio', a bare frame/list, geopandas geometries, or
    explicit ``x``/``y`` / ``row``/``col`` columns. Robust to missing fields.
    """
    portfolio = opt_result
    if isinstance(opt_result, dict):
        portfolio = opt_result.get("portfolio", opt_result.get("placements"))
    if portfolio is None:
        return []

    # geopandas / pandas DataFrame?
    cols = getattr(portfolio, "columns", None)
    records: list[dict[str, Any]]
    if cols is not None:
        try:
            records = portfolio.to_dict("records")  # type: ignore[assignment]
        except Exception:  # pragma: no cover - defensive
            return []
    elif isinstance(portfolio, (list, tuple)):
        records = [dict(r) if not isinstance(r, dict) else r for r in portfolio]  # type: ignore[arg-type]
    else:
        return []

    placements: list[dict[str, Any]] = []
    a, b, c, d, e, f = (fs.transform if fs is not None else (1, 0, 0, 0, -1, 0))
    for r in records:
        if not isinstance(r, dict):
            continue
        t = str(r.get("type", r.get("intervention", "intervention")))
        dC = r.get("delta_C", r.get("delta_c", r.get("delta_lst",
                   r.get("dC", r.get("estimated_dC", 0.0)))))
        try:
            dC = float(dC)
        except Exception:
            dC = 0.0
        x = r.get("x")
        y = r.get("y")
        if x is None or y is None:
            geom = r.get("geometry")
            cx = getattr(geom, "x", None)
            cy = getattr(geom, "y", None)
            if cx is not None and cy is not None:
                x, y = float(cx), float(cy)
            else:
                row = r.get("row")
                col = r.get("col")
                if row is not None and col is not None:
                    cc, rr = float(col) + 0.5, float(row) + 0.5
                    x = a * cc + b * rr + c
                    y = d * cc + e * rr + f
        if x is None or y is None:
            continue
        placements.append({"type": t, "x": float(x), "y": float(y), "delta_C": dC})
    return placements


# ---------------------------------------------------------------------------
# 5. Generic single-layer map
# ---------------------------------------------------------------------------
def plot_layer(
    fs: "FeatureStack",
    var: str,
    out_dir: str = "outputs",
    cmap: str = "viridis",
    title: str | None = None,
    filename: str | None = None,
) -> str | None:
    """Render any canonical driver/coefficient layer with a sensible ramp + colourbar.

    Saves ``<out_dir>/<var>.png`` (or ``filename``); returns its path or ``None``.
    [ARCHITECTURE §11.8 ``driver_map``]
    """
    plt = _import_mpl()
    if plt is None:
        return None
    if not fs.has(var):
        warnings.warn(f"plot_layer: layer {var!r} absent; skipping.", RuntimeWarning, stacklevel=2)
        return None
    data = np.asarray(fs.get(var), dtype=float)
    if not _finite_or_none(data):
        warnings.warn(f"plot_layer: layer {var!r} all-NaN; skipping.", RuntimeWarning, stacklevel=2)
        return None

    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, cmap=cmap, extent=_extent(fs), origin="upper", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(var)
    ax.set_title(title or var)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    path = os.path.join(out_dir, filename or f"{var}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Optional interactive map (folium / geemap lazy)
# ---------------------------------------------------------------------------
def interactive_map(
    fs: "FeatureStack",
    var: str = PRIORITY_SCORE,
    out_dir: str = "outputs",
    filename: str = "interactive_map.html",
) -> str | None:
    """Optional Leaflet (folium) map of a layer, saved as a self-contained HTML.

    Lazily imports ``folium``; warns and returns ``None`` if it is unavailable.
    Reprojection to WGS84 is **not** attempted here (the synthetic grid is a toy
    CRS) — this is a convenience hook; the GEE backend / app provide the real
    geemap/leafmap view. [ARCHITECTURE §11.8 interactive variant]
    """
    try:
        import folium  # noqa: F401  (lazy, optional)
    except Exception as exc:
        warnings.warn(
            f"folium/geemap unavailable ({exc!r}); skipping interactive map.",
            RuntimeWarning, stacklevel=2,
        )
        return None
    if not fs.has(var):
        warnings.warn(f"interactive_map: layer {var!r} absent; skipping.",
                      RuntimeWarning, stacklevel=2)
        return None

    _ensure_dir(out_dir)
    try:
        xmin, ymin, xmax, ymax = fs.bounds
        center = [(ymin + ymax) / 2.0, (xmin + xmax) / 2.0]
        m = folium.Map(location=center, zoom_start=11)
        folium.Rectangle(bounds=[[ymin, xmin], [ymax, xmax]],
                         tooltip=f"AOI ({var})").add_to(m)
        path = os.path.join(out_dir, filename)
        m.save(path)
        return path
    except Exception as exc:  # pragma: no cover - environment dependent
        warnings.warn(f"interactive_map failed: {exc!r}", RuntimeWarning, stacklevel=2)
        return None


# ---------------------------------------------------------------------------
# ARCHITECTURE §11.8 contract aliases (same renderers, contract names)
# ---------------------------------------------------------------------------
def lst_map(fs: "FeatureStack", interactive: bool = False, **kw: Any) -> Any:
    """§11.8 contract alias of :func:`plot_lst` (static) / :func:`interactive_map`."""
    if interactive:
        return interactive_map(fs, var=LST, **kw)
    return plot_lst(fs, **kw)


def hotspot_map(fs: "FeatureStack", interactive: bool = False, **kw: Any) -> Any:
    """§11.8 contract alias of :func:`plot_hotspots` (static) / :func:`interactive_map`."""
    if interactive:
        return interactive_map(fs, var=PRIORITY_SCORE, **kw)
    return plot_hotspots(fs, **kw)


def driver_map(fs: "FeatureStack", var: str, interactive: bool = False, **kw: Any) -> Any:
    """§11.8 contract alias of :func:`plot_layer` (static) / :func:`interactive_map`."""
    if interactive:
        return interactive_map(fs, var=var, **kw)
    return plot_layer(fs, var, **kw)


def intervention_map(fs: "FeatureStack", portfolio: Any, interactive: bool = False,
                     **kw: Any) -> Any:
    """§11.8 contract alias of :func:`plot_interventions`.

    Accepts either a portfolio frame/list directly or a full optimizer-result dict.
    """
    opt_result = portfolio if isinstance(portfolio, dict) else {"portfolio": portfolio}
    return plot_interventions(fs, opt_result, **kw)


__all__ = [
    # task-brief names
    "plot_lst", "plot_hotspots", "plot_driver_attribution",
    "plot_interventions", "plot_layer", "interactive_map",
    # ARCHITECTURE §11.8 contract names
    "lst_map", "hotspot_map", "driver_map", "intervention_map",
]
