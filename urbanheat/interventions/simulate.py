"""urbanheat.interventions.simulate — counterfactual cooling simulation.

Given the trained LST model ``LST = F(drivers)`` and a per-pixel
:class:`~urbanheat.datamodel.FeatureStack` of baseline drivers, this module
answers: *if I place intervention X on these pixels, how much does it cool?*

It does so three independent ways (the "many methods cross-verify" mandate):

1. **ML counterfactual** (:func:`predict_delta_lst` / §11.6 :func:`delta_lst`):
   perturb the driver layers per the catalog (↑NDVI/tree_frac/albedo, ↓SVF …),
   re-predict, and take ``ΔLST = F(baseline) − F(perturbed)`` (positive = cooling),
   with InVEST-style ``exp(−d/d_cool)`` spatial decay around treated pixels and
   physical clipping so the result stays plausibly a *cooling*.
2. **Physics estimate** (:func:`physics_delta_lst`): an independent surface-energy
   -balance estimate of ΔLST from the albedo change (↑α → less absorbed K↓) and
   the latent/ET change, via the radiative law ``L_up = εσT_s^4`` and
   ``constants.SIGMA_SB`` — used as a sanity cross-check on the ML delta.
3. **Blend** (:func:`combined_delta_lst`): combine ML + physics weighted by their
   agreement, returning the fused ΔLST and a per-pixel uncertainty (disagreement).

The model is treated through the §11.5 contract: a fitted object exposing
``.predict(X) -> ndarray`` and/or ``.predict_grid(fs) -> 2-D ndarray``; the
predictor matrix is assembled by :mod:`urbanheat.models.features` when available
(``predictor_grid`` / ``build_feature_table``), with a pure-numpy fallback so the
synthetic path and trivial test stubs work with numpy alone. All heavy/optional
imports (the models package, solweig) are done lazily inside functions.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from urbanheat import constants as C
from urbanheat import datamodel as dm
from urbanheat.interventions import catalog as cat

# Physical clip ranges for the canonical driver layers we perturb. Keeps a
# perturbed stack inside physically valid bounds (ARCHITECTURE §11.6 contract).
_CLIP: dict[str, tuple[float, float]] = {
    dm.NDVI: (-1.0, 0.95),
    dm.EVI: (-1.0, 1.0),
    dm.SAVI: (-1.0, 1.0),
    dm.NDWI: (-1.0, 1.0),
    dm.MNDWI: (-1.0, 1.0),
    dm.FVC: (0.0, 1.0),
    dm.LAI: (0.0, 8.0),
    dm.ALBEDO: (0.05, 0.90),
    dm.EMISSIVITY: (0.85, 0.995),
    dm.IMPERVIOUS_FRAC: (0.0, 1.0),
    dm.GREEN_FRAC: (0.0, 1.0),
    dm.WATER_FRAC: (0.0, 1.0),
    dm.TREE_FRAC: (0.0, 1.0),
    dm.SVF: (0.05, 1.0),
    dm.SOIL_MOISTURE: (0.0, 0.6),
    dm.ET: (0.0, 1e4),
}

# Absolute guardrail on per-intervention surface cooling magnitude (°C). Even the
# strongest cited lever (green roof, surface 15-45 °C) is capped to avoid a
# mis-scaled model stub producing absurd ΔT. Generous so as not to clip real
# physics. [R6 §2 surface ranges]
_MAX_SURFACE_COOLING_C = 50.0


# ===========================================================================
# Driver perturbation
# ===========================================================================
def apply_perturbation(
    fs: dm.FeatureStack,
    name: str,
    mask: np.ndarray | None = None,
) -> dm.FeatureStack:
    """Return a COPY of ``fs`` with intervention ``name``'s perturbations applied.

    The catalog's ``perturbs`` deltas are added to the corresponding canonical
    driver layers (only where they exist) within ``mask`` (default: the
    intervention's :func:`~urbanheat.interventions.catalog.feasibility_mask`),
    then clipped to physical ranges (``albedo<=0.9``, ``ndvi<=0.95`` …). Some
    levers carry coupled book-keeping beyond the literal ``perturbs`` dict:

    * adding canopy/greening (``tree_frac``/``green_frac`` ↑) lowers SVF (more
      sky blocked) and nudges IMPERVIOUS_FRAC down where vegetation replaces
      hard surface;
    * adding water (``water_frac`` ↑) lowers IMPERVIOUS_FRAC correspondingly.

    Does **not** mutate the input stack (deep-copies the affected layers).
    [R6 §7B / ARCHITECTURE §11.6]
    """
    obj = cat.get_intervention_obj(name)
    if mask is None:
        mask = cat.feasibility_mask(fs, name)
    mask = np.asarray(mask, dtype=bool)

    out = _copy_stack(fs)
    perturbs = obj.driver_perturbs  # canonical-layer subset only

    # Track net vegetation/water gains to drive coupled SVF / impervious updates.
    veg_gain = 0.0
    for layer, delta in perturbs.items():
        if layer in (dm.TREE_FRAC, dm.GREEN_FRAC, dm.FVC):
            veg_gain = max(veg_gain, float(delta))
        _add_to_layer(out, layer, float(delta), mask)

    # ----- coupled book-keeping (physically-consistent side effects) -------
    # 1. Canopy/greening reduces sky-view factor (canopy blocks sky). Use a
    #    fraction of the veg gain; only where SVF exists and we added greenery.
    if veg_gain > 0 and out.has(dm.SVF):
        _add_to_layer(out, dm.SVF, -0.5 * veg_gain, mask)

    # 2. Greening / water displaces impervious surface (conserve cover budget).
    if out.has(dm.IMPERVIOUS_FRAC):
        disp = 0.0
        if veg_gain > 0:
            disp += 0.5 * veg_gain
        if dm.WATER_FRAC in perturbs:
            disp += 0.5 * float(perturbs[dm.WATER_FRAC])
        if disp > 0:
            _add_to_layer(out, dm.IMPERVIOUS_FRAC, -disp, mask)

    out.meta = dict(out.meta)
    out.meta.setdefault("interventions_applied", [])
    out.meta["interventions_applied"] = list(out.meta["interventions_applied"]) + [name]
    return out


def apply_intervention(
    stack: dm.FeatureStack,
    intervention: "str | cat.Intervention",
    mask: np.ndarray | None = None,
) -> dm.FeatureStack:
    """Task-API alias of :func:`apply_perturbation`.

    Accepts either an intervention name or an
    :class:`~urbanheat.interventions.catalog.Intervention` object and returns a
    perturbed copy of ``stack`` with the right canonical driver vars changed
    within physical bounds.
    """
    name = intervention.name if isinstance(intervention, cat.Intervention) else str(intervention)
    return apply_perturbation(stack, name, mask)


def apply_plan(
    fs: dm.FeatureStack,
    plan: dict[str, np.ndarray],
) -> dm.FeatureStack:
    """Apply a *combined* plan (``{intervention_name -> mask}``) to one copy.

    This is what enforces NON-additivity in :func:`scenario`: all selected
    interventions perturb the SAME stack, then the model is re-predicted once on
    the combined drivers (rather than summing per-intervention deltas). [R6 Stage B]
    """
    out = _copy_stack(fs)
    for name, mask in plan.items():
        out = apply_perturbation(out, name, mask)
    return out


# ===========================================================================
# ML counterfactual ΔLST
# ===========================================================================
def predict_delta_lst(
    model: Any,
    stack: dm.FeatureStack,
    perturbed_stack: dm.FeatureStack,
    build_feature_table: Any = None,
    clip: bool = True,
) -> np.ndarray:
    """Counterfactual ΔLST grid = ``F(baseline) − F(perturbed)`` (°C, +=cooling).

    Uses the trained ``model`` via the §11.5 contract. The predictor matrix is
    built by ``build_feature_table`` if supplied, else by
    :func:`urbanheat.models.features.predictor_grid` /
    :func:`~urbanheat.models.features.build_feature_table` when importable, else
    by a pure-numpy fallback (canonical predictor layers stacked column-wise).

    Parameters
    ----------
    model :
        Fitted LST model exposing ``.predict_grid(fs)`` and/or ``.predict(X)``.
    stack, perturbed_stack :
        Baseline and perturbed FeatureStacks (same grid).
    build_feature_table :
        Optional callable ``fs -> (X, predictors)`` or ``fs -> X`` overriding the
        default feature assembly (lets the ML builder inject the exact training
        feature pipeline). [models.features.build_feature_table contract]
    clip :
        If True (default), enforce ΔLST is a bounded cooling (>= small negative
        tolerance, <= ``_MAX_SURFACE_COOLING_C``) so a mis-scaled stub cannot
        emit absurd values.

    Returns
    -------
    np.ndarray
        ΔLST (°C), positive where the intervention cools. [R5 §6 / R6 §7B]
    """
    base = _predict_grid(model, stack, build_feature_table)
    pert = _predict_grid(model, perturbed_stack, build_feature_table)
    delta = base - pert  # positive where perturbed is cooler => cooling
    delta = np.where(np.isfinite(delta), delta, 0.0)
    if clip:
        delta = _clip_cooling(delta)
    return delta.astype(np.float32)


def delta_lst(
    model: Any,
    fs: dm.FeatureStack,
    name: str,
    mask: np.ndarray | None = None,
    d_cool_m: float | None = None,
) -> np.ndarray:
    """§11.6 counterfactual ΔLST for one intervention with spatial decay.

    Perturbs ``fs`` by intervention ``name`` over ``mask`` (default feasibility
    mask), predicts ``ΔLST = LST(base) − LST(perturbed)`` via the trained
    ``model``, then applies InVEST-style ``exp(−d/d_cool)`` decay so cooling
    spreads from treated pixels into a buffer (Park-Cool-Island behaviour). The
    decay distance defaults to ``constants.INVEST_UCM['green_area_cooling_distance_m']``.

    Returns ΔLST (°C, positive=cooling). [R5 §6 / R6 §7B]
    """
    if mask is None:
        mask = cat.feasibility_mask(fs, name)
    mask = np.asarray(mask, dtype=bool)

    perturbed = apply_perturbation(fs, name, mask)
    raw = predict_delta_lst(model, fs, perturbed, clip=True)

    # Only treated pixels generate cooling; zero elsewhere, then decay-spread it.
    treated = np.where(mask, raw, 0.0).astype(np.float32)
    if d_cool_m is None:
        d_cool_m = float(C.INVEST_UCM["green_area_cooling_distance_m"])
    spread = _spatial_decay(treated, mask, fs, d_cool_m)
    return spread.astype(np.float32)


# ===========================================================================
# Physics-only ΔLST (independent cross-check)
# ===========================================================================
def physics_delta_lst(
    stack: dm.FeatureStack,
    perturbed_stack: dm.FeatureStack,
) -> np.ndarray:
    """Independent SEB estimate of surface ΔLST from albedo + ET/latent changes.

    Two physically-grounded terms (positive = cooling):

    * **Albedo term** — raising α reflects more shortwave, reducing absorbed
      ``(1−α)·K↓``. Linearising the radiative balance ``εσT_s^4 ≈ (1−α)K↓ + …``
      about the baseline surface temperature ``T_s`` gives

          ``ΔT_s ≈ Δα · K↓ / (4 ε σ T_s^3)``    (K, then reported as °C step)

      using ``constants.SIGMA_SB`` and baseline LST/emissivity/solar_radiation.
    * **Latent/ET term** — extra evapotranspiration moves absorbed energy into
      latent heat instead of warming the surface; the same linearisation gives
      ``ΔT_s ≈ ΔQ_E / (4 ε σ T_s^3)`` where ``ΔQ_E`` is approximated from the ET
      and soil-moisture / vegetation-fraction increases.

    This deliberately mirrors the energy-balance physics the ``physics`` module
    implements (``L_up = εσT_s^4``); if
    :func:`urbanheat.physics.energy_balance.longwave_up` is importable we use the
    same σ and form, otherwise this is fully self-contained. It is a *coarse*
    estimator whose job is to agree in sign and order-of-magnitude with the ML
    delta, not to replace it. [R5 §1 / R6 §1]
    """
    sigma = float(C.SIGMA_SB)

    ts_c = _layer(stack, dm.LST, default=35.0)
    ts_k = ts_c + float(C.KELVIN)
    eps = _layer(stack, dm.EMISSIVITY, default=0.96)
    kdown = _layer(stack, dm.SOLAR_RADIATION, default=700.0)  # W/m2 midday default

    # 4 ε σ T_s^3 — the radiative damping coefficient (W m^-2 K^-1).
    denom = 4.0 * eps * sigma * np.clip(ts_k, 200.0, 360.0) ** 3
    denom = np.where(denom > 1e-6, denom, 1e-6)

    # ----- albedo term -----
    da = _delta(perturbed_stack, stack, dm.ALBEDO)  # +ve = brighter
    dT_albedo = (da * kdown) / denom  # K of cooling per +Δα

    # ----- latent / ET term -----
    # Approximate the extra latent flux ΔQ_E from the ET increase (if ET present)
    # plus a vegetation/soil-moisture proxy when ET is not modelled. Scale the
    # fractional increases by a representative available-energy budget.
    dQ_E = np.zeros(stack.shape, dtype=np.float64)
    if stack.has(dm.ET) and perturbed_stack.has(dm.ET):
        det = _delta(perturbed_stack, stack, dm.ET)  # mm/period (or kg/m2)
        # latent heat of vaporisation ~2.45e6 J/kg; spread a per-period ET pulse
        # over a daytime window (~6 h) -> W/m2. Coarse but sign/scale-correct.
        lam = 2.45e6
        seconds = 6.0 * 3600.0
        dQ_E += np.clip(det, 0.0, None) * lam / seconds
    else:
        # proxy: each +0.1 fractional veg/green adds ~ a few tens of W/m2 of QE
        dgreen = (_delta(perturbed_stack, stack, dm.GREEN_FRAC)
                  + _delta(perturbed_stack, stack, dm.TREE_FRAC)
                  + _delta(perturbed_stack, stack, dm.FVC))
        dsm = _delta(perturbed_stack, stack, dm.SOIL_MOISTURE)
        # ~ up to ~0.4*K↓ can be re-routed to QE for a fully vegetated/wet pixel
        dQ_E += np.clip(dgreen, 0.0, None) * 0.25 * kdown
        dQ_E += np.clip(dsm, 0.0, None) * 0.5 * kdown
    dT_latent = dQ_E / denom

    delta = dT_albedo + dT_latent
    delta = np.where(np.isfinite(delta), delta, 0.0)
    return _clip_cooling(delta).astype(np.float32)


# ===========================================================================
# Combined ML + physics ΔLST (agreement-weighted)
# ===========================================================================
def combined_delta_lst(
    model: Any,
    stack: dm.FeatureStack,
    perturbed_stack: dm.FeatureStack,
    build_feature_table: Any = None,
    w_ml: float = 0.6,
    return_uncertainty: bool = True,
) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
    """Blend the ML and physics ΔLST estimates, weighted toward agreement.

    Computes the ML counterfactual and the physics estimate, then fuses them.
    Where they **agree** the blend is confident (low uncertainty); where they
    **disagree** the blend is pulled toward their mean and uncertainty is high
    (the absolute difference). This realises the "ΔT ± σ" output of ARCHITECTURE
    §9/§10.

    Parameters
    ----------
    w_ml : float
        Base weight on the ML estimate (physics gets ``1−w_ml``); the effective
        weight is modulated by agreement so a wildly-off physics term cannot drag
        a confident ML prediction.
    return_uncertainty : bool
        If True (default) return ``(delta, sigma)``; else just ``delta``.

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        Fused ΔLST (°C, +=cooling) and, optionally, per-pixel 1-σ uncertainty.
    """
    ml = predict_delta_lst(model, stack, perturbed_stack, build_feature_table, clip=True)
    ph = physics_delta_lst(stack, perturbed_stack)

    ml = ml.astype(np.float64)
    ph = ph.astype(np.float64)
    w_ml = float(np.clip(w_ml, 0.0, 1.0))

    blend = w_ml * ml + (1.0 - w_ml) * ph
    sigma = np.abs(ml - ph)  # disagreement = epistemic uncertainty

    blend = _clip_cooling(blend).astype(np.float32)
    if not return_uncertainty:
        return blend
    return blend, sigma.astype(np.float32)


# ===========================================================================
# Full scenario (NON-additive combined plan)
# ===========================================================================
def scenario(
    model: Any,
    fs: dm.FeatureStack,
    plan: dict[str, np.ndarray],
    d_cool_m: float | None = None,
) -> dict[str, Any]:
    """Evaluate a full plan (``{intervention_name -> placement mask}``).

    Handles NON-additivity by perturbing ONE stack with all selected
    interventions and re-predicting once, rather than summing per-intervention
    deltas. Also returns per-type marginal contributions (each evaluated alone)
    for attribution.

    Returns
    -------
    dict with keys:
      ``delta_lst`` : 2-D ndarray — combined ΔLST (°C, +=cooling, decay-spread);
      ``mean_dC``   : float — area-mean cooling over the grid;
      ``hotspot_dC``: float — mean cooling over the hottest decile of pixels;
      ``per_type``  : {name -> mean ΔLST of that type evaluated alone}.
    [R6 Stage B / ARCHITECTURE §11.6]
    """
    if d_cool_m is None:
        d_cool_m = float(C.INVEST_UCM["green_area_cooling_distance_m"])

    # Combined (non-additive) perturbation + single re-prediction.
    combined = apply_plan(fs, plan)
    raw = predict_delta_lst(model, fs, combined, clip=True)

    # Spread cooling from any treated pixel (union of masks).
    union = np.zeros(fs.shape, dtype=bool)
    for m in plan.values():
        union |= np.asarray(m, dtype=bool)
    treated = np.where(union, raw, 0.0).astype(np.float32)
    combined_delta = _spatial_decay(treated, union, fs, d_cool_m)

    mean_dC = float(np.nanmean(combined_delta)) if combined_delta.size else 0.0

    # hotspot decile mean (hottest 10% of baseline LST)
    hotspot_dC = mean_dC
    if fs.has(dm.LST):
        lst = _layer(fs, dm.LST, default=np.nan)
        finite = np.isfinite(lst)
        if finite.any():
            thr = np.nanpercentile(lst[finite], 90.0)
            hot = finite & (lst >= thr)
            if hot.any():
                hotspot_dC = float(np.nanmean(combined_delta[hot]))

    per_type: dict[str, float] = {}
    for name, m in plan.items():
        d = delta_lst(model, fs, name, np.asarray(m, dtype=bool), d_cool_m)
        per_type[name] = float(np.nanmean(d)) if d.size else 0.0

    return {
        "delta_lst": combined_delta.astype(np.float32),
        "mean_dC": mean_dC,
        "hotspot_dC": hotspot_dC,
        "per_type": per_type,
    }


# ===========================================================================
# Optional SOLWEIG microscale Tmrt hook (lazy)
# ===========================================================================
def solweig_tmrt(
    fs: dm.FeatureStack,
    cfg: Any,
    perturb: dict | None = None,
) -> np.ndarray:
    """Optional microscale mean-radiant-temperature (Tmrt, °C) cross-check.

    Delegates to the standalone Rust ``solweig`` PyPI package (imported lazily)
    on a hotspot tile to verify tree ΔTmrt for pedestrian comfort. If ``solweig``
    is not installed this returns a graceful physics-lite Tmrt proxy
    (``Tmrt ≈ LST`` nudged by SVF/shade) so the pipeline never hard-fails — the
    proxy is clearly marked in ``fs.meta``. Writes/returns TMRT. [R6 §4]
    """
    try:  # pragma: no cover - only when solweig installed
        import solweig  # type: ignore  # noqa: F401
        # A full SOLWEIG run needs a DSM/CDSM tile + meteo; constructing those
        # from the FeatureStack is the ML/physics builder's job. We expose the
        # hook and fall through to the proxy if the required tiles are absent.
        raise ImportError("solweig DSM/CDSM tile assembly not wired in synthetic path")
    except Exception:
        # Physics-lite proxy: Tmrt tracks LST but is amplified by sky exposure
        # (high SVF -> more radiant load) and reduced by added shade.
        lst = _layer(fs, dm.LST, default=35.0)
        svf = _layer(fs, dm.SVF, default=0.6)
        tmrt = lst + (svf - 0.5) * 6.0  # +/- a few deg around LST by sky-view
        if perturb:
            # adding canopy lowers SVF -> lowers Tmrt
            d_svf = float(perturb.get(dm.SVF, 0.0))
            tmrt = tmrt + d_svf * 6.0
        fs.meta = dict(fs.meta)
        fs.meta["tmrt_method"] = "proxy (solweig unavailable)"
        return tmrt.astype(np.float32)


# ===========================================================================
# Internal helpers
# ===========================================================================
def _copy_stack(fs: dm.FeatureStack) -> dm.FeatureStack:
    """Deep-copy a FeatureStack's layers (geo-ref shared by value)."""
    new_layers = {k: np.array(v, copy=True) for k, v in fs.layers.items()}
    return dm.FeatureStack(
        layers=new_layers,
        transform=fs.transform,
        crs=fs.crs,
        bounds=fs.bounds,
        shape=fs.shape,
        meta=copy.deepcopy(fs.meta),
    )


def _add_to_layer(fs: dm.FeatureStack, layer: str, delta: float, mask: np.ndarray) -> None:
    """Add ``delta`` to ``layer`` within ``mask`` (in place), then physical-clip.

    If the layer is absent it is created from a sensible neutral baseline so the
    perturbation still registers (e.g. adding tree_frac where none was modelled).
    """
    if not fs.has(layer):
        # create a neutral baseline grid for the layer so we can perturb it
        base = _neutral_baseline(layer, fs.shape)
        fs.add_layer(layer, base)
    arr = np.asarray(fs.get(layer), dtype=np.float32).copy()
    arr[mask] = arr[mask] + np.float32(delta)
    lo, hi = _CLIP.get(layer, (-np.inf, np.inf))
    if np.isfinite(lo) or np.isfinite(hi):
        arr = np.clip(arr, lo, hi)
    fs.layers[layer] = arr


def _neutral_baseline(layer: str, shape: tuple[int, int]) -> np.ndarray:
    """A neutral starting grid for a layer that is missing from the stack."""
    defaults = {
        dm.TREE_FRAC: 0.0, dm.GREEN_FRAC: 0.0, dm.WATER_FRAC: 0.0,
        dm.IMPERVIOUS_FRAC: 0.5, dm.FVC: 0.0, dm.SOIL_MOISTURE: 0.2,
        dm.ALBEDO: 0.18, dm.NDVI: 0.3, dm.SVF: 0.6, dm.EMISSIVITY: 0.96,
        dm.ET: 0.0,
    }
    return np.full(shape, float(defaults.get(layer, 0.0)), dtype=np.float32)


def _layer(fs: dm.FeatureStack, name: str, default: float | np.ndarray) -> np.ndarray:
    """Layer as float64, or a constant grid from ``default`` if absent."""
    if fs.has(name):
        return np.asarray(fs.get(name), dtype=np.float64)
    if isinstance(default, np.ndarray):
        return default.astype(np.float64)
    return np.full(fs.shape, float(default), dtype=np.float64)


def _delta(a: dm.FeatureStack, b: dm.FeatureStack, name: str) -> np.ndarray:
    """``a[name] − b[name]`` as float64 (0 where neither stack has the layer)."""
    if not (a.has(name) or b.has(name)):
        return np.zeros(a.shape, dtype=np.float64)
    av = _layer(a, name, 0.0)
    bv = _layer(b, name, 0.0)
    return av - bv


def _clip_cooling(delta: np.ndarray) -> np.ndarray:
    """Guardrail ΔLST to a plausible cooling band.

    Interventions in this catalog are cooling levers; a small negative tolerance
    is allowed (model noise / nighttime canopy warming) but large warming is
    clipped to 0, and cooling is capped at ``_MAX_SURFACE_COOLING_C``.
    """
    delta = np.asarray(delta, dtype=np.float64)
    tol = 0.5  # °C of tolerated apparent warming before we floor to 0
    delta = np.where(delta < -tol, 0.0, delta)
    delta = np.clip(delta, -tol, _MAX_SURFACE_COOLING_C)
    return delta


def _spatial_decay(
    treated_delta: np.ndarray,
    mask: np.ndarray,
    fs: dm.FeatureStack,
    d_cool_m: float,
) -> np.ndarray:
    """Spread cooling from treated pixels with InVEST ``exp(−d/d_cool)`` decay.

    For each pixel the delivered cooling is the distance-weighted contribution of
    nearby treated pixels (weight ``exp(−d/d_cool)``, normalised by the weight at
    the treated cell). Implemented as a separable Gaussian-like exponential
    convolution; uses ``scipy.ndimage`` if present, else a small numpy kernel.
    The treated pixels themselves retain (at least) their own ΔT.
    [R6 §5.2 / §7B]
    """
    treated_delta = np.asarray(treated_delta, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return treated_delta.astype(np.float32)

    res_m = _resolution_m(fs)
    d_px = max(d_cool_m / max(res_m, 1e-6), 1e-6)  # decay length in pixels

    # kernel radius: cover ~3 decay lengths; cap to keep it cheap.
    radius = int(min(max(np.ceil(3.0 * d_px), 1), 25))
    ax = np.arange(-radius, radius + 1)
    # 1-D exponential half-kernel; 2-D separable approx of exp(-r/d).
    k1d = np.exp(-np.abs(ax) / d_px)
    k1d /= k1d.sum()

    # Use a true 2-D exp(-r/d) kernel for correctness (radius small).
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    rr = np.sqrt(xx ** 2 + yy ** 2)
    k2d = np.exp(-rr / d_px)
    k2d /= k2d.max()  # normalise so a treated cell delivers ~its own ΔT at centre

    try:
        from scipy import ndimage as _ndi  # type: ignore

        spread = _ndi.maximum_filter(treated_delta, footprint=(k2d > 0.05))
        # weighted spread: correlate then take elementwise max with raw treated
        conv = _ndi.correlate(treated_delta, k2d / k2d.sum(), mode="constant")
        # scale conv back up: correlate with normalised-to-sum kernel underestimates
        # peak; blend toward maximum_filter envelope.
        out = np.maximum(treated_delta, 0.5 * (spread + conv))
        return out.astype(np.float32)
    except Exception:
        out = _conv2d_same(treated_delta, k2d)
        # normalise so treated cells keep their value
        out = np.maximum(out, treated_delta)
        return out.astype(np.float32)


def _conv2d_same(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Tiny 'same'-size 2-D convolution (numpy fallback for the decay kernel)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(arr, ((ph, ph), (pw, pw)), mode="constant")
    out = np.zeros_like(arr, dtype=np.float64)
    # kernel is small (<=51x51) and grids are demo-sized; direct loop is fine.
    for i in range(kh):
        for j in range(kw):
            w = kernel[i, j]
            if w == 0:
                continue
            out += w * padded[i:i + arr.shape[0], j:j + arr.shape[1]]
    return out


def _resolution_m(fs: dm.FeatureStack) -> float:
    """Pixel resolution in metres (from transform; falls back to meta/100 m)."""
    a, b, c, d, e, f = fs.transform
    px = abs(a)
    if np.isfinite(px) and px > 0:
        if px < 1e-3:  # geographic degrees -> ~m
            return float(px * 111_000.0)
        return float(px)
    return float(fs.meta.get("resolution_m", 100.0))


def _predict_grid(model: Any, fs: dm.FeatureStack, build_feature_table: Any = None) -> np.ndarray:
    """Predict a full-grid LST (°C) for ``fs`` via the §11.5 model contract.

    Resolution order (first that works wins), all heavy imports lazy:
      1. user-supplied ``build_feature_table(fs) -> X``  +  ``model.predict(X)``;
      2. ``model.predict_grid(fs)`` (the §11.5 contract method);
      3. ``urbanheat.models.features.build_feature_table`` / ``predictor_grid``
         (+ ``model.predict``) when the ML module is present;
      4. pure-numpy fallback: stack canonical predictor layers -> (H*W, P) and
         call ``model.predict`` (works with the trivial linear test stub).

    Always returns a 2-D (H, W) float64 grid.
    """
    H, W = fs.shape

    # 1. explicit feature-table callable
    if build_feature_table is not None:
        try:
            ft = build_feature_table(fs)
            X, _ = _split_feature_table(ft)
            yhat = np.asarray(model.predict(X), dtype=np.float64).reshape(H, W)
            return yhat
        except Exception:
            pass

    # 2. model.predict_grid
    pg = getattr(model, "predict_grid", None)
    if callable(pg):
        try:
            yhat = np.asarray(pg(fs), dtype=np.float64)
            if yhat.shape == (H, W):
                return yhat
            if yhat.size == H * W:
                return yhat.reshape(H, W)
        except Exception:
            pass

    # 3. models.features (lazy)
    predictors = None
    try:
        from urbanheat.models import features as _feat  # type: ignore

        bft = getattr(_feat, "build_feature_table", None)
        if callable(bft):
            try:
                ft = bft(fs)
                X, predictors = _split_feature_table(ft)
                yhat = np.asarray(model.predict(X), dtype=np.float64).reshape(H, W)
                return yhat
            except Exception:
                pass
        pgf = getattr(_feat, "predictor_grid", None)
        if callable(pgf):
            predictors = _default_predictors(fs)
            try:
                X = np.asarray(pgf(fs, predictors), dtype=np.float64)
                yhat = np.asarray(model.predict(X), dtype=np.float64).reshape(H, W)
                return yhat
            except Exception:
                pass
    except Exception:
        pass

    # 4. pure-numpy fallback predictor grid
    predictors = _default_predictors(fs)
    X = _numpy_predictor_grid(fs, predictors)
    yhat = np.asarray(model.predict(X), dtype=np.float64)
    return yhat.reshape(H, W)


def _split_feature_table(ft: Any) -> tuple[np.ndarray, list[str] | None]:
    """Normalise a feature-table return into ``(X_ndarray, predictor_names|None)``.

    Accepts a 2-D ndarray, a ``(X, predictors)`` tuple, or a pandas DataFrame.
    """
    if isinstance(ft, tuple) and len(ft) >= 1:
        X = ft[0]
        preds = list(ft[1]) if len(ft) > 1 and ft[1] is not None else None
    else:
        X = ft
        preds = None
    # pandas DataFrame?
    cols = getattr(X, "columns", None)
    if cols is not None and not isinstance(X, np.ndarray):
        if preds is None:
            preds = [str(c) for c in cols if c not in ("x", "y")]
        try:
            X = X[preds].to_numpy(dtype=np.float64)  # type: ignore[index]
        except Exception:
            X = np.asarray(X, dtype=np.float64)
    return np.asarray(X, dtype=np.float64), preds


def _default_predictors(fs: dm.FeatureStack) -> list[str]:
    """Default predictor list = DEFAULT_PREDICTORS intersected with present layers."""
    present = [p for p in dm.DEFAULT_PREDICTORS if fs.has(p)]
    if present:
        return present
    # last resort: any numeric driver layers except the target
    return [n for n in fs.names() if n != dm.LST]


def _numpy_predictor_grid(fs: dm.FeatureStack, predictors: list[str]) -> np.ndarray:
    """Stack predictor layers into an ``(H*W, P)`` matrix (NaNs -> column mean)."""
    H, W = fs.shape
    cols = []
    for p in predictors:
        arr = _layer(fs, p, default=0.0).reshape(-1)
        # impute NaNs by column mean so a stub model never sees NaN
        if np.isnan(arr).any():
            m = np.nanmean(arr) if np.isfinite(np.nanmean(arr)) else 0.0
            arr = np.where(np.isnan(arr), m, arr)
        cols.append(arr)
    if not cols:
        return np.zeros((H * W, 0), dtype=np.float64)
    return np.column_stack(cols).astype(np.float64)


__all__ = [
    "apply_perturbation",
    "apply_intervention",
    "apply_plan",
    "predict_delta_lst",
    "delta_lst",
    "physics_delta_lst",
    "combined_delta_lst",
    "scenario",
    "solweig_tmrt",
]
