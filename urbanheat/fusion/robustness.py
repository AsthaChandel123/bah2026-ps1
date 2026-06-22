"""urbanheat.fusion.robustness — the cross-verification & data-fusion engine.

This module is the **robustness glue** for the system (ARCHITECTURE §10, research/09).
It does three jobs:

1. **Accounting** — a :data:`METHODS_REGISTRY` reflecting the **35-entry robustness
   matrix** of research/09 §4 (≥30 cross-verifying methods/datasets). Each entry
   records its ``name``, ``type`` (``"data"`` | ``"method"``), ``role`` in the
   system, the ``gap`` it fills and ``verified_by`` (its independent cross-check).
   ``robustness_report`` summarizes this together with whatever per-layer
   agreement / uncertainty actually exists in a :class:`FeatureStack`.

2. **Array-level fusion utilities** — numpy-only primitives that implement the
   research/09 §3 toolbox at the array level so they run identically in the
   synthetic and GEE backends: :func:`weighted_ensemble`,
   :func:`uncertainty_weighted_fuse`, :func:`agreement_map`,
   :func:`ensemble_agreement`, :func:`majority_vote`, :func:`triple_collocation`,
   :func:`reconcile_sensors`, :func:`gap_fill` and :func:`monte_carlo_uncertainty`.

3. **Reporting** — :func:`robustness_report` produces the dict the report/CLI use
   to cite ">=30 cross-verifying methods" with the count actually *active* for a
   given run.

Design rule: **numpy only**. No matplotlib / pandas / ee here — this is the
honest-confidence layer that must run wherever numpy runs. All functions are
NaN-aware (gaps are encoded as ``np.nan``).

References: research/09 (the 35-row matrix, fusion/gap-filling, uncertainty);
ARCHITECTURE §5.2-5.3, §10; ``constants.ROBUSTNESS_SUMMARY``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from urbanheat.constants import ROBUSTNESS_SUMMARY
from urbanheat.datamodel import (
    AGREEMENT_SUFFIX,
    ALL_VARIABLES,
    LST,
    UNCERTAINTY_SUFFIX,
)

if TYPE_CHECKING:  # pragma: no cover - hints only
    from urbanheat.datamodel import FeatureStack


# ===========================================================================
# THE ≥30-METHOD ROBUSTNESS MATRIX  (research/09 §4, 35 entries)
# ===========================================================================
# Each dict mirrors one row of the research/09 cross-verification matrix:
#   id          : stable integer id == row number in research/09 §4
#   name        : human-readable source/method name
#   type        : "data"  (a dataset/sensor)  |  "method" (an analytical method)
#   role        : its role in the urbanheat system
#   gap         : the failure-mode / gap it fills for the others
#   verified_by : the independent source(s)/method(s) that cross-check it
#   active      : whether it participates in the offline/synthetic path
#                 (data sensors are emulated by SyntheticDataSource; the
#                  analytical methods marked active run array-level here).
METHODS_REGISTRY: list[dict[str, Any]] = [
    # ----- LST sensors (data) ------------------------------------------------
    {"id": 1, "name": "Landsat 8/9 LST (TIRS, 30 m, 16-day)", "type": "data",
     "role": "high-res LST backbone for hotspots & model target",
     "gap": "fine spatial detail of intra-urban heat",
     "verified_by": "ECOSTRESS & MODIS at matched overpass; in-situ", "active": True},
    {"id": 2, "name": "ECOSTRESS LST (70 m, diurnal/ISS)", "type": "data",
     "role": "variable-time-of-day LST; captures diurnal cycle",
     "gap": "Landsat's fixed ~10:30 overpass (diurnal gap)",
     "verified_by": "Landsat & ASTER; in-situ bias-RMSE benchmark", "active": False},
    {"id": 3, "name": "MODIS LST (MOD11/MYD11, 1 km, 4x/day)", "type": "data",
     "role": "high-temporal LST; harmonic-fit & fusion base",
     "gap": "Landsat/ECOSTRESS low revisit (temporal gap)",
     "verified_by": "triple collocation; validated MOD11 bias<0.8K", "active": True},
    {"id": 4, "name": "VIIRS LST (~750 m, daily)", "type": "data",
     "role": "extra high-temporal LST after MODIS era",
     "gap": "MODIS continuity / extra daily sample",
     "verified_by": "cross-sensor reconciliation; MODIS", "active": False},
    {"id": 5, "name": "Sentinel-3 SLSTR LST (1 km)", "type": "data",
     "role": "independent 4th LST stream",
     "gap": "adds an independent triplet member for TC",
     "verified_by": "triple collocation; MODIS/VIIRS agreement", "active": False},
    # ----- vegetation / spectral (data) -------------------------------------
    {"id": 6, "name": "Sentinel-2 MSI (10 m, 5-day)", "type": "data",
     "role": "fine vegetation/indices; LULC input; sharpening predictor",
     "gap": "fine NDVI/NDBI/NDWI for thermal sharpening & gap-fill",
     "verified_by": "Landsat indices; Dynamic World", "active": True},
    {"id": 7, "name": "Spectral indices (NDVI/NDBI/NDWI/NDISI/albedo/UI)", "type": "method",
     "role": "driver covariates; constraints on LST gap-fill; sharpening",
     "gap": "physically bounds & predicts missing LST",
     "verified_by": "mutual consistency (NDVI vs NDWI); SHAP sign check", "active": True},
    # ----- LULC (data + method) ---------------------------------------------
    {"id": 8, "name": "ESA WorldCover 10 m", "type": "data",
     "role": "LULC layer (member of vote)",
     "gap": "global consistent land cover",
     "verified_by": "majority vote vs ESRI/DW/classifier", "active": True},
    {"id": 9, "name": "ESRI 10 m Land Cover", "type": "data",
     "role": "LULC layer (member of vote)",
     "gap": "independent annual LULC",
     "verified_by": "LULC agreement map", "active": True},
    # ----- meteorology / ground truth (data) --------------------------------
    {"id": 10, "name": "IMD stations / AWS", "type": "data",
     "role": "in-situ air-temp ground truth",
     "gap": "absolute calibration of air-temp/heat-stress",
     "verified_by": "ERA5 cross-QC; CPCB", "active": False},
    {"id": 11, "name": "CPCB ambient stations", "type": "data",
     "role": "dense in-situ met in Indian metros (PS-1 named)",
     "gap": "intra-urban T_air where IMD sparse",
     "verified_by": "IMD; Netatmo; ERA5", "active": False},
    {"id": 12, "name": "Netatmo citizen weather stations", "type": "data",
     "role": "dense crowdsourced T_air (spatial UHI pattern)",
     "gap": "station-network spatial sparsity",
     "verified_by": "CrowdQC+ QC; CPCB/IMD reference", "active": False},
    {"id": 13, "name": "ERA5 / ERA5-Land reanalysis", "type": "data",
     "role": "atmospheric drivers (T_air, RH, wind) + skin temp",
     "gap": "wall-to-wall met where stations absent",
     "verified_by": "station cross-check; coarse->downscaled", "active": True},
    # ----- morphology / footprints (data) -----------------------------------
    {"id": 14, "name": "GHSL (built-up surface, height, pop)", "type": "data",
     "role": "urban morphology + exposure; built mask",
     "gap": "consistent global built-up & population",
     "verified_by": "OSM & open footprints cross-fill", "active": True},
    {"id": 15, "name": "Dynamic World (NRT 10 m LULC)", "type": "data",
     "role": "time-aware LULC + class probabilities",
     "gap": "up-to-date LULC & per-pixel uncertainty",
     "verified_by": "vote vs WorldCover/ESRI; probabilities feed MC", "active": True},
    {"id": 16, "name": "Project-trained S2/Landsat LULC classifier", "type": "method",
     "role": "local, tuned LULC for the study city",
     "gap": "local classes generic products miss",
     "verified_by": "majority vote & confusion vs WorldCover/ESRI/DW", "active": False},
    {"id": 17, "name": "OpenStreetMap (buildings/roads)", "type": "data",
     "role": "detailed morphology where mapped",
     "gap": "fine street/building geometry",
     "verified_by": "GHSL & open footprints fill OSM gaps", "active": False},
    {"id": 18, "name": "Microsoft/Google open building footprints", "type": "data",
     "role": "wide-coverage footprints",
     "gap": "OSM omissions in unmapped areas",
     "verified_by": "OSM + GHSL height reconcile", "active": False},
    {"id": 19, "name": "UT-GLOBUS urban morphology", "type": "data",
     "role": "building-level morphology params (PS-1 named)",
     "gap": "detailed 3-D morphology for SEB/SOLWEIG",
     "verified_by": "GHSL height; OSM", "active": False},
    # ----- fusion / reconciliation / error (method) -------------------------
    {"id": 20, "name": "Multi-sensor reconciliation (bias/CDF/view-angle)", "type": "method",
     "role": "harmonize all LST sensors to a common reference",
     "gap": "inter-sensor systematic bias (seam removal)",
     "verified_by": "post-harmonization overlap RMSE; TC", "active": True},
    {"id": 21, "name": "Triple Collocation (incl. Extended TC)", "type": "method",
     "role": "error variance & fusion weights for >=3 independent LST",
     "gap": "'no truth' error estimation & weighting",
     "verified_by": "independent in-situ check; ensemble spread", "active": True},
    {"id": 22, "name": "STARFM / ESTARFM / FSDAF spatiotemporal fusion", "type": "method",
     "role": "fuse hi-space x hi-time -> 30 m frequent LST",
     "gap": "cloud/orbit + revisit gaps simultaneously",
     "verified_by": "hold-out clear Landsat scene; cross-sensor", "active": False},
    {"id": 23, "name": "Thermal sharpening (TsHARP/DisTrad/DMS)", "type": "method",
     "role": "disaggregate coarse LST to 30 m via indices",
     "gap": "fine LST missing but fine indices present",
     "verified_by": "coincident Landsat LST", "active": False},
    {"id": 24, "name": "Harmonic / ATC temporal modeling (HANTS)", "type": "method",
     "role": "fit diurnal+annual cycle; fill temporal gaps",
     "gap": "cloud-obscured dates in LST series",
     "verified_by": "adjacent clear observations; GP residuals", "active": False},
    {"id": 25, "name": "Gaussian Process / (regression-)kriging / NNGP", "type": "method",
     "role": "spatial gap-fill with uncertainty; residual kriging",
     "gap": "spatial holes; residual autocorrelation",
     "verified_by": "cross-validation at withheld pixels; variogram fit", "active": True},
    {"id": 26, "name": "Building-dataset cross-fill (OSM u GHSL u open u UT-GLOBUS)", "type": "method",
     "role": "union-fill footprints, reconcile heights",
     "gap": "each footprint source's omissions",
     "verified_by": "mutual overlap agreement; imagery spot-check", "active": False},
    {"id": 27, "name": "LULC majority voting + agreement map", "type": "method",
     "role": "consensus land cover + uncertainty localization",
     "gap": "single-product classification errors",
     "verified_by": "disagreement map flags low-confidence pixels", "active": True},
    {"id": 28, "name": "CrowdQC+ / Meier QC for crowdsourced T_air", "type": "method",
     "role": "filter Netatmo solar-bias & siting outliers",
     "gap": "raw crowdsourced data unreliability",
     "verified_by": "spatial-consistency + reference stations", "active": False},
    {"id": 29, "name": "Ensemble agreement / disagreement maps", "type": "method",
     "role": "mean + spread across all LST/hotspot estimates",
     "gap": "single-estimate over-confidence",
     "verified_by": "high-spread pixels -> field check", "active": True},
    {"id": 30, "name": "Monte Carlo uncertainty propagation", "type": "method",
     "role": "propagate input errors -> output PDF & per-pixel sigma",
     "gap": "missing uncertainty on LST/attribution/cooling-dT",
     "verified_by": "TC error inputs; spatial-CV spread", "active": True},
    # ----- validation (method) ----------------------------------------------
    {"id": 31, "name": "Spatial cross-validation (BlockKFold / buffered SLOO)", "type": "method",
     "role": "honest model skill without spatial leakage",
     "gap": "random-CV optimism (~28%)",
     "verified_by": "design-based probability-sample accuracy", "active": True},
    {"id": 32, "name": "Design-based validation on probability sample", "type": "method",
     "role": "population map-accuracy estimate (Wadoux critique)",
     "gap": "spatial-CV's map-accuracy limitation",
     "verified_by": "spatial CV as the skill counterpart", "active": False},
    # ----- attribution (method) ---------------------------------------------
    {"id": 33, "name": "SHAP (global + dependence + interaction)", "type": "method",
     "role": "ML-side ranked driver importance & effects",
     "gap": "black-box opacity; interaction discovery",
     "verified_by": "LMG/variance partitioning; GWR sign", "active": True},
    {"id": 34, "name": "Variance partitioning (LMG / hierarchical / dominance)", "type": "method",
     "role": "clean additive %-share of R^2 per driver",
     "gap": "SHAP collinearity ambiguity",
     "verified_by": "agreement with SHAP ranking", "active": True},
    {"id": 35, "name": "GWR / MGWR spatially-varying coefficients", "type": "method",
     "role": "per-pixel driver coefficient & dominance maps",
     "gap": "global importance hides spatial non-stationarity",
     "verified_by": "local-R^2 map; physics sign check; SHAP", "active": False},
]

#: Total number of cross-verifying entries in the matrix (research/09 §4).
N_METHODS_TOTAL: int = len(METHODS_REGISTRY)


def methods_count(active_only: bool = False) -> int:
    """Return the number of registry entries (optionally only the active ones)."""
    if active_only:
        return sum(1 for m in METHODS_REGISTRY if m.get("active"))
    return len(METHODS_REGISTRY)


def registry_summary() -> dict[str, Any]:
    """Headline accounting of the robustness matrix for the report/CLI.

    Combines :data:`METHODS_REGISTRY` (the per-row matrix) with the
    pre-tallied :data:`constants.ROBUSTNESS_SUMMARY` headline counts.
    """
    n_data = sum(1 for m in METHODS_REGISTRY if m["type"] == "data")
    n_method = sum(1 for m in METHODS_REGISTRY if m["type"] == "method")
    return {
        "total_entries": N_METHODS_TOTAL,
        "n_data_sources": n_data,
        "n_methods": n_method,
        "n_active": methods_count(active_only=True),
        "meets_30_target": N_METHODS_TOTAL >= 30,
        "headline": dict(ROBUSTNESS_SUMMARY),
    }


# ===========================================================================
# ARRAY-LEVEL FUSION UTILITIES  (research/09 §3 — numpy only, NaN-aware)
# ===========================================================================
def _stack(layers: Sequence[np.ndarray]) -> np.ndarray:
    """Stack a sequence of equal-shaped 2-D arrays to a 3-D ``(K, H, W)`` float64."""
    if not layers:
        raise ValueError("need at least one layer")
    arrs = [np.asarray(a, dtype=np.float64) for a in layers]
    shape0 = arrs[0].shape
    for i, a in enumerate(arrs):
        if a.shape != shape0:
            raise ValueError(
                f"layer {i} shape {a.shape} != first layer shape {shape0}")
    return np.stack(arrs, axis=0)


def weighted_ensemble(
    layers: Sequence[np.ndarray],
    uncertainties: Sequence[np.ndarray] | None = None,
    weights: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Uncertainty-/skill-weighted ensemble of co-registered estimates.

    The robust core of multi-source fusion (research/09 §3.4 — BMA / weighted
    ensemble; §3.1 uncertainty-weighted LST). Combines ``K`` estimate layers of
    one quantity into a single best estimate plus its propagated 1-sigma.

    Parameters
    ----------
    layers
        Sequence of ``K`` equal-shaped ``(H, W)`` arrays (NaN = missing).
    uncertainties
        Optional matching per-pixel 1-sigma error layers. When given, weights are
        **inverse-variance** (``w_k = 1/sigma_k^2``), the statistically optimal
        linear combination; the fused 1-sigma is ``sqrt(1/sum(w_k))``.
    weights
        Optional scalar per-layer weights (e.g. Triple-Collocation skill weights)
        used when ``uncertainties`` is None. Defaults to equal weights.

    Returns
    -------
    (fused, fused_sigma)
        ``fused`` is the weighted mean (NaN where every layer is NaN);
        ``fused_sigma`` is the propagated 1-sigma (NaN where unknown).
    """
    vals = _stack(layers)
    K = vals.shape[0]
    valid = np.isfinite(vals)

    if uncertainties is not None:
        errs = _stack(uncertainties)
        if errs.shape != vals.shape:
            raise ValueError("uncertainties must match layers in shape/count")
        safe = np.where((errs > 0) & np.isfinite(errs), errs, np.nan)
        w = 1.0 / (safe ** 2)
    else:
        if weights is None:
            scal = np.ones(K, dtype=np.float64)
        else:
            scal = np.asarray(weights, dtype=np.float64)
            if scal.shape != (K,):
                raise ValueError(f"weights must have length {K}")
        w = np.broadcast_to(scal[:, None, None], vals.shape).astype(np.float64)

    w = np.where(valid & np.isfinite(w), w, 0.0)
    wsum = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fused = np.where(wsum > 0, (w * np.nan_to_num(vals)).sum(axis=0) / wsum, np.nan)

    if uncertainties is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            fused_sigma = np.where(wsum > 0, np.sqrt(1.0 / wsum), np.nan)
    else:
        # weighted sample spread as the empirical uncertainty
        diff2 = (vals - fused[None, :, :]) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.where(wsum > 0, (w * np.nan_to_num(diff2)).sum(axis=0) / wsum, np.nan)
        fused_sigma = np.sqrt(var)

    return fused.astype(np.float32), fused_sigma.astype(np.float32)


def uncertainty_weighted_fuse(
    values: Sequence[np.ndarray], errors: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-error-variance weighted fusion (ARCHITECTURE §11.7 contract name).

    Thin alias of :func:`weighted_ensemble` with explicit error layers; returns
    ``(fused_value, fused_uncertainty)``. [research/09 §3.1, R1 §3.D]
    """
    return weighted_ensemble(values, uncertainties=errors)


def agreement_map(
    layers: Sequence[np.ndarray], tol: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel ensemble agreement of continuous estimates.

    Implements the "ensemble agreement / disagreement map" (research/09 §3.6,
    matrix #29): high spread = low confidence = field-check candidate.

    Returns ``(mean, spread)`` where ``spread`` is the per-pixel standard
    deviation across the (finite) layers. If ``tol`` is given, also usable as an
    agreement count via :func:`agreement_count`.
    """
    vals = _stack(layers)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(vals, axis=0)
        spread = np.nanstd(vals, axis=0)
    return mean.astype(np.float32), spread.astype(np.float32)


def ensemble_agreement(estimates: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """ARCHITECTURE §11.7 contract name for :func:`agreement_map`.

    Stack candidate estimates of one quantity -> ``(mean, spread)`` maps.
    [research/09 §3.6]
    """
    return agreement_map(estimates)


def agreement_count(
    layers: Sequence[np.ndarray], reference: np.ndarray | None = None,
    tol: float = 1.0,
) -> np.ndarray:
    """Number of layers that agree (within ``tol``) with the ensemble (or a reference).

    A per-pixel ``n``-source agreement count, matching the FeatureStack
    :data:`AGREEMENT_SUFFIX` convention (research/09 §3.6 / R2).
    """
    vals = _stack(layers)
    ref = np.nanmedian(vals, axis=0) if reference is None else np.asarray(reference, np.float64)
    close = np.isfinite(vals) & (np.abs(vals - ref[None, :, :]) <= tol)
    return close.sum(axis=0).astype(np.float32)


def majority_vote(
    categorical_layers: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel majority vote + vote-margin for categorical layers (e.g. LULC).

    Implements LULC majority voting + agreement map (research/09 §3.5-3.6,
    matrix #27): the consensus class plus a ``[0,1]`` agreement fraction that
    localizes where classification is uncertain.

    Parameters
    ----------
    categorical_layers
        Sequence of ``K`` equal-shaped integer-coded class arrays (NaN = no vote).

    Returns
    -------
    (consensus, agreement_frac)
        ``consensus`` is the modal class per pixel (NaN where no votes);
        ``agreement_frac`` is votes_for_winner / total_votes in ``[0,1]``.
    """
    vals = _stack(categorical_layers)
    K, H, W = vals.shape
    consensus = np.full((H, W), np.nan, dtype=np.float64)
    agreement = np.zeros((H, W), dtype=np.float64)

    flat = vals.reshape(K, -1)
    for px in range(flat.shape[1]):
        col = flat[:, px]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        classes, counts = np.unique(np.round(finite).astype(np.int64),
                                    return_counts=True)
        win = int(np.argmax(counts))
        r, c = divmod(px, W)
        consensus[r, c] = float(classes[win])
        agreement[r, c] = counts[win] / finite.size
    return consensus.astype(np.float32), agreement.astype(np.float32)


def triple_collocation(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> dict[str, float]:
    """Triple-Collocation error variances of three independent datasets.

    Estimates the **random-error variance of three mutually independent
    measurements of the same geophysical variable without assuming any is truth**
    (research/09 §3.3, matrix #21; McColl 2014 Extended TC). Also returns the
    inverse-error-variance **optimal fusion weights**.

    Assumptions (must hold for validity): zero-mean errors, mutually uncorrelated
    errors, linear relationship between datasets. Uses the *covariance* (difference)
    notation::

        var_a = Cov(a,b)            ... wait — see formulae below.

    With ``X`` the (unknown) truth and ``a = X + e_a`` etc. (after rescaling b,c to
    a's scale), the classic estimator is::

        sigma_a^2 = Var(a) - Cov(a,b)*Cov(a,c)/Cov(b,c)
        sigma_b^2 = Var(b) - Cov(a,b)*Cov(b,c)/Cov(a,c)
        sigma_c^2 = Var(c) - Cov(a,c)*Cov(b,c)/Cov(a,b)

    Returns
    -------
    dict
        ``err_var_a/b/c`` (error variances, may be negative if assumptions fail —
        clipped at 0 for the weights), ``err_std_a/b/c`` (sqrt of the clipped
        variances), ``weight_a/b/c`` (inverse-error-variance fusion weights summing
        to 1), and ``n`` (number of co-located finite samples used).
    """
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    cc = np.asarray(c, dtype=np.float64).ravel()
    mask = np.isfinite(aa) & np.isfinite(bb) & np.isfinite(cc)
    aa, bb, cc = aa[mask], bb[mask], cc[mask]
    n = int(aa.size)
    out: dict[str, float] = {"n": float(n)}
    if n < 3:
        for k in ("a", "b", "c"):
            out[f"err_var_{k}"] = float("nan")
            out[f"err_std_{k}"] = float("nan")
            out[f"weight_{k}"] = float("nan")
        return out

    cov_ab = float(np.cov(aa, bb, ddof=1)[0, 1])
    cov_ac = float(np.cov(aa, cc, ddof=1)[0, 1])
    cov_bc = float(np.cov(bb, cc, ddof=1)[0, 1])
    var_a = float(np.var(aa, ddof=1))
    var_b = float(np.var(bb, ddof=1))
    var_c = float(np.var(cc, ddof=1))

    eps = 1e-12
    ev_a = var_a - (cov_ab * cov_ac) / (cov_bc if abs(cov_bc) > eps else np.nan)
    ev_b = var_b - (cov_ab * cov_bc) / (cov_ac if abs(cov_ac) > eps else np.nan)
    ev_c = var_c - (cov_ac * cov_bc) / (cov_ab if abs(cov_ab) > eps else np.nan)

    out["err_var_a"], out["err_var_b"], out["err_var_c"] = ev_a, ev_b, ev_c
    for k, ev in (("a", ev_a), ("b", ev_b), ("c", ev_c)):
        out[f"err_std_{k}"] = float(np.sqrt(ev)) if np.isfinite(ev) and ev > 0 else float("nan")

    # inverse-error-variance weights (clip negatives to a tiny variance)
    clipped = [max(ev, eps) if np.isfinite(ev) else np.inf for ev in (ev_a, ev_b, ev_c)]
    inv = np.array([1.0 / cv for cv in clipped], dtype=np.float64)
    if np.all(np.isfinite(inv)) and inv.sum() > 0:
        w = inv / inv.sum()
    else:
        w = np.array([np.nan, np.nan, np.nan])
    out["weight_a"], out["weight_b"], out["weight_c"] = (float(w[0]), float(w[1]), float(w[2]))
    return out


def reconcile_sensors(
    sensor_arrays: dict[str, np.ndarray], ref: str | None = None
) -> dict[str, np.ndarray]:
    """Bias-correct multiple LST sensor arrays to a common reference before fusion.

    Implements the multi-sensor reconciliation step (research/09 §3.2, matrix #20):
    linear-rescale (mean+std match — a lightweight CDF/quantile-match proxy) every
    sensor to a chosen ``ref`` so they blend without seams. The reference is the
    most-complete sensor by default (most finite pixels).

    Returns a new dict of bias-corrected arrays (the reference is returned
    unchanged). NaN-aware.
    """
    if not sensor_arrays:
        return {}
    names = list(sensor_arrays)
    if ref is None:
        ref = max(names, key=lambda k: int(np.isfinite(sensor_arrays[k]).sum()))
    if ref not in sensor_arrays:
        raise KeyError(f"ref {ref!r} not in sensor_arrays {names}")

    ref_arr = np.asarray(sensor_arrays[ref], dtype=np.float64)
    ref_mean = float(np.nanmean(ref_arr))
    ref_std = float(np.nanstd(ref_arr))

    out: dict[str, np.ndarray] = {}
    for name, arr in sensor_arrays.items():
        a = np.asarray(arr, dtype=np.float64)
        if name == ref:
            out[name] = a.astype(np.float32)
            continue
        m = float(np.nanmean(a))
        s = float(np.nanstd(a))
        if s > 1e-9 and np.isfinite(s):
            adj = (a - m) / s * ref_std + ref_mean
        else:
            adj = a - m + ref_mean
        out[name] = adj.astype(np.float32)
    return out


def gap_fill(
    layer: np.ndarray,
    ancillary: np.ndarray | Sequence[np.ndarray] | None = None,
    max_iter: int = 50,
) -> np.ndarray:
    """Fill NaN gaps in a 2-D field via covariate regression then diffusion infill.

    Implements the array-level gap-fill of research/09 §3.1/§3.5 (matrix #25): use
    correlated ancillary covariates (NDVI/DEM/albedo/...) to predict missing pixels
    by ordinary least squares, then fill any pixels still missing (because their
    ancillary was also NaN) by iterative neighbour-mean diffusion (a fast
    interpolation proxy). NDVI-style covariates physically constrain the fill
    (research/09 §3.5 — "NDVI constrains LST gap-fill").

    Parameters
    ----------
    layer
        2-D array to fill (NaN = gap).
    ancillary
        Optional single covariate array or sequence of them, co-registered with
        ``layer``. When None (or no usable overlap) only the diffusion infill runs.
    max_iter
        Max diffusion sweeps for residual holes.

    Returns
    -------
    np.ndarray
        A copy of ``layer`` with gaps filled (float32). Non-gap pixels are
        preserved exactly (mass/observation-preserving).
    """
    y = np.asarray(layer, dtype=np.float64).copy()
    gap = ~np.isfinite(y)
    if not gap.any():
        return y.astype(np.float32)

    # 1) covariate regression fill ------------------------------------------
    if ancillary is not None:
        anc_list = [ancillary] if isinstance(ancillary, np.ndarray) else list(ancillary)
        cols = [np.asarray(a, dtype=np.float64).ravel() for a in anc_list if a is not None]
        if cols:
            X = np.column_stack(cols)
            yf = y.ravel()
            train = np.isfinite(yf) & np.all(np.isfinite(X), axis=1)
            pred_ok = gap.ravel() & np.all(np.isfinite(X), axis=1)
            if train.sum() >= (X.shape[1] + 1) and pred_ok.any():
                A = np.column_stack([X[train], np.ones(train.sum())])
                coef, *_ = np.linalg.lstsq(A, yf[train], rcond=None)
                Ap = np.column_stack([X[pred_ok], np.ones(pred_ok.sum())])
                yf[pred_ok] = Ap @ coef
                y = yf.reshape(y.shape)

    # 2) diffusion infill for any residual gaps -----------------------------
    gap = ~np.isfinite(y)
    if gap.any():
        filled = y.copy()
        # seed remaining gaps with the global mean so the relaxation converges
        gmean = float(np.nanmean(filled)) if np.isfinite(np.nanmean(filled)) else 0.0
        filled[~np.isfinite(filled)] = gmean
        for _ in range(max_iter):
            if not gap.any():
                break
            up = np.empty_like(filled); up[:-1] = filled[1:]; up[-1] = filled[-1]
            dn = np.empty_like(filled); dn[1:] = filled[:-1]; dn[0] = filled[0]
            lf = np.empty_like(filled); lf[:, :-1] = filled[:, 1:]; lf[:, -1] = filled[:, -1]
            rt = np.empty_like(filled); rt[:, 1:] = filled[:, :-1]; rt[:, 0] = filled[:, 0]
            neigh = (up + dn + lf + rt) / 4.0
            filled = np.where(gap, neigh, filled)
        y = filled
    return y.astype(np.float32)


def monte_carlo_uncertainty(
    fn: Callable[..., np.ndarray],
    inputs: dict[str, np.ndarray],
    errors: dict[str, np.ndarray],
    n: int = 100,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo propagation of input errors through ``fn`` (research/09 §3.6/§5).

    Perturb each named input by Gaussian noise of its per-pixel (or scalar) 1-sigma
    ``errors[name]``, evaluate ``fn(**perturbed)`` ``n`` times, and return the
    per-pixel ``(mean, std)`` of the output (matrix #30). This is how LST ->
    hotspot -> cooling-dT claims acquire honest error bars.
    """
    rng = np.random.default_rng(seed)
    acc: list[np.ndarray] = []
    for _ in range(int(n)):
        sample = {}
        for k, v in inputs.items():
            arr = np.asarray(v, dtype=np.float64)
            e = errors.get(k)
            if e is None:
                sample[k] = arr
            else:
                sigma = np.asarray(e, dtype=np.float64)
                sample[k] = arr + rng.standard_normal(arr.shape) * sigma
        acc.append(np.asarray(fn(**sample), dtype=np.float64))
    cube = np.stack(acc, axis=0)
    return cube.mean(axis=0).astype(np.float32), cube.std(axis=0).astype(np.float32)


# ===========================================================================
# REPORTING
# ===========================================================================
def robustness_report(fs: "FeatureStack") -> dict[str, Any]:
    """Summarize cross-verification accounting + per-layer agreement/uncertainty.

    The dict the report/CLI consume (ARCHITECTURE §11.7). It combines:

    * the **>=30-method accounting** (:func:`registry_summary` over
      :data:`METHODS_REGISTRY`, plus ``constants.ROBUSTNESS_SUMMARY``);
    * which canonical layers in ``fs`` carry companion ``*_uncertainty`` and
      ``*_agreement`` layers (the honest-confidence layers from research/09 §5);
    * simple LST source-agreement / uncertainty statistics if those layers exist.

    Returns
    -------
    dict
        ``{"methods": {...}, "n_methods_total": int, "n_methods_active": int,
        "uncertainty_layers": [...], "agreement_layers": [...],
        "lst_uncertainty_mean": float|None, "source_agreement": {...},
        "narrative": str}``.
    """
    summary = registry_summary()
    present = set(fs.names()) if fs is not None else set()

    unc_layers = sorted(n for n in present if n.endswith(UNCERTAINTY_SUFFIX))
    agr_layers = sorted(n for n in present if n.endswith(AGREEMENT_SUFFIX))

    lst_unc_mean: float | None = None
    lst_unc_name = f"{LST}{UNCERTAINTY_SUFFIX}"
    if fs is not None and fs.has(lst_unc_name):
        with np.errstate(invalid="ignore"):
            m = float(np.nanmean(fs.get(lst_unc_name)))
        lst_unc_mean = m if np.isfinite(m) else None

    # how many of the canonical driver layers are actually populated (a proxy
    # for "sources active" in this particular run)
    populated = sorted(present.intersection(ALL_VARIABLES))

    source_agreement: dict[str, Any] = {}
    if agr_layers and fs is not None:
        for name in agr_layers:
            with np.errstate(invalid="ignore"):
                source_agreement[name] = float(np.nanmean(fs.get(name)))

    narrative = (
        f"Robustness: {summary['total_entries']} cross-verifying entries "
        f"({summary['n_data_sources']} datasets + {summary['n_methods']} methods) "
        f"from the research/09 matrix; {summary['n_active']} active in this run "
        f"(>=30 target {'MET' if summary['meets_30_target'] else 'NOT met'}). "
        f"{len(unc_layers)} uncertainty + {len(agr_layers)} agreement layers present; "
        f"{len(populated)} canonical driver layers populated."
    )

    return {
        "methods": summary,
        "registry": METHODS_REGISTRY,
        "n_methods_total": summary["total_entries"],
        "n_methods_active": summary["n_active"],
        "uncertainty_layers": unc_layers,
        "agreement_layers": agr_layers,
        "populated_layers": populated,
        "n_populated_layers": len(populated),
        "lst_uncertainty_mean": lst_unc_mean,
        "source_agreement": source_agreement,
        "narrative": narrative,
    }


__all__ = [
    "METHODS_REGISTRY",
    "N_METHODS_TOTAL",
    "methods_count",
    "registry_summary",
    "weighted_ensemble",
    "uncertainty_weighted_fuse",
    "agreement_map",
    "ensemble_agreement",
    "agreement_count",
    "majority_vote",
    "triple_collocation",
    "reconcile_sensors",
    "gap_fill",
    "monte_carlo_uncertainty",
    "robustness_report",
]
