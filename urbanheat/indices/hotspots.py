"""urbanheat.indices.hotspots — spatial hotspot statistics + the layered composite.

Implements the **statistically-defensible** hotspot machinery of research/08 (R8)
on :class:`~urbanheat.datamodel.FeatureStack` grids:

* **Getis-Ord Gi\*** (R8 §9.1) — exact z-score hot-/cold-cluster statistic with
  spatial weights from a grid neighbourhood (rook / queen / distance-band kernel
  of half-width ``k``). Writes ``GISTAR_Z``.
* **Local Moran's I** (R8 §9.2) — LISA with HH / LL / HL / LH cluster categories.
  Writes ``MORAN_LOCAL``.
* **Surface hotspots** (R8 §10) — ``HOTSPOT_MASK = (LST >= P90) AND (Gi* >= 1.96)``.
* **Heat Vulnerability Index** (R8 §11) — Exposure / Sensitivity /
  -AdaptiveCapacity (:data:`constants.HVI_DOMAINS`) combined by PCA (lazy
  ``sklearn`` else a weighted-z fallback), normalized 0-1 with quintiles. Writes
  ``HVI``.
* **The layered 5-class composite** (R8 §12) — Layer A surface score, Layer B
  human-stress agreement, Layer C vulnerability-weighted ``PriorityScore`` ->
  the Low/Moderate/High/Severe/Extreme legend (:data:`constants.HOTSPOT_LEGEND`).
  Writes ``PRIORITY_SCORE``, ``HOTSPOT_MASK`` and a 5-class categorical layer.

Design rules (ARCHITECTURE.md §7, §11.3): numpy/scipy top-level OK; ``sklearn`` /
``esda``/``pysal`` lazy with documented numpy fallbacks; every threshold / colour
from :mod:`urbanheat.constants`; canonical names and signatures matched exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from urbanheat.constants import (
    HOTSPOT_GISTAR_Z,
    HOTSPOT_LEGEND,
    HVI_DOMAIN_WEIGHTS,
    HVI_DOMAINS,
    LST_PERCENTILE_THRESHOLDS,
)
from urbanheat.datamodel import (
    AIR_TEMP,
    GISTAR_Z,
    GREEN_FRAC,
    HOTSPOT_MASK,
    HVI,
    IMPERVIOUS_FRAC,
    LST,
    LST_PERCENTILE,
    MORAN_LOCAL,
    NDVI,
    POPULATION,
    PRIORITY_SCORE,
    REL_HUMIDITY,
    SUHII,
    UTFVI,
    FeatureStack,
)
from urbanheat.indices.heat_indices import (
    HUMAN_STRESS_SCORE,
    human_stress_ensemble,
    lst_percentile,
)

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd  # noqa: F401

# Extra (diagnostic) canonical-style layer names this module writes.
PRIORITY_CLASS = "priority_class"      # integer 0-4 -> HOTSPOT_LEGEND class
HVI_QUINTILE = "hvi_quintile"          # integer 1-5 (Very Low .. Very High)
SURFACE_SCORE = "surface_score"        # 0-100 Layer-A surface heat score

# Local Moran's I category integer codes (written into MORAN_LOCAL).
MORAN_CODES: dict[str, int] = {"ns": 0, "HH": 1, "LL": 2, "HL": 3, "LH": 4}

__all__ = [
    "getis_ord_gi_star",
    "local_morans_i",
    "local_moran",
    "surface_hotspots",
    "heat_vulnerability_index",
    "layered_hotspots",
    "composite_priority",
    "classify_priority",
    "neighbourhood_weight_kernel",
    "PRIORITY_CLASS",
    "HVI_QUINTILE",
    "SURFACE_SCORE",
    "MORAN_CODES",
]


# ===========================================================================
# Spatial weights + neighbourhood sums (grid kernels)
# ===========================================================================
def neighbourhood_weight_kernel(k: int = 1, contiguity: str = "queen",
                                include_self: bool = True) -> np.ndarray:
    """Build a ``(2k+1, 2k+1)`` binary spatial-weight kernel for a regular grid.

    Parameters
    ----------
    k : int, default 1
        Neighbourhood half-width (``k=1`` -> 3x3, ``k=2`` -> 5x5 ...).
    contiguity : str, default "queen"
        * ``"queen"``    — full square (8-neighbour at k=1): all cells weight 1.
        * ``"rook"``     — von-Neumann (4-neighbour at k=1): Manhattan distance
          ``<= k`` only.
        * ``"distance"`` — Euclidean distance band: cells within radius ``k``.
    include_self : bool, default True
        Whether the centre cell (``w_ii``) is 1. **Gi\*** includes self
        (``w_ii != 0``); the classic Gi (no star) and Moran's I exclude it.

    Returns
    -------
    np.ndarray
        Float kernel of 0/1 weights, shape ``(2k+1, 2k+1)``.
    """
    size = 2 * k + 1
    yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
    if contiguity == "rook":
        w = (np.abs(yy) + np.abs(xx) <= k).astype(np.float64)
    elif contiguity == "distance":
        w = (np.sqrt(yy ** 2 + xx ** 2) <= k + 1e-9).astype(np.float64)
    else:  # queen (default): full square
        w = np.ones((size, size), dtype=np.float64)
    if not include_self:
        w[k, k] = 0.0
    return w


def _neighbour_sum(values: np.ndarray, kernel: np.ndarray,
                   valid: np.ndarray | None = None) -> np.ndarray:
    """Convolve ``values`` with ``kernel`` (sum of weighted neighbours per cell).

    Uses :func:`scipy.ndimage.correlate` when SciPy is importable (fast, exact
    boundary handling), else a documented numpy shift-and-add fallback. Boundary
    cells use only their in-grid neighbours ('constant 0' fill on the padded
    values, with a parallel pass to know which neighbours were valid).

    Parameters
    ----------
    values : np.ndarray
        2-D field (NaNs allowed; treated as 0 in the weighted sum, with the
        ``valid`` mask tracking real contributions).
    kernel : np.ndarray
        Spatial-weight kernel from :func:`neighbourhood_weight_kernel`.
    valid : np.ndarray, optional
        Boolean mask of finite/usable cells; defaults to ``isfinite(values)``.

    Returns
    -------
    np.ndarray
        Per-cell sum of ``w_ij * x_j`` over in-grid neighbours.
    """
    if valid is None:
        valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0).astype(np.float64)
    try:
        from scipy.ndimage import correlate  # type: ignore

        return correlate(filled, kernel, mode="constant", cval=0.0)
    except Exception:
        # numpy fallback: pad then shift-add each kernel offset.
        k = kernel.shape[0] // 2
        padded = np.pad(filled, k, mode="constant", constant_values=0.0)
        out = np.zeros_like(filled)
        rows, cols = filled.shape
        for dy in range(kernel.shape[0]):
            for dx in range(kernel.shape[1]):
                wgt = kernel[dy, dx]
                if wgt == 0.0:
                    continue
                out += wgt * padded[dy : dy + rows, dx : dx + cols]
        return out


def _weight_count(valid: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sum_w, sum_w2)`` of in-grid weights per cell (for Gi* variance).

    Only neighbours that are *valid* (finite) contribute, so edge cells and
    NaN-masked neighbours correctly reduce the local weight totals.
    """
    vf = valid.astype(np.float64)
    sum_w = _neighbour_sum(vf, kernel, valid=np.ones_like(valid, dtype=bool))
    sum_w2 = _neighbour_sum(vf, kernel ** 2, valid=np.ones_like(valid, dtype=bool))
    return sum_w, sum_w2


# ===========================================================================
# Getis-Ord Gi*
# ===========================================================================
def getis_ord_gi_star(
    values: Any,
    coords_or_shape: Any = None,
    k: int = 1,
    contiguity: str = "queen",
    var: str = LST,
) -> Any:
    """Getis-Ord Gi\* hot-spot z-score. [R8 §9.1] ``[verified]``

    **Dual API.** If ``values`` is a :class:`FeatureStack`, this matches the
    ARCHITECTURE §11.3 contract ``getis_ord_gi_star(fs, var=LST, k=1)``: it
    computes Gi\* on ``fs[var]`` and writes ``GISTAR_Z`` back into the stack,
    returning the stack. If ``values`` is a numpy array, it returns the Gi\*
    z-score field (the pure-array form the rest of the system uses).

    Exact ArcGIS/ESRI formula (Gi\* includes the focal cell ``w_ii``):

    ``Gi* = (Σ_j w_ij x_j - X̄ Σ_j w_ij) / (S * sqrt{[n Σ_j w_ij^2 - (Σ_j w_ij)^2]/(n-1)})``

    with global mean ``X̄ = (Σ x_j)/n`` and ``S = sqrt{(Σ x_j^2)/n - X̄^2}``.
    The output **is a z-score**: high +z = hot cluster, high -z = cold cluster
    (confidence bins in :data:`constants.HOTSPOT_GISTAR_Z`).

    Spatial weights come from a grid neighbourhood kernel
    (:func:`neighbourhood_weight_kernel`) of half-width ``k`` and the given
    ``contiguity`` (queen / rook / distance band). NaN cells are excluded from
    the global statistics and from neighbour sums; their output is NaN.

    Parameters
    ----------
    values : FeatureStack | np.ndarray
        A FeatureStack (writes ``GISTAR_Z`` on ``var``) or a 2-D / 1-D field.
    coords_or_shape : tuple | None
        If ``values`` is a 1-D array, the ``(rows, cols)`` grid shape to reshape
        to. Ignored for 2-D arrays / FeatureStacks.
    k : int, default 1
        Neighbourhood half-width (kernel is ``2k+1`` square).
    contiguity : str, default "queen"
        Kernel contiguity, see :func:`neighbourhood_weight_kernel`.
    var : str, default ``LST``
        Variable analysed when ``values`` is a FeatureStack.

    Returns
    -------
    FeatureStack | np.ndarray
        The stack (FeatureStack input) or the Gi\* z-score field (array input).
    """
    if isinstance(values, FeatureStack):
        return _gi_star_featurestack(values, var=var, k=k, contiguity=contiguity)
    a = np.asarray(values, dtype=np.float64)
    if a.ndim == 1:
        if coords_or_shape is None:
            raise ValueError("1-D values require coords_or_shape=(rows, cols)")
        a = a.reshape(tuple(coords_or_shape))
    if a.ndim != 2:
        raise ValueError(f"getis_ord_gi_star expects a 2-D field, got {a.ndim}-D")

    valid = np.isfinite(a)
    n = int(valid.sum())
    out = np.full(a.shape, np.nan, dtype=np.float64)
    if n < 2:
        return out

    vals = a[valid]
    x_sum = vals.sum()
    x2_sum = (vals * vals).sum()
    x_bar = x_sum / n
    s_var = x2_sum / n - x_bar * x_bar
    s = np.sqrt(max(s_var, 0.0))
    if s == 0.0:
        return np.where(valid, 0.0, np.nan)

    kernel = neighbourhood_weight_kernel(k=k, contiguity=contiguity, include_self=True)
    sum_wx = _neighbour_sum(a, kernel, valid=valid)        # Σ_j w_ij x_j
    sum_w, sum_w2 = _weight_count(valid, kernel)           # Σ_j w_ij , Σ_j w_ij^2

    numer = sum_wx - x_bar * sum_w
    denom_inner = (n * sum_w2 - sum_w ** 2) / (n - 1)
    denom = s * np.sqrt(np.maximum(denom_inner, 0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(denom > 0, numer / denom, 0.0)
    out[valid] = z[valid]
    return out


# ===========================================================================
# Local Moran's I (LISA)
# ===========================================================================
def local_morans_i(
    values: np.ndarray,
    coords_or_shape: Any = None,
    k: int = 1,
    contiguity: str = "queen",
    return_category: bool = True,
) -> np.ndarray:
    """Local Moran's I (LISA) cluster statistic / category. [R8 §9.2] ``[verified]``

    ``I_i = [(x_i - X̄)/m2] * Σ_j w_ij (x_j - X̄)`` with ``m2 = Σ(x_j-X̄)^2/n``
    and **row-standardized** weights (the standard LISA convention; ``w_ii = 0``).
    Each cell is categorized from the sign of its own deviation and of its
    weighted neighbour mean:

      * **HH** (1) high value, high neighbours  -> hot cluster
      * **LL** (2) low value, low neighbours     -> cool cluster
      * **HL** (3) high value, low neighbours    -> hot outlier
      * **LH** (4) low value, high neighbours     -> cool outlier
      * **ns** (0) not significant (``|I_i|`` below the permutation/normal cut).

    Significance uses an analytical normal approximation of ``I_i`` (z-score
    against the conditional-randomization mean/variance) at the 95% two-tailed
    level; this avoids a heavy permutation loop while staying defensible (the
    R8 cross-check role). When the optional ``esda``/``pysal`` stack is present it
    is **not** required — this is a self-contained numpy/scipy implementation.

    Parameters
    ----------
    values : np.ndarray
        2-D field (or 1-D + ``coords_or_shape``).
    coords_or_shape : tuple | None
        Grid ``(rows, cols)`` if ``values`` is 1-D.
    k, contiguity : see :func:`neighbourhood_weight_kernel`.
    return_category : bool, default True
        If True, return the integer category field (:data:`MORAN_CODES`); if
        False, return the raw ``I_i`` values.

    Returns
    -------
    np.ndarray
        Category codes (0-4) or raw ``I_i`` values, 2-D, same shape as input.
    """
    a = np.asarray(values, dtype=np.float64)
    if a.ndim == 1:
        if coords_or_shape is None:
            raise ValueError("1-D values require coords_or_shape=(rows, cols)")
        a = a.reshape(tuple(coords_or_shape))

    valid = np.isfinite(a)
    n = int(valid.sum())
    cat = np.zeros(a.shape, dtype=np.float64)
    i_field = np.full(a.shape, np.nan, dtype=np.float64)
    if n < 2:
        return cat if return_category else i_field

    x_bar = a[valid].mean()
    dev = np.where(valid, a - x_bar, 0.0)
    m2 = (dev[valid] ** 2).sum() / n
    if m2 == 0.0:
        return cat if return_category else i_field

    # row-standardized weights: divide neighbour-sum by the count of valid nbrs.
    kernel = neighbourhood_weight_kernel(k=k, contiguity=contiguity, include_self=False)
    sum_w, _ = _weight_count(valid, kernel)
    sum_w_safe = np.where(sum_w > 0, sum_w, np.nan)

    lagged_dev_sum = _neighbour_sum(dev, kernel, valid=valid)  # Σ_j w_ij (x_j - X̄)
    lagged_mean = lagged_dev_sum / sum_w_safe                  # row-standardized lag
    z_i = dev / np.sqrt(m2)                                    # standardized own dev
    i_local = z_i * (lagged_dev_sum / sum_w_safe) / np.sqrt(m2)
    i_field = np.where(valid & np.isfinite(sum_w_safe), i_local, np.nan)

    # analytical significance: E[I_i] = -w_i/(n-1) (row-std w_i=1) and an
    # approximate Var giving a normal z; flag |z|>=1.96.
    wi = 1.0  # row-standardized weights sum to 1 per cell
    e_i = -wi / (n - 1)
    # second-moment based variance approximation (Anselin 1995, normal approx).
    b2 = (dev[valid] ** 4).sum() / n / (m2 ** 2)
    var_i = (
        wi * (n - b2) / (n - 1)
        - wi ** 2 * (2.0 * b2 - n) / ((n - 1) * (n - 2))
        - e_i ** 2
    )
    var_i = np.where(np.isfinite(var_i) & (var_i > 0), var_i, np.nan) if np.ndim(var_i) else (
        var_i if (np.isfinite(var_i) and var_i > 0) else np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_stat = (i_field - e_i) / np.sqrt(var_i)
    sig = np.isfinite(z_stat) & (np.abs(z_stat) >= HOTSPOT_GISTAR_Z["p95"])

    if not return_category:
        return i_field

    high = dev > 0
    high_nbr = lagged_mean > 0
    codes = np.zeros(a.shape, dtype=np.float64)
    codes = np.where(sig & high & high_nbr, MORAN_CODES["HH"], codes)
    codes = np.where(sig & (~high) & (~high_nbr), MORAN_CODES["LL"], codes)
    codes = np.where(sig & high & (~high_nbr), MORAN_CODES["HL"], codes)
    codes = np.where(sig & (~high) & high_nbr, MORAN_CODES["LH"], codes)
    codes = np.where(valid, codes, MORAN_CODES["ns"])
    return codes


# ----- FeatureStack-method wrappers (ARCHITECTURE §11.3) --------------------
def _gistar_z_for(fs: FeatureStack, var: str, k: int, contiguity: str) -> np.ndarray:
    return getis_ord_gi_star(fs.get(var).astype(np.float64), k=k, contiguity=contiguity)


def local_moran(fs: FeatureStack, var: str = LST, k: int = 1,
                contiguity: str = "queen") -> FeatureStack:
    """Local Moran's I cluster category on ``var`` -> ``MORAN_LOCAL``. [R8 §9.2]

    Thin FeatureStack wrapper around :func:`local_morans_i` (HH/LL/HL/LH codes in
    :data:`MORAN_CODES`).

    Parameters
    ----------
    fs : FeatureStack
        Stack containing ``var`` (default ``LST``).
    var : str
        Variable to analyse.
    k, contiguity : kernel parameters.

    Returns
    -------
    FeatureStack
        Stack with ``MORAN_LOCAL`` added.
    """
    codes = local_morans_i(fs.get(var).astype(np.float64), k=k, contiguity=contiguity,
                           return_category=True)
    fs.add_layer(MORAN_LOCAL, codes.astype(np.float32))
    return fs


# Public method name from §11.3 is `getis_ord_gi_star`; provide the FeatureStack
# overload via the same callable by dispatching on the first argument type.
def _gi_star_featurestack(fs: FeatureStack, var: str = LST, k: int = 1,
                          contiguity: str = "queen") -> FeatureStack:
    """Compute Gi\* on ``fs[var]`` and write ``GISTAR_Z`` (FeatureStack contract)."""
    z = _gistar_z_for(fs, var, k, contiguity)
    fs.add_layer(GISTAR_Z, z.astype(np.float32))
    return fs


# ===========================================================================
# Surface hotspots (Layer A mask)
# ===========================================================================
def surface_hotspots(fs: FeatureStack, percentile: float = 90.0,
                     gi_z: float = 1.96, k: int = 1,
                     contiguity: str = "queen") -> FeatureStack:
    """Surface ``HOTSPOT_MASK = (LST_PERCENTILE >= p) AND (GISTAR_Z >= gi_z)``. [R8 §10]

    Combines magnitude (percentile band) with statistically-significant
    clustering (Gi\* z) to remove salt-and-pepper speckle and give defensible
    polygons. Computes ``LST_PERCENTILE`` and ``GISTAR_Z`` if they are missing.

    Parameters
    ----------
    fs : FeatureStack
        Stack with ``LST``.
    percentile : float, default 90.0
        Percentile gate (P90 hot; see :data:`constants.LST_PERCENTILE_THRESHOLDS`).
    gi_z : float, default 1.96
        Gi\* z-score gate (95%, :data:`constants.HOTSPOT_GISTAR_Z`).
    k, contiguity : kernel parameters for Gi\*.

    Returns
    -------
    FeatureStack
        Stack with ``HOTSPOT_MASK`` (and ``LST_PERCENTILE``/``GISTAR_Z``) added.
    """
    if not fs.has(LST_PERCENTILE):
        fs.add_layer(LST_PERCENTILE, lst_percentile(fs.get(LST).astype(np.float64)).astype(np.float32))
    if not fs.has(GISTAR_Z):
        _gi_star_featurestack(fs, var=LST, k=k, contiguity=contiguity)

    pct = fs.get(LST_PERCENTILE).astype(np.float64)
    z = fs.get(GISTAR_Z).astype(np.float64)
    mask = (pct >= percentile) & (z >= gi_z)
    fs.add_layer(HOTSPOT_MASK, mask.astype(np.float32))
    return fs


# ===========================================================================
# Heat Vulnerability Index (Exposure / Sensitivity / -Adaptive Capacity)
# ===========================================================================
def _zscore(arr: np.ndarray) -> np.ndarray:
    """Standardize a field to z-scores over its finite cells (NaN-safe)."""
    a = np.asarray(arr, dtype=np.float64)
    mu = np.nanmean(a)
    sd = np.nanstd(a)
    if not np.isfinite(sd) or sd == 0.0:
        return np.where(np.isfinite(a), 0.0, np.nan)
    return (a - mu) / sd


def _minmax01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize a field to [0, 1] over its finite cells (NaN-safe)."""
    a = np.asarray(arr, dtype=np.float64)
    lo = np.nanmin(a)
    hi = np.nanmax(a)
    if not np.isfinite(hi - lo) or hi == lo:
        return np.where(np.isfinite(a), 0.5, np.nan)
    return (a - lo) / (hi - lo)


def _quintiles(arr: np.ndarray) -> np.ndarray:
    """Classify a 0-1 field into quintiles 1..5 over finite cells (NaN -> 0)."""
    a = np.asarray(arr, dtype=np.float64)
    out = np.zeros(a.shape, dtype=np.float64)
    finite = np.isfinite(a)
    if not finite.any():
        return out
    edges = np.nanpercentile(a[finite], [20, 40, 60, 80])
    q = np.ones(a.shape, dtype=np.float64)
    for e in edges:
        q = q + (a > e).astype(np.float64)
    out[finite] = q[finite]
    return out


def _gather_domain_layers(fs: FeatureStack, domain: str) -> list[np.ndarray]:
    """Collect oriented (higher = more vulnerable) indicator fields for a domain.

    Maps the abstract indicator names of :data:`constants.HVI_DOMAINS` onto the
    FeatureStack layers that exist in a satellite-only/synthetic run, applying the
    domain ``sign`` (adaptive-capacity indicators are inverted). Indicators with
    no available proxy are skipped. Each returned field is z-scored & sign-
    oriented so higher always means more vulnerable.
    """
    info = HVI_DOMAINS[domain]
    sign = info["sign"]
    fields: list[np.ndarray] = []

    # Exposure proxies (satellite-native).
    if domain == "exposure":
        if fs.has(LST):
            fields.append(_zscore(fs.get(LST)))
        if fs.has(SUHII):
            fields.append(_zscore(fs.get(SUHII)))
        if fs.has(UTFVI):
            fields.append(_zscore(fs.get(UTFVI)))
        if fs.has(IMPERVIOUS_FRAC):
            fields.append(_zscore(fs.get(IMPERVIOUS_FRAC)))
        if fs.has(NDVI):  # low NDVI -> higher exposure -> invert
            fields.append(-_zscore(fs.get(NDVI)))
        elif fs.has(GREEN_FRAC):
            fields.append(-_zscore(fs.get(GREEN_FRAC)))

    # Sensitivity proxies: population density is the main satellite-era proxy.
    elif domain == "sensitivity":
        if fs.has(POPULATION):
            fields.append(_zscore(fs.get(POPULATION)))

    # Adaptive capacity (sign = -1): more green/AC access -> less vulnerable.
    elif domain == "adaptive_capacity":
        if fs.has(NDVI):  # green access proxy; sign flips it to reduce vulnerability
            fields.append(_zscore(fs.get(NDVI)))
        elif fs.has(GREEN_FRAC):
            fields.append(_zscore(fs.get(GREEN_FRAC)))

    # apply the domain sign so every field reads higher = more vulnerable.
    return [sign * f for f in fields]


def _pca_first_component(stack2d: np.ndarray) -> np.ndarray | None:
    """First-PC score (variance-weighted) of column-features; None if infeasible.

    ``stack2d`` is ``(n_samples, n_features)`` (finite rows only). Tries
    ``sklearn.decomposition.PCA`` lazily; falls back to a numpy SVD. The PC sign
    is oriented to correlate positively with the feature mean (so "more
    vulnerable" stays positive). Returns the per-sample score, or None if there
    are too few features/samples.
    """
    if stack2d.ndim != 2 or stack2d.shape[1] < 2 or stack2d.shape[0] < 3:
        return None
    X = stack2d - stack2d.mean(axis=0, keepdims=True)
    # 1) sklearn if present
    try:  # pragma: no cover - only when sklearn installed
        from sklearn.decomposition import PCA  # type: ignore

        pca = PCA(n_components=1)
        score = pca.fit_transform(X).ravel()
    except Exception:
        # 2) numpy SVD fallback
        try:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            score = (U[:, 0] * S[0])
        except Exception:
            return None
    # orient sign: positive correlation with the mean indicator
    ref = X.mean(axis=1)
    if np.corrcoef(score, ref)[0, 1] < 0:
        score = -score
    return score


def heat_vulnerability_index(
    fs: FeatureStack,
    census: "pd.DataFrame | None" = None,
    indicators: dict[str, list[np.ndarray]] | None = None,
    method: str = "pca",
) -> FeatureStack:
    """Build the Heat Vulnerability Index (0-1) + quintiles -> ``HVI``. [R8 §11]

    Combines the three IPCC domains of :data:`constants.HVI_DOMAINS`
    (Exposure +, Sensitivity +, Adaptive Capacity -). Each domain's available
    indicator fields are z-scored & sign-oriented (higher = more vulnerable); a
    per-domain score is formed by ``method``:

      * ``"pca"``   — first principal component of the domain's indicators
        (lazy ``sklearn`` else numpy SVD :func:`_pca_first_component`); falls back
        to the mean-z when a domain has < 2 indicators.
      * ``"equal"`` — mean of the domain's oriented z-scores.

    Domains are combined with :data:`constants.HVI_DOMAIN_WEIGHTS` (default
    1/3 each), then the composite is min-max normalized to **0-1** and classified
    into **quintiles** (written to ``HVI_QUINTILE``).

    ``census`` (a ``pandas.DataFrame`` of socio-economic indicators joined to the
    grid) is accepted for API completeness; when ``None`` the sensitivity /
    adaptive-capacity domains use the satellite proxies that exist in synthetic
    mode (population density, NDVI green-access). ``indicators`` may override the
    per-domain field lists directly (each value a list of 2-D arrays already
    oriented higher = more vulnerable).

    Parameters
    ----------
    fs : FeatureStack
        Stack with exposure layers (``LST`` / ``SUHII`` / ``IMPERVIOUS_FRAC`` /
        ``NDVI`` ...).
    census : pandas.DataFrame, optional
        Socio-economic table (unused in the synthetic-proxy path).
    indicators : dict[str, list[np.ndarray]], optional
        Explicit per-domain indicator fields, keyed by domain name.
    method : str, default "pca"
        ``"pca"`` or ``"equal"``.

    Returns
    -------
    FeatureStack
        Stack with ``HVI`` (0-1) and ``HVI_QUINTILE`` (1-5) added.
    """
    shape = fs.shape
    domain_scores: list[tuple[float, np.ndarray]] = []

    for domain, weight in HVI_DOMAIN_WEIGHTS.items():
        if indicators is not None and domain in indicators:
            fields = [np.asarray(f, dtype=np.float64) for f in indicators[domain]]
        else:
            fields = _gather_domain_layers(fs, domain)
        if not fields:
            continue

        if method == "pca" and len(fields) >= 2:
            flat = np.stack([f.ravel() for f in fields], axis=1)  # (npix, nfeat)
            finite_rows = np.all(np.isfinite(flat), axis=1)
            score_flat = np.full(flat.shape[0], np.nan, dtype=np.float64)
            pc = _pca_first_component(flat[finite_rows]) if finite_rows.any() else None
            if pc is not None:
                score_flat[finite_rows] = pc
                dscore = score_flat.reshape(shape)
            else:
                dscore = np.nanmean(np.stack(fields, axis=0), axis=0)
        else:
            dscore = np.nanmean(np.stack(fields, axis=0), axis=0)

        domain_scores.append((weight, _zscore(dscore)))

    if not domain_scores:
        # nothing to build from -> neutral 0.5 field.
        hvi = np.full(shape, 0.5, dtype=np.float64)
    else:
        total_w = sum(w for w, _ in domain_scores)
        composite = np.zeros(shape, dtype=np.float64)
        nan_acc = np.zeros(shape, dtype=bool)
        for w, ds in domain_scores:
            composite = composite + (w / total_w) * np.where(np.isfinite(ds), ds, 0.0)
            nan_acc = nan_acc | ~np.isfinite(ds)
        hvi = _minmax01(composite)

    fs.add_layer(HVI, hvi.astype(np.float32))
    fs.add_layer(HVI_QUINTILE, _quintiles(hvi).astype(np.float32))
    return fs


# ===========================================================================
# Layered 5-class composite (Layers A + B + C -> PriorityScore + legend)
# ===========================================================================
def classify_priority(score_0_100: np.ndarray) -> np.ndarray:
    """Classify a 0-100 priority score into the 5-class legend index. [R8 §12.1]

    Bands from :data:`constants.HOTSPOT_LEGEND` (Low 0-20, Moderate 20-40,
    High 40-60, Severe 60-80, Extreme 80-100). Returns the integer legend index
    ``0..4`` (NaN -> -1).

    Parameters
    ----------
    score_0_100 : np.ndarray
        Priority score in [0, 100].

    Returns
    -------
    np.ndarray
        Integer class index 0-4 (float dtype; NaN -> -1).
    """
    s = np.asarray(score_0_100, dtype=np.float64)
    out = np.full(s.shape, -1.0, dtype=np.float64)
    finite = np.isfinite(s)
    cls = np.zeros(s.shape, dtype=np.float64)
    for i, band in enumerate(HOTSPOT_LEGEND):
        # assign class i where score >= band['min']; later (higher) bands win.
        cls = np.where(s >= band["min"], float(i), cls)
    out[finite] = cls[finite]
    return out


def layered_hotspots(fs: FeatureStack, config: Any = None,
                     k: int = 1, contiguity: str = "queen") -> FeatureStack:
    """The deliverable: layered 5-class composite hotspot map. [R8 §12]

    Builds the three transparent layers and the combined priority legend:

    **Layer A - Surface Heat Hotspot** (O(1), LST only):
      ``SurfaceScore = LST_PERCENTILE`` (0-100) and
      ``HOTSPOT_MASK = (LST >= P90) AND (Gi* z >= 1.96)`` via
      :func:`surface_hotspots` (Gi\* :data:`constants.HOTSPOT_GISTAR_Z` p95).

    **Layer B - Human Heat-Stress Hotspot** (>= 3-of-5 cheap-index agreement):
      ``HumanStressScore`` (0-100) from :func:`human_stress_ensemble` when air-T
      + humidity exist; otherwise 0 (surface-only minimal build).

    **Layer C - Vulnerability-Weighted Priority**:
      ``HazardScore = max(SurfaceScore, HumanStressScore)`` and
      ``PriorityScore = 0.5*HazardScore + 0.5*HVI_norm`` (HVI from
      :func:`heat_vulnerability_index`, 0-1 -> 0-100). The result is classified to
      the Low/Moderate/High/Severe/Extreme legend (:data:`constants.HOTSPOT_LEGEND`).

    Writes ``SURFACE_SCORE``, ``HOTSPOT_MASK``, ``GISTAR_Z``, ``LST_PERCENTILE``,
    ``MORAN_LOCAL`` (cross-check), ``HVI``, ``PRIORITY_SCORE`` and
    ``PRIORITY_CLASS``.

    Parameters
    ----------
    fs : FeatureStack
        Stack with at least ``LST`` (minimal Layer-A build); air-T + humidity
        enable Layer B; exposure/population layers enrich Layer C / HVI.
    config : Any, optional
        Reserved for run-config overrides (percentile / gi_z); unused defaults
        match R8 (P90, Gi\* z>=1.96).
    k, contiguity : kernel parameters for Gi\* / Moran.

    Returns
    -------
    FeatureStack
        Stack with the composite layers written.
    """
    # ---- Layer A: surface score + significant-cluster mask ----
    if not fs.has(LST):
        raise KeyError("layered_hotspots requires an 'lst' layer for Layer A")

    surface_hotspots(fs, percentile=LST_PERCENTILE_THRESHOLDS["high"],
                     gi_z=HOTSPOT_GISTAR_Z["p95"], k=k, contiguity=contiguity)
    surface_score = fs.get(LST_PERCENTILE).astype(np.float64)  # already 0-100
    fs.add_layer(SURFACE_SCORE, surface_score.astype(np.float32))

    # Moran's I cross-check on LST (HH clusters corroborate Gi*).
    if not fs.has(MORAN_LOCAL):
        local_moran(fs, var=LST, k=k, contiguity=contiguity)

    # ---- Layer B: human heat-stress agreement score (if met data exist) ----
    if fs.has(AIR_TEMP) and fs.has(REL_HUMIDITY):
        if not fs.has(HUMAN_STRESS_SCORE):
            human_stress_ensemble(fs)
        human_score = fs.get(HUMAN_STRESS_SCORE).astype(np.float64)
    else:
        human_score = np.zeros(fs.shape, dtype=np.float64)

    # ---- Layer C: vulnerability-weighted priority ----
    if not fs.has(HVI):
        heat_vulnerability_index(fs, method="pca")
    hvi_norm = fs.get(HVI).astype(np.float64) * 100.0  # 0-1 -> 0-100

    hazard = np.fmax(surface_score, human_score)        # max(A, B), NaN-tolerant
    priority = 0.5 * hazard + 0.5 * hvi_norm
    priority = np.clip(priority, 0.0, 100.0)

    fs.add_layer(PRIORITY_SCORE, priority.astype(np.float32))
    fs.add_layer(PRIORITY_CLASS, classify_priority(priority).astype(np.float32))
    return fs


def composite_priority(fs: FeatureStack) -> FeatureStack:
    """``PRIORITY_SCORE`` (0-100) = 0.5*Hazard + 0.5*HVI, 5-class legend. [R8 §12]

    The ARCHITECTURE §11.3 entry point; delegates to :func:`layered_hotspots`
    with the R8 defaults (P90 percentile gate, Gi\* z>=1.96, queen contiguity).

    Parameters
    ----------
    fs : FeatureStack
        Stack with ``LST`` (and optionally met / exposure / population layers).

    Returns
    -------
    FeatureStack
        Stack with ``PRIORITY_SCORE`` (+ ``PRIORITY_CLASS`` and the supporting
        Layer-A/B/C fields) added.
    """
    return layered_hotspots(fs)
