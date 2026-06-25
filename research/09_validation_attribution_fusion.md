# 09 — Validation, Uncertainty, Driver Attribution & Data Fusion / Gap-Filling
### The "robustness glue" for the ISRO BAH-2026 PS-1 Urban Heat system (Research Agent R9)

> **Role of this document.** The other research streams produce *inputs* (multi-satellite LST, LULC, urban
> morphology, meteorology, spectral indices) and *models* (physics-informed ML for LST, cooling-scenario
> optimization). This document is the cross-cutting layer that (a) **validates** the AIML model defensibly,
> (b) **quantifies and ranks the drivers** of urban heat with explainable methods, and (c) **fuses, gap-fills,
> and cross-verifies** every input into one robust product. PS-1 explicitly demands a *"Validated AIML model"*
> and a *"Quantitative assessment of key drivers"* — those two outcomes live here.
>
> A stakeholder priority is **≥30 methods/datasets that cross-verify and fill each other's gaps**. Section 4
> is the numbered **robustness matrix** (35 entries) that operationalizes that priority. It is the key
> deliverable of this file.

**Status of evidence.** Every core method below was verified against peer-reviewed literature or official tool
documentation via web search (June 2026). Items I could not pull a fresh citation for are marked
`from-knowledge (verify)`. Citations with URLs are in Section 7.

---

## 0. Overview & design philosophy

Three principles drive the whole robustness layer:

1. **No single source is trusted alone.** Each LST sensor, each LULC product, each met source has a
   characteristic error and a characteristic *gap*. We pair every source with at least one independent source
   that fills its gap and at least one that can verify it. This is the literal meaning of "cross-verify and
   fill each other's gaps."
2. **Validation must respect geography and time.** Random k-fold cross-validation **leaks** spatial
   autocorrelation and over-states skill (literature reports optimistic bias up to ~28%). We use **spatial
   and temporal** cross-validation as the default, and we *report* the random-CV number only as a (deliberately
   inflated) upper bound to make the point.
3. **Attribution must be quantitative, ranked, and spatially explicit.** "Vegetation matters" is not an
   answer. We deliver a *ranked, %-contribution* table (global) **and** spatially-varying coefficient maps
   (local), with at least two independent attribution methods agreeing before any driver claim is made.

```
        INPUTS (other teams)                 R9 ROBUSTNESS LAYER                 DELIVERABLES (PS-1)
  ┌─────────────────────────────┐     ┌──────────────────────────────┐     ┌────────────────────────┐
  │ Landsat / ECOSTRESS / MODIS │     │  FUSION & GAP-FILLING        │     │ Heat-stress maps       │
  │ VIIRS / Sentinel-3  (LST)   │ ──▶ │  (STARFM, harmonic, kriging, │ ──▶ │  (gap-free, fused LST) │
  │ S2 / Landsat / Dynamic World│     │   GP, triple collocation,    │     │                        │
  │ ESA WC / ESRI / GHSL (LULC) │     │   Bayesian assimilation, OI) │     │ Ranked driver          │
  │ OSM / GHSL / UT-GLOBUS(morph)│ ──▶ │  CROSS-VERIFICATION          │ ──▶ │  attribution + maps    │
  │ ERA5 / CPCB / IMD / Netatmo │     │  (ensemble agreement, TC,    │     │                        │
  │ NDVI/NDBI/NDWI/albedo (idx) │     │   MC uncertainty)            │     │ VALIDATED AIML model   │
  └─────────────────────────────┘     │  VALIDATION (spatial CV,     │ ──▶ │  (spatial-CV metrics,  │
                                       │   metrics, ground truth)     │     │   per-LCZ error, unc.) │
                                       │  ATTRIBUTION (SHAP, GWR,     │     │                        │
                                       │   ALE, LMG, perm-imp)        │     │ Uncertainty maps       │
                                       └──────────────────────────────┘     └────────────────────────┘
```

---

## 1. Validation methodology (spatial CV, metrics, ground truth)

### 1.1 Why random cross-validation is wrong for this problem

LST, LULC, morphology and meteorology are all **spatially autocorrelated** (Tobler's first law: near things are
more similar). With random k-fold CV, a held-out test pixel almost always has a near-neighbour in the training
set, so the model is effectively interpolating between adjacent pixels rather than predicting an unseen
location. Result: **over-optimistic skill**. Roberts et al. (2017, *Ecography*) and the GeoAI-Handbook chapter
on spatial CV both document this; reported optimism can reach ~28% (MDPI *Water* 2023). The same logic applies
to **temporal** autocorrelation for time-series LST.

> **Rule for the project:** the headline validation numbers MUST come from spatial (and where relevant,
> temporal) CV. We may *additionally* report random-CV as an explicit "leaky upper bound" to demonstrate the
> gap — a strong defensive move for the jury.

### 1.2 Spatial cross-validation schemes (use all three, report the most conservative)

| Scheme | What it does | When to use | Tooling |
|---|---|---|---|
| **Spatial block k-fold** | Partition study area into contiguous blocks (rectangles/hex/polygons), assign whole blocks to folds; train/test never share a block | Default headline metric | `verde.BlockKFold` (scikit-learn-compatible); R `blockCV`; or `sklearn.model_selection.GroupKFold` with a block-ID group column |
| **Buffered / spatial leave-one-out (SLOO)** | Hold out one point (or block); remove all training points inside a *buffer* (≈ autocorrelation range) around it | Local accuracy / dense in-situ networks | `blockCV::cv_buffer` (R); custom buffer in Python via `verde` distances |
| **Environmental / feature-space blocking** | Block by covariate space (e.g., LCZ class, elevation band) to test extrapolation | Stress-test transferability to unseen conditions | custom (group by stratum) |

Block size should be tied to the empirical **range of the variogram** of the residuals (autocorrelation length)
— pick blocks at least that large so train/test are decorrelated. `verde.BlockKFold` exposes `spacing`/`shape`
to set this directly and balances folds to roughly equal N.

> **Caveat to disclose (intellectual honesty).** Wadoux et al. (2021, *Ecol. Modelling*) argue spatial CV is
> *not* the right tool for estimating **map accuracy** over the whole area when a **probability (design-based)
> sample** is available — for population-level map accuracy, design-based validation on a random/stratified-random
> probability sample is statistically defensible. We therefore use **two** validation regimes for two different
> questions:
> 1. *Model skill / transferability* → spatial CV (block + SLOO).
> 2. *Map accuracy over the city* → design-based estimate on an independent stratified-random probability sample
>    of in-situ / held-out-sensor points (when available).
> Reporting both pre-empts the obvious reviewer objection in either direction.

### 1.3 Temporal cross-validation

For the LST time series and any seasonal model: use **forward-chaining / rolling-origin** splits
(`sklearn.model_selection.TimeSeriesSplit`) so the test period is always *after* the training period. Never
shuffle dates. For combined space-time, use **leave-time-and-space-out** (block in space AND hold out whole
dates) — the strictest, and the right thing for "predict a new heatwave day at an unseen location."

### 1.4 Metrics (report the full panel, not just R²)

Compute on the spatial-CV test folds and report mean ± SD across folds:

| Metric | Formula / meaning | Why include it |
|---|---|---|
| **RMSE** | √(mean((ŷ−y)²)) | Primary error magnitude (penalizes large misses) |
| **MAE** | mean(|ŷ−y|) | Robust error magnitude (less outlier-sensitive) |
| **Bias / MBE** | mean(ŷ−y) | Systematic over/under-estimation; pairs with RMSE |
| **ubRMSE** | √(RMSE²−bias²) | *Random* error after removing bias (standard in LST/soil-moisture validation) |
| **R²** | coefficient of determination | Variance explained — but can be high with biased models, so never alone |
| **NSE** (Nash–Sutcliffe) | 1 − Σ(ŷ−y)²/Σ(y−ȳ)² | Skill vs. "predict the mean"; NSE≤0 means model is useless. Range (−∞,1] |
| **Lin's CCC** | concordance of (ŷ,y) about the 1:1 line | Combines precision *and* accuracy (bias-aware agreement); better than Pearson r for "do predictions track truth on the 1:1 line" |
| **KGE** (Kling–Gupta) | combines r, bias ratio, variability ratio | Decomposable diagnostic; complements NSE |

> Note (verified): NSE/KGE have known pathologies (Knoben et al.; recent 2025 critiques) — they are reported as
> *part of a panel*, never as the sole criterion. RMSE + bias + ubRMSE + CCC is the defensible core for LST.

### 1.5 Stratified & spatial error reporting (this is what makes attribution credible)

- **Per-LCZ stratified error.** Break every metric out by **Local Climate Zone** (Stewart & Oke LCZ scheme) —
  compact high-rise vs. open low-rise vs. dense trees vs. water. A model that is accurate over built classes but
  poor over vegetation is *not* validated for cooling-scenario claims, and only stratification reveals it.
- **Per-LULC stratified error.** Same idea by land-cover class (built / vegetation / water / bare).
- **Residual maps.** Map (ŷ−y) in space and inspect for **residual spatial autocorrelation** (Moran's I on
  residuals). Structured residuals ⇒ a missing covariate or a needed spatial term (e.g., add GWR/kriging of
  residuals — "regression-kriging"). This is both a diagnostic and a gap-fill (see §3).
- **Error vs. driver scatter.** Plot residuals against each driver to detect heteroscedastic / conditional bias.

### 1.6 Ground truth & independent reference data

Because there is no perfect "truth" for satellite LST, we use a **layered** ground-truth strategy:

1. **In-situ air temperature (T_air) networks — India-specific:**
   - **IMD** (India Meteorological Department) AWS/observatory stations.
   - **CPCB** (Central Pollution Control Board) continuous ambient air-quality stations — many report
     temperature/met variables and are dense in Indian metros (explicitly named in the PS-1 dataset list).
   - Local **AWS** / municipal smart-city sensors where available.
   - *Caveat:* T_air ≠ LST. Validate the **air-temperature / heat-stress** products directly against these, and
     validate **LST** against LST references (below). Use T_air↔LST relationships (e.g., TsHARP / regression) only
     with documented uncertainty.
2. **Crowdsourced citizen weather stations (Netatmo):** dense in cities, ±0.3 K sensor spec but known **solar
   warm-bias** and siting errors ⇒ MUST pass a quality-control pipeline (**CrowdQC+** / Meier et al. scheme:
   outlier + spatial-consistency + radiation filters) before use. Excellent for *spatial pattern* validation of
   the UHI even if absolute calibration is imperfect (Venter et al., *Sci. Adv.* 2021 used exactly this to
   contrast crowdsourced T_air against satellite LST).
3. **Independent held-out LST sensors (cross-sensor validation):** train/produce on one sensor (e.g.,
   Landsat-derived LST) and validate against a *different*, withheld sensor at matched overpass
   (ECOSTRESS 70 m, MODIS MOD11/MYD11 1 km, VIIRS, Sentinel-3 SLSTR, ASTER). Published intercomparisons give us
   *expected* agreement envelopes to benchmark against: daytime ECOSTRESS bias ≈ −0.9 K / RMSE ≈ 2.2 K vs. in-situ;
   MOD11 bias <0.8 K / RMSE <2.8 K; Landsat/VIIRS/Sentinel-3 bias <2.1 K / RMSE <4.5 K (Li et al., Corn-Belt
   benchmark; Bayat et al., Europe/Africa ECOSTRESS evaluation). If our fused product sits inside these envelopes
   against an independent sensor, that is strong evidence of validity.
4. **Field campaigns (optional, high value):** handheld IR radiometers / mobile transects / fixed thermal loggers
   on a heatwave day for a few neighbourhoods — gold-standard spot checks for the most policy-relevant hotspots.

### 1.7 Physics-based sanity checks (physics-informed validation)

Beyond statistical metrics, the model must **respect physics** — this is the "physics-informed" mandate:

- **Surface Energy Balance (SEB) closure:** Rn = H + LE + G. Check that predicted LST is consistent with the SEB
  given net radiation, sensible/latent/ground heat flux proxies; gross violations flag a bad model even if RMSE
  looks fine.
- **Sign & monotonicity of driver effects (must match physics):** higher NDVI/vegetation & water ⇒ *lower* LST
  (evapotranspiration cooling); higher NDBI/impervious/building density & lower albedo ⇒ *higher* LST; higher
  wind ⇒ generally *lower* daytime UHI. If SHAP/ALE/GWR show the *wrong sign*, the model is fitting spurious
  correlation — reject or constrain. (Optionally **enforce** these with monotonic constraints in XGBoost/LightGBM
  — a concrete physics-informed lever.)
- **Energy/units & range checks:** LST within plausible physical bounds for the scene; albedo ∈ [0,1]; emissivity
  ∈ ~[0.9,1.0].
- **Conservation in cooling scenarios:** a greening intervention cannot *increase* citywide available energy;
  predicted ΔT from interventions must be bounded by physically plausible cooling (SOLWEIG / SEB cross-check).

---

## 2. Driver attribution methodology (quantitative, ranked, explainable)

Goal: a **defensible ranked %-contribution** of the four PS-1 driver families — **LULC, urban morphology,
vegetation, atmospheric conditions** — both **globally** (one ranking for the city) and **locally** (where each
driver dominates). Rule: **no driver claim unless ≥2 independent methods agree** on sign and rough rank.

### 2.1 Global, model-agnostic importance (machine-learning side)

| Method | What it gives | Strength | Watch-out |
|---|---|---|---|
| **SHAP** (TreeSHAP for XGB/RF/LightGBM) | Per-prediction additive attributions; **mean |SHAP|** = global importance; dependence & **interaction** plots | Theoretically grounded (Shapley), local+global, handles interactions | Correlated features split credit; use clustering/grouped-SHAP for collinear drivers |
| **Permutation importance** | Drop in skill when a feature is shuffled | Simple, model-agnostic, ties importance to *predictive* value | Inflated/misleading under correlation ⇒ use **conditional** permutation |
| **Drop-column importance** | Retrain without a feature; Δskill | Most faithful "what does this feature buy us" | Expensive (retrain per feature) |
| **ALE plots** | Accumulated Local Effects = average local effect of a feature | **Unbiased under correlated features** (unlike PDP); fast | Less intuitive than PDP |
| **Partial dependence (PDP)** | Marginal effect of a feature | Intuitive shape of effect | **Biased if features correlated** (assumes independence) ⇒ pair with ALE |
| **Gain / split importance** (tree native) | Σ impurity reduction | Free from the model | Biased toward high-cardinality; use only as cross-check |

**Group the features into the four PS-1 families** and aggregate SHAP within group ⇒ a **family-level ranked
importance** directly answering "LULC vs. morphology vs. vegetation vs. atmosphere."

### 2.2 Variance partitioning / relative-importance (statistics side — gives clean % shares)

These produce a **clean, additive % decomposition** that is easy to communicate to a jury and cross-checks the
ML importances:

- **LMG / Shapley-regression relative weights** (R `relaimpo` `calc.relimp(type="lmg")`; Python
  via Shapley-value regression): partitions model R² across predictors **averaging over all orderings** —
  handles correlated predictors fairly, sums to total R². The cleanest "% of explained variance per driver."
- **Hierarchical partitioning** (R `hier.part`): splits each predictor's contribution into **independent** vs.
  **joint** components — explicitly surfaces how much is shared (collinearity) vs. unique.
- **Commonality analysis:** decomposes R² into unique + all common components — full bookkeeping of shared
  variance among, e.g., NDVI/NDBI/albedo.
- **Dominance analysis:** pairwise "which predictor dominates" — robust ranking.

> Why both ML-SHAP **and** variance partitioning? They are independent paradigms (game-theoretic on a flexible
> model vs. variance decomposition on a regression). **Agreement between them is the validation of the
> attribution itself.** Disagreement is a flag (usually collinearity) to investigate, not to hide.

### 2.3 Spatially-varying attribution (the local story)

Global importance hides that drivers matter **differently in different neighbourhoods**. Provide
**coefficient maps**:

- **GWR / MGWR** (Geographically Weighted Regression / Multiscale GWR): fits a local regression at every location,
  yielding **per-pixel coefficient maps** for each driver and a local-R² map. Verified in UHI literature:
  building density, green-view index, road density, impervious ratio are dominant and **spatially non-stationary**;
  MGWR additionally lets each driver act at its **own spatial scale**. Tools: `mgwr` (Python, PySAL), R `GWmodel`,
  R `spgwr`. Report GWR coefficient maps **per driver** + a "which-driver-dominates" map.
- **Geographically-weighted / local SHAP:** map mean |SHAP| per driver spatially (or fit GWR on SHAP values) to
  show where vegetation-deficit vs. morphology vs. albedo drives the local hotspot — directly actionable for
  *placement* of cooling interventions.
- **Spatial regimes / cluster-then-attribute:** segment the city (LCZ or LISA clusters) and run attribution per
  regime.

### 2.4 The deliverable attribution product

1. **Global ranked table:** four driver families ranked by (a) mean |SHAP| share and (b) LMG R²-share, shown
   side-by-side; agreement = confidence.
2. **Per-driver effect shapes:** ALE (primary) + PDP (intuition) + SHAP dependence, with **physics sign check**.
3. **Top interactions:** SHAP interaction values (e.g., NDVI×albedo, building-density×wind).
4. **Spatial coefficient maps:** GWR/MGWR per driver + dominant-driver map + local-R² map.
5. **Uncertainty on attribution:** repeat SHAP/LMG across spatial-CV folds / bootstrap ⇒ error bars on each
   driver's contribution (so the ranking is *itself* validated).

---

## 3. DATA FUSION & GAP-FILLING methods catalog (the ≥30-method robustness core)

This is the heart of the robustness theme. Every LST sensor has clouds/orbit gaps and a different
resolution/revisit; every LULC and morphology source has omissions. The catalog below is the toolbox; Section 4
assembles it into the numbered matrix.

### 3.1 Multi-sensor LST fusion & cloud/orbit gap-filling

- **Spatiotemporal fusion (STARFM / ESTARFM / FSDAF):** blend **high-spatial-low-temporal** (Landsat/ECOSTRESS,
  30–70 m) with **low-spatial-high-temporal** (MODIS/VIIRS/geostationary) to synthesize **30 m, frequent,
  near-gap-free** LST. ESTARFM improves over STARFM in heterogeneous (urban) scenes; **FSDAF** ("Flexible
  Spatiotemporal DAta Fusion") needs only one input pair and handles land-cover change — good for cities.
  Verified: these were built for reflectance but are routinely applied to LST.
- **Thermal sharpening / disaggregation (TsHARP, DisTrad, DMS):** sharpen coarse MODIS/VIIRS LST to Landsat
  resolution using fine NDVI/NDBI/albedo predictors — a *cross-source* fill where a fine LST is missing but
  fine indices exist.
- **Harmonic / Fourier temporal modeling (e.g., HANTS, Annual Temperature Cycle "ATC"):** fit periodic
  (diurnal+annual) harmonics to each pixel's LST time series; predict the cloud-obscured value from the fitted
  cycle. Strong, cheap **temporal** gap-fill; basis of many all-weather LST reconstructions.
- **Regression on ancillary covariates:** predict missing LST from **NDVI, DEM/elevation, slope/aspect, LULC,
  albedo, distance-to-water, land-cover class** via RF/XGBoost/GBR — the gap-fill *uses the same drivers* the
  attribution model studies, so it is internally consistent. (MDPI *RS* 2020 ARD gap-fill; Sci.Rep./PMC
  gap-filled Landsat-8 LST datasets.)
- **Gaussian Process / kriging interpolation (incl. regression-kriging, NNGP):** model LST as a GP/random field;
  interpolate gaps with a **calibrated uncertainty** at every filled pixel. Nearest-Neighbour GP scales to
  satellite rasters. Regression-kriging = covariate regression + kriging of residuals (also fixes the §1.5
  residual-autocorrelation finding).
- **All-weather LST (passive-microwave merge):** microwave (e.g., AMSR2) sees through cloud at coarse
  resolution; merge with thermal-IR LST to get cloud-penetrating LST, then sharpen. `from-knowledge (verify)`
  for the specific Indian-city implementation.

### 3.2 Multi-sensor reconciliation (bias-correct to a common reference)

Different LST sensors disagree systematically (different overpass time, viewing angle, emissivity model,
band-pass). Before fusion, **harmonize**:

- **Pairwise bias correction / histogram (CDF) matching / linear rescaling** of MODIS↔Landsat↔ECOSTRESS↔VIIRS↔
  Sentinel-3 to a chosen reference (often the most-validated sensor for that time-of-day). Account for
  **overpass-time** and **view-angle** differences (BRDF/angular normalization) explicitly.
- **Cross-sensor regression / quantile mapping** learned over overlapping clear-sky pixels.
- Outcome: an internally consistent **multi-sensor LST stack** that fusion (§3.1) can blend without seams.

### 3.3 Error estimation *without* ground truth — Triple Collocation (TC)

**Triple Collocation** estimates the **random-error variance of three (or more) mutually independent datasets**
of the same variable **without assuming any of them is truth**. For LST we can triple-collocate, e.g.,
**ECOSTRESS × MODIS × ERA5-skin-temperature** (or station-derived), getting a per-dataset error and (Extended
TC / McColl 2014) correlation-to-truth. Used to (a) **rank sensor quality** objectively, (b) derive **optimal
fusion weights** (inverse-error weighting), and (c) make **spatial error maps**. Assumptions to respect:
zero-mean errors, errors mutually uncorrelated, linearity — choose genuinely independent triplets (don't mix two
products that share the same ancillary inputs). Verified across SST/soil-moisture/precip; directly transferable.

### 3.4 Statistical/Bayesian fusion & data assimilation

- **Bayesian data assimilation / Optimal Interpolation (OI):** combine a **background** (model or harmonic-fit
  estimate) with **observations** weighted by their **error covariances** ⇒ posterior LST with uncertainty. OI =
  kriging = the linear-Gaussian case of Bayesian assimilation (one unified framework). Kalman filter/smoother for
  the sequential (time-stepping) version. This is the principled way to "combine model + observations."
- **Bayesian model averaging (BMA) / weighted ensemble:** combine multiple LST products weighted by validated
  skill (e.g., TC-derived weights) ⇒ a single best estimate + predictive variance.
- **Cokriging / multivariate GP:** interpolate LST jointly with a correlated covariate (NDVI, elevation) to
  borrow strength.

### 3.5 Cross-verification *between data sources* (the "fill each other's gaps" engine)

This is the explicit cross-checking that makes the matrix in §4 coherent:

- **ERA5 ⇄ station (CPCB/IMD/Netatmo):** use ERA5 2 m air-temp to **QC station outliers** (and the reverse: use
  stations to **bias-correct / downscale** ERA5). Each catches the other's failure mode (stations: local siting
  errors & gaps; ERA5: coarse 0.25°/~31 km, misses intra-urban gradient).
- **NDVI constrains LST gap-fill:** vegetation fraction physically bounds plausible LST (dense canopy can't be the
  hottest pixel) ⇒ use NDVI/NDWI as a **constraint/regularizer** on interpolated LST and to flag implausible
  fills.
- **Multiple building/footprint datasets cross-fill:** **OSM** (crowd, incomplete but detailed where present) +
  **GHSL built-up/height** (global, consistent) + **Microsoft/Google open building footprints** +
  **UT-GLOBUS** (urban morphology, if available) ⇒ union-fill footprints, majority-vote on built/not-built,
  reconcile heights. Each fills the others' omissions.
- **Multiple LULC products → majority voting + agreement map:** **ESA WorldCover 10 m**, **ESRI 10 m Land Cover**,
  **Dynamic World (near-real-time)**, **GHSL** built layer, plus a project-trained **S2/Landsat** classifier ⇒
  per-pixel **majority vote** for the consensus class and a **disagreement map** that localizes where LULC is
  uncertain (and where the model should distrust LULC-driven predictions).
- **Independent LST sensor verifies the fused LST** (cross-sensor, §1.6) and **station/Netatmo verifies the
  air-temp/heat-stress** product.

### 3.6 Ensemble agreement / disagreement & uncertainty propagation

- **Ensemble agreement maps:** stack all candidate LST (or hotspot) estimates; map pixel-wise **mean** and
  **spread (SD/IQR)** ⇒ "confidence map." High-spread pixels = low confidence = candidates for field check.
- **Majority voting** for categorical layers (LULC, hotspot/not-hotspot) with a per-pixel **vote-margin** =
  agreement metric.
- **Monte Carlo uncertainty propagation:** perturb inputs by their estimated error distributions (sensor noise,
  TC-derived error, classification probabilities), re-run the LST model / hotspot detection / cooling-scenario
  many times ⇒ **output probability distribution** and **per-pixel uncertainty**. Standard in RS (TOA-reflectance→
  land-cover→fractional-cover MC budgets). This is how attribution and cooling-ΔT claims get honest error bars.

---

## 4. ≥30 cross-verifying methods/datasets — ROBUSTNESS MATRIX

Each row is a distinct **data source (D)** or **analytical method (M)**. For each: its **role**, the **gap it
fills**, and **what verifies it** (the cross-check). The design intent: every source is *both* a gap-filler for
others *and* itself verified by an independent source — that is the "cross-verify and fill each other's gaps"
mandate made concrete. (35 entries.)

| # | Name | Type | Role in system | Gap it fills | What verifies / cross-checks it |
|---|---|---|---|---|---|
| 1 | **Landsat 8/9 LST** (TIRS, 30 m, 16-day) | D | High-res LST backbone for hotspots & model target | Fine spatial detail of intra-urban heat | ECOSTRESS & MODIS at matched overpass (#2,#3); in-situ (#10,#11) |
| 2 | **ECOSTRESS LST** (70 m, diurnal/ISS) | D | Variable-time-of-day LST; captures diurnal cycle | Landsat's fixed ~10:30 overpass (diurnal gap) | Landsat (#1) & ASTER; in-situ bias≈−0.9K/RMSE≈2.2K benchmark |
| 3 | **MODIS LST** (MOD11/MYD11, 1 km, 4×/day) | D | High-temporal LST; harmonic-fit & fusion base | Landsat/ECOSTRESS low revisit (temporal gap) | Triple collocation (#21); validated MOD11 bias<0.8K |
| 4 | **VIIRS LST** (~750 m, daily) | D | Extra high-temporal LST after MODIS era | MODIS continuity / extra daily sample | Cross-sensor reconciliation (#20); MODIS (#3) |
| 5 | **Sentinel-3 SLSTR LST** (1 km) | D | Independent 4th LST stream | Adds an independent triplet member for TC | TC (#21); MODIS/VIIRS agreement |
| 6 | **Sentinel-2 MSI** (10 m, 5-day) | D | Fine vegetation/indices; LULC classifier input; sharpening predictor | Fine NDVI/NDBI/NDWI for thermal sharpening & gap-fill | Landsat indices (#7); Dynamic World (#15) |
| 7 | **Spectral indices** (NDVI, NDBI, NDWI, NDISI, albedo, UI) | M/D | Driver covariates; **constraints** on LST gap-fill; sharpening | Physically bounds & predicts missing LST | Mutual consistency (NDVI vs NDWI); SHAP sign check (#24) |
| 8 | **ESA WorldCover 10 m** | D | LULC layer (member of vote) | Global consistent land cover | Majority vote vs #15,#16,#19 (#27) |
| 9 | **ESRI 10 m Land Cover** | D | LULC layer (member of vote) | Independent annual LULC | Agreement map (#27) |
| 10 | **IMD stations / AWS** | D | In-situ air-temp ground truth | Absolute calibration of air-temp/heat-stress | ERA5 cross-QC (#13); CPCB (#11) |
| 11 | **CPCB ambient stations** | D | Dense in-situ met in Indian metros (PS-1 named) | Intra-urban T_air where IMD sparse | IMD (#10); Netatmo (#12); ERA5 (#13) |
| 12 | **Netatmo citizen weather stations** | D | Dense crowdsourced T_air (spatial UHI pattern) | Station-network spatial sparsity | CrowdQC+ QC (#28); CPCB/IMD reference (#10,#11) |
| 13 | **ERA5 / ERA5-Land reanalysis** | D | Atmospheric drivers (T_air, RH, wind) + skin temp | Wall-to-wall met where stations absent | Station cross-check (#10–12); coarse→downscaled |
| 14 | **GHSL** (built-up surface, height, pop) | D | Urban morphology + exposure; built mask | Consistent global built-up & population | OSM (#17) & open footprints (#18) cross-fill (#26) |
| 15 | **Dynamic World** (near-real-time 10 m LULC) | D | Time-aware LULC + class probabilities | Up-to-date LULC & per-pixel uncertainty | Vote vs #8,#9 (#27); probabilities feed MC (#30) |
| 16 | **Project-trained S2/Landsat LULC classifier** | M | Local, tuned LULC for the study city | Local classes generic products miss | Majority vote & confusion vs #8,#9,#15 (#27) |
| 17 | **OpenStreetMap (buildings/roads)** | D | Detailed morphology where mapped | Fine street/building geometry | GHSL (#14) & open footprints (#18) fill OSM gaps (#26) |
| 18 | **Microsoft/Google open building footprints** | D | Wide-coverage footprints | OSM omissions in unmapped areas | OSM (#17) + GHSL height (#14) reconcile (#26) |
| 19 | **UT-GLOBUS** (urban morphology, if available) | D | Building-level morphology params (PS-1 named) | Detailed 3-D morphology for SEB/SOLWEIG | GHSL height (#14); OSM (#17) |
| 20 | **Multi-sensor reconciliation** (bias-corr / CDF match / view-angle norm) | M | Harmonize all LST sensors to common reference | Inter-sensor systematic bias (seam removal) | Post-harmonization overlap RMSE; TC (#21) |
| 21 | **Triple Collocation (incl. Extended TC)** | M | Error variance & fusion weights for ≥3 indep LST | "No truth" error estimation & weighting | Independent in-situ check (#10–12); ensemble spread (#29) |
| 22 | **STARFM / ESTARFM / FSDAF spatiotemporal fusion** | M | Fuse hi-space×hi-time → 30 m frequent LST | Cloud/orbit + revisit gaps simultaneously | Hold-out clear Landsat scene; cross-sensor (#1–3) |
| 23 | **Thermal sharpening** (TsHARP/DisTrad/DMS) | M | Disaggregate coarse LST to 30 m via indices | Fine LST missing but fine indices present | Coincident Landsat LST (#1) |
| 24 | **Harmonic / ATC temporal modeling (HANTS)** | M | Fit diurnal+annual cycle; fill temporal gaps | Cloud-obscured dates in LST series | Adjacent clear observations; GP residuals (#25) |
| 25 | **Gaussian Process / (regression-)kriging / NNGP** | M | Spatial gap-fill **with uncertainty**; residual kriging | Spatial holes; residual autocorrelation (§1.5) | Cross-validation at withheld pixels; variogram fit |
| 26 | **Building-dataset cross-fill** (OSM∪GHSL∪open∪UT-GLOBUS) | M | Union-fill footprints, reconcile heights | Each footprint source's omissions | Mutual overlap agreement; field/imagery spot-check |
| 27 | **LULC majority voting + agreement map** | M | Consensus land cover + uncertainty localization | Single-product classification errors | Disagreement map flags low-confidence pixels |
| 28 | **CrowdQC+ / Meier QC for crowdsourced T_air** | M | Filter Netatmo solar-bias & siting outliers | Raw crowdsourced data unreliability | Spatial-consistency + reference stations (#10,#11) |
| 29 | **Ensemble agreement / disagreement maps** | M | Mean + spread across all LST/hotspot estimates | Single-estimate over-confidence | High-spread pixels → field check (#1.6) |
| 30 | **Monte Carlo uncertainty propagation** | M | Propagate input errors → output PDF & per-pixel σ | Missing uncertainty on LST/attribution/cooling-ΔT | TC error inputs (#21); spatial-CV spread (#31) |
| 31 | **Spatial cross-validation** (BlockKFold / buffered SLOO) | M | Honest model skill without spatial leakage | Random-CV optimism (~28%) | Design-based probability-sample accuracy (#32) |
| 32 | **Design-based validation on probability sample** | M | Population map-accuracy estimate (Wadoux critique) | Spatial-CV's map-accuracy limitation | Spatial CV (#31) as the skill counterpart |
| 33 | **SHAP (global + dependence + interaction)** | M | ML-side ranked driver importance & effects | Black-box opacity; interaction discovery | LMG/variance partitioning (#34); GWR sign (#35) |
| 34 | **Variance partitioning (LMG / hierarchical / dominance)** | M | Clean additive %-share of R² per driver | SHAP collinearity ambiguity | Agreement with SHAP ranking (#33) |
| 35 | **GWR / MGWR spatially-varying coefficients** | M | Per-pixel driver coefficient & dominance maps | Global importance hides spatial non-stationarity | Local-R² map; physics sign check; SHAP (#33) |

> **How to read the matrix as "robustness":** follow any column. *Gaps filled* shows the system has redundant
> coverage for every failure mode (cloud, orbit, revisit, sparse stations, footprint omission, classification
> error, opacity, spatial leakage, missing uncertainty). *What verifies it* shows every source/method has an
> **independent** check — nothing is self-certified. 5 LST sensors + 4 LULC products + 4 footprint sources +
> 3 met sources + ~19 analytical methods = a genuinely multi-source, cross-verified pipeline (35 distinct
> entries ≥ the ≥30 target).

---

## 5. Uncertainty quantification (end-to-end budget)

We carry uncertainty through **every** stage and surface it as **maps**, not just a single number.

1. **Observation / sensor uncertainty.** Per-sensor LST error from published validation **and** from
   **Triple Collocation** (data-driven, scene-specific). → inputs to weighting & MC.
2. **Fusion / gap-fill uncertainty.** GP/kriging gives a **per-pixel posterior variance**; OI/Bayesian
   assimilation gives posterior covariance; harmonic-fit residual variance for temporally-filled pixels. → a
   "fill-confidence" raster (native pixels = 0 extra uncertainty; deeply interpolated pixels = high).
3. **Classification uncertainty.** LULC per-pixel class probabilities (Dynamic World) + majority-vote margin +
   disagreement map. → categorical uncertainty layer.
4. **Model (epistemic) uncertainty.** Variability of predictions and of SHAP/LMG **across spatial-CV folds** and
   **bootstrap**; optionally quantile-regression / NGBoost / conformal prediction for **prediction intervals**
   on LST. → error bars on metrics *and* on the driver ranking.
5. **Propagation.** **Monte Carlo**: sample from (1)–(4), re-run LST→hotspot→attribution→cooling-ΔT, summarize
   mean & σ per pixel. → final **uncertainty map** accompanying every headline product.
6. **Reporting.** Every deliverable map ships with a **paired uncertainty map**; the driver table ships with
   **confidence intervals**; cooling-intervention ΔT ships as **ΔT ± σ (°C)** so recommendations are honest.

> This directly satisfies "Validated AIML model" (skill + uncertainty, spatially stratified) and makes the
> "quantitative assessment of key drivers" *defensible* (ranked with error bars and two agreeing methods).

---

## 6. How this ties the whole project together (integration checklist)

- [ ] Receive multi-sensor LST (Landsat/ECOSTRESS/MODIS/VIIRS/Sentinel-3) → **reconcile** (#20) → **TC** error &
      weights (#21) → **fuse + gap-fill** (#22–25) → **gap-free 30 m LST + uncertainty**.
- [ ] Receive LULC products → **majority vote + agreement** (#27); receive footprints → **cross-fill** (#26).
- [ ] Receive met (ERA5/CPCB/IMD/Netatmo) → **cross-QC** (#13⇄#10–12, #28) → clean atmospheric drivers.
- [ ] Feed fused LST (target) + drivers (LULC, morphology, vegetation, atmosphere, indices) to the physics-informed
      ML model.
- [ ] **Validate** with spatial CV (#31) + design-based accuracy (#32) + per-LCZ/LULC stratified metrics +
      residual maps + **physics sanity checks** (SEB, signs).
- [ ] **Attribute** with SHAP (#33) + variance-partitioning (#34) + GWR maps (#35); require ≥2-method agreement;
      produce ranked table + coefficient maps.
- [ ] **Propagate uncertainty** (MC, #30) to every map and to cooling-scenario ΔT.

---

## 7. References (verified URLs, June 2026)

**Spatial / temporal cross-validation**
- Roberts et al. (2017) *Ecography* — CV for spatial/temporal/hierarchical structure: https://www.wsl.ch/lud/biodiversity_events/papers/Roberts_et_al-2017-Ecography.pdf
- Spatial cross-validation for GeoAI (GeoAI Handbook chapter): https://www.acsu.buffalo.edu/~yhu42/papers/2023_GeoAIHandbook_SpatialCV.pdf
- Spatial vs. random CV in ML (MDPI *Water* 2023) — optimism quantified: https://www.mdpi.com/2073-4441/15/12/2278
- Spatial+ CV method (ScienceDirect 2023): https://www.sciencedirect.com/science/article/pii/S1569843223001887
- Wadoux et al. (2021) — "Spatial cross-validation is not the right way to evaluate map accuracy": https://www.sciencedirect.com/science/article/abs/pii/S0304380021002489
- Spatial LOO CV (researchgate): https://www.researchgate.net/publication/261331249_Spatial_leave-one-out_cross-validation_for_variable_selection_in_the_presence_of_spatial_autocorrelation
- **Verde `BlockKFold`** (official docs): https://www.fatiando.org/verde/latest/api/generated/verde.BlockKFold.html and gallery: https://www.fatiando.org/verde/latest/gallery/blockkfold.html
- Verde project (GitHub): https://github.com/fatiando/verde

**Metrics**
- `hydroeval` (NSE/KGE) manual: https://cran.r-universe.dev/hydroeval/doc/manual.html
- Limitations of Lin's CCC for model accuracy (ScienceDirect 2024): https://www.sciencedirect.com/science/article/pii/S1574954124003625
- "Friends don't let friends use NSE/KGE…" (ScienceDirect 2025) — panel-of-metrics argument: https://www.sciencedirect.com/science/article/abs/pii/S1364815225003494

**Ground truth & cross-sensor LST validation**
- Continental-scale ECOSTRESS LST validation + cross-satellite comparison, Europe/Africa (ScienceDirect 2022): https://www.sciencedirect.com/science/article/abs/pii/S0034425722004023
- LST product validation MODIS/ECOSTRESS/Landsat/GOES-R/VIIRS/Sentinel-3 vs in-situ, U.S. Corn Belt (ADS/AGU): https://ui.adsabs.harvard.edu/abs/2020AGUFMB040...08L/abstract
- Four new LST products in the Corn Belt (IEEE Xplore): https://ieeexplore.ieee.org/document/9546663/
- NASA/GSFC LST&E product intercomparison references: https://lpvs.gsfc.nasa.gov/LSTE/LSTE_references.html
- Crowdsourced T_air vs satellite UHI — Venter et al., *Sci. Adv.* 2021: https://www.science.org/doi/10.1126/sciadv.abb9569 (PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC8153720/)
- Crowdsourcing citizen weather stations for urban climate (Meier et al.): https://www.sciencedirect.com/science/article/abs/pii/S2212095517300068
- CrowdQC+ quality control (researchgate): https://www.researchgate.net/publication/355981690_CrowdQC-A_Quality-Control_for_Crowdsourced_Air-Temperature_Observations_Enabling_World-Wide_Urban_Climate_Applications
- Netatmo QC over the UK (Coney et al. 2022): https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/met.2075

**Driver attribution (explainable ML + variance partitioning + GWR)**
- Explainable ML for UHI, NYC (MDPI *Buildings* 2026): https://www.mdpi.com/2075-5309/16/1/186
- Explainable AI for UHI climate-adaptation policymaking (MDPI *Land* 2026): https://www.mdpi.com/2073-445x/15/1/62
- Urban morphology → LST via explainable ML (ScienceDirect 2024): https://www.sciencedirect.com/science/article/abs/pii/S2210670724008680
- ALE — Interpretable ML Book (Molnar): https://christophm.github.io/interpretable-ml-book/ale.html
- ALE method (Alibi docs): https://docs.seldon.io/projects/alibi/en/latest/methods/ALE.html
- "Measuring Variable Importance via Accumulated Local Effects" (arXiv 2025): https://arxiv.org/pdf/2512.21124
- `iml` (PDP/ALE/permutation/interaction) vignette: https://cran.r-project.org/web/packages/iml/vignettes/intro.html
- GWR + block morphology & UHI (ScienceDirect): https://www.sciencedirect.com/science/article/abs/pii/S2210670721007046
- GWR of SUHI underlying factors (MDPI *RS* 2018): https://www.mdpi.com/2072-4292/10/9/1428
- MGWR scale-dependent UHI relationships (ScienceDirect): https://www.sciencedirect.com/science/article/abs/pii/S1364815210002008

**Data fusion & gap-filling**
- Spatiotemporal fusion + LST reconstruction review (arXiv 1909.09316): https://arxiv.org/pdf/1909.09316
- ESTARFM fusion framework for cloud-prone heterogeneous landscapes (MDPI *RS* 2016): https://www.mdpi.com/2072-4292/8/5/425
- Improved ESTARFM with surface-heterogeneity (MDPI *RS* 2020): https://www.mdpi.com/2072-4292/12/21/3673
- Gap-fill of LST/reflectance in Landsat ARD (MDPI *RS* 2020): https://www.mdpi.com/2072-4292/12/7/1192
- MODIS+Landsat LST fusion (PMC): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4346394/
- Gap-filled Landsat-8 LST for urban climate (PMC): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7570728/
- Seamless MODIS LST via multi-source fusion + multi-stage optimization (MDPI *RS* 2025): https://www.mdpi.com/2072-4292/17/19/3374
- Comparison of gap-filling methods for Landsat-like all-weather LST (ScienceDirect 2025): https://www.sciencedirect.com/science/article/abs/pii/S0924271625001650

**Triple collocation**
- Extended TC — McColl et al. (2014) *GRL*: https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2014GL061322
- TC summary diagram — Siu et al. (2024) *Front. Remote Sens.*: https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2024.1395442/full
- TC error estimation & data fusion of gridded precipitation (ScienceDirect): https://www.sciencedirect.com/science/article/abs/pii/S0022169421013573

**Bayesian assimilation / optimal interpolation / kriging / GP**
- Bayesian tutorial for data assimilation (links OI/kriging/Kalman) — ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/S016727890600354X
- Bayesian complementary kernelized learning for spatiotemporal data (arXiv): https://arxiv.org/pdf/2208.09978
- KrigR — reconciling high-resolution climate datasets (arXiv): https://arxiv.org/pdf/2108.03957

**Ensemble agreement, majority voting, Monte Carlo uncertainty**
- Monte Carlo uncertainty: TOA reflectance → plant-functional-type distributions (ScienceDirect 2025): https://www.sciencedirect.com/science/article/abs/pii/S0034425725002792
- Uncertainty-aware Bayesian ML for land-cover classification (arXiv 2025): https://arxiv.org/html/2503.21510v1
- Land-cover ensembles & majority voting (USDA FS / RSE): https://www.fs.usda.gov/rm/pubs_journals/2018/rmrs_2018_healey_s001.pdf

---

*Prepared by Research Agent R9 (Validation / Attribution / Fusion). Methods cross-checked against the sources
above (June 2026). Items labelled `from-knowledge (verify)` should be confirmed before final report inclusion.*
