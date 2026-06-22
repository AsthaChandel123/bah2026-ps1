# R5 — Physics-Informed ML: The Modeling Core (LST ↔ Drivers)

**Project:** ISRO Bharatiya Antariksh Hackathon 2026, PS-1 — Physics-informed geospatial AI/ML for urban heat: (1) map hotspots, (2) quantify drivers, (3) model LST↔drivers with **physics-informed ML**, (4) optimize cooling with °C reduction.

**Scope of this document:** the scientific heart — the *physics* of urban heat (Surface Energy Balance) and the *machine-learning* methods to relate Land Surface Temperature (LST) to its drivers, made **physics-informed** so that intervention counterfactuals (↑NDVI, ↑albedo, ↑water) extrapolate trustworthily into °C cooling.

**Status legend:** ✅ web-verified (2024–2026 sources cited); 📚 from-knowledge (verify) = standard textbook/canonical result not re-fetched verbatim from a live source this session.

> **One-paragraph thesis.** LST is the radiometric skin temperature `T_s` that the satellite sees; it is set by the **Surface Energy Balance (SEB)**. ML that merely correlates LST with NDVI/albedo can interpolate but *cannot be trusted to extrapolate* to a greened/whitened city it never saw. The fix is to **bake the SEB into the learning** — as a loss penalty (PINN), as monotonic/sign constraints (constrained GBM/GP), as a residual-on-physics hybrid, or as an emulator of a physical model (SUEWS/SOLWEIG). Physics-informed ⇒ the model obeys `Q* = Q_H + Q_E + ΔQ_S (+ Q_F)` and the radiative law `L↑ = εσT_s⁴`, so a +0.1 NDVI perturbation produces a physically bounded ΔLST instead of an arbitrary one. That is the judging differentiator.

---

## 0. Table of contents
1. Physics of urban heat (SEB) with equations
2. SUHI vs UHI; day/night; LST vs air-T; LCZ reasoning; OHM/LUMPS/SUEWS/SOLWEIG
3. Catalog of ≥18 modeling methods (table: method | what | physics-informed variant)
4. Recommended physics-informed architecture
5. Driver-attribution methodology (quantitative)
6. Intervention / counterfactual cooling simulation
7. Validation & uncertainty
8. References (URLs)

---

## 1. Physics of urban heat (Surface Energy Balance) with equations

### 1.1 The master balance
For a horizontal urban facet (or a satellite pixel) over an averaging interval, the **Surface Energy Balance** is

```
Q*  =  Q_H  +  Q_E  +  ΔQ_S  +  Q_F          [W m⁻²]            (1)
```

| Term | Name | Physical meaning | Primary driver(s) |
|---|---|---|---|
| `Q*` | Net all-wave radiation | Radiative energy available at surface | albedo α, emissivity ε, `T_s`, cloud, SVF |
| `Q_H` | Turbulent **sensible** heat | Warms the air; convective | surface–air ΔT, wind, roughness |
| `Q_E` | Turbulent **latent** heat | Evapotranspiration (cooling) | vegetation, soil moisture, water, VPD |
| `ΔQ_S` | **Storage** heat flux | Heat into/out of fabric (mass) | thermal admittance μ, impervious mass, geometry |
| `Q_F` | **Anthropogenic** heat | Traffic, AC, industry, metabolism | population, energy use, traffic |

> Sign convention: fluxes *away from* the surface positive on the RHS; `ΔQ_S` positive = surface gaining heat (daytime charging). `Q_F` is often folded into the available energy: `Q* + Q_F = Q_H + Q_E + ΔQ_S`.

### 1.2 Net radiation — the radiative driver of LST
```
Q*  =  (1 − α) K↓  +  (L↓ − L↑)                                  (2)
L↑  =  ε σ T_s⁴  +  (1 − ε) L↓                                   (3)
```
- `K↓` incoming shortwave (W m⁻²); `α` broadband **albedo** (0–1, dimensionless). Cool/white roofs ⇒ ↑α ⇒ ↓absorbed shortwave `(1−α)K↓`.
- `L↓` incoming longwave from atmosphere; `L↑` outgoing longwave.
- `ε` broadband **emissivity** (≈0.90–0.99 urban); `σ = 5.670×10⁻⁸ W m⁻² K⁻⁴` Stefan–Boltzmann; `T_s` surface (skin) temperature in **K**.
- The `(1−ε)L↓` term in (3) is reflected longwave; many simplifications drop it. **This `εσT_s⁴` law is exactly what a thermal sensor inverts to retrieve LST** — and it is the cleanest physics to embed in ML (see PINN, §3).

**LST emerges by inverting (3):** `T_s = [ (L↑ − (1−ε)L↓) / (εσ) ]^{1/4}`. So *anything* that raises absorbed radiation or cuts turbulent/storage losses raises `T_s`.

### 1.3 Turbulent fluxes and the Bowen ratio
Bulk-aerodynamic / resistance forms:
```
Q_H = ρ c_p (T_s − T_a) / r_a                                    (4)
Q_E = (ρ c_p / γ) · (e_s(T_s) − e_a) / (r_a + r_s)              (5)
β   = Q_H / Q_E      (Bowen ratio)                              (6)
```
- `ρ` air density, `c_p` specific heat, `γ` psychrometric constant, `r_a` aerodynamic resistance, `r_s` surface (stomatal/soil) resistance, `e_s−e_a` vapour-pressure deficit.
- **Vegetation enters through `Q_E`:** more leaf area / soil moisture ⇒ lower `r_s` ⇒ larger `Q_E` ⇒ **evaporative cooling** ⇒ lower `T_s`. Impervious surfaces have `r_s → ∞` ⇒ `Q_E → 0` ⇒ energy diverted into `Q_H` + `ΔQ_S` ⇒ hotter. **High β = dry/hot city; low β = vegetated/cool.** β is the single most compact "why is it hot" number. ✅ (day=impervious, night=anthropogenic+storage dominate — verified, AMS JAMC 2012).

### 1.4 Storage heat flux — the impervious-mass driver (OHM)
The diurnal **hysteresis** between `ΔQ_S` and `Q*` is captured by the **Objective Hysteresis Model** (Grimmond et al. 1991; used inside SUEWS): ✅
```
ΔQ_S = a₁ Q*  +  a₂ (∂Q*/∂t)  +  a₃                              (7)
```
- `a₁` (dimensionless) = mean dependence of storage on net radiation (typical 0.2–0.7 for built fabric).
- `a₂` (h) = phase/hysteresis term — sign & magnitude of the lead/lag of `ΔQ_S` vs `Q*` (the loop width).
- `a₃` (W m⁻²) = intercept (negative ⇒ evening release).
- For a composite pixel, OHM weights coefficients by surface-cover fraction: `ΔQ_S = Σ_i f_i (a₁ᵢ Q* + a₂ᵢ ∂Q*/∂t + a₃ᵢ)`. ✅ (SUEWS docs).

**Physical control:** `a₁,a₂` rise with **thermal admittance** `μ = √(λ C) = √(λ ρ c)` (units J m⁻² K⁻¹ s^{−1/2}) and thermal inertia. Concrete/asphalt have high `λ,C` ⇒ store huge daytime heat, release at night ⇒ **the nighttime SUHI**. **AnOHM** (Sun et al. 2017) derives `a₁,a₂,a₃` *analytically* from the 1-D heat-diffusion + SEB solution as functions of `μ`, α, ε, β, and forcing amplitude — i.e. a closed-form physics map from material properties to OHM coefficients. 📚 (analytical coefficient forms; GMD 2017 verified to exist). DyOHM extends OHM with dynamic, material/meteorology-dependent coefficients. ✅

### 1.5 Canyon geometry, SVF and longwave trapping
The **Sky View Factor** `Ψ_sky ∈ [0,1]` (fraction of sky hemisphere visible from a point) governs longwave escape:
```
L↑_net,canyon ≈ Ψ_sky · ε σ T_s⁴ − (trapped fraction)           (8)
```
Deep canyons (low `Ψ_sky`, high aspect ratio `H/W`) **trap outgoing longwave** (re-absorbed by walls) ⇒ reduced nocturnal cooling ⇒ canopy-layer UHI. Geometry also adds wall **storage** (large `ΔQ_S`) and **reduces wind** (↑`r_a` ⇒ traps `Q_H`). The classic canyon UHI parameterization: `ΔT_{u-r,max} ∝ f(Ψ_sky)` (Oke). ✅ Three 3-D-geometry sub-effects verified: wall storage, radiation trapping, wind reduction (Sci. Rep. 2019; AMS 2012).

### 1.6 Compact "driver → SEB term → ΔLST sign" map (the physics ML must respect)
| Driver ↑ | SEB pathway | Expected ∂LST sign |
|---|---|---|
| Albedo α ↑ | ↓`(1−α)K↓` ⇒ ↓`Q*` | **−** (cooler) |
| NDVI / veg ↑ | ↑`Q_E` (↓β), slight ↑ε | **−** (cooler) |
| Water fraction ↑ | ↑`Q_E`, ↑heat capacity | **−** (cooler) |
| Impervious / NDBI ↑ | ↓`Q_E`, ↑`ΔQ_S`, ↑β | **+** (hotter) |
| Building height/Σ, ↓SVF | ↑trapping, ↑wall `ΔQ_S` | **+** (esp. night) |
| Anthropogenic `Q_F` ↑ | direct + term | **+** (esp. night) |
| Emissivity ε ↑ | ↑`L↑` emission | **−** (radiates better) |
| Wind / roughness ↑ | ↓`r_a` ⇒ ↑`Q_H` export | **−** |

These signs are the **monotonicity constraints** we hand to constrained GBMs / monotone GPs / PINNs in §3. *This single table is the bridge between physics and ML.*

---

## 2. UHI vs SUHI; day/night; LST vs air-T; LCZ; physical models to inform/validate ML

- **SUHI (Surface UHI)** = ΔLST(urban − rural), from satellite thermal IR (`T_s`). **UHI (canopy/atmospheric)** = Δair-temperature `T_a` at ~2 m, from stations. They differ in magnitude and *timing*: **SUHI peaks at midday** (impervious surfaces bake; driven by `Q*`, `ΔQ_S`); **canopy UHI peaks at night** (stored heat + low-SVF trapping + `Q_F` release). ✅ Our PS-1 maps **LST/SUHI** (GEE-friendly), but we must caption that LST ≠ air-T; bridge with an LST→T_a model (e.g. SPyCer-style physics-guided, or `T_a = f(LST, NDVI, geometry)`) if heat-exposure is needed. ✅
- **Day vs night drivers** (verified, AMS JAMC 2012; Sci. Rep. 2019): **day** → impervious fraction & solar absorption dominate; **night** → anthropogenic heat `Q_F` and storage release `ΔQ_S` (+ low SVF) dominate. ⇒ *Train/interpret models separately for day and night LST* (Aqua/Terra MODIS 1:30/13:30 + Landsat ~10:30) or include an `acquisition-time/solar-geometry` covariate.
- **Local Climate Zones (Stewart & Oke 2012):** 17 classes (10 built + 7 land-cover), each with characteristic **albedo, surface (thermal) admittance, SVF, aspect ratio, building/pervious/impervious fractions, anthropogenic heat**. ✅ Use LCZ as (a) a categorical driver/stratifier in ML, (b) a physically-meaningful prior on OHM/SEB parameters per zone, and (c) a fairness/segmentation layer for hotspot mapping.

### Physical models that *inform / validate* the ML (not run live in O(1) GEE)
- **SUEWS** (Surface Urban Energy & Water Balance Scheme): solves Eq. (1) with **NARP** net-radiation, **OHM** storage (Eq. 7), **LUMPS** turbulent split, and `Q_F` schemes. ✅ Use offline to **generate physically-consistent training/validation samples** and to **sanity-check** ML-predicted fluxes and ΔLST.
- **LUMPS** (Local-scale Urban Meteorological Parameterization Scheme): a 2-parameter (`α_L,β_L`) **slab** estimate of `Q_H,Q_E` from `Q*` and vegetation fraction (de Bruin–Holtslag form): `Q_H = ((1−α_L)+γ/s)/(1+γ/s)·(Q*−ΔQ_S) − β_L`. 📚 Cheap physics prior for the turbulent split.
- **SOLWEIG / SOLWEIG-GPU**: 3-D radiation model → **Mean Radiant Temperature `T_mrt`** and shadow/SVF maps from a DSM. ✅ We **emulate** SOLWEIG with a CNN/GBM surrogate (inputs: DSM-derived SVF, shadows, `K↓`, `T_a`) to get O(1) `T_mrt`/shade fields for thermal-comfort & intervention scoring. A 2025 multimodal-PINN already models `T_mrt`. ✅

---

## 3. Catalog of ≥18 modeling methods

> Inputs (shared driver stack `X`): LST target `y`; predictors = NDVI/EVI/LAI, albedo, NDBI/NDWI, impervious %, land-cover/LCZ, building height & density, SVF/aspect ratio, DEM/slope/aspect, distance-to-water, population/`Q_F` proxy (nightlights, traffic), meteorology (`K↓,T_a,RH,wind`), and coordinates `(x,y)`. All derivable in **GEE** (R1/R2 domain). "PI-variant" = how to make it physics-informed.

| # | Method | What it does (inputs→output) | Pros / Cons | **Physics-informed (PI) variant** |
|---|---|---|---|---|
| 1 | **Random Forest (RF)** | Bagged trees; `X→LST`; gives permutation importance | + robust, nonlinear, low-tune; − no extrapolation, blocky | Constrain features to physical signs via post-hoc ALE checks; **RF on SEB residual**; add SUEWS-flux features |
| 2 | **Extra Trees** | Extremely randomized splits | + fast, low variance; − less accurate per-tree | Same as RF; best LST R² in 8-city XAI study (R²=0.908) ✅ |
| 3 | **Gradient Boosting (GBM)** | Sequential residual fitting | + accurate; − slower, overfit risk | **Monotone GBM**: enforce ∂LST/∂NDVI≤0, ∂LST/∂α≤0, ∂LST/∂impervious≥0 (Table §1.6) |
| 4 | **XGBoost** | Regularized boosting, `monotone_constraints` | + SOTA tabular, fast; − tuning | Native `monotone_constraints` per feature = physics signs; best SUHII R²=0.879 ✅ |
| 5 | **LightGBM** | Leaf-wise histogram boosting | + very fast, big data; − overfit small data | `monotone_constraints`; **physics-constrained LGBM** for gapless LST embeds SEB (arXiv 2307.04817) ✅ |
| 6 | **CatBoost** | Ordered boosting, native categoricals (LCZ!) | + handles LCZ/land-cover natively; − memory | Monotone constraints + LCZ priors; used w/ XAI for LST mechanisms ✅ |
| 7 | **OLS / Ridge / Lasso / Elastic-Net** | Linear `LST = Xβ`; shrinkage | + interpretable β, fast O(1); − misses nonlinearity | Sign-constrained / non-negative least squares to enforce physics signs on β |
| 8 | **Geographically Weighted Regression (GWR)** | Local regression, β(x,y) maps | + spatial heterogeneity, coefficient maps; − one global bandwidth | Sign-constrained GWR; interpret β(s) maps as *local* SEB sensitivities |
| 9 | **Multiscale GWR (MGWR)** | Per-covariate bandwidth via back-fitting (min AICc) | + true multiscale, beats GWR/OLS; − compute heavy | Physically: each driver acts at its own scale (veg local, geometry broad) ✅ |
| 10 | **Spatial lag / error (SAR/SEM)** | Adds spatial autocorrelation term (Wy) | + corrects autocorrelation; − global | Spatial term ≈ unresolved advection/diffusion of heat |
| 11 | **Spatial Random Forest (RF-sp / RFsp)** | RF + coordinates/buffer-distance features | + captures spatial structure non-parametrically; − big feature space | Add geometry/SVF so "space" carries physics, not just XY |
| 12 | **CNN / U-Net** | Image→image; LST mapping, gap-fill, **downscaling** | + spatial context, sharp; − data-hungry, black-box | **Physics-constrained CNN**: SEB/energy-conservation penalty (arXiv 2307.04817 style) ✅ |
| 13 | **ConvLSTM** | Spatiotemporal; convolutions in LSTM gates; LST forecast | + space+time; − heavy, needs sequences | Add diurnal-cycle / OHM hysteresis loss; forecast respects energy storage ✅ |
| 14 | **Graph Neural Net (GNN)** | Urban graph (parcels/canyons as nodes) → LST | + irregular topology, relational; − graph design | **UrbanGraph (2025)**: encodes shading/convection as dynamic causal edges; −73.8% FLOPs ✅ |
| 15 | **Vision Transformer (ViT)** | Attention over patches; LST/heat-stress | + long-range context; − very data-hungry | PINNsFormer-style: attention + PDE-residual loss ✅ |
| 16 | **Super-resolution net (SRCNN/SRGAN/ESRGAN)** | Thermal sharpening 100 m→10–30 m | + photoreal detail; − can hallucinate | **Mass/energy-preserving** SR: enforce coarse-pixel average = input (conservation) |
| 17 | **Physics-Informed Neural Net (PINN)** | NN with **PDE/SEB residual in loss** | + physically valid, extrapolates, data-efficient; − training stiffness | *This is the flagship PI method* — see §3.1 |
| 18 | **Hybrid / residual modeling** | ML learns `LST − LST_physics` | + physics backbone + ML correction; best extrapolation | Backbone = NARP/AnOHM/SUEWS; ML fits residual only |
| 19 | **Monotone-constrained NN** | NN with sign-constrained weights/lattice | + guaranteed physics monotonicity; − architecture limits | Deep Lattice / certified-monotone nets for Table §1.6 signs |
| 20 | **Gaussian Process (GP) / Kriging** | Bayesian non-param; `LST~GP(m,k)` + uncertainty | + native uncertainty, smooth; − O(n³) | **Physics-informed kernel** (PhIK / constrained GP): kernel from heat-eqn covariance; monotonicity constraints ✅ |
| 21 | **Bayesian hierarchical spatial model** | Multilevel priors, partial pooling, posterior UQ | + rigorous UQ, LCZ-level pooling; − MCMC cost | Physics priors on coefficients; propagates ΔLST uncertainty ✅ |
| 22 | **RF / GBM downscaling (TsHARP, DisTrad, SRFD)** | Coarse LST + fine indices → fine LST | + simple, strong baseline; − scale-bias | **Residual-corrected** (mass-preserving) — see §3.2 ✅ |
| 23 | **Area-to-Point Regression Kriging (ATPRK / RFATPK)** | Regression trend + ATP-kriged residual | + **mass-preserving** downscaling, geostatistical; − stationarity assumptions | Kriging residual restores energy conservation at coarse scale ✅ |
| 24 | **Geographically Weighted Downscaling (GWR/MGWR-DS)** | Local sharpening with β(s) | + spatial sharpening; − bandwidth | MGWR-based LST downscaling validated ✅ |
| 25 | **SHAP** (attribution) | Shapley game-theoretic feature credit (global+local) | + per-pixel driver attribution; − cost on big data | Check SHAP slopes match physics signs (consistency test) ✅ |
| 26 | **ALE / PDP** (attribution) | Marginal effect of a driver on LST | + de-correlated (ALE); − assumes independence (PDP) | Effect curves must be monotone per Table §1.6 |
| 27 | **Causal forest / Double ML / meta-learners** | Heterogeneous treatment effect of greening on ΔLST | + de-confounded causal cooling; − strong assumptions | Confounders = SEB drivers; estimates *causal* °C cooling ✅ |
| 28 | **Surrogate / emulator** | ML mimics SUEWS/SOLWEIG at O(1) | + physics fidelity at ML speed; − bounded by training envelope | The emulator *is* physics-informed by construction (learns a physical model) ✅ |

(28 distinct modeling methods listed — comfortably exceeds the ≥18 requirement and contributes 20+ "modeling" methods toward the team's ≥30.)

### 3.1 PINN flagship — embedding the heat/energy balance in the loss ✅
Treat LST `T(x,y,t)` (or surface temperature) as the NN output `T_θ`. The 2-D thermal-diffusion (heat) PDE with source:
```
∂T/∂t = κ (∂²T/∂x² + ∂²T/∂y²) + S(x,y,t)/(ρc·d)                  (9)
```
`κ` thermal diffusivity (m² s⁻¹), `d` active layer depth, `S` net SEB source. Composite loss:
```
L = L_data + λ_pde·L_pde + λ_seb·L_seb + λ_bc·L_bc + λ_mono·L_mono   (10)

L_data = (1/N) Σ |T_θ(xᵢ) − LST_obsᵢ|²                              (data fidelity)
L_pde  = (1/M) Σ |∂T_θ/∂t − κ∇²T_θ − S/(ρc d)|²   at collocation pts (PDE residual via autodiff)
L_seb  = (1/M) Σ |Q*_θ − (Q_H+Q_E+ΔQ_S+Q_F)|²                       (SEB closure residual, Eq.1)
L_mono = Σ ReLU(∂T_θ/∂NDVI)² + ReLU(−∂T_θ/∂α)² + ReLU(−∂T_θ/∂imp)²  (sign constraints, §1.6)
```
- `L_seb` uses Eqs. (2)–(7): `Q*` from albedo/emissivity/`T_s`, `Q_E` from veg, `ΔQ_S` from OHM. The network is *forced* to honor radiative law (3) and balance (1). ✅ (PINN loss = data + PDE-residual at collocation points — verified; multimodal PINN for `T_mrt` 2025; heat-eqn PINNs standard).
- **Why it matters for PS-1:** because the SEB residual is in the loss, perturbing α or NDVI yields a ΔLST that *satisfies energy conservation* — the counterfactual is physical, not extrapolated nonsense.

### 3.2 Mass-preserving downscaling (so sharpening conserves energy) ✅
Generic scheme (TsHARP/DisTrad/RF + residual kriging = ATPRK):
```
1) Fit trend at coarse scale:   LST_coarse = f(NDVI, NDBI, albedo, …)        (RF/GWR/regression)
2) Predict at fine scale:       LST_fine_trend = f(fine indices)
3) Residual at coarse:          r_coarse = LST_coarse_obs − ⟨LST_fine_trend⟩  (block average)
4) Area-to-point krige r → r_fine ; add back:  LST_fine = LST_fine_trend + r_fine
   Constraint: ⟨LST_fine⟩_block ≡ LST_coarse_obs   (mass/energy preservation)             (11)
```
Step 4's constraint (11) is the physics: the fine field must average back to the observed coarse pixel (radiance conservation). ML methods beat TsHARP; RFATPK & SRFD are the strong performers. ✅

---

## 4. Recommended physics-informed architecture (for PS-1)

**Design principle:** *many methods that cross-verify and fill gaps* (per project priorities), all reconciled to physics, all O(1) at inference after offline training.

```
                ┌─────────────────────────── DRIVER STACK X (GEE, O(1) tiles) ───────────────────────────┐
                │ LST(Landsat/MODIS/ECOSTRESS) ε,α | NDVI/EVI/LAI | NDBI/NDWI/impervious | LCZ           │
                │ SVF/aspect/building-height (DSM) | DEM | dist-water | Q_F proxy(VIIRS nightlights)      │
                │ meteo K↓,T_a,RH,wind                                                                    │
                └───────────────────────────────────────────────────────────────────────────────────────┘
                                              │
        ┌─────────────────────────────────────┼───────────────────────────────────────────────┐
        ▼                                     ▼                                                 ▼
 (A) PHYSICS BACKBONE              (B) CONSTRAINED ML CORE                      (C) SPATIAL LAYER
 NARP Q*  + AnOHM ΔQ_S +           Ensemble: monotone-XGBoost                  MGWR (β(s) coefficient maps)
 LUMPS Q_H,Q_E  ⇒ LST_phys         + monotone-LightGBM + Extra-Trees           + spatial residual (kriging)
 (Eqs.1–7; SUEWS-calibrated)       (signs from §1.6 Table)                     → local heterogeneity & gap-fill
        │                                     │                                                 │
        └──────────────► residual r = LST_obs − LST_phys ──────► (B) learns residual r ◄────────┘
                                              │
                                              ▼
                         (D) PINN RECONCILER (Eqs. 9–10): refines the field with
                             PDE + SEB-closure + monotonicity loss; produces the
                             *physically-consistent* LST(drivers) response surface
                                              │
                ┌──────────────────────────────┼──────────────────────────────┐
                ▼                              ▼                               ▼
        (E) SHAP / ALE attribution   (F) Counterfactual engine        (G) GP / Bayesian UQ
            (driver ranking, maps)       (perturb X → ΔLST °C)            (per-pixel uncertainty band)
```

**Why this stack (cross-verification + gap-filling, the project mandate):**
- **(A) backbone + (B) residual = hybrid** (method #18): physics gives the trend & guarantees extrapolation; ML mops up the residual. Best of both.
- **(B) monotone ensemble** (methods #4–6,#19): three learners cross-check; monotone constraints *guarantee* the §1.6 physics signs so interventions never go the wrong way.
- **(C) MGWR** (#9): supplies *spatial* coefficient maps the ensembles lack, and fills gaps geostatistically.
- **(D) PINN** (#17): the reconciler that enforces SEB closure + heat PDE — *the physics-informed differentiator for judging*.
- **(E) SHAP/ALE** (#25–26): turns the model into quantified driver attribution (PS-1 objective 2).
- **(F)** counterfactuals (#27): turns it into °C-cooling optimization (PS-1 objective 4).
- **(G) GP/Bayesian** (#20–21): every °C estimate ships with an uncertainty band.

**Minimal viable (if time-boxed):** monotone-XGBoost (B) + SHAP (E) + MGWR (C) + a thin SEB-residual penalty ≈ 80% of the value with low risk; add PINN (D) as the headline novelty.

---

## 5. Driver-attribution methodology (quantitative — PS-1 objective 2)

Goal: **rank and quantify** how LULC, morphology, vegetation, and atmosphere drive LST, globally and per-pixel.

1. **Global importance:** permutation importance + **mean(|SHAP|)** per driver. *Benchmark from literature to expect/sanity-check:* NDVI/greenspace & impervious/urban-density consistently top-ranked; NYC study: NDVI mean|SHAP|≈1.2–1.4 °C ≫ dist-coast≈0.25, water≈0.10; Da Nang: urban-density & greenspace-density dominant. ✅
2. **Direction & shape:** **ALE** (preferred over PDP under correlated drivers) — verify each curve's monotone sign matches §1.6 (a *physics-consistency audit* of the ML).
3. **Local/spatial attribution:** **SHAP maps** (per-pixel `φ_NDVI(s)`) and **MGWR β(s) maps** — both give *where* each driver matters; agreement between them = cross-verification.
4. **Variance partitioning / commonality analysis:** decompose LST R² into unique vs shared contributions of {vegetation, morphology, LULC, atmosphere} blocks → "vegetation explains X%, geometry Y%…".
5. **Report card per hotspot:** for each hotspot polygon output the dominant SEB lever (high β? low SVF? low α? low NDVI?) → directly feeds intervention choice in §6.

> Deliverable artifact: a **driver-attribution table** (driver, mean|SHAP| in °C, ALE sign, MGWR median β, % variance) + SHAP/β coefficient maps.

---

## 6. Intervention / counterfactual cooling simulation (PS-1 objective 4)

**Core idea:** a trained `LST = F(drivers)` is a *differentiable response surface*; perturb drivers, predict ΔLST.
```
ΔLST(intervention) = F(X + ΔX) − F(X)                                (12)
```
Examples of `ΔX`: ↑NDVI (+0.1–0.3 via tree canopy), ↑albedo (asphalt 0.1 → cool roof/pavement 0.4), ↑water fraction, LULC reclass (impervious→park), ↓building density / ↑SVF.

**Keeping it physically valid (the crux — why physics-informed):**
- **Monotonicity guarantee:** because the ensemble/PINN obey §1.6 signs, ↑NDVI ⇒ ΔLST ≤ 0 *always* — no sign flips on extrapolation.
- **SEB closure:** PINN's `L_seb` makes the perturbed state re-balance energy; e.g. greening simultaneously ↑`Q_E`, slightly ↑ε, ↓`ΔQ_S` — captured coherently, not as an unconstrained regression jump.
- **Physical bounds / clipping:** clip predicted ΔLST to physically attainable ranges (e.g. evaporative cooling cannot exceed available `Q*`); flag any `ΔX` outside the training envelope (GP/Bayesian uncertainty in §7 widens there as a guardrail).
- **Causal (de-confounded) °C:** use **causal forest / double-ML** (#27) so the cooling estimate is the *causal* effect of greening, not confounded by "parks happen to be near water." ✅
- **Cross-check with physics emulator:** run the same `ΔX` through the SUEWS/SOLWEIG surrogate (A/method #28); agreement between ML-counterfactual and physics-emulator ΔLST = high confidence.

**Optimization for cooling (links to R-optimization domain):** maximize Σ ΔLST cooling subject to budget/area constraints — greedy/marginal-SHAP ranking of candidate cells, or multi-objective (cooling vs cost vs equity). Interpretable-ML + multi-objective optimization for greening cooling is established (vacant-land re-greening study). ✅

> Deliverable artifact: per-scenario **ΔLST °C maps** (e.g. "+20% canopy ⇒ −1.8 °C mean, −3.2 °C in hotspots") with uncertainty bands.

---

## 7. Validation & uncertainty

**Validation**
- **Spatial cross-validation** (spatial block / leave-one-region-out) — *not* random CV — to avoid spatial-autocorrelation leakage and honestly test extrapolation.
- **Temporal hold-out** (train years → test year/season; separate day/night) for forecasting/ConvLSTM.
- **Metrics:** R², RMSE (°C), MAE, bias; per-LCZ stratified metrics. *Literature anchors to beat/compare:* Extra-Trees LST R²=0.908 / RMSE=0.745 °C; XGBoost SUHII R²=0.879. ✅
- **Physics-consistency tests (the PI checks):** (i) SEB residual `|Q* − ΣQ|` distribution ≈ 0; (ii) every ALE/SHAP curve sign matches §1.6; (iii) downscaling mass-preservation (11) holds; (iv) ML fluxes vs SUEWS within tolerance.
- **Independent ground truth:** flux-tower `Q_H,Q_E` (eddy covariance), in-situ air-T/IR, ECOSTRESS high-res LST as cross-sensor check.

**Uncertainty quantification**
- **GP / Bayesian hierarchical** posterior → per-pixel σ(LST) and σ(ΔLST). ✅
- **Quantile regression / NGBoost** (natural gradient boosting gives predictive distributions — used for UHI intensity) ✅, or **conformal prediction** for distribution-free intervals.
- **Deep ensembles / MC-dropout** for the NN/PINN/CNN components.
- **Propagate input uncertainty** (LST retrieval ±1–2 K, emissivity error) through Eq. (3) into ΔLST.
- **Out-of-envelope flag:** widen/flag uncertainty when a counterfactual `ΔX` exits the training support — prevents over-claiming cooling.

---

## 8. References (URLs)

**Physics — SEB, OHM/AnOHM, SUEWS, LUMPS, SOLWEIG, LCZ**
- SUEWS development & evaluation (Ward et al., *Urban Climate* 2016): https://www.sciencedirect.com/science/article/pii/S2212095516300256 ✅
- SUEWS parameterisations & sub-models (docs, 2026): https://docs.suews.io/stable/parameterisations-and-sub-models.html ✅
- SUEWS OHM coefficients table (docs): https://suews.readthedocs.io/stable/input_files/SUEWS_SiteInfo/SUEWS_OHMCoefficients.html ✅
- AnOHM v1.0 — analytical OHM coefficients (Sun et al., *GMD* 2017): https://gmd.copernicus.org/articles/10/2875/2017/ ✅ (preprint PDF: https://gmd.copernicus.org/preprints/gmd-2016-300/gmd-2016-300.pdf)
- Quantitative analysis of UHI-intensity factors / day-night drivers (Li et al., *J. Appl. Meteor. Climatol.* 2012): https://journals.ametsoc.org/view/journals/apme/51/5/jamc-d-11-098.1.pdf ✅
- Biophysical mechanisms of UHI vs morphology (climate variation; *Sci. Rep.* 2019): https://www.nature.com/articles/s41598-019-55847-8 ✅
- Simple thermodynamic model of UHI (*Nonlin. Processes Geophys.* 2026): https://npg.copernicus.org/articles/33/17/2026/ ✅
- SOLWEIG (UMEP) model docs: https://umep-dev.github.io/solweig/ ✅ ; SOLWEIG-GPU: https://solweig-gpu.readthedocs.io/en/latest/ ✅
- Multimodal **PINN for Mean Radiant Temperature** (arXiv 2025): https://arxiv.org/pdf/2503.08482 ✅
- Local Climate Zones (Stewart & Oke 2012; review of LCZ in modelling, *Urban Sci.* 2023): https://www.mdpi.com/2673-4834/7/1/3 ✅

**Physics-informed / constrained ML**
- **Physics-constrained ML for gapless LST** (SEB-constrained LGBM; arXiv 2307.04817): https://arxiv.org/pdf/2307.04817 ✅
- Integrating scientific knowledge with ML (theory-guided DS; Karpatne et al., arXiv 2003.04919): https://arxiv.org/pdf/2003.04919 ✅
- **UrbanGraph** — physics-informed spatio-temporal GNN for urban microclimate (arXiv 2510.00457): https://arxiv.org/pdf/2510.00457 ✅
- PINNs for PDE problems — comprehensive review (*Artif. Intell. Rev.* 2025): https://link.springer.com/article/10.1007/s10462-025-11322-7 ✅
- PINNs for heat conduction w/ phase change (arXiv 2410.14216): https://arxiv.org/abs/2410.14216 ✅
- Physics-informed GP regression (Lehigh tech report): https://engineering.lehigh.edu/sites/engineering.lehigh.edu/files/_DEPARTMENTS/ise/pdf/tech-papers/19/19T-026.pdf ✅
- Physics-Information-Aided Kriging (PhIK; arXiv 1809.03461): https://arxiv.org/pdf/1809.03461 ✅
- GP with soft inequality & monotonicity constraints (*Frontiers Mech. Eng.* 2024): https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2024.1410190/full ✅
- Smart Urban Design with PINNs — green-infrastructure cooling from thermal data (ResearchGate 2025): https://www.researchgate.net/publication/396689873 ✅
- Smart Urban Cooling — physics-informed ML heat-island mitigation (SSRN 5954787, 2025): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5954787 ✅

**ML methods, attribution (SHAP/ALE), spatial regression, causal**
- Comprehensive **review of ML for UHI assessment** (*Renew. Sustain. Energy Rev.* 2026): https://www.sciencedirect.com/science/article/pii/S1364032126002029 ✅
- Interpretable ML for urban LST + SHAP (*Sensors* 2025, MDPI 25/4/1169): https://www.mdpi.com/1424-8220/25/4/1169 ✅ (PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11859931/)
- **Disentangling climatic vs surface-physical UHI drivers, Explainable AI, 8 US cities** (XGBoost/Extra-Trees, GEE; *Sustainability* 2026, 18/8/3694): https://doi.org/10.3390/su18083694 ✅
- Explaining & reducing UHI via ML — NYC (RF/XGBoost+SHAP; *Buildings* 2026, 16/1/186): https://www.mdpi.com/2075-5309/16/1/186 ✅
- Interpretable ML for urban-heat mitigation: multi-scale driver attribution/weighting (arXiv 2507.04802): https://arxiv.org/pdf/2507.04802 ✅
- NGBoost + DNN for SUHI intensity, multi-source RS (*Sustainability* 2025, 17/10/4287): https://www.mdpi.com/2071-1050/17/10/4287 ✅
- **MGWR** for LST/urban-morphology heterogeneity (*PLOS One* 2024, Jinan): https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0307711 ✅
- Spatiotemporal decoupling of LST drivers via ensemble learning (*Sustain. Cities Soc.* 2025): https://www.sciencedirect.com/science/article/pii/S2210670725007413 ✅
- **Causal inference / double-ML** for urban LST (compactness, interpretability; *Sensors* 2025, 25/17/5380): https://www.mdpi.com/1424-8220/25/17/5380 ✅
- Causal ML for LULC change effects (*Landscape Ecol.* 2025): https://link.springer.com/article/10.1007/s10980-025-02279-7 ✅
- Optimal cooling from urban vacant-land re-greening — interpretable ML + multi-objective optimization (*Sustain. Cities Soc.* 2026): https://www.sciencedirect.com/science/article/abs/pii/S1618866726000944 ✅

**Deep learning & downscaling**
- ConvLSTM for 100 m daily near-surface air-T (*Sci. Data* 2025): https://www.nature.com/articles/s41597-025-05032-6 ✅
- Urban LST prediction integrating LSTM + geospatial info, Kunming (*J. Clean. Prod.* 2026): https://www.sciencedirect.com/science/article/abs/pii/S0959652626003951 ✅
- ML-based **Area-to-Point Regression Kriging** LST downscaling over urban areas (*Remote Sens.* 2020, 12/7/1082): https://www.mdpi.com/2072-4292/12/7/1082 ✅
- **Spatial Random Forest** LST downscaling considering spatial features (*Remote Sens.* 2021, 13/18/3645): https://www.mdpi.com/2072-4292/13/18/3645 ✅
- LST sharpening — comprehensive review (*Eur. J. Remote Sens.* 2022): https://www.tandfonline.com/doi/full/10.1080/22797254.2022.2144764 ✅
- SPyCer — semi-supervised physics-guided air-T from satellite imagery (arXiv 2603.05219): https://arxiv.org/pdf/2603.05219 ✅

**Uncertainty / Bayesian**
- Hierarchical Bayesian spatial modeling for climate uncertainty (eScholarship): https://escholarship.org/uc/item/3kq0b51t ✅
- Data-integration framework, spatial interpolation of temperature w/ UQ (*PMC* 9838203): https://pmc.ncbi.nlm.nih.gov/articles/PMC9838203/ ✅

---

### Handoff notes to the team
- **To R1/R2 (data/GEE):** I need, per pixel & per scene, the §4 driver stack including **emissivity ε, albedo α, SVF, building height, `Q_F` proxy**, and **day vs night** LST tagging — these are load-bearing for SEB and for separating day/night attribution.
- **To optimization domain:** the §6 counterfactual engine (Eq. 12) + per-cell marginal-SHAP cooling is your objective-function input; clip to physical bounds and respect uncertainty flags.
- **Judging hook:** lead with §3.1 (PINN SEB-residual loss) + §1.6 monotonicity guarantees — that is the concrete, equation-level evidence that our ML is *physics-informed*, not just ML on geodata.
