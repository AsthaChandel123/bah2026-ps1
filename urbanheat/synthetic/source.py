"""urbanheat.synthetic.source — the offline synthetic data backend.

Generates a **physically-coherent synthetic Indian city** on the config grid so
the ENTIRE pipeline (indices -> hotspots -> ML -> attribution -> intervention sim
-> optimization -> maps/report) runs end-to-end with **no GEE credentials and no
network** — the demo backbone and the unit-test fixture (ARCHITECTURE §4, §11.2).

What "physically coherent" means here
-------------------------------------
A latent urban structure is built from spatially-autocorrelated random fields
(smoothed white noise + an urban-core/sub-centre gradient + explicit water and
park masks). Every canonical driver layer is then derived from that one structure
so the drivers are mutually consistent (dense built-up has high
``impervious_frac``/``building_height``/``plan_area_frac``, low ``svf``/``ndvi``,
high ``population``/``nightlights``/``anthro_heat``; parks are green and cool;
water bodies are wet and strongly cooling; suburbs/rural fringe sit in between).

``LST`` is then synthesized from those drivers through a **surface-energy-balance
-consistent** response with the correct signs (R5 §1.6, exported by
:func:`urbanheat.physics.energy_balance.expected_lst_gradient_signs`): LST rises
with impervious fraction, anthropogenic heat, low albedo and low sky-view-factor
(night trapping), and falls with NDVI / tree fraction / vegetation cover (shade +
evapotranspiration), water fraction (evaporative) and higher albedo. The result
shows clear hotspots over dense built-up and cool islands over parks/water, so
downstream attribution and intervention counterfactuals learn sensible,
correctly-signed relationships.

Design rules (build contract, §11)
----------------------------------
* Top-level imports are **numpy + scipy only**. :mod:`urbanheat.constants`,
  :mod:`urbanheat.datamodel` and :mod:`urbanheat.physics.energy_balance` are
  imported lazily inside the functions, so importing this module is cheap and
  there is no import cycle.
* Seeded by ``config.seed`` -> fully reproducible.
* Returns a validated :class:`~urbanheat.datamodel.FeatureStack` whose ``crs`` is
  ``config.target_crs`` and whose ``transform``/``bounds`` are a real metric
  (UTM) grid derived from the config bbox + ``resolution_m``.

Units (documented per layer in :func:`make_synthetic_fields`): fractions and
indices are dimensionless (0-1 or -1..1); temperatures degC; fluxes W/m^2;
lengths/heights m; ``building_volume`` m^3/cell; ``population`` persons/cell;
``soil_moisture`` m^3/m^3; ``rel_humidity`` %; ``wind_speed`` m/s; ``pressure``
kPa; ``aod`` dimensionless; ``no2`` mol/m^2; ``nightlights`` nW/cm^2/sr.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import gaussian_filter

from urbanheat.datamodel import DataSource

if TYPE_CHECKING:  # hints only
    from urbanheat.config import Config
    from urbanheat.datamodel import FeatureStack


# ===========================================================================
# Low-level field generators (spatial autocorrelation)
# ===========================================================================
def _smooth_noise(
    shape: "tuple[int, int]",
    rng: "np.random.Generator",
    scale: float,
    octaves: int = 3,
) -> "np.ndarray":
    """Spatially-autocorrelated field in [0, 1] via multi-octave smoothed noise.

    White noise is smoothed with a Gaussian kernel (``scipy.ndimage``) at a base
    ``scale`` (pixels) and a couple of finer octaves are added for texture, then
    the result is min-max normalized. Larger ``scale`` => broader, smoother urban
    structure. Reproducible given ``rng``.
    """
    h, w = shape
    field = np.zeros(shape, dtype=np.float64)
    amp = 1.0
    total = 0.0
    s = float(scale)
    for _ in range(max(1, octaves)):
        white = rng.standard_normal(shape)
        field += amp * gaussian_filter(white, sigma=max(0.5, s), mode="reflect")
        total += amp
        amp *= 0.5
        s *= 0.5
    field /= total
    lo, hi = float(field.min()), float(field.max())
    if hi - lo < 1e-12:
        return np.zeros(shape, dtype=np.float64)
    return (field - lo) / (hi - lo)


def _radial_gradient(
    shape: "tuple[int, int]",
    center: "tuple[float, float]",
    sigma_frac: float,
) -> "np.ndarray":
    """Gaussian bump in [0, 1] centred at ``center`` (row_frac, col_frac).

    Models a monocentric urban core (or a sub-centre); ``sigma_frac`` is the
    Gaussian width as a fraction of the grid diagonal.
    """
    h, w = shape
    rr, cc = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = center[0] * h, center[1] * w
    diag = float(np.hypot(h, w))
    sig = max(1.0, sigma_frac * diag)
    d2 = (rr - cy) ** 2 + (cc - cx) ** 2
    return np.exp(-d2 / (2.0 * sig ** 2))


# ===========================================================================
# The synthetic field stack (the physics-bearing core)
# ===========================================================================
def make_synthetic_fields(
    shape: "tuple[int, int]",
    seed: int = 0,
) -> "dict[str, np.ndarray]":
    """Return a dict of canonical-name -> 2-D ``float32`` driver fields.

    Pure helper (no geo-referencing): builds a spatially-coherent synthetic city
    on ``shape`` and populates **all** canonical driver layers (vegetation /
    spectral indices, land cover & fractions, urban morphology / 3-D form,
    meteorology / atmosphere, anthropogenic) plus ``EMISSIVITY``. ``LST`` and its
    diurnal/uncertainty companions are added by :func:`synthesize_lst` (called by
    :meth:`SyntheticDataSource.load`); a self-consistent ``LST`` is also included
    here for convenience so the dict round-trips through tests.

    Reproducible for a given ``seed``. See the module docstring for units.
    """
    from urbanheat.datamodel import (
        AIR_TEMP, ALBEDO, ANTHRO_HEAT, AOD, ASPECT_RATIO, BUILDING_HEIGHT,
        BUILDING_VOLUME, DEWPOINT, DISPLACEMENT_HEIGHT, ELEVATION, EMISSIVITY, ET,
        EVI, FRONTAL_AREA_INDEX, FVC, GREEN_FRAC, IMPERVIOUS_FRAC, LAI, LCZ,
        LONGWAVE_DOWN, LULC, MNDWI, NDBAI, NDBI, NDVI, NDWI, NET_RADIATION,
        NIGHTLIGHTS, NO2, PBL_HEIGHT, PLAN_AREA_FRAC, POPULATION, PRESSURE,
        REL_HUMIDITY, ROUGHNESS_LENGTH, SAVI, SLOPE, SOIL_MOISTURE,
        SOLAR_RADIATION, SVF, TREE_FRAC, UI, WATER_FRAC, WIND_SPEED,
    )

    h, w = int(shape[0]), int(shape[1])
    rng = np.random.default_rng(int(seed))
    f: dict[str, np.ndarray] = {}

    # ---- 1. latent urban structure --------------------------------------
    # urban core + a few reproducible sub-centres + autocorrelated background.
    core = _radial_gradient((h, w), (0.5, 0.48), sigma_frac=0.16)
    n_sub = 3
    sub_centres = [(0.30, 0.30), (0.68, 0.66), (0.40, 0.75)][:n_sub]
    sub = np.zeros((h, w), dtype=np.float64)
    for (ry, rx) in sub_centres:
        sub = np.maximum(sub, 0.7 * _radial_gradient((h, w), (ry, rx), 0.08))
    background = _smooth_noise((h, w), rng, scale=max(2.0, min(h, w) / 12.0), octaves=4)
    # urbanness in [0,1]: structured cores dominate, background adds realism.
    urban = np.clip(0.62 * np.maximum(core, sub) + 0.38 * background, 0.0, 1.0)
    urban = (urban - urban.min()) / (urban.max() - urban.min() + 1e-12)

    # ---- 2. water bodies (a meandering river + a lake) ------------------
    yy = np.linspace(0.0, 1.0, h)[:, None] * np.ones((1, w))
    xx = np.ones((h, 1)) * np.linspace(0.0, 1.0, w)[None, :]
    river_axis = 0.30 + 0.12 * np.sin(2.0 * np.pi * 1.3 * yy + 0.6) \
        + 0.05 * np.sin(2.0 * np.pi * 3.1 * yy)
    river_halfwidth = 0.018
    river = np.exp(-((xx - river_axis) ** 2) / (2.0 * river_halfwidth ** 2))
    lake = _radial_gradient((h, w), (0.74, 0.30), sigma_frac=0.05)
    lake = (lake > 0.6).astype(np.float64) * lake
    water_frac = np.clip(np.maximum(0.9 * (river > 0.5) * river, lake), 0.0, 1.0)
    water_mask = water_frac > 0.25
    # carve water out of the urban field (no buildings on water).
    urban = np.where(water_mask, urban * 0.05, urban)

    # ---- 3. parks / green patches ---------------------------------------
    park_noise = _smooth_noise((h, w), rng, scale=max(2.0, min(h, w) / 18.0), octaves=2)
    big_park = _radial_gradient((h, w), (0.55, 0.40), sigma_frac=0.05)
    parks = np.clip((park_noise > 0.78).astype(np.float64) * park_noise
                    + (big_park > 0.7) * big_park, 0.0, 1.0)
    parks = np.where(water_mask, 0.0, parks)
    park_mask = parks > 0.4

    # ---- 4. land-cover fractions (sum-to-one composition) ---------------
    # impervious grows with urbanness; suppressed in parks; zero on water.
    imperv = np.clip(0.92 * urban ** 1.15, 0.0, 0.95)
    imperv = np.where(park_mask, imperv * 0.12, imperv)
    imperv = np.where(water_mask, 0.0, imperv)

    # tree canopy: high in parks, moderate in leafy suburbs, low downtown.
    tree = np.clip(0.65 * parks + 0.30 * (1.0 - urban) * background, 0.0, 0.9)
    tree = np.where(water_mask, 0.0, tree)
    # other (non-tree) vegetation / grass.
    grass = np.clip(0.55 * parks + 0.45 * (1.0 - urban) * (1.0 - background), 0.0, 0.9)
    grass = np.where(water_mask, 0.0, grass)
    green = np.clip(tree + 0.6 * grass, 0.0, 1.0)

    # normalize so impervious + green + water <= 1 (remainder = bare soil).
    total = imperv + green + water_frac
    over = np.clip(total - 1.0, 0.0, None)
    # shrink impervious then green proportionally where they overflow.
    green = np.clip(green - over * (green / np.clip(imperv + green, 1e-6, None)), 0.0, 1.0)
    imperv = np.clip(imperv - over * (imperv / np.clip(imperv + green + over, 1e-6, None)),
                     0.0, 1.0)
    imperv = np.clip(imperv, 0.0, 1.0 - water_frac)
    bare = np.clip(1.0 - imperv - green - water_frac, 0.0, 1.0)
    tree = np.clip(tree, 0.0, green)  # keep tree <= green

    f[IMPERVIOUS_FRAC] = imperv
    f[GREEN_FRAC] = green
    f[WATER_FRAC] = water_frac
    f[TREE_FRAC] = tree

    # ---- 5. spectral / vegetation indices -------------------------------
    veg_strength = np.clip(green - 0.5 * imperv, 0.0, 1.0)
    ndvi = np.clip(-0.05 + 0.85 * veg_strength - 0.25 * imperv, -0.2, 0.9)
    ndvi = np.where(water_mask, -0.25 - 0.1 * water_frac, ndvi)  # water: negative NDVI
    f[NDVI] = ndvi
    # EVI/SAVI track NDVI with characteristic scaling (EVI compresses high end).
    f[EVI] = np.clip(0.9 * ndvi * (1.0 - 0.2 * np.clip(ndvi, 0, 1)), -0.2, 0.8)
    f[SAVI] = np.clip(1.05 * ndvi, -0.25, 0.85)
    # FVC from NDVI (Carlson-Ripley square law between soil/veg thresholds).
    ndvi_soil, ndvi_veg = 0.2, 0.5
    fvc = np.clip((ndvi - ndvi_soil) / (ndvi_veg - ndvi_soil), 0.0, 1.0) ** 2
    f[FVC] = fvc
    f[LAI] = np.clip(5.0 * fvc + 0.4 * tree, 0.0, 6.0)            # m2/m2
    # water indices: high over water, negative over built-up.
    f[NDWI] = np.clip(-0.3 + 1.4 * water_frac - 0.2 * imperv, -0.4, 0.9)
    f[MNDWI] = np.clip(-0.25 + 1.4 * water_frac - 0.35 * imperv, -0.5, 0.9)
    # built-up / bare indices: high over impervious, negative over vegetation.
    f[NDBI] = np.clip(-0.25 + 0.85 * imperv - 0.45 * green, -0.5, 0.7)
    f[NDBAI] = np.clip(-0.2 + 0.6 * bare + 0.4 * imperv - 0.4 * green, -0.5, 0.7)
    f[UI] = np.clip(f[NDBI] - ndvi, -1.0, 1.0)                    # urban index
    # evapotranspiration (mm/period): scales with vegetation + soil moisture.
    sm = np.clip(0.10 + 0.40 * green + 0.45 * water_frac
                 - 0.08 * imperv + 0.05 * background, 0.03, 0.6)   # m3/m3
    f[SOIL_MOISTURE] = sm
    f[ET] = np.clip(40.0 * fvc * np.clip(sm / 0.4, 0, 1)
                    + 90.0 * water_frac, 0.0, 130.0)              # mm/period

    # ---- 6. surface radiative properties --------------------------------
    # albedo: vegetation ~0.18, impervious ~0.13 (dark asphalt/roofs), bare ~0.30,
    # water ~0.07. Add mild texture. Lower albedo over dense built-up => warming.
    albedo = (0.13 * imperv + 0.18 * green + 0.30 * bare + 0.07 * water_frac)
    albedo = np.clip(albedo + 0.02 * (background - 0.5), 0.05, 0.45)
    f[ALBEDO] = albedo
    # emissivity (NDVI-threshold, Sobrino): veg ~0.985, soil ~0.96, water ~0.99.
    emis = np.where(ndvi > ndvi_veg, 0.985,
                    np.where(ndvi < ndvi_soil, 0.960,
                             0.960 + 0.025 * fvc))
    emis = np.where(water_mask, 0.992, emis)
    f[EMISSIVITY] = np.clip(emis, 0.95, 0.995)

    # ---- 7. urban morphology / 3-D form ---------------------------------
    # building height (m): tall in cores/sub-centres, low at fringe, 0 on water.
    height = np.clip(45.0 * urban ** 1.4 + 6.0 * urban, 0.0, 70.0)
    height = np.where(water_mask | park_mask, 0.0, height)
    f[BUILDING_HEIGHT] = height
    plan_area = np.clip(0.85 * imperv, 0.0, 0.85)                 # lambda_P, 0-1
    f[PLAN_AREA_FRAC] = plan_area
    f[BUILDING_VOLUME] = (height * plan_area * 1.0e4).astype(np.float64)  # m3/cell (100m cell)
    # frontal area index lambda_F ~ plan_area * height / building width proxy.
    frontal = np.clip(plan_area * height / 30.0, 0.0, 1.2)
    f[FRONTAL_AREA_INDEX] = frontal
    # aspect ratio H/W rises with density & height.
    f[ASPECT_RATIO] = np.clip(0.1 + 2.6 * plan_area * np.clip(height / 30.0, 0, 2), 0.0, 3.5)
    # sky view factor: LOW in dense high-rise (trapping), ~1 over open/water.
    svf = np.clip(1.0 - 0.78 * plan_area - 0.18 * np.clip(height / 60.0, 0, 1), 0.18, 1.0)
    svf = np.where(water_mask, 0.99, svf)
    svf = np.where(park_mask, np.clip(svf, 0.7, 1.0), svf)
    f[SVF] = svf
    # roughness length z0 (m) & displacement height zd (m), Macdonald-flavoured.
    f[ROUGHNESS_LENGTH] = np.clip(0.03 + 0.9 * frontal * np.clip(height / 30.0, 0, 2),
                                  0.01, 2.5)
    f[DISPLACEMENT_HEIGHT] = np.clip(0.6 * height * plan_area, 0.0, 45.0)
    # terrain: gentle regional DEM + local relief; elevation in m.
    base_elev = 180.0 + 60.0 * _smooth_noise((h, w), rng, scale=max(3.0, min(h, w) / 6.0),
                                              octaves=2)
    f[ELEVATION] = (base_elev - 25.0 * water_frac).astype(np.float64)    # m
    gy, gx = np.gradient(f[ELEVATION])
    f[SLOPE] = np.clip(np.degrees(np.arctan(np.hypot(gy, gx) / 100.0)), 0.0, 25.0)  # deg

    # ---- 8. land-cover class codes (consistent with fractions) ----------
    # LULC codes (compact synthetic scheme): 1 water, 2 tree, 3 grass/veg,
    # 4 bare, 5 built. LCZ uses the constants.LCZ_TABLE class ids.
    lulc = np.full((h, w), 5, dtype=np.float64)        # default built
    lulc = np.where(bare > 0.4, 4, lulc)
    lulc = np.where(green > 0.4, 3, lulc)
    lulc = np.where(tree > 0.4, 2, lulc)
    lulc = np.where(water_mask, 1, lulc)
    f[LULC] = lulc
    # LCZ: map urbanness/height -> built LCZ 1..6, vegetation/water -> 11..17.
    lcz = np.full((h, w), 6, dtype=np.float64)         # open low-rise default
    lcz = np.where((imperv > 0.4) & (height > 8), 3, lcz)    # compact low-rise
    lcz = np.where((imperv > 0.55) & (height > 17), 2, lcz)  # compact midrise
    lcz = np.where((imperv > 0.6) & (height > 30), 1, lcz)   # compact high-rise
    lcz = np.where((imperv > 0.3) & (imperv <= 0.4) & (height > 25), 4, lcz)  # open high-rise
    lcz = np.where(tree > 0.5, 11, lcz)                # dense trees (LCZ A)
    lcz = np.where((green > 0.4) & (tree <= 0.5), 14, lcz)   # low plants (LCZ D = rural ref)
    lcz = np.where(bare > 0.5, 16, lcz)                # bare soil
    lcz = np.where(water_mask, 17, lcz)                # water (LCZ G)
    f[LCZ] = lcz

    # ---- 9. anthropogenic / socioeconomic -------------------------------
    # population density (persons/cell): peaks in dense residential, low downtown
    # core-commercial dip is realistic but keep monotone-ish with urbanness here.
    pop = np.clip(3500.0 * urban ** 1.3 * (0.6 + 0.4 * (1.0 - plan_area)), 0.0, 4000.0)
    pop = np.where(water_mask, 0.0, pop)
    pop = np.where(park_mask, pop * 0.05, pop)
    f[POPULATION] = pop
    # nightlights (nW/cm2/sr): bright commercial cores + roads.
    ntl = np.clip(55.0 * urban ** 1.1 + 8.0 * imperv, 0.0, 70.0)
    ntl = np.where(water_mask, 0.5, ntl)
    f[NIGHTLIGHTS] = ntl
    # anthropogenic heat QF (W/m2): traffic/AC/industry; from pop + nightlights +
    # built volume; strongest in dense built-up. (Day baseline; night larger.)
    qf = (0.010 * pop + 0.55 * ntl + 4.0e-5 * f[BUILDING_VOLUME] / 1.0e2)
    qf = np.clip(qf, 0.0, 90.0)
    f[ANTHRO_HEAT] = qf
    # tropospheric NO2 (mol/m2): combustion proxy, tracks QF + traffic cores.
    f[NO2] = np.clip(2.0e-5 + 1.3e-4 * (qf / 90.0) + 4.0e-5 * imperv, 0.0, 3.0e-4)

    # ---- 10. meteorology / atmosphere (downscaled to grid) --------------
    # smooth regional fields + an urban warm/dry anomaly (the canopy-layer UHI
    # signature on AIR_TEMP, which differs from the surface LST but co-varies).
    regional = _smooth_noise((h, w), rng, scale=max(4.0, min(h, w) / 5.0), octaves=2)
    air_t = 31.0 + 3.0 * (regional - 0.5) + 1.8 * urban - 0.8 * green \
        - 0.0045 * (f[ELEVATION] - 180.0)                       # degC
    f[AIR_TEMP] = air_t
    # dewpoint: wetter over vegetation/water, drier over built-up.
    dewp = 18.0 + 4.0 * (regional - 0.5) + 3.0 * water_frac + 1.5 * green - 1.5 * imperv
    f[DEWPOINT] = np.clip(dewp, 8.0, 26.0)                      # degC
    # relative humidity from T & Td (Magnus); clamp to plausible pre-monsoon band.
    es_t = 6.112 * np.exp(17.62 * air_t / (243.12 + air_t))
    es_td = 6.112 * np.exp(17.62 * f[DEWPOINT] / (243.12 + f[DEWPOINT]))
    f[REL_HUMIDITY] = np.clip(100.0 * es_td / es_t, 10.0, 95.0)  # %
    # wind: faster over open/water/rural, slower in rough dense urban canyons.
    wind = np.clip(4.5 - 2.6 * plan_area - 1.0 * f[ROUGHNESS_LENGTH]
                   + 0.8 * (regional - 0.5), 0.4, 7.0)
    wind = np.where(water_mask, np.clip(wind + 1.2, 0.8, 8.0), wind)
    f[WIND_SPEED] = wind                                        # m/s
    f[PRESSURE] = np.clip(101.3 - f[ELEVATION] * 0.0115 / 1.0, 95.0, 102.0)  # kPa
    # aerosol optical depth: hazier over dense/industrial cores.
    aod = np.clip(0.25 + 0.45 * urban + 0.1 * (regional - 0.5), 0.05, 0.9)
    f[AOD] = aod
    # incoming shortwave K_down (W/m2): high pre-monsoon insolation, AOD-dimmed.
    k_down = np.clip(880.0 * (1.0 - 0.18 * aod) - 8.0 * f[SLOPE], 450.0, 950.0)
    f[SOLAR_RADIATION] = k_down                                 # W/m2
    # incoming longwave L_down (W/m2): Swinbank clear-sky from air temp.
    ta_k = air_t + 273.15
    f[LONGWAVE_DOWN] = np.clip(5.31e-13 * ta_k ** 6, 300.0, 460.0)
    # planetary boundary layer height (m): deeper over hot dry urban cores.
    f[PBL_HEIGHT] = np.clip(900.0 + 1400.0 * urban - 500.0 * green
                            + 300.0 * (regional - 0.5), 300.0, 2800.0)  # m

    # ---- 11. net radiation Q* (consistent with the radiative drivers) ---
    # provisional skin temp (= air temp) for the Q* estimate; LST refines it.
    from urbanheat.physics.energy_balance import net_radiation_arr
    q_star = net_radiation_arr(albedo, k_down, f[LONGWAVE_DOWN], f[EMISSIVITY], air_t)
    f[NET_RADIATION] = np.clip(q_star, 50.0, 800.0)             # W/m2

    # cast everything to float32
    return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}


# ===========================================================================
# Synthetic LST from drivers via the SEB sign table
# ===========================================================================
def synthesize_lst(drivers: "dict[str, np.ndarray]") -> "np.ndarray":
    """Compute synthetic ``LST`` (degC) from driver arrays via the SEB sign table.

    Used by :meth:`SyntheticDataSource.load` and by tests as a ground-truth
    physics field. The response is built so the **signs match**
    :func:`urbanheat.physics.energy_balance.expected_lst_gradient_signs` (R5
    §1.6): warmer with impervious fraction / anthropogenic heat / low albedo /
    low SVF (nighttime longwave trapping) / built-up index, and cooler with NDVI
    / tree & vegetation cover / water fraction / higher albedo / wind /
    evapotranspiration.

    The construction blends a **radiative SEB anchor** (the equilibrium skin
    temperature that balances absorbed shortwave + longwave against emission,
    computed from the synthetic albedo/K_down/emissivity/air-temp) with explicit,
    correctly-signed driver perturbations, then adds a smooth regional gradient
    and mild noise. Returns a 2-D ``float32`` array (degC).
    """
    from urbanheat.datamodel import (
        AIR_TEMP, ALBEDO, ANTHRO_HEAT, BUILDING_HEIGHT, ELEVATION, EMISSIVITY, ET,
        GREEN_FRAC, IMPERVIOUS_FRAC, LONGWAVE_DOWN, NDBI, NDVI, SOLAR_RADIATION,
        SVF, TREE_FRAC, WATER_FRAC, WIND_SPEED,
    )
    from urbanheat.physics.energy_balance import lst_from_longwave, shortwave_net

    def _get(name: str, default: float) -> np.ndarray:
        if name in drivers:
            return np.asarray(drivers[name], dtype=np.float64)
        # shape from any present array
        any_arr = next(iter(drivers.values()))
        return np.full(np.asarray(any_arr).shape, float(default), dtype=np.float64)

    albedo = _get(ALBEDO, 0.18)
    k_down = _get(SOLAR_RADIATION, 800.0)
    emis = _get(EMISSIVITY, 0.96)
    l_down = _get(LONGWAVE_DOWN, 380.0)
    air_t = _get(AIR_TEMP, 31.0)
    imperv = _get(IMPERVIOUS_FRAC, 0.4)
    green = _get(GREEN_FRAC, 0.2)
    tree = _get(TREE_FRAC, 0.1)
    water = _get(WATER_FRAC, 0.0)
    svf = _get(SVF, 0.8)
    ndvi = _get(NDVI, 0.2)
    ndbi = _get(NDBI, 0.0)
    qf = _get(ANTHRO_HEAT, 10.0)
    wind = _get(WIND_SPEED, 2.5)
    et = _get(ET, 20.0)
    height = _get(BUILDING_HEIGHT, 5.0)
    elev = _get(ELEVATION, 180.0)
    shape = albedo.shape

    # --- (a) radiative SEB anchor ---------------------------------------
    # Daytime emitted longwave that balances absorbed short+long wave minus a
    # turbulent/storage sink that scales with available energy. Inverting the
    # radiative law gives an equilibrium skin temperature -> sets the overall
    # level and the albedo / K_down / emissivity dependence with correct signs.
    k_net = shortwave_net(albedo, k_down)                  # (1-alpha)K_down
    # fraction of net radiation NOT carried away by turbulent+storage fluxes and
    # therefore re-emitted as longwave: smaller over vegetated/wet/windy/open
    # pixels (efficient cooling), larger over dry impervious low-SVF pixels.
    sink = np.clip(
        0.30
        + 0.30 * imperv          # impervious -> storage, less Q_E (warmer)
        + 0.18 * (1.0 - svf)     # low SVF traps longwave (warmer, esp. night)
        - 0.22 * green           # vegetation -> Q_E (cooler)
        - 0.12 * tree            # canopy shade + transpiration (cooler)
        - 0.30 * water           # open water -> strong Q_E + heat capacity
        - 0.04 * np.clip(wind / 5.0, 0, 1)   # ventilation exports Q_H (cooler)
        + 0.04 * np.clip(qf / 60.0, 0, 1),   # anthropogenic input (warmer)
        0.10, 0.95,
    )
    emitted = k_net * (1.0 - sink) + emis * l_down + qf
    emitted = np.clip(emitted, emis * 5.670374419e-8 * (233.15) ** 4, None)
    lst_rad = lst_from_longwave(emitted, emis, longwave_down=l_down)   # degC
    # The radiative inversion gives a physically-correct *spatial pattern* but an
    # inflated absolute level (an isolated dry surface in radiative equilibrium
    # runs very hot; a real pixel exports much more via convection/storage). Use
    # it as a standardized, zero-mean pattern so it sets the relative structure
    # (albedo/K_down/SVF/moisture response) without dominating the magnitude.
    rad_pattern = (lst_rad - lst_rad.mean()) / (lst_rad.std() + 1e-6)

    # --- (b) explicit correctly-signed driver perturbations -------------
    # additive degC terms; magnitudes chosen for a realistic intra-urban spread
    # (~12-15 degC core-to-park) while preserving every sign in §1.6.
    dT = (
        + 9.0 * imperv           # + impervious -> hotter
        + 4.0 * np.clip(ndbi, -0.5, 0.7)   # + built-up index -> hotter
        + 5.0 * (1.0 - svf)      # + low SVF -> hotter (night trapping)
        + 4.0 * np.clip(qf / 60.0, 0, 1.5)  # + anthropogenic heat -> hotter
        + 0.04 * height          # + tall thermal mass -> hotter
        - 6.0 * np.clip(ndvi, -0.2, 0.9)    # + NDVI -> cooler
        - 5.0 * tree             # + tree canopy -> cooler
        - 4.0 * green            # + vegetation fraction -> cooler
        - 12.0 * water           # + water -> much cooler (strong evaporative island)
        - 18.0 * (albedo - 0.18)  # + albedo -> cooler (relative to typical 0.18)
        - 0.05 * et              # + evapotranspiration -> cooler
        - 0.30 * np.clip(wind - 2.5, -2.0, 4.0)   # + wind -> cooler
        - 0.0045 * (elev - float(np.mean(elev)))   # lapse-rate cooling with height
    )

    # --- (c) smooth regional gradient + mild reproducible noise ----------
    h, w = shape
    rng = np.random.default_rng(20260622)   # fixed: LST noise is deterministic
    grad = (np.linspace(-1.2, 1.2, w)[None, :] * np.ones((h, 1))
            + 0.6 * np.linspace(-1.0, 1.0, h)[:, None] * np.ones((1, w)))
    noise = gaussian_filter(rng.standard_normal(shape), sigma=1.2, mode="reflect")
    noise = 0.6 * noise / (noise.std() + 1e-9)

    # --- (d) assemble at a realistic absolute level ----------------------
    # Anchor to the downscaled air temperature plus a typical midday surface
    # excess (Ts - Ta ~ +8 degC over a mixed city; larger over dry built-up,
    # near-zero over water), add the radiative spatial pattern and the explicit
    # SEB-signed perturbations. Yields pre-monsoon daytime LST ~ low-30s to low-
    # 50s degC with clear hotspots and cool islands.
    surface_excess = 8.0 + 3.0 * imperv - 6.0 * water - 2.0 * green
    lst = air_t + surface_excess + 2.0 * rad_pattern + dT + 0.8 * grad + noise
    return np.asarray(lst, dtype=np.float32)


# ===========================================================================
# Geo-referencing helpers (dependency-free lon/lat -> UTM)
# ===========================================================================
def _utm_zone_from_crs(target_crs: str, fallback_lon: float) -> "tuple[int, bool]":
    """Return (zone_number, is_northern) for a ``EPSG:326xx/327xx`` UTM CRS.

    Falls back to deriving the zone from ``fallback_lon`` if the CRS is not a
    recognised UTM EPSG code (assumes northern hemisphere for India).
    """
    s = str(target_crs).upper().replace("EPSG:", "").strip()
    try:
        code = int(s)
        if 32601 <= code <= 32660:
            return code - 32600, True
        if 32701 <= code <= 32760:
            return code - 32700, False
    except ValueError:
        pass
    zone = int((fallback_lon + 180.0) / 6.0) + 1
    return zone, True


def _lonlat_to_utm(
    lon: float, lat: float, zone: int, northern: bool = True,
) -> "tuple[float, float]":
    """Forward WGS84 lon/lat -> UTM easting/northing (m), dependency-free.

    Standard Transverse-Mercator (Snyder/USGS) series on the WGS84 ellipsoid.
    Accurate to well under a metre across a city window — sufficient to give the
    synthetic FeatureStack a faithful metric ``transform``/``bounds`` in the
    requested UTM zone without pulling in pyproj.
    """
    a = 6378137.0                 # WGS84 semi-major axis (m)
    f = 1.0 / 298.257223563       # flattening
    e2 = f * (2.0 - f)            # first eccentricity squared
    ep2 = e2 / (1.0 - e2)
    k0 = 0.9996
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)   # central meridian
    phi = np.radians(lat)
    lam = np.radians(lon)
    N = a / np.sqrt(1.0 - e2 * np.sin(phi) ** 2)
    T = np.tan(phi) ** 2
    C = ep2 * np.cos(phi) ** 2
    A = (lam - lon0) * np.cos(phi)
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * phi)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * phi)
             - (35 * e2 ** 3 / 3072) * np.sin(6 * phi))
    easting = (k0 * N * (A + (1 - T + C) * A ** 3 / 6
                         + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120)
               + 500000.0)
    northing = (k0 * (M + N * np.tan(phi) * (A ** 2 / 2
                + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720)))
    if not northern:
        northing += 10000000.0
    return float(easting), float(northing)


def _grid_geometry(config: "Config") -> "tuple[tuple[int, int], tuple, tuple]":
    """Derive ``(shape, transform, bounds)`` in ``config.target_crs`` from the bbox.

    Shape is ``config.grid_shape`` if set, else from the bbox extent and
    ``resolution_m``. ``transform`` is a north-up rasterio-style affine
    ``(a, b, c, d, e, f)`` with pixel size = ``resolution_m`` and origin at the
    upper-left; ``bounds = (xmin, ymin, xmax, ymax)`` in UTM metres.
    """
    xmin_ll, ymin_ll, xmax_ll, ymax_ll = config.bbox
    lon_c = 0.5 * (xmin_ll + xmax_ll)
    zone, northern = _utm_zone_from_crs(config.target_crs, lon_c)
    # project the two bbox corners to UTM (axis-aligned box in UTM metres).
    e_min, n_min = _lonlat_to_utm(xmin_ll, ymin_ll, zone, northern)
    e_max, n_max = _lonlat_to_utm(xmax_ll, ymax_ll, zone, northern)
    xmin, xmax = min(e_min, e_max), max(e_min, e_max)
    ymin, ymax = min(n_min, n_max), max(n_min, n_max)

    res = float(config.resolution_m)
    if config.grid_shape is not None:
        rows, cols = int(config.grid_shape[0]), int(config.grid_shape[1])
    else:
        cols = max(2, int(round((xmax - xmin) / res)))
        rows = max(2, int(round((ymax - ymin) / res)))
    # snap bounds to an exact rows*cols*res grid anchored at the UTM upper-left.
    x_extent = cols * res
    y_extent = rows * res
    xmax_snap = xmin + x_extent
    ymin_snap = ymax - y_extent
    transform = (res, 0.0, xmin, 0.0, -res, ymax)   # north-up affine
    bounds = (xmin, ymin_snap, xmax_snap, ymax)
    return (rows, cols), transform, bounds


# ===========================================================================
# The DataSource implementation
# ===========================================================================
class SyntheticDataSource(DataSource):
    """Offline synthetic backend producing a physically-coherent FeatureStack.

    Implements the :class:`~urbanheat.datamodel.DataSource` interface so it is a
    drop-in replacement for ``GEEDataSource``; everything downstream consumes the
    returned :class:`~urbanheat.datamodel.FeatureStack` identically. No optional
    dependencies (numpy + scipy only); seeded by ``config.seed``.
    """

    name = "synthetic"

    def load(self, config: "Config") -> "FeatureStack":
        """Generate and return a fully-populated, validated FeatureStack.

        Builds the grid geometry from ``config`` (bbox + ``resolution_m`` or
        ``grid_shape``) in ``config.target_crs``, generates all canonical driver
        layers via :func:`make_synthetic_fields`, synthesizes a SEB-consistent
        ``LST`` (plus ``LST_DAY``/``LST_NIGHT``/``LST_UNCERTAINTY``) via
        :func:`synthesize_lst`, attaches CRS / transform / bounds / per-variable
        provenance metadata, validates and returns it. Reproducible for a given
        ``config.seed``.
        """
        from urbanheat.datamodel import (
            FeatureStack, LST, LST_DAY, LST_NIGHT,
            LST_UNCERTAINTY, NIGHTLIGHTS, SVF,
        )

        shape, transform, bounds = _grid_geometry(config)

        # 1. all driver fields (seeded, reproducible)
        fields = make_synthetic_fields(shape, seed=int(config.seed))

        # 2. SEB-consistent LST (degC) from the drivers
        lst = synthesize_lst(fields)
        fields[LST] = lst

        # 3. diurnal split + per-pixel uncertainty
        #    day LST tracks the surface field; night LST is cooler overall but the
        #    URBAN-RURAL contrast PERSISTS via stored heat + low-SVF trapping +
        #    anthropogenic heat (the nocturnal SUHI). [R5 §2]
        svf = np.asarray(fields[SVF], dtype=np.float64)
        ntl = np.asarray(fields[NIGHTLIGHTS], dtype=np.float64)
        ntl_n = ntl / (np.nanmax(ntl) + 1e-9)
        night_drop = 8.0 - 3.5 * (1.0 - svf) - 2.5 * ntl_n   # smaller drop downtown
        fields[LST_DAY] = lst.astype(np.float32)
        fields[LST_NIGHT] = (lst - np.clip(night_drop, 1.0, 9.0)).astype(np.float32)
        rng = np.random.default_rng(int(config.seed) + 7)
        unc = 0.6 + 0.4 * np.abs(rng.standard_normal(shape))     # ~0.6-1.5 degC
        fields[LST_UNCERTAINTY] = np.asarray(unc, dtype=np.float32)

        # 4. provenance metadata (per-variable + run-level)
        provenance = {name: "synthetic" for name in fields}
        units = _UNITS
        meta: dict[str, Any] = {
            "city": getattr(config, "city", "synthetic"),
            "city_name": _city_name(config),
            "mode": "synthetic",
            "source": "SyntheticDataSource",
            "seed": int(config.seed),
            "resolution_m": float(config.resolution_m),
            "bbox_lonlat": tuple(config.bbox),
            "start_date": getattr(config, "start_date", None),
            "end_date": getattr(config, "end_date", None),
            "provenance": provenance,
            "units": {k: units.get(k, "") for k in fields},
            "description": (
                "Physically-coherent synthetic Indian city (urban core + "
                "sub-centres, parks, river + lake) with SEB-consistent LST."
            ),
        }

        # 5. assemble + validate
        stack = FeatureStack.from_arrays(
            layers=fields, transform=transform, crs=config.target_crs,
            bounds=bounds, meta=meta,
        )
        # from_arrays already validates; call again for an explicit contract check.
        stack.validate()
        return stack

    def available_layers(self) -> "list[str]":
        """All canonical layers this backend produces (the full driver stack)."""
        from urbanheat.datamodel import ALL_VARIABLES
        return list(ALL_VARIABLES)


# ===========================================================================
# Per-variable units (documentation / metadata)
# ===========================================================================
_UNITS: dict[str, str] = {
    "lst": "degC", "lst_day": "degC", "lst_night": "degC",
    "lst_uncertainty": "degC", "emissivity": "0-1",
    "ndvi": "index", "evi": "index", "savi": "index", "ndwi": "index",
    "mndwi": "index", "ndbi": "index", "ndbai": "index", "ui": "index",
    "lai": "m2/m2", "fvc": "0-1", "et": "mm/period", "albedo": "0-1",
    "lulc": "class", "impervious_frac": "0-1", "green_frac": "0-1",
    "water_frac": "0-1", "tree_frac": "0-1", "lcz": "class",
    "building_height": "m", "building_volume": "m3/cell", "svf": "0-1",
    "aspect_ratio": "H/W", "plan_area_frac": "0-1", "frontal_area_index": "0-1",
    "roughness_length": "m", "displacement_height": "m", "elevation": "m",
    "slope": "deg", "air_temp": "degC", "dewpoint": "degC",
    "rel_humidity": "%", "wind_speed": "m/s", "pressure": "kPa",
    "solar_radiation": "W/m2", "longwave_down": "W/m2", "net_radiation": "W/m2",
    "soil_moisture": "m3/m3", "aod": "dimensionless", "pbl_height": "m",
    "population": "persons/cell", "nightlights": "nW/cm2/sr",
    "anthro_heat": "W/m2", "no2": "mol/m2",
}


def _city_name(config: "Config") -> str:
    """Resolve a human-readable city name from the config (preset or city field)."""
    try:
        from urbanheat.config import CITY_PRESETS
        key = getattr(config, "city", None)
        if key in CITY_PRESETS:
            return CITY_PRESETS[key].get("name", key)
        return str(key)
    except Exception:
        return str(getattr(config, "city", "synthetic"))


__all__ = [
    "SyntheticDataSource",
    "make_synthetic_fields",
    "synthesize_lst",
]
