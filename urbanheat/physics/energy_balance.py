"""urbanheat.physics.energy_balance — the Surface Energy Balance (SEB) backbone.

This module makes the physics of urban heat *first-class* (ARCHITECTURE §8, R5 §1).
The surface energy balance of an urban facet / satellite pixel over an averaging
interval is

    Q*  =  Q_H + Q_E + dQ_S + Q_F                               (1) balance
    Q*  =  (1 - alpha) K_down + (L_down - L_up)                 (2) net radiation
    L_up = eps * sigma * Ts^4 + (1 - eps) L_down                (3) radiative law
    Q_H  = rho cp (Ts - Ta) / r_a                               (4) sensible
    Q_E  = (rho cp / gamma)(e_s(Ts) - e_a)/(r_a + r_s)          (5) latent
    beta = Q_H / Q_E                                            (6) Bowen ratio
    dQ_S = sum_i f_i (a1_i Q* + a2_i dQ*/dt + a3_i)             (7) OHM storage

Land Surface Temperature (LST) is the radiometric skin temperature ``T_s`` that
the thermal sensor inverts from (3); *anything* that raises absorbed radiation or
cuts turbulent/storage losses raises ``T_s``. The **driver -> SEB term -> dLST
sign** map (R5 §1.6) is the bridge between this physics and the ML monotonicity
constraints, and is exported here as :func:`expected_lst_gradient_signs` so the
synthetic generator, ``models.features.monotone_constraints`` and
``models.attribution.physics_sign_audit`` all share one source of truth.

Design rules (per ARCHITECTURE §11 and the build contract)
----------------------------------------------------------
* Top-level imports are **numpy only**. Everything else — including
  :mod:`urbanheat.constants` and :mod:`urbanheat.datamodel` — is imported lazily
  inside the functions that need it, so the synthetic path imports without any
  optional dependency and there is no import cycle.
* All maths is **vectorized** and works on scalars or ndarrays alike.
* Temperatures: public array helpers take **degrees Celsius** (the FeatureStack
  ``LST``/``AIR_TEMP`` convention) and convert to Kelvin internally for the
  radiative / flux laws. Helpers are suffixed/ documented so the unit is explicit.

All physical constants (sigma, c_p, rho, gamma, kappa, Kelvin offset) come from
``constants.PHYSICAL_CONSTANTS`` — never hard-coded here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # hints only; never imported at runtime here
    from urbanheat.datamodel import FeatureStack


# ===========================================================================
# Internal constant access (lazy, cached) — keeps the top-level import numpy-only
# ===========================================================================
def _phys() -> dict:
    """Return ``constants.PHYSICAL_CONSTANTS`` (imported lazily, memoized)."""
    global _PHYS_CACHE
    try:
        return _PHYS_CACHE  # type: ignore[name-defined]
    except NameError:
        from urbanheat.constants import PHYSICAL_CONSTANTS
        _PHYS_CACHE = PHYSICAL_CONSTANTS  # noqa: F841  (module-level cache)
        return _PHYS_CACHE


def _c2k(t_c: "np.ndarray | float") -> "np.ndarray | float":
    """Celsius -> Kelvin using ``constants.KELVIN_OFFSET``."""
    return np.asarray(t_c, dtype=np.float64) + _phys()["KELVIN_OFFSET"]


def _k2c(t_k: "np.ndarray | float") -> "np.ndarray | float":
    """Kelvin -> Celsius using ``constants.KELVIN_OFFSET``."""
    return np.asarray(t_k, dtype=np.float64) - _phys()["KELVIN_OFFSET"]


def saturation_vapor_pressure(t_c: "np.ndarray | float") -> "np.ndarray | float":
    """Saturation vapour pressure ``e_s`` (kPa) from temperature (degC).

    Tetens / FAO-56 form ``e_s = 0.6108 * exp(17.27 T / (T + 237.3))`` with ``T``
    in degC and ``e_s`` in kPa. Used by the latent-heat term (Eq. 5).
    """
    t = np.asarray(t_c, dtype=np.float64)
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


# ===========================================================================
# (3) Radiative law — outgoing longwave and its inversion to LST
# ===========================================================================
def longwave_up(
    lst_c: "np.ndarray",
    emissivity: "np.ndarray",
    longwave_down: "np.ndarray | None" = None,
) -> "np.ndarray":
    """Outgoing longwave ``L_up`` (W/m^2) from the radiative law (R5 Eq. 3).

    ``L_up = eps * sigma * Ts^4 (+ (1 - eps) * L_down)``  with ``Ts`` in Kelvin
    (``lst_c`` supplied in degC and converted internally). When ``longwave_down``
    is given the reflected term ``(1 - eps) L_down`` is added (the full Eq. 3);
    otherwise only the emitted ``eps sigma Ts^4`` is returned.

    Pure vectorized array helper. ``sigma`` is ``constants.SIGMA_SB``.
    """
    sigma = _phys()["STEFAN_BOLTZMANN"]
    eps = np.asarray(emissivity, dtype=np.float64)
    ts_k = _c2k(lst_c)
    emitted = eps * sigma * ts_k ** 4
    if longwave_down is None:
        return emitted.astype(np.float64)
    ld = np.asarray(longwave_down, dtype=np.float64)
    return (emitted + (1.0 - eps) * ld).astype(np.float64)


def lst_from_longwave(
    longwave_up_arr: "np.ndarray",
    emissivity: "np.ndarray",
    longwave_down: "np.ndarray | None" = None,
) -> "np.ndarray":
    """Invert the radiative law (Eq. 3) for LST in **degC**.

    ``Ts = [ (L_up - (1 - eps) L_down) / (eps sigma) ]^(1/4)``. This is exactly
    the inversion a thermal sensor performs to retrieve LST and is the cleanest
    physics to embed downstream. Returns degC.
    """
    sigma = _phys()["STEFAN_BOLTZMANN"]
    eps = np.asarray(emissivity, dtype=np.float64)
    lu = np.asarray(longwave_up_arr, dtype=np.float64)
    if longwave_down is not None:
        lu = lu - (1.0 - eps) * np.asarray(longwave_down, dtype=np.float64)
    ts_k = np.power(np.clip(lu, 1e-6, None) / (eps * sigma), 0.25)
    return _k2c(ts_k).astype(np.float64)


# ===========================================================================
# (2) Net radiation
# ===========================================================================
def shortwave_net(albedo: "np.ndarray", k_down: "np.ndarray") -> "np.ndarray":
    """Absorbed net shortwave ``K_net = (1 - alpha) K_down`` (W/m^2).

    The dominant daytime radiative input; cool/white surfaces (high ``alpha``)
    cut the absorbed shortwave. Pure array helper. [R5 Eq. 2]
    """
    a = np.asarray(albedo, dtype=np.float64)
    k = np.asarray(k_down, dtype=np.float64)
    return ((1.0 - a) * k).astype(np.float64)


def net_radiation_arr(
    albedo: "np.ndarray",
    k_down: "np.ndarray",
    l_down: "np.ndarray",
    emissivity: "np.ndarray",
    lst_c: "np.ndarray",
) -> "np.ndarray":
    """Net all-wave radiation ``Q*`` (W/m^2) as a pure array helper (R5 Eq. 2-3).

    ``Q* = (1 - alpha) K_down + (L_down - L_up)`` with the full radiative
    ``L_up = eps sigma Ts^4 + (1 - eps) L_down`` (so the reflected-longwave term
    cancels cleanly to ``Q* = (1-alpha)K_down + eps(L_down - sigma Ts^4)``).
    ``lst_c`` in degC.

    This is the array-level twin of :func:`net_radiation`, which operates on a
    :class:`FeatureStack`. Used by the synthetic generator and tests.
    """
    eps = np.asarray(emissivity, dtype=np.float64)
    ld = np.asarray(l_down, dtype=np.float64)
    k_net = shortwave_net(albedo, k_down)
    l_up = longwave_up(lst_c, eps, longwave_down=ld)
    return (k_net + (ld - l_up)).astype(np.float64)


def net_radiation(fs: "FeatureStack") -> "FeatureStack":
    """Compute ``Q*`` and write it to the stack as ``NET_RADIATION`` (R5 Eq. 2-3).

    ``Q* = (1 - albedo) K_down + (L_down - eps sigma Ts^4 - (1-eps) L_down)``.
    Reads ``ALBEDO, SOLAR_RADIATION, LONGWAVE_DOWN, EMISSIVITY, LST``. Mutates
    ``fs`` in place and returns it (chaining), per the §11 convention.

    Missing ``LONGWAVE_DOWN`` is tolerated by estimating clear-sky atmospheric
    longwave from air temperature when present (Swinbank-style), else from LST,
    so the function still produces a sensible ``Q*`` in minimal stacks.
    """
    from urbanheat.datamodel import (
        ALBEDO, AIR_TEMP, EMISSIVITY, LONGWAVE_DOWN, LST,
        NET_RADIATION, SOLAR_RADIATION,
    )

    sigma = _phys()["STEFAN_BOLTZMANN"]
    albedo = np.asarray(fs.get(ALBEDO), dtype=np.float64)
    k_down = np.asarray(fs.get(SOLAR_RADIATION), dtype=np.float64)
    eps = np.asarray(fs.get(EMISSIVITY), dtype=np.float64)
    lst_c = np.asarray(fs.get(LST), dtype=np.float64)

    if fs.has(LONGWAVE_DOWN):
        l_down = np.asarray(fs.get(LONGWAVE_DOWN), dtype=np.float64)
    elif fs.has(AIR_TEMP):
        # Swinbank clear-sky downwelling longwave ~ 5.31e-13 * Ta^6 (Ta in K).
        ta_k = _c2k(fs.get(AIR_TEMP))
        l_down = 5.31e-13 * ta_k ** 6
    else:
        # fall back to grey-body atmosphere at surface temperature
        l_down = 0.85 * sigma * _c2k(lst_c) ** 4

    q_star = net_radiation_arr(albedo, k_down, l_down, eps, lst_c)
    fs.add_layer(NET_RADIATION, q_star.astype(np.float32))
    return fs


# ===========================================================================
# Aerodynamic resistance + (4) sensible / (5) latent fluxes
# ===========================================================================
def aerodynamic_resistance(
    wind_speed: "np.ndarray",
    roughness_length: "np.ndarray",
    measurement_height: float = 10.0,
) -> "np.ndarray":
    """Bulk aerodynamic resistance ``r_a`` (s/m) from wind & roughness.

    Neutral log-law form ``r_a = ln(z/z0)^2 / (kappa^2 u)`` with ``kappa`` the
    von-Karman constant (``constants.VON_KARMAN``). ``z0`` is clipped to a small
    positive floor and ``u`` to a light-breeze floor so ``r_a`` stays finite over
    very smooth / calm pixels. Higher wind / rougher city => smaller ``r_a`` =>
    more efficient turbulent export of ``Q_H`` (cooler), per the sign table.
    """
    kappa = _phys()["VON_KARMAN"]
    u = np.clip(np.asarray(wind_speed, dtype=np.float64), 0.3, None)
    z0 = np.clip(np.asarray(roughness_length, dtype=np.float64), 1e-3, None)
    z = float(measurement_height)
    ra = np.log(z / z0) ** 2 / (kappa ** 2 * u)
    return np.clip(ra, 1.0, 1e4).astype(np.float64)


def sensible_heat(
    lst_c: "np.ndarray",
    air_temp_c: "np.ndarray",
    wind_speed: "np.ndarray",
    roughness_length: "np.ndarray",
) -> "np.ndarray":
    """Turbulent sensible heat ``Q_H`` (W/m^2) — bulk-aerodynamic form (R5 Eq. 4).

    ``Q_H = rho cp (Ts - Ta) / r_a`` with ``rho = RHO_AIR``, ``cp = CP_AIR`` from
    constants and ``r_a`` from :func:`aerodynamic_resistance`. ``lst_c`` and
    ``air_temp_c`` in degC (the Ts-Ta difference is unit-invariant). Positive when
    the surface is warmer than the air. Pure array helper.
    """
    rho = _phys()["RHO_AIR"]
    cp = _phys()["CP_AIR"]
    ts = np.asarray(lst_c, dtype=np.float64)
    ta = np.asarray(air_temp_c, dtype=np.float64)
    ra = aerodynamic_resistance(wind_speed, roughness_length)
    return (rho * cp * (ts - ta) / ra).astype(np.float64)


def latent_heat(
    lst_c: "np.ndarray",
    air_temp_c: "np.ndarray",
    rel_humidity: "np.ndarray",
    wind_speed: "np.ndarray",
    roughness_length: "np.ndarray",
    surface_resistance: "np.ndarray | float" = 100.0,
) -> "np.ndarray":
    """Turbulent latent heat ``Q_E`` (W/m^2) — resistance form (R5 Eq. 5).

    ``Q_E = (rho cp / gamma) (e_s(Ts) - e_a) / (r_a + r_s)`` with vapour-pressure
    deficit driving evapotranspiration. ``e_a = (RH/100) e_s(Ta)``. ``gamma`` is
    the psychrometric constant (``constants.PSYCHROMETRIC_GAMMA``). Vegetation /
    moist surfaces enter through a **low** ``surface_resistance`` ``r_s`` (large
    ``Q_E`` => evaporative cooling); impervious surfaces have ``r_s -> inf`` so
    ``Q_E -> 0``. Pure array helper.
    """
    rho = _phys()["RHO_AIR"]
    cp = _phys()["CP_AIR"]
    gamma = _phys()["PSYCHROMETRIC_GAMMA"]
    e_s_ts = saturation_vapor_pressure(lst_c)
    e_a = np.asarray(rel_humidity, dtype=np.float64) / 100.0 * \
        saturation_vapor_pressure(air_temp_c)
    vpd = np.clip(e_s_ts - e_a, 0.0, None)
    ra = aerodynamic_resistance(wind_speed, roughness_length)
    rs = np.asarray(surface_resistance, dtype=np.float64)
    return (rho * cp / gamma * vpd / (ra + rs)).astype(np.float64)


def bowen_ratio(q_h: "np.ndarray", q_e: "np.ndarray") -> "np.ndarray":
    """Bowen ratio ``beta = Q_H / Q_E`` (R5 Eq. 6) — the compact "why is it hot".

    High ``beta`` = dry/hot city (energy goes to sensible heat + storage); low
    ``beta`` = vegetated/cool. ``Q_E`` is floored to a small positive value to
    avoid divide-by-zero over fully impervious pixels. Pure array helper.
    """
    qh = np.asarray(q_h, dtype=np.float64)
    qe = np.asarray(q_e, dtype=np.float64)
    return (qh / np.clip(np.abs(qe), 1.0, None) * np.sign(qe + 1e-12)).astype(np.float64)


# ===========================================================================
# (7) Storage heat — Objective Hysteresis Model (OHM)
# ===========================================================================
#: OHM (a1 dimensionless, a2 hours, a3 W/m^2) coefficients per surface-cover
#: fraction key, from SUEWS / Grimmond & Oke typical values (R5 §1.4). Built
#: fabric stores a large share of Q* and releases it at night (the nocturnal
#: SUHI); vegetation/water store little. Keyed by FeatureStack fraction names.
OHM_COEFFICIENTS: dict[str, tuple[float, float, float]] = {
    "impervious_frac": (0.70, 0.32, -38.0),   # concrete/asphalt: high storage
    "green_frac": (0.30, 0.18, -27.0),        # short vegetation
    "tree_frac": (0.27, 0.13, -23.0),         # canopy
    "water_frac": (0.50, 0.21, -39.0),        # high heat capacity, damped phase
}


def storage_heat_ohm(
    net_rad: "np.ndarray",
    fractions: "dict[str, np.ndarray]",
    dq_star_dt: "np.ndarray | float | None" = None,
) -> "np.ndarray":
    """OHM storage heat flux ``dQ_S`` (W/m^2) weighted by surface-cover fraction.

    ``dQ_S = sum_i f_i (a1_i Q* + a2_i dQ*/dt + a3_i)`` (R5 Eq. 7) with the
    per-cover ``(a1, a2, a3)`` from :data:`OHM_COEFFICIENTS`. ``fractions`` maps
    canonical fraction layer names (``impervious_frac`` ...) to weight arrays;
    unknown keys are ignored. ``dq_star_dt`` (the instantaneous tendency of
    ``Q*``) defaults to 0 (the time-mean / quasi-steady case used for daily LST).

    Physically: impervious mass has the largest ``a1`` => stores the most daytime
    heat => drives the nighttime UHI. Pure array helper.
    """
    qs = np.asarray(net_rad, dtype=np.float64)
    if dq_star_dt is None:
        dqdt = 0.0
    else:
        dqdt = np.asarray(dq_star_dt, dtype=np.float64)
    out = np.zeros_like(qs, dtype=np.float64)
    any_frac = False
    for name, (a1, a2, a3) in OHM_COEFFICIENTS.items():
        if name in fractions:
            f = np.asarray(fractions[name], dtype=np.float64)
            out = out + f * (a1 * qs + a2 * dqdt + a3)
            any_frac = True
    if not any_frac:
        # no cover info: fall back to a single built-fabric slab response
        a1, a2, a3 = OHM_COEFFICIENTS["impervious_frac"]
        out = a1 * qs + a2 * dqdt + a3
    return out.astype(np.float64)


# ===========================================================================
# Anthropogenic heat helper
# ===========================================================================
def anthropogenic_heat_flux(
    fs: "FeatureStack",
    qf_max: float = 60.0,
) -> "np.ndarray":
    """Anthropogenic heat ``Q_F`` (W/m^2) from the stack, or 0 if unavailable.

    Prefers an explicit ``ANTHRO_HEAT`` layer; otherwise builds a proxy from
    normalized ``NIGHTLIGHTS`` / ``POPULATION`` scaled to ``qf_max``. The night
    is when ``Q_F`` matters most for the UHI. Pure helper used by SEB closure.
    """
    from urbanheat.datamodel import ANTHRO_HEAT, NIGHTLIGHTS, POPULATION

    if fs.has(ANTHRO_HEAT):
        return np.asarray(fs.get(ANTHRO_HEAT), dtype=np.float64)
    proxy = None
    for name in (NIGHTLIGHTS, POPULATION):
        if fs.has(name):
            a = np.asarray(fs.get(name), dtype=np.float64)
            rng = np.nanmax(a) - np.nanmin(a)
            n = (a - np.nanmin(a)) / rng if rng > 0 else np.zeros_like(a)
            proxy = n if proxy is None else 0.5 * (proxy + n)
    if proxy is None:
        from urbanheat.datamodel import LST
        return np.zeros_like(np.asarray(fs.get(LST), dtype=np.float64))
    return (qf_max * proxy).astype(np.float64)


# ===========================================================================
# Physics-only LST backbone and SEB closure residual
# ===========================================================================
def physics_lst(fs: "FeatureStack") -> "np.ndarray":
    """Physics-only LST backbone in **degC** by inverting the SEB (R5 §4 A).

    Closes the surface energy balance for the radiating temperature: the absorbed
    radiation that is *not* exported as turbulent (``Q_H + Q_E``) or stored
    (``dQ_S``), plus the anthropogenic input ``Q_F``, must be emitted as longwave,
    so

        eps sigma Ts^4 = (1-alpha)K_down + eps L_down - Q_H - Q_E - dQ_S + Q_F

    which we invert for ``Ts``. This gives the trend ``LST_phys`` the hybrid model
    in :mod:`urbanheat.models.train` learns the residual on; it guarantees correct
    extrapolation when drivers move outside the training envelope.

    Reads ``ALBEDO, SOLAR_RADIATION, EMISSIVITY, AIR_TEMP, LST`` and, when present,
    ``LONGWAVE_DOWN, WIND_SPEED, ROUGHNESS_LENGTH, REL_HUMIDITY, SOIL_MOISTURE,
    GREEN_FRAC, TREE_FRAC, WATER_FRAC, IMPERVIOUS_FRAC``. Sensible defaults are
    used for any missing driver so it runs on minimal stacks. Returns a 2-D array.
    """
    from urbanheat.datamodel import (
        AIR_TEMP, ALBEDO, EMISSIVITY, GREEN_FRAC, IMPERVIOUS_FRAC, LONGWAVE_DOWN,
        SOIL_MOISTURE, SOLAR_RADIATION, TREE_FRAC, WATER_FRAC, WIND_SPEED,
    )

    sigma = _phys()["STEFAN_BOLTZMANN"]
    shape = fs.shape

    def _g(name: str, default: float) -> np.ndarray:
        if fs.has(name):
            return np.asarray(fs.get(name), dtype=np.float64)
        return np.full(shape, float(default), dtype=np.float64)

    albedo = _g(ALBEDO, 0.18)
    k_down = _g(SOLAR_RADIATION, 800.0)
    eps = _g(EMISSIVITY, 0.95)
    air_c = _g(AIR_TEMP, 30.0)
    wind = _g(WIND_SPEED, 2.5)

    # downwelling longwave: measured if present, else Swinbank from air temp
    if fs.has(LONGWAVE_DOWN):
        l_down = np.asarray(fs.get(LONGWAVE_DOWN), dtype=np.float64)
    else:
        l_down = 5.31e-13 * _c2k(air_c) ** 6

    green = _g(GREEN_FRAC, 0.2)
    tree = _g(TREE_FRAC, 0.1)
    water = _g(WATER_FRAC, 0.0)
    imperv = _g(IMPERVIOUS_FRAC, 0.4)
    sm = _g(SOIL_MOISTURE, 0.2)
    veg = np.clip(green + 0.5 * tree, 0.0, 1.0)

    # Available energy at the surface (NARP-style): absorbed shortwave + net
    # longwave evaluated against the AIR temperature as the radiative reference.
    k_net = shortwave_net(albedo, k_down)
    q_star = k_net + eps * (l_down - sigma * _c2k(air_c) ** 4)
    q_f = anthropogenic_heat_flux(fs)
    avail = q_star + q_f                                  # energy to be partitioned

    # LUMPS/equilibrium closure: the turbulent (Q_H + Q_E) + storage (dQ_S) export
    # is a FRACTION of the available energy rather than a flux evaluated at the
    # (unknown) skin temperature — this avoids the linearization overshoot of
    # evaluating Q_H,Q_E at Ts=Ta (which would export ~0 and bake the surface).
    # The non-emitted fraction (what is re-radiated as longwave) is SMALL over
    # well-watered/vegetated/windy surfaces (efficient turbulent cooling) and
    # LARGER over dry, impervious, calm pixels (energy piles into storage +
    # sensible heat -> hotter skin). Bounded to a physical band.
    evaporative_fraction = np.clip(
        0.30 + 0.55 * veg + 0.62 * water
        + 0.30 * np.clip(sm / 0.4, 0.0, 1.0)
        + 0.08 * np.clip(wind / 5.0, 0.0, 1.0),
        0.10, 0.92,
    )
    storage_fraction = np.clip(0.35 * imperv + 0.08, 0.0, 0.45)
    # residual fraction of available energy that must leave as ADDED longwave
    # emission above the atmospheric reference level (the daytime skin excess).
    radiated_fraction = np.clip(1.0 - evaporative_fraction - storage_fraction, 0.02, 0.70)

    # eps sigma Ts^4 = eps sigma Ta^4 + radiated_fraction * avail
    emitted = eps * sigma * _c2k(air_c) ** 4 + radiated_fraction * avail
    emitted = np.clip(emitted, eps * sigma * _c2k(-40.0) ** 4, None)
    ts_k = np.power(emitted / (eps * sigma), 0.25)
    return _k2c(ts_k).astype(np.float64)


def seb_residual(fs: "FeatureStack") -> "np.ndarray":
    """SEB closure residual ``|Q* - (Q_H + Q_E + dQ_S + Q_F)|`` (W/m^2). [R9 §1.7]

    A physics-consistency check: for an energy-conserving field this is ~0
    everywhere. Reads/derives ``NET_RADIATION`` (via :func:`net_radiation` if
    absent) and the turbulent / storage / anthropogenic terms from the stack with
    sensible defaults. Returns a 2-D array of the absolute residual.

    Note: on the **synthetic** stack the residual is *not* near zero (it sits at a
    few hundred W/m^2). The synthetic LST is generated from the driver->dLST sign
    table (:func:`expected_lst_gradient_signs`), not by inverting a closed SEB, so
    the bulk-aerodynamic ``Q_H``/``Q_E`` here are evaluated with independent
    default exchange coefficients rather than the ones implied by that LST. This
    figure is therefore a *diagnostic* of temperature-dependent flux closure under
    those defaults — useful for spotting gross sign/magnitude errors — and not a
    target to be driven to zero; doing so would require co-fitting the flux
    parameterisation to the same LST field (the job of the optional SEB-closure
    PINN loss), which is out of scope for this static check.
    """
    from urbanheat.datamodel import (
        AIR_TEMP, IMPERVIOUS_FRAC, GREEN_FRAC, LST, NET_RADIATION,
        REL_HUMIDITY, ROUGHNESS_LENGTH, SOIL_MOISTURE, TREE_FRAC, WATER_FRAC,
        WIND_SPEED,
    )

    shape = fs.shape

    def _g(name: str, default: float) -> np.ndarray:
        if fs.has(name):
            return np.asarray(fs.get(name), dtype=np.float64)
        return np.full(shape, float(default), dtype=np.float64)

    if fs.has(NET_RADIATION):
        q_star = np.asarray(fs.get(NET_RADIATION), dtype=np.float64)
    else:
        q_star = np.asarray(net_radiation(fs).get(NET_RADIATION), dtype=np.float64)

    lst_c = _g(LST, 30.0)
    air_c = _g(AIR_TEMP, 30.0)
    wind = _g(WIND_SPEED, 2.5)
    z0 = _g(ROUGHNESS_LENGTH, 0.5)
    rh = _g(REL_HUMIDITY, 45.0)
    sm = _g(SOIL_MOISTURE, 0.2)
    green = _g(GREEN_FRAC, 0.2)
    tree = _g(TREE_FRAC, 0.1)
    water = _g(WATER_FRAC, 0.0)
    imperv = _g(IMPERVIOUS_FRAC, 0.4)

    veg = np.clip(green + 0.5 * tree, 0.0, 1.0)
    r_s = 20.0 + 4000.0 * (1.0 - veg) * (1.0 - np.clip(sm / 0.4, 0.0, 1.0))
    r_s = np.where(water > 0.5, 10.0, r_s)

    q_h = sensible_heat(lst_c, air_c, wind, z0)
    q_e = latent_heat(lst_c, air_c, rh, wind, z0, surface_resistance=r_s)
    dqs = storage_heat_ohm(
        q_star,
        {"impervious_frac": imperv, "green_frac": green,
         "tree_frac": tree, "water_frac": water},
    )
    q_f = anthropogenic_heat_flux(fs)
    return np.abs(q_star - (q_h + q_e + dqs + q_f)).astype(np.float64)


# ===========================================================================
# The driver -> dLST sign table (R5 §1.6) — the physics<->ML bridge
# ===========================================================================
def expected_lst_gradient_signs() -> "dict[str, int]":
    """Expected sign of ``d LST / d driver`` for each driver (R5 §1.6 table).

    This is THE single source of truth for the SEB monotonicity directions used
    across the system: the synthetic generator builds LST to obey it,
    ``models.features.monotone_constraints`` feeds it to the GBM
    ``monotone_constraints``, ``models.attribution.physics_sign_audit`` checks
    learned effects against it, and the PINN uses it for ``L_mono``.

    Returns a dict mapping the canonical FeatureStack driver name to ``-1``
    (driver up => LST down, i.e. a cooling lever), ``+1`` (driver up => LST up,
    a warming lever), or ``0`` (no enforced monotone direction). Keys use the
    canonical constants from :mod:`urbanheat.datamodel`.
    """
    from urbanheat.datamodel import (
        AIR_TEMP, ALBEDO, ANTHRO_HEAT, AOD, ASPECT_RATIO, BUILDING_HEIGHT,
        BUILDING_VOLUME, DEWPOINT, ELEVATION, EMISSIVITY, ET, EVI, FRONTAL_AREA_INDEX,
        FVC, GREEN_FRAC, IMPERVIOUS_FRAC, LAI, LONGWAVE_DOWN, MNDWI, NDBAI, NDBI,
        NDVI, NDWI, NET_RADIATION, NIGHTLIGHTS, NO2, PLAN_AREA_FRAC, POPULATION,
        REL_HUMIDITY, ROUGHNESS_LENGTH, SAVI, SOIL_MOISTURE, SOLAR_RADIATION, SVF,
        TREE_FRAC, UI, WATER_FRAC, WIND_SPEED,
    )

    signs: dict[str, int] = {
        # --- cooling levers (driver up => cooler): - ---
        ALBEDO: -1,             # cut absorbed (1-alpha)K_down
        NDVI: -1, EVI: -1, SAVI: -1,   # vegetation -> evapotranspiration + shade
        LAI: -1, FVC: -1, ET: -1,      # canopy / latent cooling
        GREEN_FRAC: -1, TREE_FRAC: -1,  # vegetation fraction
        WATER_FRAC: -1, NDWI: -1, MNDWI: -1,  # open water -> Q_E + heat capacity
        EMISSIVITY: -1,         # radiates longwave away more efficiently
        SVF: -1,                # open sky -> longwave escapes (warming when LOW)
        WIND_SPEED: -1,         # ventilation -> Q_H export
        ROUGHNESS_LENGTH: -1,   # rougher -> more turbulent export
        SOIL_MOISTURE: -1,      # raises achievable Q_E
        REL_HUMIDITY: -1,       # damps daytime sensible heating (proxy)
        ELEVATION: -1,          # lapse-rate cooling with height
        # --- warming levers (driver up => hotter): + ---
        IMPERVIOUS_FRAC: +1,    # cuts Q_E, raises storage + Bowen ratio
        NDBI: +1, NDBAI: +1, UI: +1,   # built/bare spectral indices
        PLAN_AREA_FRAC: +1,     # lambda_P -> daytime trapping + storage
        FRONTAL_AREA_INDEX: +1,  # lambda_F (note: also raises z0; net + for LST)
        BUILDING_HEIGHT: +1, BUILDING_VOLUME: +1,  # thermal mass, low SVF (night)
        ASPECT_RATIO: +1,       # deep canyons trap longwave
        ANTHRO_HEAT: +1, NIGHTLIGHTS: +1, POPULATION: +1, NO2: +1,  # Q_F proxies
        SOLAR_RADIATION: +1,    # more incoming shortwave
        LONGWAVE_DOWN: +1,      # more incoming longwave
        AIR_TEMP: +1, DEWPOINT: +1,  # warmer/wetter air raises Ts
        NET_RADIATION: +1,      # more available energy
        AOD: 0,                 # ambiguous (dims K_down but warms atmosphere)
    }
    return signs


# ===========================================================================
# Simple physical intervention estimators (intervention cross-checks)
# ===========================================================================
def albedo_to_delta_lst(
    delta_albedo: "np.ndarray | float",
    k_down: "np.ndarray | float" = 800.0,
    emissivity: "np.ndarray | float" = 0.95,
    lst_c: "np.ndarray | float" = 35.0,
) -> "np.ndarray | float":
    """First-order surface ``dLST`` (degC) for an albedo change (cool roof/pavement).

    Linearizes the radiative law: raising albedo by ``delta_albedo`` cuts absorbed
    shortwave by ``dQ* = -delta_albedo * K_down``; equating to the change in
    emitted longwave ``d(eps sigma Ts^4) = 4 eps sigma Ts^3 dTs`` gives

        dTs = -delta_albedo * K_down / (4 eps sigma Ts^3)        [K = degC step]

    Returns a **negative** value for a positive ``delta_albedo`` (cooling), as a
    physical cross-check on the ML counterfactual (sign + order of magnitude). The
    radiative-only estimate is an upper bound; real near-surface cooling is smaller
    because part of ``dQ*`` is taken up by turbulent/storage adjustment.
    """
    sigma = _phys()["STEFAN_BOLTZMANN"]
    eps = np.asarray(emissivity, dtype=np.float64)
    ts_k = _c2k(lst_c)
    da = np.asarray(delta_albedo, dtype=np.float64)
    k = np.asarray(k_down, dtype=np.float64)
    d_ts = -da * k / (4.0 * eps * sigma * ts_k ** 3)
    return d_ts  # K step == degC step


def et_to_delta_lst(
    delta_et_mm_day: "np.ndarray | float",
    emissivity: "np.ndarray | float" = 0.95,
    lst_c: "np.ndarray | float" = 35.0,
) -> "np.ndarray | float":
    """First-order surface ``dLST`` (degC) for added evapotranspiration (greening/water).

    Extra ET of ``delta_et_mm_day`` (mm/day) is an extra latent-heat sink
    ``dQ_E = lambda_v * rho_w * (ET / 86400)`` W/m^2 (``lambda_v`` ~ 2.45e6 J/kg
    latent heat of vaporization, ``rho_w`` = 1000 kg/m^3). Removing that energy
    from emission, ``dTs = -dQ_E / (4 eps sigma Ts^3)``.

    Returns a **negative** value for positive added ET (evaporative cooling), the
    physical companion to :func:`albedo_to_delta_lst` for cross-checking the
    vegetation/water intervention counterfactuals.
    """
    sigma = _phys()["STEFAN_BOLTZMANN"]
    lambda_v = 2.45e6   # J/kg latent heat of vaporization (~35 degC)
    rho_w = 1000.0      # kg/m^3
    eps = np.asarray(emissivity, dtype=np.float64)
    ts_k = _c2k(lst_c)
    et = np.asarray(delta_et_mm_day, dtype=np.float64)
    d_qe = lambda_v * rho_w * (et / 1000.0) / 86400.0  # W/m^2 (mm/day -> m/s -> flux)
    d_ts = -d_qe / (4.0 * eps * sigma * ts_k ** 3)
    return d_ts


__all__ = [
    # radiative
    "longwave_up", "lst_from_longwave", "saturation_vapor_pressure",
    "shortwave_net", "net_radiation_arr", "net_radiation",
    # turbulent / storage fluxes
    "aerodynamic_resistance", "sensible_heat", "latent_heat", "bowen_ratio",
    "storage_heat_ohm", "OHM_COEFFICIENTS", "anthropogenic_heat_flux",
    # backbone + closure
    "physics_lst", "seb_residual",
    # physics<->ML bridge + intervention estimators
    "expected_lst_gradient_signs", "albedo_to_delta_lst", "et_to_delta_lst",
]
