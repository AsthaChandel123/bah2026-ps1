"""Tests for :mod:`urbanheat.constants` — the single-source-of-truth catalog.

Guards (ARCHITECTURE §5, §7, §9, §11.0):
  * ``GEE_DATASETS`` integrity — every entry has an ``id`` and a ``bands`` list,
    plus the scale/offset/role schema fields.
  * ``BAND_SCALE_OVERRIDES`` keys reference real datasets.
  * ``INTERVENTION_PARAMS`` — 9 levers, each with the SEB ``perturbs`` mapping
    and surface/air/Tmrt °C ranges used by the simulator/optimizer.
  * ``HOTSPOT_LEGEND`` has exactly the 5 PS-1 priority classes.
  * ``LCZ_TABLE`` has 17 Local-Climate-Zone rows.
  * Physical constants + validation/robustness panels are present.
"""

from __future__ import annotations

import math

import pytest

from urbanheat import constants as C


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
def test_physical_constants_present_and_sane() -> None:
    """Stefan-Boltzmann, Kelvin offset and friends exist with sane magnitudes."""
    pc = C.PHYSICAL_CONSTANTS
    for key in ("STEFAN_BOLTZMANN", "VON_KARMAN", "KELVIN_OFFSET", "CP_AIR",
                "RHO_AIR", "PSYCHROMETRIC_GAMMA"):
        assert key in pc, f"missing physical constant {key!r}"
    assert pc["KELVIN_OFFSET"] == pytest.approx(273.15)
    assert pc["STEFAN_BOLTZMANN"] == pytest.approx(5.670374419e-8, rel=1e-6)
    # Top-level aliases agree with the dict.
    assert C.SIGMA_SB == pc["STEFAN_BOLTZMANN"]
    assert C.KELVIN == pc["KELVIN_OFFSET"]
    assert C.VON_KARMAN == pc["VON_KARMAN"]


# ---------------------------------------------------------------------------
# GEE dataset catalog
# ---------------------------------------------------------------------------
def test_gee_datasets_nonempty() -> None:
    """The catalog must hold many cross-verifying sources (>=30 per the design)."""
    assert isinstance(C.GEE_DATASETS, dict)
    assert len(C.GEE_DATASETS) >= 30


def test_gee_datasets_schema_integrity() -> None:
    """Every dataset entry carries id, bands and the scale/offset/role schema."""
    required = {"id", "bands", "scale", "offset", "units", "role", "note"}
    valid_roles = {"primary", "secondary", "fusion", "reference", "prior"}
    for key, entry in C.GEE_DATASETS.items():
        assert isinstance(entry, dict), f"{key} entry is not a dict"
        missing = required - set(entry)
        assert not missing, f"{key} missing schema fields {missing}"
        assert isinstance(entry["id"], str), f"{key}: id must be a string"
        assert isinstance(entry["bands"], list), f"{key}: bands must be a list"
        assert isinstance(entry["scale"], (int, float)), f"{key}: scale must be numeric"
        assert isinstance(entry["offset"], (int, float)), f"{key}: offset must be numeric"
        assert entry["role"] in valid_roles, f"{key}: bad role {entry['role']!r}"


def test_key_lst_datasets_have_expected_ids() -> None:
    """Spot-check a few headline dataset IDs/bands the GEE backend depends on."""
    assert C.GEE_DATASETS["LANDSAT8_L2"]["id"] == "LANDSAT/LC08/C02/T1_L2"
    assert "ST_B10" in C.GEE_DATASETS["LANDSAT8_L2"]["bands"]
    assert C.GEE_DATASETS["MODIS_MOD11A1"]["id"] == "MODIS/061/MOD11A1"
    assert "LST_Day_1km" in C.GEE_DATASETS["MODIS_MOD11A1"]["bands"]
    # Landsat ST scale/offset -> K convention.
    assert C.GEE_DATASETS["LANDSAT8_L2"]["scale"] == pytest.approx(0.00341802)
    assert C.GEE_DATASETS["LANDSAT8_L2"]["offset"] == pytest.approx(149.0)


def test_band_scale_overrides_reference_real_datasets() -> None:
    """Override keys must point at datasets that exist and carry (scale, offset) tuples."""
    for key, bands in C.BAND_SCALE_OVERRIDES.items():
        assert key in C.GEE_DATASETS, f"override for unknown dataset {key!r}"
        for band, so in bands.items():
            assert len(so) == 2, f"{key}/{band}: expected (scale, offset)"
            assert all(isinstance(x, (int, float)) for x in so)


def test_external_sources_present() -> None:
    """Non-GEE ingest sources (ECOSTRESS-India, INSAT, CPCB, ...) are catalogued."""
    assert "INSAT_3D_LST" in C.EXTERNAL_SOURCES
    assert "ECOSTRESS_INDIA" in C.EXTERNAL_SOURCES


# ---------------------------------------------------------------------------
# Spectral index coefficients
# ---------------------------------------------------------------------------
def test_spectral_index_coeffs() -> None:
    """Albedo / emissivity / EVI coefficient sets exist with the right keys."""
    assert "const" in C.SPECTRAL_INDEX_COEFFS["ALBEDO_LANDSAT_LIANG"]
    evi = C.SPECTRAL_INDEX_COEFFS["EVI"]
    for k in ("G", "C1", "C2", "L"):
        assert k in evi
    emis = C.SPECTRAL_INDEX_COEFFS["EMISSIVITY_NDVI"]
    assert emis["eps_veg"] > emis["eps_soil"], "vegetation must be more emissive than soil"


# ---------------------------------------------------------------------------
# Hotspot classification + legends
# ---------------------------------------------------------------------------
def test_hotspot_legend_is_five_classes() -> None:
    """The deliverable priority legend has exactly the 5 PS-1 classes, ordered 0-100."""
    legend = C.HOTSPOT_LEGEND
    assert len(legend) == 5, "PS-1 priority map must be 5-class"
    names = [c["name"] for c in legend]
    assert names == ["Low", "Moderate", "High", "Severe", "Extreme"]
    # Ranges must tile [0, 100] contiguously and each carry a hex colour.
    assert legend[0]["min"] == 0 and legend[-1]["max"] == 100
    for prev, cur in zip(legend, legend[1:]):
        assert prev["max"] == cur["min"], "legend ranges must be contiguous"
    for c in legend:
        assert isinstance(c["hex"], str) and c["hex"].startswith("#")


def test_lst_color_ramp() -> None:
    """The pure-surface LST ramp is a non-empty list of hex colours."""
    assert len(C.LST_COLOR_RAMP) >= 3
    assert all(h.startswith("#") for h in C.LST_COLOR_RAMP)


def test_utfvi_classes_monotone() -> None:
    """UTFVI->EEI reclass boundaries are ascending and end at +inf."""
    maxes = [c["max"] for c in C.UTFVI_CLASSES]
    assert maxes == sorted(maxes), "UTFVI class boundaries must ascend"
    assert math.isinf(maxes[-1])
    assert {"uhi", "eei"} <= set(C.UTFVI_CLASSES[0])


def test_gistar_z_thresholds() -> None:
    """Getis-Ord Gi* significance levels match the documented two-tailed z table."""
    assert C.HOTSPOT_GISTAR_Z["p95"] == pytest.approx(1.96, abs=0.01)
    assert C.HOTSPOT_GISTAR_Z["p99"] == pytest.approx(2.58, abs=0.01)


def test_heat_stress_thresholds() -> None:
    """Wet-bulb danger ladder is present and physically ordered (28<31<35 degC)."""
    wb = C.HEAT_STRESS_THRESHOLDS["wet_bulb"]
    assert wb["danger"] < wb["extreme"] < wb["lethal"]
    assert wb["lethal"] == pytest.approx(35.0)


# ---------------------------------------------------------------------------
# LCZ table
# ---------------------------------------------------------------------------
def test_lcz_table_has_17_rows() -> None:
    """Stewart & Oke LCZ table: 17 classes (1-10 built, 11-17 = A-G natural)."""
    assert len(C.LCZ_TABLE) == 17
    assert set(C.LCZ_TABLE) == set(range(1, 18))


def test_lcz_table_schema() -> None:
    """Each LCZ row carries the morphology priors used to seed SVF/H-W/lambda_P."""
    for code, row in C.LCZ_TABLE.items():
        for field in ("name", "svf", "hw", "bsf", "isf", "height", "is_built"):
            assert field in row, f"LCZ {code} missing {field!r}"
        assert 0.0 <= row["svf"] <= 1.0, f"LCZ {code}: svf out of range"
    # The rural reference class exists and is a non-built class.
    assert C.LCZ_RURAL_REFERENCE in C.LCZ_TABLE
    assert C.LCZ_TABLE[C.LCZ_RURAL_REFERENCE]["is_built"] is False


def test_macdonald_constants() -> None:
    """Macdonald (1998) roughness constants present for z0/zd derivation."""
    for k in ("A", "Cd", "beta"):
        assert k in C.MACDONALD_CONSTANTS


# ---------------------------------------------------------------------------
# Intervention parameters
# ---------------------------------------------------------------------------
def test_intervention_params_count_and_core_levers() -> None:
    """Catalog holds the 9 cooling levers incl. the headline greening/roof ones."""
    params = C.INTERVENTION_PARAMS
    assert len(params) == 9, "PS-1 catalog is 9 intervention types"
    for lever in ("urban_trees", "cool_roof", "green_roof", "water_body", "urban_park"):
        assert lever in params, f"missing intervention {lever!r}"


def test_intervention_params_schema() -> None:
    """Every lever carries perturbs + surface/air/tmrt dC ranges + mechanism."""
    for name, p in C.INTERVENTION_PARAMS.items():
        for field in ("mechanism", "surface_dC", "air_dC", "tmrt_dC",
                      "perturbs", "feasibility"):
            assert field in p, f"{name} missing {field!r}"
        for rng_field in ("surface_dC", "air_dC", "tmrt_dC"):
            lo, hi = p[rng_field]
            assert lo <= hi, f"{name}.{rng_field} range not ordered"
        assert isinstance(p["perturbs"], dict) and p["perturbs"], \
            f"{name} must perturb at least one driver"


def test_intervention_perturbs_use_canonical_or_known_keys() -> None:
    """Perturbed driver keys are canonical FeatureStack names (or documented extras).

    The simulator applies ``perturbs`` to FeatureStack layers, so the keys must be
    canonical variable names. A small set of non-layer micro-levers ('shade') is
    explicitly allowed per the catalog notes.
    """
    from urbanheat.datamodel import ALL_VARIABLES
    allowed_extra = {"shade"}
    known = set(ALL_VARIABLES) | allowed_extra
    for name, p in C.INTERVENTION_PARAMS.items():
        for var in p["perturbs"]:
            assert var in known, f"{name}: perturb key {var!r} is not canonical"


def test_greening_perturbs_increase_vegetation() -> None:
    """Greening levers raise NDVI/tree/green (so counterfactuals cool, not warm)."""
    trees = C.INTERVENTION_PARAMS["urban_trees"]["perturbs"]
    assert trees.get("ndvi", 0) > 0
    assert trees.get("tree_frac", 0) > 0
    cool_roof = C.INTERVENTION_PARAMS["cool_roof"]["perturbs"]
    assert cool_roof.get("albedo", 0) > 0, "cool roof must raise albedo"


def test_invest_ucm_and_submodular_bound() -> None:
    """InVEST UCM weights sum to 1 and the greedy bound is (1 - 1/e)."""
    ucm = C.INVEST_UCM
    w = ucm["cc_weight_shade"] + ucm["cc_weight_albedo"] + ucm["cc_weight_eti"]
    assert w == pytest.approx(1.0)
    assert C.SUBMODULAR_GREEDY_BOUND == pytest.approx(1 - 1 / math.e, abs=1e-3)


# ---------------------------------------------------------------------------
# HVI + validation panels
# ---------------------------------------------------------------------------
def test_hvi_domains_signs() -> None:
    """HVI three-domain structure: adaptive capacity is sign-inverted (-1)."""
    d = C.HVI_DOMAINS
    assert d["exposure"]["sign"] == +1
    assert d["sensitivity"]["sign"] == +1
    assert d["adaptive_capacity"]["sign"] == -1
    assert C.HVI_DOMAIN_WEIGHTS  # weights present
    assert sum(C.HVI_DOMAIN_WEIGHTS.values()) == pytest.approx(1.0)


def test_validation_metric_panel() -> None:
    """The spatial-CV metric panel names the documented 8 metrics."""
    assert set(C.VALIDATION_METRICS) >= {"rmse", "mae", "bias", "r2"}
    assert "extra_trees_lst_r2" in C.VALIDATION_ANCHORS


def test_robustness_summary_counts() -> None:
    """The >=30-method accounting surfaces the headline source counts."""
    rs = C.ROBUSTNESS_SUMMARY
    assert rs["lst_sensors"] >= 5
    assert rs["total_matrix_entries"] >= 30
