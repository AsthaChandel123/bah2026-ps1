"""Tests for :mod:`urbanheat.config` — typed run config + Indian city presets.

Guards (ARCHITECTURE §11.0):
  * ``Config()`` zero-arg defaults (synthetic, Delhi, valid bbox/EPSG).
  * ``Config.from_city`` for all five Indian city presets, case-insensitively,
    each producing a valid lon/lat bbox and a UTM EPSG.
  * round-trip ``to_dict`` / ``from_dict`` and ``is_dataset_enabled``.
  * bbox + mode validation in ``__post_init__``.
"""

from __future__ import annotations

import pytest

from urbanheat.config import (
    CITY_PRESETS,
    DEFAULT_CITY,
    Config,
)

EXPECTED_CITIES = {"Delhi", "Mumbai", "Hyderabad", "Ahmedabad", "Bengaluru"}


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
def test_five_indian_city_presets_exist() -> None:
    """All five PS-1 India-first city presets are defined."""
    assert EXPECTED_CITIES <= set(CITY_PRESETS)
    assert DEFAULT_CITY in CITY_PRESETS


@pytest.mark.parametrize("city", sorted(EXPECTED_CITIES))
def test_preset_bbox_and_epsg_valid(city: str) -> None:
    """Each preset has an ordered lon/lat bbox in India + a UTM EPSG code."""
    preset = CITY_PRESETS[city]
    xmin, ymin, xmax, ymax = preset["bbox"]
    assert xmin < xmax and ymin < ymax, f"{city}: bbox not ordered"
    # Inside the Indian longitude/latitude envelope.
    assert 68.0 <= xmin <= 98.0 and 68.0 <= xmax <= 98.0, f"{city}: lon out of India"
    assert 6.0 <= ymin <= 38.0 and 6.0 <= ymax <= 38.0, f"{city}: lat out of India"
    epsg = preset["utm_epsg"]
    assert epsg.startswith("EPSG:"), f"{city}: utm_epsg not an EPSG string"
    # Indian mainland UTM zones are 43N/44N/45N.
    assert epsg in {"EPSG:32643", "EPSG:32644", "EPSG:32645"}, f"{city}: {epsg}"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def test_zero_arg_defaults() -> None:
    """Config() runs with zero args in synthetic mode (Delhi, 100 m)."""
    cfg = Config()
    assert cfg.mode == "synthetic"
    assert cfg.city == DEFAULT_CITY
    assert cfg.resolution_m == 100.0
    assert cfg.target_crs.startswith("EPSG:")
    xmin, ymin, xmax, ymax = cfg.bbox
    assert xmin < xmax and ymin < ymax
    # Optimizer defaults are in sensible ranges.
    assert cfg.optimizer_budget > 0
    assert 0.0 < cfg.optimizer_max_area_frac <= 1.0
    assert cfg.optimizer_method in {"greedy", "ilp", "nsga2"}


def test_bbox_coerced_to_float_tuple() -> None:
    """__post_init__ coerces the bbox into a float tuple."""
    cfg = Config(bbox=(72, 18, 73, 19))  # ints
    assert all(isinstance(x, float) for x in cfg.bbox)


# ---------------------------------------------------------------------------
# from_city
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(EXPECTED_CITIES))
def test_from_city_each_preset(name: str) -> None:
    """from_city wires the preset bbox + UTM EPSG into the Config."""
    cfg = Config.from_city(name)
    assert cfg.city == name
    assert cfg.bbox == tuple(float(x) for x in CITY_PRESETS[name]["bbox"])
    assert cfg.target_crs == CITY_PRESETS[name]["utm_epsg"]


def test_from_city_is_case_insensitive() -> None:
    """City matching ignores case/whitespace."""
    a = Config.from_city("mumbai")
    b = Config.from_city("  MUMBAI ")
    assert a.city == b.city == "Mumbai"


def test_from_city_overrides() -> None:
    """Extra kwargs override other Config fields (e.g. mode, resolution)."""
    cfg = Config.from_city("Hyderabad", mode="gee", resolution_m=30.0,
                           gee_project="proj-x")
    assert cfg.city == "Hyderabad"
    assert cfg.mode == "gee"
    assert cfg.resolution_m == 30.0
    assert cfg.gee_project == "proj-x"


def test_from_city_unknown_raises() -> None:
    """An unknown city name raises KeyError listing valid cities."""
    with pytest.raises(KeyError):
        Config.from_city("Atlantis")


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------
def test_to_dict_from_dict_roundtrip() -> None:
    """to_dict / from_dict round-trips the key identifying fields."""
    cfg = Config.from_city("Ahmedabad", resolution_m=50.0, seed=7)
    d = cfg.to_dict()
    assert isinstance(d, dict)
    cfg2 = Config.from_dict(d)
    assert cfg2.city == cfg.city
    assert cfg2.bbox == cfg.bbox
    assert cfg2.target_crs == cfg.target_crs
    assert cfg2.resolution_m == 50.0
    assert cfg2.seed == 7


def test_from_dict_ignores_unknown_keys() -> None:
    """from_dict tolerates extra keys (forward/backward compatibility)."""
    cfg = Config.from_dict({"city": "Delhi", "mode": "synthetic",
                            "not_a_field": 123})
    assert cfg.city == "Delhi"


def test_from_dict_tuples_bbox_and_grid_shape() -> None:
    """Lists are coerced to tuples for bbox and grid_shape."""
    cfg = Config.from_dict({"bbox": [72.0, 18.0, 73.0, 19.0],
                            "grid_shape": [10, 12]})
    assert cfg.bbox == (72.0, 18.0, 73.0, 19.0)
    assert cfg.grid_shape == (10, 12)


# ---------------------------------------------------------------------------
# Dataset toggles + validation
# ---------------------------------------------------------------------------
def test_is_dataset_enabled_default_true() -> None:
    """Absent dataset key => enabled; explicit False => disabled."""
    cfg = Config(datasets={"MODIS_MOD11A1": False})
    assert cfg.is_dataset_enabled("LANDSAT8_L2") is True   # absent -> on
    assert cfg.is_dataset_enabled("MODIS_MOD11A1") is False


def test_invalid_mode_raises() -> None:
    """A mode outside {synthetic, gee} is rejected."""
    with pytest.raises(ValueError):
        Config(mode="quantum")  # type: ignore[arg-type]


def test_invalid_bbox_raises() -> None:
    """A bbox with xmin>=xmax (or ymin>=ymax) is rejected."""
    with pytest.raises(ValueError):
        Config(bbox=(77.0, 28.0, 76.0, 29.0))  # xmin > xmax
