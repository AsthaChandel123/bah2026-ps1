"""urbanheat — physics-informed, multi-satellite geospatial AI/ML for urban heat.

ISRO Bharatiya Antariksh Hackathon 2026, Problem Statement 1.

Maps urban heat-stress hotspots, quantifies drivers of urban heating, models
LST<->drivers with physics-informed ML, and simulates + optimizes cooling
interventions (greening, cool roofs, albedo, water bodies) with per-intervention
degC reduction and spatial placement.

The package is **source-agnostic**: a single :class:`~urbanheat.datamodel.DataSource`
interface has two interchangeable backends —

* ``GEEDataSource`` (production, server-side Earth Engine "O(1)" compute), and
* ``SyntheticDataSource`` (offline demo + tests; no GEE credentials/network),

both returning a :class:`~urbanheat.datamodel.FeatureStack` that the entire
downstream pipeline consumes identically.

Only lightweight foundation symbols are eagerly exported here; heavy/optional
dependencies (``ee``, ``torch``, ``natcap.invest`` ...) are imported lazily
inside the functions that need them, so importing :mod:`urbanheat` and running
the synthetic path requires only numpy.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Foundation symbols (import-clean, no optional deps).
from urbanheat.config import Config, CITY_PRESETS
from urbanheat.datamodel import FeatureStack, DataSource

__all__ = [
    "__version__",
    "Config",
    "CITY_PRESETS",
    "FeatureStack",
    "DataSource",
    "get_data_source",
]


def get_data_source(config: "Config") -> "DataSource":
    """Factory: return the backend named by ``config.mode``.

    ``'synthetic'`` -> :class:`urbanheat.synthetic.source.SyntheticDataSource`;
    ``'gee'``       -> :class:`urbanheat.gee.source.GEEDataSource`.

    Imports the chosen backend lazily so that selecting the synthetic path never
    requires the Earth Engine client to be installed (and vice-versa). This is
    the single entry point the CLI / app / pipeline use to obtain data.
    """
    if config.mode == "synthetic":
        from urbanheat.synthetic.source import SyntheticDataSource
        return SyntheticDataSource()
    if config.mode == "gee":
        from urbanheat.gee.source import GEEDataSource
        return GEEDataSource()
    raise ValueError(f"unknown mode {config.mode!r}; expected 'synthetic' or 'gee'")
