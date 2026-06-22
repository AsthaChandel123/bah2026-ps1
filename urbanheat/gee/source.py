"""urbanheat.gee.source — the Earth Engine production data backend.

:class:`GEEDataSource` is the production, "O(1)" :class:`~urbanheat.datamodel.DataSource`
implementation (ARCHITECTURE.md §11.2). Its :meth:`~GEEDataSource.load` runs the
whole server-side recipe — authenticate Earth Engine, assemble the multi-band
driver image (:func:`urbanheat.gee.features.build_feature_image`), then sample it
onto the analysis grid (:func:`urbanheat.gee.features.sample_to_featurestack`) —
and returns a :class:`~urbanheat.datamodel.FeatureStack` with **the same canonical
layers as the SyntheticDataSource**, so every downstream module (indices ->
hotspots -> ML -> attribution -> intervention sim -> optimization -> maps/report)
is identical regardless of backend.

All Earth Engine work is lazy: importing this module needs only numpy + the
catalogs, so a build that runs purely in synthetic mode never imports ``ee``. If
``ee`` is missing or authentication/initialisation fails, :meth:`load` raises a
clear, actionable error that points the user at ``mode='synthetic'`` (the
offline path that needs no credentials or network).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from urbanheat.datamodel import DataSource

if TYPE_CHECKING:  # hints only; never imported at runtime
    from urbanheat.config import Config
    from urbanheat.datamodel import FeatureStack


class GEEDataSource(DataSource):
    """Google Earth Engine production backend (server-side compute).

    Implements the :class:`~urbanheat.datamodel.DataSource` interface so it is a
    drop-in alternative to :class:`urbanheat.synthetic.source.SyntheticDataSource`.
    The heavy lifting (LST fusion, spectral/morphology/meteo band math, decadal
    composites, grid sampling) runs on Google's cluster; only the reduced grid of
    values crosses the wire into the returned FeatureStack (R7, ARCHITECTURE §4).
    """

    name = "gee"

    def load(self, config: "Config") -> "FeatureStack":
        """Production path: initialise Earth Engine, build & sample the driver stack.

        Steps:

        1. **Authenticate / initialise** Earth Engine for this process via
           :func:`urbanheat.gee.auth.initialize_ee` (idempotent; honours
           ``config.gee_project`` and ``config.use_highvolume``).
        2. **Assemble + sample** the full driver stack via
           :func:`urbanheat.gee.features.build_feature_stack`
           (``build_feature_image`` -> ``sample_to_featurestack``).

        Returns a :class:`~urbanheat.datamodel.FeatureStack` whose ``crs`` is
        ``config.target_crs`` and whose grid matches ``config`` resolution/extent,
        with at minimum the :data:`urbanheat.datamodel.LST` layer populated (and
        the rest of the canonical driver stack where the GEE products are
        available; unavailable layers are present but all-NaN — see
        ``meta['missing_layers']``).

        Raises
        ------
        ImportError
            If the Earth Engine API (``earthengine-api``) is not installed, with
            install/auth instructions and a suggestion to use ``mode='synthetic'``.
        RuntimeError
            If Earth Engine cannot be initialised (not authenticated, wrong/missing
            Cloud project, ...) or if no driver group could be assembled — wrapping
            the underlying error and pointing at the offline synthetic backend.
        """
        # Lazy imports: keep this module importable without ``ee`` installed.
        from urbanheat.gee.auth import initialize_ee  # noqa: PLC0415
        from urbanheat.gee.features import build_feature_stack  # noqa: PLC0415

        # 1. Authenticate / initialise Earth Engine.
        try:
            initialize_ee(config=config)
        except ImportError as exc:
            # ``ee`` not installed -> re-raise with the synthetic-mode hint.
            raise ImportError(
                f"{exc}\n\nGEEDataSource requires Earth Engine. For an offline "
                "demo/test that needs no credentials or network, set "
                "mode='synthetic' on your Config (the SyntheticDataSource "
                "produces the same FeatureStack schema)."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - classify & re-wrap below
            raise RuntimeError(
                "GEEDataSource.load: Earth Engine initialisation failed "
                f"({type(exc).__name__}: {exc}).\n"
                "  * Authenticate once with `earthengine authenticate`, and pass "
                "a registered Cloud project via Config(gee_project=...).\n"
                "  * Or run fully offline with mode='synthetic' (no credentials/"
                "network needed)."
            ) from exc

        # 2. Build the server-side feature image and sample it onto the grid.
        try:
            return build_feature_stack(config)
        except Exception as exc:  # noqa: BLE001 - actionable wrap
            raise RuntimeError(
                "GEEDataSource.load: building the FeatureStack from Earth Engine "
                f"failed ({type(exc).__name__}: {exc}).\n"
                "  * Check Earth Engine quota / dataset access for the AOI and "
                "date window.\n"
                "  * For an offline run, set mode='synthetic'."
            ) from exc


__all__ = ["GEEDataSource"]
