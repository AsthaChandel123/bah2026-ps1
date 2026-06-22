"""urbanheat.gee.auth — Earth Engine authentication & initialisation.

Three credential paths are supported, mirroring the GEE platform guidance in
``research/07_platform_oneapi_architecture.md`` §2.1:

* **OAuth** (interactive/dev): ``ee.Authenticate()`` (browser consent, cached
  locally) then ``ee.Initialize(project=...)``.
* **Service account** (headless/production: Streamlit, cron, CI):
  ``ee.ServiceAccountCredentials(email, key_file)`` then ``ee.Initialize``.
* **Application Default Credentials** (Cloud Run / GCE / Workload Identity):
  ``google.auth.default(scopes=[...])`` then ``ee.Initialize(creds, project=...)``.

When ``high_volume`` is set (the default) we initialise against the
**high-volume endpoint** ``https://earthengine-highvolume.googleapis.com`` which
is tuned for *many, parallel, lightweight* requests — the right target for the
chip-export / thumbnail / ``getInfo`` workloads this package drives. [R7 §2.1]

All ``ee`` (and ``google.auth``) imports are **lazy** (inside functions) so this
module imports with zero optional dependencies; importing it never requires the
``earthengine-api`` package to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # hints only; never imported at runtime
    from urbanheat.config import Config

# The Earth Engine high-volume endpoint (parallel-friendly). [R7 §2.1 / §2.5]
HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"

# OAuth scopes for the ADC path. [R7 §2.1 (C)]
_EE_SCOPES = (
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/cloud-platform",
)

# Process-level flag so initialize_ee() is idempotent and is_initialized() works
# even if a caller initialised ``ee`` directly elsewhere.
_INITIALIZED = False


def _import_ee() -> Any:
    """Lazy-import the Earth Engine API with a clear, actionable error.

    Raises
    ------
    ImportError
        If ``earthengine-api`` is not installed, with install instructions.
    """
    try:
        import ee  # noqa: PLC0415 (intentional lazy import)
    except ImportError as exc:  # pragma: no cover - exercised only without ee
        raise ImportError(
            "The Earth Engine API ('ee') is not installed, which is required "
            "for mode='gee'. Install it with:\n"
            "    pip install earthengine-api\n"
            "or install the project's GEE extra:\n"
            "    pip install 'urbanheat[gee]'\n"
            "Then authenticate once with `earthengine authenticate` (or call "
            "urbanheat.gee.auth.initialize_ee(...)). For an offline demo that "
            "needs no credentials, use mode='synthetic' instead."
        ) from exc
    return ee


def initialize_ee(
    config: "Config | None" = None,
    project: str | None = None,
    service_account: str | None = None,
    key_file: str | None = None,
    high_volume: bool = True,
) -> None:
    """Initialise Earth Engine for this process (idempotent).

    Resolves the GCP project and high-volume preference from ``config`` when an
    explicit argument is not given, then initialises ``ee`` via the most specific
    credential path available:

    1. **Service account** — if both ``service_account`` (email) and ``key_file``
       (path to the JSON key) are provided, use
       :class:`ee.ServiceAccountCredentials`.
    2. **OAuth** — otherwise try ``ee.Initialize`` directly; if that fails for
       lack of credentials, run ``ee.Authenticate()`` (interactive browser
       consent, cached locally) and retry.
    3. **ADC fallback** — if OAuth credentials are unavailable in a headless
       context, fall back to :func:`google.auth.default` (Application Default
       Credentials) with the Earth Engine + cloud-platform scopes.

    Parameters
    ----------
    config : Config | None
        Optional run config; supplies ``gee_project`` and ``use_highvolume``
        defaults when the corresponding arguments are left unset.
    project : str | None
        Google Cloud project id for ``ee.Initialize`` (required by the current
        Earth Engine API). Falls back to ``config.gee_project``.
    service_account : str | None
        Service-account email for the headless/production path.
    key_file : str | None
        Path to the service-account JSON key (pairs with ``service_account``).
    high_volume : bool
        Target the high-volume endpoint (default True). Falls back to
        ``config.use_highvolume`` when ``config`` is given and this is left True.

    Raises
    ------
    ImportError
        If ``earthengine-api`` (or, for the ADC path, ``google-auth``) is not
        installed — with instructions on how to install/authenticate.
    RuntimeError
        If Earth Engine cannot be initialised (e.g. not authenticated, wrong or
        missing project) — wrapping the underlying error with a fix hint.
    """
    global _INITIALIZED

    ee = _import_ee()

    # Resolve project / high-volume from config when not explicitly provided.
    if config is not None:
        if project is None:
            project = getattr(config, "gee_project", None)
        # Only let config *disable* high-volume; an explicit high_volume=False
        # from the caller is always respected because it is the argument default
        # we cannot distinguish — config acts as the source of truth here.
        if high_volume:
            high_volume = bool(getattr(config, "use_highvolume", True))

    opt_url = HIGH_VOLUME_URL if high_volume else None

    def _init(credentials: Any = None) -> None:
        kwargs: dict[str, Any] = {}
        if credentials is not None:
            kwargs["credentials"] = credentials
        if project is not None:
            kwargs["project"] = project
        if opt_url is not None:
            kwargs["opt_url"] = opt_url
        ee.Initialize(**kwargs)

    try:
        if service_account and key_file:
            # (B) Headless / production service-account key. [R7 §2.1 (B)]
            credentials = ee.ServiceAccountCredentials(service_account, key_file)
            _init(credentials)
        else:
            # (A) OAuth interactive path; (C) ADC as a headless fallback.
            try:
                _init()
            except Exception:  # noqa: BLE001 - broad on purpose; classify below
                if not _try_oauth_then_init(ee, _init):
                    _try_adc_then_init(ee, _init, project, opt_url)
    except ImportError:
        raise
    except Exception as exc:  # noqa: BLE001 - wrap with actionable guidance
        raise RuntimeError(
            "Earth Engine initialisation failed: "
            f"{type(exc).__name__}: {exc}\n"
            "Common fixes:\n"
            "  * Authenticate once:  earthengine authenticate\n"
            "  * Pass a Cloud project that is registered for Earth Engine "
            "(project=... or Config.gee_project).\n"
            "  * For headless/CI use a service account: "
            "initialize_ee(service_account='svc@proj.iam.gserviceaccount.com', "
            "key_file='/path/key.json', project='proj').\n"
            "  * Verify access at https://code.earthengine.google.com/ ."
        ) from exc

    _INITIALIZED = True


def _try_oauth_then_init(ee: Any, init_fn: Any) -> bool:
    """Run ``ee.Authenticate()`` then retry init. Return True on success.

    Returns False (rather than raising) if interactive authentication is not
    possible in this environment, so the caller can fall through to ADC.
    """
    try:
        ee.Authenticate()
    except Exception:  # noqa: BLE001 - headless: no browser / no consent flow
        return False
    init_fn()
    return True


def _try_adc_then_init(
    ee: Any, init_fn: Any, project: str | None, opt_url: str | None
) -> None:
    """Initialise via Application Default Credentials. [R7 §2.1 (C)]"""
    try:
        import google.auth  # noqa: PLC0415 (lazy)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Application Default Credentials path needs 'google-auth'. Install "
            "it with `pip install google-auth`, or authenticate interactively "
            "with `earthengine authenticate`."
        ) from exc
    creds, adc_project = google.auth.default(scopes=list(_EE_SCOPES))
    kwargs: dict[str, Any] = {"credentials": creds}
    kwargs["project"] = project or adc_project
    if opt_url is not None:
        kwargs["opt_url"] = opt_url
    ee.Initialize(**kwargs)


def is_initialized() -> bool:
    """Return True if Earth Engine has been initialised in this process.

    Best-effort: prefers the API's own ``ee.data._initialized`` flag when the
    package is importable, otherwise falls back to this module's tracking flag.
    Never raises and never imports ``ee`` if it is not installed.
    """
    try:
        import ee  # noqa: PLC0415 (lazy)
    except ImportError:
        return False
    flag = getattr(getattr(ee, "data", None), "_initialized", None)
    if isinstance(flag, bool):
        return flag or _INITIALIZED
    return _INITIALIZED


def ee_geometry(bbox: tuple[float, float, float, float]) -> Any:
    """Return an ``ee.Geometry.Rectangle`` for an EPSG:4326 ``bbox``.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        ``(xmin, ymin, xmax, ymax)`` in lon/lat degrees (EPSG:4326).
    """
    ee = _import_ee()
    xmin, ymin, xmax, ymax = (float(v) for v in bbox)
    # geodesic=False -> planar rectangle in the given CRS (here lon/lat). [R7 §2.3]
    return ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], proj="EPSG:4326",
                                 geodesic=False)


# Backwards/contract alias: ARCHITECTURE.md §11 names this entrypoint
# ``initialize``; the builder spec names it ``initialize_ee``. Expose both so
# either contract resolves to the same implementation.
def initialize(
    project: str | None = None,
    high_volume: bool = True,
    service_account: str | None = None,
    key_file: str | None = None,
) -> None:
    """Alias of :func:`initialize_ee` matching the ARCHITECTURE.md §11 signature."""
    initialize_ee(
        config=None,
        project=project,
        service_account=service_account,
        key_file=key_file,
        high_volume=high_volume,
    )


__all__ = [
    "initialize_ee",
    "initialize",
    "is_initialized",
    "ee_geometry",
    "HIGH_VOLUME_URL",
]
