"""urbanheat.gee — Google Earth Engine production backend (server-side "O(1)").

Builds a :class:`~urbanheat.datamodel.FeatureStack` from Earth Engine by filtering
collections, running ``ee.Image.expression`` band math server-side, and pulling
back only reduced samples/exports (the data never leaves Google). See the Module
Interface Contracts in ARCHITECTURE.md for the exact public signatures of
``auth``, ``collections``, ``lst``, ``lulc``, ``meteo``, ``morphology``,
``fusion``, ``features`` and ``source``.

All ``ee`` imports are lazy (inside functions) so this subpackage is only loaded
when ``mode='gee'``.
"""
