"""urbanheat.indices — spectral/heat-stress indices and hotspot statistics.

* ``heat_indices`` — LST-derived surface metrics (SUHII, UTFVI/EEI, z-score,
  percentile) and air-temperature comfort indices (Heat Index, Humidex,
  wet-bulb, WBGT, UTCI). [R8]
* ``hotspots`` — spatial clustering (Getis-Ord Gi*, local Moran's I) and the
  layered 5-class composite hotspot definition. [R8 §9-12]

All operate on a :class:`~urbanheat.datamodel.FeatureStack` and write canonical
derived layers back into it.
"""
