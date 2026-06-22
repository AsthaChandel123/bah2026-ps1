# 07 — Platform & Cloud-Native Geospatial Architecture: The "O(1) / Fastest Platform" Compute Backbone

**Research agent:** R7 — Platform & Cloud-Native Geospatial Architecture
**Project:** ISRO Bharatiya Antariksh Hackathon 2026, PS-1 — Physics-informed geospatial AI/ML for urban heat hotspots, driver attribution, LST modelling, and cooling optimisation.
**Date:** 2026-06-22
**Status of API claims:** Verified against vendor docs via web where marked ✅; items marked *(from-knowledge — verify)* rely on expert knowledge where the live fetch was thin.

---

## 0. TL;DR — Recommended Architecture

> **Primary compute engine = Google Earth Engine (GEE).** All heavy raster work (LST retrieval, energy-balance band math, decadal composites, zonal statistics over thousands of wards) runs **server-side on Google's cluster**. The client (our laptop / Streamlit server) only ever sends a *recipe* (a graph of operations) and receives back *small results* — a few thousand numbers, a PNG tile, or a CSV of ward-level aggregates. Client effort and network transfer are therefore **near-constant regardless of how large the area of interest is** — this is the "O(1)" framing.
>
> **Orchestration** in Python via the `ee` API + `geemap`. **ML** on *sampled feature tables* (small `pandas`/`GeoDataFrame` pulled from `sampleRegions`/`reduceRegions`) using `scikit-learn`/`PyTorch`. **Serving** via Streamlit (or FastAPI + a MapLibre/deck.gl front-end). **Portability / fallback** via a STAC + `stackstac`/`odc-stac` + Dask path on **Microsoft Planetary Computer** and **AWS Open Data** — used both as a vendor-independence hedge and for sensors not (fully) in GEE, e.g. **ECOSTRESS**, whose GEE collection currently only holds Los-Angeles tiles.

```
ee.Image.expression(energy-balance) → server-side → reduceRegions(wards) → tiny CSV → sklearn/torch → Streamlit map
```

---

## 1. The "O(1) / Fastest Platform" Argument

### 1.1 What "O(1)" actually means here (and what it does not)

The stakeholder priority is the **fastest platform / O(1) technique**: server-side, planetary-scale compute so **we never download terabytes**. To be precise and defensible in front of judges, we define the claim in terms of **client-side cost**, not total FLOPs:

- **Total work is not O(1).** Computing LST over Delhi-NCR still touches millions of pixels — *somebody* pays for that. The point is *who* pays and *what crosses the wire*.
- **Client work and data egress are ≈ O(1).** With GEE you build a **lazy computation graph** on the client. Nothing executes locally. You submit the graph; Google **fans it out across its data-centre cluster** (the imagery is co-located with the compute), and returns only the **reduced** result you asked for — a tile, a thumbnail, or an aggregate table.
- Concretely: a `reduceRegion` over a 10,000 km² city returns **one dictionary of band means**. Whether the AOI is 1 km² or 100,000 km², the **client receives the same handful of numbers** and writes the **same few lines of code**. That is the operational meaning of "O(1) client effort." Doubling the area roughly doubles *Google's* internal work and your **EECU-time** bill, but **not** your code, your RAM, or your download.

This reframing is important: it is honest, and it is exactly why the approach is the *fastest* for a hackathon team — **zero data-engineering, zero storage provisioning, zero cluster ops.**

### 1.2 Why this is the fastest *and* most robust choice for PS-1

| Property | Why GEE-first wins |
|---|---|
| **Time-to-first-map** | A working LST map over any Indian city in tens of lines, minutes not days. No download, no mosaicking, no reprojection plumbing. |
| **Scale invariance** | Same code for one ward or an entire state. Judges can ask "now do all of India" and the code does not change. |
| **Data gravity** | Petabytes of Landsat/Sentinel/MODIS/ERA5/VIIRS/population/land-cover are **already in the catalogue**, co-located with compute. No ingestion. |
| **Parallelism for free** | Google auto-parallelises the graph (tile-wise, image-wise). We never write Dask/Spark for the heavy path. |
| **Reproducibility** | The recipe *is* the algorithm. Share a script; anyone re-runs identical server-side math. |
| **Physics server-side** | Energy-balance and emissivity formulas execute as `ee.Image.expression` on the cluster — the physics-informed core runs at planetary scale without us touching arrays. |

### 1.3 The cost we are trading away (and the mitigation)

GEE is a **black-box managed service** with **quotas** and **vendor lock-in**. Mitigation = a **STAC/COG fallback** (Section 4–5) that runs the *same conceptual pipeline* on open data + open libraries, giving portability and covering catalogue gaps. This dual design is the defensible, "robust" answer.

---

## 2. Google Earth Engine — Deep Dive

### 2.1 Authentication & initialisation ✅

Two paths. **OAuth** for interactive/dev; **service account** for headless/production (a Streamlit deployment, a cron job, CI).

```python
import ee

# --- (A) Interactive / notebook: OAuth (browser consent, cached locally) ---
ee.Authenticate()                       # one-time; re-run with force=True to refresh
ee.Initialize(project='my-gcp-project') # project REQUIRED in current API

# --- (B) Headless / production: service-account key (JSON) ---
SA = 'gee-runner@my-gcp-project.iam.gserviceaccount.com'
KEY = '/secrets/gee-key.json'
credentials = ee.ServiceAccountCredentials(SA, KEY)
ee.Initialize(credentials, project='my-gcp-project')

# --- (C) Application Default Credentials (Cloud Run / GCE / Workload Identity) ---
import google.auth
creds, proj = google.auth.default(
    scopes=['https://www.googleapis.com/auth/earthengine',
            'https://www.googleapis.com/auth/cloud-platform'])
ee.Initialize(creds, project=proj)
```

**High-volume endpoint** ✅ — for *many* small automated requests (thumbnail tiles, chip downloads, parallel `getInfo` from a Dask/thread pool), initialise against the high-throughput endpoint instead of the interactive one:

```python
ee.Initialize(credentials, project='my-gcp-project',
              opt_url='https://earthengine-highvolume.googleapis.com')
```

> The interactive endpoint is tuned for *few, heavy, low-latency* calls; the **high-volume** endpoint (`earthengine-highvolume.googleapis.com`) is tuned for *many, parallel, lightweight* calls and is what `xee`, `geedim`, tile servers, and chip-export loops should target. ✅

### 2.2 The mental model: lazy server-side objects

Every `ee.*` object (`ee.Image`, `ee.ImageCollection`, `ee.Feature`, `ee.FeatureCollection`, `ee.Number`, `ee.Reducer`, …) is a **handle to a computation on Google's servers**, *not* data in your RAM. Operations **compose a graph**; nothing runs until you call a **terminal** that pulls a result across the wire:

- `getInfo()` → returns the computed value as Python (use **sparingly**, never in a loop).
- `getThumbURL()` / `getDownloadURL()` → a URL to a rendered PNG / GeoTIFF (≤ **32 MB** per request). ✅
- `Export.*` → a **batch** job (Drive/Asset/Cloud Storage) for big outputs.
- map tiles requested by `geemap`/`folium` as you pan/zoom.

**The single most important performance rule:** keep everything inside the graph (`.map()`, reducers, `ee.Algorithms`) and **only cross the wire once, with an already-reduced result.**

### 2.3 Core patterns

**ImageCollection filter → map → reduce** (the canonical "stay server-side" idiom):

```python
aoi = ee.Geometry.Rectangle([77.0, 28.4, 77.4, 28.8])   # Delhi-ish bbox

col = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
       .filterBounds(aoi)
       .filterDate('2023-04-01', '2023-06-30')           # pre-monsoon heat window
       .filter(ee.Filter.lt('CLOUD_COVER', 30)))
```

**`ee.Image.expression` for physics-informed band math** — energy-balance / index formulas run on the cluster (see full LST snippet in 2.6).

**`reduceRegion`** — one aggregate over one geometry (e.g. mean LST over the AOI):

```python
mean_lst = image.select('LST').reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=aoi, scale=30, maxPixels=1e13)
```

**`reduceRegions`** — *zonal statistics over many polygons at once* (wards, grid cells). This is the workhorse for "quantify drivers per administrative unit" — it returns a **FeatureCollection**, computed entirely server-side:

```python
wards = ee.FeatureCollection('projects/my-gcp-project/assets/delhi_wards')
zonal = image.reduceRegions(
    collection=wards,
    reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
    scale=30)
```

**`sampleRegions`** — extract a **per-pixel feature table** (X = predictors, y = LST) for downstream ML; pulls a *small* labelled table, not rasters:

```python
training = stack.sampleRegions(
    collection=wards, properties=['ward_id'], scale=30,
    geometries=True, tileScale=4)            # tileScale↑ avoids memory errors
```

**`Export` + `ee.batch`** — for outputs too big for a synchronous call (full-resolution LST GeoTIFF, an asset others can reuse):

```python
task = ee.batch.Export.image.toDrive(
    image=image.select('LST'),
    description='delhi_lst_2023', folder='EE_exports',
    region=aoi, scale=30, crs='EPSG:4326', maxPixels=1e13)
task.start()
# task.status()  -> poll READY / RUNNING / COMPLETED
```

`Export.image.toAsset` persists a server-side asset (zero re-download for later runs); `Export.table.toDrive` writes the zonal CSV.

**`getThumbURL` for instant web previews** (the cheap path to a heat-map image in a Streamlit/Leaflet app):

```python
url = image.select('LST').getThumbURL({
    'min': 295, 'max': 320, 'dimensions': 1024,
    'region': aoi, 'palette': ['blue', 'yellow', 'red']})
```

### 2.4 Scaling, projections & reducers — gotchas

- **`scale` is mandatory and meaningful.** It sets the pixel size (m) at which reductions happen. Use the **native** resolution (Landsat thermal effective 30 m after resampling, Sentinel-2 10/20 m). Too-fine `scale` over a huge AOI → memory errors; too-coarse → wrong stats.
- **Default projection trap.** Computations inherit a projection; for area/zonal work pass an explicit `crs`/`scale`. For aggregations, prefer `reduceRegions`/`reduceRegion` with explicit `scale` over `reproject` (which forces eager pixel computation and is a common OOM cause).
- **`ee.Reducer` composes:** `.mean().combine(ee.Reducer.minMax(), sharedInputs=True)`, `ee.Reducer.percentile([10,50,90])`, `ee.Reducer.count()`, grouped reductions via `.group()` for land-cover-class statistics.
- **`maxPixels`** — bump to `1e13` for city/state exports or you hit the default ceiling.
- **`tileScale`** (2/4/8/16) — increases internal tiling to dodge "User memory limit exceeded" on `reduceRegions`/`sampleRegions`.

### 2.5 Limits, quotas, and how to never hit them ✅

Verified current Earth Engine quotas (per Cloud project; some adjustable):

| Quota | Value | Note |
|---|---|---|
| Concurrent **interactive** requests | **40** | adjustable |
| Concurrent **high-volume** requests | **40** | the parallel-friendly endpoint |
| Request rate | **100 req/s** (~6,000/min) | |
| **Batch** task concurrency | **~2** average | batch is *throughput*, not low-latency |
| Tasks in `READY` queue | **3,000** | |
| Aggregation result cache | **100 MiB** | a `getInfo` cannot exceed this |
| Request payload | **10 MB** | keep client→server small |
| `getDownloadURL`/`getThumbURL` result | **32 MB** | use `Export` for bigger |
| Asset storage / count | **250 GB / 10,000** | |
| Daily EECU-time | unlimited by default (set cost controls) | **EECU-time = the real "cost" unit** |

**EECU (Earth Engine Compute Unit)** is the billing/throughput currency: total work scales with EECU-time, not with your code. *This is precisely the variable our "O(1) client" trades against* — area↑ ⇒ EECU↑, client cost flat.

**Anti-patterns that destroy performance (avoid all):**

1. **`getInfo()` in a loop** — N synchronous round-trips, each blocking. ❌ Instead, build one `FeatureCollection`/`ImageCollection` and reduce/export **once**.
2. **`.getInfo()` to "look at" data, then re-process** — pulls data client-side, defeating the model.
3. **Client-side Python `for` loops over images** — use `ImageCollection.map(fn)` so Google parallelises.
4. **`reproject` for analysis** — forces eager computation at a fixed projection; prefer reducers with explicit `scale`.
5. **Pulling rasters to the client for ML** — instead `sampleRegions`/`stratifiedSample` a *table* server-side and pull only that.

> **Golden rule:** *the only things that should cross the wire are (a) metadata, (b) rendered tiles/thumbnails, and (c) already-reduced aggregates/sample tables.* Obey this and client effort stays O(1).

### 2.6 Code snippet (a): GEE auth + **server-side Landsat 8 LST** via `ee.Image.expression`

Physics-informed, single-scene LST from Landsat 8/9 Collection-2 **Level-2** (surface-temperature band already atmospherically corrected) **and** an explicit NDVI-emissivity path on the brightness temperature so reviewers see the energy-balance reasoning. Verified scaling: SR bands `×2.75e-05 − 0.2`; `ST_B10 ×0.00341802 + 149` (Kelvin); QA_PIXEL bit-3 = cloud, bit-4 = cloud shadow, bit-1 = dilated cloud. ✅

```python
import ee
ee.Initialize(project='my-gcp-project')

aoi = ee.Geometry.Rectangle([77.0, 28.4, 77.4, 28.8])

def scale_l2(img):
    """Apply USGS C2-L2 scale/offset: SR -> reflectance, ST_B10 -> Kelvin."""
    sr = img.select('SR_B.').multiply(2.75e-05).add(-0.2)
    st = img.select('ST_B10').multiply(0.00341802).add(149.0)   # Kelvin
    return img.addBands(sr, overwrite=True).addBands(st, overwrite=True)

def add_lst(img):
    # NDVI from scaled SR bands (B5 NIR, B4 Red)
    ndvi = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')

    # Fractional vegetation cover & NDVI-threshold emissivity (server-side)
    fv = img.expression(
        '((NDVI - NDVImin) / (NDVImax - NDVImin)) ** 2',
        {'NDVI': ndvi, 'NDVImin': 0.2, 'NDVImax': 0.5}).rename('FV')

    emis = img.expression('0.004 * FV + 0.986', {'FV': fv}).rename('EMIS')

    # LST from brightness temp (ST_B10 in K) corrected by emissivity (Planck form)
    # lambda = 10.895 um (B10), rho = h*c/sigma = 1.438e-2 m*K
    lst = img.expression(
        'BT / (1 + (lambda * BT / rho) * log(EMIS))',
        {'BT': img.select('ST_B10'),
         'EMIS': emis,
         'lambda': 10.895e-6,
         'rho': 1.438e-2}).rename('LST')                       # Kelvin

    return img.addBands([ndvi, fv, emis, lst])

img = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
       .filterBounds(aoi)
       .filterDate('2023-05-01', '2023-05-31')
       .filter(ee.Filter.lt('CLOUD_COVER', 20))
       .map(scale_l2).map(add_lst)
       .median().clip(aoi))                                     # composite, see (b)

lst_c = img.select('LST').subtract(273.15).rename('LST_C')      # to deg C
print('mean LST (C):',
      lst_c.reduceRegion(ee.Reducer.mean(), aoi, 30, maxPixels=1e13).getInfo())
```

> Note for the science track: the Level-2 `ST_B10` band is *already* a high-quality, atmospherically-corrected LST. The NDVI-emissivity Planck correction above is shown because PS-1 is **physics-informed** — it demonstrates the energy-balance/emissivity reasoning explicitly and lets us swap in Level-1 `TOA` brightness temperature where Level-2 ST is unavailable. For production accuracy, prefer the provided `ST_B10`.

### 2.7 Code snippet (b): **cloud masking + median composite + `reduceRegion` zonal stats**

```python
import ee
ee.Initialize(project='my-gcp-project')

aoi = ee.Geometry.Rectangle([77.0, 28.4, 77.4, 28.8])
wards = ee.FeatureCollection('projects/my-gcp-project/assets/delhi_wards')

def mask_l8_c2(img):
    """Mask cloud (bit 3), cloud shadow (bit 4), dilated cloud (bit 1) from QA_PIXEL."""
    qa = img.select('QA_PIXEL')
    cloud  = qa.bitwiseAnd(1 << 3).neq(0)
    shadow = qa.bitwiseAnd(1 << 4).neq(0)
    dilate = qa.bitwiseAnd(1 << 1).neq(0)
    clear = cloud.Or(shadow).Or(dilate).Not()
    # scale SR + ST while we are here
    sr = img.select('SR_B.').multiply(2.75e-05).add(-0.2)
    st = img.select('ST_B10').multiply(0.00341802).add(149.0)
    return (img.addBands(sr, overwrite=True).addBands(st, overwrite=True)
               .updateMask(clear))

col = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
       .filterBounds(aoi)
       .filterDate('2023-04-01', '2023-06-30')
       .filter(ee.Filter.lt('CLOUD_COVER', 60))
       .map(mask_l8_c2))

# Cloud-free median composite (server-side; per-pixel temporal median)
composite = col.median().clip(aoi)
lst_c = composite.select('ST_B10').subtract(273.15).rename('LST_C')
ndvi  = composite.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
stack = lst_c.addBands(ndvi)

# --- Single-AOI aggregate ---
print(stack.reduceRegion(
    reducer=ee.Reducer.mean().combine(ee.Reducer.percentile([90]), sharedInputs=True),
    geometry=aoi, scale=30, maxPixels=1e13).getInfo())

# --- Per-ward zonal statistics (the driver-attribution workhorse) ---
zonal = stack.reduceRegions(
    collection=wards,
    reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
    scale=30)

# Pull ONLY the small table to the client (CSV-sized), or export it:
ee.batch.Export.table.toDrive(
    collection=zonal, description='ward_lst_ndvi',
    fileFormat='CSV').start()
```

This is the heart of the "O(1)" claim in action: the rasters never leave Google; we receive a **table keyed by ward** — input to `scikit-learn`/`PyTorch` driver models.

### 2.8 `geemap` — interactive maps & DataFrame bridges

`geemap` wraps `ee` + `ipyleaflet`/`folium` and gives the glue we want:

```python
import geemap
Map = geemap.Map(center=[28.61, 77.21], zoom=10)
Map.addLayer(lst_c, {'min': 25, 'max': 48,
                     'palette': ['blue', 'yellow', 'red']}, 'LST (C)')
Map.addLayer(wards, {}, 'Wards')

gdf = geemap.ee_to_gdf(zonal)        # FeatureCollection -> GeoDataFrame (small)
df  = geemap.ee_to_df(zonal)         # -> pandas DataFrame
# geemap.zonal_stats(stack, wards, 'out.csv', stat_type='MEAN', scale=30)
```

Use `geemap.ee_to_gdf` / `ee_to_df` **only on already-reduced FeatureCollections** (zonal results, samples) — they call `getInfo` under the hood, so they must stay small.

---

## 3. Alternatives & Complements — Comparison

| Platform | Compute model | Data access | Parallelism | Free tier | Best role for PS-1 |
|---|---|---|---|---|---|
| **Google Earth Engine** ✅ | Managed, lazy **server-side graph**; you submit a recipe, Google runs the cluster | Built-in catalogue (Landsat, Sentinel, MODIS, VIIRS, ERA5, GHSL pop, ESA WorldCover…) | Automatic, transparent | Free for research/non-commercial; EECU quotas | **Primary engine** — all heavy raster + zonal + composites |
| **Microsoft Planetary Computer** ✅ | **STAC** catalogue + Hub with **Dask Gateway**; *you* write xarray/Dask | Signed COG/Zarr assets via STAC API | Explicit (you scale Dask clusters) | Free account; Hub compute | **Primary fallback** — portability, ECOSTRESS & sensors missing/partial in GEE |
| **AWS Open Data** ✅ | Bring-your-own compute (Lambda/EC2/EMR); data as **COGs in S3** | Public buckets (Landsat C2, Sentinel-2 COGs) + Element84 `earth-search` STAC | DIY (Dask/Spark) | Data free (egress costs) | COG source for the STAC path; serverless tiling |
| **Sentinel Hub** | Cloud **Processing API** (evalscripts), tiling/statistics endpoints | Sentinel/Landsat/MODIS + commercial | Server-side per-request | Limited free / paid PU | Fast WMS/statistics tiles; viz-friendly; paid at scale |
| **openEO** (e.g. Copernicus Data Space) | **Standardised API** abstracting backends; portable process graphs | Backend-dependent (CDSE = Copernicus full archive) | Backend-managed | CDSE free quota | Vendor-neutral process graphs; GEE is gaining openEO support |
| **Digital Earth Africa / Australia** (ODC) | **Open Data Cube** (`datacube`/`odc-stac`) over indexed COGs | Analysis-ready Landsat/Sentinel for the continent | Dask | Free/sandbox | Reference for ARD + ODC patterns; not India-wide |

**Why GEE primary, PC fallback (the decision):**
- GEE = **least engineering, most catalogue, automatic parallelism** → fastest to a result and scale-invariant (the stakeholder's "O(1)/fastest").
- PC + STAC + `stackstac`/`odc-stac` + Dask = **open, portable, no lock-in**, and crucially **covers catalogue gaps**. Documented example: **ECOSTRESS 70 m LST in GEE currently has only Los-Angeles tiles ingested** ✅ — so for full ECOSTRESS coverage over Indian cities we *must* go to NASA LP DAAC / a STAC path. This single fact is the strongest concrete justification for keeping the fallback in the architecture.

---

## 4. Cloud-Native Geospatial Formats

The fallback path is fast for the *same reason* GEE is: **range-reads over cloud-native formats** mean you fetch only the bytes (and resolution) you need.

| Format | What it is | Why it enables "fetch only what you need" |
|---|---|---|
| **STAC** (SpatioTemporal Asset Catalog) | JSON metadata standard cataloguing imagery as searchable *Items* with asset hrefs | Query by bbox/time/cloud-cover and get back *just the matching scene URLs* — pure metadata, tiny payload. |
| **COG** (Cloud-Optimized GeoTIFF) | GeoTIFF with internal **tiling + overviews**, laid out for HTTP range requests | Read a spatial window or a coarse overview with **partial GET** — no full-file download. Fetching 100 m from 10 m native is cheap (built-in overviews). ✅ |
| **Zarr** | Chunked, compressed N-D arrays; cloud-native sibling of NetCDF/HDF | Ideal for time-series/data-cubes (ERA5, climate); read individual chunks in parallel with Dask. |
| **(Geo)Parquet** | Columnar table format; **GeoParquet** adds geometry | Fast columnar reads/joins for ward tables, sampled feature tables, vector AOIs; partition-pruned. |

**Library stack for the cloud-native path:**
`pystac-client` (search STAC) · `planetary-computer` (sign asset URLs) · `stackstac` **or** `odc-stac` (STAC Items → lazy `xarray` cube) · `rioxarray`/`rasterio` (COG I/O, CRS, reproject) · `xarray` + `dask` (lazy, chunked, parallel) · `geopandas`/`shapely`/`pyproj` (vectors) · `numpy`/`pandas`.

### 4.1 Code snippet (c): **STAC + stackstac fallback path** (Planetary Computer & AWS)

```python
import planetary_computer
import pystac_client
import stackstac

# --- Open the Planetary Computer STAC API; sign asset URLs automatically ---
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

bbox = [77.0, 28.4, 77.4, 28.8]                       # Delhi-ish
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2023-04-01/2023-06-30",
    query={"eo:cloud_cover": {"lt": 30}},
)
items = search.item_collection()
print(f"{len(items)} scenes")

# --- Lazy, Dask-backed xarray cube straight from COGs (nothing downloaded yet) ---
cube = stackstac.stack(
    items,
    assets=["B04", "B08"],                            # Red, NIR for NDVI
    epsg=32643,                                        # UTM 43N (Delhi)
    resolution=20,                                     # coarser-than-native = cheap (overviews)
    bounds_latlon=bbox,
    chunksize=2048,
)

# Server of bytes is remote; computation is local+lazy. Reduce, THEN compute small.
red, nir = cube.sel(band="B04"), cube.sel(band="B08")
ndvi = (nir - red) / (nir + red)
ndvi_med = ndvi.median(dim="time")                    # still lazy
result = ndvi_med.mean().compute()                    # only the scalar crosses memory
print("median-NDVI mean:", float(result))
```

**AWS Open Data variant** — swap the catalogue for the Element84 `earth-search` endpoint (no signing needed): ✅

```python
catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
items = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox,
                       datetime="2023-04-01/2023-06-30",
                       query={"eo:cloud_cover": {"lt": 30}}).item_collection()
# then stackstac.stack(items, ...) exactly as above
```

`odc-stac` alternative (often nicer CRS/resampling control): `from odc.stac import load; ds = load(items, bands=["red","nir"], crs="EPSG:32643", resolution=20, chunks={})`.

> The fallback obeys the **same O(1)-flavoured discipline**: STAC search returns metadata only; `stackstac.stack` builds a **lazy Dask graph**; you **reduce first** (`median`, `mean`, zonal) and only `.compute()` the small result. On a Dask Gateway cluster (Planetary Computer Hub) the reduction itself is parallelised server-side.

---

## 5. Local / Serving Geospatial Python Stack

| Layer | Libraries |
|---|---|
| **Raster I/O & math** | `rasterio`, `rioxarray`, `xarray`, `numpy`, `dask` |
| **Vector** | `geopandas`, `shapely`, `pyproj`, `fiona` |
| **EO platform clients** | `earthengine-api` (`ee`), `geemap`, `pystac-client`, `planetary-computer`, `stackstac`, `odc-stac`, `xee` (xarray↔EE) |
| **Interactive maps** | `geemap`, `leafmap`, `folium`, `ipyleaflet`, `pydeck`/**deck.gl**, **kepler.gl**, **MapLibre/Leaflet** |
| **Tile serving** | `localtileserver`, **TiTiler** (FastAPI dynamic COG tiler), GEE map tiles, `getThumbURL` |
| **ML** | `scikit-learn` (RF/GBM driver models), `xgboost`/`lightgbm`, `PyTorch` (physics-informed nets / CNN downscaling), `statsmodels` |

**Division of labour that preserves O(1):** GEE does the heavy raster reduction → **small tables** come local → `scikit-learn`/`PyTorch` train on them → predictions pushed back as a thin layer / rendered tiles. The local machine never holds a city-scale raster in RAM.

---

## 6. Recommended Compute Architecture (ASCII)

```
                         ┌──────────────────────────────────────────────────────────┐
                         │                     USER / BROWSER                        │
                         │   Streamlit UI  ·  MapLibre/deck.gl/kepler.gl heat map    │
                         │   ward sliders · "add cool roof / +trees" intervention    │
                         └───────────────▲───────────────────────────▲──────────────┘
                                         │ small JSON / PNG tiles     │ map tiles
                                         │ (O(1) over the wire)       │
                         ┌───────────────┴───────────────────────────┴──────────────┐
                         │              APP / SERVING LAYER (our server)             │
                         │   FastAPI / Streamlit   ·   TiTiler (COG tiles)           │
                         │   orchestration: ee + geemap   ·   sklearn/torch infer    │
                         └───────┬───────────────────────────────────────┬──────────┘
                                 │ recipe (computation graph)             │ STAC query (metadata)
          (PRIMARY — server-side)│ + scale/region                        │+ bbox/time/cloud  (FALLBACK)
                                 ▼                                        ▼
        ┌─────────────────────────────────────────┐   ┌──────────────────────────────────────────┐
        │        GOOGLE EARTH ENGINE (cluster)     │   │   STAC + COG/Zarr  (open, portable)       │
        │  catalogue: Landsat C2-L2, Sentinel-2,   │   │  Microsoft Planetary Computer (Dask GW)   │
        │  MODIS, VIIRS, ERA5, GHSL pop, WorldCover│   │  AWS Open Data (S3 COGs, earth-search)    │
        │  ── runs server-side ──                  │   │  NASA LP DAAC (ECOSTRESS 70 m LST) *      │
        │  filter→.map()→ee.Image.expression(LST,  │   │  ── stackstac / odc-stac → xarray+Dask ── │
        │  energy balance)→reduceRegions(wards)    │   │  lazy cube → reduce → .compute()          │
        └───────────────┬─────────────────────────┘   └───────────────────┬──────────────────────┘
                        │ returns ONLY:                                    │ returns ONLY:
                        │  • zonal CSV / FeatureCollection                 │  • reduced arrays / stats
                        │  • thumbnail / map tiles                         │  • COG window tiles
                        ▼                                                  ▼
        ┌──────────────────────────────────────────────────────────────────────────────────────────┐
        │   SMALL FEATURE TABLES (pandas / GeoParquet)  →  ML: RandomForest / XGBoost / PINN (torch) │
        │   driver attribution (NDVI, NDBI, albedo, ISA, pop, elevation → LST)  ·  cooling optimiser │
        └──────────────────────────────────────────────────────────────────────────────────────────┘

   * ECOSTRESS in GEE currently holds only Los-Angeles tiles ✅ → use the STAC/LP-DAAC fallback for
     full Indian-city ECOSTRESS coverage. This is the concrete reason the fallback path exists.

   KEY: thick arrows up/right carry ONLY metadata, tiles, and reduced aggregates → client effort ≈ O(1),
        independent of AOI size. All pixel-scale work happens inside GEE (primary) or Dask/COG (fallback).
```

---

## 7. Serving Layer Options

| Option | Strength | Use it for |
|---|---|---|
| **Streamlit** | Fastest demo-to-app; `geemap`/`leafmap` embed directly; sliders for intervention scenarios | **Recommended primary UI** for the hackathon demo |
| **FastAPI** | Async REST; pair with **TiTiler** for dynamic COG tiles; service-account GEE init | Production API / decoupled front-end |
| **Gradio** | One-function ML demos | Quick "input AOI → LST map" widget |
| **Voila** | Turns a notebook into an app | If the team lives in notebooks |
| **GEE Apps** (`ui.*`, Apps) | Zero-infra hosting *inside* Google; pure server-side | A no-backend public viewer; JS-side though |
| **Web map** (MapLibre/Leaflet + **deck.gl**/**kepler.gl**) | GPU-rendered heat maps, large vector layers, scenario toggles | The polished hotspot/intervention front-end |

**Recommendation:** **Streamlit + `geemap`** for the demo (fastest), with a **FastAPI + TiTiler + MapLibre/deck.gl** path documented for a productionised version, and **GEE map tiles / `getThumbURL`** as the cheap heat-map image source.

---

## 8. Performance & Parallelisation Patterns (the "keep client O(1)" checklist)

1. **Push computation to the data.** Reduce *before* anything crosses the wire — `reduceRegion(s)`, `median()`, `ee.Reducer.*` (GEE) or `.median()/.mean()` then `.compute()` (Dask).
2. **Lazy evaluation everywhere.** GEE graphs and `xarray`/`dask` are lazy by design — compose, then trigger **once**.
3. **Tiling & overviews.** COGs let you read coarse overviews / windows; GEE tiles render at the zoom you view. Request the **resolution you need**, not native.
4. **Vectorise / `.map()` not Python loops.** `ImageCollection.map(fn)` and array ops parallelise; per-element Python loops serialise and round-trip.
5. **Avoid `getInfo` loops.** Batch into one `FeatureCollection`; pull/`Export` once. (#1 cause of slow GEE code.)
6. **High-volume endpoint + thread/Dask pool** for many small tile/chip pulls; interactive endpoint for few heavy calls. ✅
7. **`tileScale` / `maxPixels`** to defeat "User memory limit exceeded" on big zonal/sample ops.
8. **Cache aggressively.** `Export.*.toAsset` server-side intermediates; cache thumbnails/tiles; memoise STAC searches; persist sampled tables as **GeoParquet**.
9. **Dask cluster (fallback) = explicit horizontal scale** — Planetary Computer Hub's **Dask Gateway** spins workers so the reduction itself parallelises; choose `chunksize` to balance task count vs overhead.
10. **Sample, don't haul.** For ML, `sampleRegions`/`stratifiedSample` a small table server-side; never download rasters to train.

---

## 9. References (URLs)

**Google Earth Engine**
- Authentication & Initialization — https://developers.google.com/earth-engine/guides/auth ✅
- Service Accounts — https://developers.google.com/earth-engine/guides/service_account ✅
- Processing Environments (interactive vs batch vs high-volume) — https://developers.google.com/earth-engine/guides/processing_environments ✅
- Usage quotas & EECU — https://developers.google.com/earth-engine/guides/usage ✅
- Statistics of Image Regions (`reduceRegions`) — https://developers.google.com/earth-engine/guides/reducers_reduce_regions ✅
- Grouped Reductions & Zonal Statistics — https://developers.google.com/earth-engine/guides/reducers_grouping ✅
- Landsat 8 C2 L2 dataset (band scaling, QA_PIXEL bits) — https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2 ✅
- ECOSTRESS L2T LSTE V2 in EE (LA-only ingest note) — https://developers.google.com/earth-engine/datasets/catalog/NASA_ECOSTRESS_L2T_LSTE_V2 ✅
- "Fast(er) Downloads" (Gorelick — high-volume endpoint) — https://gorelick.medium.com/fast-er-downloads-a2abd512aa26 ✅
- `geemap` book — analysis & zonal stats — https://book.geemap.org/chapters/06_data_analysis.html ✅
- `xee` (xarray ↔ Earth Engine) — https://github.com/google/Xee ✅

**STAC / Cloud-Native / Planetary Computer / AWS**
- Planetary Computer — Reading STAC — https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/ ✅
- `stackstac` basic example — https://stackstac.readthedocs.io/en/latest/basic.html ✅
- `stackstac` on a cluster — https://stackstac.readthedocs.io/en/latest/examples/cluster.html ✅
- `pystac-client` usage — https://pystac-client.readthedocs.io/en/latest/usage.html ✅
- `odc-stac` Sentinel-2 on PC — https://odc-stac.readthedocs.io/en/latest/notebooks/stac-load-S2-ms.html ✅
- AWS Element84 earth-search STAC — https://earth-search.aws.element84.com/v1 ✅
- Microsoft PlanetaryComputerExamples — https://github.com/microsoft/PlanetaryComputerExamples ✅

**Comparisons / Alternatives**
- PC vs GEE (MapScaping) — https://mapscaping.com/microsofts-planetary-computer-vs-google-earth-engine-a-compare-and-contrast/ ✅
- openEO vs Sentinel Hub (Copernicus) — https://dataspace.copernicus.eu/news/2024-10-9-comparing-openeo-and-sentinel-hub-apis ✅
- GEE vs PC (WILDLABS) — https://wildlabs.net/discussion/google-earth-engine-vs-microsofts-planetary-computer-which-do-i-use ✅
- GEE alternatives overview — https://flypix.ai/google-earth-engine-alternatives/ ✅

**LST methodology (Landsat NDVI-emissivity)**
- MDPI — GEE open-source LST code (Landsat series) — https://www.mdpi.com/2072-4292/12/9/1471 ✅
- ECOSTRESS L2T LSTE V2 (LP DAAC) — https://lpdaac.usgs.gov/products/eco_l2t_lstev002/ ✅

*Formula notes (LST/FVC/emissivity, Planck-based single-channel) are standard in the cited Landsat-LST literature; the constants (λ=10.895 µm for B10, ρ=1.438×10⁻² m·K, emissivity = 0.004·FV + 0.986, NDVI thresholds 0.2/0.5) are widely-used defaults — tune per study area. Marked from-knowledge where not pulled verbatim; the C2-L2 band scaling and QA bits ARE verified from the EE dataset page.*
