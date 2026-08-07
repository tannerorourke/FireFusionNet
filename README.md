# FireFusion

ML modeling pipeline purpose-built for sourcing, analyzing, and predicting fire ignition and ignition cause in Washington State, at resolutions down to 500 m.

Ten geospatial products spanning terrain, fuels, weather, human activity, lightning, and fire history are aggregated onto a single daily grid spanning 2003-2020 continuously.

As a case study, we train a spatiotemporal ConvFormer on the datacube to predict, for every currently-unburned cell, the probability it ignites within the next 7 days and which of three cause classes (human, lightning, industrial) are responsible.

**Stats & Features**:

- Full daily coverage from 2003-2020, supervised over the May-October fire season.
- 38 input channels (25 from source processors, 13 derived), normalized, resampled and interpolated into trainable feature layers.
- Custom derived features: Per-cause ignition KDE's, 3x3 cell 7d rolling fire occurrence, NDVI anomalies, 2 and 5-day cumulative precipitation, 100 and 1000-hr dead fuel moisture, decayed lightning load, Fosberg FWI.
- Circular quantities (N/S and E/W aspect and wind-direction components, day-of-year) decomposed into orthogonal components, so no channel carries 0/360 discontinuities.
- Fire ignition time, cause, KDE by cause, 3x3 rolling fire occurrence, and months since last burn.
- Water and active fire masks

## Datasets

Four datacubes, available on HuggingFace, are the intended way to use this work.

Each spans 2003-2020 and includes a raw `dataset.zarr` including normalized raw and derived features, as well as a `2003-2016`/`2017-2018`/`2019-2020` train/eval/test split, used in flagship model. The `wa****` datasets cover the full state. `cascades500` narrows to a 272 x 272 km (73,984 km$^2$) window over the eastern Cascades and Okanogan, the corridor holding the bulk of the state's USFS-recorded ignitions, useful for fine-tuning.

| Dataset | Resolution | Grid (y, x) | Coverage | Dataset |
| --- | --- | --- | --- | --- |
| `wa4000` | 4000 m | 102 x 109 | Washington State | [torq1/fire-fusion-wa-4000m](https://huggingface.co/datasets/torq1/fire-fusion-wa-4000m) |
| `wa2000` | 2000 m | 204 x 217 | Washington State | [torq1/fire-fusion-wa-2000m](https://huggingface.co/datasets/torq1/fire-fusion-wa-2000m) |
| `wa1000` | 1000 m | 407 x 433 | Washington State | [torq1/fire-fusion-wa-1000m](https://huggingface.co/datasets/torq1/fire-fusion-wa-1000m) |
| `cascades500` | 500 m | 544 x 544 | Eastern Cascades-Okanogan corridor | [torq1/fire-fusion-cascades-500m](https://huggingface.co/datasets/torq1/fire-fusion-cascades-500m) |

## Sources

| Source | Type | Contributes | Native resolution |
| --- | --- | --- | --- |
| [USFS Occurrence Point](https://data-usfs.hub.arcgis.com/datasets/usfs%3A%3Anational-usfs-fire-occurrence-point-feature-layer/about) | Fire | ignition time and cause, per-cause ignition density (KDE, 20 km), 3x3 rolling occurrence | daily, point |
| [USFS Perimeter Layer](https://data-usfs.hub.arcgis.com/datasets/usfs::national-usfs-fire-perimeter-feature-layer/about) | Fire | fire extent through time | daily, <10 m |
| [PRISM AN81d](https://prism.oregonstate.edu/documents/PRISM_downloads_web_service.pdf) | Meteorology | temperature (mean/min/max), dewpoint, vapour-pressure deficit (min/max), precipitation, 2 and 5-day cumulative precipitation, 100 and 1000-hr dead fuel moisture | daily, 800 m |
| [NOAA AORC v1.1](https://registry.opendata.aws/noaa-nws-aorc/) | Meteorology | relative humidity, wind speed, wind direction decomposed E-W and N-S, Fosberg Fire Weather Index | daily, ~800 m |
| [NCEI SWDI NLDN](https://www.ncei.noaa.gov/pub/data/swdi/database-csv/v2/) | Meteorology | same-day CG strike count, decayed lightning load as a holdover proxy | daily, ~11 km |
| [MODIS MCD15A2H](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MCD15A2H) | Vegetation | leaf area index | 8-day, 500 m |
| [MODIS MOD13Q1](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD13Q1) | Vegetation | NDVI, NDVI anomaly against day-of-year climatology, land/water mask | 16-day, 250 m |
| [MODIS MCD64A1](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MCD64A1) | Fire | months since last burn | monthly, 500 m |
| [LANDFIRE](https://www.landfire.gov/viewer/) | Geography | elevation, slope, aspect decomposed E-W and N-S | annual, 30 m |
| [NLCD](https://www.mrlc.gov/viewer/) | Geography | fractional impervious surface, canopy cover | annual, 30 m |
| [USDA CONUS WUI v4](https://www.fs.usda.gov/rds/archive/catalog/RDS-2015-0012-4) | Human | housing density, WUI class index, distance to WUI interface | decadal, block-level |
| [NASA GPWv4](https://doi.org/10.7927/H4F47M65) | Human | population density | 5-year, ~1 km |
| [Census TIGER/Line](https://www.census.gov/cgi-bin/geo/shapefiles/index.php) | Human | distance to nearest road | exact, vector |

See `fire_fusion/dataset/SOURCING.md` for authoritative info on products, versions, links, and per-feature aggregation.

### Labels and masks

Masking is the core physics constraint. The model is never supervised, and must never be scored, on water, on currently-burning cells, or on unknown-cause pixels.

| Name | Meaning |
| --- | --- |
| `ign_next` | 1 if a currently-clear cell burns within the next 7 days |
| `ign_next_cause` | cause of the earliest such ignition. `dataset.zarr` carries four classes (natural/lightning, human, industrial, debris). The splits fold debris into industrial, leaving three |
| `land_mask` | 1 on land |
| `no_act_fire_mask` | 1 where the cell is not already burning |
| `valid_cause_mask` | 1 where a usable cause label exists |

## Modeling

We train a two-path spatiotemporal ConvFormer: 26 dynamic channels (weather, land state) flow through the attention trunk, and 12 static maps (terrain, human footprint) condition it through FiLM.

- Per-group CNN stems (ResNet MLPs), one per dynamic channel group, each with a learned missing-modality token
- Windowed spatial attention over the grid
- SDPA over the feature axis
- SDPA over the time axis
- A zero-initialized static branch producing per-level, spatially-varying FiLM, applied through encoder and decoder
- CNN decoder upsampling to per-cell 7-day ignition probability and cause over 3 classes, each head with a fitted calibrator

### Training

Forest fire ignition is a heavily imbalanced target, dependent on resolution (the positive rate implies a pos_weight of 944.85 at 4 km and 2683.80 at 2 km). Instead of reweighting, we absorb it by using **unit-weight BCE** and subsampling negatives in the loss. This keeps ignition loss proper under extreme imbalance.

### Experiments

Two seed profiles per tier at a uniform embed width (configured in `fire_fusion/model/params.json`), plus a wider `cascades500-optimal` headline model. Reported ladder results are seed means.

- Build: each tier's datacube is built from the raw sources, validated, and published (HuggingFace hosts the cube, B2 the training splits).
- Pre-flight: a 4 km width comparison fixes the ladder's embed width to ensure val ignorance does not degrade as dimension scales.
- Ladder: Three tiers (wa2000m, wa1000m, cascades250m) each at two seeds, plus the `cascades500-optimal` headline.

## Pipeline

Building from raw data requires gated API tokens, bulk-download requests, and patience waiting on rate limits. Please reach out me personally if you'd like to rebuild/reproduce the data.

Three staged transforms, selected by `--stage`. This keeps peak RAM to one feature layer plus a few dask chunks rather than the full cube.

1. `extract`: Raw sources $\rightarrow$ `cube.zarr`.
2. `publish`: `cube.zarr` $\rightarrow$ `dataset.zarr`.
3. `compile`: `dataset.zarr` $\rightarrow$ `{train,eval,test}.zarr` + `manifest.json` + train-fitted statistics.

*Note on transforms*: All data transforms done in the `publish` step are a fixed function of the grid (e.g., `clip`, `log1p`, `to_sin`, and `per_area`). Transforms whose parameters are estimated from data (e.g., `z_score`, `minmax`, and `scale_max`) are done in `compile`. This ensures statistics that are dependent on the train-split choice are separated, and lets `dataset.zarr` act as a redistributable product, irrespective of train/eval/test choice.

*Disclaimer*: Raw data is reprojected to EPSG:32610 (UTM Zone 10N) CRS and does not transfer cleanly to other states without rebuilding data from scratch.

## Setup

[uv](https://docs.astral.sh/uv/) or plain pip. The linux `torch` wheel carries CUDA 12.1, GPU host needs no extra CUDA setup.

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
uv run python -m fire_fusion.<...>
```

To reproduce an exact version, use `requirements.lock.txt` to pin transitive dependencies to exact versions and hashes:

```bash
uv pip sync requirements.lock.txt # reproduce
uv pip compile requirements.txt -o requirements.lock.txt --generate-hashes # regenerate
```

## Commands

- `python -m fire_fusion.dataset.build --dataset <name> --stage <extract|publish|compile|all>`: build a datacube.
- `python -m fire_fusion.train --[experiment] --[dataset] --[seed] --[init-from] --[freeze] --[alpha-ign] --[alpha-cause] --[export-b2]`: train the ConvFormer. Requires a built dataset under `data/processed`.
- `python -m fire_fusion.predict --[experiment] --[dataset] --[checkpoint] --[calib] --[split] --[batches]`: turn a trained checkpoint into per-cell ignition probabilities.



