# FireFusion

ML modeling pipeline purpose-built for sourcing, analyzing, and predicting fire ignition and ignition cause in Washington State, at resolutions down to 250 m.

Ten geospatial products spanning terrain, fuels, weather, human activity, lightning, and fire history are aggregated onto a single daily grid spanning 2003-2020 continuously.

As a case study, we train a spatiotemporal ConvFormer on the datacube to predict, for every currently-unburned cell, the probability it ignites within the next 7 days and which of three cause classes (human, lightning, industrial) are responsible.

**Stats & Features**:

- Full daily coverage from 2003-2020, supervised over the May-October fire season.
- 38 input channels (25 from source processors, 13 derived), normalized, resampled and interpolated into trainable feature layers.
- Custom derived features: Per-cause ignition KDE's, 3x3 cell 7d rolling fire occurrence, NDVI anomalies, 2 and 5-day cumulative precipitation, 100 and 1000-hr dead fuel moisture, decayed lightning load, Fosberg FWI.
- Circular quantities (N/S and E/W aspect and wind-direction components, day-of-year) decomposed into orthogonal components, so no channel carries 0/360 discontinuities.
- Fire ignition time, cause, KDE by cause, 3x3 rolling fire occurrence, and months since last burn.
- Land and active fire masks

## Datasets

Four datacubes, available on HuggingFace, are the intended way to use this work.

Each spans 2003-2020 and includes a raw `dataset.zarr` including normalized raw and derived features, as well as a `2003-2016`/`2017-2018`/`2019-2020` train/eval/test split (used in flagship model). The `wa****` datasets cover the full state. `cascades250` narrows to a 144 x 144 km (20,736 km$^2$) window over the Northeastern Cascades, which carries ~30% of Washington's ignition cell-days on 12% of its land area (3.1x increase), useful for fine-tuning. Each dataset .

| Dataset | Resolution | Grid (y, x) | Coverage | Dataset |
| --- | --- | --- | --- | --- |
| `wa4000` | 4000 m | 102 x 109 | Washington State | [torq1/fire-fusion-wa-4000m](https://huggingface.co/datasets/torq1/fire-fusion-wa-4000m) |
| `wa2000` | 2000 m | 204 x 217 | Washington State | [torq1/fire-fusion-wa-2000m](https://huggingface.co/datasets/torq1/fire-fusion-wa-2000m) |
| `wa1000` | 1000 m | 407 x 433 | Washington State | [torq1/fire-fusion-wa-1000m](https://huggingface.co/datasets/torq1/fire-fusion-wa-1000m) |
| `cascades250` | 250 m | 576 x 576 | Eastern Cascades corridor | [torq1/fire-fusion-wa-cascades-250m](https://huggingface.co/datasets/torq1/fire-fusion-wa-cascades-250m) |

Common to every tier:

| | |
| --- | --- |
| CRS | EPSG:32610 (UTM Zone 10N) |
| record | 2003-2020, supervised over the May-October fire season |
| supervised days | 184 per year, 3312 total |

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

| Name | Meaning |
| --- | --- |
| `ign_next` | 1 if a currently-clear cell burns within the next 7 days |
| `ign_next_cause` | cause of the earliest such ignition: natural/lightning, human, industrial, debris |
| `land_mask` | 1 on land |
| `no_act_fire_mask` | 1 where the cell is not already burning |
| `valid_cause_mask` | 1 where a usable cause label exists |

Masking is the core physics constraint. The model is never supervised, and must never be scored, on water, on currently-burning cells, or on unknown-cause pixels. Every mask reads 1 where the cell is usable by the head it gates, so a predicate is always an AND of `== 1` and never a negation.

Ignition is heavily imbalanced, and the imbalance itself is resolution-dependent: `ign_pos_weight` is 973.95 at 4 km and 2746.86 at 2 km. It is read from the manifest rather than restated anywhere in code.

## Modeling

As a case study, we train a spatiotemporal ConvFormer, which performs three forms of attention over each axis of the grid:

- CNN encoder (ResNet MLPs)
- Spatial-Windowed attention over the grid
- SDPA over the feature axis
- SDPA Attention over the time axis
- CNN decoder upsampling to a per-cell ignition probability

Channel count, grid size, cause-class count, and `pos_weight` all come from the manifest of the loaded dataset.

### Experiments

Configured in `fire_fusion/model/params.json`.

| Experiment | Dataset | Grid | Purpose |
| --- | --- | --- | --- |
| `smoke` (not reported) | wa2000 | 204x217 | Minimal testing run: 2 epochs, 32px crops, runs on a potato! |
| `wa4000-s{1,2}` | wa4000 | 102x109 | Full-grid training at 4 km |
| `wa2000-s{1,2}` | wa2000 | 204x217 | Full-grid training at 2 km |
| `wa1000-s{1,2}` | wa1000 | 407x433 | 1 km, supervised on 128px halo crops |
| `cascades250-s{1,2}` | cascades250 | 576x576 | 250 m Eastern Cascades corridor, 128px halo crops |

Two separate heads are fine-tuned on 7-day ignition and cause over 3 classes to output Platt-scaled prediction maps.

Reported results are the mean over the two seeds per dataset; the seed pairs are otherwise identical configurations.

## Pipeline

Three staged transforms, selected by `--stage`, so peak RAM is one feature layer plus a few dask chunks rather than the full cube.

```
extract   raw sources  ->  cube.zarr                              staging, halo days, sourced features
publish   cube.zarr    ->  dataset.zarr                           redistributable, split-agnostic
compile   dataset.zarr ->  {train,eval,test}.zarr + manifest.json splits and train-fitted statistics
```

The boundary between `publish` and `compile` is split-dependence. A transform that is a fixed function of the value or the grid (`clip`, `log1p`, `to_sin`, `per_area`) belongs to `publish`. A transform whose parameters are estimated from data (`z_score`, `minmax`, `scale_max`) stays in `compile`, where statistics come from finite train-split cells only. That is what lets `dataset.zarr` be redistributed without a consumer inheriting this project's train/eval/test choice.

Adding a channel means adding a `Feature` to `fire_fusion/config/feature_config.py`, plus a processor branch or a `DerivedProcessor` function. The builder, the normalizer, and the loader all discover channels from that config.

## Setup

[uv](https://docs.astral.sh/uv/) or plain pip. The linux `torch` wheel carries CUDA 12.1, GPU host needs no extra CUDA setup. `python -m fire_fusion.<...>` runs from any directory.

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

To reproduce an exact version, use `requirements.lock.txt` to pin transitive dependencies to exact versions and hashes:

```bash
uv pip sync requirements.lock.txt                                          # reproduce
uv pip compile requirements.txt -o requirements.lock.txt --generate-hashes # regenerate
```

## Commands

- `python -m fire_fusion.dataset.build --dataset <name> --stage <extract|publish|compile|all>`: build a datacube.
- `python -m fire_fusion.model.train --[experiment] --[dataset] --[seed] --[stage] --[init-from] --[freeze] --[alpha-ign] --[alpha-cause] --[export-s3]`: train the ConvFormer. Requires a built dataset under `data/processed`.
- `python -m fire_fusion.model.predict --[experiment] --[dataset] --[checkpoint] --[calib] --[split] --[batches]`: turn a trained checkpoint into per-cell ignition probabilities.

## Data availability

Rebuilding the dataset (i.e., at a new resolution) sits behind a mix of gated API tokens, bulk-download requests, and rate limits: MODIS access needs a NASA Earthdata token, and PRISM's service caps downloads per file per day.

If you would like to learn more, or want to dig into the data deeper, please reach out to me personally!
