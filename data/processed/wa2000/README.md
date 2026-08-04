---
license: mit
task_categories:
- image-segmentation
tags:
- wildfire
- remote-sensing
- geospatial
- zarr
- earth-observation
---

# FireFusion WA 2000m

Daily spatio-temporal datacube for **wildfire ignition and cause prediction** over Washington State, 2003-2020, at 2 km resolution. Supervision is restricted to the **May-October fire season**, the months that carry Washington's fire activity.

Each cell-day carries 38 weather / fuel / terrain / human-activity channels. The targets ask whether a currently-unburned cell ignites within the next **7** days, and if so, why.

## Files

| Artifact                                 | Definition                                                                              |
| ---------------------------------------- | --------------------------------------------------------------------------------------- |
| `dataset.zarr`                           | every supervised day, split-agnostic. Only deterministic functions of the raw sources. |
| `dataset_manifest.json`                  | grid, channel list, and every deterministic transform applied above                     |
| `train.zarr` / `eval.zarr` / `test.zarr` | the **suggested** splits, described below                                               |
| `manifest.json`                          | split years, channel order, and the train-fitted statistics applied to those splits     |

`dataset.zarr` is the primary artifact. Every transform with a data-estimated parameter (any z-score, min-max, or scale) is deferred out of it and into the split stage, so it carries no knowledge of where the split boundaries fall. The splits below are one cut of it, shipped for comparability rather than because they are the only valid choice. Cut your own with the compile stage of the [FireFusionNet](https://github.com/tannerorourke/FireFusionNet) repository, which refits every statistic on whatever you designate as training data.

## Grid

|                        |                                     |
| ---------------------- | ----------------------------------- |
| CRS                    | EPSG:32610 (UTM Zone 10N)           |
| resolution             | 2000 m                              |
| grid (y, x)            | 204 x 217                           |
| latitude               | 45.5 to 49.0                        |
| longitude              | -122.5 to -117.0                    |
| season                 | May 1 - Oct 31, each year 2003-2020 |
| supervised days / year | 184                                 |
| supervised days total  | 3312                                |

## Time axis

Only in-season days ship, so the time axis is **not contiguous**: within a year it runs May 1 to Oct 31 day-by-day, then jumps to the next May. The manifests record `time.season_months = [5, 10]` and `time.contiguous = false`.

### Halo days

Extraction is deliberately wider than what ships. Each year is pulled as a single block running **March 22 to November 10**, the fire season plus a 40-day lead and a 10-day trail, so 234 days are extracted per year against the 184 that are supervised (Over 2003-2020, 4212 days extracted, 3312 published).

The halo lets temporal derivations enter the supervised window with real history instead of restarting at zero. Every backward-looking channel (decayed lightning load, 2 and 5-day cumulative precipitation, the per-cause ignition KDEs) is computed on the wider index, and the halo is dropped before any normalization statistic or class balance is taken, so those describe exactly the days that ship. The 40-day lead is sized by the longest backward operator in the pipeline, the lightning-load IIR, which decays below 0.1% there.

Halo days are never supervised and never scored. They survive only in the build-time staging cube (`cube.zarr`), which is not distributed, and are absent from every artifact listed above.

State that legitimately spans years decays by elapsed time, not by index position. The per-cause ignition KDEs apply a 365-day half-life to the true day count between consecutive entries, so a multi-year prior crosses the roughly 4.5-month off-season gap correctly attenuated instead of stepping across it as a single day.

A sliding-window loader must still avoid building any window that straddles the year-to-year gap, see *Loading*.

## Splits

Chronological, never random. The task is forecasting, so a random split would leak future weather into training.

| Split | Years     | Days | Size   |
| ----- | --------- | ---- | ------ |
| train | 2003-2016 | 2576 | 8.9 GB |
| eval  | 2017-2018 | 368  | 1.3 GB |
| test  | 2019-2020 | 368  | 1.4 GB |

Each split is a zarr store holding `X` with shape `(time, channel, y, x)` in `float32`, plus the labels and masks below.

## Labels

| Name             | dtype | Meaning                                         |
| ---------------- | ----- | ----------------------------------------------- |
| `ign_next`       | int8  | 1 if a clear cell burns within the next 7 days  |
| `ign_next_cause` | int8  | cause id of the earliest such ignition, else -1 |

| ID | Cause | Train positives |
| --- | --- | --- |
| 0 | `NATURAL_LIGHTNING` | 15679 |
| 1 | `HUMAN` | 5862 |
| 2 | `INDUSTRIAL` (includes debris) | 813 |

Note: `dataset.zarr` carries four classes. The splits `compile` debris into industrial together, since both are a few hundred cases vs. lightning and human causes which are the overwhelming majority of cases. Regroup them by re-running compile against the published cube; `n_cause_classes` in each manifest is authoritative for the store beside it.

## Masks

| Name               | Meaning                                         |
| ------------------ | ----------------------------------------------- |
| `land_mask`        | 1 on land, derived from the MODIS water flag    |
| `no_act_fire_mask` | 1 where the cell is not already burning         |
| `valid_cause_mask` | 1 where `ign_next_cause` carries a usable label |

Ignition is heavily imbalanced: **`ign_pos_weight` = 2746.86** on the train split, so positives are ~3.6e-4 of supervised cell-days. Restricting to the fire season removes winter cell-days that are near-uniformly negative, so this is a fire-season base rate and not an annual one. Losses should apply `land_mask` and `no_act_fire_mask`; the cause head should additionally apply `valid_cause_mask`.

## Channels

Deterministic steps (`clip`, `log1p`, `to_sin`, `per_area`) are already applied in `dataset.zarr` and recorded in `dataset_manifest.json`. Statistical steps
(`z_score`, `minmax`) are fit on the **train years only** and applied in the split stores, recorded in `manifest.json`. Both are listed together below in the order they compose.

| #   | Channel                        | Normalization            |
| --- | ------------------------------ | ------------------------ |
| 0   | `canopy_cover_pct`             | clip                     |
| 1   | `d_to_road`                    | clip -> log1p -> z_score |
| 2   | `dead_fmo_1000hr`              | z_score                  |
| 3   | `dead_fmo_100hr`               | z_score                  |
| 4   | `dewpoint`                     | z_score                  |
| 5   | `doy_sin`                      | (none)                   |
| 6   | `fire_spatial_roll`            | log1p -> z_score         |
| 7   | `fosberg_fwi`                  | z_score                  |
| 8   | `frac_imp_surface`             | clip                     |
| 9   | `kde_debris`                   | per_area -> z_score      |
| 10  | `kde_human`                    | per_area -> z_score      |
| 11  | `kde_industrial`               | per_area -> z_score      |
| 12  | `kde_natural_lightning`        | per_area -> z_score      |
| 13  | `lf_aspect_ew`                 | (none)                   |
| 14  | `lf_aspect_ns`                 | (none)                   |
| 15  | `lf_elevation`                 | z_score                  |
| 16  | `lf_slope`                     | z_score                  |
| 17  | `lightning_load`               | log1p -> z_score         |
| 18  | `lightning_strikes`            | log1p -> z_score         |
| 19  | `modis_lai`                    | clip -> z_score          |
| 20  | `modis_months_since_last_burn` | log1p -> minmax          |
| 21  | `ndvi_anomaly`                 | clip -> z_score          |
| 22  | `pop_density`                  | clip -> log1p -> z_score |
| 23  | `precip_2d`                    | log1p -> z_score         |
| 24  | `precip_5d`                    | log1p -> z_score         |
| 25  | `precip_mm`                    | log1p -> z_score         |
| 26  | `rel_humidity`                 | z_score                  |
| 27  | `temp_avg`                     | z_score                  |
| 28  | `temp_max`                     | z_score                  |
| 29  | `temp_min`                     | z_score                  |
| 30  | `usda_dist_to_wui_km`          | z_score                  |
| 31  | `usda_hs_density_km2`          | log1p -> z_score         |
| 32  | `usda_wui_index`               | z_score                  |
| 33  | `vpd_max`                      | clip -> log1p -> z_score |
| 34  | `vpd_min`                      | clip -> log1p -> z_score |
| 35  | `wind_dir_ew`                  | (none)                   |
| 36  | `wind_dir_ns`                  | (none)                   |
| 37  | `wind_mph`                     | clip -> log1p -> z_score |

The four `kde_*` channels are fire-history densities in **events per km squared**, with a 20 km smoothing radius and a 365-day decay half-life. The `per_area` step is what puts them in those units; the raw accumulator is mass per cell and would rescale with cell size.

`ndvi_anomaly` is NDVI minus its day-of-year climatology, averaged over the train years only so no held-out day contributes to the mean it is measured against.

## Loading

```python
import numpy as np
import xarray as xr

ds = xr.open_zarr("train.zarr")              # local path, or an fsspec URL
x = ds["X"].isel(time=slice(0, 10))          # (10, 38, 204, 217)
y = ds["ign_next"].isel(time=9)

# the time axis skips the off-season, so build windows within a contiguous block
days = np.asarray(ds.indexes["time"], dtype="datetime64[D]")
block = np.concatenate([[0], np.cumsum(np.diff(days).astype(int) != 1)])
# a length-W window at t is valid iff block[t] == block[t + W - 1]
```

Streaming directly from the Hub works for inspection, but **not for training**: `X` chunks are `(16, 38, 204, 217)`, about 108 MB each decompressed and ~55 MB on
the wire. A 10-day window spans one or two of them, so a train epoch of ~2450 windows moves on the order of 200 GB, roughly 24x the cost of downloading the 8.9 GB split once. Download, then train.

## Resolution tiers

The same pipeline and sources at four grid sizes, with identical channels, labels, masks, and split years.

Class balance is not identical. Ignition prevalence per cell rises with cell size, so `ign_pos_weight` is 973.95 at 4 km against 2746.86 at 2 km. Read it from the manifest of the tier you load rather than carrying it across.

| Tier                                                                              | Resolution | Grid (y, x) |
| --------------------------------------------------------------------------------- | ---------- | ----------- |
| [wa4000](https://huggingface.co/datasets/torq1/fire-fusion-wa-4000m)              | 4000 m     | 102 x 109   |
| **wa2000** (this one)                                                             | 2000 m     | 204 x 217   |
| [wa1000](https://huggingface.co/datasets/torq1/fire-fusion-wa-1000m)              | 1000 m     | 407 x 433   |
| [cascades250](https://huggingface.co/datasets/torq1/fire-fusion-wa-cascades-250m) | 250 m      | 576 x 576   |

## Sources

PRISM (temperature, dewpoint, vapour-pressure deficit, precipitation), NOAA AORC (humidity, wind), MODIS MOD13Q1 / MCD15A2H / MCD64A1 (NDVI, leaf area index, burn history, water mask), LANDFIRE (elevation, slope, aspect), NLCD (canopy cover, impervious surface), USFS FOD and fire perimeters (ignition and cause labels), NOAA NCEI SWDI (lightning), GPW (population), USDA WUI (wildland-urban interface), US Census TIGER (roads).

`dataset_manifest.json` and `manifest.json` are the authoritative record of channel order, normalization steps, grid geometry, and class balance. Read them rather than hardcoding.
