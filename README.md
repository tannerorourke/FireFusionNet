# FireFusion

Modeling pipeline for wildfire ignition & cause prediction, aggregating geography, climate, and fire data from WA state to train a Spatio-Temporal ConvFormer model.

See accompanying paper: `Multi-Modal-Spatio-Temporal-Learning-for-Wildfire-Ignition-Modeling.pdf`

## Data Pipeline/Sourcing

The data sourcing for this project is inspired by the [SeasFire Datacube](https://arxiv.org/pdf/2312.07199) (Karasante et. al, 2022).

- 30 base features (6 derived) and next-day ignition + cause labels extracted from 7 sources spanning geography, population density, census, historical climatology, NASA LAADS satellite imagery, and fire ignition-point and occurence layers.
- Data interpolated onto a daily *time* and *configurable resolution* grid spanning Washington State with complete data from 2000-2020.
- Strict masking of water features and active fires (we're trying to figure out when/where a fire may come *next*, not already happened) plus cause.

The resulting dataset (*HF Dataset coming soon!*) combines 30 features covering Washington State between 2000-2020. Metadata, links, and docs listed under `fire_fusion/dataset/SOURCING.md` (a lot to keep track of!)

**Geography**:

- **NLCD** (National Landcover Database): Land cover classifications (aspect, slope, elevation) (30m res)
- **Census**: Road locations, derived to `distance to road`
- **USDA**: Wilderness-Urban Index

**Climatology**:

- **NASA GPWv4** (Gridded Population of the World): population density (30m res)
- **(NASA LAADS)**: Leaf Area Index (LAI) (*MCD15A2H*) and Normalized Difference Vegetation Index (NDVI) (*MOD13Q1*) (1km res)
- **gridMET** (Climatology Lab): Daily gridded meteorological data (30m res)

**Fire (labeling)**:

- **(NASA LAADS)**: Monthly burned-area product (*MCD64A1*) (500m res)
- **USFS** (U.S. Forest Service): Fire point occurence (plus cause) and daily perimeter layers layers

**Derived Features**:

- Sinusodial day-of-year
- Sinusodial wind direction
- 2-day and 5-day precipitation
- Fosberg Fire Weather Index

### Data Availability

The dataset was created using a multiple of gated API tokens and requested bulk downloads. If you wish to rebuild the dataset from scratch, please reach out to me personally.

## Modeling

A Spatio-Temporal ConvFormer (CNN + Transformer) performs attention over each axis of concern:

- CNN Encoder (ResNet MLPs)
- Windowed-Attention over the grid
- Attention over the feature axis
- Attention over the time axis
- Sequence of transformers performing attention on spatial, feature, and temporal shapes, respectively.
- CNN Decoder upsamples and predicts probability cell $i$, $j$ transitions to "ignition" at future timestep $t_{fire}\in [t+1, t+K-1]$:

$$
P(\text{fire}_{i,j}^{(t+1:t+K-1)}\text{ | }¬\text{fire}_{i,j}^{(t)})
$$

## Experiments

6 dataset/seed combinations are defined in `model/params.json` which the model is trained against (below). Checkpoints are written as `<experiment>_<main|specialized>_model.th`.

| Experiment | Dataset | Grid | Purpose |
| --- | --- | --- | --- |
| `smoke` (not reported) | wa2000 | 204x220 | Minimal testing run: 2 epochs, 32px crops, runs on a potato! |
| `wa2000-s{1,2}` | wa2000 | 204x220 | Full-grid training at 2km |
| `wa1000-s{1,2}` | wa1000 | 408x436 | 1km, supervised on 128px halo crops |
| `cascades250-s{1,2}` | cascades250 | 696x856 | 250m Eastern Cascades corridor, 128px halo crops |

Reported results are the mean over the two seeds per dataset. the seed pairs are otherwise identical configurations.

## Commands

- `python -m fire_fusion.dataset.build --[dataset]`: Run the data extraction pipeline.
- `python -m fire_fusion.model.train --[experiment] --[dataset] --[seed] --[stage] --[init-from] --[freeze] --[alpha-ign] --[alpha-cause] --[export-s3]`: Train the ConvFormer model. Requires a built dataset under `data/processed`.
- `python -m fire_fusion.model.predict --[experiment] --[dataset] --[checkpoint] --[calib] --[split] --[batches]`: Turn a trained checkpoint into per-cell ignition probabilities.
