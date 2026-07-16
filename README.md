# FireFusion

Modeling pipeline for wildfire ignition & cause prediction, aggregating geography, climate, and fire data from WA state to train a Spatio-Temporal ConvFormer model.

See accompanying paper: `Multi-Modal-Spatio-Temporal-Learning-for-Wildfire-Ignition-Modeling.pdf`

## Citation

Paper submission is in progress. Feel free to use the work at will under MIT Liscense.

## Data Sources

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

## Forking/Reproducing

If you wish to rebuild the dataset from scratch, I'd recommend reacing out to me personally - it was created using a variety of gated API tokens, requested bulk downloads, and some elbow grease.
