# FireFusion
Modeling pipeline for wildfire ignition & cause prediction, aggregating geography, climate, and fire data from WA state to train a Spatio-Temporal ConvFormer model.

See accompanying paper: `Multi-Modal-Spatio-Temporal-Learning-for-Wildfire-Ignition-Modeling.pdf`

### Data Sources

**Geography**
- Land cover classification from the **NLCD (National Landcover Database)** (30m)

**Climatology**
- population density from **GPWv4** (Gridded Population of the World)
- Leaf Area Index (LAI) (*MCD15A2H*) and Normalized Difference Vegetation Index (NDVI) (*MOD13Q1*) from **(NASA LAADS)** (1km)
- Daily gridded meteorolical data from **gridMET (Climatology Lab)** (30m)

**Fire (labeling)**:
- Monthly burned-area product (MCD64A1) from **(NASA LAADS)** -  (500m)
- Fire Occurence Point and Fire Perimeter layers from **USFS**.

### Spatio-Temporal ConvFormer
- Spatial CNN Encoder (ResNet MLPs)
- Sequence of transformers performing attention on spatial, feature, and temporal shapes, respectively.
- Decoder predicts the probability cell $i$, $j$ transitions to "ignition" at future timestep $t_{fire}\in [t+1, t+K-1]$:

$$P(\text{fire}_{i,j}^{(t+1:t+K-1)}\text{ | }¬\text{fire}_{i,j}^{(t)})$$

- Strict masking of water features, active fires, and active fire causes to focus loss on real-time priors.

## Forking/Reproducing

Requires > ~75gb storage to unpack data.
- Unpack each `.7z` files under `/data/raw` - many data sources were manually downloaded.
- `python -m fire_fusion.dataset.build --[resolution] --[start_year] --[end_year]`
- `python -m fire_fusion.train.py` (requires `.zarr` build files)

Note that RAM requirements scale quadratically by resolution and linearly with time. Don't get overzealous with resolution and start/end dates.
