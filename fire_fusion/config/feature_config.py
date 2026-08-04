# Who wants to deal with tuples in JSON, anyways??
from dataclasses import dataclass
import numpy as np
from rasterio.enums import Resampling
from typing import List, Optional, Tuple
from xarray.core.types import InterpOptions


CAUSAL_CLASSES = [
    "NATURAL_LIGHTNING",
    "HUMAN",
    "INDUSTRIAL",
    "DEBRIS"
]

# Ignition horizon: a clear cell is positive if it burns within this many days,
# and the cause label and its validity mask read the same forward window. A day
# leaves the positive class too sparse to supervise; a week still reads as a
# short-range forecast.
IGN_HORIZON_DAYS = 7

CAUSE_RAW_MAP = {
    "NATURAL_LIGHTNING": [
        "1", # 1, 1 - lightning
        "lightning",
        "natural",
        "other natural cause",
    ],
    "HUMAN": [
        "3", # smoking
        "4", # campfire
        "7", # arson
        "8", # children
        # text
        "campfire", "camping",
        "arson", "incendiary", "firearms/weapons",
        "children",
        "human",
        "miscellaneous",
        "other causes",
        "other human cause",
    ],
    "INDUSTRIAL": [
        "2", # equip/vehicle use
        "6", # railroad
        "9", # misc
        "equip/vehicle use", "equipment", "equipment use",
        "powgen/trans/distrib",
        "railroad", "utilities", "vehicle",
    ],
    "DEBRIS": {
        "5", # debris burning
        "debris burning", "debris/open burning", "debris"
    },
    "UNKNOWN": [
        "0",
        "cause not identified",
        "investigated but und",
        "undetermined", "undertermined",
        "",
    ],
}

WUI_CLASSES = {
    "UNINHABITED_WATER",
    "RURAL",
    "NOWUI_URBAN",
    "WUI_INTERMIX",
    "WUI_INTERFACE"
}

WUI_CLASS_MAP: dict[str, int] = {
    # uninhabited
    "Uninhabited_Veg": 0,
    "Uninhabited_NoVeg": 0,
    "Water": 0,
    # low density (rural)
    "Very_Low_Dens_Veg": 1,
    "Very_Low_Dens_NoVeg": 1,
    # non-WUI urban / town (built, no veg)
    "Low_Dens_NoVeg": 2,
    "Med_Dens_NoVeg": 2,
    "High_Dens_NoVeg": 2,
    # WUI intermix
    "Low_Dens_Intermix": 3,
    "Med_Dens_Intermix": 3,
    "High_Dens_Intermix": 3,
    # WUI interface
    "Low_Dens_Interface": 4,
    "Med_Dens_Interface": 4,
    "High_Dens_Interface": 4,
}

LAND_COVER_RAW_MAP = {
    0: [11], # water
    1: [12], # snow
    2: [21, 22], # developed, < 49%
    3: [23, 24], # developed >= 50%
    4: [31], # barren
    5: [41, 42, 43], # forest
    6: [52, 71], # farmland
    7: [90, 95], # wetlands
}

@dataclass
class Feature:
    name: str = ""
    key: Optional[str] = ""                         # unique key to access data
    clip: Optional[Tuple[float, float]] = None
    resampling: Optional[Resampling] = None         # strategy for filling missing pixels in feature grid
    time_interp: Optional[Tuple[str, InterpOptions]] = None # "time" = broadcast over time D, "existing" = fill missing
    
    # OHEs
    num_classes: Optional[int] = 0
    one_hot_encode: Optional[bool] = False

    # Special attrs
    kde_smooth_radius_km: Optional[float] = None
    kde_half_life_days: Optional[float] = None      # decay half-life for the KDE accumulator
    expand_names: Optional[List[str]] = None        # names of new features base feature is expanded into
    
    # labels and masks
    inputs: Optional[List[str]] = None
    is_label: Optional[bool] = False
    is_mask: Optional[bool] = False
    # derived features
    
    func: Optional[str] = ""                        # DerivedProcessor function signature
    # -- the derivation estimates something from the record, so it cannot ship
    # -- split-agnostic and is rebuilt against the train years at compile time.
    # -- That rebuild sees supervised days only, so a temporal-window operator
    # -- would lose its halo history and must not set this.
    train_dependent: Optional[bool] = False
    drop_inputs: Optional[List[str] | None] = None
    ds_clip: Optional[Tuple[float, float]] = None   # clip values after processing
    ds_norms: Optional[List[str]] = None            # sequence of normalizations


def get_labels():
    return [l for l in drv_feat_config() if l.is_label==True]

# -- every mask equals 1 where the cell is usable for the head it gates
def get_masks():
    return (
        [f for f in drv_feat_config() if f.is_mask==True] +
        [f for feats in base_feat_config().values() for f in feats if f.is_mask==True]
    )

def base_feat_config():
    return {
        "PRISM": [
            Feature(
                name = "temp_avg",
                key = "tmean",
                clip = (-40.0, 120.0),
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_norms = ["z_score"],
            ),
            Feature(
                name = "temp_min",
                key = "tmin",
                clip = (-40.0, 120.0),
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_norms = ["z_score"],
            ),
            Feature(
                name = "temp_max",
                key = "tmax",
                clip = (-40.0, 120.0),
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_norms = ["z_score"],
            ),
            Feature(
                name = "dewpoint",
                key = "tdmean",
                clip = (-60.0, 90.0),
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_norms = ["z_score"],
            ),
            Feature(
                name = "vpd_min",
                key = "vpdmin",
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_clip = (0.0, 200.0),
                ds_norms = ["log1p", "z_score"],
            ),
            Feature(
                name = "vpd_max",
                key = "vpdmax",
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_clip = (0.0, 200.0),
                ds_norms = ["log1p", "z_score"],
            ),
            Feature(
                name = "precip_mm",
                key = "ppt",
                clip = (0, 150),
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_norms = ["log1p", "z_score"],
            ),
        ],
        "AORC": [
            Feature(
                name = "rel_humidity",
                key = "rel_humidity",
                clip = (0.0, 100.0),
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_norms = ["z_score"],
            ),
            Feature(
                # consumed by the fuel-moisture derivation, then dropped
                name = "rh_max",
                key = "rh_max",
                clip = (0.0, 100.0),
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
            ),
            Feature(
                name = "wind_mph",
                key = "wind_mph",
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_clip = (0.0, 100.0),
                ds_norms = ["log1p", "z_score"],
            ),
            Feature(
                name = "wind_dir",
                key = "wind_dir",
                clip = (0.0, 360.0),
                # nearest: a raw angle blends through 180 degrees on 359->1
                # wraparounds under bilinear resampling
                resampling = Resampling.nearest,
                time_interp = ("existing", "nearest"),
                # dropped
            ),
        ],
        "FIRE_USFS": [
            Feature(
                # dropped for final label
                name = "usfs_perimeter",
                key = "Fire_Perimeter",
                # NO TIME INTERPOLATION
            ),
            Feature(
                # dropped for final label
                name = "usfs_burn",
                key = "Fire_Occurence",
                expand_names=["usfs_burn_occ", "usfs_burn_cause"]
                # NO TIME INTERPOLATION
            ),
            Feature(
                name = "usfs_KDE",
                # KDE names are "kde_[burn cause]"
                expand_names = ["kde_natural_lightning", "kde_human", "kde_industrial", "kde_debris"],
                key = "Fire_KDE",
                kde_smooth_radius_km = 20,
                # annual half-life keeps a multi-year spatial ignition prior while
                # holding the accumulator stationary, so a train-fit z_score stays
                # calibrated on the chronologically later eval/test years
                kde_half_life_days = 365,
                ds_norms = ["z_score"]
                # NO TIME INTERPOLATION
            ),
        ],
        "USDA_WUI": [
            Feature(
                name = "usda_hs_density_km2",
                key = "hs_density",
                # 99 percentile clip in processor
                time_interp = ("existing", "linear"),
                ds_norms = ["log1p", "z_score"]
            ),
            Feature(
                name = "usda_wui_index",
                key = "wui_index",
                # nearest: the index is ordinal; linear blends across classes
                time_interp = ("existing", "nearest"),
                ds_norms = ["z_score"]
            ),
            Feature(
                name = "usda_dist_to_wui_km",
                key = "dist_to_interface",
                time_interp = ("existing", "linear"),
                ds_norms = ["z_score"]
            )
        ],
        "GPW": [
            Feature(
                name = "pop_density",
                resampling = Resampling.nearest,
                time_interp = ("existing", "linear"),
                ds_clip = (0.0, np.inf),
                ds_norms=["log1p", "z_score"]
            )
        ],
        "LANDFIRE": [
            Feature(
                name = "lf_elevation",
                key = "_Elev",
                resampling = Resampling.bilinear,
                clip = (0.0, 5000.0),
                time_interp = ("broadcast", "linear"),
                ds_norms = ["z_score"]
            ),
            Feature(
                name = "lf_slope",
                key = "_SlpD",
                resampling = Resampling.bilinear,
                time_interp = ("broadcast", "linear"),
                ds_norms = ["z_score"]
            ),
            Feature(
                # dropped
                name = "lf_aspect",
                key = "_Asp",
                resampling = Resampling.bilinear,
                time_interp = ("broadcast", "linear"),
                
            ),
        ],
        "MODIS": [
            Feature(
                name = "modis_lai",
                key = "MCD15A2H",
                # simple reprojection
                resampling = Resampling.nearest, 
                # Ensure sharp dropoffs are captured
                time_interp = ("existing", "nearest"),
                ds_clip=(0.0, 10.0),
                ds_norms = ["z_score"],
            ),
            Feature(
                # dropped
                name = "mod13q1", # step function holds values for dropoffs (fires)
                expand_names = ["modis_ndvi", "modis_water_mask"],
                key = "MOD13Q1",
                resampling = Resampling.nearest,
                # forward fill in proc_modis
                # time_interp = ("existing", "nearest"),
            ),
            Feature(
                name = "modis_months_since_last_burn",
                key = "MCD64A1",
                # 0/1 is ordinal
                resampling = Resampling.nearest,
                # NO TIME INTERPOLATION, forward fill in proc_modis
                # The ceiling is a 'never burned in the record' sentinel on ~96% of
                # cells, not a duration. z-scoring that throws recent burns past
                # -10 sigma, so this stays bounded; log1p keeps the recent months legible.
                ds_norms = ["log1p", "minmax"],
            ),
        ],
        "NLCD": [
            # LndCov class not used
            # using other features in place
            Feature(
                name = "frac_imp_surface",
                key = "FctImp",
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_clip = (0.0, 1.0),
            ),
            Feature(
                name = "canopy_cover_pct",
                key = "tccconus",
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_clip = (0.0, 1.0),
            )
        ],
        "LIGHTNING": [
            Feature(
                name = "lightning_strikes",
                # daily CG strike count per 0.1 deg tile
                resampling = Resampling.nearest,
                # the product already carries a value (>=1 or a true zero) for
                # every day, so no temporal interpolation applied
                time_interp = None,
                ds_norms = ["log1p", "z_score"],
            )
        ],
        "CENSUSROADS": [
            Feature(
                name = "d_to_road",
                resampling = Resampling.nearest,
                time_interp = ("broadcast", "linear"),
                ds_clip=(0, 10000), # 10km
                ds_norms = ["log1p", "z_score"]
            )
        ],
    }


# -- order matters: later derivations consume the output of earlier ones
def drv_feat_config() -> List[Feature]:
    return [
        Feature(name="ign_next", is_label=True, 
            func="build_ignition_next",
            inputs=["usfs_burn_occ", "usfs_perimeter"],
        ),
        Feature(name="no_act_fire_mask", is_mask=True,
            func="build_no_act_fire_mask",
            inputs=["usfs_burn_occ", "usfs_perimeter"],
        ),
        Feature(name = "fire_spatial_roll",
            func = "build_fire_spatial_rolling",
            inputs=["usfs_burn_occ", "usfs_perimeter"],
            drop_inputs=["usfs_burn_occ", "usfs_perimeter"],
            ds_norms = ["log1p", "z_score"],
        ),
        # must be after ign_next
        Feature(name="ign_next_cause", is_label=True, 
            func="build_ign_next_cause",
            inputs=["usfs_burn_cause", "ign_next"],
        ),
        # must be after ign_next
        Feature(name="valid_cause_mask", is_mask=True,
            func="build_valid_cause_mask",
            inputs=["usfs_burn_cause", "ign_next"],
            drop_inputs=["usfs_burn_cause"],
        ),
        Feature(name="land_mask", is_mask=True,
            func="build_land_mask",
            inputs=["modis_water_mask"],
            drop_inputs=["modis_water_mask"],
        ),
        Feature(
            name = "ndvi_anomaly",
            func = "build_ndvi_anomaly",
            inputs=["modis_ndvi"],
            train_dependent=True,
            drop_inputs=["modis_ndvi"],
            # symmetric clip: negative anomalies (drier than climatology) carry fire-relevant signal 
            ds_clip=(-1.0, 1.0),
            ds_norms = ["z_score"],
        ),
        Feature(expand_names = ["precip_2d", "precip_5d"],
            func = "build_precip_cum",
            inputs=["precip_mm"], drop_inputs = None,
            ds_norms = ["log1p", "z_score"],
        ),
        Feature(expand_names = ["dead_fmo_100hr", "dead_fmo_1000hr"],
            func = "build_dead_fuel_derived",
            inputs=["temp_min", "temp_max", "rel_humidity", "rh_max", "precip_mm"],
            # EMCbar uses the diurnal extremes; rh_max is consumed and dropped.
            drop_inputs=["rh_max"],
            ds_norms = ["z_score"],
        ),
        Feature(name = "lightning_load",
            func = "build_lightning_load",
            inputs=["lightning_strikes"], drop_inputs = None,
            ds_norms = ["log1p", "z_score"],
        ),
        Feature(expand_names = ["wind_dir_ew", "wind_dir_ns"],
            func = "build_wind_ew_ns",
            inputs=["wind_dir"],
            drop_inputs=["wind_dir"],
        ),
        Feature(expand_names = ["lf_aspect_ew", "lf_aspect_ns"],
            func = "build_aspect_ew_ns",
            inputs=["lf_aspect"],
            drop_inputs=["lf_aspect"],
        ),
        Feature(
            name = "fosberg_fwi",
            func = 'build_ffwi',
            inputs=["temp_avg", "rel_humidity", "wind_mph"],
            ds_norms = ["z_score"],
        ),
        Feature(
            name = "doy_sin",
            func="build_doy_sin",
            inputs=["time"]
        ),
    ]

