from typing import List, Optional, Tuple
import pandas as pd
import xarray as xr
import numpy as np
from scipy.ndimage import maximum_filter

class DerivedProcessor:
    """
        1. Build labels and masks
        2. Build other features. Group so extracted data can be (conditionally) dropped cleanly

    Derivations that reference a statistic of the record (rather than a single
    day) take that statistic from the train split only; `train_yrs` carries the
    split boundary in. Leaving it None uses the whole record and is only
    appropriate outside of a modelling context.
    """
    def __init__(self, train_yrs: Optional[Tuple[int, int]] = None):
        self.train_yrs = train_yrs

    def _train_slice(self, da: xr.DataArray) -> xr.DataArray:
        if self.train_yrs is None or "time" not in da.dims:
            return da
        y0, y1 = self.train_yrs
        return da.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))

    # -- Labels -----------------------------------------------------------------------------------
    def build_ignition_next(self, subds: xr.Dataset, name: str) -> xr.DataArray:
        """ 1. Burning = burn label from modis, USFS occurence, or USFS perimeter
            2. shift forward 1 day
            3. Label = NO BURN @T=t >> BURN @T=t+1
        """
        burning_t = (
            (subds["usfs_burn_occ"].fillna(0) > 0) | 
            (subds["usfs_perimeter"].fillna(0) > 0)
        )
        burning_tp1 = burning_t.shift(time=-1, fill_value=0)
        ign_next_label = ((burning_t == 0) & (burning_tp1 == 1)).astype("uint8")
        ign_next_label.name = name
        return ign_next_label
    
    def build_no_act_fire_mask(self, subds: xr.Dataset, name: str) -> xr.DataArray:
        """ 1 where nothing is burning at time t -- a cell already on fire cannot
            be a fresh ignition, so it carries no ignition supervision.
        """
        burning_t = (
            (subds["usfs_burn_occ"].fillna(0) > 0) |
            (subds["usfs_perimeter"].fillna(0) > 0)
        )
        no_act_fire_mask = (burning_t == 0).astype("uint8")
        no_act_fire_mask.name = name
        return no_act_fire_mask
    
    def build_fire_spatial_rolling(self, subds: xr.Dataset, name: str, kernel = 3, t_window = 3) -> xr.DataArray:
        """ 3x3 kernel max of active fires at time = T
            (ie, is there an active fire next to me?)
        """
        burning_t = (
            (subds["usfs_burn_occ"].fillna(0) > 0) | 
            (subds["usfs_perimeter"].fillna(0) > 0)
        )
        # rolling pads with a dtype-dependent fill value; bool input breaks under dask
        burn_rolling = (
            burning_t.astype("float32")
            .rolling(time=t_window, min_periods=1).max()
            .fillna(0)
            .astype("float32")
        )

        assert burn_rolling.dims == ("time", "y", "x"), f"Unexpected dims: {burn_rolling.dims}"

        # kernel spans spatial dims only, so per-chunk application is exact as
        # long as chunks cover the full spatial extent (rolling may have split
        # them; a 3x3 max over partial spatial chunks would miss neighbors)
        if burn_rolling.chunks is not None:
            burn_rolling = burn_rolling.chunk({"y": -1, "x": -1})

        burn_filter = xr.apply_ufunc(
            maximum_filter,
            burn_rolling,
            kwargs={"size": (1, kernel, kernel)},
            dask="parallelized",
            output_dtypes=[np.float32],
        )
        burn_filter.name = name
        return burn_filter

    def build_ign_next_cause(self, subds: xr.Dataset, name: str) -> xr.DataArray:
        """ 1. One hot encode burn cause dim (fill = -1's)
            2. shift forward 1 day
            3. Next cause = cause @T=t+1 conditioned on ignition @T=t+1
        """
        burn_cause_t = subds["usfs_burn_cause"]
        ignition_next = subds["ign_next"]
        
        # one hot, fill, shift
        cause_t_oh = burn_cause_t.argmax(dim="burn_cause")
        cause_t_oh = xr.where(burn_cause_t.sum(dim="burn_cause") > 0, cause_t_oh, -1)
        cause_tp1_oh = cause_t_oh.shift(time=-1, fill_value=-1)
        
        ign_next_cause: xr.DataArray = xr.where(ignition_next == 1, cause_tp1_oh, -1)
        ign_next_cause.name = name
        return ign_next_cause
    
    def build_valid_cause_mask(self, subds: xr.Dataset, name: str) -> xr.DataArray:
        burn_cause_t = subds["usfs_burn_cause"]
        
        # one hot, fill, shift
        cause_t_oh = burn_cause_t.argmax(dim="burn_cause")
        cause_t_oh = xr.where(burn_cause_t.sum(dim="burn_cause") > 0, cause_t_oh, -1)
        cause_tp1_oh = cause_t_oh.shift(time=-1, fill_value=-1)
        
        # !!! Mask = 1 everywhere we have a "valid cause" @t+1, 0 otherwise
        valid_cause_mask: xr.DataArray = (cause_tp1_oh >= 0).astype("uint8")
        valid_cause_mask.name = name
        return valid_cause_mask
    
    def build_land_mask(self, subds: xr.Dataset, name: str) -> xr.DataArray:
        """ 1 where the cell is land. MODIS flags deep water; anything it does
            not flag (including cells it never observed) is treated as land.
        """
        land_mask = (
            subds["modis_water_mask"].fillna(0) == 0
        ).astype("uint8")
        land_mask.name = name
        return land_mask


    # -- Other Features ---------------------------------------------------------------------------
    def build_ndvi_anomaly(self, subds: xr.Dataset, name: str) -> xr.DataArray:
        """ NDVI minus its day-of-year climatology, where the climatology is an
            average over the train years only. A climatology spanning the whole
            record would carry eval/test vegetation into every training sample
            and let each held-out day contribute to the mean it is measured
            against.
        """
        ndvi = subds['modis_ndvi']
        doy = ndvi["time"].dt.dayofyear

        # Materialize the day-of-year climatology once and subtract it via a
        # per-chunk positional lookup. A groupby subtraction shatters the time
        # axis into per-day chunks, which explodes the task graph downstream.
        # Assumes spatial dims are unchunked (the build keeps full-extent
        # spatial chunks throughout).
        ref = self._train_slice(ndvi)
        clim = ref.groupby(ref["time"].dt.dayofyear).mean("time").compute()
        clim_np = clim.values.astype("float32")

        # Days the train split never observed (Feb 29 when no train year is a
        # leap year) fall back to the nearest day-of-year that it did.
        observed = clim["dayofyear"].values
        right = np.searchsorted(observed, doy.values).clip(0, len(observed) - 1)
        left = (right - 1).clip(0, len(observed) - 1)
        take_left = np.abs(observed[left] - doy.values) <= np.abs(observed[right] - doy.values)
        pos = np.where(take_left, left, right)
        pos_da = xr.DataArray(pos, dims=("time",), coords={"time": ndvi["time"]})

        def _subtract_clim(nd, p):
            return (nd - clim_np[p.reshape(p.shape[0])]).astype("float32")

        ndvi_anom = xr.apply_ufunc(
            _subtract_clim,
            ndvi, pos_da,
            dask="parallelized",
            output_dtypes=[np.float32],
        )
        ndvi_anom.name = name
        return ndvi_anom
    


    def build_precip_cum(self, subds: xr.Dataset, names: List[str]) -> xr.Dataset:
        p2d = subds['precip_mm'].rolling(time=2, min_periods=1, center=False).sum().fillna(0)
        p5d = subds['precip_mm'].rolling(time=5, min_periods=1, center=False).sum().fillna(0)

        return xr.Dataset({ names[0]: p2d, names[1]: p5d })
         
    def build_wind_ew_ns(self, subds: xr.Dataset, names: List[str]) -> xr.Dataset:
        rads = xr.apply_ufunc(np.deg2rad, subds["wind_dir"], dask="allowed")
        val_ew = - xr.apply_ufunc(np.sin, rads, dask="allowed").astype("float32")
        val_ns = - xr.apply_ufunc(np.cos, rads, dask="allowed").astype("float32")

        return xr.Dataset({ names[0]:val_ew, names[1]:val_ns })
    
    def build_aspect_ew_ns(self, subds: xr.Dataset, names: List[str]) -> xr.Dataset:
        rads = xr.apply_ufunc(np.deg2rad, subds["lf_aspect"], dask="allowed")
        val_ew = xr.apply_ufunc(np.sin, rads, dask="allowed").astype("float32")
        val_ns = xr.apply_ufunc(np.cos, rads, dask="allowed").astype("float32")

        return xr.Dataset({ names[0]:val_ew, names[1]:val_ns })
    


    def build_ffwi(self, subds: xr.Dataset, name: str) -> xr.DataArray:
        """
        FFWI = n sqrt(1 + U^2) / 0.3002, where
        - n = 1 - 2x + 1.5x^2 - 0.5x^3
        - x = EMC/30
        - EMC:
            if H < 10%:     EMC = 0.03229 + (0.281073 * H) - (0.000578 * H% & T(Far))
            if H in 10-50%: EMC = 2.22749 + (0.160107 * H) - (0.01478 * T(Far)) 
            if H >= 50%:    EMC = 21.0606 + (0.005565 * H^2) - (0.00035 * H * T(Far)) - (0.483199 * H%) 
        """
        Tf = subds["temp_avg"]
        H = subds["rel_humidity"]
        Ws = subds["wind_mph"]

        EMC_p1   = 0.03229 + (0.281073 * H) - (0.000578 * H * Tf)
        EMC_p1p5 = 2.22749 + (0.160107 * H) - (0.01478 * Tf) 
        EMC_p5   = 21.0606 + (0.005565 * (H ** 2)) - (0.00035 * H * Tf) - (0.483199 * H)

        EMC = xr.where(
            H < 10, EMC_p1, 
            xr.where(H < 50, EMC_p1p5, EMC_p5)
        )
        x = EMC.clip(0.0, 30.0) / 30.0
        eta = 1 - (2.0 * x) + 1.5 * (x ** 2) - 0.5 * (x ** 3)
        ffwi = eta * np.sqrt(1.0 + Ws ** 2) / 0.3002
        ffwi.name = name
        return ffwi
        


    def build_doy_sin(self, subds: xr.Dataset, name: str, gridref: xr.DataArray) -> xr.DataArray:
        # time index to numpy
        time_index = pd.DatetimeIndex(subds.indexes["time"])
        doy = time_index.dayofyear.to_numpy(dtype="float32")

        # sin encoding on 0, 2pi
        phase = 2.0 * np.pi * (doy - 1.0) / 365.0
        time_signal = xr.DataArray(
            np.sin(phase).astype("float32"),
            dims=("time",),
            coords={"time": time_index},
        )

        # Broadcast against a (time, y, x) variable from the dataset so the
        # result inherits its layout (and stays lazy when inputs are dask)
        template = next(
            da for da in subds.data_vars.values() if da.dims == ("time", "y", "x")
        )
        doy_map = (time_signal * xr.ones_like(template, dtype="float32"))
        doy_map = doy_map.transpose("time", "y", "x")
        doy_map.name = name
        return doy_map
    
    # def build_wui_index(self, subds: xr.Dataset, name: str, wildland_ixs = [4, 5, 7]) -> xr.DataArray:
    #     """
    #         WUI = 1 if any wildlife class
    #     """
    #     pop_norm: xr.DataArray = subds["pop_density"]
    #     lcov_class: xr.DataArray = subds["lcov_class"]
    #     wild_ohe = lcov_class[..., list(wildland_ixs)]

    #     # Reduce over the ohe to get a [0, 1] wildland value on 2d grid
    #     wildland_mask = wild_ohe.max(dim="lc_class_index")  

    #     # population is HEAVILY skewed towards cities
    #     # add sigmoid to norm'd population to further smooth out non-linearity
    #     # smoothed WUI as sigmopoid of norm'd population * wildland mask
    #     wui_smooth = wildland_mask * (1.0 / (1.0 + np.exp(-1.0 * pop_norm)))
    #     wui_smooth.name = name
    #     return wui_smooth
        
    
    