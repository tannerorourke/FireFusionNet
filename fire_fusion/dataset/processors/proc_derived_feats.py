from typing import List
import pandas as pd
import xarray as xr
import numpy as np
from scipy.ndimage import maximum_filter

class DerivedProcessor:
    """
        1. Build labels and masks
        2. Build other features. Group so extracted data can be (conditionally) dropped cleanly
    """
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
    
    def build_act_fire_mask(self, subds: xr.Dataset, name: str) -> xr.DataArray:
        burning_t = (
            (subds["usfs_burn_occ"].fillna(0) > 0) | 
            (subds["usfs_perimeter"].fillna(0) > 0)
        )
        act_ign_mask = (burning_t == 0).astype("uint8")
        act_ign_mask.name = name
        return act_ign_mask
    
    def build_fire_spatial_rolling(self, subds: xr.Dataset, name: str, kernel = 3, t_window = 3) -> xr.DataArray:
        """ 3x3 kernel max of active fires at time = T
            (ie, is there an active fire next to me?)
        """
        burning_t = (
            (subds["usfs_burn_occ"].fillna(0) > 0) | 
            (subds["usfs_perimeter"].fillna(0) > 0)
        )
        burn_rolling = burning_t.rolling(time=t_window, min_periods=1).max().fillna(0)

        assert burn_rolling.dims == ("time", "y", "x"), f"Unexpected dims: {burn_rolling.dims}"

        burn_filter = maximum_filter(burn_rolling.values.astype(np.uint8), size=(1, kernel, kernel))
        return xr.DataArray(
            burn_filter,
            dims=burn_rolling.dims,
            coords=burn_rolling.coords,
            name=name,
        )

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
    
    def build_water_mask(self, subds: xr.Dataset, name: str) -> xr.DataArray:
        water_mask = (
            subds["modis_water_mask"].fillna(0) > 0
        )
        water_mask.name = name
        return water_mask


    # -- Other Features ---------------------------------------------------------------------------
    def build_ndvi_anomaly(self, subds: xr.Dataset, name: str) -> xr.DataArray:
        ndvi_by_doy = subds['modis_ndvi'].groupby("time.dayofyear")
        ndvi_anom = ndvi_by_doy - ndvi_by_doy.mean("time")
        ndvi_anom.name = name
        return ndvi_anom
    


    def build_precip_cum(self, subds: xr.Dataset, names: List[str]) -> xr.Dataset:
        p2d = subds['precip_mm'].rolling(time=2, min_periods=1, center=False).sum().fillna(0)
        p5d = subds['precip_mm'].rolling(time=5, min_periods=1, center=False).sum().fillna(0)

        return xr.Dataset({ names[0]: p2d, names[1]: p5d })
         
    def build_wind_ew_ns(self, subds: xr.Dataset, names: List[str]) -> xr.Dataset:
        rads = xr.apply_ufunc(np.deg2rad, subds["wind_dir"])
        val_ew = - xr.apply_ufunc(np.sin, rads).astype("float32")
        val_ns = - xr.apply_ufunc(np.cos, rads).astype("float32")

        return xr.Dataset({ names[0]:val_ew, names[1]:val_ns })
    
    def build_aspect_ew_ns(self, subds: xr.Dataset, names: List[str]) -> xr.Dataset:
        rads = xr.apply_ufunc(np.deg2rad, subds["lf_aspect"])
        val_ew = xr.apply_ufunc(np.sin, rads).astype("float32")
        val_ns = xr.apply_ufunc(np.cos, rads).astype("float32")

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
        """ NOTE: Used LLM for help debugging this function"""

        # time index to numpy
        time_index = pd.DatetimeIndex(subds.indexes["time"])
        doy = time_index.dayofyear.to_numpy(dtype="float32")

        # sin encoding on 0, 2pi
        phase = 2.0 * np.pi * (doy - 1.0) / 365.0
        time_signal = np.sin(phase).astype("float32")

        # Broadcast over spatial grid
        ny = gridref.sizes["y"]
        nx = gridref.sizes["x"]
        data = np.broadcast_to(
            time_signal[:, None, None], 
            shape=(len(time_index), ny, nx)
        ).astype("float32")

        # Wrap as T, Y, X feature
        return xr.DataArray(
            data=data,
            dims=("time", "y", "x"),
            coords={
                "time": time_index,
                "y": gridref["y"].values, "x": gridref["x"].values,
            },
            name=name,
        )
    
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
        
    
    