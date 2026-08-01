from typing import List, Optional, Tuple
import pandas as pd
import xarray as xr
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import lfilter

from fire_fusion.config.feature_config import IGN_HORIZON_DAYS

# Target working size of one spatial block's full time series in the lightning
# IIR. lfilter allocates an equal-sized output, so a task costs roughly double.
LIGHTNING_BLOCK_BYTES = 256 * 1024 ** 2

# NFDRS 1978 dead fuel moisture (Bradshaw/Deeming, GTR INT-169; response factors
# and boundary constants match the WIMS/FireFamilyPlus operational code). The
# EMC branches and boundary terms are hard-wired to Fahrenheit, so temperature
# inputs must arrive in F. Per-day response fractions already encode the
# 24h/timelag ratio and are applied once per daily step.
FM100_RESPONSE = 1.0 - 0.87 * np.exp(-0.24)    # 0.315634
FM1000_RESPONSE = 1.0 - 0.82 * np.exp(-0.168)  # 0.306811
FM1000_SEED = 30.0                              # operational spin-up default (%)
# Full-time series working size of one fuel-moisture spatial block. The scan
# holds several equal-sized arrays (boundaries + output) per task.
FUEL_BLOCK_BYTES = 96 * 1024 ** 2
IGN_BLOCK_BYTES = 128 * 1024 ** 2


def _emc(Tf: np.ndarray, H: np.ndarray) -> np.ndarray:
    """ Equilibrium moisture content (% gravimetric) from temperature (F) and
        relative humidity (%). Three humidity branches; the mid branch carries no
        H*T cross term. Constants are the NFDRS operational values.
    """
    emc_low = 0.03229 + 0.281073 * H - 0.000578 * H * Tf
    emc_mid = 2.22749 + 0.160107 * H - 0.014784 * Tf
    emc_high = 21.0606 + 0.005565 * H * H - 0.00035 * H * Tf - 0.483199 * H
    return np.where(H < 10.0, emc_low, np.where(H < 50.0, emc_mid, emc_high))


def _pdur_hours(precip_mm: np.ndarray) -> np.ndarray:
    """ Precipitation duration (hours) estimated from daily precip amount.
        NFDRS ingests observed duration, not amount; with amount-only daily
        records this steps duration with rainfall and caps at the 8h
        state-of-weather reporting limit. This is a modeling choice, not an
        NFDRS constant.
    """
    conds = [precip_mm <= 0.0, precip_mm < 2.5, precip_mm < 5.0, precip_mm < 10.0, precip_mm < 25.0]
    vals = [0.0, 1.0, 2.0, 4.0, 6.0]
    return np.select(conds, vals, default=8.0).astype("float32")


def _fuel_boundaries(tmin_f, tmax_f, hmin, hmax, precip_mm):
    """ Daily 100h and 1000h boundary conditions (%). EMCbar is the 1978
        simple average of the hot-dry and cool-moist EMC endpoints; the wet
        term differs between the two timelag classes.
    """
    emc_hot = _emc(tmax_f, hmin)    # hot, dry -> lowest moisture
    emc_cool = _emc(tmin_f, hmax)   # cool, moist -> highest moisture
    emcbar = 0.5 * (emc_hot + emc_cool)
    pdur = _pdur_hours(precip_mm)
    d100 = ((24.0 - pdur) * emcbar + pdur * (0.5 * pdur + 41.0)) / 24.0
    d1000 = ((24.0 - pdur) * emcbar + pdur * (2.7 * pdur + 76.0)) / 24.0
    return d100.astype("float32"), d1000.astype("float32")


def _fm100_scan(tmin_f, tmax_f, hmin, hmax, precip_mm, reset):
    """ 100h fuel moisture recursion along axis 0 (time). `reset` marks the first
        day of each contiguous time block (year gaps in a seasonal index); the
        recursion reseeds at the day-0 boundary there rather than carrying the
        prior block's state across the gap. The 100h class forgets its seed
        within ~2 weeks.
    """
    d100, _ = _fuel_boundaries(tmin_f, tmax_f, hmin, hmax, precip_mm)
    out = np.empty_like(d100)
    prev = None
    for t in range(d100.shape[0]):
        if prev is None or reset[t]:
            cur = d100[t]
        else:
            cur = prev + (d100[t] - prev) * FM100_RESPONSE
        out[t] = cur
        prev = cur
    return np.clip(out, 0.0, 60.0).astype("float32")


def _fm1000_scan(tmin_f, tmax_f, hmin, hmax, precip_mm, reset):
    """ 1000h fuel moisture recursion along axis 0. The driver is a 7-day running
        mean of the 1000h boundary condition, so the class carries ~6 weeks of
        memory; the running mean is held within a block and cleared at each
        reset. Seeded at the operational 30% default.
    """
    _, d1000 = _fuel_boundaries(tmin_f, tmax_f, hmin, hmax, precip_mm)
    out = np.empty_like(d1000)
    prev = None
    window: List[np.ndarray] = []
    for t in range(d1000.shape[0]):
        if prev is None or reset[t]:
            prev = np.full_like(d1000[t], FM1000_SEED)
            window = []
        window.append(d1000[t])
        if len(window) > 7:
            window.pop(0)
        dbar = sum(window) / len(window)
        cur = prev + (dbar - prev) * FM1000_RESPONSE
        out[t] = cur
        prev = cur
    return np.clip(out, 0.0, 60.0).astype("float32")


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
    def build_ignition_next(self, subds: xr.Dataset, name: str,
                            horizon: int = IGN_HORIZON_DAYS) -> xr.DataArray:
        """ 1. Burning = burn label from USFS occurence or USFS perimeter
            2. Label = NO BURN @T=t and BURN on any of T=t+1 .. t+horizon
            A cell already alight cannot be a fresh ignition, so the positive is
            gated on the cell being clear today.
        """
        burning_t = (
            (subds["usfs_burn_occ"].fillna(0) > 0) |
            (subds["usfs_perimeter"].fillna(0) > 0)
        )
        # the forward window shifts along time only, so blocking y/x is exact and
        # keeps each task off the full-grid footprint that OOMs at fine resolution
        if burning_t.chunks is not None:
            nt = burning_t.sizes["time"]
            edge = max(1, int(np.sqrt(IGN_BLOCK_BYTES / nt)))
            burning_t = burning_t.chunk({
                "time": -1,
                "y": min(burning_t.sizes["y"], edge),
                "x": min(burning_t.sizes["x"], edge),
            })
        future_burn = burning_t.shift(time=-1, fill_value=False)
        for k in range(2, horizon + 1):
            future_burn = future_burn | burning_t.shift(time=-k, fill_value=False)
        ign_next_label = ((~burning_t) & future_burn).astype("uint8")
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

    def _first_future_cause(self, burn_cause_t: xr.DataArray, horizon: int) -> xr.DataArray:
        """ Cause id (0..K-1, or -1 for none) of the earliest day carrying a valid
            cause within T=t+1 .. t+horizon. Days are scanned nearest-first so the
            closest caused ignition wins.
        """
        cause_t = burn_cause_t.argmax(dim="burn_cause")
        cause_t = xr.where(burn_cause_t.sum(dim="burn_cause") > 0, cause_t, -1)
        if cause_t.chunks is not None:
            nt = cause_t.sizes["time"]
            edge = max(1, int(np.sqrt(IGN_BLOCK_BYTES / (nt * 8))))
            cause_t = cause_t.chunk({
                "time": -1,
                "y": min(cause_t.sizes["y"], edge),
                "x": min(cause_t.sizes["x"], edge),
            })

        next_cause = cause_t.shift(time=-1, fill_value=-1)
        for k in range(2, horizon + 1):
            cause_tk = cause_t.shift(time=-k, fill_value=-1)
            next_cause = xr.where(next_cause >= 0, next_cause, cause_tk)
        return next_cause

    def build_ign_next_cause(self, subds: xr.Dataset, name: str,
                             horizon: int = IGN_HORIZON_DAYS) -> xr.DataArray:
        """ Cause of the earliest caused ignition within the forward window,
            conditioned on a positive ignition label at the same cell.
        """
        ignition_next = subds["ign_next"]
        next_cause = self._first_future_cause(subds["usfs_burn_cause"], horizon)

        ign_next_cause: xr.DataArray = xr.where(ignition_next == 1, next_cause, -1)
        ign_next_cause.name = name
        return ign_next_cause

    def build_valid_cause_mask(self, subds: xr.Dataset, name: str,
                               horizon: int = IGN_HORIZON_DAYS) -> xr.DataArray:
        next_cause = self._first_future_cause(subds["usfs_burn_cause"], horizon)

        # Mask = 1 wherever a valid cause lands anywhere in the forward window
        valid_cause_mask: xr.DataArray = (next_cause >= 0).astype("uint8")
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

    def build_dead_fuel_derived(self, subds: xr.Dataset, names: List[str]) -> xr.Dataset:
        """ NFDRS 100h and 1000h dead fuel moisture (%). Inputs arrive in NFDRS
            units -- temperature in F, humidity in %, precip in mm. EMCbar uses
            the diurnal extremes: rel_humidity carries the daily-min (hot-dry)
            branch, rh_max the daily-max (cool-moist) branch.

            The recursions run along time only, so splitting y/x is exact; the
            staging cube's time-chunked full-extent layout is rechunked to the
            opposite (full time, blocked space) to bound each task.
        """
        tmin = subds["temp_min"].fillna(60.0).astype("float32")
        tmax = subds["temp_max"].fillna(60.0).astype("float32")
        hmin = subds["rel_humidity"].fillna(50.0).clip(0.0, 100.0).astype("float32")
        hmax = subds["rh_max"].fillna(50.0).clip(0.0, 100.0).astype("float32")
        precip = subds["precip_mm"].fillna(0.0).astype("float32")

        # First day of each contiguous block. A seasonally windowed index carries
        # year-to-year gaps; the recursions reseed there rather than carry the
        # prior block's fuel state across the break.
        t = pd.DatetimeIndex(tmin.indexes["time"])
        gap = np.diff(t.values).astype("timedelta64[D]").astype("int64") != 1
        reset = np.concatenate([[True], gap])

        inputs = [tmin, tmax, hmin, hmax, precip]
        if tmin.chunks is not None:
            nt = tmin.sizes["time"]
            edge = max(1, int(np.sqrt(FUEL_BLOCK_BYTES / (nt * 4))))
            chunking = {"time": -1, "y": min(tmin.sizes["y"], edge), "x": min(tmin.sizes["x"], edge)}
            inputs = [a.chunk(chunking) for a in inputs]

        fm100 = xr.apply_ufunc(
            _fm100_scan, *inputs,
            kwargs={"reset": reset},
            dask="parallelized",
            output_dtypes=[np.float32],
        )
        fm1000 = xr.apply_ufunc(
            _fm1000_scan, *inputs,
            kwargs={"reset": reset},
            dask="parallelized",
            output_dtypes=[np.float32],
        )
        return xr.Dataset({ names[0]: fm100, names[1]: fm1000 })
         
    def build_lightning_load(self, subds: xr.Dataset, name: str, half_life: float = 4.0) -> xr.DataArray:
        """ Exponentially-decayed running sum of daily CG strike counts:
                load[t] = strikes[t] + alpha * load[t-1],  alpha = 0.5 ** (1/half_life)
            A strike's contribution halves every `half_life` days, approximating
            how long a lightning-lit fire can hold before it is discovered.

            The recursion runs along time only, so splitting y/x is exact. On a
            seasonally windowed index the filter also carries state across the
            year boundary; the pre-season halo is sized so that carry-over has
            decayed below 0.1% by the first supervised day.
        """
        strikes = subds["lightning_strikes"].fillna(0.0).astype("float32")
        alpha = float(0.5 ** (1.0 / half_life))

        # The recursion needs each cell's full time series in one chunk. The
        # staging cube chunks time and holds full spatial extent, so this is the
        # exact opposite layout and a naive time rechunk collapses the array into
        # a single block -- ~10 GB at 250m, before lfilter's equal-sized output.
        # Blocking y/x keeps each task bounded regardless of grid size.
        if strikes.chunks is not None:
            nt = strikes.sizes["time"]
            edge = max(1, int(np.sqrt(LIGHTNING_BLOCK_BYTES / (nt * 4))))
            strikes = strikes.chunk({
                "time": -1,
                "y": min(strikes.sizes["y"], edge),
                "x": min(strikes.sizes["x"], edge),
            })

        def _decay(arr):
            return lfilter([1.0], [1.0, -alpha], arr, axis=0).astype("float32")

        load = xr.apply_ufunc(
            _decay,
            strikes,
            dask="parallelized",
            output_dtypes=[np.float32],
        )
        load.name = name
        return load

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
        
    
    