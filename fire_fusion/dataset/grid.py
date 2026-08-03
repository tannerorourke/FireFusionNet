import numpy as np
import pandas as pd
from rasterio.transform import from_origin
import xarray as xr, rioxarray
from pyproj import CRS, Transformer

# -- sized for a 4-stage encoder with a 2-cell attention window, the deepest in use
GRID_ALIGN = 32


def season_time_index(
    start_date: str,
    end_date: str,
    season_months=None,
    halo_lead_days: int = 40,
    halo_trail_days: int = 10,
) -> pd.DatetimeIndex:
    """
    Days to extract. With `season_months` set, this is the union of one block
    per year covering the season plus a halo on either side; without it, every
    day between the bounds.

    The halo exists so temporal derivations (decayed lightning load, rolling
    precipitation, the forward ignition horizon) enter the supervised window
    with real history behind them instead of restarting at zero. Halo days are
    dropped again before the splits are written, so they are never supervised.
    """
    full = pd.date_range(start_date, end_date, freq="D")
    if season_months is None:
        return full

    m0, m1 = season_months
    lead = pd.Timedelta(days=halo_lead_days)
    trail = pd.Timedelta(days=halo_trail_days)

    keep = pd.DatetimeIndex([])
    for year in sorted(full.year.unique()):
        block_start = pd.Timestamp(year=year, month=m0, day=1) - lead
        # month end without calendar arithmetic: first of the next month, minus a day
        month_end = pd.Timestamp(year=year + (m1 // 12), month=(m1 % 12) + 1, day=1)
        block_end = month_end - pd.Timedelta(days=1) + trail
        keep = keep.union(full[(full >= block_start) & (full <= block_end)])

    return keep


# -- which days of an extraction index carry supervision, i.e. are not halo
def supervised_mask(time_index, season_months) -> np.ndarray:
    if season_months is None:
        return np.ones(len(time_index), dtype=bool)
    m0, m1 = season_months
    months = np.asarray(pd.DatetimeIndex(time_index).month)
    return (months >= m0) & (months <= m1)


def create_coordinate_grid(
    time_index,
    resolution: float,
    lat_bounds = (45.4, 49.1),
    lon_bounds = (-124.8, -117.0),
    crs = "EPSG:32610" # UT Zone 10N (better for single state)
) -> xr.DataArray:
    """
    Defines coordinate grid to place features on top of
    - subclasses xarray.DataArray
    """
    min_lat, max_lat = min(lat_bounds), max(lat_bounds)
    min_lon, max_lon = min(lon_bounds), max(lon_bounds)

    crs_obj = CRS.from_string(crs)
    n_days = time_index.shape[0]

    # x=lon, y=lat
    transformer = Transformer.from_crs(
        "EPSG:4326", 
        crs_to=crs_obj, 
        always_xy=True
    )
    # UTM edges bow with latitude; take the envelope of all four corners so the
    # grid fully covers the requested lat/lon rectangle
    corner_xs, corner_ys = transformer.transform(
        [min_lon, max_lon, min_lon, max_lon],
        [min_lat, min_lat, max_lat, max_lat],
    )
    min_x, max_x = min(corner_xs), max(corner_xs)
    min_y, max_y = min(corner_ys), max(corner_ys)

    width_m = max_x - min_x
    height_m = max_y - min_y

    # Crop origins are multiples of the model's total stride times its attention
    # window. Where the extent is not itself such a multiple, the last aligned origin
    # stops short of the far edge and a strip of the domain becomes unreadable during
    # crop training while still being scored at evaluation.
    npx_x = int(np.ceil(np.ceil(width_m / resolution) / GRID_ALIGN) * GRID_ALIGN)
    npx_y = int(np.ceil(np.ceil(height_m / resolution) / GRID_ALIGN) * GRID_ALIGN)

    # Snap upper-right corner to exact pixel grid
    max_x_aligned = min_x + npx_x * resolution
    max_y_aligned = min_y + npx_y * resolution

    transform = from_origin(
        min_x,            # west (left)
        max_y_aligned,    # north (top)
        xsize=resolution,  # pixel width
        ysize=resolution  # pixel height
    )

    # MASTER pixel CENTERS coordinates in UTM meters
    y_coordinates = max_y_aligned - (np.arange(npx_y) + 0.5) * resolution
    x_coordinates = min_x + (np.arange(npx_x) + 0.5) * resolution

    data = np.zeros((n_days, npx_y, npx_x), dtype=np.float32)

    grid = xr.DataArray(
        data = data,
        dims = ("time", "y", "x"),
        coords= { "time": time_index, "y": y_coordinates, "x": x_coordinates },
        name = "master_grid",
        attrs={
            'resolution': resolution,
            'time_index': time_index,
            'years': sorted(time_index.year.unique().to_list()),
            'y_coordinates': y_coordinates,
            'x_coordinates': x_coordinates,
            'y_min': float(y_coordinates.min()), 'y_max': float(y_coordinates.max()),
            'x_min': float(x_coordinates.min()), 'x_max': float(x_coordinates.max()),
            'lat_min': min_lat, 'lat_max': max_lat, 
            'lon_min': min_lon, 'lon_max': max_lon
        }
    )

    grid.attrs['template'] = grid.isel(time=0)

    # attach CRS and transform for rioxarray
    grid = grid.rio.write_crs(crs_obj)
    grid = grid.rio.write_transform(transform)

    print(f"(T, Y, X) Grid Created:")
    print(f"- (Y, X) pixels: ({npx_y}, {npx_x})")
    print(f"- tot days in date range: {len(time_index)}")
    return grid