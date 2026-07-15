import numpy as np
import pandas as pd
from rasterio.transform import from_origin
import xarray as xr, rioxarray
from pyproj import CRS, Transformer


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

    # num of pixels in each direction, rounded up to multiples of 4 so the
    # model's stride-2 stages and 2x2 window partitions divide evenly
    npx_x = int(np.ceil(np.ceil(width_m / resolution) / 4) * 4)
    npx_y = int(np.ceil(np.ceil(height_m / resolution) / 4) * 4)

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