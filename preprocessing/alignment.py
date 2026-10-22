"""
Reproject, resample, and co-register all input rasters to a common grid.

Rules:
- All outputs → TARGET_CRS (EPSG:32643, UTM Zone 43N)
- All outputs → TARGET_RESOLUTION (10 m)
- Resampling method: bilinear for continuous fields, nearest-neighbour for categorical (LULC)
- All outputs clipped to the study area bounding box
- Null/nodata pixels set to np.nan for consistent masking downstream
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from pathlib import Path
from config import TARGET_CRS, TARGET_RESOLUTION


def get_study_bbox(center_lat, center_lon, buffer_km=5):
    """
    Compute bounding box around study center with buffer.
    Returns (min_lon, min_lat, max_lon, max_lat) in WGS84.
    buffer_km: extend the study area by this many km on each side.
    """
    # approx degrees per km at the given latitude
    deg_lat = buffer_km / 111.0
    deg_lon = buffer_km / (111.0 * np.cos(np.radians(center_lat)))
    return (
        center_lon - deg_lon,
        center_lat - deg_lat,
        center_lon + deg_lon,
        center_lat + deg_lat,
    )


def align_raster(src_path, dst_path, resampling=Resampling.bilinear,
                 target_crs=TARGET_CRS, resolution=TARGET_RESOLUTION,
                 reference_transform=None, reference_shape=None,
                 nodata=np.nan):
    """
    Reproject a raster to target CRS and resolution.
    If reference_transform and reference_shape are provided, snap to that exact grid.
    All outputs are float32 with nodata=nan.
    """
    with rasterio.open(src_path) as src:
        if reference_transform is not None:
            dst_transform = reference_transform
            dst_height, dst_width = reference_shape
            dst_crs = CRS.from_string(target_crs)
        else:
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height,
                *src.bounds, resolution=resolution
            )
            dst_crs = CRS.from_string(target_crs)

        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "dtype": "float32",
            "nodata": -9999.0,
            "count": src.count,
        })

        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                )

    return dst_path


def read_band(path, band=1):
    """Read a single band from a GeoTIFF as float32 numpy array. Replace nodata with nan."""
    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        transform = src.transform
        crs = src.crs
    return data, transform, crs


def get_reference_grid(reference_raster_path):
    """Extract transform and shape from a reference raster to use as the common grid."""
    with rasterio.open(reference_raster_path) as src:
        return src.transform, (src.height, src.width), src.crs


def save_raster(data, transform, crs, dst_path, nodata=-9999.0, dtype="float32"):
    """Save a numpy array as a GeoTIFF with proper georeferencing."""
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    out = np.where(np.isnan(data), nodata, data).astype(dtype)
    with rasterio.open(dst_path, "w",
                       driver="GTiff", height=data.shape[0], width=data.shape[1],
                       count=1, dtype=dtype, crs=crs,
                       transform=transform, nodata=nodata) as dst:
        dst.write(out, 1)
