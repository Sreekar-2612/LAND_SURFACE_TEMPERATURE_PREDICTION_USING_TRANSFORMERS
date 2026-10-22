"""
Extract wind speed and wind direction from ERA5 NetCDF files.

ERA5 variables used:
  - u10: 10-metre U wind component (m/s, eastward)
  - v10: 10-metre V wind component (m/s, northward)

Derived:
  - wind_speed     = sqrt(u10^2 + v10^2)              [m/s]
  - wind_direction = atan2(u10, v10) * (180/π) + 180  [degrees, meteorological convention]

ERA5 native resolution: ~0.25° (~31 km). These are spatially uniform background fields.
After computation, resample to 10 m grid (constant field within the small study area).
"""

import numpy as np
import xarray as xr


def extract_era5_variables(era5_path, target_lat, target_lon, target_date):
    """
    Extract u10, v10, tcc, and t2m from ERA5 NetCDF for the nearest pixel to study center.
    
    Returns scalar values (spatially uniform at campus scale).
    target_date: datetime object
    """
    ds = xr.open_dataset(era5_path)
    
    # Select nearest grid cell
    u10 = float(ds["u10"].sel(
        latitude=target_lat, longitude=target_lon,
        method="nearest"
    ).sel(time=target_date, method="nearest").values)
    
    v10 = float(ds["v10"].sel(
        latitude=target_lat, longitude=target_lon,
        method="nearest"
    ).sel(time=target_date, method="nearest").values)
    
    # Total cloud cover (tcc), range [0, 1]
    tcc = float(ds["tcc"].sel(
        latitude=target_lat, longitude=target_lon,
        method="nearest"
    ).sel(time=target_date, method="nearest").values)
    
    # 2m temperature (t2m) in Kelvin, convert to Celsius
    t2m_k = float(ds["t2m"].sel(
        latitude=target_lat, longitude=target_lon,
        method="nearest"
    ).sel(time=target_date, method="nearest").values)
    t2m_c = t2m_k - 273.15
    
    wind_speed = np.sqrt(u10**2 + v10**2)
    wind_direction = (np.degrees(np.arctan2(u10, v10)) + 180) % 360
    
    return float(wind_speed), float(wind_direction), float(tcc), float(t2m_c)


def create_constant_raster(value, shape):
    """Fill a 2D raster with a constant value (for ERA5 uniform fields)."""
    return np.full(shape, value, dtype=np.float32)
