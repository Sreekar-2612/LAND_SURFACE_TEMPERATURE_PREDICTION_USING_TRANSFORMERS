"""
Derive terrain parameters from ALOS-PALSAR DEM:
  - Elevation (raw DEM in metres)
  - Slope (degrees)
  - Aspect (degrees, 0–360, 0=North)
  - Curvature (planform curvature, 1/m)

Uses richdem for accurate terrain analysis.
All outputs must be aligned to the common 10 m grid before use.
"""

import numpy as np
import richdem as rd
from pathlib import Path


def compute_terrain_derivatives(dem_array, resolution=10.0):
    """
    Compute slope, aspect, and curvature from a DEM numpy array.
    
    Args:
        dem_array:   2D float32 numpy array of elevation in metres
        resolution:  grid spacing in metres
    
    Returns:
        dict with keys: elevation, slope, aspect, curvature
        All values are float32 numpy arrays.
    """
    # Mask nodata
    dem_clean = np.where(np.isnan(dem_array), -9999.0, dem_array).astype(np.float64)
    
    # Convert to richdem format
    rda = rd.rdarray(dem_clean, no_data=-9999.0)
    rda.geotransform = (0, resolution, 0, 0, 0, -resolution)  # placeholder
    
    # Slope in degrees
    slope_rda = rd.TerrainAttribute(rda, attrib='slope_degrees')
    slope = np.array(slope_rda).astype(np.float32)
    
    # Aspect in degrees (0–360)
    aspect_rda = rd.TerrainAttribute(rda, attrib='aspect')
    aspect = np.array(aspect_rda).astype(np.float32)
    aspect = np.where(aspect < 0, aspect + 360, aspect)   # normalise to 0–360
    
    # Profile curvature (second derivative)
    curv_rda = rd.TerrainAttribute(rda, attrib='curvature')
    curvature = np.array(curv_rda).astype(np.float32)
    
    # Elevation as-is
    elevation = np.where(dem_array == -9999.0, np.nan, dem_array).astype(np.float32)
    slope     = np.where(slope == -9999.0, np.nan, slope)
    aspect    = np.where(aspect == -9999.0, np.nan, aspect)
    curvature = np.where(curvature == -9999.0, np.nan, curvature)
    
    return {
        "elevation": elevation,
        "slope":     slope,
        "aspect":    aspect,
        "curvature": curvature,
    }
