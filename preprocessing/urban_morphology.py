"""
Derive urban morphology features from SRTM DSM:
  - Building Height: nDSM = DSM - bare-earth DEM (proxy)
  - Sky View Factor (SVF): proportion of visible sky from ground level

For building height:
  SRTM DSM is used as a proxy for above-ground heights.
  True nDSM = DSM - DEM (bare earth). If only SRTM available, use as proxy.
  
For SVF:
  Computed via hemispherical viewshed analysis.
  SVF = 1 − (fraction of sky blocked by surrounding structures)
  Range: 0 (fully obstructed by tall buildings) to 1 (fully open)
  
  Simplified SVF approximation for raster data:
  For each pixel, analyse the local neighbourhood to estimate sky blockage
  based on relative building height differences.
"""

import numpy as np
from scipy.ndimage import uniform_filter, minimum_filter


def compute_building_height(dsm, dem=None):
    """
    Compute normalised Digital Surface Model (nDSM) as building height proxy.
    
    If dem is provided: building_height = DSM - DEM (true nDSM)
    If dem is None: use local minimum filter as bare-earth proxy
    
    Output clipped to [0, 150] metres.
    """
    if dem is not None:
        ndsm = dsm - dem
    else:
        # Approximate bare earth as minimum in local 50m window
        window_px = max(1, int(50 / 10))  # 50m / 10m resolution
        bare_earth = minimum_filter(dsm, size=window_px)
        ndsm = dsm - bare_earth
    
    ndsm = np.clip(ndsm, 0.0, 150.0)
    return ndsm.astype(np.float32)


def compute_svf_simplified(dsm, resolution=10.0, search_radius_m=200):
    """
    Simplified Sky View Factor computation using radial elevation differences.
    
    Method: For each pixel, compute the maximum elevation angle to surrounding
    pixels within the search radius, averaged over 8 cardinal+intercardinal directions.
    SVF = 1 - mean(sin(max_elevation_angle_per_direction))
    
    This is a computational approximation. For full accuracy, use a dedicated
    tool such as SAGA GIS r.skyview or WhiteboxTools SVF function.
    
    Args:
        dsm:              2D float32 array (elevation in metres)
        resolution:       pixel size in metres
        search_radius_m:  radius to search for obstructions (metres)
    
    Returns:
        SVF array, float32, range 0–1
    """
    radius_px = int(search_radius_m / resolution)
    
    # 8 directions: N, NE, E, SE, S, SW, W, NW
    directions = [
        (0, 1), (-1, 1), (-1, 0), (-1, -1),
        (0, -1), (1, -1), (1, 0), (1, 1)
    ]
    
    h, w = dsm.shape
    svf_accumulator = np.zeros((h, w), dtype=np.float32)
    
    for dy, dx in directions:
        max_angle = np.zeros((h, w), dtype=np.float32)
        
        for step in range(1, radius_px + 1):
            # Shift DSM in this direction
            oy = dy * step
            ox = dx * step
            dist_m = step * resolution * np.sqrt(dy**2 + dx**2)
            
            shifted = np.roll(np.roll(dsm, oy, axis=0), ox, axis=1)
            elev_diff = shifted - dsm
            angle = np.degrees(np.arctan(elev_diff / dist_m))
            max_angle = np.maximum(max_angle, angle)
        
        svf_accumulator += (1 - np.sin(np.radians(np.clip(max_angle, 0, 90))))
    
    svf = svf_accumulator / 8.0
    svf = np.clip(svf, 0.0, 1.0)
    return svf.astype(np.float32)
