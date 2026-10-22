"""
Compute NDVI, NDBI, NDWI, and Albedo from Landsat 8/9 Level-2 surface reflectance bands.

Landsat Level-2 scale factors:
  Surface reflectance = (DN * 0.0000275) + (-0.2)   [Collection 2 Level-2]
  Apply this scaling before computing indices.

Band mapping for Landsat 8/9 OLI:
  B2 = Blue, B3 = Green, B4 = Red, B5 = NIR, B6 = SWIR1, B7 = SWIR2
"""

import numpy as np


def apply_landsat_c2_scale(dn_array):
    """Apply Landsat Collection 2 Level-2 surface reflectance scaling."""
    return dn_array * 0.0000275 + (-0.2)


def compute_ndvi(nir, red):
    """
    NDVI = (NIR - Red) / (NIR + Red)
    Range: -1 to +1. Higher = more/healthier vegetation.
    """
    denom = nir + red
    ndvi = np.where(denom != 0, (nir - red) / denom, 0.0)
    return ndvi.astype(np.float32)


def compute_ndbi(swir1, nir):
    """
    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    Range: -1 to +1. Positive values indicate built-up areas.
    """
    denom = swir1 + nir
    ndbi = np.where(denom != 0, (swir1 - nir) / denom, 0.0)
    return ndbi.astype(np.float32)


def compute_ndwi(green, nir):
    """
    NDWI = (Green - NIR) / (Green + NIR)
    Range: -1 to +1. Positive values indicate water bodies / surface moisture.
    """
    denom = green + nir
    ndwi = np.where(denom != 0, (green - nir) / denom, 0.0)
    return ndwi.astype(np.float32)


def compute_albedo(blue, red, nir, swir1, swir2):
    """
    Broadband surface albedo using Liang (2001) formula for Landsat:
    Albedo = 0.356*B2 + 0.130*B4 + 0.373*B5 + 0.085*B6 + 0.072*B7 - 0.0018
    All inputs are at-surface reflectance (0–1 range after scaling).
    """
    albedo = (0.356 * blue +
              0.130 * red +
              0.373 * nir +
              0.085 * swir1 +
              0.072 * swir2 -
              0.0018)
    albedo = np.clip(albedo, 0.0, 1.0)
    return albedo.astype(np.float32)


def compute_all_indices(bands_dict):
    """
    bands_dict: dict with keys matching LANDSAT_BANDS config.
    All arrays must be at-surface reflectance (apply_landsat_c2_scale first).
    Returns dict with keys: ndvi, ndbi, ndwi, albedo.
    """
    return {
        "ndvi":   compute_ndvi(bands_dict["nir"], bands_dict["red"]),
        "ndbi":   compute_ndbi(bands_dict["swir1"], bands_dict["nir"]),
        "ndwi":   compute_ndwi(bands_dict["green"], bands_dict["nir"]),
        "albedo": compute_albedo(bands_dict["blue"], bands_dict["red"],
                                 bands_dict["nir"], bands_dict["swir1"],
                                 bands_dict["swir2"]),
    }
