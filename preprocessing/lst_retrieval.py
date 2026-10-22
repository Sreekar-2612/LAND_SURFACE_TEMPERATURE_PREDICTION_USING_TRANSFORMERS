"""
Land Surface Temperature (LST) retrieval using the Mono-Window Algorithm.

Steps:
  1. Convert Band 10 DN → Top-of-Atmosphere Brightness Temperature (BT)
  2. Estimate Land Surface Emissivity (LSE) from NDVI
  3. Apply Mono-Window formula to compute LST

Landsat Collection 2 Level-2 thermal scaling:
  BT (Kelvin) = DN * 0.00341802 + 149.0   [for Band 10, Collection 2]

Mono-Window Algorithm (Sobrino et al., 2004 / Jiménez-Muñoz & Sobrino, 2003):
  LST = BT / (1 + (λ * BT / ρ) * ln(ε))

where:
  BT  = brightness temperature in Kelvin
  λ   = wavelength of emitted radiance (Band 10 centre = 10.895e-6 m)
  ρ   = h·c/σ = 0.01438 m·K  (Planck's radiation constants ratio)
  ε   = land surface emissivity
  LST output in °C (subtract 273.15)
"""

import numpy as np
from config import LST_WAVELENGTH_BAND10, RHO, NDVI_THRESH_BARE, NDVI_THRESH_VEG
from config import EMISSIVITY_BARE, EMISSIVITY_VEG, EMISSIVITY_MIX_C


def dn_to_brightness_temperature(tir_dn):
    """
    Convert Landsat Collection 2 Level-2 Band 10 DN to brightness temperature.
    Output: BT in Kelvin
    """
    return (tir_dn * 0.00341802 + 149.0).astype(np.float32)


def compute_emissivity_from_ndvi(ndvi):
    """
    Estimate land surface emissivity (LSE) from NDVI using threshold method.
    - NDVI < 0.2  → bare soil: ε = 0.97
    - NDVI > 0.5  → full vegetation: ε = 0.99
    - 0.2 ≤ NDVI ≤ 0.5 → mixed: ε = ε_veg * Pv + ε_bare * (1-Pv) + C
      where Pv = ((NDVI - 0.2)/(0.5 - 0.2))^2
    """
    Pv = np.clip((ndvi - NDVI_THRESH_BARE) / (NDVI_THRESH_VEG - NDVI_THRESH_BARE), 0, 1) ** 2

    emissivity = np.where(
        ndvi < NDVI_THRESH_BARE,
        EMISSIVITY_BARE,
        np.where(
            ndvi > NDVI_THRESH_VEG,
            EMISSIVITY_VEG,
            EMISSIVITY_VEG * Pv + EMISSIVITY_BARE * (1 - Pv) + EMISSIVITY_MIX_C
        )
    )
    return emissivity.astype(np.float32)


def mono_window_lst(bt_kelvin, emissivity):
    """
    Apply Mono-Window Algorithm to compute LST.
    
    Formula: LST = BT / (1 + (λ * BT / ρ) * ln(ε))
    
    Args:
        bt_kelvin:   brightness temperature array in Kelvin
        emissivity:  surface emissivity array (0–1)
    
    Returns:
        LST in degrees Celsius
    """
    wavelength = LST_WAVELENGTH_BAND10    # 10.895e-6 m
    rho = RHO                             # 0.01438 m·K

    # Avoid log(0): clamp emissivity
    emissivity_safe = np.clip(emissivity, 0.001, 1.0)
    
    denominator = 1.0 + (wavelength * bt_kelvin / rho) * np.log(emissivity_safe)
    lst_kelvin = bt_kelvin / denominator
    lst_celsius = lst_kelvin - 273.15

    return lst_celsius.astype(np.float32)


def retrieve_lst(tir_band10_dn, ndvi):
    """
    Full LST retrieval pipeline from raw DN to °C.
    
    Args:
        tir_band10_dn: raw Band 10 DN array (uint16 from Landsat C2 L2 product)
        ndvi:          NDVI array (same shape, already computed)
    
    Returns:
        LST in °C as float32 array
    """
    bt = dn_to_brightness_temperature(tir_band10_dn)
    emissivity = compute_emissivity_from_ndvi(ndvi)
    lst = mono_window_lst(bt, emissivity)
    
    # Sanity check: clip to physically plausible range for UAE
    lst = np.clip(lst, 10.0, 80.0)
    return lst
