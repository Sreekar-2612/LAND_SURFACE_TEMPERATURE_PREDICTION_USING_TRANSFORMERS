"""
UHI intensity computation and transect profiles.
"""

import numpy as np

def compute_uhi_intensity(lst_array, rural_mask):
    """
    Compute UHI intensity as LST - mean(Rural LST).
    """
    rural_lst = lst_array[rural_mask]
    if len(rural_lst) == 0 or np.isnan(rural_lst).all():
        return np.zeros_like(lst_array)
        
    t_rural = np.nanmean(rural_lst)
    uhi_intensity = lst_array - t_rural
    return uhi_intensity
