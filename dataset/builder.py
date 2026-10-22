"""
Build patch-wise dataset from processed feature rasters for SwinGeo.

Steps:
1. Load all 13 aligned feature rasters for each training date
2. Load corresponding LST raster (target)
3. Keep them as 2D spatial arrays (H, W, 13) and (H, W)
4. Compute Z-score standardization across valid pixels
5. Apply normalisation and return list of (H, W, 13) feature arrays and (H, W) target arrays
"""

import numpy as np
import json
from pathlib import Path
from config import FEATURE_CHANNELS

def load_feature_stack(feature_dir, date_str):
    import rasterio
    feature_path = Path(feature_dir) / date_str
    arrays = []
    
    for channel in FEATURE_CHANNELS:
        tif = feature_path / f"{channel}.tif"
        with rasterio.open(tif) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
        arrays.append(data)
    
    feature_stack = np.stack(arrays, axis=-1)
    
    lst_path = Path(feature_dir) / date_str / "lst.tif"
    with rasterio.open(lst_path) as src:
        lst = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            lst[lst == nodata] = np.nan
            
    return feature_stack, lst


def build_raster_dataset(feature_dirs, date_list):
    X_rasters = []
    y_rasters = []
    for date_str, feature_dir in zip(date_list, feature_dirs):
        features, lst = load_feature_stack(feature_dir, date_str)
        X_rasters.append(features)
        y_rasters.append(lst)
    return X_rasters, y_rasters


def compute_zscore_stats(X_rasters, y_rasters):
    """Compute per-channel mean/std for features and target across all valid pixels."""
    all_X_valid = []
    all_y_valid = []
    
    for X, y in zip(X_rasters, y_rasters):
        X_flat = X.reshape(-1, X.shape[-1])
        y_flat = y.reshape(-1)
        valid_mask = ~np.isnan(X_flat).any(axis=1) & ~np.isnan(y_flat)
        all_X_valid.append(X_flat[valid_mask])
        all_y_valid.append(y_flat[valid_mask])
        
    X_valid_concat = np.concatenate(all_X_valid, axis=0)
    y_valid_concat = np.concatenate(all_y_valid, axis=0)
    
    stats = {
        "feature_mean": X_valid_concat.mean(axis=0).tolist(),
        "feature_std": X_valid_concat.std(axis=0).tolist(),
        "lst_mean": float(y_valid_concat.mean()),
        "lst_std": float(y_valid_concat.std()),
    }
    
    # Avoid zero division
    stats["feature_std"] = [s if s > 1e-6 else 1.0 for s in stats["feature_std"]]
    if stats["lst_std"] < 1e-6:
        stats["lst_std"] = 1.0
        
    return stats


def apply_zscore_normalisation(X_rasters, y_rasters, stats):
    """
    Standardize full raster maps.
    Returns list of normalised X_rasters, y_rasters (float32).
    NaNs are preserved.
    """
    feat_mean = np.array(stats["feature_mean"], dtype=np.float32)
    feat_std = np.array(stats["feature_std"], dtype=np.float32)
    
    X_norm_rasters = []
    y_norm_rasters = []
    
    for X, y in zip(X_rasters, y_rasters):
        X_norm = (X - feat_mean) / feat_std
        y_norm = (y - stats["lst_mean"]) / stats["lst_std"]
        X_norm_rasters.append(X_norm.astype(np.float32))
        y_norm_rasters.append(y_norm.astype(np.float32))
        
    return X_norm_rasters, y_norm_rasters


def denormalise_lst(y_norm, stats):
    """Inverse transform standardized LST predictions back to °C."""
    return y_norm * stats["lst_std"] + stats["lst_mean"]


def spatial_train_val_split_rasters(X_rasters, y_rasters, val_fraction=0.20, random_seed=42):
    """
    Block-based spatial split for 2D rasters to avoid spatial leakage and ensure uniform class distribution.
    Divides the raster into a 10x10 grid of blocks. Randomly selects ~20% of blocks for validation.
    """
    rng = np.random.RandomState(random_seed)
    
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    
    for X, y in zip(X_rasters, y_rasters):
        H, W, C = X.shape
        grid_rows, grid_cols = 10, 10
        block_h = int(np.ceil(H / grid_rows))
        block_w = int(np.ceil(W / grid_cols))
        
        n_blocks = grid_rows * grid_cols
        all_blocks = np.arange(n_blocks)
        n_val_blocks = max(1, int(np.round(n_blocks * val_fraction)))
        val_blocks = rng.choice(all_blocks, size=n_val_blocks, replace=False)
        
        train_mask = np.zeros((H, W), dtype=bool)
        val_mask = np.zeros((H, W), dtype=bool)
        
        for r_idx in range(grid_rows):
            for c_idx in range(grid_cols):
                block_id = r_idx * grid_cols + c_idx
                
                r_start = r_idx * block_h
                r_end = min((r_idx + 1) * block_h, H)
                c_start = c_idx * block_w
                c_end = min((c_idx + 1) * block_w, W)
                
                if block_id in val_blocks:
                    val_mask[r_start:r_end, c_start:c_end] = True
                else:
                    train_mask[r_start:r_end, c_start:c_end] = True
        
        X_train = X.copy()
        X_train[~train_mask, :] = np.nan
        y_train = y.copy()
        y_train[~train_mask] = np.nan
        
        X_val = X.copy()
        X_val[~val_mask, :] = np.nan
        y_val = y.copy()
        y_val[~val_mask] = np.nan
        
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_val_list.append(X_val)
        y_val_list.append(y_val)
        
    return X_train_list, y_train_list, X_val_list, y_val_list


def save_normalisation_stats(stats, path):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_normalisation_stats(path):
    with open(path) as f:
        return json.load(f)
