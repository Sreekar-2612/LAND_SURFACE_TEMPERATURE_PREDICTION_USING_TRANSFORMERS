import os
import json
import logging
import numpy as np
import rasterio
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
from config import (
    BASE_DIR, DATA_PROCESSED, TRAINING_DATES, PREDICTION_DATES,
    FEATURE_CHANNELS, UHI_BINARY_PERCENTILE_THRESHOLD, METRICS_DIR, RASTER_OUT_DIR
)
from dataset.builder import load_feature_stack

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def train_rf_baseline():
    logging.info("--- Starting Random Forest Baseline ---")
    
    # 1. Gather Training Dates
    train_dates = TRAINING_DATES.get("summer", []) + TRAINING_DATES.get("winter", [])
    train_dates = [d for d in train_dates if d]
    
    if not train_dates:
        logging.error("No training dates found in config.py!")
        return
        
    feature_dir = DATA_PROCESSED / "features"
    
    X_list = []
    y_list = []
    
    # 2. Load and Flatten Training Data
    logging.info(f"Loading training data for dates: {train_dates}...")
    for date_str in train_dates:
        try:
            features, lst = load_feature_stack(feature_dir, date_str)
            valid_mask = (~np.isnan(features).any(axis=-1)) & (~np.isnan(lst))
            X_valid = features[valid_mask]
            y_valid = lst[valid_mask]
            X_list.append(X_valid)
            y_list.append(y_valid)
            logging.info(f"Loaded {len(y_valid)} valid pixels from {date_str}")
        except Exception as e:
            logging.error(f"Failed to load data for {date_str}: {e}")
            
    if not X_list:
        logging.error("No training data could be loaded!")
        return
        
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    logging.info(f"Total training pixels available: {X_all.shape[0]}")
    
    # 3. Subsample for Speed and Memory Efficiency
    max_samples = 200000
    if X_all.shape[0] > max_samples:
        logging.info(f"Subsampling training pixels to {max_samples} for efficiency...")
        idx = np.random.choice(X_all.shape[0], max_samples, replace=False)
        X_train = X_all[idx]
        y_train = y_all[idx]
    else:
        X_train = X_all
        y_train = y_all
        
    # 4. Train Random Forest
    logging.info("Training Random Forest Regressor (n_estimators=100)...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    logging.info("Random Forest training complete.")
    
    # 5. Evaluate on Prediction Dates
    pred_dates = []
    for k, v in PREDICTION_DATES.items():
        if isinstance(v, list):
            pred_dates.extend([d for d in v if d])
        elif isinstance(v, str) and v:
            pred_dates.append(v)
            
    rf_metrics = {}
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RASTER_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for date_str in pred_dates:
        logging.info(f"Evaluating RF on {date_str}...")
        try:
            features, lst_observed = load_feature_stack(feature_dir, date_str)
        except Exception as e:
            logging.error(f"Failed to load features for {date_str}: {e}")
            continue
            
        H, W, C = features.shape
        
        # Flatten for prediction
        X_test = features.reshape(-1, C)
        valid_mask_pixel = ~np.isnan(X_test).any(axis=-1)
        
        # Predict
        pred_flat = np.full(X_test.shape[0], np.nan, dtype=np.float32)
        if np.any(valid_mask_pixel):
            pred_flat[valid_mask_pixel] = rf.predict(X_test[valid_mask_pixel])
            
        pred_map = pred_flat.reshape(H, W)
        
        # Save predicted raster
        out_path = RASTER_OUT_DIR / f"predicted_lst_rf_{date_str}.tif"
        
        # Get metadata from original LST file
        lst_obs_path = feature_dir / date_str / "lst.tif"
        with rasterio.open(lst_obs_path) as src:
            meta = src.meta.copy()
            
        meta.update(dtype=rasterio.float32, nodata=np.nan)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(pred_map, 1)
        logging.info(f"Saved RF prediction map to {out_path}")
        
        # Compute metrics
        valid_mask_eval = (~np.isnan(lst_observed)) & (~np.isnan(pred_map))
        y_true_eval = lst_observed[valid_mask_eval]
        y_pred_eval = pred_map[valid_mask_eval]
        
        if len(y_true_eval) > 0:
            # Regression
            mse = mean_squared_error(y_true_eval, y_pred_eval)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_eval, y_pred_eval)
            r2 = r2_score(y_true_eval, y_pred_eval)
            
            # Classification (UHI)
            threshold = float(np.percentile(y_true_eval, UHI_BINARY_PERCENTILE_THRESHOLD))
            y_t_class = (y_true_eval >= threshold).astype(int)
            y_p_class = (y_pred_eval >= threshold).astype(int)
            
            clf_report = classification_report(y_t_class, y_p_class, target_names=["Low UHI", "High UHI"], output_dict=True, zero_division=0)
            cm = confusion_matrix(y_t_class, y_p_class)
            
            rf_metrics[date_str] = {
                "regression": {
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "r2": float(r2)
                },
                "classification": clf_report,
                "uhi_threshold": threshold
            }
            logging.info(f"RF Results for {date_str} -> RMSE: {rmse:.3f}°C, R2: {r2:.3f}")
            
    # Save metrics to JSON
    out_json = METRICS_DIR / "evaluation_metrics_rf.json"
    with open(out_json, "w") as f:
        json.dump(rf_metrics, f, indent=4)
    logging.info(f"Saved RF metrics to {out_json}")

if __name__ == "__main__":
    train_rf_baseline()
