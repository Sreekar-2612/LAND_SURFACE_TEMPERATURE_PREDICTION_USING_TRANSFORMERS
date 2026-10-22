# bias_correction.py
import json
import logging
import numpy as np
import rasterio
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
from config import (
    BASE_DIR, DATA_PROCESSED, PREDICTION_DATES,
    UHI_BINARY_PERCENTILE_THRESHOLD, METRICS_DIR, RASTER_OUT_DIR, FIGURE_DIR
)
try:
    from analysis.visualization import plot_confusion_matrix, plot_lst_map
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    logging.warning("Visualization libraries (matplotlib/seaborn) not found. Skipping plot generation, but metrics and rasters will still be saved.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_bias_correction():
    logging.info("--- Starting Mean-Bias Correction on Swin-UNet Predictions ---")
    
    feature_dir = DATA_PROCESSED / "features"
    pred_dates = ["2025-01-28", "2025-11-20", "2025-11-27", "2025-12-13"]
    
    corrected_metrics = {}
    
    for date_str in pred_dates:
        lst_obs_path = feature_dir / date_str / "lst.tif"
        lst_pred_path = RASTER_OUT_DIR / f"predicted_lst_{date_str}.tif"
        
        if not lst_obs_path.exists() or not lst_pred_path.exists():
            logging.warning(f"Missing files for {date_str}, skipping.")
            continue
            
        # Load Observed LST
        with rasterio.open(lst_obs_path) as src:
            y_true = src.read(1).astype(np.float32)
            meta = src.meta.copy()
            nodata = src.nodata
            if nodata is not None:
                y_true[y_true == nodata] = np.nan
                
        # Load Predicted LST
        with rasterio.open(lst_pred_path) as src:
            y_pred = src.read(1).astype(np.float32)
            nodata_pred = src.nodata
            if nodata_pred is not None:
                y_pred[y_pred == nodata_pred] = np.nan
            
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        if len(y_true_valid) > 0:
            # Calculate Mean Bias: Pred - True
            bias = np.mean(y_pred_valid) - np.mean(y_true_valid)
            logging.info(f"Date {date_str} - Calculated Systematic Bias: {bias:.3f}°C")
            
            # Apply Correction
            y_pred_corrected = y_pred - bias
            y_pred_corrected_valid = y_pred_corrected[valid_mask]
            
            # Save Corrected Raster
            out_path = RASTER_OUT_DIR / f"predicted_lst_corrected_{date_str}.tif"
            meta.update(dtype=rasterio.float32, nodata=np.nan)
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(y_pred_corrected, 1)
                
            # Compute New Metrics
            mse = mean_squared_error(y_true_valid, y_pred_corrected_valid)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_valid, y_pred_corrected_valid)
            r2 = r2_score(y_true_valid, y_pred_corrected_valid)
            
            # Classification
            threshold = float(np.percentile(y_true_valid, UHI_BINARY_PERCENTILE_THRESHOLD))
            y_t_class = (y_true_valid >= threshold).astype(int)
            y_p_class = (y_pred_corrected_valid >= threshold).astype(int)
            
            clf_report = classification_report(y_t_class, y_p_class, target_names=["Low UHI", "High UHI"], output_dict=True, zero_division=0)
            cm = confusion_matrix(y_t_class, y_p_class)
            
            corrected_metrics[date_str] = {
                "bias_subtracted": float(bias),
                "regression": {
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "r2": float(r2)
                },
                "classification": clf_report,
                "uhi_threshold": threshold
            }
            # Regenerate corrected plots
            if HAS_VISUALIZATION:
                plot_confusion_matrix(cm, out_filename=f"confusion_matrix_corrected_{date_str}.png")
                plot_lst_map(y_pred_corrected, out_filename=f"predicted_lst_corrected_{date_str}.png", title=f"Corrected LST - {date_str}")
            
            logging.info(f"Corrected Results for {date_str} -> RMSE: {rmse:.3f}°C, R2: {r2:.3f}")
            
    # Save Metrics
    out_json = METRICS_DIR / "evaluation_metrics_corrected.json"
    with open(out_json, "w") as f:
        json.dump(corrected_metrics, f, indent=4)
    logging.info(f"Saved corrected metrics to {out_json}")

if __name__ == "__main__":
    run_bias_correction()
