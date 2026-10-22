"""
Main entry point for the GeoViT pipeline.
"""

import sys
import argparse
import glob
import logging
import torch
from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling

from config import (
    BASE_DIR, DATA_RAW, DATA_PROCESSED, 
    TRAINING_DATES, PREDICTION_DATES, LANDSAT_BANDS, 
    TARGET_CRS, TARGET_RESOLUTION, STUDY_CENTER_LAT, STUDY_CENTER_LON,
    TRAIN_CONFIG, N_CHANNELS, ALOS_DIR, MODEL_DIR
)

from preprocessing.alignment import read_band
import torch.nn as nn
import torch.optim as optim
from dataset.builder import (
    build_raster_dataset, compute_zscore_stats, apply_zscore_normalisation,
    save_normalisation_stats, spatial_train_val_split_rasters, denormalise_lst
)
from dataset.loader import get_dataloaders
from model.swin_geo import SwinGeo

def setup_logging():
    log_file = BASE_DIR / "geovit_run.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging initialized. Writing to {log_file}")

def get_device(requested_device="auto"):
    if requested_device != "auto":
        device = torch.device(requested_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Hardware Auto-Detection: Using device '{device}'")
    return device

def mosaic_rasters(input_paths, output_path):
    if not input_paths: return None
    if len(input_paths) == 1: return input_paths[0]
    
    src_files_to_mosaic = [rasterio.open(fp) for fp in input_paths]
    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans})
    
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
        
    for src in src_files_to_mosaic: src.close()
    return output_path

from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from rasterio.crs import CRS

def get_master_grid():
    from preprocessing.alignment import get_study_bbox
    min_lon, min_lat, max_lon, max_lat = get_study_bbox(STUDY_CENTER_LAT, STUDY_CENTER_LON, buffer_km=10)
    target_crs = CRS.from_string(TARGET_CRS)
    minx, miny, maxx, maxy = transform_bounds(CRS.from_epsg(4326), target_crs, min_lon, min_lat, max_lon, max_lat)
    width = int((maxx - minx) / TARGET_RESOLUTION)
    height = int((maxy - miny) / TARGET_RESOLUTION)
    dst_transform = from_bounds(minx, miny, maxx, maxy, width, height)
    return dst_transform, (height, width)

def preprocess_data():
    from preprocessing.alignment import align_raster, get_reference_grid, save_raster
    from preprocessing.indices import compute_all_indices, apply_landsat_c2_scale
    from preprocessing.lst_retrieval import retrieve_lst
    from preprocessing.dem_derivatives import compute_terrain_derivatives
    from preprocessing.urban_morphology import compute_building_height, compute_svf_simplified
    from preprocessing.era5_processing import extract_era5_variables
    from preprocessing.lulc_classification import kmeans_lulc_fallback

    logging.info("Starting data preprocessing...")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    logging.info("Establishing Master Grid...")
    ref_transform, ref_shape = get_master_grid()

    # 1. Mosaic and Align DEM
    logging.info("Mosaicing and aligning DEM tiles...")
    dem_files = glob.glob(str(DATA_RAW / "srtm" / "*.tif"))
    dem_mosaic_path = str(DATA_PROCESSED / "dem_mosaic_raw.tif")
    dem_aligned_path = str(DATA_PROCESSED / "dem_aligned.tif")
    if dem_files:
        mosaic_rasters(dem_files, dem_mosaic_path)
        align_raster(dem_mosaic_path, dem_aligned_path, reference_transform=ref_transform, reference_shape=ref_shape)
    else:
        logging.warning("No DEM files found. Skipping.")
        return
        
    # Read aligned DEM
    dem_aligned, _, _ = read_band(dem_aligned_path)
    
    # 2. Mosaic and Align ALOS PALSAR HH/HV
    logging.info("Mosaicing and aligning ALOS PALSAR HH/HV tiles...")
    palsar_hh_files = glob.glob(str(ALOS_DIR / "*_HH_*.tif"))
    palsar_hv_files = glob.glob(str(ALOS_DIR / "*_HV_*.tif"))
    
    palsar_hh_mosaic = str(DATA_PROCESSED / "palsar_hh_mosaic_raw.tif")
    palsar_hv_mosaic = str(DATA_PROCESSED / "palsar_hv_mosaic_raw.tif")
    palsar_hh_aligned = str(DATA_PROCESSED / "palsar_hh_aligned.tif")
    palsar_hv_aligned = str(DATA_PROCESSED / "palsar_hv_aligned.tif")
    
    if palsar_hh_files and palsar_hv_files:
        mosaic_rasters(palsar_hh_files, palsar_hh_mosaic)
        align_raster(palsar_hh_mosaic, palsar_hh_aligned, reference_transform=ref_transform, reference_shape=ref_shape)
        
        mosaic_rasters(palsar_hv_files, palsar_hv_mosaic)
        align_raster(palsar_hv_mosaic, palsar_hv_aligned, reference_transform=ref_transform, reference_shape=ref_shape)
        
        palsar_hh, _, _ = read_band(palsar_hh_aligned)
        palsar_hv, _, _ = read_band(palsar_hv_aligned)
    else:
        logging.warning("No ALOS PALSAR HH/HV files found. Using default 0 values.")
        palsar_hh = np.zeros(ref_shape, dtype=np.float32)
        palsar_hv = np.zeros(ref_shape, dtype=np.float32)
        
    # 2b. Align ESA WorldCover LULC
    logging.info("Aligning ESA WorldCover LULC map...")
    lulc_files = glob.glob(str(DATA_RAW / "lulc" / "*_Map.tif"))
    lulc_aligned_path = str(DATA_PROCESSED / "lulc_aligned.tif")
    if lulc_files:
        align_raster(lulc_files[0], lulc_aligned_path, reference_transform=ref_transform, reference_shape=ref_shape, resampling=Resampling.nearest)
        lulc_aligned, _, _ = read_band(lulc_aligned_path)
    else:
        logging.warning("No ESA WorldCover LULC Map found. Using default 0 values.")
        lulc_aligned = np.zeros(ref_shape, dtype=np.float32)
    
    # Pre-calculate DEM derivatives and Urban Morphology (static across dates)
    logging.info("Computing Terrain Derivatives and Urban Morphology...")
    terrain_dict = compute_terrain_derivatives(dem_aligned, resolution=TARGET_RESOLUTION)
    building_height = compute_building_height(dem_aligned) # SRTM as proxy
    svf = compute_svf_simplified(dem_aligned, resolution=TARGET_RESOLUTION)

    # 3. Iterate through Dates
    all_dates = []
    for date_list in list(TRAINING_DATES.values()) + list(PREDICTION_DATES.values()):
        if isinstance(date_list, list):
            all_dates.extend([d for d in date_list if d])
        elif isinstance(date_list, str) and date_list:
            all_dates.append(date_list)
    # Remove duplicates while preserving order
    all_dates = list(dict.fromkeys(all_dates))

    for date_str in all_dates:
        logging.info(f"--- Processing Date: {date_str} ---")
        date_dir = DATA_PROCESSED / "features" / date_str
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # Landsat Processing
        landsat_search = list((DATA_RAW / "landsat").rglob(f"*{date_str.replace('-','')}*SR_B2.TIF"))
        if not landsat_search:
            logging.warning(f"Skipping {date_str} (No Landsat data found)")
            continue
            
        landsat_base = str(landsat_search[0]).replace("SR_B2.TIF", "")
        logging.info(f"Aligning Landsat bands for {date_str}...")
        
        aligned_bands = {}
        for band_name, band_suffix in LANDSAT_BANDS.items():
            if "tir" in band_name:
                src_path = landsat_base + f"ST_{band_suffix}.TIF"
            else:
                src_path = landsat_base + f"SR_{band_suffix}.TIF"
                
            dst_path = str(date_dir / f"{band_name}.tif")
            align_raster(src_path, dst_path, reference_transform=ref_transform, reference_shape=ref_shape)
            aligned_bands[band_name], _, _ = read_band(dst_path)

        # Landsat Indices
        logging.info(f"Computing Optical Indices for {date_str}...")
        scaled_bands = {k: apply_landsat_c2_scale(v) for k, v in aligned_bands.items() if "tir" not in k}
        indices = compute_all_indices(scaled_bands)
        
        for idx_name, idx_array in indices.items():
            save_raster(idx_array, ref_transform, CRS.from_string(TARGET_CRS), str(date_dir / f"{idx_name}.tif"))
            
        # Save LULC (using the aligned static ESA WorldCover map)
        logging.info(f"Saving ESA WorldCover LULC for {date_str}...")
        save_raster(lulc_aligned, ref_transform, CRS.from_string(TARGET_CRS), str(date_dir / "lulc.tif"))

        # Save Terrain & Urban Morphology & ALOS PALSAR
        for k, v in terrain_dict.items():
            save_raster(v, ref_transform, CRS.from_string(TARGET_CRS), str(date_dir / f"{k}.tif"))
        save_raster(building_height, ref_transform, CRS.from_string(TARGET_CRS), str(date_dir / "building_height.tif"))
        save_raster(svf, ref_transform, CRS.from_string(TARGET_CRS), str(date_dir / "svf.tif"))
        save_raster(palsar_hh, ref_transform, CRS.from_string(TARGET_CRS), str(date_dir / "palsar_hh.tif"))
        save_raster(palsar_hv, ref_transform, CRS.from_string(TARGET_CRS), str(date_dir / "palsar_hv.tif"))
        
        # ERA5 Processing
        logging.info(f"Computing Meteorological Features for {date_str}...")
        from datetime import datetime
        from preprocessing.era5_processing import create_constant_raster
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        
        era5_search = list((DATA_RAW / "era5").rglob("*.nc"))
        if era5_search:
            try:
                ws, wd, tcc, t2m = extract_era5_variables(str(era5_search[0]), STUDY_CENTER_LAT, STUDY_CENTER_LON, target_date)
                ws_raster = create_constant_raster(ws, ref_shape)
                wd_raster = create_constant_raster(wd, ref_shape)
                tcc_raster = create_constant_raster(tcc, ref_shape)
                t2m_raster = create_constant_raster(t2m, ref_shape)
            except Exception as e:
                logging.warning(f"ERA5 extraction failed: {e}. Using defaults.")
                ws_raster, wd_raster = np.zeros(ref_shape, dtype=np.float32), np.zeros(ref_shape, dtype=np.float32)
                tcc_raster, t2m_raster = np.zeros(ref_shape, dtype=np.float32), np.zeros(ref_shape, dtype=np.float32)
        else:
            logging.warning("No ERA5 files found. Using default 0 values.")
            ws_raster, wd_raster = np.zeros(ref_shape, dtype=np.float32), np.zeros(ref_shape, dtype=np.float32)
            tcc_raster, t2m_raster = np.zeros(ref_shape, dtype=np.float32), np.zeros(ref_shape, dtype=np.float32)
            
        save_raster(ws_raster, ref_transform, CRS.from_string(TARGET_CRS), str(date_dir / "wind_speed.tif"))
        save_raster(wd_raster, ref_transform, CRS.from_string(TARGET_CRS), str(date_dir / "wind_direction.tif"))
        save_raster(tcc_raster, ref_transform, CRS.from_string(TARGET_CRS), str(date_dir / "cloud_cover.tif"))
        save_raster(t2m_raster, ref_transform, CRS.from_string(TARGET_CRS), str(date_dir / "air_temperature.tif"))

        # LST Retrieval
        logging.info(f"Computing LST for {date_str}...")
        lst_dn = aligned_bands["tir1"]
        ndvi = indices["ndvi"]
        lst_celsius = retrieve_lst(lst_dn, ndvi)
        save_raster(lst_celsius, ref_transform, CRS.from_string(TARGET_CRS), str(date_dir / "lst.tif"))
        
    logging.info("Preprocessing complete. Processed rasters saved to data/processed/")

def train_model(args):
    logging.info("Starting SwinGeo model training...")
    device = get_device(args.device)
    
    dates = TRAINING_DATES.get("summer", []) + TRAINING_DATES.get("winter", [])
    dates = [d for d in dates if d]
    
    if not dates:
        logging.error("No training dates found in config.py!")
        return
        
    feature_dir = DATA_PROCESSED / "features"
    feature_dirs = [feature_dir] * len(dates)
    
    logging.info(f"Loading raster datasets for dates: {dates}...")
    X_raw, y_raw = build_raster_dataset(feature_dirs, dates)
    logging.info(f"Loaded {len(X_raw)} rasters.")
    
    logging.info("Applying Z-score standardization...")
    stats = compute_zscore_stats(X_raw, y_raw)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_normalisation_stats(stats, MODEL_DIR / "norm_stats.json")
    
    X_norm, y_norm = apply_zscore_normalisation(X_raw, y_raw, stats)
    
    logging.info("Splitting into training and validation sets (Block-based)...")
    X_train, y_train, X_val, y_val = spatial_train_val_split_rasters(
        X_norm, y_norm, val_fraction=TRAIN_CONFIG.get("val_split", 0.20)
    )
    
    logging.info("Creating patch DataLoaders with Augmentations...")
    train_loader, val_loader = get_dataloaders(X_train, y_train, X_val, y_val, patch_size=32)
    
    logging.info("Initializing SwinGeo (Swin-UNet) model...")
    model = SwinGeo(in_channels=N_CHANNELS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Using Huber Loss (SmoothL1) without reduction so we can apply mask
    criterion = nn.SmoothL1Loss(reduction='none')
    
    epochs = args.epochs
    logging.info(f"Training parameters -> Epochs: {epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    import json
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for batch_X, batch_y, batch_mask in train_loader:
            batch_X, batch_y, batch_mask = batch_X.to(device), batch_y.to(device), batch_mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            loss_map = criterion(outputs, batch_y)
            # Mask out invalid pixels
            loss = (loss_map * batch_mask).sum() / (batch_mask.sum() + 1e-8)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_count += 1
            
        train_loss /= max(1, train_count)
        
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_X, batch_y, batch_mask in val_loader:
                batch_X, batch_y, batch_mask = batch_X.to(device), batch_y.to(device), batch_mask.to(device)
                outputs = model(batch_X)
                loss_map = criterion(outputs, batch_y)
                loss = (loss_map * batch_mask).sum() / (batch_mask.sum() + 1e-8)
                val_loss += loss.item()
                val_count += 1
        
        val_loss /= max(1, val_count)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        logging.info(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = MODEL_DIR / "geovit_best.pt"
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}! Best Val Loss: {best_val_loss:.6f}")
                break
                
    final_path = MODEL_DIR / "geovit_final.pt"
    torch.save(model.state_dict(), final_path)
    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump(history, f)
        
    try:
        from analysis.visualization import plot_training_curves
        plot_training_curves(history['train_loss'], history['val_loss'], out_filename="loss_curve.png")
    except ImportError:
        pass
        
    logging.info(f"Training complete. Checkpoints saved to {MODEL_DIR}/")

def get_gaussian_window(patch_size, sigma=0.5):
    import scipy.signal
    # Create 1D Gaussian
    window_1d = scipy.signal.windows.gaussian(patch_size, std=patch_size * sigma)
    # Convert to 2D
    window_2d = np.outer(window_1d, window_1d)
    return window_2d

def run_inference(args):
    logging.info("Starting overlapping sliding window inference on test dates...")
    device = get_device(args.device)
    
    model_path = MODEL_DIR / "geovit_best.pt"
    if not model_path.exists():
        logging.error(f"Trained model not found at {model_path}!")
        return
        
    stats_path = MODEL_DIR / "norm_stats.json"
    if not stats_path.exists():
        logging.error("Normalization stats not found!")
        return
        
    from dataset.builder import load_normalisation_stats, load_feature_stack, denormalise_lst
    from preprocessing.alignment import save_raster
    from rasterio.crs import CRS
    
    stats = load_normalisation_stats(stats_path)
    model = SwinGeo(in_channels=N_CHANNELS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    out_dir = BASE_DIR / "outputs" / "rasters"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ref_transform, (H, W) = get_master_grid()
    
    pred_dates = []
    for k, v in PREDICTION_DATES.items():
        if isinstance(v, list):
            pred_dates.extend([d for d in v if d])
        elif isinstance(v, str) and v:
            pred_dates.append(v)
    
    import json
    from analysis.accuracy import compute_regression_metrics, compute_classification_metrics
    try:
        from analysis.visualization import plot_confusion_matrix, plot_lulc_lst_boxplot, plot_lst_map
        HAS_VISUALIZATION = True
    except ImportError:
        HAS_VISUALIZATION = False
        logging.warning("Visualization libraries (matplotlib/seaborn) not found. Skipping plot generation.")
    from analysis.uhi_profiling import compute_uhi_intensity
    from config import UHI_BINARY_PERCENTILE_THRESHOLD, METRICS_DIR, LULC_CLASSES, N_LULC_CLASSES, FEATURE_CHANNELS
    
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    feat_mean = np.array(stats["feature_mean"], dtype=np.float32)
    feat_std = np.array(stats["feature_std"], dtype=np.float32)

    patch_size = 32
    stride = patch_size // 2  # Overlapping stride of 16
    gaussian_mask = get_gaussian_window(patch_size)

    for date_str in pred_dates:
        logging.info(f"Predicting LST for {date_str}...")
        feature_dir = DATA_PROCESSED / "features"
        
        try:
            features, lst_observed = load_feature_stack(feature_dir, date_str)
        except Exception as e:
            logging.error(f"Failed to load features for {date_str}: {e}")
            continue
            
        valid_mask = (~np.isnan(features).any(axis=-1)) & (~np.isnan(lst_observed))
        
        # Standardize features
        X_norm = (features - feat_mean) / feat_std
        X_norm[np.isnan(X_norm)] = 0.0
        
        # Pad with 'reflect' mode to prevent edge artifacts
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        
        X_padded = np.pad(X_norm, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        H_p, W_p, _ = X_padded.shape
        
        pred_padded = np.zeros((H_p, W_p), dtype=np.float32)
        weight_padded = np.zeros((H_p, W_p), dtype=np.float32)
        
        # Overlapping Sliding window inference
        with torch.no_grad():
            for r in range(0, H_p - patch_size + 1, stride):
                for c in range(0, W_p - patch_size + 1, stride):
                    patch = X_padded[r:r+patch_size, c:c+patch_size, :]
                    patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                    out = model(patch_tensor)
                    out_np = out.squeeze().cpu().numpy()
                    
                    # Blend using Gaussian weights
                    pred_padded[r:r+patch_size, c:c+patch_size] += out_np * gaussian_mask
                    weight_padded[r:r+patch_size, c:c+patch_size] += gaussian_mask
                    
        # Avoid division by zero
        weight_padded[weight_padded == 0] = 1.0
        pred_avg = pred_padded / weight_padded
        pred_map_norm = pred_avg[:H, :W]
        
        # Denormalize
        pred_map = denormalise_lst(pred_map_norm, stats)
        
        # Apply valid mask
        pred_map[~valid_mask] = np.nan
        
        out_path = out_dir / f"predicted_lst_{date_str}.tif"
        save_raster(pred_map, ref_transform, CRS.from_string(TARGET_CRS), str(out_path))
        logging.info(f"Saved predicted UHI heat map for {date_str} to {out_path}")
        
        y_true_valid = lst_observed[valid_mask]
        y_pred_valid = pred_map[valid_mask]
        lulc_idx = FEATURE_CHANNELS.index("lulc")
        lulc_valid = features[:, :, lulc_idx][valid_mask]
        
        if len(y_true_valid) > 0:
            reg_metrics = compute_regression_metrics(y_true_valid, y_pred_valid)
            uhi_threshold = float(np.percentile(y_true_valid, UHI_BINARY_PERCENTILE_THRESHOLD))
            class_metrics = compute_classification_metrics(y_true_valid, y_pred_valid, threshold=uhi_threshold)
            all_metrics[date_str] = {"regression": reg_metrics, "classification": class_metrics["classification_report"], "uhi_threshold": uhi_threshold}
            if HAS_VISUALIZATION:
                plot_confusion_matrix(class_metrics["confusion_matrix"], out_filename=f"confusion_matrix_{date_str}.png")
                plot_lst_map(pred_map, out_filename=f"predicted_lst_{date_str}.png", title=f"Predicted LST - {date_str}")
            
            # Rural classes in ESA WorldCover: 10 (Tree cover), 20 (Shrubland), 30 (Grassland), 40 (Cropland), 60 (Bare / sparse vegetation)
            rural_mask = np.isin(features[:, :, lulc_idx], [10, 20, 30, 40, 60])
            uhi_intensity = compute_uhi_intensity(pred_map, rural_mask)
            
            if HAS_VISUALIZATION:
                plot_lst_map(uhi_intensity, out_filename=f"uhi_intensity_{date_str}.png", cmap="coolwarm", title=f"UHI Intensity - {date_str}")
                plot_lulc_lst_boxplot(y_pred_valid, lulc_valid, LULC_CLASSES, out_filename=f"lst_lulc_boxplot_{date_str}.png")

    with open(METRICS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
        
    logging.info("Inference complete. Prediction rasters saved to outputs/rasters/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoViT Pipeline Orchestrator")
    parser.add_argument("--step", choices=["preprocess", "train", "infer", "all"], default="all",
                        help="Which step of the pipeline to run")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: 'auto', 'cuda', 'mps', or 'cpu'")
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"], help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"], help="Training batch size")
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["learning_rate"], help="Learning rate")
    args = parser.parse_args()

    setup_logging()

    if args.step in ["preprocess", "all"]: preprocess_data()
    if args.step in ["train", "all"]: train_model(args)
    if args.step in ["infer", "all"]: run_inference(args)
