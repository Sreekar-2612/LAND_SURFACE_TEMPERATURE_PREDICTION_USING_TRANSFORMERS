import re

with open("main.py", "r") as f:
    code = f.read()

# Replace the imports
old_imports = """from dataset.builder import (
    build_pixel_dataset, compute_minmax_stats, apply_minmax_normalisation,
    save_normalisation_stats, spatial_train_val_split
)
from dataset.loader import get_dataloaders
from model.geovit import GeoViT"""

new_imports = """from dataset.builder import (
    build_raster_dataset, compute_minmax_stats, apply_minmax_normalisation,
    save_normalisation_stats, spatial_train_val_split_rasters, denormalise_lst
)
from dataset.loader import get_dataloaders
from model.swin_geovit import SwinGeoViT"""

code = code.replace(old_imports, new_imports)

# Replace train_model
train_model_new = """def train_model(args):
    logging.info("Starting SwinGeoViT model training...")
    device = get_device(args.device)
    
    dates = TRAINING_DATES.get("summer", []) + TRAINING_DATES.get("winter", [])
    dates = [d for d in dates if d]
    
    if not dates:
        logging.error("No training dates found in config.py!")
        return
        
    feature_dir = PROCESSED_DATA_DIR / "features"
    feature_dirs = [feature_dir] * len(dates)
    
    logging.info(f"Loading raster datasets for dates: {dates}...")
    X_raw, y_raw = build_raster_dataset(feature_dirs, dates)
    logging.info(f"Loaded {len(X_raw)} rasters.")
    
    logging.info("Applying min-max normalisation...")
    stats = compute_minmax_stats(X_raw, y_raw)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_normalisation_stats(stats, MODEL_DIR / "norm_stats.json")
    
    X_norm, y_norm = apply_minmax_normalisation(X_raw, y_raw, stats)
    
    logging.info("Splitting into training and validation sets...")
    X_train, y_train, X_val, y_val = spatial_train_val_split_rasters(
        X_norm, y_norm, val_fraction=TRAIN_CONFIG.get("val_split", 0.20)
    )
    
    logging.info("Creating patch DataLoaders...")
    train_loader, val_loader = get_dataloaders(X_train, y_train, X_val, y_val, patch_size=32)
    
    logging.info("Initializing SwinGeoViT model...")
    model = SwinGeoViT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Using MSE without reduction so we can apply mask
    criterion = nn.MSELoss(reduction='none')
    
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
        
        # Validation
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
"""

# Replace run_inference
run_inference_new = """def run_inference(args):
    logging.info("Starting dense patch inference on test dates...")
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
    model = SwinGeoViT().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    out_dir = BASE_DIR / "outputs" / "rasters"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ref_transform, (H, W) = get_master_grid()
    
    pred_dates = [d for d in [PREDICTION_DATES.get("winter"), PREDICTION_DATES.get("summer")] if d]
    
    import json
    from analysis.accuracy import compute_regression_metrics, compute_classification_metrics
    from analysis.visualization import plot_confusion_matrix, plot_lulc_lst_boxplot, plot_lst_map
    from analysis.uhi_profiling import compute_uhi_intensity
    from config import UHI_BINARY_PERCENTILE_THRESHOLD, METRICS_DIR, LULC_CLASSES, N_LULC_CLASSES, FEATURE_CHANNELS
    
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    feat_min = np.array(stats["feature_min"], dtype=np.float32)
    feat_max = np.array(stats["feature_max"], dtype=np.float32)
    denom = feat_max - feat_min
    denom[denom == 0] = 1.0

    patch_size = 32

    for date_str in pred_dates:
        logging.info(f"Predicting LST for {date_str}...")
        feature_dir = PROCESSED_DATA_DIR / "features"
        
        try:
            features, lst_observed = load_feature_stack(feature_dir, date_str)
        except Exception as e:
            logging.error(f"Failed to load features for {date_str}: {e}")
            continue
            
        valid_mask = (~np.isnan(features).any(axis=-1)) & (~np.isnan(lst_observed))
        
        # Normalize features
        X_norm = (features - feat_min) / denom
        X_norm[np.isnan(X_norm)] = 0.0
        
        # We will pad the image to be a multiple of patch_size
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        
        X_padded = np.pad(X_norm, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        H_p, W_p, _ = X_padded.shape
        
        pred_padded = np.zeros((H_p, W_p), dtype=np.float32)
        count_padded = np.zeros((H_p, W_p), dtype=np.float32)
        
        # Sliding window inference
        with torch.no_grad():
            for r in range(0, H_p, patch_size):
                for c in range(0, W_p, patch_size):
                    patch = X_padded[r:r+patch_size, c:c+patch_size, :]
                    patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                    out = model(patch_tensor)
                    out_np = out.squeeze().cpu().numpy()
                    pred_padded[r:r+patch_size, c:c+patch_size] += out_np
                    count_padded[r:r+patch_size, c:c+patch_size] += 1
                    
        pred_avg = pred_padded / np.maximum(count_padded, 1)
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
            
            plot_confusion_matrix(class_metrics["confusion_matrix"], out_filename=f"confusion_matrix_{date_str}.png")
            plot_lst_map(pred_map, out_filename=f"predicted_lst_{date_str}.png", title=f"Predicted LST - {date_str}")
            
            rural_mask = (features[:, :, lulc_idx] == 0) | (features[:, :, lulc_idx] == 2)
            uhi_intensity = compute_uhi_intensity(pred_map, rural_mask)
            plot_lst_map(uhi_intensity, out_filename=f"uhi_intensity_{date_str}.png", cmap="coolwarm", title=f"UHI Intensity - {date_str}")
            plot_lulc_lst_boxplot(y_pred_valid, lulc_valid, LULC_CLASSES, out_filename=f"lst_lulc_boxplot_{date_str}.png")

    with open(METRICS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
        
    logging.info("Inference complete. Prediction rasters saved to outputs/rasters/")
"""

# Extract the parts to replace
train_idx = code.find("def train_model(args):")
run_inf_idx = code.find("def run_inference(args):")
if__main__idx = code.find('if __name__ == "__main__":')

if train_idx != -1 and run_inf_idx != -1 and if__main__idx != -1:
    new_code = code[:train_idx] + train_model_new + "\n" + run_inference_new + "\n" + code[if__main__idx:]
    with open("main.py", "w") as f:
        f.write(new_code)
    print("main.py successfully patched.")
else:
    print("Could not find function signatures.")
