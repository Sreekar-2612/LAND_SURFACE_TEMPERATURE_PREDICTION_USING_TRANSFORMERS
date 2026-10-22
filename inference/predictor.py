"""
Inference module for running the GeoViT model on test imagery.
"""

import torch
import numpy as np
import rasterio
from pathlib import Path
from model.geovit import GeoViT
from dataset.builder import load_feature_stack, apply_minmax_normalisation, denormalise_lst, load_normalisation_stats
from config import MODEL_CONFIG, RASTER_OUT_DIR

class Predictor:
    def __init__(self, model_path, stats_path, device="cuda"):
        self.device = device
        self.model = GeoViT(MODEL_CONFIG).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.stats = load_normalisation_stats(stats_path)
        
    def predict_image(self, feature_dir, date_str):
        """Predict LST for an entire feature stack."""
        features, _ = load_feature_stack(feature_dir, date_str)
        H, W, C = features.shape
        
        X_flat = features.reshape(-1, C)
        valid_mask = ~np.isnan(X_flat).any(axis=1)
        
        X_valid = X_flat[valid_mask]
        
        # We need a dummy y to use the normalisation function, or we can just normalise X
        feat_min = np.array(self.stats["feature_min"], dtype=np.float32)
        feat_max = np.array(self.stats["feature_max"], dtype=np.float32)
        denom = feat_max - feat_min
        denom[denom == 0] = 1.0
        
        X_norm = (X_valid - feat_min) / denom
        
        X_tensor = torch.from_numpy(X_norm.astype(np.float32))
        
        batch_size = 4096
        preds_list = []
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size].to(self.device)
                preds = self.model(batch)
                preds_list.append(preds.cpu().numpy())
                
        preds_norm = np.concatenate(preds_list, axis=0).flatten()
        preds_celsius = denormalise_lst(preds_norm, self.stats)
        
        # Reconstruct spatial image
        lst_pred = np.full((H, W), np.nan, dtype=np.float32)
        lst_pred.flat[valid_mask] = preds_celsius
        
        return lst_pred
        
    def save_prediction(self, lst_pred, ref_raster_path, out_filename):
        RASTER_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RASTER_OUT_DIR / out_filename
        
        with rasterio.open(ref_raster_path) as src:
            meta = src.meta.copy()
            
        meta.update({
            "dtype": "float32",
            "count": 1,
            "nodata": -9999.0
        })
        
        lst_out = np.where(np.isnan(lst_pred), -9999.0, lst_pred)
        
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(lst_out.astype(np.float32), 1)
            
        print(f"Saved prediction to {out_path}")
