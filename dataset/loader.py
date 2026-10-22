"""
PyTorch Dataset and DataLoader for 2D patch-wise SwinGeo training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import TRAIN_CONFIG

class LSTPatchDataset(Dataset):
    """
    Dataset that extracts 2D spatial patches from full raster arrays.
    Only patches that have at least some valid pixels are yielded.
    NaNs in features/targets are replaced with 0 and tracked with a mask.
    Includes data augmentation (random flips and rotations) for training.
    """
    def __init__(self, X_rasters, y_rasters, patch_size=32, patches_per_epoch=2000, augment=False):
        self.X_rasters = X_rasters
        self.y_rasters = y_rasters
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.n_rasters = len(X_rasters)
        self.augment = augment
        
        # We precompute valid indices to sample from
        self.valid_centers = []
        for i in range(self.n_rasters):
            y = y_rasters[i]
            valid = ~np.isnan(y)
            valid_idx = np.argwhere(valid)
            for r, c in valid_idx:
                if (r >= patch_size//2 and r < y.shape[0] - patch_size//2 and
                    c >= patch_size//2 and c < y.shape[1] - patch_size//2):
                    self.valid_centers.append((i, r, c))
                    
        self.valid_centers = np.array(self.valid_centers)

    def __len__(self):
        return self.patches_per_epoch

    def apply_augmentation(self, X_patch, y_patch, mask):
        # Random horizontal flip
        if np.random.rand() > 0.5:
            X_patch = np.flip(X_patch, axis=1)
            y_patch = np.flip(y_patch, axis=1)
            mask = np.flip(mask, axis=1)
            
        # Random vertical flip
        if np.random.rand() > 0.5:
            X_patch = np.flip(X_patch, axis=0)
            y_patch = np.flip(y_patch, axis=0)
            mask = np.flip(mask, axis=0)
            
        # Random 90-degree rotation (0, 1, 2, or 3 times)
        k = np.random.randint(0, 4)
        if k > 0:
            X_patch = np.rot90(X_patch, k, axes=(0, 1))
            y_patch = np.rot90(y_patch, k, axes=(0, 1))
            mask = np.rot90(mask, k, axes=(0, 1))
            
        return X_patch.copy(), y_patch.copy(), mask.copy()

    def __getitem__(self, idx):
        # Randomly pick a valid center
        rand_idx = np.random.randint(0, len(self.valid_centers))
        raster_id, center_r, center_c = self.valid_centers[rand_idx]
        
        half_p = self.patch_size // 2
        r_start = center_r - half_p
        r_end = r_start + self.patch_size
        c_start = center_c - half_p
        c_end = c_start + self.patch_size
        
        # Extract patch
        X_patch = self.X_rasters[raster_id][r_start:r_end, c_start:c_end, :].copy() # (P, P, 13)
        y_patch = self.y_rasters[raster_id][r_start:r_end, c_start:c_end].copy()    # (P, P)
        
        # Create mask for valid pixels
        mask = (~np.isnan(y_patch)) & (~np.isnan(X_patch).any(axis=-1))
        
        # Replace NaNs with 0 (Standardization ensures 0 is the mean, making it a neutral value)
        X_patch[np.isnan(X_patch)] = 0.0
        y_patch[np.isnan(y_patch)] = 0.0
        
        # Augmentation
        if self.augment:
            X_patch, y_patch, mask = self.apply_augmentation(X_patch, y_patch, mask)
        
        # X: (13, P, P)
        X_tensor = torch.from_numpy(X_patch.transpose(2, 0, 1)).float()
        # y: (1, P, P)
        y_tensor = torch.from_numpy(y_patch).unsqueeze(0).float()
        # mask: (1, P, P)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return X_tensor, y_tensor, mask_tensor


def get_dataloaders(X_train, y_train, X_val, y_val, patch_size=32):
    """Create train and validation DataLoaders for patches."""
    train_ds = LSTPatchDataset(X_train, y_train, patch_size=patch_size, patches_per_epoch=8000, augment=True)
    val_ds   = LSTPatchDataset(X_val,   y_val,   patch_size=patch_size, patches_per_epoch=2000, augment=False)
    
    bs = min(TRAIN_CONFIG.get("batch_size", 32), 64)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=TRAIN_CONFIG.get("num_workers", 2),
        pin_memory=TRAIN_CONFIG.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=TRAIN_CONFIG.get("num_workers", 2),
        pin_memory=TRAIN_CONFIG.get("pin_memory", True),
    )
    return train_loader, val_loader
