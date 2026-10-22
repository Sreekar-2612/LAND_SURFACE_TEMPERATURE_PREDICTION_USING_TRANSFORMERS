# config.py
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
DATA_RAW         = BASE_DIR / "data" / "raw"
DATA_PROCESSED   = BASE_DIR / "data" / "processed"
OUTPUTS          = BASE_DIR / "outputs"

LANDSAT_TRAIN_DIR    = DATA_RAW / "landsat" / "training"
LANDSAT_PRED_DIR     = DATA_RAW / "landsat" / "prediction"
SENTINEL2_DIR        = DATA_RAW / "sentinel2"
ERA5_DIR             = DATA_RAW / "era5"
ALOS_DIR             = DATA_RAW / "alos_palsar"
SRTM_DIR             = DATA_RAW / "srtm"
INSITU_DIR           = DATA_RAW / "insitu"

FEATURES_DIR     = DATA_PROCESSED / "features"
LST_DIR          = DATA_PROCESSED / "lst"
LULC_DIR         = DATA_PROCESSED / "lulc"

MODEL_DIR        = OUTPUTS / "models"
RASTER_OUT_DIR   = OUTPUTS / "rasters"
FIGURE_DIR       = OUTPUTS / "figures"
METRICS_DIR      = OUTPUTS / "metrics"

# ─── Study Area ───────────────────────────────────────────────────────────────
STUDY_CENTER_LAT  = 23.0225          # Ahmedabad, Gujarat, India
STUDY_CENTER_LON  = 72.5714
TARGET_CRS        = "EPSG:32643"     # UTM Zone 43N — appropriate for Gujarat
TARGET_RESOLUTION = 10               # metres — matches Sentinel-2 native resolution

# ─── Date Configuration ───────────────────────────────────────────────────────
TRAINING_DATES = {
    "summer": ["2024-04-24", "2024-05-25", "2025-04-29"],
    "winter": ["2025-12-13", "2025-12-14", "2025-12-15"], 
}
PREDICTION_DATES = {
    "winter": ["2025-01-28", "2025-11-15", "2025-11-20", "2025-11-27", "2025-12-13"],
    "summer": [], 
}

# ─── Landsat Band Configuration (Level-2 Surface Reflectance) ─────────────────
LANDSAT_BANDS = {
    "blue":   "B2",
    "green":  "B3",
    "red":    "B4",
    "nir":    "B5",
    "swir1":  "B6",
    "swir2":  "B7",
    "tir1":   "B10",   # Thermal — 10.60–11.19 µm — used for LST
    # "tir2":   "B11",   # Thermal — 11.50–12.51 µm — used in split-window (optional)
}

# ─── LST Retrieval Constants ──────────────────────────────────────────────────
# Mono-Window Algorithm constants for Landsat 8 Band 10
LST_WAVELENGTH_BAND10 = 10.895e-6    # metres (centre of 10.60–11.19 µm)
PLANCK_H   = 6.626e-34               # J·s
SPEED_C    = 2.998e8                 # m/s
BOLTZMANN  = 1.381e-23               # J/K
RHO        = (PLANCK_H * SPEED_C) / BOLTZMANN   # = 0.01438 m·K

# Emissivity lookup by NDVI thresholds (Sobrino et al. 2004)
NDVI_THRESH_BARE   = 0.2             # below → bare soil emissivity
NDVI_THRESH_VEG    = 0.5             # above → full vegetation emissivity
EMISSIVITY_BARE    = 0.97
EMISSIVITY_VEG     = 0.99
EMISSIVITY_MIX_C   = 0.005          # fractional vegetation correction factor

# ─── Spectral Index Formulas (band names reference LANDSAT_BANDS) ─────────────
# NDVI  = (NIR - Red) / (NIR + Red)
# NDBI  = (SWIR1 - NIR) / (SWIR1 + NIR)
# NDWI  = (Green - NIR) / (Green + NIR)
# Albedo = 0.356*B2 + 0.130*B4 + 0.373*B5 + 0.085*B6 + 0.072*B7 − 0.0018
#          (Liang 2001 — standard broadband albedo formula for Landsat)

# ─── LULC Classes (ESA WorldCover 10m Classification) ─────────────────────────
LULC_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
}
N_LULC_CLASSES = len(LULC_CLASSES)

# ─── Input Feature Channels (ORDER MATTERS — must be consistent everywhere) ───
FEATURE_CHANNELS = [
    "ndvi",           # 0
    "ndbi",           # 1
    "ndwi",           # 2
    "albedo",         # 3
    "wind_speed",     # 4
    "wind_direction", # 5
    "cloud_cover",     # 6
    "air_temperature", # 7
    "palsar_hh",       # 8
    "palsar_hv",       # 9
    "elevation",      # 10
    "slope",          # 11
    "curvature",      # 12
    "aspect",         # 13
    "building_height",# 14
    "svf",            # 15
    "lulc",           # 16
]
N_CHANNELS = len(FEATURE_CHANNELS)  # = 17

# ─── Superpixel (SLIC) Parameters ─────────────────────────────────────────────
SLIC_N_SEGMENTS   = 1000            # approximate number of superpixels
SLIC_COMPACTNESS  = 10              # spatial compactness (higher = more square)
SLIC_SIGMA        = 1               # Gaussian smoothing before segmentation

# ─── GeoViT Model Hyperparameters ─────────────────────────────────────────────
MODEL_CONFIG = {
    "input_channels": N_CHANNELS,   # 13
    "patch_size":     16,           # spatial patch size in pixels
    "dim":            128,          # transformer embedding dimension
    "depth":          4,            # number of transformer encoder layers
    "heads":          4,            # number of attention heads
    "mlp_dim":        256,          # FFN hidden dimension
    "dropout":        0.0,          # dropout rate (not stated in paper; use 0)
    "output_dim":     1,            # single LST value per pixel
}

# ─── Training Hyperparameters ─────────────────────────────────────────────────
TRAIN_CONFIG = {
    "epochs":         50,
    "learning_rate":  1e-4,
    "optimizer":      "adam",
    "loss":           "mse",
    "train_split":    0.80,
    "val_split":      0.20,
    "batch_size":     512,          # not specified in paper; 512 is a practical default
    "random_seed":    42,           # not specified in paper; use 42 for reproducibility
    "num_workers":    4,
    "pin_memory":     True,
    "save_every_n_epochs": 10,
}

# ─── Adam Optimizer Defaults ──────────────────────────────────────────────────
ADAM_CONFIG = {
    "betas":   (0.9, 0.999),
    "eps":     1e-8,
    "weight_decay": 0.0,
}

# ─── UHI Classification Thresholds ───────────────────────────────────────────
# Post-hoc classification of continuous LST into thermal categories
N_THERMAL_CLASSES = 5               # natural breaks (Jenks) on full LST distribution
UHI_BINARY_CLASSES = {
    "Low UHI":  0,                  # lower thermal classes combined
    "High UHI": 1,                  # upper thermal classes combined
}
# Binary split: lower 75% → Low UHI, upper 25% → High UHI
# (inferred from confusion matrix support: 143,982 low vs 47,994 high = 75/25 split)
UHI_BINARY_PERCENTILE_THRESHOLD = 75

# ─── Expected Output Ranges (for sanity checks) ───────────────────────────────
EXPECTED_RANGES = {
    "lst_winter_observed":   (39, 56),   # °C, Landsat-derived
    "lst_summer_observed":   (48, 65),   # °C, Landsat-derived
    "lst_winter_predicted":  (45, 57),   # °C, GeoViT model output
    "lst_summer_predicted":  (44, 65),   # °C, GeoViT model output
    "uhi_summer_landsat":    (2, 14),    # °C intensity
    "uhi_winter_landsat":    (2, 11),    # °C intensity
    "uhi_summer_predicted":  (0.5, 4),   # °C intensity (smoothed)
    "uhi_winter_predicted":  (2, 8),     # °C intensity
    "rmse_normalized_max":   0.41,
}
