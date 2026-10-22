# SwinGeo: Swin-UNet for High-Resolution Urban Heat Island (UHI) Mapping

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**SwinGeo** is a deep learning framework that downscales coarse satellite thermal observations to **10 m continuous Land Surface Temperature (LST)** fields and classifies **Urban Heat Island (UHI) thermal hotspots** across heterogeneous urban landscapes (study area: Ahmedabad, Gujarat, India).

The model couples a **Swin Transformer Tiny (Swin-T)** hierarchical feature extraction backbone with a symmetric **U-Net decoder** via skip connections, preserving fine spatial features (urban roads, vegetation patches, and river corridors) while capturing multi-scale spatial contexts.

---

## 🏛️ Architecture Overview

```
Input Patch: (B, C, 32, 32)
   │
   ▼
Patch Embedding Conv2d(C, 96, k=4, s=4)  ────────► Stage 1 (B, 8, 8, 96) ──────┐
                                                           │                    │
                                                           ▼                    │
                                                   Stage 2 (B, 4, 4, 192) ──────┼───────┐
                                                           │                    │       │
                                                           ▼                    │       │
                                                   Stage 3 (B, 2, 2, 384) ──────┼───────┼───────┐
                                                           │                    │       │       │
                                                           ▼                    │       │       │
                                                   Stage 4 (B, 1, 1, 768)       │       │       │
                                                           │                    │       │       │
                                                           ▼                    │       │       │
                                                  ConvTranspose2d(2x2)          │       │       │
                                                           │                    │       │       │
                                                           ▼                    ▼       │       │
                                                      Concat ──► Conv (256, 2, 2)       │       │
                                                                       │                │       │
                                                                       ▼                │       │
                                                              ConvTranspose2d(2x2)      │       │
                                                                       │                │       │
                                                                       ▼                ▼       │
                                                                  Concat ──► Conv (128, 4, 4)   │
                                                                                   │            │
                                                                                   ▼            │
                                                                          ConvTranspose2d(2x2)  │
                                                                                   │            │
                                                                                   ▼            ▼
                                                                              Concat ──► Conv (64, 8, 8)
                                                                                               │
                                                                                               ▼
                                                                                     ConvTranspose2d(4x4)
                                                                                               │
                                                                                               ▼
                                                                                          1x1 Conv Head
                                                                                               │
                                                                                               ▼
                                                                                      Output: (B, 1, 32, 32)
```

* **Shifted Window Self-Attention:** Computes self-attention within local non-overlapping $7\times7$ windows, alternating with shifted windows to achieve $\mathcal{O}(N)$ computational scaling.
* **Skip Connections:** Encoder feature maps are permuted and concatenated into the decoder to recover high-frequency boundary details.
* **Loss Function:** Masked Huber Loss (`nn.SmoothL1Loss(reduction='none')`) with invalid pixel masking.
* **Inference:** Overlapping sliding window patch extraction with 2D Gaussian apodization blending to eliminate tiling seams.

---

## 🛰️ Input Features & Geospatial Data Sources

The model integrates multi-modal remote sensing and meteorological datasets aligned to a common $10\,\text{m}$ grid (`EPSG:32643`, UTM Zone 43N):

| Category | Source | Features / Derived Products |
| :--- | :--- | :--- |
| **Optical & Thermal** | Landsat 8/9 OLI/TIRS | Band 10 Brightness Temperature, NDVI, NDBI, NDWI, Broadband Albedo |
| **Land Cover** | ESA WorldCover (10m) | 8-class categorical land use / land cover |
| **Topography** | SRTM DEM | Elevation, Slope, Aspect, Curvature |
| **Urban Morphology** | SRTM / ALOS PALSAR | Building Height (nDSM proxy), Sky View Factor (SVF), HH/HV Polarizations |
| **Meteorology** | ERA5 Reanalysis | 10m Wind Speed, Wind Direction, 2m Air Temp, Total Cloud Cover |

---

## 📊 Performance Benchmark

Evaluated over the Ahmedabad study area:

| Test Date | Season | Model / Method | RMSE ($^\circ\text{C}$) | MAE ($^\circ\text{C}$) | $R^2$ | Hotspot Accuracy |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **2025-12-13** | Winter (Peak) | **SwinGeo (Swin-UNet)** | **$1.51$** | **$1.06$** | **$+0.55$** | **$84.3\%$** |
| **2025-11-20** | Late Autumn | **SwinGeo (Swin-UNet)** | **$2.29$** | **$1.30$** | $-0.28$ | **$80.8\%$** |
| **2025-01-28** | Mid-Winter | Random Forest Baseline | $8.26$ | $6.08$ | $-7.56$ | $55.8\%$ |
| **2025-01-28** | Mid-Winter | **SwinGeo (Swin-UNet)** | **$5.22$** | **$4.26$** | **$-2.42$** | **$67.6\%$** |

---

## 📂 Project Structure

```
geovit_uhi/
├── config.py              # Central configuration (grid, CRS, dates, channels, hyperparameters)
├── main.py                # Pipeline orchestrator: preprocessing, training, inference
├── baseline_rf.py         # Pixel-wise Random Forest baseline for comparison
├── bias_correction.py     # Seasonal calibration and evaluation metric calculations
├── submit_job.sh          # SLURM script for cluster execution with CUDA
├── model/
│   ├── swin_geo.py        # Swin-UNet (SwinGeo) architecture implementation
│   └── geovit.py          # Legacy pixel-wise ViT module
├── dataset/
│   ├── builder.py         # Multi-band raster stack ingestion & Z-score normalization
│   └── loader.py          # PyTorch patch Dataset & DataLoader with spatial augmentations
├── preprocessing/
│   ├── alignment.py       # Master grid reprojection, resampling, and raster I/O
│   ├── indices.py         # NDVI, NDBI, NDWI, and broadband albedo formulas
│   ├── lst_retrieval.py   # Mono-Window Algorithm for Landsat Band 10 LST retrieval
│   ├── dem_derivatives.py # Slope, aspect, and curvature extraction via richdem
│   ├── urban_morphology.py# Building height proxy and Sky View Factor (SVF)
│   ├── era5_processing.py # NetCDF extraction for wind vectors and temperature
│   └── lulc_classification.py # LULC helpers
├── analysis/
│   ├── accuracy.py        # Regression (RMSE, MAE, R2) and classification metrics
│   ├── uhi_profiling.py   # Urban Heat Island intensity relative to rural baselines
│   ├── visualization.py   # Plotting loss curves, confusion matrices, and LST maps
│   └── xai.py             # SHAP explainability analysis
├── requirements.txt       # Python dependencies
└── hpc_work_update.md     # Detailed HPC experimental notes and results log
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

```bash
# Clone the repository
git clone <YOUR_REPO_URL>
cd geovit_uhi

# Create and activate conda environment
conda create -n urban_env python=3.10 -y
conda activate urban_env

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Pipeline

```bash
# Full end-to-end execution (Preprocessing -> Training -> Inference)
python main.py --step all --device cuda

# Individual steps
python main.py --step preprocess
python main.py --step train --epochs 50 --batch_size 64 --lr 1e-4
python main.py --step infer

# Run Random Forest baseline
python baseline_rf.py
```

### 3. Running on SLURM / HPC Cluster

```bash
sbatch submit_job.sh
```

---

## 📜 License

This project is licensed under the MIT License.
