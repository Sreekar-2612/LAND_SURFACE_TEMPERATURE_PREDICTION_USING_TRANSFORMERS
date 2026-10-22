"""
Land Use / Land Cover (LULC) classification from Sentinel-2 MSI imagery.

6-class scheme:
  0 = Vegetation
  1 = Sand
  2 = Bare Soil
  3 = Buildings
  4 = Roads
  5 = Bricks and Concrete

Method: Random Forest classifier on Sentinel-2 spectral bands + indices.
Training labels must be provided as a GeoTIFF with integer class values.
If no training labels available, use unsupervised K-means as fallback.

The paper validates LULC against WorldView imagery at 1 m resolution.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


def prepare_feature_matrix(bands_dict):
    """
    Stack Sentinel-2 bands and indices into a (H*W, n_features) matrix.
    bands_dict: dict of 2D arrays keyed by band/index name.
    """
    stack = np.stack(list(bands_dict.values()), axis=-1)  # (H, W, C)
    h, w, c = stack.shape
    return stack.reshape(-1, c), (h, w)


def train_rf_classifier(X, y, n_estimators=100, random_state=42):
    """Train Random Forest on labelled samples."""
    mask = ~np.isnan(X).any(axis=1) & (y > -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[mask])
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state,
                                 n_jobs=-1, class_weight="balanced")
    clf.fit(X_scaled, y[mask])
    return clf, scaler


def classify_lulc(bands_dict, clf, scaler, shape):
    """Run trained classifier on the full image."""
    X, (h, w) = prepare_feature_matrix(bands_dict)
    nan_mask = np.isnan(X).any(axis=1)
    X_clean = np.nan_to_num(X, nan=0.0)
    X_scaled = scaler.transform(X_clean)
    preds = clf.predict(X_scaled).astype(np.float32)
    preds[nan_mask] = np.nan
    return preds.reshape(h, w)


def kmeans_lulc_fallback(bands_dict, n_clusters=6, random_state=42):
    """
    Unsupervised fallback when labelled training data is unavailable.
    Returns cluster labels (0–5) — manual re-labelling required afterwards.
    """
    X, (h, w) = prepare_feature_matrix(bands_dict)
    nan_mask = np.isnan(X).any(axis=1)
    X_clean = np.nan_to_num(X, nan=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled).astype(np.float32)
    labels[nan_mask] = np.nan
    return labels.reshape(h, w)
