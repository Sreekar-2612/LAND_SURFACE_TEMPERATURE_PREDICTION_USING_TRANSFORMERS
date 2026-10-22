"""
Visualization scripts.
"""

import matplotlib.pyplot as plt
import numpy as np
from config import FIGURE_DIR

def plot_lst_map(lst_array, out_filename, cmap="inferno", title="LST Map"):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(lst_array, cmap=cmap)
    plt.colorbar(label="Temperature (°C)")
    plt.title(title)
    plt.axis("off")
    plt.savefig(FIGURE_DIR / out_filename, bbox_inches="tight")
    plt.close()

import seaborn as sns

def plot_training_curves(train_losses, val_losses, out_filename="loss_curve.png"):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURE_DIR / out_filename, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(cm, out_filename="confusion_matrix.png"):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low UHI", "High UHI"], yticklabels=["Low UHI", "High UHI"])
    plt.title("UHI Classification Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.savefig(FIGURE_DIR / out_filename, bbox_inches="tight")
    plt.close()

def plot_lulc_lst_boxplot(lst_values, lulc_values, lulc_classes_dict, out_filename="lst_lulc_boxplot.png"):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    mask = ~np.isnan(lst_values) & ~np.isnan(lulc_values)
    lst_clean = lst_values[mask]
    lulc_clean = lulc_values[mask].astype(int)
    
    # Filter to only valid LULC classes
    valid_classes = list(lulc_classes_dict.keys())
    valid_mask = np.isin(lulc_clean, valid_classes)
    
    lst_clean = lst_clean[valid_mask]
    lulc_clean = lulc_clean[valid_mask]
    
    # Map class index to name
    lulc_names = [lulc_classes_dict[c] for c in lulc_clean]
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=lulc_names, y=lst_clean, order=list(lulc_classes_dict.values()), palette="Set2")
    plt.title("LST Distribution across LULC Classes", fontsize=14)
    plt.xlabel("LULC Class", fontsize=12)
    plt.ylabel("LST (°C)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(FIGURE_DIR / out_filename, bbox_inches="tight")
    plt.close()
