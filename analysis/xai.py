"""
SHAP feature importance analysis.
"""

import shap
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import FIGURE_DIR, FEATURE_CHANNELS

def compute_shap_values(model, background_data, test_data, device="cuda"):
    """
    Compute SHAP values using DeepExplainer.
    """
    model.eval()
    background = torch.from_numpy(background_data).to(device)
    test = torch.from_numpy(test_data).to(device)
    
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test)
    
    return shap_values
    
def plot_shap_summary(shap_values, test_data, out_filename="shap_summary.png"):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, test_data, feature_names=FEATURE_CHANNELS, show=False)
    plt.savefig(FIGURE_DIR / out_filename, bbox_inches="tight")
    plt.close()
