"""
Accuracy and performance metrics.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_regression_metrics(y_true, y_pred):
    """Compute MSE, RMSE, MAE, R2"""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    if len(y_t) == 0:
        return {}
        
    mse = mean_squared_error(y_t, y_p)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_t, y_p)
    r2 = r2_score(y_t, y_p)
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }

from sklearn.metrics import classification_report, confusion_matrix

def compute_classification_metrics(y_true, y_pred, threshold):
    """
    Compute Precision, Recall, F1, and Confusion Matrix.
    y_true and y_pred are continuous LST values. We threshold them to get binary classes.
    0 = Low UHI, 1 = High UHI.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    if len(y_t) == 0:
        return {}
        
    y_t_class = (y_t >= threshold).astype(int)
    y_p_class = (y_p >= threshold).astype(int)
    
    report = classification_report(y_t_class, y_p_class, target_names=["Low UHI", "High UHI"], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_t_class, y_p_class)
    
    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
