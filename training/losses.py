import torch
import torch.nn as nn

class MSELossWrapper(nn.Module):
    """
    Wrapper for MSE Loss to match the paper's specification.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)
