"""
Mask-aware evaluation metrics for MasharikiWeather,
wrapping native Torch Spatiotemporal (TSL) PyTorch metrics.
"""
import torch
from typing import Optional, Dict
from tsl.metrics.torch import MaskedMAE, MaskedMSE, MaskedMRE, MaskedMAPE

def masked_mae(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Masked Mean Absolute Error (MAE) using TSL native metric.
    """
    metric_fn = MaskedMAE()
    return metric_fn(y_hat=preds, y=targets, mask=mask)

def masked_mse(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Masked Mean Squared Error (MSE) using TSL native metric.
    """
    metric_fn = MaskedMSE()
    return metric_fn(y_hat=preds, y=targets, mask=mask)

def masked_rmse(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Masked Root Mean Squared Error (RMSE) derived from TSL MaskedMSE.
    """
    mse_val = masked_mse(preds, targets, mask=mask)
    return torch.sqrt(mse_val)

def masked_mape(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Masked Mean Absolute Percentage Error (MAPE).
    """
    metric_fn = MaskedMAPE()
    return metric_fn(y_hat=preds, y=targets, mask=mask)

class WeatherMetrics:
    """
    Collection of standardized spatiotemporal metrics.
    """
    @staticmethod
    def compute(preds: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        return {
            'mae': masked_mae(preds, targets, mask).item(),
            'mse': masked_mse(preds, targets, mask).item(),
            'rmse': masked_rmse(preds, targets, mask).item(),
            'mape': masked_mape(preds, targets, mask).item()
        }
