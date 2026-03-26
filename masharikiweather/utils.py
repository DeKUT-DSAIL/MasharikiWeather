"""
Mathematical bridging utilities for mapping dense PyG topological arrays 
into rigid, variance-protected Scientific Deep Learning paradigms.
"""
import torch
import torch.nn as nn

class SoftplusLink(nn.Module):
    """
    Scientifically robust activation module shielding Gradient Descent 
    from variance explosions when interacting with Zero-Inflated Poisson-Gamma datasets 
    (like raw extreme precipitation spanning 0-300mm+).
    
    It binds neural network predictions (which exist in normalized, stable latent bounds) 
    back into mathematically physical positive space natively. Utilizing this
    inverse-derivative guarantees exploding Tweedie deviances are damped securely 
    during `.backward()`.
    """
    def __init__(self, beta: float = 1.0, threshold: float = 20.0, epsilon: float = 1e-6):
        super().__init__()
        self.softplus = nn.Softplus(beta=beta, threshold=threshold)
        self.epsilon = epsilon
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Addition of a tiny algorithmic epsilon securely prevents zero-deviance `NaN` mathematical collapses
        return self.softplus(x) + self.epsilon


class SparseStationLoss(nn.Module):
    """
    A PyTorch module wrapper translating mathematical dense geometric predictions 
    against highly scattered, sparse observational TAHMO Station vectors.
    Executes loss strictly across valid observational topography.
    """
    def __init__(self, base_loss_fn: nn.Module):
        """
        Args:
            base_loss_fn: The raw PyTorch metric (e.g. TweedieDevianceLoss, CRPS)
                          evaluating exactly on isolated physical coordinates.
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: The mapped topological node prediction bounds precisely corresponding to station coordinates.
            targets: The raw observational bounds possessing NaNs dynamically. 
            mask: The boolean computational matrix tracking structural validity.
        """
        if predictions.shape != targets.shape:
             raise ValueError(f"Structural topology mismatch: {predictions.shape} vs {targets.shape}")
             
        if mask.shape != targets.shape:
             mask = mask.expand_as(targets)
        
        # Geometrically extract exactly the true numerical arrays
        valid_preds = predictions[mask]
        valid_targets = targets[mask]
        
        # Fault-tolerance: In epochs where structural masks fail universally offline
        if valid_targets.numel() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
            
        return self.base_loss_fn(valid_preds, valid_targets)
