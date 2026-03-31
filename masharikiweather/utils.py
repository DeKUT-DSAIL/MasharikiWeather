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


class TweedieDevianceLoss(nn.Module):
    """
    Likelihood-based loss for zero-inflated, semi-continuous precipitation.
    Alignment: Rule 3 & Hunt (2025) "Stop using root-mean-square error as a precipitation target!"
    
    Proof of Deviance:
    The Tweedie distribution with $1 < p < 2$ represents a compound Poisson-Gamma 
    process. The unit deviance $d(y, \mu)$ is derived from the log-likelihood 
    of the exponential dispersion family:
    
    $d(y, \mu) = 2 \left[ \frac{y^{2-p}}{(1-p)(2-p)} - \frac{y \mu^{1-p}}{1-p} + \frac{\mu^{2-p}}{2-p} \right]$
    
    Properties:
    1. For $y=0$ (dry days), the first two terms vanish (as $2-p > 0$), leaving 
       the third term to penalize non-zero $\mu$.
    2. For $y>0$ (wet days), the loss penalizes the distance between $\mu$ 
       and $y$ according to a Gamma-like skew.
    3. Natively handles zero-inflation without $\log(1+x)$ hacks.
    """
    def __init__(self, p: float = 1.5, epsilon: float = 1e-6):
        """
        Args:
            p: Tweedie power parameter. $p=1.5$ is typical for daily rainfall.
            epsilon: Numerical stability constant.
        """
        super().__init__()
        if not (1 < p < 2):
            raise ValueError("Tweedie index p must be in (1, 2).")
        self.p = p
        self.epsilon = epsilon

    def forward(self, mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu: Model predictions (intensities). Must be positive (Softplus).
            y: Raw precipitation targets (mm).
        """
        # Ensure mu is strictly positive for power operations
        mu = torch.clamp(mu, min=self.epsilon)
        
        # Unit Deviance components
        # Note: y=0 is handled correctly as (2-p) > 0
        term1 = torch.pow(y, 2 - self.p) / ((1 - self.p) * (2 - self.p))
        term2 = (y * torch.pow(mu, 1 - self.p)) / (1 - self.p)
        term3 = torch.pow(mu, 2 - self.p) / (2 - self.p)
        
        dev = 2 * (term1 - term2 + term3)
        return torch.mean(dev)
