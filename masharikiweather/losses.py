"""
WeatherBench 2 Standardized Atmospheric Loss and Evaluation Metrics.
Implementing exact differentiable and non-differentiable mathematically sound metrics 
assessing generative sharpness, compound probability spaces, and true threshold skill limits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TweedieDevianceLoss(nn.Module):
    """
    Computes the Tweedie deviance loss exactly suited for compound Poisson-Gamma distributions (1 < p < 2).
    Mathematically optimal for zero-inflated, heavy-tailed precipitation data natively bounding variance.
    """
    def __init__(self, p: float = 1.5, eps: float = 1e-8):
        super().__init__()
        assert 1.0 < p < 2.0, "Tweedie variance power `p` must be strictly bounded between 1 (Poisson) and 2 (Gamma)."
        self.p = p
        self.eps = eps

    def forward(self, mu: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        mu: The predicted mean (emitted strictly post-SoftplusLink), must be > 0.
        y: The true raw observation, bounding at 0.0.
        mask: Boolean mask indicating physical valid sensors vs hardware outages.
        """
        # Clamp bounds strictly preventing exponential `NaN` logic crashes
        mu = torch.clamp(mu, min=self.eps)
        
        # Guard against unpadded NaNs bypassing dataloader
        y = torch.nan_to_num(y, nan=0.0)
        y = torch.clamp(y, min=0.0)

        a = y * (torch.pow(y + self.eps, 1 - self.p) - torch.pow(mu, 1 - self.p)) / (1 - self.p)
        b = (torch.pow(y + self.eps, 2 - self.p) - torch.pow(mu, 2 - self.p)) / (2 - self.p)
        
        deviance = 2 * (a - b)
        
        # Apply sparse valid sensor boolean mask
        if mask is not None:
            deviance = deviance * mask.float()
            return deviance.sum() / (mask.float().sum() + self.eps)
            
        return deviance.mean()


class DifferentiableFSSLoss(nn.Module):
    """
    Differentiable Fractions Skill Score (FSS) computationally scaling convective thresholds.
    Utilizes 2D Average Pooling measuring spatial displacement of exact sub-grid fractional weather conditions.
    """
    def __init__(self, thresholds: list, window_sizes: list):
        super().__init__()
        self.thresholds = thresholds
        self.window_sizes = window_sizes

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        preds/targets structural constraint shape: (Batch, Channels, H, W).
        """
        loss = 0.0
        # Smoothed approximation (Sigmoid) replacing strict Boolean cutoffs making the metric perfectly differentiable
        temperature = 0.1 
        
        for t in self.thresholds:
            p_frac = torch.sigmoid((preds - t) / temperature)
            t_frac = (targets > t).float()
            
            for w in self.window_sizes:
                p_pool = F.avg_pool2d(p_frac, kernel_size=w, stride=1, padding=w//2)
                t_pool = F.avg_pool2d(t_frac, kernel_size=w, stride=1, padding=w//2)
                
                # FSS bounds = 1 - (MSE_pool / (Var_pool_p + Var_pool_t))
                mse = torch.mean((p_pool - t_pool) ** 2)
                ref_mse = torch.mean(p_pool ** 2) + torch.mean(t_pool ** 2) + 1e-8
                fss = 1.0 - (mse / ref_mse)
                
                loss += (1.0 - fss)
        return loss / (len(self.thresholds) * len(self.window_sizes))


class RAPSDLoss(nn.Module):
    """
    Radially Averaged Power Spectral Density (RAPSD) Loss matrix.
    Specifically penalizes Generative architectures executing physical 'spatial blurring' iteratively 
    by enforcing the 2D FFT energy spectrum boundaries against the true topological decay gradient.
    """
    def __init__(self):
        super().__init__()

    def _get_rapsd(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape
        fft2d = torch.fft.fftshift(torch.fft.fft2(img))
        magnitude_spectrum = torch.abs(fft2d) ** 2

        # Radial computational structure coordinates
        y, x = torch.meshgrid(torch.arange(h) - h//2, torch.arange(w) - w//2, indexing='ij')
        r = torch.sqrt(x**2 + y**2).to(img.device)
        r = torch.round(r).long()

        tbin = torch.bincount(r.flatten(), weights=magnitude_spectrum.flatten())
        nr = torch.bincount(r.flatten()).clamp(min=1)
        return tbin / nr

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        b, c, h, w = preds.shape
        total_loss = 0.0
        for i in range(b):
            for j in range(c):
                p_rapsd = self._get_rapsd(preds[i, j])
                t_rapsd = self._get_rapsd(targets[i, j])
                
                # Enforce identical decay limits mathematically scaling logarithmic spectrum coordinates
                total_loss += F.mse_loss(torch.log1p(p_rapsd), torch.log1p(t_rapsd))
        return total_loss / (b * c)


class CRPSLoss(nn.Module):
    """
    Continuous Ranked Probability Score (CRPS).
    Measures strict generic probabilistic calibration natively scaling empirical observations.
    """
    def __init__(self):
        super().__init__()

    def forward(self, preds_ensemble: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        preds_ensemble constraints arrays: (Batch, Ensemble_Members, ...)
        targets arrays: (Batch, ...)
        """
        b, m = preds_ensemble.shape[0], preds_ensemble.shape[1]
        
        targets_expanded = targets.unsqueeze(1)
        mae = torch.abs(preds_ensemble - targets_expanded).mean(dim=1)
        
        # Sort values precisely calculating mean pairwise deviations explicitly without nested looping constraints
        preds_sorted, _ = torch.sort(preds_ensemble, dim=1)
        
        i = torch.arange(1, m + 1, device=preds_ensemble.device).float()
        weights = (2 * i - m - 1) / (m * m)
        spread = torch.sum(preds_sorted * weights.view(1, -1, *([1]*(preds_sorted.ndim - 2))), dim=1)
        
        crps = mae - spread
        return crps.mean()


class SEEPSScore(nn.Module):
    """
    Stable Equitable Error in Probability Space (SEEPS).
    An evaluation scoring tool defining categorical states strictly mapped explicitly to
    dry, light, and heavy geographical conditions (equitably resolving native sampling thresholds).
    """
    def __init__(self, dry_threshold: float = 0.2, p_light: float = 0.5, p_heavy: float = 0.2):
        super().__init__()
        self.dry_threshold = dry_threshold
        self.p_light = p_light 
        self.p_heavy = p_heavy

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, heavy_thresholds: torch.Tensor) -> torch.Tensor:
        """
        heavy_thresholds: Climatological grid values separating 'light' and 'heavy' rain thresholds per-pixel.
        Returns evaluation scalar SEEPS penalty. Exact correct matrix hits are encoded as 0.0.
        """
        t_dry = (targets < self.dry_threshold)
        t_heavy = (targets >= heavy_thresholds)
        t_light = ~(t_dry | t_heavy)
        
        p_dry = (preds < self.dry_threshold)
        p_heavy = (preds >= heavy_thresholds)
        p_light = ~(p_dry | p_heavy)
        
        score = torch.zeros_like(targets, dtype=torch.float32)
        
        p1 = 1.0 - (self.p_light + self.p_heavy)
        p2 = self.p_light
        p3 = self.p_heavy
        
        # Scoring Penalties definitions
        score[t_dry & p_light] = 1.0 / (1.0 - p1)
        score[t_dry & p_heavy] = 1.0 / (1.0 - p1) + 1.0 / p3
        
        score[t_light & p_dry] = 1.0 / p1
        score[t_light & p_heavy] = 1.0 / p3
        
        score[t_heavy & p_dry] = 1.0 / p1 + 1.0 / (1.0 - p1)
        score[t_heavy & p_light] = 1.0 / (1.0 - p1)
        
        return score.mean()
