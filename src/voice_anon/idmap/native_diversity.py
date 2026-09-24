"""Native-space distribution regularization, an opt-in IDMap variant.

No forward clipping, L2 normalization, or assumption that native speakers have
low raw cosine. Statistics are fitted on training speaker centroids only.
"""
import torch
from torch import nn
from torch.nn import functional as F


class NativeDiversityLoss(nn.Module):
    def __init__(self, centroids, components=64, projections=32, seed=20260904):
        super().__init__()
        if centroids.ndim != 2 or len(centroids) < 3 or not torch.isfinite(centroids).all():
            raise ValueError("Finite native training centroids required")
        x = centroids.detach().float()
        mean = x.mean(0)
        _, singular, vh = torch.linalg.svd(x - mean, full_matrices=False)
        k = min(components, len(x)-1, x.shape[1])
        basis = vh[:k].T
        scales = (singular[:k] / (len(x)-1)**.5).clamp_min(1e-4)
        generator = torch.Generator(device=x.device).manual_seed(seed)
        directions = F.normalize(torch.randn(k, projections, generator=generator, device=x.device), dim=0)
        residual = x - mean - ((x-mean) @ basis) @ basis.T
        self.register_buffer("mean", mean)
        self.register_buffer("basis", basis)
        self.register_buffer("scales", scales)
        self.register_buffer("directions", directions)
        self.register_buffer("residual_energy", residual.square().sum(1).mean().clamp_min(1e-6))
        self.register_buffer("norm_scale", x.norm(dim=1).median().clamp_min(1e-6))

    def white(self, x):
        return ((x-self.mean) @ self.basis) / self.scales

    def forward(self, prediction, target):
        if prediction.shape != target.shape or len(prediction) < 2:
            raise ValueError("Paired equal-size native batches required")
        p, t = self.white(prediction), self.white(target.detach())
        centered = p-p.mean(0)
        covariance = centered.T @ centered / (len(p)-1)
        tc = t-t.mean(0)
        target_covariance = tc.T @ tc / (len(t)-1)
        diagonal = (covariance.diag()-target_covariance.diag()).square().mean()
        off = covariance-target_covariance
        off = off-torch.diag_embed(off.diag())
        projected = (p @ self.directions).sort(dim=0).values
        projected_target = (t @ self.directions).sort(dim=0).values
        residual = prediction-self.mean-((prediction-self.mean) @ self.basis) @ self.basis.T
        # Upper bound only: do not encourage off-manifold noise to gain rank.
        residual_penalty = F.relu(residual.square().sum(1).mean()/self.residual_energy-1).square()
        parts = {
            "mean": (p.mean(0)-t.mean(0)).square().mean(),
            "variance": diagonal,
            "covariance": off.square().mean(),
            "projected_distribution": (projected-projected_target).square().mean(),
            "off_subspace_excess": residual_penalty,
            "native_norm": ((prediction.norm(dim=1).sort().values-target.norm(dim=1).sort().values)/self.norm_scale).square().mean(),
        }
        return sum(parts.values()), parts

    def centered_geometry(self, prediction, target):
        p = F.normalize(prediction-self.mean, dim=-1)
        t = F.normalize(target.detach()-self.mean, dim=-1)
        return (p @ p.T-t @ t.T).square().mean()
