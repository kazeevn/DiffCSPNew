"""Diffusion noise schedules and wrapped normal distributions on the torus.

Implements BetaScheduler (for lattice noise diffusion) and SigmaScheduler
(for periodic fractional coordinate diffusion with wrapped normal scoring).
"""

import numpy as np
import torch
import torch.nn as nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule as proposed by Nichol & Dhariwal (2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1.0 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Standard linear beta schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Quadratic beta schedule."""
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Sigmoid beta schedule."""
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def p_wrapped_normal(
    x: torch.Tensor, sigma: torch.Tensor, n_wraps: int = 10, period: float = 1.0
) -> torch.Tensor:
    """Wrapped normal density on the 1D circle / torus of period T."""
    p_sum = torch.zeros_like(x)
    for i in range(-n_wraps, n_wraps + 1):
        diff = x + period * i
        p_sum += torch.exp(-0.5 * (diff / sigma) ** 2)
    return p_sum


def d_log_p_wrapped_normal(
    x: torch.Tensor, sigma: torch.Tensor, n_wraps: int = 10, period: float = 1.0
) -> torch.Tensor:
    """Score function (d/dx log p(x)) of the wrapped normal distribution."""
    score_numerator = torch.zeros_like(x)
    for i in range(-n_wraps, n_wraps + 1):
        diff = x + period * i
        score_numerator += (diff / (sigma**2)) * torch.exp(-0.5 * (diff / sigma) ** 2)
    p_val = p_wrapped_normal(x, sigma, n_wraps=n_wraps, period=period)
    return score_numerator / (p_val + 1e-12)


def sigma_norm(sigma: torch.Tensor, period: float = 1.0, sn: int = 10000) -> torch.Tensor:
    """Computes expected norm of wrapped score for variance normalization."""
    sigmas = sigma[None, :].repeat(sn, 1)
    x_sample = sigma * torch.randn_like(sigmas)
    x_sample = x_sample % period
    normal_score = d_log_p_wrapped_normal(x_sample, sigmas, period=period)
    return (normal_score**2).mean(dim=0)


class BetaScheduler(nn.Module):
    """Beta scheduler for continuous lattice diffusion."""

    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sigmas: torch.Tensor

    def __init__(
        self,
        timesteps: int = 1000,
        scheduler_mode: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps

        if scheduler_mode == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif scheduler_mode == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == "quadratic":
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown beta schedule mode: {scheduler_mode}")

        betas = torch.cat([torch.zeros(1), betas], dim=0)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        sigmas = torch.zeros_like(betas)
        sigmas[1:] = betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
        sigmas = torch.sqrt(torch.clamp(sigmas, min=0.0))

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sigmas", sigmas)

    def uniform_sample_t(self, batch_size: int, device: str | torch.device) -> torch.Tensor:
        """Uniformly samples discrete timesteps in [1, timesteps]."""
        return torch.randint(1, self.timesteps + 1, (batch_size,), device=device)


class SigmaScheduler(nn.Module):
    """Sigma scheduler for fractional coordinate diffusion on the torus."""

    sigmas: torch.Tensor
    sigmas_norm: torch.Tensor

    def __init__(self, timesteps: int = 1000, sigma_begin: float = 0.005, sigma_end: float = 0.5) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), timesteps)), dtype=torch.float32
        )
        sigmas_norm_vals = sigma_norm(sigmas)

        self.register_buffer("sigmas", torch.cat([torch.zeros(1), sigmas], dim=0))
        self.register_buffer("sigmas_norm", torch.cat([torch.ones(1), sigmas_norm_vals], dim=0))

    def uniform_sample_t(self, batch_size: int, device: str | torch.device) -> torch.Tensor:
        """Uniformly samples discrete timesteps in [1, timesteps]."""
        return torch.randint(1, self.timesteps + 1, (batch_size,), device=device)


class SinusoidalTimeEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Computes sinusoidal embeddings for input timesteps (B,)."""
        device = time.device
        half_dim = self.dim // 2
        emb_scale = np.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        args = time[:, None].float() * freqs[None, :]
        return torch.cat((args.sin(), args.cos()), dim=-1)
