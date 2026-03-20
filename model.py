from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyDiffusionModel(nn.Module):
    def __init__(
        self,
        frame_count: int = 20,
        latent_size: int = 8,
        hidden_dim: int = 64,
        cond_dim: int = 16,
        time_dim: int = 16,
    ) -> None:
        super().__init__()
        self.frame_count = frame_count
        self.latent_size = latent_size
        self.latent_dim = frame_count * latent_size * latent_size

        self.digit_embedding = nn.Embedding(3, cond_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(4, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim + cond_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )

    def _time_features(self, timesteps: torch.Tensor, num_steps: int) -> torch.Tensor:
        t = timesteps.float() / max(num_steps - 1, 1)
        features = torch.stack(
            [
                t,
                t * t,
                torch.sin(math.pi * t),
                torch.cos(math.pi * t),
            ],
            dim=-1,
        )
        return self.time_proj(features)

    def forward(self, noisy_latent: torch.Tensor, digits: torch.Tensor, timesteps: torch.Tensor, num_steps: int) -> torch.Tensor:
        batch_size = noisy_latent.shape[0]
        x = noisy_latent.reshape(batch_size, -1)
        cond = self.digit_embedding(digits)
        t_embed = self._time_features(timesteps, num_steps)
        output = self.net(torch.cat([x, cond, t_embed], dim=-1))
        return output.view(batch_size, self.frame_count, self.latent_size, self.latent_size)


class DiffusionSchedule:
    def __init__(self, num_steps: int = 50, beta_start: float = 1e-4, beta_end: float = 2e-2, device: str | torch.device = "cpu") -> None:
        self.num_steps = num_steps
        self.device = torch.device(device)

        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32, device=self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.alpha_bars_prev = torch.cat([torch.ones(1, device=self.device), alpha_bars[:-1]])
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
        self.sqrt_recip_alphas = torch.rsqrt(alphas)
        self.posterior_variance = betas * (1.0 - self.alpha_bars_prev) / (1.0 - alpha_bars)

    def to(self, device: str | torch.device) -> "DiffusionSchedule":
        return DiffusionSchedule(
            num_steps=self.num_steps,
            beta_start=float(self.betas[0].item()),
            beta_end=float(self.betas[-1].item()),
            device=device,
        )

    def q_sample(self, clean_latent: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        shape = (clean_latent.shape[0],) + (1,) * (clean_latent.ndim - 1)
        sqrt_alpha_bar = self.sqrt_alpha_bars[timesteps].view(shape)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[timesteps].view(shape)
        return sqrt_alpha_bar * clean_latent + sqrt_one_minus * noise

    def p_sample(self, model: TinyDiffusionModel, noisy_latent: torch.Tensor, digits: torch.Tensor, step_index: int) -> torch.Tensor:
        batch_size = noisy_latent.shape[0]
        t = torch.full((batch_size,), step_index, device=noisy_latent.device, dtype=torch.long)
        predicted_x0 = model(noisy_latent, digits, t, self.num_steps).clamp(-1.0, 1.0)

        beta_t = self.betas[step_index]
        alpha_t = self.alphas[step_index]
        alpha_bar_t = self.alpha_bars[step_index]
        alpha_bar_prev = self.alpha_bars_prev[step_index]
        posterior_mean_coef_x0 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
        posterior_mean_coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        mean = posterior_mean_coef_x0 * predicted_x0 + posterior_mean_coef_xt * noisy_latent

        if step_index == 0:
            return predicted_x0

        noise = torch.randn_like(noisy_latent)
        return mean + torch.sqrt(self.posterior_variance[step_index]) * noise


def encode_video(video: torch.Tensor, latent_size: int = 8) -> torch.Tensor:
    single_video = video.ndim == 4
    if single_video:
        video = video.unsqueeze(0)

    batch_size, frame_count, channels, _, _ = video.shape
    if channels != 1:
        raise ValueError("Expected grayscale video with a single channel.")

    flat = video.view(batch_size * frame_count, 1, video.shape[-2], video.shape[-1])
    latent = F.interpolate(flat, size=(latent_size, latent_size), mode="bilinear", align_corners=False)
    latent = latent.view(batch_size, frame_count, latent_size, latent_size)
    return latent[0] if single_video else latent


def decode_video(latent: torch.Tensor, output_size: int = 32) -> torch.Tensor:
    single_video = latent.ndim == 3
    if single_video:
        latent = latent.unsqueeze(0)

    batch_size, frame_count, _, _ = latent.shape
    flat = latent.view(batch_size * frame_count, 1, latent.shape[-2], latent.shape[-1])
    video = F.interpolate(flat, size=(output_size, output_size), mode="bilinear", align_corners=False)
    video = video.view(batch_size, frame_count, 1, output_size, output_size)
    return video[0] if single_video else video


@torch.no_grad()
def sample_video(
    model: TinyDiffusionModel,
    schedule: DiffusionSchedule,
    digit: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    model.eval()
    device = torch.device(device)
    latent = torch.randn(1, model.frame_count, model.latent_size, model.latent_size, device=device)
    digits = torch.tensor([digit], device=device, dtype=torch.long)

    for step_index in reversed(range(schedule.num_steps)):
        latent = schedule.p_sample(model, latent, digits, step_index)

    video = decode_video(latent, output_size=32)
    return video.clamp(-1.0, 1.0).cpu()


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
