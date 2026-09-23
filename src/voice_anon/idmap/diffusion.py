"""Clean-room IDMap-Diff implementation from Liu et al., Eqs. (4)–(7).

The model shares the paper's IDV sampler, pre-processor, and auxiliary
processor with IDMap-MLP.  Its generator is a conditional three-resolution
vector U-Net trained as the score model of a variance-preserving SDE.
"""

from __future__ import annotations

from collections.abc import Iterable
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .mlp import AuxiliaryProcessor, IdentityVectorSampler, PreProcessor


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int = 128) -> None:
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError("time embedding dimension must be even and at least 4")
        self.dimension = dimension

    def forward(self, time: Tensor) -> Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype)
            * (-math.log(10_000.0) / (half - 1))
        )
        phase = time[:, None] * frequencies[None, :] * 1_000.0
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class ConditionalBlock(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int, condition_dimension: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_dimension, output_dimension)
        self.condition = nn.Linear(condition_dimension, 2 * output_dimension)
        self.output = nn.Sequential(
            nn.LayerNorm(output_dimension),
            nn.SiLU(),
            nn.Linear(output_dimension, output_dimension),
        )
        self.residual = (
            nn.Identity()
            if input_dimension == output_dimension
            else nn.Linear(input_dimension, output_dimension)
        )

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        hidden = self.input(values)
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        hidden = hidden * (1.0 + scale) + shift
        return self.output(hidden) + self.residual(values)


class ConditionalVectorUNet(nn.Module):
    """Three feature-map resolutions with IDMap vector and time conditioning."""

    def __init__(
        self,
        *,
        speaker_dimension: int = 512,
        idmap_condition_dimension: int = 1024,
        widths: tuple[int, int, int] = (512, 256, 128),
        time_dimension: int = 128,
    ) -> None:
        super().__init__()
        if len(widths) != 3:
            raise ValueError("the paper configuration uses exactly three resolutions")
        condition_dimension = widths[0]
        self.time = SinusoidalTimeEmbedding(time_dimension)
        self.condition = nn.Sequential(
            nn.Linear(idmap_condition_dimension + time_dimension, condition_dimension),
            nn.SiLU(),
            nn.Linear(condition_dimension, condition_dimension),
        )
        self.input = nn.Linear(speaker_dimension, widths[0])
        self.down_1 = ConditionalBlock(widths[0], widths[0], condition_dimension)
        self.down_2 = ConditionalBlock(widths[0], widths[1], condition_dimension)
        self.down_3 = ConditionalBlock(widths[1], widths[2], condition_dimension)
        self.middle = ConditionalBlock(widths[2], widths[2], condition_dimension)
        self.up_2 = ConditionalBlock(widths[2] + widths[1], widths[1], condition_dimension)
        self.up_1 = ConditionalBlock(widths[1] + widths[0], widths[0], condition_dimension)
        self.output = nn.Sequential(
            nn.LayerNorm(widths[0]), nn.SiLU(), nn.Linear(widths[0], speaker_dimension)
        )

    def forward(self, noisy_speaker: Tensor, idmap_condition: Tensor, time: Tensor) -> Tensor:
        if noisy_speaker.ndim != 2 or idmap_condition.ndim != 2 or time.ndim != 1:
            raise ValueError("score-network inputs must be flattened batches")
        if len(noisy_speaker) != len(idmap_condition) or len(noisy_speaker) != len(time):
            raise ValueError("score-network batch sizes differ")
        condition = self.condition(torch.cat((idmap_condition, self.time(time)), dim=-1))
        first = self.down_1(self.input(noisy_speaker), condition)
        second = self.down_2(first, condition)
        third = self.down_3(second, condition)
        middle = self.middle(third, condition)
        up_second = self.up_2(torch.cat((middle, second), dim=-1), condition)
        up_first = self.up_1(torch.cat((up_second, first), dim=-1), condition)
        return self.output(up_first)


class IDMapDiff(nn.Module):
    """IDMap-Diff score model with the paper's linear VP noise schedule."""

    def __init__(
        self,
        *,
        dimension: int = 512,
        beta_min: float = 0.05,
        beta_max: float = 20.0,
        minimum_time: float = 1e-4,
    ) -> None:
        super().__init__()
        if not 0.0 < beta_min < beta_max:
            raise ValueError("expected 0 < beta_min < beta_max")
        if not 0.0 < minimum_time < 1.0:
            raise ValueError("minimum_time must be in (0, 1)")
        self.dimension = dimension
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.minimum_time = minimum_time
        self.pre_processor = PreProcessor(dimension)
        self.auxiliary_processor = AuxiliaryProcessor(dimension)
        self.score_network = ConditionalVectorUNet(
            speaker_dimension=dimension,
            idmap_condition_dimension=2 * dimension,
            widths=(dimension, dimension // 2, dimension // 4),
        )

    def condition(self, identity_vector: Tensor, auxiliary_vector: Tensor) -> Tensor:
        if identity_vector.shape != auxiliary_vector.shape:
            raise ValueError("identity and auxiliary vectors must have identical shapes")
        if identity_vector.ndim != 2 or identity_vector.shape[-1] != self.dimension:
            raise ValueError(f"expected (batch, {self.dimension}) conditioning vectors")
        return torch.cat(
            (
                self.pre_processor(identity_vector),
                self.auxiliary_processor(auxiliary_vector),
            ),
            dim=-1,
        )

    def integrated_beta(self, time: Tensor) -> Tensor:
        return self.beta_min * time + 0.5 * (self.beta_max - self.beta_min) * time.square()

    def gamma(self, time: Tensor) -> Tensor:
        return torch.exp(-0.5 * self.integrated_beta(time))

    def gamma_interval(self, start: Tensor, end: Tensor, *, power: float = 1.0) -> Tensor:
        beta_integral = (
            self.beta_min + 0.5 * (self.beta_max - self.beta_min) * (end + start)
        ) * (end - start)
        return torch.exp(-0.5 * power * beta_integral)

    def sde_ml_coefficients(
        self, time: Tensor, step_size: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        """DiffVC Eq. (10) coefficients for a reverse step from t to t-h."""

        if not 0.0 < step_size <= 1.0:
            raise ValueError("step_size must be in (0, 1]")
        start = time - step_size
        if bool((start < -1e-7).any()):
            raise ValueError("reverse SDE step crosses below time zero")
        beta = self.beta_min + (self.beta_max - self.beta_min) * time
        gamma_start = self.gamma(start)
        gamma_end = self.gamma(time)
        gamma_step_squared = self.gamma_interval(start, time, power=2.0)
        denominator = (gamma_end * beta * step_size).clamp_min(1e-12)
        kappa = gamma_start * (1.0 - gamma_step_squared) / denominator - 1.0
        variance_start = 1.0 - gamma_start.square()
        variance_end = (1.0 - gamma_end.square()).clamp_min(1e-12)
        mu = self.gamma_interval(start, time) * variance_start / variance_end
        nu = gamma_start * (1.0 - gamma_step_squared) / variance_end
        omega = (
            nu / gamma_end.clamp_min(1e-12)
            + mu
            - (0.5 * beta * step_size + 1.0)
        )
        sigma = torch.sqrt(
            (
                variance_start
                * (1.0 - gamma_step_squared)
                / variance_end
            ).clamp_min(0.0)
        )
        return kappa, omega, sigma

    def diffuse(
        self,
        target: Tensor,
        time: Tensor,
        noise: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if target.shape != noise.shape:
            raise ValueError("target and noise must have identical shapes")
        gamma = self.gamma(time)[:, None]
        variance = 1.0 - gamma.square()
        noisy = gamma * target + variance.sqrt() * noise
        score_target = -noise / variance.sqrt().clamp_min(1e-8)
        return noisy, score_target

    def training_loss(
        self,
        identity_vector: Tensor,
        auxiliary_vector: Tensor,
        target_speaker_vector: Tensor,
        *,
        time: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        condition = self.condition(identity_vector, auxiliary_vector)
        batch = len(target_speaker_vector)
        if target_speaker_vector.shape != (batch, self.dimension):
            raise ValueError(f"target must have shape (batch, {self.dimension})")
        if time is None:
            time = torch.rand(batch, device=target_speaker_vector.device)
            time = time * (1.0 - self.minimum_time) + self.minimum_time
        if time.shape != (batch,):
            raise ValueError("time must have shape (batch,)")
        if noise is None:
            noise = torch.randn_like(target_speaker_vector)
        noisy, score_target = self.diffuse(target_speaker_vector, time, noise)
        prediction = self.score_network(noisy, condition, time)
        variance = 1.0 - self.gamma(time).square()
        per_item_mse = F.mse_loss(prediction, score_target, reduction="none").mean(dim=-1)
        loss = (variance * per_item_mse).mean()
        return loss, {
            "unweighted_score_mse": per_item_mse.mean(),
            "mean_diffusion_time": time.mean(),
        }

    @torch.no_grad()
    def generate_from_indices(
        self,
        identity_indices: Iterable[int],
        fixed_auxiliary_vector: Tensor,
        sampler: IdentityVectorSampler,
        *,
        steps: int = 5,
        seed_offset: int = 0,
    ) -> Tensor:
        """Deterministic paper-style five-step maximum-likelihood SDE solver."""

        indices = tuple(int(index) for index in identity_indices)
        if steps < 1:
            raise ValueError("steps must be positive")
        if fixed_auxiliary_vector.ndim == 1:
            fixed_auxiliary_vector = fixed_auxiliary_vector.unsqueeze(0)
        if fixed_auxiliary_vector.shape != (1, self.dimension):
            raise ValueError(f"fixed auxiliary vector must have shape (1, {self.dimension})")
        identity = sampler.sample(
            indices,
            device=fixed_auxiliary_vector.device,
            dtype=fixed_auxiliary_vector.dtype,
        )
        auxiliary = fixed_auxiliary_vector.expand(len(indices), -1)
        condition = self.condition(identity, auxiliary)
        generators: list[torch.Generator] = []
        terminal: list[Tensor] = []
        for identity_index in indices:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(identity_index + seed_offset)
            generators.append(generator)
            terminal.append(torch.randn(self.dimension, generator=generator))
        values = torch.stack(terminal).to(
            device=fixed_auxiliary_vector.device,
            dtype=fixed_auxiliary_vector.dtype,
        )
        step_size = 1.0 / steps
        for step in range(steps):
            scalar_time = 1.0 - step * step_size
            time = torch.full(
                (len(indices),), scalar_time, device=values.device, dtype=values.dtype
            )
            beta = self.beta_min + (self.beta_max - self.beta_min) * time
            score = self.score_network(values, condition, time)
            kappa, omega, sigma = self.sde_ml_coefficients(time, step_size)
            stochastic = torch.stack(
                [
                    torch.randn(self.dimension, generator=generator)
                    for generator in generators
                ]
            ).to(device=values.device, dtype=values.dtype)
            increment = -values * (
                0.5 * beta[:, None] * step_size + omega[:, None]
            )
            increment = increment - score * (1.0 + kappa[:, None]) * (
                beta[:, None] * step_size
            )
            increment = increment + stochastic * sigma[:, None]
            values = values - increment
        return values


class IDMapDiffEDM(nn.Module):
    """Backend-native conditional EDM for low-dimensional speaker vectors.

    CAM++ coordinates are standardized before diffusion and restored after
    sampling.  EDM preconditioning makes the denoiser well-scaled across noise
    levels, while the deterministic Heun sampler avoids the unstable five-step
    stochastic reverse-SDE update used by the paper-faithful reproduction.
    """

    def __init__(
        self,
        *,
        dimension: int = 512,
        data_mean: Tensor | None = None,
        data_std: Tensor | None = None,
        sigma_data: float = 1.0,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        log_sigma_mean: float = -1.2,
        log_sigma_std: float = 1.2,
        geometry_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if dimension < 4:
            raise ValueError("dimension must be at least 4")
        if not 0.0 < sigma_min < sigma_max:
            raise ValueError("expected 0 < sigma_min < sigma_max")
        if sigma_data <= 0.0 or rho <= 0.0 or log_sigma_std <= 0.0:
            raise ValueError("EDM scale parameters must be positive")
        if geometry_weight < 0.0:
            raise ValueError("geometry_weight must be non-negative")
        self.dimension = dimension
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.log_sigma_mean = log_sigma_mean
        self.log_sigma_std = log_sigma_std
        self.geometry_weight = geometry_weight
        mean = torch.zeros(dimension) if data_mean is None else torch.as_tensor(data_mean)
        std = torch.ones(dimension) if data_std is None else torch.as_tensor(data_std)
        if mean.shape != (dimension,) or std.shape != (dimension,):
            raise ValueError(f"data statistics must have shape ({dimension},)")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("data statistics must be finite")
        if bool((std <= 0.0).any()):
            raise ValueError("data standard deviations must be positive")
        self.register_buffer("data_mean", mean.float().clone())
        self.register_buffer("data_std", std.float().clamp_min(1e-4).clone())
        self.pre_processor = PreProcessor(dimension)
        self.auxiliary_processor = AuxiliaryProcessor(dimension)
        self.denoise_network = ConditionalVectorUNet(
            speaker_dimension=dimension,
            idmap_condition_dimension=2 * dimension,
            widths=(dimension, dimension // 2, dimension // 4),
        )

    def normalize(self, values: Tensor) -> Tensor:
        return (values - self.data_mean.to(values)) / self.data_std.to(values)

    def denormalize(self, values: Tensor) -> Tensor:
        return values * self.data_std.to(values) + self.data_mean.to(values)

    def condition(self, identity_vector: Tensor, auxiliary_vector: Tensor) -> Tensor:
        if identity_vector.shape != auxiliary_vector.shape:
            raise ValueError("identity and auxiliary vectors must have identical shapes")
        if identity_vector.ndim != 2 or identity_vector.shape[-1] != self.dimension:
            raise ValueError(f"expected (batch, {self.dimension}) conditioning vectors")
        normalized_auxiliary = self.normalize(auxiliary_vector)
        return torch.cat(
            (
                self.pre_processor(identity_vector),
                self.auxiliary_processor(normalized_auxiliary),
            ),
            dim=-1,
        )

    def noise_level(self, sigma: Tensor) -> Tensor:
        log_min = math.log(self.sigma_min)
        log_range = math.log(self.sigma_max) - log_min
        return ((sigma.clamp_min(self.sigma_min).log() - log_min) / log_range).clamp(0.0, 1.0)

    def denoise(self, noisy: Tensor, condition: Tensor, sigma: Tensor) -> Tensor:
        if sigma.ndim != 1 or noisy.ndim != 2 or len(sigma) != len(noisy):
            raise ValueError("expected noisy=(batch, dimension) and sigma=(batch,)")
        sigma_column = sigma[:, None]
        sigma_data_squared = self.sigma_data**2
        denominator = sigma_column.square() + sigma_data_squared
        c_skip = sigma_data_squared / denominator
        c_out = sigma_column * self.sigma_data / denominator.sqrt()
        c_in = denominator.rsqrt()
        residual = self.denoise_network(
            c_in * noisy,
            condition,
            self.noise_level(sigma),
        )
        return c_skip * noisy + c_out * residual

    def training_loss(
        self,
        identity_vector: Tensor,
        auxiliary_vector: Tensor,
        target_speaker_vector: Tensor,
        *,
        sigma: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch = len(target_speaker_vector)
        if target_speaker_vector.shape != (batch, self.dimension):
            raise ValueError(f"target must have shape (batch, {self.dimension})")
        condition = self.condition(identity_vector, auxiliary_vector)
        target = self.normalize(target_speaker_vector)
        if sigma is None:
            sigma = (
                torch.randn(batch, device=target.device) * self.log_sigma_std
                + self.log_sigma_mean
            ).exp()
        if sigma.shape != (batch,):
            raise ValueError("sigma must have shape (batch,)")
        sigma = sigma.clamp(self.sigma_min, self.sigma_max)
        if noise is None:
            noise = torch.randn_like(target)
        if noise.shape != target.shape:
            raise ValueError("noise and target must have identical shapes")
        noisy = target + sigma[:, None] * noise
        denoised = self.denoise(noisy, condition, sigma)
        squared_error = (denoised - target).square().mean(dim=-1)
        weight = (sigma.square() + self.sigma_data**2) / (
            (sigma * self.sigma_data).square().clamp_min(1e-12)
        )
        loss = (weight * squared_error).mean()
        native_prediction = self.denormalize(denoised)
        cosine_distance = 1.0 - F.cosine_similarity(
            native_prediction, target_speaker_vector, dim=-1
        )
        relative_euclidean = torch.linalg.vector_norm(
            native_prediction - target_speaker_vector, dim=-1
        ) / torch.linalg.vector_norm(target_speaker_vector, dim=-1).clamp_min(1e-6)
        geometry_loss = (cosine_distance + relative_euclidean).mean()
        loss = loss + self.geometry_weight * geometry_loss
        return loss, {
            "unweighted_denoising_mse": squared_error.mean(),
            "mean_sigma": sigma.mean(),
            "native_geometry_loss": geometry_loss,
        }

    def sampling_schedule(self, steps: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if steps < 2:
            raise ValueError("EDM sampling requires at least two steps")
        ramp = torch.linspace(0.0, 1.0, steps, device=device, dtype=dtype)
        maximum = self.sigma_max ** (1.0 / self.rho)
        minimum = self.sigma_min ** (1.0 / self.rho)
        sigmas = (maximum + ramp * (minimum - maximum)).pow(self.rho)
        return torch.cat((sigmas, sigmas.new_zeros(1)))

    @torch.no_grad()
    def generate_from_indices(
        self,
        identity_indices: Iterable[int],
        fixed_auxiliary_vector: Tensor,
        sampler: IdentityVectorSampler,
        *,
        steps: int = 18,
        seed_offset: int = 0,
    ) -> Tensor:
        indices = tuple(int(index) for index in identity_indices)
        if fixed_auxiliary_vector.ndim == 1:
            fixed_auxiliary_vector = fixed_auxiliary_vector.unsqueeze(0)
        if fixed_auxiliary_vector.shape != (1, self.dimension):
            raise ValueError(f"fixed auxiliary vector must have shape (1, {self.dimension})")
        if not indices:
            return fixed_auxiliary_vector.new_empty((0, self.dimension))
        identity = sampler.sample(
            indices,
            device=fixed_auxiliary_vector.device,
            dtype=fixed_auxiliary_vector.dtype,
        )
        auxiliary = fixed_auxiliary_vector.expand(len(indices), -1)
        condition = self.condition(identity, auxiliary)
        terminal = []
        for identity_index in indices:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(identity_index + seed_offset)
            terminal.append(torch.randn(self.dimension, generator=generator))
        schedule = self.sampling_schedule(
            steps,
            device=fixed_auxiliary_vector.device,
            dtype=fixed_auxiliary_vector.dtype,
        )
        values = torch.stack(terminal).to(fixed_auxiliary_vector) * schedule[0]
        for index in range(len(schedule) - 1):
            sigma = schedule[index].expand(len(indices))
            sigma_next = schedule[index + 1]
            denoised = self.denoise(values, condition, sigma)
            derivative = (values - denoised) / sigma[:, None]
            candidate = values + (sigma_next - schedule[index]) * derivative
            if float(sigma_next) != 0.0:
                next_batch = sigma_next.expand(len(indices))
                next_denoised = self.denoise(candidate, condition, next_batch)
                next_derivative = (candidate - next_denoised) / next_batch[:, None]
                values = values + (sigma_next - schedule[index]) * (
                    0.5 * derivative + 0.5 * next_derivative
                )
            else:
                values = candidate
        return self.denormalize(values)
