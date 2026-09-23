"""Clean-room implementation of the vector-space-agnostic IDMap-MLP.

Reference: Liu et al., "IDMap: A Pseudo-Speaker Generator Framework Based
on Speaker Identity Index to Vector Mapping", Eq. (3) and Sec. IV/VI-D.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


Distribution = Literal["normal", "uniform"]


@dataclass(frozen=True, slots=True)
class IdentityVectorSampler:
    """Map integer identity indices to deterministic PCG64 vectors.

    The paper uses each identity index as the PCG64 seed, samples either
    N(0, 1) or U(-1, 1), and applies mean-variance normalization (MVN).
    """

    dimension: int = 512
    distribution: Distribution = "normal"
    apply_mvn: bool = True
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("dimension must be positive")
        if self.distribution not in ("normal", "uniform"):
            raise ValueError(f"unsupported distribution: {self.distribution}")

    def sample_numpy(self, identity_index: int) -> np.ndarray:
        if identity_index < 0:
            raise ValueError("identity_index must be non-negative")
        generator = np.random.Generator(np.random.PCG64(identity_index))
        if self.distribution == "normal":
            vector = generator.normal(0.0, 1.0, self.dimension)
        else:
            vector = generator.uniform(-1.0, 1.0, self.dimension)
        vector = vector.astype(np.float32)
        if self.apply_mvn:
            vector = (vector - vector.mean()) / max(float(vector.std()), self.epsilon)
        return vector

    def sample(
        self,
        identity_indices: int | Iterable[int],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        if isinstance(identity_indices, int):
            indices: Sequence[int] = (identity_indices,)
        else:
            indices = tuple(int(index) for index in identity_indices)
        if not indices:
            return torch.empty((0, self.dimension), device=device, dtype=dtype)
        vectors = np.stack([self.sample_numpy(index) for index in indices])
        return torch.as_tensor(vectors, device=device, dtype=dtype)


class PreProcessor(nn.Module):
    """Paper topology scaled to the active backend's vector dimension."""

    def __init__(self, dimension: int = 512) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.ReLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, identity_vector: Tensor) -> Tensor:
        return self.layers(identity_vector)


class AuxiliaryProcessor(nn.Module):
    """Three FC-ReLU-BN blocks followed by a native-dimensional output."""

    def __init__(self, dimension: int = 512, blocks: int = 3) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("blocks must be positive")
        layers: list[nn.Module] = []
        for _ in range(blocks):
            layers.extend(
                (
                    nn.Linear(dimension, dimension),
                    nn.ReLU(),
                    nn.BatchNorm1d(dimension),
                )
            )
        layers.append(nn.Linear(dimension, dimension))
        self.layers = nn.Sequential(*layers)

    def forward(self, auxiliary_speaker_vector: Tensor) -> Tensor:
        return self.layers(auxiliary_speaker_vector)


class MLPGenerator(nn.Module):
    """Paper 2D-D-D generator for a backend-native dimension D."""

    def __init__(self, dimension: int = 512) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2 * dimension, dimension),
            nn.ReLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, joint_representation: Tensor) -> Tensor:
        return self.layers(joint_representation)


class IDMapMLP(nn.Module):
    """Generate a pseudo-speaker vector from an IDV and an auxiliary vector."""

    def __init__(self, dimension: int = 512) -> None:
        super().__init__()
        self.dimension = dimension
        self.pre_processor = PreProcessor(dimension)
        self.auxiliary_processor = AuxiliaryProcessor(dimension)
        self.generator = MLPGenerator(dimension)

    def forward(self, identity_vector: Tensor, auxiliary_speaker_vector: Tensor) -> Tensor:
        self._validate_inputs(identity_vector, auxiliary_speaker_vector)
        identity_representation = self.pre_processor(identity_vector)
        auxiliary_representation = self.auxiliary_processor(auxiliary_speaker_vector)
        joint = torch.cat((identity_representation, auxiliary_representation), dim=-1)
        return self.generator(joint)

    def generate_from_indices(
        self,
        identity_indices: Iterable[int],
        fixed_auxiliary_vector: Tensor,
        sampler: IdentityVectorSampler,
    ) -> Tensor:
        """Paper inference path using the same fixed auxiliary vector for all IDs."""

        indices = tuple(identity_indices)
        if fixed_auxiliary_vector.ndim == 1:
            fixed_auxiliary_vector = fixed_auxiliary_vector.unsqueeze(0)
        if fixed_auxiliary_vector.shape != (1, self.dimension):
            raise ValueError(
                f"fixed auxiliary vector must have shape ({self.dimension},) or "
                f"(1, {self.dimension}); "
                f"got {tuple(fixed_auxiliary_vector.shape)}"
            )
        identity_vectors = sampler.sample(
            indices,
            device=fixed_auxiliary_vector.device,
            dtype=fixed_auxiliary_vector.dtype,
        )
        auxiliary_vectors = fixed_auxiliary_vector.expand(len(indices), -1)
        return self(identity_vectors, auxiliary_vectors)

    def _validate_inputs(self, identity_vector: Tensor, auxiliary_speaker_vector: Tensor) -> None:
        if identity_vector.ndim != 2 or auxiliary_speaker_vector.ndim != 2:
            raise ValueError("IDMap-MLP inputs must have shape (batch, dimension)")
        if identity_vector.shape != auxiliary_speaker_vector.shape:
            raise ValueError("identity and auxiliary vectors must have identical shapes")
        if identity_vector.shape[-1] != self.dimension:
            raise ValueError(f"expected {self.dimension}-dimensional vectors")


class IDMapMLPLoss(nn.Module):
    """Equation (3): weighted cosine distance plus Euclidean distance."""

    def __init__(self, alpha: float = 0.5, epsilon: float = 1e-8) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have identical shapes")
        cosine_distance = 1.0 - F.cosine_similarity(
            prediction, target, dim=-1, eps=self.epsilon
        )
        euclidean_distance = torch.linalg.vector_norm(prediction - target, dim=-1)
        return (
            self.alpha * cosine_distance
            + (1.0 - self.alpha) * euclidean_distance
        ).mean()
