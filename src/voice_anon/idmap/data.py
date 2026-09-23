"""Speaker-embedding data utilities for backend-native IDMap training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class SpeakerEmbeddingCorpus:
    embeddings: np.ndarray
    speaker_ids: np.ndarray
    speakers: tuple[str, ...]
    speaker_means: np.ndarray
    utterance_indices_by_speaker: tuple[np.ndarray, ...]

    @classmethod
    def load(
        cls,
        path: str | Path,
        expected_dimension: int | None = None,
    ) -> "SpeakerEmbeddingCorpus":
        with np.load(path, allow_pickle=False) as archive:
            embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
            speaker_ids = np.asarray(archive["speaker_ids"]).astype(str)
        if embeddings.ndim != 2 or embeddings.shape[1] < 1:
            raise ValueError(
                "embeddings must have shape (utterances, positive_dimension); "
                f"got {embeddings.shape}"
            )
        if expected_dimension is not None and embeddings.shape[1] != expected_dimension:
            raise ValueError(
                f"expected {expected_dimension}-dimensional embeddings; "
                f"got {embeddings.shape[1]}"
            )
        if len(embeddings) != len(speaker_ids):
            raise ValueError("embeddings and speaker_ids must have the same length")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings contain non-finite values")

        speakers = tuple(sorted(set(speaker_ids.tolist())))
        utterance_indices = tuple(
            np.flatnonzero(speaker_ids == speaker) for speaker in speakers
        )
        speaker_means = np.stack(
            [embeddings[indices].mean(axis=0) for indices in utterance_indices]
        ).astype(np.float32)
        return cls(
            embeddings=embeddings,
            speaker_ids=speaker_ids,
            speakers=speakers,
            speaker_means=speaker_means,
            utterance_indices_by_speaker=utterance_indices,
        )

    def sample_paper_batch(
        self,
        rng: np.random.Generator,
        *,
        speakers_per_batch: int = 16,
        auxiliary_utterances_per_speaker: int = 16,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return identity indices, targets, and distinct-speaker auxiliaries.

        With the paper defaults this creates 16 x 16 = 256 triplets.
        """

        if len(self.speakers) < speakers_per_batch:
            raise ValueError("corpus has fewer speakers than speakers_per_batch")
        target_indices = rng.choice(
            len(self.speakers), size=speakers_per_batch, replace=False
        )
        all_utterance_indices = np.arange(len(self.embeddings))
        batch_identity_indices: list[int] = []
        batch_targets: list[np.ndarray] = []
        batch_auxiliaries: list[np.ndarray] = []

        for target_index in target_indices:
            target_speaker = self.speakers[int(target_index)]
            candidates = all_utterance_indices[self.speaker_ids != target_speaker]
            selected = rng.choice(
                candidates,
                size=auxiliary_utterances_per_speaker,
                replace=len(candidates) < auxiliary_utterances_per_speaker,
            )
            for utterance_index in selected:
                batch_identity_indices.append(int(target_index))
                batch_targets.append(self.speaker_means[int(target_index)])
                batch_auxiliaries.append(self.embeddings[int(utterance_index)])

        return (
            np.asarray(batch_identity_indices, dtype=np.int64),
            np.stack(batch_targets).astype(np.float32),
            np.stack(batch_auxiliaries).astype(np.float32),
        )
