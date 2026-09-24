#!/usr/bin/env python3
"""Extract batched Qwen3-TTS-native speaker embeddings.

The extractor uses the Base model's own ECAPA-style speaker encoder.  Audio is
resampled to 24 kHz and deterministically center-cropped or repeated to a fixed
window so batch and single-item inference have identical tensor shapes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram


SAMPLE_RATE = 24_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--speaker-space-name",
        required=True,
        help=(
            "Versioned native speaker-space name, for example "
            "qwen3tts-12hz-1p7b-base-xvector-v1"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-single-count", type=int, default=8)
    parser.add_argument("--batch-single-atol", type=float, default=5.0e-2)
    parser.add_argument("--batch-single-min-cosine", type=float, default=0.999)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(model_dir: Path) -> str:
    candidates = sorted(model_dir.glob("*.safetensors"))
    if not candidates:
        candidates = [model_dir / "config.json"]
    digest = hashlib.sha256()
    for path in candidates:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def fixed_window(path: Path, samples: int) -> torch.Tensor:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.mean(axis=1))
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    if waveform.numel() < 1:
        raise ValueError(f"empty waveform: {path}")
    if waveform.numel() >= samples:
        start = (waveform.numel() - samples) // 2
        return waveform[start : start + samples].contiguous()
    repeats = (samples + waveform.numel() - 1) // waveform.numel()
    return waveform.repeat(repeats)[:samples].contiguous()


class AudioDataset(Dataset):
    def __init__(self, files: list[Path], root: Path, samples: int) -> None:
        self.files = files
        self.root = root
        self.samples = samples

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[str, str, torch.Tensor]:
        path = self.files[index]
        relative = path.relative_to(self.root)
        return path.stem, relative.parts[0], fixed_window(path, self.samples)


@torch.inference_mode()
def encode(model: object, waveforms: torch.Tensor) -> torch.Tensor:
    mels = mel_spectrogram(
        waveforms,
        n_fft=1024,
        num_mels=128,
        sampling_rate=SAMPLE_RATE,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12_000,
    ).transpose(1, 2)
    return model.model.speaker_encoder(
        mels.to(model.model.device, dtype=model.model.dtype)
    ).float()


def main() -> None:
    args = parse_args()
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("rank must satisfy 0 <= rank < world-size")
    files = sorted(
        path
        for path in args.audio_root.resolve().rglob("*")
        if path.is_file() and path.suffix.lower() in {".wav", ".flac"}
    )
    if args.limit is not None:
        files = files[: args.limit]
    files = [path for index, path in enumerate(files) if index % args.world_size == args.rank]
    if not files:
        raise ValueError("rank received no audio files")

    dataset = AudioDataset(
        files, args.audio_root.resolve(), round(args.window_seconds * SAMPLE_RATE)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    model = Qwen3TTSModel.from_pretrained(
        str(args.model_dir.resolve()),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.model.eval()

    utterance_ids: list[str] = []
    speaker_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    embedding_dimension: int | None = None
    validated = 0
    maximum_difference = 0.0
    minimum_cosine = 1.0
    for batch_index, (utterances, speakers, waveforms) in enumerate(loader):
        batch_embeddings = encode(model, waveforms)
        if batch_embeddings.ndim != 2 or batch_embeddings.shape[0] != len(utterances):
            raise ValueError(f"unexpected embedding shape: {tuple(batch_embeddings.shape)}")
        if embedding_dimension is None:
            embedding_dimension = int(batch_embeddings.shape[1])
        elif batch_embeddings.shape[1] != embedding_dimension:
            raise ValueError(
                "Qwen3-TTS speaker embedding dimension changed between batches"
            )
        while validated < args.validate_single_count and validated < len(utterances):
            single = encode(model, waveforms[validated : validated + 1])[0]
            difference = float((single - batch_embeddings[validated]).abs().max())
            cosine = float(torch.nn.functional.cosine_similarity(
                single.unsqueeze(0), batch_embeddings[validated].unsqueeze(0)
            ))
            maximum_difference = max(maximum_difference, difference)
            minimum_cosine = min(minimum_cosine, cosine)
            if difference > args.batch_single_atol or cosine < args.batch_single_min_cosine:
                raise ValueError(
                    "batch/single embedding mismatch: "
                    f"max_abs={difference:.8g}, cosine={cosine:.8g}"
                )
            validated += 1
        array = batch_embeddings.cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(array).all():
            raise ValueError("non-finite Qwen3-TTS speaker embedding")
        utterance_ids.extend(utterances)
        speaker_ids.extend(speakers)
        embeddings.append(array)
        if batch_index % 25 == 0:
            print(json.dumps({"rank": args.rank, "done": len(utterance_ids), "total": len(files)}), flush=True)

    fingerprint = model_fingerprint(args.model_dir.resolve())
    speaker_space = f"{args.speaker_space_name}:{fingerprint[:16]}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            embeddings=np.concatenate(embeddings),
            utterance_ids=np.asarray(utterance_ids),
            speaker_ids=np.asarray(speaker_ids),
            speaker_space=np.asarray(speaker_space),
            checkpoint_sha256=np.asarray(fingerprint),
        )
    os.replace(temporary, args.output)
    report = {
        "audio_root": str(args.audio_root.resolve()),
        "ordered_utterance_ids_sha256": hashlib.sha256(
            "\n".join(utterance_ids).encode("utf-8")
        ).hexdigest(),
        "backend": args.model_dir.resolve().name,
        "speaker_space": speaker_space,
        "embedding_dimension": embedding_dimension,
        "utterances": len(utterance_ids),
        "rank": args.rank,
        "world_size": args.world_size,
        "batch_size": args.batch_size,
        "window_seconds": args.window_seconds,
        "crop_policy": "deterministic_center_crop_or_repeat",
        "validated_batch_single": validated,
        "batch_single_max_abs": maximum_difference,
        "batch_single_min_cosine": minimum_cosine,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
