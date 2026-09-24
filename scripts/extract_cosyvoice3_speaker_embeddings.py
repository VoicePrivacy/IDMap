#!/usr/bin/env python3
"""Extract CosyVoice3-native 192-D CAM++ vectors with true ONNX batches.

Waveforms are deterministically center-cropped or repeated to one fixed window.
That makes every fbank tensor in a batch exactly the same shape and allows the
batch result to be compared directly with single-item ONNX inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import DataLoader, Dataset


EMBEDDING_DIMENSION = 192
SAMPLE_RATE = 16_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--campplus-onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-single-count", type=int, default=8)
    parser.add_argument("--batch-single-atol", type=float, default=1.0e-4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def fixed_window(waveform: torch.Tensor, samples: int) -> torch.Tensor:
    waveform = waveform.float().mean(dim=0).contiguous()
    if waveform.numel() < 1:
        raise ValueError("empty waveform")
    if waveform.numel() >= samples:
        start = (waveform.numel() - samples) // 2
        return waveform[start : start + samples]
    repeats = (samples + waveform.numel() - 1) // waveform.numel()
    return waveform.repeat(repeats)[:samples]


def make_fbank(path: Path, window_samples: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    waveform = fixed_window(waveform, window_samples)
    feature = kaldi.fbank(
        waveform.unsqueeze(0),
        num_mel_bins=80,
        dither=0,
        sample_frequency=SAMPLE_RATE,
    )
    return feature - feature.mean(dim=0, keepdim=True)


class AudioDataset(Dataset):
    def __init__(self, files: list[Path], audio_root: Path, window_samples: int) -> None:
        self.files = files
        self.audio_root = audio_root
        self.window_samples = window_samples

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[str, str, torch.Tensor]:
        path = self.files[index]
        relative = path.relative_to(self.audio_root)
        return path.stem, relative.parts[0], make_fbank(path, self.window_samples)


def main() -> None:
    args = parse_args()
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("rank must satisfy 0 <= rank < world-size")
    if args.batch_size < 1 or args.num_workers < 0 or args.window_seconds <= 0:
        raise ValueError("invalid batch, worker, or window setting")
    audio_root = args.audio_root.resolve()
    files = sorted(
        path for path in audio_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".flac", ".wav"}
    )
    if args.limit is not None:
        files = files[: args.limit]
    files = [path for position, path in enumerate(files) if position % args.world_size == args.rank]
    if not files:
        raise ValueError("rank received no audio files")

    window_samples = round(args.window_seconds * SAMPLE_RATE)
    dataset = AudioDataset(files, audio_root, window_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = max(1, min(8, os.cpu_count() or 1))
    session = onnxruntime.InferenceSession(
        str(args.campplus_onnx.resolve()),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    utterance_ids: list[str] = []
    speaker_ids: list[str] = []
    embedding_batches: list[np.ndarray] = []
    validated = 0
    for batch_index, (batch_utterances, batch_speakers, features) in enumerate(loader):
        feature_array = features.numpy().astype(np.float32, copy=False)
        embeddings = np.asarray(
            session.run(None, {input_name: feature_array})[0], dtype=np.float32
        )
        if embeddings.shape != (len(batch_utterances), EMBEDDING_DIMENSION):
            raise ValueError(f"unexpected CAM++ batch shape: {embeddings.shape}")
        if not np.isfinite(embeddings).all():
            raise ValueError("CAM++ produced non-finite embeddings")
        while validated < args.validate_single_count and validated < len(batch_utterances):
            single = np.asarray(
                session.run(None, {input_name: feature_array[validated : validated + 1]})[0],
                dtype=np.float32,
            )[0]
            difference = float(np.max(np.abs(single - embeddings[validated])))
            if difference > args.batch_single_atol:
                raise ValueError(
                    f"batch/single CAM++ mismatch: max_abs={difference:.8g}"
                )
            validated += 1
        utterance_ids.extend(batch_utterances)
        speaker_ids.extend(batch_speakers)
        embedding_batches.append(embeddings)
        if batch_index % 25 == 0:
            print(json.dumps({"rank": args.rank, "completed": len(utterance_ids), "total": len(files)}), flush=True)

    checkpoint_hash = sha256_file(args.campplus_onnx.resolve())
    speaker_space = f"cosyvoice3-campplus-v1:{checkpoint_hash[:16]}"
    atomic_npz(
        args.output.resolve(),
        embeddings=np.concatenate(embedding_batches),
        utterance_ids=np.asarray(utterance_ids),
        speaker_ids=np.asarray(speaker_ids),
        speaker_space=np.asarray(speaker_space),
        checkpoint_sha256=np.asarray(checkpoint_hash),
    )
    manifest = {
        "backend": "Fun-CosyVoice3-0.5B-2512",
        "speaker_encoder": "CosyVoice3 campplus.onnx",
        "speaker_space": speaker_space,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "rank": args.rank,
        "world_size": args.world_size,
        "utterances": len(utterance_ids),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "sample_rate": SAMPLE_RATE,
        "window_seconds": args.window_seconds,
        "crop_policy": "deterministic_center_crop_or_repeat",
        "single_batch_validation_count": validated,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
