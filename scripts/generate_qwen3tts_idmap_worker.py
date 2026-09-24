#!/usr/bin/env python3
"""True-batch Qwen3-TTS Base synthesis with model-native IDMap vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
from voice_anon.generation import load_generation_manifest
from voice_anon.idmap.diffusion import IDMapDiff, IDMapDiffEDM
from voice_anon.idmap.mlp import IDMapMLP, IdentityVectorSampler


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
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--idmap-checkpoint", type=Path, required=True)
    parser.add_argument("--speaker-adapter-checkpoint", type=Path)
    parser.add_argument(
        "--expected-speaker-space-prefix",
        required=True,
        help="Reject an IDMap trained in another Qwen3-TTS speaker space",
    )
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batching", choices=("length", "manifest"), default="length")
    parser.add_argument(
        "--attention", choices=("flash_attention_2", "sdpa"), default="flash_attention_2"
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--decode-mode",
        choices=("greedy", "official_sampling"),
        default="greedy",
    )
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--tokens-per-word", type=int, default=12)
    parser.add_argument("--token-margin", type=int, default=32)
    parser.add_argument("--minimum-new-tokens", type=int, default=64)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def valid_wave(path: Path) -> bool:
    try:
        info = sf.info(path)
        return info.samplerate == 16_000 and info.channels == 1 and info.frames >= 1600
    except Exception:
        return False


def atomic_wave(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    value = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
    if sample_rate != 16_000:
        value = torchaudio.functional.resample(value, sample_rate, 16_000)
    value = value.clamp(-1.0, 1.0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp.wav")
    sf.write(temporary, value, 16_000, subtype="PCM_16")
    os.replace(temporary, path)


def make_prompt(embedding: torch.Tensor) -> VoiceClonePromptItem:
    return VoiceClonePromptItem(
        ref_code=None,
        ref_spk_embedding=embedding,
        x_vector_only_mode=True,
        icl_mode=False,
        ref_text=None,
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("invalid rank/world-size")
    rows = load_generation_manifest(
        args.manifest, rank=args.rank, world_size=args.world_size
    )
    if args.max_records is not None:
        rows = rows[: args.max_records]
    rows = [
        row for row in rows
        if args.overwrite or not valid_wave(args.output_dir / row["output_relative_path"])
    ]
    if args.batching == "length":
        # Autoregressive batch time is set by the longest item.  Length sorting
        # avoids decoding short utterances for the duration of an unrelated
        # long one while preserving each record's identity and output path.
        rows.sort(key=lambda row: (len(str(row["text"]).split()), len(str(row["text"]))))
    print(json.dumps({"stage": "manifest_ready", "rank": args.rank, "pending": len(rows)}), flush=True)
    if not rows:
        return

    started = time.perf_counter()
    generated = 0

    model = Qwen3TTSModel.from_pretrained(
        str(args.model_dir.resolve()),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation=args.attention,
    )
    model.model.eval()
    checkpoint = torch.load(args.idmap_checkpoint, map_location="cuda:0", weights_only=False)
    config = checkpoint["config"]
    embedding_dimension = int(config["embedding_dimension"])
    if embedding_dimension != 1024:
        raise ValueError("Qwen3-TTS Base requires a native 1024-D IDMap checkpoint")
    if not str(config["speaker_space"]).startswith(
        args.expected_speaker_space_prefix + ":"
    ):
        raise ValueError(
            "IDMap speaker-space label is incompatible with the requested "
            f"Qwen3-TTS model: {config['speaker_space']!r}"
        )
    recorded_fingerprint = str(config["speaker_space"]).rsplit(":", 1)[-1]
    actual_fingerprint = model_fingerprint(args.model_dir.resolve())
    if not actual_fingerprint.startswith(recorded_fingerprint):
        raise ValueError(
            "IDMap was trained for a different Qwen3-TTS checkpoint: "
            f"recorded={recorded_fingerprint!r}, actual={actual_fingerprint[:16]!r}"
        )
    model_type = str(config.get("model_type", "idmap_mlp"))
    if model_type == "idmap_mlp":
        idmap = IDMapMLP(dimension=embedding_dimension)
    elif model_type == "idmap_diff":
        idmap = IDMapDiff(dimension=embedding_dimension)
    elif model_type == "idmap_diff_edm":
        idmap = IDMapDiffEDM(dimension=embedding_dimension)
    else:
        raise ValueError(f"unsupported IDMap checkpoint type: {model_type}")
    idmap = idmap.to("cuda:0").eval()
    idmap.load_state_dict(checkpoint["model"])
    sampler = IdentityVectorSampler(
        dimension=embedding_dimension, distribution=config.get("distribution", "normal")
    )
    auxiliary = checkpoint["fixed_auxiliary_vector"].to("cuda:0", dtype=torch.float32)
    speaker_adapter = None
    if args.speaker_adapter_checkpoint:
        from voice_anon.backends.qwen3tts import SpeakerConditioningAdapter
        adapted = torch.load(args.speaker_adapter_checkpoint, map_location="cpu", weights_only=False)
        if not adapted["config"].get("speaker_conditioning_only"):
            raise ValueError("not a speaker-conditioning adapter")
        speaker_adapter = SpeakerConditioningAdapter(embedding_dimension,
            adapted["config"].get("speaker_adapter_bottleneck", 256)).cuda().float().eval()
        speaker_adapter.load_state_dict(adapted["speaker_adapter"], strict=True)
        speaker_adapter.requires_grad_(False)
        print(json.dumps({"speaker_adapter_sha256": sha256_file(args.speaker_adapter_checkpoint),
                          "speaker_adapter_step": adapted["step"]}), flush=True)

    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        indices = [int(row["anonymous_index"]) for row in batch]
        with torch.inference_mode():
            if model_type == "idmap_mlp":
                embeddings = idmap.generate_from_indices(indices, auxiliary, sampler)
            else:
                embeddings = idmap.generate_from_indices(
                    indices, auxiliary, sampler,
                    steps=int(config.get("reverse_steps", 5 if model_type == "idmap_diff" else 18)),
                )
            if speaker_adapter is not None:
                embeddings = speaker_adapter(embeddings)
        prompts = [make_prompt(vector) for vector in embeddings]
        maximum_words = max(len(str(row["text"]).split()) for row in batch)
        batch_maximum_new_tokens = min(
            args.max_new_tokens,
            max(
                args.minimum_new_tokens,
                args.tokens_per_word * maximum_words + args.token_margin,
            ),
        )
        if args.decode_mode == "official_sampling":
            # Reproducible for this immutable manifest/batch recipe. Sampling
            # streams are batch-composition dependent, so quality equivalence
            # is checked with ASR rather than claiming PCM identity.
            batch_seed = args.seed + args.rank * 1_000_003 + offset
            torch.manual_seed(batch_seed)
            torch.cuda.manual_seed_all(batch_seed)
            decode = dict(
                do_sample=True,
                top_k=50,
                top_p=1.0,
                temperature=0.9,
                repetition_penalty=1.05,
                subtalker_dosample=True,
                subtalker_top_k=50,
                subtalker_top_p=1.0,
                subtalker_temperature=0.9,
            )
        else:
            decode = dict(do_sample=False, subtalker_dosample=False)
        waveforms, sample_rate = model.generate_voice_clone(
            text=[str(row["text"]).lower() for row in batch],
            language=["English"] * len(batch),
            voice_clone_prompt=prompts,
            max_new_tokens=batch_maximum_new_tokens,
            **decode,
        )
        if len(waveforms) != len(batch):
            raise RuntimeError("Qwen3-TTS returned the wrong batch size")
        for row, waveform in zip(batch, waveforms, strict=True):
            atomic_wave(args.output_dir / row["output_relative_path"], waveform, sample_rate)
        generated += len(batch)
        print(json.dumps({
            "rank": args.rank,
            "done": min(offset + len(batch), len(rows)),
            "total": len(rows),
            "maximum_words": maximum_words,
            "maximum_new_tokens": batch_maximum_new_tokens,
        }), flush=True)
    seconds = time.perf_counter() - started
    print(json.dumps({
        "stage": "complete",
        "rank": args.rank,
        "generated": generated,
        "seconds": seconds,
        "utterances_per_second": generated / seconds,
        "batch_size": args.batch_size,
        "decode_mode": args.decode_mode,
    }), flush=True)


if __name__ == "__main__":
    main()
