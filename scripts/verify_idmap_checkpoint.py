#!/usr/bin/env python3
"""Check one trusted backend-native IDMap export before vendor synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from voice_anon.idmap.diffusion import IDMapDiff, IDMapDiffEDM
from voice_anon.idmap.mlp import IDMapMLP, IdentityVectorSampler


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-dimension", type=int, required=True)
    parser.add_argument("--expected-speaker-space", required=True)
    args = parser.parse_args()

    actual_sha = sha256(args.checkpoint)
    if actual_sha != args.expected_sha256.lower():
        raise ValueError(f"checkpoint SHA-256 mismatch: {actual_sha}")
    # PyTorch checkpoint loading uses pickle. Only run on our verified exports
    # or files from a trusted source, after the hash check above.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    dimension = int(config["embedding_dimension"])
    if dimension != args.expected_dimension:
        raise ValueError(f"dimension mismatch: {dimension}")
    if str(config["speaker_space"]) != args.expected_speaker_space:
        raise ValueError("speaker-space label mismatch")
    model_type = str(config["model_type"])
    if model_type == "idmap_mlp":
        model = IDMapMLP(dimension=dimension)
    elif model_type == "idmap_diff":
        model = IDMapDiff(dimension=dimension)
    elif model_type == "idmap_diff_edm":
        model = IDMapDiffEDM(dimension=dimension)
    else:
        raise ValueError(f"unsupported model_type: {model_type}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    auxiliary = torch.as_tensor(checkpoint["fixed_auxiliary_vector"])
    if tuple(auxiliary.shape) != (dimension,) or not torch.isfinite(auxiliary).all():
        raise ValueError("invalid fixed auxiliary vector")
    sampler = IdentityVectorSampler(
        dimension=dimension, distribution=str(config.get("distribution", "normal"))
    )
    with torch.inference_mode():
        if model_type == "idmap_mlp":
            vectors = model.generate_from_indices([1001, 1002], auxiliary, sampler)
        else:
            vectors = model.generate_from_indices(
                [1001, 1002], auxiliary, sampler,
                steps=int(config.get("reverse_steps", 5 if model_type == "idmap_diff" else 18)),
            )
    if tuple(vectors.shape) != (2, dimension) or not torch.isfinite(vectors).all():
        raise ValueError("generated vectors have invalid shape or non-finite values")
    print(json.dumps({
        "sha256": actual_sha,
        "model_type": model_type,
        "speaker_space": config["speaker_space"],
        "dimension": dimension,
        "vector_norms": [float(v) for v in vectors.norm(dim=-1)],
        "result": "structural_check_passed_not_audio_quality",
    }, indent=2))


if __name__ == "__main__":
    main()
