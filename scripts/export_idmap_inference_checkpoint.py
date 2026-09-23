#!/usr/bin/env python3
"""Strip optimizer, RNG, speaker IDs, and local paths from a trusted run checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    # Only use with a checkpoint produced by a trusted training run. PyTorch
    # checkpoints are pickle files and loading an untrusted one is unsafe.
    original = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = dict(original["config"])
    for key in ("embeddings", "output_dir", "resume"):
        config.pop(key, None)
    required = {"model_type", "speaker_space", "embedding_dimension"}
    if not required.issubset(config):
        raise ValueError(f"checkpoint config is missing {sorted(required - set(config))}")
    state = original["model"]
    auxiliary = original["fixed_auxiliary_vector"]
    dimension = int(config["embedding_dimension"])
    if tuple(auxiliary.shape) != (dimension,):
        raise ValueError("fixed auxiliary vector does not match embedding dimension")
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("model contains non-finite weights")
    export = {
        "model": state,
        "config": config,
        "fixed_auxiliary_vector": auxiliary,
        "source_sha256": sha256(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".{os.getpid()}.tmp")
    try:
        torch.save(export, temporary)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{args.output} sha256={sha256(args.output)} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
