#!/usr/bin/env python3
"""Train IDMap-MLP in a synthesis backend's native speaker-vector space."""

from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from voice_anon.idmap.data import SpeakerEmbeddingCorpus
from voice_anon.idmap.mlp import IdentityVectorSampler, IDMapMLP, IDMapMLPLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--speaker-space",
        required=True,
        help="Immutable backend-space label, for example dots-campplus-v1",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--speakers-per-batch", type=int, default=16)
    parser.add_argument("--aux-per-speaker", type=int, default=16)
    parser.add_argument(
        "--distribution", choices=("normal", "uniform"), default="normal"
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume model, optimizer, epoch, and RNG state from a checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.steps_per_epoch < 1:
        raise ValueError("epochs and steps-per-epoch must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.Generator(np.random.PCG64(args.seed))
    device = torch.device(args.device)

    corpus = SpeakerEmbeddingCorpus.load(
        args.embeddings, expected_speaker_space=args.speaker_space
    )
    embedding_dimension = int(corpus.embeddings.shape[1])
    sampler = IdentityVectorSampler(
        dimension=embedding_dimension,
        distribution=args.distribution,
    )
    all_identity_vectors = sampler.sample(range(len(corpus.speakers)), device=device)
    model = IDMapMLP(dimension=embedding_dimension).to(device)
    criterion = IDMapMLPLoss(alpha=args.alpha)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = vars(args).copy()
    configuration.update(
        model_type="idmap_mlp",
        num_utterances=len(corpus.embeddings),
        num_speakers=len(corpus.speakers),
        embedding_dimension=embedding_dimension,
        triplets_per_batch=args.speakers_per_batch * args.aux_per_speaker,
    )
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in configuration.items()
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )

    metrics_path = args.output_dir / "metrics.jsonl"
    best_loss = float("inf")
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        checkpoint_config = checkpoint.get("config", {})
        if checkpoint_config.get("speaker_space") != args.speaker_space:
            raise ValueError("resume checkpoint speaker-space mismatch")
        if checkpoint_config.get("distribution") != args.distribution:
            raise ValueError("resume checkpoint distribution mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        completed_epoch = int(checkpoint["epoch"])
        start_epoch = completed_epoch + 1
        best_loss = float(checkpoint.get("best_loss", checkpoint["loss"]))
        existing_best_path = args.output_dir / "best.pt"
        if existing_best_path.is_file() and existing_best_path != args.resume:
            existing_best = torch.load(
                existing_best_path, map_location="cpu", weights_only=False
            )
            best_loss = min(best_loss, float(existing_best["loss"]))
        if "numpy_rng_state" in checkpoint:
            rng.bit_generator.state = checkpoint["numpy_rng_state"]
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu(), device)
        print(
            json.dumps(
                {"resumed_from": str(args.resume), "completed_epoch": completed_epoch}
            ),
            flush=True,
        )
    for epoch in range(1, args.epochs + 1):
        if epoch < start_epoch:
            continue
        model.train()
        epoch_started = time.perf_counter()
        running_loss = 0.0
        for _ in range(args.steps_per_epoch):
            identity_indices, target_numpy, auxiliary_numpy = corpus.sample_paper_batch(
                rng,
                speakers_per_batch=args.speakers_per_batch,
                auxiliary_utterances_per_speaker=args.aux_per_speaker,
            )
            identity = all_identity_vectors[
                torch.as_tensor(identity_indices, device=device)
            ]
            target = torch.as_tensor(target_numpy, device=device)
            auxiliary = torch.as_tensor(auxiliary_numpy, device=device)

            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                prediction = model(identity, auxiliary)
                loss = criterion(prediction.float(), target.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.detach())

        epoch_loss = running_loss / args.steps_per_epoch
        record = {
            "epoch": epoch,
            "loss": epoch_loss,
            "seconds": time.perf_counter() - epoch_started,
        }
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": epoch_loss,
            "config": configuration,
            "speakers": corpus.speakers,
            "fixed_auxiliary_vector": torch.from_numpy(corpus.embeddings[0]),
            "best_loss": min(best_loss, epoch_loss),
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device)
            if device.type == "cuda"
            else None,
        }
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, args.output_dir / "best.pt")
        torch.save(checkpoint, args.output_dir / "last.pt")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(checkpoint, args.output_dir / f"epoch-{epoch:04d}.pt")


if __name__ == "__main__":
    main()
