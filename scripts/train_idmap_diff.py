#!/usr/bin/env python3
"""Train IDMap-Diff in a synthesis backend's native speaker-vector space."""

from __future__ import annotations

import argparse
import copy
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from voice_anon.idmap.data import SpeakerEmbeddingCorpus
from voice_anon.idmap.diffusion import IDMapDiff, IDMapDiffEDM
from voice_anon.idmap.mlp import IdentityVectorSampler


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
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--variant",
        choices=("paper_vp_sde", "edm"),
        default="paper_vp_sde",
        help="Keep the paper reproduction or use backend-normalized EDM-v2",
    )
    parser.add_argument("--reverse-steps", type=int)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--geometry-weight", type=float, default=0.0)
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
    if args.epochs < 1 or args.steps_per_epoch < 1 or args.save_every < 1:
        raise ValueError("epoch, step, and checkpoint intervals must be positive")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema-decay must be in [0, 1)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    corpus = SpeakerEmbeddingCorpus.load(args.embeddings)
    embedding_dimension = int(corpus.embeddings.shape[1])
    sampler = IdentityVectorSampler(
        dimension=embedding_dimension,
        distribution=args.distribution,
    )
    identity_bank = sampler.sample(range(len(corpus.speakers)), device=device)
    if args.variant == "edm":
        data_mean = torch.from_numpy(corpus.speaker_means.mean(axis=0))
        data_std = torch.from_numpy(corpus.speaker_means.std(axis=0)).clamp_min(1e-4)
        model: IDMapDiff | IDMapDiffEDM = IDMapDiffEDM(
            dimension=embedding_dimension,
            data_mean=data_mean,
            data_std=data_std,
            geometry_weight=args.geometry_weight,
        ).to(device)
        model_type = "idmap_diff_edm"
        reverse_steps = args.reverse_steps or 18
        reverse_solver = "edm_heun"
        reverse_solver_reference = "Karras_EDM_Algorithm_2"
        terminal_distribution = "scaled_standard_gaussian"
    else:
        model = IDMapDiff(dimension=embedding_dimension).to(device)
        model_type = "idmap_diff"
        reverse_steps = args.reverse_steps or 5
        reverse_solver = "maximum_likelihood_sde"
        reverse_solver_reference = "DiffVC_Eqs_10_12_applied_to_IDMap_Eqs_4_7"
        terminal_distribution = "standard_gaussian"
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        model_type=model_type,
        reverse_steps=reverse_steps,
        reverse_solver=reverse_solver,
        reverse_solver_reference=reverse_solver_reference,
        terminal_distribution=terminal_distribution,
        num_speakers=len(corpus.speakers),
        num_utterances=len(corpus.embeddings),
        embedding_dimension=embedding_dimension,
    )
    if isinstance(model, IDMapDiff):
        config.update(beta_min=model.beta_min, beta_max=model.beta_max)
    else:
        config.update(
            sigma_data=model.sigma_data,
            sigma_min=model.sigma_min,
            sigma_max=model.sigma_max,
            rho=model.rho,
            log_sigma_mean=model.log_sigma_mean,
            log_sigma_std=model.log_sigma_std,
            geometry_weight=model.geometry_weight,
            normalization="per_dimension_speaker_centroid_mean_std",
        )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    best_loss = float("inf")
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        checkpoint_config = checkpoint.get("config", {})
        if checkpoint_config.get("speaker_space") != args.speaker_space:
            raise ValueError("resume checkpoint speaker-space mismatch")
        if checkpoint_config.get("distribution") != args.distribution:
            raise ValueError("resume checkpoint distribution mismatch")
        model.load_state_dict(checkpoint.get("model_raw", checkpoint["model"]))
        ema_model.load_state_dict(checkpoint["model"])
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
    metrics_path = args.output_dir / "metrics.jsonl"
    for epoch in range(1, args.epochs + 1):
        if epoch < start_epoch:
            continue
        model.train()
        running = 0.0
        for _ in range(args.steps_per_epoch):
            identity_indices, targets, auxiliaries = corpus.sample_paper_batch(
                rng,
                speakers_per_batch=args.speakers_per_batch,
                auxiliary_utterances_per_speaker=args.aux_per_speaker,
            )
            selected = torch.as_tensor(identity_indices, device=device)
            target = torch.as_tensor(targets, device=device)
            auxiliary = torch.as_tensor(auxiliaries, device=device)
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                loss, _ = model.training_loss(
                    identity_bank[selected], auxiliary, target
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                for ema_parameter, parameter in zip(
                    ema_model.parameters(), model.parameters(), strict=True
                ):
                    ema_parameter.lerp_(parameter, 1.0 - args.ema_decay)
                for ema_buffer, buffer in zip(
                    ema_model.buffers(), model.buffers(), strict=True
                ):
                    ema_buffer.copy_(buffer)
            running += float(loss.detach())

        epoch_loss = running / args.steps_per_epoch
        record = {"epoch": epoch, "loss": epoch_loss}
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        checkpoint = {
            "model": ema_model.state_dict(),
            "model_raw": model.state_dict(),
            "epoch": epoch,
            "loss": epoch_loss,
            "config": config,
            "speakers": corpus.speakers,
            "fixed_auxiliary_vector": torch.from_numpy(corpus.embeddings[0]),
            "best_loss": min(best_loss, epoch_loss),
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device)
            if device.type == "cuda"
            else None,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, args.output_dir / "best.pt")
        if epoch % args.save_every == 0:
            torch.save(checkpoint, args.output_dir / f"epoch-{epoch:04d}.pt")


if __name__ == "__main__":
    main()
