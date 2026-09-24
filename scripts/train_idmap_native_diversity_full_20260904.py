"""Full-corpus native-space IDMap fine-tuning for Qwen3-TTS 1024D vectors.

An epoch consumes every audited source-corpus utterance embedding once as
an auxiliary input.  Target speaker centroids are balanced cyclically and are
always different from the auxiliary speaker.  Selection uses a frozen unseen
identity-index development interval; the final index interval is audited only
after selection.  The intervals are index-disjoint, not speaker-disjoint,
because all native centroids are training statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from voice_anon.idmap.data import SpeakerEmbeddingCorpus
from voice_anon.idmap.mlp import IDMapMLP, IDMapMLPLoss, IdentityVectorSampler
from voice_anon.idmap.native_diversity import NativeDiversityLoss


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def balanced_different_targets(
    auxiliary_speaker_indices: np.ndarray,
    speaker_order: np.ndarray,
) -> np.ndarray:
    """Cycle balanced targets and deterministically avoid equal-speaker pairs."""
    if len(speaker_order) < 2:
        raise ValueError("At least two target speakers are required")
    tiled = np.resize(speaker_order, len(auxiliary_speaker_indices)).copy()
    # Resolve collisions by swaps so the balanced target histogram is exactly
    # preserved.  A replacement would silently delete a target class in an
    # adversarially aligned epoch.
    for index in np.flatnonzero(tiled == auxiliary_speaker_indices):
        if tiled[index] != auxiliary_speaker_indices[index]:
            continue
        candidates = np.flatnonzero(
            (tiled != auxiliary_speaker_indices[index])
            & (tiled[index] != auxiliary_speaker_indices)
            & (np.arange(len(tiled)) != index)
        )
        if not len(candidates):
            raise ValueError("Could not construct distinct balanced pairing")
        partner = int(candidates[0])
        tiled[index], tiled[partner] = tiled[partner], tiled[index]
    if np.any(tiled == auxiliary_speaker_indices):
        raise AssertionError("Target and auxiliary speaker must differ")
    return tiled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--source-audit", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 50:
        raise ValueError("epochs must be in [1, 50]")
    if args.batch_size < 32:
        raise ValueError("batch-size must be at least 32")
    args.output.mkdir(parents=True, exist_ok=False)

    started = time.time()
    state = {
        "complete": False,
        "phase": "loading",
        "started_unix": started,
        "full_corpus": True,
        "full_synthesis_authorized_by_user": True,
    }

    def save_status() -> None:
        temporary = args.output / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.output / "status.json")

    save_status()
    try:
        torch.manual_seed(20260904)
        torch.set_num_threads(8)
        rng = np.random.default_rng(20260904)
        corpus = SpeakerEmbeddingCorpus.load(args.pool, expected_dimension=1024)
        provenance = json.loads(args.source_audit.read_text())
        if not provenance.get("complete") or provenance.get("pool_sha256") != sha256(args.pool):
            raise ValueError("Missing or mismatched source-corpus audit")
        if provenance.get("utterances") != len(corpus.embeddings):
            raise ValueError("Source audit utterance count mismatch")
        initial = torch.load(args.initial, map_location="cpu", weights_only=False)
        with np.load(args.pool, allow_pickle=False) as archive:
            speaker_space = str(archive["speaker_space"].item())
        if initial["config"]["speaker_space"] != speaker_space:
            raise ValueError("Speaker-vector space mismatch")
        if tuple(initial["speakers"]) != corpus.speakers:
            raise ValueError("Speaker order mismatch")

        device = torch.device("cuda")
        model = IDMapMLP(1024).to(device)
        model.load_state_dict(initial["model"], strict=True)
        native = torch.as_tensor(corpus.speaker_means, device=device)
        utterances = torch.as_tensor(corpus.embeddings, device="cpu")
        speaker_to_index = {speaker: index for index, speaker in enumerate(corpus.speakers)}
        auxiliary_speaker_indices = np.asarray(
            [speaker_to_index[speaker] for speaker in corpus.speaker_ids], dtype=np.int64
        )
        regularizer = NativeDiversityLoss(native).to(device)
        reconstruction_loss = IDMapMLPLoss(alpha=initial["config"]["alpha"])
        sampler = IdentityVectorSampler(1024, initial["config"]["distribution"])
        known_identity_vectors = sampler.sample(range(len(native)), device=device)
        fixed_auxiliary = initial["fixed_auxiliary_vector"].to(device).float()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-5
        )

        dev_indices = np.arange(900_000, 902_048)
        final_indices = np.arange(950_000, 952_048)
        dev_ids = sampler.sample(dev_indices, device=device)
        final_ids = sampler.sample(final_indices, device=device)
        steps_per_epoch = math.ceil(len(corpus.embeddings) / args.batch_size)
        config = dict(
            initial["config"],
            variant="native_diversity_full_corpus_v1",
            initial_sha256=sha256(args.initial),
            pool_sha256=sha256(args.pool),
            source_corpus=provenance["source_corpus"],
            source_audit_sha256=sha256(args.source_audit),
            utterances_per_epoch=len(corpus.embeddings),
            speakers=len(corpus.speakers),
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            original_reconstruction_weight=1.0,
            centered_geometry_weight=0.1,
            native_distribution_weight=0.2,
            auxiliary_invariance_weight=0.05,
            target_auxiliary_speakers_distinct=True,
            inference_fixed_auxiliary=True,
            vectors_clipped_or_normalized=False,
            novel_training_indices=[len(native), 900_000],
            dev_identity_indices=[900_000, 902_048],
            final_identity_indices=[950_000, 952_048],
            index_partitions_disjoint=True,
            index_partitions_are_not_speaker_disjoint=True,
            tts_trained=False,
        )
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

        @torch.no_grad()
        def evaluate(identity_vectors: torch.Tensor, label: str) -> dict:
            model.eval()
            generated = torch.cat(
                [
                    model(chunk, fixed_auxiliary.expand(len(chunk), -1))
                    for chunk in identity_vectors.split(256)
                ]
            )
            reference = native[
                torch.arange(len(generated), device=device) % len(native)
            ]
            distribution, parts = regularizer(generated, reference)
            normalized = F.normalize(generated, dim=-1)
            half = len(generated) // 2
            pair_cosine = (normalized[:half] * normalized[half : 2 * half]).sum(-1)
            return {
                "partition": label,
                "distribution": float(distribution),
                "parts": {key: float(value) for key, value in parts.items()},
                "pair_cosine_median": float(pair_cosine.median()),
                "norm_median": float(generated.norm(dim=1).median()),
            }

        history: list[dict] = []
        best_score = float("inf")
        selected: dict | None = None
        state.update(
            phase="training",
            gpu=torch.cuda.get_device_name(),
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
        )
        save_status()
        with (args.output / "metrics.jsonl").open("x") as metrics:
            global_step = 0
            for epoch in range(1, args.epochs + 1):
                model.train()
                utterance_order = rng.permutation(len(corpus.embeddings))
                speaker_order = rng.permutation(len(corpus.speakers))
                epoch_aux_speakers = auxiliary_speaker_indices[utterance_order]
                epoch_targets = balanced_different_targets(
                    epoch_aux_speakers, speaker_order
                )
                running = {"loss": 0.0, "reconstruction": 0.0, "distribution": 0.0}
                for offset in range(0, len(utterance_order), args.batch_size):
                    selection = utterance_order[offset : offset + args.batch_size]
                    target_numpy = epoch_targets[offset : offset + len(selection)]
                    target_indices = torch.as_tensor(target_numpy, device=device)
                    auxiliary = utterances[selection].to(device, non_blocking=True)
                    target = native[target_indices]
                    ids = known_identity_vectors[target_indices]
                    novel_numbers = rng.integers(
                        len(native), 900_000, size=len(selection), dtype=np.int64
                    )
                    novel_ids = sampler.sample(novel_numbers, device=device)
                    alternative_selection = rng.integers(
                        0, len(corpus.embeddings), size=len(selection), dtype=np.int64
                    )
                    alternative = utterances[alternative_selection].to(
                        device, non_blocking=True
                    )

                    optimizer.zero_grad(set_to_none=True)
                    known_prediction = model(ids, auxiliary)
                    novel_prediction = model(novel_ids, auxiliary)
                    alternative_prediction = model(novel_ids, alternative)
                    anchor = reconstruction_loss(known_prediction, target)
                    geometry = regularizer.centered_geometry(known_prediction, target)
                    distribution, _ = regularizer(novel_prediction, target)
                    invariance = (1.0 - F.cosine_similarity(
                        novel_prediction, alternative_prediction, dim=-1
                    )).mean()
                    loss = anchor + 0.1 * geometry + 0.2 * distribution + 0.05 * invariance
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss")
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 5.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    global_step += 1
                    running["loss"] += float(loss.detach())
                    running["reconstruction"] += float(anchor.detach())
                    running["distribution"] += float(distribution.detach())

                development = evaluate(dev_ids, "development")
                divisor = steps_per_epoch
                record = {
                    "epoch": epoch,
                    "step": global_step,
                    "all_utterances_consumed_once": True,
                    "mean_loss": running["loss"] / divisor,
                    "mean_reconstruction": running["reconstruction"] / divisor,
                    "mean_distribution": running["distribution"] / divisor,
                    "gradient_norm_last": float(gradient),
                    "development": development,
                    "seconds": time.time() - started,
                }
                history.append(record)
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                print(json.dumps(record), flush=True)
                checkpoint = {
                    "model": model.state_dict(),
                    "config": config,
                    "speakers": corpus.speakers,
                    "fixed_auxiliary_vector": fixed_auxiliary.cpu(),
                    "epoch": epoch,
                    "step": global_step,
                    "complete_epoch": True,
                    "validation": development,
                    "native_regularizer": regularizer.state_dict(),
                }
                epoch_path = args.output / f"epoch-{epoch:02d}.pt"
                torch.save(checkpoint, epoch_path)
                selection_score = development["distribution"]
                if selection_score < best_score:
                    best_score = selection_score
                    selected = record
                    torch.save(checkpoint, args.output / "best.pt")
                state.update(epoch=epoch, step=global_step, latest=record)
                save_status()

        if selected is None:
            raise RuntimeError("No checkpoint selected")
        selected_checkpoint = torch.load(
            args.output / "best.pt", map_location="cpu", weights_only=False
        )
        model.load_state_dict(selected_checkpoint["model"], strict=True)
        final_audit = evaluate(final_ids, "final_index_audit")
        results = {
            "complete": True,
            "scope": "full " + provenance["source_corpus"] + " native-vector corpus",
            "utterances": len(corpus.embeddings),
            "speakers": len(corpus.speakers),
            "history": history,
            "selected": selected,
            "final_index_audit": final_audit,
            "checkpoint_sha256": sha256(args.output / "best.pt"),
            "rendered_quality_pending": True,
            "vpc_track1_full_synthesis_pending": True,
        }
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        state.update(
            complete=True,
            phase="full_vector_training_complete",
            selected_epoch=selected["epoch"],
            final_index_audit=final_audit,
            finished_unix=time.time(),
        )
        save_status()
    except Exception as error:
        state.update(phase="failed", error=repr(error), finished_unix=time.time())
        save_status()
        raise


if __name__ == "__main__":
    main()
