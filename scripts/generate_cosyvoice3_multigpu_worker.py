#!/usr/bin/env python3
"""Persistent-GPU, true-batch CosyVoice3 + native-space IDMap-MLP worker."""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time
import wave

from hyperpyyaml import load_hyperpyyaml
import numpy as np
import s3tokenizer
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer
from matcha.utils.audio import mel_spectrogram as matcha_mel_spectrogram

from voice_anon.backends.cosyvoice3 import (
    flow_inference_batch,
    hift_inference_batch,
    sample_speech_tokens_batch,
    plan_cosyvoice3_batches,
    plan_token2wav_batches,
)
from voice_anon.generation import load_generation_manifest, valid_pcm_wave
from voice_anon.idmap.diffusion import IDMapDiff, IDMapDiffEDM
from voice_anon.idmap.mlp import IDMapMLP, IdentityVectorSampler


MEL_SPECTROGRAM = partial(
    matcha_mel_spectrogram,
    n_fft=1920,
    num_mels=80,
    sampling_rate=24_000,
    hop_size=480,
    win_size=1920,
    fmin=0,
    fmax=None,
    center=False,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--hf-model-dir", type=Path, required=True)
    conditioning = parser.add_mutually_exclusive_group(required=True)
    conditioning.add_argument("--idmap-checkpoint", type=Path)
    conditioning.add_argument(
        "--speaker-pool",
        type=Path,
        help=("CosyVoice3-native 192-D real-speaker centroid archive.  "
              "Manifest rows must contain pool_speaker_id; IDMap is not loaded."),
    )
    parser.add_argument(
        "--prompt-mode", choices=("fixed", "manifest", "none"), default="fixed"
    )
    parser.add_argument("--prompt-audio", type=Path)
    parser.add_argument("--prompt-cache", type=Path)
    parser.add_argument("--prompt-text")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--llm-batch-size", type=int, default=8)
    parser.add_argument("--batching", choices=("bucket", "exact"), default="bucket")
    parser.add_argument("--sampler", choices=("fast", "legacy"), default="fast")
    parser.add_argument("--llm-input-token-budget", type=int, default=32768)
    parser.add_argument("--llm-compact-interval", type=int, default=32)
    parser.add_argument("--decode-backend", choices=("dynamic", "cudagraph"), default="dynamic")
    parser.add_argument("--token2wav-batch-size", type=int, default=4)
    parser.add_argument("--flow-precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--token2wav-token-budget",
        type=int,
        default=8192,
        help="cap max speech-token length times effective Flow/HiFT batch size",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-errors", type=int, default=10)
    return parser.parse_args()


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    audio, original_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim != 1:
        raise ValueError("prompt audio must be mono")
    value = torch.from_numpy(audio).unsqueeze(0)
    if original_rate != sample_rate:
        value = torchaudio.functional.resample(value, original_rate, sample_rate)
    return value.squeeze(0)


def atomic_write_wave(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = waveform.detach().float().cpu().clamp(-1.0, 1.0)
    pcm = (value * 32767.0).round().to(torch.int16).numpy().tobytes()
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    os.replace(temporary, path)


def prompt_tokenize(model: torch.nn.Module, audio: torch.Tensor, device: torch.device) -> list[int]:
    mel = s3tokenizer.log_mel_spectrogram(audio)
    padded, lengths = s3tokenizer.padding([mel])
    codes, code_lengths = model.quantize(padded.to(device), lengths.to(device))
    return codes[0, : int(code_lengths[0])].tolist()


def build_llm_input(tokenizer: object, prompt_text: str, target_text: str, prompt_tokens: list[int]) -> torch.Tensor:
    prompt_speech = "".join(f"<|s_{token}|>" for token in prompt_tokens)
    chat = [
        {
            "role": "user",
            "content": "You are a helpful assistant.<|endofprompt|>"
            + prompt_text
            + target_text.strip(),
        },
        {"role": "assistant", "content": prompt_speech},
    ]
    return tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        return_tensors="pt",
        continue_final_message=True,
    ).squeeze(0)


@torch.inference_mode()
def generate_batch(
    model: torch.nn.Module,
    input_items: list[torch.Tensor],
    *,
    device: torch.device,
    eos_id: int,
    maximum_new_tokens: int,
) -> list[torch.Tensor]:
    maximum = max(len(item) for item in input_items)
    padded, masks = [], []
    for item in input_items:
        missing = maximum - len(item)
        padded.append(torch.cat((torch.full((missing,), eos_id), item)))
        masks.append(torch.cat((torch.zeros(missing), torch.ones(len(item)))))
    outputs = model.generate(
        input_ids=torch.stack(padded).to(device),
        attention_mask=torch.stack(masks).long().to(device),
        max_new_tokens=maximum_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
    )
    return [row[maximum:].cpu() for row in outputs]


def speech_ids(tokens: torch.Tensor, metadata: dict[str, object]) -> list[int]:
    offset = int(metadata["speech_token_offset"])
    eos_id = int(metadata["eos_token_id"])
    size = int(metadata["base_speech_token_size"])
    result: list[int] = []
    for token in tokens.tolist():
        if token == eos_id:
            break
        value = int(token) - offset
        if 0 <= value < size:
            result.append(value)
    return result


def main() -> None:
    args = parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("invalid rank/world size")
    if args.prompt_mode == "fixed" and (not args.prompt_text or (args.prompt_cache is None and args.prompt_audio is None)):
        raise ValueError("fixed prompt mode requires prompt text and audio/cache")
    if args.prompt_mode == "none" and any(
        value is not None
        for value in (args.prompt_audio, args.prompt_cache, args.prompt_text)
    ):
        raise ValueError("none prompt mode forbids prompt text, audio, and cache")
    device = torch.device("cuda")
    records = load_generation_manifest(
        args.manifest, rank=args.rank, world_size=args.world_size,
        require_anonymous_index=args.speaker_pool is None,
    )
    pending = [
        record
        for record in records
        if args.overwrite
        or not valid_pcm_wave(
            args.output_dir / str(record["output_relative_path"]), sample_rate=16_000
        )
    ]
    if args.max_records is not None:
        pending = pending[: args.max_records]
    print(json.dumps({"stage": "manifest_ready", "pending": len(pending)}), flush=True)
    if not pending:
        return

    metadata = json.loads(
        (args.hf_model_dir / "cosyvoice3_metadata.json").read_text(encoding="utf-8")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_dir, fix_mistral_regex=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_dir, torch_dtype=torch.bfloat16
    ).to(device).eval()
    print(json.dumps({"stage": "llm_ready"}), flush=True)
    if args.prompt_mode in {"manifest", "none"}:
        prompt_tokens, prompt_mel, prompt_length = [], torch.empty(0, 80), 0
    elif args.prompt_cache is not None:
        with np.load(args.prompt_cache, allow_pickle=False) as archive:
            prompt_tokens = np.asarray(archive["prompt_tokens"], dtype=np.int64).tolist()
            prompt_mel = torch.from_numpy(
                np.asarray(archive["prompt_mel"], dtype=np.float32)
            )
        prompt_length = len(prompt_tokens)
    else:
        prompt16 = load_audio(args.prompt_audio, 16_000)
        audio_tokenizer = s3tokenizer.load_model(
            str(args.model_dir / "speech_tokenizer_v3.onnx")
        ).to(device).eval()
        prompt_tokens = prompt_tokenize(audio_tokenizer, prompt16, device)
        del audio_tokenizer
        prompt24 = load_audio(args.prompt_audio, 24_000).unsqueeze(0)
        prompt_mel = MEL_SPECTROGRAM(prompt24).transpose(1, 2).squeeze(0)
        prompt_length = min(len(prompt_tokens), int(prompt_mel.shape[0]) // 2)
        prompt_tokens = prompt_tokens[:prompt_length]
        prompt_mel = prompt_mel[: 2 * prompt_length]
    print(json.dumps({"stage": "prompt_ready", "prompt_tokens": prompt_length}), flush=True)

    with (args.model_dir / "cosyvoice3.yaml").open("r", encoding="utf-8") as handle:
        config = load_hyperpyyaml(
            handle,
            overrides={
                "qwen_pretrain_path": str(args.model_dir / "CosyVoice-BlankEN"),
                "llm": None,
                "hifigan": None,
            },
        )
    flow = config["flow"]
    flow.load_state_dict(
        torch.load(args.model_dir / "flow.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    flow.to(device).eval()
    hift = config["hift"]
    hift.load_state_dict(
        {
            key.replace("generator.", ""): value
            for key, value in torch.load(
                args.model_dir / "hift.pt", map_location="cpu", weights_only=True
            ).items()
        },
        strict=True,
    )
    hift.to(device).eval()
    print(json.dumps({"stage": "token2wav_ready"}), flush=True)

    with (args.model_dir / "campplus.onnx").open("rb") as handle:
        campplus_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    direct_pool: dict[str, torch.Tensor] | None = None
    if args.speaker_pool is not None:
        with np.load(args.speaker_pool, allow_pickle=False) as archive:
            required = {"embeddings", "speaker_ids", "speaker_space", "checkpoint_sha256"}
            if not required.issubset(archive.files):
                raise ValueError(
                    f"speaker pool missing keys: {sorted(required - set(archive.files))}"
                )
            pool_embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
            pool_speaker_ids = np.asarray(archive["speaker_ids"]).astype(str)
            speaker_space = str(np.asarray(archive["speaker_space"]).item())
            pool_checkpoint_hash = str(np.asarray(archive["checkpoint_sha256"]).item())
        if pool_embeddings.ndim != 2 or pool_embeddings.shape[1] != 192:
            raise ValueError(f"expected (N, 192) speaker pool, got {pool_embeddings.shape}")
        if len(pool_speaker_ids) != len(pool_embeddings):
            raise ValueError("speaker pool row metadata mismatch")
        if len(set(pool_speaker_ids.tolist())) != len(pool_speaker_ids):
            raise ValueError("speaker pool contains duplicate speaker IDs")
        if not np.isfinite(pool_embeddings).all():
            raise ValueError("speaker pool contains non-finite values")
        if pool_checkpoint_hash != campplus_hash:
            raise ValueError("speaker pool and CosyVoice3 use different CAM++ checkpoints")
        if speaker_space != f"cosyvoice3-campplus-v1:{campplus_hash[:16]}":
            raise ValueError(f"speaker pool has incompatible speaker space: {speaker_space}")
        direct_pool = {
            str(speaker_id): torch.from_numpy(vector.copy()).to(device)
            for speaker_id, vector in zip(pool_speaker_ids, pool_embeddings, strict=True)
        }
        checkpoint_config: dict[str, object] = {}
        idmap = sampler = auxiliary = None
        print(json.dumps({"stage": "real_speaker_pool_ready",
                          "speakers": len(direct_pool),
                          "speaker_space": speaker_space}), flush=True)
    else:
        checkpoint = torch.load(
            args.idmap_checkpoint, map_location=device, weights_only=False
        )
        checkpoint_config = checkpoint["config"]
        if int(checkpoint_config["embedding_dimension"]) != 192:
            raise ValueError("IDMap checkpoint is not in CosyVoice3's speaker space")
        speaker_space = str(checkpoint_config.get("speaker_space", ""))
        if not speaker_space.startswith("cosyvoice3-campplus-v1:"):
            raise ValueError(f"IDMap checkpoint has incompatible speaker space: {speaker_space}")
        if not campplus_hash.startswith(speaker_space.rsplit(":", 1)[1]):
            raise ValueError("IDMap and CosyVoice3 checkpoints use different CAM++ weights")
        model_type = checkpoint_config.get("model_type", "idmap_mlp")
        if model_type == "idmap_mlp":
            idmap = IDMapMLP(dimension=192)
        elif model_type == "idmap_diff":
            idmap = IDMapDiff(dimension=192)
        elif model_type == "idmap_diff_edm":
            idmap = IDMapDiffEDM(dimension=192)
        else:
            raise ValueError(f"unsupported IDMap checkpoint type: {model_type}")
        idmap = idmap.to(device).eval()
        idmap.load_state_dict(checkpoint["model"])
        sampler = IdentityVectorSampler(
            dimension=192,
            distribution=checkpoint_config.get("distribution", "normal"),
        )
        auxiliary = checkpoint["fixed_auxiliary_vector"].to(device)

    references = {}
    prepared = []
    for item in pending:
        if args.prompt_mode == "manifest":
            cache_path = str(item["prompt_cache"])
            if cache_path not in references:
                with np.load(cache_path, allow_pickle=False) as archive:
                    if str(archive["source_audio"]) != str(item["prompt_audio"]):
                        raise ValueError("reference cache audio mismatch")
                    if str(archive["source_text"]) != str(item["prompt_text"]):
                        raise ValueError("reference cache text mismatch")
                    references[cache_path] = (
                        np.asarray(archive["prompt_tokens"], dtype=np.int64).tolist(),
                        torch.from_numpy(np.asarray(archive["prompt_mel"], dtype=np.float32)),
                    )
            item_prompt_tokens = references[cache_path][0]
            item_prompt_text = str(item["prompt_text"]).strip() + " "
        elif args.prompt_mode == "none":
            item_prompt_tokens, item_prompt_text = [], ""
        else:
            item_prompt_tokens, item_prompt_text = prompt_tokens, args.prompt_text
        llm_input = build_llm_input(
            tokenizer, item_prompt_text, str(item["text"]), item_prompt_tokens
        )
        text_token_count = max(1, len(tokenizer.encode(str(item["text"]))))
        prepared.append((item, llm_input, text_token_count))
    work_batches = plan_cosyvoice3_batches(
        prepared, batch_size=args.llm_batch_size, mode=args.batching,
        input_token_budget=args.llm_input_token_budget,
    )
    print(json.dumps({"stage": "batch_plan", "sampler": args.sampler,
                      "batching": args.batching, "batches": len(work_batches),
                      "mean_batch": len(prepared) / max(1, len(work_batches)),
                      "batch_sizes": [len(b) for b in work_batches]}), flush=True)
    log_path = args.output_dir / "_worker_logs" / f"rank-{args.rank:02d}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = errors = 0
    audio_seconds = 0.0
    started_unix = time.time()
    started = time.perf_counter()
    for batch_number, prepared_batch in enumerate(work_batches):
        batch = [entry[0] for entry in prepared_batch]
        batch_started = time.perf_counter()
        print(
            json.dumps(
                {
                    "stage": "batch_start",
                    "batch_number": batch_number,
                    "batch_size": len(batch),
                    "input_length_min": min(len(entry[1]) for entry in prepared_batch),
                    "input_length_max": max(len(entry[1]) for entry in prepared_batch),
                }
            ),
            flush=True,
        )
        try:
            llm_inputs = [entry[1] for entry in prepared_batch]
            text_token_counts = [entry[2] for entry in prepared_batch]
            seeds = [
                int.from_bytes(
                    hashlib.blake2b(
                        str(item["utterance_id"]).encode(), digest_size=8
                    ).digest(),
                    "big",
                )
                for item in batch
            ]
            generated_speech = sample_speech_tokens_batch(
                model,
                llm_inputs,
                metadata=metadata,
                seeds=seeds,
                minimum_tokens=[max(10, 2 * count) for count in text_token_counts],
                maximum_tokens=[
                    min(args.max_new_tokens, 20 * count) for count in text_token_counts
                ],
                implementation=args.sampler,
                compact_interval=args.llm_compact_interval,
                decode_backend=args.decode_backend,
            )
            torch.cuda.synchronize()
            llm_seconds = time.perf_counter() - batch_started
            flow_seconds = hift_seconds = write_seconds = 0.0
            if any(not item for item in generated_speech):
                raise RuntimeError("CosyVoice3 LLM returned an empty speech-token sequence")
            acoustic_lengths = [
                len(tokens) + (len(references[str(item["prompt_cache"])][0])
                               if args.prompt_mode == "manifest" else prompt_length)
                for item, tokens in zip(batch, generated_speech, strict=True)
            ]
            acoustic_batches = plan_token2wav_batches(acoustic_lengths,
                batch_size=args.token2wav_batch_size, token_budget=args.token2wav_token_budget)
            effective_token2wav_batch = max(map(len, acoustic_batches))
            if direct_pool is not None:
                missing_pool_ids = [
                    str(item.get("pool_speaker_id")) for item in batch
                    if str(item.get("pool_speaker_id")) not in direct_pool
                ]
                if missing_pool_ids:
                    raise ValueError(
                        f"manifest references unknown pool speakers: {missing_pool_ids[:5]}"
                    )
                vectors = torch.stack(
                    [direct_pool[str(item["pool_speaker_id"])] for item in batch]
                )
            else:
                indices = [int(item["anonymous_index"]) for item in batch]
                if isinstance(idmap, (IDMapDiff, IDMapDiffEDM)):
                    vectors = idmap.generate_from_indices(
                        indices, auxiliary, sampler,
                        steps=int(checkpoint_config.get("reverse_steps", 5)),
                    )
                else:
                    vectors = idmap.generate_from_indices(indices, auxiliary, sampler)
            for group_indices in acoustic_batches:
                group = [batch[i] for i in group_indices]
                group_tokens = [generated_speech[i] for i in group_indices]
                group_vectors = vectors[group_indices]
                stage_started = time.perf_counter()
                mels = flow_inference_batch(
                    flow,
                    generated_tokens=group_tokens,
                    prompt_tokens=[references[str(item["prompt_cache"])][0] for item in group]
                    if args.prompt_mode == "manifest" else [prompt_tokens] * len(group),
                    prompt_features=[references[str(item["prompt_cache"])][1] for item in group]
                    if args.prompt_mode == "manifest" else [prompt_mel] * len(group),
                    speaker_embeddings=group_vectors,
                    estimator_precision=args.flow_precision,
                )
                torch.cuda.synchronize()
                flow_seconds += time.perf_counter() - stage_started
                stage_started = time.perf_counter()
                waveforms24 = hift_inference_batch(hift, mels)
                torch.cuda.synchronize()
                hift_seconds += time.perf_counter() - stage_started
                stage_started = time.perf_counter()
                for item, waveform24 in zip(group, waveforms24, strict=True):
                    waveform16 = torchaudio.functional.resample(
                        waveform24.squeeze(0), 24_000, 16_000
                    )
                    atomic_write_wave(
                        args.output_dir / str(item["output_relative_path"]),
                        waveform16,
                        16_000,
                    )
                    audio_seconds += waveform16.numel() / 16_000
                    completed += 1
                write_seconds += time.perf_counter() - stage_started
                del mels, waveforms24
            event = {
                "batch_number": batch_number,
                "completed": completed,
                "timestamp": time.time(),
                "batch_size": len(batch),
                "seconds": time.perf_counter() - batch_started,
                "llm_seconds": llm_seconds,
                "flow_seconds": flow_seconds,
                "hift_seconds": hift_seconds,
                "write_seconds": write_seconds,
                "sampler": args.sampler,
                "llm_compact_interval": args.llm_compact_interval,
                "decode_backend": args.decode_backend,
                "batching": args.batching,
                "speech_token_lengths": [len(item) for item in generated_speech],
                "effective_token2wav_batch_size": effective_token2wav_batch,
                "token2wav_actual_batch_sizes": [len(g) for g in acoustic_batches],
                "flow_precision": args.flow_precision,
                "token2wav_token_budget": args.token2wav_token_budget,
                "prompt_mode": args.prompt_mode,
                "utterance_ids": [str(item["utterance_id"]) for item in batch],
                "reference_utterances": [item.get("reference_utterance") for item in batch],
            }
        except Exception as error:  # noqa: BLE001
            errors += len(batch)
            event = {
                "batch_number": batch_number,
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        if errors >= args.max_errors:
            raise RuntimeError(f"worker reached max-errors={args.max_errors}")

    elapsed = time.perf_counter() - started
    result = {
        "rank": args.rank,
        "assigned": len(records),
        "pending": len(pending),
        "completed": completed,
        "errors": errors,
        "seconds": elapsed,
        "started_unix": started_unix,
        "finished_unix": time.time(),
        "generated_audio_seconds": audio_seconds,
        "rtf": elapsed / max(audio_seconds, 1e-9),
        "llm_batch_size": args.llm_batch_size,
        "token2wav_batch_size": args.token2wav_batch_size,
        "token2wav_token_budget": args.token2wav_token_budget,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "sampler": args.sampler,
        "llm_compact_interval": args.llm_compact_interval,
        "decode_backend": args.decode_backend,
        "batching": args.batching,
        "flow_precision": args.flow_precision,
    }
    print(json.dumps(result), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
