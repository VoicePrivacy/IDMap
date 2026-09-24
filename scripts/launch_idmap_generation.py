#!/usr/bin/env python3
"""Launch independent Qwen/CosyVoice IDMap synthesis workers on selected GPUs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from voice_anon.generation import load_generation_manifest, valid_pcm_wave


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("qwen", "cosy"), required=True)
    parser.add_argument("--gpus", required=True, help="comma-separated physical GPU IDs")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--idmap-checkpoint", type=Path, required=True)
    parser.add_argument("--hf-model-dir", type=Path, help="CosyVoice3 converted LLM")
    parser.add_argument("--expected-speaker-space-prefix", help="Qwen encoder family")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = [part.strip() for part in args.gpus.split(",")]
    if not gpus or any(not part.isdigit() for part in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must contain distinct nonnegative integer IDs")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.backend == "cosy" and args.hf_model_dir is None:
        raise ValueError("CosyVoice3 requires --hf-model-dir")
    if args.backend == "qwen" and not args.expected_speaker_space_prefix:
        raise ValueError("Qwen requires --expected-speaker-space-prefix")
    rows = load_generation_manifest(args.manifest, rank=0, world_size=1)
    if not rows:
        raise ValueError("manifest is empty")
    script = Path(__file__).with_name(
        "generate_qwen3tts_idmap_worker.py" if args.backend == "qwen"
        else "generate_cosyvoice3_multigpu_worker.py"
    )
    log_dir = args.output_dir / "_logs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    log_dir.mkdir(parents=True, exist_ok=False)
    processes: list[tuple[int, subprocess.Popen[bytes], object]] = []
    try:
        for rank, gpu in enumerate(gpus):
            command = [
                sys.executable, str(script), "--manifest", str(args.manifest),
                "--output-dir", str(args.output_dir), "--model-dir", str(args.model_dir),
                "--idmap-checkpoint", str(args.idmap_checkpoint),
                "--rank", str(rank), "--world-size", str(len(gpus)),
            ]
            if args.backend == "qwen":
                command += [
                    "--expected-speaker-space-prefix", args.expected_speaker_space_prefix,
                    "--batch-size", str(args.batch_size), "--attention", "sdpa",
                ]
            else:
                command += [
                    "--hf-model-dir", str(args.hf_model_dir), "--prompt-mode", "none",
                    "--llm-batch-size", str(args.batch_size),
                ]
            if args.overwrite:
                command.append("--overwrite")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            log = (log_dir / f"rank-{rank}.log").open("xb")
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            processes.append((rank, process, log))
        failures = []
        for rank, process, _ in processes:
            code = process.wait()
            if code:
                failures.append({"rank": rank, "exit_code": code})
    except KeyboardInterrupt:
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, process, _ in processes:
            process.wait()
        raise
    finally:
        for _, _, log in processes:
            log.close()
    invalid = [
        str(row["output_relative_path"]) for row in rows
        if not valid_pcm_wave(args.output_dir / str(row["output_relative_path"]), sample_rate=16_000)
    ]
    result = {
        "backend": args.backend, "gpus": gpus, "expected": len(rows),
        "valid": len(rows) - len(invalid), "invalid_count": len(invalid),
        "invalid_examples": invalid[:20], "worker_failures": failures,
        "log_directory": str(log_dir), "complete": not failures and not invalid,
    }
    (log_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
