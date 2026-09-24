#!/usr/bin/env python3
"""Turn text and external identity assignments into an IDMap synthesis manifest.

This script never analyzes source audio. For ``session`` mode, local IDs must
come from a separately evaluated diarization/tracker, not oracle labels.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def safe_relative_path(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"unsafe output_relative_path: {value!r}")
    return str(path)


def build_manifest(
    rows: list[dict[str, object]], *, mode: str, seed: int, pool_size: int
) -> list[dict[str, object]]:
    if mode not in {"utterance", "session"} or pool_size < 1:
        raise ValueError("invalid identity mode or pool size")
    seen_utterances: set[str] = set()
    seen_outputs: set[str] = set()
    keys: list[tuple[str, ...]] = []
    prepared: list[dict[str, object]] = []
    for position, row in enumerate(rows, 1):
        utterance_id = str(row.get("utterance_id", "")).strip()
        text = str(row.get("text", "")).strip()
        if not utterance_id or not text:
            raise ValueError(f"row {position}: utterance_id and text are required")
        if utterance_id in seen_utterances:
            raise ValueError(f"row {position}: duplicate utterance_id {utterance_id!r}")
        seen_utterances.add(utterance_id)
        output = safe_relative_path(
            str(row.get("output_relative_path", f"{utterance_id}.wav"))
        )
        if output in seen_outputs:
            raise ValueError(f"row {position}: duplicate output path {output!r}")
        seen_outputs.add(output)
        if mode == "utterance":
            key = (utterance_id,)
        else:
            session_id = str(row.get("session_id", "")).strip()
            local_id = str(row.get("local_speaker_id", "")).strip()
            if not session_id or not local_id:
                raise ValueError(
                    f"row {position}: session_id and local_speaker_id are required"
                )
            key = (session_id, local_id)
        keys.append(key)
        prepared.append({
            "utterance_id": utterance_id,
            "text": text,
            "output_relative_path": output,
        })
    unique_keys = sorted(set(keys))
    if len(unique_keys) > pool_size:
        raise ValueError("more requested anonymous identities than pool capacity")
    rng = random.Random(seed)
    assignment = dict(zip(unique_keys, rng.sample(range(pool_size), len(unique_keys)), strict=True))
    for row, key in zip(prepared, keys, strict=True):
        row["anonymous_index"] = assignment[key]
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("utterance", "session"), required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--pool-size", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = build_manifest(rows, mode=args.mode, seed=args.seed, pool_size=args.pool_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in manifest:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({
        "output": str(args.output), "utterances": len(manifest),
        "distinct_anonymous_indices": len({row["anonymous_index"] for row in manifest}),
        "mode": args.mode, "seed": args.seed, "pool_size": args.pool_size,
    }))


if __name__ == "__main__":
    main()
