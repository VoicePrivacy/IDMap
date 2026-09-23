"""Deterministic manifest and artifact validation for parallel synthesis."""

from __future__ import annotations

import json
import wave
from pathlib import Path


def read_kaldi_text(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        stripped = raw_line.strip()
        if not stripped:
            continue
        fields = stripped.split(maxsplit=1)
        if len(fields) != 2 or not fields[1].strip():
            raise ValueError(f"{path}:{line_number}: expected '<utterance-id> <text>'")
        utterance_id, text = fields[0], fields[1].strip()
        if "/" in utterance_id or utterance_id in {".", ".."}:
            raise ValueError(f"unsafe utterance id: {utterance_id!r}")
        if utterance_id in seen:
            raise ValueError(f"duplicate utterance id: {utterance_id}")
        seen.add(utterance_id)
        records.append((utterance_id, text))
    return sorted(records)


def load_generation_manifest(
    path: Path, *, rank: int, world_size: int,
    require_anonymous_index: bool = True,
) -> list[dict[str, object]]:
    if not 0 <= rank < world_size:
        raise ValueError("rank must satisfy 0 <= rank < world-size")
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for position, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        required = {"utterance_id", "text", "output_relative_path"}
        if require_anonymous_index:
            required.add("anonymous_index")
        missing = required.difference(record)
        if missing:
            raise ValueError(f"manifest record is missing {sorted(missing)}")
        utterance_id = str(record["utterance_id"])
        if utterance_id in seen:
            raise ValueError(f"duplicate utterance in manifest: {utterance_id}")
        seen.add(utterance_id)
        relative = Path(str(record["output_relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe output path: {relative}")
        if position % world_size == rank:
            if require_anonymous_index:
                record["anonymous_index"] = int(record["anonymous_index"])
            selected.append(record)
    return selected


def valid_pcm_wave(path: Path, *, sample_rate: int = 48_000) -> bool:
    try:
        with wave.open(str(path), "rb") as handle:
            return (
                handle.getnchannels() == 1
                and handle.getsampwidth() == 2
                and handle.getframerate() == sample_rate
                and handle.getnframes() > 0
            )
    except (FileNotFoundError, EOFError, wave.Error):
        return False
