"""Versioning, provenance, and atomic JSON helpers for the flat fast30 runtime."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

METHOD_VERSION = "V22-C-U25-semanticperturb-flat-v6-20260930"
DEFAULT_OUTPUT_NAME = "fast30_semanticperturb_flat_v6_20260930"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_provenance(code_dir: Path, filenames: list[str]) -> dict:
    files = {
        name: sha256_file(Path(code_dir) / name)
        for name in filenames
    }
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "signature": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def require_cache_signature(payload: dict, expected: str, label: str) -> None:
    got = payload.get("code_signature")
    if got != expected:
        raise RuntimeError(
            f"{label} cache code signature mismatch: expected={expected}, got={got}. "
            "Use a fresh output root or remove that seed's stale cache."
        )
