"""
Knowledge Base persistence.

A KB is a named, on-disk unit produced once by Pipeline.build_kb() and
reused by every later Pipeline.answer_query() call against it: a Chroma
persist dir plus a metadata.json recording exactly which embedding model
built it. Query time must reuse that exact model -- never a dropdown --
so metadata.json is the single source of truth Pipeline.load() reads from.

Storage root is $GLASSBOX_DATA_DIR (defaults to <repo>/data) -- this exists
as an escape hatch so KB data/venvs can be pointed outside an OneDrive-synced
checkout if that ever becomes a problem, without any code changes.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.parsing.assorted import EXT_TO_METHOD

MAX_FILES = 20
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_TOTAL_BYTES = 150 * 1024 * 1024
ALLOWED_EXTENSIONS = set(EXT_TO_METHOD)

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class UploadLimitError(ValueError):
    """Raised when an upload exceeds the platform's KB creation limits."""


def data_dir() -> Path:
    return Path(os.environ.get("GLASSBOX_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))


def kbs_dir() -> Path:
    d = data_dir() / "kbs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _validate_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise ValueError(f"Invalid KB name '{name}' -- use 1-64 characters, letters/numbers/underscore/hyphen only.")


def kb_path(name: str) -> Path:
    _validate_name(name)
    return kbs_dir() / name


def kb_exists(name: str) -> bool:
    return (kb_path(name) / "metadata.json").exists()


def list_kbs() -> list[dict]:
    """Every existing KB's metadata, for the 'use existing' picker."""
    out = []
    for p in sorted(kbs_dir().iterdir()):
        meta_path = p / "metadata.json"
        if p.is_dir() and meta_path.exists():
            out.append(json.loads(meta_path.read_text()))
    return out


def create_kb(name: str) -> Path:
    _validate_name(name)
    path = kb_path(name)
    if path.exists():
        raise FileExistsError(f"A KB named '{name}' already exists -- pick a different name or delete it first.")
    (path / "chroma").mkdir(parents=True)
    (path / "uploads").mkdir(parents=True)
    return path


def delete_kb(name: str) -> None:
    path = kb_path(name)
    if path.exists():
        shutil.rmtree(path)


def load_metadata(name: str) -> dict:
    meta_path = kb_path(name) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json for KB '{name}' -- was build_kb() ever completed for it?")
    return json.loads(meta_path.read_text())


def save_metadata(name: str, metadata: dict) -> None:
    meta_path = kb_path(name) / "metadata.json"
    now = datetime.now(timezone.utc).isoformat()
    existing_created_at = json.loads(meta_path.read_text())["created_at"] if meta_path.exists() else now
    merged = {**metadata, "name": name, "created_at": existing_created_at, "updated_at": now}
    meta_path.write_text(json.dumps(merged, indent=2))


@dataclass
class UploadedFile:
    """Minimal shape kb_store needs to validate an upload -- the API layer
    adapts whatever framework's UploadFile into this before calling
    validate_upload(), so this module has no FastAPI dependency."""
    filename: str
    size: int


def validate_upload(files: list[UploadedFile]) -> None:
    if not files:
        raise UploadLimitError("No files provided -- attach at least one document to build a KB.")
    if len(files) > MAX_FILES:
        raise UploadLimitError(f"Too many files ({len(files)}) -- max {MAX_FILES} per KB.")
    total = 0
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise UploadLimitError(f"'{f.filename}' has unsupported extension '{ext}' -- allowed: {sorted(ALLOWED_EXTENSIONS)}")
        if f.size > MAX_FILE_BYTES:
            raise UploadLimitError(f"'{f.filename}' is {f.size / 1e6:.1f}MB -- max {MAX_FILE_BYTES / 1e6:.0f}MB per file.")
        total += f.size
    if total > MAX_TOTAL_BYTES:
        raise UploadLimitError(f"Corpus totals {total / 1e6:.1f}MB -- max {MAX_TOTAL_BYTES / 1e6:.0f}MB per KB.")


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
