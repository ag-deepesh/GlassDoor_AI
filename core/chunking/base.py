from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.schemas import ParsedDoc, Chunk
from core.tokenizer import count_tokens


@dataclass
class ChunkConfig:
    """Manually configurable per the platform's requirements.
    chunk_size / overlap are in approximate tokens (see core/tokenizer.py)."""
    chunk_size: int = 512
    overlap: int = 64


class BaseChunker(ABC):
    def __init__(self, config: ChunkConfig | None = None):
        self.config = config or ChunkConfig()

    @abstractmethod
    def chunk(self, doc: ParsedDoc) -> list[Chunk]: ...

    def _mk(self, doc: ParsedDoc, text: str, n: int, page: int | None = None, **meta) -> Chunk:
        return Chunk(chunk_id=f"{doc.doc_id}_c{n}", doc_id=doc.doc_id, text=text,
                      page=page, n_tokens=count_tokens(text), meta=meta)


def split_by_tokens(text: str, size: int, overlap: int) -> list[str]:
    """Word-based sliding window sized to hit ~`size` tokens per chunk, with
    ~`overlap` tokens repeated between consecutive chunks. Shared by every
    chunker that needs a final hard-size fallback."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
        i += step
    return chunks
