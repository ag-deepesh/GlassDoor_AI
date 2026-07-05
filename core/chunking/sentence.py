from __future__ import annotations
import re
from core.registry import register
from core.schemas import ParsedDoc, Chunk
from core.chunking.base import BaseChunker
from core.tokenizer import count_tokens

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@register("chunking", "sentence")
class SentenceChunker(BaseChunker):
    """Groups whole sentences together up to ~chunk_size tokens, with the
    last `overlap`-worth of sentences repeated into the next chunk. Use when
    exact sentence-level citation matters more than uniform chunk size."""

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        out, n = [], 0
        for block in doc.text_blocks:
            sentences = [s.strip() for s in _SENTENCE_RE.split(block.text) if s.strip()]
            if not sentences:
                continue

            current: list[str] = []
            current_tokens = 0
            for sent in sentences:
                sent_tokens = count_tokens(sent)
                if current and current_tokens + sent_tokens > self.config.chunk_size:
                    n += 1
                    out.append(self._mk(doc, " ".join(current), n, page=block.page))
                    # Carry the tail sentences forward as overlap.
                    carried, carried_tokens = [], 0
                    for s in reversed(current):
                        t = count_tokens(s)
                        if carried_tokens + t > self.config.overlap:
                            break
                        carried.insert(0, s)
                        carried_tokens += t
                    current, current_tokens = carried, carried_tokens
                current.append(sent)
                current_tokens += sent_tokens

            if current:
                n += 1
                out.append(self._mk(doc, " ".join(current), n, page=block.page))
        return out
