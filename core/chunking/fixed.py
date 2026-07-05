from __future__ import annotations
from core.registry import register
from core.schemas import ParsedDoc, Chunk
from core.chunking.base import BaseChunker, split_by_tokens


@register("chunking", "fixed")
class FixedChunker(BaseChunker):
    """Uniform-size chunks, ignoring sentence/section boundaries entirely.
    Simplest option; best for short, uniform, fact-dense text (FAQs, specs)."""

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        out, n = [], 0
        for block in doc.text_blocks:
            for piece in split_by_tokens(block.text, self.config.chunk_size, self.config.overlap):
                n += 1
                out.append(self._mk(doc, piece, n, page=block.page))
        return out
