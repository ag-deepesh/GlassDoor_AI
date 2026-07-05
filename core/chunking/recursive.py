from __future__ import annotations
from core.registry import register
from core.schemas import ParsedDoc, Chunk
from core.chunking.base import BaseChunker, split_by_tokens
from core.tokenizer import count_tokens


@register("chunking", "recursive")
class RecursiveChunker(BaseChunker):
    """Splits on paragraph breaks first, then sentences, then words -- only
    recursing into a smaller separator when a piece is still too big.
    Safe general-purpose default; overlap protects against boundary splits."""
    _SEPARATORS = ["\n\n", ". ", " "]

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        out, n = [], 0
        for block in doc.text_blocks:
            for piece in self._split(block.text, self._SEPARATORS):
                if piece.strip():
                    n += 1
                    out.append(self._mk(doc, piece, n, page=block.page))
        return out

    def _split(self, text: str, seps: list[str]) -> list[str]:
        if count_tokens(text) <= self.config.chunk_size:
            return [text] if text.strip() else []
        if not seps:
            return split_by_tokens(text, self.config.chunk_size, self.config.overlap)

        sep, rest = seps[0], seps[1:]
        pieces, buf = [], ""
        for part in text.split(sep):
            candidate = (buf + sep + part) if buf else part
            if count_tokens(candidate) > self.config.chunk_size and buf:
                pieces.append(buf)
                buf = part
            else:
                buf = candidate
        if buf:
            pieces.append(buf)

        final = []
        for p in pieces:
            final.extend(self._split(p, rest) if count_tokens(p) > self.config.chunk_size else [p])
        return final
