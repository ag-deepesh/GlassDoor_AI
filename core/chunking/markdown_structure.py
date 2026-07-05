from __future__ import annotations
from core.registry import register
from core.schemas import ParsedDoc, Chunk
from core.chunking.base import BaseChunker
from core.chunking.recursive import RecursiveChunker
from core.tokenizer import count_tokens


@register("chunking", "markdown_structure")
class MarkdownStructureChunker(BaseChunker):
    """Groups text blocks by heading hierarchy (section_path, set by the
    Markdown parser) instead of a fixed token window -- one chunk per
    section, so retrieval returns whole, coherent sections. A section still
    over chunk_size falls back to recursive splitting so nothing exceeds the
    limit. Best for docs where headings carry real structure (specs, wikis,
    READMEs) rather than loose prose."""

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        sections: dict[str, list[str]] = {}
        order: list[str] = []
        for block in doc.text_blocks:
            key = block.section_path or "(no heading)"
            if key not in sections:
                sections[key] = []
                order.append(key)
            sections[key].append(block.text)

        out, n = [], 0
        fallback = RecursiveChunker(self.config)
        for key in order:
            text = "\n\n".join(sections[key])
            if count_tokens(text) <= self.config.chunk_size:
                n += 1
                out.append(self._mk(doc, text, n, section_path=key))
            else:
                for piece in fallback._split(text, RecursiveChunker._SEPARATORS):
                    if piece.strip():
                        n += 1
                        out.append(self._mk(doc, piece, n, section_path=key))
        return out
