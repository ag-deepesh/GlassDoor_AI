from __future__ import annotations
import re
from typing import Callable
from core.registry import register
from core.schemas import ParsedDoc, Chunk
from core.chunking.base import BaseChunker
from core.tokenizer import count_tokens

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _default_embed_fn():
    """Lazy-loaded local MiniLM -- only imported/downloaded if semantic
    chunking is actually selected, so it never costs anything for other
    chunking strategies. Requires internet on first run (model download)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return lambda texts: model.encode(texts, normalize_embeddings=True).tolist()


@register("chunking", "semantic")
class SemanticChunker(BaseChunker):
    """Cuts at meaning boundaries instead of fixed token counts: sentences
    are merged into a running chunk as long as each new sentence stays
    similar (cosine >= similarity_threshold) to the chunk's running average
    embedding; a drop in similarity starts a new chunk. Best for long,
    structurally loose prose (essays, transcripts) where headings don't
    exist to guide chunking. chunk_size still caps maximum chunk length as
    a safety net."""

    def __init__(self, config=None, embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
                 similarity_threshold: float = 0.6):
        super().__init__(config)
        self._embed_fn = embed_fn  # injected for testing; else lazy-loaded on first use
        self.similarity_threshold = similarity_threshold

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        embed_fn = self._embed_fn or _default_embed_fn()
        out, n = [], 0

        for block in doc.text_blocks:
            sentences = [s.strip() for s in _SENTENCE_RE.split(block.text) if s.strip()]
            if not sentences:
                continue
            if len(sentences) == 1:
                n += 1
                out.append(self._mk(doc, sentences[0], n, page=block.page))
                continue

            embeddings = embed_fn(sentences)
            current_sents = [sentences[0]]
            current_emb_sum = list(embeddings[0])

            for sent, emb in zip(sentences[1:], embeddings[1:]):
                running_avg = [x / len(current_sents) for x in current_emb_sum]
                sim = _cosine(running_avg, emb)
                candidate_text = " ".join(current_sents + [sent])

                if sim < self.similarity_threshold or count_tokens(candidate_text) > self.config.chunk_size:
                    n += 1
                    out.append(self._mk(doc, " ".join(current_sents), n, page=block.page))
                    current_sents = [sent]
                    current_emb_sum = list(emb)
                else:
                    current_sents.append(sent)
                    current_emb_sum = [x + y for x, y in zip(current_emb_sum, emb)]

            if current_sents:
                n += 1
                out.append(self._mk(doc, " ".join(current_sents), n, page=block.page))
        return out
