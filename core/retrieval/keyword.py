from __future__ import annotations
import re
from core.registry import register
from core.retrieval.base import BaseRetriever, RetrievedItem
from core.vectorstore.chroma_store import ChromaStore

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@register("retrieval", "keyword")
class KeywordRetriever(BaseRetriever):
    """BM25: ranks by weighted term overlap (TF-IDF with saturation and
    length normalization), not meaning. Good for exact terms, codes, and
    acronyms embeddings tend to blur together (e.g. 'BLEU score')."""

    def __init__(self, store: ChromaStore, top_k: int = 10):
        super().__init__(top_k)
        from rank_bm25 import BM25Okapi
        self._ids, docs = store.all_text_documents()
        self._docs = docs
        self._bm25 = BM25Okapi([_tokenize(d) for d in docs]) if docs else None

    def retrieve(self, query: str) -> list[RetrievedItem]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._ids, self._docs, scores), key=lambda x: -x[2])[:self.top_k]
        return [RetrievedItem(id=id_, text=doc, score=float(score), kind="text")
                for id_, doc, score in ranked]
