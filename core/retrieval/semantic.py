from __future__ import annotations
from core.registry import register
from core.retrieval.base import BaseRetriever, RetrievedItem
from core.embedding.base import BaseEmbedder
from core.vectorstore.chroma_store import ChromaStore


@register("retrieval", "semantic")
class SemanticRetriever(BaseRetriever):
    """Ranks by embedding cosine similarity: sim(a,b) = (a . b) / (||a|| ||b||).
    Best for conceptual/paraphrased queries where the exact wording won't
    match the source text."""

    def __init__(self, store: ChromaStore, embedder: BaseEmbedder, top_k: int = 10):
        super().__init__(top_k)
        self._store = store
        self._embedder = embedder

    def retrieve(self, query: str) -> list[RetrievedItem]:
        q_emb = self._embedder.embed([query])[0]
        res = self._store.query_text(q_emb, top_k=self.top_k)
        items = []
        for i, (id_, doc, dist, meta) in enumerate(zip(
                res["ids"][0], res["documents"][0], res["distances"][0], res["metadatas"][0])):
            # Chroma returns a distance; convert to a similarity-like score for consistent reporting.
            items.append(RetrievedItem(id=id_, text=doc, score=1 - dist, kind="text", meta=meta))
        return items
