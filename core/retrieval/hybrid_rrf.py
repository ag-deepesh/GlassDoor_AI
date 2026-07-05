from __future__ import annotations
from core.registry import register
from core.retrieval.base import BaseRetriever, RetrievedItem
from core.retrieval.semantic import SemanticRetriever
from core.retrieval.keyword import KeywordRetriever

RRF_K = 60


def rrf_fuse(rankings: list[list[RetrievedItem]], k: int = RRF_K, top_k: int = 10) -> list[RetrievedItem]:
    """Reciprocal Rank Fusion across any number of independently-ranked
    result lists:  RRF(d) = sum over each ranking of  1 / (k + rank_i(d)).
    RRF only needs each list's RANK order, not its raw score -- which
    sidesteps the problem that e.g. cosine similarity and BM25 scores live
    on totally different scales and can't be averaged directly. Shared by
    HybridRRFRetriever (semantic+keyword) and the KB+web blend step in
    Pipeline.run_retrieval (KB results+Tavily results) -- same fusion math
    either way, just a different pair of input rankings."""
    fused: dict[str, float] = {}
    by_id: dict[str, RetrievedItem] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            fused[item.id] = fused.get(item.id, 0.0) + 1.0 / (k + rank + 1)
            by_id[item.id] = item

    top_ids = sorted(fused, key=lambda id_: -fused[id_])[:top_k]
    return [RetrievedItem(id=id_, text=by_id[id_].text, score=fused[id_], kind=by_id[id_].kind, meta=by_id[id_].meta)
            for id_ in top_ids]


@register("retrieval", "hybrid-rrf")
class HybridRRFRetriever(BaseRetriever):
    """Combines Semantic and Keyword rankings via Reciprocal Rank Fusion --
    see rrf_fuse() above for the mechanism. Usually the best default: real
    queries mix exact terms and paraphrased concepts."""

    RRF_K = RRF_K

    def __init__(self, semantic: SemanticRetriever, keyword: KeywordRetriever, top_k: int = 10):
        super().__init__(top_k)
        self._semantic = semantic
        self._keyword = keyword

    def retrieve(self, query: str) -> list[RetrievedItem]:
        # Over-fetch from each side so fusion has enough candidates to re-rank from.
        fetch_k = max(self.top_k * 2, 20)
        self._semantic.top_k = fetch_k
        self._keyword.top_k = fetch_k
        sem_results = self._semantic.retrieve(query)
        kw_results = self._keyword.retrieve(query)
        return rrf_fuse([sem_results, kw_results], k=self.RRF_K, top_k=self.top_k)
