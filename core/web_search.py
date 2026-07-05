"""
Tavily web search -- the platform's one external retrieval source. Wrapped
the same way every other third-party dependency is: a thin adapter
returning the pipeline's own RetrievedItem shape (kind="web"), so
retrieval/reranking/generation never need to know a result came from the
web instead of the KB.

A missing/invalid/rate-limited key raises TavilySearchError -- callers
(Pipeline.run_retrieval, core/react_loop.py) catch this and downgrade to
KB-only results rather than let it become a hard pipeline failure, per the
platform's error-handling rule.
"""
from __future__ import annotations
from core.retrieval.base import RetrievedItem

# Approximate USD per call for Tavily's "basic" search depth -- same spirit
# as llm_clients.PRICING: rough and easy to update, used only to show an
# estimated cost in traces.
COST_PER_SEARCH_USD = 0.001


class TavilySearchError(RuntimeError):
    """Raised when a Tavily call can't be completed (missing/invalid key,
    rate limit, network) -- always non-fatal to the caller."""


class TavilyClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise TavilySearchError("no Tavily API key provided")
        try:
            import tavily
            self._client = tavily.TavilyClient(api_key=api_key)
        except Exception as e:
            raise TavilySearchError(f"{type(e).__name__}: {e}") from e

    def search(self, query: str, max_results: int = 5) -> list[RetrievedItem]:
        try:
            resp = self._client.search(query, max_results=max_results)
        except Exception as e:
            raise TavilySearchError(f"{type(e).__name__}: {e}") from e

        items = []
        for i, r in enumerate(resp.get("results", [])):
            items.append(RetrievedItem(
                id=f"web_{i}_{abs(hash(r.get('url', '')))}",
                text=(r.get("content") or "")[:2000],
                score=float(r.get("score", 1.0 / (i + 1))),
                kind="web",
                meta={"url": r.get("url", ""), "title": r.get("title", "")},
            ))
        return items
