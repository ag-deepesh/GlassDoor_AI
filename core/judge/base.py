"""
The ReAct loop's sufficiency judge -- a small, structured LLM call deciding
whether reranked context can answer the query, and if not, how to narrow
the query for another retrieval pass. Registered under the 'judge' stage
the same way every other pluggable method is (core/registry.py's decorator
pattern), so the UI's judge-model dropdown is just Registry.options('judge')
-- "add/remove an option" is "add/remove a file" here too.

Distinct from core/evaluation/ragas_eval.py's judge: that one scores a
finished answer with RAGAS metrics; this one runs mid-pipeline, before
generation, to decide whether to keep retrieving.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import re

from core.llm_clients import get_client
from core.retrieval.base import RetrievedItem


@dataclass
class JudgeVerdict:
    sufficient: bool
    rewritten_query: str | None
    reasoning: str
    tokens: int
    cost_usd: float


_SYSTEM_PROMPT = (
    "You judge whether retrieved context is sufficient to answer a question in a RAG "
    "pipeline. Respond with ONLY a JSON object: "
    '{"sufficient": true|false, "rewritten_query": "<a narrower/rewritten query, or null if sufficient>", '
    '"reasoning": "<one sentence>"}. No prose outside the JSON.'
)


class BaseJudge(ABC):
    @abstractmethod
    def judge(self, query: str, contexts: list[RetrievedItem]) -> JudgeVerdict: ...


class _ProviderJudge(BaseJudge):
    _provider: str = ""
    _default_model: str = ""

    def __init__(self, api_key: str, model: str | None = None):
        self._client = get_client(self._provider, api_key)
        self._model = model or self._default_model

    def judge(self, query: str, contexts: list[RetrievedItem]) -> JudgeVerdict:
        context_text = "\n".join(f"[#{c.id}] {c.text}" for c in contexts)
        user_msg = f"Question: {query}\n\nRetrieved context:\n{context_text or '(none)'}"
        resp = self._client.chat(_SYSTEM_PROMPT, user_msg, model=self._model, max_tokens=300)
        parsed = _parse_verdict(resp.text)
        return JudgeVerdict(sufficient=bool(parsed["sufficient"]), rewritten_query=parsed.get("rewritten_query"),
                             reasoning=parsed.get("reasoning", ""), tokens=resp.input_tokens + resp.output_tokens,
                             cost_usd=resp.cost_usd)


def _parse_verdict(text: str) -> dict:
    """Judge models sometimes wrap JSON in prose/code fences despite
    instructions -- pull out the first {...} block rather than failing the
    whole iteration over formatting. Falls back to 'sufficient' so a
    malformed response stops the loop instead of burning further iterations."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    fallback = {"sufficient": True, "rewritten_query": None, "reasoning": f"Unparseable judge response: {text[:200]}"}
    if not match:
        return fallback
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return fallback
