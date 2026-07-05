"""
Evaluation-driven development, RAG++ (W&B) style: every stage's output gets
scored immediately, not just the final answer. This module wraps the actual
`ragas` PyPI package (v0.4.x API, using its native-client `llm_factory` --
no langchain glue needed) rather than reimplementing the metrics, per your
choice for exact library fidelity.

Two tiers, matching the platform's reference-free / with-reference toggle:
  - reference_free_report(): Faithfulness, AnswerRelevancy, ContextPrecisionWithoutReference
  - reference_based_report(): + ContextPrecisionWithReference, ContextRecall, AnswerCorrectness, SemanticSimilarity

Recommendations are RULE-BASED, not another LLM call -- scoring already
costs LLM tokens; explaining the score in plain English from thresholds is
free and deterministic (this is the token-efficiency principle applied to
the eval layer itself, not just generation).
"""
from __future__ import annotations
import sys
import types
from dataclasses import dataclass, field


def _shim_missing_langchain_community_vertexai() -> None:
    """ragas 0.4.x unconditionally imports langchain_community.chat_models.
    vertexai.ChatVertexAI at module load time, purely to list it in an
    isinstance() check for n-completion support -- but langchain-community
    has since dropped that submodule entirely (it's being sunset in favor
    of standalone integration packages). We never use Vertex AI as a
    provider (get_judge below only wires up claude/openai natively), so a
    dummy placeholder class is enough to satisfy the import without pulling
    in the real integration."""
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ModuleNotFoundError:
        shim = types.ModuleType("langchain_community.chat_models.vertexai")
        shim.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules["langchain_community.chat_models.vertexai"] = shim


_shim_missing_langchain_community_vertexai()

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness, AnswerRelevancy, ContextPrecisionWithoutReference,
    ContextPrecisionWithReference, ContextRecall, AnswerCorrectness, SemanticSimilarity,
)


@dataclass
class MetricScore:
    name: str
    value: float
    note: str = ""


@dataclass
class EvalReport:
    scores: list[MetricScore] = field(default_factory=list)
    recommendation: str = ""

    def as_dict(self) -> dict:
        return {"scores": {s.name: s.value for s in self.scores}, "recommendation": self.recommendation}


# ---- Provider-agnostic judge construction --------------------------------

def _native_client(provider: str, api_key: str):
    # ragas 0.4.x's Instructor adapter drives .score() through its async
    # ascore() path, so these must be async clients -- a sync client raises
    # "Cannot use agenerate() with a synchronous client" at score time.
    if provider == "claude":
        import anthropic
        return anthropic.AsyncAnthropic(api_key=api_key), "claude-sonnet-4-6"
    if provider == "openai":
        import openai
        return openai.AsyncOpenAI(api_key=api_key), "gpt-4o-mini"
    if provider == "groq":
        # Groq's API is OpenAI-compatible -- same client used elsewhere in this
        # codebase for Groq generation (core/llm_clients.py), so ragas's
        # instructor/openai adapter works against it with no extra dependency.
        import openai
        return openai.AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1"), "llama-3.1-8b-instant"
    if provider == "gemini":
        # Google's OpenAI-compatible endpoint sidesteps the litellm/google-genai
        # SDK adapter path (which needs extra deps and has known instructor
        # issues); same trick as Groq above.
        import openai
        return openai.AsyncOpenAI(
            api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        ), "gemini-2.5-flash"
    raise ValueError(f"Unknown judge provider '{provider}'")


def get_judge(provider: str, api_key: str, model: str | None = None):
    """Returns (llm, embeddings) ready to hand to ragas metric constructors."""
    client, default_model = _native_client(provider, api_key)
    # Only Claude uses its native anthropic client/adapter; openai, groq, and
    # gemini all go through OpenAI-compatible clients above.
    llm_provider = "anthropic" if provider == "claude" else "openai"
    llm = llm_factory(model or default_model, provider=llm_provider, client=client)
    # Embeddings for AnswerRelevancy/AnswerCorrectness/SemanticSimilarity: local
    # MiniLM by default so scoring doesn't burn extra API-embedding cost/tokens.
    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    return llm, embeddings


# ---- Rule-based recommendations ------------------------------------------

def _recommend(scores: dict[str, float]) -> str:
    notes = []
    if scores.get("faithfulness", 1) < 0.75:
        notes.append("faithfulness is low -- tighten the system prompt's grounding instruction or reduce re-ranking top-k so weaker chunks don't reach generation")
    if scores.get("answer_relevancy", 1) < 0.75:
        notes.append("answer relevancy is low -- the answer may be drifting from the question; check retrieval recall (Hybrid-RRF often helps)")
    if scores.get("context_precision_without_reference", scores.get("context_precision_with_reference", 1)) < 0.6:
        notes.append("context precision is low -- retrieval is pulling too many irrelevant chunks; try re-ranking or a smaller top-k")
    if scores.get("context_recall", 1) < 0.6:
        notes.append("context recall is low -- the right chunk likely isn't being retrieved at all; try Hybrid-RRF or a smaller chunk size")
    if scores.get("answer_correctness", 1) < 0.6:
        notes.append("answer correctness is low vs. the gold answer -- inspect whether generation is citing the right chunks or hallucinating detail")
    if not notes:
        return "All scores are healthy -- no changes recommended for this run."
    return " · ".join(notes)


# ---- Public entry points ---------------------------------------------------

def reference_free_report(question: str, contexts: list[str], answer: str, judge) -> EvalReport:
    llm, embeddings = judge
    faithfulness = Faithfulness(llm=llm)
    relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    precision = ContextPrecisionWithoutReference(llm=llm)

    r1 = faithfulness.score(user_input=question, response=answer, retrieved_contexts=contexts)
    r2 = relevancy.score(user_input=question, response=answer)
    r3 = precision.score(user_input=question, response=answer, retrieved_contexts=contexts)

    scores = {
        "faithfulness": float(r1.value), "answer_relevancy": float(r2.value),
        "context_precision_without_reference": float(r3.value),
    }
    return EvalReport(
        scores=[MetricScore(k, v) for k, v in scores.items()],
        recommendation=_recommend(scores),
    )


def reference_based_report(question: str, contexts: list[str], answer: str, reference: str, judge) -> EvalReport:
    """Adds gold-reference metrics on top of the reference-free set."""
    llm, embeddings = judge
    free = reference_free_report(question, contexts, answer, judge)

    precision_ref = ContextPrecisionWithReference(llm=llm)
    recall = ContextRecall(llm=llm)
    correctness = AnswerCorrectness(llm=llm, embeddings=embeddings)
    similarity = SemanticSimilarity(embeddings=embeddings)

    r1 = precision_ref.score(user_input=question, reference=reference, retrieved_contexts=contexts)
    r2 = recall.score(user_input=question, retrieved_contexts=contexts, reference=reference)
    r3 = correctness.score(user_input=question, response=answer, reference=reference)
    r4 = similarity.score(reference=reference, response=answer)

    scores = {s.name: s.value for s in free.scores}
    scores.update({
        "context_precision_with_reference": float(r1.value), "context_recall": float(r2.value),
        "answer_correctness": float(r3.value), "semantic_similarity": float(r4.value),
    })
    return EvalReport(
        scores=[MetricScore(k, v) for k, v in scores.items()],
        recommendation=_recommend(scores),
    )
