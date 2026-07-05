"""
Cross-stage root-cause diagnosis for a single run. When generation's RAGAS
scores dip below a healthy threshold, `_recommend()` in ragas_eval.py already
gives a one-line, single-metric note -- this module goes one step further by
cross-referencing the OTHER stages' already-collected heuristic reports
(chunking's size variance, retrieval's score spread, reranking's image mix)
to guess WHICH stage is the likely culprit. Fully rule-based and reuses data
already computed this run -- zero extra LLM cost.

deep_dive() is the optional, on-demand escalation: only called when the
user clicks "Deep-dive" in the UI, since it's a real extra LLM call.
"""
from __future__ import annotations

_THRESHOLDS = {
    "faithfulness": 0.75, "answer_relevancy": 0.75,
    "context_precision_without_reference": 0.6, "context_precision_with_reference": 0.6,
    "context_recall": 0.6, "answer_correctness": 0.6,
}


def _low_scores(scores: dict) -> dict:
    return {k: v for k, v in scores.items() if v < _THRESHOLDS.get(k, 0.6)}


def diagnose(stage_reports: list[dict]) -> dict | None:
    """stage_reports: [{"stage": ..., "method": ..., "eval_reference_free": {...}}, ...]
    for one completed run, in stage order. Returns None when generation's
    scores are all healthy -- nothing to diagnose."""
    by_stage = {r["stage"]: r for r in stage_reports if r.get("stage")}
    gen = by_stage.get("generation")
    if not gen or not (gen.get("eval_reference_free") or {}).get("scores"):
        return None
    scores = gen["eval_reference_free"]["scores"]
    low = _low_scores(scores)
    if not low:
        return None

    chunking = (by_stage.get("chunking") or {}).get("eval_reference_free") or {}
    retrieval = (by_stage.get("retrieval") or {}).get("eval_reference_free") or {}
    reranking = (by_stage.get("reranking") or {}).get("eval_reference_free") or {}

    culprits, signals = [], []
    context_metrics = {"context_recall", "context_precision_without_reference", "context_precision_with_reference"}
    if low.keys() & context_metrics:
        std_tokens = chunking.get("scores", {}).get("std_tokens")
        if std_tokens and std_tokens > 150:
            culprits.append("chunking -- high chunk-size variance may be splitting or diluting facts")
            signals.append(f"chunking std_tokens={std_tokens}")
        top_score = retrieval.get("scores", {}).get("top_score")
        if top_score is not None and top_score < 0.5:
            culprits.append("retrieval -- even the top-ranked chunk scored low; the answer may be under-represented in the KB")
            signals.append(f"retrieval top_score={top_score}")
        n_images = reranking.get("scores", {}).get("n_images", 0)
        if n_images and "faithfulness" in low:
            culprits.append("reranking -- images may be crowding out a more relevant text chunk; check keep_top or the vision-rerank toggle")
            signals.append(f"reranking n_images={n_images}")
    if "faithfulness" in low and not culprits:
        culprits.append("generation -- the model may be drifting from context; tighten the system prompt's grounding instruction")
    if "answer_relevancy" in low:
        culprits.append("retrieval recall -- the right chunk may not have been retrieved at all; try Hybrid-RRF")

    summary = ("Low score(s): " + ", ".join(f"{k}={v:.2f}" for k, v in low.items()) + ". Likely culprit(s): "
               + ("; ".join(culprits) if culprits else "unclear from stage heuristics alone -- try the deep-dive."))
    return {"low_scores": low, "culprits": culprits, "signals": signals, "summary": summary}


def deep_dive(query: str, stage_reports: list[dict], provider: str, api_key: str) -> str:
    """On-demand narrative diagnosis: sends a COMPACT summary of this run's
    per-stage scores/recommendations (not raw traces or chunk text) to the
    judge LLM, for when the rule-based pass above isn't conclusive enough."""
    from core.llm_clients import get_client
    lines = [f"Query: {query}"]
    for r in stage_reports:
        rf = r.get("eval_reference_free") or {}
        if not rf:
            continue
        lines.append(f"- {r.get('stage')} ({r.get('method')}): scores={rf.get('scores', {})} "
                      f"-- {rf.get('recommendation', '')}")
    system = ("You are a RAG pipeline diagnostician. Given this run's per-stage scores and "
              "rule-based recommendations, identify the most likely root cause of any low score(s) "
              "and give 2-3 concrete, prioritized config changes. Be specific and concise (under 150 words).")
    client = get_client(provider, api_key)
    resp = client.chat(system, "\n".join(lines), max_tokens=300)
    return resp.text
