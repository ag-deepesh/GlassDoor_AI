"""
The ReAct refinement loop -- optional, sits after Reranking, before
Generation (core/pipeline.py only calls this when config.react_enabled).

Each iteration: ask the judge model whether the current reranked context is
sufficient to answer the query. If not, retrieve + rerank again with the
judge's rewritten/narrowed query. Capped at react_max_iterations (1-5,
default 3). The web toggle governs Retrieval AND this loop identically --
if it's on, web is re-queried on EVERY iteration (not just the first pass),
so an aggressive refinement loop's cost is visible in the trace, not hidden
behind a cheaper-but-silent shortcut.

When OFF, core/pipeline.py never calls this module at all -- zero added
cost or latency, exactly Milestone 2's behavior.
"""
from __future__ import annotations
import time

from core.registry import get
from core.schemas import StageReport, TraceEvent


def run_react_loop(pipeline, query: str) -> StageReport:
    t0 = time.time()
    cfg = pipeline.config
    judge_cls = get("judge", cfg.react_judge_method)
    judge = judge_cls(cfg.api_keys[judge_cls._provider])

    current_query = query
    iterations: list[dict] = []
    total_tokens, total_cost = 0, 0.0
    initial_top_score = max((i.score for i in pipeline.reranked), default=0.0)
    sufficient = False

    for i in range(cfg.react_max_iterations):
        verdict = judge.judge(current_query, pipeline.reranked)
        total_tokens += verdict.tokens
        total_cost += verdict.cost_usd
        record = {"iteration": i + 1, "query": current_query, "sufficient": verdict.sufficient,
                  "reasoning": verdict.reasoning, "judge_cost_usd": round(verdict.cost_usd, 6)}
        iterations.append(record)

        sufficient = verdict.sufficient
        if sufficient or not verdict.rewritten_query:
            break

        current_query = verdict.rewritten_query
        retrieval_report = pipeline.run_retrieval(current_query)  # re-runs web search too, if cfg.web_enabled
        pipeline.run_reranking(current_query)
        record["web_cost_usd"] = round(retrieval_report.trace.cost_usd, 6)
        total_cost += retrieval_report.trace.cost_usd

    final_top_score = max((i.score for i in pipeline.reranked), default=0.0)
    n_iterations = len(iterations)

    notes = []
    if not sufficient:
        notes.append(f"stopped after {n_iterations} iteration(s) without reaching sufficiency -- "
                      f"consider raising max iterations or a broader retrieval method")
    score_delta = round(final_top_score - initial_top_score, 3)
    if score_delta > 0:
        notes.append(f"top score improved {initial_top_score:.2f} -> {final_top_score:.2f}")
    recommendation = " · ".join(notes) if notes else "Context was sufficient on the first pass -- no refinement needed."

    eval_report = {
        "scores": {"iterations": n_iterations, "sufficient": sufficient,
                   "top_score_before": round(initial_top_score, 3), "top_score_after": round(final_top_score, 3)},
        "recommendation": recommendation,
    }
    trace = TraceEvent(stage="react", method=cfg.react_judge_method,
                        input_summary=f"query: {query[:60]}",
                        output_summary=f"{n_iterations} iteration(s), sufficient={sufficient}",
                        latency_ms=round((time.time() - t0) * 1000, 1), tokens=total_tokens,
                        cost_usd=round(total_cost, 6), extra={"iterations": iterations})
    return StageReport(stage="react", method=cfg.react_judge_method,
                        output_preview=f"{n_iterations} iteration(s) · sufficient={sufficient} · "
                                        f"score {initial_top_score:.2f}->{final_top_score:.2f}",
                        trace=trace, eval_reference_free=eval_report)
