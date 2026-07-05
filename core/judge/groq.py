from __future__ import annotations
from core.registry import register
from core.judge.base import _ProviderJudge


@register("judge", "llama-3.1-8b-instant")
class GroqJudge(_ProviderJudge):
    """Fast/cheap ReAct sufficiency judge -- Groq's inference speed makes
    this a good fit for an every-iteration check."""
    _provider = "groq"
    _default_model = "llama-3.1-8b-instant"
