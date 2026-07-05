from __future__ import annotations
from core.registry import register
from core.judge.base import _ProviderJudge


@register("judge", "gemini-2.5-flash")
class GeminiFlashJudge(_ProviderJudge):
    """Default ReAct judge, per the platform's requirement -- cheap and
    fast enough to run every iteration without dominating the run's cost."""
    _provider = "gemini"
    _default_model = "gemini-2.5-flash"
