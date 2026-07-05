from __future__ import annotations
from core.registry import register
from core.judge.base import _ProviderJudge


@register("judge", "gpt-4o-mini")
class OpenAIJudge(_ProviderJudge):
    _provider = "openai"
    _default_model = "gpt-4o-mini"
