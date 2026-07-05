from __future__ import annotations
from core.registry import register
from core.judge.base import _ProviderJudge


@register("judge", "claude-sonnet")
class ClaudeJudge(_ProviderJudge):
    _provider = "claude"
    _default_model = "claude-sonnet-4-6"
