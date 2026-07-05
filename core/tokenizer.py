"""
Token counting without a network dependency.

tiktoken's encoding files are fetched from a blob-storage URL on first use --
not bundled, and not always reachable (e.g. restricted networks, offline
demos). For chunk-sizing purposes we don't need exact token IDs, just a
consistent, fast estimate, so we use the standard ~4-chars-per-token rule of
thumb for English text. Swap this for tiktoken later if exact counts matter
(e.g. for hard API context-window limits).
"""
from __future__ import annotations
import re

_WORD_RE = re.compile(r"\S+")


def count_tokens(text: str) -> int:
    """Approximate token count. Good enough for chunk-size targeting;
    not a substitute for a real tokenizer when hitting an API's hard limit."""
    if not text:
        return 0
    chars_estimate = max(1, len(text) // 4)
    words_estimate = len(_WORD_RE.findall(text))
    # Blend both signals -- pure char/4 overcounts for short words, pure word
    # count undercounts for long/technical words -- average is more stable.
    return round((chars_estimate + words_estimate) / 2)
