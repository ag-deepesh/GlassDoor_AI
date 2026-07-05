"""
A tiny plugin registry.

Every pluggable method (a parser, a chunker, a retriever, ...) registers
itself under a stage name and a method name. The UI dropdown for a stage
is just `Registry.options("chunking")` -- so "add an option" is "write a
function/class and decorate it", and "remove an option" is "delete that
file" or call `Registry.unregister(...)`. No central list to keep in sync.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Callable

_REGISTRY: dict[str, dict[str, Callable]] = defaultdict(dict)


def register(stage: str, name: str):
    """Decorator: @register('chunking', 'recursive')"""
    def deco(fn_or_cls):
        _REGISTRY[stage][name] = fn_or_cls
        return fn_or_cls
    return deco


def get(stage: str, name: str) -> Callable:
    try:
        return _REGISTRY[stage][name]
    except KeyError:
        available = ", ".join(_REGISTRY[stage]) or "(none registered)"
        raise KeyError(f"No method '{name}' registered for stage '{stage}'. Available: {available}")


def options(stage: str) -> list[str]:
    """What the UI dropdown for this stage should show."""
    return sorted(_REGISTRY[stage].keys())


def unregister(stage: str, name: str) -> None:
    _REGISTRY[stage].pop(name, None)


def all_stages() -> dict[str, list[str]]:
    return {stage: sorted(methods) for stage, methods in _REGISTRY.items()}
