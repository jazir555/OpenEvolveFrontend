"""Model adapters for LoongFlow."""

from __future__ import annotations

from typing import Any, Dict


class LiteLLMModel:
    """Minimal model wrapper used by LoongFlow integrations."""

    def __init__(self, name: str = "gpt-4", **kwargs: Any) -> None:
        self.name = name
        self.config: Dict[str, Any] = dict(kwargs)

    def generate(self, prompt: str) -> str:
        return f"[{self.name}] {prompt}"


__all__ = ["LiteLLMModel"]
