"""Model routing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class IntelligentModelRouter:
    models: Dict[str, str] = None

    def __post_init__(self):
        if self.models is None:
            self.models = {
                "fast": "gpt-4o-mini",
                "balanced": "gpt-4o",
                "powerful": "gpt-4-turbo",
                "local": "llama-3-70b",
            }

    def route(self, task_complexity: float, cost_constraint: str, latency_requirement: float) -> str:
        if latency_requirement < 100:
            return self.models["fast"]
        if task_complexity > 0.8:
            return self.models["powerful"]
        if cost_constraint == "low":
            return self.models["local"]
        return self.models["balanced"]

