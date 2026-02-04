"""ML-specific LoongFlow PES agent."""

from __future__ import annotations

from typing import Any, Dict

from ..general_agent import GeneralEvolveAgent


class MLEvolveAgent(GeneralEvolveAgent):
    """Specialized agent for machine learning problems."""

    async def run(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        problem_data = dict(problem_data)
        problem_data.setdefault("domain", "ml")
        return await super().run(problem_data)


__all__ = ["MLEvolveAgent"]
