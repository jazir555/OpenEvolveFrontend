"""Math-specific LoongFlow PES agent."""

from __future__ import annotations

from typing import Any, Dict

from .general_agent import GeneralEvolveAgent


class MathEvolveAgent(GeneralEvolveAgent):
    """Specialized agent for math problems."""

    async def run(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        problem_data = dict(problem_data)
        problem_data.setdefault("domain", "math")
        return await super().run(problem_data)
