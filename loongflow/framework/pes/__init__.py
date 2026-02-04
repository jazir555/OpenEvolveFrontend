"""PES framework core classes."""

from __future__ import annotations

from typing import Any, Dict, Optional

from loongflow.agents.general_agent import GeneralEvolveAgent
from .register import register_runner, Worker


class PESAgent:
    """Thin wrapper around GeneralEvolveAgent for PES workflows."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._agent = GeneralEvolveAgent(config=config or {})

    async def run(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        return await self._agent.run(problem_data)


class Finalizer:
    """Base finalizer that can post-process results."""

    def finalize(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return result


class LoongFlowFinalizer(Finalizer):
    """Default finalizer for LoongFlow PES results."""

    def finalize(self, result: Dict[str, Any]) -> Dict[str, Any]:
        result["finalized"] = True
        return result


__all__ = ["PESAgent", "Finalizer", "LoongFlowFinalizer", "Worker", "register_runner"]
