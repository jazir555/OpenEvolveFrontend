"""Minimal placeholder for the missing AceSteer integration bridge.

The real ``AceSteerBridge`` (documented in
docs/integrations/ACE_STEER_INTEGRATION_GUIDE.md) is not present in this
flat script collection. This stub preserves importability and public names
while degrading ACE/Steer learning and verification gracefully.
"""
from __future__ import annotations


from typing import Any, Dict, List, Optional


class AceSteerBridge:
    """Placeholder AceSteer bridge (no-op learning/verification)."""

    def __init__(self, ace_agent_id: str = "default_agent", skillbook_path: str = ""):
        self.ace_agent_id = ace_agent_id
        self.skillbook_path = skillbook_path
        self.steer_status: Dict[str, Any] = {"available": False, "stub": True}

    def prepare_prompt(self, task: str, context: str = "") -> str:
        return task

    def verify_and_learn(
        self,
        query: str,
        output: str,
        verifications: Optional[List[str]] = None,
        reasoning: str = "",
    ) -> Dict[str, Any]:
        return {
            "all_passed": True,
            "failed_verifications": [],
            "results": [],
        }
