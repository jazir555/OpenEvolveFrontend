"""
ROMA Reliability Adapter - Connects ROMA to the reliability system.
"""

import logging
from typing import Dict, List, Any, Optional
from reliability.unified_bridge import UnifiedReliabilityBridge

logger = logging.getLogger(__name__)

class ROMAReliabilityAdapter:
    """Provides reliability assessment for ROMA solver outcomes."""

    def __init__(self):
        self.bridge = UnifiedReliabilityBridge()

    async def verify_solution(self, solution: str) -> Dict[str, Any]:
        """Verify a ROMA solver solution."""
        return await self.bridge.validate_content(solution)

    async def verify_sub_problem(self, sub_problem_result: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a ROMA sub-problem result."""
        content = sub_problem_result.get("solution", "")
        return await self.bridge.validate_content(content)
