"""
MDAP Reliability Adapter - Connects MDAP to the reliability system.
"""

import logging
from typing import Dict, List, Any, Optional
from reliability.unified_bridge import UnifiedReliabilityBridge

logger = logging.getLogger(__name__)

class MDAPReliabilityAdapter:
    """Provides reliability assessment for MDAP workflow outcomes."""

    def __init__(self):
        self.bridge = UnifiedReliabilityBridge()

    async def verify_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify an MDAP step outcome."""
        content = str(step_data.get("output", ""))
        return await self.bridge.validate_content(content)

    async def verify_run(self, run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify complete MDAP run results."""
        all_passed = True
        scores = []
        
        for step in run_results:
            result = await self.verify_step(step)
            all_passed = all_passed and result["passed"]
            scores.append(result["reliability_score"])
            
        return {
            "overall_passed": all_passed,
            "average_reliability": sum(scores) / len(scores) if scores else 1.0,
            "results": run_results
        }
