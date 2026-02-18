"""
Unified Reliability Bridge - Coordinates all reliability checks.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from .config import ReliabilityConfig
from .enhanced_redflagger import EnhancedRedflagger
from .guardrails_adapter import GuardrailsAdapter
from .lmql_adapter import LMQLAdapter

logger = logging.getLogger(__name__)

class UnifiedReliabilityBridge:
    """Central entry point for reliability and validation."""

    def __init__(self, config: Optional[ReliabilityConfig] = None):
        self.config = config or ReliabilityConfig()
        self.redflagger = EnhancedRedflagger(self.config)
        self.guardrails = GuardrailsAdapter(self.config)
        self.lmql = LMQLAdapter(self.config)

    async def validate_content(
        self, 
        content: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run all reliability checks on content.
        
        1. Scan for red flags.
        2. Validate with Guardrails.
        3. Calculate reliability score.
        """
        # Redflagger scan
        flags = self.redflagger.scan(content)
        reliability_score = self.redflagger.assess_reliability(content)
        
        # Guardrails validation
        validation = self.guardrails.validate(content)
        
        return {
            "reliability_score": reliability_score,
            "red_flags": flags,
            "validation": validation,
            "passed": reliability_score > self.config.redflagger_threshold and validation["valid"]
        }

    def get_status(self) -> Dict[str, bool]:
        """Get status of all reliability components."""
        return {
            "redflagger": True,
            "guardrails": self.guardrails._initialized,
            "lmql": self.lmql._initialized
        }
