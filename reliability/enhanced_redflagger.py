"""
Enhanced Redflagger - Identifies potential issues in generated content.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from .config import ReliabilityConfig

logger = logging.getLogger(__name__)

class EnhancedRedflagger:
    """Advanced redflagging using regex and semantic heuristics."""

    def __init__(self, config: Optional[ReliabilityConfig] = None):
        self.config = config or ReliabilityConfig()
        self.rules = [
            (r"TODO|FIXME", "Unfinished code or placeholders", 0.8),
            (r"import\s+os|import\s+subprocess", "Potentially unsafe system imports", 0.5),
            (r"password|api_key|secret", "Potential hardcoded credentials", 0.9),
            (r"localhost|127\.0\.0\.1", "Hardcoded local addresses", 0.4),
            (r"eval\(|exec\(", "Unsafe code execution", 0.95),
        ]

    def scan(self, content: str) -> List[Dict[str, Any]]:
        """Scan content for red flags."""
        flags = []
        
        for pattern, description, severity in self.rules:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                flags.append({
                    "description": description,
                    "matched_text": match.group(0),
                    "position": match.start(),
                    "severity": severity
                })
                
        return sorted(flags, key=lambda x: x["severity"], reverse=True)

    def assess_reliability(self, content: str) -> float:
        """Calculate reliability score from 0.0 to 1.0."""
        flags = self.scan(content)
        if not flags:
            return 1.0
            
        # Weighted penalty based on severity
        penalty = sum(f["severity"] for f in flags) / len(self.rules)
        return max(0.0, 1.0 - penalty)
