"""
Input Sanitizer Module

Provides input sanitization for OpenEvolve.

Author: OpenEvolve Team
Date: 2026-02-06
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InputSanitizerConfig:
    """Configuration for input sanitizer"""
    allowed_chars: str = "a-zA-Z0-9 "
    max_length: int = 10000


class InputSanitizer:
    """Input Sanitizer class"""
    
    def __init__(self, config: Optional[InputSanitizerConfig] = None):
        self.config = config or InputSanitizerConfig()
        logger.info("Input Sanitizer initialized")
    
    def sanitize(self, input_str: str) -> str:
        """Sanitize input"""
        return re.sub(f"[^{self.config.allowed_chars}]", "", input_str)
    
    def validate(self, input_str: str) -> bool:
        """Validate input"""
        return len(input_str) <= self.config.max_length


def create_input_sanitizer(config: Optional[InputSanitizerConfig] = None) -> InputSanitizer:
    """Factory function to create input sanitizer instance"""
    return InputSanitizer(config)
