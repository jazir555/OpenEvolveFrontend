"""
Input Sanitizer Module

Real input sanitization and threat detection for OpenEvolve.
Dependency-light: standard library only.

Public names preserved: InputSanitizerConfig, InputSanitizer, create_input_sanitizer.
"""
from __future__ import annotations


import html
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
    strip_html: bool = True
    allow_html_tags: tuple = ()


# Common injection / cross-site-scripting signatures
_SQLI_PATTERNS = [
    r"(?i)(\b(union|select|insert|update|delete|drop|alter|truncate)\b.*\bfrom\b)",
    r"(?i)(\bor\b\s+1\s*=\s*1)",
    r"(?i)(--|#|\/\*.*\*\/)",
    r"(?i)(\bshoot\b|\bxp_cmdshell\b)",
]
_XSS_PATTERNS = [
    r"(?i)<\s*script\b",
    r"(?i)<\s*iframe\b",
    r"(?i)on(error|load|click|mouseover)\s*=",
    r"(?i)javascript\s*:",
    r"(?i)<\s*img\b[^>]*\bsrc\s*=\s*['\"]?\s*javascript",
]
_CMDI_PATTERNS = [
    r"(?i)[;&|]\s*(rm|cat|wget|curl|nc|bash|sh|powershell|cmd)\b",
    r"(?i)`[^`]+`",
    r"(?i)\$\([^)]+\)",
]


class InputSanitizer:
    """Input Sanitizer class"""

    def __init__(self, config: Optional[InputSanitizerConfig] = None):
        self.config = config or InputSanitizerConfig()
        self._disallow_re = re.compile(f"[^{self.config.allowed_chars}]")
        self._sqli = [re.compile(p) for p in _SQLI_PATTERNS]
        self._xss = [re.compile(p) for p in _XSS_PATTERNS]
        self._cmdi = [re.compile(p) for p in _CMDI_PATTERNS]
        logger.info("Input Sanitizer initialized")

    def sanitize(self, input_str: str) -> str:
        """Sanitize input: strip disallowed characters and optional HTML."""
        if not isinstance(input_str, str):
            input_str = str(input_str)
        if self.config.strip_html and not self.config.allow_html_tags:
            input_str = self._strip_html(input_str)
        cleaned = self._disallow_re.sub("", input_str)
        if len(cleaned) > self.config.max_length:
            cleaned = cleaned[: self.config.max_length]
        return cleaned

    def _strip_html(self, text: str) -> str:
        # Remove tags entirely, then neutralize any leftover entities.
        no_tags = re.sub(r"<[^>]+>", "", text)
        return html.unescape(no_tags)

    def validate(self, input_str: str) -> bool:
        """Validate input length and absence of injected control characters."""
        if not isinstance(input_str, str):
            return False
        if len(input_str) > self.config.max_length:
            return False
        if self._disallow_re.search(input_str):
            return False
        return True

    def detect_threats(self, input_str: str) -> List[Dict[str, str]]:
        """Scan input for SQLi / XSS / command-injection signatures."""
        threats: List[Dict[str, str]] = []
        for name, patterns in (("sqli", self._sqli), ("xss", self._xss), ("cmdi", self._cmdi)):
            for pat in patterns:
                m = pat.search(input_str)
                if m:
                    threats.append({"type": name, "match": m.group(0)[:80]})
                    break
        return threats


def create_input_sanitizer(config: Optional[InputSanitizerConfig] = None) -> InputSanitizer:
    """Factory function to create input sanitizer instance"""
    return InputSanitizer(config)
