"""
Symbolic Analyzer

Extracts shared interface symbols across components to build entanglement signals.
Uses tree-sitter when available, with a regex fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class SymbolicAnalysisResult:
    symbols: Set[str]


class SymbolicAnalyzer:
    """Extract symbols from code or text to detect shared interfaces."""

    def __init__(self):
        self._tree_sitter_available = False
        self._parser = None
        self._language = None

        try:
            from tree_sitter import Parser  # type: ignore
            self._parser = Parser()
            self._tree_sitter_available = True
        except (ImportError, RuntimeError) as exc:
            logger.debug("Tree-sitter not available: %s", exc)
            self._tree_sitter_available = False

    def analyze(self, text: str) -> SymbolicAnalysisResult:
        symbols = self._extract_symbols(text)
        return SymbolicAnalysisResult(symbols=symbols)

    def _extract_symbols(self, text: str) -> Set[str]:
        if not text:
            return set()

        # Regex fallback covers most interface-level symbols
        symbols = set()

        # Function/class definitions
        for match in re.findall(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", text):
            symbols.add(match)

        # Import statements
        for match in re.findall(r"\bimport\s+([A-Za-z_][A-Za-z0-9_\.]+)", text):
            symbols.add(match.split(".")[0])

        for match in re.findall(r"\bfrom\s+([A-Za-z_][A-Za-z0-9_\.]+)\s+import\s+([A-Za-z_][A-Za-z0-9_\*,\s]*)", text):
            module = match[0].split(".")[0]
            symbols.add(module)
            imported = [i.strip() for i in match[1].split(",") if i.strip() and i.strip() != "*"]
            symbols.update(imported)

        # Common identifier patterns
        for match in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\b", text):
            if match.isupper():
                continue
            symbols.add(match)

        # Trim overly generic tokens
        blacklist = {"the", "and", "for", "with", "from", "return", "true", "false"}
        symbols = {s for s in symbols if s.lower() not in blacklist}

        return symbols

