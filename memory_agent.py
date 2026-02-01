"""
Memory Agent - Meta-cognitive analysis and disambiguation support.

Provides lightweight heuristics for:
- Failure history analysis (disambiguation constraints)
- Confusion detection in repeated dialogues
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any
import re


@dataclass
class MemoryAgent:
    """Heuristic memory agent for failure analysis and confusion detection."""

    def analyze_failure_history(self, failure_history: Iterable[str]) -> List[str]:
        """
        Analyze failure history and return disambiguation constraints.

        Args:
            failure_history: Iterable of failure narratives or critiques.

        Returns:
            List of disambiguation constraints (short, actionable strings).
        """
        merged = "\n".join(str(item) for item in failure_history if item)
        if not merged.strip():
            return ["Clarify ambiguous requirements and define success criteria explicitly."]

        # Identify repeated terms as likely ambiguity sources.
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", merged.lower())
        stop = {
            "the", "and", "for", "with", "that", "this", "from", "into", "over",
            "when", "then", "were", "was", "are", "is", "as", "but", "not",
            "should", "must", "could", "would", "about", "have", "has", "had",
            "failed", "failure", "error", "issue", "issues", "problem", "problems",
            "missing", "unclear", "ambiguous", "inconsistent", "constraint",
        }
        filtered = [t for t in tokens if t not in stop]
        common = [t for t, _ in Counter(filtered).most_common(6)]

        constraints = []
        if common:
            constraints.append(
                "Disambiguate these recurring terms: " + ", ".join(common[:4]) + "."
            )
        if "dependency" in merged.lower() or "interface" in merged.lower():
            constraints.append("Define explicit interfaces and dependency contracts.")
        if "performance" in merged.lower() or "latency" in merged.lower():
            constraints.append("Specify performance targets and acceptable trade-offs.")
        if "security" in merged.lower() or "auth" in merged.lower():
            constraints.append("Define security invariants and threat boundaries.")
        if "test" in merged.lower() or "coverage" in merged.lower():
            constraints.append("Add explicit test/validation criteria for the failing component.")

        if not constraints:
            constraints.append("Clarify edge cases and define success criteria explicitly.")

        return constraints[:5]

    def identify_confusion(self, messages: Iterable[str]) -> str:
        """
        Identify a likely point of confusion in repeated dialogue.

        Args:
            messages: Iterable of message texts.

        Returns:
            Short clarification statement.
        """
        merged = " ".join(str(m) for m in messages if m)
        if not merged.strip():
            return "Clarify the agent's role and expected output format."

        # Look for repeated phrases (2-4 word n-grams).
        words = re.findall(r"[a-zA-Z0-9_-]+", merged.lower())
        if len(words) < 6:
            return "Clarify the agent's role and expected output format."

        ngrams: Counter[str] = Counter()
        for n in (2, 3, 4):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i : i + n])
                ngrams[ngram] += 1

        common = [ng for ng, cnt in ngrams.most_common(5) if cnt >= 2]
        if common:
            return f"Resolve repeated ambiguity around: '{common[0]}' by clarifying scope and constraints."

        return "Clarify the agent's role, memory limits, and output requirements."


__all__ = ["MemoryAgent"]
