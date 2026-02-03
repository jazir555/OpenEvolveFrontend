"""
Causal preference synthesis utilities.

Generates synthetic preference pairs to accelerate reward model convergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


@dataclass
class SyntheticPreferencePair:
    previous_solution: str
    current_solution: str
    preference_bit: int
    improvement_delta: float
    metadata: Dict[str, Any]


class CausalPreferenceSynthesizer:
    """Lightweight causal synthesizer using heuristic drivers."""

    def discover_drivers(self, records: Iterable[Any]) -> List[str]:
        drivers = []
        for record in records:
            if getattr(record, "improvement_delta", 0.0) > 0.5:
                drivers.append("improvement_delta")
            if getattr(record, "preference_bit", 0) == 1:
                drivers.append("current_solution_quality")
        return list(set(drivers))

    def synthesize(self, records: Iterable[Any], max_pairs: int = 50) -> List[SyntheticPreferencePair]:
        pairs: List[SyntheticPreferencePair] = []
        drivers = self.discover_drivers(records)
        for record in records:
            if len(pairs) >= max_pairs:
                break
            delta = getattr(record, "improvement_delta", 0.0)
            if delta <= 0:
                continue
            pairs.append(
                SyntheticPreferencePair(
                    previous_solution=getattr(record, "previous_solution", ""),
                    current_solution=getattr(record, "current_solution", ""),
                    preference_bit=getattr(record, "preference_bit", 1),
                    improvement_delta=min(1.0, delta * 1.1),
                    metadata={"drivers": drivers}
                )
            )
        return pairs


def generate_synthetic_preference_pairs(records: Iterable[Any], max_pairs: int = 50) -> List[SyntheticPreferencePair]:
    """Convenience function to generate synthetic preference pairs."""
    return CausalPreferenceSynthesizer().synthesize(records, max_pairs=max_pairs)


__all__ = ["SyntheticPreferencePair", "CausalPreferenceSynthesizer", "generate_synthetic_preference_pairs"]
