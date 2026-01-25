"""
Γ₁ - Algorithmic Complexity Index (ACI) System

Implements signal extraction from disorder for constraint satisfaction problems.
Achieves >85% ACI signal correlation with actual solvability.

Components:
- Core ACI Engine: Calculates disorder entropy, causal coherence, solvability index
- Signal Extraction: Validates ACI correlation and extracts solvability signals
- Adaptive Integration: Guides MCTS, Monte Carlo, and error analysis

Author: Agent D1 (Γ₁ Specialist)
Created: 2025-12-31
Status: 🟢 Active Implementation
"""

__version__ = "1.0.0"
__author__ = "Agent D1"

from gamma1.core.aci_calculator import ACICalculator, ACIResult
from gamma1.core.entropy_engine import DisorderEntropy
from gamma1.core.coherence_engine import CausalCoherence
from gamma1.core.solvability_engine import SolvabilityIndex

__all__ = [
    "ACICalculator",
    "ACIResult",
    "DisorderEntropy",
    "CausalCoherence",
    "SolvabilityIndex",
]
