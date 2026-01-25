"""
Γ₁ Core ACI Engine

Implements the three engines for ACI calculation:
1. Disorder Entropy (H): Multi-scale entropy measurement
2. Causal Coherence (C): Graph structure + information flow
3. Solvability Index (S): Phase transition distance
"""

from gamma1.core.aci_calculator import ACICalculator, ACIResult
from gamma1.core.entropy_engine import DisorderEntropy, EntropyComponents
from gamma1.core.coherence_engine import CausalCoherence, CoherenceComponents
from gamma1.core.solvability_engine import SolvabilityIndex, SolvabilityComponents
from gamma1.core.csp_models import CSPInstance, Variable, Constraint

__all__ = [
    "ACICalculator",
    "ACIResult",
    "DisorderEntropy",
    "EntropyComponents",
    "CausalCoherence",
    "CoherenceComponents",
    "SolvabilityIndex",
    "SolvabilityComponents",
    "CSPInstance",
    "Variable",
    "Constraint",
]
