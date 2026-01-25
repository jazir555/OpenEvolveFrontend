"""
RESE Phase I: Epistemic Audit

Phase I focuses on:
- Φ₁: Symbolic Constraint Engine
- Φ₁.₅: Tacit Assumption Mining
- Φ₂: Cognitive Bias Detection
- Φ₃: Contradiction Resolution

Main Components:
    - phi15_interfaces: Φ₁.₅ interfaces and data structures
    - tacit_assumption_miner: Mine tacit assumptions from constraints
    - cognitive_biases: Detect and analyze cognitive biases
    - phi2_integration: Integrate Φ₁ (SCE) with Φ₂ (Cognitive Biases)
    - validate_phi15: Validation tools for Φ₁.₅
    - failure_database: Database of known failure modes
"""

__version__ = "1.0.0"

# Don't import submodules directly to avoid circular dependencies
# Users should import explicitly:
# from phase1.cognitive_biases import CognitiveBiasDetector
# from phase1.phi2_integration import SCEPhi2Integrator

__all__ = [
    "phi15_interfaces",
    "tacit_assumption_miner",
    "cognitive_biases",
    "phi2_integration",
    "validate_phi15",
    "failure_database",
]
