"""
RESE Phase IV: Architectural Synthesis

Phase IV focuses on:
- Δ₁: Architecture Assembly
- Δ₂: Predictive Model Generation
- Δ₃: ACI Reduction Validation

Main Components:
    - architecture_assembler: Assemble solution architectures
        * Architecture: Main architecture container
        * ComponentInterface: Component definition
        * AssemblyStrategy: Assembly strategies
    - assembly_validator: Validate architectures
        * AssemblyValidator: Validation class
    - predictive_model_generator: Generate predictive models
        * PredictiveModelGenerator: Main generator
    - aci_reduction_validator: Validate ACI reduction
        * Delta3Validator: Δ₃ validator
        * ValidationResult: Validation result
    - statistical_tests: Statistical testing utilities
    - independence_checker: Check independence assumptions
    - phase_transition: Handle phase transitions

Usage:
    from phase4.architecture_assembler import Architecture
    from phase4.assembly_validator import AssemblyValidator
    from phase4.predictive_model_generator import PredictiveModelGenerator
    from phase4.aci_reduction_validator import Delta3Validator
"""

__version__ = "1.0.0"

__all__ = [
    "architecture_assembler",
    "assembly_validator",
    "predictive_model_generator",
    "aci_reduction_validator",
    "statistical_tests",
    "independence_checker",
    "phase_transition",
]
