"""
RESE Phase II: Isomorphic Resonance

Phase II focuses on:
- Ψ₁: Constraint Inversion (via Ψ₃)
- Ψ₂: Ontology Mapping
- Ψ₃: SAT-based Constraint Solving
- I_mech: Isomorphic Mechanism Validator

Main Components:
    - imech: Isomorphic Mechanism validation
        * Core modules: FDG, causality, domain, scoring, result
        * Algorithms: VF2, Weisfeiler-Lehman, subgraph, intervention
        * Transfer: Mapper, validator, repair
        * Lean4: Proof generation
    - psi3: SAT-based constraint solving
        * Core: Constraint, Expression, ConstraintInverter
        * Solvers: SAT wrapper
        * Algorithms: Dependency analysis, preprocessing
    - ontology_mapper: Map between different ontologies
    - ontology_components:
        * Lexical matcher
        * Semantic matcher
        * Graph embedder
        * KG validator
    - ontology_imech_integration: Integrate ontology mapping with I_mech

Usage:
    from phase2.imech import IMechValidator, Domain
    from phase2.psi3 import Constraint, ConstraintInverter
    from phase2.ontology_mapper import OntologyMapper
"""

__version__ = "1.0.0"

__all__ = [
    # I_mech
    "imech",
    # Ψ₃
    "psi3",
    # Ontology
    "ontology_mapper",
    "ontology_components",
    "ontology_imech_integration",
]
