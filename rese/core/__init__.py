"""
RESE Core Components

This module provides core functionality for the RESE system including:
- Symbolic Constraint Engine (SCE)
- DITO Optimizer
- Constraint Optimizer
- Lean 4 Bridge
- LLTL Handoff
- Logic to Loss Translation
- Stage Integrations
"""

from .symbolic_constraint_engine import SymbolicConstraintEngine, Constraint, ConstraintType
from .dito_optimizer import DITOOptimizer, DITOConfig
from .dito_graphs import (
    ConstraintDependencyGraph,
    HierarchicalAbstractionGraph,
    PredicateVariableGraph,
    GraphTraversals
)
from .constraint_optimizer import ConstraintOptimizer, ResolutionStrategy, OptimizationResult
from .constraint_lean4_bridge import Lean4Bridge, Lean4Theorem
from .constraint_lltl_handoff import LLTLHandoff, LLTLSpecification, LLTLTemplate, HandoffPackage
from .logic_to_loss_translation import (
    LogicToLossTranslator,
    LossFunction,
    LossAggregationMethod,
    FuzzyLogicType,
    create_lltl_from_sce,
)
from .constraint_stage1_integration import Stage1Integrator, PromptAnalysis
from .stage5_integration import Stage5Integration, GeneratorValidator, FeedbackMode, FeedbackStrategy

__all__ = [
    # Symbolic Constraint Engine
    "SymbolicConstraintEngine",
    "Constraint",
    "ConstraintType",
    # DITO Optimizer
    "DITOOptimizer",
    "DITOConfig",
    # DITO Graphs
    "ConstraintDependencyGraph",
    "HierarchicalAbstractionGraph",
    "PredicateVariableGraph",
    "GraphTraversals",
    # Constraint Optimizer
    "ConstraintOptimizer",
    "ResolutionStrategy",
    "OptimizationResult",
    # Lean 4 Bridge
    "Lean4Bridge",
    "Lean4Theorem",
    # LLTL Handoff
    "LLTLHandoff",
    "LLTLSpecification",
    "LLTLTemplate",
    "HandoffPackage",
    # Logic to Loss Translation
    "LogicToLossTranslator",
    "LossFunction",
    "LossAggregationMethod",
    "FuzzyLogicType",
    "create_lltl_from_sce",
    # Stage Integrations
    "Stage1Integrator",
    "PromptAnalysis",
    "Stage5Integration",
    "GeneratorValidator",
    "FeedbackMode",
    "FeedbackStrategy",
]
