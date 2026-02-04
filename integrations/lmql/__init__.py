"""LMQL Integration Package for OpenEvolve.

Declarative LLM Query Language integration providing SQL-like syntax
for LLM interactions with constraint programming.

This package is the SSOT (Single Source of Truth) for LMQL integration.

Example:
    >>> from integrations.lmql import LMQLAdapter, LMQLQueryBuilder
    >>> from integrations.lmql import Constraint, ConstraintType
    
    >>> adapter = LMQLAdapter(model="gpt-4")
    >>> result = adapter.query(
    ...     "Extract entities: {text}",
    ...     context={"text": "Apple was founded by Steve Jobs."},
    ...     constraints=[Constraint(ConstraintType.LENGTH, max=100)]
    ... )
    
    >>> builder = LMQLQueryBuilder()
    >>> query = builder.with_prompt("Extract: {text}").with_variable("entities", "list").build()

Modules:
    adapter: Core LMQL adapter for query execution
    query_templates: Pre-built LMQL query templates for KG operations
    constraint_engine: Constraint evaluation engine

Author: OpenEvolve
Version: 1.0.0
License: MIT
"""

# Adapter exports
from integrations.lmql.adapter import (
    LMQLAdapter,
    LMQLQueryBuilder,
    QueryOptimizer,
    Constraint,
    ConstraintType,
    LMQLResult,
    DialogResult,
    ExtractionResult,
    EntityResult,
    get_default_adapter,
    get_default_optimizer,
    reset_defaults,
)

# Query templates exports
from integrations.lmql.query_templates import (
    # Template strings
    ENTITY_EXTRACTION_LMQL,
    ENTITY_EXTRACTION_WITH_POSITION_LMQL,
    ENTITY_LINKING_LMQL,
    ENTITY_DISAMBIGUATION_LMQL,
    RELATION_EXTRACTION_LMQL,
    RELATION_EXTRACTION_WITH_TEMPORAL_LMQL,
    TRIPLET_EXTRACTION_LMQL,
    SCHEMA_INFERENCE_LMQL,
    SCHEMA_INFERENCE_FROM_QUERIES_LMQL,
    SCHEMA_VALIDATION_LMQL,
    CYPHER_GENERATION_LMQL,
    CYPHER_GENERATION_FOR_PATH_LMQL,
    CYPHER_GENERATION_FOR_TEMPORAL_LMQL,
    CYPHER_GENERATION_FOR_AGGREGATION_LMQL,
    MULTI_HOP_REASONING_LMQL,
    CHAIN_OF_THOUGHT_LMQL,
    PATH_REASONING_LMQL,
    MULTI_TURN_DIALOG_LMQL,
    CONSTRAINED_DIALOG_LMQL,
    INFORMATION_GATHERING_LMQL,
    FACT_VERIFICATION_LMQL,
    CONSISTENCY_CHECK_LMQL,
    # Classes
    TemplateCategory,
    QueryTemplate,
    TemplateRegistry,
    # Functions
    get_default_registry,
    reset_registry,
    get_template,
    render_template,
    list_templates,
)

# Constraint engine exports
from integrations.lmql.constraint_engine import (
    ConstraintOperator,
    ConstraintType as EngineConstraintType,
    ConstraintEvaluationResult,
    BatchEvaluationResult,
    Constraint,
    LengthConstraint,
    TypeConstraint,
    RegexConstraint,
    RangeConstraint,
    EnumConstraint,
    CustomConstraint,
    StopAtConstraint,
    CompositeConstraint,
    ConstraintEvaluator,
    ConstraintParser,
    ConstraintOptimizer,
    get_default_evaluator,
    get_default_parser,
    get_default_optimizer,
    reset_defaults as reset_constraint_defaults,
)

# Version
__version__ = "1.0.0"

# Consolidated exports
__all__ = [
    # Version
    "__version__",
    
    # Adapter
    "LMQLAdapter",
    "LMQLQueryBuilder",
    "QueryOptimizer",
    "Constraint",
    "ConstraintType",
    "LMQLResult",
    "DialogResult",
    "ExtractionResult",
    "EntityResult",
    "get_default_adapter",
    "get_default_optimizer",
    "reset_defaults",
    
    # Query Templates
    "ENTITY_EXTRACTION_LMQL",
    "ENTITY_EXTRACTION_WITH_POSITION_LMQL",
    "ENTITY_LINKING_LMQL",
    "ENTITY_DISAMBIGUATION_LMQL",
    "RELATION_EXTRACTION_LMQL",
    "RELATION_EXTRACTION_WITH_TEMPORAL_LMQL",
    "TRIPLET_EXTRACTION_LMQL",
    "SCHEMA_INFERENCE_LMQL",
    "SCHEMA_INFERENCE_FROM_QUERIES_LMQL",
    "SCHEMA_VALIDATION_LMQL",
    "CYPHER_GENERATION_LMQL",
    "CYPHER_GENERATION_FOR_PATH_LMQL",
    "CYPHER_GENERATION_FOR_TEMPORAL_LMQL",
    "CYPHER_GENERATION_FOR_AGGREGATION_LMQL",
    "MULTI_HOP_REASONING_LMQL",
    "CHAIN_OF_THOUGHT_LMQL",
    "PATH_REASONING_LMQL",
    "MULTI_TURN_DIALOG_LMQL",
    "CONSTRAINED_DIALOG_LMQL",
    "INFORMATION_GATHERING_LMQL",
    "FACT_VERIFICATION_LMQL",
    "CONSISTENCY_CHECK_LMQL",
    "TemplateCategory",
    "QueryTemplate",
    "TemplateRegistry",
    "get_default_registry",
    "reset_registry",
    "get_template",
    "render_template",
    "list_templates",
    
    # Constraint Engine
    "ConstraintOperator",
    "EngineConstraintType",
    "ConstraintEvaluationResult",
    "BatchEvaluationResult",
    "LengthConstraint",
    "TypeConstraint",
    "RegexConstraint",
    "RangeConstraint",
    "EnumConstraint",
    "CustomConstraint",
    "StopAtConstraint",
    "CompositeConstraint",
    "ConstraintEvaluator",
    "ConstraintParser",
    "ConstraintOptimizer",
    "get_default_evaluator",
    "get_default_parser",
    "get_default_optimizer",
    "reset_constraint_defaults",
]
