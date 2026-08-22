"""
Knowledge Graph Query subpackage.

Exposes the parser/validator/normalizer, fluent QueryBuilder, optimizer,
planner, backends, execution engine, result cache, statistics collector and
multi-query-language translators.

Copyright 2026 OpenEvolve
Licensed under the Apache License, Version 2.0 (the "License").
"""

from .parser import (
    QueryParser, QueryValidator, QueryNormalizer, QueryBuilder,
    CypherAst, ast_to_cypher, QueryParseError, QueryValidationError,
)
from .optimizer import (
    QueryOptimizer, ExecutionPlanner, ExecutionPlan, PlanOptimizer, BackendSelector,
)
from .cache import ResultCache, StatisticsCollector
from .backend import (
    GraphBackend, InMemoryNetworkXBackend, Neo4jBackend, MemgraphBackend,
    SparqlBackend, create_backend,
)
from .languages import (
    MultiLanguageTranslator, GremlinTranslator, SparqlTranslator,
    GraphQLTranslator, CustomDSLTranslator,
)
from .executor import (
    QueryExecutionEngine, CypherCompiler, GraphTraverser, EngineConfig,
)

__all__ = [
    "QueryParser", "QueryValidator", "QueryNormalizer", "QueryBuilder",
    "CypherAst", "ast_to_cypher", "QueryParseError", "QueryValidationError",
    "QueryOptimizer", "ExecutionPlanner", "ExecutionPlan", "PlanOptimizer",
    "BackendSelector", "ResultCache", "StatisticsCollector", "GraphBackend",
    "InMemoryNetworkXBackend", "Neo4jBackend", "MemgraphBackend",
    "SparqlBackend", "create_backend", "MultiLanguageTranslator",
    "GremlinTranslator", "SparqlTranslator", "GraphQLTranslator",
    "CustomDSLTranslator", "QueryExecutionEngine", "CypherCompiler",
    "GraphTraverser", "EngineConfig",
]
