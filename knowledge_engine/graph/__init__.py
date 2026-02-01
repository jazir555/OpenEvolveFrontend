"""
Knowledge Graph Module for OpenEvolve

Core Neo4j-based knowledge graph with Pydantic validation,
connection pooling, and hybrid query capabilities.

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Core schema imports (with fallback)
try:
    from .schema import (
        NodeType, EdgeType, PropertyType,
        NodeSchema, EdgeSchema, GraphSchema
    )
    _schema_available = True
except ImportError:
    _schema_available = False
    NodeType = None
    EdgeType = None
    PropertyType = None
    NodeSchema = None
    EdgeSchema = None
    GraphSchema = None

# Core model imports (with fallback)
try:
    from .models import (
        KnowledgeNode, KnowledgeEdge, KnowledgeGraph,
        NodeProperties, EdgeProperties
    )
    _models_available = True
except ImportError:
    _models_available = False
    KnowledgeNode = None
    KnowledgeEdge = None
    KnowledgeGraph = None
    NodeProperties = None
    EdgeProperties = None

# Other core components (with fallback)
try:
    from .crud import GraphCRUD
    _crud_available = True
except ImportError:
    _crud_available = False
    GraphCRUD = None

try:
    from .connection import ConnectionPool, RetryPolicy
    _connection_available = True
except ImportError:
    _connection_available = False
    ConnectionPool = None
    RetryPolicy = None

try:
    from .cypher_builder import CypherQueryBuilder
    _cypher_available = True
except ImportError:
    _cypher_available = False
    CypherQueryBuilder = None

# Import unified KG components (for integration hub)
try:
    from .unified_kg import (
        UnifiedKnowledgeGraph,
        UnifiedTriple,
        GraphStatistics
    )
    _unified_kg_available = True
except ImportError:
    _unified_kg_available = False
    UnifiedKnowledgeGraph = None
    UnifiedTriple = None
    GraphStatistics = None

try:
    from .kg_models import (
        KnowledgeGraphModels,
        KnowledgeStatement,
        EntityProfile,
        GraphPattern,
        RelationshipDefinition,
        EntityReference,
        KnowledgeSource,
        ConfidenceLevel
    )
    _kg_models_available = True
except ImportError:
    _kg_models_available = False
    KnowledgeGraphModels = None
    KnowledgeStatement = None
    EntityProfile = None
    GraphPattern = None
    RelationshipDefinition = None
    EntityReference = None
    KnowledgeSource = None
    ConfidenceLevel = None

__all__ = []

# Add schema exports if available
if _schema_available:
    __all__.extend([
        'NodeType', 'EdgeType', 'PropertyType',
        'NodeSchema', 'EdgeSchema', 'GraphSchema',
    ])

# Add model exports if available
if _models_available:
    __all__.extend([
        'KnowledgeNode', 'KnowledgeEdge', 'KnowledgeGraph',
        'NodeProperties', 'EdgeProperties',
    ])

# Add CRUD exports if available
if _crud_available:
    __all__.extend(['GraphCRUD'])

# Add connection exports if available
if _connection_available:
    __all__.extend(['ConnectionPool', 'RetryPolicy'])

# Add cypher exports if available
if _cypher_available:
    __all__.extend(['CypherQueryBuilder'])

# Add unified KG exports if available
if _unified_kg_available:
    __all__.extend([
        'UnifiedKnowledgeGraph',
        'UnifiedTriple',
        'GraphStatistics'
    ])

if _kg_models_available:
    __all__.extend([
        'KnowledgeGraphModels',
        'KnowledgeStatement',
        'EntityProfile',
        'GraphPattern',
        'RelationshipDefinition',
        'EntityReference',
        'KnowledgeSource',
        'ConfidenceLevel'
    ])
