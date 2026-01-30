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

from .schema import (
    NodeType, EdgeType, PropertyType,
    NodeSchema, EdgeSchema, GraphSchema
)
from .models import (
    KnowledgeNode, KnowledgeEdge, KnowledgeGraph,
    NodeProperties, EdgeProperties
)
from .crud import GraphCRUD
from .connection import ConnectionPool, RetryPolicy
from .cypher_builder import CypherQueryBuilder

__all__ = [
    # Schema
    'NodeType', 'EdgeType', 'PropertyType',
    'NodeSchema', 'EdgeSchema', 'GraphSchema',
    # Models
    'KnowledgeNode', 'KnowledgeEdge', 'KnowledgeGraph',
    'NodeProperties', 'EdgeProperties',
    # CRUD
    'GraphCRUD',
    # Connection
    'ConnectionPool', 'RetryPolicy',
    # Query Builder
    'CypherQueryBuilder'
]
