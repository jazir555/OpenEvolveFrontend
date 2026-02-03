"""
Query Optimizer for Hybrid Search

Optimizes queries for efficient execution on both vector and graph stores.

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

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum, auto
import re

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries"""
    ENTITY_LOOKUP = auto()  # Find specific entity
    RELATIONSHIP = auto()   # Find relationships
    SEMANTIC = auto()       # Semantic similarity
    HYBRID = auto()         # Combination
    TRAVERSAL = auto()      # Graph traversal


@dataclass
class OptimizedQuery:
    """An optimized query with execution plan"""
    original_query: str
    query_type: QueryType
    cypher_query: Optional[str] = None
    vector_query: Optional[str] = None
    filters: Dict[str, Any] = None
    use_vector: bool = True
    use_graph: bool = True
    estimated_cost: float = 0.0
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}


class QueryOptimizer:
    """Optimizes queries for hybrid search"""
    
    def __init__(self):
        self.query_patterns = {
            QueryType.ENTITY_LOOKUP: [
                r'^who is\s+',
                r'^what is\s+',
                r'^find\s+',
                r'^lookup\s+',
            ],
            QueryType.RELATIONSHIP: [
                r'how (is|are).+related',
                r'connection between',
                r'relationship',
                r'links? (to|between)',
            ],
            QueryType.SEMANTIC: [
                r'similar to',
                r'like\s+',
                r'related to',
                r'about\s+',
            ],
            QueryType.TRAVERSAL: [
                r'neighbors? of',
                r'connected to',
                r'path (from|to)',
                r'near\s+',
            ]
        }
    
    def analyze(self, query: str) -> QueryType:
        """Analyze query to determine type"""
        query_lower = query.lower().strip()
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type
        
        # Default to hybrid
        return QueryType.HYBRID
    
    def optimize(self, query: str, context: Optional[Dict] = None) -> OptimizedQuery:
        """Optimize a query for execution"""
        query_type = self.analyze(query)
        
        # Determine best execution strategy
        use_vector = True
        use_graph = True
        
        if query_type == QueryType.ENTITY_LOOKUP:
            # Entity lookups work best with graph
            use_vector = False
            cypher = self._build_entity_lookup(query)
        elif query_type == QueryType.RELATIONSHIP:
            cypher = self._build_relationship_query(query)
        elif query_type == QueryType.SEMANTIC:
            # Semantic queries work best with vectors
            use_graph = False
            cypher = None
        elif query_type == QueryType.TRAVERSAL:
            use_vector = False
            cypher = self._build_traversal_query(query)
        else:
            cypher = self._build_generic_query(query)
        
        # Extract filters
        filters = self._extract_filters(query)
        
        # Estimate cost
        estimated_cost = self._estimate_cost(query_type, use_vector, use_graph)
        
        return OptimizedQuery(
            original_query=query,
            query_type=query_type,
            cypher_query=cypher,
            vector_query=query if use_vector else None,
            filters=filters,
            use_vector=use_vector,
            use_graph=use_graph,
            estimated_cost=estimated_cost
        )
    
    def _build_entity_lookup(self, query: str) -> str:
        """Build Cypher for entity lookup"""
        # Extract entity name from query
        patterns = [
            r'who is\s+(\w+(?:\s+\w+)*)',
            r'what is\s+(\w+(?:\s+\w+)*)',
            r'find\s+(\w+(?:\s+\w+)*)',
        ]
        
        entity_name = None
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entity_name = match.group(1)
                break
        
        if entity_name:
            return f"""
            MATCH (n)
            WHERE n.name = "{entity_name}" OR n.name CONTAINS "{entity_name}"
            RETURN n
            LIMIT 10
            """
        
        return None
    
    def _build_relationship_query(self, query: str) -> str:
        """Build Cypher for relationship query"""
        # Extract entity names
        words = re.findall(r'\b[A-Z][a-z]+\b', query)
        if len(words) >= 2:
            return f"""
            MATCH path = shortestPath(
                (a {{name: "{words[0]}"}})-[*]-(b {{name: "{words[1]}"}})
            )
            RETURN path
            LIMIT 1
            """
        return None
    
    def _build_traversal_query(self, query: str) -> str:
        """Build Cypher for traversal query"""
        patterns = [
            r'neighbors? of\s+(\w+(?:\s+\w+)*)',
            r'connected to\s+(\w+(?:\s+\w+)*)',
        ]
        
        entity_name = None
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entity_name = match.group(1)
                break
        
        if entity_name:
            return f"""
            MATCH (n {{name: "{entity_name}"}})-[*1..3]-(neighbor)
            RETURN DISTINCT neighbor
            LIMIT 20
            """
        
        return None
    
    def _build_generic_query(self, query: str) -> str:
        """Build generic Cypher query"""
        # Use full-text search if available
        return f"""
        CALL db.index.fulltext.queryNodes('entitySearch', '{query}') 
        YIELD node, score
        RETURN node, score
        LIMIT 10
        """
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract filters from query"""
        filters = {}
        
        # Check for node type filters
        type_patterns = {
            "person": ["person", "people", "who"],
            "organization": ["company", "org", "organization", "team"],
            "technology": ["technology", "software", "tool", "framework"],
            "concept": ["concept", "idea", "theory"],
        }
        
        query_lower = query.lower()
        for node_type, keywords in type_patterns.items():
            if any(kw in query_lower for kw in keywords):
                filters["node_type"] = node_type
                break
        
        return filters
    
    def _estimate_cost(
        self,
        query_type: QueryType,
        use_vector: bool,
        use_graph: bool
    ) -> float:
        """Estimate query execution cost"""
        base_costs = {
            QueryType.ENTITY_LOOKUP: 1.0,
            QueryType.RELATIONSHIP: 2.0,
            QueryType.SEMANTIC: 1.5,
            QueryType.TRAVERSAL: 3.0,
            QueryType.HYBRID: 2.5,
        }
        
        cost = base_costs.get(query_type, 2.0)
        
        if use_vector:
            cost += 1.0
        if use_graph:
            cost += 1.5
        
        return cost
    
    def rewrite_for_vector(self, query: str) -> str:
        """Rewrite query for better vector search"""
        # Remove question words
        query = re.sub(r'^(who|what|where|when|why|how)\s+(is|are|was|were|did|do|does)\s+', '', query, flags=re.IGNORECASE)
        
        # Remove stop words
        stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = query.split()
        words = [w for w in words if w.lower() not in stop_words]
        
        return ' '.join(words)
    
    def rewrite_for_graph(self, query: str) -> str:
        """Rewrite query for better graph search"""
        # Extract key entities
        # Capitalized words are likely entity names
        entities = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', query)
        
        if entities:
            return ' '.join(entities)
        
        return query
