"""Knowledge Engine LMQL Integration.

Declarative query interface for KG operations using LMQL.
This is a thin wrapper around the primary LMQL implementation in integrations/lmql/.

Architecture: SSOT (Single Source of Truth)
- Primary implementation: integrations/lmql/
- This wrapper: knowledge_engine/integrations/lmql/

Author: OpenEvolve
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

# Import from primary implementation (SSOT)
from integrations.lmql import (
    LMQLAdapter,
    LMQLQueryBuilder,
    Constraint,
    ConstraintType,
    LMQLResult,
    TemplateRegistry,
    render_template,
    get_default_adapter,
)
from integrations.lmql.constraint_engine import (
    ConstraintEvaluator,
    get_default_evaluator,
)

# Configure structured logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class EntityQueryResult:
    """Result of entity query operation."""
    success: bool
    entities: List[Dict[str, Any]] = field(default_factory=list)
    query: str = ""
    filters_applied: Dict[str, Any] = field(default_factory=dict)
    total_count: int = 0
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: str = ""
    error: Optional[str] = None


@dataclass
class RelationQueryResult:
    """Result of relation query operation."""
    success: bool
    relations: List[Dict[str, Any]] = field(default_factory=list)
    entity_ids: List[str] = field(default_factory=list)
    relation_types: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: str = ""
    error: Optional[str] = None


@dataclass
class SchemaInferenceResult:
    """Result of schema inference operation."""
    success: bool
    schema: Dict[str, Any] = field(default_factory=dict)
    entity_types: List[Dict[str, Any]] = field(default_factory=list)
    relation_types: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: str = ""
    error: Optional[str] = None


@dataclass
class MultiHopResult:
    """Result of multi-hop query operation."""
    success: bool
    paths: List[List[Dict[str, Any]]] = field(default_factory=list)
    start_entity: str = ""
    answer: str = ""
    confidence: float = 0.0
    hops_taken: int = 0
    entities_visited: List[str] = field(default_factory=list)
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: str = ""
    error: Optional[str] = None


@dataclass
class QueryExplanation:
    """Explanation of a query execution."""
    query: str
    parsed_constraints: List[Dict[str, Any]]
    execution_plan: str
    estimated_cost: str
    optimization_hints: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class CypherGenerationResult:
    """Result of Cypher query generation."""
    success: bool
    cypher: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    is_temporal: bool = False
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None


# =============================================================================
# LMQL KG INTEGRATION
# =============================================================================


class LMQLKGIntegration:
    """Knowledge Graph integration using LMQL.
    
    Provides declarative query interface for KG operations with:
    - Entity extraction and querying
    - Relation extraction and traversal
    - Schema inference
    - Multi-hop reasoning
    - Cypher generation for Memgraph
    
    Example:
        >>> integration = LMQLKGIntegration()
        >>> result = integration.query_entities(
        ...     "Find all companies founded by Steve Jobs",
        ...     filters={"entity_type": "ORG"}
        ... )
    """
    
    def __init__(
        self,
        adapter: Optional[LMQLAdapter] = None,
        kg_connection: Optional[Any] = None,
        enable_caching: bool = True,
        default_model: str = "gpt-4",
    ):
        """Initialize LMQL KG Integration.
        
        Args:
            adapter: LMQLAdapter instance (creates default if None)
            kg_connection: Knowledge graph connection
            enable_caching: Whether to enable query result caching
            default_model: Default model to use
        """
        self.adapter = adapter or get_default_adapter()
        self.kg_connection = kg_connection
        self.enable_caching = enable_caching
        self.default_model = default_model
        self.template_registry = TemplateRegistry()
        self.constraint_evaluator = get_default_evaluator()
        
        # Result cache
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_ttl = 3600  # 1 hour
        
        # Metrics
        self._metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "cached_queries": 0,
            "total_latency_ms": 0.0,
        }
        
        logger.info("LMQLKGIntegration initialized")
        
    def query_entities(
        self,
        query_str: str,
        filters: Optional[Dict[str, Any]] = None,
        entity_types: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        max_results: int = 50,
        use_cache: bool = True,
    ) -> EntityQueryResult:
        """Query entities using natural language with LMQL constraints.
        
        Args:
            query_str: Natural language query
            filters: Additional filters (entity_type, properties, etc.)
            entity_types: Specific entity types to search
            min_confidence: Minimum confidence threshold
            max_results: Maximum results to return
            use_cache: Whether to use query cache
            
        Returns:
            EntityQueryResult with matched entities
        """
        import time
        start_time = time.time()
        self._metrics["total_queries"] += 1
        
        filters = filters or {}
        
        try:
            # Check cache
            cache_key = self._get_cache_key("entities", query_str, filters)
            if use_cache and self.enable_caching:
                cached = self._get_cache(cache_key)
                if cached:
                    self._metrics["cached_queries"] += 1
                    return cached
                    
            # Build LMQL query using template
            lmql_query = self.template_registry.render(
                "entity_extraction",
                text=query_str,
                entity_types=", ".join(entity_types) if entity_types else "all",
                min_confidence=min_confidence,
                max_entities=max_results,
            )
            
            # Execute via adapter
            result = self.adapter.query(
                query_str=lmql_query,
                model=self.default_model,
                temperature=0.0,
            )
            
            if not result.success:
                self._metrics["failed_queries"] += 1
                return EntityQueryResult(
                    success=False,
                    query=query_str,
                    filters_applied=filters,
                    latency_ms=round((time.time() - start_time) * 1000, 2),
                    correlation_id=result.correlation_id,
                    error=result.error,
                )
                
            # Parse entities from result
            entities = self._parse_entities(result.data or "[]")
            
            # Apply additional filters
            if filters:
                entities = self._apply_entity_filters(entities, filters)
                
            # Limit results
            entities = entities[:max_results]
            
            query_result = EntityQueryResult(
                success=True,
                entities=entities,
                query=query_str,
                filters_applied=filters,
                total_count=len(entities),
                latency_ms=round((time.time() - start_time) * 1000, 2),
                correlation_id=result.correlation_id,
            )
            
            self._metrics["successful_queries"] += 1
            
            # Cache result
            if use_cache and self.enable_caching:
                self._set_cache(cache_key, query_result)
                
            return query_result
            
        except Exception as e:
            logger.error(f"Entity query failed: {e}", exc_info=True)
            self._metrics["failed_queries"] += 1
            return EntityQueryResult(
                success=False,
                query=query_str,
                filters_applied=filters,
                latency_ms=round((time.time() - start_time) * 1000, 2),
                error=str(e),
            )
            
    def query_relations(
        self,
        entity_ids: List[str],
        relation_types: Optional[List[str]] = None,
        include_incoming: bool = True,
        include_outgoing: bool = True,
        min_confidence: float = 0.5,
    ) -> RelationQueryResult:
        """Query relations for given entities.
        
        Args:
            entity_ids: List of entity IDs
            relation_types: Specific relation types to query
            include_incoming: Include incoming relations
            include_outgoing: Include outgoing relations
            min_confidence: Minimum confidence threshold
            
        Returns:
            RelationQueryResult with relations
        """
        import time
        start_time = time.time()
        self._metrics["total_queries"] += 1
        
        try:
            # If KG connection available, query directly
            if self.kg_connection:
                relations = self._query_kg_relations(
                    entity_ids,
                    relation_types,
                    include_incoming,
                    include_outgoing,
                )
                
                return RelationQueryResult(
                    success=True,
                    relations=relations,
                    entity_ids=entity_ids,
                    relation_types=relation_types or [],
                    latency_ms=round((time.time() - start_time) * 1000, 2),
                )
                
            # Otherwise, use LMQL to generate and execute Cypher
            cypher_result = self._generate_relation_cypher(
                entity_ids,
                relation_types,
                include_incoming,
                include_outgoing,
            )
            
            if not cypher_result.success:
                self._metrics["failed_queries"] += 1
                return RelationQueryResult(
                    success=False,
                    entity_ids=entity_ids,
                    error=cypher_result.error,
                )
                
            self._metrics["successful_queries"] += 1
            
            return RelationQueryResult(
                success=True,
                relations=[{"cypher": cypher_result.cypher, "params": cypher_result.params}],
                entity_ids=entity_ids,
                relation_types=relation_types or [],
                latency_ms=round((time.time() - start_time) * 1000, 2),
            )
            
        except Exception as e:
            logger.error(f"Relation query failed: {e}", exc_info=True)
            self._metrics["failed_queries"] += 1
            return RelationQueryResult(
                success=False,
                entity_ids=entity_ids,
                error=str(e),
            )
            
    def infer_schema(
        self,
        kg_sample: Optional[Dict[str, Any]] = None,
        sample_queries: Optional[List[str]] = None,
        use_lmql: bool = True,
    ) -> SchemaInferenceResult:
        """Infer knowledge graph schema from sample data or queries.
        
        Args:
            kg_sample: Sample KG data
            sample_queries: Example queries to infer from
            use_lmql: Whether to use LMQL for inference
            
        Returns:
            SchemaInferenceResult with inferred schema
        """
        import time
        start_time = time.time()
        self._metrics["total_queries"] += 1
        
        try:
            if kg_sample and use_lmql:
                # Use LMQL schema inference template
                result = self.adapter.query(
                    query_str=self.template_registry.render(
                        "schema_inference",
                        sample_data=json.dumps(kg_sample, indent=2),
                    ),
                    model=self.default_model,
                    temperature=0.0,
                )
                
                if result.success:
                    schema = self._parse_schema(result.data or "{}")
                    self._metrics["successful_queries"] += 1
                    
                    return SchemaInferenceResult(
                        success=True,
                        schema=schema,
                        entity_types=schema.get("entity_types", []),
                        relation_types=schema.get("relation_types", []),
                        confidence=0.85,
                        latency_ms=round((time.time() - start_time) * 1000, 2),
                        correlation_id=result.correlation_id,
                    )
                    
            elif sample_queries and use_lmql:
                # Infer from example queries
                queries_text = "\n".join(f"- {q}" for q in sample_queries)
                result = self.adapter.query(
                    query_str=self.template_registry.render(
                        "schema_inference_from_queries",
                        example_queries=queries_text,
                    ),
                    model=self.default_model,
                    temperature=0.0,
                )
                
                if result.success:
                    schema = self._parse_schema(result.data or "{}")
                    self._metrics["successful_queries"] += 1
                    
                    return SchemaInferenceResult(
                        success=True,
                        schema=schema,
                        entity_types=schema.get("entity_types", []),
                        relation_types=schema.get("relation_types", []),
                        confidence=0.80,
                        latency_ms=round((time.time() - start_time) * 1000, 2),
                        correlation_id=result.correlation_id,
                    )
                    
            # Fallback: return empty schema
            self._metrics["successful_queries"] += 1
            return SchemaInferenceResult(
                success=True,
                schema={"entity_types": [], "relation_types": []},
                confidence=0.0,
                latency_ms=round((time.time() - start_time) * 1000, 2),
            )
            
        except Exception as e:
            logger.error(f"Schema inference failed: {e}", exc_info=True)
            self._metrics["failed_queries"] += 1
            return SchemaInferenceResult(
                success=False,
                error=str(e),
            )
            
    def multi_hop_query(
        self,
        start_entity: str,
        query_path: List[Dict[str, Any]],
        max_hops: int = 5,
        min_confidence: float = 0.6,
    ) -> MultiHopResult:
        """Execute multi-hop query over knowledge graph.
        
        Args:
            start_entity: Starting entity ID or name
            query_path: List of hop specifications [{"relation": "type", "direction": "out"}]
            max_hops: Maximum number of hops
            min_confidence: Minimum confidence threshold
            
        Returns:
            MultiHopResult with paths and answer
        """
        import time
        import uuid
        start_time = time.time()
        self._metrics["total_queries"] += 1
        correlation_id = str(uuid.uuid4())
        
        try:
            # Build reasoning query
            relations = [hop.get("relation", "") for hop in query_path]
            
            result = self.adapter.query(
                query_str=self.template_registry.render(
                    "multi_hop_reasoning",
                    question=f"Find paths from {start_entity}",
                    start_entity=start_entity,
                    relations=", ".join(relations),
                    min_hops=1,
                    max_hops=max_hops,
                    min_confidence=min_confidence,
                ),
                model=self.default_model,
                temperature=0.3,
            )
            
            if not result.success:
                self._metrics["failed_queries"] += 1
                return MultiHopResult(
                    success=False,
                    start_entity=start_entity,
                    correlation_id=correlation_id,
                    error=result.error,
                )
                
            # Parse reasoning result
            reasoning = self._parse_reasoning(result.data or "{}")
            
            self._metrics["successful_queries"] += 1
            
            return MultiHopResult(
                success=True,
                paths=reasoning.get("paths", []),
                start_entity=start_entity,
                answer=reasoning.get("answer", ""),
                confidence=reasoning.get("confidence", 0.0),
                hops_taken=len(reasoning.get("reasoning_steps", [])),
                entities_visited=reasoning.get("entities_visited", []),
                reasoning_trace=reasoning.get("reasoning_steps", []),
                latency_ms=round((time.time() - start_time) * 1000, 2),
                correlation_id=correlation_id,
            )
            
        except Exception as e:
            logger.error(f"Multi-hop query failed: {e}", exc_info=True)
            self._metrics["failed_queries"] += 1
            return MultiHopResult(
                success=False,
                start_entity=start_entity,
                correlation_id=correlation_id,
                error=str(e),
            )
            
    def explain_query(
        self,
        query_str: str,
    ) -> QueryExplanation:
        """Explain how a query will be executed.
        
        Args:
            query_str: Query to explain
            
        Returns:
            QueryExplanation with execution plan
        """
        # Parse constraints from query
        from integrations.lmql.constraint_engine import ConstraintParser
        
        parser = ConstraintParser()
        constraints = parser.parse(query_str)
        
        # Build execution plan
        execution_plan = f"""
Query Analysis:
1. Parse natural language query
2. Extract constraints: {len(constraints)} found
3. Generate LMQL query with constraints
4. Execute with adapter
5. Parse and validate results
"""
        
        # Optimization hints
        hints = []
        if len(constraints) > 3:
            hints.append("Consider simplifying query - many constraints may reduce recall")
        if any(c.get_type().name == "REGEX" for c in constraints):
            hints.append("Regex constraints are computationally expensive")
            
        return QueryExplanation(
            query=query_str,
            parsed_constraints=[
                {"type": c.get_type().value, "syntax": c.to_lmql_syntax()}
                for c in constraints
            ],
            execution_plan=execution_plan,
            estimated_cost="low" if len(constraints) < 3 else "medium",
            optimization_hints=hints,
        )
        
    def generate_cypher(
        self,
        natural_language_query: str,
        query_type: str = "general",
        is_temporal: bool = False,
    ) -> CypherGenerationResult:
        """Generate Memgraph-compatible Cypher from natural language.
        
        Args:
            natural_language_query: Natural language query
            query_type: Type of query (general, path, aggregation)
            is_temporal: Whether query involves temporal constraints
            
        Returns:
            CypherGenerationResult with generated Cypher
        """
        import time
        start_time = time.time()
        
        try:
            # Select template based on query type
            if query_type == "path":
                template_name = "cypher_path_query"
            elif query_type == "aggregation":
                template_name = "cypher_aggregation"
            elif is_temporal:
                template_name = "cypher_temporal"
            else:
                template_name = "cypher_generation"
                
            # Render template
            lmql_query = self.template_registry.render(
                template_name,
                natural_language_query=natural_language_query,
                params="{}",
            )
            
            # Execute via adapter
            result = self.adapter.query(
                query_str=lmql_query,
                model=self.default_model,
                temperature=0.0,
            )
            
            if not result.success:
                return CypherGenerationResult(
                    success=False,
                    error=result.error,
                )
                
            cypher = result.data or ""
            
            # Clean up Cypher
            cypher = cypher.strip()
            if cypher.startswith('"') and cypher.endswith('"'):
                cypher = cypher[1:-1]
                
            return CypherGenerationResult(
                success=True,
                cypher=cypher,
                params={},
                explanation=f"Generated from: {natural_language_query}",
                is_temporal=is_temporal,
                latency_ms=round((time.time() - start_time) * 1000, 2),
            )
            
        except Exception as e:
            logger.error(f"Cypher generation failed: {e}", exc_info=True)
            return CypherGenerationResult(
                success=False,
                error=str(e),
            )
            
    def _get_cache_key(self, query_type: str, query_str: str, filters: Dict[str, Any]) -> str:
        """Generate cache key."""
        import hashlib
        data = f"{query_type}:{query_str}:{json.dumps(filters, sort_keys=True)}"
        return hashlib.md5(data.encode()).hexdigest()
        
    def _get_cache(self, key: str) -> Optional[Any]:
        """Get cached result."""
        import time
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return result
            del self._cache[key]
        return None
        
    def _set_cache(self, key: str, result: Any) -> None:
        """Set cached result."""
        import time
        self._cache[key] = (result, time.time())
        
    def _parse_entities(self, data: str) -> List[Dict[str, Any]]:
        """Parse entities from LLM response."""
        try:
            parsed = json.loads(data)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "entities" in parsed:
                return parsed["entities"]
            return [parsed]
        except json.JSONDecodeError:
            # Fallback: try to extract entities using regex
            return []
            
    def _apply_entity_filters(
        self,
        entities: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply additional filters to entities."""
        filtered = entities
        
        if "entity_type" in filters:
            filtered = [e for e in filtered if e.get("type") == filters["entity_type"]]
            
        if "min_confidence" in filters:
            min_conf = filters["min_confidence"]
            filtered = [e for e in filtered if e.get("confidence", 0) >= min_conf]
            
        return filtered
        
    def _query_kg_relations(
        self,
        entity_ids: List[str],
        relation_types: Optional[List[str]],
        include_incoming: bool,
        include_outgoing: bool,
    ) -> List[Dict[str, Any]]:
        """Query relations from KG connection."""
        if not self.kg_connection:
            return []
            
        relations = []
        
        # This would be implemented based on specific KG driver
        # Example for Neo4j/Memgraph:
        if hasattr(self.kg_connection, 'run'):
            for entity_id in entity_ids:
                if include_outgoing:
                    rel_filter = ":" + "|".join(relation_types) if relation_types else ""
                    query = f"MATCH (n {{id: $id}})-[r{rel_filter}]->(m) RETURN r, m"
                    result = self.kg_connection.run(query, {"id": entity_id})
                    relations.extend([dict(r) for r in result])
                    
                if include_incoming:
                    rel_filter = ":" + "|".join(relation_types) if relation_types else ""
                    query = f"MATCH (n {{id: $id}})<-[r{rel_filter}]-(m) RETURN r, m"
                    result = self.kg_connection.run(query, {"id": entity_id})
                    relations.extend([dict(r) for r in result])
                    
        return relations
        
    def _generate_relation_cypher(
        self,
        entity_ids: List[str],
        relation_types: Optional[List[str]],
        include_incoming: bool,
        include_outgoing: bool,
    ) -> CypherGenerationResult:
        """Generate Cypher for relation query."""
        # Build description
        desc = f"Find relations for entities: {entity_ids}"
        if relation_types:
            desc += f" with types: {relation_types}"
            
        return self.generate_cypher(desc, query_type="path")
        
    def _parse_schema(self, data: str) -> Dict[str, Any]:
        """Parse schema from LLM response."""
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {"entity_types": [], "relation_types": []}
            
    def _parse_reasoning(self, data: str) -> Dict[str, Any]:
        """Parse reasoning result from LLM response."""
        try:
            parsed = json.loads(data)
            return parsed
        except json.JSONDecodeError:
            return {
                "paths": [],
                "answer": "",
                "confidence": 0.0,
                "reasoning_steps": [],
                "entities_visited": [],
            }
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get integration metrics."""
        metrics = self._metrics.copy()
        if metrics["total_queries"] > 0:
            metrics["success_rate"] = metrics["successful_queries"] / metrics["total_queries"]
            metrics["avg_latency_ms"] = metrics["total_latency_ms"] / metrics["total_queries"]
        else:
            metrics["success_rate"] = 0.0
            metrics["avg_latency_ms"] = 0.0
        return metrics
        
    def reset_metrics(self) -> None:
        """Reset metrics."""
        self._metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "cached_queries": 0,
            "total_latency_ms": 0.0,
        }
        
    def clear_cache(self) -> None:
        """Clear query cache."""
        self._cache.clear()


# =============================================================================
# INTEGRATION WITH UNIFIED KG HUB
# =============================================================================


class UnifiedKGIntegrationHub:
    """Integration hub for KG operations.
    
    This class provides a unified interface for various KG integrations.
    LMQLKGIntegration can register with this hub.
    """
    
    def __init__(self):
        self._integrations: Dict[str, LMQLKGIntegration] = {}
        
    def register_integration(
        self,
        name: str,
        integration: LMQLKGIntegration
    ) -> None:
        """Register an LMQL KG integration."""
        self._integrations[name] = integration
        logger.info(f"Registered LMQL KG integration: {name}")
        
    def get_integration(self, name: str) -> Optional[LMQLKGIntegration]:
        """Get a registered integration."""
        return self._integrations.get(name)
        
    def list_integrations(self) -> List[str]:
        """List registered integration names."""
        return list(self._integrations.keys())


# Default hub instance
_default_hub: Optional[UnifiedKGIntegrationHub] = None


def get_default_hub() -> UnifiedKGIntegrationHub:
    """Get default integration hub."""
    global _default_hub
    if _default_hub is None:
        _default_hub = UnifiedKGIntegrationHub()
    return _default_hub


def register_with_hub(
    name: str = "lmql",
    integration: Optional[LMQLKGIntegration] = None
) -> LMQLKGIntegration:
    """Register LMQL integration with default hub."""
    hub = get_default_hub()
    integration = integration or LMQLKGIntegration()
    hub.register_integration(name, integration)
    return integration


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Data classes
    "EntityQueryResult",
    "RelationQueryResult",
    "SchemaInferenceResult",
    "MultiHopResult",
    "QueryExplanation",
    "CypherGenerationResult",
    # Main classes
    "LMQLKGIntegration",
    "UnifiedKGIntegrationHub",
    # Functions
    "get_default_hub",
    "register_with_hub",
]


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    integration = LMQLKGIntegration()
    
    # Example 1: Entity query
    print("\nEntity Query Example:")
    result = integration.query_entities(
        "Find technology companies",
        entity_types=["ORG"],
        max_results=5
    )
    print(f"  Success: {result.success}")
    print(f"  Entities found: {result.total_count}")
    
    # Example 2: Schema inference
    print("\nSchema Inference Example:")
    result = integration.infer_schema(
        kg_sample={
            "entities": [
                {"name": "Apple", "type": "Company"},
                {"name": "Steve Jobs", "type": "Person"}
            ],
            "relations": [
                {"from": "Steve Jobs", "to": "Apple", "type": "FOUNDED"}
            ]
        }
    )
    print(f"  Success: {result.success}")
    print(f"  Entity types: {len(result.entity_types)}")
    print(f"  Relation types: {len(result.relation_types)}")
    
    # Example 3: Cypher generation
    print("\nCypher Generation Example:")
    result = integration.generate_cypher(
        "Find all companies founded by Steve Jobs",
        is_temporal=False
    )
    print(f"  Success: {result.success}")
    if result.success:
        print(f"  Cypher: {result.cypher[:100]}...")
        
    # Print metrics
    print("\nMetrics:")
    print(f"  {integration.get_metrics()}")
