"""
Knowledge Engine Outlines Integration

Thin wrapper around SSOT implementations with KE-specific context.
Provides:
- Integration with UnifiedKGIntegrationHub
- Memgraph-compatible output formats
- Structured logging per CLAUDE.md

This module follows the SSOT pattern - primary logic is in integrations/outlines/,
this is a thin wrapper with KE-specific conveniences.
"""

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

# SSOT imports from primary implementation
from integrations.outlines import (
    OutlinesAdapter,
    OutlinesConfig,
    OutlinesResult,
    EntityExtractionSchema,
    RelationshipSchema,
    CypherQuerySchema,
    ValidationResultSchema,
    KnowledgeGraphConstraints,
    PromptTemplateManager,
    GenerationError,
    ValidationError,
)

# Try to import UnifiedKGIntegrationHub
# Per CLAUDE.md: Use adapter layer, avoid direct core-projects imports
try:
    from knowledge_engine.integrations.unified_kg_integration_hub import UnifiedKGIntegrationHub
    _HUB_AVAILABLE = True
except ImportError:
    _HUB_AVAILABLE = False
    UnifiedKGIntegrationHub = None

# Configure logger per CLAUDE.md (structured logging, UTC timestamps)
logger = logging.getLogger(__name__)


@dataclass
class KGExtractionResult:
    """Result of KG extraction with Memgraph-compatible format."""
    success: bool
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    cypher_queries: List[str]
    metadata: Dict[str, Any]
    extraction_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None
    
    def to_memgraph_format(self) -> Dict[str, Any]:
        """Convert to Memgraph import format."""
        return {
            "nodes": self.entities,
            "edges": self.relationships,
            "queries": self.cypher_queries,
            "metadata": {
                **self.metadata,
                "extraction_timestamp": self.extraction_timestamp,
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "entities": self.entities,
            "relationships": self.relationships,
            "cypher_queries": self.cypher_queries,
            "metadata": self.metadata,
            "extraction_timestamp": self.extraction_timestamp,
            "error": self.error,
        }


class OutlinesKGIntegration:
    """
    Knowledge Engine integration for Outlines structured generation.
    
    Provides KE-specific methods that wrap the SSOT OutlinesAdapter:
    - Entity extraction with KG type constraints
    - Relationship extraction with Memgraph-compatible output
    - Cypher query generation
    - Batch document processing
    
    Features:
    - Memgraph-compatible output formats
    - Structured logging with correlation IDs
    - UTC timestamps for all operations
    - Idempotent operations
    - Circuit breaker pattern
    """
    
    def __init__(
        self,
        config: Optional[OutlinesConfig] = None,
        hub: Optional[Any] = None,
    ):
        """
        Initialize the Outlines KG Integration.
        
        Args:
            config: Outlines configuration. Uses defaults if None.
            hub: Optional UnifiedKGIntegrationHub instance
        """
        self.config = config or OutlinesConfig()
        self.adapter = OutlinesAdapter(self.config)
        self.template_manager = PromptTemplateManager()
        self.constraints = KnowledgeGraphConstraints()
        
        # Integration hub
        self._hub = hub
        if hub is None and _HUB_AVAILABLE:
            try:
                self._hub = UnifiedKGIntegrationHub()
            except Exception as e:
                logger.warning({
                    "msg": "Failed to initialize UnifiedKGIntegrationHub",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
        
        # Thread pool for batch operations
        self._executor = ThreadPoolExecutor(max_workers=self.config.batch_max_workers)
        
        logger.info({
            "msg": "OutlinesKGIntegration initialized",
            "hub_connected": self._hub is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    
    def _compute_text_hash(self, text: str) -> str:
        """Compute hash for text deduplication (idempotency)."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def extract_entities_constrained(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
    ) -> EntityExtractionSchema:
        """
        Extract entities with type constraints using Outlines.
        
        Args:
            text: Text to extract entities from
            entity_types: List of allowed entity types (default: all KG types)
            correlation_id: Correlation ID for tracking
            
        Returns:
            EntityExtractionSchema with extracted entities
        """
        correlation_id = correlation_id or f"ke_entity_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        text_hash = self._compute_text_hash(text)
        
        if entity_types is None:
            entity_types = self.constraints.get_entity_types()
        
        logger.info({
            "msg": "Starting constrained entity extraction",
            "correlation_id": correlation_id,
            "text_length": len(text),
            "entity_types_count": len(entity_types),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        try:
            # Create prompt using template
            prompt = self.template_manager.create_entity_extraction_prompt(
                text=text,
                entity_types=entity_types,
                model=self.config.model_name,
            )
            
            # Generate with JSON constraint
            schema = self.constraints.get_entity_extraction_schema()
            result = self.adapter.generate_json(
                schema=schema,
                prompt=prompt,
                correlation_id=correlation_id,
            )
            
            if result.success:
                # Parse and validate
                extraction = EntityExtractionSchema(**result.output)
                extraction.text_hash = text_hash
                extraction.model_used = result.model
                
                logger.info({
                    "msg": "Entity extraction completed",
                    "correlation_id": correlation_id,
                    "entities_count": len(extraction.entities),
                    "processing_time_ms": result.processing_time_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                
                return extraction
            else:
                logger.error({
                    "msg": "Entity extraction failed",
                    "correlation_id": correlation_id,
                    "error": result.error,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                return EntityExtractionSchema(
                    entities=[],
                    text_hash=text_hash,
                    model_used=result.model,
                )
                
        except Exception as e:
            logger.error({
                "msg": "Entity extraction exception",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return EntityExtractionSchema(
                entities=[],
                text_hash=text_hash,
                error=str(e),
            )
    
    def extract_relations_constrained(
        self,
        text: str,
        relation_types: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
    ) -> RelationshipSchema:
        """
        Extract relationships with type constraints using Outlines.
        
        Args:
            text: Text to extract relationships from
            relation_types: List of allowed relation types (default: all KG types)
            entities: Optional list of known entity names
            correlation_id: Correlation ID for tracking
            
        Returns:
            RelationshipSchema with extracted relationships
        """
        correlation_id = correlation_id or f"ke_relation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        text_hash = self._compute_text_hash(text)
        
        if relation_types is None:
            relation_types = self.constraints.get_relation_types()
        
        logger.info({
            "msg": "Starting constrained relationship extraction",
            "correlation_id": correlation_id,
            "text_length": len(text),
            "relation_types_count": len(relation_types),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        try:
            # Create prompt
            prompt = self.template_manager.create_relation_extraction_prompt(
                text=text,
                relation_types=relation_types,
                entities=entities,
            )
            
            # Generate with JSON constraint
            schema = self.constraints.get_relationship_schema()
            result = self.adapter.generate_json(
                schema=schema,
                prompt=prompt,
                correlation_id=correlation_id,
            )
            
            if result.success:
                extraction = RelationshipSchema(**result.output)
                extraction.text_hash = text_hash
                extraction.model_used = result.model
                
                logger.info({
                    "msg": "Relationship extraction completed",
                    "correlation_id": correlation_id,
                    "relationships_count": len(extraction.relationships),
                    "processing_time_ms": result.processing_time_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                
                return extraction
            else:
                logger.error({
                    "msg": "Relationship extraction failed",
                    "correlation_id": correlation_id,
                    "error": result.error,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                return RelationshipSchema(
                    relationships=[],
                    text_hash=text_hash,
                    model_used=result.model,
                )
                
        except Exception as e:
            logger.error({
                "msg": "Relationship extraction exception",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return RelationshipSchema(
                relationships=[],
                text_hash=text_hash,
                error=str(e),
            )
    
    def generate_cypher_constrained(
        self,
        query_intent: str,
        schema_description: str,
        node_labels: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
    ) -> CypherQuerySchema:
        """
        Generate Memgraph Cypher query with constraints.
        
        Args:
            query_intent: Natural language description of query intent
            schema_description: Description of graph schema
            node_labels: Available node labels
            relationship_types: Available relationship types
            correlation_id: Correlation ID for tracking
            
        Returns:
            CypherQuerySchema with generated query
        """
        correlation_id = correlation_id or f"ke_cypher_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        logger.info({
            "msg": "Starting constrained Cypher generation",
            "correlation_id": correlation_id,
            "query_intent": query_intent[:100] + "..." if len(query_intent) > 100 else query_intent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        try:
            # Create prompt
            prompt = self.template_manager.create_cypher_generation_prompt(
                query_intent=query_intent,
                schema_description=schema_description,
                node_labels=node_labels,
                relationship_types=relationship_types,
            )
            
            # Generate with JSON constraint
            schema = self.constraints.get_cypher_query_schema()
            result = self.adapter.generate_json(
                schema=schema,
                prompt=prompt,
                correlation_id=correlation_id,
            )
            
            if result.success:
                query_result = CypherQuerySchema(**result.output)
                
                logger.info({
                    "msg": "Cypher generation completed",
                    "correlation_id": correlation_id,
                    "query_type": query_result.query_type,
                    "complexity": query_result.estimated_complexity,
                    "processing_time_ms": result.processing_time_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                
                return query_result
            else:
                logger.error({
                    "msg": "Cypher generation failed",
                    "correlation_id": correlation_id,
                    "error": result.error,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                return CypherQuerySchema(
                    query="",
                    explanation=f"Generation failed: {result.error}",
                )
                
        except Exception as e:
            logger.error({
                "msg": "Cypher generation exception",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return CypherQuerySchema(
                query="",
                explanation=f"Exception: {str(e)}",
            )
    
    def validate_kg_structure(
        self,
        kg_data: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ) -> ValidationResultSchema:
        """
        Validate KG structure against schema.
        
        Args:
            kg_data: Knowledge graph data to validate
            correlation_id: Correlation ID for tracking
            
        Returns:
            ValidationResultSchema with validation results
        """
        correlation_id = correlation_id or f"ke_validate_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        logger.info({
            "msg": "Starting KG structure validation",
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        result = ValidationResultSchema(is_valid=True)
        
        # Validate entities
        if "entities" in kg_data:
            for i, entity in enumerate(kg_data["entities"]):
                if "name" not in entity:
                    result.add_error(
                        message=f"Entity {i} missing required field 'name'",
                        field="entities",
                        suggestion="Add 'name' property to entity"
                    )
                if "type" not in entity:
                    result.add_error(
                        message=f"Entity {i} missing required field 'type'",
                        field="entities",
                        suggestion="Add 'type' property to entity"
                    )
        
        # Validate relationships
        if "relationships" in kg_data:
            for i, rel in enumerate(kg_data["relationships"]):
                required_fields = ["source", "target", "type"]
                for field in required_fields:
                    if field not in rel:
                        result.add_error(
                            message=f"Relationship {i} missing required field '{field}'",
                            field="relationships",
                            suggestion=f"Add '{field}' property to relationship"
                        )
        
        # Calculate confidence based on issues
        total_issues = len(result.errors) + len(result.warnings)
        if total_issues == 0:
            result.confidence = 1.0
        else:
            result.confidence = max(0.0, 1.0 - (total_issues * 0.1))
        
        logger.info({
            "msg": "KG structure validation completed",
            "correlation_id": correlation_id,
            "is_valid": result.is_valid,
            "errors_count": len(result.errors),
            "warnings_count": len(result.warnings),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        return result
    
    def batch_process_documents(
        self,
        docs: List[Union[str, Dict[str, Any]]],
        extraction_config: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> List[KGExtractionResult]:
        """
        Process multiple documents in parallel.
        
        Args:
            docs: List of documents (strings or dicts with 'text' and 'id' keys)
            extraction_config: Configuration for extraction
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of KGExtractionResult objects
        """
        correlation_id = correlation_id or f"ke_batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        extraction_config = extraction_config or {}
        
        logger.info({
            "msg": "Starting batch document processing",
            "correlation_id": correlation_id,
            "docs_count": len(docs),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        entity_types = extraction_config.get("entity_types")
        relation_types = extraction_config.get("relation_types")
        
        results = []
        
        def process_single(doc: Union[str, Dict[str, Any]], index: int) -> KGExtractionResult:
            # Extract text and id
            if isinstance(doc, str):
                text = doc
                doc_id = f"doc_{index}"
            else:
                text = doc.get("text", "")
                doc_id = doc.get("id", f"doc_{index}")
            
            doc_correlation_id = f"{correlation_id}_{doc_id}"
            
            try:
                # Extract entities
                entities_result = self.extract_entities_constrained(
                    text=text,
                    entity_types=entity_types,
                    correlation_id=doc_correlation_id,
                )
                
                # Extract relationships
                entity_names = [e.name for e in entities_result.entities]
                relations_result = self.extract_relations_constrained(
                    text=text,
                    relation_types=relation_types,
                    entities=entity_names,
                    correlation_id=doc_correlation_id,
                )
                
                # Convert to Memgraph format
                entities_memgraph = entities_result.to_memgraph_nodes()
                relationships_memgraph = relations_result.to_memgraph_edges()
                
                return KGExtractionResult(
                    success=True,
                    entities=entities_memgraph,
                    relationships=relationships_memgraph,
                    cypher_queries=[],  # Could generate Cypher here if needed
                    metadata={
                        "doc_id": doc_id,
                        "entity_count": len(entities_result.entities),
                        "relationship_count": len(relations_result.relationships),
                        "correlation_id": doc_correlation_id,
                    },
                )
                
            except Exception as e:
                logger.error({
                    "msg": f"Document {doc_id} processing failed",
                    "correlation_id": doc_correlation_id,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                return KGExtractionResult(
                    success=False,
                    entities=[],
                    relationships=[],
                    cypher_queries=[],
                    metadata={"doc_id": doc_id},
                    error=str(e),
                )
        
        # Process in parallel
        futures = [
            self._executor.submit(process_single, doc, i)
            for i, doc in enumerate(docs)
        ]
        
        for future in futures:
            try:
                result = future.result(timeout=self.config.batch_timeout_seconds)
                results.append(result)
            except Exception as e:
                results.append(KGExtractionResult(
                    success=False,
                    entities=[],
                    relationships=[],
                    cypher_queries=[],
                    metadata={},
                    error=str(e),
                ))
        
        successful_count = sum(1 for r in results if r.success)
        
        logger.info({
            "msg": "Batch document processing completed",
            "correlation_id": correlation_id,
            "docs_count": len(docs),
            "successful_count": successful_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        return results
    
    def extract_and_build_kg(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
    ) -> KGExtractionResult:
        """
        Extract entities and relationships and build KG structure.
        
        Args:
            text: Text to process
            entity_types: List of allowed entity types
            relation_types: List of allowed relation types
            correlation_id: Correlation ID for tracking
            
        Returns:
            KGExtractionResult with complete KG structure
        """
        correlation_id = correlation_id or f"ke_build_kg_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        logger.info({
            "msg": "Starting KG extraction and build",
            "correlation_id": correlation_id,
            "text_length": len(text),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        try:
            # Extract entities
            entities_result = self.extract_entities_constrained(
                text=text,
                entity_types=entity_types,
                correlation_id=f"{correlation_id}_entities",
            )
            
            # Extract relationships
            entity_names = [e.name for e in entities_result.entities]
            relations_result = self.extract_relations_constrained(
                text=text,
                relation_types=relation_types,
                entities=entity_names,
                correlation_id=f"{correlation_id}_relations",
            )
            
            # Convert to Memgraph format
            entities_memgraph = entities_result.to_memgraph_nodes()
            relationships_memgraph = relations_result.to_memgraph_edges()
            
            # Generate Cypher queries for insertion
            cypher_queries = self._generate_insert_queries(
                entities_memgraph,
                relationships_memgraph,
            )
            
            result = KGExtractionResult(
                success=True,
                entities=entities_memgraph,
                relationships=relationships_memgraph,
                cypher_queries=cypher_queries,
                metadata={
                    "entity_count": len(entities_result.entities),
                    "relationship_count": len(relations_result.relationships),
                    "correlation_id": correlation_id,
                },
            )
            
            logger.info({
                "msg": "KG extraction and build completed",
                "correlation_id": correlation_id,
                "entity_count": len(entities_result.entities),
                "relationship_count": len(relations_result.relationships),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            return result
            
        except Exception as e:
            logger.error({
                "msg": "KG extraction and build failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return KGExtractionResult(
                success=False,
                entities=[],
                relationships=[],
                cypher_queries=[],
                metadata={"correlation_id": correlation_id},
                error=str(e),
            )
    
    def _generate_insert_queries(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate Memgraph Cypher insert queries."""
        queries = []
        
        # Entity insertion queries
        for entity in entities:
            labels = ":".join(entity.get("labels", ["Entity"]))
            props = entity.get("properties", {})
            
            # Build property string
            prop_str = ", ".join([f"{k}: ${k}" for k in props.keys()])
            
            query = f"MERGE (n:{labels} {{{prop_str}}}) RETURN n"
            queries.append(query)
        
        # Relationship insertion queries
        for rel in relationships:
            rel_type = rel.get("type", "RELATED_TO")
            from_name = rel.get("from", {}).get("name", "")
            to_name = rel.get("to", {}).get("name", "")
            props = rel.get("properties", {})
            
            prop_str = ""
            if props:
                prop_str = " { " + ", ".join([f"{k}: ${k}" for k in props.keys()]) + " }"
            
            query = (
                f"MATCH (a {{name: $from_name}}), (b {{name: $to_name}}) "
                f"MERGE (a)-[r:{rel_type}{prop_str}]->(b) "
                f"RETURN r"
            )
            queries.append(query)
        
        return queries
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration status."""
        return {
            "adapter_initialized": self.adapter is not None,
            "hub_connected": self._hub is not None,
            "model_provider": self.config.model_provider.value,
            "model_name": self.config.model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    async def close(self):
        """Close resources."""
        logger.info({
            "msg": "Closing OutlinesKGIntegration resources",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        self._executor.shutdown(wait=True)
        await self.adapter.close()
        
        logger.info({
            "msg": "OutlinesKGIntegration resources closed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


__all__ = [
    "OutlinesKGIntegration",
    "KGExtractionResult",
]
