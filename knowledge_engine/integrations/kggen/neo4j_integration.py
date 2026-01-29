"""
Neo4j Integration Module for KG-Gen Pipeline

This module provides functionality for uploading knowledge graphs to Neo4j
and performing graph operations.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import uuid


logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    """Result of a Neo4j upload operation."""
    success: bool
    entities_uploaded: int = 0
    relationships_uploaded: int = 0
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class Neo4jGraphUploader:
    """
    Neo4j uploader for knowledge graphs.
    
    Provides methods for uploading entities and relationships to Neo4j,
    creating indices, and running queries.
    """
    
    def __init__(self, neo4j_driver):
        """
        Initialize the Neo4j uploader.
        
        Args:
            neo4j_driver: Neo4j driver instance
        """
        self.driver = neo4j_driver
        
        logger.info({
            "msg": "Neo4jGraphUploader initialized",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def upload_graph(
        self,
        graph: Any,  # KnowledgeGraph from extraction_pipeline
        batch_size: int = 100,
        correlation_id: Optional[str] = None
    ) -> UploadResult:
        """
        Upload a knowledge graph to Neo4j.
        
        Args:
            graph: KnowledgeGraph object with entities and relations
            batch_size: Size of batches for uploading
            correlation_id: Correlation ID for tracking
            
        Returns:
            UploadResult with success status and counts
        """
        correlation_id = correlation_id or f"upload_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting Neo4j graph upload",
            "entities_count": len(graph.entities),
            "relations_count": len(graph.relations),
            "batch_size": batch_size,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            entities_uploaded = 0
            relationships_uploaded = 0
            
            # Upload entities in batches
            for i in range(0, len(graph.entities), batch_size):
                batch = graph.entities[i:i + batch_size]
                batch_result = await self._upload_entities_batch(batch, correlation_id)
                entities_uploaded += batch_result.get('count', 0)
                
                if not batch_result.get('success'):
                    logger.warning({
                        "msg": f"Entity batch {i//batch_size} upload partially failed",
                        "correlation_id": correlation_id,
                        "error": batch_result.get('error')
                    })
            
            # Upload relationships in batches
            for i in range(0, len(graph.relations), batch_size):
                batch = graph.relations[i:i + batch_size]
                batch_result = await self._upload_relationships_batch(batch, correlation_id)
                relationships_uploaded += batch_result.get('count', 0)
                
                if not batch_result.get('success'):
                    logger.warning({
                        "msg": f"Relationship batch {i//batch_size} upload partially failed",
                        "correlation_id": correlation_id,
                        "error": batch_result.get('error')
                    })
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            result = UploadResult(
                success=True,
                entities_uploaded=entities_uploaded,
                relationships_uploaded=relationships_uploaded,
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "Neo4j graph upload completed",
                "correlation_id": correlation_id,
                "entities_uploaded": entities_uploaded,
                "relationships_uploaded": relationships_uploaded,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Neo4j graph upload failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return UploadResult(
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )
    
    async def _upload_entities_batch(
        self,
        entities: List[str],
        correlation_id: str
    ) -> Dict[str, Any]:
        """
        Upload a batch of entities to Neo4j.
        
        Args:
            entities: List of entity names to upload
            correlation_id: Correlation ID for tracking
            
        Returns:
            Dictionary with upload result
        """
        try:
            # Create a session and run the transaction
            async with self.driver.session() as session:
                # Cypher query to create entities
                query = """
                UNWIND $entities AS entityName
                MERGE (e:Entity {name: entityName})
                SET e.updated_at = datetime()
                RETURN count(e) AS createdCount
                """
                
                result = await session.run(query, entities=entities)
                record = await result.single()
                
                count = record["createdCount"] if record else 0
                
                return {
                    "success": True,
                    "count": count,
                    "correlation_id": correlation_id
                }
                
        except Exception as e:
            logger.error({
                "msg": "Entity batch upload failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "entities_count": len(entities)
            })
            return {
                "success": False,
                "error": str(e),
                "count": 0
            }
    
    async def _upload_relationships_batch(
        self,
        relations: List[Tuple[str, str, str]],
        correlation_id: str
    ) -> Dict[str, Any]:
        """
        Upload a batch of relationships to Neo4j.
        
        Args:
            relations: List of (subject, predicate, object) relationships
            correlation_id: Correlation ID for tracking
            
        Returns:
            Dictionary with upload result
        """
        try:
            # Prepare data for Cypher query
            rel_data = []
            for subj, pred, obj in relations:
                rel_data.append({
                    "subject": subj,
                    "predicate": pred,
                    "object": obj
                })
            
            # Create a session and run the transaction
            async with self.driver.session() as session:
                # Cypher query to create relationships
                query = """
                UNWIND $relations AS r
                MATCH (s:Entity {name: r.subject})
                MATCH (o:Entity {name: r.object})
                MERGE (s)-[rel:`%s`]->(o)
                SET rel.updated_at = datetime()
                RETURN count(rel) AS createdCount
                """ % "RELATED"  # Using a generic relationship type, could be dynamic
                
                result = await session.run(query, relations=rel_data)
                record = await result.single()
                
                count = record["createdCount"] if record else 0
                
                return {
                    "success": True,
                    "count": count,
                    "correlation_id": correlation_id
                }
                
        except Exception as e:
            logger.error({
                "msg": "Relationship batch upload failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "relations_count": len(relations)
            })
            return {
                "success": False,
                "error": str(e),
                "count": 0
            }
    
    async def create_indices(self) -> Dict[str, Any]:
        """
        Create recommended indices in Neo4j for better performance.
        
        Returns:
            Dictionary with creation results
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Creating Neo4j indices",
            "timestamp": start_time.isoformat()
        })
        
        try:
            async with self.driver.session() as session:
                # Create indices for better query performance
                queries = [
                    "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                    "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                    "CREATE TEXT INDEX entity_name_text_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                ]
                
                results = []
                for query in queries:
                    result = await session.run(query)
                    results.append({"query": query, "success": True})
                
                processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                logger.info({
                    "msg": "Neo4j indices created",
                    "indices_count": len(queries),
                    "processing_time_ms": processing_time_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                return {
                    "success": True,
                    "indices_created": len(queries),
                    "processing_time_ms": processing_time_ms,
                    "details": results
                }
                
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Neo4j index creation failed",
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": processing_time_ms
            }
    
    async def query_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        Query a single entity from Neo4j.
        
        Args:
            entity_name: Name of entity to query
            
        Returns:
            Entity data or None if not found
        """
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (e:Entity {name: $name})
                RETURN e {.name, .type, .created_at, .updated_at} AS entity
                LIMIT 1
                """
                
                result = await session.run(query, name=entity_name)
                record = await result.single()
                
                if record:
                    return record["entity"]
                else:
                    return None
                    
        except Exception as e:
            logger.error({
                "msg": "Entity query failed",
                "entity_name": entity_name,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return None
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph in Neo4j.
        
        Returns:
            Dictionary with graph statistics
        """
        try:
            async with self.driver.session() as session:
                # Count nodes
                node_query = "MATCH (n) RETURN count(n) AS nodeCount"
                node_result = await session.run(node_query)
                node_record = await node_result.single()
                node_count = node_record["nodeCount"] if node_record else 0
                
                # Count relationships
                rel_query = "MATCH ()-[r]->() RETURN count(r) AS relCount"
                rel_result = await session.run(rel_query)
                rel_record = await rel_result.single()
                rel_count = rel_record["relCount"] if rel_record else 0
                
                # Count entity types
                type_query = "MATCH (e:Entity) RETURN e.type AS entityType, count(e) AS count"
                type_result = await session.run(type_query)
                types = []
                async for record in type_result:
                    types.append({
                        "type": record["entityType"],
                        "count": record["count"]
                    })
                
                stats = {
                    "nodes": node_count,
                    "relationships": rel_count,
                    "entity_types": types,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                return stats
                
        except Exception as e:
            logger.error({
                "msg": "Failed to get graph statistics",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return {"error": str(e)}
    
    async def export_graph(self, format: str = 'json') -> str:
        """
        Export knowledge graph from Neo4j.
        
        Args:
            format: Export format ('json', 'csv', 'graphml')
            
        Returns:
            Exported graph data as string
        """
        try:
            async with self.driver.session() as session:
                if format.lower() == 'json':
                    # Export nodes and relationships as JSON
                    nodes_query = "MATCH (n) RETURN n"
                    rels_query = "MATCH ()-[r]->() RETURN r"
                    
                    nodes_result = await session.run(nodes_query)
                    rels_result = await session.run(rels_query)
                    
                    nodes = []
                    async for record in nodes_result:
                        node = dict(record["n"])
                        nodes.append(node)
                    
                    rels = []
                    async for record in rels_result:
                        rel = dict(record["r"])
                        rels.append(rel)
                    
                    export_data = {
                        "nodes": nodes,
                        "relationships": rels,
                        "export_format": "json",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    
                    import json
                    return json.dumps(export_data, indent=2, default=str)
                
                else:
                    # For other formats, we'd need APOC procedures or other tools
                    raise NotImplementedError(f"Export format {format} not implemented")
                    
        except Exception as e:
            logger.error({
                "msg": "Graph export failed",
                "format": format,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            raise
    
    async def run_custom_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Run a custom Cypher query against Neo4j.
        
        Args:
            query: Cypher query string
            params: Query parameters
            
        Returns:
            List of result records
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(query, **(params or {}))
                
                records = []
                async for record in result:
                    records.append(dict(record))
                
                return records
                
        except Exception as e:
            logger.error({
                "msg": "Custom query failed",
                "query": query,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            raise