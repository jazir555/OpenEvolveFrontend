"""
Graph Adapter: Bridge Arbor Graph to Knowledge Engine

Integrates Arbor's code graph with the Knowledge Engine's unified graph,
enabling queries that span both code and general knowledge.

Following CLAUDE.md principles:
- IDEMPOTENCY: Safe to re-import same graph
- ZERO TRUST: Validate all conversions
- STRUCTURED LOGGING: Track all operations
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple

from knowledge_engine.core.entity_knowledge_graph import EntityKnowledgeGraph
from knowledge_engine.schemas.base import Entity, Relationship

from .client import ArborClient
from .schema_mapping import ArborSchemaMapper
from .exceptions import ArborSyncError, ArborSchemaError

logger = logging.getLogger(__name__)


@dataclass
class MergeResult:
    """Result of merging Arbor graph into Knowledge Engine."""
    
    success: bool = False
    nodes_imported: int = 0
    nodes_updated: int = 0
    nodes_skipped: int = 0
    edges_imported: int = 0
    edges_updated: int = 0
    edges_skipped: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "nodes_imported": self.nodes_imported,
            "nodes_updated": self.nodes_updated,
            "nodes_skipped": self.nodes_skipped,
            "edges_imported": self.edges_imported,
            "edges_updated": self.edges_updated,
            "edges_skipped": self.edges_skipped,
            "errors": self.errors,
            "duration_seconds": self.duration_seconds
        }


@dataclass
class GraphDelta:
    """Represents incremental changes to the graph."""
    
    added_nodes: List[Entity] = field(default_factory=list)
    updated_nodes: List[Entity] = field(default_factory=list)
    removed_nodes: List[str] = field(default_factory=list)  # IDs
    added_edges: List[Relationship] = field(default_factory=list)
    removed_edges: List[Tuple[str, str]] = field(default_factory=list)  # (from, to) pairs
    
    @property
    def has_changes(self) -> bool:
        """Check if delta contains any changes."""
        return bool(
            self.added_nodes or self.updated_nodes or self.removed_nodes or
            self.added_edges or self.removed_edges
        )


class ArborGraphAdapter:
    """
    Adapter to integrate Arbor graph with Knowledge Engine.
    
    Provides:
    - Full graph import from Arbor
    - Incremental delta synchronization
    - Bidirectional query bridging
    - Conflict resolution
    
    Example:
        graph = EntityKnowledgeGraph()
        client = ArborClient()
        adapter = ArborGraphAdapter(graph, client)
        
        # Full import
        arbor_data = await client.export_graph()
        result = await adapter.merge_arbor_graph(arbor_data)
        
        # Incremental update
        await client.subscribe_changes(adapter.apply_delta)
    """
    
    def __init__(
        self,
        knowledge_graph: EntityKnowledgeGraph,
        arbor_client: Optional[ArborClient] = None,
        storage_prefix: str = "arbor"
    ):
        """
        Initialize graph adapter.
        
        Args:
            knowledge_graph: Knowledge Engine graph to integrate with
            arbor_client: Optional Arbor client for live sync
            storage_prefix: Prefix for stored entity IDs
        """
        self.kg = knowledge_graph
        self.client = arbor_client
        self.mapper = ArborSchemaMapper(storage_prefix=storage_prefix)
        self.storage_prefix = storage_prefix
        
        # Track imported Arbor IDs for sync
        self._imported_arbor_ids: Set[str] = set()
        self._sync_lock = asyncio.Lock()
        
        logger.info({
            "msg": "ArborGraphAdapter initialized",
            "storage_prefix": storage_prefix
        })
    
    async def merge_arbor_graph(
        self,
        arbor_graph: Dict[str, Any],
        batch_size: int = 1000
    ) -> MergeResult:
        """
        Merge full Arbor graph into Knowledge Engine.
        
        This operation is idempotent - re-importing the same graph
        will update existing entities rather than creating duplicates.
        
        Args:
            arbor_graph: Complete graph export from Arbor
            batch_size: Number of entities to process per batch
            
        Returns:
            MergeResult with statistics
        """
        import time
        start_time = time.time()
        
        result = MergeResult()
        
        try:
            # Convert Arbor graph to KE entities
            entities, relationships = self.mapper.convert_arbor_graph(arbor_graph)
            
            logger.info({
                "msg": "Starting graph merge",
                "entities": len(entities),
                "relationships": len(relationships)
            })
            
            # Process entities in batches
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                batch_result = await self._merge_entity_batch(batch)
                
                result.nodes_imported += batch_result["imported"]
                result.nodes_updated += batch_result["updated"]
                result.nodes_skipped += batch_result["skipped"]
                result.errors.extend(batch_result["errors"])
                
                logger.debug(f"Processed entity batch {i // batch_size + 1}")
            
            # Process relationships in batches
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i + batch_size]
                batch_result = await self._merge_relationship_batch(batch)
                
                result.edges_imported += batch_result["imported"]
                result.edges_updated += batch_result["updated"]
                result.edges_skipped += batch_result["skipped"]
                result.errors.extend(batch_result["errors"])
                
                logger.debug(f"Processed relationship batch {i // batch_size + 1}")
            
            # Track imported IDs
            self._imported_arbor_ids.update(
                entity.properties.get("arbor_id")
                for entity in entities
                if "arbor_id" in entity.properties
            )
            
            result.success = True
            result.duration_seconds = time.time() - start_time
            
            logger.info({
                "msg": "Graph merge completed",
                "result": result.to_dict()
            })
            
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            result.duration_seconds = time.time() - start_time
            
            logger.error({
                "msg": "Graph merge failed",
                "error": str(e)
            })
        
        return result
    
    async def _merge_entity_batch(self, entities: List[Entity]) -> Dict[str, Any]:
        """
        Merge a batch of entities into the knowledge graph.
        
        Args:
            entities: List of entities to merge
            
        Returns:
            Statistics dictionary
        """
        imported = 0
        updated = 0
        skipped = 0
        errors = []
        
        for entity in entities:
            try:
                # Check if entity already exists
                existing = await self.kg.get_entity_async(entity.entity_id)
                
                if existing:
                    # Update existing entity
                    # Note: EntityKnowledgeGraph doesn't have direct update,
                    # so we re-add which merges properties
                    await self.kg.add_entity_async(
                        name=entity.entity_id,
                        entity_type=entity.entity_type,
                        attributes=entity.properties
                    )
                    updated += 1
                else:
                    # Add new entity
                    await self.kg.add_entity_async(
                        name=entity.entity_id,
                        entity_type=entity.entity_type,
                        attributes=entity.properties
                    )
                    imported += 1
                    
            except Exception as e:
                errors.append(f"Entity {entity.entity_id}: {str(e)}")
                skipped += 1
        
        return {
            "imported": imported,
            "updated": updated,
            "skipped": skipped,
            "errors": errors
        }
    
    async def _merge_relationship_batch(
        self,
        relationships: List[Relationship]
    ) -> Dict[str, Any]:
        """
        Merge a batch of relationships into the knowledge graph.
        
        Args:
            relationships: List of relationships to merge
            
        Returns:
            Statistics dictionary
        """
        imported = 0
        updated = 0
        skipped = 0
        errors = []
        
        for rel in relationships:
            try:
                # Add relationship (idempotent)
                success = await self.kg.add_relationship_async(
                    source=rel.source_id,
                    target=rel.target_id,
                    relation_type=rel.relationship_type,
                    attributes=rel.properties
                )
                
                if success:
                    imported += 1
                else:
                    # May already exist
                    updated += 1
                    
            except Exception as e:
                errors.append(f"Relationship {rel.source_id}->{rel.target_id}: {str(e)}")
                skipped += 1
        
        return {
            "imported": imported,
            "updated": updated,
            "skipped": skipped,
            "errors": errors
        }
    
    async def apply_delta(self, delta: GraphDelta) -> None:
        """
        Apply incremental graph changes.
        
        Args:
            delta: Graph changes to apply
        """
        async with self._sync_lock:
            logger.info({
                "msg": "Applying graph delta",
                "added_nodes": len(delta.added_nodes),
                "updated_nodes": len(delta.updated_nodes),
                "removed_nodes": len(delta.removed_nodes),
                "added_edges": len(delta.added_edges)
            })
            
            # Add new nodes
            for entity in delta.added_nodes:
                await self.kg.add_entity_async(
                    name=entity.entity_id,
                    entity_type=entity.entity_type,
                    attributes=entity.properties
                )
                self._imported_arbor_ids.add(
                    entity.properties.get("arbor_id", entity.entity_id)
                )
            
            # Update existing nodes
            for entity in delta.updated_nodes:
                await self.kg.add_entity_async(
                    name=entity.entity_id,
                    entity_type=entity.entity_type,
                    attributes=entity.properties
                )
            
            # Remove deleted nodes
            for node_id in delta.removed_nodes:
                # Note: EntityKnowledgeGraph may not have remove functionality
                # Mark as deleted in properties instead
                entity = await self.kg.get_entity_async(node_id)
                if entity:
                    entity["properties"]["_deleted"] = True
                    entity["properties"]["_deleted_at"] = datetime.utcnow().isoformat()
            
            # Add new edges
            for rel in delta.added_edges:
                await self.kg.add_relationship_async(
                    source=rel.source_id,
                    target=rel.target_id,
                    relation_type=rel.relationship_type,
                    attributes=rel.properties
                )
            
            logger.info("Graph delta applied successfully")
    
    async def handle_arbor_change_event(self, event: Dict[str, Any]) -> None:
        """
        Handle real-time change event from Arbor.
        
        Args:
            event: Change event from Arbor file watcher
        """
        try:
            event_type = event.get("type")
            
            if event_type == "file_added":
                # Parse new file and add nodes
                nodes = event.get("nodes", [])
                edges = event.get("edges", [])
                
                delta = GraphDelta()
                for node in nodes:
                    delta.added_nodes.append(self.mapper.convert_arbor_node(node))
                for edge in edges:
                    delta.added_edges.append(self.mapper.convert_arbor_edge(edge))
                
                await self.apply_delta(delta)
                
            elif event_type == "file_modified":
                # Handle file modification
                old_nodes = event.get("old_nodes", [])
                new_nodes = event.get("new_nodes", [])
                
                delta = GraphDelta()
                
                # Mark old nodes as removed
                for node in old_nodes:
                    delta.removed_nodes.append(self.mapper.namespace_id(node["id"]))
                
                # Add new nodes
                for node in new_nodes:
                    delta.added_nodes.append(self.mapper.convert_arbor_node(node))
                
                await self.apply_delta(delta)
                
            elif event_type == "file_removed":
                # Mark nodes as deleted
                node_ids = event.get("node_ids", [])
                delta = GraphDelta(
                    removed_nodes=[self.mapper.namespace_id(nid) for nid in node_ids]
                )
                await self.apply_delta(delta)
            
            else:
                logger.warning(f"Unknown change event type: {event_type}")
                
        except Exception as e:
            logger.error(f"Failed to handle change event: {e}")
            raise ArborSyncError(
                sync_type="incremental",
                message=f"Change event handling failed: {str(e)}"
            )
    
    def create_delta_from_arbor_export(
        self,
        old_graph: Optional[Dict[str, Any]],
        new_graph: Dict[str, Any]
    ) -> GraphDelta:
        """
        Compute delta between two Arbor graph exports.
        
        Args:
            old_graph: Previous graph state (None for full import)
            new_graph: New graph state
            
        Returns:
            GraphDelta representing changes
        """
        delta = GraphDelta()
        
        if old_graph is None:
            # Full import - everything is new
            for node in new_graph.get("nodes", []):
                delta.added_nodes.append(self.mapper.convert_arbor_node(node))
            for edge in new_graph.get("edges", []):
                delta.added_edges.append(self.mapper.convert_arbor_edge(edge))
            return delta
        
        # Build lookup maps
        old_nodes = {n["id"]: n for n in old_graph.get("nodes", [])}
        new_nodes = {n["id"]: n for n in new_graph.get("nodes", [])}
        old_edges = {(e["from"], e["to"]): e for e in old_graph.get("edges", [])}
        new_edges = {(e["from"], e["to"]): e for e in new_graph.get("edges", [])}
        
        # Find node changes
        for node_id, node in new_nodes.items():
            if node_id not in old_nodes:
                delta.added_nodes.append(self.mapper.convert_arbor_node(node))
            elif node != old_nodes[node_id]:
                delta.updated_nodes.append(self.mapper.convert_arbor_node(node))
        
        for node_id in old_nodes:
            if node_id not in new_nodes:
                delta.removed_nodes.append(self.mapper.namespace_id(node_id))
        
        # Find edge changes
        for edge_key, edge in new_edges.items():
            if edge_key not in old_edges:
                delta.added_edges.append(self.mapper.convert_arbor_edge(edge))
        
        for edge_key in old_edges:
            if edge_key not in new_edges:
                delta.removed_edges.append(edge_key)
        
        return delta
    
    async def get_code_entities(
        self,
        file_path: Optional[str] = None,
        entity_type: Optional[str] = None,
        language: Optional[str] = None
    ) -> List[Entity]:
        """
        Query code entities from the integrated graph.
        
        Args:
            file_path: Filter by file path
            entity_type: Filter by entity type
            language: Filter by programming language
            
        Returns:
            List of matching entities
        """
        # Query from Knowledge Engine
        all_entities = self.kg._entities.values()
        
        results = []
        for entity in all_entities:
            # Only Arbor entities
            if not entity.entity_id.startswith(self.storage_prefix + ":"):
                continue
            
            # Apply filters
            if file_path:
                entity_file = entity.properties.get("file_path", "")
                if file_path not in entity_file:
                    continue
            
            if entity_type and entity.entity_type != entity_type:
                continue
            
            if language:
                entity_lang = entity.metadata.get("language", "")
                if entity_lang != language:
                    continue
            
            results.append(entity)
        
        return results
    
    def get_imported_arbor_ids(self) -> Set[str]:
        """Get set of imported Arbor node IDs."""
        return self._imported_arbor_ids.copy()
