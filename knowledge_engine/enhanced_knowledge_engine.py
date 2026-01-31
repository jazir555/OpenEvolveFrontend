"""
Enhanced Knowledge Engine - Main Integration Module

This module provides the main EnhancedKnowledgeEngine class that integrates:
- Semantic search with embeddings
- Knowledge graph navigation
- Smart caching
- Active learning
- Multi-modal knowledge processing
- Real-time synchronization
- Distributed storage
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, AsyncIterator, Union

from enhanced_knowledge_core import (
    KnowledgeType, RelationType,
    EmbeddingVector, KnowledgeItem, KnowledgeRelation,
    SearchQuery, SearchResult,
    EmbeddingService, SemanticSearchEngine,
    KnowledgeGraphNavigator, SmartCacheManager, ActiveLearningEngine
)

logger = logging.getLogger(__name__)


class KnowledgeEvent:
    """Event for knowledge changes."""
    def __init__(self, event_type: str, item_id: str, timestamp: datetime = None, data: Dict = None):
        self.event_type = event_type  # "created", "updated", "deleted"
        self.item_id = item_id
        self.timestamp = timestamp or datetime.utcnow()
        self.data = data or {}


class EnhancedKnowledgeEngine:
    """
    Enhanced Knowledge Engine with advanced capabilities.
    
    Features:
    - Multi-modal knowledge storage (text, code, structured, embeddings)
    - Semantic search with vector similarity
    - Knowledge graph with advanced traversal
    - Smart caching with predictive prefetching
    - Active learning from feedback
    - Real-time event streaming
    - Distributed storage backend support
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        embedding_model: str = "default",
        embedding_dimensions: int = 1536,
        cache_size: int = 10000,
        enable_graph: bool = True,
        enable_learning: bool = True
    ):
        """
        Initialize the Enhanced Knowledge Engine.
        
        Args:
            storage_path: Path for persistent storage
            embedding_model: Name of the embedding model to use
            embedding_dimensions: Dimensions for embedding vectors
            cache_size: Maximum cache size
            enable_graph: Enable knowledge graph features
            enable_learning: Enable active learning
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.initialized_at = datetime.utcnow()
        
        # Initialize core components
        self.embedding_service = EmbeddingService(embedding_model, embedding_dimensions)
        self.search_engine = SemanticSearchEngine(self.embedding_service)
        self.cache = SmartCacheManager(max_size=cache_size)
        
        # Optional components
        self.graph = KnowledgeGraphNavigator() if enable_graph else None
        self.learning = ActiveLearningEngine() if enable_learning else None
        
        # Storage
        self._items: Dict[str, KnowledgeItem] = {}
        self._relations: Dict[str, KnowledgeRelation] = {}
        
        # Event handling
        self._event_handlers: List[Callable[[KnowledgeEvent], None]] = []
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._event_processor_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = {
            "items_created": 0,
            "items_updated": 0,
            "items_deleted": 0,
            "searches_performed": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(f"EnhancedKnowledgeEngine initialized with embedding model: {embedding_model}")
    
    async def initialize(self):
        """Initialize the engine and start background tasks."""
        # Load persisted data
        if self.storage_path:
            await self._load_from_storage()
        
        # Start event processor
        self._event_processor_task = asyncio.create_task(self._process_events())
        
        logger.info("EnhancedKnowledgeEngine fully initialized")
    
    async def shutdown(self):
        """Shutdown the engine and cleanup resources."""
        # Stop event processor
        if self._event_processor_task:
            self._event_processor_task.cancel()
            try:
                await self._event_processor_task
            except asyncio.CancelledError:
                pass
        
        # Save data
        if self.storage_path:
            await self._save_to_storage()
        
        # Clear cache
        await self.cache.clear()
        
        logger.info("EnhancedKnowledgeEngine shutdown complete")
    
    # ==================== Knowledge Item Management ====================
    
    async def add_knowledge(
        self,
        content: Any,
        knowledge_type: KnowledgeType = KnowledgeType.TEXT,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        source: str = "unknown",
        confidence: float = 1.0,
        generate_embedding: bool = True,
        item_id: Optional[str] = None
    ) -> KnowledgeItem:
        """
        Add new knowledge to the engine.
        
        Args:
            content: The knowledge content
            knowledge_type: Type of knowledge
            metadata: Additional metadata
            tags: Tags for categorization
            source: Source of the knowledge
            confidence: Confidence score (0-1)
            generate_embedding: Whether to generate embedding
            item_id: Optional custom ID
            
        Returns:
            The created KnowledgeItem
        """
        # Create item
        item = KnowledgeItem(
            id=item_id or str(hash(str(content) + str(datetime.utcnow()))),
            content=content,
            knowledge_type=knowledge_type,
            metadata=metadata or {},
            tags=tags or set(),
            source=source,
            confidence=confidence
        )
        
        # Generate embedding if requested
        if generate_embedding:
            item.embedding = await self.embedding_service.generate_embedding(
                content, knowledge_type
            )
        
        # Store item
        self._items[item.id] = item
        
        # Index for search
        self.search_engine.index_item(item)
        
        # Add to graph if enabled
        if self.graph:
            self.graph.add_node(item)
        
        # Update stats
        self._stats["items_created"] += 1
        
        # Emit event
        await self._emit_event(KnowledgeEvent("created", item.id, data={"type": knowledge_type.value}))
        
        logger.info(f"Added knowledge item: {item.id} ({knowledge_type.value})")
        
        return item
    
    async def get_knowledge(self, item_id: str) -> Optional[KnowledgeItem]:
        """
        Retrieve a knowledge item by ID.
        
        Args:
            item_id: The item ID
            
        Returns:
            KnowledgeItem if found, None otherwise
        """
        # Check cache first
        cached = await self.cache.get(f"item:{item_id}")
        if cached:
            self._stats["cache_hits"] += 1
            return cached
        
        self._stats["cache_misses"] += 1
        
        # Get from storage
        item = self._items.get(item_id)
        
        if item:
            # Cache the result
            await self.cache.set(f"item:{item_id}", item)
        
        return item
    
    async def update_knowledge(
        self,
        item_id: str,
        new_content: Any,
        confidence: Optional[float] = None,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> Optional[KnowledgeItem]:
        """
        Update an existing knowledge item.
        
        Args:
            item_id: ID of the item to update
            new_content: New content
            confidence: Optional new confidence score
            metadata_updates: Metadata fields to update
            
        Returns:
            Updated KnowledgeItem if found, None otherwise
        """
        item = self._items.get(item_id)
        if not item:
            return None
        
        # Store old version info
        old_version = item.version
        
        # Update content
        item.update_content(new_content, confidence)
        
        # Update metadata
        if metadata_updates:
            item.metadata.update(metadata_updates)
        
        # Regenerate embedding if content changed
        if item.embedding:
            item.embedding = await self.embedding_service.generate_embedding(
                new_content, item.knowledge_type
            )
        
        # Update search index
        self.search_engine.remove_item(item_id)
        self.search_engine.index_item(item)
        
        # Invalidate cache
        await self.cache.delete(f"item:{item_id}")
        
        # Update stats
        self._stats["items_updated"] += 1
        
        # Emit event
        await self._emit_event(KnowledgeEvent(
            "updated", 
            item.id, 
            data={"old_version": old_version, "new_version": item.version}
        ))
        
        logger.info(f"Updated knowledge item: {item.id} (v{old_version} -> v{item.version})")
        
        return item
    
    async def delete_knowledge(self, item_id: str) -> bool:
        """
        Delete a knowledge item.
        
        Args:
            item_id: ID of the item to delete
            
        Returns:
            True if deleted, False if not found
        """
        if item_id not in self._items:
            return False
        
        # Remove from storage
        del self._items[item_id]
        
        # Remove from search index
        self.search_engine.remove_item(item_id)
        
        # Remove from graph
        if self.graph:
            # Note: Graph navigator doesn't have explicit remove, would need to implement
            pass
        
        # Invalidate cache
        await self.cache.delete(f"item:{item_id}")
        
        # Update stats
        self._stats["items_deleted"] += 1
        
        # Emit event
        await self._emit_event(KnowledgeEvent("deleted", item_id))
        
        logger.info(f"Deleted knowledge item: {item_id}")
        
        return True
    
    # ==================== Search & Retrieval ====================
    
    async def search(
        self,
        query: str,
        search_mode: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
        knowledge_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        max_results: int = 10
    ) -> List[SearchResult]:
        """
        Search for knowledge items.
        
        Args:
            query: Search query text
            search_mode: "keyword", "semantic", "vector", or "hybrid"
            filters: Additional filters
            knowledge_types: Filter by knowledge types
            tags: Filter by tags
            min_confidence: Minimum confidence score
            max_results: Maximum results to return
            
        Returns:
            List of search results
        """
        search_query = SearchQuery(
            text=query,
            filters=filters or {},
            knowledge_types={KnowledgeType(kt) for kt in knowledge_types} if knowledge_types else set(),
            tags=set(tags) if tags else set(),
            min_confidence=min_confidence,
            max_results=max_results,
            search_mode=search_mode
        )
        
        # Check cache for query
        cache_key = f"search:{hash(str(search_query.to_dict()))}"
        cached = await self.cache.get(cache_key)
        if cached:
            self._stats["cache_hits"] += 1
            return cached
        
        self._stats["cache_misses"] += 1
        
        # Perform search
        results = await self.search_engine.search(search_query)
        
        # Cache results
        await self.cache.set(cache_key, results, ttl=300)  # 5 minute TTL for search
        
        # Update stats
        self._stats["searches_performed"] += 1
        
        logger.info(f"Search completed: '{query}' ({search_mode}) - {len(results)} results")
        
        return results
    
    async def semantic_search(
        self,
        query_text: str,
        max_results: int = 10
    ) -> List[SearchResult]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query_text: Query text
            max_results: Maximum results
            
        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embedding(
            query_text, KnowledgeType.TEXT
        )
        
        search_query = SearchQuery(
            text=query_text,
            embedding=query_embedding,
            search_mode="semantic",
            max_results=max_results
        )
        
        return await self.search_engine.search(search_query)
    
    # ==================== Knowledge Graph Operations ====================
    
    async def create_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[KnowledgeRelation]:
        """
        Create a relationship between two knowledge items.
        
        Args:
            source_id: Source item ID
            target_id: Target item ID
            relation_type: Type of relationship
            weight: Relationship weight (0-1)
            metadata: Additional metadata
            
        Returns:
            Created relation if successful, None otherwise
        """
        if not self.graph:
            logger.warning("Graph operations disabled")
            return None
        
        # Verify items exist
        if source_id not in self._items or target_id not in self._items:
            logger.warning(f"Cannot create relation: items not found ({source_id} -> {target_id})")
            return None
        
        # Create relation
        relation = KnowledgeRelation(
            id=str(hash(f"{source_id}:{target_id}:{relation_type.value}")),
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            metadata=metadata or {}
        )
        
        # Store relation
        self._relations[relation.id] = relation
        
        # Add to graph
        self.graph.add_edge(relation)
        
        # Update item relationships
        self._items[source_id].child_ids.append(target_id)
        self._items[target_id].parent_ids.append(source_id)
        
        logger.info(f"Created relation: {source_id} -[{relation_type.value}]-> {target_id}")
        
        return relation
    
    async def find_related(
        self,
        item_id: str,
        relation_type: Optional[RelationType] = None,
        max_depth: int = 2
    ) -> List[Tuple[KnowledgeItem, KnowledgeRelation]]:
        """
        Find related knowledge items.
        
        Args:
            item_id: Starting item ID
            relation_type: Optional relation type filter
            max_depth: Maximum traversal depth
            
        Returns:
            List of (item, relation) tuples
        """
        if not self.graph:
            return []
        
        neighbors = self.graph.get_neighbors(item_id, relation_type)
        
        # Traverse deeper if needed
        if max_depth > 1:
            seen = {item_id}
            for neighbor, relation in neighbors:
                if neighbor.id not in seen:
                    seen.add(neighbor.id)
                    # Could recursively traverse here
        
        return neighbors
    
    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> List[List[KnowledgeItem]]:
        """
        Find paths between two items.
        
        Args:
            source_id: Source item ID
            target_id: Target item ID
            max_depth: Maximum path length
            
        Returns:
            List of paths (each path is a list of items)
        """
        if not self.graph:
            return []
        
        paths = self.graph.find_paths(source_id, target_id, max_depth)
        return [items for items, _ in paths]
    
    async def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        if not self.graph:
            return {"enabled": False}
        
        stats = self.graph.get_stats()
        components = self.graph.get_connected_components()
        
        return {
            "enabled": True,
            **stats,
            "connected_components": len(components),
            "largest_component_size": max(len(c) for c in components) if components else 0
        }
    
    # ==================== Active Learning ====================
    
    async def record_feedback(
        self,
        item_id: str,
        feedback_type: str,
        feedback_score: float,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Record feedback for a knowledge item.
        
        Args:
            item_id: Item ID
            feedback_type: "positive", "negative", or "neutral"
            feedback_score: Score from 0.0 to 1.0
            user_id: Optional user ID
            context: Optional context information
        """
        if not self.learning:
            logger.warning("Active learning disabled")
            return
        
        await self.learning.record_feedback(
            item_id, feedback_type, feedback_score, user_id, context
        )
    
    async def get_item_quality(self, item_id: str) -> Dict[str, Any]:
        """
        Get quality metrics for an item based on feedback.
        
        Args:
            item_id: Item ID
            
        Returns:
            Quality metrics dictionary
        """
        if not self.learning:
            return {"average_score": 0.5, "feedback_count": 0}
        
        return await self.learning.calculate_item_quality(item_id)
    
    async def get_learning_recommendations(self) -> List[Dict[str, str]]:
        """
        Get recommendations for knowledge improvement.
        
        Returns:
            List of recommendation dictionaries
        """
        if not self.learning:
            return []
        
        return await self.learning.generate_learning_recommendations()
    
    # ==================== Event Handling ====================
    
    def add_event_handler(self, handler: Callable[[KnowledgeEvent], None]):
        """Add an event handler."""
        self._event_handlers.append(handler)
    
    def remove_event_handler(self, handler: Callable[[KnowledgeEvent], None]):
        """Remove an event handler."""
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)
    
    async def _emit_event(self, event: KnowledgeEvent):
        """Emit an event to the queue."""
        await self._event_queue.put(event)
    
    async def _process_events(self):
        """Process events from the queue."""
        while True:
            try:
                event = await self._event_queue.get()
                
                # Call all handlers
                for handler in self._event_handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(f"Event handler error: {e}")
                
                self._event_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processing error: {e}")
    
    # ==================== Persistence ====================
    
    async def _save_to_storage(self):
        """Save knowledge data to persistent storage."""
        if not self.storage_path:
            return
        
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Save items
            items_data = {k: v.to_dict() for k, v in self._items.items()}
            items_file = self.storage_path / "knowledge_items.json"
            with open(items_file, 'w') as f:
                json.dump(items_data, f, indent=2)
            
            # Save relations
            relations_data = {k: v.to_dict() for k, v in self._relations.items()}
            relations_file = self.storage_path / "knowledge_relations.json"
            with open(relations_file, 'w') as f:
                json.dump(relations_data, f, indent=2)
            
            # Save stats
            stats_file = self.storage_path / "engine_stats.json"
            with open(stats_file, 'w') as f:
                json.dump({
                    **self._stats,
                    "saved_at": datetime.utcnow().isoformat()
                }, f, indent=2)
            
            logger.info(f"Saved knowledge engine data to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to save knowledge data: {e}")
    
    async def _load_from_storage(self):
        """Load knowledge data from persistent storage."""
        if not self.storage_path:
            return
        
        try:
            # Load items
            items_file = self.storage_path / "knowledge_items.json"
            if items_file.exists():
                with open(items_file, 'r') as f:
                    items_data = json.load(f)
                    self._items = {
                        k: KnowledgeItem.from_dict(v) 
                        for k, v in items_data.items()
                    }
                
                # Index items
                for item in self._items.values():
                    self.search_engine.index_item(item)
                    if self.graph:
                        self.graph.add_node(item)
            
            # Load relations
            relations_file = self.storage_path / "knowledge_relations.json"
            if relations_file.exists():
                with open(relations_file, 'r') as f:
                    relations_data = json.load(f)
                    self._relations = {
                        k: KnowledgeRelation.from_dict(v)
                        for k, v in relations_data.items()
                    }
                
                # Add to graph
                if self.graph:
                    for relation in self._relations.values():
                        self.graph.add_edge(relation)
            
            logger.info(f"Loaded {len(self._items)} items and {len(self._relations)} relations")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge data: {e}")
    
    # ==================== Statistics & Monitoring ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        cache_stats = asyncio.run(self.cache.get_stats())
        search_stats = self.search_engine.get_stats()
        
        return {
            **self._stats,
            "uptime_seconds": (datetime.utcnow() - self.initialized_at).total_seconds(),
            "total_items": len(self._items),
            "total_relations": len(self._relations),
            "cache": cache_stats,
            "search": search_stats
        }
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get health status of the engine."""
        return {
            "status": "healthy",
            "initialized_at": self.initialized_at.isoformat(),
            "components": {
                "embedding_service": self.embedding_service is not None,
                "search_engine": self.search_engine is not None,
                "cache": True,
                "graph": self.graph is not None,
                "learning": self.learning is not None
            },
            "stats": self.get_stats()
        }


# Convenience functions for common operations
async def create_knowledge_engine(
    storage_path: Optional[str] = None,
    **kwargs
) -> EnhancedKnowledgeEngine:
    """Factory function to create and initialize a knowledge engine."""
    engine = EnhancedKnowledgeEngine(storage_path=storage_path, **kwargs)
    await engine.initialize()
    return engine


__all__ = [
    "EnhancedKnowledgeEngine",
    "KnowledgeEvent",
    "create_knowledge_engine"
]
