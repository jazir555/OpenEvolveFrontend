"""
Full-Featured Knowledge Graph Backends

Complete implementations with all CRUD operations:
- delete_knowledge
- update_knowledge
- clear_all

These extend the base backends with full functionality.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .base import (
    KnowledgeGraphBackend,
    KnowledgeEntry,
    SearchResults,
    AnalysisResult,
    GraphStatistics,
    BackendType
)
from .memory_backend import MemoryBackend as InMemoryBackend
from .postgresql_backend import PostgreSQLBackend
from .qdrant_backend import QdrantBackend

logger = logging.getLogger(__name__)


class FullFeaturedInMemoryBackend(InMemoryBackend):
    """
    In-memory backend with complete CRUD operations.
    Extends the base InMemoryBackend with delete, update, and clear.
    """
    
    async def delete_knowledge(self, entry_id: str) -> bool:
        """
        Delete a knowledge entry by ID.
        
        Args:
            entry_id: ID of entry to delete
            
        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if entry_id in self._storage:
                entry = self._storage[entry_id]
                del self._storage[entry_id]
                
                # Also remove from embeddings index
                if entry_id in self._embeddings:
                    del self._embeddings[entry_id]
                
                logger.debug({
                    "msg": "Knowledge entry deleted",
                    "entry_id": entry_id
                })
                return True
            
            return False
    
    async def update_knowledge(
        self,
        entry_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update a knowledge entry by ID.
        
        Args:
            entry_id: ID of entry to update
            updates: Dictionary of fields to update
            
        Returns:
            True if updated, False if not found
        """
        async with self._lock:
            if entry_id not in self._storage:
                return False
            
            entry = self._storage[entry_id]
            
            # Update allowed fields
            allowed_fields = {'content', 'metadata', 'embedding', 'source'}
            
            for field, value in updates.items():
                if field in allowed_fields:
                    setattr(entry, field, value)
            
            # Update embedding index if embedding changed
            if 'embedding' in updates and updates['embedding'] is not None:
                self._embeddings[entry_id] = updates['embedding']
            
            logger.debug({
                "msg": "Knowledge entry updated",
                "entry_id": entry_id,
                "updated_fields": list(updates.keys())
            })
            
            return True
    
    async def clear_all(self) -> int:
        """
        Clear all knowledge from the backend.
        
        WARNING: This is a destructive operation.
        
        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._storage)
            
            self._storage.clear()
            self._embeddings.clear()
            
            logger.warning({
                "msg": "All knowledge cleared from in-memory backend",
                "entries_cleared": count
            })
            
            return count


class FullFeaturedPostgreSQLBackend(PostgreSQLBackend):
    """
    PostgreSQL backend with complete CRUD operations.
    """
    
    async def delete_knowledge(self, entry_id: str) -> bool:
        """Delete a knowledge entry by ID."""
        try:
            from asyncpg import exceptions
            
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM knowledge_entries WHERE id = $1",
                    entry_id
                )
                
                # Check if any row was deleted
                deleted = 'DELETE 1' in result
                
                if deleted:
                    logger.debug({
                        "msg": "Knowledge entry deleted from PostgreSQL",
                        "entry_id": entry_id
                    })
                
                return deleted
                
        except Exception as e:
            logger.error({
                "msg": "Failed to delete knowledge from PostgreSQL",
                "error": str(e),
                "entry_id": entry_id
            })
            raise
    
    async def update_knowledge(
        self,
        entry_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update a knowledge entry by ID."""
        try:
            async with self._pool.acquire() as conn:
                # Build dynamic update query
                set_clauses = []
                values = []
                param_idx = 1
                
                if 'content' in updates:
                    set_clauses.append(f"content = ${param_idx}")
                    values.append(updates['content'])
                    param_idx += 1
                
                if 'metadata' in updates:
                    set_clauses.append(f"metadata = ${param_idx}")
                    values.append(json.dumps(updates['metadata']))
                    param_idx += 1
                
                if 'embedding' in updates:
                    set_clauses.append(f"embedding = ${param_idx}")
                    values.append(updates['embedding'])
                    param_idx += 1
                
                if 'source' in updates:
                    set_clauses.append(f"source = ${param_idx}")
                    values.append(updates['source'])
                    param_idx += 1
                
                if not set_clauses:
                    return True  # Nothing to update
                
                # Add entry_id to values
                values.append(entry_id)
                
                query = f"""
                    UPDATE knowledge_entries 
                    SET {', '.join(set_clauses)}
                    WHERE id = ${param_idx}
                """
                
                result = await conn.execute(query, *values)
                updated = 'UPDATE 1' in result
                
                if updated:
                    logger.debug({
                        "msg": "Knowledge entry updated in PostgreSQL",
                        "entry_id": entry_id
                    })
                
                return updated
                
        except Exception as e:
            logger.error({
                "msg": "Failed to update knowledge in PostgreSQL",
                "error": str(e),
                "entry_id": entry_id
            })
            raise
    
    async def clear_all(self) -> int:
        """Clear all knowledge from PostgreSQL."""
        try:
            async with self._pool.acquire() as conn:
                # Get count before deletion
                count_result = await conn.fetchval(
                    "SELECT COUNT(*) FROM knowledge_entries"
                )
                
                # Delete all entries
                await conn.execute("DELETE FROM knowledge_entries")
                
                logger.warning({
                    "msg": "All knowledge cleared from PostgreSQL",
                    "entries_cleared": count_result
                })
                
                return count_result
                
        except Exception as e:
            logger.error({
                "msg": "Failed to clear knowledge from PostgreSQL",
                "error": str(e)
            })
            raise


class FullFeaturedQdrantBackend(QdrantBackend):
    """
    Qdrant vector database backend with complete CRUD operations.
    """
    
    async def delete_knowledge(self, entry_id: str) -> bool:
        """Delete a knowledge entry by ID from Qdrant."""
        try:
            from qdrant_client.models import PointIdsList
            
            # Delete from Qdrant collection
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[entry_id])
            )
            
            logger.debug({
                "msg": "Knowledge entry deleted from Qdrant",
                "entry_id": entry_id
            })
            
            return True
            
        except Exception as e:
            # Entry might not exist
            logger.error({
                "msg": "Failed to delete from Qdrant",
                "error": str(e),
                "entry_id": entry_id
            })
            return False
    
    async def update_knowledge(
        self,
        entry_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update a knowledge entry by ID in Qdrant."""
        try:
            from qdrant_client.models import PointStruct
            
            # Qdrant doesn't support partial updates easily
            # We need to upsert with new data
            
            # Get existing point
            existing = self._client.retrieve(
                collection_name=self.collection_name,
                ids=[entry_id]
            )
            
            if not existing:
                return False
            
            point = existing[0]
            
            # Build updated payload
            payload = point.payload or {}
            
            if 'content' in updates:
                payload['content'] = updates['content']
            
            if 'metadata' in updates:
                payload['metadata'] = updates['metadata']
            
            if 'source' in updates:
                payload['source'] = updates['source']
            
            # Get vector (use existing or updated)
            vector = updates.get('embedding', point.vector)
            
            # Upsert updated point
            self._client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=entry_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            
            logger.debug({
                "msg": "Knowledge entry updated in Qdrant",
                "entry_id": entry_id
            })
            
            return True
            
        except Exception as e:
            logger.error({
                "msg": "Failed to update knowledge in Qdrant",
                "error": str(e),
                "entry_id": entry_id
            })
            return False
    
    async def clear_all(self) -> int:
        """Clear all knowledge from Qdrant."""
        try:
            # Get count before deletion
            count_result = self._client.count(collection_name=self.collection_name)
            count = count_result.count
            
            # Delete all points
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=None  # Delete all
            )
            
            logger.warning({
                "msg": "All knowledge cleared from Qdrant",
                "entries_cleared": count
            })
            
            return count
            
        except Exception as e:
            logger.error({
                "msg": "Failed to clear knowledge from Qdrant",
                "error": str(e)
            })
            raise


def create_full_featured_backend(
    backend_type: str,
    config: Dict[str, Any]
) -> KnowledgeGraphBackend:
    """
    Factory function to create full-featured backends.
    
    Args:
        backend_type: 'memory', 'postgresql', 'qdrant', 'memgraph'
        config: Backend configuration
        
    Returns:
        Full-featured backend instance
    """
    backend_type = backend_type.lower()
    
    if backend_type == 'memory':
        return FullFeaturedInMemoryBackend(config)
    
    elif backend_type == 'postgresql':
        return FullFeaturedPostgreSQLBackend(config)
    
    elif backend_type == 'qdrant':
        return FullFeaturedQdrantBackend(config)
    
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")


__all__ = [
    'FullFeaturedInMemoryBackend',
    'FullFeaturedPostgreSQLBackend',
    'FullFeaturedQdrantBackend',
    'create_full_featured_backend'
]
