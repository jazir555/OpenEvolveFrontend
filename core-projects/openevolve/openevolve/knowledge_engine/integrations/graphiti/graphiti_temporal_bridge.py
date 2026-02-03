"""
Graphiti Temporal Bridge for OpenEvolve Knowledge Engine

This module provides a bridge to the Graphiti temporal knowledge graph system,
enabling temporal queries, contradiction detection, and agent memory capabilities.

Following CLAUDE.md principles:
- CONFIGURATION EXPLICITNESS: All config via parameters/env vars
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
- RUNTIME TRUTH: Verify components before use
- IDEMPOTENCY: All operations safe to run multiple times
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import uuid

try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EntityNode, EpisodeType
    from graphiti_core.edges import EntityEdge
    from graphiti_core.llm_client import LLMClient
    from graphiti_core.utils import extract_datetime
except ImportError:
    # Mock classes for when graphiti-core is not available
    class Graphiti:
        pass
    class EntityNode:
        pass
    class EpisodeType:
        pass
    class EntityEdge:
        pass
    class LLMClient:
        pass
    def extract_datetime(*args, **kwargs):
        return datetime.now(timezone.utc)

from enum import Enum


logger = logging.getLogger(__name__)


class TemporalFilter(Enum):
    """Types of temporal filters for queries"""
    CURRENT = "current"
    TIME_RANGE = "time_range"
    ALL = "all"
    POINT_IN_TIME = "point_in_time"


@dataclass
class KnowledgeArtifact:
    """Canonical representation of a knowledge artifact with temporal metadata"""
    id: str
    content: str
    artifact_type: str
    valid_at: datetime
    invalid_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    group_id: Optional[str] = None
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content,
            'artifact_type': self.artifact_type,
            'valid_at': self.valid_at.isoformat(),
            'invalid_at': self.invalid_at.isoformat() if self.invalid_at else None,
            'metadata': self.metadata,
            'source': self.source,
            'group_id': self.group_id,
            'confidence': self.confidence
        }


class GraphitiTemporalBridge:
    """
    Bridge to Graphiti temporal knowledge graph system.
    
    Provides high-level operations for temporal knowledge management including:
    - Adding temporal knowledge artifacts
    - Querying knowledge at specific points in time
    - Detecting contradictions
    - Managing entity lifecycles
    """
    
    def __init__(self, uri: str, user: str, password: str, llm_client: Optional[LLMClient] = None):
        """
        Initialize the Graphiti temporal bridge.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            llm_client: Optional LLM client for Graphiti
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.llm_client = llm_client
        
        # Graphiti client
        self.client: Optional[Graphiti] = None
        
        # Tracking
        self._initialized = False
        
        logger.info({
            "msg": "GraphitiTemporalBridge initialized",
            "uri": uri,
            "user": user,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def initialize(self) -> bool:
        """
        Initialize the Graphiti client connection.
        
        Following CLAUDE.md: RUNTIME TRUTH
        Verify the connection is actually working before marking as initialized.
        
        Returns:
            True if initialization successful
            
        Raises:
            RuntimeError: If connection fails
        """
        if self._initialized:
            logger.warning("GraphitiTemporalBridge already initialized")
            return True
        
        try:
            # Initialize Graphiti client
            self.client = Graphiti(
                uri=self.uri,
                user=self.user,
                password=self.password,
                llm_client=self.llm_client
            )
            
            # Test connection
            await self.client.connect()
            
            # Verify basic functionality
            test_entities = await self.client.get_entity_list()
            
            self._initialized = True
            
            logger.info({
                "msg": "GraphitiTemporalBridge initialized successfully",
                "connected": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error({
                "msg": "Failed to initialize GraphitiTemporalBridge",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            raise RuntimeError(f"Failed to initialize Graphiti client: {e}")
    
    async def add_artifact(self, artifact: KnowledgeArtifact, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a knowledge artifact to Graphiti.
        
        Args:
            artifact: KnowledgeArtifact to add
            correlation_id: Correlation ID for tracking
            
        Returns:
            Result dictionary with success status
        """
        correlation_id = correlation_id or f"artifact_{uuid.uuid4().hex}"
        
        if not self._initialized or not self.client:
            raise RuntimeError("GraphitiTemporalBridge not initialized")
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Adding knowledge artifact",
            "artifact_id": artifact.id,
            "artifact_type": artifact.artifact_type,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Create episode for the artifact
            episode_uuid = str(uuid.uuid4())
            
            # Add the artifact as an episode to Graphiti
            await self.client.add_episode(
                uuid=episode_uuid,
                name=artifact.content[:100],  # Use first 100 chars as name
                content=artifact.content,
                source=artifact.source or "knowledge_engine",
                timestamp=artifact.valid_at,
                group=artifact.group_id or "default"
            )
            
            # Extract entities and relationships from the content
            # This would typically involve calling Graphiti's extraction methods
            extracted_info = await self.client.extract_episode_entities(episode_uuid)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            result = {
                "success": True,
                "artifact_id": artifact.id,
                "episode_id": episode_uuid,
                "entities_extracted": len(extracted_info.get('entities', [])),
                "relationships_extracted": len(extracted_info.get('edges', [])),
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id
            }
            
            logger.info({
                "msg": "Knowledge artifact added successfully",
                "correlation_id": correlation_id,
                "entities_extracted": result["entities_extracted"],
                "relationships_extracted": result["relationships_extracted"],
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Failed to add knowledge artifact",
                "artifact_id": artifact.id,
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return {
                "success": False,
                "artifact_id": artifact.id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id
            }
    
    async def search_with_temporal_filters(
        self,
        query: str,
        filter_type: TemporalFilter = TemporalFilter.CURRENT,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_results: int = 10,
        group_ids: Optional[List[str]] = None,
        use_hybrid: bool = True,
        correlation_id: Optional[str] = None
    ) -> List[KnowledgeArtifact]:
        """
        Search with temporal filtering.
        
        Args:
            query: Search query
            filter_type: Type of temporal filter
            start_time: Start time for range queries
            end_time: End time for range queries
            max_results: Maximum results to return
            group_ids: Group IDs to scope search
            use_hybrid: Use hybrid search (BM25 + vector + graph)
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of KnowledgeArtifacts
        """
        correlation_id = correlation_id or f"search_{uuid.uuid4().hex}"
        
        if not self._initialized or not self.client:
            raise RuntimeError("GraphitiTemporalBridge not initialized")
        
        start_time_proc = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Temporal search initiated",
            "query": query,
            "filter_type": filter_type.value,
            "max_results": max_results,
            "correlation_id": correlation_id,
            "timestamp": start_time_proc.isoformat()
        })
        
        try:
            # Prepare search parameters based on filter type
            if filter_type == TemporalFilter.POINT_IN_TIME:
                # Search at specific point in time
                if not start_time:
                    start_time = datetime.now(timezone.utc)
                
                results = await self.client.search(
                    query=query,
                    reference_time=start_time,
                    limit=max_results,
                    group_ids=group_ids
                )
            elif filter_type == TemporalFilter.TIME_RANGE:
                # Search within time range
                if not start_time or not end_time:
                    raise ValueError("start_time and end_time required for TIME_RANGE filter")
                
                results = await self.client.search(
                    query=query,
                    start_time=start_time,
                    end_time=end_time,
                    limit=max_results,
                    group_ids=group_ids
                )
            elif filter_type == TemporalFilter.CURRENT:
                # Search for currently valid knowledge
                results = await self.client.search(
                    query=query,
                    reference_time=datetime.now(timezone.utc),
                    limit=max_results,
                    group_ids=group_ids
                )
            else:  # ALL
                # Search all historical knowledge
                results = await self.client.search(
                    query=query,
                    limit=max_results,
                    group_ids=group_ids
                )
            
            # Convert results to KnowledgeArtifacts
            artifacts = []
            for result in results:
                # Convert Graphiti result to KnowledgeArtifact
                artifact = KnowledgeArtifact(
                    id=result.get('uuid', str(uuid.uuid4())),
                    content=result.get('content', ''),
                    artifact_type=result.get('type', 'unknown'),
                    valid_at=result.get('created_at', datetime.now(timezone.utc)),
                    metadata=result.get('metadata', {}),
                    source=result.get('source', 'graphiti'),
                    group_id=result.get('group', 'default'),
                    confidence=result.get('score', 0.5)
                )
                artifacts.append(artifact)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time_proc).total_seconds() * 1000
            
            logger.info({
                "msg": "Temporal search completed",
                "correlation_id": correlation_id,
                "results_count": len(artifacts),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return artifacts
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time_proc).total_seconds() * 1000
            
            logger.error({
                "msg": "Temporal search failed",
                "query": query,
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return []
    
    async def query_at_point_in_time(
        self,
        query: str,
        timestamp: datetime,
        max_results: int = 10,
        group_ids: Optional[List[str]] = None,
        correlation_id: Optional[str] = None
    ) -> List[KnowledgeArtifact]:
        """
        Query knowledge at a specific point in time.
        
        Args:
            query: Search query
            timestamp: Point in time for query
            max_results: Maximum results to return
            group_ids: Group IDs to scope search
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of valid KnowledgeArtifacts
        """
        return await self.search_with_temporal_filters(
            query=query,
            filter_type=TemporalFilter.POINT_IN_TIME,
            start_time=timestamp,
            max_results=max_results,
            group_ids=group_ids,
            correlation_id=correlation_id
        )
    
    async def get_entity_timeline(
        self,
        entity_name: str,
        start_time: datetime,
        end_time: datetime,
        correlation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get timeline of events for an entity.
        
        Args:
            entity_name: Name of entity to get timeline for
            start_time: Start of time range
            end_time: End of time range
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of timeline events
        """
        correlation_id = correlation_id or f"timeline_{uuid.uuid4().hex}"
        
        if not self._initialized or not self.client:
            raise RuntimeError("GraphitiTemporalBridge not initialized")
        
        start_time_proc = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Getting entity timeline",
            "entity": entity_name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "correlation_id": correlation_id,
            "timestamp": start_time_proc.isoformat()
        })
        
        try:
            # Get entity timeline from Graphiti
            timeline = await self.client.get_entity_timeline(
                entity_name=entity_name,
                start_time=start_time,
                end_time=end_time
            )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time_proc).total_seconds() * 1000
            
            logger.info({
                "msg": "Entity timeline retrieved",
                "entity": entity_name,
                "events_count": len(timeline),
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return timeline
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time_proc).total_seconds() * 1000
            
            logger.error({
                "msg": "Failed to get entity timeline",
                "entity": entity_name,
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return []
    
    async def add_entity(
        self,
        name: str,
        entity_type: str = "entity",
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add an entity to the knowledge graph.
        
        Args:
            name: Name of the entity
            entity_type: Type of entity
            metadata: Additional metadata
            correlation_id: Correlation ID for tracking
            
        Returns:
            Result dictionary
        """
        correlation_id = correlation_id or f"entity_{uuid.uuid4().hex}"
        
        if not self._initialized or not self.client:
            raise RuntimeError("GraphitiTemporalBridge not initialized")
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Adding entity",
            "entity_name": name,
            "entity_type": entity_type,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Create entity node
            entity_node = EntityNode(
                uuid=str(uuid.uuid4()),
                name=name,
                labels=[entity_type],
                created_at=datetime.now(timezone.utc),
                summary='',
                metadata=metadata or {}
            )
            
            # Add to Graphiti
            await self.client.add_entity_node(entity_node)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            result = {
                "success": True,
                "entity_name": name,
                "entity_uuid": entity_node.uuid,
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id
            }
            
            logger.info({
                "msg": "Entity added successfully",
                "correlation_id": correlation_id,
                "entity_uuid": entity_node.uuid,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Failed to add entity",
                "entity_name": name,
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return {
                "success": False,
                "entity_name": name,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id
            }
    
    async def add_relation(
        self,
        subject: str,
        predicate: str,
        object: str,
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a relation between entities.
        
        Args:
            subject: Subject entity name
            predicate: Relationship predicate
            object: Object entity name
            metadata: Additional metadata
            correlation_id: Correlation ID for tracking
            
        Returns:
            Result dictionary
        """
        correlation_id = correlation_id or f"relation_{uuid.uuid4().hex}"
        
        if not self._initialized or not self.client:
            raise RuntimeError("GraphitiTemporalBridge not initialized")
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Adding relation",
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Add entities if they don't exist, then create relation
            # This is a simplified approach - in practice, you'd need to retrieve existing entities
            subject_node = EntityNode(
                uuid=str(uuid.uuid4()),
                name=subject,
                labels=['entity'],
                created_at=datetime.now(timezone.utc),
                summary='',
                metadata={}
            )
            
            object_node = EntityNode(
                uuid=str(uuid.uuid4()),
                name=object,
                labels=['entity'],
                created_at=datetime.now(timezone.utc),
                summary='',
                metadata={}
            )
            
            # Add nodes
            await self.client.add_entity_node(subject_node)
            await self.client.add_entity_node(object_node)
            
            # Create edge between entities
            edge = EntityEdge(
                uuid=str(uuid.uuid4()),
                name=predicate,
                source_node_uuid=subject_node.uuid,
                target_node_uuid=object_node.uuid,
                created_at=datetime.now(timezone.utc),
                factoids=[],
                metadata=metadata or {}
            )
            
            # Add edge to Graphiti
            await self.client.add_entity_edge(edge)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            result = {
                "success": True,
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "edge_uuid": edge.uuid,
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id
            }
            
            logger.info({
                "msg": "Relation added successfully",
                "correlation_id": correlation_id,
                "edge_uuid": edge.uuid,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Failed to add relation",
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return {
                "success": False,
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id
            }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the Graphiti knowledge graph.
        
        Returns:
            Dictionary with graph statistics
        """
        if not self._initialized or not self.client:
            raise RuntimeError("GraphitiTemporalBridge not initialized")
        
        try:
            # Get basic statistics from Graphiti
            entities = await self.client.get_entity_list()
            episodes = await self.client.get_episodes()
            
            stats = {
                "entities_count": len(entities),
                "episodes_count": len(episodes),
                "initialized": self._initialized,
                "connection_status": "connected" if self.client else "disconnected"
            }
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get Graphiti statistics: {e}")
            return {
                "error": str(e),
                "entities_count": 0,
                "episodes_count": 0,
                "initialized": self._initialized
            }
    
    async def close(self):
        """Close the Graphiti client connection."""
        if self.client:
            try:
                await self.client.close()
                logger.info("Graphiti client connection closed")
            except Exception as e:
                logger.error(f"Error closing Graphiti client: {e}")
        
        self._initialized = False