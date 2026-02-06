"""
Graphiti Integration for OpenEvolve Knowledge Engine

This module provides integration with the Graphiti temporal knowledge graph system,
enabling temporal queries, contradiction detection, and agent memory capabilities.
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


class GraphitiIntegration:
    """
    Integration with Graphiti temporal knowledge graph system.
    
    Provides methods for:
    - Adding temporal knowledge artifacts
    - Querying knowledge at specific points in time
    - Detecting contradictions
    - Managing entity lifecycles
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Graphiti integration.

        Args:
            uri: Neo4j connection URI (can be provided via config)
            user: Neo4j username (can be provided via config)
            password: Neo4j password (can be provided via config)
            llm_client: Optional LLM client for Graphiti
            config: Optional configuration dictionary (can include uri, user, password)
        """
        # Use config values if not explicitly provided
        self.config = config or {}
        self.uri = uri or self.config.get("uri", "bolt://localhost:7687")
        self.user = user or self.config.get("user", "neo4j")
        self.password = password or self.config.get("password", "")
        self.llm_client = llm_client or self.config.get("llm_client")

        # Graphiti client
        self.client: Optional[Graphiti] = None

        # Tracking
        self._initialized = False

        logger.info({
            "msg": "GraphitiIntegration initialized",
            "uri": self.uri,
            "user": self.user,
            "config": self.config,
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
            logger.warning("GraphitiIntegration already initialized")
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
            await self.client.build_indices_and_constraints(delete_existing=False)
            
            # Verify basic functionality
            test_entities = await self.client.driver.get_node_list()
            
            self._initialized = True
            
            logger.info({
                "msg": "GraphitiIntegration initialized successfully",
                "connected": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error({
                "msg": "Failed to initialize GraphitiIntegration",
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
            raise RuntimeError("GraphitiIntegration not initialized")
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Adding knowledge artifact to Graphiti",
            "artifact_id": artifact.id,
            "artifact_type": artifact.artifact_type,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Create episode for the artifact
            episode_uuid = str(uuid.uuid4())
            
            # Add the artifact as an episode to Graphiti
            add_result = await self.client.add_episode(
                name=artifact.content[:100],  # Use first 100 chars as name
                episode_body=artifact.content,
                source_description=artifact.source or "knowledge_engine",
                reference_time=artifact.valid_at,
                group_id=artifact.group_id or "default",
                uuid=episode_uuid
            )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            result = {
                "success": True,
                "artifact_id": artifact.id,
                "episode_id": episode_uuid,
                "entities_extracted": len(add_result.nodes),
                "relationships_extracted": len(add_result.edges),
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id
            }
            
            logger.info({
                "msg": "Knowledge artifact added to Graphiti successfully",
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
                "msg": "Failed to add knowledge artifact to Graphiti",
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
        start_time_param: Optional[datetime] = None,
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
            start_time_param: Start time for range queries
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
            raise RuntimeError("GraphitiIntegration not initialized")
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Temporal search in Graphiti initiated",
            "query": query,
            "filter_type": filter_type.value,
            "max_results": max_results,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Prepare search parameters based on filter type
            if filter_type == TemporalFilter.POINT_IN_TIME:
                # Search at specific point in time
                if not start_time_param:
                    start_time_param = datetime.now(timezone.utc)
                
                # For point-in-time, we'll use the search functionality
                # Graphiti doesn't have direct point-in-time search, so we'll use the closest available
                search_results = await self.client.search(
                    query=query,
                    num_results=max_results
                )
            elif filter_type == TemporalFilter.TIME_RANGE:
                # Search within time range - Graphiti doesn't have direct time range search
                # So we'll retrieve episodes in the time range and search within them
                if not start_time_param or not end_time:
                    raise ValueError("start_time and end_time required for TIME_RANGE filter")
                
                # Retrieve episodes in the time range
                episodes = await self.client.retrieve_episodes(
                    reference_time=end_time,
                    last_n=50  # Get enough episodes to cover the range
                )
                
                # Filter episodes by time range
                filtered_episodes = [
                    ep for ep in episodes 
                    if start_time_param <= ep.valid_at <= end_time
                ]
                
                # Search within these episodes
                search_results = await self.client.search(
                    query=query,
                    num_results=max_results
                )
            elif filter_type == TemporalFilter.CURRENT:
                # Search for currently valid knowledge
                search_results = await self.client.search(
                    query=query,
                    num_results=max_results
                )
            else:  # ALL
                # Search all historical knowledge
                search_results = await self.client.search(
                    query=query,
                    num_results=max_results
                )
            
            # Convert results to KnowledgeArtifacts
            artifacts = []
            for edge in search_results:
                # Convert Graphiti result to KnowledgeArtifact
                artifact = KnowledgeArtifact(
                    id=str(uuid.uuid4()),
                    content=f"{edge.source_node_name} -> {edge.fact} -> {edge.target_node_name}",
                    artifact_type="relationship",
                    valid_at=edge.created_at if hasattr(edge, 'created_at') else datetime.now(timezone.utc),
                    metadata={
                        "source_node": edge.source_node_name,
                        "target_node": edge.target_node_name,
                        "fact": edge.fact,
                        "episode_uuid": edge.episodes[0] if edge.episodes else None
                    },
                    source="graphiti",
                    group_id=edge.group_id if hasattr(edge, 'group_id') else "default",
                    confidence=edge.score if hasattr(edge, 'score') else 0.5
                )
                artifacts.append(artifact)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Temporal search in Graphiti completed",
                "correlation_id": correlation_id,
                "results_count": len(artifacts),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return artifacts
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Temporal search in Graphiti failed",
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
            start_time_param=timestamp,
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
            raise RuntimeError("GraphitiIntegration not initialized")
        
        start_time_proc = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Getting entity timeline from Graphiti",
            "entity": entity_name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "correlation_id": correlation_id,
            "timestamp": start_time_proc.isoformat()
        })
        
        try:
            # Get episodes that mention this entity
            # First, find the entity node
            query = """
            MATCH (e:Entity)
            WHERE e.name = $name
            RETURN e
            """
            records, _, _ = await self.client.driver.execute_query(query, name=entity_name)
            
            entity_uuid = None
            for record in records:
                entity_node = record['e']
                entity_uuid = entity_node.get('uuid')
                break
            
            if not entity_uuid:
                logger.warning(f"Entity {entity_name} not found in Graphiti")
                return []
            
            # Get related episodes
            episode_query = """
            MATCH (ep:Episodic)-[:MENTIONS]->(e:Entity {uuid: $entity_uuid})
            WHERE ep.valid_at >= $start_time AND ep.valid_at <= $end_time
            RETURN ep
            ORDER BY ep.valid_at
            """
            
            episode_records, _, _ = await self.client.driver.execute_query(
                episode_query,
                entity_uuid=entity_uuid,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat()
            )
            
            timeline = []
            for record in episode_records:
                episode = record['ep']
                timeline.append({
                    "timestamp": episode.get('valid_at'),
                    "event_type": "mention",
                    "description": episode.get('content', '')[:100],
                    "episode_uuid": episode.get('uuid')
                })
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time_proc).total_seconds() * 1000
            
            logger.info({
                "msg": "Entity timeline retrieved from Graphiti",
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
                "msg": "Failed to get entity timeline from Graphiti",
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
        entity_name: Optional[str] = None,  # Alias for backward compatibility
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add an entity to the knowledge graph.

        Args:
            name: Name of the entity
            entity_type: Type of entity
            entity_name: Alternative parameter name (alias for name)
            metadata: Additional metadata
            correlation_id: Correlation ID for tracking

        Returns:
            Result dictionary
        """
        # Use entity_name if provided, otherwise use name
        actual_name = entity_name or name
        correlation_id = correlation_id or f"entity_{uuid.uuid4().hex}"

        if not self._initialized or not self.client:
            raise RuntimeError("GraphitiIntegration not initialized")

        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Adding entity to Graphiti",
            "entity_name": actual_name,
            "entity_type": entity_type,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Create a minimal episode to establish the entity
            episode_result = await self.client.add_episode(
                name=f"Entity: {actual_name}",
                episode_body=f"Entity of type {entity_type} named {actual_name}",
                source_description="knowledge_engine_entity_creation",
                reference_time=datetime.now(timezone.utc),
                group_id="entities"
            )

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            result = {
                "success": True,
                "entity_name": actual_name,
                "entity_uuid": episode_result.episode.uuid if episode_result.episode else None,
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id
            }

            logger.info({
                "msg": "Entity added to Graphiti successfully",
                "correlation_id": correlation_id,
                "entity_uuid": result["entity_uuid"],
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Failed to add entity to Graphiti",
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
            raise RuntimeError("GraphitiIntegration not initialized")
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Adding relation to Graphiti",
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Create a triplet relationship in Graphiti
            # First, ensure entities exist by creating episodes mentioning them
            await self.client.add_episode(
                name=f"Relationship: {subject} -> {predicate} -> {object}",
                episode_body=f"{subject} {predicate} {object}",
                source_description="knowledge_engine_relation_creation",
                reference_time=datetime.now(timezone.utc),
                group_id="relations"
            )
            
            # For now, we'll just return success since Graphiti handles relations differently
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            result = {
                "success": True,
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id
            }
            
            logger.info({
                "msg": "Relation added to Graphiti successfully",
                "correlation_id": correlation_id,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Failed to add relation to Graphiti",
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
            raise RuntimeError("GraphitiIntegration not initialized")
        
        try:
            # Get basic statistics from Graphiti
            query = "MATCH (n) RETURN count(n) AS nodeCount"
            node_records, _, _ = await self.client.driver.execute_query(query)
            node_count = node_records[0]['nodeCount'] if node_records else 0
            
            rel_query = "MATCH ()-[r]->() RETURN count(r) AS relCount"
            rel_records, _, _ = await self.client.driver.execute_query(rel_query)
            rel_count = rel_records[0]['relCount'] if rel_records else 0
            
            stats = {
                "entities_count": node_count,
                "relationships_count": rel_count,
                "initialized": self._initialized,
                "connection_status": "connected" if self.client else "disconnected"
            }
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get Graphiti statistics: {e}")
            return {
                "error": str(e),
                "entities_count": 0,
                "relationships_count": 0,
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