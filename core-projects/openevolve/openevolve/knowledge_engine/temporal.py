"""
Temporal Knowledge Management for OpenEvolve

Integrates Graphiti's temporal knowledge graph capabilities to track the 
evolution of knowledge and record episodic events from workflows.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    from graphiti_core import Graphiti, EpisodeType
    from graphiti_core.nodes import EpisodicNode
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False

logger = logging.getLogger(__name__)

class TemporalKnowledgeManager:
    """
    Manages temporal knowledge using Graphiti.
    
    Provides:
    - Episodic event recording
    - Temporal evolution tracking
    - Time-aware search and retrieval
    """
    
    def __init__(self):
        self.available = GRAPHITI_AVAILABLE
        self.graphiti = None
        
        if self.available:
            self._initialize_graphiti()
            
        logger.info(f"TemporalKnowledgeManager initialized (Available: {self.available})")

    def _initialize_graphiti(self):
        """Initialize Graphiti with environment defaults and validation."""
        try:
            import os
            uri = os.environ.get("NEO4J_URI")
            user = os.environ.get("NEO4J_USER")
            pwd = os.environ.get("NEO4J_PASSWORD")
            
            if not all([uri, user, pwd]):
                logger.warning("Neo4j credentials missing from environment. Temporal graph disabled.")
                self.available = False
                return

            self.graphiti = Graphiti(uri=uri, user=user, password=pwd)
            logger.info(f"Graphiti core initialized against {uri}")
        except Exception as e:
            logger.error(f"Failed to initialize Graphiti core: {e}")
            self.available = False

    async def record_episode(self, name: str, content: str, source: str = "workflow") -> Optional[str]:
        """
        Record a knowledge episode with validation and error handling.
        """
        if not self.available or not self.graphiti:
            return None
            
        if not name or not content:
            logger.error("Attempted to record episode with missing name or content")
            return None

        try:
            # Graphiti's add_episode can be slow, using a timeout is recommended
            # but since we are in an async def, we'll let the caller handle higher-level timeouts
            result = await self.graphiti.add_episode(
                name=name,
                episode_body=content,
                source_description=source,
                reference_time=datetime.now(),
                source=EpisodeType.message
            )
            logger.info(f"Recorded temporal episode: {name} (UUID: {result.episode.uuid})")
            return result.episode.uuid
        except Exception as e:
            logger.error(f"Failed to record episode '{name}': {e}")
            # Do not raise here to prevent workflow failure due to telemetry issues
            return None

    async def search_with_time(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Perform temporal search with schema mapping and safety checks.
        """
        if not self.available or not self.graphiti:
            return {"nodes": [], "edges": []}
            
        try:
            results = await self.graphiti.search_(query, config=None, num_results=limit)
            
            # Map Graphiti results to unified format for OpenEvolve frontend
            nodes = []
            for node in getattr(results, 'nodes', []):
                nodes.append({
                    "id": getattr(node, 'uuid', 'unknown'),
                    "label": getattr(node, 'name', getattr(node, 'uuid', 'Unknown')),
                    "type": node.__class__.__name__,
                    "summary": getattr(node, 'summary', ""),
                    "created_at": getattr(node, 'created_at', None).isoformat() if hasattr(node, 'created_at') and node.created_at else None
                })
                
            edges = []
            for edge in getattr(results, 'edges', []):
                edges.append({
                    "source": getattr(edge, 'source_node_uuid', 'unknown'),
                    "target": getattr(edge, 'target_node_uuid', 'unknown'),
                    "label": getattr(edge, 'fact', "related"),
                    "confidence": getattr(edge, 'certainty', 1.0)
                })
                
            return {"nodes": nodes, "edges": edges}
        except Exception as e:
            logger.error(f"Temporal search failed for query '{query}': {e}")
            return {"nodes": [], "edges": []}

    def get_status(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "engine": "Graphiti",
            "connected": self.graphiti is not None
        }
