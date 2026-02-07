"""
Graphiti Integration Package for OpenEvolve Knowledge Engine

This package provides integration with Graphiti temporal knowledge graph system,
enabling temporal queries, contradiction detection, and agent memory capabilities.

Components:
- GraphitiTemporalBridge: Main bridge to Graphiti system
- GraphitiHealthChecker: Health checking utilities
- GraphitiContradictionDetector: Contradiction detection
- GraphitiConfig: Configuration for Graphiti integration
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class GraphitiConfig:
    """
    Configuration for Graphiti integration.
    
    Attributes:
        neo4j_uri: Neo4j database URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        openai_api_key: OpenAI API key for embeddings
        default_model: Default model for Graphiti
        max_hops: Maximum hops for graph traversal
        similarity_threshold: Threshold for similarity matching
        temporal_resolution: Temporal resolution (seconds or string like 'seconds')
        enable_caching: Whether to enable caching
        cache_ttl: Cache time-to-live in seconds
    """
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    openai_api_key: Optional[str] = None
    default_model: str = "gpt-4"
    max_hops: int = 3
    similarity_threshold: float = 0.8
    temporal_resolution: str = "seconds"
    enable_caching: bool = True
    cache_ttl: int = 3600
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'neo4j_uri': self.neo4j_uri,
            'neo4j_user': self.neo4j_user,
            'neo4j_password': '***' if self.neo4j_password else '',
            'openai_api_key': '***' if self.openai_api_key else None,
            'default_model': self.default_model,
            'max_hops': self.max_hops,
            'similarity_threshold': self.similarity_threshold,
            'temporal_resolution': self.temporal_resolution,
            'enable_caching': self.enable_caching,
            'cache_ttl': self.cache_ttl
        }


try:
    from .graphiti_temporal_bridge import GraphitiTemporalBridge
except ImportError:
    GraphitiTemporalBridge = None

try:
    from .health_check import GraphitiHealthChecker
except ImportError:
    GraphitiHealthChecker = None

try:
    from .contradiction_detector import GraphitiContradictionDetector
except ImportError:
    GraphitiContradictionDetector = None

__all__ = [
    'GraphitiTemporalBridge',
    'GraphitiHealthChecker',
    'GraphitiContradictionDetector',
    'GraphitiConfig'
]
class WorkflowState:
    """Stub class for WorkflowState."""
    pass

class AgentInteraction:
    """Stub class for AgentInteraction."""
    pass
