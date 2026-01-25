"""
Graphiti Bridge for OpenEvolve Knowledge Engine

This module provides a bridge that integrates the Graphiti adapter with the
OpenEvolve knowledge engine, enabling seamless temporal knowledge management.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from integrations.graphiti.adapter import GraphitiAdapter, GRAPHITI_AVAILABLE
from integrations.base.knowledge_interface import (
    TemporalFilter,
    KnowledgeGraphError,
)

logger = logging.getLogger(__name__)


class GraphitiBridge:
    """
    Bridge between Graphiti and OpenEvolve Knowledge Engine.

    This bridge provides:
    - Configuration management from YAML config
    - Singleton instance management
    - Graceful degradation when Graphiti unavailable
    - Caching layer for improved performance
    - Fallback mechanisms
    """

    _instance: Optional["GraphitiBridge"] = None
    _adapter: Optional[GraphitiAdapter] = None
    _config: Optional[Dict[str, Any]] = None
    _cache: Dict[str, Any] = {}
    _cache_enabled: bool = True
    _cache_ttl: int = 3600  # seconds

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(GraphitiBridge, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the bridge (singleton safe)."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._adapter = GraphitiAdapter()
            self._config = None
            self._cache = {}

    @classmethod
    async def get_instance(cls, config_path: Optional[str] = None) -> "GraphitiBridge":
        """
        Get or create the Graphiti bridge instance.

        Args:
            config_path: Optional path to config.yaml file

        Returns:
            GraphitiBridge instance
        """
        if cls._instance is None:
            cls._instance = cls()

        # Load config if provided and not already loaded
        if config_path and cls._instance._config is None:
            await cls._instance.load_config(config_path)

        return cls._instance

    async def load_config(self, config_path: str) -> bool:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config.yaml file

        Returns:
            True if successful
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                # Try relative path
                config_file = Path(__file__).parent / config_path

            if not config_file.exists():
                logger.error(f"Config file not found: {config_path}")
                return False

            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            self._config = config

            # Extract integration settings
            integration_config = config.get('integration', {})
            self._cache_enabled = integration_config.get('cache_enabled', True)
            self._cache_ttl = integration_config.get('cache_ttl', 3600)

            logger.info(f"Loaded Graphiti config from {config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the Graphiti adapter.

        Args:
            config: Optional configuration dict (uses loaded config if None)

        Returns:
            True if initialization successful
        """
        if not GRAPHITI_AVAILABLE:
            logger.warning("Graphiti not available - bridge in degraded mode")
            return False

        try:
            # Use provided config or loaded config
            if config is None:
                config = self._build_config_from_yaml()

            if not config:
                logger.error("No configuration available for initialization")
                return False

            # Check if auto_start is enabled
            if not config.get('auto_start', True):
                logger.info("Auto-start disabled - skipping initialization")
                return False

            # Initialize adapter
            connection_config = {
                'uri': config.get('uri'),
                'user': config.get('user', 'neo4j'),
                'password': config.get('password', ''),
                'backend': config.get('backend', 'neo4j'),
                'store_raw_episode_content': config.get('store_raw_episode_content', True),
                'max_coroutines': config.get('max_workers', 4),
            }

            success = await self._adapter.initialize(connection_config)

            if success:
                logger.info("Graphiti bridge initialized successfully")
            else:
                logger.warning("Graphiti bridge initialization failed")

            return success

        except Exception as e:
            logger.error(f"Failed to initialize Graphiti bridge: {e}")
            return False

    def _build_config_from_yaml(self) -> Optional[Dict[str, Any]]:
        """Build configuration dict from loaded YAML."""
        if self._config is None:
            return None

        project_config = self._config.get('project', {})
        connection_config = self._config.get('connection', {})
        integration_config = self._config.get('integration', {})
        performance_config = self._config.get('performance', {})

        # Check if enabled
        if not project_config.get('enabled', True):
            logger.info("Graphiti disabled in configuration")
            return None

        # Build connection config
        config = {
            'backend': connection_config.get('backend', 'neo4j'),
            'uri': connection_config.get('uri', 'bolt://localhost:7687'),
            'user': connection_config.get('user', 'neo4j'),
            'password': self._resolve_env_var(connection_config.get('password', '')),
            'auto_start': integration_config.get('auto_start', True),
            'fallback_on_error': integration_config.get('fallback_on_error', True),
            'store_raw_episode_content': True,
            'max_workers': performance_config.get('max_workers', 4),
        }

        return config

    def _resolve_env_var(self, value: str) -> str:
        """Resolve environment variable in config value."""
        if value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            return os.getenv(env_var, value)
        return value

    async def add_episode(
        self,
        name: str,
        body: str,
        reference_time: datetime,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "openevolve",
        group_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add an episode through the bridge.

        Args:
            name: Episode name
            body: Episode content
            reference_time: Reference timestamp
            metadata: Optional metadata
            source: Source identifier
            group_id: Optional group ID

        Returns:
            Episode result dictionary
        """
        if not self._adapter or not self._adapter.is_initialized:
            logger.warning("Graphiti adapter not initialized - episode not added")
            return {}

        try:
            return await self._adapter.add_episode(
                name=name,
                body=body,
                reference_time=reference_time,
                metadata=metadata,
                source=source,
                group_id=group_id
            )
        except Exception as e:
            logger.error(f"Failed to add episode: {e}")
            if self._config and self._config.get('integration', {}).get('fallback_on_error'):
                return {}
            raise

    async def search(
        self,
        query: str,
        temporal_filters: Optional[Dict[str, Any]] = None,
        num_results: int = 10,
        group_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Search through the bridge.

        Args:
            query: Search query
            temporal_filters: Optional temporal filters
            num_results: Max results
            group_ids: Optional group IDs

        Returns:
            Search results
        """
        if not self._adapter or not self._adapter.is_initialized:
            logger.warning("Graphiti adapter not initialized - returning empty results")
            return {"edges": [], "nodes": [], "context": []}

        # Check cache
        cache_key = f"search:{query}:{num_results}:{temporal_filters}"
        if self._cache_enabled and cache_key in self._cache:
            logger.debug(f"Cache hit for {cache_key}")
            return self._cache[cache_key]

        try:
            results = await self._adapter.search(
                query=query,
                temporal_filters=temporal_filters,
                num_results=num_results,
                group_ids=group_ids
            )

            # Cache results
            if self._cache_enabled:
                self._cache[cache_key] = results

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            if self._config and self._config.get('integration', {}).get('fallback_on_error'):
                return {"edges": [], "nodes": [], "context": []}
            raise

    async def get_community_detections(
        self,
        group_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get community detections through the bridge.

        Args:
            group_ids: Optional group IDs

        Returns:
            Community information
        """
        if not self._adapter or not self._adapter.is_initialized:
            logger.warning("Graphiti adapter not initialized")
            return {"communities": [], "community_edges": [], "metrics": {}}

        try:
            return await self._adapter.get_community_detections(group_ids=group_ids)
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            if self._config and self._config.get('integration', {}).get('fallback_on_error'):
                return {"communities": [], "community_edges": [], "metrics": {}}
            raise

    async def validate(self) -> Dict[str, Any]:
        """
        Validate the bridge and adapter.

        Returns:
            Validation results
        """
        if not self._adapter:
            return {
                "is_valid": False,
                "checks": {},
                "issues": ["Adapter not initialized"],
                "metrics": {}
            }

        return await self._adapter.validate()

    async def shutdown(self) -> bool:
        """
        Shutdown the bridge and adapter.

        Returns:
            True if successful
        """
        if self._adapter:
            try:
                success = await self._adapter.shutdown()
                if success:
                    self._cache.clear()
                return success
            except Exception as e:
                logger.error(f"Shutdown failed: {e}")
                return False
        return True

    @property
    def is_available(self) -> bool:
        """Check if Graphiti is available."""
        return GRAPHITI_AVAILABLE

    @property
    def is_initialized(self) -> bool:
        """Check if adapter is initialized."""
        return self._adapter is not None and self._adapter.is_initialized

    @property
    def adapter(self) -> Optional[GraphitiAdapter]:
        """Get the underlying adapter."""
        return self._adapter


# Singleton instance
_bridge_instance: Optional[GraphitiBridge] = None


async def get_bridge(config_path: Optional[str] = None) -> GraphitiBridge:
    """
    Get the Graphiti bridge singleton.

    Args:
        config_path: Optional path to config.yaml

    Returns:
        GraphitiBridge instance
    """
    global _bridge_instance

    if _bridge_instance is None:
        _bridge_instance = GraphitiBridge()
        if config_path:
            await _bridge_instance.load_config(config_path)

    return _bridge_instance
