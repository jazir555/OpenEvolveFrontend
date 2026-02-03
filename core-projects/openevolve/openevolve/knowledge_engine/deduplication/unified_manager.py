"""
Unified Deduplication Manager

Orchestrates multiple deduplication strategies with intelligent selection
and caching for optimal performance.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
import logging
import yaml
from pathlib import Path

from .base import Entity, DeduplicationResult, DeduplicationStrategy
from .strategies.semhash_strategy import SemHashStrategy
from .strategies.lm_cluster_strategy import LMClusteringStrategy
from .strategies.standardization_strategy import EntityStandardizationStrategy
from .strategies.semantic_strategy import SemanticDedupStrategy

logger = logging.getLogger(__name__)


class SimpleCache:
    """Simple in-memory cache for deduplication results."""

    def __init__(self, ttl: int = 3600):
        self.cache: Dict[str, tuple[DeduplicationResult, float]] = {}
        self.ttl = ttl

    def generate_key(self, entities: List[Entity], strategy: str) -> str:
        """Generate cache key from entity IDs and strategy."""
        entity_ids = sorted([e.id for e in entities])
        return f"{strategy}:{':'.join(entity_ids[:10])}:{len(entities)}"

    def get(self, key: str) -> Optional[DeduplicationResult]:
        """Get cached result if not expired."""
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return result
            else:
                del self.cache[key]
        return None

    def set(self, key: str, result: DeduplicationResult):
        """Cache result with current timestamp."""
        self.cache[key] = (result, time.time())

    def clear(self):
        """Clear all cached results."""
        self.cache.clear()


class UnifiedDeduplicationManager:
    """
    Manages deduplication across multiple strategies.

    Strategies:
    - semhash: Fast rule-based deduplication (kg-gen)
    - lm_cluster: ML-based clustering (kg-gen)
    - standardization: Entity normalization (ai-knowledge-graph)
    - semantic: LLM-based semantic matching (Graphiti)
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.cache = SimpleCache(
            ttl=self.config.get('cache_ttl', 3600)
        )
        self.strategies = self._initialize_strategies()
        self.canonical_mappings: Dict[str, List[str]] = {}  # canonical_id -> [variant_ids]

        logger.info(f"Initialized UnifiedDeduplicationManager with {len(self.strategies)} strategies")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            # Try multiple config locations
            possible_paths = [
                Path(__file__).parent.parent.parent.parent / 'config' / 'deduplication.yaml',
                Path(__file__).parent.parent / 'config' / 'deduplication.yaml',
                Path('config/deduplication.yaml'),
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
            else:
                config_path = None  # Use defaults

        default_config = {
            'default_strategy': 'auto',
            'cache_enabled': True,
            'cache_ttl': 3600,
            'strategies': {
                'semhash': {
                    'enabled': True,
                    'similarity_threshold': 0.95
                },
                'lm_cluster': {
                    'enabled': True,
                    'cluster_size': 128
                },
                'standardization': {
                    'enabled': True,
                    'stem_length': 4
                },
                'semantic': {
                    'enabled': True,
                    'confidence_threshold': 0.8
                }
            }
        }

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                default_config.update(config)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")

        return default_config

    def _initialize_strategies(self) -> Dict[str, DeduplicationStrategy]:
        """Initialize all enabled strategies."""
        strategies = {}

        strategy_classes = {
            'semhash': SemHashStrategy,
            'lm_cluster': LMClusteringStrategy,
            'standardization': EntityStandardizationStrategy,
            'semantic': SemanticDedupStrategy
        }

        for name, strategy_class in strategy_classes.items():
            config = self.config.get('strategies', {}).get(name, {})
            if config.get('enabled', True):
                try:
                    strategies[name] = strategy_class(config)
                    logger.info(f"Initialized strategy: {name}")
                except Exception as e:
                    logger.error(f"Failed to initialize {name}: {e}")

        return strategies

    async def deduplicate(
        self,
        entities: List[Entity],
        strategy: str = 'auto',
        use_cache: bool = True
    ) -> DeduplicationResult:
        """
        Deduplicate entities using specified or auto-selected strategy.

        Auto-selection logic:
        - < 100 entities: semhash (fastest)
        - 100-1000 entities: standardization
        - > 1000 entities: lm_cluster (most accurate)
        - Ambiguous cases: semantic (LLM-based)

        Args:
            entities: List of entities to deduplicate
            strategy: Strategy name or 'auto' for automatic selection
            use_cache: Whether to use cached results

        Returns:
            DeduplicationResult with canonical entities
        """
        if not entities:
            return DeduplicationResult(canonical_entities=[], duplicate_groups=[])

        start_time = time.time()

        # Select strategy
        if strategy == 'auto':
            strategy = self._auto_select_strategy(entities)

        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.strategies.keys())}")

        # Check cache
        if use_cache and self.config.get('cache_enabled', True):
            cache_key = self.cache.generate_key(entities, strategy)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for {len(entities)} entities using {strategy}")
                return cached_result

        # Run deduplication
        logger.info(f"Deduplicating {len(entities)} entities using {strategy} strategy")
        dedup_strategy = self.strategies[strategy]
        result = await dedup_strategy.deduplicate(entities)

        # Update stats
        processing_time = (time.time() - start_time) * 1000
        result.processing_time_ms = processing_time
        result.strategy_used = strategy

        # Track canonical mappings
        for group in result.duplicate_groups:
            if group:
                canonical = group[0]  # First entity is canonical
                variants = group[1:]  # Rest are variants
                await self.track_canonical_forms(canonical, variants)

        # Cache result
        if use_cache and self.config.get('cache_enabled', True):
            self.cache.set(cache_key, result)

        logger.info(
            f"Deduplication complete: {len(entities)} -> {len(result.canonical_entities)} "
            f"entities in {processing_time:.2f}ms"
        )

        return result

    def _auto_select_strategy(self, entities: List[Entity]) -> str:
        """Automatically select best strategy based on entity count and characteristics."""
        count = len(entities)

        # Check for ambiguous entities (need LLM)
        has_ambiguous = any(
            len(e.name) < 3 or not e.description
            for e in entities
        )

        if has_ambiguous and count < 100:
            return 'semantic'
        elif count < 100:
            return 'semhash'
        elif count < 1000:
            return 'standardization'
        else:
            return 'lm_cluster'

    async def merge_entities(
        self,
        entity_group: List[Entity]
    ) -> Entity:
        """
        Merge entity group into canonical form.

        Merges by:
        1. Keeping first entity as base
        2. Combining properties
        3. Keeping most complete description
        4. Preserving all sources
        """
        if not entity_group:
            raise ValueError("Cannot merge empty entity group")

        # Sort by timestamp (oldest first) and property completeness
        sorted_entities = sorted(
            entity_group,
            key=lambda e: (len(e.properties), e.timestamp)
        )
        canonical = sorted_entities[-1]  # Most complete

        # Merge properties
        all_sources = set()
        merged_properties = {}

        for entity in entity_group:
            # Merge properties (later entities override earlier ones)
            merged_properties.update(entity.properties)

            # Collect sources
            if entity.source:
                all_sources.add(entity.source)

        # Create merged entity
        merged_entity = Entity(
            id=canonical.id,
            name=canonical.name,
            entity_type=canonical.entity_type,
            description=canonical.description,
            properties=merged_properties,
            source=', '.join(sorted(all_sources)) if all_sources else canonical.source,
            timestamp=min(e.timestamp for e in entity_group)
        )

        return merged_entity

    async def track_canonical_forms(
        self,
        canonical: Entity,
        variants: List[Entity]
    ):
        """Track canonical-to-variant mappings for future reference."""
        if canonical.id not in self.canonical_mappings:
            self.canonical_mappings[canonical.id] = []

        for variant in variants:
            if variant.id not in self.canonical_mappings[canonical.id]:
                self.canonical_mappings[canonical.id].append(variant.id)

    def get_canonical_mapping(self) -> Dict[str, List[str]]:
        """Get all canonical-to-variant mappings."""
        return self.canonical_mappings.copy()

    def clear_cache(self):
        """Clear the deduplication cache."""
        self.cache.clear()
        logger.info("Deduplication cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about deduplication operations."""
        return {
            'strategies_available': list(self.strategies.keys()),
            'cache_size': len(self.cache.cache),
            'canonical_mappings': len(self.canonical_mappings),
            'config': self.config
        }
