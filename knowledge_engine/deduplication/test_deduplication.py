"""
Test suite for Unified Deduplication System

Tests all strategies with various entity configurations.
"""

import pytest
import asyncio
from datetime import datetime
from knowledge_engine.core.deduplication import (
    UnifiedDeduplicationManager,
    DeduplicationStrategy,
    Entity,
    DeduplicationResult
)
from knowledge_engine.core.deduplication.strategies import (
    SemHashStrategy,
    LMClusteringStrategy,
    EntityStandardizationStrategy,
    SemanticDedupStrategy
)


class TestDeduplicationBase:
    """Base test class with common fixtures."""

    @pytest.fixture
    def sample_entities(self):
        """Create sample entities for testing."""
        return [
            Entity(
                id="e1",
                name="Machine Learning",
                entity_type="concept",
                description="AI and ML technologies"
            ),
            Entity(
                id="e2",
                name="machine learning",  # Duplicate (case insensitive)
                entity_type="concept",
                description="Artificial Intelligence and ML"
            ),
            Entity(
                id="e3",
                name="Deep Learning",
                entity_type="concept",
                description="Neural networks and deep learning"
            ),
            Entity(
                id="e4",
                name="Neural Networks",
                entity_type="concept",
                description="Deep neural network architectures"
            ),
        ]

    @pytest.fixture
    def large_entity_set(self):
        """Create large set of entities for testing."""
        entities = []
        for i in range(150):
            entities.append(Entity(
                id=f"entity_{i}",
                name=f"Concept {i % 50}",  # Many duplicates
                entity_type="concept",
                description=f"Description for concept {i % 50}"
            ))
        return entities

    @pytest.fixture
    def hierarchical_entities(self):
        """Create hierarchical entities for subset testing."""
        return [
            Entity(
                id="h1",
                name="Machine Learning",
                entity_type="concept"
            ),
            Entity(
                id="h2",
                name="ML",
                entity_type="concept"
            ),
            Entity(
                id="h3",
                name="Deep Learning",
                entity_type="concept"
            ),
            Entity(
                id="h4",
                name="Machine Learning Algorithms",
                entity_type="concept"
            ),
        ]


class TestSemHashStrategy(TestDeduplicationBase):
    """Test SEMHASH strategy."""

    @pytest.mark.asyncio
    async def test_semhash_basic_deduplication(self, sample_entities):
        """Test basic deduplication with SEMHASH."""
        strategy = SemHashStrategy()
        result = await strategy.deduplicate(sample_entities)

        assert len(result.canonical_entities) < len(sample_entities)
        assert len(result.duplicate_groups) > 0
        assert result.strategy_used == "semhash"

    @pytest.mark.asyncio
    async def test_semhash_confidence_calculation(self, sample_entities):
        """Test confidence score calculation."""
        strategy = SemHashStrategy()

        # Test with similar entities
        entity1 = sample_entities[0]
        entity2 = sample_entities[1]  # Duplicate (different case)

        confidence = strategy.calculate_confidence(entity1, entity2)
        assert confidence > 0.8  # Should be high for similar entities

    @pytest.mark.asyncio
    async def test_semhash_empty_list(self):
        """Test with empty entity list."""
        strategy = SemHashStrategy()
        result = await strategy.deduplicate([])

        assert len(result.canonical_entities) == 0
        assert len(result.duplicate_groups) == 0


class TestLMClusteringStrategy(TestDeduplicationBase):
    """Test LM Clustering strategy."""

    @pytest.mark.asyncio
    async def test_lm_cluster_basic_deduplication(self, sample_entities):
        """Test basic deduplication with LM clustering."""
        strategy = LMClusteringStrategy()
        result = await strategy.deduplicate(sample_entities)

        # Should deduplicate
        assert len(result.canonical_entities) <= len(sample_entities)
        assert result.strategy_used == "lm_cluster"

    @pytest.mark.asyncio
    async def test_lm_cluster_large_dataset(self, large_entity_set):
        """Test with large dataset."""
        strategy = LMClusteringStrategy()
        result = await strategy.deduplicate(large_entity_set)

        # Should significantly reduce count due to many duplicates
        assert len(result.canonical_entities) < len(large_entity_set)

    @pytest.mark.asyncio
    async def test_lm_cluster_fallback(self, large_entity_set):
        """Test fallback behavior when embedding fails."""
        # Force fallback by disabling model
        strategy = LMClusteringStrategy()
        strategy.model = None

        result = await strategy.deduplicate(large_entity_set)

        # Should still work with fallback
        assert len(result.canonical_entities) > 0


class TestStandardizationStrategy(TestDeduplicationBase):
    """Test Entity Standardization strategy."""

    @pytest.mark.asyncio
    async def test_standardization_basic_deduplication(self, sample_entities):
        """Test basic deduplication with standardization."""
        strategy = EntityStandardizationStrategy()
        result = await strategy.deduplicate(sample_entities)

        assert len(result.canonical_entities) <= len(sample_entities)
        assert result.strategy_used == "standardization"

    @pytest.mark.asyncio
    async def test_standardization_root_words(self, hierarchical_entities):
        """Test root word extraction and grouping."""
        strategy = EntityStandardizationStrategy()
        result = await strategy.deduplicate(hierarchical_entities)

        # Should find some duplicates based on root words
        assert len(result.canonical_entities) <= len(hierarchical_entities)

    @pytest.mark.asyncio
    async def test_standardization_confidence(self, hierarchical_entities):
        """Test confidence calculation with standardization."""
        strategy = EntityStandardizationStrategy()

        entity1 = hierarchical_entities[0]  # "Machine Learning"
        entity2 = hierarchical_entities[3]  # "Machine Learning Algorithms"

        confidence = strategy.calculate_confidence(entity1, entity2)
        # Should be high due to subset relationship
        assert confidence > 0.5


class TestSemanticStrategy(TestDeduplicationBase):
    """Test Semantic Deduplication strategy."""

    @pytest.mark.asyncio
    async def test_semantic_basic_deduplication(self, sample_entities):
        """Test basic deduplication with semantic analysis."""
        strategy = SemanticDedupStrategy()
        result = await strategy.deduplicate(sample_entities)

        assert len(result.canonical_entities) <= len(sample_entities)
        assert result.strategy_used == "semantic"

    @pytest.mark.asyncio
    async def test_semantic_batch_processing(self, large_entity_set):
        """Test batch processing for large datasets."""
        strategy = SemanticDedupStrategy()
        result = await strategy.deduplicate(large_entity_set)

        # Should process in batches
        assert len(result.canonical_entities) > 0

    @pytest.mark.asyncio
    async def test_semantic_confidence(self, sample_entities):
        """Test confidence calculation."""
        strategy = SemanticDedupStrategy()

        entity1 = sample_entities[0]
        entity2 = sample_entities[2]  # Different concept

        confidence = strategy.calculate_confidence(entity1, entity2)
        # Should be low for different concepts
        assert confidence < 0.5


class TestUnifiedManager(TestDeduplicationBase):
    """Test Unified Deduplication Manager."""

    @pytest.fixture
    def manager(self):
        """Create manager instance."""
        return UnifiedDeduplicationManager()

    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert manager is not None
        assert len(manager.strategies) > 0
        assert manager.cache is not None

    def test_auto_strategy_selection(self, manager):
        """Test automatic strategy selection."""
        # Small dataset
        small_entities = [Entity(id=f"e{i}", name=f"Entity {i}", entity_type="test") for i in range(50)]
        strategy = manager._auto_select_strategy(small_entities)
        assert strategy == 'semhash' or strategy == 'standardization'

        # Large dataset
        large_entities = [Entity(id=f"e{i}", name=f"Entity {i}", entity_type="test") for i in range(1500)]
        strategy = manager._auto_select_strategy(large_entities)
        assert strategy == 'lm_cluster'

    @pytest.mark.asyncio
    async def test_deduplicate_with_auto_strategy(self, manager, sample_entities):
        """Test deduplication with automatic strategy selection."""
        result = await manager.deduplicate(sample_entities, strategy='auto')

        assert len(result.canonical_entities) > 0
        assert result.strategy_used in manager.strategies

    @pytest.mark.asyncio
    async def test_deduplicate_with_specific_strategy(self, manager, sample_entities):
        """Test deduplication with specific strategy."""
        result = await manager.deduplicate(sample_entities, strategy='semhash')

        assert len(result.canonical_entities) > 0
        assert result.strategy_used == 'semhash'

    @pytest.mark.asyncio
    async def test_cache_functionality(self, manager, sample_entities):
        """Test caching functionality."""
        # First call
        result1 = await manager.deduplicate(sample_entities, strategy='semhash', use_cache=True)

        # Second call (should use cache)
        result2 = await manager.deduplicate(sample_entities, strategy='semhash', use_cache=True)

        assert len(result1.canonical_entities) == len(result2.canonical_entities)

    @pytest.mark.asyncio
    async def test_merge_entities(self, manager):
        """Test entity merging."""
        entities = [
            Entity(
                id="e1",
                name="Test Entity",
                entity_type="test",
                properties={"key1": "value1"},
                source="source1"
            ),
            Entity(
                id="e2",
                name="Test Entity Duplicate",
                entity_type="test",
                properties={"key2": "value2"},
                source="source2"
            ),
        ]

        merged = await manager.merge_entities(entities)

        assert merged.name == "Test Entity Duplicate"  # Most complete
        assert "key1" in merged.properties
        assert "key2" in merged.properties
        assert "source1" in merged.source or "source2" in merged.source

    @pytest.mark.asyncio
    async def test_invalid_strategy(self, manager, sample_entities):
        """Test error handling for invalid strategy."""
        with pytest.raises(ValueError):
            await manager.deduplicate(sample_entities, strategy='invalid_strategy')

    def test_get_stats(self, manager):
        """Test statistics retrieval."""
        stats = manager.get_stats()

        assert 'strategies_available' in stats
        assert 'cache_size' in stats
        assert 'canonical_mappings' in stats
        assert len(stats['strategies_available']) > 0


class TestIntegration(TestDeduplicationBase):
    """Integration tests for the complete system."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, sample_entities):
        """Test complete deduplication pipeline."""
        manager = UnifiedDeduplicationManager()

        # Deduplicate
        result = await manager.deduplicate(sample_entities)

        # Verify results
        assert len(result.canonical_entities) > 0
        assert result.processing_time_ms > 0

        # Check stats
        assert result.stats['original_count'] == len(sample_entities)
        assert result.stats['canonical_count'] == len(result.canonical_entities)

    @pytest.mark.asyncio
    async def test_strategy_comparison(self, sample_entities):
        """Compare results across different strategies."""
        manager = UnifiedDeduplicationManager()

        results = {}
        for strategy_name in manager.strategies.keys():
            result = await manager.deduplicate(sample_entities, strategy=strategy_name)
            results[strategy_name] = result

        # All strategies should produce results
        assert len(results) > 0

        # Check that results vary by strategy
        canonical_counts = {
            name: len(r.canonical_entities)
            for name, r in results.items()
        }
        print(f"Canonical entity counts by strategy: {canonical_counts}")

    @pytest.mark.asyncio
    async def test_performance_benchmark(self, large_entity_set):
        """Benchmark performance with large dataset."""
        manager = UnifiedDeduplicationManager()

        import time
        start = time.time()

        result = await manager.deduplicate(large_entity_set, strategy='lm_cluster')

        elapsed = time.time() - start

        print(f"Processed {len(large_entity_set)} entities in {elapsed:.2f}s")
        print(f"Reduced to {len(result.canonical_entities)} canonical entities")
        print(f"Processing time: {result.processing_time_ms:.2f}ms")

        # Should complete in reasonable time
        assert elapsed < 60  # Less than 60 seconds


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
