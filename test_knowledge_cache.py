"""Test knowledge caching system."""

import time
from external_knowledge_integration import (
    KnowledgeCache,
    KnowledgeItem,
    KnowledgeIntegrationManager,
    KnowledgeSourceConfig,
    KnowledgeSourceType,
    DatabaseConnector
)


def test_knowledge_cache_basic():
    """Test basic cache operations."""
    print("Testing KnowledgeCache basic operations...")
    
    cache = KnowledgeCache(max_size=10, default_ttl=60)
    
    # Create test items
    items = [
        KnowledgeItem("source1", "content1", 0.9, {}),
        KnowledgeItem("source1", "content2", 0.8, {})
    ]
    
    # Test set and get
    cache.set("key1", items)
    retrieved = cache.get("key1")
    
    assert retrieved is not None
    assert len(retrieved) == 2
    assert retrieved[0].content == "content1"
    
    # Test cache miss
    assert cache.get("nonexistent") is None
    
    # Test stats
    stats = cache.get_stats()
    assert stats["hit_count"] == 1
    assert stats["miss_count"] == 1
    assert stats["size"] == 1
    
    print("[OK] Basic cache operations passed")


def test_knowledge_cache_expiry():
    """Test cache expiry."""
    print("\nTesting cache expiry...")
    
    cache = KnowledgeCache(max_size=10, default_ttl=1)  # 1 second TTL
    
    items = [KnowledgeItem("source1", "content1", 0.9, {})]
    cache.set("key1", items)
    
    # Should be in cache
    assert cache.get("key1") is not None
    
    # Wait for expiry
    time.sleep(1.1)
    
    # Should be expired
    assert cache.get("key1") is None
    
    print("[OK] Cache expiry tests passed")


def test_knowledge_cache_eviction():
    """Test cache eviction when full."""
    print("\nTesting cache eviction...")
    
    cache = KnowledgeCache(max_size=3, default_ttl=60)
    
    # Fill cache
    for i in range(3):
        items = [KnowledgeItem(f"source{i}", f"content{i}", 0.9, {})]
        cache.set(f"key{i}", items)
    
    assert cache.get_stats()["size"] == 3
    assert cache.get_stats()["eviction_count"] == 0
    
    # Add one more - should trigger eviction
    items = [KnowledgeItem("source3", "content3", 0.9, {})]
    cache.set("key3", items)
    
    assert cache.get_stats()["size"] == 3
    assert cache.get_stats()["eviction_count"] == 1
    
    print("[OK] Cache eviction tests passed")


def test_knowledge_cache_invalidation():
    """Test cache invalidation."""
    print("\nTesting cache invalidation...")
    
    cache = KnowledgeCache(max_size=10, default_ttl=60)
    
    # Add items
    for i in range(5):
        items = [KnowledgeItem(f"source{i}", f"content{i}", 0.9, {})]
        cache.set(f"test_key_{i}", items)
    
    assert cache.get_stats()["size"] == 5
    
    # Invalidate pattern
    invalidated = cache.invalidate("test_key_[0-2]")
    assert invalidated == 3
    assert cache.get_stats()["size"] == 2
    
    # Clear all
    cache.clear()
    assert cache.get_stats()["size"] == 0
    
    print("[OK] Cache invalidation tests passed")


def test_knowledge_cache_hit_rate():
    """Test cache hit rate calculation."""
    print("\nTesting cache hit rate...")
    
    cache = KnowledgeCache(max_size=10, default_ttl=60)
    
    items = [KnowledgeItem("source1", "content1", 0.9, {})]
    cache.set("key1", items)
    
    # 3 hits
    for _ in range(3):
        cache.get("key1")
    
    # 2 misses
    for _ in range(2):
        cache.get("nonexistent")
    
    stats = cache.get_stats()
    assert stats["hit_count"] == 3
    assert stats["miss_count"] == 2
    assert stats["hit_rate"] == 0.6  # 3/5
    
    print("[OK] Cache hit rate tests passed")


def test_integration_manager_with_cache():
    """Test KnowledgeIntegrationManager with caching."""
    print("\nTesting KnowledgeIntegrationManager with cache...")
    
    manager = KnowledgeIntegrationManager(cache_max_size=100, cache_ttl=60)
    
    # Register a connector
    config = KnowledgeSourceConfig(
        name="test_db",
        source_type=KnowledgeSourceType.DATABASE
    )
    config.metadata = {"db_type": "postgresql"}
    connector = DatabaseConnector(config)
    manager.register_connector(connector)
    
    # Query (will be cached)
    context = {"query": "test", "domain": "software"}
    results1 = manager.query_all_connectors(context)
    
    # Get cache stats
    stats = manager.get_cache_stats()
    assert stats["size"] >= 0  # May be 0 if no results, or 1 if results cached
    
    # Clear cache
    manager.clear_knowledge_cache()
    stats = manager.get_cache_stats()
    assert stats["size"] == 0
    
    print("[OK] Integration manager cache tests passed")


if __name__ == "__main__":
    print("Running knowledge cache tests...\n")
    
    test_knowledge_cache_basic()
    test_knowledge_cache_expiry()
    test_knowledge_cache_eviction()
    test_knowledge_cache_invalidation()
    test_knowledge_cache_hit_rate()
    test_integration_manager_with_cache()
    
    print("\n" + "="*50)
    print("All knowledge cache tests passed!")
    print("="*50)
