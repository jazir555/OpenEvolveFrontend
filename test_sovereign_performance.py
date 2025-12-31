import pytest
import time
from sovereign_performance_optimization import (
    PerformanceCache, PerformanceMonitor, LazyLoader, BatchProcessor,
    cached, timed, get_cache_stats, clear_cache
)

class TestPerformanceCache:
    def test_cache_basic_operations(self):
        cache = PerformanceCache(max_size=10, ttl_seconds=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None
        stats = cache.get_stats()
        assert stats['hit_count'] == 1
        assert stats['miss_count'] == 1
    
    def test_cache_eviction(self):
        cache = PerformanceCache(max_size=3, ttl_seconds=60)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")
        assert len(cache.cache) == 3
    
    def test_cache_invalidation(self):
        cache = PerformanceCache()
        cache.set("key1", "value1")
        cache.invalidate("key1")
        assert cache.get("key1") is None

class TestCachedDecorator:
    def test_function_caching(self):
        call_count = [0]
        @cached("test_func")
        def expensive_func(x):
            call_count[0] += 1
            return x * 2
        result1 = expensive_func(5)
        result2 = expensive_func(5)
        assert result1 == 10
        assert result2 == 10
        assert call_count[0] == 1

class TestPerformanceMonitor:
    def test_record_operations(self):
        monitor = PerformanceMonitor()
        monitor.record_operation("test_op", 0.5)
        monitor.record_operation("test_op", 0.3)
        stats = monitor.get_stats("test_op")
        assert stats['count'] == 2
        assert stats['avg_duration'] == 0.4
        assert stats['min_duration'] == 0.3
        assert stats['max_duration'] == 0.5

class TestTimedDecorator:
    def test_function_timing(self):
        @timed("slow_op")
        def slow_func():
            time.sleep(0.01)
            return "done"
        result = slow_func()
        assert result == "done"

class TestLazyLoader:
    def test_lazy_loading(self):
        loader = LazyLoader()
        load_count = [0]
        def load_resource():
            load_count[0] += 1
            return "expensive_data"
        loader.register("resource1", load_resource)
        assert not loader.is_loaded("resource1")
        result1 = loader.get("resource1")
        assert result1 == "expensive_data"
        assert load_count[0] == 1
        result2 = loader.get("resource1")
        assert result2 == "expensive_data"
        assert load_count[0] == 1

class TestBatchProcessor:
    def test_batch_operations(self):
        processor = BatchProcessor(batch_size=5)
        processor.add_operation({'type': 'insert', 'data': 'test1'})
        processor.add_operation({'type': 'insert', 'data': 'test2'})
        assert len(processor.pending_operations) == 2
        count = processor.flush()
        assert count == 2
        assert len(processor.pending_operations) == 0
    
    def test_auto_flush(self):
        processor = BatchProcessor(batch_size=3)
        processor.add_operation({'op': 1})
        processor.add_operation({'op': 2})
        processor.add_operation({'op': 3})
        assert len(processor.pending_operations) == 0

class TestGlobalUtilities:
    def test_cache_utilities(self):
        clear_cache()
        stats = get_cache_stats()
        assert stats['size'] == 0
        assert 'hit_rate' in stats
