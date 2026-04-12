import time
import os
import unittest
from llm_caching import DatabaseCache, LRUCache

class TestCacheOptimizations(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_cache.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db_cache = DatabaseCache(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_database_cache_unix_timestamps(self):
        # Set a value
        self.db_cache.set("key1", {"data": "value1"}, ttl=3600)

        # Verify it can be retrieved
        val = self.db_cache.get("key1")
        self.assertEqual(val, {"data": "value1"})

        # Verify it handles buffered hit counts
        self.db_cache.get("key1")
        self.db_cache.get("key1")

        # Manually flush hit counts
        self.db_cache._flush_hit_counts()

        # Check hit count in DB
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT hit_count, timestamp FROM llm_cache WHERE key = 'key1'")
            row = cursor.fetchone()
            self.assertEqual(row[0], 4) # 1 initial + 3 gets
            self.assertIsInstance(row[1], (int, float)) # Should be REAL (Unix timestamp)

    def test_lru_cache_periodic_cleanup(self):
        lru = LRUCache(max_size=10)
        lru._clean_interval = 0.1 # Short interval for testing

        # Add expired item
        lru.set("expired", "val", ttl=-10)

        # Immediate get should handle it (delete it)
        val = lru.get("expired")
        self.assertIsNone(val)

        # Set again
        lru.set("expired2", "val", ttl=-10)
        self.assertIn("expired2", lru.cache)

        # Wait for interval
        time.sleep(0.2)

        # Next set should trigger cleanup
        lru.set("new", "val")
        self.assertNotIn("expired2", lru.cache)

    def test_memoize_efficiency(self):
        from performance_utils import memoize

        call_count = 0
        @memoize
        def expensive_func(x, y=None):
            nonlocal call_count
            call_count += 1
            return x + (y or 0)

        # First call
        res1 = expensive_func(10, y=5)
        self.assertEqual(res1, 15)
        self.assertEqual(call_count, 1)

        # Second call (cached)
        res2 = expensive_func(10, y=5)
        self.assertEqual(res2, 15)
        self.assertEqual(call_count, 1)

        # Call with different kwargs order
        res3 = expensive_func(10, y=5)
        self.assertEqual(res3, 15)
        self.assertEqual(call_count, 1)

    def test_parallel_executor_shared_pool(self):
        from performance_utils import ParallelExecutor
        import concurrent.futures

        exec1 = ParallelExecutor(max_workers=2)
        pool1 = exec1.get_executor()

        exec2 = ParallelExecutor(max_workers=2)
        pool2 = exec2.get_executor()

        # Should be the same singleton pool
        self.assertIs(pool1, pool2)
        self.assertIsInstance(pool1, concurrent.futures.ThreadPoolExecutor)

if __name__ == "__main__":
    unittest.main()
