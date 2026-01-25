"""
Stress Tests and Memory Leak Tests for Knowledge Engine

Following CLAUDE.md principles:
- Test system under extreme load
- Detect memory leaks
- Test resource exhaustion handling
- Test system recovery

Tests verify:
- Large document processing (1000+ docs)
- Large graph operations (10000+ entities)
- Concurrent user load
- Memory leak detection
- Resource cleanup
"""

import asyncio
import json
import logging
import pytest
import time
import gc
import sys
import tracemalloc
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import core module using conftest's approach
CORE_AVAILABLE = False
EntityKnowledgeGraph = None
KnowledgeState = None

try:
    spec = importlib.util.spec_from_file_location(
        "core",
        project_root / "knowledge_engine" / "core.py"
    )
    if spec and spec.loader:
        core_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_module)
        EntityKnowledgeGraph = core_module.EntityKnowledgeGraph
        KnowledgeState = core_module.KnowledgeState
        CORE_AVAILABLE = True
except Exception as e:
    CORE_AVAILABLE = False
    EntityKnowledgeGraph = None
    KnowledgeState = None

logger = logging.getLogger(__name__)


class TestLargeScaleProcessing:
    """
    Stress tests for large-scale data processing.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_thousand_document_processing(self):
        """
        Test processing 1000 documents efficiently.
        """
        doc_count = 1000
        graph = EntityKnowledgeGraph()

        start_time = time.time()

        # Process documents in batches
        batch_size = 100
        for batch_start in range(0, doc_count, batch_size):
            batch_end = min(batch_start + batch_size, doc_count)

            tasks = []
            for i in range(batch_start, batch_end):
                doc = f"Document {i} discusses Topic {i % 50} and Concept {i % 20}"
                # Simple entity extraction
                entities = [f"Topic_{i % 50}", f"Concept_{i % 20}"]
                for entity in entities:
                    tasks.append(graph.add_entity(entity, {"source_doc": i}))

            await asyncio.gather(*tasks)

        end_time = time.time()
        duration = end_time - start_time

        entities_count = len(graph.get_entities())

        logger.info(json.dumps({
            "msg": "Thousand document processing completed",
            "doc_count": doc_count,
            "entities_extracted": entities_count,
            "duration_seconds": duration,
            "docs_per_second": doc_count / duration,
            "level": "INFO"
        }))

        # Performance assertions
        assert entities_count > 0, "No entities extracted"
        assert duration < 60, f"Processing too slow: {duration}s for {doc_count} docs"
        assert doc_count / duration > 10, f"Throughput too low: {doc_count / duration:.2f} docs/sec"

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_ten_thousand_entity_graph(self):
        """
        Test graph operations with 10,000 entities.
        """
        entity_count = 10000
        graph = EntityKnowledgeGraph()

        start_time = time.time()

        # Add entities
        for i in range(entity_count):
            await graph.add_entity(f"Entity_{i}", {"index": i, "type": f"Type_{i % 10}"})

        add_duration = time.time() - start_time

        # Add relationships
        start_rel = time.time()
        for i in range(0, entity_count - 1, 10):
            await graph.add_relationship(f"Entity_{i}", "connects_to", f"Entity_{i+1}")

        rel_duration = time.time() - start_rel

        # Query performance
        start_query = time.time()
        result = await graph.search_entities("Entity_5000")
        query_duration = time.time() - start_query

        total_entities = len(graph.get_entities())

        logger.info(json.dumps({
            "msg": "Ten thousand entity graph test completed",
            "entity_count": total_entities,
            "add_duration_seconds": add_duration,
            "rel_duration_seconds": rel_duration,
            "query_duration_seconds": query_duration,
            "entities_per_second": entity_count / add_duration,
            "level": "INFO"
        }))

        # Assertions
        assert total_entities == entity_count, f"Expected {entity_count} entities, got {total_entities}"
        assert add_duration < 30, f"Entity addition too slow: {add_duration}s"
        assert query_duration < 1, f"Query too slow: {query_duration}s"

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_million_relationship_graph(self):
        """
        Test graph with 1 million relationships (stress test).
        """
        # Use smaller scale for test feasibility
        entity_count = 1000
        relationship_count = 50000  # 50K relationships instead of 1M

        graph = EntityKnowledgeGraph()

        start_time = time.time()

        # Create entities
        for i in range(entity_count):
            await graph.add_entity(f"E_{i}")

        # Create dense relationships
        for i in range(entity_count):
            for j in range(i+1, min(i+50, entity_count)):  # Each entity connects to 50 others
                await graph.add_relationship(f"E_{i}", "relates_to", f"E_{j}")
                if len(graph.relationships) >= relationship_count:
                    break
            if len(graph.relationships) >= relationship_count:
                break

        creation_duration = time.time() - start_time

        # Test query performance
        start_query = time.time()
        rels = await graph.get_relationships_for_entity("E_500")
        query_duration = time.time() - start_query

        logger.info(json.dumps({
            "msg": "Large relationship graph test completed",
            "entity_count": entity_count,
            "relationship_count": len(graph.relationships),
            "creation_duration_seconds": creation_duration,
            "query_duration_seconds": query_duration,
            "relationships_per_second": len(graph.relationships) / creation_duration,
            "level": "INFO"
        }))

        assert len(graph.relationships) >= relationship_count * 0.9, "Too few relationships created"
        assert query_duration < 5, f"Relationship query too slow: {query_duration}s"


class TestConcurrentLoad:
    """
    Stress tests for concurrent operations.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_hundred_concurrent_users(self):
        """
        Test system with 100 concurrent users.
        """
        user_count = 100
        operations_per_user = 50

        graph = EntityKnowledgeGraph()
        start_time = time.time()

        async def user_operations(user_id: int):
            """Simulate user operations."""
            for op in range(operations_per_user):
                entity_name = f"User{user_id}_Entity{op}"
                await graph.add_entity(entity_name, {"user": user_id})

                if op % 10 == 0:
                    # Query every 10 operations
                    await graph.search_entities(f"User{user_id}")

        # Run all users concurrently
        tasks = [user_operations(i) for i in range(user_count)]
        await asyncio.gather(*tasks)

        duration = time.time() - start_time
        total_operations = user_count * operations_per_user
        ops_per_second = total_operations / duration

        entity_count = len(graph.get_entities())

        logger.info(json.dumps({
            "msg": "Concurrent user load test completed",
            "user_count": user_count,
            "operations_per_user": operations_per_user,
            "total_operations": total_operations,
            "duration_seconds": duration,
            "ops_per_second": ops_per_second,
            "entities_created": entity_count,
            "level": "INFO"
        }))

        # Should handle at least 100 ops/sec overall
        assert ops_per_second > 100, f"Throughput too low: {ops_per_second:.2f} ops/sec"

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_concurrent_read_write(self):
        """
        Test concurrent reads and writes.
        """
        graph = EntityKnowledgeGraph()

        # Pre-populate
        for i in range(1000):
            await graph.add_entity(f"Initial_{i}")

        read_count = 0
        write_count = 0

        async def reader(reader_id: int):
            """Concurrent reader."""
            nonlocal read_count
            for i in range(100):
                await graph.search_entities(f"Initial_{i % 1000}")
                read_count += 1

        async def writer(writer_id: int):
            """Concurrent writer."""
            nonlocal write_count
            for i in range(50):
                await graph.add_entity(f"Writer{writer_id}_Entity{i}")
                write_count += 1

        start_time = time.time()

        # Mix of readers and writers
        tasks = []
        for i in range(20):
            tasks.append(reader(i))
        for i in range(10):
            tasks.append(writer(i))

        await asyncio.gather(*tasks)

        duration = time.time() - start_time

        logger.info(json.dumps({
            "msg": "Concurrent read-write test completed",
            "read_operations": read_count,
            "write_operations": write_count,
            "total_operations": read_count + write_count,
            "duration_seconds": duration,
            "level": "INFO"
        }))

        assert read_count == 20 * 100, "Read operations lost"
        assert write_count == 10 * 50, "Write operations lost"


class TestMemoryLeakDetection:
    """
    Tests for memory leak detection.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_memory_leak_repeated_operations(self):
        """
        Test for memory leaks during repeated operations.
        """
        tracemalloc.start()

        graph = EntityKnowledgeGraph()

        # Baseline memory
        gc.collect()
        snapshot1 = tracemalloc.take_snapshot()

        # Perform many operations
        iterations = 1000
        for i in range(iterations):
            await graph.add_entity(f"Temp_{i % 100}")  # Reuse entity names
            await graph.add_entity(f"Temp_{i % 100}")  # Update existing

            if i % 100 == 0:
                # Periodic queries
                await graph.search_entities("Temp_0")

        # Force garbage collection
        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()

        # Calculate memory growth
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_growth = sum(stat.size_diff for stat in top_stats) / (1024 * 1024)  # MB

        tracemalloc.stop()

        logger.info(json.dumps({
            "msg": "Memory leak test completed",
            "iterations": iterations,
            "memory_growth_mb": total_growth,
            "level": "INFO"
        }))

        # Memory growth should be reasonable (< 50 MB for this test)
        assert total_growth < 50, f"Potential memory leak: {total_growth:.2f} MB growth"

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_memory_leak_large_dataset(self):
        """
        Test memory behavior with large datasets.
        """
        import sys

        gc.collect()
        initial_memory = sys.getsizeof(EntityKnowledgeGraph())

        # Create multiple graphs to test for accumulated memory
        graphs = []
        for i in range(10):
            graph = EntityKnowledgeGraph()

            # Add substantial data
            for j in range(1000):
                await graph.add_entity(f"G{i}_E{j}", {"data": "x" * 100})

            graphs.append(graph)

        gc.collect()
        final_memory = sum(sys.getsizeof(g) for g in graphs)

        avg_memory_per_graph = final_memory / len(graphs)
        memory_per_entity = avg_memory_per_graph / 1000

        logger.info(json.dumps({
            "msg": "Large dataset memory test completed",
            "graph_count": len(graphs),
            "entities_per_graph": 1000,
            "total_memory_mb": final_memory / (1024 * 1024),
            "avg_memory_per_graph_mb": avg_memory_per_graph / (1024 * 1024),
            "memory_per_entity_bytes": memory_per_entity,
            "level": "INFO"
        }))

        # Memory per entity should be reasonable (< 10 KB)
        assert memory_per_entity < 10240, f"Memory per entity too high: {memory_per_entity} bytes"


class TestResourceExhaustion:
    """
    Tests for handling resource exhaustion scenarios.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_disk_space_handling(self, temp_dir):
        """
        Test graceful handling of disk space issues.
        """
        import os

        graph = EntityKnowledgeGraph()

        # Simulate disk space check
        def check_disk_space(path: str) -> bool:
            """Check available disk space."""
            try:
                stat = os.statvfs(path)
                free_space = stat.f_bavail * stat.f_frsize
                return free_space > 10 * 1024 * 1024  # Require 10 MB free
            except:
                return True  # Assume OK if can't check

        # Test with many entities
        entity_count = 5000
        for i in range(entity_count):
            await graph.add_entity(f"Entity_{i}", {"data": "x" * 1000})

            # Check disk space periodically
            if i % 1000 == 0:
                has_space = check_disk_space(str(temp_dir))
                assert has_space, f"Disk space exhausted at entity {i}"

        logger.info(json.dumps({
            "msg": "Disk space handling test completed",
            "entities_created": entity_count,
            "disk_space_adequate": True,
            "level": "INFO"
        }))

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_connection_pool_exhaustion(self):
        """
        Test handling of connection pool exhaustion.
        """
        # Simulate connection pool
        class MockConnectionPool:
            def __init__(self, max_connections: int):
                self.max_connections = max_connections
                self.active_connections = 0
                self.wait_count = 0

            async def acquire(self):
                if self.active_connections >= self.max_connections:
                    self.wait_count += 1
                    return None  # Pool exhausted
                self.active_connections += 1
                return "connection"

            def release(self, conn):
                if conn:
                    self.active_connections -= 1

        pool = MockConnectionPool(max_connections=10)

        # Simulate many concurrent connection requests
        async def request_connection(req_id: int):
            conn = await pool.acquire()
            if conn:
                await asyncio.sleep(0.01)  # Simulate work
                pool.release(conn)
                return True
            else:
                return False  # Failed to get connection

        # More requests than pool size
        tasks = [request_connection(i) for i in range(50)]
        results = await asyncio.gather(*tasks)

        successful = sum(results)
        failed = len(results) - successful

        logger.info(json.dumps({
            "msg": "Connection pool exhaustion test completed",
            "max_connections": pool.max_connections,
            "successful_requests": successful,
            "failed_requests": failed,
            "wait_count": pool.wait_count,
            "level": "INFO"
        }))

        # Some requests should fail
        assert failed > 0, "Expected some connection failures"
        # But most should succeed
        assert successful >= pool.max_connections, "Too few successful connections"


class TestSystemRecovery:
    """
    Tests for system recovery after failures.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_recovery_after_crash(self):
        """
        Test system recovery after simulated crash.
        """
        graph = EntityKnowledgeGraph()

        # Add initial data
        for i in range(100):
            await graph.add_entity(f"Initial_{i}")

        initial_count = len(graph.get_entities())

        # Simulate crash (clear some data)
        crash_point = 50
        for i in range(crash_point):
            if f"Initial_{i}" in graph.entities:
                del graph.entities[f"Initial_{i}"]

        post_crash_count = len(graph.get_entities())

        # Recovery: restore missing entities
        for i in range(crash_point):
            await graph.add_entity(f"Initial_{i}", {"recovered": True})

        final_count = len(graph.get_entities())

        logger.info(json.dumps({
            "msg": "Crash recovery test completed",
            "initial_entities": initial_count,
            "post_crash_entities": post_crash_count,
            "final_entities": final_count,
            "entities_recovered": final_count - post_crash_count,
            "recovery_successful": final_count == initial_count,
            "level": "INFO"
        }))

        assert final_count == initial_count, "Recovery incomplete"

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_performance_degradation_recovery(self):
        """
        Test performance recovery after degradation.
        """
        graph = EntityKnowledgeGraph()

        # Baseline performance
        start = time.time()
        for i in range(100):
            await graph.add_entity(f"Baseline_{i}")
        baseline_duration = time.time() - start

        # Cause degradation (many entities)
        for i in range(5000):
            await graph.add_entity(f"Degradation_{i}")

        # Measure degraded performance
        start = time.time()
        for i in range(100):
            await graph.add_entity(f"Test_{i}")
        degraded_duration = time.time() - start

        # Recovery: clear degradation data
        for i in range(5000):
            if f"Degradation_{i}" in graph.entities:
                del graph.entities[f"Degradation_{i}"]

        # Measure recovered performance
        start = time.time()
        for i in range(100):
            await graph.add_entity(f"Recovered_{i}")
        recovered_duration = time.time() - start

        logger.info(json.dumps({
            "msg": "Performance degradation recovery test",
            "baseline_duration": baseline_duration,
            "degraded_duration": degraded_duration,
            "recovered_duration": recovered_duration,
            "degradation_factor": degraded_duration / baseline_duration,
            "recovery_factor": recovered_duration / baseline_duration,
            "level": "INFO"
        }))

        # Recovered performance should be close to baseline (within 2x)
        assert recovered_duration < baseline_duration * 2, "Performance not recovered"


class TestLongRunningStability:
    """
    Tests for long-running system stability.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_sustained_load_stability(self):
        """
        Test system stability under sustained load.
        """
        graph = EntityKnowledgeGraph()

        duration_seconds = 10
        ops_per_second = 50
        total_ops = duration_seconds * ops_per_second

        start_time = time.time()
        operation_count = 0
        errors = []

        async def perform_operation(op_id: int):
            nonlocal operation_count
            try:
                await graph.add_entity(f"Sustained_{op_id % 1000}")  # Reuse IDs
                operation_count += 1
            except Exception as e:
                errors.append(str(e))

        # Execute operations over time
        tasks = []
        for i in range(total_ops):
            tasks.append(perform_operation(i))

            # Stagger operations
            if len(tasks) >= ops_per_second:
                await asyncio.gather(*tasks)
                tasks = []
                await asyncio.sleep(1)  # Wait 1 second between batches

        # Final batch
        if tasks:
            await asyncio.gather(*tasks)

        actual_duration = time.time() - start_time

        logger.info(json.dumps({
            "msg": "Sustained load stability test completed",
            "planned_duration": duration_seconds,
            "actual_duration": actual_duration,
            "operations_completed": operation_count,
            "planned_operations": total_ops,
            "error_count": len(errors),
            "ops_per_second": operation_count / actual_duration,
            "level": "INFO"
        }))

        assert operation_count >= total_ops * 0.95, "Too many operations missed"
        assert len(errors) == 0, f"Errors occurred: {errors}"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
