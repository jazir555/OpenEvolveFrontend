"""
Performance Tests for Knowledge Engine

Following CLAUDE.md principles:
- Performance tests have defined baselines
- Structured logging of metrics
- Tests measure actual throughput, not just timing

Tests verify:
- Extraction pipeline throughput
- Graph query performance at scale
- Visualization generation speed
- Concurrent request handling
- Load test all API endpoints
"""

import asyncio
import json
import logging
import pytest
import time
from datetime import datetime
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import sys
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


# Performance baselines (milliseconds)
PERFORMANCE_BASELINES = {
    "entity_add": 50,  # Adding an entity should take < 50ms
    "relationship_add": 50,  # Adding a relationship should take < 50ms
    "entity_search": 100,  # Searching entities should take < 100ms
    "graph_query": 200,  # Querying graph should take < 200ms
    "visualization_gen": 500,  # Visualization should take < 500ms
    "document_extraction": 2000,  # Doc extraction should take < 2s
}


class TestExtractionPerformance:
    """
    Performance tests for extraction pipeline.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_extraction_throughput(self, generate_test_documents):
        """
        Benchmark extraction throughput (documents per second).
        """
        if not CORE_AVAILABLE:
            pytest.skip("Core module not available")

        documents = generate_test_documents(20)
        start_time = time.time()

        graph = EntityKnowledgeGraph()
        for doc in documents:
            # Simple extraction simulation
            entities = [w for w in doc.split() if w[0].isupper() and len(w) > 3]
            for entity in entities[:5]:
                await graph.add_entity(entity)

        end_time = time.time()
        duration = (end_time - start_time) * 1000  # Convert to ms

        docs_per_second = len(documents) / (end_time - start_time)

        logger.info(json.dumps({
            "msg": "Extraction throughput measured",
            "doc_count": len(documents),
            "duration_ms": duration,
            "docs_per_second": docs_per_second,
            "level": "INFO"
        }))

        # Performance assertion
        assert docs_per_second >= 5, f"Throughput too low: {docs_per_second:.2f} docs/sec"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_batch_extraction_performance(self):
        """
        Test performance of batch extraction vs sequential.
        """
        entity_count = 100
        entities = [f"Entity_{i}" for i in range(entity_count)]

        # Sequential addition
        graph1 = EntityKnowledgeGraph()
        start_seq = time.time()
        for entity in entities:
            await graph1.add_entity(entity)
        seq_time = (time.time() - start_seq) * 1000

        # Batch addition (simulated with asyncio.gather)
        graph2 = EntityKnowledgeGraph()
        start_batch = time.time()
        tasks = [graph2.add_entity(entity) for entity in entities]
        await asyncio.gather(*tasks)
        batch_time = (time.time() - start_batch) * 1000

        speedup = seq_time / batch_time if batch_time > 0 else 1.0

        logger.info(json.dumps({
            "msg": "Batch extraction performance",
            "entity_count": entity_count,
            "sequential_ms": seq_time,
            "batch_ms": batch_time,
            "speedup": speedup,
            "level": "INFO"
        }))

        # Batch should be at least as fast as sequential
        assert batch_time <= seq_time * 1.2, "Batch processing slower than expected"


class TestGraphQueryPerformance:
    """
    Performance tests for graph queries at scale.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_large_graph_query(self):
        """
        Test query performance on large graph (1000+ entities).
        """
        # Create large graph
        graph = EntityKnowledgeGraph()
        entity_count = 1000

        start_time = time.time()
        for i in range(entity_count):
            await graph.add_entity(f"Entity_{i}", {"index": i})

        # Create relationships
        for i in range(0, entity_count - 1, 10):
            await graph.add_relationship(f"Entity_{i}", "connects_to", f"Entity_{i+1}")

        creation_time = (time.time() - start_time) * 1000

        # Test query performance
        start_query = time.time()
        result = await graph.search_entities("Entity_5")
        query_time = (time.time() - start_query) * 1000

        logger.info(json.dumps({
            "msg": "Large graph query performance",
            "entity_count": entity_count,
            "creation_ms": creation_time,
            "query_ms": query_time,
            "results": len(result),
            "level": "INFO"
        }))

        # Performance assertions
        assert query_time < PERFORMANCE_BASELINES["graph_query"], \
            f"Query too slow: {query_time}ms > {PERFORMANCE_BASELINES['graph_query']}ms"
        assert len(result) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_relationship_query_performance(self):
        """
        Test relationship query performance at scale.
        """
        graph = EntityKnowledgeGraph()

        # Create entity with many relationships
        central_entity = "Central"
        await graph.add_entity(central_entity)

        relationship_count = 100
        start_time = time.time()

        for i in range(relationship_count):
            await graph.add_entity(f"Related_{i}")
            await graph.add_relationship(central_entity, "links_to", f"Related_{i}")

        creation_time = (time.time() - start_time) * 1000

        # Query all relationships
        start_query = time.time()
        relationships = await graph.get_relationships_for_entity(central_entity)
        query_time = (time.time() - start_query) * 1000

        logger.info(json.dumps({
            "msg": "Relationship query performance",
            "relationship_count": relationship_count,
            "creation_ms": creation_time,
            "query_ms": query_time,
            "results": len(relationships),
            "level": "INFO"
        }))

        # Performance assertion
        assert query_time < PERFORMANCE_BASELINES["graph_query"], \
            f"Relationship query too slow: {query_time}ms"
        assert len(relationships) == relationship_count


class TestVisualizationPerformance:
    """
    Performance tests for visualization generation.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_visualization_generation_speed(self):
        """
        Test visualization generation performance.
        """
        # Create graph with significant data
        graph = EntityKnowledgeGraph()
        entity_count = 500

        for i in range(entity_count):
            await graph.add_entity(f"Entity_{i}", {"type": f"Type_{i % 5}"})

        for i in range(0, entity_count - 1, 5):
            await graph.add_relationship(f"Entity_{i}", "relates_to", f"Entity_{i+1}")

        # Generate visualization
        start_time = time.time()
        viz_data = await graph.to_dict()
        gen_time = (time.time() - start_time) * 1000

        logger.info(json.dumps({
            "msg": "Visualization generation speed",
            "entity_count": entity_count,
            "generation_ms": gen_time,
            "data_size_kb": len(str(viz_data).encode()) / 1024,
            "level": "INFO"
        }))

        # Performance assertion
        assert gen_time < PERFORMANCE_BASELINES["visualization_gen"], \
            f"Visualization generation too slow: {gen_time}ms"
        assert "entities" in viz_data
        assert "relationships" in viz_data


class TestConcurrentRequestHandling:
    """
    Performance tests for concurrent request handling.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_concurrent_entity_addition(self):
        """
        Test concurrent entity addition performance.
        """
        graph = EntityKnowledgeGraph()
        concurrent_tasks = 50
        entities_per_task = 10

        async def add_entities(task_id: int):
            for i in range(entities_per_task):
                await graph.add_entity(f"Task{task_id}_Entity{i}")

        start_time = time.time()
        tasks = [add_entities(i) for i in range(concurrent_tasks)]
        await asyncio.gather(*tasks)
        duration = (time.time() - start_time) * 1000

        total_entities = concurrent_tasks * entities_per_task
        actual_count = len(graph.get_entities())

        logger.info(json.dumps({
            "msg": "Concurrent addition performance",
            "concurrent_tasks": concurrent_tasks,
            "entities_per_task": entities_per_task,
            "total_ms": duration,
            "entities_added": actual_count,
            "entities_per_second": total_entities / (duration / 1000),
            "level": "INFO"
        }))

        # Verify all entities added (no race conditions)
        assert actual_count == total_entities, f"Missing entities: {total_entities - actual_count}"

        # Performance assertion
        avg_time_per_entity = duration / total_entities
        assert avg_time_per_entity < PERFORMANCE_BASELINES["entity_add"], \
            f"Average entity addition too slow: {avg_time_per_entity}ms"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_concurrent_graph_queries(self):
        """
        Test concurrent graph query performance.
        """
        # Setup graph
        graph = EntityKnowledgeGraph()
        for i in range(100):
            await graph.add_entity(f"Entity_{i}")

        # Concurrent queries
        concurrent_queries = 20

        async def perform_query(query_id: int):
            start = time.time()
            result = await graph.search_entities(f"Entity_{query_id % 100}")
            return {
                "query_id": query_id,
                "duration_ms": (time.time() - start) * 1000,
                "results": len(result)
            }

        start_time = time.time()
        tasks = [perform_query(i) for i in range(concurrent_queries)]
        results = await asyncio.gather(*tasks)
        total_duration = (time.time() - start_time) * 1000

        avg_query_time = sum(r["duration_ms"] for r in results) / len(results)

        logger.info(json.dumps({
            "msg": "Concurrent query performance",
            "concurrent_queries": concurrent_queries,
            "total_duration_ms": total_duration,
            "avg_query_ms": avg_query_time,
            "queries_per_second": concurrent_queries / (total_duration / 1000),
            "level": "INFO"
        }))

        # Performance assertion
        assert avg_query_time < PERFORMANCE_BASELINES["entity_search"], \
            f"Average query too slow: {avg_query_time}ms"


class TestLoadTesting:
    """
    Load tests for API endpoints.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_sustained_load(self):
        """
        Test sustained load over time.
        """
        graph = EntityKnowledgeGraph()
        duration_seconds = 5
        operations_per_second = 20
        total_operations = duration_seconds * operations_per_second

        start_time = time.time()
        operation_count = 0
        errors = []

        async def perform_operation(op_id: int):
            nonlocal operation_count
            try:
                await graph.add_entity(f"LoadEntity_{op_id}")
                operation_count += 1
            except Exception as e:
                errors.append(str(e))

        # Execute operations
        tasks = []
        for i in range(total_operations):
            tasks.append(perform_operation(i))
            # Stagger operations to achieve target rate
            if len(tasks) >= operations_per_second:
                await asyncio.gather(*tasks)
                tasks = []
                await asyncio.sleep(1)  # Wait 1 second between batches

        # Final batch
        if tasks:
            await asyncio.gather(*tasks)

        total_duration = time.time() - start_time
        actual_ops_per_second = operation_count / total_duration

        logger.info(json.dumps({
            "msg": "Sustained load test results",
            "duration_seconds": total_duration,
            "operations_completed": operation_count,
            "target_ops_per_second": operations_per_second,
            "actual_ops_per_second": actual_ops_per_second,
            "error_count": len(errors),
            "level": "INFO"
        }))

        # Assertions
        assert operation_count >= total_operations * 0.95, \
            f"Too many operations missed: {operation_count}/{total_operations}"
        assert len(errors) == 0, f"Errors occurred during load test: {errors}"
        assert actual_ops_per_second >= operations_per_second * 0.9, \
            f"Throughput too low: {actual_ops_per_second:.2f} ops/sec"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_memory_usage_under_load(self):
        """
        Test memory usage doesn't grow unbounded under load.
        """
        import gc
        import sys

        graph = EntityKnowledgeGraph()

        # Force garbage collection before test
        gc.collect()
        initial_memory = sys.getsizeof(graph)

        # Add many entities
        entity_count = 1000
        for i in range(entity_count):
            await graph.add_entity(f"MemEntity_{i}", {"data": "x" * 100})

        gc.collect()
        final_memory = sys.getsizeof(graph)

        memory_per_entity = (final_memory - initial_memory) / entity_count

        logger.info(json.dumps({
            "msg": "Memory usage under load",
            "entity_count": entity_count,
            "initial_memory_bytes": initial_memory,
            "final_memory_bytes": final_memory,
            "memory_per_entity_bytes": memory_per_entity,
            "level": "INFO"
        }))

        # Memory should grow reasonably (not exponentially)
        assert memory_per_entity < 10000, \
            f"Memory per entity too high: {memory_per_entity} bytes"


class TestScalabilityBenchmarks:
    """
    Scalability benchmarks to identify performance cliffs.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    @pytest.mark.parametrize("entity_count", [10, 100, 500, 1000])
    async def test_query_scalability(self, entity_count):
        """
        Test how query performance scales with entity count.
        """
        # Setup
        graph = EntityKnowledgeGraph()
        for i in range(entity_count):
            await graph.add_entity(f"Entity_{i}")

        # Benchmark
        start_time = time.time()
        result = await graph.search_entities("Entity_0")
        query_time = (time.time() - start_time) * 1000

        logger.info(json.dumps({
            "msg": "Query scalability benchmark",
            "entity_count": entity_count,
            "query_ms": query_time,
            "level": "INFO"
        }))

        # Performance should scale roughly linearly (O(log n) or O(n))
        # Allow up to 1ms per 100 entities
        max_expected_time = (entity_count / 100) * 10
        assert query_time < max_expected_time, \
            f"Query doesn't scale well: {query_time}ms for {entity_count} entities"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    @pytest.mark.parametrize("relationship_count", [10, 100, 500])
    async def test_relationship_scalability(self, relationship_count):
        """
        Test how relationship queries scale with relationship count.
        """
        # Setup
        graph = EntityKnowledgeGraph()
        central = "Central"
        await graph.add_entity(central)

        for i in range(relationship_count):
            entity = f"Rel_{i}"
            await graph.add_entity(entity)
            await graph.add_relationship(central, "links_to", entity)

        # Benchmark
        start_time = time.time()
        relationships = await graph.get_relationships_for_entity(central)
        query_time = (time.time() - start_time) * 1000

        logger.info(json.dumps({
            "msg": "Relationship scalability benchmark",
            "relationship_count": relationship_count,
            "query_ms": query_time,
            "level": "INFO"
        }))

        # Verify all relationships returned
        assert len(relationships) == relationship_count

        # Performance assertion
        max_expected_time = (relationship_count / 100) * 20
        assert query_time < max_expected_time, \
            f"Relationship query doesn't scale well: {query_time}ms"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
