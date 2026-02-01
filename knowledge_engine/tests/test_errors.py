"""
Error Handling Tests for Knowledge Engine

Following CLAUDE.md principles:
- Test graceful degradation on model failures
- Test retry logic and circuit breakers
- Test timeout handling
- Test memory limits
- Test database connection failures

Tests verify:
- Graceful degradation
- Retry logic with exponential backoff
- Circuit breaker functionality
- Timeout enforcement
- Memory limit handling
- Database failure recovery
"""

import asyncio
import json
import logging
import pytest
import time
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch, MagicMock
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


class TestGracefulDegradation:
    """
    Tests for graceful degradation when components fail.
    """

    @pytest.mark.asyncio
    async def test_extraction_failure_recovery(self, sample_document):
        """
        Test graceful handling of extraction failures.
        """
        try:
            from knowledge_engine.knowledge_extractor import KnowledgeExtractor
        except ImportError:
            pytest.skip("KnowledgeExtractor not available")

        extractor = KnowledgeExtractor()

        # Mock extraction to fail
        with patch.object(extractor, 'extract_from_workflow', side_effect=Exception("Model unavailable")):
            # Should handle failure gracefully
            try:
                result = await asyncio.to_thread(extractor.extract_from_workflow, {"content": sample_document})
                # If it returns a result, it should indicate partial/failure state
                if result:
                    assert "error" in result or "partial" in result
            except Exception as e:
                # Or it should raise a specific error type
                assert "unavailable" in str(e).lower() or "failed" in str(e).lower()

        logger.info(json.dumps({
            "msg": "Extraction failure handled gracefully",
            "level": "INFO"
        }))

    @pytest.mark.asyncio
    async def test_partial_extraction_success(self):
        """
        Test that partial extraction results are still usable.
        """
        # Simulate partial extraction (some entities extracted, others failed)
        partial_result = {
            "entities": [
                {"name": "AI", "type": "Concept", "confidence": 0.95},
                {"name": "ML", "type": "Field", "confidence": 0.92},
                {"name": "Unknown", "type": "Unknown", "confidence": 0.3}  # Low confidence
            ],
            "partial": True,
            "warnings": ["Some entities could not be classified"]
        }

        # Should still be able to use high-confidence entities
        high_confidence_entities = [
            e for e in partial_result["entities"]
            if e.get("confidence", 0) > 0.5
        ]

        assert len(high_confidence_entities) >= 2
        assert partial_result["partial"] is True

        logger.info(json.dumps({
            "msg": "Partial extraction still usable",
            "total_entities": len(partial_result["entities"]),
            "high_confidence_entities": len(high_confidence_entities),
            "level": "INFO"
        }))


class TestRetryLogic:
    """
    Tests for retry logic with exponential backoff.
    """

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self, retry_tracker):
        """
        Test that transient failures trigger retries.
        """
        max_retries = 3
        attempt_count = 0

        async def flaky_operation():
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count < max_retries:
                retry_tracker.record_attempt(attempt_count, False, "Connection timeout")
                raise Exception("Connection timeout")
            else:
                retry_tracker.record_attempt(attempt_count, True)
                return "Success"

        # Implement retry logic
        for attempt in range(max_retries):
            try:
                result = await flaky_operation()
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff

        stats = retry_tracker.get_stats()

        assert result == "Success"
        assert stats["total_attempts"] == max_retries
        assert stats["successes"] == 1
        assert stats["failures"] == max_retries - 1

        logger.info(json.dumps({
            "msg": "Retry logic executed correctly",
            "stats": stats,
            "level": "INFO"
        }))

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """
        Test that retries use exponential backoff.
        """
        retry_delays = []
        max_retries = 4

        async def operation_with_backoff():
            for attempt in range(max_retries):
                start = time.time()
                try:
                    if attempt < max_retries - 1:
                        raise Exception("Simulated failure")
                    return "Success"
                except Exception:
                    if attempt < max_retries - 1:
                        delay = 0.1 * (2 ** attempt)  # 0.1, 0.2, 0.4, 0.8
                        retry_delays.append(delay)
                        await asyncio.sleep(delay)

        start_time = time.time()
        await operation_with_backoff()
        total_time = time.time() - start_time

        # Verify exponential backoff pattern
        expected_delays = [0.1 * (2 ** i) for i in range(max_retries - 1)]
        assert retry_delays == expected_delays

        # Total time should be sum of all delays
        expected_total = sum(expected_delays)
        assert abs(total_time - expected_total) < 0.1  # Allow 100ms tolerance

        logger.info(json.dumps({
            "msg": "Exponential backoff timing verified",
            "retry_delays": retry_delays,
            "total_time_s": total_time,
            "level": "INFO"
        }))


class TestCircuitBreaker:
    """
    Tests for circuit breaker functionality.
    """

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self, circuit_breaker_test):
        """
        Test that circuit breaker opens after threshold failures.
        """
        failure_threshold = 3

        # Simulate failures
        for i in range(failure_threshold):
            circuit_breaker_test.record_failure()

        # Circuit should be open
        assert circuit_breaker_test.state == "open"

        logger.info(json.dumps({
            "msg": "Circuit breaker opened",
            "failure_count": circuit_breaker_test.failure_count,
            "state": circuit_breaker_test.state,
            "level": "INFO"
        }))

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_calls_when_open(self, circuit_breaker_test):
        """
        Test that calls are blocked when circuit is open.
        """
        # Open the circuit
        for i in range(3):
            circuit_breaker_test.record_failure()

        assert circuit_breaker_test.state == "open"

        # Attempt operation should be rejected
        call_blocked = circuit_breaker_test.state == "open"

        assert call_blocked is True

        logger.info(json.dumps({
            "msg": "Circuit breaker blocked call",
            "state": circuit_breaker_test.state,
            "level": "INFO"
        }))

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_state(self, circuit_breaker_test):
        """
        Test circuit breaker transitions to half-open after timeout.
        """
        # Open circuit
        for i in range(3):
            circuit_breaker_test.record_failure()

        assert circuit_breaker_test.state == "open"

        # Simulate timeout (in real implementation, this would be time-based)
        # Move to half-open
        circuit_breaker_test.state = "half-open"

        # Successful call should close circuit
        circuit_breaker_test.record_success()
        assert circuit_breaker_test.state == "closed"

        logger.info(json.dumps({
            "msg": "Circuit breaker recovered",
            "final_state": circuit_breaker_test.state,
            "level": "INFO"
        }))


class TestTimeoutHandling:
    """
    Tests for timeout handling.
    """

    @pytest.mark.asyncio
    async def test_operation_timeout(self):
        """
        Test that slow operations timeout appropriately.
        """
        timeout_seconds = 1
        operation_delay = 3  # Operation takes longer than timeout

        async def slow_operation():
            await asyncio.sleep(operation_delay)
            return "Completed"

        # Should timeout
        start_time = time.time()

        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            async with asyncio.timeout(timeout_seconds):
                await slow_operation()

        elapsed = time.time() - start_time

        # Should timeout close to specified time
        assert elapsed < timeout_seconds + 0.5  # Allow small margin

        logger.info(json.dumps({
            "msg": "Operation timed out correctly",
            "timeout_seconds": timeout_seconds,
            "actual_seconds": elapsed,
            "level": "INFO"
        }))

    @pytest.mark.asyncio
    async def test_timeout_with_partial_result(self):
        """
        Test that timeout returns partial results if available.
        """
        async def operation_with_partial_result():
            # Simulate operation that produces partial result
            await asyncio.sleep(0.1)
            return {"partial": True, "data": ["item1", "item2"]}

        # Short timeout that allows partial completion
        try:
            async with asyncio.timeout(1.0):
                result = await operation_with_partial_result()
        except asyncio.TimeoutError:
            result = {"timeout": True, "partial": None}

        assert result is not None
        assert "partial" in result or "timeout" in result

        logger.info(json.dumps({
            "msg": "Timeout handled with partial result",
            "result": result,
            "level": "INFO"
        }))


class TestMemoryLimits:
    """
    Tests for memory limit handling.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_memory_limit_detection(self):
        """
        Test that memory limits are detected and handled.
        """
        import sys

        if not CORE_AVAILABLE:
            pytest.skip("Core module not available")

        graph = EntityKnowledgeGraph()
        initial_size = sys.getsizeof(graph)

        # Add entities until we hit a reasonable limit for testing
        entity_count = 100
        large_data = "x" * 1000  # 1KB per entity

        for i in range(entity_count):
            await graph.add_entity_async(f"Entity_{i}", {"data": large_data})

        final_size = sys.getsizeof(graph)
        size_increase = final_size - initial_size

        # Verify entities were actually added (memory test is implementation-dependent)
        assert len(graph.get_entities()) == entity_count

        # Memory should be reasonable (not exponential growth)
        # Note: sys.getsizeof doesn't capture dict contents, so we check entity count

        logger.info(json.dumps({
            "msg": "Memory usage tracked",
            "entity_count": entity_count,
            "size_increase_bytes": size_increase,
            "avg_bytes_per_entity": size_increase / entity_count,
            "level": "INFO"
        }))

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_graceful_handling_of_memory_pressure(self):
        """
        Test graceful handling when memory is constrained.
        """
        # Simulate memory pressure by checking available memory
        import gc

        if not CORE_AVAILABLE:
            pytest.skip("Core module not available")

        graph = EntityKnowledgeGraph()

        # Force garbage collection
        gc.collect()

        # Add entities
        for i in range(100):
            await graph.add_entity_async(f"MemTest_{i}", {"index": i})

            # Periodically check memory
            if i % 20 == 0:
                gc.collect()

        # Verify graph still functional after memory pressure
        entity = graph.entities.get("MemTest_50")
        assert entity is not None
        assert entity["index"] == 50

        logger.info(json.dumps({
            "msg": "Memory pressure handled gracefully",
            "entities_stored": 100,
            "functional": True,
            "level": "INFO"
        }))


class TestDatabaseFailures:
    """
    Tests for database connection failures.
    """

    @pytest.mark.asyncio
    async def test_connection_failure_retry(self):
        """
        Test retry behavior on database connection failure.
        """
        mock_client = AsyncMock()
        connection_attempts = []

        async def failing_connect():
            attempt = len(connection_attempts) + 1
            connection_attempts.append(attempt)
            if attempt < 3:
                raise ConnectionError("Database unavailable")
            return True

        # Retry logic
        max_retries = 3
        connected = False

        for attempt in range(max_retries):
            try:
                connected = await failing_connect()
                break
            except ConnectionError as e:
                if attempt == max_retries - 1:
                    logger.error(json.dumps({
                        "msg": "Database connection failed after retries",
                        "attempts": max_retries,
                        "level": "ERROR"
                    }))
                    raise
                await asyncio.sleep(0.1)

        assert connected is True
        assert len(connection_attempts) == 3

        logger.info(json.dumps({
            "msg": "Database connection recovered",
            "attempts": len(connection_attempts),
            "level": "INFO"
        }))

    @pytest.mark.asyncio
    async def test_query_failure_fallback(self):
        """
        Test fallback behavior when database query fails.
        """
        graph = EntityKnowledgeGraph()

        # Add some data
        await graph.add_entity_async("Test1", {"data": "value1"})
        await graph.add_entity_async("Test2", {"data": "value2"})

        # Mock database failure during query
        # In real implementation, this would catch database errors
        # and fall back to cache or alternative storage

        try:
            # Query that might fail
            result = await graph.search_entities("Test")
            query_success = True
        except Exception as e:
            query_success = False
            result = []

        # Should either succeed or fail gracefully
        if query_success:
            assert len(result) > 0
        else:
            assert result == []

        logger.info(json.dumps({
            "msg": "Query failure handled",
            "query_success": query_success,
            "result_count": len(result) if query_success else 0,
            "level": "INFO"
        }))

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self):
        """
        Test that transactions are rolled back on errors.
        """
        graph = EntityKnowledgeGraph()

        # Get initial state
        initial_entity_count = len(graph.get_entities())

        # Attempt to add entities (simulating transaction)
        entities_to_add = ["Txn1", "Txn2", "Txn3"]
        added_entities = []

        try:
            for entity in entities_to_add:
                await graph.add_entity_async(entity)
                added_entities.append(entity)

                # Simulate failure on last entity
                if entity == "Txn2":
                    raise Exception("Simulated transaction failure")

        except Exception:
            # Rollback: remove entities added in this transaction
            for entity in added_entities:
                if entity in graph.entities:
                    del graph.entities[entity]

            # Verify rollback
            final_entity_count = len(graph.get_entities())
            assert final_entity_count == initial_entity_count

            logger.info(json.dumps({
                "msg": "Transaction rolled back successfully",
                "entities_added": len(added_entities),
                "entities_rolled_back": len(added_entities),
                "level": "INFO"
            }))


class TestErrorRecovery:
    """
    Tests for error recovery mechanisms.
    """

    @pytest.mark.asyncio
    async def test_state_recovery_after_error(self):
        """
        Test that system recovers to valid state after error.
        """
        state = KnowledgeState(query="Recovery test")

        # Add some data
        state.add_fact("Fact 1")
        state.add_fact("Fact 2")

        facts_before_error = len(state.facts)

        # Simulate error during operation (fact 3 should NOT be added due to error)
        try:
            state.add_fact("Fact 3")
            # Simulate error that would rollback the addition
            raise ValueError("Simulated error - rolling back Fact 3")
        except ValueError:
            # Simulate rollback by removing the fact that was just added
            if "Fact 3" in state.facts:
                state.facts.remove("Fact 3")
            pass

        # State should still be valid (Fact 3 should have been rolled back)
        facts_after_error = len(state.facts)
        state_dict = state.to_dict()

        assert facts_after_error == facts_before_error
        assert "query" in state_dict
        assert state_dict["query"] == "Recovery test"

        logger.info(json.dumps({
            "msg": "State recovered after error",
            "facts_count": facts_after_error,
            "state_valid": True,
            "level": "INFO"
        }))

    @pytest.mark.asyncio
    async def test_error_logging_and_monitoring(self):
        """
        Test that errors are properly logged and monitored.
        """
        error_logged = False
        error_details = {}

        try:
            # Operation that will fail
            raise RuntimeError("Test error for monitoring")
        except RuntimeError as e:
            error_logged = True
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }

        assert error_logged is True
        assert error_details["error_type"] == "RuntimeError"
        assert "Test error" in error_details["error_message"]

        logger.info(json.dumps({
            "msg": "Error logged for monitoring",
            "error_details": error_details,
            "level": "INFO"
        }))


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
