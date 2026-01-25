"""
Unit Tests for Circuit Breaker Components

Tests for the circuit breaker system including level breakers,
hierarchical management, and fault isolation.
"""

import pytest
import asyncio
import time
from bubblelabs_nodes import (
    CircuitBreakerState,
    LevelCircuitBreaker,
    HierarchicalCircuitBreakerManager,
    CircuitBreakerConfig,
    create_circuit_breaker_manager,
)


class TestLevelCircuitBreaker:
    """Tests for LevelCircuitBreaker"""

    @pytest.mark.asyncio
    async def test_initial_state(self):
        """Test initial state is CLOSED"""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = LevelCircuitBreaker(level=0, config=config)

        assert breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_success_stays_closed(self):
        """Test successful calls keep breaker closed"""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = LevelCircuitBreaker(level=0, config=config)

        async def success_operation():
            return {'result': 'success'}

        # Multiple successful calls
        for _ in range(5):
            success, result, error = await breaker.execute(
                success_operation,
                context={}
            )
            assert success is True
            assert breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_failures_open_circuit(self):
        """Test failures open the circuit"""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = LevelCircuitBreaker(level=0, config=config)

        async def failing_operation():
            raise ValueError("Failed!")

        # Fail 3 times to reach threshold
        for i in range(3):
            success, result, error = await breaker.execute(
                failing_operation,
                context={}
            )

        # Circuit should be OPEN
        assert breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_open_circuit_blocks_calls(self):
        """Test open circuit blocks calls"""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = LevelCircuitBreaker(level=0, config=config)

        async def failing_operation():
            raise ValueError("Failed!")

        # Trigger circuit open
        for _ in range(2):
            await breaker.execute(failing_operation, {})

        assert breaker.state == CircuitBreakerState.OPEN

        # Next call should be blocked
        success, result, error = await breaker.execute(
            failing_operation,
            {}
        )

        assert success is False
        assert 'circuit open' in error.lower()

    @pytest.mark.asyncio
    async def test_half_open_state(self):
        """Test half-open state after recovery timeout"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=0.1,  # 100ms
            half_open_max_calls=2
        )
        breaker = LevelCircuitBreaker(level=0, config=config)

        # Open the circuit
        async def failing_operation():
            raise ValueError("Failed!")

        for _ in range(2):
            await breaker.execute(failing_operation, {})

        assert breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Next call should put in half-open
        async def success_operation():
            return {'result': 'success'}

        success, result, error = await breaker.execute(
            success_operation,
            {}
        )

        # Should be in half-open now
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        # Successful call should close circuit
        success, result, error = await breaker.execute(
            success_operation,
            {}
        )

        assert breaker.state == CircuitBreakerState.CLOSED


class TestHierarchicalCircuitBreakerManager:
    """Tests for HierarchicalCircuitBreakerManager"""

    def test_get_breaker(self):
        """Test getting breaker for level"""
        manager = create_circuit_breaker_manager()

        breaker_0 = manager.get_breaker(level=0)
        breaker_1 = manager.get_breaker(level=1)

        assert breaker_0.level == 0
        assert breaker_1.level == 1

    @pytest.mark.asyncio
    async def test_isolated_levels(self):
        """Test that levels are isolated"""
        manager = create_circuit_breaker_manager()

        config = CircuitBreakerConfig(failure_threshold=2)

        # Open level 0 breaker
        breaker_0 = manager.get_breaker(level=0, config=config)

        async def failing_operation():
            raise ValueError("Failed!")

        # Fail level 0
        for _ in range(2):
            await breaker_0.execute(failing_operation, {})

        assert breaker_0.state == CircuitBreakerState.OPEN

        # Level 1 should still be closed
        breaker_1 = manager.get_breaker(level=1, config=config)
        assert breaker_1.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_execute_at_level(self):
        """Test execute_at_level method"""
        manager = create_circuit_breaker_manager()

        async def test_operation():
            return {'result': 'success'}

        success, result, error = await manager.execute_at_level(
            level=0,
            operation=test_operation,
            context={}
        )

        assert success is True
        assert result['result'] == 'success'

    @pytest.mark.asyncio
    async def test_get_all_states(self):
        """Test getting all circuit states"""
        manager = create_circuit_breaker_manager()

        states = manager.get_all_states()

        assert 0 in states
        assert 1 in states
        assert 2 in states

        # All should start closed
        for state in states.values():
            assert state == CircuitBreakerState.CLOSED


class TestCircuitBreakerScenarios:
    """Integration tests for circuit breaker scenarios"""

    @pytest.mark.asyncio
    async def test_cascading_failure(self):
        """Test cascading failure across levels"""
        manager = create_circuit_breaker_manager()

        async def failing_operation():
            raise ValueError("Failed!")

        # Fail level 0
        breaker_0 = manager.get_breaker(level=0)
        for _ in range(5):
            await breaker_0.execute(failing_operation, {})

        assert breaker_0.state == CircuitBreakerState.OPEN

        # Level 1 should still work
        breaker_1 = manager.get_breaker(level=1)

        async def success_operation():
            return {'result': 'success'}

        success, result, error = await breaker_1.execute(
            success_operation,
            {}
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_recovery_after_timeout(self):
        """Test circuit recovery after timeout"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=0.1
        )

        breaker = LevelCircuitBreaker(level=0, config=config)

        # Open the circuit
        async def failing_operation():
            raise ValueError("Failed!")

        for _ in range(2):
            await breaker.execute(failing_operation, {})

        assert breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Successful call should close circuit
        async def success_operation():
            return {'result': 'success'}

        success, result, error = await breaker.execute(
            success_operation,
            {}
        )

        assert breaker.state == CircuitBreakerState.CLOSED


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
