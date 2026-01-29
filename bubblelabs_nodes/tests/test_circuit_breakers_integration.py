"""
Integration Tests for Circuit Breaker System

Comprehensive integration tests for the hierarchical circuit breaker,
fault isolation, and automatic recovery system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from bubblelabs_nodes.circuit_breakers import (
    CircuitBreakerState,
    CircuitBreakerConfig,
    CircuitBreakerStrategy,
    LevelCircuitBreaker,
    HierarchicalCircuitBreakerManager,
    create_circuit_breaker_manager,
)


class TestCircuitBreakerIntegration:
    """Integration tests for complete circuit breaker workflow"""

    @pytest.mark.asyncio
    async def test_complete_breaker_lifecycle(self):
        """Test complete lifecycle: closed -> open -> half-open -> closed"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=0.1,
            half_open_max_calls=2
        )

        breaker = LevelCircuitBreaker(level=0, config=config)

        # Start closed
        assert breaker.state == CircuitBreakerState.CLOSED

        # Fail until open
        async def failing_operation():
            raise ValueError("Failed!")

        for _ in range(2):
            await breaker.execute(failing_operation, {})

        assert breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Success should transition to half-open, then closed
        async def success_operation():
            return {'result': 'success'}

        await breaker.execute(success_operation, {})
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        await breaker.execute(success_operation, {})
        assert breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_hierarchical_isolation(self):
        """Test that different levels are isolated"""
        manager = create_circuit_breaker_manager()

        config = CircuitBreakerConfig(failure_threshold=2)

        # Get breakers for different levels
        breaker_0 = manager.get_breaker(level=0, config=config)
        breaker_1 = manager.get_breaker(level=1, config=config)
        breaker_2 = manager.get_breaker(level=2, config=config)

        # Fail level 0
        async def fail():
            raise ValueError("Failed!")

        for _ in range(2):
            await breaker_0.execute(fail, {})

        assert breaker_0.state == CircuitBreakerState.OPEN

        # Levels 1 and 2 should still be closed
        assert breaker_1.state == CircuitBreakerState.CLOSED
        assert breaker_2.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_cascading_failure_prevention(self):
        """Test that circuit breakers prevent cascading failures"""
        manager = create_circuit_breaker_manager()

        # Simulate multi-level problem solving
        results = []

        async def solve_level_0():
            # This will fail
            raise ValueError("Level 0 failed")

        async def solve_level_1():
            # This should still work
            return {'level': 1, 'result': 'success'}

        async def solve_level_2():
            # This should also work
            return {'level': 2, 'result': 'success'}

        # Execute all levels
        config = CircuitBreakerConfig(failure_threshold=2)

        breaker_0 = manager.get_breaker(level=0, config=config)
        breaker_1 = manager.get_breaker(level=1, config=config)
        breaker_2 = manager.get_breaker(level=2, config=config)

        # Fail level 0 multiple times
        for _ in range(2):
            success, result, error = await breaker_0.execute(solve_level_0, {})
            results.append(('level_0', success, error))

        # Try level 0 again (should be blocked)
        success, result, error = await breaker_0.execute(solve_level_0, {})
        results.append(('level_0_blocked', success, error))

        # Level 1 and 2 should still work
        success1, result1, error1 = await breaker_1.execute(solve_level_1, {})
        success2, result2, error2 = await breaker_2.execute(solve_level_2, {})

        results.append(('level_1', success1, error1))
        results.append(('level_2', success2, error2))

        # Verify isolation
        assert breaker_0.state == CircuitBreakerState.OPEN
        assert breaker_1.state == CircuitBreakerState.CLOSED
        assert breaker_2.state == CircuitBreakerState.CLOSED

        # Verify level 0 blocked after opening
        assert not results[3][1]  # level_0_blocked should fail

    @pytest.mark.asyncio
    async def test_automatic_recovery(self):
        """Test automatic recovery after timeout"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=0.2
        )

        breaker = LevelCircuitBreaker(level=0, config=config)

        # Open the circuit
        async def fail():
            raise ValueError("Failed!")

        for _ in range(2):
            await breaker.execute(fail, {})

        assert breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery
        await asyncio.sleep(0.25)

        # Next call should succeed
        async def succeed():
            return {'result': 'success'}

        success, result, error = await breaker.execute(succeed, {})

        # Should transition through half-open to closed
        assert success is True

    @pytest.mark.asyncio
    async def test_concurrent_level_execution(self):
        """Test concurrent execution across multiple levels"""
        manager = create_circuit_breaker_manager()

        config = CircuitBreakerConfig(failure_threshold=3)

        # Create operations for multiple levels
        async def operation(level):
            await asyncio.sleep(0.1)
            return {'level': level, 'result': 'success'}

        # Execute concurrently
        tasks = []
        for level in range(5):
            breaker = manager.get_breaker(level=level, config=config)
            task = breaker.execute(operation, {'level': level})
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(success for success, _, _ in results)

    @pytest.mark.asyncio
    async def test_breaker_state_monitoring(self):
        """Test monitoring breaker states"""
        manager = create_circuit_breaker_manager()

        # Get initial states
        states = manager.get_all_states()

        assert all(state == CircuitBreakerState.CLOSED for state in states.values())

        # Modify one breaker
        breaker_0 = manager.get_breaker(level=0)
        breaker_0.state = CircuitBreakerState.OPEN

        # Check updated states
        updated_states = manager.get_all_states()

        assert updated_states[0] == CircuitBreakerState.OPEN
        assert all(state == CircuitBreakerState.CLOSED
                   for level, state in updated_states.items() if level > 0)

    @pytest.mark.asyncio
    async def test_breaker_with_timeouts(self):
        """Test breaker with timeout handling"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=1.0
        )

        breaker = LevelCircuitBreaker(level=0, config=config)

        # Operation that times out
        async def timeout_operation():
            await asyncio.sleep(5)
            return {'result': 'too late'}

        # This should count as failure
        success, result, error = await asyncio.wait_for(
            breaker.execute(timeout_operation, {}),
            timeout=0.5
        )

        # Should fail due to timeout
        assert success is False or 'timeout' in str(error).lower()

    @pytest.mark.asyncio
    async def test_level_specific_configuration(self):
        """Test different configurations per level"""
        manager = create_circuit_breaker_manager()

        # Configure different thresholds per level
        config_0 = CircuitBreakerConfig(failure_threshold=2)
        config_1 = CircuitBreakerConfig(failure_threshold=5)
        config_2 = CircuitBreakerConfig(failure_threshold=10)

        breaker_0 = manager.get_breaker(level=0, config=config_0)
        breaker_1 = manager.get_breaker(level=1, config=config_1)
        breaker_2 = manager.get_breaker(level=2, config=config_2)

        async def fail():
            raise ValueError("Failed!")

        # Fail level 0 twice (should open)
        for _ in range(2):
            await breaker_0.execute(fail, {})
        assert breaker_0.state == CircuitBreakerState.OPEN

        # Fail level 1 twice (should not open yet)
        for _ in range(2):
            await breaker_1.execute(fail, {})
        assert breaker_1.state == CircuitBreakerState.CLOSED

        # Fail level 2 five times (should not open yet)
        for _ in range(5):
            await breaker_2.execute(fail, {})
        assert breaker_2.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_retry_logic(self):
        """Test retry logic in half-open state"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=0.1,
            half_open_max_calls=2
        )

        breaker = LevelCircuitBreaker(level=0, config=config)

        # Open the circuit
        async def fail():
            raise ValueError("Failed!")

        for _ in range(2):
            await breaker.execute(fail, {})

        assert breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery
        await asyncio.sleep(0.15)

        # First success in half-open
        async def succeed():
            return {'result': 'success'}

        success, result, error = await breaker.execute(succeed, {})
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        # Failure in half-open should open again
        success, result, error = await breaker.execute(fail, {})
        assert breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_strategy_variants(self):
        """Test different circuit breaker strategies"""
        # Individual strategy
        manager_individual = create_circuit_breaker_manager(
            strategy=CircuitBreakerStrategy.INDIVIDUAL
        )

        # Hierarchical strategy
        manager_hierarchical = create_circuit_breaker_manager(
            strategy=CircuitBreakerStrategy.HIERARCHICAL
        )

        # Global strategy
        manager_global = create_circuit_breaker_manager(
            strategy=CircuitBreakerStrategy.GLOBAL
        )

        # All should create managers successfully
        assert manager_individual is not None
        assert manager_hierarchical is not None
        assert manager_global is not None

    @pytest.mark.asyncio
    async def test_breaker_with_context(self):
        """Test breaker execution with context"""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = LevelCircuitBreaker(level=0, config=config)

        context = {'attempt': 1, 'problem_id': 'test_123'}

        async def operation_with_context(ctx):
            assert ctx['attempt'] == 1
            assert ctx['problem_id'] == 'test_123'
            return {'result': 'success'}

        success, result, error = await breaker.execute(
            operation_with_context,
            context
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_breaker_metrics(self):
        """Test circuit breaker metrics collection"""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = LevelCircuitBreaker(level=0, config=config)

        # Get initial metrics
        initial_metrics = breaker.get_metrics()
        assert initial_metrics['failure_count'] == 0

        # Generate some failures
        async def fail():
            raise ValueError("Failed!")

        for _ in range(2):
            await breaker.execute(fail, {})

        # Check updated metrics
        updated_metrics = breaker.get_metrics()
        assert updated_metrics['failure_count'] == 2

        # Generate successes
        async def succeed():
            return {'result': 'success'}

        for _ in range(3):
            await breaker.execute(succeed, {})

        final_metrics = breaker.get_metrics()
        assert final_metrics['success_count'] >= 3


class TestCircuitBreakerFaultInjection:
    """Fault injection tests for circuit breaker"""

    @pytest.mark.asyncio
    async def test_transient_failure_recovery(self):
        """Test recovery from transient failures"""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout_seconds=0.1
        )

        breaker = LevelCircuitBreaker(level=0, config=config)

        # Simulate transient failures
        call_count = [0]

        async def sometimes_fail():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ValueError("Transient failure")
            return {'result': 'success'}

        # First two calls fail
        success1, _, _ = await breaker.execute(sometimes_fail, {})
        success2, _, _ = await breaker.execute(sometimes_fail, {})

        # Third call succeeds (breaker still closed, only 2 failures)
        success3, result3, _ = await breaker.execute(sometimes_fail, {})

        assert success3 is True

    @pytest.mark.asyncio
    async def test_sustained_failure_handling(self):
        """Test handling of sustained failures"""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout_seconds=0.2
        )

        breaker = LevelCircuitBreaker(level=0, config=config)

        # Sustained failure
        async def always_fail():
            raise ValueError("Sustained failure")

        # Fail enough to open circuit
        for _ in range(3):
            await breaker.execute(always_fail, {})

        assert breaker.state == CircuitBreakerState.OPEN

        # Subsequent calls should be blocked
        success, result, error = await breaker.execute(always_fail, {})

        assert success is False
        assert 'circuit open' in error.lower()

        # Wait for recovery
        await asyncio.sleep(0.25)

        # Should transition to half-open
        success, result, error = await breaker.execute(always_fail, {})

        # Should still fail but circuit should be in half-open
        assert breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_burst_traffic_handling(self):
        """Test handling burst traffic"""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = LevelCircuitBreaker(level=0, config=config)

        async def handle_request():
            await asyncio.sleep(0.01)
            return {'result': 'ok'}

        # Burst of requests
        tasks = [
            breaker.execute(handle_request, {})
            for _ in range(20)
        ]

        results = await asyncio.gather(*tasks)

        # Most should succeed
        success_count = sum(1 for success, _, _ in results if success)
        assert success_count >= 15  # Allow some failures


class TestCircuitBreakerConfiguration:
    """Configuration tests for circuit breaker"""

    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout_seconds=60,
            half_open_max_calls=3
        )

        valid, errors = config.validate()
        assert valid is True
        assert len(errors) == 0

        # Invalid config
        invalid_config = CircuitBreakerConfig(
            failure_threshold=0,  # Invalid
            recovery_timeout_seconds=-1,  # Invalid
            half_open_max_calls=0  # Invalid
        )

        valid, errors = invalid_config.validate()
        assert valid is False
        assert len(errors) == 3

    def test_default_configuration(self):
        """Test default configuration values"""
        config = CircuitBreakerConfig()

        assert config.enabled is True
        assert config.failure_threshold == 5
        assert config.recovery_timeout_seconds == 60
        assert config.half_open_max_calls == 3

    def test_configuration_from_dict(self):
        """Test creating configuration from dictionary"""
        config_dict = {
            'enabled': True,
            'strategy': 'hierarchical',
            'failure_threshold': 10,
            'recovery_timeout_seconds': 120,
            'half_open_max_calls': 5
        }

        config = CircuitBreakerConfig(**config_dict)

        assert config.failure_threshold == 10
        assert config.recovery_timeout_seconds == 120
        assert config.half_open_max_calls == 5

    def test_configuration_serialization(self):
        """Test configuration serialization"""
        config = CircuitBreakerConfig(
            failure_threshold=7,
            recovery_timeout_seconds=90
        )

        # Convert to dict
        config_dict = {
            'enabled': config.enabled,
            'strategy': config.strategy.value,
            'failure_threshold': config.failure_threshold,
            'recovery_timeout_seconds': config.recovery_timeout_seconds,
            'half_open_max_calls': config.half_open_max_calls
        }

        # Recreate from dict
        recreated = CircuitBreakerConfig(**config_dict)

        assert recreated.failure_threshold == config.failure_threshold
        assert recreated.recovery_timeout_seconds == config.recovery_timeout_seconds


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
