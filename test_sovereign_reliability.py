import pytest
import time
from sovereign_reliability import (
    RetryStrategy, with_retry, ErrorHandler, HealthMonitor, CircuitBreaker,
    ErrorSeverity, SovereignError, get_error_handler, get_health_monitor
)

class TestRetryStrategy:
    def test_retry_strategy_initialization(self):
        strategy = RetryStrategy(max_attempts=3, initial_delay=1.0)
        assert strategy.max_attempts == 3
        assert strategy.initial_delay == 1.0
    
    def test_get_delay_increases(self):
        strategy = RetryStrategy(initial_delay=1.0, exponential_base=2.0, jitter=False)
        delay1 = strategy.get_delay(0)
        delay2 = strategy.get_delay(1)
        delay3 = strategy.get_delay(2)
        assert delay1 < delay2 < delay3

class TestWithRetryDecorator:
    def test_successful_call_no_retry(self):
        call_count = [0]
        @with_retry(max_attempts=3)
        def successful_func():
            call_count[0] += 1
            return "success"
        result = successful_func()
        assert result == "success"
        assert call_count[0] == 1
    
    def test_retry_on_failure(self):
        call_count = [0]
        @with_retry(max_attempts=3, retry_on=(ValueError,))
        def failing_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Temporary failure")
            return "success"
        result = failing_func()
        assert result == "success"
        assert call_count[0] == 3
    
    def test_fallback_on_all_failures(self):
        def fallback_func():
            return "fallback_result"
        @with_retry(max_attempts=2, retry_on=(ValueError,), fallback=fallback_func)
        def always_fails():
            raise ValueError("Always fails")
        result = always_fails()
        assert result == "fallback_result"

class TestErrorHandler:
    def test_error_handler_initialization(self):
        handler = ErrorHandler()
        assert len(handler.error_log) == 0
        assert len(handler.error_counts) == 0
    
    def test_handle_error(self):
        handler = ErrorHandler()
        error = ValueError("Test error")
        error_info = handler.handle_error(error, severity=ErrorSeverity.MEDIUM)
        assert error_info['type'] == 'ValueError'
        assert error_info['message'] == 'Test error'
        assert error_info['severity'] == 'medium'
    
    def test_error_stats(self):
        handler = ErrorHandler()
        handler.handle_error(ValueError("Error 1"))
        handler.handle_error(ValueError("Error 2"))
        handler.handle_error(TypeError("Error 3"))
        stats = handler.get_error_stats()
        assert stats['total_errors'] == 3
        assert stats['error_counts']['ValueError'] == 2
        assert stats['error_counts']['TypeError'] == 1

class TestHealthMonitor:
    def test_health_monitor_initialization(self):
        monitor = HealthMonitor()
        assert len(monitor.checks) == 0
    
    def test_register_and_run_checks(self):
        monitor = HealthMonitor()
        def healthy_check():
            return True
        def unhealthy_check():
            return False
        monitor.register_check("test1", healthy_check)
        monitor.register_check("test2", unhealthy_check)
        results = monitor.run_health_checks()
        assert results['overall_healthy'] is False
        assert results['checks']['test1']['healthy'] is True
        assert results['checks']['test2']['healthy'] is False
    
    def test_get_health_status(self):
        monitor = HealthMonitor()
        monitor.register_check("test", lambda: True)
        monitor.run_health_checks()
        status = monitor.get_health_status()
        assert status['status'] == 'healthy'

class TestCircuitBreaker:
    def test_circuit_breaker_initialization(self):
        cb = CircuitBreaker(failure_threshold=3, timeout=1.0)
        assert cb.state == "closed"
        assert cb.failure_count == 0
    
    def test_circuit_opens_after_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        def failing_func():
            raise ValueError("Failure")
        for i in range(3):
            try:
                cb.call(failing_func)
            except ValueError:
                pass
        assert cb.state == "open"
    
    def test_circuit_stays_closed_on_success(self):
        cb = CircuitBreaker(failure_threshold=3)
        def successful_func():
            return "success"
        result = cb.call(successful_func)
        assert result == "success"
        assert cb.state == "closed"

class TestGlobalInstances:
    def test_get_error_handler(self):
        handler = get_error_handler()
        assert isinstance(handler, ErrorHandler)
    
    def test_get_health_monitor(self):
        monitor = get_health_monitor()
        assert isinstance(monitor, HealthMonitor)
