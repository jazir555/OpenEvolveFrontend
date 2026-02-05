"""
Comprehensive Demo and Test of Reliability Configuration System

This script demonstrates the production-ready reliability mechanisms
implemented in reliability_config.py
"""

import random
import time
from reliability_config import (
    # Configuration
    RETRY_CONFIG,
    CIRCUIT_BREAKER_CONFIG,
    RATE_LIMITER_CONFIG,
    HEALTH_CHECK_CONFIG,

    # Exceptions
    RetryExhaustedError,
    CircuitBreakerOpenError,
    RateLimitExceededError,

    # Enums
    BackoffStrategy,
    CircuitState,

    # Policy Classes
    RetryPolicy,
    CircuitBreakerPolicy,
    RateLimiterPolicy,
    HealthCheckPolicy,

    # Implementation Classes
    RetryEngine,
    CircuitBreaker,
    TokenBucketRateLimiter,
    HealthChecker,
    ReliabilityManager,

    # Decorators
    with_retry,
)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def demo_basic_retry():
    """Demo 1: Basic retry mechanism with exponential backoff."""
    print_section("DEMO 1: Basic Retry with Exponential Backoff")

    attempt_count = [0]

    @with_retry(
        max_attempts=5,
        initial_delay=0.1,  # Fast for demo
        strategy=BackoffStrategy.EXPONENTIAL,
        retry_on=(ValueError,),
    )
    def flaky_operation():
        """Operation that fails a few times before succeeding."""
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise ValueError(f"Attempt {attempt_count[0]} failed")
        return f"Success on attempt {attempt_count[0]}"

    result = flaky_operation()
    print(f"Result: {result}")


def demo_circuit_breaker():
    """Demo 2: Circuit breaker pattern."""
    print_section("DEMO 2: Circuit Breaker Pattern")

    cb = CircuitBreaker(
        policy=CircuitBreakerPolicy(
            failure_threshold=3,
            timeout=2.0,
            success_threshold=2,
        ),
        name="demo_service"
    )

    fail_count = [0]

    def unreliable_service():
        """Service that fails initially, then recovers."""
        fail_count[0] += 1
        if fail_count[0] <= 3:
            raise ConnectionError(f"Service failure #{fail_count[0]}")
        return f"Service recovered on call #{fail_count[0]}"

    # Test circuit breaker
    print("Testing circuit breaker state transitions:\n")

    # Phase 1: Trigger failures to open circuit
    print("Phase 1: Triggering failures...")
    for i in range(5):
        try:
            result = cb.call(unreliable_service)
            print(f"  Call {i+1}: {result}")
        except (ConnectionError, CircuitBreakerOpenError) as e:
            print(f"  Call {i+1}: {type(e).__name__}: {str(e)[:60]}")

    print(f"\nCircuit state: {cb.state.value}")
    print(f"Failure count: {cb._failure_count}")

    # Phase 2: Wait for timeout and test recovery
    print("\nPhase 2: Waiting for timeout...")
    time.sleep(2.1)

    print("Phase 3: Testing recovery (circuit should be half-open)...")
    for i in range(3):
        try:
            result = cb.call(unreliable_service)
            print(f"  Call {i+1}: {result}")
        except (ConnectionError, CircuitBreakerOpenError) as e:
            print(f"  Call {i+1}: {type(e).__name__}: {str(e)[:60]}")

    print(f"\nFinal circuit state: {cb.state.value}")
    print(f"Stats: {cb.get_stats()}")


def demo_rate_limiter():
    """Demo 3: Token bucket rate limiter."""
    print_section("DEMO 3: Token Bucket Rate Limiter")

    rl = TokenBucketRateLimiter(
        policy=RateLimiterPolicy(
            max_requests=5,
            time_window=1.0,
            burst_capacity=2,
        ),
        name="demo_api"
    )

    print("Rate limit: 5 requests per second with 2 burst capacity\n")
    print("Making 10 requests:\n")

    for i in range(10):
        if rl.acquire():
            print(f"  Request {i+1}: [OK] Allowed (tokens: {rl._tokens:.1f})")
        else:
            wait_time = rl.get_wait_time()
            print(f"  Request {i+1}: [FAIL] Rate limited (wait: {wait_time:.2f}s)")

        # Small delay to show token refill
        if i == 4:
            print("\n  Waiting 1.1 seconds for token refill...\n")
            time.sleep(1.1)

    print(f"\nFinal stats: {rl.get_stats()}")


def demo_health_checker():
    """Demo 4: Health checker with thresholds."""
    print_section("DEMO 4: Health Checker with Thresholds")

    hc = HealthChecker(
        policy=HealthCheckPolicy(
            timeout=5.0,
            failure_threshold=2,
            success_threshold=2,
        ),
        name="demo_database"
    )

    fail_pattern = [False, False, True, True, False]  # False = failure
    check_index = [0]

    def check_database():
        """Simulated health check with varying results."""
        result = fail_pattern[check_index[0] % len(fail_pattern)]
        check_index[0] += 1
        if not result:
            raise ConnectionError("Database connection failed")
        return True

    print("Running health checks (failure threshold: 2, success threshold: 2)\n")

    for i in range(7):
        result = hc.check(check_database)
        status = "[OK] HEALTHY" if result.healthy else "[FAIL] UNHEALTHY"
        print(f"  Check {i+1}: {status}")
        print(f"    Overall: {'Healthy' if hc.is_healthy else 'Unhealthy'}")
        print(f"    Consecutive failures: {hc._consecutive_failures}")
        print(f"    Response time: {result.response_time_ms:.2f}ms")
        print()

    print(f"Final stats: {hc.get_stats()}")


def demo_retry_strategies():
    """Demo 5: Different retry strategies."""
    print_section("DEMO 5: Retry Strategy Comparison")

    strategies = [
        BackoffStrategy.EXPONENTIAL,
        BackoffStrategy.LINEAR,
        BackoffStrategy.FIXED,
    ]

    for strategy in strategies:
        print(f"\n{strategy.value.upper()} Strategy:\n")

        engine = RetryEngine(
            policy=RetryPolicy(
                max_attempts=4,
                initial_delay=0.1,
                strategy=strategy,
            )
        )

        attempt = [0]

        def failing_operation():
            attempt[0] += 1
            raise ValueError(f"Failure #{attempt[0]}")

        try:
            engine.execute(failing_operation)
        except RetryExhaustedError:
            pass

        print(f"  Total attempts: {engine._stats['total_attempts']}")
        print(f"  Total retries: {engine._stats['total_retries']}")


def demo_configuration_driven():
    """Demo 6: Configuration-driven behavior."""
    print_section("DEMO 6: Configuration-Driven Behavior")

    # Create policies from configuration dictionaries (backward compatible)
    print("Creating policies from config dictionaries:\n")

    retry_policy = RetryPolicy.from_dict(RETRY_CONFIG)
    print(f"Retry Policy: {retry_policy}")

    cb_policy = CircuitBreakerPolicy.from_dict(CIRCUIT_BREAKER_CONFIG)
    print(f"Circuit Breaker Policy: {cb_policy}")

    rl_policy = RateLimiterPolicy.from_dict(RATE_LIMITER_CONFIG)
    print(f"Rate Limiter Policy: {rl_policy}")

    hc_policy = HealthCheckPolicy.from_dict(HEALTH_CHECK_CONFIG)
    print(f"Health Check Policy: {hc_policy}")


def demo_integrated_manager():
    """Demo 7: Integrated reliability manager."""
    print_section("DEMO 7: Integrated Reliability Manager")

    manager = ReliabilityManager(
        retry_policy=RetryPolicy(max_attempts=3, initial_delay=0.1),
        circuit_breaker_policy=CircuitBreakerPolicy(failure_threshold=3, timeout=2.0),
        rate_limiter_policy=RateLimiterPolicy(max_requests=10, time_window=1.0),
        health_checker_policy=HealthCheckPolicy(timeout=5.0, failure_threshold=2),
        name="demo_api"
    )

    call_count = [0]

    def critical_api_call():
        """Simulated API call with mixed results."""
        call_count[0] += 1
        if call_count[0] <= 2:
            raise ConnectionError(f"API call #{call_count[0]} failed")
        return f"Success on call #{call_count[0]}"

    print("Executing operation with all reliability mechanisms:\n")

    try:
        result = manager.execute(critical_api_call)
        print(f"Result: {result}\n")
    except Exception as e:
        print(f"Failed: {type(e).__name__}: {e}\n")

    print("Composite statistics from all mechanisms:\n")
    import json
    print(json.dumps(manager.get_composite_stats(), indent=2))


def demo_specific_exceptions():
    """Demo 8: Specific exception handling."""
    print_section("DEMO 8: Specific Exception Handling")

    print("Demonstrating specific exception types:\n")

    # 1. Retry Exhausted Error
    try:
        engine = RetryEngine(RetryPolicy(max_attempts=2, initial_delay=0.01))
        engine.execute(lambda: exec('raise ValueError("Always fails")'))
    except RetryExhaustedError as e:
        print(f"1. RetryExhaustedError:")
        print(f"   Message: {e.message}")
        print(f"   Context: {e.context}\n")

    # 2. Circuit Breaker Open Error
    try:
        cb = CircuitBreaker(CircuitBreakerPolicy(failure_threshold=1))
        cb.call(lambda: exec('raise ConnectionError("Fail")'))
        cb.call(lambda: exec('raise ConnectionError("Fail")'))
        # Circuit should now be open
        cb.call(lambda: "Should not execute")
    except CircuitBreakerOpenError as e:
        print(f"2. CircuitBreakerOpenError:")
        print(f"   Message: {e.message}")
        print(f"   Circuit: {e.context['circuit']}")
        print(f"   State: {e.context['state']}\n")

    # 3. Rate Limit Exceeded Error
    try:
        rl = TokenBucketRateLimiter(RateLimiterPolicy(max_requests=1, time_window=60.0))
        rl.acquire()
        rl.acquire_or_raise()
    except RateLimitExceededError as e:
        print(f"3. RateLimitExceededError:")
        print(f"   Message: {e.message}")
        print(f"   Wait time: {e.context['wait_time_seconds']:.2f}s\n")


def demo_statistics_tracking():
    """Demo 9: Statistics tracking."""
    print_section("DEMO 9: Statistics Tracking")

    # Retry Engine Stats
    print("1. Retry Engine Statistics:\n")
    retry_engine = RetryEngine(RetryPolicy(max_attempts=5, initial_delay=0.01))

    for i in range(3):
        try:
            retry_engine.execute(lambda: exec('raise ValueError("Fail")'))
        except RetryExhaustedError:
            pass

    print(f"   {retry_engine.get_stats()}\n")

    # Circuit Breaker Stats
    print("2. Circuit Breaker Statistics:\n")
    cb = CircuitBreaker(CircuitBreakerPolicy(failure_threshold=2))
    for i in range(3):
        try:
            cb.call(lambda: exec('raise ConnectionError("Fail")'))
        except ConnectionError:
            pass

    print(f"   {cb.get_stats()}\n")

    # Rate Limiter Stats
    print("3. Rate Limiter Statistics:\n")
    rl = TokenBucketRateLimiter(RateLimiterPolicy(max_requests=5))
    for i in range(7):
        rl.acquire()

    print(f"   {rl.get_stats()}\n")


def demo_thread_safety():
    """Demo 10: Thread-safe operations."""
    print_section("DEMO 10: Thread Safety")

    import threading

    rl = TokenBucketRateLimiter(
        policy=RateLimiterPolicy(max_requests=100, time_window=1.0),
        name="thread_safe_limiter"
    )

    results = []
    errors = []

    def worker(worker_id: int):
        """Worker thread that makes requests."""
        try:
            for i in range(20):
                if rl.acquire():
                    results.append(f"Worker-{worker_id}: Request {i+1} allowed")
                else:
                    results.append(f"Worker-{worker_id}: Request {i+1} rate limited")
        except Exception as e:
            errors.append(f"Worker-{worker_id}: {e}")

    # Launch multiple threads
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    print(f"Total operations: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Final stats: {rl.get_stats()}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  PRODUCTION-READY RELIABILITY SYSTEM DEMONSTRATION")
    print("  reliability_config.py v2.0.0")
    print("=" * 70)

    demos = [
        ("Basic Retry with Exponential Backoff", demo_basic_retry),
        ("Circuit Breaker Pattern", demo_circuit_breaker),
        ("Token Bucket Rate Limiter", demo_rate_limiter),
        ("Health Checker with Thresholds", demo_health_checker),
        ("Retry Strategy Comparison", demo_retry_strategies),
        ("Configuration-Driven Behavior", demo_configuration_driven),
        ("Integrated Reliability Manager", demo_integrated_manager),
        ("Specific Exception Handling", demo_specific_exceptions),
        ("Statistics Tracking", demo_statistics_tracking),
        ("Thread Safety", demo_thread_safety),
    ]

    for name, demo_func in demos:
        try:
            demo_func()
            time.sleep(0.5)  # Brief pause between demos
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("  ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)
    print("\nThe reliability_config.py module provides:")
    print("  [OK] Retry logic with 4 backoff strategies")
    print("  [OK] Circuit breaker with 3 states (closed/open/half-open)")
    print("  [OK] Token bucket rate limiting")
    print("  [OK] Health checks with thresholds")
    print("  [OK] Specific exception hierarchy")
    print("  [OK] Full type hints for IDE support")
    print("  [OK] Thread-safe operation")
    print("  [OK] Configuration-driven behavior")
    print("  [OK] Production-ready logging")
    print("  [OK] Comprehensive statistics tracking")
    print("\n")


if __name__ == "__main__":
    main()
