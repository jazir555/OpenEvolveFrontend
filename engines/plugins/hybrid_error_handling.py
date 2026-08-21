"""
Error Handling for Hybrid MAKER Integration System

This module provides robust error handling:
- Custom exceptions
- Retry mechanisms
- Circuit breakers
- Error recovery

Author: OpenEvolve Hybrid Error Handling Team
Created: 2025-01-07
Version: 1.0.0
"""
from __future__ import annotations


import asyncio
import functools
import logging
import random
import time
from typing import Any, Dict, List, Optional, Callable, Type
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class HybridMakerError(Exception):
    """Base exception for hybrid MAKER system"""

    def __init__(self, message: str, error_code: str = "HYBRID_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class StrategyNotFoundError(HybridMakerError):
    """Raised when hybrid strategy is not found"""

    def __init__(self, strategy_name: str):
        super().__init__(
            f"Strategy not found: {strategy_name}",
            "STRATEGY_NOT_FOUND"
        )
        self.strategy_name = strategy_name


class PopulationError(HybridMakerError):
    """Raised when population operations fail"""

    def __init__(self, message: str):
        super().__init__(message, "POPULATION_ERROR")


class FitnessEvaluationError(HybridMakerError):
    """Raised when fitness evaluation fails"""

    def __init__(self, individual: Any, reason: str):
        super().__init__(
            f"Fitness evaluation failed: {reason}",
            "FITNESS_ERROR"
        )
        self.individual = individual


class DecompositionError(HybridMakerError):
    """Raised when task decomposition fails"""

    def __init__(self, theorem: str, reason: str):
        super().__init__(
            f"Decomposition failed for '{theorem}': {reason}",
            "DECOMPOSITION_ERROR"
        )
        self.theorem = theorem


class VotingError(HybridMakerError):
    """Raised when MAKER voting fails"""

    def __init__(self, reason: str):
        super().__init__(f"Voting failed: {reason}", "VOTING_ERROR")


# =============================================================================
# RETRY DECORATOR
# =============================================================================

def retry_on_error(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exceptions: tuple = (HybridMakerError,)
):
    """
    Decorator for retrying hybrid operations

    Example:
        @retry_on_error(max_retries=3)
        async def generate_proof(theorem):
            # Proof generation
            return result
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt >= max_retries - 1:
                        logger.error(f"Max retries exceeded for {func.__name__}")
                        raise

                    # Calculate delay
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    delay *= 0.5 + random.random()  # Jitter

                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {delay:.2f}s: {e}"
                    )

                    await asyncio.sleep(delay)

            raise last_exception

        return async_wrapper
    return decorator


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class HybridCircuitBreaker:
    """
    Circuit breaker for hybrid operations

    Prevents cascading failures by stopping operations
    when failure threshold is reached
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = HybridMakerError
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open

    def __call__(self, func: Callable) -> Callable:
        """Decorator usage"""

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Check if circuit is open
            if self.state == "open":
                if self.last_failure_time:
                    time_since_failure = time.time() - self.last_failure_time
                    if time_since_failure > self.recovery_timeout:
                        self.state = "half-open"
                        logger.info("Circuit breaker entering half-open state")
                    else:
                        raise HybridMakerError(
                            f"Circuit breaker is OPEN. Try again in "
                            f"{self.recovery_timeout - time_since_failure:.0f}s"
                        )

            try:
                result = await func(*args, **kwargs)

                # Success - close circuit if half-open
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                    logger.info("Circuit breaker closed after successful request")

                return result

            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.error(
                        f"Circuit breaker opened after {self.failure_count} failures. "
                        f"Will remain open for {self.recovery_timeout}s"
                    )

                raise

        return async_wrapper


# =============================================================================
# SAFE EXECUTION
# =============================================================================

async def safe_hybrid_execute(
    func: Callable,
    *args,
    fallback: Any = None,
    timeout: float = 30.0,
    **kwargs
) -> Any:
    """
    Safely execute hybrid function with error handling

    Args:
        func: Function to execute
        *args: Function arguments
        fallback: Fallback value on error
        timeout: Timeout in seconds
        **kwargs: Function keyword arguments

    Returns:
        Result or fallback
    """
    try:
        result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        return result
    except asyncio.TimeoutError:
        logger.error(f"Timeout in {func.__name__}")
        return fallback
    except HybridMakerError as e:
        logger.error(f"Hybrid error in {func.__name__}: {e}")
        return fallback
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"Unexpected error in {func.__name__}: {e}")
        return fallback


# =============================================================================
# DEMO / MAIN
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("Hybrid MAKER Error Handling")
    print("=" * 60)

    # Demo 1: Custom exceptions
    print("\n1. Custom Exceptions")
    print("-" * 40)

    try:
        raise StrategyNotFoundError("UnknownStrategy")
    except HybridMakerError as e:
        print(f"[OK] Caught: {e.error_code} - {e.message}")

    # Demo 2: Retry decorator
    print("\n2. Retry Decorator")
    print("-" * 40)

    call_count = [0]

    @retry_on_error(max_retries=3)
    async def flaky_hybrid_operation():
        call_count[0] += 1
        print(f"  Attempt {call_count[0]}...")

        if call_count[0] < 2:
            raise HybridMakerError("Temporary failure")

        print("  [OK] Success!")
        return {"success": True}

    result = asyncio.run(flaky_hybrid_operation())
    print(f"Result: {result}")

    # Demo 3: Circuit breaker
    async def demo_circuit_breaker():
        print("\n3. Circuit Breaker")
        print("-" * 40)

        circuit_breaker = HybridCircuitBreaker(failure_threshold=2, recovery_timeout=5.0)

        @circuit_breaker
        async def unreliable_hybrid_service():
            raise HybridMakerError("Service unavailable")

        # Trigger failures
        try:
            await unreliable_hybrid_service()
        except HybridMakerError as e:
            print(f"  [OK] Failure 1: {e.message}")

        try:
            await unreliable_hybrid_service()
        except HybridMakerError as e:
            print(f"  [OK] Failure 2: {e.message}")

        # Circuit should be open
        try:
            await unreliable_hybrid_service()
        except HybridMakerError as e:
            print(f"  [OK] Circuit breaker: {e.message}")

        print(f"  Circuit state: {circuit_breaker.state}")

    import asyncio
    asyncio.run(demo_circuit_breaker())

    # Demo 4: Safe execution
    print("\n4. Safe Execution")
    print("-" * 40)

    async def risky_hybrid_task():
        raise HybridMakerError("Task failed")

    result = asyncio.run(safe_hybrid_execute(
        risky_hybrid_task,
        fallback={"fallback": True}
    ))
    print(f"Result with fallback: {result}")

    print("\n" + "=" * 60)
    print("Error handling demo complete!")
