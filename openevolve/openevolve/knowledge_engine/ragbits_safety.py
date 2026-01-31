"""
Safety Wrapper for RAGBits Integration

Provides comprehensive safety checks, graceful degradation, and fallback behavior
for all RAGBits integration components. Ensures the system works correctly even when
RAGBits is unavailable or errors occur.
"""

import logging
import functools
from typing import Any, Callable, Optional, TypeVar
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


def safe_execute(
    fallback_value: Any = None,
    log_errors: bool = True,
    reraise: bool = False
) -> Callable:
    """
    Decorator for safe execution of functions that may fail.

    Ensures that decorated functions:
    - Never raise exceptions to callers (unless reraise=True)
    - Always return fallback_value on error
    - Log all errors appropriately
    - Handle cancellation gracefully

    Args:
        fallback_value: Value to return on error
        log_errors: Whether to log errors
        reraise: Whether to re-raise exceptions (default: False)

    Example:
        @safe_execute(fallback_value=[])
        async def search_function(query):
            # May fail, but will return [] on error
            return await ragbits.search(query)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(
                        f"❌ Error in {func.__name__}: {e}",
                        exc_info=True
                    )
                if reraise:
                    raise
                return fallback_value

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(
                        f"❌ Error in {func.__name__}: {e}",
                        exc_info=True
                    )
                if reraise:
                    raise
                return fallback_value

        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def validate_query(query: Any) -> bool:
    """
    Validate search query.

    Args:
        query: Query to validate

    Returns:
        True if query is valid, False otherwise
    """
    if not query:
        return False
    if not isinstance(query, str):
        return False
    if len(query.strip()) == 0:
        return False
    if len(query) > 10000:  # Prevent excessively long queries
        logger.warning("⚠️ Query too long, truncating to 10000 chars")
        return False
    return True


def validate_top_k(top_k: Any) -> int:
    """
    Validate and normalize top_k parameter.

    Args:
        top_k: top_k value to validate

    Returns:
        Normalized top_k value (between 1 and 100)
    """
    # Handle None or non-numeric types first
    if top_k is None:
        logger.warning("⚠️ top_k is None, using default 5")
        return 5

    if not isinstance(top_k, (int, float, str)):
        logger.warning(f"⚠️ Invalid top_k type {type(top_k)}, using default 5")
        return 5

    # Try to convert to int
    try:
        top_k_int = int(top_k)
    except (ValueError, TypeError):
        logger.warning("⚠️ Could not convert top_k to int, using default 5")
        return 5

    # Validate range
    if top_k_int < 1:
        logger.warning("⚠️ top_k too small, using minimum 1")
        return 1
    if top_k_int > 100:
        logger.warning("⚠️ top_k too large, capping at 100")
        return 100

    return top_k_int


def validate_filters(filters: Any) -> dict:
    """
    Validate and normalize filters parameter.

    Args:
        filters: Filters to validate

    Returns:
        Validated filters dictionary
    """
    if filters is None:
        return {}
    if not isinstance(filters, dict):
        logger.warning("⚠️ Invalid filters type, using empty dict")
        return {}

    # Sanitize filter values
    validated = {}
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool, list)):
            validated[key] = value
        elif isinstance(value, dict):
            validated[key] = validate_filters(value)
        else:
            logger.warning(f"⚠️ Invalid filter value for {key}, skipping")

    return validated


def generate_fallback_result(
    query: str,
    result_type: str = "search"
) -> dict:
    """
    Generate a fallback result when RAGBits is unavailable.

    Args:
        query: Original query
        result_type: Type of result (search, pattern, etc.)

    Returns:
        Fallback result dictionary
    """
    return {
        "content": f"Fallback result for: {query[:100]}",
        "score": 0.5,
        "metadata": {
            "source": "fallback",
            "type": result_type,
            "timestamp": datetime.utcnow().isoformat(),
            "fallback_reason": "RAGBits not available"
        }
    }


def generate_fallback_artifact_id() -> str:
    """
    Generate a fallback artifact ID.

    Returns:
        Fallback artifact ID
    """
    return f"fallback_artifact_{datetime.utcnow().timestamp()}"


class RAGBitsSafetyManager:
    """
    Centralized safety management for RAGBits operations.

    Provides:
    - Availability checking
    - Graceful degradation
    - Fallback generation
    - Error tracking
    """

    def __init__(self):
        self._error_counts = {}
        self._last_error_time = {}
        self._circuit_breaker_timeout = 60  # seconds
        self._circuit_breaker_until = {}

    def is_available(self, service: str = "ragbits") -> bool:
        """
        Check if a service is available.

        Args:
            service: Service name to check

        Returns:
            True if service is available, False otherwise
        """
        # Check circuit breaker
        if service in self._circuit_breaker_until:
            if datetime.utcnow().timestamp() < self._circuit_breaker_until[service]:
                logger.info(f"⚠️ Service '{service}' is in circuit breaker until "
                          f"{datetime.fromtimestamp(self._circuit_breaker_until[service])}")
                return False
            else:
                # Circuit breaker timeout expired
                logger.info(f"✅ Circuit breaker reset for '{service}'")
                del self._circuit_breaker_until[service]

        return True

    def record_error(self, service: str, error: Exception):
        """
        Record an error for a service.

        Args:
            service: Service name
            error: Exception that occurred
        """
        self._error_counts[service] = self._error_counts.get(service, 0) + 1
        self._last_error_time[service] = datetime.utcnow().timestamp()

        # Trigger circuit breaker after 3 consecutive errors
        if self._error_counts[service] >= 3:
            timeout = datetime.utcnow().timestamp() + self._circuit_breaker_timeout
            self._circuit_breaker_until[service] = timeout
            logger.warning(f"⚠️ Circuit breaker triggered for '{service}' for "
                          f"{self._circuit_breaker_timeout} seconds")

    def reset_errors(self, service: str):
        """
        Reset error count for a service.

        Args:
            service: Service name
        """
        self._error_counts[service] = 0
        logger.info(f"✅ Error count reset for '{service}'")

    def get_error_count(self, service: str) -> int:
        """
        Get error count for a service.

        Args:
            service: Service name

        Returns:
            Number of errors recorded
        """
        return self._error_counts.get(service, 0)


# Global safety manager instance
_safety_manager = RAGBitsSafetyManager()


def get_safety_manager() -> RAGBitsSafetyManager:
    """Get the global safety manager instance"""
    return _safety_manager


class SafeRAGBitsWrapper:
    """
    Safe wrapper for RAGBits operations with automatic fallback handling.

    All methods return sensible defaults on error and never raise exceptions.
    """

    def __init__(self, retriever=None):
        """
        Initialize the safe wrapper.

        Args:
            retriever: RAGBits retriever instance (optional)
        """
        self.retriever = retriever
        self.safety_manager = get_safety_manager()
        self.logger = logging.getLogger(__name__)

    @safe_execute(fallback_value=[])
    async def safe_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
        **kwargs
    ) -> list:
        """
        Safely execute search operation.

        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            **kwargs: Additional parameters

        Returns:
            Search results (empty list on error)
        """
        # Validate inputs
        if not validate_query(query):
            self.logger.warning("⚠️ Invalid query, returning empty results")
            return []

        top_k = validate_top_k(top_k)
        filters = validate_filters(filters)

        # Check if service is available
        if not self.safety_manager.is_available("ragbits"):
            self.logger.info("ℹ️ RAGBits service unavailable, using fallback")
            return [generate_fallback_result(query, "search")]

        # Execute search if retriever exists
        if self.retriever:
            try:
                results = await self.retriever.search_similar_solutions(
                    query=query,
                    top_k=top_k,
                    filters=filters
                )
                self.safety_manager.reset_errors("ragbits")
                return results
            except Exception as e:
                self.safety_manager.record_error("ragbits", e)
                raise
        else:
            self.logger.info("ℹ️ No retriever configured, using fallback")
            return [generate_fallback_result(query, "search")]

    @safe_execute(fallback_value="")
    async def safe_ingest(
        self,
        content: str,
        metadata: dict,
        artifact_type: str = "solution"
    ) -> str:
        """
        Safely execute ingest operation.

        Args:
            content: Artifact content
            metadata: Artifact metadata
            artifact_type: Type of artifact

        Returns:
            Artifact ID (empty string on error)
        """
        # Validate inputs
        if not content or not isinstance(content, str):
            self.logger.warning("⚠️ Invalid content, returning empty ID")
            return ""

        if not metadata or not isinstance(metadata, dict):
            metadata = {}

        if not artifact_type or not isinstance(artifact_type, str):
            artifact_type = "general"

        # Check if service is available
        if not self.safety_manager.is_available("ragbits"):
            self.logger.info("ℹ️ RAGBits service unavailable, using fallback ID")
            return generate_fallback_artifact_id()

        # Execute ingest if retriever exists
        if self.retriever:
            try:
                artifact_id = await self.retriever.ingest_artifact(
                    content=content,
                    metadata=metadata,
                    artifact_type=artifact_type
                )
                self.safety_manager.reset_errors("ragbits")
                return artifact_id
            except Exception as e:
                self.safety_manager.record_error("ragbits", e)
                raise
        else:
            self.logger.info("ℹ️ No retriever configured, using fallback ID")
            return generate_fallback_artifact_id()


def create_safe_wrapper(retriever=None) -> SafeRAGBitsWrapper:
    """
    Create a safe wrapper for RAGBits operations.

    Args:
        retriever: RAGBits retriever instance (optional)

    Returns:
        SafeRAGBitsWrapper instance
    """
    return SafeRAGBitsWrapper(retriever)
