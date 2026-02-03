"""Resilient LLM client with retry logic and error classification."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

from ..llm import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

# Try to import OpenAI-specific errors
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None  # type: ignore[assignment]


@dataclass
class CallInfo:
    """Detailed information about an LLM API call.

    Attributes:
        role: The role making the call (e.g., "agent", "reflector", "skill_manager")
        call_id: Unique identifier for this call
        model: Model name used
        prompt: Input prompt text
        response: Output response text
        total_time: Total time taken for the call (including retries)
        prompt_tokens: Number of tokens in the prompt
        response_tokens: Number of tokens in the response
        error: Error message if the call failed
    """

    role: str
    call_id: str
    model: str
    prompt: str
    response: str
    total_time: float
    prompt_tokens: int = 0
    response_tokens: int = 0
    error: Optional[str] = None


class ResilientLLMClient(LLMClient):
    """
    Resilient LLM client wrapper with retry logic and error classification.

    Wraps any LLMClient with intelligent retry logic for transient failures.
    Implements exponential backoff with jitter to avoid thundering herd problems.

    Example:
        >>> base_client = LiteLLMClient(model="gpt-4")
        >>> resilient_client = ResilientLLMClient(base_client, max_retries=3)
        >>> response, call_info = resilient_client.complete(
        ...     "What is ACE?",
        ...     role="agent",
        ...     call_id="call_123"
        ... )
        >>> print(f"Got response in {call_info.total_time:.2f}s")
    """

    def __init__(
        self,
        base_client: LLMClient,
        max_retries: int = 3,
        base_sleep: float = 1.0,
        timeout: float = 60.0,
    ) -> None:
        """
        Initialize the resilient client.

        Args:
            base_client: The underlying LLM client to wrap
            max_retries: Maximum number of retry attempts (default: 3)
            base_sleep: Base sleep time in seconds for exponential backoff (default: 1.0)
            timeout: Maximum total time allowed for all attempts (default: 60.0)
        """
        super().__init__(model=base_client.model)
        self.base_client = base_client
        self.max_retries = max_retries
        self.base_sleep = base_sleep
        self.timeout = timeout

    def complete(
        self,
        prompt: str,
        role: str = "unknown",
        call_id: str = "unknown",
        **kwargs: Any,
    ) -> tuple[str, CallInfo]:
        """
        Generate completion with retry logic and detailed call tracking.

        Args:
            prompt: Input prompt text
            role: Role identifier for tracking (e.g., "agent", "reflector")
            call_id: Unique identifier for this call
            **kwargs: Additional parameters to pass to base client

        Returns:
            Tuple of (response_text, call_info)

        Raises:
            Exception: The last exception if all retries are exhausted
        """
        start_time = time.time()
        last_error: Optional[Exception] = None
        attempt = 0
        response_text = ""
        prompt_tokens = 0
        response_tokens = 0

        while attempt <= self.max_retries:
            attempt += 1
            attempt_start = time.time()

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                logger.error(
                    f"Call {call_id} exceeded timeout {self.timeout}s after {attempt} attempts"
                )
                break

            try:
                logger.debug(
                    f"Attempt {attempt}/{self.max_retries + 1} for call {call_id}"
                )

                # Call base client
                response: LLMResponse = self.base_client.complete(prompt, **kwargs)
                response_text = response.text

                # Extract token usage if available
                if response.raw and "usage" in response.raw:
                    usage = response.raw["usage"]
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    response_tokens = usage.get("completion_tokens", 0)

                # Check for empty response
                if not response_text or not response_text.strip():
                    logger.warning(
                        f"Empty response on attempt {attempt} for call {call_id}"
                    )
                    if attempt <= self.max_retries:
                        # Retry on empty response
                        backoff = self._calculate_backoff("unknown", attempt)
                        logger.info(f"Sleeping {backoff:.2f}s before retry")
                        time.sleep(backoff)
                        continue

                # Success!
                total_time = time.time() - start_time
                logger.info(
                    f"Call {call_id} succeeded on attempt {attempt} in {total_time:.2f}s"
                )

                call_info = CallInfo(
                    role=role,
                    call_id=call_id,
                    model=self.model or "unknown",
                    prompt=prompt,
                    response=response_text,
                    total_time=total_time,
                    prompt_tokens=prompt_tokens,
                    response_tokens=response_tokens,
                    error=None,
                )

                return response_text, call_info

            except Exception as e:
                last_error = e
                error_type = self._classify_error(e)

                logger.warning(
                    f"Attempt {attempt} failed for call {call_id}: {error_type} - {e}"
                )

                # Check if we should retry
                if attempt > self.max_retries:
                    logger.error(
                        f"Max retries ({self.max_retries}) exceeded for call {call_id}"
                    )
                    break

                # Don't retry on auth errors
                if error_type == "auth_error":
                    logger.error(f"Auth error for call {call_id}, not retrying")
                    break

                # Calculate backoff and sleep
                backoff = self._calculate_backoff(error_type, attempt)
                logger.info(f"Sleeping {backoff:.2f}s before retry")
                time.sleep(backoff)

        # All retries exhausted
        total_time = time.time() - start_time
        error_message = str(last_error) if last_error else "Unknown error"

        logger.error(
            f"Call {call_id} failed after {attempt} attempts in {total_time:.2f}s: {error_message}"
        )

        call_info = CallInfo(
            role=role,
            call_id=call_id,
            model=self.model or "unknown",
            prompt=prompt,
            response=response_text,
            total_time=total_time,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            error=error_message,
        )

        # Raise the last error
        if last_error:
            raise last_error

        return response_text, call_info

    def _classify_error(self, error: Exception) -> str:
        """
        Classify an error to determine retry strategy.

        Args:
            error: The exception to classify

        Returns:
            Error type: "timeout", "rate_limit", "server_error", "auth_error", or "unknown"
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Check for rate limit errors
        if (
            "429" in error_str
            or "rate" in error_str
            or "limit" in error_str
            or "quota" in error_str
        ):
            return "rate_limit"

        # Check for OpenAI-specific rate limit error
        if OPENAI_AVAILABLE and openai:
            if isinstance(error, openai.RateLimitError):
                return "rate_limit"

        # Check for timeout errors
        if (
            "timeout" in error_str
            or "timed out" in error_str
            or "timedout" in error_str
            or error_type == "timeout"
        ):
            return "timeout"

        # Check for connection errors
        if (
            "connection" in error_str
            or "network" in error_str
            or "connect" in error_str
        ):
            return "timeout"

        # Check for server errors (5xx)
        if "500" in error_str or "502" in error_str or "503" in error_str:
            return "server_error"

        # Check for OpenAI-specific server error
        if OPENAI_AVAILABLE and openai:
            if isinstance(error, openai.InternalServerError):
                return "server_error"

        # Check for authentication errors (don't retry)
        if (
            "401" in error_str
            or "403" in error_str
            or "auth" in error_str
            or "unauthorized" in error_str
            or "permission" in error_str
            or "api key" in error_str
            or "invalid" in error_str
        ):
            return "auth_error"

        # Check for OpenAI-specific auth error
        if OPENAI_AVAILABLE and openai:
            if isinstance(error, openai.AuthenticationError):
                return "auth_error"

        # Unknown error - will retry with conservative backoff
        return "unknown"

    def _calculate_backoff(self, error_type: str, attempt: int) -> float:
        """
        Calculate exponential backoff with jitter.

        Uses different multipliers based on error type:
        - rate_limit: 2.0 (longer backoff for rate limits)
        - server_error: 1.5 (moderate backoff for server issues)
        - timeout: 1.0 (shorter backoff for timeouts)
        - unknown: 1.0 (conservative default)

        Jitter: Random factor between 0.5x and 1.5x to avoid thundering herd.

        Args:
            error_type: The classified error type
            attempt: The current attempt number (1-indexed)

        Returns:
            Sleep time in seconds
        """
        # Choose multiplier based on error type
        multipliers = {
            "rate_limit": 2.0,
            "server_error": 1.5,
            "timeout": 1.0,
            "unknown": 1.0,
            "auth_error": 0.0,  # Don't retry auth errors
        }

        multiplier = multipliers.get(error_type, 1.0)

        # Calculate base backoff with exponential growth
        base_backoff = self.base_sleep * (multiplier**attempt)

        # Add jitter: 0.5x to 1.5x of base backoff
        jitter = random.uniform(0.5, 1.5)
        backoff = base_backoff * jitter

        logger.debug(
            f"Backoff calculation: error_type={error_type}, attempt={attempt}, "
            f"base={base_backoff:.2f}s, jitter={jitter:.2f}, final={backoff:.2f}s"
        )

        return backoff
