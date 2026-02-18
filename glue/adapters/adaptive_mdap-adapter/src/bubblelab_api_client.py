"""
BubbleLab API Client for Adaptive MDAP/MAKER

Federation Constitution Compliant API client for communicating with
the BubbleLab OpenEvolve API server.

Law 5: Configuration Explicitness - All URLs and timeouts must be configured
Law 6: UTC - All timestamps in UTC ISO-8601
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class BubbleLabAPIClientConfig:
    """Configuration for BubbleLab API client.

    Environment Variables:
        BUBBLELAB_API_URL: Base URL of BubbleLab API (REQUIRED)
        BUBBLELAB_API_KEY: API key for authentication (optional)
        BUBBLELAB_TIMEOUT_MS: Request timeout in milliseconds (default: 30000)
        BUBBLELAB_MAX_RETRIES: Maximum retry attempts (default: 3)
    """
    api_url: str
    api_key: Optional[str] = None
    timeout_ms: int = 30000
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "BubbleLabAPIClientConfig":
        """Load configuration from environment variables."""
        api_url = os.getenv("BUBBLELAB_API_URL")
        if api_url is None:
            raise ValueError(
                "BUBBLELAB_API_URL is required. "
                "Set the BubbleLab API base URL."
            )

        return cls(
            api_url=api_url.rstrip('/'),
            api_key=os.getenv("BUBBLELAB_API_KEY"),
            timeout_ms=int(os.getenv("BUBBLELAB_TIMEOUT_MS", "30000")),
            max_retries=int(os.getenv("BUBBLELAB_MAX_RETRIES", "3"))
        )


class BubbleLabAPIClientError(Exception):
    """Base exception for API client errors."""
    pass


class BubbleLabAPIConnectionError(BubbleLabAPIClientError):
    """Connection error."""
    pass


class BubbleLabAPIResponseError(BubbleLabAPIClientError):
    """API response error."""

    def __init__(self, status_code: int, message: str, response_body: Optional[str] = None):
        self.status_code = status_code
        self.message = message
        self.response_body = response_body
        super().__init__(f"API Error {status_code}: {message}")


class BubbleLabAPIClient:
    """
    API Client for BubbleLab OpenEvolve API.

    Provides methods for interacting with the MDAP/MAKER endpoints
    exposed by the BubbleLab OpenEvolve API server.
    """

    def __init__(self, config: Optional[BubbleLabAPIClientConfig] = None):
        """Initialize the API client."""
        self.config = config or BubbleLabAPIClientConfig.from_env()
        self.logger = logging.getLogger("BubbleLabAPIClient")

        if not REQUESTS_AVAILABLE:
            self.logger.warning("requests library not available, client will be limited")

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        # Add correlation tracking
        headers["X-Request-Start"] = datetime.now(timezone.utc).isoformat()

        return headers

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to BubbleLab API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters

        Returns:
            Response data as dictionary

        Raises:
            BubbleLabAPIConnectionError: Connection error
            BubbleLabAPIResponseError: API error response
        """
        if not REQUESTS_AVAILABLE:
            raise BubbleLabAPIConnectionError("requests library not available")

        url = f"{self.config.api_url}{endpoint}"
        timeout_seconds = self.config.timeout_ms / 1000.0

        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(
                    f"API Request: {method} {url}",
                    attempt=attempt + 1,
                    max_retries=self.config.max_retries
                )

                response = requests.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    headers=self._get_headers(),
                    timeout=timeout_seconds
                )

                # Handle error responses
                if response.status_code >= 400:
                    raise BubbleLabAPIResponseError(
                        status_code=response.status_code,
                        message=response.reason or "Unknown error",
                        response_body=response.text
                    )

                return response.json()

            except requests.exceptions.Timeout as e:
                if attempt < self.config.max_retries - 1:
                    self.logger.warning(f"Request timeout, retrying: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise BubbleLabAPIConnectionError(f"Request timeout: {e}")

            except requests.exceptions.ConnectionError as e:
                if attempt < self.config.max_retries - 1:
                    self.logger.warning(f"Connection error, retrying: {e}")
                    time.sleep(2 ** attempt)
                    continue
                raise BubbleLabAPIConnectionError(f"Connection error: {e}")

            except BubbleLabAPIResponseError:
                # Don't retry API errors
                raise

            except Exception as e:
                raise BubbleLabAPIConnectionError(f"Unexpected error: {e}")

    # ========================================================================
    # MDAP/MAKER Endpoints
    # ========================================================================

    def get_mdap_maker_status(self) -> Dict[str, Any]:
        """
        Get MDAP/MAKER system status.

        Returns:
            Status dictionary with availability flags:
            - mdap_available: bool
            - maker_available: bool
            - associative_available: bool
            - ground_truth_available: bool
            - full_system_available: bool
        """
        self.logger.info("Fetching MDAP/MAKER status")
        return self._make_request("GET", "/api/openevolve/mdap-maker/status")

    def solve_with_mdap_maker(
        self,
        problem_statement: str,
        sub_solutions: Optional[Dict[str, Any]] = None,
        conflicts: Optional[List[Any]] = None,
        use_mdap: Optional[bool] = None,
        use_associative: Optional[bool] = None,
        num_mdap_agents: Optional[int] = None,
        llm_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Solve a problem using MDAP/MAKER workflow.

        Args:
            problem_statement: The problem to solve
            sub_solutions: Optional dictionary of sub-solutions
            conflicts: Optional list of conflicts to resolve
            use_mdap: Whether to use MDAP (default from server config)
            use_associative: Whether to use associative memory (default from server config)
            num_mdap_agents: Number of MDAP agents (default from server config)
            llm_config: Optional LLM configuration overrides

        Returns:
            Result dictionary with 'success' and 'result' keys
        """
        self.logger.info(
            "Solving with MDAP/MAKER",
            problem_statement_length=len(problem_statement)
        )

        request_data = {
            "problem_statement": problem_statement,
            "sub_solutions": sub_solutions or {},
            "conflicts": conflicts or [],
        }

        # Add optional parameters
        if use_mdap is not None:
            request_data["use_mdap"] = use_mdap
        if use_associative is not None:
            request_data["use_associative"] = use_associative
        if num_mdap_agents is not None:
            request_data["num_mdap_agents"] = num_mdap_agents
        if llm_config is not None:
            request_data["llm_config"] = llm_config

        return self._make_request("POST", "/api/openevolve/mdap-maker/solve", request_data)

    def get_roma_mdap_maker_status(self) -> Dict[str, Any]:
        """
        Get ROMA-MDAP-MAKER system status.

        Returns:
            Status dictionary with component availability
        """
        self.logger.info("Fetching ROMA-MDAP-MAKER status")
        return self._make_request("GET", "/api/openevolve/roma-mdap-maker/status")

    def solve_with_roma_mdap_maker(
        self,
        problem_statement: str,
        context: Optional[Dict[str, Any]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        recursive: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Solve a problem using ROMA-MDAP-MAKER workflow.

        Args:
            problem_statement: The problem to solve
            context: Optional context dictionary
            config_overrides: Optional configuration overrides
            recursive: Whether to use recursive solving

        Returns:
            Result dictionary with 'success' and 'result' keys
        """
        self.logger.info(
            "Solving with ROMA-MDAP-MAKER",
            problem_statement_length=len(problem_statement)
        )

        request_data = {
            "problem_statement": problem_statement,
        }

        # Add optional parameters
        if context is not None:
            request_data["context"] = context
        if config_overrides is not None:
            request_data["config_overrides"] = config_overrides
        if recursive is not None:
            request_data["recursive"] = recursive

        return self._make_request("POST", "/api/openevolve/roma-mdap-maker/solve", request_data)

    # ========================================================================
    # Health Check
    # ========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on BubbleLab API.

        Returns:
            Health status dictionary
        """
        try:
            # Try to fetch status as health check
            status = self.get_mdap_maker_status()
            return {
                "status": "healthy",
                "api_url": self.config.api_url,
                "mdap_maker_status": status,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "api_url": self.config.api_url,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


# ============================================================================
# Convenience Functions
# ============================================================================

_default_client: Optional[BubbleLabAPIClient] = None


def get_bubblelab_client(config: Optional[BubbleLabAPIClientConfig] = None) -> BubbleLabAPIClient:
    """Get or create the singleton API client instance."""
    global _default_client
    if _default_client is None:
        _default_client = BubbleLabAPIClient(config)
    return _default_client
