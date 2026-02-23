"""
ROMA Python Bridge - Canonical Adapter Interface

This module provides a Python wrapper around the ROMA canonical adapter,
ensuring Air Gap compliance while providing a clean Python API.

This replaces direct imports from core-projects/ROMA/ with HTTP API calls.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Configuration
ROMA_SERVER_URL = os.getenv('ROMA_SERVER_URL', 'http://localhost:8000')
ROMA_API_KEY = os.getenv('ROMA_API_KEY', '')
ROMA_TIMEOUT = int(os.getenv('ROMA_TIMEOUT', '30000'))

# Availability flag
ROMA_AVAILABLE = True


@dataclass
class RomaExecutionRequest:
    """Canonical ROMA execution request."""
    goal: str
    max_depth: Optional[int] = None
    config_profile: Optional[str] = None
    execution_method: Optional[str] = None
    timeout_ms: Optional[int] = None
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RomaExecutionResponse:
    """Canonical ROMA execution response."""
    execution_id: str
    status: str
    initial_goal: str
    result: Optional[Any]
    statistics: Dict[str, Any]
    timestamp: str
    error: Optional[str] = None


class RomaCanonicalBridge:
    """
    Canonical bridge to ROMA following Air Gap principles.

    This class provides a clean Python API that communicates with ROMA
    via HTTP API calls, avoiding direct imports from core-projects/ROMA/.

    Usage:
        bridge = RomaCanonicalBridge()
        result = await bridge.execute_task("Solve problem X", max_depth=3)
    """

    def __init__(self, server_url: str = None, api_key: str = None):
        self.server_url = server_url or ROMA_SERVER_URL
        self.api_key = api_key or ROMA_API_KEY
        self.timeout = ROMA_TIMEOUT // 1000  # Convert to seconds

        # Configure session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=3)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {
            'Content-Type': 'application/json',
        }
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def _get_url(self, endpoint: str) -> str:
        """Get full URL for API endpoint."""
        return f"{self.server_url}{endpoint}"

    async def execute_task(
        self,
        goal: str,
        max_depth: int = 3,
        config_profile: str = 'general',
        execution_method: str = 'auto',
        correlation_id: Optional[str] = None
    ) -> RomaExecutionResponse:
        """
        Execute a ROMA task via canonical HTTP API.

        Replaces: from roma_dspy.core.engine.solve import solve
        """
        request_body = {
            'goal': goal,
            'max_depth': max_depth,
            'config_profile': config_profile,
            'execution_method': execution_method,
            'timeout': self.timeout,
            'metadata': {
                'correlation_id': correlation_id or self._generate_correlation_id(),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }
        }

        try:
            response = self.session.post(
                self._get_url('/api/v1/executions'),
                json=request_body,
                headers=self._get_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()

            return RomaExecutionResponse(
                execution_id=data['execution_id'],
                status=data['status'],
                initial_goal=data['goal'],
                result=data.get('result'),
                statistics=data.get('statistics', {}),
                timestamp=data.get('timestamp', datetime.now(timezone.utc).isoformat()),
                error=data.get('error')
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"ROMA API request failed: {e}")
            raise

    async def get_execution(self, execution_id: str) -> RomaExecutionResponse:
        """Get execution details by ID."""
        try:
            response = self.session.get(
                self._get_url(f'/api/v1/executions/{execution_id}'),
                headers=self._get_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()

            return RomaExecutionResponse(
                execution_id=data['execution_id'],
                status=data['status'],
                initial_goal=data['goal'],
                result=data.get('result'),
                statistics=data.get('statistics', {}),
                timestamp=data.get('timestamp', datetime.now(timezone.utc).isoformat()),
                error=data.get('error')
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get ROMA execution {execution_id}: {e}")
            raise

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution."""
        try:
            response = self.session.post(
                self._get_url(f'/api/v1/executions/{execution_id}/cancel'),
                headers=self._get_headers(),
                timeout=10
            )
            response.raise_for_status()
            return response.json().get('status') == 'cancelled'
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to cancel ROMA execution {execution_id}: {e}")
            return False

    async def get_status(self) -> Dict[str, Any]:
        """Get ROMA server health status."""
        try:
            response = self.session.get(
                self._get_url('/health'),
                headers=self._get_headers(),
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get ROMA status: {e}")
            return {'status': 'unhealthy', 'error': str(e)}

    def _generate_correlation_id(self) -> str:
        """Generate a unique correlation ID."""
        import uuid
        return f"roma-{uuid.uuid4().hex}"


# Singleton instance
_bridge_instance: Optional[RomaCanonicalBridge] = None


def get_roma_bridge() -> RomaCanonicalBridge:
    """
    Get singleton instance of the ROMA canonical bridge.

    This provides a global access point similar to the old pattern:
        # OLD (direct import - violates Air Gap):
        from roma_dspy.core.engine.solve import RecursiveSolver
        solver = RecursiveSolver()

        # NEW (canonical bridge - Air Gap compliant):
        from glue.adapters.roma_bridge import get_roma_bridge
        bridge = get_roma_bridge()
    """
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = RomaCanonicalBridge()
    return _bridge_instance


async def solve_with_roma(goal: str, **kwargs) -> RomaExecutionResponse:
    """
    Convenience function for solving problems with ROMA.

    Replaces: from roma_dspy.core.engine.solve import solve
    """
    bridge = get_roma_bridge()
    return await bridge.execute_task(goal, **kwargs)


async def recursive_solve(goal: str, max_depth: int = 3, **kwargs) -> RomaExecutionResponse:
    """
    Recursive problem solving with ROMA.

    Replaces: from roma_dspy.core.engine.solve import RecursiveSolver
    """
    bridge = get_roma_bridge()
    return await bridge.execute_task(goal, max_depth=max_depth, **kwargs)


def reset_roma_bridge() -> None:
    """Reset the singleton bridge instance (mainly for testing)."""
    global _bridge_instance
    _bridge_instance = None


__all__ = [
    'RomaExecutionRequest',
    'RomaExecutionResponse',
    'RomaCanonicalBridge',
    'get_roma_bridge',
    'solve_with_roma',
    'recursive_solve',
    'reset_roma_bridge',
    'ROMA_AVAILABLE',
]
