"""
LeanAide client (stub) for ``integrations.bubblelabs``.

``stub: implement`` - the real client is an HTTP client for a running LeanAide
server. Verification cannot be faked, so every request method fails closed.
Consumers in this package guard on ``ImportError``/``LEAN_AVAILABLE``, so the
absence of a live server degrades gracefully.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:
    from ._stub_support import STUB, raise_stub
except ImportError:
    from _stub_support import STUB, raise_stub

logger = logging.getLogger(__name__)

__all__ = ["STUB", "LeanAideClient"]


class LeanAideClient:
    """
    Client for a LeanAide formal-verification server.

    Args:
        base_url: Base URL of the LeanAide server.
        timeout: Request timeout in seconds.

    Attributes:
        base_url: Base URL of the LeanAide server.
        timeout: Request timeout in seconds.
    """

    def __init__(self, base_url: str = "http://localhost:7654", timeout: float = 30.0) -> None:
        self.base_url = base_url
        self.timeout = timeout

    def verify(self, target: str, criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Verify a statement with the Lean theorem prover.

        Args:
            target: Statement or artefact to verify.
            criteria: Optional verification criteria.

        Returns:
            Mapping with at least a ``verified`` flag.

        Raises:
            StubNotImplementedError: Always - requires a running LeanAide server.
        """
        raise_stub(
            "LeanAideClient.verify",
            hint="POST the target to the LeanAide server and return its verification verdict",
        )

    def translate(self, statement: str) -> Dict[str, Any]:
        """
        Translate a natural-language statement into Lean.

        Args:
            statement: Natural-language mathematical statement.

        Returns:
            Mapping containing the generated Lean code.

        Raises:
            StubNotImplementedError: Always - requires a running LeanAide server.
        """
        raise_stub(
            "LeanAideClient.translate",
            hint="call the LeanAide translation endpoint",
        )

    def health(self) -> Dict[str, Any]:
        """
        Report server reachability.

        Returns:
            Mapping with a ``healthy`` flag and a diagnostic ``detail`` string.
            This is intentionally a safe default rather than an exception so
            callers can probe availability cheaply.
        """
        return {
            "healthy": False,
            "detail": "stub: implement - no LeanAide server configured",
            "base_url": self.base_url,
        }
