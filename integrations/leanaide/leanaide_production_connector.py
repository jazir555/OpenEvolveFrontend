"""LeanAide Production Connector.

Connector that talks to a production Lean 4 verification service (configured
via ``LEAN4_URL`` / ``LEAN4_API_KEY``). It performs a genuine HTTP request
when the service is reachable and otherwise degrades gracefully to the
structural checker. It never claims success without a real check.
"""

import logging
import os
import urllib.request
import urllib.error
import json
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from leanaide_systems import check_lean_proof_structural
except ImportError:  # pragma: no cover
    from integrations.leanaide.leanaide_systems import check_lean_proof_structural  # type: ignore


class LeanAideProductionConnector:
    """Talk to a deployed Lean 4 verification endpoint."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.url = url or os.environ.get("LEAN4_URL")
        self.api_key = api_key or os.environ.get("LEAN4_API_KEY")
        self.timeout = timeout

    def status(self) -> Dict[str, Any]:
        return {
            "configured": bool(self.url),
            "url": self.url,
            "has_api_key": bool(self.api_key),
        }

    def verify(self, code: str) -> Dict[str, Any]:
        structural = check_lean_proof_structural(code)
        if not self.url:
            structural["method"] = "structural"
            structural["warnings"] = list(structural.get("warnings", [])) + [
                "Production Lean4 endpoint not configured; structural check only"
            ]
            return structural

        payload = json.dumps({"code": code}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.url, data=payload, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return {
                "valid": bool(body.get("success", body.get("valid", False))),
                "method": "production_lean4",
                "errors": body.get("errors", []),
                "warnings": body.get("warnings", []),
                "remote": True,
            }
        except (urllib.error.URLError, OSError, ValueError) as exc:
            structural["method"] = "structural"
            structural["warnings"] = list(structural.get("warnings", [])) + [
                f"Production endpoint unreachable ({exc}); structural check only"
            ]
            return structural
