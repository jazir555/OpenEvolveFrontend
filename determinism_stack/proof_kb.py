"""
Proof Knowledge Base - Python Bridge.

Bridges to the @openevolve/proof-knowledge-base TypeScript library.
Following ADR-007: Centralized storage for formal proofs.
"""

import os
import uuid
import logging
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from .deps import optional_import

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ProofKnowledgeBase:
    """Python bridge to the Proof Knowledge Base service."""

    def __init__(self):
        self.api_url = os.environ.get("PROOF_KB_URL")
        self.timeout_ms = int(os.environ.get("TIMEOUT_MS", "5000"))
        
        if not self.api_url:
            logger.debug("PROOF_KB_URL not set. Proof storage will be disabled.")

    def is_available(self) -> bool:
        return REQUESTS_AVAILABLE and self.api_url is not None

    def store_proof(self, proof_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store a formal proof in the KB."""
        if not self.is_available():
            return {"success": False, "error": "Proof KB API not available"}

        correlation_id = str(uuid.uuid4())
        payload = {
            "proof": proof_data,
            "metadata": {
                "correlation_id": correlation_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
        }

        try:
            response = requests.post(
                f"{self.api_url}/api/proofs",
                json=payload,
                timeout=self.timeout_ms / 1000
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.error(f"Proof KB storage failed: {exc}")
            return {"success": False, "error": str(exc), "correlation_id": correlation_id}

    def search_similar(self, theorem_statement: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar proofs."""
        if not self.is_available():
            return []

        try:
            response = requests.get(
                f"{self.api_url}/api/proofs/search",
                params={"query": theorem_statement, "limit": limit},
                timeout=self.timeout_ms / 1000
            )
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as exc:
            logger.error(f"Proof KB search failed: {exc}")
            return []
