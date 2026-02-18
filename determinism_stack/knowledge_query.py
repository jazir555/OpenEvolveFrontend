"""
Unified Knowledge Query Python Bridge.

Bridges to the @openevolve/unified-knowledge-query TypeScript engine.
Following Federation Constitution:
- Law of Air Gap: HTTP API interaction.
- Law of UTC: ISO-8601 timestamps.
"""

import os
import uuid
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class UnifiedKnowledgeQuery:
    """Python bridge to the TypeScript Unified Knowledge Query Engine."""

    def __init__(self):
        self.api_url = os.environ.get("UNIFIED_KNOWLEDGE_QUERY_URL")
        self.timeout_ms = int(os.environ.get("TIMEOUT_MS", "5000"))
        
        if not self.api_url:
            logger.debug("UNIFIED_KNOWLEDGE_QUERY_URL not set. Unified query will be disabled.")

    def is_available(self) -> bool:
        """Check if the Unified Knowledge Query API is available."""
        return REQUESTS_AVAILABLE and self.api_url is not None

    def query(
        self,
        query_text: str,
        domains: Optional[List[str]] = None,
        max_results: int = 5,
        temporal_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a cross-system knowledge query.
        
        Args:
            query_text: The query string.
            domains: Optional list of domains to search.
            max_results: Maximum results to return.
            temporal_filter: Optional temporal constraints.
            
        Returns:
            Unified query result with fused knowledge.
        """
        if not self.is_available():
            return {"success": False, "error": "Unified Knowledge Query API not available"}

        correlation_id = str(uuid.uuid4())
        payload = {
            "query": query_text,
            "domains": domains or ["general"],
            "maxResults": max_results,
            "temporalFilter": temporal_filter,
            "metadata": {
                "correlation_id": correlation_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
        }

        try:
            # Note: This assumes the service is running at this endpoint
            # following the canonical glue layer pattern.
            response = requests.post(
                f"{self.api_url}/api/query",
                json=payload,
                timeout=self.timeout_ms / 1000
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.error(f"Unified Knowledge Query failed: {exc}")
            return {"success": False, "error": str(exc), "correlation_id": correlation_id}
