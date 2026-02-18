"""
ICR Python Adapter - Bridge to the 7-mode ICR system.

Following Federation Constitution:
- Law of Air Gap: Calls ICR API via HTTP
- Law of Runtime Truth: Verifies API connectivity
- Law of Configuration Explicitness: All config via environment variables
- Law of UTC: ISO-8601 timestamps
"""

import os
import time
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

class ICRAdapter:
    """Python bridge to the TypeScript ICR Adapter service."""

    def __init__(self):
        self.api_url = os.environ.get("OPENEVOLVE_ICR_API_URL")
        self.timeout_ms = int(os.environ.get("TIMEOUT_MS", "5000"))
        self.correlation_id_prefix = "python-icr-"
        
        if not self.api_url:
            logger.debug("OPENEVOLVE_ICR_API_URL not set. ICR integration will be disabled.")

    def is_available(self) -> bool:
        """Check if the ICR API is configured and reachable."""
        if not REQUESTS_AVAILABLE or not self.api_url:
            return False
        
        try:
            response = requests.get(f"{self.api_url}/api/health", timeout=self.timeout_ms / 1000)
            return response.status_code == 200
        except Exception:
            return False

    def _execute_mode(self, mode: str, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Internal helper to execute an ICR mode."""
        if not self.is_available():
            return {"success": False, "error": "ICR API not available or requests not installed"}

        correlation_id = f"{self.correlation_id_prefix}{uuid.uuid4()}"
        payload = {
            "mode": mode,
            "prompt": prompt,
            "options": options or {},
            "metadata": {
                "correlation_id": correlation_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "source_service": "determinism-stack-python"
            }
        }

        try:
            response = requests.post(
                f"{self.api_url}/api/modes/{mode}",
                json=payload,
                timeout=self.timeout_ms / 1000
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.error(f"ICR {mode} mode failed: {exc}")
            return {"success": False, "error": str(exc), "correlation_id": correlation_id}

    def refine(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """ICR Refine Mode: Traditional iterative refinements."""
        return self._execute_mode("refine", prompt, options)

    def react(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """ICR React Mode: React application development."""
        return self._execute_mode("react", prompt, options)

    def deepthink(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """ICR Deepthink Mode: Strategic problem decomposition."""
        return self._execute_mode("deepthink", prompt, options)

    def adaptive_deepthink(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """ICR Adaptive Deepthink Mode: Enhanced deepthink access."""
        return self._execute_mode("adaptive_deepthink", prompt, options)

    def agentic(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """ICR Agentic Mode: Tool-based manipulation."""
        return self._execute_mode("agentic", prompt, options)

    def contextual(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """ICR Contextual Mode: Collaborative agent refinement."""
        return self._execute_mode("contextual", prompt, options)

    def generative_ui(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """ICR Generative UI Mode: Interactive UI development."""
        return self._execute_mode("generative_ui", prompt, options)
