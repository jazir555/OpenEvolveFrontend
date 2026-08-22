"""LeanAide BubbleLab Integration adapter.

Adapter that bridges verified LeanAIDE proofs into the BubbleLab workspace.
It degrades gracefully: when the BubbleLab SDK is unavailable, operations are
recorded locally and reported as ``degraded`` rather than failing hard.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from leanaide_systems import check_lean_proof_structural
except ImportError:  # pragma: no cover
    from integrations.leanaide.leanaide_systems import check_lean_proof_structural  # type: ignore

try:
    import bubblelabs_sdk  # type: ignore

    BUBBLELAB_AVAILABLE = True
except Exception:  # pragma: no cover
    BUBBLELAB_AVAILABLE = False
    bubblelabs_sdk = None


class LeanAideBubbleLabIntegration:
    """Bridge LeanAIDE verification results into BubbleLab."""

    def __init__(self, workspace_id: Optional[str] = None):
        self.workspace_id = workspace_id
        self._client = None
        self._published: list = []
        if BUBBLELAB_AVAILABLE:
            try:
                self._client = bubblelabs_sdk.Client() if hasattr(bubblelabs_sdk, "Client") else None
            except Exception as exc:  # pragma: no cover
                logger.warning("BubbleLab client init failed: %s", exc)
                self._client = None

    def status(self) -> Dict[str, Any]:
        return {
            "bubblelab_available": BUBBLELAB_AVAILABLE,
            "client_connected": self._client is not None,
            "workspace_id": self.workspace_id,
            "published_count": len(self._published),
        }

    def publish_proof(self, name: str, code: str) -> Dict[str, Any]:
        """Verify a proof structurally and publish it to BubbleLab if possible."""
        check = check_lean_proof_structural(code)
        record = {
            "name": name,
            "verified": check["valid"],
            "errors": check["errors"],
            "warnings": check["warnings"],
        }
        if self._client is not None:
            try:
                self._client.publish(
                    workspace=self.workspace_id, title=name, lean_code=code,
                    verified=check["valid"],
                )
                record["published"] = True
            except Exception as exc:  # pragma: no cover
                record["published"] = False
                record["publish_error"] = str(exc)
        else:
            record["published"] = False
            record["note"] = "BubbleLab SDK unavailable; recorded locally only"
        self._published.append(record)
        return record
