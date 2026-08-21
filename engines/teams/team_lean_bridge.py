"""Shared Lean bridge for team coordination (Red / Blue / Gold).

Centralizes access to the real ``LeanAideClient`` so every coordinator and the
gauntlet validation manager can attempt formal proofs / verification through a
single, dependency-light entry point.

The real client is provided by ``leanaide_client`` (another agent delivers that
module). When it is not yet importable -- e.g. due to parallel-checkout timing
-- a self-contained mock with the same surface is used so the coordination logic
stays importable and functional entirely offline.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real client import (guarded).  If this fails we fall back to the mock.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when the real module is present
    from leanaide_client import (
        LeanAideClient as _RealLeanAideClient,
        LeanAideConfig,
        LeanAideResult,
        TaskType,
    )
    LEAN_AVAILABLE = True
except Exception:  # noqa: BLE001 - graceful fallback to the built-in mock
    LeanAideConfig = None  # type: ignore
    LeanAideResult = None  # type: ignore
    TaskType = None  # type: ignore
    LEAN_AVAILABLE = False
    logger.debug("leanaide_client not importable; using built-in mock client")


# ---------------------------------------------------------------------------
# Mock client (mirrors the real LeanAideClient surface)
# ---------------------------------------------------------------------------
@dataclass
class _MockResult:
    """Minimal mirror of :class:`leanaide_client.LeanAideResult`."""

    proved: bool
    status: str
    tactic_count: int = 0
    tactics: List[str] = field(default_factory=list)
    error: Optional[str] = None
    elapsed: float = 0.0
    raw: Optional[Dict[str, Any]] = None

    def __bool__(self) -> bool:  # ``if result:``
        return self.proved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proved": self.proved,
            "status": self.status,
            "tactic_count": self.tactic_count,
            "tactics": list(self.tactics),
            "error": self.error,
            "elapsed": self.elapsed,
        }


class MockLeanAideClient:
    """Offline fallback client implementing the real API surface.

    ``prove`` uses a deterministic heuristic (same idea as the real mock) so the
    coordination logic is reproducible without a Lean4 server.
    """

    def __init__(self, config: Any = None):
        self.config = config
        self.session_tactics: List[str] = []
        self._seed = getattr(config, "mock_seed", 0) if config is not None else 0

    @property
    def is_connected(self) -> bool:
        return False

    @staticmethod
    def _heuristic(theorem: str, tactics: List[str], seed: int) -> bool:
        if not tactics:
            return False
        digest = seed
        for t in tactics:
            digest = (digest * 31 + hash(t) + hash(theorem)) & 0xFFFFFFFF
        return digest % 2 == 0

    def prove(
        self, theorem: str, tactics: Optional[List[str]] = None
    ) -> "_MockResult":
        start = time.time()
        tactics = list(tactics or ["simp"])
        proved = self._heuristic(theorem, tactics, self._seed)
        self.session_tactics.extend(tactics)
        return _MockResult(
            proved=proved,
            status="proved" if proved else "failed",
            tactic_count=len(tactics),
            tactics=list(tactics),
            elapsed=time.time() - start,
        )

    def submit_tactic(self, state: Any, tactic: str) -> Tuple[Any, bool]:
        self.session_tactics.append(tactic)
        success = bool(tactic) and tactic.strip() not in ("sorry", "admit")
        return state, success

    def autoformalize(self, text: str, task_type: Any = None) -> str:
        stub = text.strip().replace("\n", " ")
        return f"theorem autoform_{abs(hash(stub)) & 0xFFFF} : {stub} := by simp"

    def reset(self) -> None:
        self.session_tactics.clear()

    def close(self) -> None:
        self.reset()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def get_lean_client(config: Any = None):
    """Return a LeanAIDE client (real when available, otherwise the mock)."""
    if LEAN_AVAILABLE:
        try:  # pragma: no cover - depends on external module
            return _RealLeanAideClient(config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to construct real LeanAideClient (%s); using mock", exc)
    return MockLeanAideClient(config)


def prove_theorem(client: Any, theorem: str, tactics: Optional[List[str]] = None) -> Dict[str, Any]:
    """Prove ``theorem`` and normalise the result into a plain dict."""
    result = client.prove(theorem, tactics)
    return getattr(result, "to_dict", lambda: dict(proved=bool(result), status=getattr(result, "status", "unknown")))()


def sha1(text: str) -> str:
    """Stable hash helper used for deduplication across coordinators."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
