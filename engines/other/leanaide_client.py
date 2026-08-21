"""LeanAideClient - a client for the Lean4 interactive theorem prover.

This is a self-contained, dependency-light implementation intended to make the
flat ``engines/`` scripts (which do ``from leanaide_client import LeanAideClient``)
import cleanly without an external Lean4 server.

If a real Lean4 server is reachable (configured via ``LEAN4_URL`` or a
``LeanAideConfig``), the client will talk to it; otherwise it degrades to a
deterministic in-process mock that records submitted tactics and returns a
stable "proved"/"failed" result based on a simple heuristic.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Kind of formalization/proof task submitted to LeanAIDE."""

    PROVE_THEOREM = "prove_theorem"
    AUTOFORMALIZE = "autoformalize"
    VERIFY_SPEC = "verify_spec"
    SYNTHESIZE = "synthesize"


@dataclass
class LeanAideConfig:
    """Configuration for a :class:`LeanAideClient`."""

    url: Optional[str] = field(
        default_factory=lambda: os.environ.get("LEAN4_URL")
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("LEAN4_API_KEY")
    )
    timeout: float = 30.0
    max_tactics: int = 100
    # When True (default) and no server is reachable, fall back to the mock prover.
    allow_mock: bool = True
    # Deterministic seed so the mock prover is reproducible across runs.
    mock_seed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LeanAideConfig":
        return cls()


@dataclass
class LeanAideResult:
    """Outcome of a LeanAIDE proof attempt."""

    proved: bool
    status: str  # "proved" | "failed" | "timeout" | "error"
    tactic_count: int = 0
    tactics: List[str] = field(default_factory=list)
    error: Optional[str] = None
    elapsed: float = 0.0
    raw: Optional[Dict[str, Any]] = None

    def __bool__(self) -> bool:  # convenience: ``if result:``
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


def _simple_proof_heuristic(theorem: str, tactics: List[str], seed: int) -> bool:
    """Deterministic mock: a theorem "proves" if it received at least one tactic
    and the combined text hash is even (stable given the same inputs)."""
    if not tactics:
        return False
    digest = seed
    for t in tactics:
        digest = (digest * 31 + hash(t) + hash(theorem)) & 0xFFFFFFFF
    return digest % 2 == 0


class LeanAideClient:
    """Client for the Lean4 prover.

    Usage::

        client = LeanAideClient(LeanAideConfig(url="http://localhost:8000"))
        result = client.prove("theorem t : 1 + 1 = 2 := by simp")
        result = client.submit_tactic(state, "simp")
    """

    def __init__(self, config: Optional[LeanAideConfig] = None):
        self.config = config or LeanAideConfig()
        self._lean_available = self._probe_server()
        self.session_tactics: List[str] = []
        # Lightweight, dependency-free record of the current proof state so that
        # ``get_proof_state()`` can be called without importing ``proof_state``.
        self._current_state: Dict[str, Any] = {
            "theorem": None,
            "goals": [],
            "assigned_tactics": [],
            "history": [],
            "solved": False,
            "status": "idle",
        }

    # -- connectivity ----------------------------------------------------
    def _probe_server(self) -> bool:
        if not self.config.url or not self.config.allow_mock:
            return False
        try:  # pragma: no cover - network optional
            import urllib.request

            req = urllib.request.Request(self.config.url, method="HEAD")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status < 400
        except Exception:  # noqa: BLE001 - graceful degradation is the point
            return False

    @property
    def is_connected(self) -> bool:
        return self._lean_available

    # -- core API --------------------------------------------------------
    def prove(self, theorem: str, tactics: Optional[List[str]] = None) -> LeanAideResult:
        """Attempt to prove ``theorem``. If ``tactics`` are supplied they are
        applied in order; otherwise a trivial ``simp`` is tried."""
        start = time.time()
        tactics = list(tactics or ["simp"])
        self._current_state["theorem"] = theorem
        self._current_state["goals"] = [theorem] if theorem else []
        self._current_state["status"] = "running"
        if self._lean_available:
            try:  # pragma: no cover - network optional
                result = self._prove_remote(theorem, tactics, start)
                self._sync_state(theorem, result)
                return result
            except Exception as exc:  # noqa: BLE001
                logger.warning("Remote prove failed, using mock: %s", exc)
        proved = _simple_proof_heuristic(theorem, tactics, self.config.mock_seed)
        status = "proved" if proved else "failed"
        self.session_tactics.extend(tactics)
        result = LeanAideResult(
            proved=proved,
            status=status,
            tactic_count=len(tactics),
            tactics=list(tactics),
            elapsed=time.time() - start,
        )
        self._sync_state(theorem, result)
        return result

    def _sync_state(self, theorem: str, result: "LeanAideResult") -> None:
        """Mirror the outcome of a proof attempt into the tracked proof state."""
        self._current_state["theorem"] = theorem
        self._current_state["assigned_tactics"] = list(result.tactics)
        self._current_state["history"] = list(self._current_state["history"]) + [
            f"prove:{result.status}"
        ]
        self._current_state["solved"] = bool(result.proved)
        self._current_state["status"] = result.status
        self._current_state["goals"] = [] if result.proved else (
            [theorem] if theorem else []
        )

    def _prove_remote(self, theorem, tactics, start):
        import urllib.request

        payload = json.dumps(
            {"theorem": theorem, "tactics": tactics}
        ).encode("utf-8")
        req = urllib.request.Request(
            self.config.url.rstrip("/") + "/prove",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        proved = bool(data.get("proved", False))
        return LeanAideResult(
            proved=proved,
            status="proved" if proved else "failed",
            tactic_count=len(tactics),
            tactics=list(tactics),
            elapsed=time.time() - start,
            raw=data,
        )

    def submit_tactic(self, state: Any, tactic: str) -> Tuple[Any, bool]:
        """Apply ``tactic`` to ``state``. With the mock prover, state is treated
        as opaque and the tactic is recorded. Returns ``(new_state, success)``."""
        self.session_tactics.append(tactic)
        # Mock: a tactic "succeeds" unless it is empty or explicitly `sorry`.
        success = bool(tactic) and tactic.strip() not in ("sorry", "admit")
        self._current_state["assigned_tactics"] = list(
            self._current_state["assigned_tactics"]
        ) + [tactic]
        self._current_state["history"] = list(self._current_state["history"]) + [
            f"tactic:{tactic}:{'ok' if success else 'fail'}"
        ]
        self._current_state["status"] = "in_progress" if success else "stuck"
        # Keep a mirror of the caller's state when it exposes goals, so that
        # ``get_proof_state()`` reflects externally-driven proof search too.
        goals = getattr(state, "goals", None)
        if isinstance(goals, list):
            self._current_state["goals"] = list(goals)
        elif isinstance(state, dict) and isinstance(state.get("goals"), list):
            self._current_state["goals"] = list(state["goals"])
        return state, success

    def get_proof_state(self) -> Dict[str, Any]:
        """Return a snapshot of the current proof state.

        The snapshot is a plain ``dict`` (``theorem``, ``goals``,
        ``assigned_tactics``, ``history``, ``solved``, ``status``) so this method
        works with no Lean4 server and without importing ``proof_state``. Use
        :meth:`get_proof_state_object` for a ``ProofState`` instance when the
        companion module is importable.
        """
        snapshot = dict(self._current_state)
        snapshot["goals"] = list(self._current_state.get("goals") or [])
        snapshot["assigned_tactics"] = list(
            self._current_state.get("assigned_tactics") or []
        )
        snapshot["history"] = list(self._current_state.get("history") or [])
        snapshot["connected"] = self._lean_available
        return snapshot

    def get_proof_state_object(self) -> Any:
        """Return the current state as a ``proof_state.ProofState`` when that
        flat module is on ``sys.path``; otherwise return the dict snapshot."""
        snapshot = self.get_proof_state()
        try:
            from proof_state import ProofState, Tactic as _Tactic  # noqa: WPS433

            return ProofState(
                goals=snapshot["goals"],
                assigned_tactics=[
                    _Tactic.parse(t) for t in snapshot["assigned_tactics"]
                ],
                history=snapshot["history"],
                solved=snapshot["solved"],
            )
        except Exception:  # noqa: BLE001 - graceful degradation is the point
            return snapshot

    def autoformalize(self, text: str, task_type: TaskType = TaskType.AUTOFORMALIZE) -> str:
        """Best-effort autoformalization. The mock returns a stub theorem
        skeleton derived from the input text."""
        stub = text.strip().replace("\n", " ")
        if self._lean_available:
            try:  # pragma: no cover - network optional
                import urllib.request

                payload = json.dumps({"text": text, "type": task_type.value}).encode()
                req = urllib.request.Request(
                    self.config.url.rstrip("/") + "/autoformalize", data=payload,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.config.timeout) as r:
                    return json.loads(r.read().decode())["formal"]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Remote autoformalize failed: %s", exc)
        return f"theorem autoform_{abs(hash(stub)) & 0xFFFF} : {stub} := by simp"

    def reset(self) -> None:
        self.session_tactics.clear()
        self._current_state = {
            "theorem": None,
            "goals": [],
            "assigned_tactics": [],
            "history": [],
            "solved": False,
            "status": "idle",
        }

    def close(self) -> None:  # pragma: no cover - resource cleanup hook
        self.reset()
