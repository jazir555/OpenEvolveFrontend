"""
Resource management and optimization for the Sovereign-Grade Decomposition
Workflow (doc §6.2 "resource management and optimization").

This module is PURE and OFFLINE: it has no external dependencies and performs no
network / API calls. It tracks token / time / step budgets across a workflow run
and helps the orchestrator decide how many independent sub-problems to run in
parallel given the remaining budget.

It is deliberately dependency-light (stdlib only) so it can be imported inside
worker subprocesses and unit-tested without spinning up the full OpenEvolve
import graph.

§6.2 (doc "resource management and optimization") ties into the
``DecompositionPlan`` knobs ``resource_limits``, ``max_parallel_sub_problems``
and ``parallel_processing_enabled``; this manager is the runtime counterpart
that enforces those limits and recommends batch sizing for the executor in
``workflow_distributed.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


class ResourceExhaustedError(RuntimeError):
    """Raised when a requested consumption would exceed the configured budget."""


@dataclass
class ResourceBudget:
    """Declarative budget for a single workflow run.

    A dimension is *unbounded* when its ``total_*`` value is ``None``. The
    ``*_per_sub_problem`` fields are estimates used by :meth:`ResourceManager.
    recommend_batch_size` to translate the remaining budget into a safe number
    of parallel sub-problems.
    """

    total_tokens: Optional[int] = None
    total_time_seconds: Optional[float] = None
    total_steps: Optional[int] = None

    max_parallel: int = 1

    tokens_per_sub_problem: int = 0
    time_per_sub_problem: float = 0.0
    steps_per_sub_problem: int = 0

    # When True, ``consume`` is allowed to overshoot silently and only
    # ``consume_or_raise`` blocks. When False, even ``consume`` raises. Kept
    # False by default so enforcement is strict and testable.
    allow_overshoot: bool = False

    def __post_init__(self) -> None:
        if self.max_parallel is not None and self.max_parallel < 1:
            self.max_parallel = 1


@dataclass
class ResourceUsage:
    """Accumulated consumption across all dimensions."""

    tokens: int = 0
    time_seconds: float = 0.0
    steps: int = 0

    def __add__(self, other: "ResourceUsage") -> "ResourceUsage":
        return ResourceUsage(
            tokens=self.tokens + other.tokens,
            time_seconds=self.time_seconds + other.time_seconds,
            steps=self.steps + other.steps,
        )


class ResourceManager:
    """Tracks and enforces a :class:`ResourceBudget` over a workflow run.

    The manager is intentionally tiny and side-effect free: it only accumulates
    numbers. Time is measured via a monotonic clock and can also be supplied
    explicitly (``record_actual``) so callers that already measured a real
    sub-problem duration can report it instead of wall-clock deltas.
    """

    def __init__(self, budget: ResourceBudget, clock: Optional[callable] = None):
        self.budget = budget
        self.used = ResourceUsage()
        # ``clock`` is injectable for deterministic tests.
        self._clock = clock or time.monotonic

    # ------------------------------------------------------------------
    # Remaining budget queries
    # ------------------------------------------------------------------
    def remaining_tokens(self) -> Optional[int]:
        if self.budget.total_tokens is None:
            return None
        return max(0, self.budget.total_tokens - self.used.tokens)

    def remaining_time(self) -> Optional[float]:
        if self.budget.total_time_seconds is None:
            return None
        return max(0.0, self.budget.total_time_seconds - self.used.time_seconds)

    def remaining_steps(self) -> Optional[int]:
        if self.budget.total_steps is None:
            return None
        return max(0, self.budget.total_steps - self.used.steps)

    # ------------------------------------------------------------------
    # Affordability / enforcement
    # ------------------------------------------------------------------
    def can_afford(self, tokens: int = 0, time_seconds: float = 0.0, steps: int = 0) -> bool:
        """True when the requested consumption fits within the remaining budget."""
        rt = self.remaining_tokens()
        if rt is not None and tokens > rt:
            return False
        rtime = self.remaining_time()
        if rtime is not None and time_seconds > rtime:
            return False
        rs = self.remaining_steps()
        if rs is not None and steps > rs:
            return False
        return True

    def consume(self, tokens: int = 0, time_seconds: float = 0.0, steps: int = 0) -> None:
        """Accumulate consumption without raising (unless overshoot is forbidden)."""
        if not self.budget.allow_overshoot and not self.can_afford(
            tokens, time_seconds, steps
        ):
            raise ResourceExhaustedError(
                f"Resource budget exceeded: requested tokens={tokens}, "
                f"time={time_seconds}, steps={steps} but remaining "
                f"tokens={self.remaining_tokens()}, time={self.remaining_time()}, "
                f"steps={self.remaining_steps()}"
            )
        self.used.tokens += tokens
        self.used.time_seconds += time_seconds
        self.used.steps += steps

    def consume_or_raise(
        self, tokens: int = 0, time_seconds: float = 0.0, steps: int = 0
    ) -> None:
        """Public enforcement entry point; always blocks on exhaustion."""
        if not self.can_afford(tokens, time_seconds, steps):
            raise ResourceExhaustedError(
                f"Resource budget exceeded: requested tokens={tokens}, "
                f"time={time_seconds}, steps={steps} but remaining "
                f"tokens={self.remaining_tokens()}, time={self.remaining_time()}, "
                f"steps={self.remaining_steps()}"
            )
        self.used.tokens += tokens
        self.used.time_seconds += time_seconds
        self.used.steps += steps

    def record_actual(
        self,
        tokens: int = 0,
        time_seconds: float = 0.0,
        steps: int = 0,
        raise_on_exhaust: bool = False,
    ) -> None:
        """Record a *measured* consumption (e.g. a real sub-problem duration).

        Unlike ``consume`` this never blocks by default (it reflects reality
        that already happened); set ``raise_on_exhaust=True`` to also enforce.
        """
        if raise_on_exhaust and not self.can_afford(tokens, time_seconds, steps):
            raise ResourceExhaustedError("Recorded consumption exceeded budget")
        self.used.tokens += tokens
        self.used.time_seconds += time_seconds
        self.used.steps += steps

    def is_exhausted(self) -> bool:
        return not self.can_afford(1, 1e-9, 1)

    # ------------------------------------------------------------------
    # Optimization helper: how many to run in parallel?
    # ------------------------------------------------------------------
    def recommend_batch_size(
        self, pending_count: int, max_parallel: Optional[int] = None
    ) -> int:
        """Suggest how many *independent* sub-problems to run concurrently.

        The recommendation is the tightest of:
          * the parallelism cap (``max_parallel`` or ``budget.max_parallel``),
          * the number of still-pending sub-problems,
          * the budget headroom expressed in whole sub-problems using the
            ``*_per_sub_problem`` estimates.

        Returns a non-negative integer; ``0`` means the budget is fully spent.
        """
        if pending_count <= 0:
            return 0
        cap = max_parallel if max_parallel is not None else self.budget.max_parallel
        if cap is None or cap < 1:
            cap = 1
        batch = min(int(cap), max(0, int(pending_count)))

        b = self.budget
        if b.total_tokens is not None and b.tokens_per_sub_problem > 0:
            headroom = self.remaining_tokens() // b.tokens_per_sub_problem
            batch = min(batch, max(0, int(headroom)))
        if b.total_time_seconds is not None and b.time_per_sub_problem > 0:
            headroom = int(self.remaining_time() // b.time_per_sub_problem)
            batch = min(batch, max(0, headroom))
        if b.total_steps is not None and b.steps_per_sub_problem > 0:
            headroom = self.remaining_steps() // b.steps_per_sub_problem
            batch = min(batch, max(0, int(headroom)))
        return batch

    def utilization(self) -> dict:
        """Return a small snapshot useful for monitoring / reporting."""
        return {
            "used_tokens": self.used.tokens,
            "used_time": self.used.time_seconds,
            "used_steps": self.used.steps,
            "remaining_tokens": self.remaining_tokens(),
            "remaining_time": self.remaining_time(),
            "remaining_steps": self.remaining_steps(),
        }
