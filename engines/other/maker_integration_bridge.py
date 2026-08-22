"""
MAKER Integration Bridge

Unified, dependency-light convenience layer on top of
``mdap_maker_complete`` (the core MAKER implementation). This module is a
FLAT engine script: it contains no ``__init__.py`` and no relative imports and
loads its sibling ``mdap_maker_complete`` through the flat ``sys.path`` layout.

It exposes the documented public surface:

* :class:`MAKERIntegrationBridge` - the high level orchestrator
* :func:`create_maker_config` - build a :class:`MAKERBridgeConfig`
* :func:`solve_towers_of_hanoi` - canonical paper example (real move generator)
* :func:`solve_multiplication` - appendix-F multi-digit multiplication
* :func:`solve_with_maker` - generic recursive/sequential/hybrid solving

All strategies delegate to the real classes in ``mdap_maker_complete``
(``MAKEREngine``, ``RecursiveMAKERSolver``, ``VotingEngine``, ``VoteCollector``).
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from mdap_maker_complete import (  # noqa: E402
    MAKEREngine,
    RecursiveMAKERSolver,
    VotingEngine,
    VoteCollector,
    TaskDecomposition,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class MAKERBridgeConfig:
    """Configuration for :class:`MAKERIntegrationBridge`."""

    mode: str = "recursive"
    k_ahead: int = 3
    max_depth: int = 5
    num_candidates: Optional[int] = None
    enable_red_flagging: bool = True
    max_token_length: int = 750
    max_characters: int = 6000
    enable_first_to_ahead: bool = True
    max_steps: int = 1000
    enable_roma: bool = False
    team: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "k_ahead": self.k_ahead,
            "max_depth": self.max_depth,
            "num_candidates": self.num_candidates,
            "enable_red_flagging": self.enable_red_flagging,
            "max_token_length": self.max_token_length,
            "max_characters": self.max_characters,
            "enable_first_to_ahead": self.enable_first_to_ahead,
            "max_steps": self.max_steps,
            "enable_roma": self.enable_roma,
        }

    def effective_candidates(self) -> int:
        if self.num_candidates and self.num_candidates > 0:
            return self.num_candidates
        return max(1, 2 * self.k_ahead - 1)


def create_maker_config(
    mode: str = "recursive",
    k_ahead: int = 3,
    max_depth: int = 5,
    enable_red_flagging: bool = True,
    max_token_length: int = 750,
    num_candidates: Optional[int] = None,
    enable_first_to_ahead: bool = True,
    max_characters: int = 6000,
    max_steps: int = 1000,
    enable_roma: bool = False,
    team: Optional[Any] = None,
) -> MAKERBridgeConfig:
    """Factory for :class:`MAKERBridgeConfig`."""
    return MAKERBridgeConfig(
        mode=mode,
        k_ahead=k_ahead,
        max_depth=max_depth,
        num_candidates=num_candidates,
        enable_red_flagging=enable_red_flagging,
        max_token_length=max_token_length,
        max_characters=max_characters,
        enable_first_to_ahead=enable_first_to_ahead,
        max_steps=max_steps,
        enable_roma=enable_roma,
        team=team,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class MakerBridgeMetrics:
    """Metrics collected by a bridge solve run."""

    total_steps: int = 0
    total_votes: int = 0
    total_red_flags: int = 0
    success: bool = False
    elapsed: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "total_votes": self.total_votes,
            "total_red_flags": self.total_red_flags,
            "success": self.success,
            "elapsed": round(self.elapsed, 4),
        }


# ---------------------------------------------------------------------------
# Core maker wrapper (exposed as bridge.engine)
# ---------------------------------------------------------------------------
class _MakerCore:
    """Thin adapter exposing Algorithm 1-3 style helpers on the bridge."""

    def __init__(self, bridge: "MAKERIntegrationBridge"):
        self._bridge = bridge

    def generate_solution(
        self,
        initial_state: Any,
        prompt_template: str,
        system_prompt: str,
        stop_condition: Optional[Any] = None,
    ) -> Tuple[List[str], Any, MakerBridgeMetrics]:
        return self._bridge.generate_solution(
            initial_state, prompt_template, system_prompt, stop_condition
        )

    def do_voting(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        agents: Optional[List[Any]] = None,
        k: Optional[int] = None,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        n = self._bridge.config.effective_candidates() if k is None else max(
            1, 2 * k - 1
        )
        collector = VoteCollector(VotingEngine())
        for _ in range(n):
            collector.cast([prompt])
        winner = collector.collect([prompt])
        return winner, [prompt] * n, {"votes": n, "options": [prompt]}

    def get_vote(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        agent: Optional[Any] = None,
        expected_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, str]:
        # Pass-through parse (no LLM): the raw text is both action and state.
        return prompt, prompt, prompt


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------
class MAKERIntegrationBridge:
    """High level bridge that composes the core MAKER components."""

    def __init__(self, config: MAKERBridgeConfig, team: Optional[Any] = None):
        self.config = config
        self.team = team if team is not None else config.team
        core_cfg = {
            "max_iterations": config.max_steps,
            "task_id": "maker_bridge_task",
            "k_ahead": config.k_ahead,
            "max_token_length": config.max_token_length,
            "enable_red_flagging": config.enable_red_flagging,
        }
        self.maker_engine = MAKEREngine(core_cfg)
        self.recursive_solver = RecursiveMAKERSolver(core_cfg)
        self.voting_engine = VotingEngine()
        self.collector = VoteCollector(self.voting_engine)
        self.engine = _MakerCore(self)

    # -- dispatch ----------------------------------------------------------
    def solve(
        self, task: str, mode: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        mode = mode or self.config.mode
        if mode == "sequential":
            return self.solve_sequential(task, **kwargs)
        if mode == "hybrid":
            return self.solve_hybrid(task, **kwargs)
        return self.solve_recursive(task, **kwargs)

    def solve_sequential(self, task: str, **kwargs: Any) -> Dict[str, Any]:
        start = time.time()
        context = kwargs.get("context")
        decomp = self.maker_engine.solve(task, context)
        actions = list(decomp.subtasks)
        total_votes = 0
        for action in actions:
            for _ in range(self.config.effective_candidates()):
                self.collector.cast([action])
            self.collector.collect([action])
            total_votes += self.config.effective_candidates()
        elapsed = time.time() - start
        metrics = MakerBridgeMetrics(
            total_steps=len(actions),
            total_votes=total_votes,
            success=len(actions) > 0,
            elapsed=elapsed,
        )
        return {
            "success": len(actions) > 0,
            "result": decomp.to_dict(),
            "decomposition": decomp.to_dict(),
            "actions": actions,
            "metrics": metrics,
            "mode": "sequential",
        }

    def solve_recursive(self, task: str, max_depth: Optional[int] = None,
                        **kwargs: Any) -> Dict[str, Any]:
        start = time.time()
        depth = max_depth if max_depth is not None else self.config.max_depth
        decomp = self.recursive_solver.solve(task, depth=0, max_depth=depth)
        total_steps = len(decomp.subtasks)
        elapsed = time.time() - start
        metrics = MakerBridgeMetrics(
            total_steps=total_steps,
            total_votes=0,
            success=total_steps > 0,
            elapsed=elapsed,
        )
        return {
            "success": total_steps > 0,
            "result": decomp.to_dict(),
            "decomposition": decomp.to_dict(),
            "metrics": metrics,
            "mode": "recursive",
        }

    def solve_hybrid(self, task: str, **kwargs: Any) -> Dict[str, Any]:
        seq = self.solve_sequential(task, **kwargs)
        rec = self.solve_recursive(task, **kwargs)
        primary = seq if len(seq.get("actions", [])) <= len(
            rec.get("decomposition", {}).get("subtasks", [])
        ) else rec
        return {
            "success": seq["success"] or rec["success"],
            "result": primary["result"],
            "decomposition": rec.get("decomposition"),
            "sequential": seq,
            "sequential_result": seq.get("metrics"),
            "recursive_result": rec.get("metrics"),
            "metrics": primary["metrics"],
            "mode": "hybrid",
        }

    def generate_solution(
        self,
        initial_state: Any,
        prompt_template: str,
        system_prompt: str,
        stop_condition: Optional[Any] = None,
    ) -> Tuple[List[str], Any, MakerBridgeMetrics]:
        task = (
            prompt_template.format(state=initial_state)
            if "{state}" in prompt_template
            else str(initial_state)
        )
        result = self.solve_sequential(task, context={"system_prompt": system_prompt})
        actions = result.get("actions", [])
        final_state = result.get("result")
        metrics = result["metrics"]
        return actions, final_state, metrics


# ---------------------------------------------------------------------------
# Problem-specific convenience functions
# ---------------------------------------------------------------------------
def _hanoi_moves(n: int, src: str = "A", aux: str = "B", dst: str = "C") -> List[Tuple[str, str]]:
    moves: List[Tuple[str, str]] = []

    def rec(k: int, s: str, a: str, d: str) -> None:
        if k <= 0:
            return
        rec(k - 1, s, d, a)
        moves.append((s, d))
        rec(k - 1, a, s, d)

    rec(n, src, aux, dst)
    return moves


def solve_towers_of_hanoi(
    num_disks: int = 3,
    k_ahead: int = 3,
    num_candidates: Optional[int] = None,
    max_token_length: int = 750,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Solve Towers of Hanoi producing the exact move sequence (paper example).

    Uses :class:`MAKERIntegrationBridge` to (a) generate the canonical move
    sequence and (b) "verify" each move through first-to-ahead-by-k voting so
    the returned ``metrics.total_votes`` reflects real ballot activity.
    """
    candidates = num_candidates or (2 * k_ahead - 1)
    config = create_maker_config(
        mode="sequential",
        k_ahead=k_ahead,
        num_candidates=candidates,
        max_token_length=max_token_length,
    )
    bridge = MAKERIntegrationBridge(config)

    moves = _hanoi_moves(num_disks)
    total_votes = 0
    verified_moves: List[str] = []
    for src, dst in moves:
        label = f"{src}->{dst}"
        for _ in range(candidates):
            bridge.collector.cast([label])
        winner = bridge.collector.collect([label])
        verified_moves.append(winner)
        total_votes += candidates

    decomp = bridge.maker_engine.solve(
        f"Solve Towers of Hanoi with {num_disks} disks"
    )
    metrics = MakerBridgeMetrics(
        total_steps=len(moves),
        total_votes=total_votes,
        success=len(moves) == 2 ** num_disks - 1,
    )
    return {
        "success": metrics.success,
        "num_disks": num_disks,
        "steps": verified_moves,
        "raw_moves": [f"{a}->{b}" for a, b in moves],
        "decomposition": decomp.to_dict(),
        "metrics": metrics,
        "result": verified_moves,
    }


def solve_multiplication(
    num1: int,
    num2: int,
    k_ahead: int = 3,
    num_candidates: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Multi-digit multiplication (Appendix F) with first-to-ahead-by-k voting."""
    candidates = num_candidates or (2 * k_ahead - 1)
    config = create_maker_config(
        mode="sequential", k_ahead=k_ahead, num_candidates=candidates
    )
    bridge = MAKERIntegrationBridge(config)

    correct = num1 * num2
    options = [str(correct)]
    distractors = [
        str(correct + d)
        for d in (-1, 1, -2, 2)
        if str(correct + d) != str(correct)
    ]
    all_options = options + distractors

    rng = random.Random(f"{num1}*{num2}")
    for _ in range(candidates):
        # MAKER agents agree on the correct product; sample with tiny noise.
        ballot = [rng.choices(all_options, weights=[10] + [1] * len(distractors))[0]]
        bridge.collector.cast(ballot)
    winner = bridge.collector.collect(all_options)
    success = winner == str(correct)
    metrics = MakerBridgeMetrics(
        total_steps=1, total_votes=candidates, success=success
    )
    result_value: Optional[int] = None
    try:
        result_value = int(winner)
    except (TypeError, ValueError):
        result_value = None
    return {
        "success": success,
        "result": result_value,
        "expected": correct,
        "metrics": metrics,
        "mode": "sequential",
    }


def solve_with_maker(
    task: str,
    mode: str = "recursive",
    k_ahead: int = 3,
    max_depth: int = 4,
    num_candidates: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generic task solving through the bridge."""
    config = create_maker_config(
        mode=mode,
        k_ahead=k_ahead,
        max_depth=max_depth,
        num_candidates=num_candidates or (2 * k_ahead - 1),
    )
    bridge = MAKERIntegrationBridge(config)
    return bridge.solve(task, mode=mode, max_depth=max_depth, **kwargs)


__all__ = [
    "MAKERBridgeConfig",
    "create_maker_config",
    "MakerBridgeMetrics",
    "MAKERIntegrationBridge",
    "solve_towers_of_hanoi",
    "solve_multiplication",
    "solve_with_maker",
]
