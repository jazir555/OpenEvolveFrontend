"""leanaide_mcts - MCTS proof-search components for LeanAIDE integration.

Flat-script module providing the proof-state / tactic data structures and a
minimal but functional Monte-Carlo Tree Search over proof states. Used by
``mcts_evolved_policies`` / ``mcts_evolutionary_nodes`` via
``from leanaide_mcts import ProofState, Tactic, ...``.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# Canonical shared proof-state / tactic primitives (prefers the project-wide
# ``proof_state`` module, falling back to a local fully-functional definition).
from mcts_mdap_bases import ProofState, Tactic, ProofHint  # noqa: E402,F401

logger = logging.getLogger(__name__)


@dataclass
class MCTSConfig:
    max_iterations: int = 200
    exploration_constant: float = 1.4
    max_depth: int = 30
    random_seed: int = 0


@dataclass
class MCTSResult:
    success: bool
    tactic_sequence: List[str] = field(default_factory=list)
    iterations: int = 0
    nodes_expanded: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "tactic_sequence": list(self.tactic_sequence),
            "iterations": self.iterations,
        }


@dataclass
class MCTSNode:
    state: ProofState
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0

    def ucb(self, exploration: float) -> float:
        if self.visits == 0:
            return float("inf")
        return self.value / self.visits + exploration * (2 * (self.parent.visits if self.parent else 0) ** 0.5) / (1 + self.visits)

    def add_child(self, state: ProofState) -> "MCTSNode":
        child = MCTSNode(state=state, parent=self)
        self.children.append(child)
        return child


@dataclass
class MCTSTree:
    root: MCTSNode
    config: MCTSConfig = field(default_factory=MCTSConfig)


class RolloutPolicy:
    """Default uniform-random rollout over a provided tactic vocabulary."""

    def __init__(self, tactics: Optional[List[Tactic]] = None, seed: int = 0):
        self.tactics = tactics or [Tactic("simp"), Tactic("intro"), Tactic("rw"), Tactic("exact")]
        self._rng = random.Random(seed)

    def sample(self, state: ProofState) -> Tactic:
        return self._rng.choice(self.tactics)


class MCTSSelection:
    @staticmethod
    def select(node: MCTSNode, config: MCTSConfig) -> MCTSNode:
        cur = node
        while cur.children:
            cur = max(cur.children, key=lambda c: c.ucb(config.exploration_constant))
        return cur


class MCTSExpansion:
    @staticmethod
    def expand(node: MCTSNode, policy: RolloutPolicy) -> MCTSNode:
        tactic = policy.sample(node.state)
        new_state = node.state.apply(tactic)
        return node.add_child(new_state)


class MCTSSimulation:
    def __init__(self, policy: RolloutPolicy):
        self.policy = policy

    def simulate(self, state: ProofState, max_depth: int = 20) -> float:
        cur = state
        depth = 0
        while not cur.is_solved() and depth < max_depth:
            tactic = self.policy.sample(cur)
            cur = cur.apply(tactic)
            depth += 1
        return 1.0 if cur.is_solved() else 0.0


class MCTSBackpropagation:
    @staticmethod
    def backprop(node: Optional[MCTSNode], value: float) -> None:
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value += value
            cur = cur.parent


class MCTS:
    """A compact MCTS proof searcher."""

    def __init__(self, config: Optional[MCTSConfig] = None,
                 policy: Optional[RolloutPolicy] = None, seed: int = 0):
        self.config = config or MCTSConfig()
        self.policy = policy or RolloutPolicy(seed=self.config.random_seed)
        self._rng = random.Random(self.config.random_seed)

    def search(self, initial: ProofState) -> MCTSResult:
        root = MCTSNode(state=initial)
        expanded = 0
        for i in range(self.config.max_iterations):
            leaf = MCTSSelection.select(root, self.config)
            if not leaf.state.is_solved() and leaf.state.depth < self.config.max_depth:
                child = MCTSExpansion.expand(leaf, self.policy)
                expanded += 1
                leaf = child
            value = MCTSSimulation(self.policy).simulate(leaf.state, self.config.max_depth)
            MCTSBackpropagation.backprop(leaf, value)
            if leaf.state.is_solved():
                return MCTSResult(success=True, tactic_sequence=self._path(leaf),
                                  iterations=i + 1, nodes_expanded=expanded)
        return MCTSResult(success=False, iterations=self.config.max_iterations,
                          nodes_expanded=expanded)

    @staticmethod
    def _path(node: MCTSNode) -> List[str]:
        seq: List[str] = []
        cur: Optional[MCTSNode] = node
        while cur and cur.state.tactic:
            seq.append(cur.state.tactic.to_string())
            cur = cur.parent
        return list(reversed(seq))


class LeanProofMCTS(MCTS):
    """MCTS specialised for Lean proofs (alias with Lean-aware helpers)."""


def run_mcts_search(initial: ProofState, config: Optional[MCTSConfig] = None,
                    seed: int = 0) -> MCTSResult:
    return MCTS(config=config, seed=seed).search(initial)


def record_failure_lineage(state: ProofState, reason: str = "") -> Dict[str, Any]:
    """Record a failed proof attempt lineage for later analysis."""
    lineage: List[str] = []
    cur: Optional[ProofState] = state
    while cur is not None:
        if cur.tactic:
            lineage.append(cur.tactic.to_string())
        cur = cur.parent
    return {"lineage": list(reversed(lineage)), "reason": reason, "depth": state.depth}
