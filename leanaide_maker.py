"""
LeanAide MAKER - Multi-Agent Voting System for Lean 4 Proofs

Production-ready MAKER (Multi-Agent Knowledge Enhanced Reasoning) implementation
specialized for Lean 4 theorem proving. Uses voting-based consensus to select
tactics step-by-step during proof construction.

Architecture:
    - LeanMakerStep: MAKER step specialized for Lean 4 proofs
    - LeanMakerConfig: Configuration for Lean 4 MAKER system
    - LeanProofState: Represents state during MAKER execution
    - LeanMakerEngine: Main MAKER orchestrator for Lean 4
    - LeanTacticVoter: Agent that votes on next tactic
    - TacticVote: Single vote for a tactic
    - LeanAggregator: Aggregates votes and selects winning tactic
    - LeanRedFlagRules: Red-flag rules for Lean 4 tactics
    - LeanMakerRunResult: Result of MAKER proof construction

Voting Pipeline:
    1. Get current proof state
    2. Collect votes from N agents (each proposes a tactic)
    3. Red-flag invalid tactics
    4. Aggregate valid votes (first-k-ahead)
    5. Select winning tactic
    6. Apply tactic and update state
    7. Repeat until proof complete or max steps

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import json
import logging
import random
import time
import uuid
import threading
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union
)
import sqlite3
import hashlib
from pathlib import Path

# Import base MAKER engine
try:
    from maker_engine import MakerStep, MakerConfig, MakerEngine
    from mdap_engine import RedFlagRules, canonicalize_candidate
    from workflow_structures import ModelConfig, Team
    MAKER_ENGINE_AVAILABLE = True
except ImportError:
    MAKER_ENGINE_AVAILABLE = False
    logging.warning("Base MAKER engine not available - using standalone mode")

# Import LeanAide components
try:
    from leanaide_client import LeanAideClient
    from leanaide_evolution import Tactic, LeanProof
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logging.warning("LeanAide integration not available - using simulation mode")


logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================

class VoterType(Enum):
    """Types of tactic voters."""
    RANDOM = "random"
    HEURISTIC = "heuristic"
    EVOLUTIONARY = "evolutionary"
    MCTS = "mcts"
    DIRECT = "direct"
    ENSEMBLE = "ensemble"


class AggregationStrategy(Enum):
    """Vote aggregation strategies."""
    FIRST_K_AHEAD = "first_k_ahead"
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    THRESHOLD = "threshold"
    CONDORCET = "condorcet"
    BORDA = "borda"


class TerminationReason(Enum):
    """Reasons for MAKER termination."""
    PROOF_COMPLETE = "proof_complete"
    MAX_STEPS_REACHED = "max_steps_reached"
    NO_CONSENSUS = "no_consensus"
    TIME_LIMIT = "time_limit"
    ALL_TACTICS_FLAGGED = "all_tactics_flagged"
    PROOF_STUCK = "proof_stuck"
    ERROR = "error"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TacticVote:
    """
    A single vote for a tactic.

    Attributes:
        tactic: The tactic being voted for
        confidence: Confidence score (0.0 to 1.0)
        rationale: Explanation for why this tactic
        voter_id: ID of the voter agent
        voter_type: Type of voter that cast this vote
        estimated_success: Estimated probability of success
        proof_state_hash: Hash of proof state when vote was cast
        metadata: Additional metadata
    """
    tactic: str
    confidence: float
    rationale: str
    voter_id: str
    voter_type: VoterType
    estimated_success: float = 0.5
    proof_state_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tactic": self.tactic,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "voter_id": self.voter_id,
            "voter_type": self.voter_type.value,
            "estimated_success": self.estimated_success,
            "proof_state_hash": self.proof_state_hash,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TacticVote':
        """Create from dictionary."""
        return cls(
            tactic=data["tactic"],
            confidence=data["confidence"],
            rationale=data["rationale"],
            voter_id=data["voter_id"],
            voter_type=VoterType(data["voter_type"]),
            estimated_success=data.get("estimated_success", 0.5),
            proof_state_hash=data.get("proof_state_hash", ""),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time())
        )


@dataclass
class LeanProofState:
    """
    Represents a Lean 4 proof state during MAKER execution.

    Attributes:
        goals: Current unsolved goals
        context: Current proof context (hypotheses, assumptions, definitions)
        tactic_sequence: Sequence of tactics applied so far
        depth: Depth in the proof tree
        is_complete: Whether all goals are solved
        metadata: Additional metadata
    """
    goals: List[str] = field(default_factory=list)
    context: List[str] = field(default_factory=list)
    tactic_sequence: List[Tactic] = field(default_factory=list)
    depth: int = 0
    is_complete: bool = False
    hash: str = field(default="")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute hash after initialization."""
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute unique hash of the proof state."""
        state_str = f"{json.dumps(self.goals, sort_keys=True)}:{json.dumps(self.context, sort_keys=True)}"
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "goals": self.goals,
            "context": self.context,
            "tactic_sequence": [t.to_dict() for t in self.tactic_sequence],
            "depth": self.depth,
            "is_complete": self.is_complete,
            "hash": self.hash,
            "metadata": self.metadata
        }

    def is_complete(self) -> bool:
        """Check if proof is complete."""
        return len(self.goals) == 0 or self.is_complete

    def get_applicable_tactics(self) -> List[str]:
        """Get tactics applicable to current state."""
        # Basic applicable tactics based on goals
        tactics = []

        for goal in self.goals:
            # Check for quantifiers
            if any(q in goal for q in ["forall", "∀", "→", "->"]):
                tactics.append("intros")

            # Check for equality
            if "=" in goal:
                tactics.extend(["simp", "rw", "linarith", "ring"])

            # Check for logical connectives
            if any(conn in goal for conn in ["∧", "and", "\\and"]):
                tactics.append("constructor")

            if any(conn in goal for conn in ["∨", "or", "\\or"]):
                tactics.append("cases")

            # Check for existence
            if any(q in goal for q in ["exists", "∃"]):
                tactics.append("use")

            # Check for natural numbers
            if "Nat" in goal or "ℕ" in goal:
                tactics.extend(["induction", "cases"])

            # Check for negation
            if any(q in goal for q in ["not", "¬", "Nonempty"]):
                tactics.extend(["push_neg", "contrapose"])

        # Add general tactics
        tactics.extend(["simp", "aesop", "done", "trivial"])

        # Deduplicate while preserving order
        seen = set()
        unique_tactics = []
        for t in tactics:
            if t not in seen:
                seen.add(t)
                unique_tactics.append(t)

        return unique_tactics

    def apply_tactic(self, tactic: str) -> Optional['LeanProofState']:
        """
        Apply a tactic to get new proof state.

        Args:
            tactic: Tactic to apply

        Returns:
            New proof state, or None if tactic invalid
        """
        # Create new state
        new_state = LeanProofState(
            goals=self.goals.copy(),
            context=self.context.copy(),
            tactic_sequence=self.tactic_sequence.copy(),
            depth=self.depth + 1
        )

        # Parse tactic
        tactic_obj = self._parse_tactic(tactic)
        new_state.tactic_sequence.append(tactic_obj)

        # Simulate tactic application
        new_state = self._simulate_tactic(new_state, tactic)

        return new_state

    def _parse_tactic(self, tactic_str: str) -> Tactic:
        """Parse a tactic string into a Tactic object."""
        parts = tactic_str.strip().split(maxsplit=1)
        name = parts[0] if parts else "unknown"
        arguments = parts[1].split() if len(parts) > 1 else []

        return Tactic(name=name, arguments=arguments)

    def _simulate_tactic(self, new_state: 'LeanProofState', tactic: str) -> 'LeanProofState':
        """Simulate tactic application heuristically."""
        tactic_name = tactic.split()[0] if tactic else ""

        # Simulate based on tactic type
        if tactic_name in ["intros", "intro"]:
            # Intros typically introduce variables from goal
            if new_state.goals:
                # Remove forall quantifiers heuristically
                new_state.goals = [g.split("->")[-1].strip() for g in new_state.goals if "->" in g]

        elif tactic_name in ["simp", "simp_all"]:
            # Simplification may simplify goals
            new_state.goals = [self._simplify_goal(g) for g in new_state.goals]

        elif tactic_name in ["rw", "rewrite"]:
            # Rewrite might simplify
            if new_state.goals:
                new_state.goals = [new_state.goals[0]]  # Simplified: keep one goal

        elif tactic_name in ["cases", "induction"]:
            # Case analysis typically creates multiple goals
            if new_state.goals:
                # Simplified: duplicate goal
                new_state.goals = new_state.goals + new_state.goals.copy()

        elif tactic_name in ["aesop", "trivial", "done"]:
            # Automation might solve goals
            if random.random() > 0.7:  # 30% chance of solving
                new_state.goals = []
                new_state.is_complete = True

        # Check if complete
        new_state.is_complete = len(new_state.goals) == 0 or new_state.is_complete

        # Update hash
        new_state.hash = new_state._compute_hash()

        return new_state

    def _simplify_goal(self, goal: str) -> str:
        """Simplify a goal string heuristically."""
        # Remove common redundant patterns
        simplified = goal
        simplified = simplified.replace("Nat.succ", "succ")
        simplified = simplified.replace("Nat.zero", "0")
        simplified = simplified.replace("Nat.add", "Nat.add")
        return simplified

    def estimate_distance_to_proof(self) -> int:
        """
        Estimate the number of tactics needed to complete proof.

        Returns:
            Estimated distance (lower is better)
        """
        if self.is_complete:
            return 0

        # Estimate based on goal complexity
        total_complexity = 0
        for goal in self.goals:
            # Count quantifiers
            quantifiers = goal.count("forall") + goal.count("∀") + goal.count("exists") + goal.count("∃")
            # Count connectives
            connectives = goal.count("->") + goal.count("∧") + goal.count("∨")
            # Count nested structures
            depth = goal.count("(")

            total_complexity += quantifiers * 2 + connectives + depth // 2

        return max(1, total_complexity)


@dataclass
class LeanMakerConfig(MakerConfig if MAKER_ENGINE_AVAILABLE else object):
    """
    Configuration for Lean 4 MAKER system.

    Extends MakerConfig with Lean-specific parameters.

    Attributes:
        k_min: Minimum K value for first-k-ahead voting
        k_max: Maximum K value for first-k-ahead voting
        max_votes_per_step: Maximum votes to collect per step
        max_steps: Maximum proof steps
        timeout_seconds: Timeout per step
        checkpoint_interval: Checkpoint save interval
        red_flag_rules: Red-flag rules for Lean 4
        tactic_preferences: Bias toward certain tactics
        aggregation_strategy: Strategy for aggregating votes
        voter_types: Types of voters to use
        adaptive_k: Whether to adapt K based on progress
        proof_length_penalty: Penalty for longer proofs
        diversity_weight: Weight for voter diversity
        confidence_threshold: Minimum confidence to accept vote
    """
    k_min: int = 2
    k_max: int = 5
    max_votes_per_step: int = 20
    max_steps: int = 100
    timeout_seconds: int = 90
    checkpoint_interval: int = 25
    red_flag_rules: Optional['LeanRedFlagRules'] = None
    tactic_preferences: Dict[str, float] = field(default_factory=lambda: {
        "simp": 1.0,
        "intros": 1.0,
        "rw": 1.0,
        "apply": 0.9,
        "exact": 0.9,
        "aesop": 0.8,
        "linarith": 0.7
    })
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FIRST_K_AHEAD
    voter_types: List[VoterType] = field(default_factory=lambda: [
        VoterType.HEURISTIC,
        VoterType.EVOLUTIONARY,
        VoterType.MCTS,
        VoterType.RANDOM
    ])
    adaptive_k: bool = True
    proof_length_penalty: float = 0.01
    diversity_weight: float = 0.1
    confidence_threshold: float = 0.3
    enable_leanaide: bool = True
    leanaide_url: str = "http://localhost:7654"
    parallel_voting: bool = True
    max_parallel_voters: int = 5

    def get_tactic_preferences(self) -> Dict[str, float]:
        """Get tactic preference weights."""
        return self.tactic_preferences.copy()

    def compute_k_value(self, step: int, progress: float) -> int:
        """
        Compute adaptive K value for first-k-ahead voting.

        Args:
            step: Current step number
            progress: Proof progress (0.0 to 1.0)

        Returns:
            K value for voting
        """
        if self.adaptive_k:
            # Adapt K based on progress
            # Higher K early, lower K later
            base_k = self.k_max - int(progress * (self.k_max - self.k_min))
        else:
            base_k = (self.k_min + self.k_max) // 2

        return max(self.k_min, min(self.k_max, base_k))


@dataclass
class LeanMakerStep(MakerStep if MAKER_ENGINE_AVAILABLE else object):
    """
    MAKER step specialized for Lean 4 proofs.

    Extends MakerStep with Lean-specific prompt templates
    and state representation.

    Attributes:
        step_id: Unique step identifier
        prompt_template: Template for generating prompts
        expected_schema: Expected schema for responses
        task_type: Type of task (e.g., "tactic_selection")
        priority: Step priority
        system_prompt: System prompt for LLM
        metadata: Additional metadata
    """
    step_id: str
    prompt_template: str
    expected_schema: Optional[Dict[str, Any]] = None
    task_type: str = "tactic_selection"
    priority: int = 0
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render_prompt(self, state: LeanProofState, history: List[Dict[str, Any]]) -> str:
        """
        Render prompt for current proof state.

        Args:
            state: Current proof state
            history: History of previous steps

        Returns:
            Formatted prompt string
        """
        state_str = json.dumps(state.to_dict(), indent=2, ensure_ascii=True)
        history_str = json.dumps(history, indent=2, ensure_ascii=True)

        return self.prompt_template.format(
            state=state_str,
            history=history_str,
            goals="\n".join(f"  {i+1}. {g}" for i, g in enumerate(state.goals)),
            context="\n".join(f"  {i+1}. {c}" for i, c in enumerate(state.context)),
            depth=state.depth
        )


# =============================================================================
# Red Flag Rules
# =============================================================================

@dataclass
class LeanRedFlagRules(RedFlagRules if MAKER_ENGINE_AVAILABLE else object):
    """
    Enhanced red-flag rules for Lean 4 tactics.

    Rules for filtering invalid or dangerous tactics:
    - Invalid tactic syntax
    - Tactic not applicable to current goal
    - Proof too long (max steps exceeded)
    - Circular reasoning
    - Contradictory tactics

    Attributes:
        max_tactic_length: Maximum tactic string length
        max_proof_steps: Maximum proof steps before flagging
        blocked_tactics: Tactics that should never be used
        required_tactic_patterns: Required patterns for valid tactics
        circular_detection: Enable circular reasoning detection
    """
    max_tactic_length: int = 200
    max_proof_steps: int = 200
    blocked_tactics: Set[str] = field(default_factory=set)
    required_tactic_patterns: List[str] = field(default_factory=list)
    circular_detection: bool = True

    def is_flagged(
        self,
        tactic: str,
        state: LeanProofState,
        schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if a tactic should be red-flagged.

        Args:
            tactic: Tactic to check
            state: Current proof state
            schema: Optional schema for validation

        Returns:
            Tuple of (is_flagged, list_of_reasons)
        """
        reasons: List[str] = []

        # Check tactic length
        if len(tactic) > self.max_tactic_length:
            reasons.append("tactic_too_long")

        # Check for blocked tactics
        if any(blocked in tactic for blocked in self.blocked_tactics):
            reasons.append("blocked_tactic")

        # Check proof length
        if state.depth > self.max_proof_steps:
            reasons.append("proof_too_long")

        # Check for circular reasoning
        if self.circular_detection and self._detect_circular(tactic, state):
            reasons.append("circular_reasoning")

        # Validate tactic syntax
        if not self._validate_tactic_syntax(tactic):
            reasons.append("invalid_syntax")

        # Check tactic applicability
        if not self.validate_tactic_applicability(tactic, state):
            reasons.append("tactic_not_applicable")

        return len(reasons) > 0, reasons

    def validate_tactic_applicability(self, tactic: str, state: LeanProofState) -> bool:
        """
        Validate that a tactic is applicable to current state.

        Args:
            tactic: Tactic to validate
            state: Current proof state

        Returns:
            True if tactic is applicable
        """
        tactic_name = tactic.split()[0] if tactic else ""

        # Check if we have goals
        if not state.goals and tactic_name not in ["done", "trivial"]:
            return False

        # Check applicability based on tactic type
        if tactic_name == "intros":
            # Only applicable if goals have forall/implication
            return any(any(q in g for q in ["forall", "∀", "→", "->"]) for g in state.goals)

        elif tactic_name in ["cases", "induction"]:
            # Only applicable if we have a variable to case on
            return any("Nat" in g or "ℕ" in g for g in state.goals)

        elif tactic_name in ["linarith", "ring"]:
            # Only applicable for arithmetic goals
            return any(any(op in g for op in ["=", "+", "*", "-", "<", ">"]) for g in state.goals)

        elif tactic_name in ["constructor"]:
            # Only applicable for existence/conjunction goals
            return any(any(q in g for q in ["∧", "and", "∃", "exists"]) for g in state.goals)

        return True

    def _validate_tactic_syntax(self, tactic: str) -> bool:
        """Validate basic tactic syntax."""
        if not tactic or not tactic.strip():
            return False

        # Check for balanced brackets
        brackets = {"(": ")", "[": "]", "{": "}"}
        stack = []

        for char in tactic:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return False
                opening = stack.pop()
                if brackets[opening] != char:
                    return False

        return len(stack) == 0

    def _detect_circular(self, tactic: str, state: LeanProofState) -> bool:
        """Detect circular reasoning in tactic application."""
        # Check if tactic undoes previous work
        tactic_lower = tactic.lower()

        # Check for undo patterns
        undo_patterns = ["rw [←", "simp only"]
        if any(pattern in tactic_lower for pattern in undo_patterns):
            # Check if we're reverting a recent tactic
            recent_tactics = state.tactic_sequence[-3:]  # Last 3 tactics
            recent_names = [t.name for t in recent_tactics]
            return any(name in tactic for name in recent_names)

        return False


# =============================================================================
# Tactic Voters
# =============================================================================

class LeanTacticVoter:
    """
    Base class for agents that vote on next tactic.

    Voter types:
    - RandomVoter: Selects random applicable tactic
    - HeuristicVoter: Uses heuristics to select tactic
    - EvolutionaryVoter: Uses evolutionary strategy
    - MCTSVoter: Uses MCTS-guided selection
    - DirectVoter: Direct LeanAide suggestion
    - EnsembleVoter: Combines multiple voters
    """

    def __init__(
        self,
        voter_id: str,
        voter_type: VoterType,
        config: LeanMakerConfig
    ):
        self.voter_id = voter_id
        self.voter_type = voter_type
        self.config = config
        self.vote_count = 0
        self.success_count = 0

    def vote(self, state: LeanProofState) -> TacticVote:
        """
        Cast a vote for the next tactic.

        Args:
            state: Current proof state

        Returns:
            TacticVote with selected tactic and rationale
        """
        raise NotImplementedError

    def get_tactic_preferences(self) -> Dict[str, float]:
        """Get tactic preference weights."""
        return self.config.get_tactic_preferences()

    def update_stats(self, was_successful: bool):
        """Update voter statistics."""
        self.vote_count += 1
        if was_successful:
            self.success_count += 1

    def get_success_rate(self) -> float:
        """Get voter success rate."""
        if self.vote_count == 0:
            return 0.0
        return self.success_count / self.vote_count


class RandomVoter(LeanTacticVoter):
    """Random tactic selection voter."""

    def vote(self, state: LeanProofState) -> TacticVote:
        """Vote for a random applicable tactic."""
        applicable = state.get_applicable_tactics()

        if not applicable:
            # Fallback to basic tactics
            applicable = ["simp", "intros", "done"]

        # Apply tactic preferences
        preferences = self.get_tactic_preferences()
        weighted_tactics = []
        for tactic in applicable:
            weight = preferences.get(tactic, 0.5)
            weighted_tactics.extend([tactic] * int(weight * 10))

        selected = random.choice(weighted_tactics if weighted_tactics else applicable)

        return TacticVote(
            tactic=selected,
            confidence=random.uniform(0.3, 0.6),
            rationale=f"Random selection from {len(applicable)} applicable tactics",
            voter_id=self.voter_id,
            voter_type=self.voter_type,
            proof_state_hash=state.hash
        )


class HeuristicVoter(LeanTacticVoter):
    """Heuristic-guided tactic selection voter."""

    def vote(self, state: LeanProofState) -> TacticVote:
        """Vote for heuristic-based tactic selection."""
        applicable = state.get_applicable_tactics()

        # Score each tactic
        scores = []
        for tactic in applicable:
            score = self._score_tactic(tactic, state)
            scores.append((score, tactic))

        # Sort by score
        scores.sort(reverse=True)

        # Select from top-3 with bias toward top
        top_k = min(3, len(scores))
        if top_k > 0:
            weights = [1.0 / (i + 1) for i in range(top_k)]
            total = sum(weights)
            probs = [w / total for w in weights]
            idx = random.choices(range(top_k), weights=probs)[0]
            selected = scores[idx][1]
            confidence = scores[idx][0] / 10.0  # Normalize to 0-1
        else:
            selected = "simp"
            confidence = 0.5

        return TacticVote(
            tactic=selected,
            confidence=min(0.9, confidence),
            rationale=f"Heuristic selection: {selected} scored {scores[0][0]:.2f}",
            voter_id=self.voter_id,
            voter_type=self.voter_type,
            proof_state_hash=state.hash
        )

    def _score_tactic(self, tactic: str, state: LeanProofState) -> float:
        """Score a tactic based on heuristics."""
        score = 5.0  # Base score

        # Preference weight
        preferences = self.get_tactic_preferences()
        score += preferences.get(tactic, 0.0) * 2.0

        # Goal-specific bonuses
        for goal in state.goals:
            # Intros bonus for quantifiers
            if tactic == "intros" and any(q in goal for q in ["forall", "∀", "→", "->"]):
                score += 3.0

            # Simplification bonus for complex goals
            if tactic in ["simp", "simp_all"] and len(goal) > 50:
                score += 2.0

            # Automation bonus for simple goals
            if tactic in ["aesop", "trivial"] and len(goal) < 30:
                score += 2.0

            # Case analysis bonus for inductive types
            if tactic in ["cases", "induction"] and any(typ in goal for typ in ["Nat", "ℕ", "List"]):
                score += 2.5

        # Depth adjustments
        if state.depth < 5:
            # Early: prefer intros and simplification
            if tactic in ["intros", "simp"]:
                score += 1.5
        elif state.depth > 20:
            # Late: prefer automation
            if tactic in ["aesop", "trivial", "done"]:
                score += 2.0

        return score


class EvolutionaryVoter(LeanTacticVoter):
    """Evolutionary strategy voter."""

    def __init__(self, *args, population_size: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.population_size = population_size
        self.history: List[str] = []

    def vote(self, state: LeanProofState) -> TacticVote:
        """Vote using evolutionary strategy."""
        applicable = state.get_applicable_tactics()

        if not applicable:
            return TacticVote(
                tactic="simp",
                confidence=0.4,
                rationale="No applicable tactics, using fallback",
                voter_id=self.voter_id,
                voter_type=self.voter_type,
                proof_state_hash=state.hash
            )

        # Generate population of tactic sequences
        population = self._generate_population(state, applicable)

        # Evaluate fitness
        scored_population = [(self._fitness(seq, state), seq) for seq in population]
        scored_population.sort(reverse=True, key=lambda x: x[0])

        # Select best tactic from best sequence
        best_sequence = scored_population[0][1] if scored_population else [applicable[0]]
        selected = best_sequence[0]

        # Confidence based on fitness
        confidence = min(0.9, scored_population[0][0] if scored_population else 0.5)

        return TacticVote(
            tactic=selected,
            confidence=confidence,
            rationale=f"Evolutionary: best of {self.population_size} sequences",
            voter_id=self.voter_id,
            voter_type=self.voter_type,
            proof_state_hash=state.hash
        )

    def _generate_population(self, state: LeanProofState, applicable: List[str]) -> List[List[str]]:
        """Generate initial population of tactic sequences."""
        population = []

        for _ in range(self.population_size):
            # Random sequence length (1-3 tactics)
            length = random.randint(1, 3)
            sequence = [random.choice(applicable) for _ in range(length)]
            population.append(sequence)

        return population

    def _fitness(self, sequence: List[str], state: LeanProofState) -> float:
        """Evaluate fitness of a tactic sequence."""
        score = 0.0

        # Prefer shorter sequences (Occam's razor)
        score -= len(sequence) * 0.1

        # Prefer preferred tactics
        preferences = self.get_tactic_preferences()
        for tactic in sequence:
            score += preferences.get(tactic, 0.0)

        # Simulate application
        simulated_state = state
        for tactic in sequence:
            simulated_state = simulated_state.apply_tactic(tactic)
            if not simulated_state:
                return 0.0  # Invalid sequence

            if simulated_state.is_complete:
                score += 10.0  # Big bonus for completing proof
                break

        # Prefer sequences that reduce goals
        if len(state.goals) > 0 and len(simulated_state.goals) < len(state.goals):
            score += 2.0

        return max(0.0, score)


class MCTSVoter(LeanTacticVoter):
    """MCTS-guided tactic selection voter."""

    def __init__(self, *args, simulations: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulations = simulations

    def vote(self, state: LeanProofState) -> TacticVote:
        """Vote using simplified MCTS."""
        applicable = state.get_applicable_tactics()

        if not applicable:
            return TacticVote(
                tactic="simp",
                confidence=0.4,
                rationale="No applicable tactics, using fallback",
                voter_id=self.voter_id,
                voter_type=self.voter_type,
                proof_state_hash=state.hash
            )

        # Run simplified MCTS for each applicable tactic
        scores = {}
        for tactic in applicable:
            score = self._simulate_tactic(tactic, state)
            scores[tactic] = score

        # Select best
        best_tactic = max(scores, key=scores.get)
        best_score = scores[best_tactic]
        confidence = min(0.9, best_score / 10.0)

        return TacticVote(
            tactic=best_tactic,
            confidence=confidence,
            rationale=f"MCTS: {best_tactic} scored {best_score:.2f} in {self.simulations} simulations",
            voter_id=self.voter_id,
            voter_type=self.voter_type,
            proof_state_hash=state.hash
        )

    def _simulate_tactic(self, tactic: str, state: LeanProofState) -> float:
        """Simulate applying a tactic multiple times."""
        total_score = 0.0

        for _ in range(self.simulations):
            new_state = state.apply_tactic(tactic)
            if not new_state:
                continue

            if new_state.is_complete:
                total_score += 10.0
            else:
                # Score based on goal reduction
                reduction = len(state.goals) - len(new_state.goals)
                total_score += reduction

        return total_score / self.simulations


class DirectVoter(LeanTacticVoter):
    """Direct LeanAide suggestion voter."""

    def __init__(self, *args, leanaide_client: Optional[LeanAideClient] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.leanaide_client = leanaide_client

    async def vote_async(self, state: LeanProofState) -> TacticVote:
        """Vote using direct LeanAide suggestion."""
        if not self.leanaide_client or not LEANAIDE_AVAILABLE:
            # Fallback to heuristic
            return HeuristicVoter(
                voter_id=self.voter_id,
                voter_type=self.voter_type,
                config=self.config
            ).vote(state)

        try:
            # Create Lean code for current state
            lean_code = self._state_to_lean_code(state)

            # Use LeanAide to get suggestion
            result = await self.leanaide_client.elaborate(lean_code)

            if result.success and result.data:
                # Extract suggested tactic
                tactic = self._extract_suggested_tactic(result.data)

                return TacticVote(
                    tactic=tactic,
                    confidence=0.8,
                    rationale="Direct LeanAide suggestion",
                    voter_id=self.voter_id,
                    voter_type=self.voter_type,
                    proof_state_hash=state.hash
                )
        except Exception as e:
            logger.warning(f"LeanAide vote failed: {e}")

        # Fallback
        return TacticVote(
            tactic="simp",
            confidence=0.5,
            rationale="LeanAide unavailable, using fallback",
            voter_id=self.voter_id,
            voter_type=self.voter_type,
            proof_state_hash=state.hash
        )

    def vote(self, state: LeanProofState) -> TacticVote:
        """Synchronous vote wrapper."""
        # For compatibility, return a basic vote
        # In practice, use vote_async
        return TacticVote(
            tactic="simp",
            confidence=0.6,
            rationale="Direct voter (async required for full functionality)",
            voter_id=self.voter_id,
            voter_type=self.voter_type,
            proof_state_hash=state.hash
        )

    def _state_to_lean_code(self, state: LeanProofState) -> str:
        """Convert proof state to Lean code."""
        code = "import Mathlib\n\n"

        # Add context
        for i, hyp in enumerate(state.context):
            code += f"have h{i} : {hyp}\n"

        # Add goals
        if state.goals:
            code += f"\ntheorem temp_goal : {state.goals[0]} := by\n"

        return code

    def _extract_suggested_tactic(self, data: Dict[str, Any]) -> str:
        """Extract suggested tactic from LeanAide response."""
        # Try various fields
        if "tactic" in data:
            return data["tactic"]
        if "suggestion" in data:
            return data["suggestion"]
        if "nextTactic" in data:
            return data["nextTactic"]

        return "simp"  # Fallback


class EnsembleVoter(LeanTacticVoter):
    """Ensemble voter that combines multiple voters."""

    def __init__(self, *args, sub_voters: List[LeanTacticVoter] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub_voters = sub_voters or []

    def vote(self, state: LeanProofState) -> TacticVote:
        """Vote by combining sub-voter preferences."""
        if not self.sub_voters:
            return TacticVote(
                tactic="simp",
                confidence=0.5,
                rationale="No sub-voters configured",
                voter_id=self.voter_id,
                voter_type=self.voter_type,
                proof_state_hash=state.hash
            )

        # Collect votes from sub-voters
        votes: List[TacticVote] = []
        for voter in self.sub_voters:
            vote = voter.vote(state)
            votes.append(vote)

        # Aggregate votes
        tactic_counts = Counter(v.tactic for v in votes)
        most_common = tactic_counts.most_common(1)[0]

        selected = most_common[0]
        count = most_common[1]

        # Average confidence for selected tactic
        selected_votes = [v for v in votes if v.tactic == selected]
        avg_confidence = sum(v.confidence for v in selected_votes) / len(selected_votes)

        return TacticVote(
            tactic=selected,
            confidence=avg_confidence,
            rationale=f"Ensemble: {count}/{len(votes)} voters selected this tactic",
            voter_id=self.voter_id,
            voter_type=self.voter_type,
            proof_state_hash=state.hash
        )


# =============================================================================
# Vote Aggregator
# =============================================================================

class LeanAggregator:
    """
    Aggregates votes and selects winning tactic.

    Strategies:
    - First-K-Ahead: First tactic to get K votes ahead
    - Majority: Tactic with >50% votes
    - Weighted: Weighted by voter confidence
    - Threshold: Tactic with votes above threshold
    - Condorcet: Condorcet winner (beats all others pairwise)
    - Borda: Borda count (ranked voting)
    """

    def __init__(self, strategy: AggregationStrategy = AggregationStrategy.FIRST_K_AHEAD):
        self.strategy = strategy

    def aggregate(
        self,
        votes: List[TacticVote],
        k_value: int = 3,
        threshold: float = 0.5
    ) -> Optional[str]:
        """
        Aggregate votes and select winning tactic.

        Args:
            votes: List of votes to aggregate
            k_value: K value for first-k-ahead
            threshold: Threshold for threshold-based selection

        Returns:
            Winning tactic, or None if no winner
        """
        if not votes:
            return None

        if self.strategy == AggregationStrategy.FIRST_K_AHEAD:
            return self._first_k_ahead(votes, k_value)
        elif self.strategy == AggregationStrategy.MAJORITY:
            return self._majority(votes, threshold)
        elif self.strategy == AggregationStrategy.WEIGHTED:
            return self._weighted(votes)
        elif self.strategy == AggregationStrategy.THRESHOLD:
            return self._threshold(votes, threshold)
        elif self.strategy == AggregationStrategy.CONDORCET:
            return self._condorcet(votes)
        elif self.strategy == AggregationStrategy.BORDA:
            return self._borda(votes)
        else:
            return self._first_k_ahead(votes, k_value)

    def _first_k_ahead(self, votes: List[TacticVote], k_value: int) -> Optional[str]:
        """First-K-Ahead: first tactic to be K votes ahead."""
        counts: Dict[str, int] = Counter(v.tactic for v in votes)

        if not counts:
            return None

        # Find tactic that is K ahead
        while counts:
            leader = max(counts, key=counts.get)
            leader_count = counts[leader]

            # Find second place
            sorted_counts = sorted(counts.values(), reverse=True)
            second_count = sorted_counts[1] if len(sorted_counts) > 1 else 0

            if leader_count >= second_count + k_value:
                return leader

            # Remove leader and continue
            del counts[leader]

        return None

    def _majority(self, votes: List[TacticVote], threshold: float) -> Optional[str]:
        """Majority: tactic with > threshold of votes."""
        counts: Dict[str, int] = Counter(v.tactic for v in votes)
        total = len(votes)

        if total == 0:
            return None

        for tactic, count in counts.items():
            if count / total > threshold:
                return tactic

        return None

    def _weighted(self, votes: List[TacticVote]) -> Optional[str]:
        """Weighted: sum of confidences for each tactic."""
        weighted_scores: Dict[str, float] = defaultdict(float)

        for vote in votes:
            weighted_scores[vote.tactic] += vote.confidence

        if not weighted_scores:
            return None

        return max(weighted_scores, key=weighted_scores.get)

    def _threshold(self, votes: List[TacticVote], threshold: float) -> Optional[str]:
        """Threshold: tactic with votes >= threshold."""
        counts: Dict[str, int] = Counter(v.tactic for v in votes)

        for tactic, count in counts.items():
            if count >= threshold:
                return tactic

        return None

    def _condorcet(self, votes: List[TacticVote]) -> Optional[str]:
        """Condorcet: tactic that beats all others in pairwise comparison."""
        tactics = list(set(v.tactic for v in votes))

        for tactic in tactics:
            wins = 0
            for other in tactics:
                if tactic == other:
                    continue

                # Count head-to-head
                tactic_votes = sum(1 for v in votes if v.tactic == tactic)
                other_votes = sum(1 for v in votes if v.tactic == other)

                if tactic_votes > other_votes:
                    wins += 1

            if wins == len(tactics) - 1:
                return tactic

        return None

    def _borda(self, votes: List[TacticVote]) -> Optional[str]:
        """Borda count: ranked voting."""
        # Sort votes by confidence
        sorted_votes = sorted(votes, key=lambda v: v.confidence, reverse=True)

        # Assign points: n-1 for first, n-2 for second, etc.
        scores: Dict[str, float] = defaultdict(float)
        n = len(sorted_votes)

        for i, vote in enumerate(sorted_votes):
            points = n - 1 - i
            scores[vote.tactic] += points

        if not scores:
            return None

        return max(scores, key=scores.get)

    def get_vote_statistics(self, votes: List[TacticVote]) -> Dict[str, Any]:
        """Get statistics about votes."""
        if not votes:
            return {
                "total_votes": 0,
                "unique_tactics": 0,
                "tactic_counts": {},
                "avg_confidence": 0.0
            }

        tactic_counts = Counter(v.tactic for v in votes)
        confidences = [v.confidence for v in votes]

        return {
            "total_votes": len(votes),
            "unique_tactics": len(tactic_counts),
            "tactic_counts": dict(tactic_counts),
            "avg_confidence": sum(confidences) / len(confidences),
            "max_confidence": max(confidences),
            "min_confidence": min(confidences),
            "voter_types": Counter(v.voter_type.value for v in votes)
        }


# =============================================================================
# MAKER Engine
# =============================================================================

@dataclass
class LeanMakerRunResult:
    """
    Result of MAKER proof construction.

    Attributes:
        final_state: Final proof state
        tactic_sequence: Sequence of tactics applied
        voting_history: History of voting at each step
        metrics: Performance metrics
        termination_reason: Why the process terminated
        success: Whether proof was completed
    """
    final_state: LeanProofState
    tactic_sequence: List[Tactic]
    voting_history: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    termination_reason: TerminationReason
    success: bool
    timestamp: float = field(default_factory=time.time)

    def get_proof(self) -> LeanProof:
        """Get the final proof."""
        lean_code = "import Mathlib\n\n"
        lean_code += f"theorem maker_proof : {' ∧ '.join(self.final_state.context)} := by\n" if self.final_state.context else "theorem maker_proof := by\n"

        for tactic in self.tactic_sequence:
            lean_code += f"  {tactic}\n"

        return LeanProof(
            theorem_name="maker_proof",
            theorem_statement="Generated by MAKER",
            lean_code=lean_code,
            tactics=self.tactic_sequence
        )

    def get_voting_trace(self) -> List[Dict[str, Any]]:
        """Get the voting trace."""
        return self.voting_history

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "final_state": self.final_state.to_dict(),
            "tactic_sequence": [t.to_dict() for t in self.tactic_sequence],
            "voting_history": self.voting_history,
            "metrics": self.metrics,
            "termination_reason": self.termination_reason.value,
            "success": self.success,
            "timestamp": self.timestamp
        }


class LeanMakerEngine:
    """
    Main MAKER engine for Lean 4 proofs.

    Orchestrates voting-based tactic selection using multiple agents.
    Features:
    - Multiple agents propose tactics
    - Voting selects best tactic
    - Red-flagging filters invalid tactics
    - Step-by-step proof construction
    - Checkpointing and recovery
    - Comprehensive logging and metrics
    """

    def __init__(
        self,
        config: LeanMakerConfig,
        team: Optional['Team'] = None,
        leanaide_client: Optional[LeanAideClient] = None
    ):
        """
        Initialize MAKER engine.

        Args:
            config: MAKER configuration
            team: Optional team of LLM agents
            leanaide_client: Optional LeanAide client
        """
        self.config = config
        self.team = team
        self.leanaide_client = leanaide_client

        # Initialize red flag rules
        self.red_flag_rules = config.red_flag_rules or LeanRedFlagRules()

        # Initialize aggregator
        self.aggregator = LeanAggregator(strategy=config.aggregation_strategy)

        # Initialize voters
        self.voters: List[LeanTacticVoter] = self._initialize_voters()

        # Metrics
        self.metrics = {
            "steps": 0,
            "votes_cast": 0,
            "red_flags": 0,
            "errors": 0,
            "voter_success_rates": {},
            "tactic_usage": Counter(),
            "time_per_step": []
        }

    def _initialize_voters(self) -> List[LeanTacticVoter]:
        """Initialize tactic voters."""
        voters = []

        voter_counts = Counter(vt for vt in self.config.voter_types)

        for voter_type, count in voter_counts.items():
            for i in range(count):
                voter_id = f"{voter_type.value}_{i}"

                if voter_type == VoterType.RANDOM:
                    voter = RandomVoter(
                        voter_id=voter_id,
                        voter_type=voter_type,
                        config=self.config
                    )
                elif voter_type == VoterType.HEURISTIC:
                    voter = HeuristicVoter(
                        voter_id=voter_id,
                        voter_type=voter_type,
                        config=self.config
                    )
                elif voter_type == VoterType.EVOLUTIONARY:
                    voter = EvolutionaryVoter(
                        voter_id=voter_id,
                        voter_type=voter_type,
                        config=self.config
                    )
                elif voter_type == VoterType.MCTS:
                    voter = MCTSVoter(
                        voter_id=voter_id,
                        voter_type=voter_type,
                        config=self.config
                    )
                elif voter_type == VoterType.DIRECT:
                    voter = DirectVoter(
                        voter_id=voter_id,
                        voter_type=voter_type,
                        config=self.config,
                        leanaide_client=self.leanaide_client
                    )
                elif voter_type == VoterType.ENSEMBLE:
                    # Create ensemble of sub-voters
                    sub_voters = [
                        HeuristicVoter(f"{voter_id}_h", VoterType.HEURISTIC, self.config),
                        RandomVoter(f"{voter_id}_r", VoterType.RANDOM, self.config)
                    ]
                    voter = EnsembleVoter(
                        voter_id=voter_id,
                        voter_type=voter_type,
                        config=self.config,
                        sub_voters=sub_voters
                    )
                else:
                    logger.warning(f"Unknown voter type: {voter_type}")
                    continue

                voters.append(voter)

        return voters

    def solve_with_voting(
        self,
        initial_state: LeanProofState,
        checkpoint_path: Optional[str] = None
    ) -> LeanMakerRunResult:
        """
        Solve proof using voting-based MAKER.

        Args:
            initial_state: Initial proof state
            checkpoint_path: Optional path for checkpointing

        Returns:
            LeanMakerRunResult with final state and metrics
        """
        start_time = time.time()
        current_state = initial_state
        voting_history = []
        terminated_reason = TerminationReason.MAX_STEPS_REACHED

        logger.info(f"Starting MAKER proof construction with {len(self.voters)} voters")
        logger.info(f"Initial state: {len(initial_state.goals)} goals")

        try:
            for step in range(self.config.max_steps):
                step_start = time.time()

                # Check if proof is complete
                if current_state.is_complete or len(current_state.goals) == 0:
                    terminated_reason = TerminationReason.PROOF_COMPLETE
                    logger.info(f"Proof complete at step {step}")
                    break

                # Get votes
                votes = self.get_tactic_votes(current_state)

                if not votes:
                    logger.warning(f"No votes collected at step {step}")
                    terminated_reason = TerminationReason.NO_CONSENSUS
                    break

                # Select best tactic
                k_value = self.config.compute_k_value(step, current_state.depth / self.config.max_steps)
                winning_tactic = self.select_best_tactic(votes, k_value)

                if not winning_tactic:
                    logger.warning(f"No winning tactic at step {step}")
                    terminated_reason = TerminationReason.NO_CONSENSUS
                    break

                # Apply tactic with verification
                new_state = self.apply_tactic_with_verification(current_state, winning_tactic)

                if not new_state:
                    # Tactic failed
                    self.metrics["errors"] += 1
                    logger.warning(f"Tactic application failed at step {step}: {winning_tactic}")

                    # Try alternative
                    if votes:
                        # Try second best
                        sorted_votes = sorted(votes, key=lambda v: v.confidence, reverse=True)
                        for vote in sorted_votes[1:3]:  # Try top 2-3
                            new_state = self.apply_tactic_with_verification(current_state, vote.tactic)
                            if new_state:
                                winning_tactic = vote.tactic
                                break

                    if not new_state:
                        terminated_reason = TerminationReason.ALL_TACTICS_FLAGGED
                        break

                # Record voting history
                voting_history.append({
                    "step": step,
                    "state_hash": current_state.hash,
                    "votes": [v.to_dict() for v in votes],
                    "winning_tactic": winning_tactic,
                    "vote_stats": self.aggregator.get_vote_statistics(votes)
                })

                # Update metrics
                self.metrics["steps"] += 1
                self.metrics["tactic_usage"][winning_tactic.split()[0]] += 1
                step_time = time.time() - step_start
                self.metrics["time_per_step"].append(step_time)

                # Update state
                current_state = new_state

                # Checkpoint
                if checkpoint_path and step % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(checkpoint_path, current_state, voting_history)

                logger.info(
                    f"Step {step}: Applied '{winning_tactic}', "
                    f"{len(current_state.goals)} goals remaining, "
                    f"{step_time:.2f}s"
                )

                # Check for time limit
                if time.time() - start_time > self.config.timeout_seconds * self.config.max_steps:
                    terminated_reason = TerminationReason.TIME_LIMIT
                    break

        except Exception as e:
            logger.error(f"MAKER error: {e}", exc_info=True)
            self.metrics["errors"] += 1
            terminated_reason = TerminationReason.ERROR

        # Compile result
        elapsed_time = time.time() - start_time
        self.metrics["total_time"] = elapsed_time
        self.metrics["avg_time_per_step"] = elapsed_time / max(1, self.metrics["steps"])

        result = LeanMakerRunResult(
            final_state=current_state,
            tactic_sequence=current_state.tactic_sequence,
            voting_history=voting_history,
            metrics=self.metrics,
            termination_reason=terminated_reason,
            success=(terminated_reason == TerminationReason.PROOF_COMPLETE)
        )

        logger.info(f"MAKER completed: {terminated_reason.value} in {elapsed_time:.2f}s")

        return result

    def get_tactic_votes(self, state: LeanProofState) -> List[TacticVote]:
        """
        Collect votes from all voters.

        Args:
            state: Current proof state

        Returns:
            List of tactic votes
        """
        votes: List[TacticVote] = []

        # Collect votes from each voter
        if self.config.parallel_voting:
            votes = self._collect_votes_parallel(state)
        else:
            votes = self._collect_votes_sequential(state)

        # Filter by red flags
        valid_votes = []
        for vote in votes:
            is_flagged, reasons = self.red_flag_rules.is_flagged(vote.tactic, state)

            if is_flagged:
                self.metrics["red_flags"] += 1
                logger.debug(f"Vote red-flagged: {vote.tactic} - {reasons}")
            else:
                valid_votes.append(vote)

        self.metrics["votes_cast"] += len(votes)

        return valid_votes

    def _collect_votes_sequential(self, state: LeanProofState) -> List[TacticVote]:
        """Collect votes sequentially."""
        votes = []

        for voter in self.voters:
            try:
                vote = voter.vote(state)
                votes.append(vote)
            except Exception as e:
                logger.error(f"Voter {voter.voter_id} failed: {e}")

        return votes

    def _collect_votes_parallel(self, state: LeanProofState) -> List[TacticVote]:
        """Collect votes in parallel."""
        votes = []

        with ThreadPoolExecutor(max_workers=self.config.max_parallel_voters) as executor:
            futures = {executor.submit(voter.vote, state): voter for voter in self.voters}

            for future in as_completed(futures):
                voter = futures[future]
                try:
                    vote = future.result()
                    votes.append(vote)
                except Exception as e:
                    logger.error(f"Voter {voter.voter_id} failed: {e}")

        return votes

    def select_best_tactic(self, votes: List[TacticVote], k_value: int) -> Optional[str]:
        """
        Select best tactic from votes.

        Args:
            votes: List of votes
            k_value: K value for first-k-ahead

        Returns:
            Best tactic, or None
        """
        # Filter by confidence threshold
        filtered = [v for v in votes if v.confidence >= self.config.confidence_threshold]

        if not filtered:
            filtered = votes  # Use all if none pass threshold

        # Aggregate using configured strategy
        return self.aggregator.aggregate(filtered, k_value=k_value)

    def apply_tactic_with_verification(
        self,
        state: LeanProofState,
        tactic: str
    ) -> Optional[LeanProofState]:
        """
        Apply tactic with verification.

        Args:
            state: Current proof state
            tactic: Tactic to apply

        Returns:
            New proof state, or None if tactic invalid
        """
        # Check red flags first
        is_flagged, reasons = self.red_flag_rules.is_flagged(tactic, state)
        if is_flagged:
            logger.debug(f"Tactic red-flagged: {reasons}")
            return None

        # Apply tactic
        new_state = state.apply_tactic(tactic)

        if not new_state:
            logger.debug(f"Tactic application failed: {tactic}")
            return None

        return new_state

    def _save_checkpoint(self, path: str, state: LeanProofState, history: List[Dict[str, Any]]):
        """Save checkpoint."""
        try:
            checkpoint_data = {
                "state": state.to_dict(),
                "history": history,
                "metrics": self.metrics,
                "timestamp": time.time()
            }

            with open(path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

            logger.debug(f"Checkpoint saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, path: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            logger.info(f"Checkpoint loaded from {path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None


# =============================================================================
# Convenience Functions
# =============================================================================

def solve_lean_proof_with_maker(
    theorem: str,
    context: Optional[List[str]] = None,
    config: Optional[LeanMakerConfig] = None,
    leanaide_url: str = "http://localhost:7654"
) -> LeanMakerRunResult:
    """
    Convenience function to solve a Lean proof using MAKER.

    Args:
        theorem: Theorem statement to prove
        context: Optional list of hypotheses/assumptions
        config: Optional MAKER configuration
        leanaide_url: LeanAide server URL

    Returns:
        LeanMakerRunResult with proof and metrics
    """
    # Create initial state
    initial_state = LeanProofState(
        goals=[theorem],
        context=context or [],
        tactic_sequence=[],
        depth=0
    )

    # Create config
    if config is None:
        config = LeanMakerConfig(
            k_min=2,
            k_max=5,
            max_votes_per_step=20,
            max_steps=100,
            enable_leanaide=True,
            leanaide_url=leanaide_url
        )

    # Initialize LeanAide client if enabled
    leanaide_client = None
    if config.enable_leanaide and LEANAIDE_AVAILABLE:
        try:
            leanaide_client = LeanAideClient()
            leanaide_client.config.base_url = config.leanaide_url
        except Exception as e:
            logger.warning(f"Failed to initialize LeanAide: {e}")

    # Create engine
    engine = LeanMakerEngine(
        config=config,
        leanaide_client=leanaide_client
    )

    # Solve
    result = engine.solve_with_voting(initial_state)

    return result


# =============================================================================
# Example Usage
# =============================================================================

async def example_usage():
    """Example usage of LeanAide MAKER."""

    print("=" * 80)
    print("LeanAide MAKER Example")
    print("=" * 80)

    # Simple theorem
    theorem = "forall (n : Nat), n + 0 = n"

    print(f"\nTheorem: {theorem}\n")

    # Solve with MAKER
    result = solve_lean_proof_with_maker(
        theorem=theorem,
        config=LeanMakerConfig(
            k_min=2,
            k_max=4,
            max_votes_per_step=15,
            max_steps=50,
            voter_types=[
                VoterType.HEURISTIC,
                VoterType.HEURISTIC,
                VoterType.EVOLUTIONARY,
                VoterType.RANDOM,
                VoterType.RANDOM
            ],
            aggregation_strategy=AggregationStrategy.FIRST_K_AHEAD
        )
    )

    # Print results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    print(f"\nSuccess: {result.success}")
    print(f"Termination: {result.termination_reason.value}")
    print(f"Steps: {result.metrics['steps']}")
    print(f"Votes cast: {result.metrics['votes_cast']}")
    print(f"Red flags: {result.metrics['red_flags']}")
    print(f"Time: {result.metrics.get('total_time', 0):.2f}s")

    if result.tactic_sequence:
        print("\n" + "=" * 80)
        print("Proof")
        print("=" * 80)
        proof = result.get_proof()
        print(f"\n{proof.lean_code}")

    if result.voting_history:
        print("\n" + "=" * 80)
        print("Voting Statistics")
        print("=" * 80)

        for step_info in result.voting_history[:5]:  # First 5 steps
            step = step_info["step"]
            stats = step_info["vote_stats"]
            winning = step_info["winning_tactic"]

            print(f"\nStep {step}:")
            print(f"  Winning tactic: {winning}")
            print(f"  Total votes: {stats['total_votes']}")
            print(f"  Unique tactics: {stats['unique_tactics']}")
            print(f"  Avg confidence: {stats['avg_confidence']:.2f}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
