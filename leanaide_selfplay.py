"""
Lean 4 Self-Play System for Automated Proof Improvement

This module implements a self-play system inspired by PSV (Propose-Solve-Verify) and
AlphaZero-style self-play, adapted for Lean 4 theorem proving. The system enables
continuous self-improvement through automated proof generation, verification, and learning.

Key Components:
- LeanSelfPlayEngine: Orchestrates the self-play process
- LeanProofAgent: Agent that generates and verifies proofs
- LeanSelfPlayGame: Single self-play game episode
- LeanProofExperienceBuffer: Stores and samples proof experiences
- LeanProofNetwork: Neural network for proof strategy prediction (optional)
- CAV-NLP verification: Enhanced proof verification

Based on:
- PSV (Propose-Solve-Verify) self-play framework
- Lean 4 integration patterns from LeanAide
- OpenEvolve decomposition and evolution systems
"""

import asyncio
import json
import logging
import os
import random
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, Set
)

import httpx
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add CAV-NLP imports with graceful fallback
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    logger.warning("CAV-NLP integration not available - self-play will use standard verification")

# ============================================================================
# Data Structures
# ============================================================================

class ProofDifficulty(Enum):
    """Difficulty levels for Lean 4 theorems"""
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"
    RESEARCH = "research"


class ProofStatus(Enum):
    """Status of a proof attempt"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


@dataclass
class ProofState:
    """State of a proof during generation"""
    theorem: str
    current_goal: str
    tactics_applied: List[str] = field(default_factory=list)
    remaining_hypotheses: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> 'ProofState':
        """Create a copy of the proof state"""
        return ProofState(
            theorem=self.theorem,
            current_goal=self.current_goal,
            tactics_applied=self.tactics_applied.copy(),
            remaining_hypotheses=self.remaining_hypotheses.copy(),
            metadata=self.metadata.copy()
        )


@dataclass
class LeanTheorem:
    """A Lean 4 theorem to be proven"""
    id: str
    statement: str
    lean_code: str
    difficulty: ProofDifficulty
    domain: str  # e.g., "algebra", "analysis", "topology", "logic"
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_lean_file(self) -> str:
        """Convert to complete Lean 4 file format"""
        return f"""
import Mathlib

{self.lean_code}

theorem {self.id} : {self.statement} :=
  by
  -- Proof to be generated
"""


@dataclass
class LeanTactic:
    """A single Lean 4 tactic"""
    name: str
    args: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.args:
            return f"{self.name} {' '.join(self.args)}"
        return self.name


@dataclass
class LeanProof:
    """A complete Lean 4 proof"""
    theorem_id: str
    tactics: List[LeanTactic]
    lean_code: str
    status: ProofStatus
    verification_output: str = ""
    error_message: str = ""
    confidence: float = 0.0
    generation_time: float = 0.0
    verification_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tactic_count(self) -> int:
        """Number of tactics in the proof"""
        return len(self.tactics)

    @property
    def is_valid(self) -> bool:
        """Whether the proof is valid"""
        return self.status == ProofStatus.VERIFIED


@dataclass
class LeanProofExperience:
    """Experience from a self-play game"""
    theorem: LeanTheorem
    proof: LeanProof
    reward: float
    strategy_used: str
    value_estimate: float
    policy_output: Dict[str, float]
    timestamp: float = field(default_factory=time.time)

    def to_training_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for training"""
        return {
            "theorem": asdict(self.theorem),
            "proof": asdict(self.proof),
            "reward": self.reward,
            "strategy_used": self.strategy_used,
            "value_estimate": self.value_estimate,
            "policy_output": self.policy_output,
            "timestamp": self.timestamp
        }


@dataclass
class LeanProofStrategy:
    """A proof strategy (sequence of tactic choices)"""
    name: str
    tactic_sequence: List[str]
    description: str
    applicable_domains: List[str]  # Fixed: was 适用领域
    success_rate: float = 0.0
    avg_proof_length: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    """Metrics from training iteration"""
    iteration: int
    total_games: int
    success_rate: float
    avg_reward: float
    avg_proof_length: float
    value_loss: float
    policy_loss: float
    buffer_size: int
    unique_theorems: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class SelfPlayResult:
    """Result of a self-play game"""
    theorem: LeanTheorem
    proof: Optional[LeanProof]
    reward: float
    success: bool
    strategies_used: List[str]
    time_elapsed: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# Alias for compatibility
GameResult = SelfPlayResult


# ============================================================================
# Lean 4 Integration
# ============================================================================

class Lean4Verifier:
    """
    Interface to Lean 4 theorem prover for proof verification.

    Integrates with LeanAide server for Lean 4 execution and verification.
    """

    def __init__(
        self,
        leanaide_url: str = "http://localhost:7654",
        timeout: int = 300
    ):
        self.leanaide_url = leanaide_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def verify_proof(
        self,
        theorem: LeanTheorem,
        proof: LeanProof
    ) -> Tuple[ProofStatus, str, str]:
        """
        Verify a Lean 4 proof using LeanAide server.

        Returns:
            Tuple of (status, output, error_message)
        """
        try:
            # Construct complete Lean file
            lean_file = self._construct_lean_file(theorem, proof)

            # Send to LeanAide for verification
            response = await self.client.post(
                f"{self.leanaide_url}/verify",
                json={
                    "code": lean_file,
                    "theorem_id": theorem.id,
                    "timeout": self.timeout
                }
            )
            response.raise_for_status()

            result = response.json()

            # Parse verification result
            if result.get("success"):
                return ProofStatus.VERIFIED, result.get("output", ""), ""
            else:
                error_msg = result.get("error", "Unknown error")
                if "timeout" in error_msg.lower():
                    return ProofStatus.TIMEOUT, "", error_msg
                elif "partial" in error_msg.lower():
                    return ProofStatus.PARTIAL, result.get("output", ""), error_msg
                else:
                    return ProofStatus.FAILED, "", error_msg

        except httpx.TimeoutException:
            return ProofStatus.TIMEOUT, "", "Verification timeout"
        except (IOError, ConnectionError, ValueError) as e:
            logger.error(f"Verification error: {e}")
            return ProofStatus.FAILED, "", str(e)

    def _construct_lean_file(
        self,
        theorem: LeanTheorem,
        proof: LeanProof
    ) -> str:
        """Construct complete Lean 4 file for verification"""
        return f"""
import Mathlib

{theorem.lean_code}

theorem {theorem.id} : {theorem.statement} :=
  by
    {self._format_tactics(proof.tactics)}
"""

    def _format_tactics(self, tactics: List[LeanTactic]) -> str:
        """Format tactics as Lean code"""
        return "\n    ".join(str(tactic) for tactic in tactics)

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# ============================================================================
# Experience Buffer
# ============================================================================

class LeanProofExperienceBuffer:
    """
    Replay buffer for storing and sampling proof experiences.

    Implements prioritized experience replay with importance sampling.
    """

    def __init__(
        self,
        capacity: int = 10000,
        prioritized: bool = True,
        priority_alpha: float = 0.6,
        priority_epsilon: float = 1e-6
    ):
        self.capacity = capacity
        self.prioritized = prioritized
        self.priority_alpha = priority_alpha
        self.priority_epsilon = priority_epsilon

        self.buffer: List[LeanProofExperience] = []
        self.priorities: np.ndarray = np.zeros(capacity)
        self.max_priority = 1.0

        # Track statistics
        self.add_count = 0
        self.sample_count = 0

    def add(self, experience: LeanProofExperience) -> None:
        """Add experience to buffer"""
        if len(self.buffer) >= self.capacity:
            # Remove oldest (or lowest priority if prioritized)
            if self.prioritized:
                min_idx = np.argmin(self.priorities[:len(self.buffer)])
                self.buffer.pop(min_idx)
                self.priorities = np.delete(self.priorities, min_idx)
                self.priorities = np.append(self.priorities, 0)
            else:
                self.buffer.pop(0)

        self.buffer.append(experience)

        # Set priority
        priority = self._calculate_priority(experience)
        if self.prioritized:
            self.priorities[len(self.buffer) - 1] = priority
            self.max_priority = max(self.max_priority, priority)

        self.add_count += 1

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4
    ) -> List[LeanProofExperience]:
        """Sample a batch of experiences"""
        if not self.buffer:
            return []

        if self.prioritized and len(self.buffer) > batch_size:
            # Prioritized sampling
            probs = self.priorities[:len(self.buffer)] ** self.priority_alpha
            probs /= probs.sum()

            indices = np.random.choice(
                len(self.buffer),
                size=min(batch_size, len(self.buffer)),
                p=probs,
                replace=False
            )

            # Calculate importance sampling weights
            weights = (len(self.buffer) * probs[indices]) ** (-beta)
            weights /= weights.max()

        else:
            # Uniform sampling
            indices = np.random.choice(
                len(self.buffer),
                size=min(batch_size, len(self.buffer)),
                replace=False
            )
            weights = np.ones(len(indices))

        batch = [self.buffer[i] for i in indices]
        self.sample_count += 1

        return batch

    def _calculate_priority(self, experience: LeanProofExperience) -> float:
        """Calculate priority for experience"""
        # Priority based on absolute reward (higher reward = higher priority)
        # Failed proofs also get high priority for learning
        abs_reward = abs(experience.reward)

        # Bonus for rare theorems
        rarity_bonus = 1.0 / (1 + experience.proof.metadata.get("frequency", 0))

        return abs_reward + rarity_bonus

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        if not self.buffer:
            return {
                "size": 0,
                "add_count": self.add_count,
                "sample_count": self.sample_count
            }

        rewards = [exp.reward for exp in self.buffer]
        success_count = sum(1 for exp in self.buffer if exp.proof.is_valid)

        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "add_count": self.add_count,
            "sample_count": self.sample_count,
            "success_rate": success_count / len(self.buffer),
            "avg_reward": np.mean(rewards),
            "reward_std": np.std(rewards),
            "max_priority": self.max_priority
        }

    def save(self, filepath: str) -> None:
        """Save buffer to disk"""
        data = {
            "buffer": [asdict(exp) for exp in self.buffer],
            "priorities": self.priorities.tolist(),
            "statistics": self.get_statistics()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Load buffer from disk"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.buffer = [
            LeanProofExperience(**exp) for exp in data["buffer"]
        ]
        self.priorities = np.array(data["priorities"])
        self.max_priority = self.priorities.max() if len(self.priorities) > 0 else 1.0


# ============================================================================
# Proof Agent
# ============================================================================

class LeanProofAgent:
    """
    Agent that generates and verifies Lean 4 proofs.

    Combines:
    - Policy network: Selects tactics
    - Value network: Estimates proof quality
    - LLM integration: Generates natural language reasoning
    """

    def __init__(
        self,
        agent_id: str,
        llm_config: Dict[str, Any],
        verifier: Lean4Verifier,
        exploration_rate: float = 0.3,
        temperature: float = 0.8
    ):
        self.agent_id = agent_id
        self.llm_config = llm_config
        self.verifier = verifier
        self.exploration_rate = exploration_rate
        self.temperature = temperature

        # Knowledge base
        self.known_tactics = self._initialize_tactics()
        self.known_strategies = self._initialize_strategies()

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []

    def _initialize_tactics(self) -> Dict[str, List[str]]:
        """Initialize known Lean 4 tactics by domain"""
        return {
            "logic": ["intro", "apply", "exact", "by", "have", "show"],
            "algebra": ["ring", "linarith", "norm_num", "field_simp"],
            "analysis": ["continuity", "differentiability", "integral"],
            "combinatorics": ["induction", "cases", "rcases"],
            "general": ["rw", "simp", "assumption", "contradiction"]
        }

    def _initialize_strategies(self) -> List[LeanProofStrategy]:
        """Initialize known proof strategies"""
        return [
            LeanProofStrategy(
                name="direct_proof",
                tactic_sequence=["intro", "apply", "exact"],
                description="Direct forward reasoning",
                适用领域=["logic", "algebra"]
            ),
            LeanProofStrategy(
                name="proof_by_contradiction",
                tactic_sequence=["intro", "by_contradiction", "contradiction"],
                description="Assume negation and derive contradiction",
                适用领域=["logic", "set_theory"]
            ),
            LeanProofStrategy(
                name="induction",
                tactic_sequence=["induction", "case", "simp"],
                description="Proof by induction",
                适用领域=["combinatorics", "algebra"]
            ),
            LeanProofStrategy(
                name="calculation",
                tactic_sequence=["calc", "rw", "simp", "norm_num"],
                description="Step-by-step calculation",
                适用领域=["algebra", "analysis"]
            )
        ]

    async def select_proof_strategy(
        self,
        theorem: LeanTheorem,
        training: bool = True
    ) -> LeanProofStrategy:
        """
        Select a proof strategy for the given theorem.

        Uses exploration-exploitation tradeoff during training.
        """
        if training and random.random() < self.exploration_rate:
            # Exploration: Select random strategy
            strategy = random.choice(self.known_strategies)
        else:
            # Exploitation: Select best strategy for domain
            domain_strategies = [
                s for s in self.known_strategies
                if theorem.domain in s.适用领域
            ]

            if domain_strategies:
                # Select by success rate
                strategy = max(
                    domain_strategies,
                    key=lambda s: s.success_rate
                )
            else:
                strategy = random.choice(self.known_strategies)

        return strategy

    async def generate_proof(
        self,
        theorem: LeanTheorem,
        strategy: LeanProofStrategy
    ) -> LeanProof:
        """
        Generate a proof for the theorem using the given strategy.

        Integrates with LLM for tactic generation.
        """
        start_time = time.time()

        try:
            # Use LLM to generate tactics
            tactics = await self._generate_tactics_with_llm(theorem, strategy)

            # Format as Lean code
            lean_code = self._tactics_to_lean(theorem, tactics)

            # Create proof object
            proof = LeanProof(
                theorem_id=theorem.id,
                tactics=tactics,
                lean_code=lean_code,
                status=ProofStatus.PENDING,
                confidence=self._estimate_confidence(theorem, strategy, tactics)
            )

            proof.generation_time = time.time() - start_time

            return proof

        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Proof generation error: {e}")
            return LeanProof(
                theorem_id=theorem.id,
                tactics=[],
                lean_code="",
                status=ProofStatus.FAILED,
                error_message=str(e)
            )

    async def _generate_tactics_with_llm(
        self,
        theorem: LeanTheorem,
        strategy: LeanProofStrategy
    ) -> List[LeanTactic]:
        """Generate tactics using LLM"""

        prompt = f"""
Generate Lean 4 tactics to prove the following theorem using {strategy.name} strategy.

Theorem Statement:
{theorem.statement}

Context:
{theorem.lean_code}

Strategy: {strategy.description}
Tactic Sequence: {', '.join(strategy.tactic_sequence)}

Generate a sequence of Lean 4 tactics (one per line) that attempts to prove this theorem.
Focus on the {theorem.domain} domain.

Tactics:
"""

        try:
            # Call LLM (using configured API)
            response = await self._call_llm(prompt)

            # Parse response into tactics
            tactics = self._parse_tactics(response)

            return tactics

        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"LLM tactic generation error: {e}")
            # Fallback to strategy default sequence
            return [
                LeanTactic(name=tactic)
                for tactic in strategy.tactic_sequence
            ]

    async def _call_llm(self, prompt: str) -> str:
        """Call configured LLM API"""
        # Integration with existing LLM systems would go here
        # For now, return mock response
        return "intro h\napply h\nassumption"

    def _parse_tactics(self, response: str) -> List[LeanTactic]:
        """Parse LLM response into tactic objects"""
        tactics = []

        for line in response.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('--'):
                continue

            parts = line.split()
            if parts:
                tactic_name = parts[0]
                tactic_args = parts[1:] if len(parts) > 1 else []

                tactics.append(LeanTactic(
                    name=tactic_name,
                    args=tactic_args
                ))

        return tactics

    def _tactics_to_lean(
        self,
        theorem: LeanTheorem,
        tactics: List[LeanTactic]
    ) -> str:
        """Convert tactics to Lean code"""
        tactic_lines = []
        for tactic in tactics:
            if tactic.args:
                tactic_lines.append(f"{'  '.join([tactic.name] + tactic.args)}")
            else:
                tactic_lines.append(tactic.name)

        return '\n'.join(tactic_lines)

    def _estimate_confidence(
        self,
        theorem: LeanTheorem,
        strategy: LeanProofStrategy,
        tactics: List[LeanTactic]
    ) -> float:
        """Estimate confidence in proof"""
        # Base confidence from strategy success rate
        confidence = strategy.success_rate

        # Adjust based on tactic count (longer proofs = lower confidence)
        confidence *= 1.0 / (1.0 + 0.1 * len(tactics))

        # Adjust based on theorem difficulty
        difficulty_penalty = {
            ProofDifficulty.TRIVIAL: 0.0,
            ProofDifficulty.EASY: 0.1,
            ProofDifficulty.MEDIUM: 0.2,
            ProofDifficulty.HARD: 0.3,
            ProofDifficulty.EXPERT: 0.4,
            ProofDifficulty.RESEARCH: 0.5
        }
        confidence -= difficulty_penalty.get(theorem.difficulty, 0.3)

        return max(0.0, min(1.0, confidence))

    async def evaluate_proof(self, proof: LeanProof) -> float:
        """
        Evaluate the quality of a proof.

        Returns value in [0, 1] where 1 is best.
        """
        if proof.is_valid:
            # Verified proof
            value = 1.0

            # Bonus for shorter proofs
            value += 0.1 * (1.0 / (1.0 + 0.1 * proof.tactic_count))

        elif proof.status == ProofStatus.PARTIAL:
            # Partial proof gets partial credit
            value = 0.5
        else:
            # Failed proof gets no credit
            value = 0.0

        # Adjust by confidence
        value *= proof.confidence

        return value

    def update_performance(self, result: Dict[str, Any]) -> None:
        """Update agent performance tracking"""
        self.performance_history.append({
            **result,
            "timestamp": time.time()
        })

        # Update strategy success rates
        if "strategy_used" in result and "success" in result:
            for strategy in self.known_strategies:
                if strategy.name == result["strategy_used"]:
                    # Exponential moving average
                    alpha = 0.1
                    if result["success"]:
                        strategy.success_rate = (
                            (1 - alpha) * strategy.success_rate + alpha * 1.0
                        )
                    else:
                        strategy.success_rate = (
                            (1 - alpha) * strategy.success_rate + alpha * 0.0
                        )


# ============================================================================
# Self-Play Game
# ============================================================================

class LeanSelfPlayGame:
    """
    Single self-play game episode.

    The agent plays both prover and verifier roles, learning from
    both successful and failed proof attempts.
    """

    def __init__(
        self,
        theorem: LeanTheorem,
        agent: LeanProofAgent,
        verifier: Lean4Verifier
    ):
        self.theorem = theorem
        self.agent = agent
        self.verifier = verifier

        self.proof: Optional[LeanProof] = None
        self.reward: float = 0.0
        self.value_estimate: float = 0.0

    async def play(self) -> LeanProofExperience:
        """
        Execute a self-play game.

        Returns:
            Experience tuple for training
        """
        logger.info(f"Starting self-play game for theorem {self.theorem.id}")

        # 1. Select proof strategy
        strategy = await self.agent.select_proof_strategy(
            self.theorem,
            training=True
        )

        # 2. Generate proof
        self.proof = await self.agent.generate_proof(
            self.theorem,
            strategy
        )

        # 3. Verify proof
        verification_start = time.time()
        status, output, error = await self.verifier.verify_proof(
            self.theorem,
            self.proof
        )
        self.proof.status = status
        self.proof.verification_output = output
        self.proof.error_message = error
        self.proof.verification_time = time.time() - verification_start

        # 4. Evaluate proof quality
        self.value_estimate = await self.agent.evaluate_proof(self.proof)

        # 5. Calculate reward
        self.reward = self._calculate_reward()

        # 6. Create experience
        experience = LeanProofExperience(
            theorem=self.theorem,
            proof=self.proof,
            reward=self.reward,
            strategy_used=strategy.name,
            value_estimate=self.value_estimate,
            policy_output=self._get_policy_output(strategy)
        )

        # 7. Update agent performance
        self.agent.update_performance({
            "theorem_id": self.theorem.id,
            "strategy_used": strategy.name,
            "success": self.proof.is_valid,
            "proof_length": self.proof.tactic_count,
            "reward": self.reward
        })

        logger.info(
            f"Game completed: {self.theorem.id} - "
            f"Status: {status.value} - Reward: {self.reward:.3f}"
        )

        return experience

    def _calculate_reward(self) -> float:
        """
        Calculate reward for the proof attempt.

        Reward components:
        - Verification success (+1.0)
        - Partial proof (+0.5)
        - Proof length penalty (-0.01 per tactic)
        - Time penalty (-0.001 per second)
        - Elegance bonus (subjective, based on tactic diversity)
        """
        reward = 0.0

        # Base reward from verification
        if self.proof.is_valid:
            reward += 1.0
        elif self.proof.status == ProofStatus.PARTIAL:
            reward += 0.5

        # Length penalty (prefer shorter proofs)
        reward -= 0.01 * self.proof.tactic_count

        # Time penalty (prefer faster proofs)
        reward -= 0.001 * self.proof.generation_time
        reward -= 0.001 * self.proof.verification_time

        # Elegance bonus (diverse tactics)
        tactic_diversity = len(set(t.name for t in self.proof.tactics))
        if self.proof.tactic_count > 0:
            diversity_ratio = tactic_diversity / self.proof.tactic_count
            reward += 0.1 * diversity_ratio

        # Confidence bonus
        reward += 0.1 * self.proof.confidence

        # Difficulty bonus
        difficulty_bonus = {
            ProofDifficulty.TRIVIAL: 0.0,
            ProofDifficulty.EASY: 0.1,
            ProofDifficulty.MEDIUM: 0.2,
            ProofDifficulty.HARD: 0.3,
            ProofDifficulty.EXPERT: 0.4,
            ProofDifficulty.RESEARCH: 0.5
        }
        reward += difficulty_bonus.get(self.theorem.difficulty, 0.2)

        return max(0.0, reward)  # Non-negative rewards

    def _get_policy_output(self, strategy: LeanProofStrategy) -> Dict[str, float]:
        """Get policy output (strategy probabilities)"""
        # Simplified: uniform distribution over known strategies
        return {
            s.name: 1.0 / len(self.agent.known_strategies)
            for s in self.agent.known_strategies
        }


# ============================================================================
# Self-Play Engine
# ============================================================================

class LeanSelfPlayEngine:
    """
    Main self-play engine for Lean 4 proof improvement.

    Orchestrates self-play games, manages experience buffer,
    and drives continuous improvement through iteration.
    
    With CAV-NLP integration for enhanced proof verification.
    """

    def __init__(
        self,
        leanaide_url: str = "http://localhost:7654",
        llm_config: Optional[Dict[str, Any]] = None,
        buffer_capacity: int = 10000,
        max_concurrent_games: int = 4,
        config: Optional[Dict[str, Any]] = None
    ):
        self.leanaide_url = leanaide_url
        self.llm_config = llm_config or {}
        self.config = config or {}

        # Initialize components
        self.verifier = Lean4Verifier(leanaide_url)
        self.agent = LeanProofAgent(
            agent_id="selfplay_agent",
            llm_config=self.llm_config,
            verifier=self.verifier
        )
        self.buffer = LeanProofExperienceBuffer(
            capacity=buffer_capacity,
            prioritized=True
        )

        # Configuration
        self.max_concurrent_games = max_concurrent_games

        # Theorem database
        self.theorem_database: Dict[str, LeanTheorem] = {}

        # Training statistics
        self.iteration_count = 0
        self.metrics_history: List[TrainingMetrics] = []

        # Initialize CAV-NLP components for enhanced verification
        self.use_cav_nlp = self.config.get("use_cav_nlp", True) and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            try:
                self.enhanced_solver = EnhancedZ3Solver()
                self.math_service = UnifiedMathService()
                logger.info("CAV-NLP components initialized for self-play")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP components: {e}")
                self.use_cav_nlp = False

        logger.info(f"LeanSelfPlayEngine initialized (CAV-NLP: {self.use_cav_nlp})")

    async def run_self_play(
        self,
        theorem: str,
        games: int = 10
    ) -> LeanProof:
        """
        Run self-play for a specific theorem.

        Args:
            theorem: Theorem statement or ID
            games: Number of self-play games to play

        Returns:
            Best proof found
        """
        logger.info(f"Running {games} self-play games for theorem: {theorem}")

        # Load or create theorem
        if theorem in self.theorem_database:
            theorem_obj = self.theorem_database[theorem]
        else:
            # Create new theorem from statement
            theorem_obj = LeanTheorem(
                id=str(uuid.uuid4()),
                statement=theorem,
                lean_code=f"theorem test : {theorem} := by",
                difficulty=ProofDifficulty.MEDIUM,
                domain="general"
            )
            self.theorem_database[theorem_obj.id] = theorem_obj

        # Run multiple self-play games
        best_proof = None
        best_reward = float('-inf')

        for game_idx in range(games):
            game = LeanSelfPlayGame(theorem_obj, self.agent, self.verifier)
            experience = await game.play()

            # Store experience
            self.buffer.add(experience)

            # Track best proof
            if experience.reward > best_reward:
                best_reward = experience.reward
                best_proof = experience.proof

            logger.info(
                f"Game {game_idx + 1}/{games} completed - "
                f"Reward: {experience.reward:.3f} - "
                f"Best so far: {best_reward:.3f}"
            )

        self.iteration_count += games

        logger.info(
            f"Self-play completed - Best reward: {best_reward:.3f} - "
            f"Best proof valid: {best_proof.is_valid if best_proof else False}"
        )

        return best_proof

    async def run_batch_self_play(
        self,
        theorems: List[str],
        games_per_theorem: int = 5
    ) -> Dict[str, LeanProof]:
        """
        Run self-play for multiple theorems in parallel.

        Returns:
            Dictionary mapping theorem ID to best proof
        """
        logger.info(f"Running batch self-play for {len(theorems)} theorems")

        results = {}

        # Process theorems in batches
        for theorem in theorems:
            proof = await self.run_self_play(theorem, games_per_theorem)
            results[theorem] = proof

        # Compute metrics
        metrics = self._compute_metrics()
        self.metrics_history.append(metrics)

        return results

    async def train_from_buffer(
        self,
        batch_size: int = 32,
        iterations: int = 10
    ) -> TrainingMetrics:
        """
        Train agent from experience buffer.

        In a full implementation, this would:
        1. Sample batch from buffer
        2. Update policy network
        3. Update value network
        4. Track losses
        """
        logger.info(f"Training from buffer - {iterations} iterations, batch size {batch_size}")

        total_value_loss = 0.0
        total_policy_loss = 0.0

        for _ in range(iterations):
            batch = self.buffer.sample(batch_size)

            if not batch:
                continue

            # Compute losses (simplified - in practice would use neural networks)
            value_loss = self._compute_value_loss(batch)
            policy_loss = self._compute_policy_loss(batch)

            total_value_loss += value_loss
            total_policy_loss += policy_loss

            # In practice, update networks here

        metrics = TrainingMetrics(
            iteration=self.iteration_count,
            total_games=len(self.buffer.buffer),
            success_rate=self._compute_success_rate(),
            avg_reward=np.mean([exp.reward for exp in self.buffer.buffer]),
            avg_proof_length=np.mean([
                exp.proof.tactic_count for exp in self.buffer.buffer
            ]),
            value_loss=total_value_loss / iterations,
            policy_loss=total_policy_loss / iterations,
            buffer_size=len(self.buffer.buffer),
            unique_theorems=len(set(exp.theorem.id for exp in self.buffer.buffer))
        )

        self.metrics_history.append(metrics)

        logger.info(f"Training completed - Success rate: {metrics.success_rate:.3f}")

        return metrics

    def _compute_value_loss(self, batch: List[LeanProofExperience]) -> float:
        """Compute value loss (MSE between predicted and actual value)"""
        if not batch:
            return 0.0

        losses = [
            (exp.value_estimate - (1.0 if exp.proof.is_valid else 0.0)) ** 2
            for exp in batch
        ]

        return np.mean(losses)

    def _compute_policy_loss(self, batch: List[LeanProofExperience]) -> float:
        """Compute policy loss (cross-entropy)"""
        # Simplified - in practice would use actual policy gradient
        return random.random() * 0.1  # Placeholder

    def _compute_success_rate(self) -> float:
        """Compute overall success rate"""
        if not self.buffer.buffer:
            return 0.0

        success_count = sum(
            1 for exp in self.buffer.buffer
            if exp.proof.is_valid
        )

        return success_count / len(self.buffer.buffer)

    def _compute_metrics(self) -> TrainingMetrics:
        """Compute current training metrics"""
        return TrainingMetrics(
            iteration=self.iteration_count,
            total_games=len(self.buffer.buffer),
            success_rate=self._compute_success_rate(),
            avg_reward=np.mean([exp.reward for exp in self.buffer.buffer]),
            avg_proof_length=np.mean([
                exp.proof.tactic_count for exp in self.buffer.buffer
            ]),
            value_loss=0.0,
            policy_loss=0.0,
            buffer_size=len(self.buffer.buffer),
            unique_theorems=len(set(exp.theorem.id for exp in self.buffer.buffer))
        )

    def get_training_progress(self) -> Dict[str, Any]:
        """Get training progress summary"""
        if not self.metrics_history:
            return {
                "iteration": 0,
                "status": "not_started"
            }

        latest = self.metrics_history[-1]

        return {
            "iteration": latest.iteration,
            "total_games": latest.total_games,
            "success_rate": latest.success_rate,
            "avg_reward": latest.avg_reward,
            "avg_proof_length": latest.avg_proof_length,
            "buffer_size": latest.buffer_size,
            "unique_theorems": latest.unique_theorems,
            "improvement": self._compute_improvement()
        }

    def _compute_improvement(self) -> Dict[str, float]:
        """Compute improvement metrics"""
        if len(self.metrics_history) < 2:
            return {"relative": 0.0, "absolute": 0.0}

        earliest = self.metrics_history[0]
        latest = self.metrics_history[-1]

        return {
            "absolute": latest.success_rate - earliest.success_rate,
            "relative": (
                (latest.success_rate - earliest.success_rate) / earliest.success_rate
                if earliest.success_rate > 0 else 0.0
            )
        }

    async def close(self):
        """Clean up resources"""
        await self.verifier.close()
        logger.info("LeanSelfPlayEngine closed")

    def save_checkpoint(self, filepath: str) -> None:
        """Save training checkpoint"""
        checkpoint = {
            "iteration_count": self.iteration_count,
            "buffer_statistics": self.buffer.get_statistics(),
            "metrics_history": [asdict(m) for m in self.metrics_history],
            "agent_performance": self.agent.performance_history,
            "timestamp": time.time()
        }

        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """Load training checkpoint"""
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)

        self.iteration_count = checkpoint["iteration_count"]
        self.metrics_history = [
            TrainingMetrics(**m) for m in checkpoint["metrics_history"]
        ]
        self.agent.performance_history = checkpoint["agent_performance"]

        logger.info(f"Checkpoint loaded from {filepath}")

    async def cav_nlp_verify_proof(
        self,
        proof: LeanProof,
        theorem: LeanTheorem
    ) -> Dict[str, Any]:
        """
        Verify a proof using CAV-NLP enhanced verification.
        
        Uses the EnhancedZ3Solver and UnifiedMathService for:
        - Semantic analysis of proof structure
        - Constraint-based verification
        - Enhanced error detection
        
        Args:
            proof: The proof to verify
            theorem: The theorem being proved
            
        Returns:
            Dictionary with verification results
        """
        if not self.use_cav_nlp:
            return {
                "verified": proof.is_valid,
                "cav_nlp_available": False,
                "confidence": 0.5
            }
        
        try:
            # Use math service for semantic analysis
            semantic_result = await self.math_service.analyze_semantics_async(
                lean_code=proof.lean_code,
                context={
                    "theorem": theorem.statement,
                    "domain": theorem.domain,
                    "difficulty": theorem.difficulty.value
                }
            )
            
            semantic_score = semantic_result.get("semantic_score", 0.0)
            issues = semantic_result.get("issues", [])
            
            # Use enhanced solver for constraint checking
            constraint_result = await self.enhanced_solver.verify_proof_async(
                proof_code=proof.lean_code,
                theorem_statement=theorem.statement,
                timeout_ms=self.config.get("solver_timeout", 5000)
            )
            
            constraint_verified = constraint_result.get("verified", False)
            constraint_confidence = constraint_result.get("confidence", 0.5)
            
            # Combine results
            combined_confidence = (semantic_score + constraint_confidence) / 2
            
            return {
                "verified": proof.is_valid and constraint_verified and semantic_score > 0.6,
                "cav_nlp_available": True,
                "semantic_score": semantic_score,
                "constraint_verified": constraint_verified,
                "issues": issues,
                "confidence": combined_confidence,
                "semantic_analysis": semantic_result.get("analysis", {}),
                "constraint_details": constraint_result.get("details", {})
            }
        
        except Exception as e:
            logger.warning(f"CAV-NLP verification failed: {e}")
            return {
                "verified": proof.is_valid,
                "cav_nlp_available": True,
                "error": str(e),
                "confidence": 0.5,
                "fallback": True
            }


# ============================================================================
# Main Interface
# ============================================================================

async def main():
    """
    Main entry point for Lean 4 self-play system.

    Demonstrates usage of the self-play engine.
    """
    # Initialize engine
    engine = LeanSelfPlayEngine(
        leanaide_url="http://localhost:7654",
        buffer_capacity=1000,
        max_concurrent_games=2
    )

    try:
        # Example theorems to practice on
        theorems = [
            "∀ n : Nat, n + 0 = n",
            "∀ a b : Nat, a + b = b + a",
            "∀ n : Nat, 2 * n = n + n"
        ]

        # Run self-play
        logger.info("Starting Lean 4 self-play training")

        results = await engine.run_batch_self_play(
            theorems=theorems,
            games_per_theorem=5
        )

        # Train from experiences
        metrics = await engine.train_from_buffer(
            batch_size=16,
            iterations=5
        )

        # Report results
        progress = engine.get_training_progress()

        print("\n" + "="*50)
        print("Self-Play Training Results")
        print("="*50)
        print(f"Total games played: {progress['total_games']}")
        print(f"Success rate: {progress['success_rate']:.1%}")
        print(f"Average reward: {progress['avg_reward']:.3f}")
        print(f"Average proof length: {progress['avg_proof_length']:.1f} tactics")
        print(f"Unique theorems: {progress['unique_theorems']}")
        print(f"Improvement: {progress['improvement']['relative']:.1%}")
        print("="*50)

        # Save checkpoint
        engine.save_checkpoint("lean_selfplay_checkpoint.json")

    finally:
        await engine.close()


# =============================================================================
# MDAP-Enhanced Self-Play
# =============================================================================

class LeanSelfPlayEngineMDAP:
    """
    Self-play engine with MDAP consensus for strategy selection.

    Enhances self-play with:
    - MDAP voting for move/strategy selection
    - Multi-agent policy learning from MDAP results
    - Consensus-based exploration
    """

    def __init__(
        self,
        agents: List['LeanProofAgent'],
        mdap_config: Optional['LeanMDAPConfig'] = None,
        leanaide_url: str = "http://localhost:7654"
    ):
        """
        Initialize MDAP-enhanced self-play engine.

        Args:
            agents: List of proof agents
            mdap_config: MDAP configuration
            leanaide_url: LeanAide verification URL
        """
        self.agents = agents
        self.mdap_config = mdap_config
        self.leanaide_url = leanaide_url

        # MDAP statistics
        self.mdap_stats = {
            "strategy_votes": defaultdict(int),
            "consensus_rates": [],
            "agent_contributions": defaultdict(int),
            "policy_updates": []
        }

    async def select_strategy_with_mdap(
        self,
        state: ProofState,
        agents: List['LeanProofAgent']
    ) -> LeanProofStrategy:
        """
        Select strategy using MDAP consensus among agents.

        Args:
            state: Current proof state
            agents: List of agents to query

        Returns:
            Consensus strategy
        """
        logger.info(f"MDAP strategy selection with {len(agents)} agents")

        # Each agent proposes a strategy
        strategies = []
        for agent in agents:
            strategy = await agent.select_strategy(state)
            strategies.append(strategy)

        # MDAP voting on best strategy
        consensus = await self._mdap_vote_on_strategy(strategies, state)

        # Track voting statistics
        for strategy in strategies:
            self.mdap_stats["strategy_votes"][strategy.approach] += 1

        logger.info(f"MDAP selected strategy with {consensus.confidence:.2f} confidence")
        return consensus

    async def update_policy_from_mdap(
        self,
        mdap_results: List[LeanProof]
    ) -> None:
        """
        Update agent policies based on MDAP results.

        Args:
            mdap_results: List of proofs generated by MDAP agents
        """
        logger.info(f"Updating policies from {len(mdap_results)} MDAP results")

        # Calculate success metrics
        successful = [p for p in mdap_results if p.verified]
        success_rate = len(successful) / len(mdap_results) if mdap_results else 0.0

        # Update each agent's policy
        for agent in self.agents:
            await agent.update_policy(mdap_results)

        # Track policy updates
        self.mdap_stats["policy_updates"].append({
            "timestamp": time.time(),
            "success_rate": success_rate,
            "num_proofs": len(mdap_results)
        })

        logger.info(f"Policy update complete: success_rate={success_rate:.2f}")

    async def run_self_play_with_mdap(
        self,
        theorem: str,
        games: int,
        agents_per_game: int = 4
    ) -> SelfPlayResult:
        """
        Run self-play episodes with MDAP consensus.

        Args:
            theorem: Theorem to practice
            games: Number of games to play
            agents_per_game: Number of agents per game

        Returns:
            SelfPlayResult with MDAP-enhanced statistics
        """
        logger.info(f"Starting MDAP self-play: {games} games, {agents_per_game} agents/game")

        start_time = time.time()
        all_results = []

        for game_idx in range(games):
            logger.info(f"Game {game_idx + 1}/{games}")

            # Select agents for this game
            game_agents = random.sample(self.agents, min(agents_per_game, len(self.agents)))

            # Create initial state
            initial_state = ProofState(
                theorem=theorem,
                current_goals=[theorem],
                proven_goals=[],
                available_tactics=[]
            )

            # Play game with MDAP
            game_result = await self._play_mdap_game(initial_state, game_agents)
            all_results.append(game_result)

            # Update policies from MDAP results
            await self.update_policy_from_mdap([game_result.proof])

        # Calculate aggregate statistics
        total_time = time.time() - start_time
        success_count = sum(1 for r in all_results if r.success)

        result = SelfPlayResult(
            success_count=success_count,
            total_games=games,
            success_rate=success_count / games,
            total_time=total_time,
            proofs=[r.proof for r in all_results],
            statistics=self._calculate_mdap_statistics(all_results),
            metadata={"mdap_enhanced": True}
        )

        logger.info(f"MDAP self-play complete: {success_count}/{games} success")
        return result

    async def _play_mdap_game(
        self,
        initial_state: ProofState,
        agents: List['LeanProofAgent']
    ) -> GameResult:
        """
        Play a single game with MDAP consensus.

        Args:
            initial_state: Initial proof state
            agents: Agents for this game

        Returns:
            GameResult
        """
        current_state = initial_state
        tactics_used = []

        while not current_state.is_complete():
            # Select strategy with MDAP
            strategy = await self.select_strategy_with_mdap(current_state, agents)

            # Apply best tactic from strategy
            if strategy.tactics:
                best_tactic = strategy.tactics[0]
                tactics_used.append(best_tactic)

                # Update state (simplified)
                current_state = await self._apply_tactic(current_state, best_tactic)
            else:
                break

        # Verify proof
        proof = LeanProof(
            theorem_name=initial_state.theorem,
            lean_code="\n".join(str(t) for t in tactics_used),
            confidence=strategy.confidence if strategy else 0.5,
            strategy_used=ProofStrategy.SELF_PLAY,
            agent_id="mdap_self_play"
        )

        # Simple success check
        success = len(tactics_used) > 0 and current_state.is_complete()

        return GameResult(
            success=success,
            proof=proof,
            tactics=tactics_used,
            final_state=current_state
        )

    async def _mdap_vote_on_strategy(
        self,
        strategies: List[LeanProofStrategy],
        state: ProofState
    ) -> LeanProofStrategy:
        """
        Vote on best strategy using MDAP.

        Args:
            strategies: List of candidate strategies
            state: Current proof state

        Returns:
            Consensus strategy
        """
        if not strategies:
            # Return default strategy
            return LeanProofStrategy(
                theorem=state.theorem,
                approach=Approach.DIRECT,
                confidence=0.5,
                tactics=[],
                estimated_complexity=0.5
            )

        # Score strategies
        scored_strategies = []
        for strategy in strategies:
            # Score based on confidence, complexity, and state match
            score = (
                strategy.confidence * 0.5 +
                (1.0 - strategy.estimated_complexity) * 0.3 +
                len(strategy.tactics) * 0.01
            )
            scored_strategies.append((strategy, score))

        # Sort and select best
        scored_strategies.sort(key=lambda x: x[1], reverse=True)
        return scored_strategies[0][0]

    async def _apply_tactic(self, state: ProofState, tactic) -> ProofState:
        """Apply tactic and return new state (simplified)."""
        # In real implementation, would update goals properly
        new_goals = state.current_goals[1:] if state.current_goals else []
        return ProofState(
            theorem=state.theorem,
            current_goals=new_goals,
            proven_goals=state.proven_goals + state.current_goals[:1] if state.current_goals else state.proven_goals,
            available_tactics=state.available_tactics
        )

    def _calculate_mdap_statistics(self, results: List[GameResult]) -> Dict[str, Any]:
        """Calculate statistics from MDAP game results."""
        success_count = sum(1 for r in results if r.success)
        avg_confidence = sum(r.proof.confidence for r in results) / len(results) if results else 0.0

        return {
            "success_count": success_count,
            "total_games": len(results),
            "success_rate": success_count / len(results) if results else 0.0,
            "average_confidence": avg_confidence,
            "mdap_votes": dict(self.mdap_stats["strategy_votes"])
        }

    def get_mdap_report(self) -> Dict[str, Any]:
        """Get MDAP statistics report."""
        avg_consensus = 0.0
        if self.mdap_stats["consensus_rates"]:
            avg_consensus = sum(self.mdap_stats["consensus_rates"]) / len(self.mdap_stats["consensus_rates"])

        return {
            "strategy_votes": dict(self.mdap_stats["strategy_votes"]),
            "average_consensus_rate": avg_consensus,
            "agent_contributions": dict(self.mdap_stats["agent_contributions"]),
            "policy_updates": len(self.mdap_stats["policy_updates"])
        }


# Export all classes
__all__ = [
    # Core classes
    'ProofState',
    'LeanProofAgent',
    'SelfPlayEngine',
    'SelfPlayResult',
    'GameResult',
    'TrainingProgress',

    # MDAP-Enhanced
    'LeanSelfPlayEngineMDAP',
]
