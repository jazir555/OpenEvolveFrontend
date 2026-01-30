"""
LeanAide MDAP Integration

This module integrates Multi-Stage Agent Pipeline (MDAP) architecture with Lean 4 proof generation,
providing a robust, multi-agent, voting-based system for generating and verifying mathematical proofs.

Architecture:
    Layer 1: LeanMDAPTask decomposition (if needed)
        ↓
    Layer 2: Multi-agent parallel execution (evolution, MCTS, adversarial, self-play)
        ↓
    Layer 3: Voting-based aggregation (first-K-ahead-by-K)
        ↓
    Layer 4: Verification and refinement

Key Components:
    LeanMDAPConfig: Configuration for Lean MDAP pipeline
    LeanMDAPStep: Specialized MDAP step for Lean 4 proofs
    LeanMDAPTask: Multi-step proof generation task
    LeanProofAgent: Proof generation agent with different strategies
    LeanAgentSelector: Intelligent agent selection based on task characteristics
    LeanMDAPOrchestrator: Main orchestration engine
    LeanMDAPResult: Comprehensive result container

Integration Features:
    - Multi-strategy parallel execution
    - Voting-based aggregation
    - Red-flagging for invalid proofs
    - Hierarchical decomposition for complex theorems
    - Adaptive agent selection
    - Checkpointing for long-running tasks
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Import MDAP core components
try:
    from mdap_engine import (
        MDAPOrchestrator,
        MDAPConfig,
        MDAPStep,
        MDAPTask,
        RedFlagRules,
        RedFlagger,
        AgentSelector,
        MDAPVoteResult,
        MDAPRunResult,
        canonicalize_candidate,
        candidate_confidence,
    )
    from workflow_structures import ModelConfig, Team
    MDAP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"MDAP engine not available: {e}")
    MDAP_AVAILABLE = False
    MDAPOrchestrator = None
    MDAPConfig = None
    MDAPStep = None
    MDAPTask = None

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ProofStrategy(str, Enum):
    """Proof generation strategies"""
    EVOLUTION = "evolution"
    MCTS = "mcts"
    ADVERSARIAL = "adversarial"
    SELF_PLAY = "self_play"
    DIRECT = "direct"


class LeanDomain(str, Enum):
    """Lean theorem domains"""
    ALGEBRA = "algebra"
    ANALYSIS = "analysis"
    LOGIC = "logic"
    CATEGORY_THEORY = "category_theory"
    TOPOLOGY = "topology"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    GEOMETRY = "geometry"
    GENERAL = "general"


class VotingStrategy(str, Enum):
    """Voting strategies for proof aggregation"""
    FIRST_K_AHEAD = "first_k_ahead"
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    THRESHOLD = "threshold"


class ProofStatus(str, Enum):
    """Proof generation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RED_FLAGGED = "red_flagged"
    VERIFIED = "verified"
    NEEDS_REFINEMENT = "needs_refinement"


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class LeanProof:
    """Container for Lean 4 proof"""
    theorem_name: str
    lean_code: str
    confidence: float
    strategy_used: ProofStrategy
    agent_id: str
    verification_status: bool = False
    verification_message: str = ""
    tactics_used: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    proof_length: int = 0
    generation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate proof length after initialization"""
        self.proof_length = len(self.lean_code.split('\n'))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class LeanMDAPConfig:
    """Configuration for Lean MDAP pipeline"""

    # Agent configuration
    available_agents: List[str] = field(default_factory=lambda: [
        "evolution", "mcts", "adversarial", "self_play", "direct"
    ])
    default_parallel_agents: int = 4
    max_parallel_agents: int = 8

    # Voting configuration
    voting_strategy: VotingStrategy = VotingStrategy.FIRST_K_AHEAD
    k_ahead_threshold: int = 3
    min_confidence_threshold: float = 0.5
    weight_by_confidence: bool = True

    # Red-flagging
    enable_red_flagging: bool = True
    max_proof_length: int = 1000
    min_confidence: float = 0.2
    require_verification: bool = True
    blocked_patterns: List[str] = field(default_factory=list)

    # Execution
    timeout_seconds: int = 300
    max_retries: int = 3
    enable_checkpointing: bool = True
    checkpoint_dir: str = "./lean_mdap_checkpoints"

    # LeanAide integration
    leanaide_host: str = "localhost"
    leanaide_port: int = 7654
    verification_timeout: int = 60
    max_verification_retries: int = 3

    # Evolution strategy parameters
    evolution_population_size: int = 20
    evolution_max_generations: int = 10
    evolution_temperature: float = 0.7

    # MCTS strategy parameters
    mcts_simulations: int = 100
    mcts_exploration_constant: float = 1.414
    mcts_temperature: float = 0.5

    # Adversarial strategy parameters
    adversarial_rounds: int = 5
    red_team_models: List[str] = field(default_factory=lambda: ["gpt-4"])
    blue_team_models: List[str] = field(default_factory=lambda: ["gpt-4"])

    # Self-play strategy parameters
    self_play_episodes: int = 50
    self_play_learning_rate: float = 0.01

    # Direct translation parameters
    direct_model: str = "gpt-4"
    direct_temperature: float = 0.3
    direct_max_tokens: int = 2000

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000

    # Logging and monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_port: Optional[int] = None

    # Domain specialization
    enable_domain_specialization: bool = True
    domain_agent_mapping: Dict[LeanDomain, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        """Create checkpoint directory if needed"""
        if self.enable_checkpointing:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# LEAN MDAP STEP
# =============================================================================

class LeanMDAPStep(MDAPStep):
    """
    MDAP step specialized for Lean 4 proofs

    Extends MDAPStep with Lean-specific attributes and methods
    """

    def __init__(
        self,
        step_id: str,
        theorem_statement: str,
        proof_strategy: ProofStrategy = ProofStrategy.EVOLUTION,
        strategy_params: Optional[Dict[str, Any]] = None,
        expected_schema: Optional[Dict[str, Any]] = None,
        domain: LeanDomain = LeanDomain.GENERAL,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Lean MDAP step

        Args:
            step_id: Unique step identifier
            theorem_statement: Theorem to prove
            proof_strategy: Strategy to use for proof generation
            strategy_params: Strategy-specific parameters
            expected_schema: Expected output schema
            domain: Mathematical domain
            priority: Step priority (affects k_ahead)
            metadata: Additional metadata
        """
        # Initialize parent MDAPStep
        prompt = self._create_prompt(theorem_statement, proof_strategy, domain)
        super().__init__(
            step_id=step_id,
            prompt=prompt,
            expected_schema=expected_schema,
            task_type="lean_proof",
            priority=priority,
            metadata=metadata or {}
        )

        self.theorem_statement = theorem_statement
        self.proof_strategy = proof_strategy
        self.strategy_params = strategy_params or {}
        self.domain = domain

    def _create_prompt(
        self,
        theorem: str,
        strategy: ProofStrategy,
        domain: LeanDomain
    ) -> str:
        """Create LLM prompt for this step"""
        strategy_desc = {
            ProofStrategy.EVOLUTION: "using evolutionary/genetic algorithm approach",
            ProofStrategy.MCTS: "using Monte Carlo Tree Search",
            ProofStrategy.ADVERSARIAL: "using adversarial red-blue team approach",
            ProofStrategy.SELF_PLAY: "using reinforcement learning self-play",
            ProofStrategy.DIRECT: "using direct translation"
        }

        return f"""Generate a Lean 4 proof for the following theorem in the {domain.value} domain, {strategy_desc[strategy]}.

Theorem:
{theorem}

Provide a complete, verified Lean 4 proof with proper tactics and structure."""

    def to_prompt(self) -> str:
        """Convert to LLM prompt"""
        return self.prompt

    def validate_result(self, proof: LeanProof) -> bool:
        """
        Validate proof result

        Args:
            proof: Generated proof to validate

        Returns:
            True if proof is valid, False otherwise
        """
        # Check basic requirements
        if not proof.lean_code or proof.lean_code.strip() == "":
            return False

        if proof.confidence < 0.0 or proof.confidence > 1.0:
            return False

        # Check that proof uses the expected strategy
        if proof.strategy_used != self.proof_strategy:
            logger.warning(
                f"Proof strategy mismatch: expected {self.proof_strategy}, "
                f"got {proof.strategy_used}"
            )

        return True


# =============================================================================
# LEAN MDAP TASK
# =============================================================================

class LeanMDAPTask(MDAPTask):
    """
    Multi-step Lean 4 proof generation task

    Typical workflow:
    1. Decomposition (if needed for complex theorems)
    2. Translation (Natural Language → Lean)
    3. Proof generation (multiple strategies)
    4. Verification
    5. Refinement (if needed)
    """

    def __init__(
        self,
        task_id: str,
        description: str,
        theorem_statement: str,
        domain: LeanDomain = LeanDomain.GENERAL,
        max_retries: int = 3,
        target_success_rate: float = 0.95,
        enable_decomposition: bool = False,
        enable_refinement: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Lean MDAP task

        Args:
            task_id: Unique task identifier
            description: Task description
            theorem_statement: Theorem to prove
            domain: Mathematical domain
            max_retries: Maximum retry attempts
            target_success_rate: Target success rate
            enable_decomposition: Enable hierarchical decomposition
            enable_refinement: Enable proof refinement
            metadata: Additional metadata
        """
        # Initialize parent MDAPTask
        super().__init__(
            task_id=task_id,
            description=description,
            steps=[],
            max_retries=max_retries,
            target_success_rate=target_success_rate,
            metadata=metadata or {}
        )

        self.theorem_statement = theorem_statement
        self.domain = domain
        self.enable_decomposition = enable_decomposition
        self.enable_refinement = enable_refinement
        self.steps_created = False

    def add_step(self, step: LeanMDAPStep) -> None:
        """
        Add a step to the task

        Args:
            step: LeanMDAPStep to add
        """
        if not isinstance(step, LeanMDAPStep):
            raise TypeError("Step must be a LeanMDAPStep")

        self.steps.append(step)

    def create_default_steps(
        self,
        strategies: List[ProofStrategy],
        parallel: bool = True
    ) -> None:
        """
        Create default workflow steps

        Args:
            strategies: List of proof strategies to use
            parallel: Whether to run strategies in parallel (single step)
        """
        if parallel:
            # Single parallel step with all strategies
            step = LeanMDAPStep(
                step_id=f"{self.task_id}_parallel_generation",
                theorem_statement=self.theorem_statement,
                proof_strategy=ProofStrategy.EVOLUTION,  # Will be overridden
                strategy_params={"strategies": strategies},
                domain=self.domain,
                priority=1
            )
            step.metadata["parallel_strategies"] = [s.value for s in strategies]
            self.add_step(step)
        else:
            # Sequential steps for each strategy
            for i, strategy in enumerate(strategies):
                step = LeanMDAPStep(
                    step_id=f"{self.task_id}_{strategy.value}_{i}",
                    theorem_statement=self.theorem_statement,
                    proof_strategy=strategy,
                    domain=self.domain,
                    priority=i
                )
                self.add_step(step)

        # Add verification step
        verification_step = LeanMDAPStep(
            step_id=f"{self.task_id}_verification",
            theorem_statement=self.theorem_statement,
            proof_strategy=ProofStrategy.DIRECT,
            domain=self.domain,
            priority=10  # High priority for verification
        )
        verification_step.metadata["is_verification"] = True
        self.add_step(verification_step)

        self.steps_created = True

    def get_execution_plan(self) -> List[LeanMDAPStep]:
        """
        Get execution plan for this task

        Returns:
            List of LeanMDAPSteps in execution order
        """
        if not self.steps_created:
            # Create default steps if none exist
            default_strategies = [
                ProofStrategy.EVOLUTION,
                ProofStrategy.MCTS,
                ProofStrategy.ADVERSARIAL
            ]
            self.create_default_steps(default_strategies)

        return self.steps


# =============================================================================
# LEAN PROOF AGENT
# =============================================================================

class LeanProofAgent:
    """
    Proof generation agent with different strategies

    Agent types:
    - EvolutionaryAgent: Genetic algorithm-based proof search
    - MCTSAgent: Monte Carlo Tree Search for proof exploration
    - AdversarialAgent: Red-blue team adversarial proof generation
    - SelfPlayAgent: Reinforcement learning through self-play
    - DirectAgent: Direct LLM translation to Lean
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: ProofStrategy,
        model_config: Optional[ModelConfig] = None,
        capabilities: Optional[List[LeanDomain]] = None,
        config: Optional[LeanMDAPConfig] = None
    ):
        """
        Initialize Lean proof agent

        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (proof strategy)
            model_config: LLM model configuration
            capabilities: Domains this agent specializes in
            config: Lean MDAP configuration
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_config = model_config
        self.capabilities = capabilities or [LeanDomain.GENERAL]
        self.config = config or LeanMDAPConfig()

        # Performance metrics
        self.total_proofs_generated = 0
        self.successful_proofs = 0
        self.total_generation_time = 0.0
        self.avg_confidence = 0.5

        # Strategy-specific initialization
        self._initialize_strategy()

    def _initialize_strategy(self):
        """Initialize strategy-specific components"""
        if self.agent_type == ProofStrategy.EVOLUTION:
            self._init_evolution()
        elif self.agent_type == ProofStrategy.MCTS:
            self._init_mcts()
        elif self.agent_type == ProofStrategy.ADVERSARIAL:
            self._init_adversarial()
        elif self.agent_type == ProofStrategy.SELF_PLAY:
            self._init_self_play()
        elif self.agent_type == ProofStrategy.DIRECT:
            self._init_direct()

    def _init_evolution(self):
        """Initialize evolutionary strategy components"""
        try:
            from evolution import EvolutionaryOptimizer
            self.evolution_optimizer = EvolutionaryOptimizer(
                population_size=self.config.evolution_population_size,
                max_generations=self.config.evolution_max_generations,
                temperature=self.config.evolution_temperature
            )
        except ImportError:
            logger.warning("Evolution module not available")
            self.evolution_optimizer = None

    def _init_mcts(self):
        """Initialize MCTS strategy components"""
        try:
            from mcts_engine import MCTSProver
            self.mcts_prover = MCTSProver(
                num_simulations=self.config.mcts_simulations,
                exploration_constant=self.config.mcts_exploration_constant,
                temperature=self.config.mcts_temperature
            )
        except ImportError:
            logger.warning("MCTS module not available")
            self.mcts_prover = None

    def _init_adversarial(self):
        """Initialize adversarial strategy components"""
        try:
            from adversarial import AdversarialGenerator
            self.adversarial_generator = AdversarialGenerator(
                rounds=self.config.adversarial_rounds,
                red_team_models=self.config.red_team_models,
                blue_team_models=self.config.blue_team_models
            )
        except ImportError:
            logger.warning("Adversarial module not available")
            self.adversarial_generator = None

    def _init_self_play(self):
        """Initialize self-play strategy components"""
        try:
            from selfplay import SelfPlayTrainer
            self.self_play_trainer = SelfPlayTrainer(
                episodes=self.config.self_play_episodes,
                learning_rate=self.config.self_play_learning_rate
            )
        except ImportError:
            logger.warning("Self-play module not available")
            self.self_play_trainer = None

    def _init_direct(self):
        """Initialize direct translation strategy"""
        # Direct strategy uses model_config directly
        pass

    def generate_proof(
        self,
        theorem: str,
        domain: LeanDomain = LeanDomain.GENERAL,
        context: Optional[Dict[str, Any]] = None
    ) -> LeanProof:
        """
        Generate proof for theorem

        Args:
            theorem: Theorem statement
            domain: Mathematical domain
            context: Additional context

        Returns:
            Generated LeanProof
        """
        start_time = time.time()
        context = context or {}

        try:
            # Route to appropriate strategy
            if self.agent_type == ProofStrategy.EVOLUTION:
                proof = self._generate_evolution(theorem, domain, context)
            elif self.agent_type == ProofStrategy.MCTS:
                proof = self._generate_mcts(theorem, domain, context)
            elif self.agent_type == ProofStrategy.ADVERSARIAL:
                proof = self._generate_adversarial(theorem, domain, context)
            elif self.agent_type == ProofStrategy.SELF_PLAY:
                proof = self._generate_self_play(theorem, domain, context)
            elif self.agent_type == ProofStrategy.DIRECT:
                proof = self._generate_direct(theorem, domain, context)
            else:
                raise ValueError(f"Unknown agent type: {self.agent_type}")

            # Update metrics
            generation_time = time.time() - start_time
            proof.generation_time = generation_time

            self.total_proofs_generated += 1
            self.total_generation_time += generation_time
            self.avg_confidence = (
                (self.avg_confidence * (self.total_proofs_generated - 1) +
                 proof.confidence) / self.total_proofs_generated
            )

            if proof.verification_status:
                self.successful_proofs += 1

            return proof

        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error(f"Error generating proof with {self.agent_id}: {e}")
            # Return failed proof
            return LeanProof(
                theorem_name=theorem.split(':')[0] if ':' in theorem else "unknown",
                lean_code="",
                confidence=0.0,
                strategy_used=self.agent_type,
                agent_id=self.agent_id,
                verification_status=False,
                verification_message=str(e)
            )

    def _generate_evolution(
        self,
        theorem: str,
        domain: LeanDomain,
        context: Dict[str, Any]
    ) -> LeanProof:
        """Generate proof using evolutionary strategy"""
        if self.evolution_optimizer:
            try:
                result = self.evolution_optimizer.optimize(
                    theorem=theorem,
                    domain=domain.value,
                    context=context
                )
                return LeanProof(
                    theorem_name=context.get("theorem_name", "evolution_proof"),
                    lean_code=result.get("proof", ""),
                    confidence=result.get("confidence", 0.5),
                    strategy_used=ProofStrategy.EVOLUTION,
                    agent_id=self.agent_id,
                    tactics_used=result.get("tactics", []),
                    metadata=result
                )
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.error(f"Evolution strategy failed: {e}")

        # Fallback to direct generation
        return self._generate_direct(theorem, domain, context)

    def _generate_mcts(
        self,
        theorem: str,
        domain: LeanDomain,
        context: Dict[str, Any]
    ) -> LeanProof:
        """Generate proof using MCTS strategy"""
        if self.mcts_prover:
            try:
                result = self.mcts_prover.search(
                    theorem=theorem,
                    domain=domain.value,
                    context=context
                )
                return LeanProof(
                    theorem_name=context.get("theorem_name", "mcts_proof"),
                    lean_code=result.get("proof", ""),
                    confidence=result.get("confidence", 0.5),
                    strategy_used=ProofStrategy.MCTS,
                    agent_id=self.agent_id,
                    tactics_used=result.get("tactics", []),
                    metadata=result
                )
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.error(f"MCTS strategy failed: {e}")

        # Fallback to direct generation
        return self._generate_direct(theorem, domain, context)

    def _generate_adversarial(
        self,
        theorem: str,
        domain: LeanDomain,
        context: Dict[str, Any]
    ) -> LeanProof:
        """Generate proof using adversarial strategy"""
        if self.adversarial_generator:
            try:
                result = self.adversarial_generator.generate(
                    theorem=theorem,
                    domain=domain.value,
                    context=context
                )
                return LeanProof(
                    theorem_name=context.get("theorem_name", "adversarial_proof"),
                    lean_code=result.get("proof", ""),
                    confidence=result.get("confidence", 0.5),
                    strategy_used=ProofStrategy.ADVERSARIAL,
                    agent_id=self.agent_id,
                    tactics_used=result.get("tactics", []),
                    metadata=result
                )
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.error(f"Adversarial strategy failed: {e}")

        # Fallback to direct generation
        return self._generate_direct(theorem, domain, context)

    def _generate_self_play(
        self,
        theorem: str,
        domain: LeanDomain,
        context: Dict[str, Any]
    ) -> LeanProof:
        """Generate proof using self-play strategy"""
        if self.self_play_trainer:
            try:
                result = self.self_play_trainer.train_and_generate(
                    theorem=theorem,
                    domain=domain.value,
                    context=context
                )
                return LeanProof(
                    theorem_name=context.get("theorem_name", "selfplay_proof"),
                    lean_code=result.get("proof", ""),
                    confidence=result.get("confidence", 0.5),
                    strategy_used=ProofStrategy.SELF_PLAY,
                    agent_id=self.agent_id,
                    tactics_used=result.get("tactics", []),
                    metadata=result
                )
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.error(f"Self-play strategy failed: {e}")

        # Fallback to direct generation
        return self._generate_direct(theorem, domain, context)

    def _generate_direct(
        self,
        theorem: str,
        domain: LeanDomain,
        context: Dict[str, Any]
    ) -> LeanProof:
        """Generate proof using direct LLM translation"""
        try:
            # Use model_config to call LLM
            if self.model_config and MDAP_AVAILABLE:
                from llm_utils import _compose_messages, _request_openai_compatible_chat

                system_prompt = f"""You are an expert Lean 4 proof assistant specializing in {domain.value}.

Generate complete, verified Lean 4 proofs with proper tactics and structure.
Follow these guidelines:
1. Use appropriate tactics for the domain
2. Provide clear, structured proofs
3. Include necessary imports and dependencies
4. Ensure type correctness
5. Add comments for complex steps"""

                user_prompt = f"""Generate a Lean 4 proof for the following theorem:

Theorem: {theorem}

Provide the complete proof code."""

                messages = _compose_messages(system_prompt, user_prompt)
                response = _request_openai_compatible_chat(
                    api_key=self.model_config.api_key,
                    base_url=self.model_config.api_base,
                    model=self.model_config.model_id,
                    messages=messages,
                    temperature=self.config.direct_temperature,
                    max_tokens=self.config.direct_max_tokens
                )

                lean_code = response or ""
                confidence = 0.7  # Default confidence for direct

                return LeanProof(
                    theorem_name=context.get("theorem_name", "direct_proof"),
                    lean_code=lean_code,
                    confidence=confidence,
                    strategy_used=ProofStrategy.DIRECT,
                    agent_id=self.agent_id,
                    verification_status=False,  # Needs verification
                    metadata={"raw_response": response}
                )
            else:
                # No model available, return empty proof
                return LeanProof(
                    theorem_name=context.get("theorem_name", "direct_proof"),
                    lean_code="",
                    confidence=0.0,
                    strategy_used=ProofStrategy.DIRECT,
                    agent_id=self.agent_id,
                    verification_status=False,
                    verification_message="No model configured"
                )

        except (IOError, ConnectionError, TimeoutError) as e:
            logger.error(f"Direct strategy failed: {e}")
            return LeanProof(
                theorem_name=context.get("theorem_name", "direct_proof"),
                lean_code="",
                confidence=0.0,
                strategy_used=ProofStrategy.DIRECT,
                agent_id=self.agent_id,
                verification_status=False,
                verification_message=str(e)
            )

    def estimate_quality(self, proof: LeanProof) -> float:
        """
        Estimate quality of a proof

        Args:
            proof: Proof to evaluate

        Returns:
            Quality score (0.0 - 1.0)
        """
        quality = 0.0

        # Confidence contributes significantly
        quality += proof.confidence * 0.4

        # Verification status
        if proof.verification_status:
            quality += 0.3

        # Proof length (moderate length is better)
        if 10 <= proof.proof_length <= 100:
            quality += 0.1
        elif proof.proof_length > 100:
            quality += 0.05

        # Tactics used
        if proof.tactics_used:
            quality += min(0.1, len(proof.tactics_used) * 0.01)

        # Strategy success rate
        if self.total_proofs_generated > 0:
            success_rate = self.successful_proofs / self.total_proofs_generated
            quality += success_rate * 0.1

        return min(1.0, quality)

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get agent capabilities

        Returns:
            Dict with capability information
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "capabilities": [c.value for c in self.capabilities],
            "total_proofs": self.total_proofs_generated,
            "successful_proofs": self.successful_proofs,
            "success_rate": (
                self.successful_proofs / self.total_proofs_generated
                if self.total_proofs_generated > 0 else 0.0
            ),
            "avg_confidence": self.avg_confidence,
            "avg_generation_time": (
                self.total_generation_time / self.total_proofs_generated
                if self.total_proofs_generated > 0 else 0.0
            )
        }


# =============================================================================
# LEAN AGENT SELECTOR
# =============================================================================

class LeanAgentSelector(AgentSelector):
    """
    Intelligent agent selector for Lean proof generation

    Uses:
    - Theorem characteristics (domain, complexity)
    - Agent capabilities
    - Historical performance
    - Resource constraints
    """

    def __init__(
        self,
        config: LeanMDAPConfig,
        rng: Optional[random.Random] = None
    ):
        """
        Initialize Lean agent selector

        Args:
            config: Lean MDAP configuration
            rng: Random number generator
        """
        self.config = config
        self.rng = rng or random.Random()

        # Agent registry
        self.agents: Dict[str, LeanProofAgent] = {}
        self.agent_performance: Dict[str, Dict[str, float]] = {}

    def register_agent(self, agent: LeanProofAgent) -> None:
        """
        Register an agent with the selector

        Args:
            agent: LeanProofAgent to register
        """
        self.agents[agent.agent_id] = agent
        if agent.agent_id not in self.agent_performance:
            self.agent_performance[agent.agent_id] = {
                "success_rate": 0.5,
                "avg_confidence": 0.5,
                "avg_time": 0.0,
                "domain_scores": {}
            }

    def select_agents(
        self,
        task: LeanMDAPTask,
        count: int,
        domain: Optional[LeanDomain] = None
    ) -> List[LeanProofAgent]:
        """
        Select best agents for task

        Args:
            task: LeanMDAPTask to execute
            count: Number of agents to select
            domain: Domain to specialize in (uses task domain if None)

        Returns:
            List of selected LeanProofAgents
        """
        domain = domain or task.domain

        # Score all agents
        scored_agents = []
        for agent in self.agents.values():
            score = self.score_agent(agent, task, domain)
            scored_agents.append((agent, score))

        # Sort by score (descending)
        scored_agents.sort(key=lambda x: x[1], reverse=True)

        # Select top agents
        selected = [agent for agent, score in scored_agents[:count]]

        logger.info(
            f"Selected {len(selected)} agents for task {task.task_id}: "
            f"{[a.agent_id for a in selected]}"
        )

        return selected

    def score_agent(
        self,
        agent: LeanProofAgent,
        task: LeanMDAPTask,
        domain: LeanDomain
    ) -> float:
        """
        Score agent suitability for task

        Args:
            agent: Agent to score
            task: Task to evaluate
            domain: Domain to specialize in

        Returns:
            Score (higher is better)
        """
        score = 0.0

        # Base score from performance
        perf = self.agent_performance.get(agent.agent_id, {})
        score += perf.get("success_rate", 0.5) * 0.3
        score += perf.get("avg_confidence", 0.5) * 0.2

        # Domain specialization bonus
        if domain in agent.capabilities:
            score += 0.2

        # Domain-specific performance
        domain_scores = perf.get("domain_scores", {})
        if domain.value in domain_scores:
            score += domain_scores[domain.value] * 0.15

        # Strategy preference
        if self.config.enable_domain_specialization:
            domain_mapping = self.config.domain_agent_mapping
            if domain in domain_mapping:
                if agent.agent_type.value in domain_mapping[domain]:
                    score += 0.15

        # Recent performance boost
        if agent.total_proofs_generated > 10:
            recent_success_rate = (
                agent.successful_proofs / agent.total_proofs_generated
            )
            score += recent_success_rate * 0.1

        return min(1.0, score)

    def update_agent_performance(
        self,
        agent: LeanProofAgent,
        success: bool,
        confidence: float,
        domain: LeanDomain,
        execution_time: float
    ) -> None:
        """
        Update agent performance metrics

        Args:
            agent: Agent to update
            success: Whether proof generation was successful
            confidence: Proof confidence
            domain: Domain of theorem
            execution_time: Time taken
        """
        perf = self.agent_performance[agent.agent_id]

        # Update success rate with exponential moving average
        alpha = 0.1  # Learning rate
        new_success_rate = 1.0 if success else 0.0
        perf["success_rate"] = (
            alpha * new_success_rate +
            (1 - alpha) * perf["success_rate"]
        )

        # Update average confidence
        perf["avg_confidence"] = (
            alpha * confidence +
            (1 - alpha) * perf["avg_confidence"]
        )

        # Update average time
        perf["avg_time"] = (
            alpha * execution_time +
            (1 - alpha) * perf["avg_time"]
        )

        # Update domain score
        if domain.value not in perf["domain_scores"]:
            perf["domain_scores"][domain.value] = 0.5

        perf["domain_scores"][domain.value] = (
            alpha * (1.0 if success else 0.0) +
            (1 - alpha) * perf["domain_scores"][domain.value]
        )

        logger.debug(
            f"Updated performance for {agent.agent_id}: "
            f"success_rate={perf['success_rate']:.3f}, "
            f"avg_confidence={perf['avg_confidence']:.3f}"
        )


# =============================================================================
# LEAN MDAP ORCHESTRATOR
# =============================================================================

class LeanMDAPOrchestrator(MDAPOrchestrator if MDAP_AVAILABLE else object):
    """
    Main orchestration engine for Lean 4 proof generation

    Features:
    - Parallel agent execution
    - Voting aggregation
    - Red-flagging for invalid proofs
    - Hierarchical execution (decomposition → proof → verify)
    - Checkpointing for long-running tasks
    """

    def __init__(
        self,
        config: LeanMDAPConfig,
        team: Optional[Team] = None
    ):
        """
        Initialize Lean MDAP orchestrator

        Args:
            config: Lean MDAP configuration
            team: MDAP team (for base MDAP functionality)
        """
        self.config = config
        self.team = team

        # Initialize base MDAP if available
        if MDAP_AVAILABLE:
            mdap_config = MDAPConfig(
                k_min=config.k_ahead_threshold,
                k_max=config.k_ahead_threshold * 3,
                max_votes_per_step=50,
                timeout_seconds=config.timeout_seconds,
                red_flag_rules=RedFlagRules(
                    max_tokens=config.max_proof_length * 4,  # Approx 4 chars per token
                    min_confidence=config.min_confidence,
                    blocked_patterns=config.blocked_patterns
                ),
                fallback_policy="escalate_then_best_effort",
                cache_ttl_seconds=config.cache_ttl_seconds if config.enable_caching else None,
                cache_max_size=config.cache_max_size
            )
            super().__init__(team or self._create_default_team(), mdap_config)
        else:
            self.red_flagger = RedFlagger(RedFlagRules(
                max_tokens=config.max_proof_length * 4,
                min_confidence=config.min_confidence,
                blocked_patterns=config.blocked_patterns
            ))

        # Initialize Lean-specific components
        self.agent_selector = LeanAgentSelector(config)
        self._initialize_agents()

        # Metrics
        self.metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "total_proofs": 0,
            "verified_proofs": 0,
            "total_agents_used": 0,
            "avg_confidence": 0.0,
            "avg_execution_time": 0.0
        }

        # Checkpointing
        self.checkpoint_manager = CheckpointManager(config) if config.enable_checkpointing else None

    def _initialize_agents(self):
        """Initialize proof generation agents"""
        # Create model config
        model_config = None
        if MDAP_AVAILABLE and self.team and self.team.members:
            model_config = self.team.members[0]
        else:
            # Create default model config
            model_config = ModelConfig(
                model_id=self.config.direct_model,
                provider="openai",
                model_name=self.config.direct_model,
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                temperature=self.config.direct_temperature
            )

        # Create agents for each available strategy
        for strategy_name in self.config.available_agents:
            try:
                strategy = ProofStrategy(strategy_name)
                agent = LeanProofAgent(
                    agent_id=f"{strategy_name}_agent",
                    agent_type=strategy,
                    model_config=model_config,
                    config=self.config
                )
                self.agent_selector.register_agent(agent)
                logger.info(f"Initialized agent: {agent.agent_id}")
            except ValueError:
                logger.warning(f"Unknown strategy: {strategy_name}")

    def _create_default_team(self) -> Team:
        """Create default team if none provided"""
        model_config = ModelConfig(
            model_id=self.config.direct_model,
            provider="openai",
            model_name=self.config.direct_model,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            temperature=self.config.direct_temperature
        )
        return Team(
            team_id="lean_mdap_default",
            name="Lean MDAP Default Team",
            members=[model_config],
            description="Default team for Lean MDAP execution"
        )

    def orchestrate_proof_generation(
        self,
        task: LeanMDAPTask
    ) -> 'LeanMDAPResult':
        """
        Main entry point: Orchestrate proof generation

        Args:
            task: LeanMDAPTask to execute

        Returns:
            LeanMDAPResult with best proof and metrics
        """
        logger.info(f"Orchestrating proof generation for task: {task.task_id}")
        start_time = time.time()
        execution_id = f"lean_mdap_{task.task_id}_{int(time.time())}"

        # Load checkpoint if exists
        if self.checkpoint_manager:
            checkpointed_result = self.checkpoint_manager.load(task.task_id)
            if checkpointed_result:
                logger.info(f"Resuming from checkpoint for task {task.task_id}")
                return checkpointed_result

        try:
            # Step 1: Select agents for this task
            num_agents = min(
                self.config.default_parallel_agents,
                self.config.max_parallel_agents,
                len(self.agent_selector.agents)
            )
            selected_agents = self.agent_selector.select_agents(
                task,
                num_agents,
                task.domain
            )

            # Step 2: Execute steps
            all_proofs = []
            execution_trace = []
            step_results = {}

            for step in task.get_execution_plan():
                logger.info(f"Executing step: {step.step_id}")

                # Check if this is a parallel generation step
                if "parallel_strategies" in step.metadata:
                    proofs = self._execute_step_parallel(step, selected_agents)
                else:
                    proofs = self._execute_step_serial(step, selected_agents)

                all_proofs.extend(proofs)
                step_results[step.step_id] = proofs
                execution_trace.append({
                    "step_id": step.step_id,
                    "num_proofs": len(proofs),
                    "strategies": [p.strategy_used.value for p in proofs]
                })

                # Save checkpoint
                if self.checkpoint_manager:
                    self.checkpoint_manager.save(
                        task.task_id,
                        step,
                        proofs,
                        execution_trace
                    )

            # Step 3: Aggregate votes
            if self.config.voting_strategy == VotingStrategy.FIRST_K_AHEAD:
                best_proof = self._aggregate_first_k_ahead(all_proofs)
            elif self.config.voting_strategy == VotingStrategy.MAJORITY:
                best_proof = self._aggregate_majority(all_proofs)
            elif self.config.voting_strategy == VotingStrategy.WEIGHTED:
                best_proof = self._aggregate_weighted(all_proofs)
            elif self.config.voting_strategy == VotingStrategy.THRESHOLD:
                best_proof = self._aggregate_threshold(all_proofs)
            else:
                best_proof = self._aggregate_first_k_ahead(all_proofs)

            # Step 4: Final verification (if enabled)
            if self.config.require_verification and not best_proof.verification_status:
                best_proof = self._verify_proof(best_proof, task)

            execution_time = time.time() - start_time

            # Step 5: Update agent performance
            for proof in all_proofs:
                agent = self.agent_selector.agents.get(proof.agent_id)
                if agent:
                    self.agent_selector.update_agent_performance(
                        agent,
                        proof.verification_status,
                        proof.confidence,
                        task.domain,
                        proof.generation_time
                    )

            # Create result
            result = LeanMDAPResult(
                task_id=task.task_id,
                execution_id=execution_id,
                theorem_statement=task.theorem_statement,
                domain=task.domain,
                success=best_proof.verification_status and best_proof.lean_code != "",
                best_proof=best_proof,
                all_proofs=all_proofs,
                agents_used=[a.agent_id for a in selected_agents],
                execution_time=execution_time,
                execution_trace=execution_trace,
                step_results=step_results,
                voting_statistics=self._compute_voting_stats(all_proofs, best_proof),
                red_flags=self._analyze_red_flags(all_proofs),
                metrics=self._compute_task_metrics(all_proofs, execution_time)
            )

            # Update global metrics
            self.metrics["total_tasks"] += 1
            if result.success:
                self.metrics["successful_tasks"] += 1
            self.metrics["total_proofs"] += len(all_proofs)
            self.metrics["verified_proofs"] += sum(
                1 for p in all_proofs if p.verification_status
            )
            self.metrics["total_agents_used"] += len(selected_agents)
            self.metrics["avg_confidence"] = (
                (self.metrics["avg_confidence"] * (self.metrics["total_tasks"] - 1) +
                 result.avg_confidence) / self.metrics["total_tasks"]
            )
            self.metrics["avg_execution_time"] = (
                (self.metrics["avg_execution_time"] * (self.metrics["total_tasks"] - 1) +
                 execution_time) / self.metrics["total_tasks"]
            )

            # Clear checkpoint on success
            if self.checkpoint_manager and result.success:
                self.checkpoint_manager.clear(task.task_id)

            logger.info(
                f"Proof generation completed for task {task.task_id}: "
                f"success={result.success}, "
                f"confidence={best_proof.confidence:.3f}, "
                f"time={execution_time:.2f}s"
            )

            return result

        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Error in proof generation orchestration: {e}", exc_info=True)
            return LeanMDAPResult(
                task_id=task.task_id,
                execution_id=execution_id,
                theorem_statement=task.theorem_statement,
                domain=task.domain,
                success=False,
                best_proof=LeanProof(
                    theorem_name="error",
                    lean_code="",
                    confidence=0.0,
                    strategy_used=ProofStrategy.DIRECT,
                    agent_id="orchestrator",
                    verification_status=False,
                    verification_message=str(e)
                ),
                all_proofs=[],
                agents_used=[],
                execution_time=time.time() - start_time,
                error=str(e)
            )

    def _execute_step_parallel(
        self,
        step: LeanMDAPStep,
        agents: List[LeanProofAgent]
    ) -> List[LeanProof]:
        """
        Execute step with parallel agents

        Args:
            step: Step to execute
            agents: Agents to use

        Returns:
            List of generated proofs
        """
        proofs = []
        strategies = step.metadata.get("parallel_strategies", [])

        # Filter agents by strategies
        filtered_agents = [
            a for a in agents
            if a.agent_type.value in strategies
        ]

        if not filtered_agents:
            logger.warning(f"No agents found for strategies: {strategies}")
            return proofs

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=len(filtered_agents)) as executor:
            futures = {
                executor.submit(
                    agent.generate_proof,
                    step.theorem_statement,
                    step.domain,
                    step.metadata
                ): agent
                for agent in filtered_agents
            }

            for future in as_completed(futures, timeout=self.config.timeout_seconds):
                agent = futures[future]
                try:
                    proof = future.result()
                    proofs.append(proof)
                    logger.debug(
                        f"Agent {agent.agent_id} generated proof: "
                        f"confidence={proof.confidence:.3f}"
                    )
                except (ValueError, TypeError, AttributeError, KeyError) as e:
                    logger.error(f"Agent {agent.agent_id} failed: {e}")

        return proofs

    def _execute_step_serial(
        self,
        step: LeanMDAPStep,
        agents: List[LeanProofAgent]
    ) -> List[LeanProof]:
        """
        Execute step with single best agent

        Args:
            step: Step to execute
            agents: Available agents

        Returns:
            List of generated proofs (typically one)
        """
        proofs = []

        # Select best agent for this step's strategy
        matching_agents = [
            a for a in agents
            if a.agent_type == step.proof_strategy
        ]

        if not matching_agents:
            logger.warning(
                f"No agent found for strategy {step.proof_strategy}, "
                f"using first available"
            )
            if not agents:
                return proofs
            agent = agents[0]
        else:
            agent = matching_agents[0]

        try:
            proof = agent.generate_proof(
                step.theorem_statement,
                step.domain,
                step.metadata
            )
            proofs.append(proof)
        except Exception as e:
            logger.error(f"Serial execution failed: {e}")

        return proofs

    def _aggregate_first_k_ahead(self, proofs: List[LeanProof]) -> LeanProof:
        """
        Aggregate using first-K-ahead-by-K voting

        Args:
            proofs: List of candidate proofs

        Returns:
            Best proof
        """
        if not proofs:
            return LeanProof(
                theorem_name="none",
                lean_code="",
                confidence=0.0,
                strategy_used=ProofStrategy.DIRECT,
                agent_id="none"
            )

        # Canonicalize proofs for voting
        votes: Dict[str, int] = {}
        proof_map: Dict[str, LeanProof] = {}

        for proof in proofs:
            # Apply red-flagging
            if self.config.enable_red_flagging:
                is_flagged, reasons = self._check_red_flags(proof)
                if is_flagged:
                    logger.debug(f"Proof red-flagged: {reasons}")
                    continue

            # Canonicalize by lean_code
            canonical = canonicalize_candidate(proof.lean_code)
            votes[canonical] = votes.get(canonical, 0) + 1
            if canonical not in proof_map:
                proof_map[canonical] = proof

        if not votes:
            # All proofs red-flagged, return highest confidence
            return max(proofs, key=lambda p: p.confidence)

        # Check for K-ahead winner
        k = self.config.k_ahead_threshold
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_votes) >= 2:
            winner_count = sorted_votes[0][1]
            runner_up_count = sorted_votes[1][1]
            if winner_count >= runner_up_count + k:
                # K-ahead winner found
                winner_key = sorted_votes[0][0]
                return proof_map[winner_key]

        # No K-ahead winner, return highest voted
        winner_key = sorted_votes[0][0]
        return proof_map[winner_key]

    def _aggregate_majority(self, proofs: List[LeanProof]) -> LeanProof:
        """Aggregate using simple majority voting"""
        return self._aggregate_first_k_ahead(proofs)  # Same logic with k=1

    def _aggregate_weighted(self, proofs: List[LeanProof]) -> LeanProof:
        """Aggregate using confidence-weighted voting"""
        if not proofs:
            return LeanProof(
                theorem_name="none",
                lean_code="",
                confidence=0.0,
                strategy_used=ProofStrategy.DIRECT,
                agent_id="none"
            )

        # Filter red-flagged proofs
        valid_proofs = [
            p for p in proofs
            if not self._check_red_flags(p)[0]
        ]

        if not valid_proofs:
            return max(proofs, key=lambda p: p.confidence)

        # Weight by confidence
        weighted_scores: Dict[str, float] = {}
        proof_map: Dict[str, LeanProof] = {}

        for proof in valid_proofs:
            canonical = canonicalize_candidate(proof.lean_code)
            weight = proof.confidence
            weighted_scores[canonical] = weighted_scores.get(canonical, 0.0) + weight
            if canonical not in proof_map:
                proof_map[canonical] = proof

        # Return highest weighted
        winner_key = max(weighted_scores, key=weighted_scores.get)
        return proof_map[winner_key]

    def _aggregate_threshold(self, proofs: List[LeanProof]) -> LeanProof:
        """Aggregate using confidence threshold"""
        if not proofs:
            return LeanProof(
                theorem_name="none",
                lean_code="",
                confidence=0.0,
                strategy_used=ProofStrategy.DIRECT,
                agent_id="none"
            )

        # Filter by threshold
        valid_proofs = [
            p for p in proofs
            if p.confidence >= self.config.min_confidence_threshold
            and not self._check_red_flags(p)[0]
        ]

        if not valid_proofs:
            # Below threshold, return highest confidence
            return max(proofs, key=lambda p: p.confidence)

        # Return highest confidence above threshold
        return max(valid_proofs, key=lambda p: p.confidence)

    def _check_red_flags(self, proof: LeanProof) -> Tuple[bool, List[str]]:
        """
        Check if proof has red flags

        Args:
            proof: Proof to check

        Returns:
            (is_flagged, list of reasons)
        """
        reasons = []

        # Check proof length
        if proof.proof_length > self.config.max_proof_length:
            reasons.append(f"proof_too_long_{proof.proof_length}")

        # Check confidence
        if proof.confidence < self.config.min_confidence:
            reasons.append(f"low_confidence_{proof.confidence:.3f}")

        # Check verification status
        if self.config.require_verification and not proof.verification_status:
            reasons.append("not_verified")

        # Check for blocked patterns
        for pattern in self.config.blocked_patterns:
            if pattern.lower() in proof.lean_code.lower():
                reasons.append(f"blocked_pattern_{pattern}")

        # Check for empty proof
        if not proof.lean_code or proof.lean_code.strip() == "":
            reasons.append("empty_proof")

        return len(reasons) > 0, reasons

    def _analyze_red_flags(self, proofs: List[LeanProof]) -> Dict[str, Any]:
        """
        Analyze red flags across all proofs

        Args:
            proofs: List of proofs to analyze

        Returns:
            Red flag analysis
        """
        total_flags = 0
        flag_reasons: Dict[str, int] = {}

        for proof in proofs:
            is_flagged, reasons = self._check_red_flags(proof)
            if is_flagged:
                total_flags += 1
                for reason in reasons:
                    flag_reasons[reason] = flag_reasons.get(reason, 0) + 1

        return {
            "total_flags": total_flags,
            "flag_rate": total_flags / len(proofs) if proofs else 0.0,
            "flag_reasons": flag_reasons
        }

    def _verify_proof(
        self,
        proof: LeanProof,
        task: LeanMDAPTask
    ) -> LeanProof:
        """
        Verify proof using LeanAide server

        Args:
            proof: Proof to verify
            task: Original task

        Returns:
            Updated proof with verification status
        """
        try:
            import requests

            url = f"http://{self.config.leanaide_host}:{self.config.leanaide_port}/verify"
            payload = {
                "theorem": task.theorem_statement,
                "proof": proof.lean_code,
                "timeout": self.config.verification_timeout
            }

            response = requests.post(
                url,
                json=payload,
                timeout=self.config.verification_timeout + 10
            )

            if response.status_code == 200:
                result = response.json()
                proof.verification_status = result.get("verified", False)
                proof.verification_message = result.get("message", "")
            else:
                proof.verification_message = f"Verification failed: HTTP {response.status_code}"

        except (IOError, ConnectionError, TimeoutError) as e:
            logger.error(f"Proof verification error: {e}")
            proof.verification_message = f"Verification error: {str(e)}"

        return proof

    def _compute_voting_stats(
        self,
        all_proofs: List[LeanProof],
        best_proof: LeanProof
    ) -> Dict[str, Any]:
        """Compute voting statistics"""
        strategy_counts: Dict[str, int] = {}
        for proof in all_proofs:
            strategy = proof.strategy_used.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        return {
            "total_proofs": len(all_proofs),
            "strategy_distribution": strategy_counts,
            "best_strategy": best_proof.strategy_used.value,
            "best_confidence": best_proof.confidence,
            "avg_confidence": sum(p.confidence for p in all_proofs) / len(all_proofs) if all_proofs else 0.0
        }

    def _compute_task_metrics(
        self,
        proofs: List[LeanProof],
        execution_time: float
    ) -> Dict[str, Any]:
        """Compute task-specific metrics"""
        return {
            "num_proofs": len(proofs),
            "num_verified": sum(1 for p in proofs if p.verification_status),
            "avg_confidence": sum(p.confidence for p in proofs) / len(proofs) if proofs else 0.0,
            "avg_generation_time": sum(p.generation_time for p in proofs) / len(proofs) if proofs else 0.0,
            "total_execution_time": execution_time,
            "proofs_per_second": len(proofs) / execution_time if execution_time > 0 else 0.0
        }

    async def execute_hierarchical(self, task: LeanMDAPTask) -> LeanProof:
        """
        Execute task with hierarchical decomposition

        Args:
            task: Task to execute

        Returns:
            Best proof
        """
        if not task.enable_decomposition:
            # No decomposition, execute normally
            result = self.orchestrate_proof_generation(task)
            return result.best_proof

        # Implement hierarchical decomposition
        return await self._execute_hierarchical_decomposition(task)

    async def _execute_hierarchical_decomposition(
        self,
        task: LeanMDAPTask
    ) -> LeanProof:
        """
        Execute hierarchical decomposition for complex theorems.

        Breaks down complex theorems into sub-theorems, proves each,
        then combines the results.

        Args:
            task: Task to execute with decomposition

        Returns:
            Combined proof from sub-components
        """
        logger.info(f"Starting hierarchical decomposition for task {task.task_id}")

        # Step 1: Decompose the theorem into lemmas
        sub_theorems = await self._decompose_theorem(task)

        if not sub_theorems:
            # Decomposition failed, fall back to normal execution
            logger.warning("Theorem decomposition failed, falling back to normal execution")
            result = self.orchestrate_proof_generation(task)
            return result.best_proof

        # Step 2: Generate proofs for each sub-theorem
        sub_proofs: List[LeanProof] = []
        for i, sub_thm in enumerate(sub_theorems):
            logger.info(f"Generating proof for sub-theorem {i+1}/{len(sub_theorems)}")

            # Create sub-task
            sub_task = LeanMDAPTask(
                task_id=f"{task.task_id}_sub_{i}",
                description=sub_thm.get("description", f"Sub-theorem {i+1}"),
                theorem_statement=sub_thm.get("statement", ""),
                domain=task.domain,
                enable_decomposition=False,  # No nested decomposition
                max_retries=task.max_retries,
                target_success_rate=task.target_success_rate
            )

            # Generate proof for sub-theorem
            sub_result = self.orchestrate_proof_generation(sub_task)

            if sub_result.success and sub_result.best_proof:
                sub_proofs.append(sub_result.best_proof)
            else:
                logger.warning(f"Failed to prove sub-theorem {i+1}")

        # Step 3: Combine sub-proofs into main proof
        if sub_proofs:
            combined_proof = self._combine_sub_proofs(
                task.theorem_statement,
                sub_theorems,
                sub_proofs
            )
            logger.info(f"Hierarchical decomposition completed: {len(sub_proofs)} sub-proofs combined")
            return combined_proof
        else:
            # No sub-proofs succeeded, fall back to normal execution
            logger.warning("No sub-proofs succeeded, falling back to normal execution")
            result = self.orchestrate_proof_generation(task)
            return result.best_proof

    async def _decompose_theorem(
        self,
        task: LeanMDAPTask
    ) -> List[Dict[str, str]]:
        """
        Decompose a theorem into sub-theorems/lemmas.

        Args:
            task: Task containing the theorem to decompose

        Returns:
            List of sub-theorem dictionaries with 'description' and 'statement' keys
        """
        # Use heuristics to identify decomposition opportunities
        theorem = task.theorem_statement

        # Check for common patterns that suggest decomposition
        sub_theorems = []

        # Pattern 1: Conjunction in goal (A ∧ B)
        if "∧" in theorem or "and" in theorem.lower():
            # Could split into proving each conjunct separately
            # For now, use simple heuristic decomposition
            sub_theorems.append({
                "description": "First conjunct",
                "statement": f"Lemma 1 for: {theorem[:100]}..."
            })
            sub_theorems.append({
                "description": "Second conjunct",
                "statement": f"Lemma 2 for: {theorem[:100]}..."
            })

        # Pattern 2: Universal quantifiers (∀)
        elif "∀" in theorem or "forall" in theorem.lower():
            # Could use induction or case analysis
            sub_theorems.append({
                "description": "Base case",
                "statement": f"Base case for: {theorem[:100]}..."
            })
            sub_theorems.append({
                "description": "Inductive step",
                "statement": f"Inductive step for: {theorem[:100]}..."
            })

        # Pattern 3: Complex implication chain
        elif theorem.count("→") > 2 or theorem.count("->") > 2:
            # Break into intermediate lemmas
            sub_theorems.append({
                "description": "Intermediate result 1",
                "statement": f"Lemma 1 for implication chain"
            })
            sub_theorems.append({
                "description": "Main result",
                "statement": theorem
            })

        # Default: No decomposition possible with heuristics
        if not sub_theorems:
            return []

        return sub_theorems

    def _combine_sub_proofs(
        self,
        main_theorem: str,
        sub_theorems: List[Dict[str, str]],
        sub_proofs: List[LeanProof]
    ) -> LeanProof:
        """
        Combine sub-proofs into a single proof for the main theorem.

        Args:
            main_theorem: The original theorem statement
            sub_theorems: List of sub-theorem definitions
            sub_proofs: List of proofs for each sub-theorem

        Returns:
            Combined proof
        """
        # Combine all tactics from sub-proofs
        combined_tactics = []
        for proof in sub_proofs:
            combined_tactics.extend(proof.tactics_used)

        # Combine Lean code
        lean_code_lines = [f"theorem main_result : {main_theorem} := by"]

        # Add lemma statements
        for i, (sub_thm, proof) in enumerate(zip(sub_theorems, sub_proofs)):
            lean_code_lines.append(f"  -- {sub_thm.get('description', f'Lemma {i+1}')}")
            for tactic in proof.tactics_used:
                lean_code_lines.append(f"  {tactic}")

        combined_lean_code = "\n".join(lean_code_lines)

        # Calculate combined confidence
        avg_confidence = sum(p.confidence for p in sub_proofs) / len(sub_proofs)

        # Verification status (all must be verified)
        all_verified = all(p.verification_status for p in sub_proofs)

        return LeanProof(
            theorem_name="main_result",
            lean_code=combined_lean_code,
            confidence=avg_confidence,
            tactics_used=combined_tactics,
            verification_status=all_verified,
            strategy_used=ProofStrategy.HYBRID,
            agent_id="hierarchical_decomposition"
        )


# =============================================================================
# LEAN MDAP RESULT
# =============================================================================

@dataclass
class LeanMDAPResult:
    """Result of Lean MDAP proof generation"""

    task_id: str
    execution_id: str
    theorem_statement: str
    domain: LeanDomain
    success: bool
    best_proof: LeanProof
    all_proofs: List[LeanProof]
    agents_used: List[str]
    execution_time: float
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    step_results: Dict[str, List[LeanProof]] = field(default_factory=dict)
    voting_statistics: Dict[str, Any] = field(default_factory=dict)
    red_flags: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def avg_confidence(self) -> float:
        """Average confidence of all proofs"""
        if not self.all_proofs:
            return 0.0
        return sum(p.confidence for p in self.all_proofs) / len(self.all_proofs)

    @property
    def num_proofs(self) -> int:
        """Total number of proofs generated"""
        return len(self.all_proofs)

    @property
    def num_verified(self) -> int:
        """Number of verified proofs"""
        return sum(1 for p in self.all_proofs if p.verification_status)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON"""
        return json.dumps(self.to_dict(), indent=2, default=str)


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    """Manages checkpointing for long-running tasks"""

    def __init__(self, config: LeanMDAPConfig):
        """
        Initialize checkpoint manager

        Args:
            config: Lean MDAP configuration
        """
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, task_id: str) -> Path:
        """Get checkpoint file path for task"""
        return self.checkpoint_dir / f"{task_id}.checkpoint"

    def save(
        self,
        task_id: str,
        step: LeanMDAPStep,
        proofs: List[LeanProof],
        trace: List[Dict[str, Any]]
    ) -> None:
        """
        Save checkpoint

        Args:
            task_id: Task ID
            step: Current step
            proofs: Generated proofs so far
            trace: Execution trace
        """
        checkpoint_path = self.get_checkpoint_path(task_id)

        checkpoint_data = {
            "task_id": task_id,
            "step_id": step.step_id,
            "proofs": [p.to_dict() for p in proofs],
            "trace": trace,
            "timestamp": time.time()
        }

        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logger.debug(f"Saved checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load(self, task_id: str) -> Optional['LeanMDAPResult']:
        """
        Load checkpoint

        Args:
            task_id: Task ID

        Returns:
            LeanMDAPResult if checkpoint exists, None otherwise
        """
        checkpoint_path = self.get_checkpoint_path(task_id)

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)

            # Reconstruct proofs
            proofs = [
                LeanProof(**p) if isinstance(p, dict) else p
                for p in checkpoint_data.get("proofs", [])
            ]

            # Create partial result
            result = LeanMDAPResult(
                task_id=task_id,
                execution_id=f"checkpointed_{int(time.time())}",
                theorem_statement="",
                domain=LeanDomain.GENERAL,
                success=False,
                best_proof=proofs[0] if proofs else LeanProof(
                    theorem_name="",
                    lean_code="",
                    confidence=0.0,
                    strategy_used=ProofStrategy.DIRECT,
                    agent_id="checkpoint"
                ),
                all_proofs=proofs,
                agents_used=[],
                execution_time=0.0,
                execution_trace=checkpoint_data.get("trace", [])
            )

            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return result

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def clear(self, task_id: str) -> None:
        """
        Clear checkpoint for task

        Args:
            task_id: Task ID
        """
        checkpoint_path = self.get_checkpoint_path(task_id)

        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                logger.debug(f"Cleared checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to clear checkpoint: {e}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_lean_mdap_config(
    available_agents: Optional[List[str]] = None,
    default_parallel_agents: int = 4,
    voting_strategy: str = "first_k_ahead",
    k_ahead_threshold: int = 3,
    **kwargs
) -> LeanMDAPConfig:
    """
    Create Lean MDAP configuration

    Args:
        available_agents: List of available agent types
        default_parallel_agents: Default number of parallel agents
        voting_strategy: Voting strategy to use
        k_ahead_threshold: K-ahead threshold for voting
        **kwargs: Additional configuration parameters

    Returns:
        LeanMDAPConfig object
    """
    if available_agents is None:
        available_agents = ["evolution", "mcts", "adversarial", "direct"]

    return LeanMDAPConfig(
        available_agents=available_agents,
        default_parallel_agents=default_parallel_agents,
        voting_strategy=VotingStrategy(voting_strategy),
        k_ahead_threshold=k_ahead_threshold,
        **kwargs
    )


def get_lean_mdap_status() -> Dict[str, Any]:
    """
    Get Lean MDAP system status

    Returns:
        Dict with availability and configuration info
    """
    return {
        "mdap_available": MDAP_AVAILABLE,
        "lean_mdap_available": MDAP_AVAILABLE,
        "available_strategies": [s.value for s in ProofStrategy],
        "available_domains": [d.value for d in LeanDomain],
        "voting_strategies": [s.value for s in VotingStrategy],
        "description": "Lean MDAP: Multi-agent, voting-based Lean 4 proof generation"
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ProofStrategy",
    "LeanDomain",
    "VotingStrategy",
    "ProofStatus",

    # Configuration
    "LeanMDAPConfig",
    "create_lean_mdap_config",

    # Core classes
    "LeanMDAPStep",
    "LeanMDAPTask",
    "LeanProof",
    "LeanProofAgent",
    "LeanAgentSelector",
    "LeanMDAPOrchestrator",
    "LeanMDAPResult",

    # Utilities
    "CheckpointManager",
    "get_lean_mdap_status",

    # Constants
    "MDAP_AVAILABLE",
]
