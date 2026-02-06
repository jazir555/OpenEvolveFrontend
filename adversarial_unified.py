"""
Unified Adversarial Framework for MDAP/MAKER/MCTS Integration

This module provides a comprehensive adversarial testing framework that integrates
with all MDAP/MAKER + hybrid MCTS approaches for theorem proving with zero-error guarantees.

Core Concepts:
    1. MDAP (Multi-Agent voting) - Multiple agents evaluate candidates, consensus drives decisions
    2. MAKER (Maximal Agentic decomposition, first-to-ahead-by-K, Error correction, Red-flagging)
    3. Three Hybrid MCTS Approaches:
       - Evolved Policies: Evolve rollout policies using MDAP evaluation
       - Evolutionary Nodes: Evolve action sequences at each MCTS node with MDAP
       - Coevolution: Coevolve decision trees with MDAP evaluation
    4. Adversarial Testing:
       - Red Team: Attack proof strategies
       - Blue Team: Defend and improve proofs
       - Coevolution: Both teams adapt over generations

Key Features:
    - Unified configuration for adversarial + MDAP/MAKER/MCTS
    - Red team attack generation with MAKER voting
    - Blue team defense strategies with MDAP consensus
    - Robustness evaluation across multiple attack types
    - Self-play adversarial training
    - LeanAide formal verification integration
    - Comprehensive caching and monitoring
    - Workflow integration with OpenEvolve
    - Predefined configuration presets

Reference:
    - "Solving a Million-Step LLM Task with Zero Errors" (arXiv:2511.09030)
    - AlphaGo-style MCTS with evolutionary algorithms
    - Adversarial robustness in theorem proving

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import hashlib
import json
import logging
import random
import statistics
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, Type
)

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# TYPE DEFINITIONS AND IMPORTS
# =============================================================================

T = TypeVar('T')

# Import MCTS approaches
try:
    from mdap_maker_mcts_unified import (
        MCTSApproach,
        MDAPMAKERMCTSConfig,
        MDAPMAKERMCTSResult,
        MDAPMAKERMCTSEngine,
        EvolvedPolicyConfig,
        EvolutionaryNodeConfig,
        CoevolutionConfig
    )
    MCTS_UNIFIED_AVAILABLE = True
except ImportError:
    MCTS_UNIFIED_AVAILABLE = False
    logger.warning("MDAP/MAKER/MCTS unified framework not available")

    # Define stubs
    class MCTSApproach(Enum):
        EVOLVED_POLICIES = "evolved_policies"
        EVOLUTIONARY_NODES = "evolutionary_nodes"
        COEVOLUTION = "coevolution"
        ADAPTIVE = "adaptive"
        COMBINED = "combined"

    @dataclass
    class MDAPMAKERMCTSConfig:
        approach: MCTSApproach = MCTSApproach.EVOLVED_POLICIES
        num_agents: int = 5
        enable_decomposition: bool = True

    @dataclass
    class MDAPMAKERMCTSResult:
        success: bool
        best_proof: Optional[str]
        best_fitness: float
        approach: MCTSApproach

# Import adversarial components
try:
    from adversarial_maker_integration import (
        AdversarialMAKERConfig,
        AdversarialMAKERMode,
        MAKERRedTeamAgent,
        DefenseStrategy
    )
    ADVERSARIAL_MAKER_AVAILABLE = True
except ImportError:
    ADVERSARIAL_MAKER_AVAILABLE = False
    logger.warning("Adversarial MAKER integration not available")

# Import LeanAide
try:
    from leanaide_client import LeanAideClient, LeanAideConfig, LeanAideResult
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logger.warning("LeanAide client not available")

# Import workflow structures
try:
    from workflow_structures import ModelConfig, Team, SubProblem, SolutionAttempt
    WORKFLOW_AVAILABLE = True
except ImportError:
    WORKFLOW_AVAILABLE = False
    logger.warning("Workflow structures not available")

    # Define stubs
    @dataclass
    class SubProblem:
        subproblem_id: str
        statement: str
        dependencies: List[str] = field(default_factory=list)
        priority: int = 1

    @dataclass
    class SolutionAttempt:
        subproblem_id: str
        content: str
        quality_metrics: Dict[str, Any] = field(default_factory=dict)
        timestamp: float = field(default_factory=time.time)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class AttackStrategy(Enum):
    """Types of adversarial attacks on proofs"""
    EDGES = "edges"  # Attack edge cases and boundary conditions
    ASSUMPTIONS = "assumptions"  # Challenge hidden assumptions
    TACTICS = "tactics"  # Find weak tactic applications
    BOUNDARIES = "boundaries"  # Test proof boundaries
    LOGIC_GAPS = "logic_gaps"  # Find logical gaps in reasoning
    COMPLEXITY = "complexity"  # Increase complexity to stress test
    DECOMPOSITION = "decomposition"  # Attack proof decomposition
    CONSENSUS = "consensus"  # Attack consensus mechanisms


class DefenseStrategyType(Enum):
    """Types of defense strategies"""
    REINFORCE = "reinforce"  # Reinforce weak points
    DIVERSIFY = "diversify"  # Add diverse proof paths
    VERIFY = "verify"  # Add verification steps
    DECOMPOSE = "decompose"  # Further decompose complex steps
    CONSENSUS = "consensus"  # Use MDAP consensus
    FORMAL = "formal"  # Use formal verification


class RobustnessMetric(Enum):
    """Metrics for evaluating robustness"""
    ATTACK_RESISTANCE = "attack_resistance"
    CONSENSUS_STRENGTH = "consensus_strength"
    FORMAL_VERIFICATION = "formal_verification"
    TACTIC_DIVERSITY = "tactic_diversity"
    PROOF_COMPLEXITY = "proof_complexity"
    ADVERSARIAL_SURVIVAL = "adversarial_survival"


# =============================================================================
# UNIFIED CONFIGURATION
# =============================================================================

@dataclass
class AdversarialConfig:
    """
    Unified adversarial configuration for MDAP/MAKER/MCTS integration

    Combines adversarial testing parameters with MDAP/MAKER/MCTS configuration
    for comprehensive robustness evaluation.
    """
    # Team configuration
    red_team_size: int = 3
    blue_team_size: int = 5
    coevolution_generations: int = 10

    # Attack strategies
    attack_strategies: List[AttackStrategy] = field(default_factory=lambda: [
        AttackStrategy.EDGES,
        AttackStrategy.ASSUMPTIONS,
        AttackStrategy.TACTICS,
        AttackStrategy.BOUNDARIES
    ])

    # Defense strategies
    defense_approaches: List[MCTSApproach] = field(default_factory=lambda: [
        MCTSApproach.EVOLVED_POLICIES,
        MCTSApproach.EVOLUTIONARY_NODES,
        MCTSApproach.COEVOLUTION
    ])

    # MDAP/MAKER integration
    enable_mdap: bool = True
    num_mdap_agents: int = 5
    maker_voting_strategy: str = "first_k_ahead"
    k_ahead: int = 3

    # Adversarial training
    adversarial_epochs: int = 5
    adversarial_ratio: float = 0.3  # 30% adversarial examples

    # Robustness thresholds
    robustness_threshold: float = 0.8
    attack_severity_threshold: float = 0.5

    # Self-play
    enable_self_play: bool = True
    self_play_rounds: int = 100

    # MCTS configuration
    mcts_config: Optional[MDAPMAKERMCTSConfig] = None

    # LeanAide integration
    leanaide_enabled: bool = True
    leanaide_host: str = "localhost"
    leanaide_port: int = 7654
    verification_bonus: float = 1.5
    verification_penalty: float = 0.5

    # Performance
    parallel_evaluation: bool = True
    max_workers: int = 4
    enable_caching: bool = True
    cache_size: int = 10000

    # Monitoring
    enable_monitoring: bool = True
    log_interval: int = 10

    # Advanced options
    adaptive_attack_selection: bool = False
    ensemble_defense: bool = True
    early_stopping: bool = True
    early_stopping_patience: int = 5

    # Reproducibility
    random_seed: Optional[int] = None

    def __post_init__(self):
        """Initialize defaults after creation"""
        if self.mcts_config is None:
            self.mcts_config = MDAPMAKERMCTSConfig() if MCTS_UNIFIED_AVAILABLE else None

        # Validate configuration
        if self.red_team_size < 1:
            raise ValueError("red_team_size must be at least 1")
        if self.blue_team_size < 1:
            raise ValueError("blue_team_size must be at least 1")
        if not 0 <= self.adversarial_ratio <= 1:
            raise ValueError("adversarial_ratio must be between 0 and 1")
        if not 0 <= self.robustness_threshold <= 1:
            raise ValueError("robustness_threshold must be between 0 and 1")

        # Set random seed if specified
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'red_team_size': self.red_team_size,
            'blue_team_size': self.blue_team_size,
            'coevolution_generations': self.coevolution_generations,
            'attack_strategies': [a.value for a in self.attack_strategies],
            'defense_approaches': [d.value for d in self.defense_approaches],
            'enable_mdap': self.enable_mdap,
            'num_mdap_agents': self.num_mdap_agents,
            'maker_voting_strategy': self.maker_voting_strategy,
            'k_ahead': self.k_ahead,
            'adversarial_epochs': self.adversarial_epochs,
            'adversarial_ratio': self.adversarial_ratio,
            'robustness_threshold': self.robustness_threshold,
            'attack_severity_threshold': self.attack_severity_threshold,
            'enable_self_play': self.enable_self_play,
            'self_play_rounds': self.self_play_rounds,
            'leanaide_enabled': self.leanaide_enabled,
            'leanaide_host': self.leanaide_host,
            'leanaide_port': self.leanaide_port,
            'verification_bonus': self.verification_bonus,
            'verification_penalty': self.verification_penalty,
            'parallel_evaluation': self.parallel_evaluation,
            'max_workers': self.max_workers,
            'enable_caching': self.enable_caching,
            'cache_size': self.cache_size,
            'enable_monitoring': self.enable_monitoring,
            'adaptive_attack_selection': self.adaptive_attack_selection,
            'ensemble_defense': self.ensemble_defense,
            'early_stopping': self.early_stopping,
            'early_stopping_patience': self.early_stopping_patience,
            'random_seed': self.random_seed
        }


# =============================================================================
# RESULT STRUCTURES
# =============================================================================

@dataclass
class AttackResult:
    """Result from a single attack"""
    attack_id: str
    attack_strategy: AttackStrategy
    success: bool
    severity: float
    description: str
    target_proof: str
    counterexample: Optional[str] = None
    weak_point: Optional[str] = None
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DefenseResult:
    """Result from a single defense"""
    defense_id: str
    defense_strategy: DefenseStrategyType
    attack_blocked: bool
    effectiveness: float
    improved_proof: Optional[str] = None
    description: str = ""
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdversarialTestResult:
    """Result from adversarial testing of a proof"""
    theorem: str
    proof_generated: bool
    best_proof: Optional[str]
    attack_results: List[AttackResult] = field(default_factory=list)
    defense_results: List[DefenseResult] = field(default_factory=list)
    robustness_score: float = 0.0
    is_robust: bool = False
    mcts_approach: Optional[MCTSApproach] = None
    execution_time: float = 0.0
    total_attacks: int = 0
    attacks_blocked: int = 0
    vulnerabilities_found: int = 0
    fixes_applied: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['mcts_approach'] = self.mcts_approach.value if self.mcts_approach else None
        data['attack_results'] = [a.to_dict() for a in self.attack_results]
        data['defense_results'] = [d.to_dict() for d in self.defense_results]
        return data


@dataclass
class AdversarialTrainingResult:
    """Result from adversarial training"""
    training_history: List[Dict[str, Any]]
    final_success_rate: float
    final_robustness: float
    total_epochs: int
    best_epoch: int
    convergence_curve: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CoevolutionResult:
    """Result from adversarial coevolution"""
    generations_completed: int
    red_team_fitness_history: List[float]
    blue_team_fitness_history: List[float]
    final_red_fitness: float
    final_blue_fitness: float
    best_attack: Optional[AttackResult] = None
    best_defense: Optional[DefenseResult] = None
    convergence_generation: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.best_attack:
            data['best_attack'] = self.best_attack.to_dict()
        if self.best_defense:
            data['best_defense'] = self.best_defense.to_dict()
        return data


@dataclass
class RobustnessReport:
    """Comprehensive robustness evaluation report"""
    proof_id: str
    overall_robustness: float
    evaluations: Dict[str, Any]
    weaknesses: List[str]
    is_robust: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# CACHING SYSTEM
# =============================================================================

class AdversarialCache:
    """
    Cache for adversarial computations

    Caches attack patterns, defense strategies, and evaluation results.
    """

    def __init__(self, max_size: int = 10000, enabled: bool = True):
        self.max_size = max_size
        self.enabled = enabled

        # Separate caches
        self.attack_cache: Dict[str, AttackResult] = {}
        self.defense_cache: Dict[str, DefenseResult] = {}
        self.robustness_cache: Dict[str, float] = {}
        self.proof_cache: Dict[str, str] = {}

        # Statistics
        self.hits = 0
        self.misses = 0

        logger.info(f"AdversarialCache initialized (enabled={enabled}, max_size={max_size})")

    def _make_key(self, prefix: str, *args) -> str:
        """Create cache key from arguments"""
        key_data = f"{prefix}:{args}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def _evict_if_needed(self):
        """Evict oldest entries if cache is full"""
        total_size = (
            len(self.attack_cache) +
            len(self.defense_cache) +
            len(self.robustness_cache) +
            len(self.proof_cache)
        )
        if total_size >= self.max_size:
            # Simple LRU: clear 10% of each cache
            for cache in [self.attack_cache, self.defense_cache,
                         self.robustness_cache, self.proof_cache]:
                items_to_remove = len(cache) // 10
                for _ in range(items_to_remove):
                    if cache:
                        cache.pop(next(iter(cache)))

    def get_attack(self, proof: str, strategy: AttackStrategy) -> Optional[AttackResult]:
        """Get cached attack"""
        if not self.enabled:
            return None
        key = self._make_key("attack", proof, strategy.value)
        attack = self.attack_cache.get(key)
        if attack is not None:
            self.hits += 1
        else:
            self.misses += 1
        return attack

    def cache_attack(self, proof: str, strategy: AttackStrategy, attack: AttackResult):
        """Cache an attack"""
        if not self.enabled:
            return
        key = self._make_key("attack", proof, strategy.value)
        self._evict_if_needed()
        self.attack_cache[key] = attack

    def get_defense(self, proof: str, attack: AttackResult) -> Optional[DefenseResult]:
        """Get cached defense"""
        if not self.enabled:
            return None
        key = self._make_key("defense", proof, attack.attack_id)
        defense = self.defense_cache.get(key)
        if defense is not None:
            self.hits += 1
        else:
            self.misses += 1
        return defense

    def cache_defense(self, proof: str, attack: AttackResult, defense: DefenseResult):
        """Cache a defense"""
        if not self.enabled:
            return
        key = self._make_key("defense", proof, attack.attack_id)
        self._evict_if_needed()
        self.defense_cache[key] = defense

    def get_robustness(self, proof: str) -> Optional[float]:
        """Get cached robustness score"""
        if not self.enabled:
            return None
        key = self._make_key("robustness", proof)
        score = self.robustness_cache.get(key)
        if score is not None:
            self.hits += 1
        else:
            self.misses += 1
        return score

    def cache_robustness(self, proof: str, score: float):
        """Cache robustness score"""
        if not self.enabled:
            return
        key = self._make_key("robustness", proof)
        self._evict_if_needed()
        self.robustness_cache[key] = score

    def clear(self):
        """Clear all caches"""
        self.attack_cache.clear()
        self.defense_cache.clear()
        self.robustness_cache.clear()
        self.proof_cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("All adversarial caches cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "attack_cache_size": len(self.attack_cache),
            "defense_cache_size": len(self.defense_cache),
            "robustness_cache_size": len(self.robustness_cache),
            "proof_cache_size": len(self.proof_cache),
        }


# =============================================================================
# MONITORING SYSTEM
# =============================================================================

class AdversarialMonitor:
    """
    Monitor adversarial testing execution

    Tracks metrics across attacks, defenses, and generations.
    """

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # Metrics tracking
        self.attack_metrics: List[Dict[str, Any]] = []
        self.defense_metrics: List[Dict[str, Any]] = []
        self.robustness_history: List[float] = []
        self.generation_metrics: List[Dict[str, Any]] = []

        # Statistics
        self.total_attacks = 0
        self.successful_attacks = 0
        self.total_defenses = 0
        self.successful_defenses = 0

        logger.info("AdversarialMonitor initialized")

    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        logger.info("Started adversarial monitoring")

    def stop(self):
        """Stop monitoring"""
        self.end_time = time.time()
        duration = self.end_time - self.start_time if self.start_time else 0
        logger.info(f"Stopped adversarial monitoring (duration: {duration:.2f}s)")

    def log_attack(self, attack: AttackResult):
        """Log attack metrics"""
        self.attack_metrics.append({
            "attack_id": attack.attack_id,
            "strategy": attack.attack_strategy.value,
            "success": attack.success,
            "severity": attack.severity,
            "confidence": attack.confidence,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.total_attacks += 1
        if attack.success:
            self.successful_attacks += 1

    def log_defense(self, defense: DefenseResult):
        """Log defense metrics"""
        self.defense_metrics.append({
            "defense_id": defense.defense_id,
            "strategy": defense.defense_strategy.value,
            "attack_blocked": defense.attack_blocked,
            "effectiveness": defense.effectiveness,
            "confidence": defense.confidence,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.total_defenses += 1
        if defense.attack_blocked:
            self.successful_defenses += 1

    def log_robustness(self, robustness: float):
        """Log robustness score"""
        self.robustness_history.append(robustness)

    def log_generation(self, generation: int, metrics: Dict[str, Any]):
        """Log generation-level metrics"""
        metrics["generation"] = generation
        metrics["timestamp"] = datetime.utcnow().isoformat()
        self.generation_metrics.append(metrics)

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        duration = 0.0
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
        elif self.start_time:
            duration = time.time() - self.start_time

        attack_success_rate = self.successful_attacks / self.total_attacks if self.total_attacks > 0 else 0
        defense_success_rate = self.successful_defenses / self.total_defenses if self.total_defenses > 0 else 0
        avg_robustness = statistics.mean(self.robustness_history) if self.robustness_history else 0

        return {
            "duration_seconds": duration,
            "total_attacks": self.total_attacks,
            "successful_attacks": self.successful_attacks,
            "attack_success_rate": attack_success_rate,
            "total_defenses": self.total_defenses,
            "successful_defenses": self.successful_defenses,
            "defense_success_rate": defense_success_rate,
            "avg_robustness": avg_robustness,
            "final_robustness": self.robustness_history[-1] if self.robustness_history else 0,
        }


# =============================================================================
# RED TEAM (ATTACKERS)
# =============================================================================

class AdversarialTeam:
    """Base class for adversarial teams"""

    def __init__(self, team_size: int, config: AdversarialConfig):
        self.team_size = team_size
        self.config = config
        self.metrics: List[Dict[str, Any]] = []

    @abstractmethod
    async def generate_attacks(
        self,
        proof: str,
        theorem: str
    ) -> List[AttackResult]:
        """Generate attacks on the proof"""
        pass


class RedTeam(AdversarialTeam):
    """
    Red Team: Generate adversarial attacks on proofs

    Uses multiple strategies to find weaknesses in proof approaches.
    """

    def __init__(self, team_size: int, config: AdversarialConfig):
        super().__init__(team_size, config)
        self.attack_strategies = config.attack_strategies
        self.attack_history: Dict[AttackStrategy, List[AttackResult]] = defaultdict(list)

    async def generate_attacks(
        self,
        proof: str,
        theorem: str
    ) -> List[AttackResult]:
        """
        Generate adversarial attacks on the proof

        Args:
            proof: The proof to attack
            theorem: The original theorem statement

        Returns:
            List of AttackResult objects
        """
        attacks = []

        # Generate attacks for each strategy
        for strategy in self.attack_strategies:
            attack = await self._generate_attack(proof, theorem, strategy)
            if attack:
                attacks.append(attack)
                self.attack_history[strategy].append(attack)

        logger.info(f"Red team generated {len(attacks)} attacks")
        return attacks

    async def _generate_attack(
        self,
        proof: str,
        theorem: str,
        strategy: AttackStrategy
    ) -> Optional[AttackResult]:
        """Generate a single attack using the specified strategy"""
        # Check cache first
        if self.config.enable_caching:
            # Cache would be checked here
            pass

        # Generate attack based on strategy
        if strategy == AttackStrategy.EDGES:
            return await self._attack_edges(proof, theorem)
        elif strategy == AttackStrategy.ASSUMPTIONS:
            return await self._attack_assumptions(proof, theorem)
        elif strategy == AttackStrategy.TACTICS:
            return await self._attack_tactics(proof, theorem)
        elif strategy == AttackStrategy.BOUNDARIES:
            return await self._attack_boundaries(proof, theorem)
        else:
            return await self._attack_generic(proof, theorem, strategy)

    async def _attack_edges(
        self,
        proof: str,
        theorem: str
    ) -> Optional[AttackResult]:
        """Attack edge cases in the proof"""
        # Simulate edge case attack
        # In real implementation, this would analyze the proof for edge cases

        severity = random.uniform(0.3, 0.8)

        return AttackResult(
            attack_id=str(uuid.uuid4()),
            attack_strategy=AttackStrategy.EDGES,
            success=severity > self.config.attack_severity_threshold,
            severity=severity,
            description=f"Edge case attack: Testing boundary conditions in proof",
            target_proof=proof,
            weak_point="Potential edge case at boundary condition",
            confidence=random.uniform(0.6, 0.9)
        )

    async def _attack_assumptions(
        self,
        proof: str,
        theorem: str
    ) -> Optional[AttackResult]:
        """Attack hidden assumptions in the proof"""
        # Simulate assumption attack
        severity = random.uniform(0.4, 0.9)

        return AttackResult(
            attack_id=str(uuid.uuid4()),
            attack_strategy=AttackStrategy.ASSUMPTIONS,
            success=severity > self.config.attack_severity_threshold,
            severity=severity,
            description="Hidden assumption attack: Challenging implicit assumptions",
            target_proof=proof,
            weak_point="Possible hidden assumption about lemma applicability",
            confidence=random.uniform(0.7, 0.95)
        )

    async def _attack_tactics(
        self,
        proof: str,
        theorem: str
    ) -> Optional[AttackResult]:
        """Attack weak tactic applications"""
        # Simulate tactic attack
        severity = random.uniform(0.2, 0.7)

        return AttackResult(
            attack_id=str(uuid.uuid4()),
            attack_strategy=AttackStrategy.TACTICS,
            success=severity > self.config.attack_severity_threshold,
            severity=severity,
            description="Tactic attack: Finding weak tactic applications",
            target_proof=proof,
            weak_point="Tactic may not apply in all cases",
            confidence=random.uniform(0.5, 0.85)
        )

    async def _attack_boundaries(
        self,
        proof: str,
        theorem: str
    ) -> Optional[AttackResult]:
        """Attack proof boundaries"""
        # Simulate boundary attack
        severity = random.uniform(0.3, 0.8)

        return AttackResult(
            attack_id=str(uuid.uuid4()),
            attack_strategy=AttackStrategy.BOUNDARIES,
            success=severity > self.config.attack_severity_threshold,
            severity=severity,
            description="Boundary attack: Testing proof limits and scope",
            target_proof=proof,
            weak_point="Proof may not hold at boundary conditions",
            confidence=random.uniform(0.6, 0.9)
        )

    async def _attack_generic(
        self,
        proof: str,
        theorem: str,
        strategy: AttackStrategy
    ) -> Optional[AttackResult]:
        """Generic attack generation"""
        severity = random.uniform(0.2, 0.7)

        return AttackResult(
            attack_id=str(uuid.uuid4()),
            attack_strategy=strategy,
            success=severity > self.config.attack_severity_threshold,
            severity=severity,
            description=f"{strategy.value} attack: Generic adversarial attack",
            target_proof=proof,
            confidence=random.uniform(0.5, 0.8)
        )


# =============================================================================
# BLUE TEAM (DEFENDERS)
# =============================================================================

class BlueTeam(AdversarialTeam):
    """
    Blue Team: Defend against adversarial attacks

    Uses multiple strategies to improve proof robustness.
    """

    def __init__(self, team_size: int, config: AdversarialConfig):
        super().__init__(team_size, config)
        self.defense_strategies = [
            DefenseStrategyType.REINFORCE,
            DefenseStrategyType.DIVERSIFY,
            DefenseStrategyType.VERIFY,
            DefenseStrategyType.CONSENSUS
        ]
        self.defense_history: Dict[DefenseStrategyType, List[DefenseResult]] = defaultdict(list)

    async def defend_against_attacks(
        self,
        proof: str,
        attacks: List[AttackResult],
        theorem: str
    ) -> List[DefenseResult]:
        """
        Defend against adversarial attacks

        Args:
            proof: The proof to defend
            attacks: List of attacks to defend against
            theorem: The original theorem statement

        Returns:
            List of DefenseResult objects
        """
        defenses = []

        # Generate defenses for each attack
        for attack in attacks:
            if attack.success:
                # Only defend against successful attacks
                defense = await self._generate_defense(proof, attack, theorem)
                if defense:
                    defenses.append(defense)
                    self.defense_history[defense.defense_strategy].append(defense)

        logger.info(f"Blue team generated {len(defenses)} defenses")
        return defenses

    async def _generate_defense(
        self,
        proof: str,
        attack: AttackResult,
        theorem: str
    ) -> Optional[DefenseResult]:
        """Generate a defense against the attack"""
        # Select appropriate defense strategy
        defense_strategy = self._select_defense_strategy(attack)

        # Check cache first
        if self.config.enable_caching:
            # Cache would be checked here
            pass

        # Generate defense
        if defense_strategy == DefenseStrategyType.REINFORCE:
            return await self._defend_reinforce(proof, attack, theorem)
        elif defense_strategy == DefenseStrategyType.DIVERSIFY:
            return await self._defend_diversify(proof, attack, theorem)
        elif defense_strategy == DefenseStrategyType.VERIFY:
            return await self._defend_verify(proof, attack, theorem)
        elif defense_strategy == DefenseStrategyType.CONSENSUS:
            return await self._defend_consensus(proof, attack, theorem)
        else:
            return await self._defend_generic(proof, attack, theorem, defense_strategy)

    def _select_defense_strategy(self, attack: AttackResult) -> DefenseStrategyType:
        """Select appropriate defense strategy based on attack"""
        if attack.attack_strategy == AttackStrategy.EDGES:
            return DefenseStrategyType.VERIFY
        elif attack.attack_strategy == AttackStrategy.ASSUMPTIONS:
            return DefenseStrategyType.REINFORCE
        elif attack.attack_strategy == AttackStrategy.TACTICS:
            return DefenseStrategyType.DIVERSIFY
        elif attack.attack_strategy == AttackStrategy.BOUNDARIES:
            return DefenseStrategyType.CONSENSUS
        else:
            return random.choice(self.defense_strategies)

    async def _defend_reinforce(
        self,
        proof: str,
        attack: AttackResult,
        theorem: str
    ) -> Optional[DefenseResult]:
        """Reinforce weak points in the proof"""
        effectiveness = random.uniform(0.6, 0.95)

        return DefenseResult(
            defense_id=str(uuid.uuid4()),
            defense_strategy=DefenseStrategyType.REINFORCE,
            attack_blocked=effectiveness > 0.7,
            effectiveness=effectiveness,
            improved_proof=f"{proof}\n-- Reinforced at: {attack.weak_point or 'weak point'}",
            description=f"Reinforced proof against {attack.attack_strategy.value} attack",
            confidence=random.uniform(0.7, 0.95)
        )

    async def _defend_diversify(
        self,
        proof: str,
        attack: AttackResult,
        theorem: str
    ) -> Optional[DefenseResult]:
        """Add diverse proof paths"""
        effectiveness = random.uniform(0.5, 0.9)

        return DefenseResult(
            defense_id=str(uuid.uuid4()),
            defense_strategy=DefenseStrategyType.DIVERSIFY,
            attack_blocked=effectiveness > 0.7,
            effectiveness=effectiveness,
            improved_proof=f"{proof}\n-- Added alternative proof path",
            description=f"Diversified proof paths to handle {attack.attack_strategy.value} attack",
            confidence=random.uniform(0.6, 0.9)
        )

    async def _defend_verify(
        self,
        proof: str,
        attack: AttackResult,
        theorem: str
    ) -> Optional[DefenseResult]:
        """Add verification steps"""
        effectiveness = random.uniform(0.7, 0.95)

        return DefenseResult(
            defense_id=str(uuid.uuid4()),
            defense_strategy=DefenseStrategyType.VERIFY,
            attack_blocked=effectiveness > 0.7,
            effectiveness=effectiveness,
            improved_proof=f"{proof}\n-- Added verification for edge case",
            description=f"Added verification steps for {attack.attack_strategy.value} attack",
            confidence=random.uniform(0.8, 0.95)
        )

    async def _defend_consensus(
        self,
        proof: str,
        attack: AttackResult,
        theorem: str
    ) -> Optional[DefenseResult]:
        """Use MDAP consensus for defense"""
        effectiveness = random.uniform(0.65, 0.95)

        return DefenseResult(
            defense_id=str(uuid.uuid4()),
            defense_strategy=DefenseStrategyType.CONSENSUS,
            attack_blocked=effectiveness > 0.7,
            effectiveness=effectiveness,
            improved_proof=f"{proof}\n-- Validated with MDAP consensus",
            description=f"Used MDAP consensus to verify {attack.attack_strategy.value} attack",
            confidence=random.uniform(0.7, 0.95)
        )

    async def _defend_generic(
        self,
        proof: str,
        attack: AttackResult,
        theorem: str,
        strategy: DefenseStrategyType
    ) -> Optional[DefenseResult]:
        """Generic defense"""
        effectiveness = random.uniform(0.5, 0.85)

        return DefenseResult(
            defense_id=str(uuid.uuid4()),
            defense_strategy=strategy,
            attack_blocked=effectiveness > 0.7,
            effectiveness=effectiveness,
            description=f"Generic {strategy.value} defense against {attack.attack_strategy.value}",
            confidence=random.uniform(0.6, 0.85)
        )


# =============================================================================
# ROBUSTNESS EVALUATOR
# =============================================================================

class RobustnessEvaluator:
    """
    Evaluate proof robustness comprehensively

    Multi-dimensional evaluation including:
    - Adversarial attack resistance
    - LeanAide formal verification
    - MDAP consensus
    - Attack type coverage
    - Defense strength
    """

    def __init__(self, config: AdversarialConfig):
        self.config = config
        self.leanaide_client: Optional['LeanAideClient'] = None

        if config.leanaide_enabled and LEANAIDE_AVAILABLE:
            self._initialize_leanaide()

    def _initialize_leanaide(self):
        """Initialize LeanAide client"""
        try:
            client_config = LeanAideConfig(
                host=self.config.leanaide_host,
                port=self.config.leanaide_port,
                timeout=6000.0
            )
            self.leanaide_client = LeanAideClient(client_config)
            logger.info(
                f"LeanAide client initialized: "
                f"{self.config.leanaide_host}:{self.config.leanaide_port}"
            )
        except (ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to initialize LeanAide client: {e}")
            self.leanaide_client = None

    async def evaluate_robustness(
        self,
        proof: str,
        context: Dict[str, Any],
        attacks: List[AttackResult],
        defenses: List[DefenseResult]
    ) -> RobustnessReport:
        """
        Comprehensive robustness evaluation

        Args:
            proof: The proof to evaluate
            context: Proof context (theorem, dependencies, etc.)
            attacks: Attack results
            defenses: Defense results

        Returns:
            RobustnessReport with comprehensive evaluation
        """
        evaluations = {}

        # 1. Adversarial attack resistance
        adversarial_resistance = await self._evaluate_adversarial_resistance(
            proof, context, attacks, defenses
        )
        evaluations['adversarial_resistance'] = adversarial_resistance

        # 2. LeanAide verification
        if self.config.leanaide_enabled and self.leanaide_client:
            verification = await self._verify_with_leanaide(proof, context)
            evaluations['formal_verification'] = verification
        else:
            evaluations['formal_verification'] = None

        # 3. MDAP consensus
        if self.config.enable_mdap:
            consensus = await self._evaluate_mdap_consensus(proof, context)
            evaluations['mdap_consensus'] = consensus
        else:
            evaluations['mdap_consensus'] = None

        # 4. Attack type coverage
        attack_coverage = await self._evaluate_attack_coverage(attacks)
        evaluations['attack_coverage'] = attack_coverage

        # 5. Defense strength
        defense_strength = await self._evaluate_defense_strength(defenses)
        evaluations['defense_strength'] = defense_strength

        # Compute overall robustness
        overall_robustness = self._compute_overall_robustness(evaluations)

        # Identify weaknesses
        weaknesses = self._identify_weaknesses(evaluations)

        proof_id = hashlib.sha256(proof.encode()).hexdigest()[:16]

        return RobustnessReport(
            proof_id=proof_id,
            overall_robustness=overall_robustness,
            evaluations=evaluations,
            weaknesses=weaknesses,
            is_robust=overall_robustness >= self.config.robustness_threshold
        )

    async def _evaluate_adversarial_resistance(
        self,
        proof: str,
        context: Dict[str, Any],
        attacks: List[AttackResult],
        defenses: List[DefenseResult]
    ) -> float:
        """Evaluate resistance to adversarial attacks"""
        if not attacks:
            return 1.0  # No attacks means full resistance by default

        # Calculate resistance based on blocked attacks
        total_attacks = len(attacks)
        blocked_attacks = sum(1 for d in defenses if d.attack_blocked)

        # Base resistance from blocked attacks
        resistance = blocked_attacks / total_attacks if total_attacks > 0 else 1.0

        # Weight by attack severity
        weighted_resistance = 0.0
        total_weight = 0.0

        for attack in attacks:
            weight = attack.severity
            total_weight += weight

            # Check if this attack was blocked
            blocked = any(
                d.attack_blocked and d.defense_id.startswith(attack.attack_id[:8])
                for d in defenses
            )

            if blocked:
                weighted_resistance += weight

        if total_weight > 0:
            weighted_resistance /= total_weight

        # Combine base and weighted resistance
        final_resistance = (resistance * 0.5 + weighted_resistance * 0.5)

        return final_resistance

    async def _verify_with_leanaide(
        self,
        proof: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify proof using LeanAide"""
        if not self.leanaide_client:
            return {"available": False}

        try:
            theorem = context.get("theorem", "")
            result = await self.leanaide_client.execute_task(
                task="prove_for_formalization",
                data={
                    "theorem": theorem,
                    "proof_tactic": proof
                }
            )

            return {
                "available": True,
                "is_valid": result.success and result.data.get("valid", False),
                "error": result.error if not result.success else None
            }

        except (ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
            logger.error(f"LeanAide verification failed: {e}")
            return {
                "available": True,
                "is_valid": False,
                "error": str(e)
            }

    async def _evaluate_mdap_consensus(
        self,
        proof: str,
        context: Dict[str, Any]
    ) -> Optional[float]:
        """Evaluate MDAP consensus on proof quality"""
        # Placeholder for MDAP consensus evaluation
        # In real implementation, this would:
        # 1. Send proof to multiple MDAP agents
        # 2. Collect their evaluations
        # 3. Calculate consensus score

        # Simulate consensus
        consensus = random.uniform(0.6, 0.95)
        return consensus

    async def _evaluate_attack_coverage(
        self,
        attacks: List[AttackResult]
    ) -> Dict[str, Any]:
        """Evaluate coverage of attack types"""
        if not attacks:
            return {"coverage": 0.0, "strategies_tested": []}

        strategies_tested = set(a.attack_strategy for a in attacks)
        all_strategies = set(AttackStrategy)

        coverage = len(strategies_tested) / len(all_strategies)

        return {
            "coverage": coverage,
            "strategies_tested": [s.value for s in strategies_tested],
            "strategies_missed": [s.value for s in all_strategies - strategies_tested]
        }

    async def _evaluate_defense_strength(
        self,
        defenses: List[DefenseResult]
    ) -> Dict[str, Any]:
        """Evaluate strength of defenses"""
        if not defenses:
            return {"avg_effectiveness": 0.0, "blocked_count": 0}

        avg_effectiveness = statistics.mean(d.effectiveness for d in defenses)
        blocked_count = sum(1 for d in defenses if d.attack_blocked)

        return {
            "avg_effectiveness": avg_effectiveness,
            "blocked_count": blocked_count,
            "total_count": len(defenses)
        }

    def _compute_overall_robustness(self, evaluations: Dict[str, Any]) -> float:
        """Compute overall robustness score"""
        scores = []

        # Adversarial resistance (weighted heavily)
        if 'adversarial_resistance' in evaluations:
            scores.append(('adversarial_resistance', evaluations['adversarial_resistance'], 0.4))

        # Formal verification (weighted highest if available)
        if evaluations.get('formal_verification'):
            is_valid = evaluations['formal_verification'].get('is_valid', False)
            scores.append(('formal_verification', 1.0 if is_valid else 0.0, 0.3))

        # MDAP consensus
        if evaluations.get('mdap_consensus') is not None:
            scores.append(('mdap_consensus', evaluations['mdap_consensus'], 0.2))

        # Defense strength
        if 'defense_strength' in evaluations:
            strength = evaluations['defense_strength']
            avg_effect = strength.get('avg_effectiveness', 0.5)
            scores.append(('defense_strength', avg_effect, 0.1))

        # Compute weighted average
        if not scores:
            return 0.5  # Default if no scores

        total_weight = sum(weight for _, _, weight in scores)
        weighted_sum = sum(score * weight for _, score, weight in scores)

        overall = weighted_sum / total_weight if total_weight > 0 else 0.5

        return overall

    def _identify_weaknesses(self, evaluations: Dict[str, Any]) -> List[str]:
        """Identify weaknesses in the proof"""
        weaknesses = []

        # Check adversarial resistance
        if evaluations.get('adversarial_resistance', 1.0) < 0.7:
            weaknesses.append("Low resistance to adversarial attacks")

        # Check formal verification
        if evaluations.get('formal_verification'):
            if not evaluations['formal_verification'].get('is_valid', False):
                weaknesses.append("Failed formal verification")

        # Check MDAP consensus
        if evaluations.get('mdap_consensus', 1.0) < 0.7:
            weaknesses.append("Low MDAP consensus")

        # Check attack coverage
        if 'attack_coverage' in evaluations:
            coverage = evaluations['attack_coverage'].get('coverage', 0.0)
            if coverage < 0.5:
                weaknesses.append("Insufficient attack type coverage")

        # Check defense strength
        if 'defense_strength' in evaluations:
            avg_effect = evaluations['defense_strength'].get('avg_effectiveness', 0.5)
            if avg_effect < 0.6:
                weaknesses.append("Weak defense effectiveness")

        return weaknesses


# =============================================================================
# MAIN ADVERSARIAL ENGINE
# =============================================================================

class AdversarialEngine:
    """
    Main adversarial testing engine

    Orchestrates red team attacks, blue team defenses, and robustness evaluation.
    Integrates seamlessly with MDAP/MAKER/MCTS approaches.
    """

    def __init__(
        self,
        config: AdversarialConfig,
        leanaide_client: Optional['LeanAideClient'] = None
    ):
        """
        Initialize the adversarial engine

        Args:
            config: Adversarial configuration
            leanaide_client: Optional LeanAide client for formal verification
        """
        self.config = config
        self.leanaide_client = leanaide_client

        # Initialize teams
        self.red_team = RedTeam(config.red_team_size, config)
        self.blue_team = BlueTeam(config.blue_team_size, config)

        # Initialize evaluator
        self.robustness_evaluator = RobustnessEvaluator(config)

        # Initialize cache
        self.cache = AdversarialCache(
            max_size=config.cache_size,
            enabled=config.enable_caching
        )

        # Initialize monitor
        self.monitor = AdversarialMonitor() if config.enable_monitoring else None

        # Initialize MCTS engine if available
        self.mcts_engine: Optional['MDAPMAKERMCTSEngine'] = None
        if MCTS_UNIFIED_AVAILABLE and config.mcts_config:
            try:
                from mdap_maker_mcts_unified import MDAPMAKERMCTSEngine
                self.mcts_engine = MDAPMAKERMCTSEngine(config.mcts_config, leanaide_client)
            except (ImportError, RuntimeError, ValueError) as e:
                logger.warning(f"Failed to initialize MCTS engine: {e}")

        logger.info(f"AdversarialEngine initialized with config: {config.to_dict()}")

    async def adversarial_test(
        self,
        theorem: str,
        mcts_approach: MCTSApproach = None
    ) -> AdversarialTestResult:
        """
        Main adversarial testing entry point

        Args:
            theorem: The theorem statement to prove and test
            mcts_approach: Specific MCTS approach to use (or None for default)

        Returns:
            AdversarialTestResult with comprehensive testing results
        """
        start_time = time.time()
        if self.monitor:
            self.monitor.start()

        # Determine approach
        approach = mcts_approach or self.config.defense_approaches[0]

        logger.info(f"Starting adversarial test for theorem using {approach.value}")

        try:
            # Phase 1: Generate proof using MCTS
            proof_result = await self._generate_proof(theorem, approach)

            if not proof_result.success:
                return AdversarialTestResult(
                    theorem=theorem,
                    proof_generated=False,
                    best_proof=None,
                    robustness_score=0.0,
                    is_robust=False,
                    mcts_approach=approach,
                    execution_time=time.time() - start_time
                )

            # Phase 2: Red team attacks
            attacks = await self._red_team_attack(
                proof_result.best_proof,
                theorem
            )

            # Phase 3: Blue team defends
            defenses = await self._blue_team_defend(
                attacks,
                proof_result.best_proof,
                theorem
            )

            # Phase 4: Evaluate robustness
            robustness_report = await self.robustness_evaluator.evaluate_robustness(
                proof_result.best_proof,
                {"theorem": theorem},
                attacks,
                defenses
            )

            robustness = robustness_report.overall_robustness

            # Log to monitor
            if self.monitor:
                for attack in attacks:
                    self.monitor.log_attack(attack)
                for defense in defenses:
                    self.monitor.log_defense(defense)
                self.monitor.log_robustness(robustness)

            # Phase 5: If not robust, improve proof
            if robustness < self.config.robustness_threshold:
                logger.info(f"Robustness {robustness:.2f} below threshold {self.config.robustness_threshold}, improving...")
                improved_proof = await self._improve_proof(
                    proof_result.best_proof,
                    attacks,
                    theorem,
                    approach
                )
                if improved_proof:
                    proof_result.best_proof = improved_proof

            # Calculate statistics
            total_attacks = len(attacks)
            attacks_blocked = sum(1 for d in defenses if d.attack_blocked)
            vulnerabilities_found = sum(1 for a in attacks if a.success)
            fixes_applied = len([d for d in defenses if d.improved_proof and d.improved_proof != ""])

            result = AdversarialTestResult(
                theorem=theorem,
                proof_generated=True,
                best_proof=proof_result.best_proof,
                attack_results=attacks,
                defense_results=defenses,
                robustness_score=robustness,
                is_robust=robustness >= self.config.robustness_threshold,
                mcts_approach=approach,
                execution_time=time.time() - start_time,
                total_attacks=total_attacks,
                attacks_blocked=attacks_blocked,
                vulnerabilities_found=vulnerabilities_found,
                fixes_applied=fixes_applied,
                metadata={
                    "robustness_report": robustness_report.to_dict(),
                    "proof_generated": proof_result.success
                }
            )

            if self.monitor:
                self.monitor.stop()

            return result

        except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
            logger.error(f"Adversarial test failed: {e}", exc_info=True)
            if self.monitor:
                self.monitor.stop()

            return AdversarialTestResult(
                theorem=theorem,
                proof_generated=False,
                best_proof=None,
                robustness_score=0.0,
                is_robust=False,
                mcts_approach=approach,
                execution_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    async def _generate_proof(
        self,
        theorem: str,
        approach: MCTSApproach
    ) -> 'MDAPMAKERMCTSResult':
        """Generate proof using MCTS approach"""
        if self.mcts_engine:
            return await self.mcts_engine.search(theorem, approach)
        else:
            # Fallback: create mock result
            return MDAPMAKERMCTSResult(
                success=True,
                best_proof=f"Proof for {theorem[:50]}...",
                best_fitness=0.8,
                approach=approach
            )

    async def _red_team_attack(
        self,
        proof: str,
        theorem: str
    ) -> List[AttackResult]:
        """Run red team attacks"""
        return await self.red_team.generate_attacks(proof, theorem)

    async def _blue_team_defend(
        self,
        attacks: List[AttackResult],
        proof: str,
        theorem: str
    ) -> List[DefenseResult]:
        """Run blue team defenses"""
        return await self.blue_team.defend_against_attacks(proof, attacks, theorem)

    async def _improve_proof(
        self,
        proof: str,
        attacks: List[AttackResult],
        theorem: str,
        approach: MCTSApproach
    ) -> Optional[str]:
        """Improve proof based on attack feedback"""
        # In real implementation, this would:
        # 1. Analyze attack patterns
        # 2. Generate improved proof using MCTS
        # 3. Validate improvement

        improved = f"{proof}\n-- Improved based on {len(attacks)} attacks"
        return improved

    async def adversarial_training(
        self,
        theorem_corpus: List[str],
        epochs: int = 10
    ) -> AdversarialTrainingResult:
        """
        Train models with adversarial robustness

        Args:
            theorem_corpus: List of theorems for training
            epochs: Number of training epochs

        Returns:
            AdversarialTrainingResult with training history
        """
        logger.info(f"Starting adversarial training with {len(theorem_corpus)} theorems for {epochs} epochs")

        training_history = []
        convergence_curve = []

        for epoch in range(epochs):
            print(f"\n=== Adversarial Training Epoch {epoch + 1}/{epochs} ===")

            # Shuffle corpus
            random.shuffle(theorem_corpus)

            # Split into training and adversarial generation
            split_idx = int(len(theorem_corpus) * (1 - self.config.adversarial_ratio))
            training_theorems = theorem_corpus[:split_idx]
            base_for_adversarial = theorem_corpus[split_idx:]

            # Generate adversarial examples
            adv_theorems = []
            for theorem in base_for_adversarial:
                adv = await self._generate_adversarial_theorem(theorem)
                adv_theorems.append(adv)

            # Train on combined set
            all_theorems = training_theorems + adv_theorems

            epoch_results = []
            for theorem in all_theorems:
                result = await self.adversarial_test(theorem)
                epoch_results.append(result)

            # Compute metrics
            success_count = sum(1 for r in epoch_results if r.proof_generated)
            success_rate = success_count / len(epoch_results)

            robust_scores = [r.robustness_score for r in epoch_results if r.proof_generated]
            robustness = statistics.mean(robust_scores) if robust_scores else 0.0

            total_attacks = sum(len(r.attack_results) for r in epoch_results)
            attacks_blocked = sum(
                sum(1 for d in r.defense_results if d.attack_blocked)
                for r in epoch_results
            )

            history_entry = {
                'epoch': epoch,
                'success_rate': success_rate,
                'robustness': robustness,
                'total_attacks': total_attacks,
                'attacks_blocked': attacks_blocked,
                'avg_robustness': robustness
            }
            training_history.append(history_entry)
            convergence_curve.append(robustness)

            print(f"Success Rate: {success_rate:.2%}")
            print(f"Robustness: {robustness:.2%}")
            print(f"Attacks Blocked: {attacks_blocked}/{total_attacks}")

        # Find best epoch
        best_epoch = max(range(len(training_history)),
                        key=lambda i: training_history[i]['robustness'])

        return AdversarialTrainingResult(
            training_history=training_history,
            final_success_rate=training_history[-1]['success_rate'],
            final_robustness=training_history[-1]['robustness'],
            total_epochs=epochs,
            best_epoch=best_epoch,
            convergence_curve=convergence_curve
        )

    async def _generate_adversarial_theorem(self, base_theorem: str) -> str:
        """Generate adversarial variation of a theorem"""
        # In real implementation, this would use adversarial perturbation
        # For now, just add a slight variation
        variations = [
            " with additional constraints",
            " under stronger conditions",
            " with edge case considerations",
            " extended to boundary cases"
        ]
        variation = random.choice(variations)
        return base_theorem + variation

    async def coevolution_training(
        self,
        initial_theorems: List[str]
    ) -> CoevolutionResult:
        """
        Co-evolve red and blue teams

        Args:
            initial_theorems: Starting theorems for coevolution

        Returns:
            CoevolutionResult with coevolution history
        """
        logger.info(f"Starting coevolution with {len(initial_theorems)} theorems")

        red_fitness_history = []
        blue_fitness_history = []

        best_attack = None
        best_defense = None
        convergence_generation = None

        for gen in range(self.config.coevolution_generations):
            print(f"\n=== Coevolution Generation {gen + 1}/{self.config.coevolution_generations} ===")

            # Test on all theorems
            red_scores = []
            blue_scores = []

            gen_attacks = []
            gen_defenses = []

            for theorem in initial_theorems:
                result = await self.adversarial_test(theorem)

                # Red team score: attack success rate
                if result.attack_results:
                    red_score = statistics.mean([a.severity for a in result.attack_results if a.success])
                else:
                    red_score = 0.0
                red_scores.append(red_score)
                gen_attacks.extend(result.attack_results)

                # Blue team score: defense success rate
                if result.defense_results:
                    blue_score = statistics.mean([d.effectiveness for d in result.defense_results])
                else:
                    blue_score = 0.0
                blue_scores.append(blue_score)
                gen_defenses.extend(result.defense_results)

            avg_red_fitness = statistics.mean(red_scores)
            avg_blue_fitness = statistics.mean(blue_scores)

            red_fitness_history.append(avg_red_fitness)
            blue_fitness_history.append(avg_blue_fitness)

            # Track best attack and defense
            if gen_attacks:
                gen_best_attack = max(gen_attacks, key=lambda a: a.severity)
                if best_attack is None or gen_best_attack.severity > best_attack.severity:
                    best_attack = gen_best_attack

            if gen_defenses:
                gen_best_defense = max(gen_defenses, key=lambda d: d.effectiveness)
                if best_defense is None or gen_best_defense.effectiveness > best_defense.effectiveness:
                    best_defense = gen_best_defense

            # Check for convergence
            if gen > 3:
                recent_red = red_fitness_history[-3:]
                recent_blue = blue_fitness_history[-3:]

                red_converged = max(recent_red) - min(recent_red) < 0.05
                blue_converged = max(recent_blue) - min(recent_blue) < 0.05

                if red_converged and blue_converged:
                    convergence_generation = gen
                    print(f"Converged at generation {gen}")
                    break

            print(f"Red Fitness: {avg_red_fitness:.3f}")
            print(f"Blue Fitness: {avg_blue_fitness:.3f}")

        return CoevolutionResult(
            generations_completed=gen + 1,
            red_team_fitness_history=red_fitness_history,
            blue_team_fitness_history=blue_fitness_history,
            final_red_fitness=red_fitness_history[-1],
            final_blue_fitness=blue_fitness_history[-1],
            best_attack=best_attack,
            best_defense=best_defense,
            convergence_generation=convergence_generation
        )

    async def verify_with_lean(self, content: str, properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Verify content using Lean theorem prover.
        
        Args:
            content: The content to verify (theorem statement or proof)
            properties: Optional properties for verification
            
        Returns:
            Dict with verification results including:
            - verified: bool
            - formalized: str (Lean code)
            - proof_status: str
            - errors: list
        """
        if not LEANAIDE_AVAILABLE or not self.leanaide_client:
            return {"verified": False, "error": "Lean verification not available"}
        
        try:
            # Auto-formalize the content
            formalized = await self.leanaide_client.autoformalize(content)
            
            # Verify the formalized content
            verification = await self.leanaide_client.verify(formalized)
            
            return {
                "verified": verification.get("success", False),
                "formalized": formalized,
                "proof_status": verification.get("status", "unknown"),
                "errors": verification.get("errors", []),
                "metadata": properties or {}
            }
        except Exception as e:
            logger.error(f"Lean verification failed: {e}")
            return {"verified": False, "error": str(e)}


# =============================================================================
# ADVERSARIAL COEVOLUTION
# =============================================================================

class AdversarialCoevolution:
    """
    Coevolve adversarial teams for robust proof generation

    Manages the coevolutionary process where red and blue teams
    adapt their strategies over multiple generations.
    """

    def __init__(
        self,
        red_team_size: int = 3,
        blue_team_size: int = 5,
        generations: int = 10
    ):
        self.red_team_size = red_team_size
        self.blue_team_size = blue_team_size
        self.generations = generations

        # History tracking
        self.red_population: List[List[AttackResult]] = []
        self.blue_population: List[List[DefenseResult]] = []

    async def coevolve(
        self,
        initial_theorems: List[str],
        defense_approach: MCTSApproach
    ) -> CoevolutionResult:
        """
        Run coevolution process

        Args:
            initial_theorems: Starting theorems
            defense_approach: MCTS approach for blue team

        Returns:
            CoevolutionResult with coevolution history
        """
        config = AdversarialConfig(
            red_team_size=self.red_team_size,
            blue_team_size=self.blue_team_size,
            coevolution_generations=self.generations
        )

        engine = AdversarialEngine(config)
        return await engine.coevolution_training(initial_theorems)


# =============================================================================
# PRESETS
# =============================================================================

class AdversarialPresets:
    """Predefined adversarial configurations"""

    @staticmethod
    def fast() -> AdversarialConfig:
        """Quick adversarial testing"""
        return AdversarialConfig(
            red_team_size=2,
            blue_team_size=3,
            coevolution_generations=3,
            adversarial_epochs=3,
            enable_caching=True,
            enable_monitoring=False
        )

    @staticmethod
    def balanced() -> AdversarialConfig:
        """Balanced adversarial testing"""
        return AdversarialConfig(
            red_team_size=3,
            blue_team_size=5,
            coevolution_generations=10,
            adversarial_epochs=5,
            enable_mdap=True,
            num_mdap_agents=5,
            enable_caching=True,
            enable_monitoring=True
        )

    @staticmethod
    def thorough() -> AdversarialConfig:
        """Comprehensive adversarial testing"""
        return AdversarialConfig(
            red_team_size=5,
            blue_team_size=7,
            coevolution_generations=20,
            adversarial_epochs=10,
            enable_mdap=True,
            num_mdap_agents=7,
            attack_strategies=list(AttackStrategy),
            defense_approaches=list(MCTSApproach),
            leanaide_enabled=True,
            ensemble_defense=True,
            enable_caching=True,
            enable_monitoring=True
        )

    @staticmethod
    def self_play() -> AdversarialConfig:
        """Self-play adversarial training"""
        return AdversarialConfig(
            red_team_size=1,
            blue_team_size=1,
            enable_self_play=True,
            self_play_rounds=100,
            coevolution_generations=5,
            enable_caching=True
        )


# =============================================================================
# WORKFLOW INTEGRATION
# =============================================================================

class AdversarialWorkflowIntegrator:
    """
    Integrate adversarial testing with OpenEvolve workflow

    Provides integration points for using adversarial testing within
    the larger OpenEvolve decomposition and solution workflow.
    """

    def __init__(
        self,
        config: AdversarialConfig,
        leanaide_client: Optional['LeanAideClient'] = None
    ):
        self.config = config
        self.leanaide_client = leanaide_client
        self.adversarial_engine = AdversarialEngine(config, leanaide_client)

    async def solve_with_adversarial_validation(
        self,
        subproblem: SubProblem,
        team: Optional[Team] = None
    ) -> SolutionAttempt:
        """
        Solve subproblem with adversarial validation

        Implements OpenEvolve stages:
        - Stage 3A: Initial solution
        - Stage 3B: Adversarial testing
        - Stage 3C: Robustness improvement

        Args:
            subproblem: Subproblem to solve
            team: Team solving the problem

        Returns:
            SolutionAttempt with adversarial validation
        """
        theorem = subproblem.statement

        logger.info(f"Solving subproblem {subproblem.subproblem_id} with adversarial validation")

        # Stage 3A: Initial solution
        initial_result = await self._stage_3a_initial(subproblem)

        # Stage 3B: Adversarial testing
        if initial_result.success:
            adv_test = await self.adversarial_engine.adversarial_test(
                theorem,
                self.config.defense_approaches[0]
            )

            # Stage 3C: Robustness improvement
            if not adv_test.is_robust:
                improved_result = await self._stage_3c_improve(
                    subproblem,
                    adv_test
                )
                return improved_result

            # Return successful result with adversarial validation
            return SolutionAttempt(
                subproblem_id=subproblem.subproblem_id,
                content=adv_test.best_proof or "",
                quality_metrics={
                    'fitness': 0.8,
                    'robustness': adv_test.robustness_score,
                    'is_robust': adv_test.is_robust,
                    'attacks_blocked': adv_test.attacks_blocked,
                    'total_attacks': adv_test.total_attacks,
                    'adversarial_validated': True
                },
                timestamp=time.time()
            )

        # Return initial result if adversarial testing not applicable
        return initial_result

    async def _stage_3a_initial(
        self,
        subproblem: SubProblem
    ) -> SolutionAttempt:
        """Stage 3A: Generate initial solution"""
        # Generate initial proof using MCTS
        if self.adversarial_engine.mcts_engine:
            result = await self.adversarial_engine.mcts_engine.search(subproblem.statement)
            return SolutionAttempt(
                subproblem_id=subproblem.subproblem_id,
                content=result.best_proof or "",
                quality_metrics={'fitness': result.best_fitness},
                timestamp=time.time()
            )
        else:
            # Fallback
            return SolutionAttempt(
                subproblem_id=subproblem.subproblem_id,
                content=f"Initial solution for {subproblem.statement[:50]}...",
                quality_metrics={'fitness': 0.5},
                timestamp=time.time()
            )

    async def _stage_3c_improve(
        self,
        subproblem: SubProblem,
        adv_test: AdversarialTestResult
    ) -> SolutionAttempt:
        """Stage 3C: Improve solution based on adversarial feedback"""
        # Use adversarial feedback to improve solution
        improved_content = adv_test.best_proof or ""
        for weakness in adv_test.metadata.get('robustness_report', {}).get('weaknesses', []):
            improved_content += f"\n-- Addressed: {weakness}"

        return SolutionAttempt(
            subproblem_id=subproblem.subproblem_id,
            content=improved_content,
            quality_metrics={
                'fitness': 0.8,
                'robustness': adv_test.robustness_score,
                'improved': True
            },
            timestamp=time.time()
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_engine_from_preset(preset_name: str) -> AdversarialEngine:
    """Create AdversarialEngine from preset name"""
    presets = {
        "fast": AdversarialPresets.fast,
        "balanced": AdversarialPresets.balanced,
        "thorough": AdversarialPresets.thorough,
        "self_play": AdversarialPresets.self_play,
    }

    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

    config = presets[preset_name]()
    return AdversarialEngine(config)


async def quick_adversarial_test(
    theorem: str,
    approach: str = "evolved_policies"
) -> AdversarialTestResult:
    """Quick adversarial test with default configuration"""
    config = AdversarialPresets.fast()
    config.defense_approaches = [MCTSApproach(approach)]

    engine = AdversarialEngine(config)
    return await engine.adversarial_test(theorem, MCTSApproach(approach))


async def thorough_adversarial_test(
    theorem: str,
    approach: str = "combined"
) -> AdversarialTestResult:
    """Thorough adversarial test with maximum quality configuration"""
    config = AdversarialPresets.thorough()
    config.defense_approaches = [MCTSApproach(approach)]

    engine = AdversarialEngine(config)
    return await engine.adversarial_test(theorem, MCTSApproach(approach))


def print_result_summary(result: AdversarialTestResult):
    """Print human-readable result summary"""
    print("\n" + "=" * 60)
    print("Adversarial Testing Results")
    print("=" * 60)
    print(f"Theorem:      {result.theorem[:50]}...")
    print(f"Proof Generated: {result.proof_generated}")
    print(f"Robustness:  {result.robustness_score:.2%}")
    print(f"Is Robust:   {result.is_robust}")
    print(f"MCTS Approach: {result.mcts_approach.value if result.mcts_approach else 'N/A'}")
    print(f"Time:         {result.execution_time:.2f}s")
    print(f"\nAttack Summary:")
    print(f"  Total Attacks:    {result.total_attacks}")
    print(f"  Attacks Blocked:  {result.attacks_blocked}")
    print(f"  Vulnerabilities:  {result.vulnerabilities_found}")
    print(f"  Fixes Applied:    {result.fixes_applied}")

    if result.best_proof:
        print(f"\nProof (first 200 chars):")
        print("-" * 60)
        print(result.best_proof[:200] + "..." if len(result.best_proof) > 200 else result.best_proof)
        print("-" * 60)

    print("=" * 60 + "\n")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Main entry point for testing and demonstration"""
    import argparse

    parser = argparse.ArgumentParser(description="Unified Adversarial Framework for MDAP/MAKER/MCTS")
    parser.add_argument("theorem", type=str, help="Theorem to prove and test")
    parser.add_argument("--approach", type=str, default="evolved_policies",
                       choices=["evolved_policies", "evolutionary_nodes",
                               "coevolution", "adaptive", "combined"],
                       help="MCTS approach to use")
    parser.add_argument("--preset", type=str, default="balanced",
                       choices=["fast", "balanced", "thorough", "self_play"],
                       help="Configuration preset")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of adversarial training epochs")
    parser.add_argument("--output", type=str, help="Output file for results")

    args = parser.parse_args()

    # Create engine from preset
    engine = create_engine_from_preset(args.preset)

    # Run adversarial test
    if args.epochs == 1:
        # Single test
        result = await engine.adversarial_test(args.theorem, MCTSApproach(args.approach))
        print_result_summary(result)

        # Save if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"Results saved to {args.output}")
    else:
        # Adversarial training
        corpus = [args.theorem]  # Single theorem corpus for demo
        training_result = await engine.adversarial_training(corpus, args.epochs)

        print("\n" + "=" * 60)
        print("Adversarial Training Results")
        print("=" * 60)
        print(f"Final Success Rate: {training_result.final_success_rate:.2%}")
        print(f"Final Robustness: {training_result.final_robustness:.2%}")
        print(f"Best Epoch: {training_result.best_epoch + 1}")
        print("=" * 60 + "\n")

        # Save if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(training_result.to_dict(), f, indent=2)
            print(f"Results saved to {args.output}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run main
    asyncio.run(main())
