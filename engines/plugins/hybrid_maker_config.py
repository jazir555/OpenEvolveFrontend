"""
Hybrid MAKER Strategy Configuration System

This module provides comprehensive configuration management for hybrid MAKER strategies
that combine multiple problem-solving approaches including LeanAide verification,
MAKER voting, MCTS search, evolutionary optimization, and MDAP decomposition.

Author: OpenEvolve Frontend Team
Version: 1.0.0
"""
from __future__ import annotations


import json
import logging
import os
import yaml
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import copy

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Enumeration of available strategy types"""
    LEANAIDE = "leanaide"
    MAKER = "maker"
    MCTS = "mcts"
    EVOLUTION = "evolution"
    MDAP = "mdap"
    ADAPTIVE = "adaptive"


class PerformanceThreshold(Enum):
    """Performance threshold levels for strategy switching"""
    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"


@dataclass
class LeanAideConfig:
    """Configuration for LeanAide verification strategy"""
    # Server configuration
    server_url: str = "http://localhost:8080"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

    # Verification settings
    verify_tactics: bool = True
    verify_theorems: bool = True
    strict_verification: bool = False

    # Lean environment
    lean_project_path: Optional[str] = None
    lean_version: str = "4.0"
    import_search_path: List[str] = field(default_factory=list)

    # Performance
    parallel_verifications: int = 1
    cache_verification_results: bool = True
    cache_ttl_seconds: int = 3600

    # Resource limits
    max_memory_mb: Optional[int] = None
    max_proof_depth: int = 100

    # Output
    output_format: str = "json"  # json, plain, detailed
    include_proof_trace: bool = False
    include_lean_code: bool = True

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate LeanAide configuration"""
        errors = []

        if self.timeout <= 0:
            errors.append("timeout must be positive")

        if self.max_retries < 0:
            errors.append("max_retries cannot be negative")

        if self.retry_delay < 0:
            errors.append("retry_delay cannot be negative")

        if self.parallel_verifications < 1:
            errors.append("parallel_verifications must be at least 1")

        if self.max_proof_depth < 1:
            errors.append("max_proof_depth must be at least 1")

        if self.output_format not in ["json", "plain", "detailed"]:
            errors.append(f"invalid output_format: {self.output_format}")

        return len(errors) == 0, errors


@dataclass
class MakerConfig:
    """Configuration for MAKER voting strategy"""
    # Voting parameters
    k_min: int = 2
    k_max: int = 8
    max_votes_per_step: int = 50
    max_steps: int = 1000

    # Agent configuration
    min_agents: int = 3
    max_agents: int = 10
    agent_selection_strategy: str = "weighted"  # weighted, random, round_robin

    # Timeout and checkpoints
    timeout_seconds: int = 60
    checkpoint_interval: int = 25

    # Red flag rules
    max_tokens: int = 750
    max_characters: Optional[int] = 6000
    min_confidence: float = 0.2
    require_schema_match: bool = True

    # Performance
    parallel_voting: bool = True
    vote_batch_size: int = 5

    # Fallback behavior
    fallback_policy: str = "escalate_then_best_effort"  # escalate, best_effort, escalate_then_best_effort

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate MAKER configuration"""
        errors = []

        if self.k_min < 1:
            errors.append("k_min must be at least 1")

        if self.k_max < self.k_min:
            errors.append("k_max must be >= k_min")

        if self.max_votes_per_step < 1:
            errors.append("max_votes_per_step must be at least 1")

        if self.max_steps < 1:
            errors.append("max_steps must be at least 1")

        if self.min_agents < 1:
            errors.append("min_agents must be at least 1")

        if self.max_agents < self.min_agents:
            errors.append("max_agents must be >= min_agents")

        if self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")

        if self.checkpoint_interval < 1:
            errors.append("checkpoint_interval must be at least 1")

        if self.min_confidence < 0 or self.min_confidence > 1:
            errors.append("min_confidence must be between 0 and 1")

        if self.vote_batch_size < 1:
            errors.append("vote_batch_size must be at least 1")

        if self.fallback_policy not in ["escalate", "best_effort", "escalate_then_best_effort"]:
            errors.append(f"invalid fallback_policy: {self.fallback_policy}")

        return len(errors) == 0, errors


@dataclass
class MCTSConfig:
    """Configuration for MCTS search strategy"""
    # Search parameters
    num_simulations: int = 1000
    exploration_constant: float = 1.414  # UCB1 constant (sqrt(2))
    discount_factor: float = 0.99

    # Tree configuration
    max_tree_depth: int = 100
    min_visits_for_expansion: int = 1
    virtual_loss: int = 1

    # Selection policy
    selection_policy: str = "ucb1"  # ucb1, thompson, epsilon_greedy
    epsilon: float = 0.1  # for epsilon_greedy

    # Action space
    max_actions_per_node: int = 10
    action_pruning_threshold: float = 0.01

    # Performance
    parallel_simulations: bool = True
    num_workers: int = 4

    # Rollout policy
    rollout_depth: int = 10
    rollout_policy: str = "random"  # random, heuristic, model

    # Termination
    early_termination: bool = True
    convergence_threshold: float = 0.001
    convergence_window: int = 100

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate MCTS configuration"""
        errors = []

        if self.num_simulations < 1:
            errors.append("num_simulations must be at least 1")

        if self.exploration_constant < 0:
            errors.append("exploration_constant cannot be negative")

        if self.discount_factor < 0 or self.discount_factor > 1:
            errors.append("discount_factor must be between 0 and 1")

        if self.max_tree_depth < 1:
            errors.append("max_tree_depth must be at least 1")

        if self.min_visits_for_expansion < 1:
            errors.append("min_visits_for_expansion must be at least 1")

        if self.selection_policy not in ["ucb1", "thompson", "epsilon_greedy"]:
            errors.append(f"invalid selection_policy: {self.selection_policy}")

        if self.epsilon < 0 or self.epsilon > 1:
            errors.append("epsilon must be between 0 and 1")

        if self.max_actions_per_node < 1:
            errors.append("max_actions_per_node must be at least 1")

        if self.action_pruning_threshold < 0 or self.action_pruning_threshold > 1:
            errors.append("action_pruning_threshold must be between 0 and 1")

        if self.num_workers < 1:
            errors.append("num_workers must be at least 1")

        if self.rollout_depth < 1:
            errors.append("rollout_depth must be at least 1")

        if self.rollout_policy not in ["random", "heuristic", "model"]:
            errors.append(f"invalid rollout_policy: {self.rollout_policy}")

        return len(errors) == 0, errors


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary optimization strategy"""
    # Population parameters
    population_size: int = 100
    generations: int = 100
    elite_ratio: float = 0.1

    # Genetic operators
    mutation_rate: float = 0.1
    mutation_strength: float = 0.2
    crossover_rate: float = 0.7
    crossover_method: str = "uniform"  # uniform, single_point, two_point

    # Selection
    selection_method: str = "tournament"  # tournament, roulette, rank
    tournament_size: int = 3
    selection_pressure: float = 2.0

    # Diversity maintenance
    diversity_metric: str = "hamming"  # hamming, euclidean, cosine
    diversity_threshold: float = 0.1
    niching: bool = False
    niche_radius: float = 0.2

    # Island model (optional)
    num_islands: int = 1
    migration_interval: int = 20
    migration_rate: float = 0.1
    migration_topology: str = "ring"  # ring, fully_connected, random

    # Performance
    parallel_evaluation: bool = True
    evaluation_workers: int = 4

    # Stopping conditions
    early_stopping: bool = True
    patience: int = 20
    min_improvement: float = 0.001

    # Archive
    use_archive: bool = True
    archive_size: int = 100

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate Evolution configuration"""
        errors = []

        if self.population_size < 10:
            errors.append("population_size must be at least 10")

        if self.generations < 1:
            errors.append("generations must be at least 1")

        if self.elite_ratio < 0 or self.elite_ratio > 0.5:
            errors.append("elite_ratio must be between 0 and 0.5")

        if self.mutation_rate < 0 or self.mutation_rate > 1:
            errors.append("mutation_rate must be between 0 and 1")

        if self.mutation_strength < 0 or self.mutation_strength > 1:
            errors.append("mutation_strength must be between 0 and 1")

        if self.crossover_rate < 0 or self.crossover_rate > 1:
            errors.append("crossover_rate must be between 0 and 1")

        if self.crossover_method not in ["uniform", "single_point", "two_point"]:
            errors.append(f"invalid crossover_method: {self.crossover_method}")

        if self.selection_method not in ["tournament", "roulette", "rank"]:
            errors.append(f"invalid selection_method: {self.selection_method}")

        if self.tournament_size < 2:
            errors.append("tournament_size must be at least 2")

        if self.selection_pressure < 1:
            errors.append("selection_pressure must be at least 1")

        if self.diversity_metric not in ["hamming", "euclidean", "cosine"]:
            errors.append(f"invalid diversity_metric: {self.diversity_metric}")

        if self.diversity_threshold < 0 or self.diversity_threshold > 1:
            errors.append("diversity_threshold must be between 0 and 1")

        if self.niche_radius < 0 or self.niche_radius > 1:
            errors.append("niche_radius must be between 0 and 1")

        if self.num_islands < 1:
            errors.append("num_islands must be at least 1")

        if self.migration_interval < 1:
            errors.append("migration_interval must be at least 1")

        if self.migration_rate < 0 or self.migration_rate > 1:
            errors.append("migration_rate must be between 0 and 1")

        if self.migration_topology not in ["ring", "fully_connected", "random"]:
            errors.append(f"invalid migration_topology: {self.migration_topology}")

        if self.evaluation_workers < 1:
            errors.append("evaluation_workers must be at least 1")

        if self.patience < 1:
            errors.append("patience must be at least 1")

        if self.archive_size < 10:
            errors.append("archive_size must be at least 10")

        return len(errors) == 0, errors


@dataclass
class MDAPConfig:
    """Configuration for MDAP decomposition strategy"""
    # Decomposition parameters
    decomposition_depth: int = 3
    min_subproblem_size: int = 5
    max_subproblem_size: int = 50

    # Agent configuration
    agent_count: int = 5
    agent_specialization: bool = True

    # Task execution
    max_retries_per_task: int = 3
    task_timeout: int = 120
    parallel_tasks: int = 3

    # Assembly
    assembly_strategy: str = "hierarchical"  # hierarchical, sequential, parallel
    verify_assembly: bool = True
    max_assembly_retries: int = 2

    # Quality control
    quality_threshold: float = 0.8
    validation_rate: float = 0.2
    cross_validation: bool = True

    # Caching
    cache_solutions: bool = True
    cache_size: int = 1000

    # Resource limits
    max_memory_mb: Optional[int] = None
    max_total_time: Optional[int] = None

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate MDAP configuration"""
        errors = []

        if self.decomposition_depth < 1:
            errors.append("decomposition_depth must be at least 1")

        if self.min_subproblem_size < 1:
            errors.append("min_subproblem_size must be at least 1")

        if self.max_subproblem_size < self.min_subproblem_size:
            errors.append("max_subproblem_size must be >= min_subproblem_size")

        if self.agent_count < 1:
            errors.append("agent_count must be at least 1")

        if self.max_retries_per_task < 0:
            errors.append("max_retries_per_task cannot be negative")

        if self.task_timeout <= 0:
            errors.append("task_timeout must be positive")

        if self.parallel_tasks < 1:
            errors.append("parallel_tasks must be at least 1")

        if self.assembly_strategy not in ["hierarchical", "sequential", "parallel"]:
            errors.append(f"invalid assembly_strategy: {self.assembly_strategy}")

        if self.quality_threshold < 0 or self.quality_threshold > 1:
            errors.append("quality_threshold must be between 0 and 1")

        if self.validation_rate < 0 or self.validation_rate > 1:
            errors.append("validation_rate must be between 0 and 1")

        if self.cache_size < 1:
            errors.append("cache_size must be at least 1")

        return len(errors) == 0, errors


@dataclass
class HybridStrategyProfile:
    """Profile configuration for a single strategy"""
    strategy_type: StrategyType
    enabled: bool = True
    performance_weight: float = 1.0
    priority: int = 0

    # Resource allocation
    cpu_allocation: float = 1.0  # Fraction of available CPUs
    memory_allocation_mb: Optional[int] = None

    # Timeout settings
    max_time_seconds: Optional[int] = None
    soft_timeout_seconds: Optional[int] = None

    # Execution parameters
    parallel_instances: int = 1
    retry_on_failure: bool = True
    max_retries: int = 3

    # Quality control
    quality_threshold: float = 0.7
    confidence_threshold: float = 0.5

    # Metadata
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate strategy profile"""
        errors = []

        if self.performance_weight < 0:
            errors.append("performance_weight cannot be negative")

        if self.priority < 0:
            errors.append("priority cannot be negative")

        if self.cpu_allocation <= 0 or self.cpu_allocation > 1:
            errors.append("cpu_allocation must be between 0 and 1")

        if self.max_time_seconds is not None and self.max_time_seconds <= 0:
            errors.append("max_time_seconds must be positive")

        if self.soft_timeout_seconds is not None and self.soft_timeout_seconds <= 0:
            errors.append("soft_timeout_seconds must be positive")

        if self.parallel_instances < 1:
            errors.append("parallel_instances must be at least 1")

        if self.max_retries < 0:
            errors.append("max_retries cannot be negative")

        if self.quality_threshold < 0 or self.quality_threshold > 1:
            errors.append("quality_threshold must be between 0 and 1")

        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            errors.append("confidence_threshold must be between 0 and 1")

        return len(errors) == 0, errors


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive strategy selection"""
    # Adaptation parameters
    enable_adaptive_selection: bool = True
    adaptation_interval: int = 10
    warmup_period: int = 5

    # Performance tracking
    track_performance_history: bool = True
    performance_history_size: int = 100

    # Switching thresholds
    min_performance_threshold: float = 0.6
    performance_degradation_threshold: float = 0.1

    # Resource-aware adaptation
    resource_aware: bool = True
    cpu_threshold: float = 0.8
    memory_threshold: float = 0.8

    # Strategy combination
    allow_strategy_combination: bool = True
    max_combined_strategies: int = 3
    combination_method: str = "weighted_voting"  # weighted_voting, stacking, ensemble

    # Exploration
    exploration_rate: float = 0.1
    exploration_decay: float = 0.99
    min_exploration_rate: float = 0.01

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate adaptive configuration"""
        errors = []

        if self.adaptation_interval < 1:
            errors.append("adaptation_interval must be at least 1")

        if self.warmup_period < 0:
            errors.append("warmup_period cannot be negative")

        if self.performance_history_size < 10:
            errors.append("performance_history_size must be at least 10")

        if self.min_performance_threshold < 0 or self.min_performance_threshold > 1:
            errors.append("min_performance_threshold must be between 0 and 1")

        if self.performance_degradation_threshold < 0 or self.performance_degradation_threshold > 1:
            errors.append("performance_degradation_threshold must be between 0 and 1")

        if self.cpu_threshold <= 0 or self.cpu_threshold > 1:
            errors.append("cpu_threshold must be between 0 and 1")

        if self.memory_threshold <= 0 or self.memory_threshold > 1:
            errors.append("memory_threshold must be between 0 and 1")

        if self.max_combined_strategies < 2:
            errors.append("max_combined_strategies must be at least 2")

        if self.combination_method not in ["weighted_voting", "stacking", "ensemble"]:
            errors.append(f"invalid combination_method: {self.combination_method}")

        if self.exploration_rate < 0 or self.exploration_rate > 1:
            errors.append("exploration_rate must be between 0 and 1")

        if self.exploration_decay <= 0 or self.exploration_decay > 1:
            errors.append("exploration_decay must be between 0 and 1")

        if self.min_exploration_rate < 0 or self.min_exploration_rate > 1:
            errors.append("min_exploration_rate must be between 0 and 1")

        return len(errors) == 0, errors


@dataclass
class PerformanceThresholds:
    """Performance thresholds for strategy switching"""
    # Time thresholds (seconds)
    fast_time_threshold: int = 60
    balanced_time_threshold: int = 300
    thorough_time_threshold: int = 1800

    # Quality thresholds
    fast_quality_threshold: float = 0.6
    balanced_quality_threshold: float = 0.8
    thorough_quality_threshold: float = 0.95

    # Resource thresholds
    max_cpu_usage: float = 0.9
    max_memory_usage: float = 0.9
    max_execution_time: int = 3600

    # Switching conditions
    enable_time_based_switching: bool = True
    enable_quality_based_switching: bool = True
    enable_resource_based_switching: bool = True

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate performance thresholds"""
        errors = []

        if self.fast_time_threshold <= 0:
            errors.append("fast_time_threshold must be positive")

        if self.balanced_time_threshold <= self.fast_time_threshold:
            errors.append("balanced_time_threshold must be > fast_time_threshold")

        if self.thorough_time_threshold <= self.balanced_time_threshold:
            errors.append("thorough_time_threshold must be > balanced_time_threshold")

        if self.fast_quality_threshold < 0 or self.fast_quality_threshold > 1:
            errors.append("fast_quality_threshold must be between 0 and 1")

        if self.balanced_quality_threshold < 0 or self.balanced_quality_threshold > 1:
            errors.append("balanced_quality_threshold must be between 0 and 1")

        if self.thorough_quality_threshold < 0 or self.thorough_quality_threshold > 1:
            errors.append("thorough_quality_threshold must be between 0 and 1")

        if self.balanced_quality_threshold <= self.fast_quality_threshold:
            errors.append("balanced_quality_threshold should be > fast_quality_threshold")

        if self.thorough_quality_threshold <= self.balanced_quality_threshold:
            errors.append("thorough_quality_threshold should be > balanced_quality_threshold")

        if self.max_cpu_usage <= 0 or self.max_cpu_usage > 1:
            errors.append("max_cpu_usage must be between 0 and 1")

        if self.max_memory_usage <= 0 or self.max_memory_usage > 1:
            errors.append("max_memory_usage must be between 0 and 1")

        if self.max_execution_time <= 0:
            errors.append("max_execution_time must be positive")

        return len(errors) == 0, errors


@dataclass
class HybridMakerConfig:
    """Main configuration class for hybrid MAKER strategies"""
    # Strategy configurations
    leanaide_config: LeanAideConfig = field(default_factory=LeanAideConfig)
    maker_config: MakerConfig = field(default_factory=MakerConfig)
    mcts_config: MCTSConfig = field(default_factory=MCTSConfig)
    evolution_config: EvolutionConfig = field(default_factory=EvolutionConfig)
    mdap_config: MDAPConfig = field(default_factory=MDAPConfig)

    # Strategy profiles
    strategy_profiles: Dict[str, HybridStrategyProfile] = field(default_factory=dict)

    # Adaptive configuration
    adaptive_config: AdaptiveConfig = field(default_factory=AdaptiveConfig)

    # Performance thresholds
    performance_thresholds: PerformanceThresholds = field(default_factory=PerformanceThresholds)

    # Global settings
    default_strategy: StrategyType = StrategyType.MAKER
    enable_parallel_strategies: bool = True
    max_parallel_strategies: int = 3
    global_timeout: int = 3600
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 100
    checkpoint_dir: str = "./checkpoints"

    # Logging and monitoring
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_metrics: bool = True
    metrics_port: Optional[int] = None

    # Metadata
    config_name: str = "default"
    config_version: str = "1.0.0"
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize default strategy profiles after creation"""
        if not self.strategy_profiles:
            self._init_default_profiles()

    def _init_default_profiles(self):
        """Initialize default strategy profiles"""
        self.strategy_profiles = {
            "leanaide": HybridStrategyProfile(
                strategy_type=StrategyType.LEANAIDE,
                enabled=True,
                performance_weight=1.0,
                priority=1,
                description="Lean theorem proving and verification"
            ),
            "maker": HybridStrategyProfile(
                strategy_type=StrategyType.MAKER,
                enabled=True,
                performance_weight=1.0,
                priority=2,
                description="Multi-agent voting and consensus"
            ),
            "mcts": HybridStrategyProfile(
                strategy_type=StrategyType.MCTS,
                enabled=True,
                performance_weight=0.8,
                priority=3,
                description="Monte Carlo Tree Search exploration"
            ),
            "evolution": HybridStrategyProfile(
                strategy_type=StrategyType.EVOLUTION,
                enabled=True,
                performance_weight=0.7,
                priority=4,
                description="Evolutionary optimization"
            ),
            "mdap": HybridStrategyProfile(
                strategy_type=StrategyType.MDAP,
                enabled=True,
                performance_weight=0.9,
                priority=5,
                description="Multi-agent Decomposition and Assembly"
            ),
        }

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate entire configuration"""
        all_errors = []

        # Validate each sub-configuration
        configs = [
            ("leanaide", self.leanaide_config),
            ("maker", self.maker_config),
            ("mcts", self.mcts_config),
            ("evolution", self.evolution_config),
            ("mdap", self.mdap_config),
        ]

        for name, config in configs:
            valid, errors = config.validate()
            if not valid:
                all_errors.extend([f"{name}.{err}" for err in errors])

        # Validate strategy profiles
        for profile_name, profile in self.strategy_profiles.items():
            valid, errors = profile.validate()
            if not valid:
                all_errors.extend([f"profile.{profile_name}.{err}" for err in errors])

        # Validate adaptive config
        valid, errors = self.adaptive_config.validate()
        if not valid:
            all_errors.extend([f"adaptive.{err}" for err in errors])

        # Validate performance thresholds
        valid, errors = self.performance_thresholds.validate()
        if not valid:
            all_errors.extend([f"thresholds.{err}" for err in errors])

        # Validate global settings
        if self.max_parallel_strategies < 1:
            all_errors.append("max_parallel_strategies must be at least 1")

        if self.global_timeout <= 0:
            all_errors.append("global_timeout must be positive")

        if self.checkpoint_interval < 1:
            all_errors.append("checkpoint_interval must be at least 1")

        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            all_errors.append(f"invalid log_level: {self.log_level}")

        return len(all_errors) == 0, all_errors

    def estimate_runtime(self, strategy: Optional[StrategyType] = None) -> Dict[str, float]:
        """
        Estimate runtime for a given strategy or all strategies

        Returns dict with strategy names and estimated runtimes in seconds
        """
        estimates = {}

        if strategy is None or strategy == StrategyType.LEANAIDE:
            lean_time = (
                self.leanaide_config.timeout *
                (1 + self.leanaide_config.max_retries * self.leanaide_config.retry_delay)
            )
            estimates["leanaide"] = lean_time

        if strategy is None or strategy == StrategyType.MAKER:
            maker_time = (
                self.maker_config.max_steps *
                self.maker_config.timeout_seconds /
                max(1, self.maker_config.min_agents)
            )
            estimates["maker"] = maker_time

        if strategy is None or strategy == StrategyType.MCTS:
            mcts_time = (
                self.mcts_config.num_simulations *
                self.mcts_config.rollout_depth /
                max(1, self.mcts_config.num_workers)
            )
            estimates["mcts"] = mcts_time

        if strategy is None or strategy == StrategyType.EVOLUTION:
            evo_time = (
                self.evolution_config.population_size *
                self.evolution_config.generations /
                max(1, self.evolution_config.evaluation_workers)
            )
            estimates["evolution"] = evo_time

        if strategy is None or strategy == StrategyType.MDAP:
            mdap_time = (
                self.mdap_config.decomposition_depth *
                self.mdap_config.agent_count *
                self.mdap_config.task_timeout
            )
            estimates["mdap"] = mdap_time

        return estimates

    def estimate_resource_usage(self, strategy: Optional[StrategyType] = None) -> Dict[str, Dict[str, float]]:
        """
        Estimate resource usage for strategies

        Returns dict with strategy names and resource estimates (CPU, memory in MB)
        """
        usage = {}

        if strategy is None or strategy == StrategyType.LEANAIDE:
            usage["leanaide"] = {
                "cpu": self.leanaide_config.parallel_verifications * 0.5,
                "memory_mb": self.leanaide_config.max_memory_mb or 512
            }

        if strategy is None or strategy == StrategyType.MAKER:
            usage["maker"] = {
                "cpu": self.maker_config.min_agents * 0.3,
                "memory_mb": 256 * self.maker_config.min_agents
            }

        if strategy is None or strategy == StrategyType.MCTS:
            usage["mcts"] = {
                "cpu": self.mcts_config.num_workers * 0.8,
                "memory_mb": 1024
            }

        if strategy is None or strategy == StrategyType.EVOLUTION:
            usage["evolution"] = {
                "cpu": self.evolution_config.evaluation_workers * 0.6,
                "memory_mb": self.evolution_config.population_size * 0.5
            }

        if strategy is None or strategy == StrategyType.MDAP:
            usage["mdap"] = {
                "cpu": self.mdap_config.parallel_tasks * 0.7,
                "memory_mb": self.mdap_config.max_memory_mb or 2048
            }

        return usage

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        def convert_value(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_value(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            elif hasattr(obj, '__dict__'):
                # Handle dataclasses and objects
                return {k: convert_value(v) for k, v in obj.__dict__.items()}
            return obj

        result = {}
        for key, value in asdict(self).items():
            if key == 'strategy_profiles':
                # Convert strategy_profiles properly
                result['strategy_profiles'] = {
                    k: convert_value(v) for k, v in self.strategy_profiles.items()
                }
            else:
                result[key] = convert_value(value)

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HybridMakerConfig':
        """Create configuration from dictionary"""
        # Create nested configs
        leanaide = LeanAideConfig(**data.get('leanaide_config', {}))
        maker = MakerConfig(**data.get('maker_config', {}))
        mcts = MCTSConfig(**data.get('mcts_config', {}))
        evolution = EvolutionConfig(**data.get('evolution_config', {}))
        mdap = MDAPConfig(**data.get('mdap_config', {}))

        # Create strategy profiles
        profiles_data = data.get('strategy_profiles', {})
        strategy_profiles = {}
        for name, profile_data in profiles_data.items():
            if isinstance(profile_data, dict):
                profile_data['strategy_type'] = StrategyType(profile_data.get('strategy_type', 'maker'))
                strategy_profiles[name] = HybridStrategyProfile(**profile_data)

        # Create adaptive config
        adaptive_data = data.get('adaptive_config', {})
        adaptive = AdaptiveConfig(**adaptive_data)

        # Create performance thresholds
        thresholds_data = data.get('performance_thresholds', {})
        thresholds = PerformanceThresholds(**thresholds_data)

        # Create main config
        config = cls(
            leanaide_config=leanaide,
            maker_config=maker,
            mcts_config=mcts,
            evolution_config=evolution,
            mdap_config=mdap,
            strategy_profiles=strategy_profiles,
            adaptive_config=adaptive,
            performance_thresholds=thresholds,
        )

        # Override global settings
        for key in ['default_strategy', 'enable_parallel_strategies', 'max_parallel_strategies',
                    'global_timeout', 'checkpoint_enabled', 'checkpoint_interval', 'checkpoint_dir',
                    'log_level', 'log_file', 'enable_metrics', 'metrics_port',
                    'config_name', 'config_version', 'description', 'tags']:
            if key in data:
                if key == 'default_strategy':
                    setattr(config, key, StrategyType(data[key]))
                else:
                    setattr(config, key, data[key])

        return config

    def save_to_file(self, filepath: Union[str, Path], format: str = "yaml") -> bool:
        """Save configuration to file"""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            data = self.to_dict()

            if format.lower() == "json":
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
            elif format.lower() == "yaml":
                with open(filepath, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Configuration saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> Optional['HybridMakerConfig']:
        """Load configuration from file"""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                logger.error(f"Configuration file not found: {filepath}")
                return None

            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif filepath.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    # Try YAML first, fall back to JSON
                    content = f.read()
                    f.seek(0)
                    try:
                        data = yaml.safe_load(content)
                    except yaml.YAMLError:
                        data = json.loads(content)

            config = cls.from_dict(data)
            logger.info(f"Configuration loaded from {filepath}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return None

    def merge_with(self, other: 'HybridMakerConfig') -> 'HybridMakerConfig':
        """Merge this configuration with another, preferring other's values"""
        merged = copy.deepcopy(self)

        # Merge each sub-config
        merged.leanaide_config = copy.deepcopy(other.leanaide_config)
        merged.maker_config = copy.deepcopy(other.maker_config)
        merged.mcts_config = copy.deepcopy(other.mcts_config)
        merged.evolution_config = copy.deepcopy(other.evolution_config)
        merged.mdap_config = copy.deepcopy(other.mdap_config)

        # Merge strategy profiles
        for name, profile in other.strategy_profiles.items():
            merged.strategy_profiles[name] = copy.deepcopy(profile)

        # Merge adaptive config
        merged.adaptive_config = copy.deepcopy(other.adaptive_config)

        # Merge performance thresholds
        merged.performance_thresholds = copy.deepcopy(other.performance_thresholds)

        # Merge global settings (only if not default in other)
        if other.config_name != "default":
            merged.config_name = other.config_name
        if other.description is not None:
            merged.description = other.description
        if other.tags:
            merged.tags = copy.deepcopy(other.tags)
        if other.config_version != "1.0.0":
            merged.config_version = other.config_version

        # Always merge these settings
        merged.default_strategy = other.default_strategy
        merged.enable_parallel_strategies = other.enable_parallel_strategies
        merged.max_parallel_strategies = other.max_parallel_strategies
        merged.global_timeout = other.global_timeout
        merged.checkpoint_enabled = other.checkpoint_enabled
        merged.checkpoint_interval = other.checkpoint_interval
        merged.checkpoint_dir = other.checkpoint_dir
        merged.log_level = other.log_level
        merged.log_file = other.log_file
        merged.enable_metrics = other.enable_metrics
        merged.metrics_port = other.metrics_port

        return merged


class HybridMakerConfigPreset:
    """Predefined configuration presets for common use cases"""

    @staticmethod
    def fast() -> HybridMakerConfig:
        """Fast exploration configuration - minimal computation"""
        config = HybridMakerConfig(config_name="fast", description="Fast exploration configuration")

        # LeanAide - minimal retries
        config.leanaide_config.timeout = 10
        config.leanaide_config.max_retries = 1
        config.leanaide_config.parallel_verifications = 1

        # MAKER - minimal voting
        config.maker_config.k_min = 2
        config.maker_config.k_max = 3
        config.maker_config.max_votes_per_step = 10
        config.maker_config.min_agents = 3
        config.maker_config.timeout_seconds = 20

        # MCTS - minimal simulations
        config.mcts_config.num_simulations = 100
        config.mcts_config.max_tree_depth = 20
        config.mcts_config.num_workers = 2
        config.mcts_config.rollout_depth = 5

        # Evolution - small population
        config.evolution_config.population_size = 20
        config.evolution_config.generations = 10
        config.evolution_config.evaluation_workers = 2

        # MDAP - minimal decomposition
        config.mdap_config.decomposition_depth = 2
        config.mdap_config.agent_count = 3
        config.mdap_config.parallel_tasks = 1

        # Performance thresholds
        config.performance_thresholds.fast_time_threshold = 30
        config.performance_thresholds.fast_quality_threshold = 0.5
        config.global_timeout = 300

        # Disable some features for speed
        config.adaptive_config.enable_adaptive_selection = False
        config.checkpoint_enabled = False

        return config

    @staticmethod
    def balanced() -> HybridMakerConfig:
        """Balanced configuration - good trade-off between speed and quality"""
        config = HybridMakerConfig(config_name="balanced", description="Balanced configuration")

        # LeanAide - balanced settings
        config.leanaide_config.timeout = 30
        config.leanaide_config.max_retries = 2
        config.leanaide_config.parallel_verifications = 2

        # MAKER - moderate voting
        config.maker_config.k_min = 3
        config.maker_config.k_max = 6
        config.maker_config.max_votes_per_step = 30
        config.maker_config.min_agents = 5
        config.maker_config.timeout_seconds = 45

        # MCTS - moderate simulations
        config.mcts_config.num_simulations = 500
        config.mcts_config.max_tree_depth = 50
        config.mcts_config.num_workers = 4
        config.mcts_config.rollout_depth = 10

        # Evolution - moderate population
        config.evolution_config.population_size = 50
        config.evolution_config.generations = 30
        config.evolution_config.evaluation_workers = 4

        # MDAP - moderate decomposition
        config.mdap_config.decomposition_depth = 3
        config.mdap_config.agent_count = 5
        config.mdap_config.parallel_tasks = 2

        # Performance thresholds
        config.performance_thresholds.balanced_time_threshold = 300
        config.performance_thresholds.balanced_quality_threshold = 0.75
        config.global_timeout = 1800

        return config

    @staticmethod
    def thorough() -> HybridMakerConfig:
        """Thorough configuration - maximum quality"""
        config = HybridMakerConfig(config_name="thorough", description="Maximum quality configuration")

        # LeanAide - maximum verification
        config.leanaide_config.timeout = 60
        config.leanaide_config.max_retries = 5
        config.leanaide_config.parallel_verifications = 4
        config.leanaide_config.strict_verification = True

        # MAKER - extensive voting
        config.maker_config.k_min = 5
        config.maker_config.k_max = 10
        config.maker_config.max_votes_per_step = 100
        config.maker_config.min_agents = 10
        config.maker_config.timeout_seconds = 120

        # MCTS - extensive simulations
        config.mcts_config.num_simulations = 5000
        config.mcts_config.max_tree_depth = 200
        config.mcts_config.num_workers = 8
        config.mcts_config.rollout_depth = 20

        # Evolution - large population
        config.evolution_config.population_size = 200
        config.evolution_config.generations = 100
        config.evolution_config.evaluation_workers = 8
        config.evolution_config.niching = True
        config.evolution_config.num_islands = 5

        # MDAP - deep decomposition
        config.mdap_config.decomposition_depth = 5
        config.mdap_config.agent_count = 10
        config.mdap_config.parallel_tasks = 5
        config.mdap_config.cross_validation = True

        # Performance thresholds
        config.performance_thresholds.thorough_time_threshold = 3600
        config.performance_thresholds.thorough_quality_threshold = 0.95
        config.global_timeout = 14400  # 4 hours

        # Enhanced features
        config.adaptive_config.enable_adaptive_selection = True
        config.adaptive_config.track_performance_history = True
        config.checkpoint_enabled = True
        config.checkpoint_interval = 50

        return config

    @staticmethod
    def leanaide_focused() -> HybridMakerConfig:
        """LeanAide-focused configuration"""
        config = HybridMakerConfig(config_name="leanaide_focused", description="LeanAide-focused configuration")
        config.default_strategy = StrategyType.LEANAIDE

        # Prioritize LeanAide
        config.leanaide_config.timeout = 90
        config.leanaide_config.max_retries = 5
        config.leanaide_config.parallel_verifications = 4
        config.leanaide_config.strict_verification = True
        config.leanaide_config.include_proof_trace = True

        # Disable other strategies
        config.strategy_profiles["maker"].enabled = False
        config.strategy_profiles["mcts"].enabled = False
        config.strategy_profiles["evolution"].enabled = False
        config.strategy_profiles["mdap"].enabled = False

        config.enable_parallel_strategies = False

        return config

    @staticmethod
    def maker_focused() -> HybridMakerConfig:
        """MAKER-focused configuration"""
        config = HybridMakerConfig(config_name="maker_focused", description="MAKER-focused configuration")
        config.default_strategy = StrategyType.MAKER

        # Prioritize MAKER
        config.maker_config.k_min = 4
        config.maker_config.k_max = 8
        config.maker_config.max_votes_per_step = 80
        config.maker_config.min_agents = 8
        config.maker_config.timeout_seconds = 90

        # Use MDAP for decomposition support
        config.mdap_config.decomposition_depth = 2
        config.mdap_config.agent_count = 5

        # Disable other strategies
        config.strategy_profiles["leanaide"].enabled = False
        config.strategy_profiles["mcts"].enabled = False
        config.strategy_profiles["evolution"].enabled = False

        return config

    @staticmethod
    def adaptive() -> HybridMakerConfig:
        """Adaptive configuration with automatic strategy selection"""
        config = HybridMakerConfig(config_name="adaptive", description="Adaptive strategy configuration")
        config.default_strategy = StrategyType.ADAPTIVE

        # Enable all strategies
        for profile in config.strategy_profiles.values():
            profile.enabled = True

        # Adaptive settings
        config.adaptive_config.enable_adaptive_selection = True
        config.adaptive_config.adaptation_interval = 5
        config.adaptive_config.track_performance_history = True
        config.adaptive_config.performance_history_size = 200
        config.adaptive_config.resource_aware = True
        config.adaptive_config.allow_strategy_combination = True
        config.adaptive_config.max_combined_strategies = 3

        # Enable parallel strategies
        config.enable_parallel_strategies = True
        config.max_parallel_strategies = 2

        # Moderate defaults for all strategies
        config.maker_config.k_min = 3
        config.maker_config.k_max = 6
        config.maker_config.max_votes_per_step = 40
        config.maker_config.min_agents = 5

        config.mcts_config.num_simulations = 500
        config.mcts_config.num_workers = 4

        config.evolution_config.population_size = 50
        config.evolution_config.generations = 20
        config.evolution_config.evaluation_workers = 4

        config.mdap_config.decomposition_depth = 3
        config.mdap_config.agent_count = 5

        return config

    @staticmethod
    def research() -> HybridMakerConfig:
        """Research configuration for experimentation"""
        config = HybridMakerConfig(config_name="research", description="Research configuration")

        # Enable all advanced features
        config.adaptive_config.enable_adaptive_selection = True
        config.adaptive_config.allow_strategy_combination = True
        config.adaptive_config.max_combined_strategies = 5

        config.evolution_config.niching = True
        config.evolution_config.num_islands = 7
        config.evolution_config.use_archive = True

        # Extensive logging and metrics
        config.log_level = "DEBUG"
        config.enable_metrics = True
        config.checkpoint_enabled = True
        config.checkpoint_interval = 25

        # High resource allocation
        config.enable_parallel_strategies = True
        config.max_parallel_strategies = 5

        # Long execution time
        config.global_timeout = 28800  # 8 hours

        return config


def create_config_from_preset(preset_name: str) -> Optional[HybridMakerConfig]:
    """Create a configuration from a preset name"""
    preset_method = getattr(HybridMakerConfigPreset, preset_name.lower(), None)
    if preset_method and callable(preset_method):
        return preset_method()

    logger.error(f"Unknown preset: {preset_name}")
    return None


def merge_configs(*configs: HybridMakerConfig) -> HybridMakerConfig:
    """Merge multiple configurations, later configs take precedence"""
    if not configs:
        return HybridMakerConfig()

    merged = configs[0]
    for config in configs[1:]:
        merged = merged.merge_with(config)

    return merged


def validate_and_create_config(data: Dict[str, Any]) -> Tuple[Optional[HybridMakerConfig], List[str]]:
    """
    Validate configuration data and create config if valid

    Returns:
        Tuple of (config or None, list of errors)
    """
    try:
        config = HybridMakerConfig.from_dict(data)
        valid, errors = config.validate()

        if valid:
            return config, []
        else:
            return None, errors
    except Exception as e:
        return None, [str(e)]


# Utility functions for configuration management
def get_available_presets() -> List[str]:
    """Get list of available preset names"""
    return [
        "fast",
        "balanced",
        "thorough",
        "leanaide_focused",
        "maker_focused",
        "adaptive",
        "research"
    ]


def compare_configs(config1: HybridMakerConfig, config2: HybridMakerConfig) -> Dict[str, Any]:
    """
    Compare two configurations and return differences

    Returns:
        Dict with keys 'added', 'removed', 'changed', 'unchanged'
    """
    dict1 = config1.to_dict()
    dict2 = config2.to_dict()

    def compare_dicts(d1: Dict, d2: Dict, prefix: str = "") -> Dict[str, Any]:
        result = {
            "added": [],
            "removed": [],
            "changed": [],
            "unchanged": []
        }

        all_keys = set(d1.keys()) | set(d2.keys())

        for key in all_keys:
            full_key = f"{prefix}.{key}" if prefix else key

            if key not in d1:
                result["added"].append(full_key)
            elif key not in d2:
                result["removed"].append(full_key)
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                nested = compare_dicts(d1[key], d2[key], full_key)
                result["added"].extend(nested["added"])
                result["removed"].extend(nested["removed"])
                result["changed"].extend(nested["changed"])
                result["unchanged"].extend(nested["unchanged"])
            elif d1[key] != d2[key]:
                result["changed"].append(full_key)
            else:
                result["unchanged"].append(full_key)

        return result

    return compare_dicts(dict1, dict2)


def export_config_summary(config: HybridMakerConfig) -> str:
    """Export a human-readable summary of the configuration"""
    lines = [
        f"# Hybrid MAKER Configuration Summary",
        f"# Name: {config.config_name}",
        f"# Version: {config.config_version}",
        f"# Description: {config.description or 'N/A'}",
        f"# Tags: {', '.join(config.tags) if config.tags else 'None'}",
        f"",
        "## Strategy Configuration",
        f"Default Strategy: {config.default_strategy.value}",
        f"Parallel Strategies: {'Enabled' if config.enable_parallel_strategies else 'Disabled'}",
        f"Max Parallel Strategies: {config.max_parallel_strategies}",
        f"Global Timeout: {config.global_timeout}s",
        f"",
        "## Enabled Strategies",
    ]

    for name, profile in config.strategy_profiles.items():
        if profile.enabled:
            lines.append(f"  - {name}: weight={profile.performance_weight}, priority={profile.priority}")

    lines.extend([
        "",
        "## Resource Estimates",
    ])

    runtime = config.estimate_runtime()
    for strategy, time in runtime.items():
        lines.append(f"  {strategy}: ~{time:.1f}s")

    lines.extend([
        "",
        "## Performance Thresholds",
        f"Fast: {config.performance_thresholds.fast_time_threshold}s @ {config.performance_thresholds.fast_quality_threshold:.0%} quality",
        f"Balanced: {config.performance_thresholds.balanced_time_threshold}s @ {config.performance_thresholds.balanced_quality_threshold:.0%} quality",
        f"Thorough: {config.performance_thresholds.thorough_time_threshold}s @ {config.performance_thresholds.thorough_quality_threshold:.0%} quality",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create a configuration from preset
    config = HybridMakerConfigPreset.balanced()

    # Validate it
    valid, errors = config.validate()
    if valid:
        print("Configuration is valid!")
    else:
        print(f"Configuration errors: {errors}")

    # Print summary
    print("\n" + export_config_summary(config))

    # Save to file
    config.save_to_file("hybrid_maker_config_example.yaml")
    print("\nConfiguration saved to hybrid_maker_config_example.yaml")

    # Load from file
    loaded_config = HybridMakerConfig.load_from_file("hybrid_maker_config_example.yaml")
    if loaded_config:
        print("\nConfiguration loaded successfully!")
        print(f"Loaded config name: {loaded_config.config_name}")
