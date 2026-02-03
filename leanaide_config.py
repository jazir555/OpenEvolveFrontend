"""
LeanAide Configuration Module

Provides comprehensive configuration management for LeanAide integration with:
- Configuration dataclasses for all LeanAide components
- Multiple configuration sources (YAML, environment variables, Python API)
- Configuration validation with helpful error messages
- Sensible defaults that work out of the box
- Configuration migration support for future changes

Configuration Precedence:
1. Environment variables (LEANAIDE_*) - highest priority
2. YAML configuration files (leanaide_config.yaml, config.yaml)
3. Python API parameters
4. Default values - lowest priority

Usage:
    from leanaide_config import load_leanaide_config, get_leanaide_config

    # Load configuration
    config = load_leanaide_config()

    # Access configuration
    print(config.server.host)
    print(config.verification.complexity_threshold)

    # Or use the global instance
    config = get_leanaide_config()
"""

import os
import logging
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, field, asdict
from datetime import timedelta

from env_helpers import (
    env_var_str,
    env_var_int,
    env_var_float,
    env_var_bool,
    env_var_list,
    env_var_path,
    env_var_url,
    check_required_env_vars,
    is_production,
    is_development,
    ValidationError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class LeanAideServerConfig:
    """
    LeanAide server connection settings.

    Attributes:
        host: Server hostname or IP address
        port: Server port number
        timeout: Request timeout in seconds
        max_retries: Maximum number of connection retries
        retry_delay: Delay between retries in seconds
        use_ssl: Whether to use SSL/TLS encryption
        verify_ssl: Whether to verify SSL certificates
        api_version: API version to use
        health_check_interval: Seconds between health checks (0 = disabled)
    """
    host: str = "localhost"
    port: int = 8080
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    use_ssl: bool = False
    verify_ssl: bool = True
    api_version: str = "v1"
    health_check_interval: int = 60

    def get_base_url(self) -> str:
        """Get the base URL for LeanAide API."""
        scheme = "https" if self.use_ssl else "http"
        return f"{scheme}://{self.host}:{self.port}/{self.api_version}"


@dataclass
class LeanAideVerificationConfig:
    """
    LeanAide verification settings.

    Attributes:
        enable_auto: Enable automatic verification
        complexity_threshold: Minimum complexity to trigger verification (0-100)
        domains: List of Lean 4 domains to verify (e.g., ["mathlib", "analysis"])
        max_proof_depth: Maximum proof depth to verify
        timeout_per_proof: Timeout per proof verification in seconds
        parallel_verifications: Number of parallel verification threads
        strict_mode: Enable strict verification (fails on warnings)
        cache_verified_proofs: Cache successfully verified proofs
        verification_strategy: Strategy to use ("quick", "thorough", "adaptive")
        fallback_on_timeout: Whether to fallback on timeout
    """
    enable_auto: bool = True
    complexity_threshold: int = 50
    domains: List[str] = field(default_factory=lambda: ["mathlib"])
    max_proof_depth: int = 100
    timeout_per_proof: float = 120.0
    parallel_verifications: int = 4
    strict_mode: bool = False
    cache_verified_proofs: bool = True
    verification_strategy: str = "adaptive"
    fallback_on_timeout: bool = True

    # Advanced settings
    trust_level: float = 0.95  # Minimum trust level for verified proofs
    use_external_prover: bool = False  # Use external prover if available
    prover_timeout_multiplier: float = 2.0  # Multiplier for external prover timeout


@dataclass
class LeanAideCacheConfig:
    """
    LeanAide caching settings.

    Attributes:
        enable: Enable caching
        ttl: Cache time-to-live in seconds
        cache_dir: Directory for cache storage
        max_cache_size_mb: Maximum cache size in megabytes
        cache_proof_objects: Cache individual proof objects
        cache_dependencies: Cache proof dependencies
        compression_enabled: Enable cache compression
        persistent_cache: Use persistent cache (survives restarts)
    """
    enable: bool = True
    ttl: int = 86400  # 24 hours in seconds
    cache_dir: str = "./leanaide_cache"
    max_cache_size_mb: int = 500
    cache_proof_objects: bool = True
    cache_dependencies: bool = True
    compression_enabled: bool = True
    persistent_cache: bool = True

    # Cache invalidation settings
    invalidate_on_proof_change: bool = True
    invalidate_on_dependency_update: bool = True
    min_cache_hits_before_persist: int = 2


@dataclass
class LeanAideWorkflowConfig:
    """
    LeanAide workflow integration settings.

    Attributes:
        stage_3c_enabled: Enable LeanAide verification in Stage 3C (Decomposition Verification)
        stage_5_enabled: Enable LeanAide verification in Stage 5 (Solution Verification)
        stage_3c_priority: Priority for Stage 3C (1-10, higher = more important)
        stage_5_priority: Priority for Stage 5 (1-10, higher = more important)
        async_verification: Enable asynchronous verification
        block_on_verification: Block workflow until verification completes
        verification_timeout: Overall workflow verification timeout in seconds
        failure_action: Action on verification failure ("warn", "error", "continue", "fallback")
        progress_reporting: Enable verification progress reporting
    """
    stage_3c_enabled: bool = True
    stage_5_enabled: bool = True
    stage_3c_priority: int = 7
    stage_5_priority: int = 8
    async_verification: bool = True
    block_on_verification: bool = False
    verification_timeout: float = 600.0
    failure_action: str = "warn"  # warn, error, continue, fallback
    progress_reporting: bool = True

    # Integration settings
    inject_proof_hints: bool = True  # Inject verified proofs as hints
    use_verified_tactics: bool = True  # Use only verified tactics
    verification_results_in_output: bool = True  # Include verification results in final output


@dataclass
class LeanAideLean4Config:
    """
    Lean 4 environment configuration.

    Attributes:
        lean_path: Path to Lean 4 executable
        lean_pkg_path: Path to leanpkg tool
        mathlib_path: Path to MathLib library
        lake_path: Path to Lake build tool
        project_root: Root directory for Lean 4 projects
        output_dir: Directory for generated Lean 4 files
        import_paths: Additional import paths for Lean 4
        prelude: Custom prelude file (None = use default)
        use_lake: Use Lake for building (vs leanpkg)
    """
    lean_path: str = "lean"
    lean_pkg_path: str = "leanpkg"
    mathlib_path: Optional[str] = None
    lake_path: str = "lake"
    project_root: str = "./lean4_projects"
    output_dir: str = "./lean4_output"
    import_paths: List[str] = field(default_factory=list)
    prelude: Optional[str] = None
    use_lake: bool = True


@dataclass
class LeanAideLoggingConfig:
    """
    LeanAide logging configuration.

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None = stdout only)
        log_format: Log format string
        log_verification_details: Log detailed verification information
        log_proof_attempts: Log all proof attempts
        log_cache_hits: Log cache hits
        max_log_size_mb: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
    """
    level: str = "INFO"
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_verification_details: bool = False
    log_proof_attempts: bool = True
    log_cache_hits: bool = False
    max_log_size_mb: int = 100
    backup_count: int = 5


@dataclass
class LeanAideSecurityConfig:
    """
    LeanAide security configuration.

    Attributes:
        enable_sandboxing: Enable sandboxing for untrusted code
        sandbox_timeout: Sandbox execution timeout in seconds
        max_memory_mb: Maximum memory for sandboxed processes
        allow_network_access: Allow network access in sandbox
        trusted_domains: List of trusted domains for imports
        verify_imports: Verify all imports before execution
        enable_resource_limits: Enable resource limits
        max_cpu_time: Maximum CPU time per verification (seconds)
    """
    enable_sandboxing: bool = True
    sandbox_timeout: float = 300.0
    max_memory_mb: int = 2048
    allow_network_access: bool = False
    trusted_domains: List[str] = field(default_factory=lambda: ["mathlib", "std"])
    verify_imports: bool = True
    enable_resource_limits: bool = True
    max_cpu_time: float = 600.0


@dataclass
class LeanAidePerformanceConfig:
    """
    LeanAide performance tuning configuration.

    Attributes:
        worker_threads: Number of worker threads for parallel processing
        queue_size: Size of the task queue
        batch_size: Batch size for proof verification
        enable_profiling: Enable performance profiling
        profile_dir: Directory for profile data
        enable_optimization: Enable performance optimizations
        optimization_level: Optimization level (0-3)
        preload_mathlib: Preload common MathLib modules
        parallel_import_processing: Process imports in parallel
    """
    worker_threads: int = 4
    queue_size: int = 100
    batch_size: int = 10
    enable_profiling: bool = False
    profile_dir: str = "./leanaide_profiles"
    enable_optimization: bool = True
    optimization_level: int = 2
    preload_mathlib: bool = True
    parallel_import_processing: bool = True


@dataclass
class LeanAideMDAPMCTSConfig:
    """
    MDAP-MCTS hybrid configuration for LeanAide.

    This configuration integrates Multi-Agent Distributed Agreement Protocol (MDAP)
    with Monte Carlo Tree Search (MCTS) for enhanced automated theorem proving.

    Attributes:
        # MDAP settings
        mdap_enabled: Enable MDAP multi-agent system (default: True)
        mdap_num_agents: Number of MDAP agents to use (default: 4)
        mdap_agent_types: Types of agents (default: ["evolution", "mcts", "adversarial", "self_play"])
        mdap_voting_strategy: Voting strategy for MDAP (default: "first_k_ahead")
        mdap_k_ahead: K parameter for first-K-ahead voting (default: 3)
        mdap_consensus_threshold: Minimum confidence for consensus (default: 0.6)

        # MCTS settings
        mcts_enabled: Enable MCTS tree search (default: True)
        mcts_iterations: Number of MCTS iterations (default: 100)
        mcts_time_budget: Time budget for MCTS in seconds (default: 30.0)
        mcts_c_param: UCT exploration constant (default: 1.414)
        mcts_rollout_depth: Maximum rollout depth (default: 100)
        mcts_parallel_simulations: Number of parallel simulations (default: 4)

        # Hybrid settings
        hybrid_mode: Hybrid mode ("mcts_then_mdap", "mdap_then_mcts", "parallel", "adaptive") (default: "mcts_then_mdap")
        hybrid_ratio: Ratio of MCTS vs MDAP iterations (default: 0.5)
        agent_weight_bonus: Bonus weight for agent votes in UCT (default: 0.3)
        enable_mdap_selection: Use MDAP in MCTS selection (default: True)
        enable_mdap_expansion: Use MDAP in MCTS expansion (default: True)
        enable_mdap_simulation: Use MDAP in MCTS simulation (default: False)

        # Performance tracking
        track_agent_performance: Track individual agent performance (default: True)
        track_voting_statistics: Track voting statistics (default: True)
        track_convergence_rates: Track convergence rates (default: True)
        log_agent_decisions: Log individual agent decisions (default: False)

        # Advanced settings
        adaptive_agent_weights: Dynamically adjust agent weights (default: True)
        progressive_widening: Enable progressive widening (default: True)
        transposition_table: Enable transposition table (default: True)
        amaf_enabled: Enable AMAF (All-Moves-As-First) (default: True)
        amaf_alpha: AMAF mixing parameter (default: 0.5)
    """
    # MDAP settings
    mdap_enabled: bool = True
    mdap_num_agents: int = 4
    mdap_agent_types: List[str] = field(default_factory=lambda: ["evolution", "mcts", "adversarial", "self_play"])
    mdap_voting_strategy: str = "first_k_ahead"
    mdap_k_ahead: int = 3
    mdap_consensus_threshold: float = 0.6

    # MCTS settings
    mcts_enabled: bool = True
    mcts_iterations: int = 100
    mcts_time_budget: float = 30.0
    mcts_c_param: float = 1.414
    mcts_rollout_depth: int = 100
    mcts_parallel_simulations: int = 4

    # Hybrid settings
    hybrid_mode: str = "mcts_then_mdap"
    hybrid_ratio: float = 0.5
    agent_weight_bonus: float = 0.3
    enable_mdap_selection: bool = True
    enable_mdap_expansion: bool = True
    enable_mdap_simulation: bool = False

    # Performance tracking
    track_agent_performance: bool = True
    track_voting_statistics: bool = True
    track_convergence_rates: bool = True
    log_agent_decisions: bool = False

    # Advanced settings
    adaptive_agent_weights: bool = True
    progressive_widening: bool = True
    transposition_table: bool = True
    amaf_enabled: bool = True
    amaf_alpha: float = 0.5

    def validate(self) -> List[str]:
        """
        Validate MDAP-MCTS configuration.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Validate MDAP settings
        if self.mdap_num_agents < 1:
            errors.append(f"MDAP num_agents must be >= 1, got {self.mdap_num_agents}")
        if not (0.0 <= self.mdap_consensus_threshold <= 1.0):
            errors.append(f"MDAP consensus_threshold must be 0-1, got {self.mdap_consensus_threshold}")
        if self.mdap_k_ahead < 1:
            errors.append(f"MDAP k_ahead must be >= 1, got {self.mdap_k_ahead}")
        if self.mdap_voting_strategy not in ["first_k_ahead", "majority", "unanimous", "weighted"]:
            errors.append(f"Invalid MDAP voting_strategy: {self.mdap_voting_strategy}")

        # Validate MCTS settings
        if self.mcts_iterations < 1:
            errors.append(f"MCTS iterations must be >= 1, got {self.mcts_iterations}")
        if self.mcts_time_budget <= 0:
            errors.append(f"MCTS time_budget must be positive, got {self.mcts_time_budget}")
        if self.mcts_c_param <= 0:
            errors.append(f"MCTS c_param must be positive, got {self.mcts_c_param}")
        if self.mcts_rollout_depth < 1:
            errors.append(f"MCTS rollout_depth must be >= 1, got {self.mcts_rollout_depth}")
        if self.mcts_parallel_simulations < 1:
            errors.append(f"MCTS parallel_simulations must be >= 1, got {self.mcts_parallel_simulations}")

        # Validate hybrid settings
        if self.hybrid_mode not in ["mcts_then_mdap", "mdap_then_mcts", "parallel", "adaptive"]:
            errors.append(f"Invalid hybrid_mode: {self.hybrid_mode}")
        if not (0.0 <= self.hybrid_ratio <= 1.0):
            errors.append(f"Hybrid ratio must be 0-1, got {self.hybrid_ratio}")
        if not (0.0 <= self.agent_weight_bonus <= 1.0):
            errors.append(f"Agent weight bonus must be 0-1, got {self.agent_weight_bonus}")

        # Validate advanced settings
        if not (0.0 <= self.amaf_alpha <= 1.0):
            errors.append(f"AMAF alpha must be 0-1, got {self.amaf_alpha}")

        return errors

    def to_mcts_config(self):
        """Convert to MCTSConfig for leanaide_mcts."""
        from leanaide_mcts import MCTSConfig

        return MCTSConfig(
            max_iterations=self.mcts_iterations,
            time_budget=self.mcts_time_budget,
            c_param=self.mcts_c_param,
            rollout_depth=self.mcts_rollout_depth,
            parallel_simulations=self.mcts_parallel_simulations,
            enable_transposition_table=self.transposition_table,
            enable_amaf=self.amaf_enabled,
            amaf_alpha=self.amaf_alpha,
            progressive_widening=self.progressive_widening,
        )

    def to_mdap_mcts_config(self):
        """Convert to MDAPMCTSConfig for leanaide_mcts."""
        from leanaide_mcts import MDAPMCTSConfig

        return MDAPMCTSConfig(
            base_mcts_config=self.to_mcts_config(),
            use_mdap_selection=self.enable_mdap_selection,
            use_mdap_expansion=self.enable_mdap_expansion,
            use_mdap_simulation=self.enable_mdap_simulation,
            num_mdap_agents=self.mdap_num_agents,
            mdap_agent_types=self.mdap_agent_types,
            mdap_voting_strategy=self.mdap_voting_strategy,
            mdap_k_ahead=self.mdap_k_ahead,
            agent_weight_bonus=self.agent_weight_bonus,
            enable_mdap_mcts_hybrid=True,
            mdap_mcts_ratio=self.hybrid_ratio,
        )


@dataclass
class LeanAideMDAPEvolutionConfig:
    """
    MDAP-Evolution hybrid configuration for LeanAide.

    This configuration integrates Multi-Agent Distributed Agreement Protocol (MDAP)
    with evolutionary proof generation strategies.

    Attributes:
        # Evolution settings
        evolution_population_size: Population size for evolution (default: 20)
        evolution_max_generations: Maximum generations (default: 20)
        evolution_mutation_rate: Mutation rate (default: 0.1)
        evolution_crossover_rate: Crossover rate (default: 0.8)
        evolution_elitism_ratio: Elitism ratio (default: 0.1)
        evolution_selection_method: Selection method (default: "tournament")

        # MDAP settings
        mdap_enabled: Enable MDAP voting (default: True)
        mdap_num_agents: Number of MDAP agents (default: 4)
        mdap_agent_types: Agent types (default: ["evolution", "mcts", "adversarial", "self_play"])
        mdap_voting_strategy: Voting strategy (default: "first_k_ahead")
        mdap_k_ahead: K parameter for first-K-ahead (default: 3)
        mdap_consensus_threshold: Consensus threshold (default: 0.6)

        # Hybrid execution modes
        hybrid_mode: Hybrid execution mode (default: "mcts_then_mdap")
        hybrid_ratio: Ratio of evolution vs MDAP (default: 0.5)
        enable_mdap_parent_selection: Use MDAP for parent selection (default: True)
        enable_mdap_crossover: Use MDAP for crossover (default: True)
        enable_mdap_mutation: Use MDAP for mutation (default: True)

        # Performance tracking
        track_mdap_vs_pure: Track MDAP vs pure evolution (default: True)
        track_agent_contributions: Track agent contributions (default: True)
        track_voting_statistics: Track voting statistics (default: True)
        track_convergence_rates: Track convergence rates (default: True)
        log_agent_decisions: Log individual agent decisions (default: False)

        # Advanced settings
        adaptive_agent_weights: Adaptive agent weighting (default: True)
        progressive_widening: Progressive widening (default: True)
        enable_seeding: Enable MDAP seeding (default: True)
        seed_population_ratio: Ratio of population to seed (default: 0.3)
    """
    # Evolution settings
    evolution_population_size: int = 20
    evolution_max_generations: int = 20
    evolution_mutation_rate: float = 0.1
    evolution_crossover_rate: float = 0.8
    evolution_elitism_ratio: float = 0.1
    evolution_selection_method: str = "tournament"

    # MDAP settings
    mdap_enabled: bool = True
    mdap_num_agents: int = 4
    mdap_agent_types: List[str] = field(default_factory=lambda: ["evolution", "mcts", "adversarial", "self_play"])
    mdap_voting_strategy: str = "first_k_ahead"
    mdap_k_ahead: int = 3
    mdap_consensus_threshold: float = 0.6

    # Hybrid execution modes
    hybrid_mode: str = "mcts_then_mdap"
    hybrid_ratio: float = 0.5
    enable_mdap_parent_selection: bool = True
    enable_mdap_crossover: bool = True
    enable_mdap_mutation: bool = True

    # Performance tracking
    track_mdap_vs_pure: bool = True
    track_agent_contributions: bool = True
    track_voting_statistics: bool = True
    track_convergence_rates: bool = True
    log_agent_decisions: bool = False

    # Advanced settings
    adaptive_agent_weights: bool = True
    progressive_widening: bool = True
    enable_seeding: bool = True
    seed_population_ratio: float = 0.3

    def validate(self) -> List[str]:
        """
        Validate MDAP-evolution configuration.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Validate evolution settings
        if self.evolution_population_size < 1:
            errors.append(f"Evolution population size must be >= 1, got {self.evolution_population_size}")
        if self.evolution_max_generations < 1:
            errors.append(f"Evolution max generations must be >= 1, got {self.evolution_max_generations}")
        if not (0.0 <= self.evolution_mutation_rate <= 1.0):
            errors.append(f"Evolution mutation rate must be 0-1, got {self.evolution_mutation_rate}")
        if not (0.0 <= self.evolution_crossover_rate <= 1.0):
            errors.append(f"Evolution crossover rate must be 0-1, got {self.evolution_crossover_rate}")
        if not (0.0 <= self.evolution_elitism_ratio <= 1.0):
            errors.append(f"Evolution elitism ratio must be 0-1, got {self.evolution_elitism_ratio}")

        # Validate MDAP settings
        if self.mdap_num_agents < 1:
            errors.append(f"MDAP num agents must be >= 1, got {self.mdap_num_agents}")
        if not (0.0 <= self.mdap_consensus_threshold <= 1.0):
            errors.append(f"MDAP consensus threshold must be 0-1, got {self.mdap_consensus_threshold}")
        if self.mdap_k_ahead < 1:
            errors.append(f"MDAP k_ahead must be >= 1, got {self.mdap_k_ahead}")
        if self.mdap_voting_strategy not in ["first_k_ahead", "majority", "unanimous", "weighted"]:
            errors.append(f"Invalid MDAP voting strategy: {self.mdap_voting_strategy}")

        # Validate hybrid settings
        if self.hybrid_mode not in ["mcts_then_mdap", "mdap_then_mcts", "parallel", "adaptive"]:
            errors.append(f"Invalid hybrid mode: {self.hybrid_mode}")
        if not (0.0 <= self.hybrid_ratio <= 1.0):
            errors.append(f"Hybrid ratio must be 0-1, got {self.hybrid_ratio}")
        if not (0.0 <= self.seed_population_ratio <= 1.0):
            errors.append(f"Seed population ratio must be 0-1, got {self.seed_population_ratio}")

        return errors

    def to_evolution_config(self):
        """Convert to evolution config for leanaide_evolution."""
        from leanaide_evolution import MDAPMCTSGenerationConfig

        return MDAPMCTSGenerationConfig(
            mdap_num_agents=self.mdap_num_agents,
            mdap_agent_types=self.mdap_agent_types,
            mdap_voting_strategy=self.mdap_voting_strategy,
            mdap_k_ahead=self.mdap_k_ahead,
            mdap_consensus_threshold=self.mdap_consensus_threshold,
            hybrid_mode=self.hybrid_mode,
            hybrid_ratio=self.hybrid_ratio,
            agent_weight_bonus=0.3,
            enable_mdap_selection=self.enable_mdap_parent_selection,
            enable_mdap_expansion=self.enable_mdap_crossover,
            enable_mdap_simulation=self.enable_mdap_mutation,
            enable_mdap_mcts_hybrid=True,
            mdap_mcts_ratio=self.hybrid_ratio,
        )


@dataclass
class LeanAideAdversarialMDAPConfig:
    """
    MDAP-enhanced adversarial evolution configuration.

    Attributes:
        # Blue team MDAP settings
        blue_team_mdap_enabled: Enable MDAP for blue team (default: True)
        blue_team_agents: Number of blue team agents (default: 3)
        blue_team_voting: Blue team voting strategy (default: "first_k_ahead")

        # Red team MDAP settings
        red_team_mdap_enabled: Enable MDAP for red team (default: True)
        red_team_agents: Number of red team agents (default: 3)
        red_team_voting: Red team voting strategy (default: "weighted")

        # Consensus tracking
        track_consensus_rate: Track consensus rates (default: True)
        min_consensus_threshold: Minimum consensus threshold (default: 0.7)

        # Rounds
        adversarial_rounds: Number of adversarial rounds (default: 5)
        round_timeout: Round timeout in seconds (default: 60.0)
    """
    # Blue team MDAP settings
    blue_team_mdap_enabled: bool = True
    blue_team_agents: int = 3
    blue_team_voting: str = "first_k_ahead"

    # Red team MDAP settings
    red_team_mdap_enabled: bool = True
    red_team_agents: int = 3
    red_team_voting: str = "weighted"

    # Consensus tracking
    track_consensus_rate: bool = True
    min_consensus_threshold: float = 0.7

    # Rounds
    adversarial_rounds: int = 5
    round_timeout: float = 60.0

    def validate(self) -> List[str]:
        """Validate adversarial MDAP configuration."""
        errors = []

        if self.blue_team_agents < 1:
            errors.append(f"Blue team agents must be >= 1, got {self.blue_team_agents}")
        if self.red_team_agents < 1:
            errors.append(f"Red team agents must be >= 1, got {self.red_team_agents}")
        if not (0.0 <= self.min_consensus_threshold <= 1.0):
            errors.append(f"Min consensus threshold must be 0-1, got {self.min_consensus_threshold}")
        if self.adversarial_rounds < 1:
            errors.append(f"Adversarial rounds must be >= 1, got {self.adversarial_rounds}")

        return errors


@dataclass
class LeanAideSelfPlayMDAPConfig:
    """
    MDAP-enhanced self-play configuration.

    Attributes:
        # Self-play settings
        self_play_episodes: Number of self-play episodes (default: 50)
        agents_per_game: Agents per game (default: 4)
        learning_rate: Learning rate (default: 0.01)

        # MDAP consensus settings
        mdap_strategy_selection: Use MDAP for strategy selection (default: True)
        consensus_policy_updates: Consensus-based policy updates (default: True)
        voting_weight: Voting weight for policy updates (default: 0.7)

        # Exploration
        exploration_rate: Exploration rate (default: 0.3)
        exploration_decay: Exploration decay (default: 0.995)
    """
    # Self-play settings
    self_play_episodes: int = 50
    agents_per_game: int = 4
    learning_rate: float = 0.01

    # MDAP consensus settings
    mdap_strategy_selection: bool = True
    consensus_policy_updates: bool = True
    voting_weight: float = 0.7

    # Exploration
    exploration_rate: float = 0.3
    exploration_decay: float = 0.995

    def validate(self) -> List[str]:
        """Validate self-play MDAP configuration."""
        errors = []

        if self.self_play_episodes < 1:
            errors.append(f"Self-play episodes must be >= 1, got {self.self_play_episodes}")
        if self.agents_per_game < 1:
            errors.append(f"Agents per game must be >= 1, got {self.agents_per_game}")
        if not (0.0 <= self.learning_rate <= 1.0):
            errors.append(f"Learning rate must be 0-1, got {self.learning_rate}")
        if not (0.0 <= self.voting_weight <= 1.0):
            errors.append(f"Voting weight must be 0-1, got {self.voting_weight}")
        if not (0.0 <= self.exploration_rate <= 1.0):
            errors.append(f"Exploration rate must be 0-1, got {self.exploration_rate}")

        return errors


@dataclass
class LeanAideConfig:
    """
    Main LeanAide configuration container.

    This is the primary configuration class that contains all sub-configurations.
    Use load_leanaide_config() to create an instance with proper validation.
    """
    server: LeanAideServerConfig = field(default_factory=LeanAideServerConfig)
    verification: LeanAideVerificationConfig = field(default_factory=LeanAideVerificationConfig)
    cache: LeanAideCacheConfig = field(default_factory=LeanAideCacheConfig)
    workflow: LeanAideWorkflowConfig = field(default_factory=LeanAideWorkflowConfig)
    lean4: LeanAideLean4Config = field(default_factory=LeanAideLean4Config)
    logging: LeanAideLoggingConfig = field(default_factory=LeanAideLoggingConfig)
    security: LeanAideSecurityConfig = field(default_factory=LeanAideSecurityConfig)
    performance: LeanAidePerformanceConfig = field(default_factory=LeanAidePerformanceConfig)
    mdap_mcts: LeanAideMDAPMCTSConfig = field(default_factory=LeanAideMDAPMCTSConfig)
    mdap_evolution: LeanAideMDAPEvolutionConfig = field(default_factory=LeanAideMDAPEvolutionConfig)
    mdap_adversarial: LeanAideAdversarialMDAPConfig = field(default_factory=LeanAideAdversarialMDAPConfig)
    mdap_selfplay: LeanAideSelfPlayMDAPConfig = field(default_factory=LeanAideSelfPlayMDAPConfig)

    # Global settings
    enabled: bool = True
    environment: str = "development"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)."""
        return asdict(self)

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Validate server config
        if not (1 <= self.server.port <= 65535):
            errors.append(f"Server port must be 1-65535, got {self.server.port}")
        if self.server.timeout <= 0:
            errors.append(f"Server timeout must be positive, got {self.server.timeout}")

        # Validate verification config
        if not (0 <= self.verification.complexity_threshold <= 100):
            errors.append(f"Complexity threshold must be 0-100, got {self.verification.complexity_threshold}")
        if self.verification.timeout_per_proof <= 0:
            errors.append(f"Proof timeout must be positive, got {self.verification.timeout_per_proof}")
        if self.verification.parallel_verifications < 1:
            errors.append(f"Parallel verifications must be >= 1, got {self.verification.parallel_verifications}")
        if self.verification.verification_strategy not in ["quick", "thorough", "adaptive"]:
            errors.append(f"Invalid verification strategy: {self.verification.verification_strategy}")

        # Validate cache config
        if self.cache.ttl <= 0:
            errors.append(f"Cache TTL must be positive, got {self.cache.ttl}")
        if self.cache.max_cache_size_mb <= 0:
            errors.append(f"Max cache size must be positive, got {self.cache.max_cache_size_mb}")

        # Validate workflow config
        if not (1 <= self.workflow.stage_3c_priority <= 10):
            errors.append(f"Stage 3C priority must be 1-10, got {self.workflow.stage_3c_priority}")
        if not (1 <= self.workflow.stage_5_priority <= 10):
            errors.append(f"Stage 5 priority must be 1-10, got {self.workflow.stage_5_priority}")
        if self.workflow.failure_action not in ["warn", "error", "continue", "fallback"]:
            errors.append(f"Invalid failure action: {self.workflow.failure_action}")

        # Validate performance config
        if self.performance.worker_threads < 1:
            errors.append(f"Worker threads must be >= 1, got {self.performance.worker_threads}")
        if not (0 <= self.performance.optimization_level <= 3):
            errors.append(f"Optimization level must be 0-3, got {self.performance.optimization_level}")

        # Validate MDAP-MCTS config
        errors.extend(self.mdap_mcts.validate())

        # Validate MDAP-evolution config
        errors.extend(self.mdap_evolution.validate())

        # Validate MDAP-adversarial config
        errors.extend(self.mdap_adversarial.validate())

        # Validate MDAP-selfplay config
        errors.extend(self.mdap_selfplay.validate())

        return errors


# =============================================================================
# Configuration Loader
# =============================================================================

class LeanAideConfigLoader:
    """
    Load and validate LeanAide configuration from multiple sources.

    Configuration sources (in order of precedence):
    1. Environment variables (LEANAIDE_*)
    2. YAML files (leanaide_config.yaml, config.yaml)
    3. Python API parameters
    4. Default values
    """

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        env_prefix: str = "LEANAIDE_"
    ):
        """
        Initialize configuration loader.

        Args:
            config_dir: Directory containing configuration files
            env_prefix: Prefix for environment variables
        """
        self.config_dir = config_dir or Path.cwd()
        self.env_prefix = env_prefix
        self.config_file = self.config_dir / "leanaide_config.yaml"
        self.fallback_config_file = self.config_dir / "config.yaml"
        self._raw_config: Dict[str, Any] = {}

    def load(
        self,
        **overrides
    ) -> LeanAideConfig:
        """
        Load configuration from all sources.

        Args:
            **overrides: Python API overrides (highest precedence)

        Returns:
            Validated LeanAideConfig object

        Raises:
            ValidationError: If configuration is invalid
        """
        logger.info("Loading LeanAide configuration...")

        # Load from files
        self._load_from_files()

        # Create config with proper precedence
        config = self._create_config(overrides)

        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValidationError(
                f"LeanAide configuration validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

        # Log successful load
        self._log_config(config)

        return config

    def _load_from_files(self) -> None:
        """Load configuration from YAML files."""
        # Try leanaide_config.yaml first
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    data = yaml.safe_load(f)
                    if data:
                        # Extract leanaide section if it exists
                        if "leanaide" in data:
                            self._raw_config.update(data["leanaide"])
                        else:
                            self._raw_config.update(data)
                logger.info(f"Loaded configuration from {self.config_file}")
            except (IOError, yaml.YAMLError, TypeError) as e:
                logger.warning(f"Failed to load {self.config_file}: {e}")

        # Try config.yaml as fallback
        elif self.fallback_config_file.exists():
            try:
                with open(self.fallback_config_file, "r") as f:
                    data = yaml.safe_load(f)
                    if data and "leanaide" in data:
                        self._raw_config.update(data["leanaide"])
                logger.info(f"Loaded LeanAide configuration from {self.fallback_config_file}")
            except (IOError, yaml.YAMLError, TypeError) as e:
                logger.warning(f"Failed to load {self.fallback_config_file}: {e}")
        else:
            logger.info(f"No LeanAide configuration file found, using defaults")

    def _create_config(self, overrides: Dict[str, Any]) -> LeanAideConfig:
        """Create configuration object with proper precedence."""
        # Helper to get value with precedence
        def get_value(
            section: str,
            key: str,
            env_type: str,
            default: Any,
            **kwargs
        ) -> Any:
            # 1. Check Python API overrides first (highest precedence)
            value = None
            if section:
                # Try both section__key and section_key formats
                override_key_double = f"{section}__{key}"
                override_key_single = f"{section}_{key}"
                if override_key_double in overrides:
                    value = overrides[override_key_double]
                elif override_key_single in overrides:
                    value = overrides[override_key_single]
            if value is None and key in overrides:
                value = overrides[key]

            # 2. Check environment variables
            env_name = f"{self.env_prefix}{section.upper()}_{key.upper()}"
            env_var_flat = f"{self.env_prefix}{key.upper()}"  # Also try flat name

            if value is None:
                # Determine if env var is set
                if os.environ.get(env_name) is not None:
                    try:
                        if env_type == "str":
                            value = env_var_str(env_name, default=None)
                        elif env_type == "int":
                            value = env_var_int(env_name, default=None, **kwargs)
                        elif env_type == "float":
                            value = env_var_float(env_name, default=None, **kwargs)
                        elif env_type == "bool":
                            value = env_var_bool(env_name, default=None)
                        elif env_type == "list":
                            value = env_var_list(env_name, default=None)
                    except ValidationError:
                        # Re-raise if env var is set but invalid
                        raise
                # Try flat env_var
                elif os.environ.get(env_var_flat) is not None:
                    try:
                        if env_type == "str":
                            value = env_var_str(env_var_flat, default=None)
                        elif env_type == "int":
                            value = env_var_int(env_var_flat, default=None, **kwargs)
                        elif env_type == "float":
                            value = env_var_float(env_var_flat, default=None, **kwargs)
                        elif env_type == "bool":
                            value = env_var_bool(env_var_flat, default=None)
                        elif env_type == "list":
                            value = env_var_list(env_var_flat, default=None)
                    except ValidationError:
                        # Re-raise if env var is set but invalid
                        raise

            # 3. Check YAML config
            if value is None:
                section_data = self._raw_config.get(section, {})
                if key in section_data:
                    value = section_data[key]

            # 4. Use default
            if value is None:
                value = default

            return value

        # Extract sections from raw config
        server_data = self._raw_config.get("server", {})
        verification_data = self._raw_config.get("verification", {})
        cache_data = self._raw_config.get("cache", {})
        workflow_data = self._raw_config.get("workflow", {})
        lean4_data = self._raw_config.get("lean4", {})
        logging_data = self._raw_config.get("logging", {})
        security_data = self._raw_config.get("security", {})
        performance_data = self._raw_config.get("performance", {})
        mdap_mcts_data = self._raw_config.get("mdap_mcts", {})
        mdap_evolution_data = self._raw_config.get("mdap_evolution", {})
        mdap_adversarial_data = self._raw_config.get("mdap_adversarial", {})
        mdap_selfplay_data = self._raw_config.get("mdap_selfplay", {})

        # Server config
        server = LeanAideServerConfig(
            host=get_value("server", "host", "str", server_data.get("host", "localhost")),
            port=get_value("server", "port", "int", server_data.get("port", 8080), min_val=1, max_val=65535),
            timeout=get_value("server", "timeout", "float", server_data.get("timeout", 30.0), min_val=0.1),
            max_retries=get_value("server", "max_retries", "int", server_data.get("max_retries", 3), min_val=0),
            retry_delay=get_value("server", "retry_delay", "float", server_data.get("retry_delay", 1.0), min_val=0),
            use_ssl=get_value("server", "use_ssl", "bool", server_data.get("use_ssl", False)),
            verify_ssl=get_value("server", "verify_ssl", "bool", server_data.get("verify_ssl", True)),
            api_version=get_value("server", "api_version", "str", server_data.get("api_version", "v1")),
            health_check_interval=get_value("server", "health_check_interval", "int", server_data.get("health_check_interval", 60), min_val=0),
        )

        # Verification config
        verification = LeanAideVerificationConfig(
            enable_auto=get_value("verification", "enable_auto", "bool", verification_data.get("enable_auto", True)),
            complexity_threshold=get_value("verification", "complexity_threshold", "int", verification_data.get("complexity_threshold", 50), min_val=0, max_val=100),
            domains=get_value("verification", "domains", "list", verification_data.get("domains", ["mathlib"])),
            max_proof_depth=get_value("verification", "max_proof_depth", "int", verification_data.get("max_proof_depth", 100), min_val=1),
            timeout_per_proof=get_value("verification", "timeout_per_proof", "float", verification_data.get("timeout_per_proof", 120.0), min_val=1.0),
            parallel_verifications=get_value("verification", "parallel_verifications", "int", verification_data.get("parallel_verifications", 4), min_val=1),
            strict_mode=get_value("verification", "strict_mode", "bool", verification_data.get("strict_mode", False)),
            cache_verified_proofs=get_value("verification", "cache_verified_proofs", "bool", verification_data.get("cache_verified_proofs", True)),
            verification_strategy=get_value("verification", "verification_strategy", "str", verification_data.get("verification_strategy", "adaptive")),
            fallback_on_timeout=get_value("verification", "fallback_on_timeout", "bool", verification_data.get("fallback_on_timeout", True)),
            trust_level=get_value("verification", "trust_level", "float", verification_data.get("trust_level", 0.95), min_val=0.0, max_val=1.0),
            use_external_prover=get_value("verification", "use_external_prover", "bool", verification_data.get("use_external_prover", False)),
            prover_timeout_multiplier=get_value("verification", "prover_timeout_multiplier", "float", verification_data.get("prover_timeout_multiplier", 2.0), min_val=1.0),
        )

        # Cache config
        cache = LeanAideCacheConfig(
            enable=get_value("cache", "enable", "bool", cache_data.get("enable", True)),
            ttl=get_value("cache", "ttl", "int", cache_data.get("ttl", 86400), min_val=1),
            cache_dir=get_value("cache", "cache_dir", "str", cache_data.get("cache_dir", "./leanaide_cache")),
            max_cache_size_mb=get_value("cache", "max_cache_size_mb", "int", cache_data.get("max_cache_size_mb", 500), min_val=1),
            cache_proof_objects=get_value("cache", "cache_proof_objects", "bool", cache_data.get("cache_proof_objects", True)),
            cache_dependencies=get_value("cache", "cache_dependencies", "bool", cache_data.get("cache_dependencies", True)),
            compression_enabled=get_value("cache", "compression_enabled", "bool", cache_data.get("compression_enabled", True)),
            persistent_cache=get_value("cache", "persistent_cache", "bool", cache_data.get("persistent_cache", True)),
            invalidate_on_proof_change=get_value("cache", "invalidate_on_proof_change", "bool", cache_data.get("invalidate_on_proof_change", True)),
            invalidate_on_dependency_update=get_value("cache", "invalidate_on_dependency_update", "bool", cache_data.get("invalidate_on_dependency_update", True)),
            min_cache_hits_before_persist=get_value("cache", "min_cache_hits_before_persist", "int", cache_data.get("min_cache_hits_before_persist", 2), min_val=1),
        )

        # Workflow config
        workflow = LeanAideWorkflowConfig(
            stage_3c_enabled=get_value("workflow", "stage_3c_enabled", "bool", workflow_data.get("stage_3c_enabled", True)),
            stage_5_enabled=get_value("workflow", "stage_5_enabled", "bool", workflow_data.get("stage_5_enabled", True)),
            stage_3c_priority=get_value("workflow", "stage_3c_priority", "int", workflow_data.get("stage_3c_priority", 7), min_val=1, max_val=10),
            stage_5_priority=get_value("workflow", "stage_5_priority", "int", workflow_data.get("stage_5_priority", 8), min_val=1, max_val=10),
            async_verification=get_value("workflow", "async_verification", "bool", workflow_data.get("async_verification", True)),
            block_on_verification=get_value("workflow", "block_on_verification", "bool", workflow_data.get("block_on_verification", False)),
            verification_timeout=get_value("workflow", "verification_timeout", "float", workflow_data.get("verification_timeout", 600.0), min_val=1.0),
            failure_action=get_value("workflow", "failure_action", "str", workflow_data.get("failure_action", "warn")),
            progress_reporting=get_value("workflow", "progress_reporting", "bool", workflow_data.get("progress_reporting", True)),
            inject_proof_hints=get_value("workflow", "inject_proof_hints", "bool", workflow_data.get("inject_proof_hints", True)),
            use_verified_tactics=get_value("workflow", "use_verified_tactics", "bool", workflow_data.get("use_verified_tactics", True)),
            verification_results_in_output=get_value("workflow", "verification_results_in_output", "bool", workflow_data.get("verification_results_in_output", True)),
        )

        # Lean4 config
        lean4 = LeanAideLean4Config(
            lean_path=get_value("lean4", "lean_path", "str", lean4_data.get("lean_path", "lean")),
            lean_pkg_path=get_value("lean4", "lean_pkg_path", "str", lean4_data.get("lean_pkg_path", "leanpkg")),
            mathlib_path=get_value("lean4", "mathlib_path", "str", lean4_data.get("mathlib_path")),
            lake_path=get_value("lean4", "lake_path", "str", lean4_data.get("lake_path", "lake")),
            project_root=get_value("lean4", "project_root", "str", lean4_data.get("project_root", "./lean4_projects")),
            output_dir=get_value("lean4", "output_dir", "str", lean4_data.get("output_dir", "./lean4_output")),
            import_paths=get_value("lean4", "import_paths", "list", lean4_data.get("import_paths", [])),
            prelude=get_value("lean4", "prelude", "str", lean4_data.get("prelude")),
            use_lake=get_value("lean4", "use_lake", "bool", lean4_data.get("use_lake", True)),
        )

        # Logging config
        logging_config = LeanAideLoggingConfig(
            level=get_value("logging", "level", "str", logging_data.get("level", "INFO")),
            log_file=get_value("logging", "log_file", "str", logging_data.get("log_file")),
            log_format=get_value("logging", "log_format", "str", logging_data.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")),
            log_verification_details=get_value("logging", "log_verification_details", "bool", logging_data.get("log_verification_details", False)),
            log_proof_attempts=get_value("logging", "log_proof_attempts", "bool", logging_data.get("log_proof_attempts", True)),
            log_cache_hits=get_value("logging", "log_cache_hits", "bool", logging_data.get("log_cache_hits", False)),
            max_log_size_mb=get_value("logging", "max_log_size_mb", "int", logging_data.get("max_log_size_mb", 100), min_val=1),
            backup_count=get_value("logging", "backup_count", "int", logging_data.get("backup_count", 5), min_val=0),
        )

        # Security config
        security = LeanAideSecurityConfig(
            enable_sandboxing=get_value("security", "enable_sandboxing", "bool", security_data.get("enable_sandboxing", True)),
            sandbox_timeout=get_value("security", "sandbox_timeout", "float", security_data.get("sandbox_timeout", 300.0), min_val=1.0),
            max_memory_mb=get_value("security", "max_memory_mb", "int", security_data.get("max_memory_mb", 2048), min_val=128),
            allow_network_access=get_value("security", "allow_network_access", "bool", security_data.get("allow_network_access", False)),
            trusted_domains=get_value("security", "trusted_domains", "list", security_data.get("trusted_domains", ["mathlib", "std"])),
            verify_imports=get_value("security", "verify_imports", "bool", security_data.get("verify_imports", True)),
            enable_resource_limits=get_value("security", "enable_resource_limits", "bool", security_data.get("enable_resource_limits", True)),
            max_cpu_time=get_value("security", "max_cpu_time", "float", security_data.get("max_cpu_time", 600.0), min_val=1.0),
        )

        # Performance config
        performance = LeanAidePerformanceConfig(
            worker_threads=get_value("performance", "worker_threads", "int", performance_data.get("worker_threads", 4), min_val=1),
            queue_size=get_value("performance", "queue_size", "int", performance_data.get("queue_size", 100), min_val=1),
            batch_size=get_value("performance", "batch_size", "int", performance_data.get("batch_size", 10), min_val=1),
            enable_profiling=get_value("performance", "enable_profiling", "bool", performance_data.get("enable_profiling", False)),
            profile_dir=get_value("performance", "profile_dir", "str", performance_data.get("profile_dir", "./leanaide_profiles")),
            enable_optimization=get_value("performance", "enable_optimization", "bool", performance_data.get("enable_optimization", True)),
            optimization_level=get_value("performance", "optimization_level", "int", performance_data.get("optimization_level", 2), min_val=0, max_val=3),
            preload_mathlib=get_value("performance", "preload_mathlib", "bool", performance_data.get("preload_mathlib", True)),
            parallel_import_processing=get_value("performance", "parallel_import_processing", "bool", performance_data.get("parallel_import_processing", True)),
        )

        # MDAP-MCTS config
        mdap_mcts = LeanAideMDAPMCTSConfig(
            mdap_enabled=get_value("mdap_mcts", "mdap_enabled", "bool", mdap_mcts_data.get("mdap_enabled", True)),
            mdap_num_agents=get_value("mdap_mcts", "mdap_num_agents", "int", mdap_mcts_data.get("mdap_num_agents", 4), min_val=1),
            mdap_agent_types=get_value("mdap_mcts", "mdap_agent_types", "list", mdap_mcts_data.get("mdap_agent_types", ["evolution", "mcts", "adversarial", "self_play"])),
            mdap_voting_strategy=get_value("mdap_mcts", "mdap_voting_strategy", "str", mdap_mcts_data.get("mdap_voting_strategy", "first_k_ahead")),
            mdap_k_ahead=get_value("mdap_mcts", "mdap_k_ahead", "int", mdap_mcts_data.get("mdap_k_ahead", 3), min_val=1),
            mdap_consensus_threshold=get_value("mdap_mcts", "mdap_consensus_threshold", "float", mdap_mcts_data.get("mdap_consensus_threshold", 0.6), min_val=0.0, max_val=1.0),
            mcts_enabled=get_value("mdap_mcts", "mcts_enabled", "bool", mdap_mcts_data.get("mcts_enabled", True)),
            mcts_iterations=get_value("mdap_mcts", "mcts_iterations", "int", mdap_mcts_data.get("mcts_iterations", 100), min_val=1),
            mcts_time_budget=get_value("mdap_mcts", "mcts_time_budget", "float", mdap_mcts_data.get("mcts_time_budget", 30.0), min_val=0.1),
            mcts_c_param=get_value("mdap_mcts", "mcts_c_param", "float", mdap_mcts_data.get("mcts_c_param", 1.414), min_val=0.1),
            mcts_rollout_depth=get_value("mdap_mcts", "mcts_rollout_depth", "int", mdap_mcts_data.get("mcts_rollout_depth", 100), min_val=1),
            mcts_parallel_simulations=get_value("mdap_mcts", "mcts_parallel_simulations", "int", mdap_mcts_data.get("mcts_parallel_simulations", 4), min_val=1),
            hybrid_mode=get_value("mdap_mcts", "hybrid_mode", "str", mdap_mcts_data.get("hybrid_mode", "mcts_then_mdap")),
            hybrid_ratio=get_value("mdap_mcts", "hybrid_ratio", "float", mdap_mcts_data.get("hybrid_ratio", 0.5), min_val=0.0, max_val=1.0),
            agent_weight_bonus=get_value("mdap_mcts", "agent_weight_bonus", "float", mdap_mcts_data.get("agent_weight_bonus", 0.3), min_val=0.0, max_val=1.0),
            enable_mdap_selection=get_value("mdap_mcts", "enable_mdap_selection", "bool", mdap_mcts_data.get("enable_mdap_selection", True)),
            enable_mdap_expansion=get_value("mdap_mcts", "enable_mdap_expansion", "bool", mdap_mcts_data.get("enable_mdap_expansion", True)),
            enable_mdap_simulation=get_value("mdap_mcts", "enable_mdap_simulation", "bool", mdap_mcts_data.get("enable_mdap_simulation", False)),
            track_agent_performance=get_value("mdap_mcts", "track_agent_performance", "bool", mdap_mcts_data.get("track_agent_performance", True)),
            track_voting_statistics=get_value("mdap_mcts", "track_voting_statistics", "bool", mdap_mcts_data.get("track_voting_statistics", True)),
            track_convergence_rates=get_value("mdap_mcts", "track_convergence_rates", "bool", mdap_mcts_data.get("track_convergence_rates", True)),
            log_agent_decisions=get_value("mdap_mcts", "log_agent_decisions", "bool", mdap_mcts_data.get("log_agent_decisions", False)),
            adaptive_agent_weights=get_value("mdap_mcts", "adaptive_agent_weights", "bool", mdap_mcts_data.get("adaptive_agent_weights", True)),
            progressive_widening=get_value("mdap_mcts", "progressive_widening", "bool", mdap_mcts_data.get("progressive_widening", True)),
            transposition_table=get_value("mdap_mcts", "transposition_table", "bool", mdap_mcts_data.get("transposition_table", True)),
            amaf_enabled=get_value("mdap_mcts", "amaf_enabled", "bool", mdap_mcts_data.get("amaf_enabled", True)),
            amaf_alpha=get_value("mdap_mcts", "amaf_alpha", "float", mdap_mcts_data.get("amaf_alpha", 0.5), min_val=0.0, max_val=1.0),
        )

        # MDAP-Evolution config
        mdap_evolution = LeanAideMDAPEvolutionConfig(
            evolution_population_size=get_value("mdap_evolution", "evolution_population_size", "int", mdap_evolution_data.get("evolution_population_size", 20), min_val=1),
            evolution_max_generations=get_value("mdap_evolution", "evolution_max_generations", "int", mdap_evolution_data.get("evolution_max_generations", 20), min_val=1),
            evolution_mutation_rate=get_value("mdap_evolution", "evolution_mutation_rate", "float", mdap_evolution_data.get("evolution_mutation_rate", 0.1), min_val=0.0, max_val=1.0),
            evolution_crossover_rate=get_value("mdap_evolution", "evolution_crossover_rate", "float", mdap_evolution_data.get("evolution_crossover_rate", 0.8), min_val=0.0, max_val=1.0),
            evolution_elitism_ratio=get_value("mdap_evolution", "evolution_elitism_ratio", "float", mdap_evolution_data.get("evolution_elitism_ratio", 0.1), min_val=0.0, max_val=1.0),
            evolution_selection_method=get_value("mdap_evolution", "evolution_selection_method", "str", mdap_evolution_data.get("evolution_selection_method", "tournament")),
            mdap_enabled=get_value("mdap_evolution", "mdap_enabled", "bool", mdap_evolution_data.get("mdap_enabled", True)),
            mdap_num_agents=get_value("mdap_evolution", "mdap_num_agents", "int", mdap_evolution_data.get("mdap_num_agents", 4), min_val=1),
            mdap_agent_types=get_value("mdap_evolution", "mdap_agent_types", "list", mdap_evolution_data.get("mdap_agent_types", ["evolution", "mcts", "adversarial", "self_play"])),
            mdap_voting_strategy=get_value("mdap_evolution", "mdap_voting_strategy", "str", mdap_evolution_data.get("mdap_voting_strategy", "first_k_ahead")),
            mdap_k_ahead=get_value("mdap_evolution", "mdap_k_ahead", "int", mdap_evolution_data.get("mdap_k_ahead", 3), min_val=1),
            mdap_consensus_threshold=get_value("mdap_evolution", "mdap_consensus_threshold", "float", mdap_evolution_data.get("mdap_consensus_threshold", 0.6), min_val=0.0, max_val=1.0),
            hybrid_mode=get_value("mdap_evolution", "hybrid_mode", "str", mdap_evolution_data.get("hybrid_mode", "mcts_then_mdap")),
            hybrid_ratio=get_value("mdap_evolution", "hybrid_ratio", "float", mdap_evolution_data.get("hybrid_ratio", 0.5), min_val=0.0, max_val=1.0),
            enable_mdap_parent_selection=get_value("mdap_evolution", "enable_mdap_parent_selection", "bool", mdap_evolution_data.get("enable_mdap_parent_selection", True)),
            enable_mdap_crossover=get_value("mdap_evolution", "enable_mdap_crossover", "bool", mdap_evolution_data.get("enable_mdap_crossover", True)),
            enable_mdap_mutation=get_value("mdap_evolution", "enable_mdap_mutation", "bool", mdap_evolution_data.get("enable_mdap_mutation", True)),
            track_mdap_vs_pure=get_value("mdap_evolution", "track_mdap_vs_pure", "bool", mdap_evolution_data.get("track_mdap_vs_pure", True)),
            track_agent_contributions=get_value("mdap_evolution", "track_agent_contributions", "bool", mdap_evolution_data.get("track_agent_contributions", True)),
            track_voting_statistics=get_value("mdap_evolution", "track_voting_statistics", "bool", mdap_evolution_data.get("track_voting_statistics", True)),
            track_convergence_rates=get_value("mdap_evolution", "track_convergence_rates", "bool", mdap_evolution_data.get("track_convergence_rates", True)),
            log_agent_decisions=get_value("mdap_evolution", "log_agent_decisions", "bool", mdap_evolution_data.get("log_agent_decisions", False)),
            adaptive_agent_weights=get_value("mdap_evolution", "adaptive_agent_weights", "bool", mdap_evolution_data.get("adaptive_agent_weights", True)),
            progressive_widening=get_value("mdap_evolution", "progressive_widening", "bool", mdap_evolution_data.get("progressive_widening", True)),
            enable_seeding=get_value("mdap_evolution", "enable_seeding", "bool", mdap_evolution_data.get("enable_seeding", True)),
            seed_population_ratio=get_value("mdap_evolution", "seed_population_ratio", "float", mdap_evolution_data.get("seed_population_ratio", 0.3), min_val=0.0, max_val=1.0),
        )

        # MDAP-Adversarial config
        mdap_adversarial = LeanAideAdversarialMDAPConfig(
            blue_team_mdap_enabled=get_value("mdap_adversarial", "blue_team_mdap_enabled", "bool", mdap_adversarial_data.get("blue_team_mdap_enabled", True)),
            blue_team_agents=get_value("mdap_adversarial", "blue_team_agents", "int", mdap_adversarial_data.get("blue_team_agents", 3), min_val=1),
            blue_team_voting=get_value("mdap_adversarial", "blue_team_voting", "str", mdap_adversarial_data.get("blue_team_voting", "first_k_ahead")),
            red_team_mdap_enabled=get_value("mdap_adversarial", "red_team_mdap_enabled", "bool", mdap_adversarial_data.get("red_team_mdap_enabled", True)),
            red_team_agents=get_value("mdap_adversarial", "red_team_agents", "int", mdap_adversarial_data.get("red_team_agents", 3), min_val=1),
            red_team_voting=get_value("mdap_adversarial", "red_team_voting", "str", mdap_adversarial_data.get("red_team_voting", "weighted")),
            track_consensus_rate=get_value("mdap_adversarial", "track_consensus_rate", "bool", mdap_adversarial_data.get("track_consensus_rate", True)),
            min_consensus_threshold=get_value("mdap_adversarial", "min_consensus_threshold", "float", mdap_adversarial_data.get("min_consensus_threshold", 0.7), min_val=0.0, max_val=1.0),
            adversarial_rounds=get_value("mdap_adversarial", "adversarial_rounds", "int", mdap_adversarial_data.get("adversarial_rounds", 5), min_val=1),
            round_timeout=get_value("mdap_adversarial", "round_timeout", "float", mdap_adversarial_data.get("round_timeout", 60.0), min_val=1.0),
        )

        # MDAP-SelfPlay config
        mdap_selfplay = LeanAideSelfPlayMDAPConfig(
            self_play_episodes=get_value("mdap_selfplay", "self_play_episodes", "int", mdap_selfplay_data.get("self_play_episodes", 50), min_val=1),
            agents_per_game=get_value("mdap_selfplay", "agents_per_game", "int", mdap_selfplay_data.get("agents_per_game", 4), min_val=1),
            learning_rate=get_value("mdap_selfplay", "learning_rate", "float", mdap_selfplay_data.get("learning_rate", 0.01), min_val=0.0, max_val=1.0),
            mdap_strategy_selection=get_value("mdap_selfplay", "mdap_strategy_selection", "bool", mdap_selfplay_data.get("mdap_strategy_selection", True)),
            consensus_policy_updates=get_value("mdap_selfplay", "consensus_policy_updates", "bool", mdap_selfplay_data.get("consensus_policy_updates", True)),
            voting_weight=get_value("mdap_selfplay", "voting_weight", "float", mdap_selfplay_data.get("voting_weight", 0.7), min_val=0.0, max_val=1.0),
            exploration_rate=get_value("mdap_selfplay", "exploration_rate", "float", mdap_selfplay_data.get("exploration_rate", 0.3), min_val=0.0, max_val=1.0),
            exploration_decay=get_value("mdap_selfplay", "exploration_decay", "float", mdap_selfplay_data.get("exploration_decay", 0.995), min_val=0.0, max_val=1.0),
        )

        # Global settings
        enabled = get_value("", "enabled", "bool", self._raw_config.get("enabled", True))
        environment = get_value("", "environment", "str", self._raw_config.get("environment", "development"))

        return LeanAideConfig(
            server=server,
            verification=verification,
            cache=cache,
            workflow=workflow,
            lean4=lean4,
            logging=logging_config,
            security=security,
            performance=performance,
            mdap_mcts=mdap_mcts,
            mdap_evolution=mdap_evolution,
            mdap_adversarial=mdap_adversarial,
            mdap_selfplay=mdap_selfplay,
            enabled=enabled,
            environment=environment,
        )

    def _log_config(self, config: LeanAideConfig) -> None:
        """Log configuration summary."""
        logger.info(f"LeanAide Configuration Loaded:")
        logger.info(f"  Enabled: {config.enabled}")
        logger.info(f"  Environment: {config.environment}")
        logger.info(f"  Server: {config.server.get_base_url()}")
        logger.info(f"  Auto-verification: {config.verification.enable_auto}")
        logger.info(f"  Caching: {config.cache.enable}")
        logger.info(f"  Stage 3C integration: {config.workflow.stage_3c_enabled}")
        logger.info(f"  Stage 5 integration: {config.workflow.stage_5_enabled}")


# =============================================================================
# Global Configuration Instance
# =============================================================================

_leanaide_config: Optional[LeanAideConfig] = None
_leanaide_loader: Optional[LeanAideConfigLoader] = None


def load_leanaide_config(
    config_dir: Optional[Path] = None,
    force_reload: bool = False,
    **overrides
) -> LeanAideConfig:
    """
    Load LeanAide configuration from all sources.

    Args:
        config_dir: Directory containing configuration files
        force_reload: Force reload even if already loaded
        **overrides: Python API configuration overrides

    Returns:
        Validated LeanAideConfig object

    Example:
        config = load_leanaide_config(
            server_port=9090,
            verification_complexity_threshold=75
        )
    """
    global _leanaide_config, _leanaide_loader

    if _leanaide_config is None or force_reload:
        _leanaide_loader = LeanAideConfigLoader(config_dir)
        _leanaide_config = _leanaide_loader.load(**overrides)

    return _leanaide_config


def get_leanaide_config() -> LeanAideConfig:
    """
    Get the currently loaded LeanAide configuration.

    Returns:
        LeanAideConfig object (loads if not already loaded)

    Raises:
        ValidationError: If configuration is invalid
    """
    global _leanaide_config

    if _leanaide_config is None:
        return load_leanaide_config()

    return _leanaide_config


def reload_leanaide_config(**overrides) -> LeanAideConfig:
    """
    Force reload LeanAide configuration from all sources.

    Args:
        **overrides: Python API configuration overrides

    Returns:
        Reloaded LeanAideConfig object
    """
    return load_leanaide_config(force_reload=True, **overrides)


def get_leanaide_config_summary() -> Dict[str, Any]:
    """
    Get a summary of the current LeanAide configuration (safe for logging).

    Returns:
        Dictionary with non-sensitive configuration info
    """
    config = get_leanaide_config()

    return {
        "enabled": config.enabled,
        "environment": config.environment,
        "server": {
            "base_url": config.server.get_base_url(),
            "api_version": config.server.api_version,
            "timeout": config.server.timeout,
        },
        "verification": {
            "enable_auto": config.verification.enable_auto,
            "complexity_threshold": config.verification.complexity_threshold,
            "domains": config.verification.domains,
            "strategy": config.verification.verification_strategy,
        },
        "cache": {
            "enabled": config.cache.enable,
            "ttl_seconds": config.cache.ttl,
            "cache_dir": config.cache.cache_dir,
        },
        "workflow": {
            "stage_3c_enabled": config.workflow.stage_3c_enabled,
            "stage_5_enabled": config.workflow.stage_5_enabled,
            "async_verification": config.workflow.async_verification,
            "failure_action": config.workflow.failure_action,
        },
        "lean4": {
            "lean_path": config.lean4.lean_path,
            "project_root": config.lean4.project_root,
            "use_lake": config.lean4.use_lake,
        },
        "performance": {
            "worker_threads": config.performance.worker_threads,
            "optimization_level": config.performance.optimization_level,
        },
    }


# =============================================================================
# Configuration Migration Support
# =============================================================================

class LeanAideConfigMigrator:
    """
    Handle configuration schema migrations between versions.

    This class provides utilities to migrate configuration from older
    versions to newer formats automatically.
    """

    # Current schema version
    CURRENT_VERSION = "1.0.0"

    # Migration paths
    MIGRATIONS = {
        "0.9.0": "migrate_from_090",
        "0.8.0": "migrate_from_080",
    }

    @classmethod
    def migrate(cls, config_data: Dict[str, Any], from_version: str) -> Dict[str, Any]:
        """
        Migrate configuration data to current version.

        Args:
            config_data: Configuration dictionary to migrate
            from_version: Version of the configuration data

        Returns:
            Migrated configuration dictionary

        Raises:
            ValidationError: If migration is not possible
        """
        if from_version == cls.CURRENT_VERSION:
            return config_data

        # Apply migrations in sequence
        version = from_version
        while version != cls.CURRENT_VERSION:
            if version not in cls.MIGRATIONS:
                raise ValidationError(
                    f"Cannot migrate LeanAide config from version {from_version} to {cls.CURRENT_VERSION}. "
                    f"Missing migration path for version {version}"
                )

            migration_method = getattr(cls, cls.MIGRATIONS[version])
            config_data = migration_method(config_data)

            # Update version
            version = cls._get_next_version(version)

        logger.info(f"Migrated LeanAide configuration from {from_version} to {cls.CURRENT_VERSION}")
        return config_data

    @classmethod
    def migrate_from_090(cls, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from version 0.9.0 to 1.0.0."""
        # Add new fields introduced in 1.0.0
        if "server" in config_data:
            if "health_check_interval" not in config_data["server"]:
                config_data["server"]["health_check_interval"] = 60

        if "verification" in config_data:
            if "use_external_prover" not in config_data["verification"]:
                config_data["verification"]["use_external_prover"] = False
            if "prover_timeout_multiplier" not in config_data["verification"]:
                config_data["verification"]["prover_timeout_multiplier"] = 2.0

        if "cache" in config_data:
            if "invalidate_on_proof_change" not in config_data["cache"]:
                config_data["cache"]["invalidate_on_proof_change"] = True
            if "invalidate_on_dependency_update" not in config_data["cache"]:
                config_data["cache"]["invalidate_on_dependency_update"] = True
            if "min_cache_hits_before_persist" not in config_data["cache"]:
                config_data["cache"]["min_cache_hits_before_persist"] = 2

        return config_data

    @classmethod
    def migrate_from_080(cls, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from version 0.8.0 to 0.9.0."""
        # First migrate to 0.9.0
        config_data = cls.migrate_from_090(config_data)
        return config_data

    @classmethod
    def _get_next_version(cls, version: str) -> str:
        """Get next version in migration path."""
        versions = sorted(cls.MIGRATIONS.keys())
        try:
            idx = versions.index(version)
            if idx + 1 < len(versions):
                return versions[idx + 1]
        except ValueError:
            pass
        return cls.CURRENT_VERSION


# =============================================================================
# Configuration Schema Documentation
# =============================================================================

LEANADE_CONFIG_SCHEMA_DOCS = """
LeanAide Configuration Schema Documentation
===========================================

Overview:
The LeanAide configuration system supports multiple configuration sources with
clear precedence rules. All settings have sensible defaults for development use.

Configuration Precedence (highest to lowest):
1. Environment variables with LEANAIDE_ prefix
2. YAML configuration files
3. Python API parameters
4. Default values

Configuration Files:
- Primary: leanaide_config.yaml
- Fallback: config.yaml (leanaide section)

Environment Variables:
All settings can be overridden via environment variables using the pattern:
- LEANAIDE_<SECTION>_<KEY> (e.g., LEANAIDE_SERVER_HOST)
- LEANAIDE_<KEY> for flat settings (e.g., LEANAIDE_ENABLED)

Configuration Sections:

1. Server Configuration (server)
   - host: Server hostname (default: "localhost")
   - port: Server port (default: 8080, range: 1-65535)
   - timeout: Request timeout in seconds (default: 30.0)
   - max_retries: Maximum connection retries (default: 3)
   - retry_delay: Delay between retries in seconds (default: 1.0)
   - use_ssl: Enable SSL/TLS (default: False)
   - verify_ssl: Verify SSL certificates (default: True)
   - api_version: API version (default: "v1")
   - health_check_interval: Health check interval in seconds (default: 60)

2. Verification Configuration (verification)
   - enable_auto: Enable automatic verification (default: True)
   - complexity_threshold: Minimum complexity for verification (default: 50, range: 0-100)
   - domains: Lean 4 domains to verify (default: ["mathlib"])
   - max_proof_depth: Maximum proof depth (default: 100)
   - timeout_per_proof: Timeout per proof in seconds (default: 120.0)
   - parallel_verifications: Parallel verification threads (default: 4)
   - strict_mode: Enable strict verification (default: False)
   - cache_verified_proofs: Cache verified proofs (default: True)
   - verification_strategy: Strategy - "quick", "thorough", or "adaptive" (default: "adaptive")
   - fallback_on_timeout: Fallback on timeout (default: True)
   - trust_level: Minimum trust level (default: 0.95, range: 0.0-1.0)
   - use_external_prover: Use external prover (default: False)
   - prover_timeout_multiplier: External prover timeout multiplier (default: 2.0)

3. Cache Configuration (cache)
   - enable: Enable caching (default: True)
   - ttl: Cache TTL in seconds (default: 86400 = 24 hours)
   - cache_dir: Cache directory (default: "./leanaide_cache")
   - max_cache_size_mb: Maximum cache size in MB (default: 500)
   - cache_proof_objects: Cache proof objects (default: True)
   - cache_dependencies: Cache dependencies (default: True)
   - compression_enabled: Enable compression (default: True)
   - persistent_cache: Persistent cache across restarts (default: True)
   - invalidate_on_proof_change: Invalidate on proof change (default: True)
   - invalidate_on_dependency_update: Invalidate on dependency update (default: True)
   - min_cache_hits_before_persist: Minimum cache hits before persist (default: 2)

4. Workflow Configuration (workflow)
   - stage_3c_enabled: Enable Stage 3C verification (default: True)
   - stage_5_enabled: Enable Stage 5 verification (default: True)
   - stage_3c_priority: Stage 3C priority (default: 7, range: 1-10)
   - stage_5_priority: Stage 5 priority (default: 8, range: 1-10)
   - async_verification: Asynchronous verification (default: True)
   - block_on_verification: Block workflow on verification (default: False)
   - verification_timeout: Verification timeout in seconds (default: 600.0)
   - failure_action: Action on failure - "warn", "error", "continue", or "fallback" (default: "warn")
   - progress_reporting: Enable progress reporting (default: True)
   - inject_proof_hints: Inject verified proof hints (default: True)
   - use_verified_tactics: Use only verified tactics (default: True)
   - verification_results_in_output: Include results in output (default: True)

5. Lean 4 Configuration (lean4)
   - lean_path: Path to Lean 4 executable (default: "lean")
   - lean_pkg_path: Path to leanpkg (default: "leanpkg")
   - mathlib_path: Path to MathLib (default: None)
   - lake_path: Path to Lake (default: "lake")
   - project_root: Project root directory (default: "./lean4_projects")
   - output_dir: Output directory (default: "./lean4_output")
   - import_paths: Additional import paths (default: [])
   - prelude: Custom prelude file (default: None)
   - use_lake: Use Lake builder (default: True)

6. Logging Configuration (logging)
   - level: Log level - DEBUG, INFO, WARNING, ERROR, CRITICAL (default: "INFO")
   - log_file: Log file path (default: None = stdout only)
   - log_format: Log format string (default: standard format)
   - log_verification_details: Log verification details (default: False)
   - log_proof_attempts: Log proof attempts (default: True)
   - log_cache_hits: Log cache hits (default: False)
   - max_log_size_mb: Max log file size in MB (default: 100)
   - backup_count: Number of backup logs (default: 5)

7. Security Configuration (security)
   - enable_sandboxing: Enable sandboxing (default: True)
   - sandbox_timeout: Sandbox timeout in seconds (default: 300.0)
   - max_memory_mb: Maximum memory in MB (default: 2048)
   - allow_network_access: Allow network access (default: False)
   - trusted_domains: Trusted domains (default: ["mathlib", "std"])
   - verify_imports: Verify imports (default: True)
   - enable_resource_limits: Enable resource limits (default: True)
   - max_cpu_time: Maximum CPU time in seconds (default: 600.0)

8. Performance Configuration (performance)
   - worker_threads: Worker threads (default: 4)
   - queue_size: Task queue size (default: 100)
   - batch_size: Batch size (default: 10)
   - enable_profiling: Enable profiling (default: False)
   - profile_dir: Profile directory (default: "./leanaide_profiles")
   - enable_optimization: Enable optimizations (default: True)
   - optimization_level: Optimization level (default: 2, range: 0-3)
   - preload_mathlib: Preload MathLib (default: True)
   - parallel_import_processing: Parallel import processing (default: True)

Global Settings:
- enabled: Enable LeanAide integration (default: True)
- environment: Environment name (default: "development")

Example Environment Variables:
export LEANAIDE_SERVER_HOST="leanaide.example.com"
export LEANAIDE_SERVER_PORT=9090
export LEANAIDE_VERIFICATION_ENABLE_AUTO=true
export LEANAIDE_VERIFICATION_COMPLEXITY_THRESHOLD=75
export LEANAIDE_CACHE_ENABLE=true
export LEANAIDE_WORKFLOW_STAGE_3C_ENABLED=true

Example Usage (Python):
    from leanaide_config import load_leanaide_config, get_leanaide_config

    # Load with custom settings
    config = load_leanaide_config(
        server_port=9090,
        verification_complexity_threshold=75
    )

    # Use configuration
    print(f"Server URL: {config.server.get_base_url()}")
    print(f"Auto-verification: {config.verification.enable_auto}")

Validation:
All configurations are validated automatically. Common validation errors:
- Port out of range (must be 1-65535)
- Invalid verification strategy (must be "quick", "thorough", or "adaptive")
- Invalid failure action (must be "warn", "error", "continue", or "fallback")
- Negative timeouts or thresholds
- Optimization level out of range (must be 0-3)

For more examples, see leanaide_config.example.yaml
"""


if __name__ == "__main__":
    # Print configuration schema documentation
    print(LEANADE_CONFIG_SCHEMA_DOCS)
