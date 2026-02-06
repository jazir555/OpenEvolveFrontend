"""
RESE Configuration System

Centralized configuration management for all RESE components.
Supports environment-specific settings, parameter tuning, and runtime configuration.

Author: Agent Z1 (Integration Specialist)
Created: 2025-12-31
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from enum import Enum


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# =============================================================================
# Phase I Configuration (Epistemic Audit)
# =============================================================================

@dataclass
class Phase1Config:
    """Configuration for Phase I: Epistemic Audit"""

    # Φ₁: Symbolic Constraint Engine
    sce_max_constraints: int = 10000
    sce_enable_caching: bool = True
    sce_conflict_detection: bool = True
    sce_dependency_tracking: bool = True

    # Φ₁.₅: Tacit Assumption Mining
    phi15_enabled: bool = True
    phi15_assumption_threshold: float = 0.6
    phi15_max_assumptions: int = 100
    phi15_use_failure_database: bool = True

    # Φ₂: Cognitive Bias Detection
    phi2_enabled: bool = True
    phi2_bias_threshold: float = 0.5
    phi2_auto_debias: bool = False
    phi2_log_all_detections: bool = True

    # Φ₃: Contradiction Resolution
    phi3_enabled: bool = True
    phi3_resolution_strategy: str = "priority"  # priority, probabilistic, manual
    phi3_max_iterations: int = 100


# =============================================================================
# Phase II Configuration (Isomorphic Resonance)
# =============================================================================

@dataclass
class Phase2Config:
    """Configuration for Phase II: Isomorphic Resonance"""

    # Ψ₁: Constraint Inversion
    psi1_enabled: bool = True
    psi1_complexity_reduction_target: float = 0.1  # 2^n -> 2^(n/10)
    psi1_max_inversion_depth: int = 5

    # Ψ₂: Ontology Mapping
    psi2_enabled: bool = True
    psi2_similarity_threshold: float = 0.7
    psi2_use_embeddings: bool = True
    psi2_embedding_model: str = "text-embedding-ada-002"

    # Ψ₃: Isomorphism Validation (I_mech)
    psi3_enabled: bool = True
    psi3_target_accuracy: float = 0.80
    psi3_use_lean4_proofs: bool = True
    psi3_parallel_isomorphism_check: bool = True

    # I_mech: Mechanistic Isomorphism
    imech_algorithm: str = "weisfeiler_lehman"  # weisfeiler_lehman, vf2
    imech_use_causal_structure: bool = True
    imech_interventional_testing: bool = True


# =============================================================================
# Phase III Configuration (Monte Carlo Refinement)
# =============================================================================

@dataclass
class Phase3Config:
    """Configuration for Phase III: Monte Carlo Refinement"""

    # Γ₁: ACI Analyzer
    gamma1_enabled: bool = True
    gamma1_signal_threshold: float = 0.5
    gamma1_use_entropy_engine: bool = True
    gamma1_use_coherence_engine: bool = True

    # Γ₂: MCTS Search
    gamma2_enabled: bool = True
    gamma2_iterations: int = 1000
    gamma2_playout_depth: int = 100
    gamma2_exploration_constant: float = 1.41
    gamma2_adaptive_c: bool = True
    gamma2_parallel_agents: int = 4
    gamma2_aci_guided: bool = True

    # Γ₃: Statistical Validator
    gamma3_enabled: bool = True
    gamma3_confidence_level: float = 0.95
    gamma3_bootstrap_iterations: int = 1000
    gamma3_significance_level: float = 0.05

    # Convergence Control
    convergence_enabled: bool = True
    convergence_patience: int = 50
    convergence_min_delta: float = 0.001


# =============================================================================
# Phase IV Configuration (Architectural Synthesis)
# =============================================================================

@dataclass
class Phase4Config:
    """Configuration for Phase IV: Architectural Synthesis"""

    # Δ₁: Architecture Assembly
    delta1_enabled: bool = True
    delta1_max_components: int = 50
    delta1_integration_strategy: str = "hierarchical"  # hierarchical, flat, hybrid

    # Δ₂: Predictive Model Generation
    delta2_enabled: bool = True
    delta2_prediction_horizon: int = 10
    delta2_model_type: str = "ensemble"  # ensemble, neural, symbolic

    # Δ₃: ACI Reduction Validator
    delta3_enabled: bool = True
    delta3_validation_threshold: float = 0.7
    delta3_min_aci_reduction: float = 0.2  # 20%
    delta3_holdout_ratio: float = 0.2
    delta3_significance_level: float = 0.05


# =============================================================================
# Pipeline Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for RESE pipeline execution"""

    # Execution mode
    sequential_phases: bool = False  # False = parallel when possible
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    checkpoint_interval: int = 300  # 5 minutes

    # Error handling
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    continue_on_error: bool = False
    rollback_on_failure: bool = True

    # Resource limits
    max_memory_gb: float = 16.0
    max_time_seconds: float = 86400  # 24 hours
    max_parallel_tasks: int = 4

    # Monitoring
    enable_monitoring: bool = True
    metrics_collection_interval: float = 1.0  # seconds
    enable_profiling: bool = False


# =============================================================================
# API Configuration
# =============================================================================

@dataclass
class APIConfig:
    """Configuration for REST API"""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 4

    # Security
    enable_auth: bool = True
    api_key_required: bool = True
    api_key_header: str = "X-API-Key"
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60

    # CORS
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # WebSocket
    enable_websocket: bool = True
    websocket_ping_interval: float = 20.0
    websocket_ping_timeout: float = 20.0


# =============================================================================
# Lean 4 / LeanAide Configuration
# =============================================================================

@dataclass
class LeanAideConfig:
    """
    Configuration for Lean 4 / LeanAide integration.
    
    LeanAide provides autoformalization and proof verification
    capabilities using Lean 4 theorem prover.
    """
    
    # Enable/disable Lean integration
    enabled: bool = True
    
    # Executable paths
    lean_executable: str = "lean"
    lake_executable: str = "lake"
    
    # Verification settings
    auto_verify_proofs: bool = True
    verification_depth: int = 100
    timeout_seconds: float = 120.0
    
    # Mathlib configuration
    mathlib_path: Optional[str] = None
    mathlib_auto_detect: bool = True
    
    # Domain support
    domains: List[str] = field(default_factory=lambda: [
        "mathlib",
        "analysis",
        "algebra",
        "topology",
        "number_theory"
    ])
    
    # Stage integration
    stage_3c_enabled: bool = True  # PES Stage 3C
    stage_5_enabled: bool = True   # PES Stage 5
    
    # Server configuration
    server_host: str = "localhost"
    server_port: int = 7654
    server_auto_start: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    # Performance
    max_concurrent_requests: int = 4
    memory_limit_mb: int = 4096
    
    def __post_init__(self):
        """Auto-detect mathlib path if not specified."""
        if self.mathlib_auto_detect and self.mathlib_path is None:
            possible_paths = [
                Path.cwd() / "lean_workspace" / "mathlib_project",
                Path.cwd() / "mathlib_project",
                Path.home() / ".lean" / "mathlib4",
                Path.home() / "Documents" / "OpenEvolve" / "Frontend" / "lean_workspace" / "mathlib_project",
            ]
            for path in possible_paths:
                if path.exists():
                    self.mathlib_path = str(path)
                    break


# =============================================================================
# PES Enhanced Configuration
# =============================================================================

@dataclass
class PESEnhancedConfig:
    """Configuration for PES Enhanced integration.
    
    Adds cost optimization, early stopping, planning, and summarization
    capabilities to the RESE pipeline.
    """
    
    # Cost Optimization
    enable_cost_optimization: bool = False
    max_cost_usd: float = 10.0
    cost_warning_threshold: float = 0.7
    cost_critical_threshold: float = 0.9
    prompt_token_price: float = 0.00001  # $0.01 per 1K tokens
    completion_token_price: float = 0.00003  # $0.03 per 1K tokens
    
    # Early Stopping
    enable_early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_improvement: float = 0.001
    early_stopping_plateau_threshold: float = 0.001
    
    # PES Phases
    pes_planning_enabled: bool = True
    pes_summarization_enabled: bool = True
    pes_auto_select_strategy: bool = True
    
    # Budget allocation percentages
    planning_budget_pct: float = 0.05
    evolution_budget_pct: float = 0.85
    verification_budget_pct: float = 0.10
    
    # Model selection for cost optimization
    use_cheap_models_for_execution: bool = True
    cheap_model: str = "gpt-3.5-turbo"
    expensive_model: str = "gpt-4o"


# =============================================================================
# Monitoring Configuration
# =============================================================================

@dataclass
class MonitoringConfig:
    """Configuration for monitoring system"""

    # Metrics
    enable_metrics: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_rotation: bool = True
    log_max_bytes: int = 10485760  # 10MB
    log_backup_count: int = 5

    # Tracing
    enable_tracing: bool = False
    trace_sample_rate: float = 0.1

    # Alerts
    enable_alerts: bool = True
    alert_threshold_aci: float = 0.8
    alert_threshold_error_rate: float = 0.05
    alert_threshold_latency_ms: float = 5000


# =============================================================================
# Main RESE Configuration
# =============================================================================

@dataclass
class RESEConfig:
    """Master configuration for entire RESE system"""

    # Environment
    environment: str = "development"
    project_name: str = "rese"
    version: str = "1.0.0"

    # Phase configurations
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    phase3: Phase3Config = field(default_factory=Phase3Config)
    phase4: Phase4Config = field(default_factory=Phase4Config)

    # System configurations
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    pes_enhanced: PESEnhancedConfig = field(default_factory=PESEnhancedConfig)
    lean_aide: LeanAideConfig = field(default_factory=LeanAideConfig)

    # Paths
    base_path: Path = field(default_factory=lambda: Path.cwd())
    data_path: Optional[Path] = None
    cache_path: Optional[Path] = None
    log_path: Optional[Path] = None

    # Feature flags
    feature_use_gpu: bool = False
    feature_distributed: bool = False
    feature_experimental: bool = False

    def __post_init__(self):
        """Initialize derived paths"""
        base = Path(self.base_path)

        if self.data_path is None:
            self.data_path = base / "data"
        if self.cache_path is None:
            self.cache_path = base / "cache"
        if self.log_path is None:
            self.log_path = base / "logs"

        # Create directories
        for path in [self.data_path, self.cache_path, self.log_path]:
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_file(cls, config_path: Path) -> "RESEConfig":
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to configuration JSON file

        Returns:
            RESEConfig instance
        """
        with open(config_path, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RESEConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            RESEConfig instance
        """
        # Extract nested configs
        phase1 = Phase1Config(**data.get('phase1', {}))
        phase2 = Phase2Config(**data.get('phase2', {}))
        phase3 = Phase3Config(**data.get('phase3', {}))
        phase4 = Phase4Config(**data.get('phase4', {}))
        pipeline = PipelineConfig(**data.get('pipeline', {}))
        api = APIConfig(**data.get('api', {}))
        monitoring = MonitoringConfig(**data.get('monitoring', {}))
        pes_enhanced = PESEnhancedConfig(**data.get('pes_enhanced', {}))
        lean_aide = LeanAideConfig(**data.get('lean_aide', {}))

        # Create main config
        config_data = {k: v for k, v in data.items()
                      if k not in ['phase1', 'phase2', 'phase3', 'phase4',
                                   'pipeline', 'api', 'monitoring', 'pes_enhanced', 'lean_aide']}

        return cls(
            phase1=phase1,
            phase2=phase2,
            phase3=phase3,
            phase4=phase4,
            pipeline=pipeline,
            api=api,
            monitoring=monitoring,
            pes_enhanced=pes_enhanced,
            lean_aide=lean_aide,
            **config_data
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            'environment': self.environment,
            'project_name': self.project_name,
            'version': self.version,
            'phase1': asdict(self.phase1),
            'phase2': asdict(self.phase2),
            'phase3': asdict(self.phase3),
            'phase4': asdict(self.phase4),
            'pipeline': asdict(self.pipeline),
            'api': asdict(self.api),
            'monitoring': asdict(self.monitoring),
            'pes_enhanced': asdict(self.pes_enhanced),
            'lean_aide': asdict(self.lean_aide),
            'base_path': str(self.base_path),
            'data_path': str(self.data_path) if self.data_path else None,
            'cache_path': str(self.cache_path) if self.cache_path else None,
            'log_path': str(self.log_path) if self.log_path else None,
            'feature_use_gpu': self.feature_use_gpu,
            'feature_distributed': self.feature_distributed,
            'feature_experimental': self.feature_experimental,
        }

    def save(self, config_path: Optional[Path] = None) -> None:
        """
        Save configuration to JSON file.

        Args:
            config_path: Optional path (default: data_path/config.json)
        """
        if config_path is None:
            config_path = self.data_path / "config.json"

        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "RESEConfig":
        """
        Load configuration from default locations.

        Search order:
        1. Environment variable RESE_CONFIG
        2. ./config.json
        3. ~/.rese/config.json
        4. Default configuration

        Args:
            config_path: Optional explicit path

        Returns:
            RESEConfig instance
        """
        if config_path:
            if config_path.exists():
                return cls.from_file(config_path)

        # Check environment variable
        env_config = os.environ.get('RESE_CONFIG')
        if env_config and Path(env_config).exists():
            return cls.from_file(Path(env_config))

        # Check default locations
        default_locations = [
            Path.cwd() / "config.json",
            Path.home() / ".rese" / "config.json",
        ]

        for location in default_locations:
            if location.exists():
                return cls.from_file(location)

        # Return default configuration
        return cls()

    def for_environment(self, environment: Environment) -> "RESEConfig":
        """
        Create configuration for specific environment.

        Args:
            environment: Target environment

        Returns:
            RESEConfig adjusted for environment
        """
        config = RESEConfig(**self.to_dict())
        config.environment = environment.value

        # Environment-specific adjustments
        if environment == Environment.PRODUCTION:
            config.api.debug = False
            config.api.workers = 8
            config.monitoring.log_level = "WARNING"
            config.pipeline.enable_profiling = False

        elif environment == Environment.DEVELOPMENT:
            config.api.debug = True
            config.api.workers = 1
            config.monitoring.log_level = "DEBUG"
            config.pipeline.enable_profiling = True

        elif environment == Environment.TESTING:
            config.api.debug = True
            config.api.workers = 1
            config.monitoring.log_level = "DEBUG"
            config.pipeline.enable_caching = False

        return config


# =============================================================================
# Configuration Manager
# =============================================================================

class ConfigManager:
    """
    Manages RESE configuration lifecycle.

    Provides singleton access, hot-reloading, and validation.
    """

    _instance: Optional["ConfigManager"] = None
    _config: RESEConfig

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._config = RESEConfig.load()
            self._initialized = True

    @property
    def config(self) -> RESEConfig:
        """Get current configuration"""
        return self._config

    def reload(self) -> None:
        """Reload configuration from file"""
        self._config = RESEConfig.load()

    def update(self, **kwargs) -> None:
        """
        Update configuration parameters.

        Args:
            **kwargs: Parameters to update
        """
        config_dict = self._config.to_dict()
        config_dict.update(kwargs)
        self._config = RESEConfig.from_dict(config_dict)

    def save_current(self) -> None:
        """Save current configuration to file"""
        self._config.save()


# =============================================================================
# Convenience Functions
# =============================================================================

def get_config() -> RESEConfig:
    """
    Get current RESE configuration (singleton).

    Returns:
        RESEConfig instance
    """
    manager = ConfigManager()
    return manager.config


def load_config(config_path: Optional[Path] = None) -> RESEConfig:
    """
    Load RESE configuration from file.

    Args:
        config_path: Optional path to config file

    Returns:
        RESEConfig instance
    """
    return RESEConfig.load(config_path)


def create_default_config(output_path: Optional[Path] = None) -> RESEConfig:
    """
    Create and save default configuration.

    Args:
        output_path: Optional path to save config

    Returns:
        RESEConfig instance
    """
    config = RESEConfig()
    config.save(output_path)
    return config


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main configuration
    'RESEConfig',
    'ConfigManager',
    'get_config',
    'load_config',
    'create_default_config',

    # Phase configurations
    'Phase1Config',
    'Phase2Config',
    'Phase3Config',
    'Phase4Config',

    # System configurations
    'PipelineConfig',
    'APIConfig',
    'MonitoringConfig',
    'PESEnhancedConfig',
    'LeanAideConfig',

    # Enums
    'Environment',
    'LogLevel',
]
