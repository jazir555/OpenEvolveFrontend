"""
ROMA Configuration System

This module provides comprehensive configuration classes for ROMA integration
at every stage of the CREWAI workflow.

Architecture:
    CREWAIConfig (Top-level)
        ├── Phase1Config (Problem Setup)
        ├── Phase2Config (Solution Generation)
        ├── Phase3Config (Adversarial Critique)
        ├── Phase4Config (Verification)
        ├── Phase5Config (Reassembly)
        └── Phase6Config (Final Validation)

Each config can be used independently or as part of the full workflow.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from enum import Enum


# =============================================================================
# CONFIGURATION ENUMS
# =============================================================================

class ROMAExecutionMode(Enum):
    """ROMA execution modes"""
    RECURSIVE = "recursive"  # Depth-first recursive execution
    EVENT_DRIVEN = "event_driven"  # Parallel DAG-based execution


class ROMAProvider(Enum):
    """Supported ROMA providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"


class ROMAAnalysisType(Enum):
    """ROMA analysis types"""
    DECOMPOSITION = "decomposition"
    COMPLEXITY = "complexity"
    DEPENDENCIES = "dependencies"


class ROMACritiqueFocus(Enum):
    """ROMA critique focus areas"""
    COMPREHENSIVE = "comprehensive"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CORRECTNESS = "correctness"


# =============================================================================
# BASE CONFIGURATION
# =============================================================================

@dataclass
class ROMABaseConfig:
    """Base configuration for all ROMA operations"""

    # Provider settings
    provider: Optional[str] = None  # None = use ROMA default
    model: Optional[str] = None
    api_key: Optional[str] = None

    # Execution settings
    execution_mode: str = "recursive"  # "recursive" or "event_driven"
    max_depth: int = 2

    # Feature flags
    enable_checkpoints: bool = False
    enable_logging: bool = False
    enable_observability: bool = False  # MLflow tracking

    # Resource limits
    max_concurrency: int = 10
    timeout_seconds: int = 300

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    def validate(self) -> List[str]:
        """Validate configuration, return list of errors (empty if valid)"""
        errors = []

        if self.execution_mode not in ["recursive", "event_driven"]:
            errors.append(f"Invalid execution_mode: {self.execution_mode}")

        if self.max_depth < 1 or self.max_depth > 10:
            errors.append(f"max_depth must be between 1 and 10, got {self.max_depth}")

        if self.max_concurrency < 1 or self.max_concurrency > 100:
            errors.append(f"max_concurrency must be between 1 and 100, got {self.max_concurrency}")

        return errors


# =============================================================================
# PHASE-SPECIFIC CONFIGURATIONS
# =============================================================================

@dataclass
class ROMAPhase1Config(ROMABaseConfig):
    """
    Configuration for Phase 1: Problem Setup with ROMA

    Uses ROMA's analysis capabilities to understand and decompose the problem.
    """

    # Analysis settings
    analysis_type: str = "decomposition"  # "decomposition", "complexity", "dependencies"
    max_depth_analysis: int = 3  # Higher depth for analysis phase

    # Decomposition settings
    decomposition_strategy: str = "automatic"  # ROMA chooses strategy
    enable_hierarchical_breakdown: bool = True
    identify_dependencies: bool = True
    calculate_complexity_scores: bool = True

    # Output settings
    include_dag_visualization: bool = False
    include_token_usage: bool = True
    include_execution_trace: bool = False

    # Integration with Decomposition Workflow
    fallback_to_traditional: bool = True  # Fall back to traditional if ROMA fails
    merge_with_traditional_analysis: bool = False  # Combine ROMA + traditional


@dataclass
class ROMAPhase2Config(ROMABaseConfig):
    """
    Configuration for Phase 2: Solution Generation with ROMA

    Uses ROMA's recursive solving to generate solutions for sub-problems.
    """

    # Solving settings
    max_depth_solving: int = 2  # Depth for solution generation
    enable_recursive_solving: bool = True
    enable_parallel_execution: bool = False  # Use event_driven mode for parallel

    # Team integration
    team_name: Optional[str] = None  # Blue Team name
    enable_blue_team_coordination: bool = False  # Coordinate with Blue Team agents

    # Solution generation
    solution_format: str = "detailed"  # "concise", "detailed", "comprehensive"
    include_implementation_details: bool = True
    include_test_cases: bool = False
    include_alternatives: bool = False

    # Evolution integration
    enable_evolution: bool = False  # Use OpenEvolve with ROMA
    evolution_iterations: int = 50

    # Output settings
    include_dag_info: bool = True
    include_token_usage: bool = True
    include_stage_results: bool = True


@dataclass
class ROMAPhase3Config(ROMABaseConfig):
    """
    Configuration for Phase 3: Adversarial Critique with ROMA

    Uses ROMA's critique capabilities for Red Team adversarial review.
    """

    # Critique settings
    critique_focus: str = "comprehensive"  # "comprehensive", "security", "performance", "correctness"
    max_depth_critique: int = 1  # Shallow depth for critique

    # Red Team integration
    enable_red_team_coordination: bool = False  # Coordinate with Red Team agents
    red_team_gauntlet: bool = False  # Use Red Team gauntlet with ROMA

    # Critique intensity
    critique_intensity: str = "balanced"  # "mild", "balanced", "strict"
    max_critique_iterations: int = 1

    # Output settings
    include_improvement_suggestions: bool = True
    include_security_analysis: bool = True
    include_performance_analysis: bool = True


@dataclass
class ROMAPhase4Config(ROMABaseConfig):
    """
    Configuration for Phase 4: Verification with ROMA

    Uses ROMA's verification capabilities for Gold Team quality assurance.
    """

    # Verification settings
    max_depth_verification: int = 1  # Shallow depth for verification
    verification_criteria: Optional[List[str]] = None  # Custom criteria

    # Gold Team integration
    enable_gold_team_coordination: bool = False  # Coordinate with Gold Team agents
    gold_team_gauntlet: bool = False  # Use Gold Team gauntlet with ROMA

    # Verification strictness
    verification_strictness: str = "balanced"  # "lenient", "balanced", "strict"
    require_all_criteria: bool = True

    # Output settings
    include_pass_fail_matrix: bool = True
    include_verification_report: bool = True
    include_gap_analysis: bool = True


@dataclass
class ROMAPhase5Config(ROMABaseConfig):
    """
    Configuration for Phase 5: Reassembly with ROMA

    Uses ROMA's aggregation capabilities to combine solutions.
    """

    # Aggregation settings
    aggregation_method: str = "automatic"  # "automatic", "manual", "hybrid"
    enable_roma_aggregation: bool = True  # Use ROMA's automatic aggregation

    # Reassembly strategy
    reassembly_strategy: str = "hierarchical"  # "hierarchical", "sequential", "parallel"
    conflict_resolution: str = "merge"  # "merge", "prioritize", "ask"

    # Output settings
    include_integration_plan: bool = True
    include_dependencies: bool = True
    include_deployment_guide: bool = False


@dataclass
class ROMAPhase6Config(ROMABaseConfig):
    """
    Configuration for Phase 6: Final Validation with ROMA

    Uses ROMA for final validation and knowledge extraction.
    """

    # Validation settings
    validation_depth: int = 1
    enable_roma_validation: bool = True

    # Knowledge extraction
    extract_knowledge: bool = True
    knowledge_format: str = "structured"  # "structured", "narrative", "both"

    # Final quality checks
    run_comprehensive_tests: bool = False
    generate_validation_report: bool = True

    # Output settings
    include_metrics: bool = True
    include_recommendations: bool = True


# =============================================================================
# HYBRID CONFIGURATION
# =============================================================================

@dataclass
class ROMAHybridConfig:
    """
    Configuration for ROMA-Decomposition Hybrid mode

    Combines ROMA's automatic decomposition with Decomposition Workflow's
    team-based quality assurance.
    """

    # ROMA settings
    roma_max_depth_analysis: int = 3
    roma_max_depth_solving: int = 2
    roma_execution_mode: str = "recursive"
    roma_provider: Optional[str] = None
    roma_model: Optional[str] = None
    roma_api_key: Optional[str] = None

    # Decomposition Workflow settings
    enable_gauntlets: bool = True
    enable_evolution: bool = True
    evolution_iterations: int = 50

    # Team settings
    blue_team_name: str = "roma_blue_team"
    red_team_name: str = "roma_red_team"
    gold_team_name: str = "roma_gold_team"

    # Hybrid orchestration
    auto_aggregate: bool = True  # Use ROMA's aggregation
    parallel_stages: bool = False  # Run critique/verify in parallel

    # Quality settings
    critique_intensity: str = "balanced"
    verification_strictness: str = "balanced"

    # Output settings
    include_stage_breakdown: bool = True
    include_dag_info: bool = True
    include_token_usage: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)


# =============================================================================
# FULL WORKFLOW CONFIGURATION
# =============================================================================

@dataclass
class CREWAIROMAConfig:
    """
    Complete configuration for CREWAI workflow with ROMA

    This is the top-level configuration that encompasses all phases.
    """

    # Workflow selection
    execution_method: str = "auto"  # "traditional", "claudiomiro", "datapizza", "roma", "hybrid", "auto"
    use_roma_native_workflow: bool = False  # Use ROMA's native workflow instead of Decomposition

    # Phase-specific configurations
    phase1: ROMAPhase1Config = field(default_factory=ROMAPhase1Config)
    phase2: ROMAPhase2Config = field(default_factory=ROMAPhase2Config)
    phase3: ROMAPhase3Config = field(default_factory=ROMAPhase3Config)
    phase4: ROMAPhase4Config = field(default_factory=ROMAPhase4Config)
    phase5: ROMAPhase5Config = field(default_factory=ROMAPhase5Config)
    phase6: ROMAPhase6Config = field(default_factory=ROMAPhase6Config)

    # Hybrid configuration (if using hybrid mode)
    hybrid: Optional[ROMAHybridConfig] = None

    # Global settings
    enable_evolution: bool = True
    evolution_iterations: int = 100

    # Claudiomiro settings (if using claudiomiro)
    use_claudiomiro: bool = False
    claudiomiro_provider: str = "claude"
    claudiomiro_backend: Optional[str] = None
    claudiomiro_frontend: Optional[str] = None
    working_dir: str = "."
    max_cycles: int = 20

    # DataPizza settings (if using datapizza)
    use_datapizza: bool = False
    datapizza_provider: str = "openai"
    datapizza_api_key: Optional[str] = None
    datapizza_model: Optional[str] = None
    datapizza_tools: Optional[List[str]] = None
    datapizza_planning_interval: int = 3
    datapizza_max_steps: int = 20

    # Feature flags
    enable_auto_selection: bool = True  # Auto-select best method
    enable_graceful_fallback: bool = True  # Fall back if method unavailable
    enable_detailed_logging: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        config = asdict(self)
        # Convert dataclass fields to dicts
        config['phase1'] = self.phase1.to_dict()
        config['phase2'] = self.phase2.to_dict()
        config['phase3'] = self.phase3.to_dict()
        config['phase4'] = self.phase4.to_dict()
        config['phase5'] = self.phase5.to_dict()
        config['phase6'] = self.phase6.to_dict()
        if self.hybrid:
            config['hybrid'] = self.hybrid.to_dict()
        return config

    def validate(self) -> Dict[str, List[str]]:
        """Validate all configurations, return dict of phase->errors"""
        errors = {}

        errors['phase1'] = self.phase1.validate()
        errors['phase2'] = self.phase2.validate()
        errors['phase3'] = self.phase3.validate()
        errors['phase4'] = self.phase4.validate()
        errors['phase5'] = self.phase5.validate()
        errors['phase6'] = self.phase6.validate()

        return errors

    def get_phase_config(self, phase: int) -> Union[ROMAPhase1Config, ROMAPhase2Config,
                                                    ROMAPhase3Config, ROMAPhase4Config,
                                                    ROMAPhase5Config, ROMAPhase6Config]:
        """Get configuration for specific phase"""
        phases = {
            1: self.phase1,
            2: self.phase2,
            3: self.phase3,
            4: self.phase4,
            5: self.phase5,
            6: self.phase6,
        }
        return phases.get(phase)


# =============================================================================
# CONFIGURATION BUILDERS
# =============================================================================

class ROMAConfigBuilder:
    """Builder pattern for creating ROMA configurations"""

    @staticmethod
    def default() -> CREWAIROMAConfig:
        """Create default configuration"""
        return CREWAIROMAConfig()

    @staticmethod
    def roma_native() -> CREWAIROMAConfig:
        """Create configuration for ROMA native workflow"""
        config = CREWAIROMAConfig()
        config.execution_method = "roma"
        config.use_roma_native_workflow = True
        return config

    @staticmethod
    def roma_hybrid() -> CREWAIROMAConfig:
        """Create configuration for ROMA-Decomposition hybrid"""
        config = CREWAIROMAConfig()
        config.execution_method = "hybrid"
        config.hybrid = ROMAHybridConfig()
        return config

    @staticmethod
    def for_phase(phase: int, **kwargs) -> Union[ROMAPhase1Config, ROMAPhase2Config,
                                                 ROMAPhase3Config, ROMAPhase4Config,
                                                 ROMAPhase5Config, ROMAPhase6Config]:
        """Create configuration for specific phase with custom settings"""
        phase_configs = {
            1: ROMAPhase1Config,
            2: ROMAPhase2Config,
            3: ROMAPhase3Config,
            4: ROMAPhase4Config,
            5: ROMAPhase5Config,
            6: ROMAPhase6Config,
        }

        config_class = phase_configs.get(phase)
        if not config_class:
            raise ValueError(f"Invalid phase: {phase}")

        return config_class(**kwargs)

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> CREWAIROMAConfig:
        """Create configuration from dictionary"""
        # Extract phase configs if present
        phase_configs = {}
        for phase_num in range(1, 7):
            phase_key = f'phase{phase_num}'
            if phase_key in config_dict:
                phase_class = {
                    1: ROMAPhase1Config,
                    2: ROMAPhase2Config,
                    3: ROMAPhase3Config,
                    4: ROMAPhase4Config,
                    5: ROMAPhase5Config,
                    6: ROMAPhase6Config,
                }[phase_num]
                phase_configs[phase_key] = phase_class(**config_dict[phase_key])

        # Extract hybrid config if present
        hybrid_config = None
        if 'hybrid' in config_dict and config_dict['hybrid']:
            hybrid_config = ROMAHybridConfig(**config_dict['hybrid'])

        # Create main config
        config = CREWAIROMAConfig(**{k: v for k, v in config_dict.items()
                                         if k not in [f'phase{i}' for i in range(1, 7)] and k != 'hybrid'})

        # Apply phase configs
        for phase_key, phase_config in phase_configs.items():
            setattr(config, phase_key, phase_config)

        if hybrid_config:
            config.hybrid = hybrid_config

        return config


# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

class ROMAConfigPresets:
    """Pre-configured settings for common use cases"""

    @staticmethod
    def fast_development() -> CREWAIROMAConfig:
        """Fast development - lower depth, fewer features"""
        config = CREWAIROMAConfig()

        config.phase1.max_depth_analysis = 2
        config.phase2.max_depth_solving = 1
        config.phase2.enable_evolution = False
        config.phase2.include_test_cases = False
        config.phase3.red_team_gauntlet = False
        config.phase4.gold_team_gauntlet = False

        return config

    @staticmethod
    def comprehensive_analysis() -> CREWAIROMAConfig:
        """Comprehensive analysis - maximum depth, all features"""
        config = CREWAIROMAConfig()

        config.phase1.max_depth_analysis = 5
        config.phase1.include_dag_visualization = True
        config.phase1.include_execution_trace = True

        config.phase2.max_depth_solving = 3
        config.phase2.enable_parallel_execution = True
        config.phase2.solution_format = "comprehensive"
        config.phase2.include_test_cases = True
        config.phase2.include_alternatives = True
        config.phase2.enable_evolution = True
        config.phase2.evolution_iterations = 100

        config.phase3.critique_intensity = "strict"
        config.phase3.max_critique_iterations = 3
        config.phase3.red_team_gauntlet = True

        config.phase4.verification_strictness = "strict"
        config.phase4.gold_team_gauntlet = True

        config.phase6.run_comprehensive_tests = True

        return config

    @staticmethod
    def security_focused() -> CREWAIROMAConfig:
        """Security-focused analysis"""
        config = CREWAIROMAConfig()

        config.phase1.analysis_type = "dependencies"
        config.phase1.identify_dependencies = True

        config.phase3.critique_focus = "security"
        config.phase3.critique_intensity = "strict"
        config.phase3.include_security_analysis = True
        config.phase3.red_team_gauntlet = True

        config.phase4.verification_strictness = "strict"
        config.phase4.verification_criteria = [
            "security_best_practices",
            "input_validation",
            "output_encoding",
            "authentication",
            "authorization",
        ]

        return config

    @staticmethod
    def performance_focused() -> CREWAIROMAConfig:
        """Performance-focused analysis"""
        config = CREWAIROMAConfig()

        config.phase1.analysis_type = "complexity"
        config.phase1.calculate_complexity_scores = True

        config.phase2.enable_parallel_execution = True
        config.phase2.max_concurrency = 20

        config.phase3.critique_focus = "performance"
        config.phase3.include_performance_analysis = True

        config.phase4.verification_criteria = [
            "response_time",
            "throughput",
            "scalability",
            "resource_usage",
        ]

        return config

    @staticmethod
    def minimal_resource_usage() -> CREWAIROMAConfig:
        """Minimal resource usage - lowest depth, essential features only"""
        config = CREWAIROMAConfig()

        config.phase1.max_depth_analysis = 1
        config.phase1.include_dag_visualization = False
        config.phase1.include_execution_trace = False

        config.phase2.max_depth_solving = 1
        config.phase2.enable_parallel_execution = False
        config.phase2.solution_format = "concise"
        config.phase2.include_test_cases = False
        config.phase2.include_alternatives = False
        config.phase2.enable_evolution = False

        config.phase3.max_critique_iterations = 1
        config.phase3.red_team_gauntlet = False

        config.phase4.gold_team_gauntlet = False

        config.phase6.run_comprehensive_tests = False

        return config
