"""
Claudiomiro Configuration System

This module provides comprehensive configuration classes for Claudiomiro integration
at every stage of the CREWAI workflow.

Architecture:
    CREWAIConfig (Top-level)
        ├── ClaudiomiroPhase1Config (Problem Setup)
        ├── ClaudiomiroPhase2Config (Solution Generation)
        ├── ClaudiomiroPhase3Config (Adversarial Critique)
        ├── ClaudiomiroPhase4Config (Verification)
        ├── ClaudiomiroPhase5Config (Reassembly)
        └── ClaudiomiroPhase6Config (Final Validation)

Each config can be used independently or as part of the full workflow.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from enum import Enum


# =============================================================================
# CREWAI CLAUDIOMIRO CONFIG
# =============================================================================

@dataclass
class CrewAIClaudiomiroConfig:
    """Configuration for CrewAI-Claudiomiro integration"""
    provider: str = "claude"
    backend: str = "local"
    frontend: str = "cli"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # Execution settings
    max_iterations: int = 10
    timeout_seconds: int = 300
    enable_parallel: bool = True
    max_workers: int = 4
    
    # Feature flags
    enable_analytics: bool = True
    enable_logging: bool = True
    enable_checkpoints: bool = True
    
    # Resource limits
    max_memory_mb: int = 4096
    max_disk_gb: float = 10.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    def validate(self) -> List[str]:
        """Validate configuration, return list of errors (empty if valid)"""
        errors = []
        valid_providers = ["claude", "codex", "gemini", "deep-seek", "glm"]
        if self.provider not in valid_providers:
            errors.append(f"Invalid provider: {self.provider}")
        if self.max_iterations < 1:
            errors.append(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.timeout_seconds < 1:
            errors.append(f"timeout_seconds must be >= 1, got {self.timeout_seconds}")
        return errors


# =============================================================================
# CONFIGURATION ENUMS
# =============================================================================

class ClaudiomiroProvider(Enum):
    """Supported Claudiomiro providers"""
    CLAUDE = "claude"
    CODEX = "codex"
    GEMINI = "gemini"
    DEEPSEEK = "deep-seek"
    GLM = "glm"


class ClaudiomiroBackend(Enum):
    """Backend types for Claudiomiro"""
    LOCAL = "local"
    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"


class ClaudiomiroFrontend(Enum):
    """Frontend types for Claudiomiro"""
    CLI = "cli"
    WEB = "web"
    API = "api"


class ClaudiomiroMode(Enum):
    """Claudiomiro execution modes"""
    AUTONOMOUS = "autonomous"  # Fully autonomous development
    ASSISTED = "assisted"      # User-assisted development
    PLANNING = "planning"      # Planning mode only


# =============================================================================
# BASE CONFIGURATION
# =============================================================================

@dataclass
class ClaudiomiroBaseConfig:
    """Base configuration for all Claudiomiro operations"""

    # Provider settings
    provider: str = "claude"
    backend: Optional[str] = None
    frontend: Optional[str] = None

    # Working directory
    working_dir: str = "."

    # Execution settings
    max_cycles: int = 20
    enable_git_integration: bool = True
    enable_auto_commit: bool = True

    # Resource limits
    timeout_seconds: int = 600
    max_files_per_cycle: int = 100

    # Observability
    enable_tracing: bool = False
    enable_logging: bool = False
    log_level: str = "info"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    def validate(self) -> List[str]:
        """Validate configuration, return list of errors (empty if valid)"""
        errors = []

        valid_providers = ["claude", "codex", "gemini", "deep-seek", "glm"]
        if self.provider not in valid_providers:
            errors.append(f"Invalid provider: {self.provider}")

        if self.max_cycles < 1 or self.max_cycles > 100:
            errors.append(f"max_cycles must be between 1 and 100, got {self.max_cycles}")

        if self.timeout_seconds < 10 or self.timeout_seconds > 3600:
            errors.append(f"timeout_seconds must be between 10 and 3600, got {self.timeout_seconds}")

        valid_log_levels = ["debug", "info", "warning", "error"]
        if self.log_level not in valid_log_levels:
            errors.append(f"Invalid log_level: {self.log_level}")

        return errors


# =============================================================================
# PHASE-SPECIFIC CONFIGURATIONS
# =============================================================================

@dataclass
class ClaudiomiroPhase1Config(ClaudiomiroBaseConfig):
    """
    Configuration for Phase 1: Problem Setup with Claudiomiro

    Note: Claudiomiro is primarily used for Phase 2 (implementation).
    Phase 1 with Claudiomiro uses it for planning and analysis.
    """

    # Planning mode
    enable_planning_mode: bool = True
    planning_depth: str = "standard"  # "quick", "standard", "detailed"

    # Analysis settings
    enable_codebase_analysis: bool = True
    enable_dependency_analysis: bool = True
    max_files_to_analyze: int = 50

    # Output settings
    include_file_tree: bool = True
    include_dependency_graph: bool = False
    include_technical_assessment: bool = True

    # Git integration
    enable_git_history_analysis: bool = False
    git_history_depth: int = 10


@dataclass
class ClaudiomiroPhase2Config(ClaudiomiroBaseConfig):
    """
    Configuration for Phase 2: Solution Generation with Claudiomiro

    Uses Claudiomiro's autonomous development capabilities.
    """

    # Development settings
    development_mode: str = "autonomous"  # "autonomous", "assisted", "planning"
    enable_implementation: bool = True
    enable_testing: bool = True
    enable_documentation: bool = True

    # Code generation settings
    code_style: str = "follow_project"  # "follow_project", "custom", "minimal"
    include_comments: str = "standard"  # "minimal", "standard", "extensive"
    include_type_hints: bool = True
    include_docstrings: bool = True

    # Testing settings
    test_framework: Optional[str] = None  # "pytest", "unittest", "jest", etc.
    generate_test_cases: bool = True
    test_coverage_target: float = 0.8  # 80% coverage

    # Git integration
    commit_message_style: str = "conventional"  # "conventional", "simple", "detailed"
    enable_branch_creation: bool = True
    branch_name_pattern: str = "claudiomiro/{feature}"

    # Quality settings
    enable_linting: bool = True
    enable_formatting: bool = True
    linter: Optional[str] = None  # "eslint", "pylint", "flake8", etc.
    formatter: Optional[str] = None  # "prettier", "black", "autopep8", etc.

    # Multi-repo settings
    backend_dir: Optional[str] = None
    frontend_dir: Optional[str] = None

    # Resource management
    max_files_per_cycle: int = 100
    enable_incremental_builds: bool = True

    # Output settings
    include_build_log: bool = True
    include_test_results: bool = True
    include_commit_info: bool = True


@dataclass
class ClaudiomiroPhase3Config(ClaudiomiroBaseConfig):
    """
    Configuration for Phase 3: Adversarial Critique with Claudiomiro

    Uses Claudiomiro for code review and critique.
    """

    # Critique settings
    critique_mode: str = "comprehensive"  # "quick", "comprehensive", "security"
    critique_intensity: str = "balanced"  # "mild", "balanced", "strict"

    # Analysis depth
    enable_security_analysis: bool = True
    enable_performance_analysis: bool = True
    enable_code_quality_analysis: bool = True
    enable_best_practices_check: bool = True

    # Output settings
    include_line_by_line_feedback: bool = False
    include_refactoring_suggestions: bool = True
    include_security_advisories: bool = True
    include_criticality_scores: bool = False

    # Reporting
    report_format: str = "detailed"  # "summary", "detailed", "extensive"


@dataclass
class ClaudiomiroPhase4Config(ClaudiomiroBaseConfig):
    """
    Configuration for Phase 4: Verification with Claudiomiro

    Uses Claudiomiro for verification and validation.
    """

    # Verification settings
    verification_mode: str = "automated"  # "automated", "manual", "hybrid"
    verification_strictness: str = "balanced"  # "lenient", "balanced", "strict"

    # Testing
    enable_automated_testing: bool = True
    test_framework: Optional[str] = None
    run_integration_tests: bool = False
    run_e2e_tests: bool = False

    # Code quality checks
    enable_linting_verification: bool = True
    enable_type_checking: bool = True
    enable_security_scanning: bool = False

    # Standards compliance
    enable_style_check: bool = True
    enable_documentation_check: bool = True

    # Output settings
    include_test_report: bool = True
    include_coverage_report: bool = False
    include_compliance_report: bool = True


@dataclass
class ClaudiomiroPhase5Config(ClaudiomiroBaseConfig):
    """
    Configuration for Phase 5: Reassembly with Claudiomiro

    Uses Claudiomiro for code integration and reassembly.
    """

    # Integration settings
    integration_mode: str = "automated"  # "automated", "assisted", "manual"
    enable_conflict_resolution: bool = True
    conflict_resolution_strategy: str = "ask"  # "auto", "ask", "priority"

    # Build settings
    enable_build_verification: bool = True
    build_command: Optional[str] = None  # e.g., "npm run build", "mvn package"

    # Output settings
    include_integration_log: bool = True
    include_build_artifacts: bool = False


@dataclass
class ClaudiomiroPhase6Config(ClaudiomiroBaseConfig):
    """
    Configuration for Phase 6: Final Validation with Claudiomiro

    Uses Claudiomiro for final validation and deployment preparation.
    """

    # Validation settings
    enable_pre_deployment_checks: bool = True
    enable_smoke_tests: bool = True
    enable_regression_tests: bool = True

    # Deployment preparation
    enable_deployment_package: bool = False
    deployment_package_format: str = "docker"  # "docker", "zip", "tar"

    # Documentation
    generate_release_notes: bool = False
    generate_changelog: bool = False
    include_upgrade_guide: bool = False

    # Output settings
    include_validation_summary: bool = True
    include_deployment_checklist: bool = True


# =============================================================================
# MULTI-REPO CONFIGURATION
# =============================================================================

@dataclass
class ClaudiomiroMultiRepoConfig:
    """
    Configuration for Claudiomiro multi-repo projects

    Manages backend/frontend repository coordination.
    """

    # Repository structure
    backend_type: str = "python"  # "python", "node", "java", "go", etc.
    frontend_type: str = "react"  # "react", "vue", "angular", "svelte", etc.
    enable_monorepo: bool = False

    # Directory structure
    backend_dir: Optional[str] = None
    frontend_dir: Optional[str] = None
    shared_dir: Optional[str] = None  # For shared types/utilities

    # Coordination settings
    enable_cross_repo_communication: bool = True
    enable_shared_types: bool = True
    api_first_development: bool = True

    # Build settings
    build_order: List[str] = field(default_factory=lambda: ["backend", "frontend"])
    enable_parallel_builds: bool = False

    # Output settings
    include_per_repo_results: bool = True
    include_integration_results: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)


# =============================================================================
# FULL WORKFLOW CONFIGURATION
# =============================================================================

@dataclass
class CREWAIClaudiomiroConfig:
    """
    Complete configuration for CREWAI workflow with Claudiomiro

    This is the top-level configuration that encompasses all phases.
    """

    # Workflow selection
    execution_method: str = "claudiomiro"
    use_claudiomiro_workflow: bool = True

    # Phase-specific configurations
    phase1: ClaudiomiroPhase1Config = field(default_factory=ClaudiomiroPhase1Config)
    phase2: ClaudiomiroPhase2Config = field(default_factory=ClaudiomiroPhase2Config)
    phase3: ClaudiomiroPhase3Config = field(default_factory=ClaudiomiroPhase3Config)
    phase4: ClaudiomiroPhase4Config = field(default_factory=ClaudiomiroPhase4Config)
    phase5: ClaudiomiroPhase5Config = field(default_factory=ClaudiomiroPhase5Config)
    phase6: ClaudiomiroPhase6Config = field(default_factory=ClaudiomiroPhase6Config)

    # Multi-repo configuration
    multi_repo: Optional[ClaudiomiroMultiRepoConfig] = None

    # Global settings
    enable_evolution: bool = False  # Claudiomiro has its own evolution
    evolution_iterations: int = 50

    # Feature flags
    enable_auto_selection: bool = True
    enable_graceful_fallback: bool = True
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
        if self.multi_repo:
            config['multi_repo'] = self.multi_repo.to_dict()
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

    def get_phase_config(self, phase: int) -> Union[ClaudiomiroPhase1Config, ClaudiomiroPhase2Config,
                                                    ClaudiomiroPhase3Config, ClaudiomiroPhase4Config,
                                                    ClaudiomiroPhase5Config, ClaudiomiroPhase6Config]:
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

class ClaudiomiroConfigBuilder:
    """Builder pattern for creating Claudiomiro configurations"""

    @staticmethod
    def default() -> CREWAIClaudiomiroConfig:
        """Create default configuration"""
        return CREWAIClaudiomiroConfig()

    @staticmethod
    def autonomous_development() -> CREWAIClaudiomiroConfig:
        """Create configuration for autonomous development"""
        config = CREWAIClaudiomiroConfig()
        config.phase2.development_mode = "autonomous"
        config.phase2.enable_implementation = True
        config.phase2.enable_testing = True
        config.phase2.enable_auto_commit = True
        return config

    @staticmethod
    def assisted_development() -> CREWAIClaudiomiroConfig:
        """Create configuration for assisted development"""
        config = CREWAIClaudiomiroConfig()
        config.phase2.development_mode = "assisted"
        config.phase2.max_cycles = 10
        return config

    @staticmethod
    def multi_repo() -> CREWAIClaudiomiroConfig:
        """Create configuration for multi-repo project"""
        config = CREWAIClaudiomiroConfig()
        config.multi_repo = ClaudiomiroMultiRepoConfig(
            backend_type="python",
            frontend_type="react",
            backend_dir="backend",
            frontend_dir="frontend",
            enable_cross_repo_communication=True,
        )
        config.phase2.backend_dir = "backend"
        config.phase2.frontend_dir = "frontend"
        return config

    @staticmethod
    def for_phase(phase: int, **kwargs) -> Union[ClaudiomiroPhase1Config, ClaudiomiroPhase2Config,
                                                 ClaudiomiroPhase3Config, ClaudiomiroPhase4Config,
                                                 ClaudiomiroPhase5Config, ClaudiomiroPhase6Config]:
        """Create configuration for specific phase with custom settings"""
        phase_configs = {
            1: ClaudiomiroPhase1Config,
            2: ClaudiomiroPhase2Config,
            3: ClaudiomiroPhase3Config,
            4: ClaudiomiroPhase4Config,
            5: ClaudiomiroPhase5Config,
            6: ClaudiomiroPhase6Config,
        }

        config_class = phase_configs.get(phase)
        if not config_class:
            raise ValueError(f"Invalid phase: {phase}")

        return config_class(**kwargs)

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> CREWAIClaudiomiroConfig:
        """Create configuration from dictionary"""
        # Extract phase configs if present
        phase_configs = {}
        for phase_num in range(1, 7):
            phase_key = f'phase{phase_num}'
            if phase_key in config_dict:
                phase_class = {
                    1: ClaudiomiroPhase1Config,
                    2: ClaudiomiroPhase2Config,
                    3: ClaudiomiroPhase3Config,
                    4: ClaudiomiroPhase4Config,
                    5: ClaudiomiroPhase5Config,
                    6: ClaudiomiroPhase6Config,
                }[phase_num]
                phase_configs[phase_key] = phase_class(**config_dict[phase_key])

        # Extract multi-repo config if present
        multi_repo_config = None
        if 'multi_repo' in config_dict and config_dict['multi_repo']:
            multi_repo_config = ClaudiomiroMultiRepoConfig(**config_dict['multi_repo'])

        # Create main config
        config = CREWAIClaudiomiroConfig(**{k: v for k, v in config_dict.items()
                                                  if k not in [f'phase{i}' for i in range(1, 7)] and k != 'multi_repo'})

        # Apply phase configs
        for phase_key, phase_config in phase_configs.items():
            setattr(config, phase_key, phase_config)

        if multi_repo_config:
            config.multi_repo = multi_repo_config

        return config


# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

class ClaudiomiroConfigPresets:
    """Pre-configured settings for common use cases"""

    @staticmethod
    def rapid_prototyping() -> CREWAIClaudiomiroConfig:
        """Rapid prototyping - quick development, minimal checks"""
        config = CREWAIClaudiomiroConfig()

        config.phase1.planning_depth = "quick"
        config.phase1.max_files_to_analyze = 20

        config.phase2.development_mode = "autonomous"
        config.phase2.max_cycles = 10
        config.phase2.enable_testing = False
        config.phase2.enable_documentation = False
        config.phase2.include_comments = "minimal"

        config.phase3.critique_mode = "quick"
        config.phase3.critique_intensity = "mild"

        config.phase4.verification_strictness = "lenient"
        config.phase4.enable_automated_testing = False

        return config

    @staticmethod
    def production_ready() -> CREWAIClaudiomiroConfig:
        """Production ready - comprehensive checks, testing, documentation"""
        config = CREWAIClaudiomiroConfig()

        config.phase1.planning_depth = "detailed"
        config.phase1.max_files_to_analyze = 100
        config.phase1.include_dependency_graph = True

        config.phase2.development_mode = "autonomous"
        config.phase2.max_cycles = 30
        config.phase2.enable_implementation = True
        config.phase2.enable_testing = True
        config.phase2.enable_documentation = True
        config.phase2.include_comments = "extensive"
        config.phase2.include_type_hints = True
        config.phase2.include_docstrings = True
        config.phase2.test_coverage_target = 0.9
        config.phase2.enable_linting = True
        config.phase2.enable_formatting = True

        config.phase3.critique_mode = "comprehensive"
        config.phase3.critique_intensity = "strict"
        config.phase3.enable_security_analysis = True
        config.phase3.enable_performance_analysis = True

        config.phase4.verification_strictness = "strict"
        config.phase4.enable_automated_testing = True
        config.phase4.run_integration_tests = True
        config.phase4.run_e2e_tests = True
        config.phase4.enable_type_checking = True

        return config

    @staticmethod
    def security_focused() -> CREWAIClaudiomiroConfig:
        """Security focused - emphasis on security analysis and best practices"""
        config = CREWAIClaudiomiroConfig()

        config.phase2.enable_linting = True
        config.phase2.linter = "bandit"  # Python security linter
        config.phase2.enable_security_scanning = True

        config.phase3.critique_mode = "security"
        config.phase3.critique_intensity = "strict"
        config.phase3.enable_security_analysis = True
        config.phase3.include_security_advisories = True

        config.phase4.enable_security_scanning = True
        config.phase4.verification_strictness = "strict"

        return config

    @staticmethod
    def testing_focused() -> CREWAIClaudiomiroConfig:
        """Testing focused - emphasis on test generation and quality"""
        config = CREWAIClaudiomiroConfig()

        config.phase2.enable_testing = True
        config.phase2.test_framework = "pytest"
        config.phase2.generate_test_cases = True
        config.phase2.test_coverage_target = 0.95

        config.phase4.enable_automated_testing = True
        config.phase4.test_framework = "pytest"
        config.phase4.run_integration_tests = True
        config.phase4.run_e2e_tests = True
        config.phase4.include_test_report = True
        config.phase4.include_coverage_report = True

        config.phase6.enable_smoke_tests = True
        config.phase6.enable_regression_tests = True

        return config

    @staticmethod
    def minimal_resource_usage() -> CREWAIClaudiomiroConfig:
        """Minimal resource usage - single cycle, no extra features"""
        config = CREWAIClaudiomiroConfig()

        config.phase1.enable_planning_mode = False
        config.phase1.enable_codebase_analysis = False

        config.phase2.development_mode = "autonomous"
        config.phase2.max_cycles = 5
        config.phase2.enable_testing = False
        config.phase2.enable_documentation = False
        config.phase2.include_comments = "minimal"
        config.phase2.enable_linting = False
        config.phase2.enable_formatting = False
        config.phase2.enable_auto_commit = False

        config.phase3.critique_mode = "quick"
        config.phase3.critique_intensity = "mild"

        config.phase4.verification_strictness = "lenient"
        config.phase4.enable_automated_testing = False
        config.phase4.enable_linting_verification = False

        return config
