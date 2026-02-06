"""
DataPizza Configuration System

This module provides comprehensive configuration classes for DataPizza integration
at every stage of the CREWAI workflow.

Architecture:
    CREWAIConfig (Top-level)
        ├── DataPizzaPhase1Config (Problem Setup)
        ├── DataPizzaPhase2Config (Solution Generation)
        ├── DataPizzaPhase3Config (Adversarial Critique)
        ├── DataPizzaPhase4Config (Verification)
        ├── DataPizzaPhase5Config (Reassembly)
        └── DataPizzaPhase6Config (Final Validation)

Each config can be used independently or as part of the full workflow.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from enum import Enum


# =============================================================================
# CONFIGURATION ENUMS
# =============================================================================

class DataPizzaProvider(Enum):
    """Supported DataPizza providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class DataPizzaTool(Enum):
    """Available DataPizza tools"""
    FILESYSTEM = "filesystem"
    DUCKDUCKGO = "duckduckgo"
    SQL = "sql"
    WEB_FETCH = "web_fetch"


class DataPizzaAgentRole(Enum):
    """DataPizza agent roles"""
    BLUE = "blue"  # Solution architect
    RED = "red"    # Critiquer
    GOLD = "gold"  # Verifier


class DataPizzaWorkflow(Enum):
    """DataPizza workflow types"""
    PARALLEL = "parallel"  # All agents work in parallel
    BLUE_RED_GOLD = "blue_red_gold"  # Sequential workflow
    AUTO = "auto"  # Automatic selection


# =============================================================================
# CREWAI DATAPizza CONFIG
# =============================================================================

@dataclass
class CrewAIDataPizzaConfig:
    """Configuration for CrewAI-DataPizza integration"""
    provider: str = "openai"
    model: Optional[str] = None
    api_key: Optional[str] = None
    max_steps: int = 20
    planning_interval: int = 3
    enable_planning: bool = True
    tools: Optional[List[str]] = None
    timeout_seconds: int = 300
    max_concurrency: int = 5
    enable_tracing: bool = False
    enable_logging: bool = False
    workflow_type: str = "blue_red_gold"  # parallel, blue_red_gold, auto
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    def validate(self) -> List[str]:
        """Validate configuration, return list of errors (empty if valid)"""
        errors = []
        if self.provider not in ["openai", "anthropic", "google"]:
            errors.append(f"Invalid provider: {self.provider}")
        if self.max_steps < 1 or self.max_steps > 100:
            errors.append(f"max_steps must be between 1 and 100, got {self.max_steps}")
        return errors


# =============================================================================
# BASE CONFIGURATION
# =============================================================================

@dataclass
class DataPizzaBaseConfig:
    """Base configuration for all DataPizza operations"""

    # Provider settings
    provider: str = "openai"
    model: Optional[str] = None
    api_key: Optional[str] = None

    # Execution settings
    max_steps: int = 20
    planning_interval: int = 3
    enable_planning: bool = True

    # Tool settings
    tools: Optional[List[str]] = None  # tools to enable

    # Resource limits
    timeout_seconds: int = 300
    max_concurrency: int = 5

    # Observability
    enable_tracing: bool = False
    enable_logging: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    def validate(self) -> List[str]:
        """Validate configuration, return list of errors (empty if valid)"""
        errors = []

        if self.provider not in ["openai", "anthropic", "google"]:
            errors.append(f"Invalid provider: {self.provider}")

        if self.max_steps < 1 or self.max_steps > 100:
            errors.append(f"max_steps must be between 1 and 100, got {self.max_steps}")

        if self.planning_interval < 1 or self.planning_interval > 20:
            errors.append(f"planning_interval must be between 1 and 20, got {self.planning_interval}")

        valid_tools = ["filesystem", "duckduckgo", "sql", "web_fetch"]
        if self.tools:
            for tool in self.tools:
                if tool not in valid_tools:
                    errors.append(f"Invalid tool: {tool}")

        return errors


# =============================================================================
# PHASE-SPECIFIC CONFIGURATIONS
# =============================================================================

@dataclass
class DataPizzaPhase1Config(DataPizzaBaseConfig):
    """
    Configuration for Phase 1: Problem Setup with DataPizza

    Uses DataPizza's multi-agent analysis capabilities.
    """

    # Analysis settings
    agent_role: str = "blue"  # Use blue agents for analysis
    workflow: str = "parallel"  # Parallel analysis

    # Multi-agent coordination
    enable_multi_agent_analysis: bool = True
    num_parallel_agents: int = 3

    # Analysis depth
    analysis_depth: str = "standard"  # "quick", "standard", "thorough"
    include_research: bool = True
    include_dependency_analysis: bool = True

    # Tool configuration
    enable_web_search: bool = True
    enable_filesystem_access: bool = False
    enable_sql_queries: bool = False

    # Output settings
    include_agent_reasoning: bool = False
    include_tool_usage: bool = True
    include_step_by_step: bool = True


@dataclass
class DataPizzaPhase2Config(DataPizzaBaseConfig):
    """
    Configuration for Phase 2: Solution Generation with DataPizza

    Uses DataPizza's Blue Agents for solution generation.
    """

    # Agent settings
    agent_role: str = "blue"
    team_name: Optional[str] = None

    # Planning settings
    planning_interval: int = 3  # Plan every N steps
    enable_replanning: bool = True

    # Solution generation
    solution_detail_level: str = "detailed"  # "concise", "detailed", "comprehensive"
    include_implementation_code: bool = True
    include_explanations: bool = True
    include_alternatives: bool = False

    # Tool configuration
    enable_filesystem_writes: bool = True
    enable_web_search: bool = True
    enable_sql_execution: bool = False
    enable_web_fetch: bool = True

    # Evolution integration
    enable_evolution: bool = False
    evolution_iterations: int = 50

    # Output settings
    include_tool_results: bool = True
    include_step_trace: bool = True
    include_token_usage: bool = True


@dataclass
class DataPizzaPhase3Config(DataPizzaBaseConfig):
    """
    Configuration for Phase 3: Adversarial Critique with DataPizza

    Uses DataPizza's Red Agents for adversarial critique.
    """

    # Agent settings
    agent_role: str = "red"
    critique_intensity: str = "balanced"  # "mild", "balanced", "strict"

    # Critique settings
    critique_focus: List[str] = field(default_factory=lambda: ["correctness", "security", "performance"])
    enable_deep_critique: bool = False
    max_critique_iterations: int = 1

    # Tool configuration
    enable_web_validation: bool = True  # Use web to validate claims
    enable_code_analysis: bool = True

    # Output settings
    include_improvement_suggestions: bool = True
    include_security_findings: bool = True
    include_performance_issues: bool = True
    include_criticality_scores: bool = False


@dataclass
class DataPizzaPhase4Config(DataPizzaBaseConfig):
    """
    Configuration for Phase 4: Verification with DataPizza

    Uses DataPizza's Gold Agents for verification.
    """

    # Agent settings
    agent_role: str = "gold"
    verification_strictness: str = "balanced"  # "lenient", "balanced", "strict"

    # Verification settings
    verification_criteria: Optional[List[str]] = None
    require_all_criteria: bool = True
    enable_test_generation: bool = False

    # Tool configuration
    enable_code_execution: bool = False
    enable_web_validation: bool = True

    # Output settings
    include_pass_fail_matrix: bool = True
    include_verification_report: bool = True
    include_gap_analysis: bool = True


@dataclass
class DataPizzaPhase5Config(DataPizzaBaseConfig):
    """
    Configuration for Phase 5: Reassembly with DataPizza

    Uses DataPizza's multi-agent coordination for reassembly.
    """

    # Reassembly settings
    workflow: str = "blue_red_gold"  # Sequential workflow
    enable_agent_coordination: bool = True

    # Aggregation strategy
    aggregation_method: str = "consensus"  # "consensus", "voting", "priority"
    conflict_resolution: str = "merge"  # "merge", "priority", "ask"

    # Output settings
    include_integration_plan: bool = True
    include_coordination_log: bool = True


@dataclass
class DataPizzaPhase6Config(DataPizzaBaseConfig):
    """
    Configuration for Phase 6: Final Validation with DataPizza

    Uses DataPizza's multi-agent workflow for final validation.
    """

    # Validation settings
    workflow: str = "blue_red_gold"
    enable_full_team_review: bool = True

    # Quality checks
    run_agent_consensus: bool = True
    require_unanimous_approval: bool = False

    # Output settings
    include_team_votes: bool = True
    include_consensus_report: bool = True
    include_final_recommendations: bool = True


# =============================================================================
# MULTI-AGENT CONFIGURATION
# =============================================================================

@dataclass
class DataPizzaMultiAgentConfig:
    """
    Configuration for DataPizza multi-agent coordination

    Manages Blue/Red/Gold team coordination.
    """

    # Team settings
    blue_team_name: str = "datapizza_blue_team"
    red_team_name: str = "datapizza_red_team"
    gold_team_name: str = "datapizza_gold_team"

    # Workflow settings
    workflow: str = "blue_red_gold"  # "parallel", "blue_red_gold", "auto"
    enable_sequential_execution: bool = True
    enable_parallel_execution: bool = False

    # Coordination settings
    enable_agent_to_agent_communication: bool = True
    enable_shared_context: bool = True
    enable_result_aggregation: bool = True

    # Quality settings
    consensus_threshold: float = 0.7  # 70% agreement required
    enable_voting: bool = True

    # Output settings
    include_per_agent_results: bool = True
    include_coordination_trace: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)


# =============================================================================
# FULL WORKFLOW CONFIGURATION
# =============================================================================

@dataclass
class CREWAIDataPizzaConfig:
    """
    Complete configuration for CREWAI workflow with DataPizza

    This is the top-level configuration that encompasses all phases.
    """

    # Workflow selection
    execution_method: str = "datapizza"
    use_datapizza_workflow: bool = True

    # Phase-specific configurations
    phase1: DataPizzaPhase1Config = field(default_factory=DataPizzaPhase1Config)
    phase2: DataPizzaPhase2Config = field(default_factory=DataPizzaPhase2Config)
    phase3: DataPizzaPhase3Config = field(default_factory=DataPizzaPhase3Config)
    phase4: DataPizzaPhase4Config = field(default_factory=DataPizzaPhase4Config)
    phase5: DataPizzaPhase5Config = field(default_factory=DataPizzaPhase5Config)
    phase6: DataPizzaPhase6Config = field(default_factory=DataPizzaPhase6Config)

    # Multi-agent configuration
    multi_agent: Optional[DataPizzaMultiAgentConfig] = None

    # Global settings
    enable_evolution: bool = True
    evolution_iterations: int = 100

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
        if self.multi_agent:
            config['multi_agent'] = self.multi_agent.to_dict()
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

    def get_phase_config(self, phase: int) -> Union[DataPizzaPhase1Config, DataPizzaPhase2Config,
                                                    DataPizzaPhase3Config, DataPizzaPhase4Config,
                                                    DataPizzaPhase5Config, DataPizzaPhase6Config]:
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

class DataPizzaConfigBuilder:
    """Builder pattern for creating DataPizza configurations"""

    @staticmethod
    def default() -> CREWAIDataPizzaConfig:
        """Create default configuration"""
        return CREWAIDataPizzaConfig()

    @staticmethod
    def multi_agent() -> CREWAIDataPizzaConfig:
        """Create configuration for multi-agent workflow"""
        config = CREWAIDataPizzaConfig()
        config.multi_agent = DataPizzaMultiAgentConfig(
            workflow="blue_red_gold",
            enable_agent_to_agent_communication=True,
            enable_shared_context=True,
        )
        return config

    @staticmethod
    def parallel() -> CREWAIDataPizzaConfig:
        """Create configuration for parallel execution"""
        config = CREWAIDataPizzaConfig()
        config.phase1.workflow = "parallel"
        config.phase5.workflow = "parallel"
        config.multi_agent = DataPizzaMultiAgentConfig(
            workflow="parallel",
            enable_parallel_execution=True,
        )
        return config

    @staticmethod
    def for_phase(phase: int, **kwargs) -> Union[DataPizzaPhase1Config, DataPizzaPhase2Config,
                                                 DataPizzaPhase3Config, DataPizzaPhase4Config,
                                                 DataPizzaPhase5Config, DataPizzaPhase6Config]:
        """Create configuration for specific phase with custom settings"""
        phase_configs = {
            1: DataPizzaPhase1Config,
            2: DataPizzaPhase2Config,
            3: DataPizzaPhase3Config,
            4: DataPizzaPhase4Config,
            5: DataPizzaPhase5Config,
            6: DataPizzaPhase6Config,
        }

        config_class = phase_configs.get(phase)
        if not config_class:
            raise ValueError(f"Invalid phase: {phase}")

        return config_class(**kwargs)

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> CREWAIDataPizzaConfig:
        """Create configuration from dictionary"""
        # Extract phase configs if present
        phase_configs = {}
        for phase_num in range(1, 7):
            phase_key = f'phase{phase_num}'
            if phase_key in config_dict:
                phase_class = {
                    1: DataPizzaPhase1Config,
                    2: DataPizzaPhase2Config,
                    3: DataPizzaPhase3Config,
                    4: DataPizzaPhase4Config,
                    5: DataPizzaPhase5Config,
                    6: DataPizzaPhase6Config,
                }[phase_num]
                phase_configs[phase_key] = phase_class(**config_dict[phase_key])

        # Extract multi-agent config if present
        multi_agent_config = None
        if 'multi_agent' in config_dict and config_dict['multi_agent']:
            multi_agent_config = DataPizzaMultiAgentConfig(**config_dict['multi_agent'])

        # Create main config
        config = CREWAIDataPizzaConfig(**{k: v for k, v in config_dict.items()
                                               if k not in [f'phase{i}' for i in range(1, 7)] and k != 'multi_agent'})

        # Apply phase configs
        for phase_key, phase_config in phase_configs.items():
            setattr(config, phase_key, phase_config)

        if multi_agent_config:
            config.multi_agent = multi_agent_config

        return config


# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

class DataPizzaConfigPresets:
    """Pre-configured settings for common use cases"""

    @staticmethod
    def fast_analysis() -> CREWAIDataPizzaConfig:
        """Fast analysis - fewer steps, less detail"""
        config = CREWAIDataPizzaConfig()

        config.phase1.analysis_depth = "quick"
        config.phase1.num_parallel_agents = 1

        config.phase2.max_steps = 10
        config.phase2.solution_detail_level = "concise"
        config.phase2.include_alternatives = False

        config.phase3.max_critique_iterations = 1
        config.phase3.critique_intensity = "mild"

        config.phase4.verification_strictness = "lenient"

        return config

    @staticmethod
    def comprehensive_analysis() -> CREWAIDataPizzaConfig:
        """Comprehensive analysis - maximum detail, all features"""
        config = CREWAIDataPizzaConfig()

        config.phase1.analysis_depth = "thorough"
        config.phase1.num_parallel_agents = 5
        config.phase1.enable_research = True
        config.phase1.include_dependency_analysis = True

        config.phase2.max_steps = 30
        config.phase2.solution_detail_level = "comprehensive"
        config.phase2.include_alternatives = True
        config.phase2.enable_evolution = True
        config.phase2.evolution_iterations = 100

        config.phase3.critique_intensity = "strict"
        config.phase3.max_critique_iterations = 3
        config.phase3.enable_deep_critique = True

        config.phase4.verification_strictness = "strict"
        config.phase4.enable_test_generation = True

        config.phase5.enable_agent_coordination = True
        config.phase5.include_coordination_log = True

        config.phase6.enable_full_team_review = True
        config.phase6.require_unanimous_approval = True

        config.multi_agent = DataPizzaMultiAgentConfig(
            workflow="blue_red_gold",
            enable_agent_to_agent_communication=True,
            enable_shared_context=True,
            consensus_threshold=0.8,
        )

        return config

    @staticmethod
    def research_focused() -> CREWAIDataPizzaConfig:
        """Research-focused - heavy use of web search and analysis"""
        config = CREWAIDataPizzaConfig()

        config.phase1.analysis_depth = "thorough"
        config.phase1.include_research = True
        config.phase1.enable_web_search = True

        config.phase2.enable_web_search = True
        config.phase2.enable_web_fetch = True

        config.phase3.enable_web_validation = True

        config.phase4.enable_web_validation = True
        config.phase4.enable_code_execution = False

        # Tools configuration
        tools = ["duckduckgo", "web_fetch"]
        config.phase1.tools = tools
        config.phase2.tools = tools
        config.phase3.tools = tools
        config.phase4.tools = tools

        return config

    @staticmethod
    def code_generation() -> CREWAIDataPizzaConfig:
        """Code generation focused - filesystem access, implementation"""
        config = CREWAIDataPizzaConfig()

        config.phase2.solution_detail_level = "comprehensive"
        config.phase2.include_implementation_code = True
        config.phase2.enable_filesystem_writes = True
        config.phase2.enable_evolution = True

        config.phase3.enable_code_analysis = True

        config.phase4.enable_test_generation = True
        config.phase4.enable_code_execution = True

        # Tools configuration
        tools = ["filesystem"]
        config.phase2.tools = tools

        return config

    @staticmethod
    def minimal_resource_usage() -> CREWAIDataPizzaConfig:
        """Minimal resource usage - single agent, fewer steps"""
        config = CREWAIDataPizzaConfig()

        config.phase1.num_parallel_agents = 1
        config.phase1.analysis_depth = "quick"
        config.phase1.enable_web_search = False
        config.phase1.enable_filesystem_access = False

        config.phase2.max_steps = 10
        config.phase2.solution_detail_level = "concise"
        config.phase2.enable_evolution = False
        config.phase2.enable_web_search = False
        config.phase2.enable_filesystem_writes = False

        config.phase3.max_critique_iterations = 1
        config.phase3.critique_intensity = "mild"
        config.phase3.enable_web_validation = False
        config.phase3.enable_code_analysis = False

        config.phase4.verification_strictness = "lenient"
        config.phase4.enable_code_execution = False
        config.phase4.enable_web_validation = False

        config.multi_agent = None

        return config
