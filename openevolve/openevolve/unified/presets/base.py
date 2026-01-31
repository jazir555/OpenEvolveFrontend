"""
Base classes for configuration presets.

Provides the foundation for all preset configurations.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
from pydantic import BaseModel


@dataclass
class PresetInfo:
    """Information about a preset configuration."""
    name: str
    category: str
    description: str
    when_to_use: str
    trade_offs: Dict[str, str]
    related_presets: List[str]
    example_usage: str


@dataclass
class ValidationResult:
    """Result of preset validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)


@dataclass
class PresetComparison:
    """Comparison between two presets."""
    preset1: str
    preset2: str
    differences: Dict[str, tuple[Any, Any]] = field(default_factory=dict)
    similarities: List[str] = field(default_factory=list)


class BasePreset(BaseModel):
    """
    Base class for all preset configurations.

    Presets provide pre-configured settings for common use cases.
    Each preset inherits from this and defines its specific parameters.
    """

    # === Metadata ===
    name: str = Field(
        ...,
        description="Unique name for this preset"
    )
    category: str = Field(
        ...,
        description="Category: performance, domain, use_case, system, problem_type"
    )
    description: str = Field(
        ...,
        description="Human-readable description of when to use this preset"
    )

    # === Core Evolution Parameters (CommonConfig) ===
    max_iterations: int = Field(
        default=100,
        ge=1,
        description="Maximum number of evolution iterations"
    )
    random_seed: Optional[int] = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility"
    )
    checkpoint_interval: int = Field(
        default=50,
        ge=1,
        description="Save checkpoints every N iterations"
    )

    # === Logging ===
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_to_console: bool = Field(
        default=True,
        description="Enable console logging"
    )
    log_to_file: bool = Field(
        default=True,
        description="Enable file logging"
    )

    # === Workspace ===
    workspace_path: str = Field(
        default="./evolve_run_output",
        description="Root directory for outputs"
    )
    task_name: str = Field(
        default="evolution_task",
        description="Task name"
    )

    # === Concurrency ===
    concurrency: int = Field(
        default=5,
        ge=1,
        description="Number of concurrent evaluations"
    )
    timeout: int = Field(
        default=300,
        ge=1,
        description="Default timeout in seconds"
    )

    # === Evolution Mode ===
    evolution_mode: str = Field(
        default="openevolve",
        description="Evolution mode to use"
    )

    # === Database (simplified for presets) ===
    population_size: int = Field(
        default=1000,
        ge=10,
        description="Population size"
    )
    elite_archive_size: int = Field(
        default=100,
        ge=1,
        description="Elite archive size"
    )
    num_islands: int = Field(
        default=5,
        ge=1,
        description="Number of islands"
    )

    # === LLM (simplified for presets) ===
    llm_model: str = Field(
        default="gpt-4o",
        description="Primary LLM model"
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature"
    )

    def get_info(self) -> PresetInfo:
        """Get information about this preset."""
        raise NotImplementedError("Subclasses must implement get_info()")

    def to_unified_config(self) -> Dict[str, Any]:
        """
        Convert preset to unified configuration dictionary.

        Returns a dictionary compatible with UnifiedEvolutionConfig.from_dict()
        """
        return {
            "evolution_mode": self.evolution_mode,
            "enable_modes": [self.evolution_mode],
            "common": {
                "max_iterations": self.max_iterations,
                "random_seed": self.random_seed,
                "checkpoint_interval": self.checkpoint_interval,
                "log_level": self.log_level,
                "log_to_console": self.log_to_console,
                "log_to_file": self.log_to_file,
                "workspace_path": self.workspace_path,
                "task_name": self.task_name,
                "concurrency": self.concurrency,
                "timeout": self.timeout,
            },
            "llm": {
                "models": [
                    {
                        "name": self.llm_model,
                        "weight": 1.0,
                        "temperature": self.llm_temperature,
                    }
                ]
            },
            "database": {
                "population_size": self.population_size,
                "elite_archive_size": self.elite_archive_size,
                "num_islands": self.num_islands,
            }
        }

    def validate(self) -> ValidationResult:
        """
        Validate the preset configuration.

        Returns:
            ValidationResult with validation status and messages
        """
        result = ValidationResult(is_valid=True)

        # Check evolution mode
        valid_modes = ["openevolve", "pes", "qd", "mo", "adversarial", "hybrid"]
        if self.evolution_mode not in valid_modes:
            result.is_valid = False
            result.errors.append(
                f"Invalid evolution_mode '{self.evolution_mode}'. "
                f"Must be one of: {valid_modes}"
            )

        # Check log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            result.warnings.append(
                f"Log level '{self.log_level}' should be one of: {valid_log_levels}"
            )

        # Check concurrency
        if self.concurrency > 20:
            result.warnings.append(
                f"High concurrency ({self.concurrency}) may cause API rate limiting"
            )

        # Check population size vs iterations
        if self.population_size < self.max_iterations * 0.1:
            result.info.append(
                f"Population size ({self.population_size}) is small compared to "
                f"iterations ({self.max_iterations}). Consider increasing."
            )

        return result

    def get_parameter_summary(self) -> Dict[str, Any]:
        """Get a summary of key parameters."""
        return {
            "name": self.name,
            "category": self.category,
            "evolution_mode": self.evolution_mode,
            "max_iterations": self.max_iterations,
            "concurrency": self.concurrency,
            "population_size": self.population_size,
            "llm_model": self.llm_model,
        }


# Import Field after BaseModel to avoid circular dependency
from pydantic import Field
