"""
Performance-oriented presets for different speed/resource requirements.

These presets optimize for different trade-offs between:
- Speed vs Quality
- Resource usage vs Thoroughness
- Cost vs Completeness
"""

from typing import Dict, List
from .base import BasePreset, PresetInfo, Field


class FastPreset(BasePreset):
    """
    Maximum speed preset for rapid prototyping and quick iterations.

    Optimized for:
    - Fast feedback loops
    - Early development
    - Idea validation
    - Resource-constrained environments

    Trade-offs:
    - Lower quality solutions
    - May miss optimal solutions
    - No validation/gauntlet checks
    """

    name: str = "fast"
    category: str = "performance"
    description: str = "Maximum speed for rapid prototyping and quick iterations"

    # Core parameters - minimized for speed
    max_iterations: int = Field(default=20, description="Quick iterations")
    population_size: int = Field(default=100, description="Small population")
    concurrency: int = Field(default=3, description="Lower concurrency")
    timeout: int = Field(default=120, description="Short timeout")

    # Performance optimizations
    checkpoint_interval: int = Field(default=100, description="Infrequent checkpoints")
    log_level: str = Field(default="WARNING", description="Minimal logging")

    # Skip expensive features
    # These are applied when converting to unified config

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use=(
                "Early development and prototyping, testing ideas quickly, "
                "resource-constrained environments, proof-of-concept work"
            ),
            trade_offs={
                "Speed": "⚡ Very fast - completes in seconds/minutes",
                "Quality": "[WARN] Lower quality - may miss optimal solutions",
                "Validation": "[WARN] No validation - gauntlet skipped",
                "Cost": "[OK] Low cost - minimal API calls"
            },
            related_presets=["balanced", "budget"],
            example_usage="""
from openevolve.unified.presets import FastPreset
from openevolve.unified.config import UnifiedEvolutionConfig

# Create fast preset
preset = FastPreset()

# Convert to unified config
config_dict = preset.to_unified_config()
config = UnifiedEvolutionConfig.from_dict(config_dict)

# Use in evolution
result = await evolve(problem_code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        # Add OpenEvolve-specific optimizations
        config["openevolve"] = {
            "early_stopping_patience": 5,
            "enable_novelty_search": False,
            "enable_quality_diversity": False,
            "use_crossover": False,
        }
        return config


class BalancedPreset(BasePreset):
    """
    Balanced preset - default configuration for most use cases.

    Optimized for:
    - General-purpose evolution
    - Production workflows
    - Balanced speed/quality

    Trade-offs:
    - Moderate speed
    - Good quality solutions
    - Standard validation
    """

    name: str = "balanced"
    category: str = "performance"
    description: str = "Balanced configuration for most use cases"

    # Core parameters - balanced defaults
    max_iterations: int = Field(default=100, description="Standard iterations")
    population_size: int = Field(default=500, description="Medium population")
    concurrency: int = Field(default=5, description="Moderate concurrency")
    timeout: int = Field(default=300, description="Standard timeout")

    # Standard logging and checkpoints
    checkpoint_interval: int = Field(default=50, description="Regular checkpoints")
    log_level: str = Field(default="INFO", description="Standard logging")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use=(
                "General evolution tasks, production workflows, "
                "when unsure which preset to use"
            ),
            trade_offs={
                "Speed": "⚡ Moderate speed - completes in minutes/hours",
                "Quality": "[OK] Good quality - finds solid solutions",
                "Validation": "[OK] Standard validation - basic checks",
                "Cost": "⚖️ Moderate cost - balanced API usage"
            },
            related_presets=["fast", "thorough"],
            example_usage="""
from openevolve.unified.presets import BalancedPreset

# Use balanced preset (recommended default)
preset = BalancedPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

result = await evolve(problem_code, config=config)
"""
        )


class ThoroughPreset(BasePreset):
    """
    Maximum quality preset - thorough search regardless of time.

    Optimized for:
    - Production-critical systems
    - Final optimization passes
    - Research publications
    - Maximum quality assurance

    Trade-offs:
    - Very slow
    - Higher cost
    - Best possible solutions
    """

    name: str = "thorough"
    category: str = "performance"
    description: str = "Maximum quality regardless of time or cost"

    # Core parameters - maximized for quality
    max_iterations: int = Field(default=500, description="Extensive iterations")
    population_size: int = Field(default=2000, description="Large population")
    concurrency: int = Field(default=10, description="High concurrency")
    timeout: int = Field(default=600, description="Long timeout")

    # Comprehensive logging and checkpoints
    checkpoint_interval: int = Field(default=25, description="Frequent checkpoints")
    log_level: str = Field(default="DEBUG", description="Detailed logging")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use=(
                "Production-critical systems, final optimization passes, "
                "research publications, when quality is paramount"
            ),
            trade_offs={
                "Speed": "🐌 Very slow - completes in hours/days",
                "Quality": "[OK] Maximum quality - finds optimal solutions",
                "Validation": "[OK] Comprehensive validation - all checks enabled",
                "Cost": "💰 High cost - extensive API usage"
            },
            related_presets=["balanced", "quality_critical"],
            example_usage="""
from openevolve.unified.presets import ThoroughPreset

# Use thorough preset for final optimization
preset = ThoroughPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

result = await evolve(problem_code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        # Enable all quality features
        config["openevolve"] = {
            "early_stopping_patience": None,  # No early stopping
            "enable_novelty_search": True,
            "enable_quality_diversity": True,
            "use_crossover": True,
            "use_mutation": True,
            "use_embedding": True,
        }
        config["qd"] = {
            "enable_map_elites": True,
            "grid_resolution": 20,
            "archive_elitism": True,
        }
        return config


class BudgetPreset(BasePreset):
    """
    Resource-constrained preset - work within strict resource limits.

    Optimized for:
    - Limited API budgets
    - Free tier accounts
    - Rate-limited environments
    - Cost-sensitive applications

    Trade-offs:
    - Very limited exploration
    - May not find optimal solutions
    - Minimal overhead
    """

    name: str = "budget"
    category: str = "performance"
    description: str = "Work within strict resource limits"

    # Core parameters - minimized for budget
    max_iterations: int = Field(default=10, description="Minimal iterations")
    population_size: int = Field(default=50, description="Tiny population")
    concurrency: int = Field(default=1, description="Sequential execution")
    timeout: int = Field(default=60, description="Short timeout")

    # Minimal overhead
    checkpoint_interval: int = Field(default=1000, description="Rare checkpoints")
    log_level: str = Field(default="ERROR", description="Minimal logging")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use=(
                "Free tier accounts, limited API budgets, "
                "rate-limited environments, cost-sensitive applications"
            ),
            trade_offs={
                "Speed": "⚡ Fast - minimal computation",
                "Quality": "[WARN] Very limited - may not find good solutions",
                "Validation": "[FAIL] None - all validation disabled",
                "Cost": "[OK] Minimal cost - very few API calls"
            },
            related_presets=["fast", "resource_constrained"],
            example_usage="""
from openevolve.unified.presets import BudgetPreset

# Use budget preset for cost-constrained evolution
preset = BudgetPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

result = await evolve(problem_code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        # Disable all expensive features
        config["openevolve"] = {
            "early_stopping_patience": 3,
            "enable_novelty_search": False,
            "enable_quality_diversity": False,
            "use_crossover": False,
            "use_embedding": False,
            "evolution_trace_enabled": False,
        }
        config["database"] = {
            **config["database"],
            "log_prompts": False,
            "enable_artifacts": False,
        }
        return config
