"""
Use-case specific configuration presets.

These presets are organized by common usage scenarios rather than domains.
Each preset is optimized for a specific use case.
"""

from typing import Dict
from .base import BasePreset, PresetInfo, Field


class QuickPrototypePreset(BasePreset):
    """
    Rapid prototyping preset.

    Optimized for:
    - Fast iteration cycles
    - Idea validation
    - Proof of concept development
    - Early stage development

    Trade-offs:
    - Speed over quality
    - Minimal validation
    - Quick feedback
    """

    name: str = "quick_prototype"
    category: str = "use_case"
    description: str = "Rapid prototyping with fast feedback loops"

    # Speed-optimized parameters
    max_iterations: int = Field(default=10, description="Very few iterations")
    population_size: int = Field(default=50, description="Small population")
    concurrency: int = Field(default=2, description="Low concurrency for speed")
    timeout: int = Field(default=60, description="Quick evaluations")
    log_level: str = Field(default="ERROR", description="Minimal logging")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use=(
                "Early development, idea validation, proof of concept, "
                "when you need results in seconds/minutes"
            ),
            trade_offs={
                "Speed": "⚡⚡⚡ Very fast - seconds to minutes",
                "Quality": "⚠️ Low - basic solutions",
                "Validation": "❌ None - skipped",
                "Cost": "✅ Very low - minimal API calls"
            },
            related_presets=["fast", "resource_constrained"],
            example_usage="""
from openevolve.unified.presets import QuickPrototypePreset

preset = QuickPrototypePreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Get quick proof of concept
poc = await evolve(code, config=config)
"""
        )


class ProductionPreset(BasePreset):
    """
    Production deployment preset.

    Optimized for:
    - Production-quality code
    - Comprehensive validation
    - Reliability and robustness
    - Maximum quality assurance

    Trade-offs:
    - Slower but thorough
    - Higher quality
    - Production-ready
    """

    name: str = "production"
    category: str = "use_case"
    description: str = "Production deployment with comprehensive validation"

    # Quality-optimized parameters
    max_iterations: int = Field(default=200, description="Thorough search")
    population_size: int = Field(default=800, description="Large population")
    concurrency: int = Field(default=8, description="High concurrency")
    timeout: int = Field(default=600, description="Long timeout for quality")
    log_level: str = Field(default="INFO", description="Comprehensive logging")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use=(
                "Production deployment, when code quality is critical, "
                "final optimization before release"
            ),
            trade_offs={
                "Speed": "🐌 Slow - hours to days",
                "Quality": "✅✅✅ Maximum quality",
                "Validation": "✅✅✅ Comprehensive - all checks enabled",
                "Cost": "💰💰 High - extensive API usage"
            },
            related_presets=["thorough", "quality_critical"],
            example_usage="""
from openevolve.unified.presets import ProductionPreset

preset = ProductionPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Get production-ready code
production_code = await evolve(code, config=config)
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
            "enable_simplification": True,
        }
        config["evaluator"] = {
            "use_llm_feedback": True,
            "llm_feedback_weight": 0.2,  # High weight for quality
            "cascade_evaluation": True,
            "parallel_evaluations": 8,
        }
        return config


class ResearchPreset(BasePreset):
    """
    Research and exploration preset.

    Optimized for:
    - Exploring solution space
    - Novel algorithm discovery
    - Academic research
    - Publication-quality results

    Trade-offs:
    - Diversity-focused
    - Novelty emphasis
    - Longer exploration
    """

    name: str = "research"
    category: str = "use_case"
    description: str = "Research exploration with novelty emphasis"

    # Diversity-optimized parameters
    max_iterations: int = Field(default=200, description="Extensive exploration")
    population_size: int = Field(default=1000, description="Maximum diversity")
    concurrency: int = Field(default=6, description="Good throughput")
    timeout: int = Field(default=300, description="Standard timeout")
    log_level: str = Field(default="DEBUG", description="Detailed logging for analysis")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use=(
                "Academic research, novel algorithm discovery, "
                "exploring solution space, publication work"
            ),
            trade_offs={
                "Focus": "🔬 Novelty and diversity",
                "Output": "📊 Archive of diverse solutions",
                "Time": "⏰ Long exploration phase",
                "Quality": "✅ High quality with diversity"
            },
            related_presets=["thorough", "science_discovery"],
            example_usage="""
from openevolve.unified.presets import ResearchPreset

preset = ResearchPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Discover novel algorithms
novel_solutions = await evolve(research_code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        # Quality diversity mode
        config["evolution_mode"] = "qd"
        config["qd"] = {
            "enable_map_elites": True,
            "grid_resolution": 30,
            "use_novelty": True,
            "novelty_threshold": 0.3,
            "archive_size_limit": 10000,  # Large archive
            "archive_elitism": True,
        }
        config["openevolve"] = {
            "enable_novelty_search": True,
            "enable_quality_diversity": True,
            "evolution_trace_enabled": True,
            "evolution_trace_include_code": True,
            "evolution_trace_include_prompts": True,
        }
        return config


class ResourceConstrainedPreset(BasePreset):
    """
    Resource-constrained preset.

    Optimized for:
    - Limited compute resources
    - Free tier accounts
    - Rate-limited APIs
    - Budget constraints

    Trade-offs:
    - Minimal resource usage
    - Sequential execution
    - Basic quality
    """

    name: str = "resource_constrained"
    category: str = "use_case"
    description: str = "Work within strict resource limits"

    # Resource-optimized parameters
    max_iterations: int = Field(default=15, description="Minimal iterations")
    population_size: int = Field(default=40, description="Tiny population")
    concurrency: int = Field(default=1, description="Sequential only")
    timeout: int = Field(default=60, description="Short timeout")
    log_level: str = Field(default="WARNING", description="Minimal logging")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use=(
                "Free tier accounts, limited API budget, "
                "rate-limited environments, minimal compute"
            ),
            trade_offs={
                "Resources": "💾 Minimal - sequential execution",
                "Speed": "⚡ Fast - minimal computation",
                "Quality": "⚠️ Basic - limited exploration",
                "Cost": "✅✅ Very low - few API calls"
            },
            related_presets=["budget", "fast"],
            example_usage="""
from openevolve.unified.presets import ResourceConstrainedPreset

preset = ResourceConstrainedPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Evolve with minimal resources
result = await evolve(code, config=config)
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
            "log_prompts": False,
            "enable_artifacts": False,
        }
        return config


class QualityCriticalPreset(BasePreset):
    """
    Quality-critical preset with maximum validation.

    Optimized for:
    - Safety-critical systems
    - High-stakes applications
    - Formal verification
    - Maximum quality assurance

    Trade-offs:
    - Very slow
    - Maximum validation
    - Formal proofs
    """

    name: str = "quality_critical"
    category: str = "use_case"
    description: str = "Maximum quality assurance for critical systems"

    # Quality-maximized parameters
    max_iterations: int = Field(default=300, description="Extensive search")
    population_size: int = Field(default=1500, description="Very large population")
    concurrency: int = Field(default=10, description="Maximum parallelism")
    timeout: int = Field(default=900, description="15-minute timeout")
    log_level: str = Field(default="DEBUG", description="Maximum logging")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use=(
                "Safety-critical systems, medical devices, aerospace, "
                "any application where failures are unacceptable"
            ),
            trade_offs={
                "Speed": "🐌🐌 Very slow - extensive validation",
                "Quality": "✅✅✅ Maximum - formal verification",
                "Validation": "✅✅✅✅ Comprehensive - all checks",
                "Cost": "💰💰💰 Very high - extensive resources"
            },
            related_presets=["thorough", "production", "safety_critical"],
            example_usage="""
from openevolve.unified.presets import QualityCriticalPreset

preset = QualityCriticalPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Get verified, production-critical code
verified_code = await evolve(critical_code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        # Enable all validation and formal verification
        config["evolution_mode"] = "adversarial"
        config["adversarial"] = {
            "enable_adversarial": True,
            "adversarial_mode": "generator_discriminator",
            "num_adversaries": 3,
            "adversarial_rounds": 50,  # Extensive adversarial testing
            "use_coevolution": True,
        }
        config["openevolve"] = {
            "early_stopping_patience": None,
            "enable_novelty_search": True,
            "enable_quality_diversity": True,
            "use_crossover": True,
            "use_mutation": True,
            "use_embedding": True,
            "enable_simplification": True,
        }
        config["evaluator"] = {
            "use_llm_feedback": True,
            "llm_feedback_weight": 0.3,
            "cascade_evaluation": True,
            "parallel_evaluations": 10,
        }
        return config
