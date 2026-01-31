"""
System mode configuration presets.

These presets control which evolutionary system(s) to use:
- Pure OpenEvolve
- Pure LoongFlow PES
- Hybrid (auto-selection)
- Custom (user-defined)
"""

from typing import Dict
from .base import BasePreset, PresetInfo, Field


class PureOpenEvolvePreset(BasePreset):
    """
    Use only OpenEvolve (original evolutionary algorithm).

    Optimized for:
    - Code evolution via diffs
    - LLM-driven optimization
    - Standard evolutionary operators

    When to use:
    - You want pure OpenEvolve
    - You don't need planning
    - Standard code evolution
    """

    name: str = "pure_openevolve"
    category: str = "system"
    description: str = "Use only OpenEvolve (no LoongFlow features)"
    evolution_mode: str = "openevolve"

    # Standard OpenEvolve parameters
    max_iterations: int = Field(default=100, description="Standard evolution")
    population_size: int = Field(default=500, description="Medium population")
    concurrency: int = Field(default=5, description="Moderate concurrency")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use=(
                "Standard code evolution, when you don't need planning, "
                "pure OpenEvolve workflow"
            ),
            trade_offs={
                "System": "OpenEvolve only",
                "Planning": "❌ No planning phase",
                "Features": "Standard evolutionary operators",
                "Speed": "⚡ Fast - direct evolution"
            },
            related_presets=["pure_loongflow", "hybrid_auto"],
            example_usage="""
from openevolve.unified.presets import PureOpenEvolvePreset

preset = PureOpenEvolvePreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Evolve with pure OpenEvolve
result = await evolve(code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        config["enable_modes"] = ["openevolve"]
        config["pes"] = None
        config["qd"] = None
        config["mo"] = None
        config["adversarial"] = None
        # OpenEvolve-specific settings
        config["openevolve"] = {
            "diff_based_evolution": True,
            "enable_simplification": True,
            "use_template_stochasticity": True,
        }
        return config


class PureLoongFlowPreset(BasePreset):
    """
    Use only LoongFlow PES (Plan-Evolve-Summarize).

    Optimized for:
    - Structured problem-solving
    - Planning-based evolution
    - Memory and summarization

    When to use:
    - You want PES workflow
    - Planning is beneficial
    - Complex, multi-step problems
    """

    name: str = "pure_loongflow"
    category: str = "system"
    description: str = "Use only LoongFlow PES (Plan-Evolve-Summarize)"
    evolution_mode: str = "pes"

    # PES-optimized parameters
    max_iterations: int = Field(default=100, description="PES iterations")
    population_size: int = Field(default=400, description="PES population")
    concurrency: int = Field(default=5, description="Standard concurrency")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use=(
                "Complex problems, when planning helps, "
                "structured problem-solving workflow"
            ),
            trade_offs={
                "System": "LoongFlow PES only",
                "Planning": "✅ Planning phase enabled",
                "Memory": "✅ Long-term memory",
                "Summarization": "✅ Evolution summarization"
            },
            related_presets=["pure_openevolve", "hybrid_auto"],
            example_usage="""
from openevolve.unified.presets import PureLoongFlowPreset

preset = PureLoongFlowPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Evolve with PES workflow
result = await evolve(code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        config["enable_modes"] = ["pes"]
        config["qd"] = None
        config["mo"] = None
        config["adversarial"] = None
        config["openevolve"] = None
        # PES-specific settings
        config["pes"] = {
            "enable_planning": True,
            "planner_type": "evolve_planner",
            "planning_iterations": 1,
            "use_refinement": True,
            "enable_memory": True,
            "enable_summary": True,
            "summary_type": "evolve_summary",
        }
        return config


class HybridAutoPreset(BasePreset):
    """
    Auto-select best system (hybrid mode).

    Optimized for:
    - Automatic system selection
    - Adaptive evolution
    - Best of both systems

    When to use:
    - You're unsure which system to use
    - You want adaptive selection
    - Maximum flexibility
    """

    name: str = "hybrid_auto"
    category: str = "system"
    description: str = "Auto-select the best evolutionary system"
    evolution_mode: str = "hybrid"

    # Hybrid parameters
    max_iterations: int = Field(default=100, description="Hybrid iterations")
    population_size: int = Field(default=500, description="Hybrid population")
    concurrency: int = Field(default=5, description="Standard concurrency")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use=(
                "When unsure which system to use, want automatic selection, "
                "maximum flexibility and adaptation"
            ),
            trade_offs={
                "System": "Hybrid - auto-selection",
                "Flexibility": "✅✅ Maximum flexibility",
                "Adaptation": "✅ Adaptive to problem",
                "Overhead": "⚠️ Slight overhead from auto-selection"
            },
            related_presets=["pure_openevolve", "pure_loongflow"],
            example_usage="""
from openevolve.unified.presets import HybridAutoPreset

preset = HybridAutoPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Auto-select best system
result = await evolve(code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        # Enable all modes - system will auto-select
        config["enable_modes"] = ["openevolve", "pes", "qd"]
        config["evolution_mode"] = "hybrid"
        # Configure all systems
        config["pes"] = {
            "enable_planning": True,
            "enable_memory": True,
        }
        config["qd"] = {
            "enable_map_elites": True,
        }
        config["openevolve"] = {
            "diff_based_evolution": True,
        }
        return config


class CustomPreset(BasePreset):
    """
    User-defined custom configuration.

    This is a template for creating custom presets.
    Users can override any parameters as needed.

    When to use:
    - You have specific requirements
    - None of the presets fit
    - You want full control
    """

    name: str = "custom"
    category: str = "system"
    description: str = "User-defined custom configuration"

    # All parameters user-configurable
    evolution_mode: str = Field(default="openevolve", description="Choose mode")
    max_iterations: int = Field(default=100, description="Set iterations")
    population_size: int = Field(default=500, description="Set population")
    concurrency: int = Field(default=5, description="Set concurrency")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use=(
                "When you have specific requirements, "
                "none of the presets fit, need full control"
            ),
            trade_offs={
                "Control": "✅✅✅ Full control",
                "Complexity": "⚠️ Higher - you configure everything",
                "Flexibility": "✅✅✅ Maximum flexibility"
            },
            related_presets=["balanced", "pure_openevolve"],
            example_usage="""
from openevolve.unified.presets import CustomPreset

# Create custom preset with your settings
preset = CustomPreset(
    evolution_mode="pes",
    max_iterations=150,
    population_size=600,
    concurrency=8
)

config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Evolve with your custom settings
result = await evolve(code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        # Return base config - users can extend this
        return super().to_unified_config()
