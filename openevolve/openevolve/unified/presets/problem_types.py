"""
Problem-type specific configuration presets.

These presets are optimized for different problem characteristics:
- Single vs multi-objective
- Expensive vs fast evaluation
- Safety-critical vs standard problems
"""

from typing import Dict
from .base import BasePreset, PresetInfo, Field


class SingleObjectivePreset(BasePreset):
    """
    Single objective optimization preset.

    Optimized for:
    - Problems with one clear objective
    - Simple maximization/minimization
    - Score-based optimization

    When to use:
    - You have a single metric to optimize
    - Problems with clear success criteria
    - Standard optimization tasks
    """

    name: str = "single_objective"
    category: str = "problem_type"
    description: str = "Single objective optimization"
    evolution_mode: str = "openevolve"

    # Single-objective parameters
    max_iterations: int = Field(default=100, description="Standard search")
    population_size: int = Field(default=400, description="Good coverage")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use=(
                "Single metric optimization, clear success criteria, "
                "standard maximization/minimization"
            ),
            trade_offs={
                "Objectives": "1 - single objective",
                "Method": "Standard evolution",
                "Output": "Best solution found",
                "Complexity": "✅ Simple - straightforward optimization"
            },
            related_presets=["multi_objective", "balanced"],
            example_usage="""
from openevolve.unified.presets import SingleObjectivePreset

preset = SingleObjectivePreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Optimize single objective (e.g., accuracy, score)
best = await evolve(code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        # Explicitly disable multi-objective
        config["mo"] = None
        config["openevolve"] = {
            "enable_novelty_search": False,
            "enable_quality_diversity": False,
        }
        return config


class MultiObjectivePreset(BasePreset):
    """
    Multi-objective optimization preset.

    Optimized for:
    - Problems with multiple objectives
    - Pareto optimization
    - Trade-off analysis

    When to use:
    - You have multiple competing objectives
    - You need Pareto front analysis
    - Trade-off exploration is important
    """

    name: str = "multi_objective"
    category: str = "problem_type"
    description: str = "Multi-objective Pareto optimization"
    evolution_mode: str = "mo"  # Multi-objective mode

    # Multi-objective parameters
    max_iterations: int = Field(default=150, description="For Pareto convergence")
    population_size: int = Field(default=600, description="For diverse Pareto front")
    concurrency: int = Field(default=6, description="Good throughput")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use=(
                "Multiple objectives, need Pareto analysis, "
                "exploring trade-offs between metrics"
            ),
            trade_offs={
                "Objectives": "2+ - multiple objectives",
                "Method": "NSGA-II / NSGA-III",
                "Output": "Pareto front of solutions",
                "Complexity": "⚠️ Higher - multi-objective optimization"
            },
            related_presets=["single_objective", "finance_portfolio"],
            example_usage="""
from openevolve.unified.presets import MultiObjectivePreset

preset = MultiObjectivePreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Get Pareto-optimal solutions
pareto_front = await evolve(code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        # Multi-objective configuration
        config["mo"] = {
            "objectives": ["objective1", "objective2"],  # User should customize
            "optimization_direction": {
                "objective1": "maximize",
                "objective2": "minimize"
            },
            "use_pareto": True,
            "selection_method": "nsga2",
            "pareto_archive_size": 100,
            "pareto_pruning_method": "crowding_distance",
        }
        return config


class ExpensiveEvaluationPreset(BasePreset):
    """
    Very expensive evaluation preset.

    Optimized for:
    - Expensive fitness functions
    - Time-consuming evaluations
    - Limited evaluation budgets

    When to use:
    - Each evaluation takes minutes/hours
    - You have limited evaluation budget
    - Computational simulations
    """

    name: str = "expensive_evaluation"
    category: str = "problem_type"
    description: str = "Optimization with very expensive evaluations"
    evolution_mode: str = "pes"  # Planning helps reduce evaluations

    # Expensive evaluation parameters
    max_iterations: int = Field(default=20, description="Few iterations")
    population_size: int = Field(default=30, description="Tiny population")
    concurrency: int = Field(default=2, description="Low parallelism")
    timeout: int = Field(default=3600, description="1-hour timeout per evaluation")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use=(
                "Very expensive evaluations, limited budget, "
                "computational simulations, hours per evaluation"
            ),
            trade_offs={
                "Evaluations": "💰💰 Very expensive - minimize count",
                "Method": "Planning-based - intelligent search",
                "Parallelism": "Low - resource constrained",
                "Time": "🐌 Very slow - each evaluation is expensive"
            },
            related_presets=["fast_evaluation", "budget"],
            example_usage="""
from openevolve.unified.presets import ExpensiveEvaluationPreset

preset = ExpensiveEvaluationPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Optimize with minimal evaluations
result = await evolve(expensive_code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        # Planning to reduce wasted evaluations
        config["pes"] = {
            "enable_planning": True,
            "planning_iterations": 2,  # More planning to guide search
            "use_refinement": True,
            "enable_memory": True,  # Remember what works
        }
        config["openevolve"] = {
            "early_stopping_patience": 5,  # Stop early if no improvement
            "enable_novelty_search": False,  # Disable expensive features
            "enable_quality_diversity": False,
        }
        config["database"] = {
            "use_sampling_weight": True,
            "boltzmann_temperature": 0.5,  # More exploitation
            "exploration_rate": 0.1,  # Less exploration
        }
        return config


class FastEvaluationPreset(BasePreset):
    """
    Fast evaluation preset.

    Optimized for:
    - Quick fitness functions
    - Millisecond-scale evaluations
    - Massive exploration

    When to use:
    - Evaluations are very fast
    - You can afford many iterations
    - Want extensive exploration
    """

    name: str = "fast_evaluation"
    category: str = "problem_type"
    description: str = "Optimization with very fast evaluations"
    evolution_mode: str = "qd"  # QD for extensive exploration

    # Fast evaluation parameters
    max_iterations: int = Field(default=500, description="Many iterations")
    population_size: int = Field(default=2000, description="Huge population")
    concurrency: int = Field(default=20, description="High parallelism")
    timeout: int = Field(default=10, description="Short timeout")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use=(
                "Very fast evaluations, can afford many iterations, "
                "want extensive exploration of solution space"
            ),
            trade_offs={
                "Evaluations": "⚡⚡⚡ Very fast - maximize exploration",
                "Method": "QD - explore entire space",
                "Parallelism": "High - leverage speed",
                "Time": "✅ Efficient - fast evaluations"
            },
            related_presets=["expensive_evaluation", "thorough"],
            example_usage="""
from openevolve.unified.presets import FastEvaluationPreset

preset = FastEvaluationPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Extensive exploration with fast evaluations
archive = await evolve(fast_code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        # Quality diversity for extensive exploration
        config["qd"] = {
            "enable_map_elites": True,
            "grid_resolution": 50,  # Very fine grid
            "archive_size_limit": 50000,  # Huge archive
            "archive_elitism": True,
            "use_novelty": True,
        }
        config["database"] = {
            "population_size": 2000,
            "elite_archive_size": 500,
        }
        return config


class SafetyCriticalPreset(BasePreset):
    """
    Safety-critical system optimization preset.

    Optimized for:
    - Safety-critical applications
    - Medical devices
    - Aerospace systems
    - High-stakes optimization

    When to use:
    - Failures are unacceptable
    - Safety is paramount
    - Formal verification needed
    """

    name: str = "safety_critical"
    category: str = "problem_type"
    description: str = "Optimization for safety-critical systems"
    evolution_mode: str = "adversarial"  # Adversarial for robustness

    # Safety-critical parameters
    max_iterations: int = Field(default=250, description="Thorough validation")
    population_size: int = Field(default=1000, description="Large population")
    concurrency: int = Field(default=8, description="Good parallelism")
    timeout: int = Field(default=600, description="Long timeout")

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
                "Safety": "✅✅✅ Maximum - adversarial testing",
                "Validation": "✅✅✅ Comprehensive - all checks",
                "Method": "Adversarial - robust solutions",
                "Cost": "💰💰💰 Very high - extensive validation"
            },
            related_presets=["quality_critical", "production"],
            example_usage="""
from openevolve.unified.presets import SafetyCriticalPreset

preset = SafetyCriticalPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Get safety-verified solutions
verified = await evolve(critical_code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        # Adversarial evolution for robustness
        config["adversarial"] = {
            "enable_adversarial": True,
            "adversarial_mode": "generator_discriminator",
            "num_adversaries": 3,
            "adversarial_rounds": 50,
            "use_coevolution": True,
            "use_arms_race": True,
        }
        config["evaluator"] = {
            "use_llm_feedback": True,
            "llm_feedback_weight": 0.3,
            "cascade_evaluation": True,
        }
        config["openevolve"] = {
            "early_stopping_patience": None,  # Don't stop early
            "enable_simplification": True,  # Simplify for verification
        }
        return config
