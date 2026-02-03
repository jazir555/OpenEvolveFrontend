"""
Domain-specific configuration presets.

Each domain has 3 specialized presets:
1. General: General-purpose optimization for the domain
2. Specialized A: Focused on a specific sub-domain or use case
3. Specialized B: Focused on another sub-domain or use case

Total: 18 domain presets (6 domains × 3 presets each)
"""

from typing import Dict, List
from .base import BasePreset, PresetInfo, Field


# ============================================================================
# FINANCE DOMAIN (3 presets)
# ============================================================================

class FinanceGeneralPreset(BasePreset):
    """
    General finance optimization preset.

    Optimized for:
    - Portfolio optimization
    - Risk management
    - Trading strategy development
    - Financial modeling

    Domain specifics:
    - PES mode for planning-based evolution
    - Strict evaluation limits
    - Risk-aware optimization
    """

    name: str = "finance_general"
    category: str = "domain"
    description: str = "General finance optimization tasks"
    evolution_mode: str = "pes"  # Use planning

    # Finance-optimized parameters
    max_iterations: int = Field(default=50, description="Moderate iterations")
    population_size: int = Field(default=200, description="Medium population")
    concurrency: int = Field(default=3, description="Conservative concurrency")
    timeout: int = Field(default=300, description="5-minute evaluation limit")

    # Risk-aware settings
    random_seed: int = Field(default=42, description="For reproducibility")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use=(
                "General finance optimization, portfolio management, "
                "trading strategies, risk analysis"
            ),
            trade_offs={
                "Mode": "PES (planning-based) for structured evolution",
                "Risk": "Risk-aware with reproducible results",
                "Speed": "Moderate speed with quality focus"
            },
            related_presets=["finance_portfolio", "finance_risk"],
            example_usage="""
from openevolve.unified.presets import FinanceGeneralPreset

preset = FinanceGeneralPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Optimize portfolio allocation
result = await evolve(portfolio_code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        config["pes"] = {
            "enable_planning": True,
            "enable_memory": True,
            "enable_code_execution": True,
            "sandbox_mode": True,  # Critical for financial code
        }
        config["evaluator"] = {
            "timeout": self.timeout,
            "use_llm_feedback": True,
            "llm_feedback_weight": 0.15,  # Higher weight for quality
        }
        return config


class FinancePortfolioPreset(BasePreset):
    """
    Portfolio optimization preset (multi-objective).

    Optimized for:
    - Multi-objective optimization (return vs risk)
    - Pareto front analysis
    - Asset allocation
    """

    name: str = "finance_portfolio"
    category: str = "domain"
    description: str = "Multi-objective portfolio optimization"
    evolution_mode: str = "mo"  # Multi-objective

    # Multi-objective parameters
    max_iterations: int = Field(default=100, description="For Pareto convergence")
    population_size: int = Field(default=300, description="Larger for diversity")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use="Portfolio optimization with multiple objectives",
            trade_offs={
                "Objectives": "Return maximization, risk minimization",
                "Output": "Pareto front of optimal portfolios",
                "Method": "NSGA-II for multi-objective optimization"
            },
            related_presets=["finance_general", "finance_risk"],
            example_usage="""
preset = FinancePortfolioPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Get Pareto-optimal portfolios
pareto_front = await evolve(portfolio_code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        config["mo"] = {
            "objectives": ["return", "risk", "sharpe_ratio"],
            "optimization_direction": {
                "return": "maximize",
                "risk": "minimize",
                "sharpe_ratio": "maximize"
            },
            "use_pareto": True,
            "selection_method": "nsga2",
            "pareto_archive_size": 100,
        }
        return config


class FinanceRiskPreset(BasePreset):
    """
    Risk analysis and VaR optimization preset.

    Optimized for:
    - Value at Risk (VaR) optimization
    - Conditional VaR (CVaR)
    - Stress testing
    - Risk management
    """

    name: str = "finance_risk"
    category: str = "domain"
    description: str = "Risk analysis and VaR optimization"
    evolution_mode: str = "qd"  # Quality diversity for risk scenarios

    # QD parameters for diverse risk scenarios
    max_iterations: int = Field(default=75, description="For scenario coverage")
    population_size: int = Field(default=400, description="For diversity")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use="Risk analysis, VaR optimization, stress testing",
            trade_offs={
                "Focus": "Risk metrics (VaR, CVaR)",
                "Diversity": "QD for diverse scenarios",
                "Output": "Archive of risk-optimized strategies"
            },
            related_presets=["finance_general", "finance_portfolio"],
            example_usage="""
preset = FinanceRiskPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Optimize for various risk scenarios
risk_archive = await evolve(risk_code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        config["qd"] = {
            "enable_map_elites": True,
            "grid_dimensions": ["var", "cvar", "max_drawdown"],
            "grid_resolution": 15,
            "archive_elitism": True,
        }
        return config


# ============================================================================
# TRADING DOMAIN (3 presets)
# ============================================================================

class TradingGeneralPreset(BasePreset):
    """
    General trading strategy development preset.

    Optimized for:
    - Trading signal generation
    - Strategy development
    - Backtesting optimization
    - Adversarial robustness
    """

    name: str = "trading_general"
    category: str = "domain"
    description: str = "Trading strategy development"
    evolution_mode: str = "adversarial"  # For robust strategies

    # Adversarial parameters
    max_iterations: int = Field(default=40, description="For adversarial rounds")
    population_size: int = Field(default=150, description="Manageable size")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use="Trading strategy development, signal optimization",
            trade_offs={
                "Robustness": "Adversarial training for robustness",
                "Validation": "Rigorous out-of-sample testing",
                "Focus": "Strategy performance and stability"
            },
            related_presets=["trading_signal", "trading_parameter"],
            example_usage="""
preset = TradingGeneralPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Develop robust trading strategy
strategy = await evolve(trading_code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        config["adversarial"] = {
            "enable_adversarial": True,
            "adversarial_mode": "generator_discriminator",
            "num_adversaries": 2,
            "adversarial_rounds": 20,
            "use_arms_race": True,  # Progressive difficulty
        }
        return config


class TradingSignalPreset(BasePreset):
    """
    Signal optimization preset.

    Optimized for:
    - Trading signal generation
    - Feature engineering
    - Signal-to-noise optimization
    """

    name: str = "trading_signal"
    category: str = "domain"
    description: str = "Trading signal optimization"
    evolution_mode: str = "qd"  # For diverse signal types

    max_iterations: int = Field(default=60, description="For signal diversity")
    population_size: int = Field(default=300, description="For variety")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use="Signal optimization, feature engineering",
            trade_offs={
                "Diversity": "QD for diverse signal types",
                "Metrics": "Sharpe ratio, win rate, profit factor",
                "Output": "Archive of signal strategies"
            },
            related_presets=["trading_general", "trading_parameter"],
            example_usage="""
preset = TradingSignalPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Generate diverse trading signals
signals = await evolve(signal_code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        config["qd"] = {
            "enable_map_elites": True,
            "grid_dimensions": ["sharpe_ratio", "win_rate", "profit_factor"],
            "grid_resolution": 12,
        }
        return config


class TradingParameterPreset(BasePreset):
    """
    Parameter tuning preset for existing strategies.

    Optimized for:
    - Hyperparameter optimization
    - Strategy calibration
    - Fine-tuning existing logic
    """

    name: str = "trading_parameter"
    category: str = "domain"
    description: str = "Parameter tuning for trading strategies"
    evolution_mode: str = "pes"  # Planning-based for efficiency

    max_iterations: int = Field(default=30, description="Quick tuning")
    population_size: int = Field(default=100, description="Focused search")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use="Parameter tuning, strategy calibration",
            trade_offs={
                "Focus": "Parameter optimization only",
                "Speed": "Fast convergence",
                "Method": "Planning-based for efficiency"
            },
            related_presets=["trading_general", "trading_signal"],
            example_usage="""
preset = TradingParameterPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Tune strategy parameters
tuned = await evolve(strategy_code, config=config)
"""
        )


# ============================================================================
# SCIENCE DOMAIN (3 presets)
# ============================================================================

class ScienceGeneralPreset(BasePreset):
    """
    General scientific optimization preset.

    Optimized for:
    - Scientific computing
    - Numerical optimization
    - Algorithm discovery
    """

    name: str = "science_general"
    category: str = "domain"
    description: str = "General scientific computing optimization"
    evolution_mode: str = "openevolve"

    max_iterations: int = Field(default=80, description="Standard science tasks")
    population_size: int = Field(default=300, description="Good coverage")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use="Scientific computing, numerical optimization",
            trade_offs={
                "Precision": "High precision numerical computation",
                "Reproducibility": "Fixed random seed",
                "Method": "Standard evolution"
            },
            related_presets=["science_optimization", "science_discovery"],
            example_usage="""
preset = ScienceGeneralPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Optimize scientific algorithm
result = await evolve(science_code, config=config)
"""
        )


class ScienceOptimizationPreset(BasePreset):
    """
    Scientific optimization preset.

    Optimized for:
    - Function optimization
    - Numerical methods
    - Performance tuning
    """

    name: str = "science_optimization"
    category: str = "domain"
    description: str = "Numerical optimization and function maximization"
    evolution_mode: str = "qd"  # For diverse optima

    max_iterations: int = Field(default=100, description="Thorough search")
    population_size: int = Field(default=500, description="Extensive coverage")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use="Function optimization, finding global optima",
            trade_offs={
                "Coverage": "QD for multiple optima",
                "Diversity": "Explore solution landscape",
                "Output": "Archive of diverse solutions"
            },
            related_presets=["science_general", "science_discovery"],
            example_usage="""
preset = ScienceOptimizationPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Find multiple optima
optima = await evolve(function_code, config=config)
"""
        )


class ScienceDiscoveryPreset(BasePreset):
    """
    Scientific discovery preset.

    Optimized for:
    - Novel algorithm discovery
    - Exploration of solution space
    - Research applications
    """

    name: str = "science_discovery"
    category: str = "domain"
    description: str = "Novel algorithm discovery for research"
    evolution_mode: str = "qd"  # Maximum diversity

    max_iterations: int = Field(default=150, description="Extensive exploration")
    population_size: int = Field(default=800, description="Maximum diversity")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use="Research, novel algorithm discovery",
            trade_offs={
                "Novelty": "Emphasis on novelty search",
                "Diversity": "Maximum solution diversity",
                "Time": "Longer exploration phase"
            },
            related_presets=["science_general", "science_optimization"],
            example_usage="""
preset = ScienceDiscoveryPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Discover novel algorithms
novel = await evolve(research_code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        config["qd"] = {
            "enable_map_elites": True,
            "use_novelty": True,  # Enable novelty search
            "novelty_threshold": 0.3,
            "grid_resolution": 25,
        }
        config["openevolve"] = {
            "enable_novelty_search": True,
        }
        return config


# ============================================================================
# ENGINEERING DOMAIN (3 presets)
# ============================================================================

class EngineeringGeneralPreset(BasePreset):
    """
    General engineering optimization preset.

    Optimized for:
    - Engineering design optimization
    - Control systems
    - Performance tuning
    """

    name: str = "engineering_general"
    category: str = "domain"
    description: str = "General engineering optimization"
    evolution_mode: str = "pes"

    max_iterations: int = Field(default=70, description="Standard engineering")
    population_size: int = Field(default=250, description="Good coverage")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use="Engineering design, optimization",
            trade_offs={
                "Method": "Planning-based for structured design",
                "Validation": "Engineering constraints enforced",
                "Output": "Practical, implementable solutions"
            },
            related_presets=["engineering_design", "engineering_control"],
            example_usage="""
preset = EngineeringGeneralPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Optimize engineering design
design = await evolve(engineering_code, config=config)
"""
        )


class EngineeringDesignPreset(BasePreset):
    """
    Engineering design optimization preset.

    Optimized for:
    - Multi-constraint design
    - Pareto optimization
    - Design space exploration
    """

    name: str = "engineering_design"
    category: str = "domain"
    description: str = "Multi-objective engineering design"
    evolution_mode: str = "mo"  # Multi-objective

    max_iterations: int = Field(default=120, description="For convergence")
    population_size: int = Field(default=400, description="For Pareto front")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use="Multi-constraint design optimization",
            trade_offs={
                "Objectives": "Cost, performance, reliability",
                "Output": "Pareto front of designs",
                "Constraints": "Engineering constraints enforced"
            },
            related_presets=["engineering_general", "engineering_control"],
            example_usage="""
preset = EngineeringDesignPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Get Pareto-optimal designs
designs = await evolve(design_code, config=config)
"""
        )


class EngineeringControlPreset(BasePreset):
    """
    Control systems optimization preset.

    Optimized for:
    - PID controller tuning
    - Control system design
    - Stability optimization
    """

    name: str = "engineering_control"
    category: str = "domain"
    description: str = "Control systems optimization"
    evolution_mode: str = "openevolve"

    max_iterations: int = Field(default=60, description="For convergence")
    population_size: int = Field(default=200, description="Focused search")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use="Control system tuning, optimization",
            trade_offs={
                "Focus": "Stability and performance",
                "Method": "Standard evolution",
                "Output": "Optimized control parameters"
            },
            related_presets=["engineering_general", "engineering_design"],
            example_usage="""
preset = EngineeringControlPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Optimize control system
controller = await evolve(control_code, config=config)
"""
        )


# ============================================================================
# PHARMA DOMAIN (3 presets)
# ============================================================================

class PharmaGeneralPreset(BasePreset):
    """
    General pharmaceutical optimization preset.

    Optimized for:
    - Drug discovery
    - Molecular optimization
    - Clinical trial analysis
    """

    name: str = "pharma_general"
    category: str = "domain"
    description: str = "General pharmaceutical research optimization"
    evolution_mode: str = "qd"  # For diverse candidates

    max_iterations: int = Field(default=100, description="Thorough search")
    population_size: int = Field(default=500, description="Diverse candidates")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use="Drug discovery, molecular optimization",
            trade_offs={
                "Diversity": "QD for diverse molecular candidates",
                "Safety": "Safety constraints enforced",
                "Output": "Archive of candidates"
            },
            related_presets=["pharma_drug_discovery", "pharma_clinical"],
            example_usage="""
preset = PharmaGeneralPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Discover drug candidates
candidates = await evolve(molecular_code, config=config)
"""
        )


class PharmaDrugDiscoveryPreset(BasePreset):
    """
    Drug discovery preset.

    Optimized for:
    - Lead optimization
    - ADMET prediction
    - Molecular design
    """

    name: str = "pharma_drug_discovery"
    category: str = "domain"
    description: str = "Lead optimization and drug discovery"
    evolution_mode: str = "qd"

    max_iterations: int = Field(default=150, description="Extensive search")
    population_size: int = Field(default=800, description="Large library")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use="Lead optimization, molecular design",
            trade_offs={
                "Objectives": "Efficacy, safety, ADMET properties",
                "Diversity": "Explore chemical space",
                "Output": "Ranked candidates"
            },
            related_presets=["pharma_general", "pharma_clinical"],
            example_usage="""
preset = PharmaDrugDiscoveryPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Optimize lead compounds
leads = await evolve(lead_code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        config["qd"] = {
            "enable_map_elites": True,
            "grid_dimensions": ["efficacy", "safety", "admet_score"],
            "grid_resolution": 20,
            "archive_elitism": True,
        }
        return config


class PharmaClinicalPreset(BasePreset):
    """
    Clinical trial analysis preset.

    Optimized for:
    - Clinical trial optimization
    - Treatment protocol design
    - Outcome prediction
    """

    name: str = "pharma_clinical"
    category: str = "domain"
    description: str = "Clinical trial optimization and analysis"
    evolution_mode: str = "pes"

    max_iterations: int = Field(default=80, description="Standard analysis")
    population_size: int = Field(default=300, description="Good coverage")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use="Clinical trial design, analysis",
            trade_offs={
                "Planning": "Structured trial design",
                "Safety": "Patient safety prioritized",
                "Method": "Planning-based evolution"
            },
            related_presets=["pharma_general", "pharma_drug_discovery"],
            example_usage="""
preset = PharmaClinicalPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Optimize clinical trial
trial = await evolve(trial_code, config=config)
"""
        )


# ============================================================================
# WEB DESIGN DOMAIN (3 presets)
# ============================================================================

class WebDesignGeneralPreset(BasePreset):
    """
    General web design optimization preset.

    Optimized for:
    - Frontend optimization
    - UX improvement
    - Performance tuning
    """

    name: str = "web_design_general"
    category: str = "domain"
    description: str = "General web design and frontend optimization"
    evolution_mode: str = "openevolve"

    max_iterations: int = Field(default=60, description="Quick iterations")
    population_size: int = Field(default=200, description="Focused variety")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use="Frontend optimization, UX improvement",
            trade_offs={
                "Speed": "Fast iteration for quick feedback",
                "Quality": "Good quality solutions",
                "Method": "Standard evolution"
            },
            related_presets=["web_design_ux", "web_design_performance"],
            example_usage="""
preset = WebDesignGeneralPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Optimize frontend code
frontend = await evolve(web_code, config=config)
"""
        )


class WebDesignUxPreset(BasePreset):
    """
    UX optimization preset.

    Optimized for:
    - User experience optimization
    - Accessibility improvement
    - User interface design
    """

    name: str = "web_design_ux"
    category: str = "domain"
    description: str = "User experience and accessibility optimization"
    evolution_mode: str = "mo"  # Multiple UX objectives

    max_iterations: int = Field(default=80, description="For UX objectives")
    population_size: int = Field(default=300, description="For diversity")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use="UX optimization, accessibility improvements",
            trade_offs={
                "Objectives": "Accessibility, usability, aesthetics",
                "Method": "Multi-objective optimization",
                "Output": "Pareto-optimal UX solutions"
            },
            related_presets=["web_design_general", "web_design_performance"],
            example_usage="""
preset = WebDesignUxPreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Optimize for multiple UX metrics
ux = await evolve(ux_code, config=config)
"""
        )

    def to_unified_config(self) -> Dict:
        config = super().to_unified_config()
        config["mo"] = {
            "objectives": ["accessibility", "usability", "performance"],
            "use_pareto": True,
        }
        return config


class WebDesignPerformancePreset(BasePreset):
    """
    Web performance optimization preset.

    Optimized for:
    - Page load speed
    - Resource optimization
    - Core Web Vitals
    """

    name: str = "web_design_performance"
    category: str = "domain"
    description: str = "Web performance and Core Web Vitals optimization"
    evolution_mode: str = "openevolve"

    max_iterations: int = Field(default=70, description="Performance tuning")
    population_size: int = Field(default=250, description="Good coverage")

    def get_info(self) -> PresetInfo:
        return PresetInfo(
            name=self.name,
            category=self.category,
            description=self.description,
            when_to_use="Performance optimization, Core Web Vitals",
            trade_offs={
                "Focus": "Load time, responsiveness, stability",
                "Method": "Standard evolution",
                "Output": "Optimized frontend code"
            },
            related_presets=["web_design_general", "web_design_ux"],
            example_usage="""
preset = WebDesignPerformancePreset()
config = UnifiedEvolutionConfig.from_dict(preset.to_unified_config())

# Optimize web performance
performance = await evolve(perf_code, config=config)
"""
        )
