"""
Default Configurations for Different Domains

Provides pre-configured settings optimized for:
- Finance problems
- Trading problems
- Scientific experiments
- Engineering optimization
- Pharmaceutical development
- Web design optimization
"""

from typing import Dict, Any
from .config import (
    UnifiedEvolutionConfig,
    CommonConfig,
    LLMConfig,
    LLMModelConfig,
    DatabaseConfig,
    EvaluatorConfig,
    PESConfig,
    QDConfig,
    MOConfig,
    OpenEvolveConfig,
)


def get_finance_config() -> UnifiedEvolutionConfig:
    """
    Configuration optimized for financial problems:
    - Risk analysis
    - Portfolio optimization
    - Fraud detection
    - Credit scoring

    Characteristics:
    - High precision required
    - Multi-objective (return vs risk)
    - Conservative exploration
    - Strong validation
    """
    return UnifiedEvolutionConfig(
        evolution_mode="mo",
        enable_modes=["openevolve", "mo", "qd"],

        # Common settings - conservative approach
        common=CommonConfig(
            max_iterations=500,
            random_seed=42,
            checkpoint_interval=25,
            log_level="INFO",
            workspace_path="./finance_evolution",
            task_name="financial_optimization",
            concurrency=3,  # Lower concurrency for stability
            timeout=600,  # 10 min per evaluation
        ),

        # LLM config - use best models for accuracy
        llm=LLMConfig(
            models=[
                LLMModelConfig(
                    name="gpt-4o",
                    weight=0.7,
                    temperature=0.3,  # Lower temperature for precision
                    max_tokens=8192,
                    reasoning_effort="high"
                ),
                LLMModelConfig(
                    name="claude-3-5-sonnet-20241022",
                    weight=0.3,
                    temperature=0.3,
                    max_tokens=8192,
                ),
            ],
            default_temperature=0.3,
        ),

        # Database - larger population for diversity
        database=DatabaseConfig(
            storage_type="in_memory",
            population_size=2000,
            elite_archive_size=200,
            num_islands=5,
            migration_interval=50,
            migration_rate=0.15,
            feature_dimensions=["return", "risk", "volatility", "sharpe_ratio"],
            feature_bins={"return": 20, "risk": 20, "volatility": 15, "sharpe_ratio": 15},
            elite_selection_ratio=0.15,
            exploration_rate=0.15,  # Lower exploration for stability
            exploitation_ratio=0.70,
        ),

        # Evaluator - strict validation
        evaluator=EvaluatorConfig(
            timeout=600,
            max_retries=5,
            cascade_evaluation=True,
            cascade_thresholds=[0.6, 0.8, 0.95],
            parallel_evaluations=3,
            use_llm_feedback=True,
            llm_feedback_weight=0.15,
        ),

        # Multi-objective optimization
        mo=MOConfig(
            objectives=["return", "risk", "volatility", "sharpe_ratio"],
            objective_weights={"return": 0.4, "risk": 0.3, "volatility": 0.15, "sharpe_ratio": 0.15},
            optimization_direction={
                "return": "maximize",
                "risk": "minimize",
                "volatility": "minimize",
                "sharpe_ratio": "maximize"
            },
            use_pareto=True,
            pareto_archive_size=200,
            selection_method="nsga2",
            tournament_size=4,
        ),

        # Quality Diversity
        qd=QDConfig(
            enable_map_elites=True,
            grid_resolution=20,
            grid_dimensions=["return", "risk"],
            archive_elitism=True,
            use_novelty=False,
        ),

        # OpenEvolve-specific
        openevolve=OpenEvolveConfig(
            system_message="You are an expert financial analyst optimizing investment strategies.",
            num_top_programs=5,
            num_diverse_programs=3,
            diff_based_evolution=True,
            max_code_length=15000,
            early_stopping_patience=50,
            convergence_threshold=0.0001,
            use_meta_prompting=True,
            enable_novelty_search=False,
        ),
    )


def get_trading_config() -> UnifiedEvolutionConfig:
    """
    Configuration optimized for trading problems:
    - Strategy optimization
    - Signal generation
    - Order execution
    - Market making

    Characteristics:
    - Fast iteration required
    - Adaptive to market conditions
    - High concurrency
    - Real-time feedback
    """
    return UnifiedEvolutionConfig(
        evolution_mode="hybrid",
        enable_modes=["openevolve", "pes", "qd"],

        # Common settings - fast iteration
        common=CommonConfig(
            max_iterations=1000,
            random_seed=None,  # Random for diversity
            checkpoint_interval=50,
            log_level="INFO",
            workspace_path="./trading_evolution",
            task_name="trading_strategy",
            concurrency=10,  # High concurrency for speed
            timeout=120,  # 2 min per evaluation
        ),

        # LLM config - fast models
        llm=LLMConfig(
            models=[
                LLMModelConfig(
                    name="gpt-4o",
                    weight=0.5,
                    temperature=0.6,
                    max_tokens=4096,
                ),
                LLMModelConfig(
                    name="gemini-2.0-flash",
                    weight=0.5,
                    temperature=0.6,
                    max_tokens=4096,
                ),
            ],
        ),

        # Database - smaller, faster population
        database=DatabaseConfig(
            storage_type="in_memory",
            population_size=500,
            elite_archive_size=50,
            num_islands=10,  # Many islands for diversity
            migration_interval=20,
            migration_rate=0.2,
            feature_dimensions=["profit_factor", "max_drawdown", "win_rate", "signal_quality"],
            feature_bins=10,
            exploration_rate=0.3,  # Higher exploration
            exploitation_ratio=0.6,
        ),

        # Evaluator - fast evaluation
        evaluator=EvaluatorConfig(
            timeout=120,
            max_retries=2,
            cascade_evaluation=True,
            cascade_thresholds=[0.4, 0.7, 0.9],
            parallel_evaluations=10,
            use_llm_feedback=False,  # Skip for speed
        ),

        # PES for planning strategies
        pes=PESConfig(
            enable_planning=True,
            planner_type="evolve_planner",
            planning_iterations=2,
            enable_refinement=True,
            executor_type="evolve_executor",
            execution_mode="parallel",
            enable_code_execution=True,
            execution_timeout=120,
            enable_summary=True,
            summary_detail_level="low",
        ),

        # Quality Diversity for strategy variants
        qd=QDConfig(
            enable_map_elites=True,
            grid_resolution=15,
            grid_dimensions=["profit_factor", "max_drawdown"],
            adaptive_grid=True,
            grid_update_interval=50,
            use_novelty=True,
            novelty_threshold=0.3,
        ),

        # OpenEvolve-specific
        openevolve=OpenEvolveConfig(
            system_message="You are an expert algorithmic trader developing profitable strategies.",
            num_top_programs=3,
            num_diverse_programs=5,
            diff_based_evolution=True,
            max_code_length=8000,
            early_stopping_patience=100,
            enable_novelty_search=True,
            use_crossover=True,
            crossover_method="uniform",
        ),
    )


def get_scientific_config() -> UnifiedEvolutionConfig:
    """
    Configuration optimized for scientific experiments:
    - Parameter tuning
    - Experiment design
    - Data analysis
    - Hypothesis testing

    Characteristics:
    - High precision required
    - Reproducibility critical
    - Thorough exploration
    - Statistical validation
    """
    return UnifiedEvolutionConfig(
        evolution_mode="qd",
        enable_modes=["openevolve", "qd", "mo"],

        # Common settings - thorough exploration
        common=CommonConfig(
            max_iterations=2000,
            random_seed=42,  # Fixed for reproducibility
            checkpoint_interval=100,
            log_level="DEBUG",  # Detailed logging
            workspace_path="./scientific_evolution",
            task_name="scientific_experiment",
            concurrency=5,
            timeout=300,  # 5 min per evaluation
        ),

        # LLM config - use reasoning models
        llm=LLMConfig(
            models=[
                LLMModelConfig(
                    name="o1-preview",  # Reasoning model
                    weight=0.6,
                    temperature=0.2,
                    max_tokens=16384,
                    reasoning_effort="high",
                ),
                LLMModelConfig(
                    name="claude-3-5-sonnet-20241022",
                    weight=0.4,
                    temperature=0.2,
                    max_tokens=8192,
                ),
            ],
            default_temperature=0.2,
        ),

        # Database - very large population for thorough exploration
        database=DatabaseConfig(
            storage_type="in_memory",
            population_size=5000,
            elite_archive_size=500,
            num_islands=8,
            migration_interval=100,
            migration_rate=0.1,
            feature_dimensions=["accuracy", "precision", "recall", "f1_score", "complexity"],
            feature_bins=25,  # High resolution
            elite_selection_ratio=0.1,
            exploration_rate=0.25,
            exploitation_ratio=0.65,
        ),

        # Evaluator - thorough validation
        evaluator=EvaluatorConfig(
            timeout=300,
            max_retries=5,
            cascade_evaluation=True,
            cascade_thresholds=[0.5, 0.75, 0.9, 0.95],
            parallel_evaluations=5,
            use_llm_feedback=True,
            llm_feedback_weight=0.2,
        ),

        # Quality Diversity - explore full parameter space
        qd=QDConfig(
            enable_map_elites=True,
            grid_resolution=25,
            grid_dimensions=["accuracy", "complexity"],
            adaptive_grid=True,
            grid_update_interval=100,
            archive_type="map_elites",
            archive_elitism=True,
            use_novelty=True,
            novelty_threshold=0.4,
            feature_extraction_method="auto",
            use_feature_learning=True,
            feature_learning_rate=0.001,
            use_niching=True,
            niche_radius=0.15,
        ),

        # Multi-objective - balance multiple metrics
        mo=MOConfig(
            objectives=["accuracy", "precision", "recall", "f1_score", "complexity"],
            objective_weights=None,  # Equal weight
            optimization_direction={
                "accuracy": "maximize",
                "precision": "maximize",
                "recall": "maximize",
                "f1_score": "maximize",
                "complexity": "minimize"
            },
            use_pareto=True,
            pareto_archive_size=500,
            selection_method="nsga3",
            use_hypervolume=True,
        ),

        # OpenEvolve-specific
        openevolve=OpenEvolveConfig(
            system_message="You are an expert scientist conducting rigorous experiments.",
            num_top_programs=5,
            num_diverse_programs=5,
            diff_based_evolution=True,
            max_code_length=20000,
            early_stopping_patience=200,
            convergence_threshold=0.00001,
            use_meta_prompting=True,
            meta_prompt_weight=0.15,
            enable_novelty_search=True,
            use_embedding=True,
            embedding_model="text-embedding-3-small",
        ),
    )


def get_engineering_config() -> UnifiedEvolutionConfig:
    """
    Configuration optimized for engineering problems:
    - Design optimization
    - Parameter tuning
    - Performance optimization
    - Resource allocation

    Characteristics:
    - Practical solutions preferred
    - Resource constraints important
    - Balanced exploration/exploitation
    - Multi-objective common
    """
    return UnifiedEvolutionConfig(
        evolution_mode="mo",
        enable_modes=["openevolve", "mo"],

        # Common settings - balanced approach
        common=CommonConfig(
            max_iterations=800,
            random_seed=42,
            checkpoint_interval=40,
            log_level="INFO",
            workspace_path="./engineering_evolution",
            task_name="engineering_optimization",
            concurrency=5,
            timeout=300,
        ),

        # LLM config - practical models
        llm=LLMConfig(
            models=[
                LLMModelConfig(
                    name="gpt-4o",
                    weight=0.6,
                    temperature=0.5,
                    max_tokens=8192,
                ),
                LLMModelConfig(
                    name="claude-3-5-sonnet-20241022",
                    weight=0.4,
                    temperature=0.5,
                    max_tokens=8192,
                ),
            ],
        ),

        # Database - moderate size
        database=DatabaseConfig(
            storage_type="in_memory",
            population_size=1500,
            elite_archive_size=150,
            num_islands=6,
            migration_interval=60,
            migration_rate=0.12,
            feature_dimensions=["performance", "cost", "reliability", "efficiency"],
            feature_bins=15,
            elite_selection_ratio=0.12,
            exploration_rate=0.2,
            exploitation_ratio=0.68,
        ),

        # Evaluator - practical evaluation
        evaluator=EvaluatorConfig(
            timeout=300,
            max_retries=3,
            cascade_evaluation=True,
            cascade_thresholds=[0.5, 0.75, 0.9],
            parallel_evaluations=5,
            use_llm_feedback=True,
            llm_feedback_weight=0.1,
        ),

        # Multi-objective - balance practical constraints
        mo=MOConfig(
            objectives=["performance", "cost", "reliability", "efficiency"],
            objective_weights={"performance": 0.35, "cost": 0.25, "reliability": 0.25, "efficiency": 0.15},
            optimization_direction={
                "performance": "maximize",
                "cost": "minimize",
                "reliability": "maximize",
                "efficiency": "maximize"
            },
            use_pareto=True,
            pareto_archive_size=150,
            selection_method="nsga2",
            use_scalarization=True,
            scalarization_method="weighted_sum",
        ),

        # OpenEvolve-specific
        openevolve=OpenEvolveConfig(
            system_message="You are an expert engineer optimizing practical solutions.",
            num_top_programs=4,
            num_diverse_programs=3,
            diff_based_evolution=True,
            max_code_length=12000,
            early_stopping_patience=75,
            enable_simplification=True,
            suggest_simplification_after_chars=1000,
        ),
    )


def get_pharmaceutical_config() -> UnifiedEvolutionConfig:
    """
    Configuration optimized for pharmaceutical development:
    - Drug discovery
    - Molecular optimization
    - Clinical trial design
    - Treatment optimization

    Characteristics:
    - Very high precision required
    - Safety critical
    - Extensive validation
    - Regulatory compliance
    """
    return UnifiedEvolutionConfig(
        evolution_mode="qd",
        enable_modes=["openevolve", "qd", "mo"],

        # Common settings - thorough and careful
        common=CommonConfig(
            max_iterations=3000,
            random_seed=42,
            checkpoint_interval=50,
            log_level="DEBUG",
            workspace_path="./pharma_evolution",
            task_name="drug_discovery",
            concurrency=3,  # Lower for safety
            timeout=900,  # 15 min per evaluation
        ),

        # LLM config - best reasoning models
        llm=LLMConfig(
            models=[
                LLMModelConfig(
                    name="o1-preview",
                    weight=0.7,
                    temperature=0.1,  # Very low for precision
                    max_tokens=32768,
                    reasoning_effort="high",
                ),
                LLMModelConfig(
                    name="claude-3-5-sonnet-20241022",
                    weight=0.3,
                    temperature=0.1,
                    max_tokens=16384,
                ),
            ],
            default_temperature=0.1,
        ),

        # Database - largest population
        database=DatabaseConfig(
            storage_type="in_memory",
            population_size=10000,
            elite_archive_size=1000,
            num_islands=10,
            migration_interval=150,
            migration_rate=0.08,
            feature_dimensions=["efficacy", "safety", "stability", "bioavailability", "toxicity"],
            feature_bins=30,  # Very high resolution
            elite_selection_ratio=0.08,
            exploration_rate=0.12,  # Conservative
            exploitation_ratio=0.80,
        ),

        # Evaluator - extremely thorough
        evaluator=EvaluatorConfig(
            timeout=900,
            max_retries=10,
            cascade_evaluation=True,
            cascade_thresholds=[0.6, 0.8, 0.9, 0.95, 0.98],
            parallel_evaluations=3,
            use_llm_feedback=True,
            llm_feedback_weight=0.25,
        ),

        # Quality Diversity - full exploration
        qd=QDConfig(
            enable_map_elites=True,
            grid_resolution=30,
            grid_dimensions=["efficacy", "safety"],
            adaptive_grid=True,
            grid_update_interval=200,
            archive_type="cvt_map_elites",
            cvt_samples=50000,
            archive_elitism=True,
            use_novelty=True,
            novelty_threshold=0.5,
            feature_extraction_method="learned",
            use_feature_learning=True,
            feature_learning_rate=0.0005,
            use_niching=True,
            niche_radius=0.08,
        ),

        # Multi-objective - safety critical
        mo=MOConfig(
            objectives=["efficacy", "safety", "stability", "bioavailability", "toxicity"],
            objective_weights={
                "efficacy": 0.3,
                "safety": 0.35,  # Safety weighted highest
                "stability": 0.15,
                "bioavailability": 0.1,
                "toxicity": 0.1
            },
            optimization_direction={
                "efficacy": "maximize",
                "safety": "maximize",
                "stability": "maximize",
                "bioavailability": "maximize",
                "toxicity": "minimize"
            },
            use_pareto=True,
            pareto_archive_size=1000,
            selection_method="nsga3",
            use_hypervolume=True,
        ),

        # OpenEvolve-specific
        openevolve=OpenEvolveConfig(
            system_message="You are an expert pharmaceutical researcher developing safe, effective treatments.",
            num_top_programs=7,
            num_diverse_programs=5,
            diff_based_evolution=True,
            max_code_length=25000,
            early_stopping_patience=150,
            convergence_threshold=0.000001,
            use_meta_prompting=True,
            meta_prompt_weight=0.2,
            enable_novelty_search=True,
            use_embedding=True,
            embedding_model="text-embedding-3-large",
            embedding_dimension=3072,
        ),
    )


def get_web_design_config() -> UnifiedEvolutionConfig:
    """
    Configuration optimized for web design optimization:
    - A/B testing
    - UX optimization
    - Conversion optimization
    - Layout optimization

    Characteristics:
    - Fast iteration
    - User feedback important
    - Visual diversity
    - Practical constraints
    """
    return UnifiedEvolutionConfig(
        evolution_mode="hybrid",
        enable_modes=["openevolve", "qd", "pes"],

        # Common settings - fast iteration
        common=CommonConfig(
            max_iterations=500,
            random_seed=None,
            checkpoint_interval=25,
            log_level="INFO",
            workspace_path="./web_design_evolution",
            task_name="ux_optimization",
            concurrency=8,
            timeout=180,  # 3 min per evaluation
        ),

        # LLM config - fast, creative models
        llm=LLMConfig(
            models=[
                LLMModelConfig(
                    name="gpt-4o",
                    weight=0.5,
                    temperature=0.8,  # Higher for creativity
                    max_tokens=6144,
                ),
                LLMModelConfig(
                    name="gemini-2.0-flash",
                    weight=0.5,
                    temperature=0.8,
                    max_tokens=6144,
                ),
            ],
        ),

        # Database - moderate size, diverse
        database=DatabaseConfig(
            storage_type="in_memory",
            population_size=800,
            elite_archive_size=80,
            num_islands=8,
            migration_interval=30,
            migration_rate=0.2,
            feature_dimensions=["conversion_rate", "user_engagement", "visual_appeal", "accessibility"],
            feature_bins=12,
            elite_selection_ratio=0.12,
            exploration_rate=0.35,  # High exploration for diversity
            exploitation_ratio=0.53,
        ),

        # Evaluator - include user feedback
        evaluator=EvaluatorConfig(
            timeout=180,
            max_retries=3,
            cascade_evaluation=True,
            cascade_thresholds=[0.4, 0.7, 0.85],
            parallel_evaluations=8,
            use_llm_feedback=True,
            llm_feedback_weight=0.15,
        ),

        # PES for planning UX improvements
        pes=PESConfig(
            enable_planning=True,
            planner_type="evolve_planner",
            planning_iterations=2,
            enable_refinement=True,
            executor_type="evolve_executor",
            execution_mode="parallel",
            enable_code_execution=True,
            enable_summary=True,
            summary_detail_level="medium",
        ),

        # Quality Diversity - diverse designs
        qd=QDConfig(
            enable_map_elites=True,
            grid_resolution=12,
            grid_dimensions=["conversion_rate", "visual_appeal"],
            adaptive_grid=True,
            grid_update_interval=50,
            archive_elitism=True,
            use_novelty=True,
            novelty_threshold=0.35,
        ),

        # OpenEvolve-specific
        openevolve=OpenEvolveConfig(
            system_message="You are an expert UX/UI designer creating beautiful, effective web experiences.",
            num_top_programs=4,
            num_diverse_programs=6,  # More diverse for creativity
            diff_based_evolution=True,
            max_code_length=10000,
            early_stopping_patience=60,
            use_template_stochasticity=True,
            template_variations={
                "improvement_suggestion": [
                    "Let's enhance this design:",
                    "Here's a UX improvement:",
                    "We can improve the user experience by:",
                    "Consider this design change:",
                ]
            },
            enable_novelty_search=True,
            use_crossover=True,
            crossover_method="uniform",
        ),
    )


# Domain configuration registry
DOMAIN_CONFIGS: Dict[str, callable] = {
    "finance": get_finance_config,
    "trading": get_trading_config,
    "scientific": get_scientific_config,
    "engineering": get_engineering_config,
    "pharmaceutical": get_pharmaceutical_config,
    "web_design": get_web_design_config,
}


def get_domain_config(domain: str) -> UnifiedEvolutionConfig:
    """
    Get default configuration for a specific domain

    Args:
        domain: Domain name (finance, trading, scientific, engineering, pharmaceutical, web_design)

    Returns:
        UnifiedEvolutionConfig with domain-specific defaults

    Raises:
        ValueError: If domain is not recognized
    """
    if domain not in DOMAIN_CONFIGS:
        raise ValueError(
            f"Unknown domain '{domain}'. Available domains: {list(DOMAIN_CONFIGS.keys())}"
        )

    return DOMAIN_CONFIGS[domain]()


def list_domains() -> list[str]:
    """List available domain configurations"""
    return list(DOMAIN_CONFIGS.keys())
