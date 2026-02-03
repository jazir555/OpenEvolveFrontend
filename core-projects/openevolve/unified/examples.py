"""
Unified Configuration Examples
Demonstrates how to configure for different evolution modes and domains

Author: AI Architecture Team
Date: 2026-01-30
"""

from openevolve.unified import (
    UnifiedEvolutionConfig,
    EvolutionMode,
    DomainType,
    PESConfig,
    QDConfig,
    MOConfig,
    AdversarialConfig,
    LLMModelConfig,
    ConfigValidator,
    ConfigMapper
)


# ============================================================================
# EXAMPLE 1: PES MODE (LoongFlow) - Mathematical Optimization
# ============================================================================

def example_pes_math_optimization():
    """
    PES mode for mathematical optimization (e.g., circle packing)

    Best for: Expensive evaluations, reasoning-heavy problems
    """
    config = UnifiedEvolutionConfig(
        # Mode selection
        evolution_mode=EvolutionMode.PES,
        domain=DomainType.MATH,

        # Iteration control
        max_iterations=100,
        time_limit_seconds=3600,  # 1 hour

        # PES configuration
        pes=PESConfig(
            enabled=True,
            enable_planning=True,
            max_rounds=3,
            parallel_candidates=1,
        ),

        # LLM configuration
        llm={
            "models": [
                LLMModelConfig(name="gpt-4", weight=1.0),
                LLMModelConfig(name="claude-3-opus", weight=1.0),
            ],
            "temperature": 0.7,
            "plan_temperature": 0.7,
            "summary_temperature": 0.7,
            "timeout": 60,
        },

        # Database / Memory
        database={
            "num_islands": 3,
            "population_size": 100,
            "enable_memory": True,
            "adaptive_exploration": True,
            "exploration_rate": 0.2,
        },

        # Evaluator
        evaluator={
            "timeout": 300,
            "early_stopping": True,
            "early_stopping_patience": 5,
            "early_stopping_threshold": 0.01,
        },
    )

    # Validate
    validator = ConfigValidator(config)
    errors, warnings = validator.validate()

    print("=== PES Math Optimization ===")
    print(f"Valid: {len(errors) == 0}")
    if errors:
        print("Errors:", [str(e) for e in errors])
    if warnings:
        print("Warnings:", [str(w) for w in warnings])

    # Convert to LoongFlow format
    pes_dict = ConfigMapper.to_pes_config(config)
    print(f"\nPES Config Keys: {list(pes_dict.keys())}")

    return config


# ============================================================================
# EXAMPLE 2: QD MODE (OpenEvolve) - Trading Strategy Discovery
# ============================================================================

def example_qd_trading_strategy():
    """
    QD mode for trading strategy discovery

    Best for: Multi-modal problems, behavioral diversity
    """
    config = UnifiedEvolutionConfig(
        # Mode selection
        evolution_mode=EvolutionMode.QD,
        domain=DomainType.TRADING,

        # Iteration control
        max_iterations=1000,
        early_stopping_patience=50,
        convergence_threshold=0.001,

        # QD configuration
        qd=QDConfig(
            enabled=True,
            grid_resolution=10,
            feature_dimensions=["sharpe_ratio", "max_drawdown"],
            archive_size=1000,
        ),

        # Database
        database={
            "population_size": 1000,
            "num_islands": 10,
            "archive_size": 100,
            "feature_dimensions": ["sharpe_ratio", "max_drawdown"],
            "feature_bins": 10,
            "elite_selection_ratio": 0.1,
            "exploration_ratio": 0.2,
            "exploitation_ratio": 0.7,
            "migration_interval": 50,
            "migration_rate": 0.1,
        },

        # LLM
        llm={
            "models": [
                LLMModelConfig(name="gpt-4", weight=1.0),
            ],
            "temperature": 0.9,  # High creativity for novel strategies
            "diff_based_evolution": True,
        },

        # Evaluator
        evaluator={
            "timeout": 600,  # 10 minutes for backtest
            "cascade_evaluation": False,  # Full backtests only
            "parallel_evaluations": 5,
            "enable_gauntlets": True,
            "gauntlet_strictness": "strict",
        },
    )

    # Validate
    validator = ConfigValidator(config)
    errors, warnings = validator.validate()

    print("=== QD Trading Strategy ===")
    print(f"Valid: {len(errors) == 0}")
    if warnings:
        print("Warnings:", [str(w) for w in warnings])

    # Convert to OpenEvolve format
    oe_dict = ConfigMapper.to_openevolve_config(config)
    print(f"\nOpenEvolve Config Keys: {list(oe_dict.keys())}")

    return config


# ============================================================================
# EXAMPLE 3: MO MODE - Portfolio Optimization
# ============================================================================

def example_mo_portfolio_optimization():
    """
    Multi-objective mode for portfolio optimization

    Best for: Problems with multiple competing objectives
    """
    config = UnifiedEvolutionConfig(
        # Mode selection
        evolution_mode=EvolutionMode.MO,
        domain=DomainType.FINANCE,

        # Iteration control
        max_iterations=500,

        # MO configuration
        mo=MOConfig(
            enabled=True,
            objectives=["return", "risk", "liquidity"],
            objective_weights={"return": 0.5, "risk": 0.3, "liquidity": 0.2},
            algorithm="nsga2",
            pareto_size=100,
        ),

        # Database
        database={
            "population_size": 500,
            "num_islands": 5,
            "feature_dimensions": ["return", "risk"],
            "feature_bins": {"return": 10, "risk": 10},
        },

        # Evaluator
        evaluator={
            "timeout": 300,
            "parallel_evaluations": 10,
            "enable_gauntlets": True,
        },
    )

    # Validate
    validator = ConfigValidator(config)
    errors, warnings = validator.validate()

    print("=== MO Portfolio Optimization ===")
    print(f"Valid: {len(errors) == 0}")
    if warnings:
        print("Warnings:", [str(w) for w in warnings])

    return config


# ============================================================================
# EXAMPLE 4: ADVERSARIAL MODE - Security Testing
# ============================================================================

def example_adversarial_security():
    """
    Adversarial mode for security testing

    Best for: Robustness testing, adversarial attacks
    """
    config = UnifiedEvolutionConfig(
        # Mode selection
        evolution_mode=EvolutionMode.ADVERSARIAL,
        domain=DomainType.GENERAL,

        # Iteration control
        max_iterations=100,

        # Adversarial configuration
        adversarial=AdversarialConfig(
            enabled=True,
            adversarial_rounds=20,
            red_team_models=["gpt-4", "claude-3-opus"],
            blue_team_models=["gpt-4", "claude-3-opus"],
            robustness_threshold=0.8,
        ),

        # Database
        database={
            "population_size": 100,
            "num_islands": 2,  # Red team vs Blue team
        },

        # Evaluator
        evaluator={
            "timeout": 120,
            "enable_gauntlets": True,
            "gauntlet_strictness": "strict",
        },
    )

    # Validate
    validator = ConfigValidator(config)
    errors, warnings = validator.validate()

    print("=== Adversarial Security Testing ===")
    print(f"Valid: {len(errors) == 0}")

    return config


# ============================================================================
# EXAMPLE 5: PES MODE - Scientific Experiment Design
# ============================================================================

def example_pes_science_experiment():
    """
    PES mode for scientific experiment design

    Best for: Expensive evaluations, domain knowledge integration
    """
    config = UnifiedEvolutionConfig(
        # Mode selection
        evolution_mode=EvolutionMode.PES,
        domain=DomainType.SCIENCE,

        # Iteration control
        max_iterations=50,
        time_limit_seconds=7200,  # 2 hours

        # PES configuration
        pes=PESConfig(
            enabled=True,
            enable_planning=True,
            enable_summary=True,
            max_rounds=5,  # More rounds for complex experiments
            use_memory=True,
            memory_top_k=10,  # Retrieve more past experiments
        ),

        # LLM
        llm={
            "models": [
                LLMModelConfig(name="claude-3-opus", weight=1.0),  # Best for reasoning
            ],
            "temperature": 0.7,
            "timeout": 120,
        },

        # Database
        database={
            "num_islands": 5,
            "population_size": 50,
            "enable_memory": True,
            "exploration_rate": 0.3,  # Higher exploration for novel experiments
        },

        # Evaluator
        evaluator={
            "timeout": 1800,  # 30 minutes for simulation
            "early_stopping": True,
            "early_stopping_patience": 3,
        },
    )

    # Validate
    validator = ConfigValidator(config)
    errors, warnings = validator.validate()

    print("=== PES Scientific Experiment ===")
    print(f"Valid: {len(errors) == 0}")
    if warnings:
        print("Warnings:", [str(w) for w in warnings])

    return config


# ============================================================================
# EXAMPLE 6: AUTO MODE - Automatic Mode Selection
# ============================================================================

def example_auto_mode():
    """
    Auto mode - automatically select based on configuration

    System will detect which mode to use based on enabled configs
    """
    config = UnifiedEvolutionConfig(
        # Auto mode
        evolution_mode=EvolutionMode.AUTO,
        domain=DomainType.GENERAL,

        # Enable PES - this will trigger PES mode
        pes=PESConfig(enabled=True),

        # Other parameters
        max_iterations=100,
        database={
            "num_islands": 3,
            "population_size": 100,
        },
    )

    # Auto-detection will set evolution_mode to PES
    print(f"Detected Mode: {config.evolution_mode}")

    return config


# ============================================================================
# EXAMPLE 7: Domain-Specific Presets
# ============================================================================

def example_domain_presets():
    """Show domain-specific configuration presets"""

    presets = {
        "finance": {
            "evolution_mode": EvolutionMode.MO,
            "domain": DomainType.FINANCE,
            "mo": MOConfig(
                enabled=True,
                objectives=["return", "risk"],
            ),
        },
        "trading": {
            "evolution_mode": EvolutionMode.QD,
            "domain": DomainType.TRADING,
            "qd": QDConfig(
                enabled=True,
                feature_dimensions=["return", "drawdown"],
            ),
        },
        "science": {
            "evolution_mode": EvolutionMode.PES,
            "domain": DomainType.SCIENCE,
            "pes": PESConfig(enabled=True),
        },
        "math": {
            "evolution_mode": EvolutionMode.PES,
            "domain": DomainType.MATH,
            "pes": PESConfig(enabled=True),
        },
        "ml": {
            "evolution_mode": EvolutionMode.PES,
            "domain": DomainType.ML,
            "pes": PESConfig(enabled=True),
        },
    }

    print("=== Domain Presets ===")
    for domain_name, preset in presets.items():
        config = UnifiedEvolutionConfig(**preset)
        print(f"{domain_name}: {config.evolution_mode.value}")

    return presets


# ============================================================================
# EXAMPLE 8: Configuration Conversion
# ============================================================================

def example_conversion():
    """Show conversion between different config formats"""

    # Start with OpenEvolve-style dict
    oe_dict = {
        "max_iterations": 1000,
        "database": {
            "population_size": 500,
            "num_islands": 5,
            "feature_dimensions": ["complexity", "diversity"],
        },
        "llm": {
            "temperature": 0.7,
            "models": [{"name": "gpt-4", "weight": 1.0}],
        },
    }

    # Convert to unified config
    unified = ConfigMapper.from_openevolve_dict(oe_dict)

    print("=== Configuration Conversion ===")
    print(f"OpenEvolve -> Unified: {unified.evolution_mode}")

    # Convert back to OpenEvolve format
    oe_converted = ConfigMapper.to_openevolve_config(unified)
    print(f"Unified -> OpenEvolve: {list(oe_converted.keys())}")

    # Convert to PES format
    pes_converted = ConfigMapper.to_pes_config(unified)
    print(f"Unified -> PES: {list(pes_converted.keys())}")

    return unified


# ============================================================================
# MAIN: Run all examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("UNIFIED CONFIGURATION EXAMPLES")
    print("=" * 80)

    # Run examples
    example_pes_math_optimization()
    print("\n" + "=" * 80 + "\n")

    example_qd_trading_strategy()
    print("\n" + "=" * 80 + "\n")

    example_mo_portfolio_optimization()
    print("\n" + "=" * 80 + "\n")

    example_adversarial_security()
    print("\n" + "=" * 80 + "\n")

    example_pes_science_experiment()
    print("\n" + "=" * 80 + "\n")

    example_auto_mode()
    print("\n" + "=" * 80 + "\n")

    example_domain_presets()
    print("\n" + "=" * 80 + "\n")

    example_conversion()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)
