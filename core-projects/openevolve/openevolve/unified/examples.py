"""
Unified Configuration System - Usage Examples

This file demonstrates how to use the unified configuration system
for various evolutionary modes and use cases.
"""

from openevolve.unified import (
    UnifiedEvolutionConfig,
    CommonConfig,
    LLMConfig,
    LLMModelConfig,
    DatabaseConfig,
    EvaluatorConfig,
    PESConfig,
    QDConfig,
    MOConfig,
    AdversarialConfig,
    OpenEvolveConfig,
    ConfigMapper,
    ConfigValidator,
    get_finance_config,
    get_trading_config,
    get_scientific_config,
    get_engineering_config,
    get_pharmaceutical_config,
    get_web_design_config,
    get_domain_config,
    list_domains,
)


# ============================================================================
# Example 1: Basic OpenEvolve Configuration
# ============================================================================

def example_basic_openevolve():
    """Create a basic OpenEvolve configuration"""
    print("Example 1: Basic OpenEvolve Configuration")
    print("-" * 80)

    config = UnifiedEvolutionConfig(
        evolution_mode="openevolve",
        common=CommonConfig(
            max_iterations=100,
            random_seed=42,
            workspace_path="./basic_evolution",
        ),
        database=DatabaseConfig(
            population_size=500,
            num_islands=3,
        ),
        openevolve=OpenEvolveConfig(
            system_message="You are an expert Python programmer.",
            diff_based_evolution=True,
        ),
    )

    print(f"Mode: {config.evolution_mode}")
    print(f"Max iterations: {config.common.max_iterations}")
    print(f"Population size: {config.database.population_size}")
    print()

    # Save to file
    config.save_yaml("./examples/basic_config.yaml")
    print("Saved to ./examples/basic_config.yaml")
    print()


# ============================================================================
# Example 2: Multi-Objective Optimization
# ============================================================================

def example_multi_objective():
    """Configure multi-objective optimization"""
    print("Example 2: Multi-Objective Optimization")
    print("-" * 80)

    config = UnifiedEvolutionConfig(
        evolution_mode="mo",
        enable_modes=["openevolve", "mo"],
        common=CommonConfig(
            max_iterations=500,
            task_name="multi_objective_optimization",
        ),
        mo=MOConfig(
            objectives=["accuracy", "efficiency", "cost"],
            objective_weights={
                "accuracy": 0.5,
                "efficiency": 0.3,
                "cost": 0.2,
            },
            optimization_direction={
                "accuracy": "maximize",
                "efficiency": "maximize",
                "cost": "minimize",
            },
            use_pareto=True,
            pareto_archive_size=100,
            selection_method="nsga2",
        ),
    )

    print(f"Objectives: {config.mo.objectives}")
    print(f"Use Pareto: {config.mo.use_pareto}")
    print(f"Archive size: {config.mo.pareto_archive_size}")
    print()

    # Validate
    validator = ConfigValidator(config)
    if validator.is_valid():
        print("[OK] Configuration is valid!")
    else:
        print("[FAIL] Configuration has errors:")
        print(validator.get_validation_report())
    print()


# ============================================================================
# Example 3: Quality Diversity with MAP-Elites
# ============================================================================

def example_quality_diversity():
    """Configure Quality Diversity optimization"""
    print("Example 3: Quality Diversity (MAP-Elites)")
    print("-" * 80)

    config = UnifiedEvolutionConfig(
        evolution_mode="qd",
        enable_modes=["openevolve", "qd"],
        qd=QDConfig(
            enable_map_elites=True,
            grid_resolution=20,
            grid_dimensions=["complexity", "novelty"],
            adaptive_grid=True,
            use_novelty=True,
            novelty_threshold=0.3,
        ),
        database=DatabaseConfig(
            population_size=2000,
            feature_dimensions=["complexity", "novelty"],
            feature_bins=20,
        ),
    )

    print(f"Grid resolution: {config.qd.grid_resolution}")
    print(f"Grid dimensions: {config.qd.grid_dimensions}")
    print(f"Adaptive grid: {config.qd.adaptive_grid}")
    print()

    # Convert to OpenEvolve format
    oe_config = ConfigMapper.to_qd_config(config)
    print("Converted to OpenEvolve format")
    print(f"Feature bins: {oe_config['database']['feature_bins']}")
    print()


# ============================================================================
# Example 4: LoongFlow PES Configuration
# ============================================================================

def example_pes_configuration():
    """Configure Plan-Evolve-Summarize mode"""
    print("Example 4: LoongFlow PES Configuration")
    print("-" * 80)

    config = UnifiedEvolutionConfig(
        evolution_mode="pes",
        enable_modes=["pes"],
        pes=PESConfig(
            enable_planning=True,
            planner_type="evolve_planner",
            planning_iterations=3,
            use_refinement=True,
            max_refinement_iterations=5,
            executor_type="evolve_executor",
            execution_mode="parallel",
            enable_code_execution=True,
            enable_summary=True,
            summary_detail_level="high",
        ),
    )

    print(f"Planning enabled: {config.pes.enable_planning}")
    print(f"Planning iterations: {config.pes.planning_iterations}")
    print(f"Execution mode: {config.pes.execution_mode}")
    print()

    # Convert to PES format
    pes_config = ConfigMapper.to_pes_config(config)
    print("Converted to PES format")
    print(f"Planners: {list(pes_config.get('planners', {}).keys())}")
    print(f"Executors: {list(pes_config.get('executors', {}).keys())}")
    print()


# ============================================================================
# Example 5: Adversarial Evolution
# ============================================================================

def example_adversarial_evolution():
    """Configure adversarial evolution"""
    print("Example 5: Adversarial Evolution")
    print("-" * 80)

    config = UnifiedEvolutionConfig(
        evolution_mode="adversarial",
        enable_modes=["openevolve", "adversarial"],
        adversarial=AdversarialConfig(
            enable_adversarial=True,
            num_adversaries=3,
            adversarial_mode="generator_discriminator",
            generator_objective="fool_discriminator",
            discriminator_objective="detect_fake",
            balance_factor=0.5,
            use_coevolution=True,
            coevolution_frequency=5,
        ),
    )

    print(f"Adversarial mode: {config.adversarial.adversarial_mode}")
    print(f"Number of adversaries: {config.adversarial.num_adversaries}")
    print(f"Balance factor: {config.adversarial.balance_factor}")
    print()

    # Validate
    validator = ConfigValidator(config)
    errors, warnings = validator.validate()
    if errors:
        print("Errors found:")
        for error in errors:
            print(f"  - {error}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print()


# ============================================================================
# Example 6: Hybrid Mode
# ============================================================================

def example_hybrid_mode():
    """Configure hybrid mode combining multiple approaches"""
    print("Example 6: Hybrid Mode (QD + MO)")
    print("-" * 80)

    config = UnifiedEvolutionConfig(
        evolution_mode="hybrid",
        enable_modes=["openevolve", "qd", "mo"],
        qd=QDConfig(
            enable_map_elites=True,
            grid_resolution=15,
            grid_dimensions=["accuracy", "robustness"],
        ),
        mo=MOConfig(
            objectives=["accuracy", "robustness", "efficiency"],
            use_pareto=True,
        ),
    )

    print(f"Evolution mode: {config.evolution_mode}")
    print(f"Enabled modes: {config.enable_modes}")
    print(f"QD grid dimensions: {config.qd.grid_dimensions}")
    print(f"MO objectives: {config.mo.objectives}")
    print()


# ============================================================================
# Example 7: Domain-Specific Presets
# ============================================================================

def example_domain_presets():
    """Use domain-specific configuration presets"""
    print("Example 7: Domain-Specific Presets")
    print("-" * 80)

    # List available domains
    print("Available domains:")
    for domain in list_domains():
        print(f"  - {domain}")
    print()

    # Get finance configuration
    finance_config = get_finance_config()
    print("Finance configuration:")
    print(f"  Max iterations: {finance_config.common.max_iterations}")
    print(f"  Population size: {finance_config.database.population_size}")
    print(f"  Objectives: {finance_config.mo.objectives}")
    print()

    # Get scientific configuration
    scientific_config = get_scientific_config()
    print("Scientific configuration:")
    print(f"  Max iterations: {scientific_config.common.max_iterations}")
    print(f"  Grid resolution: {scientific_config.qd.grid_resolution}")
    print()

    # Get domain config by name
    trading_config = get_domain_config("trading")
    print("Trading configuration:")
    print(f"  Concurrency: {trading_config.common.concurrency}")
    print(f"  Timeout: {trading_config.common.timeout}")
    print()


# ============================================================================
# Example 8: Configuration Mapping
# ============================================================================

def example_config_mapping():
    """Convert between configuration formats"""
    print("Example 8: Configuration Mapping")
    print("-" * 80)

    # Create unified config
    unified_config = get_finance_config()

    # Convert to OpenEvolve format
    oe_config = ConfigMapper.to_openevolve_config(unified_config)
    print("Converted to OpenEvolve format:")
    print(f"  Type: {type(oe_config)}")
    print(f"  Has 'llm' key: {'llm' in oe_config}")
    print(f"  Has 'database' key: {'database' in oe_config}")
    print()

    # Convert to PES format
    pes_config = ConfigMapper.to_pes_config(unified_config)
    print("Converted to PES format:")
    print(f"  Type: {type(pes_config)}")
    print(f"  Has 'evolve' key: {'evolve' in pes_config}")
    print(f"  Has 'planners' key: {'planners' in pes_config}")
    print()

    # Convert back from OpenEvolve
    restored_config = ConfigMapper.from_openevolve_config(oe_config)
    print("Restored from OpenEvolve format:")
    print(f"  Evolution mode: {restored_config.evolution_mode}")
    print(f"  Max iterations: {restored_config.common.max_iterations}")
    print()


# ============================================================================
# Example 9: Serialization
# ============================================================================

def example_serialization():
    """Serialize and deserialize configurations"""
    print("Example 9: Serialization")
    print("-" * 80)

    config = get_trading_config()

    # To YAML
    yaml_str = config.to_yaml()
    print("YAML output (first 500 chars):")
    print(yaml_str[:500] + "...")
    print()

    # From YAML
    restored_config = UnifiedEvolutionConfig.from_yaml(yaml_str)
    print(f"Restored from YAML: {restored_config.evolution_mode}")
    print()

    # To JSON
    json_str = config.to_json()
    print(f"JSON length: {len(json_str)} characters")
    print()

    # To dict
    config_dict = config.to_dict()
    print(f"Dict keys: {list(config_dict.keys())}")
    print()


# ============================================================================
# Example 10: Validation and Error Reporting
# ============================================================================

def example_validation():
    """Demonstrate validation and error reporting"""
    print("Example 10: Validation and Error Reporting")
    print("-" * 80)

    # Create a config with some issues
    config = UnifiedEvolutionConfig(
        evolution_mode="mo",
        mo=MOConfig(
            objectives=[],  # Empty objectives - this will cause an error
        ),
        database=DatabaseConfig(
            population_size=10,  # Very small - this will cause a warning
            num_islands=5,
        ),
    )

    # Validate
    validator = ConfigValidator(config)
    errors, warnings = validator.validate()

    print("Validation Results:")
    print(f"  Errors: {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    print()

    if errors or warnings:
        print(validator.get_validation_report())
    print()

    # Fix the config
    config.mo.objectives = ["accuracy", "efficiency"]
    config.database.population_size = 1000

    # Validate again
    validator = ConfigValidator(config)
    if validator.is_valid():
        print("[OK] Fixed configuration is now valid!")
    print()


# ============================================================================
# Example 11: Custom LLM Ensemble
# ============================================================================

def example_custom_llm_ensemble():
    """Configure custom LLM ensemble"""
    print("Example 11: Custom LLM Ensemble")
    print("-" * 80)

    config = UnifiedEvolutionConfig(
        evolution_mode="openevolve",
        llm=LLMConfig(
            models=[
                LLMModelConfig(
                    name="gpt-4o",
                    weight=0.6,
                    temperature=0.7,
                    max_tokens=8192,
                    reasoning_effort="high",
                ),
                LLMModelConfig(
                    name="claude-3-5-sonnet-20241022",
                    weight=0.3,
                    temperature=0.7,
                    max_tokens=8192,
                ),
                LLMModelConfig(
                    name="gemini-2.0-flash",
                    weight=0.1,
                    temperature=0.8,
                    max_tokens=4096,
                ),
            ],
            evaluator_models=[
                LLMModelConfig(
                    name="o1-preview",
                    weight=1.0,
                    temperature=0.2,
                    reasoning_effort="high",
                ),
            ],
        ),
    )

    print("Evolution models:")
    for model in config.llm.models:
        print(f"  - {model.name} (weight: {model.weight})")
    print()

    print("Evaluator models:")
    for model in config.llm.evaluator_models:
        print(f"  - {model.name} (weight: {model.weight})")
    print()


# ============================================================================
# Example 12: Advanced Feature Dimensions
# ============================================================================

def example_advanced_features():
    """Configure advanced feature dimensions for MAP-Elites"""
    print("Example 12: Advanced Feature Dimensions")
    print("-" * 80)

    config = UnifiedEvolutionConfig(
        evolution_mode="qd",
        qd=QDConfig(
            enable_map_elites=True,
            grid_resolution=25,
            grid_dimensions=["performance", "efficiency", "robustness"],
            adaptive_grid=True,
            use_feature_learning=True,
            feature_learning_rate=0.001,
        ),
        database=DatabaseConfig(
            feature_dimensions=["performance", "efficiency", "robustness"],
            feature_bins={
                "performance": 25,
                "efficiency": 20,
                "robustness": 15,
            },
            feature_scaling_method="standard",
        ),
    )

    print("Feature dimensions:")
    for dim, bins in config.database.feature_bins.items():
        print(f"  - {dim}: {bins} bins")
    print()

    print(f"Grid resolution: {config.qd.grid_resolution}")
    print(f"Feature learning: {config.qd.use_feature_learning}")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("UNIFIED CONFIGURATION SYSTEM - USAGE EXAMPLES")
    print("=" * 80)
    print()

    # Run all examples
    example_basic_openevolve()
    example_multi_objective()
    example_quality_diversity()
    example_pes_configuration()
    example_adversarial_evolution()
    example_hybrid_mode()
    example_domain_presets()
    example_config_mapping()
    example_serialization()
    example_validation()
    example_custom_llm_ensemble()
    example_advanced_features()

    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
