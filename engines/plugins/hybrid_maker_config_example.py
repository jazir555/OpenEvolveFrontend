"""
Hybrid MAKER Configuration System - Usage Examples

This file demonstrates how to use the hybrid_maker_config system
for managing hybrid MAKER strategy configurations.
"""
from __future__ import annotations


from hybrid_maker_config import (
    HybridMakerConfig,
    HybridMakerConfigPreset,
    StrategyType,
    create_config_from_preset,
    export_config_summary,
    get_available_presets,
)


def example_1_using_presets():
    """Example 1: Using predefined presets"""
    print("=" * 60)
    print("Example 1: Using Predefined Presets")
    print("=" * 60)

    # Show available presets
    presets = get_available_presets()
    print(f"\nAvailable presets: {', '.join(presets)}")

    # Create a fast configuration
    fast_config = HybridMakerConfigPreset.fast()
    print(f"\nFast Configuration:")
    print(f"  - MAKER k_min: {fast_config.maker_config.k_min}")
    print(f"  - MCTS simulations: {fast_config.mcts_config.num_simulations}")
    print(f"  - Evolution population: {fast_config.evolution_config.population_size}")
    print(f"  - Global timeout: {fast_config.global_timeout}s")

    # Create a balanced configuration
    balanced_config = HybridMakerConfigPreset.balanced()
    print(f"\nBalanced Configuration:")
    print(f"  - MAKER k_min: {balanced_config.maker_config.k_min}")
    print(f"  - MCTS simulations: {balanced_config.mcts_config.num_simulations}")
    print(f"  - Evolution population: {balanced_config.evolution_config.population_size}")
    print(f"  - Global timeout: {balanced_config.global_timeout}s")

    # Create a thorough configuration
    thorough_config = HybridMakerConfigPreset.thorough()
    print(f"\nThorough Configuration:")
    print(f"  - MAKER k_min: {thorough_config.maker_config.k_min}")
    print(f"  - MCTS simulations: {thorough_config.mcts_config.num_simulations}")
    print(f"  - Evolution population: {thorough_config.evolution_config.population_size}")
    print(f"  - Global timeout: {thorough_config.global_timeout}s")


def example_2_custom_configuration():
    """Example 2: Creating custom configuration"""
    print("\n" + "=" * 60)
    print("Example 2: Creating Custom Configuration")
    print("=" * 60)

    # Create a custom configuration
    config = HybridMakerConfig(
        config_name="custom_research",
        description="Custom research configuration",
        tags=["research", "experiment"],
        default_strategy=StrategyType.MAKER,
        global_timeout=7200,  # 2 hours
    )

    # Customize MAKER settings
    config.maker_config.k_min = 4
    config.maker_config.k_max = 7
    config.maker_config.min_agents = 5
    config.maker_config.max_agents = 12

    # Customize MCTS settings
    config.mcts_config.num_simulations = 1500
    config.mcts_config.exploration_constant = 1.5
    config.mcts_config.num_workers = 6

    # Customize Evolution settings
    config.evolution_config.population_size = 100
    config.evolution_config.generations = 50
    config.evolution_config.mutation_rate = 0.15

    # Enable specific strategies
    config.strategy_profiles["maker"].enabled = True
    config.strategy_profiles["maker"].performance_weight = 1.0
    config.strategy_profiles["maker"].priority = 1

    config.strategy_profiles["mcts"].enabled = True
    config.strategy_profiles["mcts"].performance_weight = 0.8
    config.strategy_profiles["mcts"].priority = 2

    config.strategy_profiles["evolution"].enabled = True
    config.strategy_profiles["evolution"].performance_weight = 0.6
    config.strategy_profiles["evolution"].priority = 3

    # Disable other strategies
    config.strategy_profiles["leanaide"].enabled = False
    config.strategy_profiles["mdap"].enabled = False

    # Validate configuration
    valid, errors = config.validate()
    if valid:
        print("\nCustom configuration is valid!")
    else:
        print(f"\nConfiguration has errors: {errors}")

    print(f"\nCustom Configuration:")
    print(f"  - Name: {config.config_name}")
    print(f"  - Description: {config.description}")
    print(f"  - Tags: {', '.join(config.tags)}")
    print(f"  - Default strategy: {config.default_strategy.value}")


def example_3_runtime_estimation():
    """Example 3: Estimating runtime and resource usage"""
    print("\n" + "=" * 60)
    print("Example 3: Runtime and Resource Estimation")
    print("=" * 60)

    config = HybridMakerConfigPreset.balanced()

    # Estimate runtime for all strategies
    runtime = config.estimate_runtime()
    print("\nEstimated runtime per strategy:")
    for strategy, time in runtime.items():
        print(f"  - {strategy}: ~{time:.1f}s")

    # Estimate runtime for single strategy
    maker_runtime = config.estimate_runtime(StrategyType.MAKER)
    print(f"\nMAKER-only runtime: ~{maker_runtime['maker']:.1f}s")

    # Estimate resource usage
    resources = config.estimate_resource_usage()
    print("\nEstimated resource usage:")
    for strategy, usage in resources.items():
        print(f"  - {strategy}:")
        print(f"      CPU: {usage['cpu']:.1f} cores")
        print(f"      Memory: {usage['memory_mb']:.0f} MB")


def example_4_serialization():
    """Example 4: Saving and loading configurations"""
    print("\n" + "=" * 60)
    print("Example 4: Saving and Loading Configurations")
    print("=" * 60)

    # Create and save configuration
    config = HybridMakerConfigPreset.adaptive()
    config.config_name = "my_adaptive_config"
    config.description = "My custom adaptive configuration"

    # Save to YAML
    yaml_path = "example_config.yaml"
    success = config.save_to_file(yaml_path, format="yaml")
    if success:
        print(f"\nConfiguration saved to {yaml_path}")

    # Save to JSON
    json_path = "example_config.json"
    success = config.save_to_file(json_path, format="json")
    if success:
        print(f"Configuration saved to {json_path}")

    # Load from file
    loaded_config = HybridMakerConfig.load_from_file(yaml_path)
    if loaded_config:
        print(f"\nConfiguration loaded from {yaml_path}")
        print(f"  - Name: {loaded_config.config_name}")
        print(f"  - Description: {loaded_config.description}")
        print(f"  - Adaptive selection: {loaded_config.adaptive_config.enable_adaptive_selection}")


def example_5_strategy_profiles():
    """Example 5: Working with strategy profiles"""
    print("\n" + "=" * 60)
    print("Example 5: Strategy Profiles")
    print("=" * 60)

    config = HybridMakerConfig()

    print("\nDefault strategy profiles:")
    for name, profile in config.strategy_profiles.items():
        status = "enabled" if profile.enabled else "disabled"
        print(f"  - {name}:")
        print(f"      Status: {status}")
        print(f"      Weight: {profile.performance_weight}")
        print(f"      Priority: {profile.priority}")
        if profile.description:
            print(f"      Description: {profile.description}")

    # Customize a profile
    print("\nCustomizing MAKER profile...")
    config.strategy_profiles["maker"].performance_weight = 1.5
    config.strategy_profiles["maker"].priority = 1
    config.strategy_profiles["maker"].max_time_seconds = 600
    config.strategy_profiles["maker"].parallel_instances = 2

    print(f"  - New weight: {config.strategy_profiles['maker'].performance_weight}")
    print(f"  - New priority: {config.strategy_profiles['maker'].priority}")
    print(f"  - Max time: {config.strategy_profiles['maker'].max_time_seconds}s")
    print(f"  - Parallel instances: {config.strategy_profiles['maker'].parallel_instances}")


def example_6_focused_configurations():
    """Example 6: Using focused configurations"""
    print("\n" + "=" * 60)
    print("Example 6: Focused Configurations")
    print("=" * 60)

    # LeanAide focused
    leanaide_config = HybridMakerConfigPreset.leanaide_focused()
    print("\nLeanAide-focused configuration:")
    print(f"  - Default strategy: {leanaide_config.default_strategy.value}")
    print(f"  - LeanAide timeout: {leanaide_config.leanaide_config.timeout}s")
    print(f"  - Strict verification: {leanaide_config.leanaide_config.strict_verification}")
    enabled_strategies = [name for name, profile in leanaide_config.strategy_profiles.items()
                          if profile.enabled]
    print(f"  - Enabled strategies: {', '.join(enabled_strategies)}")

    # MAKER focused
    maker_config = HybridMakerConfigPreset.maker_focused()
    print("\nMAKER-focused configuration:")
    print(f"  - Default strategy: {maker_config.default_strategy.value}")
    print(f"  - MAKER k_min: {maker_config.maker_config.k_min}")
    print(f"  - MAKER k_max: {maker_config.maker_config.k_max}")
    enabled_strategies = [name for name, profile in maker_config.strategy_profiles.items()
                          if profile.enabled]
    print(f"  - Enabled strategies: {', '.join(enabled_strategies)}")

    # Adaptive
    adaptive_config = HybridMakerConfigPreset.adaptive()
    print("\nAdaptive configuration:")
    print(f"  - Default strategy: {adaptive_config.default_strategy.value}")
    print(f"  - Adaptive selection: {adaptive_config.adaptive_config.enable_adaptive_selection}")
    print(f"  - Resource-aware: {adaptive_config.adaptive_config.resource_aware}")
    print(f"  - Strategy combination: {adaptive_config.adaptive_config.allow_strategy_combination}")


def example_7_performance_thresholds():
    """Example 7: Configuring performance thresholds"""
    print("\n" + "=" * 60)
    print("Example 7: Performance Thresholds")
    print("=" * 60)

    config = HybridMakerConfig()

    print("\nDefault performance thresholds:")
    print(f"  - Fast: {config.performance_thresholds.fast_time_threshold}s @ "
          f"{config.performance_thresholds.fast_quality_threshold:.0%} quality")
    print(f"  - Balanced: {config.performance_thresholds.balanced_time_threshold}s @ "
          f"{config.performance_thresholds.balanced_quality_threshold:.0%} quality")
    print(f"  - Thorough: {config.performance_thresholds.thorough_time_threshold}s @ "
          f"{config.performance_thresholds.thorough_quality_threshold:.0%} quality")

    # Customize thresholds
    config.performance_thresholds.fast_time_threshold = 120
    config.performance_thresholds.balanced_time_threshold = 600
    config.performance_thresholds.thorough_time_threshold = 3600

    config.performance_thresholds.fast_quality_threshold = 0.7
    config.performance_thresholds.balanced_quality_threshold = 0.85
    config.performance_thresholds.thorough_quality_threshold = 0.98

    print("\nCustomized performance thresholds:")
    print(f"  - Fast: {config.performance_thresholds.fast_time_threshold}s @ "
          f"{config.performance_thresholds.fast_quality_threshold:.0%} quality")
    print(f"  - Balanced: {config.performance_thresholds.balanced_time_threshold}s @ "
          f"{config.performance_thresholds.balanced_quality_threshold:.0%} quality")
    print(f"  - Thorough: {config.performance_thresholds.thorough_time_threshold}s @ "
          f"{config.performance_thresholds.thorough_quality_threshold:.0%} quality")


def example_8_configuration_summary():
    """Example 8: Exporting configuration summary"""
    print("\n" + "=" * 60)
    print("Example 8: Configuration Summary Export")
    print("=" * 60)

    config = HybridMakerConfig(
        config_name="production",
        description="Production-ready configuration",
        tags=["production", "optimized"],
    )

    # Customize for production
    config.enable_parallel_strategies = False
    config.global_timeout = 1800
    config.checkpoint_enabled = True
    config.log_level = "WARNING"

    config.maker_config.k_min = 3
    config.maker_config.k_max = 5
    config.maker_config.timeout_seconds = 30

    # Export summary
    summary = export_config_summary(config)
    print(summary)


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("HYBRID MAKER CONFIGURATION SYSTEM - USAGE EXAMPLES")
    print("=" * 60)

    example_1_using_presets()
    example_2_custom_configuration()
    example_3_runtime_estimation()
    example_4_serialization()
    example_5_strategy_profiles()
    example_6_focused_configurations()
    example_7_performance_thresholds()
    example_8_configuration_summary()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
