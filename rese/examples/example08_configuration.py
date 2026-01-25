"""
Example 8: Configuration Management

This example demonstrates how to configure and customize RESE behavior.
"""

import sys
sys.path.insert(0, r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend')

from config import (
    RESEConfig,
    Phase1Config,
    Phase2Config,
    Phase3Config,
    Phase4Config,
    PipelineConfig,
    Environment,
    get_config
)
from pathlib import Path

def main():
    print("=" * 60)
    print("Example 8: Configuration Management")
    print("=" * 60)
    print()

    # Example 1: Load default configuration
    print("Example 1: Load Default Configuration")
    print("-" * 60)

    default_config = get_config()
    print(f"Environment: {default_config.environment}")
    print(f"Version: {default_config.version}")
    print()

    # Example 2: Create custom configuration
    print("Example 2: Create Custom Configuration")
    print("-" * 60)

    custom_config = RESEConfig(
        environment="development",
        phase1=Phase1Config(
            sce_max_constraints=5000,
            phi15_assumption_threshold=0.7,
            phi2_bias_threshold=0.6
        ),
        phase3=Phase3Config(
            gamma2_iterations=500,
            gamma2_parallel_agents=2
        )
    )

    print("Custom Configuration:")
    print(f"  Phase I max constraints: {custom_config.phase1.sce_max_constraints}")
    print(f"  Phase I assumption threshold: {custom_config.phase1.phi15_assumption_threshold}")
    print(f"  Phase III iterations: {custom_config.phase3.gamma2_iterations}")
    print()

    # Example 3: Environment-specific configuration
    print("Example 3: Environment-Specific Configuration")
    print("-" * 60)

    dev_config = RESEConfig().for_environment(Environment.DEVELOPMENT)
    prod_config = RESEConfig().for_environment(Environment.PRODUCTION)

    print("Development Configuration:")
    print(f"  API Debug: {dev_config.api.debug}")
    print(f"  Log Level: {dev_config.monitoring.log_level}")
    print()

    print("Production Configuration:")
    print(f"  API Debug: {prod_config.api.debug}")
    print(f"  Log Level: {prod_config.monitoring.log_level}")
    print()

    # Example 4: Save and load configuration
    print("Example 4: Save and Load Configuration")
    print("-" * 60)

    config_path = Path("C:/Users/mmeadow/Documents/OpenEvolve/Frontend/rese/examples/my_config.json")

    # Save configuration
    custom_config.save(config_path)
    print(f"Saved configuration to: {config_path}")
    print()

    # Load configuration
    loaded_config = RESEConfig.from_file(config_path)
    print(f"Loaded configuration:")
    print(f"  Environment: {loaded_config.environment}")
    print(f"  Phase I max constraints: {loaded_config.phase1.sce_max_constraints}")
    print()

    # Example 5: Update configuration
    print("Example 5: Update Configuration")
    print("-" * 60)

    config = get_config()

    print("Before Update:")
    print(f"  Phase III iterations: {config.phase3.gamma2_iterations}")

    # Update
    config.phase3.gamma2_iterations = 2000

    print("After Update:")
    print(f"  Phase III iterations: {config.phase3.gamma2_iterations}")
    print()

    # Example 6: View all configuration
    print("Example 6: Complete Configuration Overview")
    print("-" * 60)

    config_dict = custom_config.to_dict()

    print("Configuration Structure:")
    for key, value in config_dict.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey in list(value.keys())[:3]:  # Show first 3
                print(f"    - {subkey}")
            if len(value) > 3:
                print(f"    ... and {len(value) - 3} more")
        else:
            print(f"  {key}: {value}")
    print()

    # Example 7: Pipeline configuration
    print("Example 7: Pipeline Configuration")
    print("-" * 60)

    pipeline_config = PipelineConfig(
        enable_caching=True,
        cache_ttl_seconds=1800,
        max_retries=3,
        max_parallel_tasks=4,
        enable_monitoring=True
    )

    print("Pipeline Settings:")
    print(f"  Caching Enabled: {pipeline_config.enable_caching}")
    print(f"  Cache TTL: {pipeline_config.cache_ttl_seconds}s")
    print(f"  Max Retries: {pipeline_config.max_retries}")
    print(f"  Max Parallel Tasks: {pipeline_config.max_parallel_tasks}")
    print(f"  Monitoring Enabled: {pipeline_config.enable_monitoring}")
    print()

    # Example 8: Feature flags
    print("Example 8: Feature Flags")
    print("-" * 60)

    config_with_features = RESEConfig(
        feature_use_gpu=True,
        feature_distributed=True,
        feature_experimental=False
    )

    print("Feature Flags:")
    print(f"  Use GPU: {config_with_features.feature_use_gpu}")
    print(f"  Distributed: {config_with_features.feature_distributed}")
    print(f"  Experimental: {config_with_features.feature_experimental}")
    print()

    print("=" * 60)
    print("Example 8 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
