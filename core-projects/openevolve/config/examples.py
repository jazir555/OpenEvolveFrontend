#!/usr/bin/env python3
"""
OpenEvolve Configuration System - Usage Examples

Demonstrates various features of the configuration system.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openevolve.config import ConfigManager


def example_1_basic_usage():
    """Example 1: Basic configuration loading"""
    print("=" * 60)
    print("Example 1: Basic Configuration Loading")
    print("=" * 60)

    manager = ConfigManager()

    # Load with default settings
    config = manager.load_config()

    print(f"Loaded {len(config)} parameters")
    print(f"Max iterations: {config.get('max_iterations', 'not set')}")
    print(f"Temperature: {config.get('temperature', 'not set')}")
    print()


def example_2_using_profiles():
    """Example 2: Using built-in profiles"""
    print("=" * 60)
    print("Example 2: Using Built-in Profiles")
    print("=" * 60)

    manager = ConfigManager()

    # List available profiles
    profiles = manager.list_profiles()
    print(f"Available profiles: {', '.join(profiles)}")
    print()

    # Load development profile
    dev_config = manager.load_config(profile='development')
    print("Development Profile:")
    print(f"  Max iterations: {dev_config['max_iterations']}")
    print(f"  Population size: {dev_config['population_size']}")
    print(f"  Log level: {dev_config['log_level']}")
    print()

    # Load production profile
    prod_config = manager.load_config(profile='production')
    print("Production Profile:")
    print(f"  Max iterations: {prod_config['max_iterations']}")
    print(f"  Population size: {prod_config['population_size']}")
    print(f"  Log level: {prod_config['log_level']}")
    print()


def example_3_environment_variables():
    """Example 3: Using environment variables"""
    print("=" * 60)
    print("Example 3: Environment Variables")
    print("=" * 60)

    import os

    # Set some environment variables
    os.environ['EVOLVE_MAX_ITERATIONS'] = '250'
    os.environ['EVOLVE_TEMPERATURE'] = '0.9'
    os.environ['EVOLVE_ENABLE_PLANNING'] = 'true'

    manager = ConfigManager()

    # Load with environment override
    config = manager.load_config(
        profile='development',
        env_override=True
    )

    print("Configuration with environment overrides:")
    print(f"  Max iterations: {config['max_iterations']} (from env)")
    print(f"  Temperature: {config['temperature']} (from env)")
    print(f"  Enable planning: {config['enable_planning']} (from env)")

    # Cleanup
    del os.environ['EVOLVE_MAX_ITERATIONS']
    del os.environ['EVOLVE_TEMPERATURE']
    del os.environ['EVOLVE_ENABLE_PLANNING']
    print()


def example_4_runtime_overrides():
    """Example 4: Runtime parameter overrides"""
    print("=" * 60)
    print("Example 4: Runtime Overrides")
    print("=" * 60)

    manager = ConfigManager()

    # Load with runtime overrides
    config = manager.load_config(
        profile='development',
        runtime_overrides={
            'max_iterations': 500,
            'custom_param': 'my_value'
        }
    )

    print("Configuration with runtime overrides:")
    print(f"  Max iterations: {config['max_iterations']}")
    print(f"  Custom param: {config['custom_param']}")
    print()


def example_5_hierarchical_priority():
    """Example 5: Demonstrating priority hierarchy"""
    print("=" * 60)
    print("Example 5: Configuration Priority Hierarchy")
    print("=" * 60)

    import os
    os.environ['EVOLVE_MAX_ITERATIONS'] = '150'

    manager = ConfigManager()

    # Demonstrates priority: runtime > env > profile
    config = manager.load_config(
        profile='development',  # Sets max_iterations=20
        env_override=True,      # Sets max_iterations=150
        runtime_overrides={     # Sets max_iterations=300 (highest priority)
            'max_iterations': 300
        }
    )

    print("Priority demonstration (max_iterations):")
    print(f"  Profile (development): 20")
    print(f"  Environment variable: 150")
    print(f"  Runtime override: 300")
    print(f"  Final value: {config['max_iterations']} (runtime wins)")
    print()

    # Cleanup
    del os.environ['EVOLVE_MAX_ITERATIONS']


def example_6_saving_config():
    """Example 6: Saving configuration to file"""
    print("=" * 60)
    print("Example 6: Saving Configuration")
    print("=" * 60)

    import tempfile
    import os

    manager = ConfigManager()

    # Load configuration
    config = manager.load_config(profile='development')

    # Save to JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_file = f.name

    try:
        manager.save_config(config, json_file, format='json')
        print(f"Saved configuration to: {json_file}")

        # Read and display
        with open(json_file, 'r') as f:
            content = f.read()
        print(f"File size: {len(content)} bytes")
    finally:
        if os.path.exists(json_file):
            os.remove(json_file)

    print()


def example_7_validation():
    """Example 7: Configuration validation"""
    print("=" * 60)
    print("Example 7: Configuration Validation")
    print("=" * 60)

    manager = ConfigManager()

    # Valid configuration
    valid_config = {
        'max_iterations': 50,
        'temperature': 0.7,
        'enable_planning': True
    }

    result = manager.validate_config(valid_config)
    print(f"Valid config: {result.is_valid}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    print()

    # Invalid configuration (temperature out of range)
    invalid_config = {
        'temperature': 5.0  # Out of range [0.0, 2.0]
    }

    try:
        result = manager.validate_config(invalid_config)
    except Exception as e:
        print(f"Validation error: {e}")
    print()


def example_8_parameter_info():
    """Example 8: Getting parameter information"""
    print("=" * 60)
    print("Example 8: Parameter Information")
    print("=" * 60)

    manager = ConfigManager()

    # Get info about specific parameter
    info = manager.get_parameter_info('max_iterations')
    print(f"Parameter: {info['name']}")
    print(f"Environment variable: {info['env_var']}")
    print(f"Type: {info['type']}")
    if 'range' in info:
        print(f"Valid range: {info['range']}")
    print()

    # List all parameters
    all_params = manager.list_all_parameters()
    print(f"Total parameters: {len(all_params)}")
    print(f"First 10: {', '.join(all_params[:10])}")
    print()


def example_9_comparing_configs():
    """Example 9: Comparing configurations"""
    print("=" * 60)
    print("Example 9: Comparing Configurations")
    print("=" * 60)

    manager = ConfigManager()

    # Load two profiles
    dev_config = manager.load_config(profile='development')
    prod_config = manager.load_config(profile='production')

    # Compare
    diff = manager.compare_configs(dev_config, prod_config)

    print("Differences between development and production:")
    print(f"  Parameters only in dev: {len(diff['only_in_first'])}")
    print(f"  Parameters only in prod: {len(diff['only_in_second'])}")
    print(f"  Parameters with different values: {len(diff['different_values'])}")
    print()

    # Show some differences
    if 'max_iterations' in diff['different_values']:
        diff_info = diff['different_values']['max_iterations']
        print(f"  max_iterations:")
        print(f"    dev: {diff_info['config1']}")
        print(f"    prod: {diff_info['config2']}")
    print()


def example_10_custom_profile():
    """Example 10: Creating custom profile"""
    print("=" * 60)
    print("Example 10: Creating Custom Profile")
    print("=" * 60)

    import tempfile
    import shutil

    # Create temporary profile directory
    profile_dir = tempfile.mkdtemp()

    try:
        manager = ConfigManager()

        # Create custom profile
        custom_config = manager.create_profile(
            name='my_custom',
            base='development',
            overrides={
                'max_iterations': 15,
                'experiment_name': 'my_experiment'
            }
        )

        print("Created custom profile:")
        print(f"  Max iterations: {custom_config['max_iterations']}")
        print(f"  Experiment name: {custom_config['experiment_name']}")
        print(f"  Still has dev defaults: {custom_config['log_level']}")
        print()

    finally:
        # Cleanup
        shutil.rmtree(profile_dir, ignore_errors=True)


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "OpenEvolve Configuration System Examples" + " " * 10 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    example_1_basic_usage()
    example_2_using_profiles()
    example_3_environment_variables()
    example_4_runtime_overrides()
    example_5_hierarchical_priority()
    example_6_saving_config()
    example_7_validation()
    example_8_parameter_info()
    example_9_comparing_configs()
    example_10_custom_profile()

    print("=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
