"""
RESE Framework - Configuration Usage Example

This module demonstrates how to use the configuration loader in your application.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from glue.adapters.rese_integration.config_loader import (
    load_config,
    get_config,
    ConfigurationError,
)


def example_basic_usage():
    """Example: Basic configuration loading and access."""
    print("=" * 60)
    print("Example 1: Basic Configuration Loading")
    print("=" * 60)

    try:
        # Load configuration (will read from environment variables)
        config = load_config()

        # Access configuration values
        print(f"\nEnvironment: {config.env}")
        print(f"Log Level: {config.log_level}")
        print(f"Phase I Timeout: {config.phase1_timeout_ms}ms")
        print(f"Phase III Iterations: {config.phase3_iterations}")
        print(f"Enable Metrics: {config.enable_metrics}")

        # Export as dictionary
        config_dict = config.to_dict()
        print(f"\nConfiguration exported as dict with {len(config_dict)} sections")

    except ConfigurationError as e:
        print(f"\n❌ Configuration Error: {e}")
        sys.exit(1)


def example_phase_configuration():
    """Example: Accessing phase-specific configuration."""
    print("\n" + "=" * 60)
    print("Example 2: Phase-Specific Configuration")
    print("=" * 60)

    config = get_config()

    print("\nPhase I (Epistemic Audit):")
    print(f"  Timeout: {config.phase1_timeout_ms}ms")
    print(f"  Max Assumptions: {config.phase1_max_assumptions}")
    print(f"  Min Confidence: {config.phase1_min_assumption_confidence}")
    print(f"  Enable Tacit Mining: {config.phase1_enable_tacit_mining}")
    print(f"  Enable Red Team: {config.phase1_enable_red_team}")

    print("\nPhase II (Isomorphic Mapping):")
    print(f"  Timeout: {config.phase2_timeout_ms}ms")
    print(f"  IMech Threshold: {config.phase2_imech_threshold}")
    print(f"  Max Target Domains: {config.phase2_max_target_domains}")
    print(f"  Search Depth: {config.phase2_search_depth}")

    print("\nPhase III (MCTS Search):")
    print(f"  Timeout: {config.phase3_timeout_ms}ms")
    print(f"  Iterations: {config.phase3_iterations}")
    print(f"  UCB1 C: {config.phase3_ucb1_c}")
    print(f"  Parallel Workers: {config.phase3_parallel_workers}")

    print("\nPhase IV (Architecture Assembly):")
    print(f"  Timeout: {config.phase4_timeout_ms}ms")
    print(f"  Beam Width: {config.phase4_beam_width}")
    print(f"  Validation Level: {config.phase4_validation_level}")
    print(f"  Integration Strategy: {config.phase4_integration_strategy}")


def example_external_services():
    """Example: External service configuration."""
    print("\n" + "=" * 60)
    print("Example 3: External Services Configuration")
    print("=" * 60)

    config = get_config()

    print("\nOpenAI:")
    print(f"  Model: {config.openai_model}")
    print(f"  API Key: {'*' * 20}{config.openai_api_key[-4:]}")  # Hide most of key

    print("\nRedis:")
    print(f"  URL: {config.redis_url}")
    print(f"  Key TTL: {config.redis_key_ttl}s")

    print("\nTelemetry:")
    print(f"  Metrics Enabled: {config.enable_metrics}")
    if config.enable_metrics:
        print(f"  Metrics Port: {config.metrics_port}")
    print(f"  Tracing Enabled: {config.enable_tracing}")
    if config.enable_tracing and config.jaeger_endpoint:
        print(f"  Jaeger Endpoint: {config.jaeger_endpoint}")


def example_failure_handling():
    """Example: Failure handling configuration."""
    print("\n" + "=" * 60)
    print("Example 4: Failure Handling Configuration")
    print("=" * 60)

    config = get_config()

    print("\nCircuit Breakers:")
    print(f"  Enabled: {config.enable_circuit_breakers}")
    print(f"  Reset Timeout: {config.circuit_breaker_reset_timeout_ms}ms")

    print("\nRetry Logic:")
    print(f"  Enabled: {config.enable_retry}")
    print(f"  Max Attempts: {config.max_retry_attempts}")
    print(f"  Base Delay: {config.retry_base_delay_ms}ms")

    print("\nDead Letter Queue:")
    print(f"  Enabled: {config.enable_dlq}")
    print(f"  Queue Name: {config.dlq_name}")


def example_environment_specific():
    """Example: Environment-specific configuration."""
    print("\n" + "=" * 60)
    print("Example 5: Environment-Specific Configuration")
    print("=" * 60)

    config = get_config()

    env = config.env
    print(f"\nCurrent Environment: {env}")

    if env == "development":
        print("  → Fast iteration, detailed logging, permissive validation")
        print(f"  → Log Level: {config.log_level}")
        print(f"  → Profiling Enabled: {config.enable_profiling}")
    elif env == "staging":
        print("  → Production-like with comprehensive testing")
        print(f"  → Log Level: {config.log_level}")
        print(f"  → Tracing Enabled: {config.enable_tracing}")
    elif env == "production":
        print("  → Maximum quality, resilience, and observability")
        print(f"  → Log Level: {config.log_level}")
        print(f"  → Circuit Breakers: {config.enable_circuit_breakers}")
        print(f"  → DLQ Enabled: {config.enable_dlq}")


def example_conditional_logic():
    """Example: Conditional logic based on configuration."""
    print("\n" + "=" * 60)
    print("Example 6: Conditional Logic Based on Configuration")
    print("=" * 60)

    config = get_config()

    print("\nFeature Flags:")

    # Phase I features
    if config.phase1_enable_tacit_mining:
        print("  ✓ Tacit assumption mining ENABLED")
    else:
        print("  ✗ Tacit assumption mining DISABLED")

    if config.phase1_enable_red_team:
        print("  ✓ Red team mode ENABLED")
    else:
        print("  ✗ Red team mode DISABLED")

    if config.phase1_enable_lean4_integration:
        print(f"  ✓ Lean4 integration ENABLED (path: {config.lean4_exec_path})")
    else:
        print("  ✗ Lean4 integration DISABLED")

    # Phase II features
    if config.phase2_enable_constraint_inversion:
        print("  ✓ Constraint inversion ENABLED")
    else:
        print("  ✗ Constraint inversion DISABLED")

    # Phase IV strategy
    strategy = config.phase4_integration_strategy
    if strategy == "conservative":
        print("  ✓ Integration strategy: CONSERVATIVE (100% confidence required)")
    elif strategy == "balanced":
        print("  ✓ Integration strategy: BALANCED (80%+ confidence required)")
    elif strategy == "aggressive":
        print("  ✓ Integration strategy: AGGRESSIVE (all viable components)")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("RESE Framework - Configuration Usage Examples")
    print("=" * 60)

    try:
        example_basic_usage()
        example_phase_configuration()
        example_external_services()
        example_failure_handling()
        example_environment_specific()
        example_conditional_logic()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except ConfigurationError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\n💡 Tip: Make sure you have set the required environment variables")
        print("   or created a .env file from .env.example")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
