#!/usr/bin/env python3
"""
LeanAide Configuration Module Demo

This script demonstrates the key features of the LeanAide configuration system.
"""

import json
from leanaide_config import (
    load_leanaide_config,
    get_leanaide_config,
    get_leanaide_config_summary,
    ValidationError
)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def demo_default_config():
    """Demonstrate loading configuration with defaults."""
    print_section("1. Loading Configuration with Defaults")

    config = load_leanaide_config()

    print(f"Server URL: {config.server.get_base_url()}")
    print(f"Auto-verification: {config.verification.enable_auto}")
    print(f"Complexity threshold: {config.verification.complexity_threshold}")
    print(f"Caching enabled: {config.cache.enable}")
    print(f"Cache TTL: {config.cache.ttl} seconds ({config.cache.ttl // 3600} hours)")
    print(f"Stage 3C integration: {config.workflow.stage_3c_enabled}")
    print(f"Stage 5 integration: {config.workflow.stage_5_enabled}")
    print(f"Worker threads: {config.performance.worker_threads}")
    print(f"Environment: {config.environment}")


def demo_python_overrides():
    """Demonstrate overriding configuration via Python API."""
    print_section("2. Overriding Configuration via Python API")

    # Clear global config
    import leanaide_config
    leanaide_config._leanaide_config = None

    config = load_leanaide_config(
        server__host="custom.example.com",
        server__port=9090,
        verification__complexity_threshold=75,
        verification__parallel_verifications=8,
        cache__ttl=7200,
        workflow__failure_action="error",
        performance__worker_threads=8
    )

    print(f"Custom server: {config.server.get_base_url()}")
    print(f"Custom threshold: {config.verification.complexity_threshold}")
    print(f"Parallel verifications: {config.verification.parallel_verifications}")
    print(f"Custom cache TTL: {config.cache.ttl} seconds")
    print(f"Failure action: {config.workflow.failure_action}")
    print(f"Worker threads: {config.performance.worker_threads}")


def demo_validation():
    """Demonstrate configuration validation."""
    print_section("3. Configuration Validation")

    # Valid configuration
    try:
        config = load_leanaide_config(server__port=8080)
        print("[OK] Valid configuration accepted")
    except ValidationError as e:
        print(f"[ERROR] Unexpected error: {e}")

    # Invalid port
    try:
        config = load_leanaide_config(server__port=99999)
        print("[ERROR] Invalid port should have been rejected")
    except ValidationError as e:
        print(f"[OK] Invalid port correctly rejected: {str(e).split(':')[0]}")

    # Invalid verification strategy
    try:
        config = load_leanaide_config(verification__verification_strategy="invalid")
        print("[ERROR] Invalid strategy should have been rejected")
    except ValidationError as e:
        print(f"[OK] Invalid strategy correctly rejected: {str(e).split(':')[0]}")


def demo_config_summary():
    """Demonstrate getting configuration summary."""
    print_section("4. Configuration Summary (Safe for Logging)")

    config = load_leanaide_config(
        server__host="production.example.com",
        server__use_ssl=True
    )

    summary = get_leanaide_config_summary()
    print(json.dumps(summary, indent=2))


def demo_config_dict():
    """Demonstrate converting configuration to dictionary."""
    print_section("5. Configuration as Dictionary")

    config = load_leanaide_config()

    config_dict = config.to_dict()
    print(f"Number of sections: {len(config_dict)}")
    print(f"Sections: {', '.join(config_dict.keys())}")

    # Show one section in detail
    print("\nServer configuration:")
    for key, value in config_dict['server'].items():
        print(f"  {key}: {value}")


def demo_development_vs_production():
    """Demonstrate development vs production configurations."""
    print_section("6. Development vs Production Configurations")

    # Clear global config
    import leanaide_config
    leanaide_config._leanaide_config = None

    # Development
    dev_config = load_leanaide_config(
        environment="development",
        server__host="localhost",
        server__port=8080,
        verification__strict_mode=False,
        logging__level="DEBUG"
    )

    print("Development Configuration:")
    print(f"  Environment: {dev_config.environment}")
    print(f"  Server: {dev_config.server.host}:{dev_config.server.port}")
    print(f"  Strict mode: {dev_config.verification.strict_mode}")
    print(f"  Log level: {dev_config.logging.level}")

    # Clear global config for production
    leanaide_config._leanaide_config = None

    # Production
    prod_config = load_leanaide_config(
        environment="production",
        server__host="leanaide.prod.example.com",
        server__port=443,
        server__use_ssl=True,
        verification__strict_mode=True,
        logging__level="WARNING",
        security__enable_sandboxing=True
    )

    print("\nProduction Configuration:")
    print(f"  Environment: {prod_config.environment}")
    print(f"  Server: {prod_config.server.host}:{prod_config.server.port}")
    print(f"  SSL: {prod_config.server.use_ssl}")
    print(f"  Strict mode: {prod_config.verification.strict_mode}")
    print(f"  Log level: {prod_config.logging.level}")
    print(f"  Sandboxing: {prod_config.security.enable_sandboxing}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  LeanAide Configuration Module - Feature Demonstration")
    print("=" * 70)

    demo_default_config()
    demo_python_overrides()
    demo_validation()
    demo_config_summary()
    demo_config_dict()
    demo_development_vs_production()

    print_section("Demonstration Complete")
    print("\nThe LeanAide configuration module is ready to use!")
    print("\nNext steps:")
    print("  1. Copy leanaide_config.example.yaml to leanaide_config.yaml")
    print("  2. Customize the configuration for your environment")
    print("  3. Import and use: from leanaide_config import load_leanaide_config")
    print()


if __name__ == "__main__":
    main()
