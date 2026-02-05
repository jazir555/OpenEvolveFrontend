"""
LoongFlow Graceful Fallback Example

This example demonstrates how to use OpenEvolve with the LoongFlow integration
and graceful fallback system. The system works seamlessly whether LoongFlow
is installed or not.
"""

import asyncio
from openevolve.integrations import (
    LoongFlowAdapter,
    LoongFlowChecker
)


async def example_1_check_availability():
    """Example 1: Check LoongFlow availability."""
    print("\n" + "=" * 70)
    print("Example 1: Checking LoongFlow Availability")
    print("=" * 70)

    # Check if LoongFlow is installed
    installed = LoongFlowChecker.is_installed()
    print(f"LoongFlow Installed: {installed}")

    # Get version if available
    version = LoongFlowChecker.get_version()
    print(f"LoongFlow Version: {version or 'N/A'}")

    # Check availability (with deep requirement check)
    available = LoongFlowChecker.is_available(requirement_check=True)
    print(f"LoongFlow Available: {available}")

    # Get comprehensive diagnostics
    diagnostics = LoongFlowChecker.get_diagnostics()
    print(f"\nDiagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")


async def example_2_default_configuration():
    """Example 2: Use default configuration (automatic fallback)."""
    print("\n" + "=" * 70)
    print("Example 2: Default Configuration")
    print("=" * 70)

    # Create adapter with default configuration
    # LoongFlow will be used if available, otherwise falls back to OpenEvolve
    config = {
        "max_iterations": 10,
        "population_size": 5,
    }

    adapter = LoongFlowAdapter(config)

    # Check status
    status = adapter.get_status()
    print(f"\nAdapter Status:")
    print(f"  Mode: {status['mode']}")
    print(f"  Using LoongFlow: {status['using_loongflow']}")
    print(f"  System: {status['capabilities']['system']}")

    # Run evolution (works seamlessly regardless of which system is used)
    result = await adapter.evolve(
        problem="Optimize function: f(x) = x^2",
        domain="math"
    )

    print(f"\nEvolution Result:")
    print(f"  System Used: {result['system_used']}")
    print(f"  Mode Used: {result['mode_used']}")
    print(f"  Best Fitness: {result['best_fitness']}")


async def example_3_explicit_openevolve_mode():
    """Example 3: Explicitly use OpenEvolve-only mode."""
    print("\n" + "=" * 70)
    print("Example 3: OpenEvolve-Only Mode")
    print("=" * 70)

    # Disable LoongFlow explicitly
    config = {
        "enable_loongflow": False,
        "mode": "qd",  # Quality-Diversity mode
        "max_iterations": 10,
        "show_messages": True  # Show user-friendly messages
    }

    adapter = LoongFlowAdapter(config)

    # Run evolution
    result = await adapter.evolve(
        problem="Find diverse solutions for sorting",
        domain="code"
    )

    print(f"\nResult:")
    print(f"  System: {result['system_used']}")
    print(f"  Mode: {result['mode_used']}")


async def example_4_different_openevolve_modes():
    """Example 4: Try different OpenEvolve modes."""
    print("\n" + "=" * 70)
    print("Example 4: Different OpenEvolve Modes")
    print("=" * 70)

    modes = ["standard", "qd", "mo", "adversarial"]

    for mode in modes:
        print(f"\n--- Testing {mode.upper()} mode ---")

        config = {
            "enable_loongflow": False,
            "mode": mode,
            "max_iterations": 5,
            "show_messages": False
        }

        adapter = LoongFlowAdapter(config)
        status = adapter.get_status()

        print(f"  Evolution Mode: {mode}")
        print(f"  System: {status['capabilities']['system']}")
        print(f"  Supports QD: {status['capabilities']['supports_qd']}")
        print(f"  Supports MO: {status['capabilities']['supports_mo']}")


async def example_5_production_ready_configuration():
    """Example 5: Production-ready configuration with proper settings."""
    print("\n" + "=" * 70)
    print("Example 5: Production-Ready Configuration")
    print("=" * 70)

    # Production configuration with all settings
    config = {
        # LoongFlow settings
        "enable_loongflow": True,  # Try to use LoongFlow if available
        "require_loongflow": False,  # Don't fail if LoongFlow unavailable

        # OpenEvolve settings (for fallback)
        "mode": "standard",  # Fallback mode
        "max_iterations": 100,
        "population_size": 20,

        # Feature flags
        "enable_planning": True,
        "enable_memory": True,

        # User experience
        "show_messages": True,  # Show helpful status messages

        # LLM configuration
        "llm_config": {
            "model": "gpt-4",
            "temperature": 0.7
        },

        # Execution settings
        "timeout": 300,
    }

    adapter = LoongFlowAdapter(config)

    # Print detailed status
    print("\nAdapter Configuration:")
    adapter.print_status()

    # Run a sample evolution
    result = await adapter.evolve(
        problem="Optimize a trading algorithm",
        domain="scientific"
    )

    print(f"\nEvolution Complete:")
    print(f"  System Used: {result['system_used']}")
    print(f"  Best Fitness: {result['best_fitness']:.4f}")
    print(f"  Iterations: {result['iterations_performed']}")


async def example_6_error_handling():
    """Example 6: Proper error handling and recovery."""
    print("\n" + "=" * 70)
    print("Example 6: Error Handling and Recovery")
    print("=" * 70)

    # Try with strict requirement (will fail if LoongFlow not installed)
    if not LoongFlowChecker.is_installed():
        print("\nAttempting strict LoongFlow requirement...")

        config = {
            "enable_loongflow": True,
            "require_loongflow": True  # This will fail if LoongFlow not installed
        }

        try:
            adapter = LoongFlowAdapter(config)
            print("  [FAIL] Should have failed but didn't!")
        except RuntimeError as e:
            print(f"  [OK] Correctly failed with: {e}")

    # Show proper configuration with fallback
    print("\nUsing proper fallback configuration...")
    config = {
        "enable_loongflow": True,
        "require_loongflow": False,  # Allow fallback
        "show_messages": False
    }

    adapter = LoongFlowAdapter(config)
    print(f"  [OK] Successfully initialized in {adapter.mode} mode")


async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("LOONGFLOW GRACEFUL FALLBACK EXAMPLES")
    print("=" * 70)
    print("\nThis demonstrates OpenEvolve's seamless integration with LoongFlow")
    print("and automatic fallback to OpenEvolve-native mode when needed.")

    # Run examples
    await example_1_check_availability()
    await example_2_default_configuration()
    await example_3_explicit_openevolve_mode()
    await example_4_different_openevolve_modes()
    await example_5_production_ready_configuration()
    await example_6_error_handling()

    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. The system works seamlessly with or without LoongFlow")
    print("  2. Automatic fallback is transparent and preserves functionality")
    print("  3. User can explicitly disable LoongFlow if desired")
    print("  4. All OpenEvolve modes are available in fallback mode")
    print("  5. Production-ready configuration handles all scenarios")
    print("  6. Error handling is robust and user-friendly")
    print("\n[OK] OpenEvolve is ready for production use!")


if __name__ == "__main__":
    asyncio.run(main())
