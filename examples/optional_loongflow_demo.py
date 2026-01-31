"""
Optional LoongFlow Configuration Examples

This script demonstrates the different ways to configure LoongFlow as optional
in the unified evolution system.

Author: AI Architecture Team
Date: 2026-01-30
"""

import sys
sys.path.insert(0, 'openevolve')

from unified.config import (
    UnifiedEvolutionConfig,
    EvolutionMode,
    DomainType,
    PESConfig,
    QDConfig
)


def example_1_default_configuration():
    """
    Example 1: Default Configuration

    By default, LoongFlow is enabled with fallback allowed.
    """
    print("\n" + "="*80)
    print("Example 1: Default Configuration")
    print("="*80)

    config = UnifiedEvolutionConfig(
        domain=DomainType.FINANCE,
        max_iterations=1000
    )

    print(f"LoongFlow Enabled: {config.enable_loongflow}")
    print(f"Fallback Enabled: {config.loongflow_fallback_enabled}")
    print(f"Require LoongFlow: {config.require_loongflow}")
    print(f"Should Use LoongFlow: {config.should_use_loongflow()}")
    print()


def example_2_explicitly_disable_loongflow():
    """
    Example 2: Explicitly Disable LoongFlow

    Disable LoongFlow and use only OpenEvolve modes.
    """
    print("\n" + "="*80)
    print("Example 2: Explicitly Disable LoongFlow")
    print("="*80)

    config = UnifiedEvolutionConfig(
        enable_loongflow=False,
        domain=DomainType.TRADING,
        evolution_mode=EvolutionMode.QD,
        qd=QDConfig(enabled=True)
    )

    print(f"LoongFlow Enabled: {config.enable_loongflow}")
    print(f"Should Use LoongFlow: {config.should_use_loongflow()}")
    print(f"Evolution Mode: {config.evolution_mode}")
    print()


def example_3_openevolve_only_convenience():
    """
    Example 3: Using openevolve_only() Convenience Method

    Quick way to create OpenEvolve-only configuration.
    """
    print("\n" + "="*80)
    print("Example 3: OpenEvolve Only (Convenience Method)")
    print("="*80)

    config = UnifiedEvolutionConfig.openevolve_only(
        max_iterations=500,
        domain=DomainType.SCIENCE
    )

    print(f"LoongFlow Enabled: {config.enable_loongflow}")
    print(f"Fallback Enabled: {config.loongflow_fallback_enabled}")
    print(f"Require LoongFlow: {config.require_loongflow}")
    print(f"Max Iterations: {config.max_iterations}")
    print(f"Domain: {config.domain}")
    print()


def example_4_require_loongflow():
    """
    Example 4: Require LoongFlow (No Fallback)

    Require LoongFlow to be available. Will raise error if not installed.
    """
    print("\n" + "="*80)
    print("Example 4: Require LoongFlow (No Fallback)")
    print("="*80)

    try:
        config = UnifiedEvolutionConfig.loongflow_required(
            domain=DomainType.ENGINEERING,
            evolution_mode=EvolutionMode.PES,
            pes=PESConfig(enabled=True)
        )

        print(f"LoongFlow Enabled: {config.enable_loongflow}")
        print(f"Require LoongFlow: {config.require_loongflow}")
        print(f"Fallback Enabled: {config.loongflow_fallback_enabled}")
        print(f"Evolution Mode: {config.evolution_mode}")
        print(f"Should Use LoongFlow: {config.should_use_loongflow()}")

    except RuntimeError as e:
        print(f"Error: {e}")
        print("LoongFlow is not installed. Install it to use this configuration.")

    print()


def example_5_enable_with_fallback():
    """
    Example 5: Enable LoongFlow with Fallback

    Try to use LoongFlow, but gracefully fallback to OpenEvolve if unavailable.
    """
    print("\n" + "="*80)
    print("Example 5: Enable LoongFlow with Graceful Fallback")
    print("="*80)

    config = UnifiedEvolutionConfig(
        enable_loongflow=True,
        loongflow_fallback_enabled=True,
        domain=DomainType.PHARMA
    )

    print(f"LoongFlow Enabled: {config.enable_loongflow}")
    print(f"Fallback Enabled: {config.loongflow_fallback_enabled}")

    should_use = config.should_use_loongflow()
    print(f"Should Use LoongFlow: {should_use}")

    if not should_use:
        print("Note: LoongFlow not available, will use OpenEvolve modes")
    else:
        print("Note: LoongFlow is available and will be used")

    print()


def example_6_disable_fallback():
    """
    Example 6: Enable LoongFlow but Disable Fallback

    Use LoongFlow if available, but don't fallback to OpenEvolve.
    """
    print("\n" + "="*80)
    print("Example 6: Enable LoongFlow, Disable Fallback")
    print("="*80)

    config = UnifiedEvolutionConfig(
        enable_loongflow=True,
        loongflow_fallback_enabled=False,
        require_loongflow=False,
        domain=DomainType.MATH
    )

    print(f"LoongFlow Enabled: {config.enable_loongflow}")
    print(f"Fallback Enabled: {config.loongflow_fallback_enabled}")
    print(f"Require LoongFlow: {config.require_loongflow}")

    should_use = config.should_use_loongflow()
    print(f"Should Use LoongFlow: {should_use}")

    if not should_use:
        print("Note: LoongFlow not available and fallback is disabled")
        print("OpenEvolve modes will be used instead")

    print()


def example_7_configuration_validation():
    """
    Example 7: Configuration Validation

    Demonstrates validation of conflicting settings.
    """
    print("\n" + "="*80)
    print("Example 7: Configuration Validation")
    print("="*80)

    try:
        # This should raise an error: can't require if disabled
        config = UnifiedEvolutionConfig(
            enable_loongflow=False,
            require_loongflow=True
        )
        print("ERROR: Validation should have failed!")
    except ValueError as e:
        print(f"Validation Error (Expected): {e}")

    print()


def example_8_check_availability():
    """
    Example 8: Check LoongFlow Availability

    Shows how to programmatically check if LoongFlow is available.
    """
    print("\n" + "="*80)
    print("Example 8: Check LoongFlow Availability")
    print("="*80)

    config = UnifiedEvolutionConfig()

    is_available = config._check_loongflow_availability()
    print(f"LoongFlow Available: {is_available}")

    if is_available:
        print("LoongFlow package is installed and ready to use")
    else:
        print("LoongFlow package is not installed")
        print("Install it with: pip install loongflow")

    print()


def example_9_qd_mode_without_loongflow():
    """
    Example 9: Use QD Mode Without LoongFlow

    Typical use case: Quality-Diversity evolution using OpenEvolve only.
    """
    print("\n" + "="*80)
    print("Example 9: QD Mode Without LoongFlow")
    print("="*80)

    config = UnifiedEvolutionConfig.openevolve_only(
        evolution_mode=EvolutionMode.QD,
        qd=QDConfig(
            enabled=True,
            grid_resolution=10,
            archive_size=500
        ),
        domain=DomainType.GENERAL
    )

    print(f"Evolution Mode: {config.evolution_mode}")
    print(f"LoongFlow Enabled: {config.enable_loongflow}")
    print(f"QD Grid Resolution: {config.qd.grid_resolution}")
    print(f"QD Archive Size: {config.qd.archive_size}")
    print()


def example_10_pes_mode_with_loongflow():
    """
    Example 10: Use PES Mode with LoongFlow

    Typical use case: Plan-Execute-Summarize evolution using LoongFlow.
    """
    print("\n" + "="*80)
    print("Example 10: PES Mode with LoongFlow")
    print("="*80)

    config = UnifiedEvolutionConfig(
        evolution_mode=EvolutionMode.PES,
        pes=PESConfig(
            enabled=True,
            max_rounds=5,
            enable_planning=True,
            enable_summary=True
        ),
        enable_loongflow=True,
        domain=DomainType.WEB
    )

    print(f"Evolution Mode: {config.evolution_mode}")
    print(f"LoongFlow Enabled: {config.enable_loongflow}")
    print(f"PES Max Rounds: {config.pes.max_rounds}")
    print(f"PES Planning: {config.pes.enable_planning}")
    print(f"PES Summary: {config.pes.enable_summary}")

    should_use = config.should_use_loongflow()
    print(f"Should Use LoongFlow: {should_use}")

    if not should_use:
        print("Note: LoongFlow not available, PES mode will not work as expected")

    print()


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("OPTIONAL LOONGFLOW CONFIGURATION EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate different ways to configure LoongFlow")
    print("as an optional component in the unified evolution system.")

    example_1_default_configuration()
    example_2_explicitly_disable_loongflow()
    example_3_openevolve_only_convenience()
    example_4_require_loongflow()
    example_5_enable_with_fallback()
    example_6_disable_fallback()
    example_7_configuration_validation()
    example_8_check_availability()
    example_9_qd_mode_without_loongflow()
    example_10_pes_mode_with_loongflow()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. LoongFlow is enabled by default with graceful fallback")
    print("2. Use .openevolve_only() for OpenEvolve-exclusive configurations")
    print("3. Use .loongflow_required() to enforce LoongFlow availability")
    print("4. Check .should_use_loongflow() to determine what will be used")
    print("5. Configuration validation prevents contradictory settings")
    print()


if __name__ == "__main__":
    main()
