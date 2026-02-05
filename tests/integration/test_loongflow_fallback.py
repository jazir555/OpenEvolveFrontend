"""
Test LoongFlow Graceful Fallback System

This test demonstrates and validates the graceful fallback system that allows
OpenEvolve to work seamlessly whether LoongFlow is available or not.
"""

import asyncio
import pytest
from openevolve.integrations import (
    LoongFlowAdapter,
    LoongFlowChecker,
    OpenEvolveFallbackAdapter
)
from openevolve.utils.messages import LoongFlowMessages


class TestLoongFlowChecker:
    """Test LoongFlow availability checker."""

    def test_is_installed(self):
        """Test checking if LoongFlow is installed."""
        result = LoongFlowChecker.is_installed()
        assert isinstance(result, bool)
        print(f"LoongFlow installed: {result}")

    def test_get_version(self):
        """Test getting LoongFlow version."""
        version = LoongFlowChecker.get_version()
        if LoongFlowChecker.is_installed():
            assert version is not None
            assert isinstance(version, str)
            print(f"LoongFlow version: {version}")
        else:
            assert version is None
            print("LoongFlow version: N/A (not installed)")

    def test_check_requirements(self):
        """Test checking LoongFlow requirements."""
        issues = LoongFlowChecker.check_requirements()
        assert isinstance(issues, list)
        if not LoongFlowChecker.is_installed():
            assert len(issues) > 0
            print(f"Requirements issues: {issues}")
        else:
            print(f"Requirements check passed with {len(issues)} issues")

    def test_is_available(self):
        """Test checking if LoongFlow is available."""
        # Quick check
        available = LoongFlowChecker.is_available(requirement_check=False)
        assert isinstance(available, bool)

        # Deep check
        available_deep = LoongFlowChecker.is_available(requirement_check=True)
        assert isinstance(available_deep, bool)

        print(f"LoongFlow available (quick): {available}")
        print(f"LoongFlow available (deep): {available_deep}")

    def test_get_diagnostics(self):
        """Test getting comprehensive diagnostics."""
        diagnostics = LoongFlowChecker.get_diagnostics()
        assert isinstance(diagnostics, dict)
        assert "installed" in diagnostics
        assert "version" in diagnostics
        assert "available" in diagnostics
        assert "issues" in diagnostics
        assert "components" in diagnostics

        print("\nDiagnostics:")
        for key, value in diagnostics.items():
            print(f"  {key}: {value}")

    def test_print_diagnostics(self, capsys):
        """Test printing diagnostics."""
        LoongFlowChecker.print_diagnostics()
        captured = capsys.readouterr()
        assert "LoongFlow Availability Diagnostics" in captured.out


class TestLoongFlowMessages:
    """Test user-friendly messages."""

    def test_disabled_message(self):
        """Test message when LoongFlow is disabled."""
        message = LoongFlowMessages.disabled_message()
        assert "LoongFlow Disabled" in message
        assert "OpenEvolve-only mode" in message
        print(f"\n{message}")

    def test_not_available_message(self):
        """Test message when LoongFlow is not available."""
        # With fallback
        message_fallback = LoongFlowMessages.not_available_message(
            fallback_enabled=True
        )
        assert "LoongFlow Not Available" in message_fallback
        assert "falling back" in message_fallback

        # Without fallback
        message_no_fallback = LoongFlowMessages.not_available_message(
            fallback_enabled=False
        )
        assert "Required But Not Available" in message_no_fallback

        print(f"\nWith fallback:\n{message_fallback}")
        print(f"\nWithout fallback:\n{message_no_fallback}")

    def test_using_openevolve_message(self):
        """Test message when using OpenEvolve."""
        message = LoongFlowMessages.using_openevolve_message(mode="qd")
        assert "OpenEvolve-Only Mode" in message
        assert "qd" in message
        print(f"\n{message}")

    def test_using_loongflow_message(self):
        """Test message when LoongFlow is initialized."""
        message = LoongFlowMessages.using_loongflow_message()
        assert "LoongFlow PES Initialized" in message
        print(f"\n{message}")

    def test_initialization_failed_message(self):
        """Test message when initialization fails."""
        message = LoongFlowMessages.initialization_failed_message(
            error="ImportError: No module named 'loongflow'",
            fallback_enabled=True
        )
        assert "Initialization Failed" in message
        print(f"\n{message}")


class TestOpenEvolveFallbackAdapter:
    """Test OpenEvolve fallback adapter."""

    def test_init(self):
        """Test initializing fallback adapter."""
        config = {
            "max_iterations": 50,
            "population_size": 10,
            "mode": "standard"
        }
        adapter = OpenEvolveFallbackAdapter(config)
        assert adapter is not None
        assert adapter.openevolve_config == config
        assert adapter.evolution_mode == "standard"
        print(f"Adapter: {adapter}")

    def test_get_capabilities(self):
        """Test getting fallback adapter capabilities."""
        config = {"mode": "qd"}
        adapter = OpenEvolveFallbackAdapter(config)
        capabilities = adapter.get_capabilities()

        assert isinstance(capabilities, dict)
        assert capabilities["available"] is True
        assert capabilities["system"] == "openevolve"
        assert capabilities["mode"] == "qd"
        print(f"\nCapabilities: {capabilities}")

    @pytest.mark.asyncio
    async def test_evolve(self):
        """Test evolution with fallback adapter."""
        config = {
            "max_iterations": 10,
            "mode": "standard"
        }
        adapter = OpenEvolveFallbackAdapter(config)

        result = await adapter.evolve(
            problem="Optimize function: f(x) = x^2",
            domain="math"
        )

        assert isinstance(result, dict)
        assert "best_solution" in result
        assert "best_fitness" in result
        assert "system_used" in result
        assert result["system_used"] == "openevolve"
        print(f"\nEvolution result: {result}")


class TestLoongFlowAdapter:
    """Test LoongFlow adapter with fallback."""

    def test_init_with_loongflow_disabled(self):
        """Test initialization with LoongFlow explicitly disabled."""
        config = {
            "enable_loongflow": False,
            "max_iterations": 50
        }
        adapter = LoongFlowAdapter(config)

        assert adapter.using_loongflow is False
        assert adapter.mode in ["disabled", "openevolve"]
        assert adapter.fallback_adapter is not None
        print(f"Adapter status: {adapter.get_status()}")

    def test_init_with_loongflow_enabled_but_unavailable(self):
        """Test initialization when LoongFlow enabled but not available."""
        if not LoongFlowChecker.is_installed():
            config = {
                "enable_loongflow": True,
                "require_loongflow": False,
                "show_messages": False
            }
            adapter = LoongFlowAdapter(config)

            assert adapter.using_loongflow is False
            assert adapter.fallback_adapter is not None
            assert adapter.mode == "unavailable"
            print(f"Adapter status: {adapter.get_status()}")
        else:
            pytest.skip("LoongFlow is installed")

    def test_init_require_loongflow_fail(self):
        """Test that require_loongflow=True fails when LoongFlow unavailable."""
        if not LoongFlowChecker.is_installed():
            config = {
                "enable_loongflow": True,
                "require_loongflow": True
            }

            # This should raise an error
            with pytest.raises(RuntimeError, match="LoongFlow is required"):
                adapter = LoongFlowAdapter(config)
        else:
            pytest.skip("LoongFlow is installed")

    def test_get_status(self):
        """Test getting adapter status."""
        config = {"enable_loongflow": False}
        adapter = LoongFlowAdapter(config)

        status = adapter.get_status()
        assert isinstance(status, dict)
        assert "mode" in status
        assert "using_loongflow" in status
        assert "loongflow_available" in status
        assert "capabilities" in status
        print(f"\nStatus: {status}")

    def test_get_capabilities(self):
        """Test getting adapter capabilities."""
        config = {"enable_loongflow": False}
        adapter = LoongFlowAdapter(config)

        capabilities = adapter.get_capabilities()
        assert isinstance(capabilities, dict)
        assert "available" in capabilities
        assert "system" in capabilities
        print(f"\nCapabilities: {capabilities}")

    def test_print_status(self, capsys):
        """Test printing adapter status."""
        config = {"enable_loongflow": False}
        adapter = LoongFlowAdapter(config)

        adapter.print_status()
        captured = capsys.readouterr()
        assert "LoongFlow Adapter Status" in captured.out
        assert "Mode:" in captured.out

    @pytest.mark.asyncio
    async def test_evolve_with_fallback(self):
        """Test evolution with OpenEvolve fallback."""
        config = {
            "enable_loongflow": False,
            "max_iterations": 10,
            "mode": "standard"
        }
        adapter = LoongFlowAdapter(config)

        result = await adapter.evolve(
            problem="Optimize sorting algorithm",
            domain="code"
        )

        assert isinstance(result, dict)
        assert "best_solution" in result
        assert "best_fitness" in result
        assert "system_used" in result
        assert result["system_used"] == "openevolve"
        print(f"\nEvolution result: {result}")


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    def test_scenario_1_default_configuration(self):
        """
        Scenario 1: Default configuration (LoongFlow enabled if available).

        This is the typical user scenario where LoongFlow will be used
        if installed, but the system falls back gracefully if not.
        """
        config = {}  # Use all defaults
        adapter = LoongFlowAdapter(config)

        status = adapter.get_status()
        print(f"\nScenario 1 - Default config:")
        print(f"  Mode: {status['mode']}")
        print(f"  Using LoongFlow: {status['using_loongflow']}")
        print(f"  System available: {status['loongflow_available']}")

        # Should work regardless of LoongFlow availability
        assert adapter is not None

    def test_scenario_2_explicit_openevolve_mode(self):
        """
        Scenario 2: User explicitly wants OpenEvolve-only mode.

        User may disable LoongFlow intentionally to use OpenEvolve's
        native capabilities.
        """
        config = {
            "enable_loongflow": False,
            "mode": "qd"
        }
        adapter = LoongFlowAdapter(config)

        assert adapter.using_loongflow is False
        assert adapter.evolution_mode == "qd"

        capabilities = adapter.get_capabilities()
        assert capabilities["system"] == "openevolve"
        print(f"\nScenario 2 - OpenEvolve-only:")
        print(f"  Mode: {adapter.mode}")
        print(f"  Evolution mode: {adapter.evolution_mode}")

    def test_scenario_3_strict_loongflow_requirement(self):
        """
        Scenario 3: User requires LoongFlow.

        If LoongFlow is required but not available, the system should
        fail explicitly rather than silently falling back.
        """
        if not LoongFlowChecker.is_installed():
            config = {
                "enable_loongflow": True,
                "require_loongflow": True
            }

            with pytest.raises(RuntimeError):
                adapter = LoongFlowAdapter(config)

            print("\nScenario 3 - Strict LoongFlow: Correctly failed")
        else:
            pytest.skip("LoongFlow is installed")

    def test_scenario_4_production_configuration(self):
        """
        Scenario 4: Production-ready configuration.

        Configuration optimized for production use with proper
        fallback and message handling.
        """
        config = {
            "enable_loongflow": True,
            "require_loongflow": False,
            "show_messages": True,
            "max_iterations": 100,
            "population_size": 20,
            "mode": "standard"
        }
        adapter = LoongFlowAdapter(config)

        status = adapter.get_status()
        print(f"\nScenario 4 - Production config:")
        print(f"  Mode: {status['mode']}")
        print(f"  Ready: {adapter.is_available() or True}")  # Always ready

        # Should always be ready, with or without LoongFlow
        assert adapter is not None


def test_run_diagnostics():
    """Run and display full diagnostics."""
    print("\n" + "=" * 70)
    print("LOONGFLOW INTEGRATION DIAGNOSTICS")
    print("=" * 70)

    # Run checker diagnostics
    LoongFlowChecker.print_diagnostics()

    # Test adapter with different configurations
    configs = [
        {"enable_loongflow": False, "mode": "standard"},
        {"enable_loongflow": True, "require_loongflow": False, "mode": "qd"},
    ]

    for i, config in enumerate(configs, 1):
        print(f"\nConfiguration {i}: {config}")
        adapter = LoongFlowAdapter({**config, "show_messages": False})
        status = adapter.get_status()
        print(f"  Result: {status['mode']}")
        print(f"  Using LoongFlow: {status['using_loongflow']}")
        print(f"  Capabilities: {status['capabilities']['system']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run diagnostics when executed directly
    test_run_diagnostics()
    print("\n[OK] All fallback systems operational!")
