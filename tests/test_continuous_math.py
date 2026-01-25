"""
Test Suite for LeanAide Continuous Mathematics Bridge

This test suite validates the continuous mathematics components implemented
from the Gap Analysis Implementation Plan (System 1: Continuous Mathematics Bridge).

Author: OpenEvolve
Created: 2026-01-02
"""

import asyncio
import pytest
import math
from typing import Dict, List, Tuple

# Import continuous math bridge
from leanaide_continuous_math import (
    ContinuousMathBridge,
    VerifiedIntegral,
    VerifiedODE,
    VerifiedLimit,
    CASBackend,
    Interval,
    BatchContinuousMath
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
async def math_bridge():
    """Create a continuous mathematics bridge for testing"""
    bridge = ContinuousMathBridge(
        cas_backend=CASBackend.SYMPY,
        default_epsilon=1e-8
    )
    return bridge


@pytest.fixture
def batch_processor(math_bridge):
    """Create a batch processor for testing"""
    return BatchContinuousMath(math_bridge)


# ============================================================================
# Interval Arithmetic Tests
# ============================================================================

class TestIntervalArithmetic:
    """Test interval arithmetic operations"""

    def test_interval_creation(self):
        """Test creating intervals"""
        interval = Interval(0.0, 1.0)
        assert interval.lower == 0.0
        assert interval.upper == 1.0

    def test_interval_midpoint(self):
        """Test interval midpoint calculation"""
        interval = Interval(0.0, 2.0)
        assert interval.midpoint == 1.0

    def test_interval_width(self):
        """Test interval width calculation"""
        interval = Interval(0.0, 1.0)
        assert interval.width == 1.0

    def test_interval_addition(self):
        """Test interval addition"""
        interval1 = Interval(0.0, 1.0)
        interval2 = Interval(2.0, 3.0)
        result = interval1 + interval2
        assert result.lower == 2.0
        assert result.upper == 4.0

    def test_interval_multiplication_positive(self):
        """Test interval multiplication with positive scalar"""
        interval = Interval(1.0, 2.0)
        result = interval * 2.0
        assert result.lower == 2.0
        assert result.upper == 4.0

    def test_interval_multiplication_negative(self):
        """Test interval multiplication with negative scalar"""
        interval = Interval(1.0, 2.0)
        result = interval * -1.0
        assert result.lower == -2.0
        assert result.upper == -1.0


# ============================================================================
# Verified Integration Tests
# ============================================================================

class TestVerifiedIntegration:
    """Test verified integral computation"""

    @pytest.mark.asyncio
    async def test_simple_polynomial_integral(self, math_bridge):
        """Test integration of simple polynomial"""
        result = await math_bridge.integrate_verified(
            "x**2",
            0.0,
            1.0,
            epsilon=1e-8
        )

        assert isinstance(result, VerifiedIntegral)
        assert result.integrand == "x**2"
        assert result.bounds == (0.0, 1.0)
        # ∫₀¹ x² dx = 1/3
        assert abs(result.value - 1.0/3.0) < 1e-6
        assert result.error_bound > 0
        assert result.computation_time >= 0

    @pytest.mark.asyncio
    async def test_gaussian_integral(self, math_bridge):
        """Test Gaussian integral: ∫₀^∞ x² exp(-x²) dx"""
        result = await math_bridge.integrate_verified(
            "x**2 * exp(-x**2)",
            0.0,
            float('inf'),
            epsilon=1e-8
        )

        assert isinstance(result, VerifiedIntegral)
        # ∫₀^∞ x² e^(-x²) dx = √π / 4 ≈ 0.443
        expected = math.sqrt(math.pi) / 4
        assert abs(result.value - expected) < 0.01
        assert result.error_bound > 0

    @pytest.mark.asyncio
    async def test_sinc_integral(self, math_bridge):
        """Test sinc function integral: ∫₀^∞ sin(x)/x dx"""
        result = await math_bridge.integrate_verified(
            "sin(x)/x",
            0.0,
            float('inf'),
            epsilon=1e-6
        )

        assert isinstance(result, VerifiedIntegral)
        # ∫₀^∞ sin(x)/x dx = π/2
        expected = math.pi / 2
        assert abs(result.value - expected) < 0.1

    @pytest.mark.asyncio
    async def test_exponential_integral(self, math_bridge):
        """Test exponential integral"""
        result = await math_bridge.integrate_verified(
            "exp(-x)",
            0.0,
            10.0,
            epsilon=1e-8
        )

        assert isinstance(result, VerifiedIntegral)
        # ∫₀^10 e^(-x) dx = 1 - e^(-10)
        expected = 1 - math.exp(-10)
        assert abs(result.value - expected) < 1e-4


# ============================================================================
# Verified ODE Tests
# ============================================================================

class TestVerifiedODE:
    """Test verified ODE solving"""

    @pytest.mark.asyncio
    async def test_exponential_decay_ode(self, math_bridge):
        """Test exponential decay: dy/dt = -y, y(0) = 1"""
        result = await math_bridge.solve_ode_verified(
            "dy/dt = -y",
            {"y": 1.0, "t": 0.0},
            (0.0, 1.0),
            method="runge_kutta_4",
            step_size=0.01
        )

        assert isinstance(result, VerifiedODE)
        assert result.equation == "dy/dt = -y"
        assert len(result.solution_points) > 0
        assert result.error_bound > 0

        # Check first point (initial condition)
        t0, y0 = result.solution_points[0]
        assert abs(t0 - 0.0) < 1e-10
        assert abs(y0 - 1.0) < 1e-6

        # Check last point (should be near e^(-1) ≈ 0.368)
        t_final, y_final = result.solution_points[-1]
        assert abs(t_final - 1.0) < 0.01
        expected_y = math.exp(-1.0)
        assert abs(y_final - expected_y) < 0.1

    @pytest.mark.asyncio
    async def test_simple_growth_ode(self, math_bridge):
        """Test simple growth: dy/dt = y, y(0) = 1"""
        result = await math_bridge.solve_ode_verified(
            "dy/dt = y",
            {"y": 1.0, "t": 0.0},
            (0.0, 1.0),
            method="runge_kutta_4",
            step_size=0.01
        )

        assert isinstance(result, VerifiedODE)
        assert len(result.solution_points) > 0

        # Check final value (should be near e^1 ≈ 2.718)
        t_final, y_final = result.solution_points[-1]
        expected_y = math.exp(1.0)
        assert abs(y_final - expected_y) < 0.2


# ============================================================================
# Verified Limit Tests
# ============================================================================

class TestVerifiedLimit:
    """Test verified limit computation"""

    @pytest.mark.asyncio
    async def test_sinc_limit(self, math_bridge):
        """Test limit: lim(x→0) sin(x)/x = 1"""
        result = await math_bridge.limit_verified(
            "sin(x)/x",
            "x",
            0.0,
            epsilon=1e-10
        )

        assert isinstance(result, VerifiedLimit)
        assert result.expression == "sin(x)/x"
        assert result.variable == "x"
        assert result.point == 0.0
        assert abs(result.limit_value - 1.0) < 1e-6
        assert result.delta > 0
        assert result.epsilon == 1e-10

    @pytest.mark.asyncio
    async def test_polynomial_limit(self, math_bridge):
        """Test limit: lim(x→1) (x² - 1)/(x - 1) = 2"""
        result = await math_bridge.limit_verified(
            "(x**2 - 1)/(x - 1)",
            "x",
            1.0,
            epsilon=1e-10
        )

        assert isinstance(result, VerifiedLimit)
        assert abs(result.limit_value - 2.0) < 1e-6

    @pytest.mark.asyncio
    async def test_exponential_limit(self, math_bridge):
        """Test limit: lim(x→0) (e^x - 1)/x = 1"""
        result = await math_bridge.limit_verified(
            "(exp(x) - 1)/x",
            "x",
            0.0,
            epsilon=1e-10
        )

        assert isinstance(result, VerifiedLimit)
        assert abs(result.limit_value - 1.0) < 1e-6


# ============================================================================
# Batch Operations Tests
# ============================================================================

class TestBatchOperations:
    """Test batch operations"""

    @pytest.mark.asyncio
    async def test_batch_integration(self, batch_processor):
        """Test batch integral computation"""
        integrals = [
            ("x**2", 0.0, 1.0),
            ("x", 0.0, 1.0),
            ("exp(-x)", 0.0, 1.0),
        ]

        results = await batch_processor.batch_integrate(integrals, epsilon=1e-8)

        assert len(results) == 3
        assert all(isinstance(r, VerifiedIntegral) for r in results)

        # Check results
        assert abs(results[0].value - 1.0/3.0) < 1e-6
        assert abs(results[1].value - 0.5) < 1e-6
        assert abs(results[2].value - (1 - math.exp(-1))) < 1e-6

    @pytest.mark.asyncio
    async def test_batch_ode_solving(self, batch_processor):
        """Test batch ODE solving"""
        odes = [
            ("dy/dt = -y", {"y": 1.0, "t": 0.0}, (0.0, 1.0)),
            ("dy/dt = y", {"y": 1.0, "t": 0.0}, (0.0, 1.0)),
        ]

        results = await batch_processor.batch_solve_odes(odes, method="runge_kutta_4")

        assert len(results) == 2
        assert all(isinstance(r, VerifiedODE) for r in results)
        assert all(len(r.solution_points) > 0 for r in results)


# ============================================================================
# Integration with LeanAide Client Tests
# ============================================================================

class TestLeanAideClientIntegration:
    """Test integration with LeanAide client"""

    @pytest.mark.asyncio
    async def test_client_continuous_math_methods(self):
        """Test that LeanAide client has continuous math methods"""
        from leanaide_client import LeanAideClient

        client = LeanAideClient()

        # Check methods exist
        assert hasattr(client, 'integrate_verified')
        assert hasattr(client, 'solve_ode_verified')
        assert hasattr(client, 'compute_limit_verified')
        assert hasattr(client, 'get_continuous_math_status')

    @pytest.mark.asyncio
    async def test_continuous_math_status(self):
        """Test getting continuous math status"""
        from leanaide_client import LeanAideClient

        client = LeanAideClient()
        status = await client.get_continuous_math_status()

        assert isinstance(status, dict)
        assert "enabled" in status
        assert "bridge_available" in status
        assert "cas_backend" in status


# ============================================================================
# MCP Tools Tests
# ============================================================================

class TestMCPTools:
    """Test MCP tools for continuous mathematics"""

    def test_integrate_verified_mcp_tool(self):
        """Test leanaide_integrate_verified MCP tool"""
        from leanaide_mcp_tools import leanaide_integrate_verified

        result = leanaide_integrate_verified(
            integrand="x**2",
            lower_bound=0.0,
            upper_bound=1.0,
            epsilon=1e-8
        )

        assert isinstance(result, dict)
        assert "success" in result
        if result.get("success"):
            assert "value" in result
            assert "error_bound" in result
            assert abs(result["value"] - 1.0/3.0) < 1e-6

    def test_solve_ode_verified_mcp_tool(self):
        """Test leanaide_solve_ode_verified MCP tool"""
        from lenaide_mcp_tools import leanaide_solve_ode_verified

        result = leanaide_solve_ode_verified(
            ode="dy/dt = -y",
            initial_conditions={"y": 1.0, "t": 0.0},
            time_span_start=0.0,
            time_span_end=1.0,
            step_size=0.01
        )

        assert isinstance(result, dict)
        assert "success" in result
        if result.get("success"):
            assert "solution_points" in result
            assert "num_points" in result

    def test_compute_limit_verified_mcp_tool(self):
        """Test leanaide_compute_limit_verified MCP tool"""
        from leanaide_mcp_tools import leanaide_compute_limit_verified

        result = leanaide_compute_limit_verified(
            expression="sin(x)/x",
            variable="x",
            point=0.0,
            epsilon=1e-10
        )

        assert isinstance(result, dict)
        assert "success" in result
        if result.get("success"):
            assert "limit_value" in result
            assert "delta" in result
            assert abs(result["limit_value"] - 1.0) < 1e-6

    def test_continuous_math_status_mcp_tool(self):
        """Test get_leanaide_continuous_math_status MCP tool"""
        from leanaide_mcp_tools import get_leanaide_continuous_math_status

        status = get_leanaide_continuous_math_status()

        assert isinstance(status, dict)
        assert "enabled" in status
        assert "bridge_available" in status
        assert "sympy_available" in status
        assert "scipy_available" in status
        assert "numpy_available" in status


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling"""

    @pytest.mark.asyncio
    async def test_invalid_expression(self, math_bridge):
        """Test handling of invalid mathematical expression"""
        with pytest.raises(Exception):
            await math_bridge.integrate_verified(
                "invalid_function(x)",
                0.0,
                1.0
            )

    @pytest.mark.asyncio
    async def test_invalid_ode_format(self, math_bridge):
        """Test handling of invalid ODE format"""
        with pytest.raises(ValueError):
            await math_bridge.solve_ode_verified(
                "not_a_valid_ode",
                {"y": 1.0, "t": 0.0},
                (0.0, 1.0)
            )


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics"""

    @pytest.mark.asyncio
    async def test_integration_performance(self, math_bridge):
        """Test that integration completes in reasonable time"""
        import time

        start_time = time.time()
        result = await math_bridge.integrate_verified(
            "x**2 * exp(-x**2)",
            0.0,
            float('inf'),
            epsilon=1e-8
        )
        elapsed_time = time.time() - start_time

        assert elapsed_time < 30.0  # Should complete in under 30 seconds
        assert result.computation_time < elapsed_time

    @pytest.mark.asyncio
    async def test_ode_performance(self, math_bridge):
        """Test that ODE solving completes in reasonable time"""
        import time

        start_time = time.time()
        result = await math_bridge.solve_ode_verified(
            "dy/dt = -y",
            {"y": 1.0, "t": 0.0},
            (0.0, 1.0),
            step_size=0.01
        )
        elapsed_time = time.time() - start_time

        assert elapsed_time < 10.0  # Should complete in under 10 seconds


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
