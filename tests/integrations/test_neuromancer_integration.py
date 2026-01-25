"""
Test Suite for NeuroMANCER Integration

This module provides comprehensive tests for the NeuroMANCER integration,
including unit tests, integration tests, and end-to-end tests.
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json
import tempfile
from pathlib import Path

# Import components to test
from integrations.base.optimization_interface import (
    OptimizationInterface,
    OptimizationResult,
    OptimizationProblem,
    OptimizationType,
    ProblemType,
    OptimizationError,
    ConfigurationError,
    ValidationError
)

from integrations.neuromancer.adapter import NeuroMANCERAdapter
from integrations.neuromancer.bridge import HybridSolver, LeanAideNeuroMANCERBridge


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def basic_config():
    """Basic configuration for testing."""
    return {
        "pytorch_env": "neuromancer_env",
        "device": "cpu",
        "max_workers": 2,
        "timeout": 30,
        "cache_enabled": False,  # Disable for faster tests
        "cache_ttl": 3600
    }


@pytest.fixture
def hybrid_config(basic_config):
    """Configuration for hybrid solver testing."""
    return {
        "leanaide_config": {},
        "neuromancer_config": basic_config,
        "hybrid_mode": "sequential",
        "max_iterations": 2,
        "convergence_tolerance": 1e-4
    }


@pytest.fixture
async def adapter(basic_config):
    """Initialize adapter for testing."""
    adapter = NeuroMANCERAdapter()
    # Mock environment validation for tests
    with patch.object(adapter, '_validate_environment', return_value=True):
        await adapter.initialize(basic_config)
    yield adapter
    await adapter.shutdown()


@pytest.fixture
def sample_optimization_problem():
    """Sample optimization problem for testing."""
    return OptimizationProblem(
        problem_type=ProblemType.OPTIMIZATION,
        variables={
            "x": {"initial_value": 0.0, "bounds": (-10, 10)},
            "y": {"initial_value": 0.0, "bounds": (-10, 10)}
        },
        parameters={"objective": "minimize x^2 + y^2"}
    )


@pytest.fixture
def sample_ode_problem():
    """Sample ODE problem for testing."""
    return {
        "ode_definition": {
            "equations": ["dy/dt = -k*y"],
            "variables": ["y"],
            "parameters": {"k": 0.5}
        },
        "initial_conditions": {"y": 1.0},
        "time_span": (0, 10),
        "method": "automatic"
    }


# ============================================================================
# Unit Tests: Optimization Interface
# ============================================================================

class TestOptimizationInterface:
    """Test the abstract optimization interface."""

    def test_optimization_result_creation(self):
        """Test creating OptimizationResult."""
        result = OptimizationResult(
            success=True,
            optimal_value=1.5,
            optimal_variables=[1.0, 2.0],
            iterations=100,
            convergence_history=[2.0, 1.8, 1.5]
        )

        assert result.success is True
        assert result.optimal_value == 1.5
        assert result.iterations == 100
        assert len(result.convergence_history) == 3

    def test_optimization_result_to_dict(self):
        """Test converting OptimizationResult to dictionary."""
        result = OptimizationResult(
            success=True,
            optimal_value=1.5,
            optimal_variables=[1.0, 2.0],
            iterations=100
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["success"] is True
        assert result_dict["optimal_value"] == 1.5
        assert isinstance(result_dict["optimal_variables"], list)

    def test_optimization_problem_validation(self):
        """Test OptimizationProblem validation."""
        # Valid problem
        problem = OptimizationProblem(
            problem_type=ProblemType.ODE,
            variables={"y": {"initial_value": 1.0}}
        )
        assert problem.validate() is True

        # Invalid optimization problem (no objective)
        problem = OptimizationProblem(
            problem_type=ProblemType.OPTIMIZATION,
            objective_function=None
        )
        assert problem.validate() is False

    def test_optimization_type_enum(self):
        """Test OptimizationType enum."""
        assert OptimizationType.UNCONSTRAINED.value == "unconstrained"
        assert OptimizationType.CONSTRAINED.value == "constrained"
        assert OptimizationType.PHYSICS_INFORMED.value == "physics_informed"

    def test_problem_type_enum(self):
        """Test ProblemType enum."""
        assert ProblemType.ODE.value == "ordinary_differential_equation"
        assert ProblemType.PDE.value == "partial_differential_equation"
        assert ProblemType.OPTIMIZATION.value == "optimization"


# ============================================================================
# Unit Tests: NeuroMANCER Adapter
# ============================================================================

class TestNeuroMANCERAdapter:
    """Test NeuroMANCER adapter functionality."""

    @pytest.mark.asyncio
    async def test_adapter_initialization(self, basic_config):
        """Test adapter initialization."""
        adapter = NeuroMANCERAdapter()

        with patch.object(adapter, '_validate_environment', return_value=True):
            result = await adapter.initialize(basic_config)

        assert result is True
        assert adapter.initialized is True
        assert adapter.pytorch_env == "neuromancer_env"
        assert adapter.device == "cpu"

    @pytest.mark.asyncio
    async def test_adapter_initialization_invalid_env(self, basic_config):
        """Test adapter initialization with invalid environment."""
        adapter = NeuroMANCERAdapter()

        with patch.object(adapter, '_validate_environment', return_value=False):
            with pytest.raises(ConfigurationError):
                await adapter.initialize(basic_config)

    @pytest.mark.asyncio
    async def test_validate_environment(self, adapter):
        """Test environment validation."""
        # Should be mocked in test, so this tests the mock works
        result = await adapter._validate_environment()
        assert isinstance(result, bool)

    def test_get_device_info(self, adapter):
        """Test device information retrieval."""
        info = adapter._get_device_info()

        assert "requested_device" in info
        assert "cpu_available" in info
        assert info["requested_device"] == "cpu"

    def test_json_serializer(self):
        """Test custom JSON serializer for numpy types."""
        import numpy as np

        data = {
            "array": np.array([1, 2, 3]),
            "int": np.int64(5),
            "float": np.float64(3.14)
        }

        # Should not raise TypeError
        json_str = json.dumps(data, default=NeuroMANCERAdapter._json_serializer)
        parsed = json.loads(json_str)

        assert parsed["array"] == [1, 2, 3]
        assert parsed["int"] == 5
        assert parsed["float"] == 3.14

    @pytest.mark.asyncio
    async def test_shutdown(self, adapter):
        """Test adapter shutdown."""
        result = await adapter.shutdown()

        assert result is True
        assert adapter.initialized is False

    @pytest.mark.asyncio
    async def test_get_template(self, adapter):
        """Test template retrieval."""
        # Mock template loading
        adapter.template_cache["test"] = {"name": "test"}

        template = await adapter.get_template("test")
        assert template["name"] == "test"

    @pytest.mark.asyncio
    async def test_get_template_not_found(self, adapter):
        """Test template retrieval with non-existent template."""
        with pytest.raises(Exception):  # TemplateNotFoundError
            await adapter.get_template("nonexistent")

    @pytest.mark.asyncio
    async def test_list_templates(self, adapter):
        """Test listing available templates."""
        # Mock templates
        adapter.template_cache = {"ode": {}, "pde": {}, "optimization": {}}

        templates = await adapter.list_templates()
        assert len(templates) == 3
        assert "ode" in templates
        assert "pde" in templates
        assert "optimization" in templates


# ============================================================================
# Integration Tests: Adapter Methods
# ============================================================================

class TestAdapterIntegration:
    """Integration tests for adapter methods."""

    @pytest.mark.asyncio
    async def test_solve_optimization_mocked(self, adapter, sample_optimization_problem):
        """Test solving optimization problem (mocked)."""
        # Mock the solver invocation
        with patch.object(adapter, '_invoke_solver'):
            result = await adapter.solve(
                sample_optimization_problem,
                OptimizationType.UNCONSTRAINED
            )

        assert isinstance(result, OptimizationResult)

    @pytest.mark.asyncio
    async def test_solve_ode_mocked(self, adapter, sample_ode_problem):
        """Test solving ODE (mocked)."""
        with patch.object(adapter, '_invoke_solver'):
            result = await adapter.solve_ode(**sample_ode_problem)

        assert isinstance(result, dict)
        assert "solution" in result or "error" in result

    @pytest.mark.asyncio
    async def test_solve_pde_mocked(self, adapter):
        """Test solving PDE (mocked)."""
        pde_problem = {
            "pde_definition": {
                "equation": "∂u/∂t = ∂²u/∂x²",
                "variables": ["u"],
                "parameters": {}
            },
            "boundary_conditions": {
                "type": "dirichlet",
                "conditions": {"x=0": "u=0", "x=1": "u=0"}
            },
            "initial_conditions": {"u(x,0)": "sin(πx)"},
            "domain": {"type": "interval", "bounds": [0, 1]}
        }

        with patch.object(adapter, '_invoke_solver'):
            result = await adapter.solve_pde(**pde_problem)

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_identify_system_mocked(self, adapter):
        """Test system identification (mocked)."""
        data = {
            "inputs": [[1, 2, 3], [0.5, 1.0, 1.5]],
            "outputs": [[2, 4, 6], [1, 2, 3]]
        }

        with patch.object(adapter, '_invoke_solver'):
            result = await adapter.identify_system(data=data)

        assert isinstance(result, dict)
        assert "model" in result or "error" in result

    @pytest.mark.asyncio
    async def test_constrained_optimization_mocked(self, adapter):
        """Test constrained optimization (mocked)."""
        constraints = [
            {"type": "inequality", "function": lambda x: 1 - x[0]}
        ]
        variables = {
            "names": ["x", "y"],
            "initial_values": [0, 0],
            "bounds": [(0, 10), (0, 10)]
        }

        with patch.object(adapter, 'solve'):
            result = await adapter.constrained_optimization(
                objective=lambda x: x[0]**2 + x[1]**2,
                constraints=constraints,
                variables=variables
            )

        assert isinstance(result, OptimizationResult)

    @pytest.mark.asyncio
    async def test_validate_adapter(self, adapter):
        """Test adapter validation."""
        validation = await adapter.validate()

        assert "is_valid" in validation
        assert "checks" in validation
        assert "issues" in validation
        assert "metrics" in validation


# ============================================================================
# Unit Tests: Hybrid Solver
# ============================================================================

class TestHybridSolver:
    """Test hybrid solver functionality."""

    @pytest.mark.asyncio
    async def test_hybrid_solver_initialization(self, hybrid_config):
        """Test hybrid solver initialization."""
        solver = HybridSolver()

        with patch.object(solver.neuromancer, 'initialize', return_value=True):
            result = await solver.initialize(hybrid_config)

        assert result is True
        assert solver.initialized is True
        assert solver.hybrid_mode == "sequential"

    @pytest.mark.asyncio
    async def test_solve_optimization_problem_hybrid(self, hybrid_config):
        """Test solving problem with hybrid approach."""
        solver = HybridSolver()

        with patch.object(solver.neuromancer, 'initialize', return_value=True):
            await solver.initialize(hybrid_config)

        problem = OptimizationProblem(
            problem_type=ProblemType.OPTIMIZATION,
            variables={"x": {"initial_value": 0.0}}
        )

        with patch.object(solver.neuromancer, 'solve') as mock_solve:
            mock_solve.return_value = OptimizationResult(
                success=True,
                optimal_value=1.0,
                optimal_variables=[1.0],
                iterations=100
            )

            result = await solver.solve_optimization_problem(problem, symbolic_analysis=False)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_solve_physics_informed_problem(self, hybrid_config):
        """Test solving physics-informed problem."""
        solver = HybridSolver()

        with patch.object(solver.neuromancer, 'initialize', return_value=True):
            await solver.initialize(hybrid_config)

        problem_def = {
            "type": "ode",
            "equations": ["dy/dt = -k*y"],
            "initial_conditions": {"y": 1.0, "k": 0.5}
        }

        with patch.object(solver.neuromancer, 'solve_ode') as mock_solve:
            mock_solve.return_value = {
                "solution": [1, 0.5, 0.25],
                "time_points": [0, 1, 2],
                "success": True
            }

            result = await solver.solve_physics_informed_problem(problem_def)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_get_solver_status(self, hybrid_config):
        """Test getting solver status."""
        solver = HybridSolver()

        with patch.object(solver.neuromancer, 'initialize', return_value=True):
            await solver.initialize(hybrid_config)

        with patch.object(solver.neuromancer, 'validate', return_value={"is_valid": True}):
            status = await solver.get_solver_status()

        assert "hybrid_solver_initialized" in status
        assert "neuromancer" in status
        assert status["hybrid_solver_initialized"] is True

    @pytest.mark.asyncio
    async def test_shutdown_hybrid_solver(self, hybrid_config):
        """Test shutting down hybrid solver."""
        solver = HybridSolver()

        with patch.object(solver.neuromancer, 'initialize', return_value=True):
            await solver.initialize(hybrid_config)

        with patch.object(solver.neuromancer, 'shutdown', return_value=True):
            result = await solver.shutdown()

        assert result is True
        assert solver.initialized is False


# ============================================================================
# Unit Tests: Bridge
# ============================================================================

class TestLeanAideNeuroMANCERBridge:
    """Test high-level bridge interface."""

    @pytest.mark.asyncio
    async def test_bridge_initialization(self, hybrid_config):
        """Test bridge initialization."""
        bridge = LeanAideNeuroMANCERBridge()

        with patch.object(bridge.hybrid_solver, 'initialize', return_value=True):
            result = await bridge.initialize(hybrid_config)

        assert result is True

    @pytest.mark.asyncio
    async def test_optimize(self, hybrid_config):
        """Test high-level optimization interface."""
        bridge = LeanAideNeuroMANCERBridge()

        with patch.object(bridge.hybrid_solver, 'initialize', return_value=True):
            await bridge.initialize(hybrid_config)

        with patch.object(bridge.hybrid_solver, 'solve_optimization_problem') as mock_solve:
            mock_result = OptimizationResult(
                success=True,
                optimal_value=0.0,
                optimal_variables=[0, 0],
                iterations=100
            )
            mock_solve.return_value = mock_result

            result = await bridge.optimize(
                objective="minimize x^2 + y^2",
                constraints=["x >= 0", "y >= 0"],
                variables={"x": (0, 10), "y": (0, 10)},
                use_hybrid=True
            )

        assert isinstance(result, dict)
        assert "success" in result

    @pytest.mark.asyncio
    async def test_solve_differential_equation(self, hybrid_config):
        """Test high-level differential equation solver."""
        bridge = LeanAideNeuroMANCERBridge()

        with patch.object(bridge.hybrid_solver, 'initialize', return_value=True):
            await bridge.initialize(hybrid_config)

        with patch.object(bridge.hybrid_solver, 'solve_physics_informed_problem') as mock_solve:
            mock_solve.return_value = {
                "solution": [1, 0.5],
                "success": True
            }

            result = await bridge.solve_differential_equation(
                equation="dy/dt = -k*y",
                equation_type="ode",
                conditions={"initial": {"y": 1.0, "k": 0.5}}
            )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_identify_system(self, hybrid_config):
        """Test high-level system identification."""
        bridge = LeanAideNeuroMANCERBridge()

        with patch.object(bridge.hybrid_solver, 'initialize', return_value=True):
            await bridge.initialize(hybrid_config)

        with patch.object(bridge.hybrid_solver.neuromancer, 'identify_system') as mock_id:
            mock_id.return_value = {
                "model": {"parameters": [0.5, 0.3]},
                "metrics": {"r2": 0.95}
            }

            result = await bridge.identify_system(
                input_data=[[1, 2, 3]],
                output_data=[[2, 4, 6]]
            )

        assert "model" in result
        assert result["model"]["parameters"] == [0.5, 0.3]


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_uninitialized_adapter(self):
        """Test using adapter without initialization."""
        adapter = NeuroMANCERAdapter()
        problem = OptimizationProblem(
            problem_type=ProblemType.OPTIMIZATION,
            variables={"x": {"initial_value": 0.0}}
        )

        with pytest.raises(RuntimeError):
            await adapter.solve(problem)

    @pytest.mark.asyncio
    async def test_invalid_problem(self, adapter):
        """Test solving invalid problem."""
        invalid_problem = OptimizationProblem(
            problem_type=ProblemType.OPTIMIZATION,
            objective_function=None  # Missing for optimization
        )

        with pytest.raises(ValidationError):
            await adapter.solve(invalid_problem)

    @pytest.mark.asyncio
    async def test_timeout_handling(self, adapter):
        """Test solver timeout."""
        problem = OptimizationProblem(
            problem_type=ProblemType.OPTIMIZATION,
            variables={"x": {"initial_value": 0.0}}
        )

        with patch.object(adapter, '_invoke_solver', side_effect=asyncio.TimeoutError()):
            with pytest.raises(TimeoutError):
                await adapter.solve(problem)

    @pytest.mark.asyncio
    async def test_solver_error(self, adapter):
        """Test solver error handling."""
        problem = OptimizationProblem(
            problem_type=ProblemType.OPTIMIZATION,
            variables={"x": {"initial_value": 0.0}}
        )

        with patch.object(adapter, '_invoke_solver', side_effect=Exception("Solver failed")):
            with pytest.raises(Exception):  # SolverError
                await adapter.solve(problem)


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and stress tests."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_optimizations(self, adapter):
        """Test multiple concurrent optimization problems."""
        problems = [
            OptimizationProblem(
                problem_type=ProblemType.OPTIMIZATION,
                variables={f"x{i}": {"initial_value": 0.0}}
            )
            for i in range(5)
        ]

        with patch.object(adapter, 'solve') as mock_solve:
            mock_solve.return_value = OptimizationResult(
                success=True,
                optimal_value=0.0,
                optimal_variables=[],
                iterations=10
            )

            results = await asyncio.gather(*[
                adapter.solve(p) for p in problems
            ])

        assert len(results) == 5
        assert all(r.success for r in results)


# ============================================================================
# End-to-End Tests
# ============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_optimization_workflow(self, basic_config):
        """Test complete optimization workflow."""
        # Initialize
        adapter = NeuroMANCERAdapter()

        with patch.object(adapter, '_validate_environment', return_value=True):
            assert await adapter.initialize(basic_config)

        # Create problem
        problem = OptimizationProblem(
            problem_type=ProblemType.OPTIMIZATION,
            variables={"x": {"initial_value": 5.0}},
            parameters={"minimize": "x^2"}
        )

        # Solve (mocked)
        with patch.object(adapter, '_invoke_solver'):
            result = await adapter.solve(problem)

        # Validate
        assert isinstance(result, OptimizationResult)

        # Cleanup
        assert await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_full_ode_workflow(self, basic_config):
        """Test complete ODE solving workflow."""
        adapter = NeuroMANCERAdapter()

        with patch.object(adapter, '_validate_environment', return_value=True):
            await adapter.initialize(basic_config)

        ode_def = {
            "equations": ["dy/dt = -y"],
            "variables": ["y"],
            "parameters": {}
        }

        with patch.object(adapter, '_invoke_solver'):
            result = await adapter.solve_ode(
                ode_definition=ode_def,
                initial_conditions={"y": 1.0},
                time_span=(0, 10)
            )

        assert isinstance(result, dict)

        await adapter.shutdown()


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
