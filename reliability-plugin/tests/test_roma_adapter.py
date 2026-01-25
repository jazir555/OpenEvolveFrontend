"""
Comprehensive test suite for ROMA Reliability Adapter.

This test suite covers all functionality of the ROMA adapter including:
- Adapter initialization and dual-mode operation
- Core integration mode testing
- MCP tool fallback testing
- Solve with constraints functionality
- Analyze with constraints functionality
- Health checks and layer availability
- Error handling and graceful degradation
- Integration with other reliability components
"""

import pytest
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock, call, AsyncMock
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import classes to test (with proper error handling for missing modules)
try:
    from reliability.roma_adapter import (
        RomaReliabilityAdapter,
        RomaSolutionResult,
        RomaAnalysisResult,
        create_roma_adapter,
        get_default_adapter
    )
    ROMA_ADAPTER_AVAILABLE = True
except ImportError:
    ROMA_ADAPTER_AVAILABLE = False
    # Create mock classes for testing when ROMA is not available
    class RomaSolutionResult:
        def __init__(self, success: bool, result: Dict = None, task: str = None,
                     error: str = None, layers_used: List[str] = None,
                     constraint_violations: List[str] = None,
                     validation_failures: List[Dict] = None,
                     remediation_applied: List[str] = None,
                     correlation_id: str = None, metadata: Dict = None):
            self.success = success
            self.result = result
            self.task = task
            self.error = error
            self.layers_used = layers_used or []
            self.constraint_violations = constraint_violations or []
            self.validation_failures = validation_failures or []
            self.remediation_applied = remediation_applied or []
            self.correlation_id = correlation_id or ""
            self.metadata = metadata or {}

    class RomaAnalysisResult:
        def __init__(self, success: bool, analysis: Dict = None, task: str = None,
                     error: str = None, layers_used: List[str] = None,
                     validation_failures: List[Dict] = None,
                     correlation_id: str = None, metadata: Dict = None):
            self.success = success
            self.analysis = analysis
            self.task = task
            self.error = error
            self.layers_used = layers_used or []
            self.validation_failures = validation_failures or []
            self.correlation_id = correlation_id or ""
            self.metadata = metadata or {}

    class RomaReliabilityAdapter:
        def __init__(self, config: Optional[Dict] = None):
            pass

    def create_roma_adapter(config: Optional[Dict] = None):
        return RomaReliabilityAdapter(config)

    def get_default_adapter():
        return RomaReliabilityAdapter()


class TestRomaAdapterInitialization:
    """Test suite for ROMA adapter initialization."""

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_initialization_with_config(self, mock_config):
        """Test adapter initialization with configuration."""
        config = mock_config
        adapter = RomaReliabilityAdapter(config)

        assert adapter is not None
        assert adapter.config == config
        # Check that adapters are initialized
        assert hasattr(adapter, 'lmql_adapter')
        assert hasattr(adapter, 'guardrails_adapter')

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_initialization_without_config(self):
        """Test adapter initialization without configuration."""
        adapter = RomaReliabilityAdapter()

        assert adapter is not None
        assert adapter.config is None  # Should handle missing config

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_initialization_with_custom_adapters(self, mock_lmql_adapter, mock_guardrails_adapter):
        """Test adapter initialization with custom adapters."""
        adapter = RomaReliabilityAdapter(
            lmql_adapter=mock_lmql_adapter,
            guardrails_adapter=mock_guardrails_adapter
        )

        assert adapter is not None
        assert adapter.lmql_adapter == mock_lmql_adapter
        assert adapter.guardrails_adapter == mock_guardrails_adapter

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_initialization_status_attributes(self, mock_config):
        """Test initialization of status attributes."""
        adapter = RomaReliabilityAdapter(mock_config)

        # Check that all status attributes are initialized
        assert hasattr(adapter, 'roma_core_available')
        assert hasattr(adapter, 'roma_mcp_available')
        assert hasattr(adapter, 'roma_available')
        assert hasattr(adapter, 'lmql_enabled')
        assert hasattr(adapter, 'guardrails_enabled')

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_initialization_logging(self, mock_config):
        """Test initialization logs appropriate events."""
        with patch('reliability.roma_adapter.logger') as mock_logger:
            adapter = RomaReliabilityAdapter(mock_config)

            # Check that initialization was logged
            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args[0][0]
            assert "roma_reliability_adapter_initialized" in call_args.get("event", "")

    def test_initialization_when_roma_unavailable(self):
        """Test adapter initialization when ROMA is completely unavailable."""
        with patch.dict('sys.modules', {'roma_dspy': None, 'roma_mcp_tools': None}):
            adapter = RomaReliabilityAdapter()

            assert adapter is not None
            assert adapter.roma_available is False

    @pytest.mark.parametrize("lmql_enabled", [True, False])
    @pytest.mark.parametrize("guardrails_enabled", [True, False])
    def test_initialization_with_different_layer_combinations(self, lmql_enabled, guardrails_enabled):
        """Test initialization with different combinations of layers."""
        with patch('reliability.roma_adapter.LMQL_AVAILABLE', lmql_enabled), \
             patch('reliability.roma_adapter.GUARDRAILS_AVAILABLE', guardrails_enabled):
            adapter = RomaReliabilityAdapter()

            assert adapter is not None
            assert adapter.lmql_enabled == lmql_enabled
            assert adapter.guardrails_enabled == guardrails_enabled


class TestDualModeOperation:
    """Test suite for dual-mode operation (core integration vs MCP fallback)."""

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_core_integration_mode(self, mock_roma_core, mock_roma_mcp_tools):
        """Test core integration mode when available."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True), \
             patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            assert adapter.roma_core_available is True
            assert adapter.roma_mcp_available is True
            assert adapter.roma_available is True

            # Should prefer core integration
            status = adapter.get_status()
            assert status['execution_mode'] == 'core_preferred_with_mcp_fallback'

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_mcp_fallback_mode(self, mock_roma_mcp_tools):
        """Test MCP fallback mode when core is unavailable."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', False), \
             patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            assert adapter.roma_core_available is False
            assert adapter.roma_mcp_available is True
            assert adapter.roma_available is True

            # Should use MCP mode
            status = adapter.get_status()
            assert status['execution_mode'] == 'mcp_only'

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_unavailable_mode(self):
        """Test when both core and MCP are unavailable."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', False), \
             patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', False):
            adapter = RomaReliabilityAdapter()

            assert adapter.roma_core_available is False
            assert adapter.roma_mcp_available is False
            assert adapter.roma_available is False

            # Should be unavailable
            status = adapter.get_status()
            assert status['execution_mode'] == 'unavailable'

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_mode_switching(self, mock_roma_core, mock_roma_mcp_tools):
        """Test adapter mode switching based on availability."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True), \
             patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Initially should prefer core
            status = adapter.get_status()
            assert status['execution_mode'] == 'core_preferred_with_mcp_fallback'

            # Simulate core becoming unavailable
            adapter.roma_core_available = False
            status = adapter.get_status()
            assert status['execution_mode'] == 'mcp_only'


class TestCoreIntegration:
    """Test suite for core integration functionality."""

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_solve_with_core_integration(self, mock_roma_core):
        """Test solving with core integration."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            result = adapter._solve_with_core_integration(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={"max_depth": 3},
                execution_mode="recursive",
                enable_checkpoints=True,
                provider="openai",
                model="gpt-4",
                api_key="test",
                correlation_id="test_corr_id"
            )

            # Should succeed and use core integration
            assert result["success"] is True
            assert "lmql_constraints" in result.get("layers_used", [])
            assert "roma_core" in result.get("layers_used", [])

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_core_integration_with_lmql_constraints(self, mock_lmql_adapter, mock_roma_core):
        """Test core integration with LMQL constraints."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True), \
             patch('reliability.roma_adapter.LMQL_AVAILABLE', True):
            adapter = RomaReliabilityAdapter(lmql_adapter=mock_lmql_adapter)

            # Configure mock solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                result="Test solution",
                status=Mock(value="completed")
            )
            mock_roma_core.RecursiveSolver.return_value = mock_solver

            result = adapter._solve_with_core_integration(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={"max_depth": 3},
                execution_mode="recursive",
                enable_checkpoints=True,
                provider="openai",
                model="gpt-4",
                api_key="test",
                correlation_id="test_corr_id"
            )

            # Should use LMQL constraints
            assert result["success"] is True
            assert "lmql_constraints" in result.get("layers_used", [])

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_core_integration_event_driven_mode(self, mock_roma_core):
        """Test core integration with event-driven mode."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure solver with event_solve capability
            mock_solver = Mock()
            mock_solver.event_solve.return_value = Mock(
                result="Event-driven solution",
                status=Mock(value="completed")
            )
            mock_roma_core.RecursiveSolver.return_value = mock_solver

            result = adapter._solve_with_core_integration(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={},
                execution_mode="event_driven",
                enable_checkpoints=True,
                provider="openai",
                model="gpt-4",
                api_key="test",
                correlation_id="test_corr_id"
            )

            # Should use event_solve
            assert result["success"] is True

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_core_integration_failure_fallback(self, mock_roma_core, mock_roma_mcp_tools):
        """Test core integration failure falls back to MCP."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True), \
             patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure core to fail
            mock_roma_core.RecursiveSolver.side_effect = Exception("Core failed")

            # Configure MCP to succeed
            mock_result = {
                "result": "MCP solution",
                "status": "completed",
                "token_usage": {"input": 100, "output": 200}
            }
            mock_roma_mcp_tools.solve_with_roma.return_value = mock_result

            result = adapter.solve_with_constraints(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            # Should succeed with MCP fallback
            assert result.success is True
            assert result.result["result"] == "MCP solution"

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_enhanced_atomizer_creation(self, mock_lmql_adapter, mock_roma_core):
        """Test creation of enhanced atomizer with LMQL."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True), \
             patch('reliability.roma_adapter.LMQL_AVAILABLE', True):
            adapter = RomaReliabilityAdapter(lmql_adapter=mock_lmql_adapter)

            # Test enhanced atomizer creation
            atomizer = adapter._create_enhanced_atomizer(max_depth=3, use_lmql=True)

            assert atomizer is not None
            # Should be wrapped with LMQL functionality

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_enhanced_planner_creation(self, mock_lmql_adapter, mock_roma_core):
        """Test creation of enhanced planner with LMQL."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True), \
             patch('reliability.roma_adapter.LMQL_AVAILABLE', True):
            adapter = RomaReliabilityAdapter(lmql_adapter=mock_lmql_adapter)

            # Test enhanced planner creation
            planner = adapter._create_enhanced_planner(max_subtasks=10, use_lmql=True)

            assert planner is not None
            # Should be wrapped with LMQL functionality

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_roma_config_creation(self, mock_roma_core):
        """Test ROMA configuration creation."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Test config creation with provider
            config = adapter._create_roma_config(
                provider="openai",
                model="gpt-4",
                api_key="test_key"
            )

            assert config is not None


class TestMCPFallback:
    """Test suite for MCP fallback functionality."""

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_solve_with_mcp_tools(self, mock_roma_mcp_tools):
        """Test solving with MCP tools."""
        with patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure MCP result
            mock_result = {
                "result": "MCP solution",
                "status": "completed",
                "token_usage": {"input": 100, "output": 200}
            }
            mock_roma_mcp_tools.solve_with_roma.return_value = mock_result

            result = adapter._solve_with_mcp_tools(
                task="Solve: 2 + 2",
                max_depth=3,
                execution_mode="recursive",
                enable_checkpoints=True,
                provider="openai",
                model="gpt-4",
                api_key="test",
                correlation_id="test_corr_id"
            )

            # Should succeed
            assert result["success"] is True
            assert result["result"]["result"] == "MCP solution"
            assert "roma_mcp" in result.get("layers_used", [])

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_mcp_tools_error_handling(self, mock_roma_mcp_tools):
        """Test MCP tools error handling."""
        with patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure MCP to return error
            mock_roma_mcp_tools.solve_with_roma.return_value = {
                "error": "MCP tool error"
            }

            result = adapter._solve_with_mcp_tools(
                task="Solve: 2 + 2",
                max_depth=3,
                execution_mode="recursive",
                enable_checkpoints=True,
                provider="openai",
                model="gpt-4",
                api_key="test",
                correlation_id="test_corr_id"
            )

            # Should fail gracefully
            assert result["success"] is False
            assert result["error"] == "MCP tool error"

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_mcp_tools_exception_handling(self, mock_roma_mcp_tools):
        """Test MCP tools exception handling."""
        with patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure MCP to raise exception
            mock_roma_mcp_tools.solve_with_roma.side_effect = Exception("Network error")

            result = adapter._solve_with_mcp_tools(
                task="Solve: 2 + 2",
                max_depth=3,
                execution_mode="recursive",
                enable_checkpoints=True,
                provider="openai",
                model="gpt-4",
                api_key="test",
                correlation_id="test_corr_id"
            )

            # Should fail gracefully
            assert result["success"] is False
            assert "Network error" in result["error"]


class TestSolveWithConstraints:
    """Test suite for solve_with_constraints functionality."""

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_solve_with_constraints_success(self, mock_roma_core):
        """Test successful solve with constraints."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure core solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                result="Test solution",
                status=Mock(value="completed")
            )
            mock_roma_core.RecursiveSolver.return_value = mock_solver

            result = adapter.solve_with_constraints(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={"max_depth": 3, "max_subtasks": 10},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            # Should succeed
            assert result.success is True
            assert result.result["result"] == "Test solution"
            assert "roma_core" in result.layers_used

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_solve_with_input_validation(self, mock_guardrails_adapter, mock_roma_core):
        """Test solve with input validation."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True), \
             patch('reliability.roma_adapter.GUARDRAILS_AVAILABLE', True):
            adapter = RomaReliabilityAdapter(guardrails_adapter=mock_guardrails_adapter)

            # Configure input validation to pass
            mock_guardrails_adapter.validate_input.return_value = Mock(
                is_valid=True,
                failures=[]
            )

            # Configure core solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                result="Test solution",
                status=Mock(value="completed")
            )
            mock_roma_core.RecursiveSolver.return_value = mock_solver

            result = adapter.solve_with_constraints(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            # Should succeed and use input validation
            assert result.success is True
            assert "guardrails_input" in result.layers_used

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_solve_with_input_validation_failure(self, mock_guardrails_adapter):
        """Test solve with input validation failure."""
        with patch('reliability.roma_adapter.GUARDRAILS_AVAILABLE', True):
            adapter = RomaReliabilityAdapter(guardrails_adapter=mock_guardrails_adapter)

            # Configure input validation to fail
            mock_guardrails_adapter.validate_input.return_value = Mock(
                is_valid=False,
                failures=[{"validator": "length", "message": "Too long"}]
            )

            result = adapter.solve_with_constraints(
                task="This is a very long task that exceeds the maximum allowed length",
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            # Should fail at input validation
            assert result.success is False
            assert "Input validation failed" in result.error
            assert "guardrails_input" in result.layers_used

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_solve_with_output_validation(self, mock_guardrails_adapter, mock_roma_core):
        """Test solve with output validation."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE'), \
             patch('reliability.roma_adapter.GUARDRAILS_AVAILABLE', True):
            adapter = RomaReliabilityAdapter(guardrails_adapter=mock_guardrails_adapter)

            # Configure output validation
            mock_guardrails_adapter.validate_output.return_value = Mock(
                is_valid=True,
                failures=[],
                remediation_applied=None,
                output=None
            )

            # Configure core solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                result="Test solution",
                status=Mock(value="completed")
            )
            mock_roma_core.RecursiveSolver.return_value = mock_solver

            result = adapter.solve_with_constraints(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            # Should succeed and use output validation
            assert result.success is True
            assert "guardrails_output" in result.layers_used

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_solve_with_output_remediation(self, mock_guardrails_adapter, mock_roma_core):
        """Test solve with output remediation."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE'), \
             patch('reliability.roma_adapter.GUARDRAILS_AVAILABLE', True):
            adapter = RomaReliabilityAdapter(guardrails_adapter=mock_guardrails_adapter)

            # Configure output validation with remediation
            mock_guardrails_adapter.validate_output.return_value = Mock(
                is_valid=True,
                failures=[{"validator": "json_structure"}],
                remediation_applied="json_fix",
                output='{"fixed": "true"}'
            )

            # Configure core solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                result='{"original": "data"}',
                status=Mock(value="completed")
            )
            mock_roma_core.RecursiveSolver.return_value = mock_solver

            result = adapter.solve_with_constraints(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            # Should succeed and apply remediation
            assert result.success is True
            assert "output_remediated" in result.remediation_applied
            assert "guardrails_output" in result.layers_used

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_solve_with_constraints_parameterized(self, test_constraints):
        """Test solve with different constraint configurations."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure core solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                result="Test solution",
                status=Mock(value="completed")
            )
            from unittest.mock import patch
            with patch('reliability.roma_adapter.RecursiveSolver') as mock_recursive:
                mock_recursive.return_value = mock_solver

            for constraint_name, constraints in test_constraints.items():
                result = adapter.solve_with_constraints(
                    task=f"Test task with {constraint_name} constraints",
                    max_depth=3,
                    constraints=constraints,
                    execution_mode="recursive",
                    enable_checkpoints=True
                )

                assert result.success is True

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    @pytest.mark.parametrize("execution_mode", ["recursive", "event_driven"])
    def test_solve_with_different_execution_modes(self, execution_mode, mock_roma_core):
        """Test solve with different execution modes."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure solver
            if execution_mode == "event_driven":
                mock_solver = Mock()
                mock_solver.event_solve.return_value = Mock(
                    result="Event-driven solution",
                    status=Mock(value="completed")
                )
            else:
                mock_solver = Mock()
                mock_solver.solve.return_value = Mock(
                    result="Recursive solution",
                    status=Mock(value="completed")
                )
            mock_roma_core.RecursiveSolver.return_value = mock_solver

            result = adapter.solve_with_constraints(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={},
                execution_mode=execution_mode,
                enable_checkpoints=True
            )

            assert result.success is True


class TestAnalyzeWithConstraints:
    """Test suite for analyze_with_constraints functionality."""

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_analyze_with_constraints_success(self, mock_roma_mcp_tools):
        """Test successful analyze with constraints."""
        with patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure MCP analyze result
            mock_result = {
                "analysis": {"complexity": "medium", "decomposition": ["step1", "step2"]},
                "status": "completed"
            }
            mock_roma_mcp_tools.analyze_with_roma.return_value = mock_result

            result = adapter.analyze_with_constraints(
                task="Analyze quicksort algorithm",
                analysis_type="decomposition",
                max_depth=3
            )

            # Should succeed
            assert result.success is True
            assert result.analysis == mock_result["analysis"]
            assert "roma_core" in result.layers_used

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    @pytest.mark.parametrize("analysis_type", ["decomposition", "complexity", "dependencies"])
    def test_analyze_with_different_analysis_types(self, analysis_type, mock_roma_mcp_tools):
        """Test analyze with different analysis types."""
        with patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure MCP analyze result
            mock_result = {
                "analysis": {"type": analysis_type, "result": "analysis_data"},
                "status": "completed"
            }
            mock_roma_mcp_tools.analyze_with_roma.return_value = mock_result

            result = adapter.analyze_with_constraints(
                task="Test analysis",
                analysis_type=analysis_type,
                max_depth=3
            )

            assert result.success is True
            assert result.analysis["type"] == analysis_type

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_analyze_with_input_validation(self, mock_guardrails_adapter, mock_roma_mcp_tools):
        """Test analyze with input validation."""
        with patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', True), \
             patch('reliability.roma_adapter.GUARDRAILS_AVAILABLE', True):
            adapter = RomaReliabilityAdapter(guardrails_adapter=mock_guardrails_adapter)

            # Configure input validation to pass
            mock_guardrails_adapter.validate_input.return_value = Mock(
                is_valid=True,
                failures=[]
            )

            # Configure MCP analyze result
            mock_result = {
                "analysis": {"complexity": "low"},
                "status": "completed"
            }
            mock_roma_mcp_tools.analyze_with_roma.return_value = mock_result

            result = adapter.analyze_with_constraints(
                task="Analyze task",
                analysis_type="complexity",
                max_depth=3
            )

            assert result.success is True
            assert "guardrails_input" in result.layers_used


class TestVerifyWithConstraints:
    """Test suite for verify_with_constraints functionality."""

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_verify_with_constraints_success(self, mock_roma_mcp_tools):
        """Test successful verify with constraints."""
        with patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure MCP verify result
            mock_result = {
                "verification": {"valid": True, "confidence": 0.95},
                "status": "completed"
            }
            mock_roma_mcp_tools.verify_with_roma.return_value = mock_result

            result = adapter.verify_with_constraints(
                solution="2 + 2 = 4",
                original_task="Solve: 2 + 2"
            )

            assert result.success is True
            assert result.result == mock_result

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_verify_with_criteria(self, mock_roma_mcp_tools):
        """Test verify with specific criteria."""
        with patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure MCP verify result
            mock_result = {
                "verification": {"valid": True, "criteria_met": ["correctness", "completeness"]},
                "status": "completed"
            }
            mock_roma_mcp_tools.verify_with_roma.return_value = mock_result

            result = adapter.verify_with_constraints(
                solution="2 + 2 = 4",
                original_task="Solve: 2 + 2",
                verification_criteria=["correctness", "completeness"]
            )

            assert result.success is True


class TestCritiqueWithConstraints:
    """Test suite for critique_with_constraints functionality."""

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_critique_with_constraints_success(self, mock_roma_mcp_tools):
        """Test successful critique with constraints."""
        with patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure MCP critique result
            mock_result = {
                "critique": {"focus": "comprehensive", "issues": ["performance"]},
                "status": "completed"
            }
            mock_roma_mcp_tools.critique_with_roma.return_value = mock_result

            result = adapter.critique_with_constraints(
                solution="Test solution",
                original_task="Original task",
                critique_focus="comprehensive"
            )

            assert result.success is True
            assert result.result == mock_result

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    @pytest.mark.parametrize("critique_focus", [
        "comprehensive", "security", "performance", "correctness"
    ])
    def test_critique_with_different_foci(self, critique_focus, mock_roma_mcp_tools):
        """Test critique with different focus areas."""
        with patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure MCP critique result
            mock_result = {
                "critique": {"focus": critique_focus, "issues": []},
                "status": "completed"
            }
            mock_roma_mcp_tools.critique_with_roma.return_value = mock_result

            result = adapter.critique_with_constraints(
                solution="Test solution",
                original_task="Original task",
                critique_focus=critique_focus
            )

            assert result.success is True
            assert result.result["critique"]["focus"] == critique_focus


class TestHealthChecks:
    """Test suite for health checking functionality."""

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_get_status(self, mock_roma_core, mock_roma_mcp_tools):
        """Test getting adapter status."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True), \
             patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', True), \
             patch('reliability.roma_adapter.get_roma_status') as mock_status:
            # Configure MCP status
            mock_status.return_value = {
                "available": True,
                "version": "1.0.0"
            }
            adapter = RomaReliabilityAdapter()

            status = adapter.get_status()

            assert isinstance(status, dict)
            assert "roma_available" in status
            assert "roma_core_available" in status
            assert "roma_mcp_available" in status
            assert "execution_mode" in status
            assert "lmql_available" in status
            assert "guardrails_available" in status
            assert "config" in status

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_health_check(self, mock_roma_core, mock_roma_mcp_tools):
        """Test comprehensive health check."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True), \
             patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', True), \
             patch('reliability.roma_adapter.get_roma_status') as mock_status:
            # Configure MCP status
            mock_status.return_value = {
                "available": True,
                "version": "1.0.0"
            }
            adapter = RomaReliabilityAdapter()

            health = adapter.health_check()

            assert isinstance(health, dict)
            assert "adapter_healthy" in health
            assert "execution_mode" in health
            assert "components" in health
            assert health["components"]["roma_core"]["healthy"] is True
            assert health["components"]["roma_mcp"]["healthy"] is True

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_health_check_unavailable_components(self):
        """Test health check with unavailable components."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', False), \
             patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', False):
            adapter = RomaReliabilityAdapter()

            health = adapter.health_check()

            assert health["adapter_healthy"] is False
            assert health["execution_mode"] == "unavailable"
            assert health["components"]["roma_core"]["healthy"] is False
            assert health["components"]["roma_mcp"]["healthy"] is False

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_is_available(self, mock_roma_core):
        """Test is_available method."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            assert adapter.is_available() is True

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_is_available_when_unavailable(self):
        """Test is_available method when ROMA is unavailable."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', False), \
             patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', False):
            adapter = RomaReliabilityAdapter()

            assert adapter.is_available() is False


class TestErrorHandling:
    """Test suite for error handling and edge cases."""

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_solve_with_empty_task(self, mock_roma_core):
        """Test solve with empty task."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            result = adapter.solve_with_constraints(
                task="",
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            # Should handle empty task gracefully
            assert result is not None

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_solve_with_none_parameters(self, mock_roma_core):
        """Test solve with None parameters."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            result = adapter.solve_with_constraints(
                task="Solve: 2 + 2",
                max_depth=None,
                constraints=None,
                execution_mode=None,
                enable_checkpoints=None
            )

            # Should handle None parameters gracefully
            assert result is not None

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_core_integration_exception_handling(self, mock_roma_core):
        """Test core integration exception handling."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure core to raise exception
            mock_roma_core.RecursiveSolver.side_effect = Exception("Core exception")

            result = adapter.solve_with_constraints(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            # Should fail gracefully
            assert result.success is False
            assert "Core exception" in result.error

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_mcp_exception_handling(self, mock_roma_mcp_tools):
        """Test MCP exception handling."""
        with patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure MCP to raise exception
            mock_roma_mcp_tools.solve_with_roma.side_effect = Exception("MCP exception")

            result = adapter.solve_with_constraints(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            # Should fail gracefully
            assert result.success is False
            assert "MCP exception" in result.error

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_unavailable_roma_handling(self):
        """Test handling when ROMA is completely unavailable."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', False), \
             patch('reliability.roma_adapter.ROMA_MCP_AVAILABLE', False):
            adapter = RomaReliabilityAdapter()

            result = adapter.solve_with_constraints(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            # Should fail gracefully with appropriate error
            assert result.success is False
            assert "ROMA not available" in result.error

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_agent_registration_failure(self, mock_roma_core):
        """Test agent registration failure handling."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure registry to fail
            adapter.registry = Mock()
            adapter.registry.register_agent.side_effect = Exception("Registration failed")

            # Should still work without registration
            result = adapter._solve_with_core_integration(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True,
                provider="openai",
                model="gpt-4",
                api_key="test",
                correlation_id="test_corr_id"
            )

            # Should still succeed without registration
            assert result["success"] is True


class TestStatisticsAndLogging:
    """Test suite for statistics tracking and logging."""

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_correlation_id_generation(self, mock_roma_core):
        """Test correlation ID generation."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            result = adapter.solve_with_constraints(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            assert result.correlation_id is not None
            assert "roma_solve_" in result.correlation_id

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_layer_usage_tracking(self, mock_roma_core):
        """Test layer usage tracking."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            result = adapter.solve_with_constraints(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            assert len(result.layers_used) > 0
            assert "roma_core" in result.layers_used

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_constraint_violations_tracking(self, mock_lmql_adapter, mock_roma_core):
        """Test constraint violations tracking."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True), \
             patch('reliability.roma_adapter.LMQL_AVAILABLE', True):
            adapter = RomaReliabilityAdapter(lmql_adapter=mock_lmql_adapter)

            # Configure mock result with violations
            mock_lmql_adapter.constrained_generation.return_value = Mock(
                success=True,
                text="Generated text",
                tokens_used=1500,  # Exceeds constraint
                constraint_violations=["max_tokens exceeded"]
            )

            result = adapter._solve_with_core_integration(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={"max_tokens": 1000},
                execution_mode="recursive",
                enable_checkpoints=True,
                provider="openai",
                model="gpt-4",
                api_key="test",
                correlation_id="test_corr_id"
            )

            assert len(result.get("constraint_violations", [])) > 0

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_validation_failures_tracking(self, mock_guardrails_adapter, mock_roma_core):
        """Test validation failures tracking."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True), \
             patch('reliability.roma_adapter.GUARDRAILS_AVAILABLE', True):
            adapter = RomaReliabilityAdapter(guardrails_adapter=mock_guardrails_adapter)

            # Configure output validation to fail
            mock_guardrails_adapter.validate_output.return_value = Mock(
                is_valid=False,
                failures=[{"validator": "json_structure", "message": "Invalid JSON"}],
                remediation_applied=None,
                output=None
            )

            result = adapter.solve_with_constraints(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            assert len(result.validation_failures) > 0

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_remediation_tracking(self, mock_guardrails_adapter, mock_roma_core):
        """Test remediation tracking."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True), \
             patch('reliability.roma_adapter.GUARDRAILS_AVAILABLE', True):
            adapter = RomaReliabilityAdapter(guardrails_adapter=mock_guardrails_adapter)

            # Configure output validation with remediation
            mock_guardrails_adapter.validate_output.return_value = Mock(
                is_valid=True,
                failures=[],
                remediation_applied="json_fix",
                output='{"fixed": "true"}'
            )

            result = adapter.solve_with_constraints(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            assert len(result.remediation_applied) > 0
            assert "output_remediated" in result.remediation_applied


class TestIntegration:
    """Test suite for integration with other reliability components."""

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_integration_with_lmql_adapter(self, mock_lmql_adapter, mock_roma_core):
        """Test integration with LMQL adapter."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter(lmql_adapter=mock_lmql_adapter)

            # Configure constraints
            mock_lmql_adapter.create_constraint.return_value = Mock(
                type="max_tokens",
                value=1000
            )

            result = adapter.solve_with_constraints(
                task="Generate text with constraints",
                max_depth=3,
                constraints={"max_tokens": 1000},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            assert result.success is True

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_integration_with_guardrails_adapter(self, mock_guardrails_adapter, mock_roma_core):
        """Test integration with Guardrails adapter."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter(guardrails_adapter=mock_guardrails_adapter)

            # Configure validation
            mock_guardrails_adapter.validate_input.return_value = Mock(
                is_valid=True,
                failures=[]
            )
            mock_guardrails_adapter.validate_output.return_value = Mock(
                is_valid=True,
                failures=[],
                remediation_applied=None,
                output=None
            )

            result = adapter.solve_with_constraints(
                task="Safe task",
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            assert result.success is True
            assert "guardrails_input" in result.layers_used
            assert "guardrails_output" in result.layers_used

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_unified_bridge_workflow(self, mock_lmql_adapter, mock_guardrails_adapter, mock_roma_core):
        """Test integration with Unified Bridge workflow."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter(
                lmql_adapter=mock_lmql_adapter,
                guardrails_adapter=mock_guardrails_adapter
            )

            # Configure all components
            mock_lmql_adapter.create_constraint.return_value = Mock(
                type="max_depth",
                value=3
            )

            mock_guardrails_adapter.validate_input.return_value = Mock(
                is_valid=True,
                failures=[]
            )

            mock_guardrails_adapter.validate_output.return_value = Mock(
                is_valid=True,
                failures=[],
                remediation_applied=None,
                output=None
            )

            # Configure solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                result="Test solution",
                status=Mock(value="completed")
            )
            mock_roma_core.RecursiveSolver.return_value = mock_solver

            result = adapter.solve_with_constraints(
                task="Unified bridge task",
                max_depth=3,
                constraints={"max_depth": 3},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            # Should use all layers
            assert result.success is True
            assert "guardrails_input" in result.layers_used
            assert "lmql_constraints" in result.layers_used
            assert "roma_core" in result.layers_used
            assert "guardrails_output" in result.layers_used


class TestPerformance:
    """Test suite for performance and optimization."""

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_batch_generation(self, mock_roma_core):
        """Test batch generation for performance."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                result="Batch solution",
                status=Mock(value="completed")
            )
            mock_roma_core.RecursiveSolver.return_value = mock_solver

            tasks = [f"Solve: {i} + {i}" for i in range(5)]
            results = []

            for task in tasks:
                result = adapter.solve_with_constraints(
                    task=task,
                    max_depth=3,
                    constraints={},
                    execution_mode="recursive",
                    enable_checkpoints=True
                )
                results.append(result)

            # Should process all tasks
            assert len(results) == 5
            assert all(result.success for result in results)

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_caching_mechanism(self, mock_roma_core):
        """Test caching mechanism for performance."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                result="Cached solution",
                status=Mock(value="completed")
            )
            mock_roma_core.RecursiveSolver.return_value = mock_solver

            # Same task multiple times
            for _ in range(3):
                result = adapter.solve_with_constraints(
                    task="Same task",
                    max_depth=3,
                    constraints={},
                    execution_mode="recursive",
                    enable_checkpoints=True
                )

            # Should cache results (implementation dependent)
            stats = adapter.get_performance_stats()
            assert isinstance(stats, dict)


class TestParameterizedTests:
    """Parameterized tests for different configurations."""

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    @pytest.mark.parametrize("max_depth", [1, 3, 5, 10])
    def test_different_max_depths(self, max_depth, mock_roma_core):
        """Test solve with different max depths."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                result=f"Solution with depth {max_depth}",
                status=Mock(value="completed")
            )
            mock_roma_core.RecursiveSolver.return_value = mock_solver

            result = adapter.solve_with_constraints(
                task="Solve: 2 + 2",
                max_depth=max_depth,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            assert result.success is True

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    @pytest.mark.parametrize("provider", ["openai", "anthropic", "google", "openrouter"])
    def test_different_providers(self, provider, mock_roma_core):
        """Test solve with different LLM providers."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                result=f"Solution with {provider}",
                status=Mock(value="completed")
            )
            mock_roma_core.RecursiveSolver.return_value = mock_solver

            result = adapter.solve_with_constraints(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True,
                provider=provider,
                model="gpt-4",
                api_key="test"
            )

            assert result.success is True

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    @pytest.mark.parametrize("enable_checkpoints", [True, False])
    def test_checkpoint_enabling(self, enable_checkpoints, mock_roma_core):
        """Test with checkpoints enabled/disabled."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                result="Solution",
                status=Mock(value="completed")
            )
            mock_roma_core.RecursiveSolver.return_value = mock_solver

            result = adapter.solve_with_constraints(
                task="Solve: 2 + 2",
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=enable_checkpoints
            )

            assert result.success is True


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_extremely_long_task(self, mock_roma_core):
        """Test with extremely long task."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                result="Solution for long task",
                status=Mock(value="completed")
            )
            mock_roma_core.RecursiveSolver.return_value = mock_solver

            very_long_task = "x" * 10000  # Very long task

            result = adapter.solve_with_constraints(
                task=very_long_task,
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            assert result.success is True

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_unicode_task(self, mock_roma_core):
        """Test with unicode task."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                result="Solution",
                status=Mock(value="completed")
            )
            mock_roma_core.RecursiveSolver.return_value = mock_solver

            unicode_task = "测试 task with 🚀 emojis and ñáéíóú"

            result = adapter.solve_with_constraints(
                task=unicode_task,
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            assert result.success is True

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_special_characters_task(self, mock_roma_core):
        """Test with special characters."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                result="Solution",
                status=Mock(value="completed")
            )
            mock_roma_core.RecursiveSolver.return_value = mock_solver

            special_task = "Task with !@#$%^&*()_+-=[]{}|;':\",./<>?`~"

            result = adapter.solve_with_constraints(
                task=special_task,
                max_depth=3,
                constraints={},
                execution_mode="recursive",
                enable_checkpoints=True
            )

            assert result.success is True


# =============================================================================
# UTILITIES
# =============================================================================

class TestUtilities:
    """Test suite for utility functions."""

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_result_conversion_to_dict(self):
        """Test result conversion to dictionary."""
        result = RomaSolutionResult(
            success=True,
            result={"test": "data"},
            task="Test task",
            layers_used=["roma_core"],
            correlation_id="test_corr"
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["success"] is True
        assert result_dict["result"] == {"test": "data"}
        assert result_dict["task"] == "Test task"
        assert result_dict["layers_used"] == ["roma_core"]

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_violation_detection(self):
        """Test violation detection utilities."""
        result = RomaSolutionResult(
            success=True,
            result={"test": "data"},
            task="Test task",
            constraint_violations=["max_depth exceeded", "token limit exceeded"]
        )

        assert result.has_violations() is True
        assert result.has_validation_failures() is False

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_validation_failure_detection(self):
        """Test validation failure detection."""
        result = RomaSolutionResult(
            success=True,
            result={"test": "data"},
            task="Test task",
            validation_failures=[{"validator": "toxic_language", "message": "Bad content"}]
        )

        assert result.has_validation_failures() is True

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_remediation_detection(self):
        """Test remediation detection."""
        result = RomaSolutionResult(
            success=True,
            result={"test": "data"},
            task="Test task",
            remediation_applied=["json_fix", "output_remediated"]
        )

        assert result.was_remediated() is True

    @pytest.mark.skipif(not ROMA_ADAPTER_AVAILABLE, reason="ROMA adapter not available")
    def test_convenience_functions(self, mock_roma_core):
        """Test convenience functions."""
        with patch('reliability.roma_adapter.ROMA_CORE_AVAILABLE', True):
            adapter = RomaReliabilityAdapter()

            # Configure solver
            mock_solver = Mock()
            mock_solver.solve.return_value = Mock(
                result="Solution",
                status=Mock(value="completed")
            )
            mock_roma_core.RecursiveSolver.return_value = mock_solver

            # Test solve_with_constraints function
            result = solve_with_constraints("Solve: 2 + 2", max_depth=3)

            assert result.success is True

            # Test get_default_adapter function
            default_adapter = get_default_adapter()
            assert default_adapter is not None


# =============================================================================
# SETUP/TEARDOWN
# =============================================================================

@pytest.fixture(scope="class", autouse=True)
def setup_class():
    """Setup for test class."""
    yield
    # Cleanup after all tests in class


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Clear any state between tests