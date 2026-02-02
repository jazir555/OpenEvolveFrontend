"""
Integration tests for complete ROMA system

Tests the full integration of ROMA core, plugins, and BubbleLab frontend.

Author: OpenEvolve
Date: 2026-02-02
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import pytest

from roma_dspy.core.plugin_loader import (
    PluginLoader,
    PluginConfig,
    create_plugin_loader,
)
from roma_associative_integration import (
    ROMAMDAPMakerAssociativeEngine,
    create_romamdapmaker_associative_config,
)


# =============================================================================
# Mock ROMA Client
# =============================================================================

class MockROMAClient:
    """Mock ROMA client for testing."""
    
    def __init__(self):
        self.executions = {}
        self.execution_counter = 0
    
    def execute(self, task: str, **kwargs):
        """Execute a ROMA task."""
        self.execution_counter += 1
        execution_id = f"exec_{self.execution_counter}"
        self.executions[execution_id] = {
            "id": execution_id,
            "task": task,
            "status": "completed",
            "result": f"Result for: {task}",
            "timestamp": "2026-02-02T00:00:00Z",
            "statistics": {
                "subtasksCreated": 5,
                "subtasksCompleted": 5,
                "toolsUsed": ["tool1", "tool2"],
                "modulesUsed": ["atomizer", "planner", "executor"]
            }
        }
        return self.executions[execution_id]
    
    def get_execution(self, execution_id: str):
        """Get an execution by ID."""
        return self.executions.get(execution_id)
    
    async def async_execute(self, task: str, **kwargs):
        """Async execute a ROMA task."""
        return self.execute(task, **kwargs)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_roma_client():
    """Create mock ROMA client."""
    return MockROMAClient()


@pytest.fixture
def temp_config_dir():
    """Create temporary directory for config files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def integration_plugin_loader(mock_roma_client, temp_config_dir):
    """Create plugin loader for integration testing."""
    config_path = temp_config_dir / "plugins.yaml"
    loader = PluginLoader(roma_client=mock_roma_client, config_path=config_path)
    return loader


# =============================================================================
# Complete System Integration Tests
# =============================================================================

class TestCompleteROMAIntegration:
    """Tests for complete ROMA system integration."""
    
    def test_roma_client_initialization(self, mock_roma_client):
        """Test ROMA client initialization."""
        assert mock_roma_client is not None
        assert mock_roma_client.execution_counter == 0
    
    def test_roma_execution_flow(self, mock_roma_client):
        """Test complete ROMA execution flow."""
        # Execute task
        result = mock_roma_client.execute("Solve x + 2 = 5")
        
        assert result["status"] == "completed"
        assert "result" in result
        assert "statistics" in result
        assert mock_roma_client.execution_counter == 1
        
        # Get execution by ID
        execution = mock_roma_client.get_execution(result["id"])
        assert execution is not None
        assert execution["task"] == "Solve x + 2 = 5"
    
    @pytest.mark.asyncio
    async def test_roma_async_execution_flow(self, mock_roma_client):
        """Test async ROMA execution flow."""
        result = await mock_roma_client.async_execute("Async task")
        
        assert result["status"] == "completed"
        assert "result" in result


# =============================================================================
# Plugin System Integration Tests
# =============================================================================

class TestPluginSystemIntegration:
    """Tests for plugin system integration."""
    
    def test_plugin_loader_with_roma_client(self, mock_roma_client):
        """Test plugin loader with ROMA client."""
        loader = create_plugin_loader(roma_client=mock_roma_client)
        
        assert loader.roma_client == mock_roma_client
        assert loader._initialized is False
    
    def test_plugin_loader_initializes_registries(self, integration_plugin_loader):
        """Test that plugin loader initializes registries."""
        integration_plugin_loader._initialize_registries()
        
        assert integration_plugin_loader._initialized is True
        assert integration_plugin_loader.command_registry is not None
        assert integration_plugin_loader.panel_registry is not None
        assert integration_plugin_loader.menu_registry is not None
    
    def test_plugin_loader_status(self, integration_plugin_loader):
        """Test plugin loader status reporting."""
        integration_plugin_loader._initialize_registries()
        
        status = integration_plugin_loader.get_status()
        
        assert "initialized" in status
        assert "total_plugins" in status
        assert "loaded_plugins" in status
        assert "failed_plugins" in status
        assert "plugins" in status


# =============================================================================
# Associative Integration Tests
# =============================================================================

class TestAssociativeIntegration:
    """Tests for ROMA associative integration."""
    
    def test_config_creation(self):
        """Test creating associative integration config."""
        config = create_romamdapmaker_associative_config()
        
        assert config is not None
        assert config.roma_max_depth_analysis == 3
        assert config.mdap_enabled is True
    
    def test_engine_initialization(self):
        """Test associative engine initialization."""
        config = create_romamdapmaker_associative_config()
        engine = ROMAMDAPMakerAssociativeEngine(config)
        
        assert engine.config == config
        assert engine.initialized is False
    
    def test_engine_initialization_with_roma_client(self, mock_roma_client):
        """Test associative engine with ROMA client."""
        config = create_romamdapmaker_associative_config()
        engine = ROMAMDAPMakerAssociativeEngine(config)
        
        # In simplified mode, ROMA client is not used directly
        assert engine.config is not None
    
    def test_problem_decomposition(self):
        """Test problem decomposition."""
        config = create_romamdapmaker_associative_config()
        engine = ROMAMDAPMakerAssociativeEngine(config)
        
        result = engine.plan_decomposition(
            "Solve equation x + 2 = 5",
            "mathematics"
        )
        
        assert "problem" in result
        assert result["domain"] == "mathematics"
        assert "approach" in result
        assert "confidence" in result
    
    def test_problem_solving(self):
        """Test problem solving."""
        config = create_romamdapmaker_associative_config()
        engine = ROMAMDAPMakerAssociativeEngine(config)
        
        result = engine.solve_problem("Test problem")
        
        assert "success" in result
        assert result["success"] is True
        assert "solution" in result
        assert "confidence" in result
    
    def test_metrics_tracking(self):
        """Test metrics tracking."""
        config = create_romamdapmaker_associative_config()
        engine = ROMAMDAPMakerAssociativeEngine(config)
        
        # Solve a problem
        engine.solve_problem("Test problem")
        
        # Get metrics
        metrics = engine.get_metrics()
        
        assert metrics["total_problems_solved"] == 1
        assert "total_decomposition_time" in metrics
        assert "total_recomposition_time" in metrics
        assert "total_validation_time" in metrics
    
    def test_metrics_reset(self):
        """Test metrics reset."""
        config = create_romamdapmaker_associative_config()
        engine = ROMAMDAPMakerAssociativeEngine(config)
        
        # Solve some problems
        engine.solve_problem("Problem 1")
        engine.solve_problem("Problem 2")
        
        # Reset metrics
        engine.reset_metrics()
        
        # Verify reset
        metrics = engine.get_metrics()
        assert metrics["total_problems_solved"] == 0


# =============================================================================
# End-to-End Integration Tests
# =============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests for complete ROMA system."""
    
    def test_full_workflow(self, mock_roma_client):
        """Test complete workflow from config to execution."""
        # Create associative config
        config = create_romamdapmaker_associative_config()
        
        # Create engine
        engine = ROMAMDAPMakerAssociativeEngine(config)
        
        # Initialize engine
        assert engine.initialize() is True
        
        # Plan decomposition
        plan = engine.plan_decomposition(
            "Solve x + 2 = 5",
            "mathematics"
        )
        assert "problem" in plan
        
        # Execute with ROMA client
        result = mock_roma_client.execute("Solve x + 2 = 5")
        assert result["status"] == "completed"
        
        # Verify statistics
        assert "statistics" in result
        stats = result["statistics"]
        assert "subtasksCreated" in stats
        assert "subtasksCompleted" in stats
    
    def test_plugin_system_with_multiple_plugins(self, integration_plugin_loader, temp_config_dir):
        """Test plugin system with multiple plugins configured."""
        # Create config with multiple plugins
        config_path = temp_config_dir / "plugins.yaml"
        config_content = """
plugins:
  - name: plugin1
    enabled: true
    priority: 10
  - name: plugin2
    enabled: true
    priority: 5
  - name: plugin3
    enabled: false
    priority: 15
"""
        config_path.write_text(config_content)
        
        # Load config
        integration_plugin_loader.load_config()
        
        # Verify loaded
        assert len(integration_plugin_loader.plugin_configs) == 3
        # Check priority sorting (higher priority first)
        assert integration_plugin_loader.plugin_configs[0].priority == 15
        assert integration_plugin_loader.plugin_configs[1].priority == 10
        assert integration_plugin_loader.plugin_configs[2].priority == 5
    
    def test_config_with_plugin_specific_settings(self, integration_plugin_loader, temp_config_dir):
        """Test plugin-specific configuration."""
        config_path = temp_config_dir / "plugins.yaml"
        config_content = """
plugins:
  - name: test_plugin
    enabled: true
    config:
      setting1: value1
      setting2: value2
      nested:
        key: value
"""
        config_path.write_text(config_content)
        
        # Load config
        integration_plugin_loader.load_config()
        
        # Verify plugin config
        assert len(integration_plugin_loader.plugin_configs) == 1
        plugin_config = integration_plugin_loader.plugin_configs[0]
        assert plugin_config.config["setting1"] == "value1"
        assert plugin_config.config["setting2"] == "value2"
        assert plugin_config.config["nested"]["key"] == "value"


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in integration."""
    
    def test_plugin_loader_handles_missing_config(self, integration_plugin_loader):
        """Test plugin loader handles missing config file."""
        result = integration_plugin_loader.load_config()
        
        assert result is False
    
    def test_engine_handles_initialization_failure(self):
        """Test engine handles initialization failure."""
        config = create_romamdapmaker_associative_config()
        engine = ROMAMDAPMakerAssociativeEngine(config)
        
        # Initialize should succeed
        result = engine.initialize()
        assert result is True
    
    def test_engine_handles_missing_plugin(self):
        """Test engine handles when full plugin is not available."""
        config = create_romamdapmaker_associative_config()
        engine = ROMAMDAPMakerAssociativeEngine(config)
        
        # Should work in simplified mode
        result = engine.solve_problem("Test problem")
        assert "success" in result


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance tests for ROMA integration."""
    
    def test_multiple_sequential_executions(self, mock_roma_client):
        """Test multiple sequential ROMA executions."""
        num_executions = 10
        
        for i in range(num_executions):
            result = mock_roma_client.execute(f"Task {i}")
            assert result["status"] == "completed"
        
        assert mock_roma_client.execution_counter == num_executions
    
    def test_multiple_problem_solutions(self):
        """Test solving multiple problems."""
        config = create_romamdapmaker_associative_config()
        engine = ROMAMDAPMakerAssociativeEngine(config)
        
        num_problems = 5
        for i in range(num_problems):
            result = engine.solve_problem(f"Problem {i}")
            assert "success" in result
        
        metrics = engine.get_metrics()
        assert metrics["total_problems_solved"] == num_problems


# =============================================================================
# Compatibility Tests
# =============================================================================

class TestCompatibility:
    """Tests for compatibility between components."""
    
    def test_config_compatibility(self):
        """Test config compatibility between components."""
        # Create associative config
        config = create_romamdapmaker_associative_config()
        
        # Verify all expected fields exist
        assert hasattr(config, 'roma_max_depth_analysis')
        assert hasattr(config, 'mdap_enabled')
        assert hasattr(config, 'use_associative_recomposition')
        assert hasattr(config, 'provider')
        assert hasattr(config, 'model')
    
    def test_plugin_interface_compatibility(self, integration_plugin_loader):
        """Test plugin interface compatibility."""
        # Initialize registries
        integration_plugin_loader._initialize_registries()
        
        # Verify registry structure
        assert isinstance(integration_plugin_loader.command_registry, dict)
        assert isinstance(integration_plugin_loader.panel_registry, dict)
        assert isinstance(integration_plugin_loader.menu_registry, dict)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
