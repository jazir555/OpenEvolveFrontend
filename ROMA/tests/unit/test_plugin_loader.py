"""
Unit tests for ROMA Plugin Loader

Tests the plugin loading, initialization, and registration functionality.

Author: OpenEvolve
Date: 2026-02-02
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from dataclasses import dataclass

import pytest

from roma_dspy.core.plugin_loader import (
    PluginLoader,
    PluginStatus,
    PluginMetadata,
    PluginConfig,
    LoadedPlugin,
    create_plugin_loader,
)


# =============================================================================
# Mock Plugin for Testing
# =============================================================================

@dataclass
class MockPlugin:
    """Mock plugin for testing."""
    
    def __init__(self):
        self.initialized = False
        self.commands_registered = False
        self.panels_registered = False
        self.menus_registered = False
        self.shutdown_called = False
    
    def initialize(self, roma_client, config):
        """Initialize mock plugin."""
        self.initialized = True
        self.roma_client = roma_client
        self.config = config
        return True
    
    def register_commands(self, command_registry):
        """Register mock commands."""
        self.commands_registered = True
        command_registry['mock_command'] = self
        return True
    
    def register_panels(self, panel_registry):
        """Register mock panels."""
        self.panels_registered = True
        panel_registry['mock_panel'] = self
        return True
    
    def register_menus(self, menu_registry):
        """Register mock menus."""
        self.menus_registered = True
        menu_registry['mock_menu'] = self
        return True
    
    def get_info(self):
        """Get plugin info."""
        return {
            'name': 'mock_plugin',
            'version': '1.0.0',
            'description': 'Mock plugin for testing',
            'author': 'Test Author',
            'dependencies': [],
        }
    
    def shutdown(self):
        """Shutdown mock plugin."""
        self.shutdown_called = True


def create_plugin():
    """Factory function for mock plugin."""
    return MockPlugin()


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_roma_client():
    """Create mock ROMA client."""
    client = Mock()
    client.execute = Mock(return_value={'result': 'success'})
    return client


@pytest.fixture
def temp_config_dir():
    """Create temporary directory for config files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def plugin_loader(mock_roma_client, temp_config_dir):
    """Create plugin loader instance."""
    config_path = temp_config_dir / "plugins.yaml"
    loader = PluginLoader(roma_client=mock_roma_client, config_path=config_path)
    return loader


# =============================================================================
# PluginConfig Tests
# =============================================================================

class TestPluginConfig:
    """Tests for PluginConfig dataclass."""
    
    def test_default_config(self):
        """Test creating default plugin config."""
        config = PluginConfig(name="test_plugin")
        assert config.name == "test_plugin"
        assert config.enabled is True
        assert config.module_path is None
        assert config.config == {}
        assert config.priority == 0
    
    def test_config_with_values(self):
        """Test creating plugin config with values."""
        config = PluginConfig(
            name="test_plugin",
            enabled=False,
            module_path="/path/to/plugin",
            config={"key": "value"},
            priority=10
        )
        assert config.name == "test_plugin"
        assert config.enabled is False
        assert config.module_path == "/path/to/plugin"
        assert config.config == {"key": "value"}
        assert config.priority == 10


# =============================================================================
# PluginMetadata Tests
# =============================================================================

class TestPluginMetadata:
    """Tests for PluginMetadata dataclass."""
    
    def test_default_metadata(self):
        """Test creating default plugin metadata."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test Author"
        )
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test plugin"
        assert metadata.author == "Test Author"
        assert metadata.dependencies == []
        assert metadata.min_roma_version is None
        assert metadata.max_roma_version is None
    
    def test_metadata_with_dependencies(self):
        """Test creating metadata with dependencies."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            dependencies=["dep1", "dep2"],
            min_roma_version="1.0.0",
            max_roma_version="2.0.0"
        )
        assert metadata.dependencies == ["dep1", "dep2"]
        assert metadata.min_roma_version == "1.0.0"
        assert metadata.max_roma_version == "2.0.0"


# =============================================================================
# LoadedPlugin Tests
# =============================================================================

class TestLoadedPlugin:
    """Tests for LoadedPlugin dataclass."""
    
    def test_default_loaded_plugin(self):
        """Test creating default loaded plugin."""
        plugin = LoadedPlugin(name="test_plugin", instance=None)
        assert plugin.name == "test_plugin"
        assert plugin.instance is None
        assert plugin.status == PluginStatus.LOADING
        assert plugin.metadata is None
        assert plugin.error is None
        assert plugin.commands_registered == 0
        assert plugin.panels_registered == 0
        assert plugin.menus_registered == 0


# =============================================================================
# PluginLoader Tests - Configuration Loading
# =============================================================================

class TestPluginLoaderConfig:
    """Tests for plugin loader configuration loading."""
    
    def test_load_config_file_not_found(self, plugin_loader):
        """Test loading config when file doesn't exist."""
        result = plugin_loader.load_config()
        assert result is False
    
    def test_load_config_valid_yaml(self, plugin_loader, temp_config_dir):
        """Test loading valid YAML config."""
        config_path = temp_config_dir / "plugins.yaml"
        config_content = """
plugins:
  - name: test_plugin
    enabled: true
    priority: 10
"""
        config_path.write_text(config_content)
        
        result = plugin_loader.load_config()
        assert result is True
        assert len(plugin_loader.plugin_configs) == 1
        assert plugin_loader.plugin_configs[0].name == "test_plugin"
        assert plugin_loader.plugin_configs[0].enabled is True
        assert plugin_loader.plugin_configs[0].priority == 10
    
    def test_load_config_multiple_plugins(self, plugin_loader, temp_config_dir):
        """Test loading config with multiple plugins."""
        config_path = temp_config_dir / "plugins.yaml"
        config_content = """
plugins:
  - name: plugin1
    enabled: true
    priority: 10
  - name: plugin2
    enabled: false
    priority: 5
  - name: plugin3
    enabled: true
    priority: 15
"""
        config_path.write_text(config_content)
        
        result = plugin_loader.load_config()
        assert result is True
        assert len(plugin_loader.plugin_configs) == 3
        # Check sorted by priority
        assert plugin_loader.plugin_configs[0].name == "plugin3"
        assert plugin_loader.plugin_configs[0].priority == 15
        assert plugin_loader.plugin_configs[1].name == "plugin1"
        assert plugin_loader.plugin_configs[1].priority == 10
        assert plugin_loader.plugin_configs[2].name == "plugin2"
        assert plugin_loader.plugin_configs[2].priority == 5
    
    def test_load_config_with_plugin_config(self, plugin_loader, temp_config_dir):
        """Test loading config with plugin-specific config."""
        config_path = temp_config_dir / "plugins.yaml"
        config_content = """
plugins:
  - name: test_plugin
    enabled: true
    config:
      key1: value1
      key2: value2
"""
        config_path.write_text(config_content)
        
        result = plugin_loader.load_config()
        assert result is True
        assert plugin_loader.plugin_configs[0].config == {"key1": "value1", "key2": "value2"}


# =============================================================================
# PluginLoader Tests - Plugin Loading
# =============================================================================

class TestPluginLoaderLoading:
    """Tests for plugin loading functionality."""
    
    @patch('roma_dspy.core.plugin_loader.importlib.import_module')
    def test_load_plugin_success(self, mock_import, plugin_loader):
        """Test successful plugin loading."""
        # Create mock module
        mock_module = Mock()
        mock_module.create_plugin = Mock(return_value=MockPlugin())
        mock_import.return_value = mock_module
        
        config = PluginConfig(name="test_plugin")
        plugin = plugin_loader._load_plugin(config)
        
        assert plugin is not None
        assert plugin.name == "test_plugin"
        assert plugin.status == PluginStatus.REGISTERED
        assert plugin.error is None
    
    @patch('roma_dspy.core.plugin_loader.importlib.import_module')
    def test_load_plugin_missing_create_plugin(self, mock_import, plugin_loader):
        """Test plugin loading when create_plugin is missing."""
        mock_module = Mock()
        del mock_module.create_plugin  # Remove create_plugin
        mock_import.return_value = mock_module
        
        config = PluginConfig(name="test_plugin")
        plugin = plugin_loader._load_plugin(config)
        
        assert plugin is not None
        assert plugin.status == PluginStatus.ERROR
        assert "missing 'create_plugin'" in plugin.error
    
    @patch('roma_dspy.core.plugin_loader.importlib.import_module')
    def test_load_plugin_import_error(self, mock_import, plugin_loader):
        """Test plugin loading when import fails."""
        mock_import.side_effect = ImportError("Module not found")
        
        config = PluginConfig(name="test_plugin")
        plugin = plugin_loader._load_plugin(config)
        
        assert plugin is not None
        assert plugin.status == PluginStatus.ERROR
        assert "Could not import" in plugin.error
    
    @patch('roma_dspy.core.plugin_loader.importlib.import_module')
    def test_load_plugin_initialization_error(self, mock_import, plugin_loader):
        """Test plugin loading when initialization fails."""
        mock_plugin = MockPlugin()
        mock_plugin.initialize = Mock(side_effect=Exception("Init failed"))
        
        mock_module = Mock()
        mock_module.create_plugin = Mock(return_value=mock_plugin)
        mock_import.return_value = mock_module
        
        config = PluginConfig(name="test_plugin")
        plugin = plugin_loader._load_plugin(config)
        
        assert plugin is not None
        assert plugin.status == PluginStatus.ERROR
        assert "Init failed" in plugin.error


# =============================================================================
# PluginLoader Tests - Plugin Registration
# =============================================================================

class TestPluginLoaderRegistration:
    """Tests for plugin registration functionality."""
    
    def test_register_plugin_commands(self, plugin_loader):
        """Test registering plugin commands."""
        mock_plugin = MockPlugin()
        loaded_plugin = LoadedPlugin(
            name="test_plugin",
            instance=mock_plugin,
            status=PluginStatus.INITIALIZED
        )
        
        plugin_loader._register_plugin(loaded_plugin)
        
        assert loaded_plugin.status == PluginStatus.REGISTERED
        assert loaded_plugin.commands_registered == 1
        assert 'mock_command' in plugin_loader.command_registry
    
    def test_register_plugin_panels(self, plugin_loader):
        """Test registering plugin panels."""
        mock_plugin = MockPlugin()
        loaded_plugin = LoadedPlugin(
            name="test_plugin",
            instance=mock_plugin,
            status=PluginStatus.INITIALIZED
        )
        
        plugin_loader._register_plugin(loaded_plugin)
        
        assert loaded_plugin.panels_registered == 1
        assert 'mock_panel' in plugin_loader.panel_registry
    
    def test_register_plugin_menus(self, plugin_loader):
        """Test registering plugin menus."""
        mock_plugin = MockPlugin()
        loaded_plugin = LoadedPlugin(
            name="test_plugin",
            instance=mock_plugin,
            status=PluginStatus.INITIALIZED
        )
        
        plugin_loader._register_plugin(loaded_plugin)
        
        assert loaded_plugin.menus_registered == 1
        assert 'mock_menu' in plugin_loader.menu_registry


# =============================================================================
# PluginLoader Tests - Plugin Management
# =============================================================================

class TestPluginLoaderManagement:
    """Tests for plugin management functionality."""
    
    def test_get_plugin(self, plugin_loader):
        """Test getting a loaded plugin."""
        plugin = LoadedPlugin(name="test_plugin", instance=None)
        plugin_loader.plugins["test_plugin"] = plugin
        
        result = plugin_loader.get_plugin("test_plugin")
        assert result is not None
        assert result.name == "test_plugin"
    
    def test_get_plugin_not_found(self, plugin_loader):
        """Test getting a non-existent plugin."""
        result = plugin_loader.get_plugin("nonexistent")
        assert result is None
    
    def test_get_all_plugins(self, plugin_loader):
        """Test getting all loaded plugins."""
        plugin1 = LoadedPlugin(name="plugin1", instance=None)
        plugin2 = LoadedPlugin(name="plugin2", instance=None)
        plugin_loader.plugins["plugin1"] = plugin1
        plugin_loader.plugins["plugin2"] = plugin2
        
        result = plugin_loader.get_all_plugins()
        assert len(result) == 2
        assert "plugin1" in result
        assert "plugin2" in result
    
    def test_get_status(self, plugin_loader):
        """Test getting plugin loader status."""
        plugin_loader._initialize_registries()
        
        plugin1 = LoadedPlugin(
            name="plugin1",
            instance=MockPlugin(),
            status=PluginStatus.REGISTERED
        )
        plugin2 = LoadedPlugin(
            name="plugin2",
            instance=None,
            status=PluginStatus.ERROR
        )
        plugin_loader.plugins["plugin1"] = plugin1
        plugin_loader.plugins["plugin2"] = plugin2
        
        status = plugin_loader.get_status()
        
        assert status["initialized"] is True
        assert status["total_plugins"] == 2
        assert status["loaded_plugins"] == 1
        assert status["failed_plugins"] == 1
        assert "plugins" in status


# =============================================================================
# PluginLoader Tests - Async Methods
# =============================================================================

class TestPluginLoaderAsync:
    """Tests for async plugin loader methods."""
    
    @pytest.mark.asyncio
    async def test_async_plugin_initialization(self, plugin_loader):
        """Test async plugin initialization."""
        async_plugin = Mock()
        async_plugin.initialize = AsyncMock(return_value=True)
        async_plugin.get_info = Mock(return_value={
            'name': 'async_plugin',
            'version': '1.0.0',
            'description': 'Async plugin',
            'author': 'Test Author',
            'dependencies': [],
        })
        
        with patch('roma_dspy.core.plugin_loader.importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.create_plugin = Mock(return_value=async_plugin)
            mock_import.return_value = mock_module
            
            config = PluginConfig(name="async_plugin")
            plugin = plugin_loader._load_plugin(config)
            
            assert plugin is not None
            assert plugin.status == PluginStatus.REGISTERED
            assert async_plugin.initialize.called
    
    @pytest.mark.asyncio
    async def test_async_plugin_shutdown(self, plugin_loader):
        """Test async plugin shutdown."""
        async_plugin = Mock()
        async_plugin.shutdown = AsyncMock()
        
        plugin_loader.plugins["async_plugin"] = LoadedPlugin(
            name="async_plugin",
            instance=async_plugin,
            status=PluginStatus.REGISTERED
        )
        
        await plugin_loader.shutdown()
        
        assert async_plugin.shutdown.called


# =============================================================================
# PluginLoader Tests - Factory Function
# =============================================================================

class TestPluginLoaderFactory:
    """Tests for plugin loader factory function."""
    
    def test_create_plugin_loader(self, mock_roma_client):
        """Test creating plugin loader via factory function."""
        loader = create_plugin_loader(roma_client=mock_roma_client)
        
        assert loader is not None
        assert isinstance(loader, PluginLoader)
        assert loader.roma_client == mock_roma_client
    
    def test_create_plugin_loader_with_config_path(self, mock_roma_client, temp_config_dir):
        """Test creating plugin loader with custom config path."""
        config_path = temp_config_dir / "custom_plugins.yaml"
        loader = create_plugin_loader(
            roma_client=mock_roma_client,
            config_path=config_path
        )
        
        assert loader.config_path == config_path


# =============================================================================
# Integration Tests
# =============================================================================

class TestPluginLoaderIntegration:
    """Integration tests for plugin loader."""
    
    def test_full_plugin_loading_workflow(self, plugin_loader, temp_config_dir):
        """Test complete plugin loading workflow."""
        # Create config file
        config_path = temp_config_dir / "plugins.yaml"
        config_content = """
plugins:
  - name: plugin1
    enabled: true
    priority: 10
  - name: plugin2
    enabled: false
    priority: 5
"""
        config_path.write_text(config_content)
        
        # Load config
        plugin_loader.load_config()
        assert len(plugin_loader.plugin_configs) == 2
        
        # Check status
        status = plugin_loader.get_status()
        assert status["initialized"] is True
    
    @pytest.mark.asyncio
    async def test_shutdown_all_plugins(self, plugin_loader):
        """Test shutting down all plugins."""
        # Add plugins
        plugin1 = MockPlugin()
        plugin2 = MockPlugin()
        plugin_loader.plugins["plugin1"] = LoadedPlugin(
            name="plugin1",
            instance=plugin1,
            status=PluginStatus.REGISTERED
        )
        plugin_loader.plugins["plugin2"] = LoadedPlugin(
            name="plugin2",
            instance=plugin2,
            status=PluginStatus.REGISTERED
        )
        
        # Shutdown
        await plugin_loader.shutdown()
        
        # Check shutdown called
        assert plugin1.shutdown_called is True
        assert plugin2.shutdown_called is True
        assert len(plugin_loader.plugins) == 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
