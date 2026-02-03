"""
ROMA Plugin Loader

This module provides a plugin system for ROMA that allows loading and managing
external plugins without modifying ROMA core code. Plugins follow the Air Gap
principle - they don't directly import ROMA internals and receive dependencies
through dependency injection.

Plugin Interface:
    - create_plugin(): Factory function that returns plugin instance
    - Plugin class with methods:
        * initialize(roma_client, config): Initialize plugin with dependencies
        * register_commands(command_registry): Register commands with ROMA
        * register_panels(panel_registry): Register panels with ROMA
        * register_menus(menu_registry): Register menus with ROMA
        * get_info(): Return plugin metadata
        * shutdown(): Cleanup plugin resources

Author: OpenEvolve
Date: 2026-02-02
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Plugin Loader
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Callable
from enum import Enum

from loguru import logger


class PluginStatus(Enum):
    """Plugin status enumeration."""
    LOADING = "loading"
    LOADED = "loaded"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    REGISTERING = "registering"
    REGISTERED = "registered"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    min_roma_version: Optional[str] = None
    max_roma_version: Optional[str] = None


@dataclass
class PluginConfig:
    """Configuration for a plugin."""
    name: str
    enabled: bool = True
    module_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher priority loads first


@dataclass
class LoadedPlugin:
    """Represents a loaded plugin instance."""
    name: str
    instance: Any
    status: PluginStatus = PluginStatus.LOADING
    metadata: Optional[PluginMetadata] = None
    config: PluginConfig = None
    error: Optional[str] = None
    commands_registered: int = 0
    panels_registered: int = 0
    menus_registered: int = 0


class PluginLoader:
    """
    ROMA Plugin Loader - Manages loading, initialization, and registration of plugins.
    
    The plugin loader follows the Air Gap principle:
    - Plugins don't directly import ROMA internals
    - All dependencies are injected
    - Plugins are isolated from ROMA core
    """

    def __init__(
        self,
        roma_client: Optional[Any] = None,
        config_path: Optional[Path] = None
    ):
        """
        Initialize the plugin loader.
        
        Args:
            roma_client: ROMA client instance (injected)
            config_path: Path to plugins configuration file
        """
        self.roma_client = roma_client
        self.config_path = config_path or Path("config/plugins.yaml")
        self.plugins: Dict[str, LoadedPlugin] = {}
        self.plugin_configs: List[PluginConfig] = []
        
        # Registries for plugin components
        self.command_registry: Dict[str, Any] = {}
        self.panel_registry: Dict[str, Any] = {}
        self.menu_registry: Dict[str, Any] = {}
        
        self._initialized = False

    def load_config(self) -> bool:
        """
        Load plugin configuration from file.
        
        Returns:
            True if config loaded successfully, False otherwise
        """
        if not self.config_path.exists():
            logger.warning(f"Plugin config file not found: {self.config_path}")
            return False
            
        try:
            import yaml
            
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data or 'plugins' not in config_data:
                logger.warning("No plugins found in config file")
                return False
                
            self.plugin_configs = []
            for plugin_data in config_data['plugins']:
                self.plugin_configs.append(PluginConfig(
                    name=plugin_data.get('name'),
                    enabled=plugin_data.get('enabled', True),
                    module_path=plugin_data.get('module_path'),
                    config=plugin_data.get('config', {}),
                    priority=plugin_data.get('priority', 0)
                ))
            
            # Sort by priority (higher priority first)
            self.plugin_configs.sort(key=lambda x: x.priority, reverse=True)
            
            logger.info(f"Loaded {len(self.plugin_configs)} plugin configurations")
            return True
            
        except ImportError:
            logger.warning("PyYAML not installed, cannot load plugin config")
            return False
        except Exception as e:
            logger.error(f"Error loading plugin config: {e}")
            return False

    def load_plugins(self) -> Dict[str, LoadedPlugin]:
        """
        Load all enabled plugins from configuration.
        
        Returns:
            Dictionary of loaded plugins by name
        """
        if not self._initialized:
            self._initialize_registries()
            
        if not self.plugin_configs:
            self.load_config()
            
        loaded = {}
        
        for plugin_config in self.plugin_configs:
            if not plugin_config.enabled:
                logger.info(f"Plugin {plugin_config.name} is disabled, skipping")
                continue
                
            plugin = self._load_plugin(plugin_config)
            if plugin:
                loaded[plugin.name] = plugin
                self.plugins[plugin.name] = plugin
                
        logger.info(f"Loaded {len(loaded)} plugins successfully")
        return loaded

    def _load_plugin(self, config: PluginConfig) -> Optional[LoadedPlugin]:
        """
        Load a single plugin.
        
        Args:
            config: Plugin configuration
            
        Returns:
            LoadedPlugin instance or None if loading failed
        """
        logger.info(f"Loading plugin: {config.name}")
        
        plugin = LoadedPlugin(
            name=config.name,
            instance=None,
            status=PluginStatus.LOADING,
            config=config
        )
        
        try:
            # Import the plugin module
            if config.module_path:
                # Load from specified module path
                module = self._import_module_from_path(config.module_path)
            else:
                # Try to import from installed packages
                module = self._import_plugin_module(config.name)
                
            if module is None:
                raise ImportError(f"Could not import plugin module: {config.name}")
            
            plugin.status = PluginStatus.LOADED
            
            # Get the create_plugin factory function
            if not hasattr(module, 'create_plugin'):
                raise AttributeError(
                    f"Plugin {config.name} missing 'create_plugin' factory function"
                )
            
            # Create plugin instance
            plugin_instance = module.create_plugin()
            plugin.instance = plugin_instance
            plugin.status = PluginStatus.INITIALIZING
            
            # Initialize plugin with dependencies
            if hasattr(plugin_instance, 'initialize'):
                import asyncio
                if asyncio.iscoroutinefunction(plugin_instance.initialize):
                    # Async initialization
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(
                        plugin_instance.initialize(
                            roma_client=self.roma_client,
                            config=config.config
                        )
                    )
                else:
                    # Sync initialization
                    plugin_instance.initialize(
                        roma_client=self.roma_client,
                        config=config.config
                    )
            
            plugin.status = PluginStatus.INITIALIZED
            
            # Get plugin metadata
            if hasattr(plugin_instance, 'get_info'):
                info = plugin_instance.get_info()
                plugin.metadata = PluginMetadata(
                    name=info.get('name', config.name),
                    version=info.get('version', '0.0.0'),
                    description=info.get('description', ''),
                    author=info.get('author', 'Unknown'),
                    dependencies=info.get('dependencies', []),
                    min_roma_version=info.get('min_roma_version'),
                    max_roma_version=info.get('max_roma_version')
                )
            
            # Register plugin components
            self._register_plugin(plugin)
            
            plugin.status = PluginStatus.REGISTERED
            logger.info(f"Plugin {config.name} loaded and registered successfully")
            
            return plugin
            
        except Exception as e:
            plugin.status = PluginStatus.ERROR
            plugin.error = str(e)
            logger.error(f"Error loading plugin {config.name}: {e}", exc_info=True)
            return None

    def _import_module_from_path(self, module_path: str):
        """
        Import a module from a file path.
        
        Args:
            module_path: Path to the module file
            
        Returns:
            Module object or None
        """
        try:
            path = Path(module_path)
            if not path.exists():
                path = Path.cwd() / module_path
                
            if not path.exists():
                logger.error(f"Module path not found: {module_path}")
                return None
            
            spec = importlib.util.spec_from_file_location(
                path.stem,
                path
            )
            if spec is None or spec.loader is None:
                logger.error(f"Could not load spec for: {module_path}")
                return None
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[path.stem] = module
            spec.loader.exec_module(module)
            
            return module
            
        except Exception as e:
            logger.error(f"Error importing module from path: {e}")
            return None

    def _import_plugin_module(self, plugin_name: str):
        """
        Import a plugin module by name.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Module object or None
        """
        try:
            # Try common naming patterns
            patterns = [
                f"{plugin_name}",
                f"{plugin_name}_plugin",
                f"roma_{plugin_name}_plugin",
                f"roma_kg_plugin",  # Special case for KG plugin
            ]
            
            for pattern in patterns:
                try:
                    return importlib.import_module(pattern)
                except ImportError:
                    continue
                    
            return None
            
        except Exception as e:
            logger.error(f"Error importing plugin module: {e}")
            return None

    def _register_plugin(self, plugin: LoadedPlugin):
        """
        Register plugin components with registries.
        
        Args:
            plugin: LoadedPlugin instance
        """
        plugin.status = PluginStatus.REGISTERING
        instance = plugin.instance
        
        # Register commands
        if hasattr(instance, 'register_commands'):
            try:
                import asyncio
                if asyncio.iscoroutinefunction(instance.register_commands):
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(
                        instance.register_commands(self.command_registry)
                    )
                else:
                    result = instance.register_commands(self.command_registry)
                    
                if result:
                    plugin.commands_registered = len(self.command_registry) - plugin.commands_registered
                    logger.info(f"Registered {plugin.commands_registered} commands for {plugin.name}")
            except Exception as e:
                logger.error(f"Error registering commands for {plugin.name}: {e}")
        
        # Register panels
        if hasattr(instance, 'register_panels'):
            try:
                import asyncio
                if asyncio.iscoroutinefunction(instance.register_panels):
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(
                        instance.register_panels(self.panel_registry)
                    )
                else:
                    result = instance.register_panels(self.panel_registry)
                    
                if result:
                    plugin.panels_registered = len(self.panel_registry) - plugin.panels_registered
                    logger.info(f"Registered {plugin.panels_registered} panels for {plugin.name}")
            except Exception as e:
                logger.error(f"Error registering panels for {plugin.name}: {e}")
        
        # Register menus
        if hasattr(instance, 'register_menus'):
            try:
                import asyncio
                if asyncio.iscoroutinefunction(instance.register_menus):
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(
                        instance.register_menus(self.menu_registry)
                    )
                else:
                    result = instance.register_menus(self.menu_registry)
                    
                if result:
                    plugin.menus_registered = len(self.menu_registry) - plugin.menus_registered
                    logger.info(f"Registered {plugin.menus_registered} menus for {plugin.name}")
            except Exception as e:
                logger.error(f"Error registering menus for {plugin.name}: {e}")

    def _initialize_registries(self):
        """Initialize the component registries."""
        self.command_registry = {}
        self.panel_registry = {}
        self.menu_registry = {}
        self._initialized = True

    def get_plugin(self, name: str) -> Optional[LoadedPlugin]:
        """
        Get a loaded plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            LoadedPlugin instance or None
        """
        return self.plugins.get(name)

    def get_all_plugins(self) -> Dict[str, LoadedPlugin]:
        """Get all loaded plugins."""
        return self.plugins.copy()

    def get_status(self) -> Dict[str, Any]:
        """
        Get plugin loader status.
        
        Returns:
            Status dictionary with plugin information
        """
        return {
            "initialized": self._initialized,
            "total_plugins": len(self.plugins),
            "loaded_plugins": len([p for p in self.plugins.values() 
                                 if p.status == PluginStatus.REGISTERED]),
            "failed_plugins": len([p for p in self.plugins.values() 
                                 if p.status == PluginStatus.ERROR]),
            "plugins": {
                name: {
                    "status": plugin.status.value,
                    "metadata": {
                        "name": plugin.metadata.name if plugin.metadata else name,
                        "version": plugin.metadata.version if plugin.metadata else "unknown",
                        "description": plugin.metadata.description if plugin.metadata else ""
                    } if plugin.metadata else None,
                    "commands_registered": plugin.commands_registered,
                    "panels_registered": plugin.panels_registered,
                    "menus_registered": plugin.menus_registered,
                    "error": plugin.error
                }
                for name, plugin in self.plugins.items()
            }
        }

    async def shutdown(self):
        """Shutdown all plugins and cleanup resources."""
        logger.info("Shutting down plugin loader...")
        
        for plugin in self.plugins.values():
            try:
                if hasattr(plugin.instance, 'shutdown'):
                    if asyncio.iscoroutinefunction(plugin.instance.shutdown):
                        await plugin.instance.shutdown()
                    else:
                        plugin.instance.shutdown()
                logger.info(f"Plugin {plugin.name} shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down plugin {plugin.name}: {e}")
        
        self.plugins.clear()
        self._initialized = False
        logger.info("Plugin loader shutdown complete")


def create_plugin_loader(
    roma_client: Optional[Any] = None,
    config_path: Optional[Path] = None
) -> PluginLoader:
    """
    Factory function to create a plugin loader instance.
    
    Args:
        roma_client: ROMA client instance (injected)
        config_path: Path to plugins configuration file
        
    Returns:
        PluginLoader instance
    """
    return PluginLoader(roma_client=roma_client, config_path=config_path)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PluginLoader",
    "PluginStatus",
    "PluginMetadata",
    "PluginConfig",
    "LoadedPlugin",
    "create_plugin_loader"
]
