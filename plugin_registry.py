"""
Plugin Registry - License: Apache 2.0

Dynamic plugin loading system for OpenEvolve integrations.
Supports loading plugins from files, modules, and packages.

Dependencies:
- pydantic: MIT License
- importlib: Python Standard Library

Author: OpenEvolve
Date: 2026-02-02
"""

import asyncio
import importlib
import importlib.util
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from enum import Enum
import json
import sys

# Pydantic - MIT License
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PluginType(Enum):
    """Types of plugins supported."""
    DECOMPOSITION = "decomposition"
    KNOWLEDGE = "knowledge"
    INTEGRATION = "integration"
    TOOL = "tool"
    WORKFLOW = "workflow"
    MCP_TOOL = "mcp_tool"
    CUSTOM = "custom"


class PluginStatus(Enum):
    """Plugin lifecycle status."""
    REGISTERED = "registered"
    LOADING = "loading"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ERROR = "error"
    UNLOADED = "unloaded"


class PluginCapability(Enum):
    """Plugin capabilities that can be advertised."""
    DECOMPOSITION = "decomposition"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    REASONING = "reasoning"
    VERIFICATION = "verification"
    VISUALIZATION = "visualization"
    WORKFLOW = "workflow"


@dataclass
class PluginMetadata:
    """Plugin metadata information."""
    name: str
    version: str
    description: str
    author: str
    license: str
    plugin_type: PluginType
    capabilities: List[PluginCapability] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    homepage: Optional[str] = None
    repository: Optional[str] = None


@dataclass
class PluginInfo:
    """Runtime plugin information."""
    metadata: PluginMetadata
    status: PluginStatus = PluginStatus.REGISTERED
    instance: Optional[Any] = None
    error_message: Optional[str] = None
    load_time: Optional[float] = None
    module_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


class OpenEvolvePlugin(ABC):
    """
    Base class for all OpenEvolve plugins.
    
    Plugins must inherit from this class and implement required methods.
    
    Example:
        class MyPlugin(OpenEvolvePlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="my_plugin",
                    version="1.0.0",
                    description="My custom plugin",
                    author="Me",
                    license="Apache-2.0",
                    plugin_type=PluginType.CUSTOM
                )
            
            async def initialize(self, config: Dict[str, Any]) -> bool:
                # Initialize plugin
                return True
            
            async def shutdown(self) -> bool:
                # Cleanup
                return True
    """
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the plugin with configuration.
        
        Args:
            config: Plugin configuration dictionary
            
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Shutdown the plugin and cleanup resources.
        
        Returns:
            True if shutdown successful
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Health status dictionary
        """
        return {"status": "healthy"}
    
    def get_capabilities(self) -> List[PluginCapability]:
        """Get plugin capabilities."""
        return self.metadata.capabilities


class MCPToolPlugin(OpenEvolvePlugin):
    """
    Base class for MCP tool plugins.
    
    Provides easy way to register new MCP tools dynamically.
    """
    
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
    
    def register_tool(self, name: str, handler: Callable, schema: Dict[str, Any]) -> None:
        """Register an MCP tool."""
        self._tools[name] = {"handler": handler, "schema": schema}
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Register tools with MCP server."""
        try:
            from unified_mcp_server import get_unified_mcp_server
            from unified_mcp_server import ToolCategory
            
            server = get_unified_mcp_server()
            
            for name, tool_info in self._tools.items():
                server.register_tool(
                    name=name,
                    category=ToolCategory.CUSTOM,
                    description=tool_info["schema"].get("description", "Custom tool"),
                    handler=tool_info["handler"],
                    input_schema=tool_info["schema"]
                )
            
            return True
        except Exception as e:
            logger.error(f"Failed to register MCP tools: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Unregister tools."""
        # Tools remain registered until server restart
        return True


class PluginRegistry:
    """
    Central registry for dynamic plugin loading.
    
    Features:
    - Load plugins from Python modules
    - Load plugins from file paths
    - Hot reload support
    - Dependency management
    - Configuration management
    
    License: Apache 2.0
    """
    
    def __init__(self):
        self._plugins: Dict[str, PluginInfo] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._capabilities: Dict[PluginCapability, List[str]] = {
            cap: [] for cap in PluginCapability
        }
        
    def register(
        self,
        plugin_class: Type[OpenEvolvePlugin],
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a plugin class.
        
        Args:
            plugin_class: Plugin class (must inherit OpenEvolvePlugin)
            config: Optional configuration
            
        Returns:
            True if registered successfully
        """
        try:
            # Create instance to get metadata
            instance = plugin_class()
            metadata = instance.metadata
            
            if metadata.name in self._plugins:
                logger.warning(f"Plugin '{metadata.name}' already registered")
                return False
            
            info = PluginInfo(
                metadata=metadata,
                instance=instance,
                config=config or {}
            )
            
            self._plugins[metadata.name] = info
            
            # Index capabilities
            for cap in metadata.capabilities:
                self._capabilities[cap].append(metadata.name)
            
            logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register plugin: {e}")
            return False
    
    async def load_from_module(
        self,
        module_name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Load a plugin from a Python module.
        
        Args:
            module_name: Full module path (e.g., "my_package.my_plugin")
            config: Optional configuration
            
        Returns:
            True if loaded successfully
        """
        try:
            module = importlib.import_module(module_name)
            
            # Find plugin classes in module
            plugin_classes = []
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, OpenEvolvePlugin)
                    and obj is not OpenEvolvePlugin
                    and obj is not MCPToolPlugin
                ):
                    plugin_classes.append(obj)
            
            if not plugin_classes:
                logger.error(f"No plugin classes found in module: {module_name}")
                return False
            
            # Register first plugin class found
            success = self.register(plugin_classes[0], config)
            
            if success:
                info = self._plugins[plugin_classes[0]().metadata.name]
                info.module_path = module_name
                info.status = PluginStatus.LOADED
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to load plugin from module {module_name}: {e}")
            return False
    
    async def load_from_file(
        self,
        file_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Load a plugin from a Python file.
        
        Args:
            file_path: Path to Python file
            config: Optional configuration
            
        Returns:
            True if loaded successfully
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"Plugin file not found: {file_path}")
                return False
            
            # Load module from file
            spec = importlib.util.spec_from_file_location(
                file_path.stem,
                file_path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[file_path.stem] = module
            spec.loader.exec_module(module)
            
            # Find plugin classes
            plugin_classes = []
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, OpenEvolvePlugin)
                    and obj is not OpenEvolvePlugin
                    and obj is not MCPToolPlugin
                ):
                    plugin_classes.append(obj)
            
            if not plugin_classes:
                logger.error(f"No plugin classes found in file: {file_path}")
                return False
            
            # Register plugin
            success = self.register(plugin_classes[0], config)
            
            if success:
                info = self._plugins[plugin_classes[0]().metadata.name]
                info.module_path = str(file_path)
                info.status = PluginStatus.LOADED
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to load plugin from file {file_path}: {e}")
            return False
    
    async def load_from_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True
    ) -> List[str]:
        """
        Load all plugins from a directory.
        
        Args:
            directory: Directory path
            recursive: Search subdirectories
            
        Returns:
            List of loaded plugin names
        """
        directory = Path(directory)
        loaded = []
        
        pattern = "**/*.py" if recursive else "*.py"
        
        for file_path in directory.glob(pattern):
            if file_path.name.startswith("_"):
                continue
            
            success = await self.load_from_file(file_path)
            if success:
                # Get plugin name from loaded plugins
                for name, info in self._plugins.items():
                    if info.module_path == str(file_path):
                        loaded.append(name)
                        break
        
        return loaded
    
    async def initialize_plugin(self, name: str) -> bool:
        """
        Initialize a registered plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            True if initialized successfully
        """
        if name not in self._plugins:
            logger.error(f"Plugin not found: {name}")
            return False
        
        info = self._plugins[name]
        
        if info.status == PluginStatus.INITIALIZED:
            return True
        
        try:
            info.status = PluginStatus.LOADING
            
            import time
            start = time.time()
            
            success = await info.instance.initialize(info.config)
            
            info.load_time = time.time() - start
            
            if success:
                info.status = PluginStatus.INITIALIZED
                logger.info(f"Initialized plugin: {name}")
                
                # Execute hooks
                await self._execute_hooks("plugin_initialized", info)
                
                return True
            else:
                info.status = PluginStatus.ERROR
                info.error_message = "Initialization returned False"
                return False
                
        except Exception as e:
            info.status = PluginStatus.ERROR
            info.error_message = str(e)
            logger.error(f"Failed to initialize plugin {name}: {e}")
            return False
    
    async def initialize_all(self) -> Dict[str, bool]:
        """
        Initialize all registered plugins.
        
        Returns:
            Dict mapping plugin name to success status
        """
        results = {}
        
        # Sort by dependencies (simple approach)
        plugin_names = list(self._plugins.keys())
        
        for name in plugin_names:
            results[name] = await self.initialize_plugin(name)
        
        return results
    
    async def unload_plugin(self, name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            True if unloaded successfully
        """
        if name not in self._plugins:
            return False
        
        info = self._plugins[name]
        
        try:
            if info.status == PluginStatus.INITIALIZED:
                await info.instance.shutdown()
            
            # Remove from capabilities index
            for cap in info.metadata.capabilities:
                if name in self._capabilities[cap]:
                    self._capabilities[cap].remove(name)
            
            del self._plugins[name]
            logger.info(f"Unloaded plugin: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {name}: {e}")
            return False
    
    def get_plugin(self, name: str) -> Optional[PluginInfo]:
        """Get plugin information."""
        return self._plugins.get(name)
    
    def get_plugin_instance(self, name: str) -> Optional[OpenEvolvePlugin]:
        """Get plugin instance."""
        info = self._plugins.get(name)
        return info.instance if info else None
    
    def list_plugins(self, status: Optional[PluginStatus] = None) -> List[PluginInfo]:
        """List all plugins, optionally filtered by status."""
        plugins = list(self._plugins.values())
        if status:
            plugins = [p for p in plugins if p.status == status]
        return plugins
    
    def get_plugins_by_capability(self, capability: PluginCapability) -> List[str]:
        """Get plugin names that provide a specific capability."""
        return self._capabilities.get(capability, [])
    
    def register_hook(self, event: str, handler: Callable) -> None:
        """Register a hook for an event."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(handler)
    
    async def _execute_hooks(self, event: str, *args, **kwargs) -> None:
        """Execute hooks for an event."""
        for handler in self._hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(*args, **kwargs)
                else:
                    handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Hook error: {e}")
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status of all plugins."""
        return {
            name: {
                "status": info.status.value,
                "capabilities": [c.value for c in info.metadata.capabilities],
                "health": info.instance.health_check() if info.instance and info.status == PluginStatus.INITIALIZED else None
            }
            for name, info in self._plugins.items()
        }


# Global registry instance
_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """Get or create global plugin registry."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry


# Example plugins

class ExampleToolPlugin(MCPToolPlugin):
    """Example plugin showing MCP tool registration."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_tools",
            version="1.0.0",
            description="Example MCP tools plugin",
            author="OpenEvolve",
            license="Apache-2.0",
            plugin_type=PluginType.MCP_TOOL,
            capabilities=[PluginCapability.WORKFLOW]
        )
    
    def __init__(self):
        super().__init__()
        # Register tools in constructor
        self.register_tool(
            "example_hello",
            self.handle_hello,
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name to greet"}
                },
                "required": ["name"],
                "description": "Say hello to someone"
            }
        )
    
    async def handle_hello(self, args: Dict[str, Any]) -> str:
        """Handle hello tool."""
        name = args.get("name", "World")
        return f"Hello, {name}! This is an example plugin tool."
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize plugin."""
        # Register all tools with MCP server
        return await super().initialize(config)
    
    async def shutdown(self) -> bool:
        """Shutdown plugin."""
        return True


if __name__ == "__main__":
    # Demo usage
    async def main():
        registry = get_plugin_registry()
        
        # Register example plugin
        registry.register(ExampleToolPlugin)
        
        # Initialize all plugins
        results = await registry.initialize_all()
        
        print("Plugin initialization results:")
        for name, success in results.items():
            print(f"  {name}: {'✓' if success else '✗'}")
        
        # Print health
        print("\nPlugin health:")
        print(registry.get_health())
    
    asyncio.run(main())
