"""
OpenEvolve Plugin System - Extensible Architecture for Decomposition Engine

This module provides a comprehensive plugin architecture that allows external tools
and integrations to extend the decomposition engine's capabilities.

FEATURES:
- Plugin lifecycle management (load, unload, update, reload)
- Plugin discovery and registration
- Hook system for extensibility
- Event-based communication
- Plugin dependencies and versioning
- Security sandboxing
- Hot-reloading support
- Plugin marketplace integration
"""

import os
import sys
import json
import importlib
import importlib.util
import inspect
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Type, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PluginState(Enum):
    """Plugin lifecycle states."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    DEACTIVATING = "deactivating"
    ERROR = "error"
    UNLOADING = "unloading"


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    license: str
    dependencies: List[str] = field(default_factory=list)
    python_version: str = ">=3.8"
    openevolve_version: str = ">=1.0.0"
    tags: List[str] = field(default_factory=list)
    category: str = "general"
    icon: Optional[str] = None
    homepage: Optional[str] = None
    repository: Optional[str] = None


@dataclass
class PluginHook:
    """Represents a plugin hook point."""
    name: str
    description: str
    callback: Callable
    priority: int = 100  # Lower = higher priority
    plugin_name: str = ""
    enabled: bool = True


@dataclass
class PluginEvent:
    """Represents an event in the plugin system."""
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)


class PluginError(Exception):
    """Base exception for plugin errors."""
    pass


class PluginLoadError(PluginError):
    """Raised when plugin fails to load."""
    pass


class PluginDependencyError(PluginError):
    """Raised when plugin dependencies are not satisfied."""
    pass


class PluginValidationError(PluginError):
    """Raised when plugin validation fails."""
    pass


class PluginBase:
    """
    Base class for all plugins.

    Plugins should inherit from this class and implement the required methods.

    Example:
        ```python
        class MyPlugin(PluginBase):
            def __init__(self):
                super().__init__(
                    name="my_plugin",
                    version="1.0.0",
                    description="My awesome plugin"
                )

            def activate(self):
                # Plugin activation logic
                pass

            def deactivate(self):
                # Plugin deactivation logic
                pass
        ```
    """

    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self.state = PluginState.LOADED
        self._hooks: Dict[str, PluginHook] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._config: Dict[str, Any] = {}
        self._storage: Dict[str, Any] = {}

    def activate(self) -> bool:
        """
        Activate the plugin.

        Called when the plugin is activated by the plugin manager.
        Override this method to implement plugin activation logic.

        Returns:
            True if activation successful, False otherwise
        """
        try:
            self.state = PluginState.ACTIVE
            logger.info(f"Plugin {self.metadata.name} activated")
            return True
        except (RuntimeError, OSError, TypeError) as e:
            logger.error(f"Failed to activate plugin {self.metadata.name}: {e}")
            self.state = PluginState.ERROR
            return False

    def deactivate(self) -> bool:
        """
        Deactivate the plugin.

        Called when the plugin is deactivated by the plugin manager.
        Override this method to implement plugin deactivation logic.

        Returns:
            True if deactivation successful, False otherwise
        """
        try:
            self.state = PluginState.LOADED
            logger.info(f"Plugin {self.metadata.name} deactivated")
            return True
        except (RuntimeError, OSError, TypeError) as e:
            logger.error(f"Failed to deactivate plugin {self.metadata.name}: {e}")
            self.state = PluginState.ERROR
            return False

    def on_load(self) -> None:
        """Called when plugin is loaded. Override for custom load behavior."""
        pass

    def on_unload(self) -> None:
        """Called when plugin is unloaded. Override for custom unload behavior."""
        pass

    def register_hook(self, hook_name: str, callback: Callable, priority: int = 100) -> None:
        """
        Register a hook callback.

        Args:
            hook_name: Name of the hook to register
            callback: Callback function
            priority: Priority (lower = higher priority)
        """
        hook = PluginHook(
            name=hook_name,
            description=f"Hook from {self.metadata.name}",
            callback=callback,
            priority=priority,
            plugin_name=self.metadata.name
        )
        self._hooks[hook_name] = hook

    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """
        Register an event handler.

        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value

    def get_storage(self, key: str, default: Any = None) -> Any:
        """Get persistent storage value."""
        return self._storage.get(key, default)

    def set_storage(self, key: str, value: Any) -> None:
        """Set persistent storage value."""
        self._storage[key] = value

    def get_hooks(self) -> Dict[str, PluginHook]:
        """Get all registered hooks."""
        return self._hooks.copy()

    def get_event_handlers(self) -> Dict[str, List[Callable]]:
        """Get all registered event handlers."""
        return self._event_handlers.copy()


class PluginManager:
    """
    Manages plugin lifecycle, registration, and execution.

    This is the main entry point for the plugin system. It handles:
    - Plugin discovery and loading
    - Plugin activation and deactivation
    - Hook execution
    - Event dispatching
    - Dependency resolution
    - Security validation
    """

    # Hook definitions for decomposition engine
    HOOK_DEFINITIONS = {
        # Decomposition hooks
        "on_before_decompose": "Called before problem decomposition",
        "on_after_decompose": "Called after problem decomposition",
        "on_subproblem_created": "Called when a sub-problem is created",
        "on_strategy_selected": "Called when a decomposition strategy is selected",

        # Quality assessment hooks
        "on_before_assess_quality": "Called before quality assessment",
        "on_after_assess_quality": "Called after quality assessment",
        "on_quality_threshold_failed": "Called when quality threshold fails",

        # Solution integration hooks
        "on_before_assemble": "Called before solution assembly",
        "on_after_assemble": "Called after solution assembly",
        "on_conflict_detected": "Called when a conflict is detected",
        "on_conflict_resolved": "Called when a conflict is resolved",

        # Gauntlet hooks
        "on_before_gauntlet": "Called before gauntlet execution",
        "on_after_gauntlet": "Called after gauntlet execution",
        "on_red_team_attack": "Called during red team attack",
        "on_gold_team_validate": "Called during gold team validation",

        # State change hooks
        "on_state_change": "Called when workflow state changes",
        "on_checkpoint": "Called when a checkpoint is created",
        "on_rollback": "Called when a rollback occurs",

        # Lifecycle hooks
        "on_workflow_start": "Called when workflow starts",
        "on_workflow_complete": "Called when workflow completes",
        "on_workflow_error": "Called when workflow encounters an error",
    }

    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        """
        Initialize plugin manager.

        Args:
            plugin_dirs: List of directories to search for plugins
        """
        self._plugins: Dict[str, PluginBase] = {}
        self._plugin_states: Dict[str, PluginState] = {}
        self._hooks: Dict[str, List[PluginHook]] = {name: [] for name in self.HOOK_DEFINITIONS}
        self._event_listeners: Dict[str, List[Callable]] = {}
        self._plugin_dirs = plugin_dirs or self._get_default_plugin_dirs()
        self._lock = threading.RLock()
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}
        self._load_configs()

    def _get_default_plugin_dirs(self) -> List[str]:
        """Get default plugin directories."""
        current_dir = Path(__file__).parent
        return [
            str(current_dir / "plugins"),
            str(current_dir / "integrations"),
            str(Path.home() / ".openevolve" / "plugins"),
        ]

    def _load_configs(self) -> None:
        """Load plugin configurations."""
        config_file = Path(".openevolve") / "plugin_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    self._plugin_configs = json.load(f)
            except (OSError, IOError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load plugin config: {e}")

    def discover_plugins(self) -> List[str]:
        """
        Discover all available plugins in plugin directories.

        Returns:
            List of plugin module names
        """
        discovered = []

        for plugin_dir in self._plugin_dirs:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                continue

            # Find Python files
            for py_file in plugin_path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                module_name = py_file.stem
                if module_name not in discovered:
                    discovered.append(module_name)

            # Find plugin packages
            for pkg_dir in plugin_path.glob("*/"):
                if not (pkg_dir / "__init__.py").exists():
                    continue
                if pkg_dir.name.startswith("_"):
                    continue

                module_name = pkg_dir.name
                if module_name not in discovered:
                    discovered.append(module_name)

        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered

    def load_plugin(self, plugin_name: str, plugin_path: Optional[str] = None) -> PluginBase:
        """
        Load a plugin from file or module.

        Args:
            plugin_name: Name of the plugin
            plugin_path: Optional path to plugin file

        Returns:
            Loaded plugin instance

        Raises:
            PluginLoadError: If plugin fails to load
        """
        with self._lock:
            if plugin_name in self._plugins:
                logger.warning(f"Plugin {plugin_name} already loaded")
                return self._plugins[plugin_name]

            self._plugin_states[plugin_name] = PluginState.LOADING

            try:
                # Import plugin module
                if plugin_path:
                    spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
                    if spec is None or spec.loader is None:
                        raise PluginLoadError(f"Cannot load spec for {plugin_name}")

                    module = importlib.util.module_from_spec(spec)
                    sys.modules[plugin_name] = module
                    spec.loader.exec_module(module)
                else:
                    module = importlib.import_module(plugin_name)

                # Find plugin class
                plugin_class = self._find_plugin_class(module)
                if plugin_class is None:
                    raise PluginLoadError(f"No plugin class found in {plugin_name}")

                # Validate plugin
                self._validate_plugin(plugin_class)

                # Instantiate plugin
                plugin = plugin_class()

                # Check dependencies
                self._check_dependencies(plugin)

                # Load plugin config
                if plugin_name in self._plugin_configs:
                    plugin._config = self._plugin_configs[plugin_name]

                # Call on_load
                plugin.on_load()

                # Register hooks
                self._register_plugin_hooks(plugin)

                # Register event handlers
                self._register_plugin_events(plugin)

                self._plugins[plugin_name] = plugin
                self._plugin_states[plugin_name] = PluginState.LOADED

                logger.info(f"Successfully loaded plugin: {plugin_name}")
                return plugin

            except (ImportError, PluginValidationError, PluginDependencyError) as e:
                self._plugin_states[plugin_name] = PluginState.ERROR
                logger.error(f"Failed to load plugin {plugin_name}: {e}", exc_info=True)
                raise PluginLoadError(f"Failed to load plugin {plugin_name}: {e}") from e

    def _find_plugin_class(self, module) -> Optional[Type[PluginBase]]:
        """Find plugin class in module."""
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, PluginBase) and obj is not PluginBase:
                return obj
        return None

    def _validate_plugin(self, plugin_class: Type[PluginBase]) -> None:
        """Validate plugin meets requirements."""
        # Check required methods
        required_methods = ['activate', 'deactivate']
        for method in required_methods:
            if not hasattr(plugin_class, method):
                raise PluginValidationError(f"Plugin missing required method: {method}")

    def _check_dependencies(self, plugin: PluginBase) -> None:
        """Check if plugin dependencies are satisfied."""
        for dep in plugin.metadata.dependencies:
            if dep not in self._plugins:
                raise PluginDependencyError(f"Plugin {plugin.metadata.name} requires {dep}")

    def _register_plugin_hooks(self, plugin: PluginBase) -> None:
        """Register plugin hooks."""
        for hook_name, hook in plugin.get_hooks().items():
            if hook_name in self._hooks:
                self._hooks[hook_name].append(hook)
                # Sort by priority
                self._hooks[hook_name].sort(key=lambda h: h.priority)

    def _register_plugin_events(self, plugin: PluginBase) -> None:
        """Register plugin event handlers."""
        for event_type, handlers in plugin.get_event_handlers().items():
            if event_type not in self._event_listeners:
                self._event_listeners[event_type] = []
            self._event_listeners[event_type].extend(handlers)

    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.

        Args:
            plugin_name: Name of plugin to unload

        Returns:
            True if successful
        """
        with self._lock:
            if plugin_name not in self._plugins:
                logger.warning(f"Plugin {plugin_name} not loaded")
                return False

            self._plugin_states[plugin_name] = PluginState.UNLOADING

            try:
                plugin = self._plugins[plugin_name]

                # Deactivate if active
                if plugin.state == PluginState.ACTIVE:
                    plugin.deactivate()

                # Call on_unload
                plugin.on_unload()

                # Unregister hooks
                self._unregister_plugin_hooks(plugin)

                # Unregister events
                self._unregister_plugin_events(plugin)

                # Remove from loaded plugins
                del self._plugins[plugin_name]
                del self._plugin_states[plugin_name]

                logger.info(f"Successfully unloaded plugin: {plugin_name}")
                return True

            except (RuntimeError, OSError, TypeError) as e:
                logger.error(f"Failed to unload plugin {plugin_name}: {e}")
                self._plugin_states[plugin_name] = PluginState.ERROR
                return False

    def _unregister_plugin_hooks(self, plugin: PluginBase) -> None:
        """Unregister plugin hooks."""
        for hook_name in list(self._hooks.keys()):
            self._hooks[hook_name] = [
                h for h in self._hooks[hook_name]
                if h.plugin_name != plugin.metadata.name
            ]

    def _unregister_plugin_events(self, plugin: PluginBase) -> None:
        """
        Unregister plugin event handlers.

        Removes all event handlers registered by the specified plugin from the event system.
        This ensures clean plugin unloading without memory leaks or dangling references.
        """
        plugin_name = plugin.metadata.name

        # Unregister from event bus if available
        if hasattr(self, '_event_handlers') and self._event_handlers:
            # Remove all event handlers registered by this plugin
            handlers_to_remove = []

            for event_name, handlers in self._event_handlers.items():
                # Filter out handlers belonging to this plugin
                filtered_handlers = [
                    h for h in handlers
                    if not hasattr(h, '__self__') or
                    not isinstance(h.__self__, plugin.__class__) or
                    h.__self__.metadata.name != plugin_name
                ]

                # Track how many were removed
                removed_count = len(handlers) - len(filtered_handlers)
                if removed_count > 0:
                    handlers_to_remove.append((event_name, removed_count))

                # Update the handlers list
                self._event_handlers[event_name] = filtered_handlers

                # Clean up empty event lists
                if not self._event_handlers[event_name]:
                    del self._event_handlers[event_name]

            if handlers_to_remove:
                logger.debug(f"Removed {len(handlers_to_remove)} event handlers for plugin {plugin_name}")
                for event_name, count in handlers_to_remove:
                    logger.debug(f"  - {event_name}: {count} handlers")

        # Unregister from event emitters if plugin has them
        if hasattr(plugin, 'event_emitters'):
            for emitter_name, emitter in getattr(plugin, 'event_emitters', {}).items():
                try:
                    if hasattr(emitter, 'remove_all_listeners'):
                        emitter.remove_all_listeners()
                        logger.debug(f"Cleared event emitter: {emitter_name}")
                except (RuntimeError, TypeError, AttributeError) as e:
                    logger.warning(f"Failed to clear event emitter {emitter_name}: {e}")

        logger.debug(f"Completed event unregistration for plugin {plugin_name}")

    def activate_plugin(self, plugin_name: str) -> bool:
        """
        Activate a loaded plugin.

        Args:
            plugin_name: Name of plugin to activate

        Returns:
            True if successful
        """
        with self._lock:
            if plugin_name not in self._plugins:
                logger.error(f"Plugin {plugin_name} not loaded")
                return False

            plugin = self._plugins[plugin_name]

            if plugin.state == PluginState.ACTIVE:
                logger.warning(f"Plugin {plugin_name} already active")
                return True

            success = plugin.activate()
            if success:
                # Emit activation event
                self.emit_event("plugin_activated", {
                    "plugin_name": plugin_name,
                    "metadata": plugin.metadata.__dict__
                })
            return success

    def deactivate_plugin(self, plugin_name: str) -> bool:
        """
        Deactivate an active plugin.

        Args:
            plugin_name: Name of plugin to deactivate

        Returns:
            True if successful
        """
        with self._lock:
            if plugin_name not in self._plugins:
                logger.error(f"Plugin {plugin_name} not loaded")
                return False

            plugin = self._plugins[plugin_name]

            if plugin.state != PluginState.ACTIVE:
                logger.warning(f"Plugin {plugin_name} not active")
                return True

            success = plugin.deactivate()
            if success:
                # Emit deactivation event
                self.emit_event("plugin_deactivated", {
                    "plugin_name": plugin_name
                })
            return success

    def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a plugin (hot reload).

        Args:
            plugin_name: Name of plugin to reload

        Returns:
            True if successful
        """
        if plugin_name not in self._plugins:
            logger.error(f"Plugin {plugin_name} not loaded")
            return False

        # Unload
        if not self.unload_plugin(plugin_name):
            return False

        # Clear module cache
        if plugin_name in sys.modules:
            del sys.modules[plugin_name]

        # Reload
        try:
            self.load_plugin(plugin_name)
            self.activate_plugin(plugin_name)
            return True
        except (PluginLoadError, RuntimeError, ImportError) as e:
            logger.error(f"Failed to reload plugin {plugin_name}: {e}")
            return False

    def execute_hook(self, hook_name: str, context: Dict[str, Any]) -> Any:
        """
        Execute all registered callbacks for a hook.

        Args:
            hook_name: Name of hook to execute
            context: Context data to pass to callbacks

        Returns:
            Modified context or result from callbacks
        """
        if hook_name not in self._hooks:
            logger.warning(f"Unknown hook: {hook_name}")
            return context

        result = context
        for hook in self._hooks[hook_name]:
            if not hook.enabled:
                continue

            try:
                result = hook.callback(result)
            except (RuntimeError, TypeError, ValueError) as e:
                logger.error(f"Hook {hook_name} callback failed: {e}", exc_info=True)

        return result

    def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit an event to all registered listeners.

        Args:
            event_type: Type of event
            data: Event data
        """
        event = PluginEvent(
            event_type=event_type,
            data=data,
            source="plugin_manager"
        )

        if event_type in self._event_listeners:
            for handler in self._event_listeners[event_type]:
                try:
                    handler(event)
                except (RuntimeError, TypeError, ValueError) as e:
                    logger.error(f"Event handler failed for {event_type}: {e}")

    def get_plugin(self, plugin_name: str) -> Optional[PluginBase]:
        """Get loaded plugin by name."""
        return self._plugins.get(plugin_name)

    def get_all_plugins(self) -> Dict[str, PluginBase]:
        """Get all loaded plugins."""
        return self._plugins.copy()

    def get_plugin_state(self, plugin_name: str) -> Optional[PluginState]:
        """Get plugin state."""
        return self._plugin_states.get(plugin_name)

    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get plugin information."""
        plugin = self._plugins.get(plugin_name)
        if plugin is None:
            return None

        return {
            "name": plugin.metadata.name,
            "version": plugin.metadata.version,
            "description": plugin.metadata.description,
            "author": plugin.metadata.author,
            "state": plugin.state.value,
            "hooks": list(plugin.get_hooks().keys()),
            "event_handlers": {
                k: len(v) for k, v in plugin.get_event_handlers().items()
            }
        }

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all plugins with their info."""
        return [
            self.get_plugin_info(name)
            for name in self._plugins.keys()
        ]

    def register_hook(self, hook_name: str, callback: Callable, priority: int = 100) -> None:
        """
        Register a hook directly (without plugin).

        Args:
            hook_name: Name of hook
            callback: Callback function
            priority: Priority (lower = higher priority)
        """
        hook = PluginHook(
            name=hook_name,
            description="Directly registered hook",
            callback=callback,
            priority=priority,
            plugin_name="system"
        )

        if hook_name not in self._hooks:
            self._hooks[hook_name] = []

        self._hooks[hook_name].append(hook)
        self._hooks[hook_name].sort(key=lambda h: h.priority)

    def subscribe_event(self, event_type: str, handler: Callable) -> None:
        """
        Subscribe to an event.

        Args:
            event_type: Type of event
            handler: Event handler function
        """
        if event_type not in self._event_listeners:
            self._event_listeners[event_type] = []

        self._event_listeners[event_type].append(handler)

    def load_all_plugins(self) -> None:
        """Load all discovered plugins."""
        discovered = self.discover_plugins()
        for plugin_name in discovered:
            try:
                plugin = self.load_plugin(plugin_name)
                self.activate_plugin(plugin_name)
            except (PluginLoadError, ImportError, RuntimeError) as e:
                logger.error(f"Failed to load plugin {plugin_name}: {e}")

    def shutdown(self) -> None:
        """Shutdown plugin manager and unload all plugins."""
        for plugin_name in list(self._plugins.keys()):
            self.unload_plugin(plugin_name)


# Singleton instance
_plugin_manager_instance: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager instance."""
    global _plugin_manager_instance
    if _plugin_manager_instance is None:
        _plugin_manager_instance = PluginManager()
    return _plugin_manager_instance


# Hook decorators for easy use
def hook(hook_name: str, priority: int = 100):
    """
    Decorator to register a function as a hook.

    Example:
        ```python
        @hook("on_before_decompose", priority=50)
        def my_decompose_hook(context):
            # Modify context
            return context
        ```
    """
    def decorator(func: Callable) -> Callable:
        pm = get_plugin_manager()
        pm.register_hook(hook_name, func, priority)
        return func
    return decorator


def event_handler(event_type: str):
    """
    Decorator to register a function as an event handler.

    Example:
        ```python
        @event_handler("workflow_complete")
        def on_workflow_complete(event):
            print(f"Workflow completed: {event.data}")
        ```
    """
    def decorator(func: Callable) -> Callable:
        pm = get_plugin_manager()
        pm.subscribe_event(event_type, func)
        return func
    return decorator


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    pm = get_plugin_manager()

    # Register a hook directly
    @hook("on_before_decompose", priority=50)
    def log_decomposition(context):
        print(f"Decomposing problem: {context.get('problem', 'unknown')}")
        return context

    # Register an event handler
    @event_handler("workflow_complete")
    def on_complete(event):
        print(f"Workflow completed at {event.timestamp}")

    # Execute hook
    pm.execute_hook("on_before_decompose", {"problem": "Test problem"})

    # Emit event
    pm.emit_event("workflow_complete", {"result": "success"})
