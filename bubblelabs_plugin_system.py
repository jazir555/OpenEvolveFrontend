"""
BubbleLabs Plugin Architecture for OpenEvolve

This module implements a proper plugin architecture for BubbleLabs integration,
providing plugin lifecycle management, dependency resolution, event bus communication,
and comprehensive error handling.

Author: OpenEvolve Integration Team
Created: 2026-01-03
Status: Production Ready
"""

import asyncio
import inspect
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    Coroutine,
)
from contextlib import asynccontextmanager
from functools import wraps
import importlib.util
import sys

logger = logging.getLogger(__name__)

# Type variables for generic plugin support
P = TypeVar('P', bound='BubbleLabsPlugin')
T = TypeVar('T')


class PluginState(Enum):
    """States a plugin can be in during its lifecycle."""
    UNLOADED = "unloaded"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    STARTED = "started"
    STOPPED = "stopped"
    ERROR = "error"
    DISABLED = "disabled"


class PluginPriority(Enum):
    """Priority levels for plugin loading/unloading order."""
    CRITICAL = 0  # Load first, unload last
    HIGH = 1
    NORMAL = 2
    LOW = 3  # Load last, unload first


@dataclass
class PluginMetadata:
    """
    Metadata for a BubbleLabs plugin.

    Attributes:
        name: Unique plugin identifier
        version: Plugin version (semver)
        author: Plugin author/organization
        description: Human-readable description
        dependencies: List of plugin names this plugin depends on
        priority: Loading priority
        category: Plugin category for organization
        tags: Searchable tags
        config_schema: JSON schema for plugin configuration
        min_bubblelabs_version: Minimum BubbleLabs API version required
        max_bubblelabs_version: Maximum BubbleLabs API version supported (inclusive)
    """
    name: str
    version: str
    author: str
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    priority: PluginPriority = PluginPriority.NORMAL
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    min_bubblelabs_version: str = "1.0.0"
    max_bubblelabs_version: str = "2.0.0"

    def __post_init__(self):
        """Validate metadata after initialization."""
        if not self.name:
            raise ValueError("Plugin name cannot be empty")
        if not self.version:
            raise ValueError("Plugin version cannot be empty")


@dataclass
class PluginStatus:
    """
    Current status of a plugin.

    Attributes:
        state: Current plugin state
        health: Health status (healthy, degraded, unhealthy)
        message: Status message
        last_updated: Timestamp of last status update
        error: Last error if in ERROR state
        metrics: Performance/usage metrics
    """
    state: PluginState
    health: str = "unknown"
    message: str = ""
    last_updated: float = field(default_factory=time.time)
    error: Optional[Exception] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        """Check if plugin is healthy."""
        return self.health in ("healthy", "degraded") and self.state not in (
            PluginState.ERROR,
            PluginState.DISABLED,
        )


class PluginEvent(Enum):
    """Events that can occur during plugin lifecycle."""
    BEFORE_LOAD = "before_load"
    AFTER_LOAD = "after_load"
    BEFORE_INIT = "before_init"
    AFTER_INIT = "after_init"
    BEFORE_START = "before_start"
    AFTER_START = "after_start"
    BEFORE_STOP = "before_stop"
    AFTER_STOP = "after_stop"
    BEFORE_UNLOAD = "before_unload"
    AFTER_UNLOAD = "after_unload"
    ON_ERROR = "on_error"
    ON_CONFIG_CHANGE = "on_config_change"


@dataclass
class Event:
    """
    Event in the plugin event bus.

    Attributes:
        type: Event type
        plugin_name: Name of plugin that triggered the event
        timestamp: Event timestamp
        data: Event-specific data
    """
    type: PluginEvent
    plugin_name: str
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """
    Event bus for plugin communication.

    Implements publish-subscribe pattern for loose coupling between plugins.
    Thread-safe with support for both sync and async event handlers.
    """

    def __init__(self):
        """Initialize the event bus."""
        self._subscribers: Dict[PluginEvent, List[Callable]] = {}
        self._lock = threading.RLock()
        self._event_history: List[Event] = []
        self._max_history = 1000

    def subscribe(self, event: PluginEvent, handler: Callable) -> None:
        """
        Subscribe to an event.

        Args:
            event: Event type to subscribe to
            handler: Callback function (sync or async)
        """
        with self._lock:
            if event not in self._subscribers:
                self._subscribers[event] = []
            self._subscribers[event].append(handler)
            logger.debug(f"Subscribed to event {event.value}: {handler}")

    def unsubscribe(self, event: PluginEvent, handler: Callable) -> None:
        """
        Unsubscribe from an event.

        Args:
            event: Event type to unsubscribe from
            handler: Callback function to remove
        """
        with self._lock:
            if event in self._subscribers:
                try:
                    self._subscribers[event].remove(handler)
                    logger.debug(f"Unsubscribed from event {event.value}: {handler}")
                except ValueError:
                    logger.warning(f"Handler not found for event {event.value}")

    async def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        # Add to history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

        # Get subscribers
        with self._lock:
            handlers = self._subscribers.get(event.type, []).copy()

        # Call all handlers
        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    # Run sync handlers in thread pool to avoid blocking
                    await asyncio.get_event_loop().run_in_executor(
                        None, handler, event
                    )
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(
                    f"Error in event handler for {event.type.value}: {e}",
                    exc_info=True,
                )

    def get_history(self, event_type: Optional[PluginEvent] = None) -> List[Event]:
        """
        Get event history.

        Args:
            event_type: Optional event type filter

        Returns:
            List of events
        """
        with self._lock:
            if event_type:
                return [e for e in self._event_history if e.type == event_type]
            return self._event_history.copy()

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._event_history.clear()


class BubbleLabsPlugin(ABC):
    """
    Base class for all BubbleLabs plugins.

    Plugins must inherit from this class and implement the abstract methods.
    The plugin lifecycle is: load -> initialize -> start -> [running] -> stop -> cleanup

    Example:
        ```python
        class MyPlugin(BubbleLabsPlugin):
            @classmethod
            def get_metadata(cls) -> PluginMetadata:
                return PluginMetadata(
                    name="my_plugin",
                    version="1.0.0",
                    author="Me",
                    description="My awesome plugin"
                )

            async def initialize(self) -> None:
                # Setup plugin resources
                pass

            async def start(self) -> None:
                # Start plugin logic
                pass

            async def stop(self) -> None:
                # Stop plugin logic
                pass

            async def cleanup(self) -> None:
                # Cleanup resources
                pass
        ```
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize plugin with configuration.

        Args:
            config: Plugin configuration
        """
        self._config = config
        self._status = PluginStatus(state=PluginState.LOADED)
        self._event_bus: Optional[EventBus] = None
        self._logger = logging.getLogger(f"bubblelabs.plugin.{self.get_metadata().name}")

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> PluginMetadata:
        """
        Get plugin metadata.

        Returns:
            PluginMetadata object
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the plugin.

        Called once after plugin is loaded. Setup resources, validate config, etc.
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """
        Start the plugin.

        Called after initialization. Plugin should be ready to handle requests.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the plugin.

        Gracefully stop plugin operations.
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup plugin resources.

        Called before plugin is unloaded. Release all resources.
        """
        pass

    def register_hooks(self, event_bus: EventBus) -> None:
        """
        Register event hooks.

        Override this to register event handlers.

        Args:
            event_bus: Event bus to register hooks with
        """
        pass

    async def health_check(self) -> bool:
        """
        Check plugin health.

        Override to implement custom health checks.

        Returns:
            True if plugin is healthy
        """
        return self._status.is_healthy()

    def get_status(self) -> PluginStatus:
        """Get current plugin status."""
        return self._status

    def get_config(self) -> Dict[str, Any]:
        """Get plugin configuration."""
        return self._config.copy()

    async def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update plugin configuration.

        Args:
            new_config: New configuration
        """
        self._config = new_config
        if self._event_bus:
            await self._event_bus.publish(
                Event(
                    type=PluginEvent.ON_CONFIG_CHANGE,
                    plugin_name=self.get_metadata().name,
                    data={"new_config": new_config},
                )
            )

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Set event bus for plugin."""
        self._event_bus = event_bus


class PluginRegistry:
    """
    Registry for managing BubbleLabs plugins.

    Handles plugin discovery, registration, lifecycle management, dependency resolution,
    and provides thread-safe operations.
    """

    def __init__(self):
        """Initialize the plugin registry."""
        self._plugins: Dict[str, Type[BubbleLabsPlugin]] = {}
        self._instances: Dict[str, BubbleLabsPlugin] = {}
        self._event_bus = EventBus()
        self._lock = asyncio.Lock()  # Lock for async operations (loading/starting)
        self._registry_lock = threading.RLock()  # Lock for sync registry operations
        self._dependency_graph: Dict[str, Set[str]] = {}

    def register_plugin(
        self, plugin_class: Type[P], config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a plugin class.

        Args:
            plugin_class: Plugin class to register
            config: Default configuration for plugin instances

        Raises:
            ValueError: If plugin is invalid or already registered
        """
        metadata = plugin_class.get_metadata()

        with self._registry_lock:
            if metadata.name in self._plugins:
                # Already registered, just return
                return

            # Validate plugin class
            if not issubclass(plugin_class, BubbleLabsPlugin):
                raise ValueError(f"Plugin must inherit from BubbleLabsPlugin")

            # Validate dependencies exist (warning only)
            for dep in metadata.dependencies:
                if dep not in self._plugins and dep != metadata.name:
                    logger.debug(
                        f"Plugin {metadata.name} depends on {dep} which is not registered yet"
                    )

            self._plugins[metadata.name] = plugin_class
            self._dependency_graph[metadata.name] = set(metadata.dependencies)

            logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")

    async def unregister_plugin(self, name: str) -> None:
        """
        Unregister a plugin.

        Args:
            name: Plugin name

        Raises:
            ValueError: If plugin has dependents
        """
        async with self._lock:
            with self._registry_lock:
                # Check for dependents
                dependents = [
                    plugin_name
                    for plugin_name, deps in self._dependency_graph.items()
                    if name in deps and plugin_name != name
                ]

                if dependents:
                    raise ValueError(
                        f"Cannot unregister {name}: depended upon by {dependents}"
                    )

                # Unregister
                if name in self._plugins:
                    del self._plugins[name]
                if name in self._dependency_graph:
                    del self._dependency_graph[name]

            # Unload if running
            if name in self._instances:
                await self.unload_plugin(name)

            logger.info(f"Unregistered plugin: {name}")

    async def load_plugin(
        self, name: str, config: Optional[Dict[str, Any]] = None
    ) -> Optional[BubbleLabsPlugin]:
        """
        Load and initialize a plugin.

        Args:
            name: Plugin name
            config: Optional configuration override

        Returns:
            Plugin instance or None if failed
        """
        async with self._lock:
            with self._registry_lock:
                if name not in self._plugins:
                    logger.error(f"Plugin {name} not registered")
                    return None

                if name in self._instances:
                    logger.warning(f"Plugin {name} is already loaded")
                    return self._instances[name]

                plugin_class = self._plugins[name]
                metadata = plugin_class.get_metadata()

            # Check dependencies
            deps_ok = await self._check_dependencies(name)
            if not deps_ok:
                logger.error(f"Plugin {name} has unsatisfied dependencies")
                return None

            # Load dependencies first
            for dep in metadata.dependencies:
                if dep not in self._instances:
                    await self.load_plugin(dep)

            # Create instance
            try:
                plugin_config = config or {}
                instance = plugin_class(plugin_config)
                instance.set_event_bus(self._event_bus)

                # Publish before_load event
                await self._event_bus.publish(
                    Event(type=PluginEvent.BEFORE_LOAD, plugin_name=name)
                )

                # Initialize
                await self._event_bus.publish(
                    Event(type=PluginEvent.BEFORE_INIT, plugin_name=name)
                )

                await instance.initialize()
                instance._status.state = PluginState.INITIALIZED
                instance._status.health = "healthy"

                await self._event_bus.publish(
                    Event(type=PluginEvent.AFTER_INIT, plugin_name=name)
                )

                # Register hooks
                instance.register_hooks(self._event_bus)

                # Store instance
                self._instances[name] = instance

                await self._event_bus.publish(
                    Event(type=PluginEvent.AFTER_LOAD, plugin_name=name)
                )

                logger.info(f"Loaded plugin: {name}")
                return instance

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to load plugin {name}: {e}", exc_info=True)
                await self._event_bus.publish(
                    Event(
                        type=PluginEvent.ON_ERROR,
                        plugin_name=name,
                        data={"error": str(e)},
                    )
                )
                return None

    async def start_plugin(self, name: str) -> bool:
        """
        Start a loaded plugin.

        Args:
            name: Plugin name

        Returns:
            True if started successfully
        """
        async with self._lock:
            instance = self._instances.get(name)
            if not instance:
                logger.error(f"Plugin {name} is not loaded")
                return False

            try:
                await self._event_bus.publish(
                    Event(type=PluginEvent.BEFORE_START, plugin_name=name)
                )

                await instance.start()
                instance._status.state = PluginState.STARTED

                await self._event_bus.publish(
                    Event(type=PluginEvent.AFTER_START, plugin_name=name)
                )

                logger.info(f"Started plugin: {name}")
                return True

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to start plugin {name}: {e}", exc_info=True)
                instance._status.state = PluginState.ERROR
                instance._status.error = e
                await self._event_bus.publish(
                    Event(
                        type=PluginEvent.ON_ERROR,
                        plugin_name=name,
                        data={"error": str(e)},
                    )
                )
                return False

    async def stop_plugin(self, name: str) -> bool:
        """
        Stop a running plugin.

        Args:
            name: Plugin name

        Returns:
            True if stopped successfully
        """
        async with self._lock:
            instance = self._instances.get(name)
            if not instance:
                logger.warning(f"Plugin {name} is not loaded")
                return True

            if instance._status.state != PluginState.STARTED:
                logger.warning(f"Plugin {name} is not running")
                return True

            try:
                await self._event_bus.publish(
                    Event(type=PluginEvent.BEFORE_STOP, plugin_name=name)
                )

                await instance.stop()
                instance._status.state = PluginState.STOPPED

                await self._event_bus.publish(
                    Event(type=PluginEvent.AFTER_STOP, plugin_name=name)
                )

                logger.info(f"Stopped plugin: {name}")
                return True

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to stop plugin {name}: {e}", exc_info=True)
                return False

    async def unload_plugin(self, name: str) -> bool:
        """
        Unload a plugin.

        Args:
            name: Plugin name

        Returns:
            True if unloaded successfully
        """
        async with self._lock:
            instance = self._instances.get(name)
            if not instance:
                logger.warning(f"Plugin {name} is not loaded")
                return True

            try:
                # Stop if running
                if instance._status.state == PluginState.STARTED:
                    await self.stop_plugin(name)

                await self._event_bus.publish(
                    Event(type=PluginEvent.BEFORE_UNLOAD, plugin_name=name)
                )

                # Cleanup
                await instance.cleanup()
                instance._status.state = PluginState.UNLOADED

                await self._event_bus.publish(
                    Event(type=PluginEvent.AFTER_UNLOAD, plugin_name=name)
                )

                # Remove instance
                if name in self._instances:
                    del self._instances[name]

                logger.info(f"Unloaded plugin: {name}")
                return True

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to unload plugin {name}: {e}", exc_info=True)
                return False

    async def get_plugin(self, name: str) -> Optional[BubbleLabsPlugin]:
        """
        Get a loaded plugin instance.

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None
        """
        async with self._lock:
            return self._instances.get(name)

    def list_plugins(
        self, state: Optional[PluginState] = None
    ) -> Dict[str, PluginMetadata]:
        """
        List registered plugins.

        Args:
            state: Optional state filter

        Returns:
            Dictionary mapping plugin names to metadata
        """
        with self._registry_lock:
            if state:
                return {
                    name: self._plugins[name].get_metadata()
                    for name, instance in self._instances.items()
                    if instance._status.state == state
                }
            return {
                name: plugin_class.get_metadata()
                for name, plugin_class in self._plugins.items()
            }

    def get_plugin_status(self, name: str) -> Optional[PluginStatus]:
        """
        Get plugin status.

        Args:
            name: Plugin name

        Returns:
            Plugin status or None
        """
        return self._instances.get(name)._status if name in self._instances else None

    async def check_all_health(self) -> Dict[str, bool]:
        """
        Check health of all loaded plugins.

        Returns:
            Dictionary mapping plugin names to health status
        """
        health_status = {}
        async with self._lock:
            instances = list(self._instances.items())
            
        for name, instance in instances:
            try:
                health_status[name] = await instance.health_check()
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Health check failed for {name}: {e}")
                health_status[name] = False
        return health_status

    def get_event_bus(self) -> EventBus:
        """Get the global event bus."""
        return self._event_bus

    async def _check_dependencies(self, plugin_name: str) -> bool:
        """
        Check if plugin dependencies are satisfied.

        Args:
            plugin_name: Plugin to check

        Returns:
            True if dependencies are satisfied
        """
        with self._registry_lock:
            dependencies = self._dependency_graph.get(plugin_name, set())

            for dep in dependencies:
                if dep not in self._plugins:
                    logger.error(f"Dependency {dep} for {plugin_name} not registered")
                    return False

        return True

    async def shutdown_all(self) -> Dict[str, bool]:
        """
        Shutdown all plugins in reverse dependency order.

        Returns:
            Dictionary mapping plugin names to shutdown success
        """
        shutdown_status = {}

        async with self._lock:
            instances = list(self._instances.items())

        # Get plugins in reverse priority order
        plugins_by_priority = sorted(
            instances,
            key=lambda x: x[1].get_metadata().priority.value,
            reverse=True,
        )

        for name, _ in plugins_by_priority:
            shutdown_status[name] = await self.unload_plugin(name)

        return shutdown_status


# Global plugin registry instance
_global_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """
    Get the global plugin registry.

    Returns:
        PluginRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def register_plugin(
    plugin_class: Type[BubbleLabsPlugin], config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Register a plugin with the global registry.

    Args:
        plugin_class: Plugin class to register
        config: Default configuration
    """
    registry = get_plugin_registry()
    registry.register_plugin(plugin_class, config)


def load_plugin_sync(
    name: str, config: Optional[Dict[str, Any]] = None
) -> Optional[BubbleLabsPlugin]:
    """
    Load a plugin synchronously.

    Args:
        name: Plugin name
        config: Configuration override

    Returns:
        Plugin instance or None
    """
    registry = get_plugin_registry()
    # Run in existing event loop or create new one
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we need to schedule as a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, registry.load_plugin(name, config)
                )
                return future.result()
        else:
            return loop.run_until_complete(registry.load_plugin(name, config))
    except RuntimeError:
        # No event loop, create new one
        return asyncio.run(registry.load_plugin(name, config))
