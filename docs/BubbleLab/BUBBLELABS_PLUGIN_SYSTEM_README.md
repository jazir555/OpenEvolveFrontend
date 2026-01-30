# BubbleLabs Plugin System

A robust, enterprise-grade plugin architecture for BubbleLabs integration with OpenEvolve workflows.

## Features

- **🔄 Plugin Lifecycle Management**: Complete lifecycle with initialization, startup, shutdown, and cleanup
- **📊 Event-Driven Architecture**: Event bus for loose coupling between plugins
- **🔗 Dependency Management**: Automatic dependency resolution and ordered loading
- **🔒 Thread Safety**: All operations are thread-safe with proper locking
- **🚦 Health Monitoring**: Built-in health checks and status tracking
- **🔥 Hot-Reloading**: Load/unload plugins at runtime without restarting
- **📝 Comprehensive Logging**: Detailed logging for all lifecycle events
- **⚙️ Configuration Management**: Per-plugin configuration with validation
- **🔌 Backward Compatible**: Drop-in replacement for existing integration code
- **🎯 Type Safety**: Full type hints throughout for better IDE support

## Installation

No additional installation required - the plugin system is included with OpenEvolve.

```bash
# Already part of the OpenEvolve Frontend
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend
```

## Quick Start

### Basic Usage

```python
import asyncio
from bubblelabs_plugin_system import get_plugin_registry

async def main():
    registry = get_plugin_registry()

    # Load OpenEvolve plugin
    plugin = await registry.load_plugin("openevolve")
    await registry.start_plugin("openevolve")

    # Create workflow
    definition = await plugin.create_workflow_definition(
        problem_statement="Design quantum-resistant protocols",
        team_config={"content_analyzer_team": "RedTeam"},
        gauntlet_config={"sub_problem_red_gauntlet": "PhysicsGauntlet"}
    )

    print(f"Created workflow: {definition.id}")

    # Cleanup
    await registry.unload_plugin("openevolve")

if __name__ == "__main__":
    asyncio.run(main())
```

### Backward Compatible Usage

```python
# Old code still works!
from openevolve_bubblelabs_plugin import bubblelabs_integration

definition = bubblelabs_integration.create_workflow_definition_from_openevolve(
    problem_statement="Solve world hunger",
    team_config={"content_analyzer_team": "RedTeam"},
    gauntlet_config={"sub_problem_red_gauntlet": "PhysicsGauntlet"}
)
```

## Architecture

### Plugin Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                    Plugin Lifecycle                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  UNLOADED ──load──> LOADED ──init──> INITIALIZED            │
│                      ▲               │                       │
│                      │               ▼                       │
│                      │           STARTED ──stop──> STOPPED   │
│                      │               │                       │
│                      └─────unload────┘                       │
│                                                               │
│  Any state ──error──> ERROR                                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Event Flow

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Plugin A  │         │  Event Bus   │         │   Plugin B  │
└──────┬──────┘         └──────┬───────┘         └──────┬──────┘
       │                        │                        │
       │  Publish Event         │                        │
       │───────────────────────>│                        │
       │                        │                        │
       │                        │  Distribute Event      │
       │                        │───────────────────────>│
       │                        │                        │
       │                        │  Call Handler          │
       │                        │<───────────────────────│
       │                        │                        │
```

## Core Components

### 1. BubbleLabsPlugin (Base Class)

All plugins must inherit from this base class:

```python
from bubblelabs_plugin_system import BubbleLabsPlugin

class MyPlugin(BubbleLabsPlugin):
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="my_plugin",
            version="1.0.0",
            author="Your Name",
            description="My awesome plugin"
        )

    async def initialize(self) -> None:
        """Setup resources"""
        pass

    async def start(self) -> None:
        """Start operations"""
        pass

    async def stop(self) -> None:
        """Stop operations"""
        pass

    async def cleanup(self) -> None:
        """Release resources"""
        pass
```

### 2. PluginRegistry

Manages plugin registration, loading, and lifecycle:

```python
from bubblelabs_plugin_system import get_plugin_registry

registry = get_plugin_registry()

# Register plugin
await registry.register_plugin(MyPlugin)

# Load plugin
plugin = await registry.load_plugin("my_plugin")

# Start plugin
await registry.start_plugin("my_plugin")

# Stop plugin
await registry.stop_plugin("my_plugin")

# Unload plugin
await registry.unload_plugin("my_plugin")
```

### 3. EventBus

Event-driven communication between plugins:

```python
from bubblelabs_plugin_system import PluginEvent

def register_hooks(self, event_bus: EventBus) -> None:
    async def on_workflow_start(event):
        print(f"Workflow started: {event.data}")

    event_bus.subscribe(PluginEvent.AFTER_START, on_workflow_start)
```

## Available Plugins

### OpenEvolve Plugin

The main OpenEvolve workflow integration plugin.

**Metadata:**
- Name: `openevolve`
- Version: `1.0.0`
- Category: `workflow`
- Priority: `HIGH`

**Configuration:**
```python
config = {
    "max_instance_age_seconds": 7 * 24 * 3600,  # 7 days
    "max_instances": 1000,
    "enable_auto_cleanup": True,
    "cleanup_interval_seconds": 3600,
}
```

**Methods:**
- `create_workflow_definition()`: Create workflow definition
- `control_workflow()`: Control workflow instances
- `get_metrics()`: Get plugin metrics
- `health_check()`: Check plugin health

### WorkflowAnalytics Plugin

Real-time workflow analytics and reporting.

**Metadata:**
- Name: `workflow_analytics`
- Version: `1.0.0`
- Category: `analytics`
- Dependencies: `["openevolve"]`

**Features:**
- Track workflow creation
- Monitor control actions
- Generate reports
- Calculate success rates

## Creating Custom Plugins

### Step 1: Define Plugin Class

```python
from bubblelabs_plugin_system import BubbleLabsPlugin, PluginMetadata

class MyCustomPlugin(BubbleLabsPlugin):
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="my_custom_plugin",
            version="1.0.0",
            author="Your Name",
            description="My custom plugin",
            dependencies=["openevolve"],  # Optional dependencies
            category="custom",
            tags=["custom", "plugin"],
        )
```

### Step 2: Implement Lifecycle Methods

```python
    async def initialize(self) -> None:
        """Initialize plugin resources"""
        self._logger.info("Initializing plugin")

        # Setup resources
        # Validate configuration
        # Connect to services

        self._status.state = PluginState.INITIALIZED
        self._status.health = "healthy"

    async def start(self) -> None:
        """Start plugin operations"""
        self._logger.info("Starting plugin")

        # Start background tasks
        # Subscribe to events
        # Begin operations

        self._status.state = PluginState.STARTED
        self._status.health = "healthy"

    async def stop(self) -> None:
        """Stop plugin operations"""
        self._logger.info("Stopping plugin")

        # Stop background tasks
        # Unsubscribe from events
        # Save state

        self._status.state = PluginState.STOPPED

    async def cleanup(self) -> None:
        """Cleanup plugin resources"""
        self._logger.info("Cleaning up plugin")

        # Close connections
        # Release resources
        # Cleanup temporary files

        self._status.state = PluginState.UNLOADED
```

### Step 3: Register Event Hooks

```python
    def register_hooks(self, event_bus: EventBus) -> None:
        """Register event hooks"""

        async def on_workflow_created(event):
            if event.plugin_name == "openevolve":
                self._logger.info(f"Workflow created: {event.data}")
                # Handle event

        event_bus.subscribe(PluginEvent.AFTER_START, on_workflow_created)
```

### Step 4: Register Plugin

```python
from bubblelabs_plugin_system import register_plugin

register_plugin(MyCustomPlugin)
```

## Configuration

### Plugin Configuration

```python
plugin = await registry.load_plugin("openevolve", config={
    "max_instance_age_seconds": 14 * 24 * 3600,
    "max_instances": 5000,
    "enable_auto_cleanup": True,
    "cleanup_interval_seconds": 1800,
})
```

### Runtime Configuration Updates

```python
plugin = await registry.get_plugin("openevolve")
await plugin.update_config({
    "max_instances": 10000,
    "new_feature_enabled": True,
})
```

## Health Monitoring

### Check Single Plugin

```python
plugin = await registry.get_plugin("openevolve")
is_healthy = await plugin.health_check()
status = plugin.get_status()

print(f"Healthy: {is_healthy}")
print(f"State: {status.state.value}")
print(f"Health: {status.health}")
```

### Check All Plugins

```python
health_status = await registry.check_all_health()

for name, is_healthy in health_status.items():
    print(f"{name}: {'✓' if is_healthy else '✗'}")
```

## Event Types

| Event | Description | When Fired |
|-------|-------------|------------|
| `BEFORE_LOAD` | Before plugin loads | Just before load |
| `AFTER_LOAD` | After plugin loads | Just after load |
| `BEFORE_INIT` | Before initialization | Just before initialize() |
| `AFTER_INIT` | After initialization | Just after initialize() |
| `BEFORE_START` | Before start | Just before start() |
| `AFTER_START` | After start | Just after start() |
| `BEFORE_STOP` | Before stop | Just before stop() |
| `AFTER_STOP` | After stop | Just after stop() |
| `BEFORE_UNLOAD` | Before unload | Just before unload |
| `AFTER_UNLOAD` | After unload | Just after unload |
| `ON_ERROR` | On error | When error occurs |
| `ON_CONFIG_CHANGE` | On config change | When config updates |

## Error Handling

### Graceful Degradation

```python
plugin = await registry.load_plugin("openevolve")

if plugin is None:
    logger.error("Failed to load plugin")
    # Use fallback behavior
else:
    # Use plugin
    await plugin.start()
```

### Error Recovery

```python
status = registry.get_plugin_status("openevolve")

if status.state == PluginState.ERROR:
    logger.error(f"Plugin error: {status.error}")

    # Attempt recovery
    await registry.unload_plugin("openevolve")
    plugin = await registry.load_plugin("openevolve")
```

## Testing

### Unit Test Example

```python
import pytest
from bubblelabs_plugin_system import get_plugin_registry

@pytest.mark.asyncio
async def test_openevolve_plugin():
    registry = get_plugin_registry()

    # Load plugin
    plugin = await registry.load_plugin("openevolve")
    assert plugin is not None

    # Start plugin
    success = await registry.start_plugin("openevolve")
    assert success

    # Health check
    is_healthy = await plugin.health_check()
    assert is_healthy

    # Cleanup
    await registry.unload_plugin("openevolve")
```

## API Reference

### BubbleLabsPlugin

**Abstract Methods:**
- `get_metadata() -> PluginMetadata`: Get plugin metadata
- `initialize() -> None`: Initialize plugin
- `start() -> None`: Start plugin
- `stop() -> None`: Stop plugin
- `cleanup() -> None`: Cleanup resources

**Optional Methods:**
- `register_hooks(event_bus: EventBus) -> None`: Register event hooks
- `health_check() -> bool`: Check plugin health

**Properties:**
- `_config`: Plugin configuration
- `_status`: Current plugin status
- `_logger`: Plugin logger
- `_event_bus`: Event bus reference

### PluginRegistry

**Methods:**
- `register_plugin(plugin_class, config) -> None`: Register plugin
- `unregister_plugin(name) -> None`: Unregister plugin
- `load_plugin(name, config) -> Optional[Plugin]`: Load plugin
- `start_plugin(name) -> bool`: Start plugin
- `stop_plugin(name) -> bool`: Stop plugin
- `unload_plugin(name) -> bool`: Unload plugin
- `get_plugin(name) -> Optional[Plugin]`: Get plugin instance
- `list_plugins(state=None) -> Dict[str, PluginMetadata]`: List plugins
- `get_plugin_status(name) -> Optional[PluginStatus]`: Get plugin status
- `check_all_health() -> Dict[str, bool]`: Check all plugin health
- `shutdown_all() -> Dict[str, bool]`: Shutdown all plugins

## Best Practices

### 1. Plugin Design
- Keep plugins focused on a single responsibility
- Use dependency injection for external dependencies
- Handle errors gracefully
- Log all operations

### 2. Lifecycle Management
- Initialize resources in `initialize()`
- Start operations in `start()`
- Stop operations in `stop()`
- Cleanup resources in `cleanup()`

### 3. Event Handling
- Register hooks in `register_hooks()`
- Always wrap event handlers in try/except
- Use async for event handlers
- Clean up event subscriptions

### 4. Configuration
- Validate configuration in `initialize()`
- Provide sensible defaults
- Document configuration schema
- Handle runtime configuration changes

### 5. Dependencies
- Declare all dependencies in metadata
- Check dependency availability
- Use priority to control load order
- Avoid circular dependencies

## Migration Guide

See [MIGRATION_GUIDE.md](BUBBLELABS_PLUGIN_MIGRATION_GUIDE.md) for detailed migration instructions from the old integration system.

## Examples

See [examples/bubblelabs_plugin_examples.py](examples/bubblelabs_plugin_examples.py) for comprehensive usage examples.

## Troubleshooting

### Plugin Won't Load

```python
# Check if registered
registry = get_plugin_registry()
plugins = registry.list_plugins()
print(plugins.keys())

# Check status
status = registry.get_plugin_status("openevolve")
print(status)
```

### Dependency Issues

```python
# Check dependency graph
registry = get_plugin_registry()
print(registry._dependency_graph)
```

### Event Handlers Not Firing

```python
# Check event history
event_bus = registry.get_event_bus()
history = event_bus.get_history()
for event in history:
    print(f"{event.type} - {event.plugin_name}: {event.data}")
```

## Contributing

To contribute a new plugin:

1. Create plugin class inheriting from `BubbleLabsPlugin`
2. Implement all required methods
3. Add comprehensive tests
4. Update documentation
5. Submit pull request

## License

Part of the OpenEvolve project. See main project LICENSE file.

## Support

- Documentation: [BUBBLELABS_PLUGIN_MIGRATION_GUIDE.md](BUBBLELABS_PLUGIN_MIGRATION_GUIDE.md)
- Examples: [examples/bubblelabs_plugin_examples.py](examples/bubblelabs_plugin_examples.py)
- Issues: OpenEvolve GitHub Issues

## Changelog

### Version 1.0.0 (2026-01-03)
- Initial release
- Plugin lifecycle management
- Event bus implementation
- Dependency resolution
- Health monitoring
- Backward compatibility layer
- Comprehensive documentation
- Complete test suite

---

**Built with ❤️ by the OpenEvolve Integration Team**
