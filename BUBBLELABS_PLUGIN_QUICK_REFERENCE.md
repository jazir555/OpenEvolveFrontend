# BubbleLabs Plugin System - Quick Reference

## Quick Start (30 seconds)

```python
import asyncio
from bubblelabs_plugin_system import get_plugin_registry

async def main():
    registry = get_plugin_registry()
    plugin = await registry.load_plugin("openevolve")
    await registry.start_plugin("openevolve")

    definition = await plugin.create_workflow_definition(
        problem_statement="Your problem here",
        team_config={"content_analyzer_team": "RedTeam"},
        gauntlet_config={"sub_problem_red_gauntlet": "PhysicsGauntlet"}
    )

    print(f"Created: {definition.id}")

asyncio.run(main())
```

## Plugin Template (Copy & Paste)

```python
from bubblelabs_plugin_system import (
    BubbleLabsPlugin,
    PluginMetadata,
    PluginEvent,
    EventBus,
    PluginState,
)
from typing import Dict, Any

class MyPlugin(BubbleLabsPlugin):
    """One-line description of plugin."""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="my_plugin",
            version="1.0.0",
            author="Your Name",
            description="Detailed description",
            dependencies=[],  # ["openevolve"] if depends on it
            priority=PluginPriority.NORMAL,
            category="general",
            tags=["tag1", "tag2"],
        )

    async def initialize(self) -> None:
        """Setup resources."""
        self._logger.info("Initializing")
        # Setup code here
        self._status.state = PluginState.INITIALIZED
        self._status.health = "healthy"

    async def start(self) -> None:
        """Start operations."""
        self._logger.info("Starting")
        # Start code here
        self._status.state = PluginState.STARTED
        self._status.health = "healthy"

    async def stop(self) -> None:
        """Stop operations."""
        self._logger.info("Stopping")
        # Stop code here
        self._status.state = PluginState.STOPPED

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._logger.info("Cleaning up")
        # Cleanup code here
        self._status.state = PluginState.UNLOADED

    def register_hooks(self, event_bus: EventBus) -> None:
        """Register event hooks."""
        super().register_hooks(event_bus)

        async def on_event(event):
            self._logger.info(f"Event: {event.data}")

        event_bus.subscribe(PluginEvent.AFTER_START, on_event)

    async def health_check(self) -> bool:
        """Check health."""
        return True

# Register
from bubblelabs_plugin_system import register_plugin
register_plugin(MyPlugin)
```

## Common Operations

### Loading & Starting

```python
# Load
plugin = await registry.load_plugin("openevolve", config={})

# Start
await registry.start_plugin("openevolve")

# Check status
status = plugin.get_status()
print(f"State: {status.state.value}")
print(f"Healthy: {status.is_healthy()}")
```

### Creating Workflows

```python
definition = await plugin.create_workflow_definition(
    problem_statement="Your problem",
    team_config={
        "content_analyzer_team": "RedTeam",
        "planner_team": "PlannerTeam",
        "solver_team": "SolverTeam",
        "assembler_team": "AssemblerTeam",
    },
    gauntlet_config={
        "sub_problem_red_gauntlet": "PhysicsGauntlet",
        "final_gold_gauntlet": "GoldGauntlet",
    }
)
```

### Controlling Workflows

```python
result = await plugin.control_workflow(instance_id, "start")
result = await plugin.control_workflow(instance_id, "pause")
result = await plugin.control_workflow(instance_id, "resume")
result = await plugin.control_workflow(instance_id, "cancel")
```

### Health Checks

```python
# Single plugin
is_healthy = await plugin.health_check()

# All plugins
health = await registry.check_all_health()

# Detailed status
status = plugin.get_status()
print(f"State: {status.state}")
print(f"Health: {status.health}")
print(f"Metrics: {status.metrics}")
```

### Event Subscription

```python
def register_hooks(self, event_bus: EventBus) -> None:
    async def handler(event):
        if event.plugin_name == "openevolve":
            # Handle event
            pass

    event_bus.subscribe(PluginEvent.AFTER_START, handler)
```

## Plugin States

| State | Description |
|-------|-------------|
| `UNLOADED` | Plugin not loaded |
| `LOADED` | Plugin loaded, not initialized |
| `INITIALIZED` | Plugin initialized, not started |
| `STARTED` | Plugin running |
| `STOPPED` | Plugin stopped |
| `ERROR` | Plugin in error state |
| `DISABLED` | Plugin disabled |

## Event Types

| Event | When |
|-------|------|
| `BEFORE_LOAD` | Before loading |
| `AFTER_LOAD` | After loading |
| `BEFORE_INIT` | Before initialization |
| `AFTER_INIT` | After initialization |
| `BEFORE_START` | Before starting |
| `AFTER_START` | After starting |
| `BEFORE_STOP` | Before stopping |
| `AFTER_STOP` | After stopping |
| `BEFORE_UNLOAD` | Before unloading |
| `AFTER_UNLOAD` | After unloading |
| `ON_ERROR` | On error |
| `ON_CONFIG_CHANGE` | On config change |

## Configuration Options

### OpenEvolve Plugin

```python
config = {
    "max_instance_age_seconds": 7 * 24 * 3600,  # 7 days
    "max_instances": 1000,
    "enable_auto_cleanup": True,
    "cleanup_interval_seconds": 3600,  # 1 hour
}
```

## Error Handling

```python
# Check if plugin loaded
plugin = await registry.load_plugin("openevolve")
if plugin is None:
    logger.error("Failed to load")
    return

# Check for errors
status = plugin.get_status()
if status.state == PluginState.ERROR:
    logger.error(f"Error: {status.error}")
    # Handle error

# Try-except for operations
try:
    await plugin.start()
except Exception as e:
    logger.error(f"Start failed: {e}")
```

## Cleanup

```python
# Unload single plugin
await registry.unload_plugin("openevolve")

# Shutdown all
await registry.shutdown_all()
```

## Backward Compatibility

```python
# Old API still works
from openevolve_bubblelabs_plugin import bubblelabs_integration

definition = bubblelabs_integration.create_workflow_definition_from_openevolve(
    problem_statement="Problem",
    team_config={},
    gauntlet_config={}
)
```

## Registry Operations

```python
registry = get_plugin_registry()

# List all plugins
plugins = registry.list_plugins()

# List by state
started = registry.list_plugins(state=PluginState.STARTED)

# Get plugin instance
plugin = await registry.get_plugin("openevolve")

# Get plugin info
info = registry.get_plugin_info("openevolve")

# Check availability
available = await registry.is_available("openevolve")
```

## Event Bus

```python
event_bus = registry.get_event_bus()

# Get history
history = event_bus.get_history()
recent = event_bus.get_history(event_type=PluginEvent.AFTER_START)

# Clear history
event_bus.clear_history()
```

## Metrics

```python
# Get metrics
metrics = await plugin.get_metrics()
print(metrics)

# Reset metrics
await plugin.reset_metrics()
```

## Common Patterns

### With Resource Cleanup

```python
async def with_plugin():
    registry = get_plugin_registry()
    try:
        plugin = await registry.load_plugin("openevolve")
        await registry.start_plugin("openevolve")
        # Use plugin
    finally:
        await registry.shutdown_all()
```

### Event Handler with Error Handling

```python
async def handler(event):
    try:
        # Handle event
        pass
    except Exception as e:
        self._logger.error(f"Handler error: {e}")
        # Don't raise

event_bus.subscribe(PluginEvent.AFTER_START, handler)
```

### Periodic Task in Plugin

```python
async def start(self):
    self._task = asyncio.create_task(self._periodic_loop())

async def _periodic_loop(self):
    while True:
        try:
            await asyncio.sleep(60)
            await self._do_work()
        except asyncio.CancelledError:
            break
        except Exception as e:
            self._logger.error(f"Task error: {e}")

async def stop(self):
    if self._task:
        self._task.cancel()
        await self._task
```

## Troubleshooting

### Plugin Not Loading

```python
# Check registered
plugins = registry.list_plugins()
print("Registered:", list(plugins.keys()))

# Check status
status = registry.get_plugin_status("openevolve")
print("Status:", status)
```

### Dependency Issues

```python
# Check deps
deps = registry._dependency_graph
print("Dependencies:", deps)
```

### Event Issues

```python
# Check history
event_bus = registry.get_event_bus()
history = event_bus.get_history()
for event in history[-10:]:
    print(f"{event.type} - {event.plugin_name}")
```

## File Structure

```
Frontend/
├── bubblelabs_plugin_system.py          # Core plugin system
├── openevolve_bubblelabs_plugin.py      # OpenEvolve plugin
├── bubblelabs_integration.py            # Legacy integration (keep)
├── BUBBLELABS_PLUGIN_SYSTEM_README.md   # Full documentation
├── BUBBLELABS_PLUGIN_MIGRATION_GUIDE.md # Migration guide
├── BUBBLELABS_PLUGIN_QUICK_REFERENCE.md # This file
└── examples/
    └── bubblelabs_plugin_examples.py    # Complete examples
```

## Import Paths

```python
# Core system
from bubblelabs_plugin_system import (
    BubbleLabsPlugin,
    PluginMetadata,
    PluginPriority,
    PluginState,
    PluginEvent,
    PluginRegistry,
    EventBus,
    get_plugin_registry,
    register_plugin,
)

# OpenEvolve plugin
from openevolve_bubblelabs_plugin import (
    OpenEvolveBubbleLabsPlugin,
    bubblelabs_integration,  # Backward compat
)

# Legacy (still works)
from bubblelabs_integration import (
    BubbleLabsIntegration,
    BubbleWorkflowDefinition,
)
```

## Tips

1. **Always use async/await** for plugin operations
2. **Check for None** when loading plugins
3. **Handle errors** in event handlers
4. **Cleanup resources** in cleanup() method
5. **Log everything** with self._logger
6. **Validate config** in initialize()
7. **Use health checks** for monitoring
8. **Subscribe to events** in register_hooks()
9. **Declare dependencies** in metadata
10. **Test lifecycle** in isolation

## More Info

- Full Docs: `BUBBLELABS_PLUGIN_SYSTEM_README.md`
- Migration: `BUBBLELABS_PLUGIN_MIGRATION_GUIDE.md`
- Examples: `examples/bubblelabs_plugin_examples.py`
