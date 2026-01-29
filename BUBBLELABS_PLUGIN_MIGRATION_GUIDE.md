# BubbleLabs Plugin Architecture - Migration Guide

## Overview

This guide helps you migrate from the old `bubblelabs_integration.py` to the new plugin architecture.

## What's New

### Plugin System Features

1. **Plugin Lifecycle Management**
   - Proper initialization, startup, shutdown, and cleanup
   - State tracking (UNLOADED, LOADED, INITIALIZED, STARTED, STOPPED, ERROR)
   - Health monitoring and automatic recovery

2. **Event-Driven Architecture**
   - Event bus for plugin communication
   - Subscribe to lifecycle events
   - Loose coupling between plugins

3. **Dependency Management**
   - Automatic dependency resolution
   - Ordered loading based on dependencies
   - Circular dependency detection

4. **Configuration Management**
   - Per-plugin configuration with validation
   - Runtime configuration updates
   - Configuration change events

5. **Hot-Reloading Support**
   - Load/unload plugins at runtime
   - Automatic dependency handling
   - Graceful shutdown

6. **Comprehensive Logging**
   - Detailed logging for all lifecycle events
   - Per-plugin loggers
   - Event history tracking

7. **Thread Safety**
   - All operations are thread-safe
   - Async/await support throughout
   - Proper locking mechanisms

## Migration Guide

### Step 1: Update Imports

**Before (Old Code):**
```python
from bubblelabs_integration import (
    BubbleLabsIntegration,
    bubblelabs_integration,
    BubbleWorkflowDefinition,
)

# Create instance
integration = BubbleLabsIntegration()
```

**After (New Code):**
```python
from bubblelabs_plugin_system import (
    BubbleLabsPlugin,
    PluginMetadata,
    PluginPriority,
    register_plugin,
    get_plugin_registry,
)
from openevolve_bubblelabs_plugin import OpenEvolveBubbleLabsPlugin

# Or use backward compatible wrapper
from openevolve_bubblelabs_plugin import bubblelabs_integration
```

### Step 2: Plugin-Based Usage

**Basic Usage:**
```python
import asyncio
from bubblelabs_plugin_system import get_plugin_registry

async def main():
    registry = get_plugin_registry()

    # Load plugin
    plugin = await registry.load_plugin("openevolve", config={
        "max_instance_age_seconds": 7 * 24 * 3600,
        "max_instances": 1000,
        "enable_auto_cleanup": True,
        "cleanup_interval_seconds": 3600,
    })

    # Start plugin
    await registry.start_plugin("openevolve")

    # Create workflow definition
    definition = await plugin.create_workflow_definition(
        problem_statement="Optimize quantum entanglement protocols",
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

    print(f"Created workflow: {definition.id}")

    # Control workflow
    instance_id = "some-instance-id"
    result = await plugin.control_workflow(instance_id, "start")
    print(f"Control result: {result}")

    # Get metrics
    metrics = await plugin.get_metrics()
    print(f"Metrics: {metrics}")

    # Health check
    is_healthy = await plugin.health_check()
    print(f"Healthy: {is_healthy}")

    # Shutdown
    await registry.unload_plugin("openevolve")

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 3: Creating Custom Plugins

**Example: Analytics Plugin**

```python
from bubblelabs_plugin_system import BubbleLabsPlugin, PluginMetadata, PluginPriority
from typing import Dict, Any

class AnalyticsPlugin(BubbleLabsPlugin):
    """Plugin for workflow analytics and reporting."""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="analytics",
            version="1.0.0",
            author="Data Team",
            description="Workflow analytics and reporting plugin",
            dependencies=["openevolve"],  # Depends on OpenEvolve plugin
            priority=PluginPriority.NORMAL,
            category="analytics",
            tags=["analytics", "reporting", "metrics"],
        )

    async def initialize(self) -> None:
        """Initialize analytics plugin."""
        self._logger.info("Initializing Analytics plugin")

        # Setup database connection
        # Create tables
        # Initialize metrics collectors

        self._status.state = PluginState.INITIALIZED
        self._status.health = "healthy"

    async def start(self) -> None:
        """Start analytics collection."""
        self._logger.info("Starting Analytics plugin")

        # Start background collection tasks
        # Subscribe to workflow events
        # Begin aggregating metrics

        self._status.state = PluginState.STARTED
        self._status.health = "healthy"

    async def stop(self) -> None:
        """Stop analytics collection."""
        self._logger.info("Stopping Analytics plugin")

        # Stop background tasks
        # Flush metrics to database
        # Close connections

        self._status.state = PluginState.STOPPED
        self._status.health = "healthy"

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._logger.info("Cleaning up Analytics plugin")

        # Close database connections
        # Cleanup temporary files

        self._status.state = PluginState.UNLOADED

    def register_hooks(self, event_bus: EventBus) -> None:
        """Register event hooks."""
        super().register_hooks(event_bus)

        # Subscribe to OpenEvolve workflow events
        async def on_workflow_created(event):
            if event.plugin_name == "openevolve":
                self._logger.info(f"Workflow created: {event.data.get('definition_id')}")
                # Track in analytics

        async def on_workflow_controlled(event):
            if event.plugin_name == "openevolve":
                action = event.data.get("action_type")
                self._logger.info(f"Workflow {action}: {event.data.get('instance_id')}")
                # Update metrics

        event_bus.subscribe(PluginEvent.AFTER_START, on_workflow_created)
        event_bus.subscribe(PluginEvent.AFTER_START, on_workflow_controlled)

    async def generate_report(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """Generate analytics report."""
        # Query database for metrics in time range
        # Aggregate statistics
        # Generate visualizations
        return {
            "total_workflows": 100,
            "successful": 95,
            "failed": 5,
            "average_duration": 300.5,
        }

# Register the plugin
from bubblelabs_plugin_system import register_plugin
register_plugin(AnalyticsPlugin)
```

### Step 4: Backward Compatibility

The new system maintains backward compatibility through wrapper classes:

```python
# Old code still works
from openevolve_bubblelabs_plugin import bubblelabs_integration

# This will automatically use the plugin system under the hood
definition = bubblelabs_integration.create_workflow_definition_from_openevolve(
    problem_statement="Solve world hunger",
    team_config={"content_analyzer_team": "RedTeam"},
    gauntlet_config={"sub_problem_red_gauntlet": "PhysicsGauntlet"}
)

# Control workflows
result = bubblelabs_integration.control_workflow_local(instance_id, "start")
```

### Step 5: Event-Based Plugin Communication

**Example: Plugin that reacts to workflow events**

```python
from bubblelabs_plugin_system import BubbleLabsPlugin, PluginEvent, EventBus

class NotificationPlugin(BubbleLabsPlugin):
    """Plugin for sending notifications on workflow events."""

    def register_hooks(self, event_bus: EventBus) -> None:
        """Register event hooks."""

        async def on_workflow_complete(event):
            if event.plugin_name == "openevolve":
                instance_id = event.data.get("instance_id")
                result = event.data.get("result")

                # Send notification
                await self.send_notification(
                    f"Workflow {instance_id} completed with result: {result}"
                )

        async def on_workflow_error(event):
            if event.plugin_name == "openevolve":
                error = event.data.get("error")

                # Send alert
                await self.send_alert(f"Workflow error: {error}")

        event_bus.subscribe(PluginEvent.AFTER_START, on_workflow_complete)
        event_bus.subscribe(PluginEvent.ON_ERROR, on_workflow_error)

    async def send_notification(self, message: str):
        """Send notification."""
        # Implementation
        pass

    async def send_alert(self, message: str):
        """Send alert."""
        # Implementation
        pass

    # ... implement other required methods ...

# Register plugin
register_plugin(NotificationPlugin)
```

## Configuration Management

### Plugin Configuration

```python
# Load plugin with custom configuration
registry = get_plugin_registry()

plugin = await registry.load_plugin("openevolve", config={
    "max_instance_age_seconds": 14 * 24 * 3600,  # 14 days
    "max_instances": 5000,
    "enable_auto_cleanup": True,
    "cleanup_interval_seconds": 1800,  # 30 minutes
    "custom_setting": "value",
})
```

### Runtime Configuration Updates

```python
# Update plugin configuration at runtime
plugin = await registry.get_plugin("openevolve")
await plugin.update_config({
    "max_instances": 10000,
    "new_feature_enabled": True,
})
```

## Health Monitoring

### Check Plugin Health

```python
# Check single plugin
plugin = await registry.get_plugin("openevolve")
is_healthy = await plugin.health_check()
status = plugin.get_status()

print(f"Healthy: {is_healthy}")
print(f"State: {status.state}")
print(f"Health: {status.health}")
print(f"Message: {status.message}")
print(f"Metrics: {status.metrics}")
```

### Check All Plugins

```python
# Check health of all loaded plugins
health_status = await registry.check_all_health()

for plugin_name, is_healthy in health_status.items():
    print(f"{plugin_name}: {'✓' if is_healthy else '✗'}")
```

## Error Handling

### Graceful Degradation

```python
# Plugin loading handles errors gracefully
plugin = await registry.load_plugin("openevolve")

if plugin is None:
    logger.error("Failed to load OpenEvolve plugin")
    # Use fallback behavior
else:
    # Use plugin
    await plugin.start()
```

### Error Recovery

```python
# Get plugin status to check for errors
status = registry.get_plugin_status("openevolve")

if status.state == PluginState.ERROR:
    logger.error(f"Plugin error: {status.error}")

    # Attempt to reload
    await registry.unload_plugin("openevolve")
    plugin = await registry.load_plugin("openevolve")
```

## Testing

### Unit Testing Plugins

```python
import pytest
from bubblelabs_plugin_system import get_plugin_registry

@pytest.mark.asyncio
async def test_openevolve_plugin():
    """Test OpenEvolve plugin lifecycle."""
    registry = get_plugin_registry()

    # Load plugin
    plugin = await registry.load_plugin("openevolve")
    assert plugin is not None
    assert plugin.get_status().state == PluginState.INITIALIZED

    # Start plugin
    success = await registry.start_plugin("openevolve")
    assert success
    assert plugin.get_status().state == PluginState.STARTED

    # Create workflow
    definition = await plugin.create_workflow_definition(
        problem_statement="Test",
        team_config={},
        gauntlet_config={}
    )
    assert definition is not None

    # Health check
    is_healthy = await plugin.health_check()
    assert is_healthy

    # Unload
    success = await registry.unload_plugin("openevolve")
    assert success
```

## Best Practices

### 1. Plugin Design

- **Keep plugins focused**: Each plugin should have a single responsibility
- **Use dependency injection**: Pass dependencies through constructor
- **Handle errors gracefully**: Never let exceptions escape plugin methods
- **Log everything**: Use the plugin's logger for all operations

### 2. Lifecycle Management

- **Initialize in initialize()**: Setup resources in initialize(), not __init__
- **Start in start()**: Begin operations in start()
- **Stop in stop()**: Gracefully stop operations in stop()
- **Cleanup in cleanup()**: Release resources in cleanup()

### 3. Event Handling

- **Subscribe early**: Register hooks in register_hooks()
- **Handle errors**: Always wrap event handler code in try/except
- **Be async-aware**: Use async for event handlers
- **Unsubscribe when done**: Clean up event subscriptions

### 4. Configuration

- **Validate config**: Validate configuration in initialize()
- **Provide defaults**: Always have sensible defaults
- **Document schema**: Use JSON schema for complex configs
- **Support updates**: Handle runtime configuration changes

### 5. Dependencies

- **Declare explicitly**: List all dependencies in metadata
- **Check availability**: Verify dependencies are loaded
- **Order carefully**: Use priority to control load order
- **Avoid circular dependencies**: They will cause load failures

## Troubleshooting

### Plugin Won't Load

```python
# Check if plugin is registered
registry = get_plugin_registry()
plugins = registry.list_plugins()
print(f"Registered plugins: {list(plugins.keys())}")

# Check plugin status
status = registry.get_plugin_status("openevolve")
print(f"Status: {status}")
```

### Dependency Issues

```python
# Check dependency graph
registry = get_plugin_registry()
print(registry._dependency_graph)

# Verify dependencies are satisfied
for name, metadata in registry.list_plugins().items():
    print(f"{name}: depends on {metadata.dependencies}")
```

### Event Handlers Not Firing

```python
# Check event history
event_bus = registry.get_event_bus()
history = event_bus.get_history()

for event in history:
    print(f"{event.type.value} - {event.plugin_name}: {event.data}")
```

## Migration Checklist

- [ ] Update imports to use new plugin system
- [ ] Replace direct instantiation with plugin loading
- [ ] Update sync code to async (use wrappers if needed)
- [ ] Add configuration validation
- [ ] Implement health checks
- [ ] Add event handlers for inter-plugin communication
- [ ] Update tests
- [ ] Update documentation
- [ ] Test backward compatibility
- [ ] Deploy and monitor

## Support

For issues or questions:
- Check the event history for debugging
- Use the health check endpoints
- Review plugin logs
- Consult the API documentation

## Summary

The new plugin architecture provides:
- ✅ Proper lifecycle management
- ✅ Event-driven communication
- ✅ Dependency resolution
- ✅ Hot-reloading support
- ✅ Comprehensive logging
- ✅ Health monitoring
- ✅ Thread safety
- ✅ Backward compatibility

Migrate your code today to take advantage of these improvements!
