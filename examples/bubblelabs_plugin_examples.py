"""
BubbleLabs Plugin System - Complete Examples

This file contains comprehensive examples of using the BubbleLabs plugin architecture.

Author: OpenEvolve Integration Team
Created: 2026-01-03
"""

import asyncio
import logging
from typing import Dict, Any, List

from bubblelabs_plugin_system import (
    BubbleLabsPlugin,
    PluginEvent,
    PluginMetadata,
    PluginPriority,
    PluginState,
    PluginStatus,
    EventBus,
    get_plugin_registry,
    register_plugin,
)
from openevolve_bubblelabs_plugin import OpenEvolveBubbleLabsPlugin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: Basic Plugin Usage
# ============================================================================

async def example_1_basic_usage():
    """Example 1: Basic plugin loading and usage."""

    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Plugin Usage")
    print("=" * 80 + "\n")

    registry = get_plugin_registry()

    # Load the OpenEvolve plugin
    print("Loading OpenEvolve plugin...")
    plugin = await registry.load_plugin("openevolve", config={
        "max_instance_age_seconds": 7 * 24 * 3600,
        "max_instances": 1000,
        "enable_auto_cleanup": True,
        "cleanup_interval_seconds": 3600,
    })

    if not plugin:
        print("[FAIL] Failed to load plugin")
        return

    print(f"[OK] Plugin loaded: {plugin.get_metadata().name}")
    print(f"  Version: {plugin.get_metadata().version}")
    print(f"  Author: {plugin.get_metadata().author}")
    print(f"  Description: {plugin.get_metadata().description}")

    # Start the plugin
    print("\nStarting plugin...")
    success = await registry.start_plugin("openevolve")
    print(f"[OK] Plugin started: {success}")

    # Check status
    status = plugin.get_status()
    print(f"\nPlugin Status:")
    print(f"  State: {status.state.value}")
    print(f"  Health: {status.health}")
    print(f"  Message: {status.message}")

    # Health check
    is_healthy = await plugin.health_check()
    print(f"\n[OK] Health check: {'PASS' if is_healthy else 'FAIL'}")

    # Create a workflow definition
    print("\nCreating workflow definition...")
    definition = await plugin.create_workflow_definition(
        problem_statement="Design quantum-resistant cryptographic protocols",
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

    print(f"[OK] Workflow definition created:")
    print(f"  ID: {definition.id}")
    print(f"  Name: {definition.name}")
    print(f"  Description: {definition.description}")
    print(f"  Nodes: {len(definition.nodes)}")
    print(f"  Edges: {len(definition.edges)}")

    # Get metrics
    metrics = await plugin.get_metrics()
    print(f"\nPlugin Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Cleanup
    print("\nUnloading plugin...")
    await registry.unload_plugin("openevolve")
    print("[OK] Plugin unloaded")


# ============================================================================
# EXAMPLE 2: Custom Plugin
# ============================================================================

class WorkflowAnalyticsPlugin(BubbleLabsPlugin):
    """
    Custom plugin for workflow analytics.

    Tracks workflow events and generates reports.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._workflows_created = 0
        self._workflows_started = 0
        self._workflows_cancelled = 0
        self._errors = 0

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="workflow_analytics",
            version="1.0.0",
            author="Analytics Team",
            description="Real-time workflow analytics and reporting",
            dependencies=["openevolve"],  # Depends on OpenEvolve
            priority=PluginPriority.NORMAL,
            category="analytics",
            tags=["analytics", "reporting", "monitoring"],
        )

    async def initialize(self) -> None:
        """Initialize analytics plugin."""
        self._logger.info("Initializing Workflow Analytics plugin")

        # Setup analytics infrastructure
        self._status.state = PluginState.INITIALIZED
        self._status.health = "healthy"
        self._status.message = "Analytics plugin initialized"

    async def start(self) -> None:
        """Start analytics collection."""
        self._logger.info("Starting Workflow Analytics plugin")

        self._status.state = PluginState.STARTED
        self._status.health = "healthy"
        self._status.message = "Analytics collection started"

    async def stop(self) -> None:
        """Stop analytics collection."""
        self._logger.info("Stopping Workflow Analytics plugin")

        # Generate final report
        report = self.generate_report()
        self._logger.info(f"Final report: {report}")

        self._status.state = PluginState.STOPPED
        self._status.health = "healthy"

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._logger.info("Cleaning up Workflow Analytics plugin")
        self._status.state = PluginState.UNLOADED

    def register_hooks(self, event_bus: EventBus) -> None:
        """Register event hooks."""
        super().register_hooks(event_bus)

        # Track workflow creation
        async def on_workflow_created(event):
            if event.plugin_name == "openevolve":
                if event.data.get("action") == "workflow_definition_created":
                    self._workflows_created += 1
                    self._logger.info(
                        f"Workflow created: {event.data.get('definition_id')}"
                    )

        # Track workflow control actions
        async def on_workflow_controlled(event):
            if event.plugin_name == "openevolve":
                if event.data.get("action") == "workflow_controlled":
                    action = event.data.get("action_type")
                    if action == "start":
                        self._workflows_started += 1
                    elif action == "cancel":
                        self._workflows_cancelled += 1

                    self._logger.info(
                        f"Workflow {action}: {event.data.get('instance_id')}"
                    )

        # Track errors
        async def on_error(event):
            if event.plugin_name == "openevolve":
                self._errors += 1
                self._logger.error(f"Workflow error: {event.data.get('error')}")

        event_bus.subscribe(PluginEvent.AFTER_START, on_workflow_created)
        event_bus.subscribe(PluginEvent.AFTER_START, on_workflow_controlled)
        event_bus.subscribe(PluginEvent.ON_ERROR, on_error)

    def generate_report(self) -> Dict[str, Any]:
        """Generate analytics report."""
        return {
            "workflows_created": self._workflows_created,
            "workflows_started": self._workflows_started,
            "workflows_cancelled": self._workflows_cancelled,
            "errors": self._errors,
            "success_rate": (
                self._workflows_started / max(self._workflows_created, 1) * 100
                if self._workflows_created > 0
                else 0
            ),
        }


async def example_2_custom_plugin():
    """Example 2: Creating and using a custom plugin."""

    print("\n" + "=" * 80)
    print("EXAMPLE 2: Custom Plugin with Event Hooks")
    print("=" * 80 + "\n")

    registry = get_plugin_registry()

    # Register the custom plugin
    print("Registering Workflow Analytics plugin...")
    register_plugin(WorkflowAnalyticsPlugin)

    # Load OpenEvolve (dependency)
    print("Loading OpenEvolve plugin...")
    await registry.load_plugin("openevolve")
    await registry.start_plugin("openevolve")

    # Load analytics plugin
    print("Loading Analytics plugin...")
    analytics = await registry.load_plugin("workflow_analytics")
    await registry.start_plugin("workflow_analytics")

    print(f"[OK] Analytics plugin loaded: {analytics.get_metadata().name}")

    # Create some workflows to trigger events
    print("\nCreating workflows...")
    openevolve = await registry.get_plugin("openevolve")

    for i in range(3):
        await openevolve.create_workflow_definition(
            problem_statement=f"Test problem {i}",
            team_config={"content_analyzer_team": "RedTeam"},
            gauntlet_config={"sub_problem_red_gauntlet": "PhysicsGauntlet"}
        )

    print("[OK] Created 3 workflows")

    # Get analytics report
    report = analytics.generate_report()
    print(f"\nAnalytics Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")

    # Cleanup
    print("\nCleaning up...")
    await registry.shutdown_all()
    print("[OK] All plugins unloaded")


# ============================================================================
# EXAMPLE 3: Plugin Discovery and Management
# ============================================================================

async def example_3_plugin_management():
    """Example 3: Plugin discovery, health checks, and management."""

    print("\n" + "=" * 80)
    print("EXAMPLE 3: Plugin Management")
    print("=" * 80 + "\n")

    registry = get_plugin_registry()

    # Register multiple plugins
    print("Registering plugins...")
    register_plugin(WorkflowAnalyticsPlugin)

    # List all registered plugins
    print("\nRegistered Plugins:")
    for name, metadata in registry.list_plugins().items():
        print(f"\n  {name}:")
        print(f"    Version: {metadata.version}")
        print(f"    Author: {metadata.author}")
        print(f"    Category: {metadata.category}")
        print(f"    Priority: {metadata.priority.name}")
        print(f"    Dependencies: {metadata.dependencies}")
        print(f"    Tags: {', '.join(metadata.tags)}")

    # Load all plugins
    print("\n" + "-" * 80)
    print("Loading plugins in dependency order...")
    print("-" * 80 + "\n")

    # Load OpenEvolve first (dependency)
    await registry.load_plugin("openevolve")
    await registry.start_plugin("openevolve")

    # Then load analytics (depends on OpenEvolve)
    await registry.load_plugin("workflow_analytics")
    await registry.start_plugin("workflow_analytics")

    # Check health of all plugins
    print("\n" + "-" * 80)
    print("Health Check Results")
    print("-" * 80 + "\n")

    health_status = await registry.check_all_health()
    for name, is_healthy in health_status.items():
        status = registry.get_plugin_status(name)
        print(f"{name}:")
        print(f"  Healthy: {'[OK]' if is_healthy else '[FAIL]'}")
        print(f"  State: {status.state.value if status else 'N/A'}")
        print(f"  Health: {status.health if status else 'N/A'}")
        print()

    # Get event history
    print("-" * 80)
    print("Event History")
    print("-" * 80 + "\n")

    event_bus = registry.get_event_bus()
    history = event_bus.get_history()

    print(f"Total events: {len(history)}\n")
    for event in history[-10:]:  # Show last 10 events
        print(f"{event.type.value} - {event.plugin_name}")
        if event.data:
            print(f"  Data: {event.data}")

    # Get plugin statistics
    print("\n" + "-" * 80)
    print("Plugin Statistics")
    print("-" * 80 + "\n")

    for name, plugin in registry._instances.items():
        metrics = await plugin.get_metrics() if hasattr(plugin, "get_metrics") else {}
        print(f"{name}:")
        if metrics:
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        print()

    # Cleanup
    print("\nShutting down all plugins...")
    shutdown_status = await registry.shutdown_all()
    for name, success in shutdown_status.items():
        print(f"  {name}: {'[OK]' if success else '[FAIL]'}")


# ============================================================================
# EXAMPLE 4: Error Handling and Recovery
# ============================================================================

async def example_4_error_handling():
    """Example 4: Error handling and recovery."""

    print("\n" + "=" * 80)
    print("EXAMPLE 4: Error Handling and Recovery")
    print("=" * 80 + "\n")

    registry = get_plugin_registry()

    # Try to load non-existent plugin
    print("Attempting to load non-existent plugin...")
    plugin = await registry.load_plugin("nonexistent")
    print(f"Result: {plugin}")  # Should be None
    print("[OK] Gracefully handled missing plugin\n")

    # Load OpenEvolve successfully
    print("Loading OpenEvolve plugin...")
    plugin = await registry.load_plugin("openevolve")
    await registry.start_plugin("openevolve")
    print(f"[OK] Plugin loaded: {plugin.get_metadata().name}\n")

    # Simulate error condition
    print("Simulating error condition...")
    status = plugin.get_status()
    print(f"Current state: {status.state.value}")
    print(f"Current health: {status.health}\n")

    # Check health
    is_healthy = await plugin.health_check()
    print(f"Health check result: {'PASS' if is_healthy else 'FAIL'}")

    # Get detailed status
    status = plugin.get_status()
    print(f"\nDetailed Status:")
    print(f"  State: {status.state.value}")
    print(f"  Health: {status.health}")
    print(f"  Message: {status.message}")
    print(f"  Error: {status.error}")
    print(f"  Metrics: {status.metrics}")

    # Recovery: unload and reload
    print("\nAttempting recovery...")
    await registry.unload_plugin("openevolve")
    plugin = await registry.load_plugin("openevolve")
    await registry.start_plugin("openevolve")
    print(f"[OK] Plugin recovered: {plugin is not None}")

    # Cleanup
    await registry.shutdown_all()
    print("\n[OK] All plugins unloaded")


# ============================================================================
# EXAMPLE 5: Backward Compatibility
# ============================================================================

async def example_5_backward_compatibility():
    """Example 5: Using backward compatible wrapper."""

    print("\n" + "=" * 80)
    print("EXAMPLE 5: Backward Compatibility")
    print("=" * 80 + "\n")

    from openevolve_bubblelabs_plugin import bubblelabs_integration

    print("Using backward compatible wrapper...")
    print("(This maintains the old API while using the new plugin system)\n")

    # Old API - still works!
    definition = bubblelabs_integration.create_workflow_definition_from_openevolve(
        problem_statement="Design quantum-resistant cryptographic protocols",
        team_config={
            "content_analyzer_team": "RedTeam",
            "planner_team": "PlannerTeam",
        },
        gauntlet_config={
            "sub_problem_red_gauntlet": "PhysicsGauntlet",
        }
    )

    print(f"[OK] Created workflow definition: {definition.id}")
    print(f"  Name: {definition.name}")
    print(f"  Nodes: {len(definition.nodes)}")
    print(f"  Edges: {len(definition.edges)}")

    print("\n[OK] Backward compatibility maintained!")


# ============================================================================
# MAIN RUNNER
# ============================================================================

async def main():
    """Run all examples."""

    print("\n")
    print("=" * 80)
    print(" BUBBLELABS PLUGIN SYSTEM - COMPLETE EXAMPLES")
    print("=" * 80)

    # Run examples
    await example_1_basic_usage()
    await example_2_custom_plugin()
    await example_3_plugin_management()
    await example_4_error_handling()
    await example_5_backward_compatibility()

    print("\n")
    print("=" * 80)
    print(" ALL EXAMPLES COMPLETED")
    print("=" * 80)
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
