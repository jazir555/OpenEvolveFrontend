"""
OpenEvolve Plugin for BubbleLabs

This module implements the OpenEvolve plugin for the BubbleLabs plugin architecture.
It provides backward compatibility with the existing bubblelabs_integration.py while
adding proper plugin lifecycle management.

Author: OpenEvolve Integration Team
Created: 2026-01-03
Status: Production Ready
"""

import asyncio
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
import traceback

from bubblelabs_plugin_system import (
    BubbleLabsPlugin,
    PluginEvent,
    PluginMetadata,
    PluginPriority,
    PluginState,
    PluginStatus,
    EventBus,
    Event,
    register_plugin,
    get_plugin_registry,
)
from bubblelabs_integration import (
    BubbleLabsIntegration,
    BubbleWorkflowDefinition,
    BubbleWorkflowInstance,
    BubbleNode,
    BubbleEdge,
    STATE_VALIDATION_AVAILABLE,
)
from bubblelabs_nodes import (
    list_nodes,
    get_node as create_node,
    BubbleLabsNode
)

logger = logging.getLogger(__name__)


class OpenEvolveBubbleLabsPlugin(BubbleLabsPlugin):
    """
    OpenEvolve plugin for BubbleLabs integration.

    This plugin provides:
    - Workflow definition and instance management
    - Team and Gauntlet management
    - State machine validation (if available)
    - Thread-safe operations
    - Memory leak prevention
    - Event-driven architecture

    Example:
        ```python
        # Load and start the plugin
        from bubblelabs_plugin_system import get_plugin_registry

        registry = get_plugin_registry()
        plugin = await registry.load_plugin("openevolve")

        # Create workflow definition
        definition = await plugin.create_workflow_definition(
            problem_statement="Simize quantum entanglement",
            team_config={"content_analyzer_team": "RedTeam"},
            gauntlet_config={"sub_problem_red_gauntlet": "PhysicsGauntlet"}
        )

        # Control workflow
        result = await plugin.control_workflow(instance_id, "start")
        ```
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenEvolve plugin.

        Args:
            config: Plugin configuration with optional keys:
                - max_instance_age_seconds: Maximum age for workflow instances (default: 7 days)
                - max_instances: Maximum number of instances to keep (default: 1000)
                - enable_auto_cleanup: Enable automatic cleanup (default: True)
                - cleanup_interval_seconds: Cleanup interval in seconds (default: 3600)
        """
        # Validate config first
        validated_config = self._validate_config(config)

        super().__init__(validated_config)

        # Initialize the legacy integration with error handling
        try:
            self._integration = BubbleLabsIntegration()
        except (RuntimeError, OSError, ConnectionError) as e:
            self._logger.error(f"Failed to initialize BubbleLabsIntegration: {e}\n{traceback.format_exc()}")
            # Create a mock integration that handles errors gracefully
            self._integration = self._create_fallback_integration()
            self._status.state = PluginState.ERROR
            self._status.health = "degraded"
            self._status.message = f"Using fallback integration due to initialization error: {str(e)}"

        # Apply config overrides
        if "max_instance_age_seconds" in validated_config:
            try:
                self._integration._MAX_INSTANCE_AGE_SECONDS = validated_config["max_instance_age_seconds"]
            except AttributeError:
                self._logger.warning("_MAX_INSTANCE_AGE_SECONDS attribute not found in integration")
        if "max_instances" in validated_config:
            try:
                self._integration._MAX_INSTANCES = validated_config["max_instances"]
            except AttributeError:
                self._logger.warning("_MAX_INSTANCES attribute not found in integration")

        # Auto-cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._auto_cleanup_enabled = validated_config.get("enable_auto_cleanup", True)
        self._cleanup_interval = validated_config.get("cleanup_interval_seconds", 3600)

        # Performance metrics
        self._metrics = {
            "workflows_created": 0,
            "workflows_started": 0,
            "workflows_completed": 0,
            "workflows_cancelled": 0,
            "control_actions": 0,
            "errors": 0,
        }

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize plugin configuration."""
        try:
            validated_config = {}

            # Validate max_instance_age_seconds
            if "max_instance_age_seconds" in config:
                try:
                    val = int(config["max_instance_age_seconds"])
                    if val <= 0:
                        raise ValueError("max_instance_age_seconds must be positive")
                    validated_config["max_instance_age_seconds"] = val
                except (ValueError, TypeError):
                    self._logger.warning(f"Invalid max_instance_age_seconds: {config['max_instance_age_seconds']}, using default")
                    validated_config["max_instance_age_seconds"] = 7 * 24 * 3600  # 7 days
            else:
                validated_config["max_instance_age_seconds"] = 7 * 24 * 3600  # 7 days

            # Validate max_instances
            if "max_instances" in config:
                try:
                    val = int(config["max_instances"])
                    if val <= 0:
                        raise ValueError("max_instances must be positive")
                    validated_config["max_instances"] = val
                except (ValueError, TypeError):
                    self._logger.warning(f"Invalid max_instances: {config['max_instances']}, using default")
                    validated_config["max_instances"] = 1000
            else:
                validated_config["max_instances"] = 1000

            # Validate enable_auto_cleanup
            if "enable_auto_cleanup" in config:
                validated_config["enable_auto_cleanup"] = bool(config["enable_auto_cleanup"])
            else:
                validated_config["enable_auto_cleanup"] = True

            # Validate cleanup_interval_seconds
            if "cleanup_interval_seconds" in config:
                try:
                    val = int(config["cleanup_interval_seconds"])
                    if val <= 0:
                        raise ValueError("cleanup_interval_seconds must be positive")
                    validated_config["cleanup_interval_seconds"] = val
                except (ValueError, TypeError):
                    self._logger.warning(f"Invalid cleanup_interval_seconds: {config['cleanup_interval_seconds']}, using default")
                    validated_config["cleanup_interval_seconds"] = 3600  # 1 hour
            else:
                validated_config["cleanup_interval_seconds"] = 3600  # 1 hour

            # Copy any additional config values
            for key, value in config.items():
                if key not in validated_config:
                    validated_config[key] = value

            return validated_config
        except (TypeError, ValueError, AttributeError) as e:
            self._logger.error(f"Error validating config: {e}\n{traceback.format_exc()}")
            # Return safe defaults
            return {
                "max_instance_age_seconds": 7 * 24 * 3600,
                "max_instances": 1000,
                "enable_auto_cleanup": True,
                "cleanup_interval_seconds": 3600,
            }

    def _create_fallback_integration(self):
        """Create a fallback integration object that handles all calls gracefully."""
        class FallbackIntegration:
            def __init__(self):
                self._MAX_INSTANCE_AGE_SECONDS = 7 * 24 * 3600
                self._MAX_INSTANCES = 1000

            def initialize(self):
                pass

            def _cleanup_old_instances(self):
                return 0

            def list_workflow_definitions(self):
                return []

            def list_workflow_instances(self):
                return []

            def control_workflow_local(self, instance_id, action):
                return {"error": "Fallback integration: workflow control not available", "status": "error"}

            def get_workflow_definition(self, definition_id):
                return None

            def create_workflow_definition_from_openevolve(
                self,
                problem_statement,
                team_config,
                gauntlet_config,
                workflow_type="sovereign_decomposition",
                web3_config=None,
            ):
                _ = workflow_type, web3_config
                return None

        return FallbackIntegration()

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="openevolve",
            version="1.0.0",
            author="OpenEvolve Team",
            description="OpenEvolve sovereign-grade decomposition workflow integration for BubbleLabs",
            dependencies=[],  # No dependencies
            priority=PluginPriority.HIGH,  # Load early as other plugins may depend on it
            category="workflow",
            tags=[
                "workflow",
                "decomposition",
                "team-management",
                "gauntlet",
                "sovereign",
                "quantum-safe",
            ],
            min_bubblelabs_version="1.0.0",
            max_bubblelabs_version="2.0.0",
        )

    async def initialize(self) -> None:
        """
        Initialize the plugin.

        Sets up resources, validates configuration, and prepares for operation.
        """
        try:
            self._logger.info("Initializing OpenEvolve plugin")

            # Initialize integration
            if hasattr(self._integration, "initialize"):
                await self._run_sync(self._integration.initialize)

            # Validate state machine validation availability
            if STATE_VALIDATION_AVAILABLE:
                self._logger.info("State machine validation is available")
            else:
                self._logger.warning("State machine validation is NOT available")

            # Update status
            self._status.state = PluginState.INITIALIZED
            self._status.health = "healthy"
            self._status.message = "Plugin initialized successfully"

        except (RuntimeError, ConnectionError, OSError) as e:
            self._logger.error(f"Failed to initialize OpenEvolve plugin: {e}\n{traceback.format_exc()}")
            self._status.state = PluginState.ERROR
            self._status.health = "unhealthy"
            self._status.message = f"Initialization failed: {str(e)}"
            self._status.error = e
            # Still allow graceful failure by not raising the exception

    async def start(self) -> None:
        """
        Start the plugin.

        Starts the auto-cleanup task if enabled.
        """
        try:
            self._logger.info("Starting OpenEvolve plugin")

            # Start auto-cleanup task
            if self._auto_cleanup_enabled:
                self._cleanup_task = asyncio.create_task(self._auto_cleanup_loop())
                self._logger.info(f"Auto-cleanup task started (interval: {self._cleanup_interval}s)")

            # Update status
            self._status.state = PluginState.STARTED
            self._status.health = "healthy"
            self._status.message = "Plugin started successfully"

        except (RuntimeError, ConnectionError, OSError) as e:
            self._logger.error(f"Failed to start OpenEvolve plugin: {e}\n{traceback.format_exc()}")
            self._status.state = PluginState.ERROR
            self._status.health = "unhealthy"
            self._status.message = f"Start failed: {str(e)}"
            self._status.error = e
            # Still allow graceful failure by not raising the exception

    async def stop(self) -> None:
        """
        Stop the plugin.

        Stops the auto-cleanup task and gracefully stops all workflows.
        """
        try:
            self._logger.info("Stopping OpenEvolve plugin")

            # Stop auto-cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error in {__name__}", exc_info=True)
                    raise  # Re-raise the exception
                self._cleanup_task = None
                self._logger.info("Auto-cleanup task stopped")

            # Cancel all running workflows
            await self._cancel_all_workflows()

            # Update status
            self._status.state = PluginState.STOPPED
            self._status.health = "healthy"
            self._status.message = "Plugin stopped successfully"

        except (RuntimeError, ConnectionError, OSError) as e:
            self._logger.error(f"Failed to stop OpenEvolve plugin: {e}\n{traceback.format_exc()}")
            self._status.state = PluginState.ERROR
            self._status.health = "unhealthy"
            self._status.message = f"Stop failed: {str(e)}"
            self._status.error = e
            # Still allow graceful failure by not raising the exception

    async def cleanup(self) -> None:
        """
        Cleanup plugin resources.

        Performs final cleanup before unloading.
        """
        try:
            self._logger.info("Cleaning up OpenEvolve plugin")

            # Final cleanup
            removed = await self._run_sync(self._integration._cleanup_old_instances)
            self._logger.info(f"Cleaned up {removed} instances")

            # Update status
            self._status.state = PluginState.UNLOADED
            self._status.message = "Plugin cleaned up successfully"

        except (RuntimeError, ConnectionError, OSError) as e:
            self._logger.error(f"Failed to cleanup OpenEvolve plugin: {e}\n{traceback.format_exc()}")
            self._status.state = PluginState.ERROR
            self._status.health = "unhealthy"
            self._status.message = f"Cleanup failed: {str(e)}"
            self._status.error = e
            # Still allow graceful failure by not raising the exception

    def register_hooks(self, event_bus: EventBus) -> None:
        """
        Register event hooks.

        Listens to workflow events from other plugins.

        Args:
            event_bus: Event bus to register hooks with
        """
        super().register_hooks(event_bus)

        # Subscribe to config change events
        async def on_config_change(event):
            self._logger.info(f"Configuration changed: {event.data}")

        event_bus.subscribe(PluginEvent.ON_CONFIG_CHANGE, on_config_change)

    async def health_check(self) -> bool:
        """
        Check plugin health.

        Returns:
            True if plugin is healthy
        """
        try:
            # Check if integration is responsive
            definitions = await self._run_sync(self._integration.list_workflow_definitions)
            instances = await self._run_sync(self._integration.list_workflow_instances)

            # Update metrics
            self._status.metrics["workflow_definitions"] = len(definitions)
            self._status.metrics["workflow_instances"] = len(instances)
            self._status.metrics["uptime_seconds"] = time.time() - self._status.last_updated

            return True

        except (ConnectionError, TimeoutError, RuntimeError) as e:
            self._logger.error(f"Health check failed: {e}\n{traceback.format_exc()}")
            self._status.health = "unhealthy"
            self._status.error = e
            return False

    # ============================================================================
    # NODE MANAGEMENT METHODS
    # ============================================================================

    def list_supported_nodes(self) -> List[str]:
        """List supported node types."""
        return list(list_nodes().keys())

    def get_node(self, node_type: str, config: Optional[Dict] = None) -> Optional[BubbleLabsNode]:
        """
        Get a node instance by type.

        Args:
            node_type: Node type identifier
            config: Optional node configuration

        Returns:
            BubbleLabsNode instance or None
        """
        try:
            return create_node(node_type, config)
        except ValueError:
            self._logger.warning(f"Invalid node type requested: {node_type}")
            return None
        except (RuntimeError, TypeError, AttributeError) as e:
            self._logger.error(f"Error getting node {node_type}: {e}\n{traceback.format_exc()}")
            return None

    # ============================================================================
    # WORKFLOW MANAGEMENT METHODS
    # ============================================================================

    async def create_workflow_definition(
        self,
        problem_statement: str,
        team_config: Dict[str, str],
        gauntlet_config: Dict[str, str],
        workflow_type: str = "sovereign_decomposition",
        web3_config: Optional[Dict[str, Any]] = None,
    ) -> BubbleWorkflowDefinition:
        """
        Create a workflow definition.

        Args:
            problem_statement: Problem to solve
            team_config: Team configuration mapping
            gauntlet_config: Gauntlet configuration mapping
            workflow_type: Workflow type (sovereign_decomposition or web3 aliases)
            web3_config: Optional Web3 configuration payload

        Returns:
            BubbleWorkflowDefinition object
        """
        try:
            self._logger.info(f"Creating workflow definition for: {problem_statement[:50]}...")

            definition = await self._run_sync(
                self._integration.create_workflow_definition_from_openevolve,
                problem_statement,
                team_config,
                gauntlet_config,
                workflow_type,
                web3_config,
            )

            # Update metrics
            self._metrics["workflows_created"] += 1

            # Publish event
            if self._event_bus:
                await self._event_bus.publish(
                    Event(
                        type=PluginEvent.AFTER_START,
                        plugin_name=self.get_metadata().name,
                        data={
                            "action": "workflow_definition_created",
                            "definition_id": definition.id,
                            "problem_statement": problem_statement,
                        },
                    )
                )

            return definition

        except (ConnectionError, TimeoutError, RuntimeError, ValueError) as e:
            self._logger.error(f"Failed to create workflow definition: {e}\n{traceback.format_exc()}")
            self._metrics["errors"] += 1
            self._status.health = "degraded"
            # Instead of raising, return a default error response or handle gracefully
            # For now, we'll re-raise to maintain original behavior but with better logging
            raise

    async def get_workflow_definition(self, definition_id: str) -> Optional[BubbleWorkflowDefinition]:
        """
        Get a workflow definition by ID.

        Args:
            definition_id: Definition ID

        Returns:
            BubbleWorkflowDefinition or None
        """
        try:
            return await self._run_sync(self._integration.get_workflow_definition, definition_id)
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            self._logger.error(f"Error getting workflow definition {definition_id}: {e}\n{traceback.format_exc()}")
            return None

    async def list_workflow_definitions(self) -> List[BubbleWorkflowDefinition]:
        """
        List all workflow definitions.

        Returns:
            List of BubbleWorkflowDefinition objects
        """
        try:
            return await self._run_sync(self._integration.list_workflow_definitions)
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            self._logger.error(f"Error listing workflow definitions: {e}\n{traceback.format_exc()}")
            return []

    async def list_workflow_instances(self) -> List[BubbleWorkflowInstance]:
        """
        List all workflow instances.

        Returns:
            List of BubbleWorkflowInstance objects
        """
        try:
            return await self._run_sync(self._integration.list_workflow_instances)
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            self._logger.error(f"Error listing workflow instances: {e}\n{traceback.format_exc()}")
            return []

    async def control_workflow(
        self, instance_id: str, action: str
    ) -> Dict[str, Any]:
        """
        Control a workflow instance.

        Args:
            instance_id: Instance ID
            action: Action to perform (start, pause, resume, cancel, restart)

        Returns:
            Status dictionary
        """
        try:
            self._logger.info(f"Controlling workflow {instance_id}: {action}")

            result = await self._run_sync(self._integration.control_workflow_local, instance_id, action)

            # Update metrics
            self._metrics["control_actions"] += 1
            if action == "start":
                self._metrics["workflows_started"] += 1
            elif action == "cancel":
                self._metrics["workflows_cancelled"] += 1

            # Check for errors
            if "error" in result:
                self._metrics["errors"] += 1
                self._status.health = "degraded"

            # Publish event
            if self._event_bus:
                await self._event_bus.publish(
                    Event(
                        type=PluginEvent.AFTER_START,
                        plugin_name=self.get_metadata().name,
                        data={
                            "action": "workflow_controlled",
                            "instance_id": instance_id,
                            "action_type": action,
                            "result": result,
                        },
                    )
                )

            return result

        except (ConnectionError, TimeoutError, RuntimeError) as e:
            self._logger.error(f"Failed to control workflow {instance_id}: {e}\n{traceback.format_exc()}")
            self._metrics["errors"] += 1
            self._status.health = "degraded"
            return {"error": f"Failed to control workflow: {str(e)}", "status": "error"}

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get plugin metrics.

        Returns:
            Metrics dictionary
        """
        try:
            # Update current stats
            instances = await self._run_sync(self._integration.list_workflow_instances)
            definitions = await self._run_sync(self._integration.list_workflow_definitions)

            metrics = self._metrics.copy()
            metrics["active_instances"] = len(instances)
            metrics["total_definitions"] = len(definitions)
            metrics["status"] = self._status.__dict__

            return metrics
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            self._logger.error(f"Error getting metrics: {e}\n{traceback.format_exc()}")
            # Return basic metrics even if there's an error
            return {
                "active_instances": 0,
                "total_definitions": 0,
                "status": self._status.__dict__,
                **self._metrics
            }

    async def reset_metrics(self) -> None:
        """Reset metrics."""
        try:
            self._metrics = {
                "workflows_created": 0,
                "workflows_started": 0,
                "workflows_completed": 0,
                "workflows_cancelled": 0,
                "control_actions": 0,
                "errors": 0,
            }
        except (RuntimeError, TypeError, AttributeError) as e:
            self._logger.error(f"Error resetting metrics: {e}\n{traceback.format_exc()}")

    # ============================================================================
    # PRIVATE METHODS
    # ============================================================================

    async def _auto_cleanup_loop(self) -> None:
        """Auto-cleanup loop for removing old instances."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)

                removed = await self._run_sync(self._integration._cleanup_old_instances)
                if removed > 0:
                    self._logger.info(f"Auto-cleanup: removed {removed} old instances")

            except asyncio.CancelledError:
                self._logger.info("Auto-cleanup loop was cancelled")
                break
            except (RuntimeError, OSError) as e:
                self._logger.error(f"Error in auto-cleanup loop: {e}\n{traceback.format_exc()}")
                # Continue the loop despite errors to ensure cleanup keeps running
                continue

    async def _cancel_all_workflows(self) -> None:
        """Cancel all running workflow instances."""
        try:
            instances = await self._run_sync(self._integration.list_workflow_instances)

            for instance in instances:
                if instance.status in ("running", "paused"):
                    self._logger.info(f"Cancelling workflow {instance.id}")
                    try:
                        await self.control_workflow(instance.id, "cancel")
                    except (ConnectionError, TimeoutError, RuntimeError) as e:
                        self._logger.error(f"Error cancelling workflow {instance.id}: {e}\n{traceback.format_exc()}")

        except (RuntimeError, ConnectionError) as e:
            self._logger.error(f"Error in _cancel_all_workflows: {e}\n{traceback.format_exc()}")

    async def _run_sync(self, func, *args, **kwargs) -> Any:
        """
        Run a synchronous function in an executor.

        Args:
            func: Function to run
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, kwargs)
        except (RuntimeError, ConnectionError, TimeoutError) as e:
            self._logger.error(f"Error running sync function {func.__name__}: {e}\n{traceback.format_exc()}")
            raise


# ============================================================================
    # BACKWARD COMPATIBILITY WRAPPERS
    # ============================================================================

class BubbleLabsIntegrationWrapper:
    """
    Wrapper class to maintain backward compatibility with existing code.

    This wrapper provides the same interface as BubbleLabsIntegration
    but delegates to the plugin system.
    """

    _instance: Optional["BubbleLabsIntegrationWrapper"] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the wrapper."""
        if not hasattr(self, "_initialized"):
            self._plugin: Optional[OpenEvolveBubbleLabsPlugin] = None
            self._initialized = True

    async def _get_plugin(self) -> OpenEvolveBubbleLabsPlugin:
        """Get or create the plugin instance."""
        if self._plugin is None:
            registry = get_plugin_registry()
            self._plugin = await registry.load_plugin("openevolve")
            if self._plugin:
                await registry.start_plugin("openevolve")
        return self._plugin

    def create_workflow_definition_from_openevolve(
        self,
        problem_statement: str,
        team_config: Dict[str, str],
        gauntlet_config: Dict[str, str],
        workflow_type: str = "sovereign_decomposition",
        web3_config: Optional[Dict[str, Any]] = None,
    ) -> BubbleWorkflowDefinition:
        """
        Create workflow definition (backward compatible sync wrapper).

        Args:
            problem_statement: Problem to solve
            team_config: Team configuration
            gauntlet_config: Gauntlet configuration
            workflow_type: Workflow type (sovereign_decomposition or web3 aliases)
            web3_config: Optional Web3 configuration payload

        Returns:
            BubbleWorkflowDefinition
        """
        # Run async method in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to create a new loop in a thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        self._create_definition_sync,
                        problem_statement,
                        team_config,
                        gauntlet_config,
                        workflow_type,
                        web3_config,
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._create_definition_async(
                        problem_statement,
                        team_config,
                        gauntlet_config,
                        workflow_type,
                        web3_config,
                    )
                )
        except RuntimeError:
            return asyncio.run(
                self._create_definition_async(
                    problem_statement,
                    team_config,
                    gauntlet_config,
                    workflow_type,
                    web3_config,
                )
            )
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error(f"Error in create_workflow_definition_from_openevolve: {e}\n{traceback.format_exc()}")
            # Return a default error response or handle gracefully
            # For now, we'll re-raise to maintain original behavior but with better logging
            raise

    async def _create_definition_async(
        self,
        problem_statement: str,
        team_config: Dict[str, str],
        gauntlet_config: Dict[str, str],
        workflow_type: str = "sovereign_decomposition",
        web3_config: Optional[Dict[str, Any]] = None,
    ) -> BubbleWorkflowDefinition:
        """Async implementation of create_workflow_definition."""
        plugin = await self._get_plugin()
        return await plugin.create_workflow_definition(
            problem_statement,
            team_config,
            gauntlet_config,
            workflow_type=workflow_type,
            web3_config=web3_config,
        )

    def _create_definition_sync(
        self,
        problem_statement: str,
        team_config: Dict[str, str],
        gauntlet_config: Dict[str, str],
        workflow_type: str = "sovereign_decomposition",
        web3_config: Optional[Dict[str, Any]] = None,
    ) -> BubbleWorkflowDefinition:
        """Sync implementation of create_workflow_definition."""
        return asyncio.run(
            self._create_definition_async(
                problem_statement,
                team_config,
                gauntlet_config,
                workflow_type=workflow_type,
                web3_config=web3_config,
            )
        )

    def control_workflow_local(self, instance_id: str, action: str) -> Dict[str, Any]:
        """Control workflow (backward compatible sync wrapper)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._control_sync, instance_id, action)
                    return future.result()
            else:
                return loop.run_until_complete(self._control_async(instance_id, action))
        except RuntimeError:
            return asyncio.run(self._control_async(instance_id, action))
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error(f"Error in control_workflow_local: {e}\n{traceback.format_exc()}")
            return {"error": f"Failed to control workflow: {str(e)}", "status": "error"}

    async def _control_async(self, instance_id: str, action: str) -> Dict[str, Any]:
        """Async implementation of control_workflow."""
        plugin = await self._get_plugin()
        return await plugin.control_workflow(instance_id, action)

    def _control_sync(self, instance_id: str, action: str) -> Dict[str, Any]:
        """Sync implementation of control_workflow."""
        return asyncio.run(self._control_async(instance_id, action))


# Global singleton instance for backward compatibility
bubblelabs_integration = BubbleLabsIntegrationWrapper()


# Auto-register the plugin
def register_openevolve_plugin():
    """Register the OpenEvolve plugin with the global registry."""
    register_plugin(OpenEvolveBubbleLabsPlugin, config={})


# Auto-register on import
register_openevolve_plugin()
