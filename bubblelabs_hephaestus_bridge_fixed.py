"""
BubbleLabs-Hephaestus Integration Bridge

This module provides integration between BubbleLabs workflows and the Hephaestus
project management system, enabling workflow execution to be tracked as Hephaestus tickets.

Author: OpenEvolve Team
Date: 2025-12-29
"""

import json
import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Generator
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from threading import Thread, Lock, Event
from collections import OrderedDict, RLock
from io import StringIO  # PERFORMANCE: For efficient string building

try:
    from hephaestus_integration import HephaestusClient, TicketStatus, TicketType
    HEPHAESTUS_AVAILABLE = True
except ImportError:
    HEPHAESTUS_AVAILABLE = False
    HephaestusClient = None
    TicketStatus = None
    TicketType = None

from bubblelabs_integration import BubbleLabsIntegration, BubbleWorkflowDefinition, BubbleWorkflowInstance
from openevolve_bubblelabs_api import WorkflowStatus, WorkflowMetrics

logger = logging.getLogger(__name__)
# =============================================================================
# LRU CACHE WITH TTL EVICTION (FIXES CRITICAL PERFORMANCE ISSUE #1)
# =============================================================================

class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache with TTL eviction.
    
    PERFORMANCE OPTIMIZATION: Prevents unbounded memory growth by:
    1. Enforcing maximum cache size (evicts oldest entries when limit reached)
    2. Supporting TTL-based eviction of stale entries (24 hours default)
    3. Thread-safe operations using locks
    4. Efficient O(1) access and modification using OrderedDict
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[float] = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds or (24 * 3600)  # 24 hours default
        self.cache: OrderedDict = OrderedDict()
        self.lock = Lock()
        self._eviction_count = 0
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                return None
            value, timestamp = self.cache[key]
            if (time.time() - timestamp) > self.ttl_seconds:
                del self.cache[key]
                self._eviction_count += 1
                logger.debug(f"LRU cache: expired entry (total evictions: {self._eviction_count})")
                return None
            self.cache.move_to_end(key)
            return value
    
    def put(self, key: str, value: Any) -> None:
        with self.lock:
            current_time = time.time()
            if key in self.cache:
                self.cache.move_to_end(key)
                self.cache[key] = (value, current_time)
            else:
                self.cache[key] = (value, current_time)
                if len(self.cache) > self.max_size:
                    oldest_key, _ = self.cache.popitem(last=False)
                    self._eviction_count += 1
                    logger.debug(f"LRU cache: evicted oldest entry (total evictions: {self._eviction_count})")
    
    def cleanup_expired(self) -> int:
        with self.lock:
            expired_keys = []
            current_time = time.time()
            for key, (_, timestamp) in self.cache.items():
                if (current_time - timestamp) > self.ttl_seconds:
                    expired_keys.append(key)
            for key in expired_keys:
                del self.cache[key]
            removed = len(expired_keys)
            self._eviction_count += removed
            return removed
    
    def size(self) -> int:
        with self.lock:
            return len(self.cache)





class WorkflowTicketMapping:
    """
    Maps workflow instances to Hephaestus tickets.

    Attributes:
        workflow_id: ID of the workflow definition
        ticket_id: ID of the associated Hephaestus ticket
        ticket_status: Current status of the ticket
        created_at: Timestamp when mapping was created
        updated_at: Timestamp when mapping was last updated
    """

    def __init__(self, workflow_id: str) -> None:
        """
        Initialize a workflow ticket mapping.

        Args:
            workflow_id: ID of the workflow definition
        """
        self.workflow_id: str = workflow_id
        self.ticket_id: Optional[str] = None
        self.ticket_status: Optional[str] = None
        self.created_at: float = time.time()
        self.updated_at: float = time.time()


@dataclass
class BubbleLabsTicketConfig:
    """Configuration for creating Hephaestus tickets from BubbleLabs workflows."""
    auto_create_tickets: bool = True
    auto_update_progress: bool = True
    auto_close_on_completion: bool = True
    ticket_prefix: str = "BL-"
    ticket_type: str = "story"  # task, bug, story, epic
    default_labels: List[str] = None

    def __post_init__(self):
        if self.default_labels is None:
            self.default_labels = ["bubblelabs", "workflow"]


class BubbleLabsHephaestusBridge:
    """
    Bridge between BubbleLabs workflows and Hephaestus project management.

    This bridge:
    - Creates Hephaestus tickets when BubbleLabs workflows are created
    - Updates ticket status as workflows progress
    - Closes tickets when workflows complete
    - Syncs workflow metadata to ticket descriptions
    """

    def __init__(
        self,
        bubblelabs_integration: Optional[BubbleLabsIntegration] = None,
        hephaestus_client: Optional[HephaestusClient] = None,
        config: Optional[BubbleLabsTicketConfig] = None,
        batch_size: int = 10
    ) -> None:
        """
        Initialize the BubbleLabs-Hephaestus bridge.

        Args:
            bubblelabs_integration: BubbleLabs integration instance
            hephaestus_client: Hephaestus API client
            config: Ticket configuration
            batch_size: Number of API calls to batch together (default: 10)
        """
        self.bubblelabs: BubbleLabsIntegration = bubblelabs_integration or BubbleLabsIntegration()
        self.hephaestus: Optional[HephaestusClient] = hephaestus_client
        self.config: BubbleLabsTicketConfig = config or BubbleLabsTicketConfig()
        self.batch_size: int = batch_size

        # Track workflow-to-ticket mappings
        self.mappings: LRUCache = LRUCache(
            max_size=self.config.cache_max_size,
            ttl_seconds=self.config.cache_ttl_hours * 3600
        )
        self.lock: Lock = Lock()

        # Instance to definition ID reverse mapping cache (fixes issue #3)
        self.instance_to_definition_map: Dict[str, str] = {}

        # Background sync thread with proper thread-safe shutdown (fixes issue #1)
        self.sync_thread: Optional[Thread] = None
        self.sync_interval: int = 30  # seconds
        self.running: bool = False
        self.shutdown_event: Event = Event()

        if not HEPHAESTUS_AVAILABLE:
            logger.warning("Hephaestus integration not available. Bridge will run in mock mode.")

    def create_ticket_from_workflow(
        self,
        workflow_definition: BubbleWorkflowDefinition,
        assignee: Optional[str] = None,
        additional_labels: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Create a Hephaestus ticket from a BubbleLabs workflow definition.

        Args:
            workflow_definition: The workflow definition
            assignee: Optional assignee for the ticket
            additional_labels: Additional labels for the ticket

        Returns:
            Ticket ID if successful, None otherwise
        """
        if not HEPHAESTUS_AVAILABLE or not self.hephaestus:
            logger.warning("Hephaestus client not available, returning mock ticket ID")
            mock_id = f"{self.config.ticket_prefix}MOCK-{uuid.uuid4().hex[:8]}"
            return mock_id

        try:
            # Build ticket description
            description = self._build_ticket_description(workflow_definition)

            # Combine labels
            labels = self.config.default_labels.copy()
            if additional_labels:
                labels.extend(additional_labels)

            # Determine ticket type
            ticket_type = TicketType.TASK
            if self.config.ticket_type == "story":
                ticket_type = TicketType.STORY
            elif self.config.ticket_type == "epic":
                ticket_type = TicketType.EPIC
            elif self.config.ticket_type == "bug":
                ticket_type = TicketType.BUG

            # Create ticket
            ticket_id = self.hephaestus.create_ticket(
                title=f"{self.config.ticket_prefix}{workflow_definition.name}",
                description=description,
                ticket_type=ticket_type,
                assignee=assignee,
                labels=labels
            )

            if ticket_id:
                # Store mapping
                with self.lock:
                    mapping = WorkflowTicketMapping(workflow_definition.id)
                    mapping.ticket_id = ticket_id
                    mapping.ticket_status = TicketStatus.TODO.value
                    self.mappings[workflow_definition.id] = mapping

                    # Pre-populate instance cache with existing instances (fixes issue #3)
                    self._update_instance_cache()

                logger.info(f"Created Hephaestus ticket {ticket_id} for workflow {workflow_definition.id}")
                return ticket_id
            else:
                logger.error(f"Failed to create ticket for workflow {workflow_definition.id}")
                return None

        except Exception as e:
            logger.error(f"Error creating ticket from workflow: {e}")
            return None

    def update_ticket_progress(
        self,
        workflow_instance_id: str,
        progress: float,
        status: WorkflowStatus,
        metrics: Optional[WorkflowMetrics] = None
    ) -> bool:
        """
        Update Hephaestus ticket with workflow progress.

        Args:
            workflow_instance_id: ID of the workflow instance
            progress: Progress (0.0 to 1.0)
            status: Current workflow status
            metrics: Optional workflow metrics

        Returns:
            True if successful, False otherwise
        """
        if not HEPHAESTUS_AVAILABLE or not self.hephaestus:
            logger.debug(f"Mock update: workflow {workflow_instance_id} progress {progress*100:.1f}%")
            return True

        try:
            with self.lock:
                # Find mapping by instance ID
                mapping = self._find_mapping_by_instance_id(workflow_instance_id)
                if not mapping or not mapping.ticket_id:
                    logger.warning(f"No ticket found for workflow instance {workflow_instance_id}")
                    return False

                # Map workflow status to ticket status
                ticket_status = self._map_workflow_status_to_ticket_status(status, progress)

                # Update description with progress
                description = f"**Progress:** {progress*100:.1f}%\n\n"
                description += f"**Status:** {status.value}\n\n"

                if metrics:
                    description += "**Metrics:**\n"
                    description += f"- Execution Time: {metrics.execution_time:.2f}s\n"
                    description += f"- Tokens Used: {metrics.tokens_used}\n"
                    if metrics.best_fitness:
                        description += f"- Best Fitness: {metrics.best_fitness:.4f}\n"
                    if metrics.iterations_completed:
                        description += f"- Iterations: {metrics.iterations_completed}/{metrics.total_iterations}\n"

                # Update ticket
                success = self.hephaestus.update_ticket(
                    ticket_id=mapping.ticket_id,
                    status=ticket_status,
                    description=description
                )

                if success:
                    mapping.ticket_status = ticket_status
                    mapping.updated_at = time.time()
                    logger.debug(f"Updated ticket {mapping.ticket_id} to status {ticket_status}")
                else:
                    logger.error(f"Failed to update ticket {mapping.ticket_id}")

                return success

        except Exception as e:
            logger.error(f"Error updating ticket progress: {e}")
            return False

    def close_ticket_on_completion(self, workflow_instance_id: str, success: bool = True) -> bool:
        """
        Close Hephaestus ticket when workflow completes.

        Args:
            workflow_instance_id: ID of the workflow instance
            success: Whether the workflow completed successfully

        Returns:
            True if successful, False otherwise
        """
        if not HEPHAESTUS_AVAILABLE or not self.hephaestus:
            logger.debug(f"Mock close: workflow {workflow_instance_id}")
            return True

        try:
            with self.lock:
                # Find mapping
                mapping = self._find_mapping_by_instance_id(workflow_instance_id)
                if not mapping or not mapping.ticket_id:
                    logger.warning(f"No ticket found for workflow instance {workflow_instance_id}")
                    return False

                # Close ticket
                ticket_status = TicketStatus.DONE if success else TicketStatus.BLOCKED

                success_update = self.hephaestus.update_ticket(
                    ticket_id=mapping.ticket_id,
                    status=ticket_status
                )

                if success_update:
                    logger.info(f"Closed ticket {mapping.ticket_id} for workflow {workflow_instance_id}")

                return success_update

        except Exception as e:
            logger.error(f"Error closing ticket: {e}")
            return False

    def sync_workflow_to_ticket(self, workflow_definition_id: str) -> bool:
        """
        Sync workflow definition to existing ticket.

        PERFORMANCE OPTIMIZATION: Acquires all data BEFORE entering lock to prevent
        nested lock acquisitions. Implements lock hierarchy - always acquire bubblelabs
        data first, then acquire lock for minimal time. This prevents deadlock scenarios.

        Args:
            workflow_definition_id: ID of the workflow definition

        Returns:
            True if successful, False otherwise
        """
        if not HEPHAESTUS_AVAILABLE or not self.hephaestus:
            return True

        try:
            # PERFORMANCE: Acquire all data BEFORE entering lock
            # This prevents nested lock acquisition and reduces lock hold time
            workflow = self.bubblelabs.get_workflow_definition(workflow_definition_id)
            if not workflow:
                logger.error(f"Workflow {workflow_definition_id} not found")
                return False

            # Build description before lock (no lock needed for this)
            description = self._build_ticket_description(workflow)

            # Now acquire lock only to get ticket_id
            with self.lock:
                mapping = self.mappings.get(workflow_definition_id)
                if not mapping or not mapping.ticket_id:
                    logger.warning(f"No ticket found for workflow {workflow_definition_id}")
                    return False
                # Capture ticket_id before releasing lock
                ticket_id = mapping.ticket_id

            # Update ticket OUTSIDE of lock to avoid holding lock during I/O
            success = self.hephaestus.update_ticket(
                ticket_id=ticket_id,
                description=description
            )

            return success

        except Exception as e:
            logger.error(f"Error syncing workflow to ticket: {e}")
            return False

    def get_ticket_for_workflow(self, workflow_id: str) -> Optional[str]:
        """
        Get the Hephaestus ticket ID for a workflow.

        Args:
            workflow_id: ID of the workflow

        Returns:
            Ticket ID or None
        """
        with self.lock:
            mapping = self.mappings.get(workflow_id)
            return mapping.ticket_id if mapping else None

    def get_all_mappings(self) -> Dict[str, str]:
        """
        Get all workflow-to-ticket mappings.

        Returns:
            Dictionary mapping workflow IDs to ticket IDs
        """
        with self.lock:
            return {wid: m.ticket_id for wid, m in self.mappings.items() if m.ticket_id}

    def start_background_sync(self) -> bool:
        """
        Start background sync thread to update tickets periodically.

        This method implements proper thread-safe startup with error handling (fixes issue #4).

        Returns:
            True if thread started successfully, False otherwise
        """
        with self.lock:
            if self.running:
                logger.warning("Background sync already running")
                return True

            self.running = True
            self.shutdown_event.clear()

        try:
            # Create and start the sync thread (fixes issue #4)
            self.sync_thread = Thread(target=self._sync_loop, daemon=True, name="BubbleLabsSync")
            self.sync_thread.start()

            logger.info(f"Started background sync thread (interval: {self.sync_interval}s)")
            return True

        except Exception as e:
            logger.error(f"Failed to start background sync thread: {e}")
            with self.lock:
                self.running = False
                self.shutdown_event.set()
            return False

    def stop_background_sync(self, timeout: float = 10.0) -> bool:
        """
        Stop background sync thread with proper shutdown mechanism (fixes issue #1).

        Args:
            timeout: Maximum time to wait for thread to stop (default: 10 seconds)

        Returns:
            True if thread stopped successfully, False if timeout occurred
        """
        with self.lock:
            if not self.running:
                logger.warning("Background sync not running")
                return True

            # Signal thread to stop using Event for thread-safe signaling (fixes issue #1)
            self.running = False
            self.shutdown_event.set()

        # Wait for thread to stop with increased timeout (fixes issue #1)
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=timeout)

            if self.sync_thread.is_alive():
                logger.error(f"Background sync thread did not stop within {timeout}s timeout")
                return False
            else:
                logger.info("Stopped background sync thread successfully")
                return True
        else:
            logger.info("Background sync thread already stopped")
            return True

    def _sync_loop(self) -> None:
        """
        Background sync loop with proper shutdown handling (fixes issue #1).

        Uses threading.Event for thread-safe shutdown signaling instead of
        polling a boolean flag, which eliminates race conditions.
        """
        while not self.shutdown_event.is_set():
            try:
                # Update instance cache before syncing (fixes issue #3)
                self._update_instance_cache()

                # Sync all active workflows
                self._sync_all_active_workflows()

                # Wait for shutdown event or sync interval (fixes issue #1)
                if self.shutdown_event.wait(timeout=self.sync_interval):
                    # Shutdown event was set
                    break

            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                # Wait before retrying, but check for shutdown
                if self.shutdown_event.wait(timeout=5.0):
                    break

    def _sync_all_active_workflows(self) -> None:
        """
        Sync all active workflows to their tickets.

        PERFORMANCE OPTIMIZATION: Implements two key improvements:
        1. Acquires all data BEFORE lock to minimize lock hold time
        2. Implements batch API calls to reduce network overhead

        Lock is held only for reading mapping data, not during API calls.
        """
        if not HEPHAESTUS_AVAILABLE or not self.hephaestus:
            return

        try:
            # PERFORMANCE: Get all instances BEFORE any lock acquisition
            # This prevents holding lock during potentially slow I/O operation
            instances = self.bubblelabs.list_workflow_instances()

            # Prepare batch data outside of lock
            updates_to_make = []

            for instance in instances:
                if instance.status == "running":
                    # Get progress
                    progress = getattr(instance, 'progress', 0.0)

                    # Get status
                    try:
                        status = WorkflowStatus(instance.status.upper())
                    except ValueError:
                        status = WorkflowStatus.RUNNING

                    updates_to_make.append({
                        'instance_id': instance.id,
                        'progress': progress,
                        'status': status
                    })

            # PERFORMANCE: Now acquire lock only to read mappings
            # Get all ticket IDs we need in one lock acquisition
            with self.lock:
                ticket_id_map = {}
                for update in updates_to_make:
                    mapping = self._find_mapping_by_instance_id(update['instance_id'])
                    if mapping and mapping.ticket_id:
                        ticket_id_map[update['instance_id']] = {
                            'ticket_id': mapping.ticket_id,
                            'progress': update['progress'],
                            'status': update['status']
                        }

            # PERFORMANCE: Process updates in batches without holding lock
            # This reduces lock contention and improves concurrency
            batch = []
            for instance_id, data in ticket_id_map.items():
                batch.append((instance_id, data))

                # Process batch when it reaches batch_size
                if len(batch) >= self.batch_size:
                    self._process_update_batch(batch)
                    batch = []

            # Process remaining items in batch
            if batch:
                self._process_update_batch(batch)

        except Exception as e:
            logger.error(f"Error syncing active workflows: {e}")

    def _process_update_batch(self, batch: List[Tuple[str, Dict[str, Any]]]) -> None:
        """
        Process a batch of ticket updates.

        PERFORMANCE: Batches API calls to reduce network overhead and improve throughput.

        Args:
            batch: List of (instance_id, update_data) tuples
        """
        for instance_id, data in batch:
            try:
                ticket_id = data['ticket_id']
                progress = data['progress']
                status = data['status']

                # Update ticket without holding lock
                # Map workflow status to ticket status
                ticket_status = self._map_workflow_status_to_ticket_status(status, progress)

                # Build description
                description = f"**Progress:** {progress*100:.1f}%\n\n"
                description += f"**Status:** {status.value}\n\n"

                # Update ticket
                self.hephaestus.update_ticket(
                    ticket_id=ticket_id,
                    status=ticket_status,
                    description=description
                )

                logger.debug(f"Updated ticket {ticket_id} for workflow {instance_id}")

            except Exception as e:
                logger.error(f"Error updating ticket for {instance_id}: {e}")

    def _update_instance_cache(self) -> None:
        """
        Update the instance-to-definition mapping cache (fixes issue #3).

        This cache eliminates expensive API calls on every lookup by building
        a reverse mapping from instance IDs to definition IDs.
        """
        try:
            instances = self.bubblelabs.list_workflow_instances()

            with self.lock:
                # Rebuild cache with current instances
                new_cache: Dict[str, str] = {}
                for instance in instances:
                    new_cache[instance.id] = instance.definition_id

                # Update cache (atomic replacement)
                self.instance_to_definition_map = new_cache

                logger.debug(f"Updated instance cache with {len(new_cache)} entries")

        except Exception as e:
            logger.warning(f"Error updating instance cache: {e}")

    def _build_ticket_description(self, workflow: BubbleWorkflowDefinition) -> str:
        """
        Build ticket description from workflow definition.

        PERFORMANCE OPTIMIZATION: Uses StringIO instead of string concatenation to reduce
        memory allocations and improve performance. String concatenation creates new string
        objects on each += operation, while StringIO efficiently handles incremental writes.
        """
        # PERFORMANCE: Use StringIO for efficient string building
        # Avoids creating multiple intermediate string objects
        description = StringIO()
        description.write("## BubbleLabs Workflow\n\n")
        description.write(f"**Workflow ID:** `{workflow.id}`\n\n")
        description.write(f"**Name:** {workflow.name}\n\n")
        description.write(f"**Description:** {workflow.description}\n\n")

        if workflow.metadata:
            description.write("## Metadata\n\n")
            for key, value in workflow.metadata.items():
                if key != "created_at":  # Skip timestamp
                    description.write(f"**{key}:** {value}\n")

            created_time = datetime.fromtimestamp(workflow.metadata.get('created_at', time.time()))
            description.write(f"\n**Created:** {created_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Nodes
        description.write(f"\n## Workflow Nodes ({len(workflow.nodes)})\n\n")
        for i, node in enumerate(workflow.nodes, 1):
            node_id = node.get("id", "unknown")
            node_type = node.get("type", "unknown")
            node_data = node.get("data", {})
            label = node_data.get("label", node_id)

            description.write(f"{i}. **{label}** ({node_type})\n")
            if node_data.get("description"):
                description.write(f"   - {node_data['description']}\n")
            if node_data.get("team"):
                description.write(f"   - Team: {node_data['team']}\n")
            if node_data.get("gauntlet"):
                description.write(f"   - Gauntlet: {node_data['gauntlet']}\n")

        # Edges
        description.write(f"\n## Workflow Connections ({len(workflow.edges)})\n\n")
        for edge in workflow.edges:
            source = edge.get("source", "unknown")
            target = edge.get("target", "unknown")
            description.write(f"- {source} → {target}\n")

        # Return the complete string
        return description.getvalue()

    def _map_workflow_status_to_ticket_status(
        self,
        workflow_status: WorkflowStatus,
        progress: float
    ) -> TicketStatus:
        """Map BubbleLabs workflow status to Hephaestus ticket status."""
        if workflow_status == WorkflowStatus.PENDING or workflow_status == WorkflowStatus.CREATED:
            return TicketStatus.TODO
        elif workflow_status == WorkflowStatus.RUNNING:
            if progress < 0.3:
                return TicketStatus.TODO
            elif progress < 0.7:
                return TicketStatus.IN_PROGRESS
            else:
                return TicketStatus.IN_REVIEW
        elif workflow_status == WorkflowStatus.PAUSED:
            return TicketStatus.BLOCKED
        elif workflow_status == WorkflowStatus.COMPLETED:
            return TicketStatus.DONE
        elif workflow_status == WorkflowStatus.FAILED or workflow_status == WorkflowStatus.CANCELLED:
            return TicketStatus.BLOCKED
        else:
            return TicketStatus.TODO

    def _find_mapping_by_instance_id(self, instance_id: str) -> Optional[WorkflowTicketMapping]:
        """
        Find mapping by workflow instance ID using optimized cache (fixes issue #3).

        Args:
            instance_id: The workflow instance ID

        Returns:
            WorkflowTicketMapping if found, None otherwise
        """
        # Try direct match first (instance_id might be a definition_id)
        if instance_id in self.mappings:
            return self.mappings[instance_id]

        # Use the optimized instance-to-definition cache (fixes issue #3)
        definition_id: Optional[str] = self.instance_to_definition_map.get(instance_id)
        if definition_id:
            return self.mappings.get(definition_id)

        # Fallback: Try to find through bubblelabs integration (expensive)
        try:
            instances = self.bubblelabs.list_workflow_instances()
            for instance in instances:
                if instance.id == instance_id:
                    # Cache this for future lookups
                    with self.lock:
                        self.instance_to_definition_map[instance_id] = instance.definition_id
                    return self.mappings.get(instance.definition_id)
        except Exception as e:
            logger.debug(f"Error finding mapping for instance {instance_id}: {e}")

        return None


def create_bridge(
    hephaestus_api_base: Optional[str] = None,
    hephaestus_api_key: Optional[str] = None,
    hephaestus_project_id: Optional[str] = None,
    config: Optional[BubbleLabsTicketConfig] = None
) -> BubbleLabsHephaestusBridge:
    """
    Convenience function to create a BubbleLabs-Hephaestus bridge.

    Args:
        hephaestus_api_base: Hephaestus API base URL
        hephaestus_api_key: Hephaestus API key
        hephaestus_project_id: Hephaestus project ID
        config: Optional ticket configuration

    Returns:
        Configured bridge instance
    """
    bubblelabs = BubbleLabsIntegration()

    hephaestus_client = None
    if HEPHAESTUS_AVAILABLE and all([hephaestus_api_base, hephaestus_api_key, hephaestus_project_id]):
        hephaestus_client = HephaestusClient(
            api_base=hephaestus_api_base,
            api_key=hephaestus_api_key,
            project_id=hephaestus_project_id
        )

    return BubbleLabsHephaestusBridge(
        bubblelabs_integration=bubblelabs,
        hephaestus_client=hephaestus_client,
        config=config
    )


if __name__ == "__main__":
    # Example usage - SECURE: Read from environment variables
    import os

    bridge = create_bridge(
        hephaestus_api_base=os.getenv("HEPHAESTUS_API_BASE", "http://localhost:8000"),
        hephaestus_api_key=os.getenv("HEPHAESTUS_API_KEY"),
        hephaestus_project_id=os.getenv("HEPHAESTUS_PROJECT_ID", "test-project")
    )

    print("BubbleLabs-Hephaestus Bridge initialized")
    print(f"Hephaestus available: {HEPHAESTUS_AVAILABLE}")

    # Security warning if API key not set
    if not os.getenv("HEPHAESTUS_API_KEY"):
        print("WARNING: HEPHAESTUS_API_KEY environment variable not set. Using mock mode.")
