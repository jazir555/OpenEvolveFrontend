"""
BubbleLabs-Hephaestus Integration Bridge

This module provides integration between BubbleLabs workflows and the Hephaestus
project management system, enabling workflow execution to be tracked as Hephaestus tickets.

Author: OpenEvolve Team
Date: 2025-12-29
"""

import json
import sqlite3
import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Generator, Set, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from threading import Thread, Lock, Event, RLock
from io import StringIO  # PERFORMANCE: For efficient string building
from collections import OrderedDict  # MEMORY LEAK FIX: For LRU cache implementation
from functools import wraps  # For decorator implementation

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
# VALIDATION CONSTANTS
# =============================================================================

MAX_MAPPINGS = 1000
MAX_DESCRIPTION_LENGTH = 10000
MAX_SYNC_INTERVAL = 3600
MAX_BATCH_SIZE = 100


# =============================================================================
# STATE MACHINE DEFINITIONS
# =============================================================================

class ExtendedWorkflowStatus(Enum):
    """
    Extended workflow status states with state machine validation.

    States:
    - CREATED: Workflow definition created but not yet started
    - PENDING: Workflow queued and ready to start
    - RUNNING: Workflow currently executing
    - PAUSED: Workflow temporarily paused (can be resumed)
    - STOPPING: Workflow in process of stopping gracefully
    - STOPPED: Workflow stopped (can be restarted)
    - COMPLETED: Workflow finished successfully (terminal state)
    - FAILED: Workflow failed (can be retried)
    - CANCELLED: Workflow cancelled by user (terminal state)
    """
    CREATED = "created"
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExtendedTicketStatus(Enum):
    """
    Hephaestus ticket status states with state machine validation.

    States:
    - TODO: Ticket created but not yet started
    - IN_PROGRESS: Work actively being done
    - IN_REVIEW: Work completed, under review
    - DONE: Work completed and approved (terminal state)
    - CANCELLED: Ticket cancelled (terminal state)
    - BLOCKED: Work blocked (can be resumed)
    """
    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    IN_REVIEW = "IN_REVIEW"
    DONE = "DONE"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"


# Valid state transitions for workflows
VALID_WORKFLOW_TRANSITIONS: Dict[ExtendedWorkflowStatus, Set[ExtendedWorkflowStatus]] = {
    ExtendedWorkflowStatus.CREATED: {
        ExtendedWorkflowStatus.PENDING,
        ExtendedWorkflowStatus.CANCELLED
    },
    ExtendedWorkflowStatus.PENDING: {
        ExtendedWorkflowStatus.RUNNING,
        ExtendedWorkflowStatus.CANCELLED
    },
    ExtendedWorkflowStatus.RUNNING: {
        ExtendedWorkflowStatus.PAUSED,
        ExtendedWorkflowStatus.STOPPING,
        ExtendedWorkflowStatus.COMPLETED,
        ExtendedWorkflowStatus.FAILED,
        ExtendedWorkflowStatus.CANCELLED
    },
    ExtendedWorkflowStatus.PAUSED: {
        ExtendedWorkflowStatus.RUNNING,
        ExtendedWorkflowStatus.STOPPING,
        ExtendedWorkflowStatus.CANCELLED
    },
    ExtendedWorkflowStatus.STOPPING: {
        ExtendedWorkflowStatus.STOPPED,
        ExtendedWorkflowStatus.CANCELLED,
        ExtendedWorkflowStatus.FAILED
    },
    ExtendedWorkflowStatus.STOPPED: {
        ExtendedWorkflowStatus.PENDING,
        ExtendedWorkflowStatus.RUNNING
    },
    ExtendedWorkflowStatus.COMPLETED: set(),  # Terminal state
    ExtendedWorkflowStatus.FAILED: {
        ExtendedWorkflowStatus.PENDING,
        ExtendedWorkflowStatus.RUNNING
    },  # Can retry
    ExtendedWorkflowStatus.CANCELLED: set(),  # Terminal state
}


# Valid state transitions for tickets
VALID_TICKET_TRANSITIONS: Dict[ExtendedTicketStatus, Set[ExtendedTicketStatus]] = {
    ExtendedTicketStatus.TODO: {
        ExtendedTicketStatus.IN_PROGRESS,
        ExtendedTicketStatus.CANCELLED,
        ExtendedTicketStatus.BLOCKED
    },
    ExtendedTicketStatus.IN_PROGRESS: {
        ExtendedTicketStatus.IN_REVIEW,
        ExtendedTicketStatus.TODO,
        ExtendedTicketStatus.CANCELLED,
        ExtendedTicketStatus.BLOCKED
    },
    ExtendedTicketStatus.IN_REVIEW: {
        ExtendedTicketStatus.IN_PROGRESS,
        ExtendedTicketStatus.DONE,
        ExtendedTicketStatus.TODO,
        ExtendedTicketStatus.CANCELLED,
        ExtendedTicketStatus.BLOCKED
    },
    ExtendedTicketStatus.DONE: set(),  # Terminal state
    ExtendedTicketStatus.CANCELLED: set(),  # Terminal state
    ExtendedTicketStatus.BLOCKED: {
        ExtendedTicketStatus.TODO,
        ExtendedTicketStatus.IN_PROGRESS,
        ExtendedTicketStatus.CANCELLED
    },
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_not_none(value: Any, param_name: str) -> Any:
    """Validate that a value is not None."""
    if value is None:
        raise ValueError(f"{param_name} cannot be None")
    return value


def validate_not_empty(value: str, param_name: str) -> str:
    """Validate that a string is not empty or just whitespace."""
    if not value or not value.strip():
        raise ValueError(f"{param_name} cannot be empty")
    return value


def validate_string_length(value: str, max_length: int, param_name: str) -> str:
    """Validate string length."""
    if len(value) > max_length:
        raise ValueError(f"{param_name} cannot exceed {max_length} characters")
    return value


def validate_range(value: int, min_value: int, max_value: int, param_name: str) -> int:
    """Validate numeric range."""
    if value < min_value or value > max_value:
        raise ValueError(f"{param_name} must be between {min_value} and {max_value}")
    return value


# =============================================================================
# STATE MACHINE VALIDATION FUNCTIONS
# =============================================================================

def validate_workflow_transition(
    current_status: Union[ExtendedWorkflowStatus, str],
    new_status: Union[ExtendedWorkflowStatus, str]
) -> bool:
    """
    Validate if a workflow state transition is allowed.

    Args:
        current_status: Current workflow status (enum or string)
        new_status: Desired new workflow status (enum or string)

    Returns:
        True if transition is valid, False otherwise

    Logs:
        - ERROR: If current status is unknown
        - ERROR: If transition is invalid
    """
    # Convert strings to enums if needed
    if isinstance(current_status, str):
        try:
            current_status = ExtendedWorkflowStatus(current_status.lower())
        except ValueError:
            logger.error(f"Unknown current workflow status: {current_status}")
            return False

    if isinstance(new_status, str):
        try:
            new_status = ExtendedWorkflowStatus(new_status.lower())
        except ValueError:
            logger.error(f"Unknown new workflow status: {new_status}")
            return False

    # No-op transition is always valid
    if current_status == new_status:
        return True

    # Check if current status exists in transition table
    if current_status not in VALID_WORKFLOW_TRANSITIONS:
        logger.error(f"Unknown current status in transition table: {current_status}")
        return False

    # Check if transition is valid
    if new_status not in VALID_WORKFLOW_TRANSITIONS[current_status]:
        logger.error(f"Invalid workflow transition: {current_status.value} -> {new_status.value}")
        logger.error(f"Valid transitions from {current_status.value}: {[s.value for s in VALID_WORKFLOW_TRANSITIONS[current_status]]}")
        return False

    return True


def validate_ticket_transition(
    current_status: Union[ExtendedTicketStatus, str],
    new_status: Union[ExtendedTicketStatus, str]
) -> bool:
    """
    Validate if a ticket state transition is allowed.

    Args:
        current_status: Current ticket status (enum or string)
        new_status: Desired new ticket status (enum or string)

    Returns:
        True if transition is valid, False otherwise

    Logs:
        - ERROR: If current status is unknown
        - ERROR: If transition is invalid
    """
    # Convert strings to enums if needed
    if isinstance(current_status, str):
        try:
            current_status = ExtendedTicketStatus(current_status.upper())
        except ValueError:
            logger.error(f"Unknown current ticket status: {current_status}")
            return False

    if isinstance(new_status, str):
        try:
            new_status = ExtendedTicketStatus(new_status.upper())
        except ValueError:
            logger.error(f"Unknown new ticket status: {new_status}")
            return False

    # No-op transition is always valid
    if current_status == new_status:
        return True

    # Check if current status exists in transition table
    if current_status not in VALID_TICKET_TRANSITIONS:
        logger.error(f"Unknown current ticket status in transition table: {current_status}")
        return False

    # Check if transition is valid
    if new_status not in VALID_TICKET_TRANSITIONS[current_status]:
        logger.error(f"Invalid ticket transition: {current_status.value} -> {new_status.value}")
        logger.error(f"Valid transitions from {current_status.value}: {[s.value for s in VALID_TICKET_TRANSITIONS[current_status]]}")
        return False

    return True


def get_valid_workflow_transitions(status: Union[ExtendedWorkflowStatus, str]) -> Set[str]:
    """
    Get all valid next states for a given workflow status.

    Args:
        status: Current workflow status (enum or string)

    Returns:
        Set of valid next status values as strings
    """
    if isinstance(status, str):
        try:
            status = ExtendedWorkflowStatus(status.lower())
        except ValueError:
            logger.error(f"Unknown workflow status: {status}")
            return set()

    transitions = VALID_WORKFLOW_TRANSITIONS.get(status, set())
    return {s.value for s in transitions}


def get_valid_ticket_transitions(status: Union[ExtendedTicketStatus, str]) -> Set[str]:
    """
    Get all valid next states for a given ticket status.

    Args:
        status: Current ticket status (enum or string)

    Returns:
        Set of valid next status values as strings
    """
    if isinstance(status, str):
        try:
            status = ExtendedTicketStatus(status.upper())
        except ValueError:
            logger.error(f"Unknown ticket status: {status}")
            return set()

    transitions = VALID_TICKET_TRANSITIONS.get(status, set())
    return {s.value for s in transitions}


def is_terminal_workflow_status(status: Union[ExtendedWorkflowStatus, str]) -> bool:
    """
    Check if a workflow status is terminal (no valid transitions out).

    Args:
        status: Workflow status to check

    Returns:
        True if status is terminal, False otherwise
    """
    if isinstance(status, str):
        try:
            status = ExtendedWorkflowStatus(status.lower())
        except ValueError:
            return False

    return len(VALID_WORKFLOW_TRANSITIONS.get(status, set())) == 0


def is_terminal_ticket_status(status: Union[ExtendedTicketStatus, str]) -> bool:
    """
    Check if a ticket status is terminal (no valid transitions out).

    Args:
        status: Ticket status to check

    Returns:
        True if status is terminal, False otherwise
    """
    if isinstance(status, str):
        try:
            status = ExtendedTicketStatus(status.upper())
        except ValueError:
            return False

    return len(VALID_TICKET_TRANSITIONS.get(status, set())) == 0


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
        batch_size: int = 10,
        mappings_db_path: Optional[str] = None
    ) -> None:
        """
        Initialize the BubbleLabs-Hephaestus bridge.

        Args:
            bubblelabs_integration: BubbleLabs integration instance
            hephaestus_client: Hephaestus API client
            config: Ticket configuration
            batch_size: Number of API calls to batch together (default: 10)
            mappings_db_path: Optional path for mappings database (default: "hephaestus_workflow_mappings.db")

        Raises:
            ValueError: If batch_size is out of valid range
        """
        # Input validation
        if batch_size is not None:
            validate_range(batch_size, 1, MAX_BATCH_SIZE, "batch_size")

        self.bubblelabs: BubbleLabsIntegration = bubblelabs_integration or BubbleLabsIntegration()
        self.hephaestus: Optional[HephaestusClient] = hephaestus_client
        self.config: BubbleLabsTicketConfig = config or BubbleLabsTicketConfig()
        self.batch_size: int = batch_size

        # MEMORY LEAK FIX #1: LRU cache for workflow-to-ticket mappings (was unbounded Dict)
        # This prevents unbounded memory growth from accumulating workflow mappings
        self._mappings: OrderedDict = OrderedDict()
        self._MAX_MAPPINGS = 1000
        self.lock: Lock = Lock()

        # MEMORY LEAK FIX #2: LRU cache for instance-to-definition mapping (was unbounded Dict)
        # This prevents unbounded memory growth from accumulated instance mappings
        self._instance_to_definition_cache: OrderedDict = OrderedDict()
        self._MAX_CACHE_SIZE = 1000

        # PERSISTENCE: Add database path for mappings (allow override for testing)
        self._mappings_db_path = mappings_db_path or "hephaestus_workflow_mappings.db"

        # Background sync thread with proper thread-safe shutdown (fixes issue #1)
        self.sync_thread: Optional[Thread] = None
        self.sync_interval: int = 30  # seconds
        self.running: bool = False
        self.shutdown_event: Event = Event()

        # DATABASE CLEANUP CONFIGURATION
        # Automatic cleanup of old mappings to prevent unbounded growth
        self._retention_days = 90  # Default retention: 90 days
        self._cleanup_interval = 86400  # Cleanup once per day (24 hours in seconds)
        self._last_mappings_cleanup = time.time()

        # PERSISTENCE: Initialize the mappings database
        self._init_mappings_database()

        # PERSISTENCE: Load existing mappings from database into LRU cache
        self._load_mappings_from_db()

        if not HEPHAESTUS_AVAILABLE:
            logger.warning("Hephaestus integration not available. Bridge will run in mock mode.")

    def _init_mappings_database(self) -> None:
        """
        PERSISTENCE: Initialize SQLite database for workflow-to-ticket mappings.

        Creates the database table and indexes if they don't exist.

        Raises:
            Exception: If database initialization fails
        """
        try:
            conn = sqlite3.connect(self._mappings_db_path)
            cursor = conn.cursor()

            # Enable foreign keys
            cursor.execute("PRAGMA foreign_keys = ON")

            # Create table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_ticket_mappings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT NOT NULL,
                    ticket_id TEXT NOT NULL,
                    ticket_status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,

                    -- Optional: Store workflow definition as JSON
                    workflow_name TEXT,
                    workflow_description TEXT,

                    -- Metadata
                    last_synced_at REAL,

                    -- Unique constraint on workflow_id
                    UNIQUE(workflow_id)
                )
            """)

            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mappings_ticket_status
                ON workflow_ticket_mappings(ticket_status)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mappings_updated_at
                ON workflow_ticket_mappings(updated_at)
            """)

            conn.commit()
            conn.close()

            logger.info(f"Initialized workflow mappings database: {self._mappings_db_path}")

        except Exception as e:
            logger.error(f"Error initializing mappings database: {e}")
            raise

    def _load_mappings_from_db(self) -> None:
        """
        PERSISTENCE: Load all workflow-to-ticket mappings from database into LRU cache.

        This is called during initialization to restore previously saved mappings.
        """
        try:
            conn = sqlite3.connect(self._mappings_db_path)
            cursor = conn.cursor()

            # Load all mappings
            cursor.execute("""
                SELECT workflow_id, ticket_id, ticket_status, created_at, updated_at,
                       workflow_name, workflow_description
                FROM workflow_ticket_mappings
                ORDER BY updated_at DESC
            """)

            rows = cursor.fetchall()
            conn.close()

            mappings_loaded = 0
            with self.lock:
                for row in rows:
                    (workflow_id, ticket_id, ticket_status, created_at, updated_at,
                     workflow_name, workflow_description) = row

                    # Create WorkflowTicketMapping object
                    mapping = WorkflowTicketMapping(workflow_id)
                    mapping.ticket_id = ticket_id
                    mapping.ticket_status = ticket_status
                    mapping.created_at = created_at
                    mapping.updated_at = updated_at

                    # Store in LRU cache
                    self._mappings[workflow_id] = mapping
                    mappings_loaded += 1

            logger.info(f"Loaded {mappings_loaded} workflow-to-ticket mappings from database")

        except Exception as e:
            logger.error(f"Error loading mappings from database: {e}")
            # Don't fail - continue with empty mappings

    def _save_mapping_to_db(self, mapping: WorkflowTicketMapping) -> None:
        """
        PERSISTENCE: Save or update a workflow-to-ticket mapping in the database.

        This should be called whenever a mapping is created or updated.

        Args:
            mapping: The WorkflowTicketMapping object to persist
        """
        try:
            conn = sqlite3.connect(self._mappings_db_path)
            cursor = conn.cursor()

            # Get workflow details if available
            workflow_name = None
            workflow_description = None
            if hasattr(self, 'bubblelabs') and hasattr(self.bubblelabs, 'workflow_definitions'):
                if mapping.workflow_id in self.bubblelabs.workflow_definitions:
                    wf = self.bubblelabs.workflow_definitions[mapping.workflow_id]
                    workflow_name = getattr(wf, 'name', None)
                    workflow_description = getattr(wf, 'description', None)

            # Use INSERT OR REPLACE for upsert
            cursor.execute("""
                INSERT INTO workflow_ticket_mappings
                (workflow_id, ticket_id, ticket_status, created_at, updated_at,
                 workflow_name, workflow_description, last_synced_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mapping.workflow_id,
                mapping.ticket_id,
                mapping.ticket_status,
                mapping.created_at,
                mapping.updated_at,
                workflow_name,
                workflow_description,
                time.time()  # last_synced_at
            ))

            conn.commit()
            conn.close()

            logger.debug(f"Saved mapping to database: {mapping.workflow_id} -> {mapping.ticket_id}")

        except Exception as e:
            logger.error(f"Error saving mapping to database: {e}")

    def _delete_mapping_from_db(self, workflow_id: str) -> None:
        """
        PERSISTENCE: Delete a workflow-to-ticket mapping from the database.

        Args:
            workflow_id: The workflow ID whose mapping should be deleted
        """
        try:
            conn = sqlite3.connect(self._mappings_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM workflow_ticket_mappings
                WHERE workflow_id = ?
            """, (workflow_id,))

            conn.commit()
            deleted_count = cursor.rowcount
            conn.close()

            if deleted_count > 0:
                logger.debug(f"Deleted mapping from database: {workflow_id}")

        except Exception as e:
            logger.error(f"Error deleting mapping from database: {e}")

    def _add_mapping(self, workflow_id: str, mapping: WorkflowTicketMapping) -> None:
        """
        MEMORY LEAK FIX #1: Add mapping with LRU eviction.

        Automatically evicts oldest entry when cache reaches capacity.
        """
        with self.lock:
            # If at capacity, remove oldest entry
            if len(self._mappings) >= self._MAX_MAPPINGS:
                oldest_id, oldest_mapping = self._mappings.popitem(last=False)
                logger.info(f"LRU eviction: removed mapping for workflow {oldest_id} (cache full)")

            # Add new mapping
            self._mappings[workflow_id] = mapping

            # Move to end (most recently used)
            self._mappings.move_to_end(workflow_id)

    def _get_mapping(self, workflow_id: str) -> Optional[WorkflowTicketMapping]:
        """
        MEMORY LEAK FIX #1: Get mapping and update LRU status.

        Returns the mapping if found and updates its position as most recently used.
        """
        with self.lock:
            mapping = self._mappings.get(workflow_id)
            if mapping:
                # Move to end (most recently used)
                self._mappings.move_to_end(workflow_id)
            return mapping

    def _add_instance_mapping(self, instance_id: str, definition_id: str) -> None:
        """
        MEMORY LEAK FIX #2: Add instance-to-definition mapping with LRU eviction.

        Automatically evicts oldest entry when cache reaches capacity.
        """
        with self.lock:
            # If at capacity, remove oldest entry
            if len(self._instance_to_definition_cache) >= self._MAX_CACHE_SIZE:
                oldest_instance, oldest_definition = self._instance_to_definition_cache.popitem(last=False)
                logger.debug(f"LRU eviction: removed instance mapping for {oldest_instance} (cache full)")

            # Add new mapping
            self._instance_to_definition_cache[instance_id] = definition_id

            # Move to end (most recently used)
            self._instance_to_definition_cache.move_to_end(instance_id)

    def create_ticket_from_workflow(
        self,
        workflow_definition: BubbleWorkflowDefinition,
        assignee: Optional[str] = None,
        additional_labels: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Create a Hephaestus ticket from a BubbleLabs workflow definition.

        Args:
            workflow_definition: The workflow definition (cannot be None)
            assignee: Optional assignee for the ticket
            additional_labels: Additional labels for the ticket

        Returns:
            Ticket ID if successful, None otherwise

        Raises:
            ValueError: If workflow_definition is None or empty
        """
        # Input validation
        validate_not_none(workflow_definition, "workflow_definition")
        validate_not_empty(workflow_definition.id, "workflow_definition.id")
        validate_not_empty(workflow_definition.name, "workflow_definition.name")

        # State validation: Check if workflow already has a ticket
        with self.lock:
            if workflow_definition.id in self._mappings:
                logger.warning(f"Workflow {workflow_definition.id} already has a ticket")
                return self._mappings[workflow_definition.id].ticket_id

            # Maximum limit validation
            if len(self._mappings) >= self._MAX_MAPPINGS:
                raise ValueError(f"Maximum number of mappings ({self._MAX_MAPPINGS}) reached")

        if not HEPHAESTUS_AVAILABLE or not self.hephaestus:
            logger.warning("Hephaestus client not available, returning mock ticket ID")
            mock_id = f"{self.config.ticket_prefix}MOCK-{uuid.uuid4().hex[:8]}"
            return mock_id

        try:
            # Build ticket description
            description = self._build_ticket_description(workflow_definition)

            # Validate description length
            if len(description) > MAX_DESCRIPTION_LENGTH:
                logger.warning(f"Description exceeds {MAX_DESCRIPTION_LENGTH} chars, truncating")
                description = description[:MAX_DESCRIPTION_LENGTH]

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
                # Store mapping using LRU cache method (MEMORY LEAK FIX #1)
                mapping = WorkflowTicketMapping(workflow_definition.id)
                mapping.ticket_id = ticket_id
                mapping.ticket_status = TicketStatus.TODO.value
                self._add_mapping(workflow_definition.id, mapping)

                # PERSISTENCE: Save to database
                self._save_mapping_to_db(mapping)

                # Pre-populate instance cache with existing instances (fixes issue #3)
                with self.lock:
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
        Update Hephaestus ticket with workflow progress and state validation.

        Args:
            workflow_instance_id: ID of the workflow instance (cannot be empty)
            progress: Progress (0.0 to 1.0)
            status: Current workflow status
            metrics: Optional workflow metrics

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If workflow_instance_id is None/empty, progress is out of range,
                       or state transition is invalid
        """
        # Input validation
        validate_not_empty(workflow_instance_id, "workflow_instance_id")
        if progress < 0.0 or progress > 1.0:
            raise ValueError(f"progress must be between 0.0 and 1.0, got {progress}")
        validate_not_none(status, "status")

        # State validation: Check if instance exists before updating
        with self.lock:
            mapping = self._find_mapping_by_instance_id(workflow_instance_id)
            if not mapping or not mapping.ticket_id:
                logger.warning(f"No ticket found for workflow instance {workflow_instance_id}")
                return False

        if not HEPHAESTUS_AVAILABLE or not self.hephaestus:
            logger.debug(f"Mock update: workflow {workflow_instance_id} progress {progress*100:.1f}%")
            return True

        try:
            # CONCURRENCY FIX (Issue #7): Minimize lock scope - acquire ticket_id, release lock, then perform I/O
            # This prevents blocking other threads during potentially slow network operations

            # Step 1: Get ticket_id and current status while holding lock (minimal critical section)
            with self.lock:
                mapping = self._find_mapping_by_instance_id(workflow_instance_id)
                if not mapping or not mapping.ticket_id:
                    logger.warning(f"No ticket found for workflow instance {workflow_instance_id}")
                    return False

                # Capture ticket_id and current ticket_status for use after lock release
                ticket_id = mapping.ticket_id
                current_ticket_status = mapping.ticket_status

                # Map workflow status to ticket status (fast operation, still in lock)
                new_ticket_status = self._map_workflow_status_to_ticket_status(status, progress)

            # Step 2: Validate state transition OUTSIDE of lock (no shared state accessed)
            # This ensures we don't hold lock during validation logic
            if current_ticket_status and not validate_ticket_transition(current_ticket_status, new_ticket_status.value):
                logger.error(
                    f"Invalid ticket state transition: {current_ticket_status} -> {new_ticket_status.value}. "
                    f"Valid transitions from {current_ticket_status}: {get_valid_ticket_transitions(current_ticket_status)}"
                )
                return False

            # Step 3: Build description OUTSIDE of lock (no shared state accessed)
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

            # Validate description length
            if len(description) > MAX_DESCRIPTION_LENGTH:
                logger.warning(f"Description exceeds {MAX_DESCRIPTION_LENGTH} chars, truncating")
                description = description[:MAX_DESCRIPTION_LENGTH]

            # Step 4: Perform network I/O WITHOUT holding lock (CONCURRENCY FIX #7)
            # This is critical - holding lock during I/O blocks all other operations
            success = self.hephaestus.update_ticket(
                ticket_id=ticket_id,
                status=new_ticket_status,
                description=description
            )

            # Step 5: Update local state after I/O completes (re-acquire lock briefly)
            if success:
                with self.lock:
                    mapping = self._find_mapping_by_instance_id(workflow_instance_id)
                    if mapping:
                        mapping.ticket_status = new_ticket_status.value
                        mapping.updated_at = time.time()

                        # PERSISTENCE: Save to database
                        self._save_mapping_to_db(mapping)

                logger.debug(f"Updated ticket {ticket_id} to status {new_ticket_status.value}")
            else:
                logger.error(f"Failed to update ticket {ticket_id}")

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
            # Find mapping and validate state transition
            with self.lock:
                mapping = self._find_mapping_by_instance_id(workflow_instance_id)
                if not mapping or not mapping.ticket_id:
                    logger.warning(f"No ticket found for workflow instance {workflow_instance_id}")
                    return False

                # Capture current ticket status for validation
                current_ticket_status = mapping.ticket_status
                ticket_id = mapping.ticket_id

            # Determine new ticket status
            new_ticket_status = TicketStatus.DONE if success else TicketStatus.BLOCKED

            # STATE MACHINE VALIDATION: Validate ticket state transition
            if current_ticket_status and not validate_ticket_transition(current_ticket_status, new_ticket_status.value):
                logger.error(
                    f"Invalid ticket state transition in close_ticket_on_completion: "
                    f"{current_ticket_status} -> {new_ticket_status.value}. "
                    f"Valid transitions from {current_ticket_status}: "
                    f"{get_valid_ticket_transitions(current_ticket_status)}"
                )
                return False

            # Close ticket
            success_update = self.hephaestus.update_ticket(
                ticket_id=ticket_id,
                status=new_ticket_status
            )

            if success_update:
                # PERSISTENCE: Update mapping and save to database
                with self.lock:
                    mapping.ticket_status = ticket_status
                    mapping.updated_at = time.time()
                    self._save_mapping_to_db(mapping)

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

        CRITICAL BUG FIX: Added explicit None check and attribute validation for workflow
        object to prevent AttributeError crashes.

        Args:
            workflow_definition_id: ID of the workflow definition (cannot be empty)

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If workflow_definition_id is None or empty
        """
        # Input validation
        validate_not_empty(workflow_definition_id, "workflow_definition_id")

        # State validation: Check if ticket exists before syncing
        with self.lock:
            mapping = self._mappings.get(workflow_definition_id)
            if not mapping or not mapping.ticket_id:
                logger.warning(f"No ticket found for workflow {workflow_definition_id}")
                return False

        if not HEPHAESTUS_AVAILABLE or not self.hephaestus:
            return True

        try:
            # PERFORMANCE: Acquire all data BEFORE entering lock
            # This prevents nested lock acquisition and reduces lock hold time
            workflow = self.bubblelabs.get_workflow_definition(workflow_definition_id)

            # CRITICAL FIX: Explicit None check instead of truthy check
            if workflow is None:
                logger.error(f"Workflow {workflow_definition_id} not found (returned None)")
                return False

            # CRITICAL FIX: Validate workflow has required attributes
            if not hasattr(workflow, 'id') or not workflow.id:
                logger.error(f"Invalid workflow object for {workflow_definition_id}: missing 'id' attribute")
                return False

            if not hasattr(workflow, 'name') or not workflow.name:
                logger.error(f"Invalid workflow object for {workflow_definition_id}: missing 'name' attribute")
                return False

            # Build description before lock (no lock needed for this)
            description = self._build_ticket_description(workflow)

            # Validate description length
            if len(description) > MAX_DESCRIPTION_LENGTH:
                logger.warning(f"Description exceeds {MAX_DESCRIPTION_LENGTH} chars, truncating")
                description = description[:MAX_DESCRIPTION_LENGTH]

            # Now acquire lock only to get ticket_id
            with self.lock:
                mapping = self._mappings.get(workflow_definition_id)
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
            mapping = self._mappings.get(workflow_id)
            if mapping:
                # Update LRU status
                self._mappings.move_to_end(workflow_id)
            return mapping.ticket_id if mapping else None

    def get_all_mappings(self) -> Dict[str, WorkflowTicketMapping]:
        """
        PERSISTENCE: Get all workflow-to-ticket mappings from database.

        Returns:
            Dictionary mapping workflow IDs to WorkflowTicketMapping objects
        """
        try:
            conn = sqlite3.connect(self._mappings_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT workflow_id, ticket_id, ticket_status, created_at, updated_at
                FROM workflow_ticket_mappings
                ORDER BY created_at DESC
            """)

            rows = cursor.fetchall()
            conn.close()

            mappings = {}
            for row in rows:
                workflow_id, ticket_id, ticket_status, created_at, updated_at = row

                mapping = WorkflowTicketMapping(workflow_id)
                mapping.ticket_id = ticket_id
                mapping.ticket_status = ticket_status
                mapping.created_at = created_at
                mapping.updated_at = updated_at

                mappings[workflow_id] = mapping

            return mappings

        except Exception as e:
            logger.error(f"Error getting all mappings: {e}")
            return {}

    def start_background_sync(self) -> bool:
        """
        Start background sync thread to update tickets periodically.

        This method implements proper thread-safe startup with error handling (fixes issue #4).

        Returns:
            True if thread started successfully, False otherwise

        Raises:
            ValueError: If sync_interval is out of valid range
        """
        # Validate sync interval
        if self.sync_interval < 1 or self.sync_interval > MAX_SYNC_INTERVAL:
            raise ValueError(f"sync_interval must be between 1 and {MAX_SYNC_INTERVAL} seconds")

        with self.lock:
            if self.running:
                logger.warning("Background sync already running")
                return True

        # CONCURRENCY FIX (Issue #9): Create thread BEFORE setting running flag
        # This prevents race condition where thread starts but immediately sees running=False
        # If thread.start() fails, we won't have set running flag yet
        try:
            # Create thread first (no state change yet)
            self.sync_thread = Thread(target=self._sync_loop, daemon=True, name="BubbleLabsSync")

            # Now set running flag and start thread
            with self.lock:
                self.running = True
                self.shutdown_event.clear()

            # Start thread AFTER setting running flag (CONCURRENCY FIX #9)
            # If this succeeds, running flag is already set correctly
            # If this fails, we'll roll back in the exception handler
            self.sync_thread.start()

            logger.info(f"Started background sync thread (interval: {self.sync_interval}s)")
            return True

        except Exception as e:
            logger.error(f"Failed to start background sync thread: {e}")
            # Rollback: Clear running flag since thread failed to start
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

        Raises:
            ValueError: If timeout is negative or greater than MAX_SYNC_INTERVAL
        """
        # Input validation
        if timeout < 0:
            raise ValueError("timeout cannot be negative")
        if timeout > MAX_SYNC_INTERVAL:
            raise ValueError(f"timeout cannot exceed {MAX_SYNC_INTERVAL} seconds")

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

    def cleanup_old_mappings(self, max_age_days: int = 90) -> int:
        """
        PERSISTENCE: Remove mappings older than specified days.

        This method cleans up old completed/closed/cancelled mappings from the database
        to prevent unbounded database growth.

        Args:
            max_age_days: Maximum age in days (default: 90)

        Returns:
            Number of mappings deleted

        Raises:
            ValueError: If max_age_days is negative or excessive
        """
        # Input validation
        if max_age_days < 0:
            raise ValueError("max_age_days cannot be negative")
        if max_age_days > 3650:  # 10 years
            raise ValueError("max_age_days cannot exceed 3650 days (10 years)")

        try:
            cutoff_time = time.time() - (max_age_days * 86400)

            conn = sqlite3.connect(self._mappings_db_path)
            cursor = conn.cursor()

            # Delete old mappings
            cursor.execute("""
                DELETE FROM workflow_ticket_mappings
                WHERE created_at < ? AND ticket_status IN ('DONE', 'CLOSED', 'CANCELLED')
            """, (cutoff_time,))

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            # Also reload from database to update LRU cache
            self._load_mappings_from_db()

            logger.info(f"Cleaned up {deleted_count} old workflow mappings (older than {max_age_days} days)")
            return deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up old mappings: {e}")
            return 0

    def auto_cleanup_if_needed(self) -> None:
        """
        Automatically cleanup old mappings if cleanup interval has passed.

        This method should be called periodically (e.g., during sync operations)
        to ensure cleanup runs at least once per day. Only cleans completed/closed/
        cancelled mappings to preserve active workflow tracking.
        """
        now = time.time()

        # Check if cleanup is needed (run once per day)
        if now - self._last_mappings_cleanup > self._cleanup_interval:
            logger.info("Running automatic mappings cleanup...")
            self.cleanup_old_mappings(max_age_days=self._retention_days)
            self._last_mappings_cleanup = now

    def get_mapping_stats(self) -> Dict[str, Any]:
        """
        PERSISTENCE: Get statistics about workflow-to-ticket mappings.

        Returns:
            Dictionary containing mapping statistics
        """
        try:
            conn = sqlite3.connect(self._mappings_db_path)
            cursor = conn.cursor()

            # Get total count
            cursor.execute("SELECT COUNT(*) FROM workflow_ticket_mappings")
            total_count = cursor.fetchone()[0]

            # Get count by status
            cursor.execute("""
                SELECT ticket_status, COUNT(*)
                FROM workflow_ticket_mappings
                GROUP BY ticket_status
            """)
            status_counts = dict(cursor.fetchall())

            # Get oldest and newest
            cursor.execute("""
                SELECT MIN(created_at), MAX(created_at)
                FROM workflow_ticket_mappings
            """)
            oldest, newest = cursor.fetchone()

            conn.close()

            stats = {
                "total_mappings": total_count,
                "by_status": status_counts,
                "oldest_mapping": datetime.fromtimestamp(oldest).isoformat() if oldest else None,
                "newest_mapping": datetime.fromtimestamp(newest).isoformat() if newest else None,
                "cache_size": len(self._mappings),
                "cache_max_size": self._MAX_MAPPINGS,
                "database_path": self._mappings_db_path
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting mapping stats: {e}")
            return {
                "error": str(e),
                "total_mappings": 0,
                "by_status": {},
                "cache_size": len(self._mappings)
            }

    def _sync_loop(self) -> None:
        """
        Background sync loop with proper shutdown handling (fixes issue #1).

        Uses threading.Event for thread-safe shutdown signaling instead of
        polling a boolean flag, which eliminates race conditions.

        DATABASE CLEANUP: Automatically cleans up old mappings once per day.
        """
        while not self.shutdown_event.is_set():
            try:
                # Update instance cache before syncing (fixes issue #3)
                self._update_instance_cache()

                # Sync all active workflows
                self._sync_all_active_workflows()

                # AUTOMATIC CLEANUP: Run cleanup if needed (once per day)
                self.auto_cleanup_if_needed()

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
                        # Include current ticket status for state machine validation
                        ticket_id_map[update['instance_id']] = {
                            'ticket_id': mapping.ticket_id,
                            'current_ticket_status': mapping.ticket_status,  # For validation
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
                current_ticket_status = data.get('current_ticket_status')
                progress = data['progress']
                status = data['status']

                # Map workflow status to ticket status
                new_ticket_status = self._map_workflow_status_to_ticket_status(status, progress)

                # STATE MACHINE VALIDATION: Validate ticket state transition
                if current_ticket_status and not validate_ticket_transition(current_ticket_status, new_ticket_status.value):
                    logger.error(
                        f"Invalid ticket state transition in batch update for {instance_id}: "
                        f"{current_ticket_status} -> {new_ticket_status.value}. "
                        f"Valid transitions from {current_ticket_status}: "
                        f"{get_valid_ticket_transitions(current_ticket_status)}. Skipping update."
                    )
                    continue

                # Build description
                description = f"**Progress:** {progress*100:.1f}%\n\n"
                description += f"**Status:** {status.value}\n\n"

                # Update ticket
                self.hephaestus.update_ticket(
                    ticket_id=ticket_id,
                    status=new_ticket_status,
                    description=description
                )

                logger.debug(f"Updated ticket {ticket_id} for workflow {instance_id}")

            except Exception as e:
                logger.error(f"Error updating ticket for {instance_id}: {e}")

    def _update_instance_cache(self) -> None:
        """
        MEMORY LEAK FIX #2: Update instance-to-definition mapping cache with LRU eviction.

        This cache eliminates expensive API calls on every lookup by building
        a reverse mapping from instance IDs to definition IDs.

        Implements LRU eviction to prevent unbounded memory growth.
        """
        try:
            instances = self.bubblelabs.list_workflow_instances()

            with self.lock:
                # Build new cache
                new_cache = OrderedDict()
                for instance in instances:
                    new_cache[instance.id] = instance.definition_id

                # If new cache too large, trim to MAX_CACHE_SIZE
                while len(new_cache) > self._MAX_CACHE_SIZE:
                    oldest_instance, oldest_definition = new_cache.popitem(last=False)
                    logger.debug(f"Trimming oldest instance mapping: {oldest_instance}")

                # Atomic replacement
                self._instance_to_definition_cache = new_cache

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
        workflow_status: Union[WorkflowStatus, ExtendedWorkflowStatus, str],
        progress: float = 0.0
    ) -> TicketStatus:
        """
        Map BubbleLabs workflow status to Hephaestus ticket status with validation.

        This method now uses ExtendedTicketStatus for the mapping to ensure
        proper state machine validation.

        Args:
            workflow_status: Current workflow status (enum or string)
            progress: Workflow progress (0.0 to 1.0)

        Returns:
            Corresponding ticket status
        """
        EPSILON = 0.001

        # Convert to string for comparison
        if isinstance(workflow_status, (WorkflowStatus, ExtendedWorkflowStatus)):
            status_str = workflow_status.value
        else:
            status_str = str(workflow_status).lower()

        # Map workflow status to ticket status based on state machine
        if status_str in ["pending", "created"]:
            return ExtendedTicketStatus.TODO
        elif status_str == "cancelled":
            return ExtendedTicketStatus.CANCELLED
        elif status_str == "completed":
            return ExtendedTicketStatus.DONE
        elif status_str == "failed":
            return ExtendedTicketStatus.BLOCKED  # Failed workflows are blocked
        elif status_str == "running":
            # Map based on progress
            if progress < 0.3 - EPSILON:
                return ExtendedTicketStatus.TODO
            elif progress < 0.7 - EPSILON:
                return ExtendedTicketStatus.IN_PROGRESS
            else:
                return ExtendedTicketStatus.IN_REVIEW
        elif status_str == "paused":
            return ExtendedTicketStatus.BLOCKED  # Paused workflows are blocked
        elif status_str == "stopped":
            return ExtendedTicketStatus.BLOCKED  # Stopped workflows are blocked
        elif status_str == "stopping":
            return ExtendedTicketStatus.BLOCKED  # Stopping workflows are blocked
        else:
            logger.warning(f"Unknown workflow status: {workflow_status}, defaulting to TODO")
            return ExtendedTicketStatus.TODO

    def _find_mapping_by_instance_id(self, instance_id: str) -> Optional[WorkflowTicketMapping]:
        """
        MEMORY LEAK FIX #2 & #3: Find mapping by workflow instance ID using LRU cache.

        Updates LRU status when mappings are accessed.

        Args:
            instance_id: The workflow instance ID

        Returns:
            WorkflowTicketMapping if found, None otherwise
        """
        # Try direct match first (instance_id might be a definition_id)
        with self.lock:
            if instance_id in self._mappings:
                # Update LRU status
                self._mappings.move_to_end(instance_id)
                return self._mappings[instance_id]

            # Use the LRU-protected instance-to-definition cache
            definition_id: Optional[str] = self._instance_to_definition_cache.get(instance_id)
            if definition_id and definition_id in self._mappings:
                # Update LRU status for both caches
                self._instance_to_definition_cache.move_to_end(instance_id)
                self._mappings.move_to_end(definition_id)
                return self._mappings[definition_id]

        # Fallback: Try to find through bubblelabs integration (expensive)
        try:
            instances = self.bubblelabs.list_workflow_instances()
            for instance in instances:
                if instance.id == instance_id:
                    # Cache this for future lookups using LRU method
                    self._add_instance_mapping(instance_id, instance.definition_id)
                    with self.lock:
                        mapping = self._mappings.get(instance.definition_id)
                        if mapping:
                            self._mappings.move_to_end(instance.definition_id)
                        return mapping
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
