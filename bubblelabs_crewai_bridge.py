"""
BubbleLabs-CrewAI Bridge

This module provides integration between BubbleLabs workflows and CrewAI execution,
replacing the AGPL-licensed Hephaestus integration with MIT-licensed CrewAI.

This replaces bubblelabs_hephaestus_bridge.py with local CrewAI execution.

IMPORTANT: BubbleLabs workflows are now tracked as CrewAI workflows instead of
Hephaestus tickets. The bridge maintains the same API for compatibility.

License: MIT (replaces AGPL Hephaestus)
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
from io import StringIO
from collections import OrderedDict
from functools import wraps

# Import CrewAI zero-error workflow (replaces Hephaestus)
from crewai_zero_error_workflow import (
    CrewAIZeroErrorWorkflow,
    ZeroErrorConfig,
    create_zero_error_workflow,
    create_zero_error_config,
)

# Import state management
from crewai_state_management import (
    WorkflowState,
    SubProblem,
    DecompositionPlan,
    StateManager,
)

# BubbleLabs integration (maintained)
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


class ExtendedCrewAIStatus(Enum):
    """
    CrewAI workflow status states with state machine validation.

    States:
    - TODO: Workflow created but not yet started
    - IN_PROGRESS: Work actively being done
    - IN_REVIEW: Work completed, under review
    - DONE: Work completed and approved (terminal state)
    - CANCELLED: Workflow cancelled (terminal state)
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


# Valid state transitions for CrewAI workflows
VALID_CREWAI_TRANSITIONS: Dict[ExtendedCrewAIStatus, Set[ExtendedCrewAIStatus]] = {
    ExtendedCrewAIStatus.TODO: {
        ExtendedCrewAIStatus.IN_PROGRESS,
        ExtendedCrewAIStatus.CANCELLED,
        ExtendedCrewAIStatus.BLOCKED
    },
    ExtendedCrewAIStatus.IN_PROGRESS: {
        ExtendedCrewAIStatus.IN_REVIEW,
        ExtendedCrewAIStatus.TODO,
        ExtendedCrewAIStatus.CANCELLED,
        ExtendedCrewAIStatus.BLOCKED
    },
    ExtendedCrewAIStatus.IN_REVIEW: {
        ExtendedCrewAIStatus.IN_PROGRESS,
        ExtendedCrewAIStatus.DONE,
        ExtendedCrewAIStatus.TODO,
        ExtendedCrewAIStatus.CANCELLED,
        ExtendedCrewAIStatus.BLOCKED
    },
    ExtendedCrewAIStatus.DONE: set(),  # Terminal state
    ExtendedCrewAIStatus.CANCELLED: set(),  # Terminal state
    ExtendedCrewAIStatus.BLOCKED: {
        ExtendedCrewAIStatus.TODO,
        ExtendedCrewAIStatus.IN_PROGRESS,
        ExtendedCrewAIStatus.CANCELLED
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
    """
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

    if current_status == new_status:
        return True

    if current_status not in VALID_WORKFLOW_TRANSITIONS:
        logger.error(f"Unknown current status in transition table: {current_status}")
        return False

    if new_status not in VALID_WORKFLOW_TRANSITIONS[current_status]:
        logger.error(f"Invalid workflow transition: {current_status.value} -> {new_status.value}")
        return False

    return True


def validate_crewai_transition(
    current_status: Union[ExtendedCrewAIStatus, str],
    new_status: Union[ExtendedCrewAIStatus, str]
) -> bool:
    """
    Validate if a CrewAI workflow state transition is allowed.

    Args:
        current_status: Current CrewAI status (enum or string)
        new_status: Desired new CrewAI status (enum or string)

    Returns:
        True if transition is valid, False otherwise
    """
    if isinstance(current_status, str):
        try:
            current_status = ExtendedCrewAIStatus(current_status.upper())
        except ValueError:
            logger.error(f"Unknown current CrewAI status: {current_status}")
            return False

    if isinstance(new_status, str):
        try:
            new_status = ExtendedCrewAIStatus(new_status.upper())
        except ValueError:
            logger.error(f"Unknown new CrewAI status: {new_status}")
            return False

    if current_status == new_status:
        return True

    if current_status not in VALID_CREWAI_TRANSITIONS:
        logger.error(f"Unknown current status in transition table: {current_status}")
        return False

    if new_status not in VALID_CREWAI_TRANSITIONS[current_status]:
        logger.error(f"Invalid CrewAI transition: {current_status.value} -> {new_status.value}")
        return False

    return True


def get_valid_workflow_transitions(status: Union[ExtendedWorkflowStatus, str]) -> Set[str]:
    """Get all valid next states for a given workflow status."""
    if isinstance(status, str):
        try:
            status = ExtendedWorkflowStatus(status.lower())
        except ValueError:
            logger.error(f"Unknown workflow status: {status}")
            return set()

    transitions = VALID_WORKFLOW_TRANSITIONS.get(status, set())
    return {s.value for s in transitions}


def get_valid_crewai_transitions(status: Union[ExtendedCrewAIStatus, str]) -> Set[str]:
    """Get all valid next states for a given CrewAI status."""
    if isinstance(status, str):
        try:
            status = ExtendedCrewAIStatus(status.upper())
        except ValueError:
            logger.error(f"Unknown CrewAI status: {status}")
            return set()

    transitions = VALID_CREWAI_TRANSITIONS.get(status, set())
    return {s.value for s in transitions}


def is_terminal_workflow_status(status: Union[ExtendedWorkflowStatus, str]) -> bool:
    """
    Check if a workflow status is terminal.

    Terminal states indicate no further transitions are allowed.
    """
    if isinstance(status, str):
        try:
            status = ExtendedWorkflowStatus(status.lower())
        except ValueError:
            logger.error(f"Unknown workflow status: {status}")
            return False

    return status in {
        ExtendedWorkflowStatus.COMPLETED,
        ExtendedWorkflowStatus.FAILED,
        ExtendedWorkflowStatus.CANCELLED,
    }


class WorkflowCrewAIMapping:
    """
    Maps workflow instances to CrewAI workflows.

    Attributes:
        workflow_id: ID of the workflow definition
        crewai_workflow_id: ID of the associated CrewAI workflow
        crewai_status: Current status of the CrewAI workflow
        created_at: Timestamp when mapping was created
        updated_at: Timestamp when mapping was last updated
    """

    def __init__(self, workflow_id: str) -> None:
        """
        Initialize a workflow CrewAI mapping.

        Args:
            workflow_id: ID of the workflow definition
        """
        self.workflow_id: str = workflow_id
        self.crewai_workflow_id: Optional[str] = None
        self.crewai_status: Optional[str] = None
        self.created_at: float = time.time()
        self.updated_at: float = time.time()


@dataclass
class BubbleLabsCrewAIConfig:
    """Configuration for creating CrewAI workflows from BubbleLabs workflows."""
    auto_create_workflows: bool = True
    auto_update_progress: bool = True
    auto_close_on_completion: bool = True
    workflow_prefix: str = "BL-"
    enable_zero_error: bool = True
    default_labels: List[str] = None

    def __post_init__(self):
        if self.default_labels is None:
            self.default_labels = ["bubblelabs", "workflow", "crewai"]


class BubbleLabsCrewAIBridge:
    """
    Bridge between BubbleLabs workflows and CrewAI execution.

    This bridge:
    - Creates CrewAI workflows when BubbleLabs workflows are created
    - Updates workflow status as workflows progress
    - Closes workflows when workflows complete
    - Syncs workflow metadata to workflow descriptions

    Replaces BubbleLabsHephaestusBridge with local CrewAI execution.
    """

    def __init__(
        self,
        bubblelabs_integration: Optional[BubbleLabsIntegration] = None,
        config: Optional[BubbleLabsCrewAIConfig] = None,
        batch_size: int = 10,
        mappings_db_path: Optional[str] = None,
        state_storage_dir: Optional[str] = None
    ) -> None:
        """
        Initialize the BubbleLabs-CrewAI bridge.

        Args:
            bubblelabs_integration: BubbleLabs integration instance
            config: Workflow configuration
            batch_size: Number of operations to batch together
            mappings_db_path: Optional path for mappings database
            state_storage_dir: Directory for CrewAI state storage

        Raises:
            ValueError: If batch_size is out of valid range
        """
        if batch_size is not None:
            validate_range(batch_size, 1, MAX_BATCH_SIZE, "batch_size")

        self.bubblelabs: BubbleLabsIntegration = bubblelabs_integration or BubbleLabsIntegration()
        self.config: BubbleLabsCrewAIConfig = config or BubbleLabsCrewAIConfig()
        self.batch_size: int = batch_size

        # LRU cache for workflow-to-CrewAI mappings
        self._mappings: OrderedDict = OrderedDict()
        self._MAX_MAPPINGS = 1000
        self.lock: Lock = Lock()

        # LRU cache for instance-to-definition mapping
        self._instance_to_definition_cache: OrderedDict = OrderedDict()
        self._MAX_CACHE_SIZE = 1000

        # Database path for mappings
        self._mappings_db_path = mappings_db_path or "crewai_workflow_mappings.db"

        # State storage directory
        self._state_storage_dir = state_storage_dir or "./crewai_states"
        self.state_manager: Optional[StateManager] = None

        # Background sync thread
        self.sync_thread: Optional[Thread] = None
        self.sync_interval: int = 30  # seconds
        self.running: bool = False
        self.shutdown_event: Event = Event()

        # Database cleanup configuration
        self._retention_days = 90
        self._cleanup_interval = 86400
        self._last_mappings_cleanup = time.time()

        # Initialize state manager
        self._init_state_manager()

        # Initialize database
        self._init_mappings_database()

        # Load existing mappings
        self._load_mappings_from_db()

        logger.info("BubbleLabs-CrewAI Bridge initialized (MIT-licensed)")

    def _init_state_manager(self) -> None:
        """Initialize CrewAI state manager for local execution."""
        try:
            self.state_manager = StateManager(self._state_storage_dir)
            logger.info(f"CrewAI StateManager initialized: {self._state_storage_dir}")
        except Exception as e:
            logger.error(f"Error initializing StateManager: {e}")
            # Continue without state manager

    def _init_mappings_database(self) -> None:
        """Initialize SQLite database for workflow-to-CrewAI mappings."""
        try:
            with sqlite3.connect(self._mappings_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA foreign_keys = ON")

                # Create table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS workflow_crewai_mappings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        workflow_id TEXT NOT NULL,
                        crewai_workflow_id TEXT NOT NULL,
                        crewai_status TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,

                        workflow_name TEXT,
                        workflow_description TEXT,

                        last_synced_at REAL,

                        UNIQUE(workflow_id)
                    )
                """)

                # Create indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_crewai_mappings_status
                    ON workflow_crewai_mappings(crewai_status)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_crewai_mappings_updated_at
                    ON workflow_crewai_mappings(updated_at)
                """)

                conn.commit()

            logger.info(f"Initialized CrewAI workflow mappings database: {self._mappings_db_path}")

        except Exception as e:
            logger.error(f"Error initializing mappings database: {e}")
            raise

    def _load_mappings_from_db(self) -> None:
        """Load all workflow-to-CrewAI mappings from database into LRU cache."""
        try:
            with sqlite3.connect(self._mappings_db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT workflow_id, crewai_workflow_id, crewai_status, created_at, updated_at,
                           workflow_name, workflow_description
                    FROM workflow_crewai_mappings
                    ORDER BY updated_at DESC
                """)

                rows = cursor.fetchall()

            mappings_loaded = 0
            with self.lock:
                for row in rows:
                    (workflow_id, crewai_workflow_id, crewai_status, created_at, updated_at,
                     workflow_name, workflow_description) = row

                    mapping = WorkflowCrewAIMapping(workflow_id)
                    mapping.crewai_workflow_id = crewai_workflow_id
                    mapping.crewai_status = crewai_status
                    mapping.created_at = created_at
                    mapping.updated_at = updated_at

                    self._mappings[workflow_id] = mapping
                    mappings_loaded += 1

            logger.info(f"Loaded {mappings_loaded} workflow-to-CrewAI mappings from database")

        except Exception as e:
            logger.error(f"Error loading mappings from database: {e}")

    def _save_mapping_to_db(self, mapping: WorkflowCrewAIMapping) -> None:
        """Save or update a workflow-to-CrewAI mapping in the database."""
        try:
            with sqlite3.connect(self._mappings_db_path) as conn:
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
                    INSERT INTO workflow_crewai_mappings
                    (workflow_id, crewai_workflow_id, crewai_status, created_at, updated_at,
                     workflow_name, workflow_description, last_synced_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    mapping.workflow_id,
                    mapping.crewai_workflow_id,
                    mapping.crewai_status,
                    mapping.created_at,
                    mapping.updated_at,
                    workflow_name,
                    workflow_description,
                    time.time()
                ))

                conn.commit()

            logger.debug(f"Saved mapping to database: {mapping.workflow_id} -> {mapping.crewai_workflow_id}")

        except Exception as e:
            logger.error(f"Error saving mapping to database: {e}")

    def _add_mapping(self, workflow_id: str, mapping: WorkflowCrewAIMapping) -> None:
        """Add mapping with LRU eviction."""
        with self.lock:
            if len(self._mappings) >= self._MAX_MAPPINGS:
                oldest_id, oldest_mapping = self._mappings.popitem(last=False)
                logger.info(f"LRU eviction: removed mapping for workflow {oldest_id}")

            self._mappings[workflow_id] = mapping
            self._mappings.move_to_end(workflow_id)

    def create_workflow_from_bubblelabs(
        self,
        workflow_definition: BubbleWorkflowDefinition,
        assignee: Optional[str] = None,
        additional_labels: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Create a CrewAI workflow from a BubbleLabs workflow definition.

        Args:
            workflow_definition: The workflow definition
            assignee: Optional assignee (not used in CrewAI, kept for API compatibility)
            additional_labels: Additional labels (not used in CrewAI, kept for API compatibility)

        Returns:
            CrewAI workflow ID if successful, None otherwise
        """
        validate_not_none(workflow_definition, "workflow_definition")
        validate_not_empty(workflow_definition.id, "workflow_definition.id")
        validate_not_empty(workflow_definition.name, "workflow_definition.name")

        # Check if workflow already has a CrewAI workflow
        with self.lock:
            if workflow_definition.id in self._mappings:
                logger.warning(f"Workflow {workflow_definition.id} already has a CrewAI workflow")
                return self._mappings[workflow_definition.id].crewai_workflow_id

            if len(self._mappings) >= self._MAX_MAPPINGS:
                raise ValueError(f"Maximum number of mappings ({self._MAX_MAPPINGS}) reached")

        try:
            # Create CrewAI workflow ID
            crewai_workflow_id = f"{self.config.workflow_prefix}{workflow_definition.id}"

            # Create CrewAI zero-error config
            config = create_zero_error_config(
                enable_red_flagging=self.config.enable_zero_error,
                enable_first_to_ahead=self.config.enable_zero_error,
            )

            # Create CrewAI workflow
            workflow = create_zero_error_workflow(
                config=config,
                workflow_id=crewai_workflow_id,
            )

            # Create initial workflow state
            if self.state_manager:
                workflow_state = WorkflowState(
                    workflow_id=crewai_workflow_id,
                    problem_statement=workflow_definition.description or workflow_definition.name,
                    execution_method="roma_mdap_maker" if self.config.enable_zero_error else "traditional",
                )

                # Save initial state
                self.state_manager.save_state(crewai_workflow_id, workflow_state)

            # Store mapping
            mapping = WorkflowCrewAIMapping(workflow_definition.id)
            mapping.crewai_workflow_id = crewai_workflow_id
            mapping.crewai_status = ExtendedCrewAIStatus.TODO.value
            self._add_mapping(workflow_definition.id, mapping)

            # Persist to database
            self._save_mapping_to_db(mapping)

            # Update instance cache
            with self.lock:
                self._update_instance_cache()

            logger.info(f"Created CrewAI workflow {crewai_workflow_id} for BubbleLabs workflow {workflow_definition.id}")
            return crewai_workflow_id

        except Exception as e:
            logger.error(f"Error creating CrewAI workflow from BubbleLabs: {e}")
            return None

    def update_workflow_progress(
        self,
        workflow_instance_id: str,
        progress: float,
        status: WorkflowStatus,
        metrics: Optional[WorkflowMetrics] = None
    ) -> bool:
        """
        Update CrewAI workflow with BubbleLabs workflow progress.

        Args:
            workflow_instance_id: ID of the workflow instance
            progress: Progress (0.0 to 1.0)
            status: Current workflow status
            metrics: Optional workflow metrics

        Returns:
            True if successful, False otherwise
        """
        validate_not_empty(workflow_instance_id, "workflow_instance_id")
        if progress < 0.0 or progress > 1.0:
            raise ValueError(f"progress must be between 0.0 and 1.0, got {progress}")
        validate_not_none(status, "status")

        # Find mapping
        with self.lock:
            mapping = self._find_mapping_by_instance_id(workflow_instance_id)
            if not mapping or not mapping.crewai_workflow_id:
                logger.warning(f"No CrewAI workflow found for BubbleLabs instance {workflow_instance_id}")
                return False

        try:
            # Get current state
            if self.state_manager:
                workflow_state = self.state_manager.load_state(mapping.crewai_workflow_id)

                if workflow_state:
                    # Update progress and status
                    workflow_state.status = status.value
                    if metrics:
                        # Update metrics in state
                        if hasattr(workflow_state, 'metrics'):
                            workflow_state.metrics.update({
                                'execution_time': metrics.execution_time,
                                'tokens_used': metrics.tokens_used,
                                'best_fitness': metrics.best_fitness,
                                'iterations_completed': metrics.iterations_completed,
                            })

                    # Save updated state
                    self.state_manager.save_state(mapping.crewai_workflow_id, workflow_state)

            # Update mapping status
            with self.lock:
                new_crewai_status = self._map_workflow_status_to_crewai_status(status, progress)
                mapping = self._find_mapping_by_instance_id(workflow_instance_id)
                if mapping:
                    mapping.crewai_status = new_crewai_status.value
                    mapping.updated_at = time.time()
                    self._save_mapping_to_db(mapping)

            logger.debug(f"Updated CrewAI workflow {mapping.crewai_workflow_id} to status {new_crewai_status.value}")
            return True

        except Exception as e:
            logger.error(f"Error updating CrewAI workflow progress: {e}")
            return False

    def close_workflow_on_completion(self, workflow_instance_id: str, success: bool = True) -> bool:
        """
        Close CrewAI workflow when BubbleLabs workflow completes.

        Args:
            workflow_instance_id: ID of the workflow instance
            success: Whether the workflow completed successfully

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.lock:
                mapping = self._find_mapping_by_instance_id(workflow_instance_id)
                if not mapping or not mapping.crewai_workflow_id:
                    logger.warning(f"No CrewAI workflow found for instance {workflow_instance_id}")
                    return False

                current_crewai_status = mapping.crewai_status
                crewai_workflow_id = mapping.crewai_workflow_id

            # Determine new status
            new_crewai_status = ExtendedCrewAIStatus.DONE if success else ExtendedCrewAIStatus.BLOCKED

            # Validate state transition
            if current_crewai_status and not validate_crewai_transition(current_crewai_status, new_crewai_status.value):
                logger.error(
                    f"Invalid CrewAI state transition: {current_crewai_status} -> {new_crewai_status.value}"
                )
                return False

            # Update state
            if self.state_manager:
                workflow_state = self.state_manager.load_state(crewai_workflow_id)
                if workflow_state:
                    workflow_state.status = "completed" if success else "failed"
                    self.state_manager.save_state(crewai_workflow_id, workflow_state)

            # Update mapping
            with self.lock:
                mapping.crewai_status = new_crewai_status.value
                mapping.updated_at = time.time()
                self._save_mapping_to_db(mapping)

            logger.info(f"Closed CrewAI workflow {mapping.crewai_workflow_id} for BubbleLabs instance {workflow_instance_id}")
            return True

        except Exception as e:
            logger.error(f"Error closing CrewAI workflow: {e}")
            return False

    def sync_workflow_to_crewai(self, workflow_definition_id: str) -> bool:
        """
        Sync BubbleLabs workflow definition to existing CrewAI workflow.

        Args:
            workflow_definition_id: ID of the workflow definition

        Returns:
            True if successful, False otherwise
        """
        validate_not_empty(workflow_definition_id, "workflow_definition_id")

        with self.lock:
            mapping = self._mappings.get(workflow_definition_id)
            if not mapping or not mapping.crewai_workflow_id:
                logger.warning(f"No CrewAI workflow found for BubbleLabs workflow {workflow_definition_id}")
                return False

        try:
            workflow = self.bubblelabs.get_workflow_definition(workflow_definition_id)

            if workflow is None:
                logger.error(f"BubbleLabs workflow {workflow_definition_id} not found")
                return False

            # Update CrewAI workflow state if needed
            if self.state_manager:
                workflow_state = self.state_manager.load_state(mapping.crewai_workflow_id)

                if workflow_state:
                    # Update problem statement
                    workflow_state.problem_statement = workflow.description or workflow.name
                    self.state_manager.save_state(mapping.crewai_workflow_id, workflow_state)

            return True

        except Exception as e:
            logger.error(f"Error syncing BubbleLabs workflow to CrewAI: {e}")
            return False

    def get_crewai_workflow_for_bubblelabs(self, workflow_id: str) -> Optional[str]:
        """
        Get the CrewAI workflow ID for a BubbleLabs workflow.

        Args:
            workflow_id: ID of the BubbleLabs workflow

        Returns:
            CrewAI workflow ID or None
        """
        with self.lock:
            mapping = self._mappings.get(workflow_id)
            if mapping:
                self._mappings.move_to_end(workflow_id)
            return mapping.crewai_workflow_id if mapping else None

    def get_all_mappings(self) -> Dict[str, WorkflowCrewAIMapping]:
        """Get all BubbleLabs-to-CrewAI mappings from database."""
        try:
            with sqlite3.connect(self._mappings_db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT workflow_id, crewai_workflow_id, crewai_status, created_at, updated_at
                    FROM workflow_crewai_mappings
                    ORDER BY created_at DESC
                """)

                rows = cursor.fetchall()

            mappings = {}
            for row in rows:
                workflow_id, crewai_workflow_id, crewai_status, created_at, updated_at = row

                mapping = WorkflowCrewAIMapping(workflow_id)
                mapping.crewai_workflow_id = crewai_workflow_id
                mapping.crewai_status = crewai_status
                mapping.created_at = created_at
                mapping.updated_at = updated_at

                mappings[workflow_id] = mapping

            return mappings

        except Exception as e:
            logger.error(f"Error getting all mappings: {e}")
            return {}

    def start_background_sync(self) -> bool:
        """Start background sync thread to update workflows periodically."""
        if self.sync_interval < 1 or self.sync_interval > MAX_SYNC_INTERVAL:
            raise ValueError(f"sync_interval must be between 1 and {MAX_SYNC_INTERVAL} seconds")

        with self.lock:
            if self.running:
                logger.warning("Background sync already running")
                return True

        try:
            self.sync_thread = Thread(target=self._sync_loop, daemon=True, name="BubbleLabsCrewAISync")

            with self.lock:
                self.running = True
                self.shutdown_event.clear()

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
        """Stop background sync thread."""
        if timeout < 0:
            raise ValueError("timeout cannot be negative")
        if timeout > MAX_SYNC_INTERVAL:
            raise ValueError(f"timeout cannot exceed {MAX_SYNC_INTERVAL} seconds")

        with self.lock:
            if not self.running:
                logger.warning("Background sync not running")
                return True

            self.running = False
            self.shutdown_event.set()

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
        """Background sync loop."""
        while not self.shutdown_event.is_set():
            try:
                self._update_instance_cache()
                self._sync_all_active_workflows()

                if self.shutdown_event.wait(timeout=self.sync_interval):
                    break

            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                if self.shutdown_event.wait(timeout=5.0):
                    break

    def _sync_all_active_workflows(self) -> None:
        """Sync all active BubbleLabs workflows to their CrewAI workflows."""
        try:
            instances = self.bubblelabs.list_workflow_instances()

            updates_to_make = []

            for instance in instances:
                if instance.status == "running":
                    progress = getattr(instance, 'progress', 0.0)

                    try:
                        status = WorkflowStatus(instance.status.upper())
                    except ValueError:
                        status = WorkflowStatus.RUNNING

                    updates_to_make.append({
                        'instance_id': instance.id,
                        'progress': progress,
                        'status': status
                    })

            with self.lock:
                for update in updates_to_make:
                    mapping = self._find_mapping_by_instance_id(update['instance_id'])
                    if mapping and mapping.crewai_workflow_id:
                        # Update status
                        new_status = self._map_workflow_status_to_crewai_status(
                            update['status'],
                            update['progress']
                        )
                        mapping.crewai_status = new_status.value
                        mapping.updated_at = time.time()
                        self._save_mapping_to_db(mapping)

        except Exception as e:
            logger.error(f"Error syncing active workflows: {e}")

    def _update_instance_cache(self) -> None:
        """Update instance-to-definition mapping cache with LRU eviction."""
        try:
            instances = self.bubblelabs.list_workflow_instances()

            with self.lock:
                new_cache = OrderedDict()
                for instance in instances:
                    new_cache[instance.id] = instance.definition_id

                while len(new_cache) > self._MAX_CACHE_SIZE:
                    oldest_instance, oldest_definition = new_cache.popitem(last=False)
                    logger.debug(f"Trimming oldest instance mapping: {oldest_instance}")

                self._instance_to_definition_cache = new_cache

                logger.debug(f"Updated instance cache with {len(new_cache)} entries")

        except Exception as e:
            logger.warning(f"Error updating instance cache: {e}")

    def _map_workflow_status_to_crewai_status(
        self,
        workflow_status: Union[WorkflowStatus, ExtendedWorkflowStatus, str],
        progress: float = 0.0
    ) -> ExtendedCrewAIStatus:
        """
        Map BubbleLabs workflow status to CrewAI workflow status.

        Args:
            workflow_status: Current workflow status (enum or string)
            progress: Workflow progress (0.0 to 1.0)

        Returns:
            Corresponding CrewAI status
        """
        EPSILON = 0.001

        if isinstance(workflow_status, (WorkflowStatus, ExtendedWorkflowStatus)):
            status_str = workflow_status.value
        else:
            status_str = str(workflow_status).lower()

        if status_str in ["pending", "created"]:
            return ExtendedCrewAIStatus.TODO
        elif status_str == "cancelled":
            return ExtendedCrewAIStatus.CANCELLED
        elif status_str == "completed":
            return ExtendedCrewAIStatus.DONE
        elif status_str == "failed":
            return ExtendedCrewAIStatus.BLOCKED
        elif status_str == "running":
            if progress < 0.3 - EPSILON:
                return ExtendedCrewAIStatus.TODO
            elif progress < 0.7 - EPSILON:
                return ExtendedCrewAIStatus.IN_PROGRESS
            else:
                return ExtendedCrewAIStatus.IN_REVIEW
        elif status_str in ["paused", "stopped", "stopping"]:
            return ExtendedCrewAIStatus.BLOCKED
        else:
            logger.warning(f"Unknown workflow status: {workflow_status}, defaulting to TODO")
            return ExtendedCrewAIStatus.TODO

    def _find_mapping_by_instance_id(self, instance_id: str) -> Optional[WorkflowCrewAIMapping]:
        """Find mapping by workflow instance ID using LRU cache."""
        with self.lock:
            if instance_id in self._mappings:
                self._mappings.move_to_end(instance_id)
                return self._mappings[instance_id]

            definition_id = self._instance_to_definition_cache.get(instance_id)
            if definition_id and definition_id in self._mappings:
                self._instance_to_definition_cache.move_to_end(instance_id)
                self._mappings.move_to_end(definition_id)
                return self._mappings[definition_id]

        # Fallback: Try to find through bubblelabs integration
        try:
            instances = self.bubblelabs.list_workflow_instances()
            for instance in instances:
                if instance.id == instance_id:
                    self._add_instance_mapping(instance_id, instance.definition_id)
                    with self.lock:
                        mapping = self._mappings.get(instance.definition_id)
                        if mapping:
                            self._mappings.move_to_end(instance.definition_id)
                        return mapping
        except Exception as e:
            logger.debug(f"Error finding mapping for instance {instance_id}: {e}")

        return None

    def _add_instance_mapping(self, instance_id: str, definition_id: str) -> None:
        """Add instance-to-definition mapping with LRU eviction."""
        with self.lock:
            if len(self._instance_to_definition_cache) >= self._MAX_CACHE_SIZE:
                oldest_instance, oldest_definition = self._instance_to_definition_cache.popitem(last=False)
                logger.debug(f"LRU eviction: removed instance mapping for {oldest_instance}")

            self._instance_to_definition_cache[instance_id] = definition_id
            self._instance_to_definition_cache.move_to_end(instance_id)


def create_bridge(
    config: Optional[BubbleLabsCrewAIConfig] = None,
    state_storage_dir: Optional[str] = None
) -> BubbleLabsCrewAIBridge:
    """
    Convenience function to create a BubbleLabs-CrewAI bridge.

    Args:
        config: Optional workflow configuration
        state_storage_dir: Directory for CrewAI state storage

    Returns:
        Configured bridge instance
    """
    bubblelabs = BubbleLabsIntegration()

    return BubbleLabsCrewAIBridge(
        bubblelabs_integration=bubblelabs,
        config=config,
        state_storage_dir=state_storage_dir
    )


if __name__ == "__main__":
    # Example usage
    bridge = create_bridge()

    print("BubbleLabs-CrewAI Bridge initialized (MIT-licensed)")
    print(f"Bridge ready for local execution")
