"""
BubbleLabs Integration for OpenEvolve Workflows

This module provides integration between OpenEvolve workflows and the BubbleLabs UI,
enabling visualization, interaction, and control of workflows through the BubbleLabs interface.

CIRCULAR IMPORT FIX: All imports from api_server are now lazy (inside functions)
to prevent circular import issues with the Z3 service chain.
"""

import json
from typing import Dict, Any, List, Optional, Set, Union
import asyncio
import logging
import threading
import time
import uuid

from workflow_structures import WorkflowState
from team_manager import TeamManager
from gauntlet_manager import GauntletManager

# Lazy imports from api_server to prevent circular imports
# These are imported inside functions where needed instead of at module level
_api_server_team_manager = None
_api_server_gauntlet_manager = None

def _get_api_server_managers():
    """Lazy import api_server managers to prevent circular imports.
    
    This function uses lazy imports to break the circular dependency chain:
    z3_api_server -> z3_leanaide_openevolve_integration -> bubblelabs_integration -> api_server
    
    Returns:
        Tuple of (team_manager, gauntlet_manager)
    """
    global _api_server_team_manager, _api_server_gauntlet_manager
    if _api_server_team_manager is None or _api_server_gauntlet_manager is None:
        try:
            # Lazy import to prevent circular import
            from api_server import team_manager, gauntlet_manager
            _api_server_team_manager = team_manager
            _api_server_gauntlet_manager = gauntlet_manager
        except Exception as e:
            # If api_server is not available (circular import or other error),
            # create local instances as fallback
            logger.debug(f"Using local TeamManager/GauntletManager due to: {e}")
            _api_server_team_manager = TeamManager()
            _api_server_gauntlet_manager = GauntletManager()
    return _api_server_team_manager, _api_server_gauntlet_manager

# Import LeanAide integration
try:
    from bubblelabs_leanaide_integration import (
        get_leanaide_bridge,
        LeanAideIntegrationBridge,
        LEANAIDE_AVAILABLE,
        MCTS_AVAILABLE,
        MDAP_AVAILABLE
    )
    LEANAIDE_INTEGRATION_AVAILABLE = True
except ImportError:
    LEANAIDE_INTEGRATION_AVAILABLE = False
    get_leanaide_bridge = None
    LeanAideIntegrationBridge = None

# Import state machine validation
try:
    from bubblelabs_crewai_bridge import (
        ExtendedWorkflowStatus,
        validate_workflow_transition,
        get_valid_workflow_transitions,
        is_terminal_workflow_status
    )
    STATE_VALIDATION_AVAILABLE = True
except ImportError:
    STATE_VALIDATION_AVAILABLE = False
    ExtendedWorkflowStatus = None
    validate_workflow_transition = None
    get_valid_workflow_transitions = None
    is_terminal_workflow_status = None



logger = logging.getLogger(__name__)

# Data models for BubbleLabs integration (simplified for local use)
class BubbleNode:
    """Represents a node in a BubbleLabs workflow graph."""
    def __init__(self, id: str, type: str, position: Dict[str, float], data: Dict[str, Any]):
        self.id = id
        self.type = type
        self.position = position
        self.data = data


class BubbleEdge:
    """Represents an edge in a BubbleLabs workflow graph."""
    def __init__(self, id: str, source: str, target: str, sourceHandle: str = None, targetHandle: str = None):
        self.id = id
        self.source = source
        self.target = target
        self.sourceHandle = sourceHandle
        self.targetHandle = targetHandle


class BubbleWorkflowDefinition:
    """Definition of a workflow for BubbleLabs visualization."""
    def __init__(self, id: str, name: str, description: str, nodes: List[Dict], edges: List[Dict], metadata: Dict[str, Any]):
        self.id = id
        self.name = name
        self.description = description
        self.nodes = nodes
        self.edges = edges
        self.metadata = metadata


class BubbleWorkflowInstance:
    """Represents a running instance of a BubbleLabs workflow."""
    def __init__(self, id: str, definition_id: str, status: str, created_at: float, updated_at: float, 
                 progress: float, current_node: str = None, data: Dict[str, Any] = None):
        self.id = id
        self.definition_id = definition_id
        self.status = status
        self.created_at = created_at
        self.updated_at = updated_at
        self.progress = progress
        self.current_node = current_node
        self.data = data or {}


class BubbleLabsIntegration:
    """
    Manages integration between OpenEvolve workflows and BubbleLabs UI.
    This is a local integration that works within the Streamlit application.

    CONCURRENCY FIX (Issues #3, #4): Thread-safe with proper locking hierarchy.

    Lock Hierarchy (to prevent deadlock):
    1. _state_lock (top-level) - protects all state changes
    2. Individual locks (_instances_lock, _definitions_lock, _threads_lock) are secondary
    3. Never acquire individual locks while holding _state_lock
    4. When acquiring multiple individual locks, always acquire in alphabetical order:
       - _definitions_lock
       - _instances_lock
       - _threads_lock
    """

    def __init__(self):
        self.workflow_instances: Dict[str, BubbleWorkflowInstance] = {}
        self.workflow_definitions: Dict[str, BubbleWorkflowDefinition] = {}
        self.running_threads: Dict[str, threading.Thread] = {}
        # Use lazy imports to avoid circular import issues
        team_mgr, gauntlet_mgr = _get_api_server_managers()
        self.team_manager = team_mgr
        self.gauntlet_manager = gauntlet_mgr

        # CONCURRENCY FIX (Issue #3): Use RLock for reentrancy and establish lock hierarchy
        # Using separate RLocks allows fine-grained locking while preventing deadlock
        self._instances_lock = threading.RLock()
        self._definitions_lock = threading.RLock()
        self._threads_lock = threading.RLock()

        # Lock order documentation for preventing deadlock
        self._lock_order = ["_definitions_lock", "_instances_lock", "_threads_lock"]

        # MEMORY LEAK FIX #3: TTL-based eviction for workflow instances
        # Prevents unbounded memory growth from accumulated instances
        self._MAX_INSTANCE_AGE_SECONDS = 7 * 24 * 3600  # 7 days
        self._MAX_INSTANCES = 1000

    def _cleanup_old_instances(self) -> int:
        """
        MEMORY LEAK FIX #3: Remove instances older than 7 days and enforce max limit.

        Returns:
            Number of instances removed
        """
        now = time.time()
        to_remove = []

        with self._instances_lock:
            # Check instance age
            for instance_id, instance in self.workflow_instances.items():
                # Get instance creation time
                instance_age = now - getattr(instance, 'created_at', now)
                if instance_age > self._MAX_INSTANCE_AGE_SECONDS:
                    to_remove.append(instance_id)

            # Remove old instances
            for instance_id in to_remove:
                del self.workflow_instances[instance_id]
                logger.info(f"Cleaned up old instance (age > 7 days): {instance_id}")

            # Also enforce max limit
            if len(self.workflow_instances) > self._MAX_INSTANCES:
                # Sort by created_at and remove oldest
                sorted_instances = sorted(
                    self.workflow_instances.items(),
                    key=lambda x: getattr(x[1], 'created_at', 0)
                )
                # Remove excess oldest instances
                excess = len(self.workflow_instances) - self._MAX_INSTANCES
                for instance_id, _ in sorted_instances[:excess]:
                    del self.workflow_instances[instance_id]
                    logger.info(f"Cleaned up excess instance (max limit): {instance_id}")
                to_remove.extend([inst_id for inst_id, _ in sorted_instances[:excess]])

        return len(to_remove)

    def _add_workflow_instance(self, instance_id: str, instance: BubbleWorkflowInstance) -> None:
        """
        MEMORY LEAK FIX #3: Add instance and trigger cleanup if needed.

        Automatically triggers periodic cleanup every 100 additions.
        """
        with self._instances_lock:
            self.workflow_instances[instance_id] = instance

        # Periodically cleanup (every 100 additions)
        if len(self.workflow_instances) % 100 == 0:
            removed = self._cleanup_old_instances()
            if removed > 0:
                logger.info(f"Periodic cleanup: removed {removed} old instances")

    def create_workflow_definition_from_openevolve(
        self, 
        problem_statement: str, 
        team_config: Dict[str, str],  # Maps roles to team names
        gauntlet_config: Dict[str, str]  # Maps gauntlet types to gauntlet names
    ) -> BubbleWorkflowDefinition:
        """
        Create a BubbleLabs workflow definition from OpenEvolve workflow parameters.
        
        Args:
            problem_statement: The problem to be solved by the workflow
            team_config: Configuration mapping roles to team names
            gauntlet_config: Configuration mapping gauntlet types to gauntlet names
            
        Returns:
            BubbleWorkflowDefinition: The workflow definition for BubbleLabs
        """
        workflow_id = str(uuid.uuid4())
        
        # Create nodes for the OpenEvolve workflow
        nodes = [
            {
                "id": "content_analysis",
                "type": "content_analyzer",
                "position": {"x": 0, "y": 0},
                "data": {
                    "label": "Content Analysis",
                    "team": team_config.get("content_analyzer_team", ""),
                    "description": "Analyze the problem statement and extract structured context"
                }
            },
            {
                "id": "decomposition",
                "type": "decomposer",
                "position": {"x": 300, "y": 0},
                "data": {
                    "label": "Problem Decomposition",
                    "team": team_config.get("planner_team", ""),
                    "description": "Break down the problem into sub-problems"
                }
            },
            {
                "id": "subproblem_solver",
                "type": "solver",
                "position": {"x": 600, "y": 0},
                "data": {
                    "label": "Sub-problem Solving",
                    "team": team_config.get("solver_team", ""),
                    "gauntlet": gauntlet_config.get("sub_problem_red_gauntlet", ""),
                    "description": "Solve each sub-problem with specified gauntlet validation"
                }
            },
            {
                "id": "final_verification",
                "type": "verifier",
                "position": {"x": 900, "y": 0},
                "data": {
                    "label": "Final Verification",
                    "team": team_config.get("assembler_team", ""),
                    "gauntlet": gauntlet_config.get("final_gold_gauntlet", ""),
                    "description": "Verify the final solution with gold team gauntlet"
                }
            }
        ]
        
        # Create edges connecting the nodes
        edges = [
            {
                "id": "edge_1",
                "source": "content_analysis",
                "target": "decomposition",
                "sourceHandle": "output",
                "targetHandle": "input"
            },
            {
                "id": "edge_2",
                "source": "decomposition",
                "target": "subproblem_solver",
                "sourceHandle": "output",
                "targetHandle": "input"
            },
            {
                "id": "edge_3",
                "source": "subproblem_solver",
                "target": "final_verification",
                "sourceHandle": "output",
                "targetHandle": "input"
            }
        ]
        
        definition = BubbleWorkflowDefinition(
            id=workflow_id,
            name=f"OpenEvolve Workflow: {problem_statement[:30]}...",
            description=f"OpenEvolve sovereign-grade decomposition for: {problem_statement}",
            nodes=nodes,
            edges=edges,
            metadata={
                "problem_statement": problem_statement,
                "team_config": team_config,
                "gauntlet_config": gauntlet_config,
                "created_at": time.time(),
                "workflow_type": "openevolve_sovereign_decomposition"
            }
        )

        # Thread-safe: use lock when modifying workflow_definitions
        with self._definitions_lock:
            self.workflow_definitions[workflow_id] = definition

        return definition
    
    def get_workflow_definition(self, definition_id: str) -> Optional[BubbleWorkflowDefinition]:
        """
        Get a workflow definition by ID.

        CONCURRENCY FIX (Issue #4): Protected with lock for thread-safe access.
        """
        with self._definitions_lock:
            return self.workflow_definitions.get(definition_id)

    def list_workflow_definitions(self) -> List[BubbleWorkflowDefinition]:
        """
        List all workflow definitions.

        CONCURRENCY FIX (Issue #4): Protected with lock to prevent race condition
        during dictionary iteration and list creation.
        """
        with self._definitions_lock:
            return list(self.workflow_definitions.values())

    def list_workflow_instances(self) -> List[BubbleWorkflowInstance]:
        """
        List all workflow instances.

        CONCURRENCY FIX (Issue #4): Protected with lock to prevent race condition
        during dictionary iteration and list creation.
        """
        with self._instances_lock:
            return list(self.workflow_instances.values())
    
    # =========================================================================
    # LeanAide Integration
    # =========================================================================
    
    def get_leanaide_bridge(self) -> Optional[LeanAideIntegrationBridge]:
        """
        Get the LeanAide integration bridge.
        
        Returns:
            LeanAideIntegrationBridge instance or None if not available
        """
        if not LEANAIDE_INTEGRATION_AVAILABLE:
            logger.warning("LeanAide integration not available")
            return None
        
        return get_leanaide_bridge()
    
    def is_leanaide_available(self) -> bool:
        """
        Check if LeanAide integration is available.
        
        Returns:
            True if LeanAide is available
        """
        return LEANAIDE_INTEGRATION_AVAILABLE and LEANAIDE_AVAILABLE
    
    def get_leanaide_status(self) -> Dict[str, Any]:
        """
        Get LeanAide integration status.
        
        Returns:
            Dictionary with LeanAide status information
        """
        if not LEANAIDE_INTEGRATION_AVAILABLE:
            return {"available": False, "reason": "Integration not available"}
        
        bridge = get_leanaide_bridge()
        if bridge is None:
            return {"available": False, "reason": "Bridge initialization failed"}
        
        return bridge.get_status()
    
    def control_workflow_local(self, instance_id: str, action: str) -> Dict[str, Any]:
        """
        Control a running workflow instance locally with state machine validation.

        CONCURRENCY FIX (Issue #3): Thread-safe with proper lock ordering to prevent deadlock.
        Follows lock hierarchy: never acquire _threads_lock while holding _instances_lock.

        STATE VALIDATION: All state transitions are validated before being applied.
        Invalid transitions are rejected with detailed error messages.

        Lock Order (alphabetical):
        1. _definitions_lock (if needed)
        2. _instances_lock
        3. _threads_lock

        Args:
            instance_id: ID of the workflow instance
            action: Action to perform (start, pause, resume, cancel, restart)

        Returns:
            Status of the control operation

        Raises:
            ValueError: If state transition is invalid
        """
        # CONCURRENCY FIX (Issue #3): Acquire thread info BEFORE instances lock
        # This prevents nested lock acquisition which could cause deadlock
        with self._threads_lock:
            has_thread = instance_id in self.running_threads
            thread = self.running_threads.get(instance_id) if has_thread else None

        # Now acquire instances lock separately
        with self._instances_lock:
            if instance_id not in self.workflow_instances:
                return {"error": "Workflow instance not found"}

            instance = self.workflow_instances[instance_id]
            current_status = instance.status
            new_status = None

            if action == "start":
                new_status = "running"
                if current_status not in ["pending", "created"]:
                    return {"error": f"Cannot start workflow from status: {current_status}"}

                # Validate transition if state validation is available
                if STATE_VALIDATION_AVAILABLE and not validate_workflow_transition(current_status, new_status):
                    valid_transitions = get_valid_workflow_transitions(current_status)
                    return {
                        "error": f"Invalid state transition: {current_status} -> {new_status}",
                        "valid_transitions": list(valid_transitions)
                    }

                instance.status = new_status
                instance.updated_at = time.time()
                return {"message": "Workflow started", "status": instance.status}

            elif action == "pause":
                new_status = "paused"
                if current_status != "running":
                    return {"error": f"Cannot pause workflow from status: {current_status}"}

                # Validate transition if state validation is available
                if STATE_VALIDATION_AVAILABLE and not validate_workflow_transition(current_status, new_status):
                    valid_transitions = get_valid_workflow_transitions(current_status)
                    return {
                        "error": f"Invalid state transition: {current_status} -> {new_status}",
                        "valid_transitions": list(valid_transitions)
                    }

                instance.status = new_status
                instance.updated_at = time.time()
                return {"message": "Workflow paused", "status": instance.status}

            elif action == "resume":
                new_status = "running"
                if current_status != "paused":
                    return {"error": f"Cannot resume workflow from status: {current_status}"}

                # Validate transition if state validation is available
                if STATE_VALIDATION_AVAILABLE and not validate_workflow_transition(current_status, new_status):
                    valid_transitions = get_valid_workflow_transitions(current_status)
                    return {
                        "error": f"Invalid state transition: {current_status} -> {new_status}",
                        "valid_transitions": list(valid_transitions)
                    }

                instance.status = new_status
                instance.updated_at = time.time()
                return {"message": "Workflow resumed", "status": instance.status}

            elif action == "cancel":
                new_status = "cancelled"

                # Validate transition if state validation is available
                if STATE_VALIDATION_AVAILABLE and not validate_workflow_transition(current_status, new_status):
                    valid_transitions = get_valid_workflow_transitions(current_status)
                    return {
                        "error": f"Invalid state transition: {current_status} -> {new_status}",
                        "valid_transitions": list(valid_transitions)
                    }

                instance.status = new_status
                instance.updated_at = time.time()

                # CONCURRENCY FIX (Issue #3): Thread cleanup done outside instances lock
                # to prevent nested lock acquisition
                instance.data["cancel_requested"] = True

                # Release instances lock before acquiring threads lock (prevents deadlock)
                # We'll handle thread cleanup after releasing this lock

            elif action == "restart":
                # Cancel the current instance
                new_status = "cancelled"

                # Validate transition if state validation is available
                if STATE_VALIDATION_AVAILABLE and not validate_workflow_transition(current_status, new_status):
                    valid_transitions = get_valid_workflow_transitions(current_status)
                    return {
                        "error": f"Invalid state transition: {current_status} -> {new_status}",
                        "valid_transitions": list(valid_transitions)
                    }

                instance.status = new_status
                instance.updated_at = time.time()

                # Return a message to create a new instance
                return {"message": "Please create a new instance to restart workflow"}

        # MEMORY LEAK FIX (Leak #1): Proper thread cleanup with join and verification
        # This prevents thread leakage and resource exhaustion
        if action == "cancel" and thread:
            with self._threads_lock:
                # Signal thread to stop
                if hasattr(thread, "cancel_event"):
                    try:
                        thread.cancel_event.set()
                    except (RuntimeError, AttributeError):
                        logger.debug(f"Failed to set cancel_event for {instance_id}")
                if hasattr(thread, "stop_event"):
                    try:
                        thread.stop_event.set()
                    except (RuntimeError, AttributeError):
                        logger.debug(f"Failed to set stop_event for {instance_id}")

            # CRITICAL FIX: Join thread with timeout to ensure it stops
            # Do this OUTSIDE threads_lock to avoid deadlock
            try:
                thread.join(timeout=30)
                if thread.is_alive():
                    logger.warning(
                        f"Thread for instance {instance_id} did not stop within 30s timeout. "
                        f"This may indicate a hung thread that continues consuming resources."
                    )
                    # As last resort, we could consider thread termination methods
                    # but Python doesn't provide safe thread termination
                else:
                    logger.debug(f"Thread for instance {instance_id} stopped successfully")
            except Exception as e:
                logger.error(f"Error joining thread for instance {instance_id}: {e}")

            # Only remove from running_threads after confirming thread stopped
            with self._threads_lock:
                if not thread.is_alive():
                    self.running_threads.pop(instance_id, None)
                    logger.debug(f"Removed thread for instance {instance_id} from running_threads")
                else:
                    # Keep in dict for monitoring/cleanup, but log warning
                    logger.warning(
                        f"Thread for {instance_id} still running after cancel. "
                        f"Keeping in running_threads for later cleanup."
                    )

            return {"message": "Workflow cancelled", "status": "cancelled"}

        # For non-cancel actions, return status from instance
        with self._instances_lock:
            instance = self.workflow_instances.get(instance_id)
            if instance:
                return {"message": f"Action '{action}' performed", "status": instance.status}

        return {"error": "Workflow instance not found"}

    def get_knowledge_graph_visualization(
        self,
        use_pygraphistry: bool = True,
        max_nodes: int = 500,
        apply_clustering: bool = True,
        clustering_method: str = "dbscan",
        embedding_method: str = "umap"
    ) -> Optional[str]:
        """
        Get knowledge graph visualization using PyGraphistry or fallback method.

        Args:
            use_pygraphistry: Whether to use PyGraphistry for visualization
            max_nodes: Maximum number of nodes to include
            apply_clustering: Whether to apply clustering pipeline (PyGraphistry only)
            clustering_method: Clustering method ('dbscan', 'kmeans') (PyGraphistry only)
            embedding_method: Embedding method ('umap', 'pca') (PyGraphistry only)

        Returns:
            Path or URL to the visualization, or None if failed
        """
        try:
            from knowledge_graph_visualizer import KnowledgeGraphVisualizer

            # Create visualizer with PyGraphistry support
            visualizer = KnowledgeGraphVisualizer(use_pygraphistry=use_pygraphistry)

            # Build the graph
            stats = visualizer.build_graph(max_nodes=max_nodes)

            if stats.get("nodes", 0) == 0:
                print("No nodes in knowledge graph to visualize")
                return None

            # Create output path
            import tempfile
            import os
            output_path = os.path.join(tempfile.gettempdir(), f"knowledge_graph_viz_{hash(str(stats))}.html")

            # Visualize with the specified parameters
            success = visualizer.visualize_interactive(
                output_path=output_path,
                apply_clustering=apply_clustering,
                clustering_method=clustering_method,
                embedding_method=embedding_method
            )

            if success:
                return output_path
            else:
                print("Failed to create knowledge graph visualization")
                return None

        except ImportError as e:
            print(f"Knowledge graph visualization not available: {e}")
            return None
        except Exception as e:
            print(f"Error in get_knowledge_graph_visualization: {e}")
            return None


# Initialize the integration manager (lazy initialization)
_bubblelabs_integration_instance = None

def get_bubblelabs_integration():
    """Get the singleton BubbleLabsIntegration instance (lazy initialization)."""
    global _bubblelabs_integration_instance
    if _bubblelabs_integration_instance is None:
        _bubblelabs_integration_instance = BubbleLabsIntegration()
    return _bubblelabs_integration_instance


# Backward compatibility: module-level access will use lazy initialization
# Note: Direct use of bubblelabs_integration at module level is deprecated.
# Use get_bubblelabs_integration() instead.
class _LazyIntegrationProxy:
    """Proxy that lazily initializes the real integration on first access."""
    
    def __getattr__(self, name):
        instance = get_bubblelabs_integration()
        return getattr(instance, name)
    
    def __setattr__(self, name, value):
        instance = get_bubblelabs_integration()
        return setattr(instance, name, value)
    
    def __call__(self, *args, **kwargs):
        instance = get_bubblelabs_integration()
        return instance(*args, **kwargs)


bubblelabs_integration = _LazyIntegrationProxy()


if __name__ == "__main__":
    # This module is primarily a library, but can be used for testing
    print("BubbleLabs Integration module loaded successfully")
