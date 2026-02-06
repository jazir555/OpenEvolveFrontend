"""
bubblelabs_maker_integration.py - CrewAI Integration

This file uses CrewAI (MIT) for workflow orchestration.

For questions, see: CREWAI_MIGRATION_MASTER_TASKLIST.md
"""

"""
BubbleLabs Integration with Maker Engine and CrewAI

This module provides comprehensive UI components and integration logic for
connecting BubbleLabs with:
- Maker Engine (MAKER framework for zero-error task solving)
- CrewAI workflow system (project management and ticket tracking)
- ROMA MDAP Maker integration (unified task execution)

Features:
- Tool creation workflow visualization
- CrewAI task status tracking
- MDAP maker step-by-step progress
- Tool testing and validation interface
- Tool repository browser

Based on papers:
- "Solving a Million-Step LLM Task with Zero Errors" (arXiv:2511.09030)
"""

import json
import logging
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
from ui_shim import ui as st
import pandas as pd

# Import Maker Engine components
try:
    from maker_engine import MakerEngine, MakerConfig, MakerState, MakerStep, MakerRunResult
    from mdap_maker_complete import (
        MAKEREngine,
        RecursiveMAKERSolver,
        VotingEngine,
        VoteCollector,
        TaskDecomposition,
        MAKERRunMetrics
    )
    from maker_integration_bridge import (
        MAKERIntegrationBridge,
        MAKERIntegrationConfig,
        create_maker_config
    )
    MAKER_AVAILABLE = True
except ImportError as e:
    MAKER_AVAILABLE = False
    logging.warning(f"Maker Engine not available: {e}")

# Import CrewAI integration components
try:
    from crewai_integration import (
        CrewAIIntegrationManager,
        CrewAIClient,
        TicketStatus,
        TicketType
    )
    CREWAI_INTEGRATION_AVAILABLE = True
except ImportError as e:
    CREWAI_INTEGRATION_AVAILABLE = False
    logging.warning(f"CrewAI integration not available: {e}")

# Import ROMA MDAP
try:
    from mdap_engine import MDAPTask, MDAPConfig
    ROMA_AVAILABLE = True
except ImportError:
    ROMA_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ToolStatus(Enum):
    """Status of tools in the repository"""
    DRAFT = "draft"
    TESTING = "testing"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"


class DelegationStatus(Enum):
    """Status of delegated CrewAI workflow tasks"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    REVIEW = "in_review"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class ToolDefinition:
    """Represents a tool created by Maker"""
    tool_id: str
    name: str
    description: str
    version: str
    status: ToolStatus
    maker_mode: str  # "sequential", "recursive", "hybrid"
    config: Dict[str, Any]
    prompt_template: Optional[str] = None
    system_prompt: Optional[str] = None
    expected_schema: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = "system"
    test_results: Optional[Dict[str, Any]] = None
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolDefinition':
        """Create from dictionary"""
        # Handle enum conversion
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = ToolStatus(data['status'])
        return cls(**data)


@dataclass
class CrewAIDelegation:
    """Represents a task delegated to CrewAI workflow"""
    delegation_id: str
    task_id: str  # CrewAI ticket ID
    title: str
    description: str
    status: DelegationStatus
    delegation_type: str  # "maker_run", "mdap_task", "custom_tool"
    tool_id: Optional[str] = None
    workflow_epic_id: Optional[str] = None
    assigned_to: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CrewAIDelegation':
        """Create from dictionary"""
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = DelegationStatus(data['status'])
        return cls(**data)


@dataclass
class ToolExecutionResult:
    """Result of executing a tool"""
    tool_id: str
    execution_id: str
    input_data: Dict[str, Any]
    output_data: Any
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    CREWAI_ticket_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# TOOL REPOSITORY
# =============================================================================

class ToolRepository:
    """
    Manages the repository of tools created by Maker Engine.

    Provides functionality for:
    - Tool registration and storage
    - Tool versioning
    - Tool discovery and search
    - Tool usage tracking
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "./tool_repository.json"
        self.tools: Dict[str, ToolDefinition] = {}
        self.lock = threading.Lock()
        self._load_repository()

    def register_tool(
        self,
        name: str,
        description: str,
        maker_mode: str,
        config: Dict[str, Any],
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        expected_schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ToolDefinition:
        """Register a new tool in the repository"""
        with self.lock:
            tool_id = f"tool_{int(time.time() * 1000)}"

            tool = ToolDefinition(
                tool_id=tool_id,
                name=name,
                description=description,
                version="1.0.0",
                status=ToolStatus.DRAFT,
                maker_mode=maker_mode,
                config=config,
                prompt_template=prompt_template,
                system_prompt=system_prompt,
                expected_schema=expected_schema,
                metadata=metadata or {}
            )

            self.tools[tool_id] = tool
            self._save_repository()

            logger.info(f"Registered tool: {tool_id} - {name}")
            return tool

    def update_tool(
        self,
        tool_id: str,
        status: Optional[ToolStatus] = None,
        test_results: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing tool"""
        with self.lock:
            if tool_id not in self.tools:
                return False

            tool = self.tools[tool_id]

            if status:
                tool.status = status
            if test_results:
                tool.test_results = test_results
            if metadata:
                tool.metadata.update(metadata)

            self._save_repository()
            return True

    def get_tool(self, tool_id: str) -> Optional[ToolDefinition]:
        """Get a tool by ID"""
        return self.tools.get(tool_id)

    def list_tools(
        self,
        status_filter: Optional[ToolStatus] = None,
        maker_mode_filter: Optional[str] = None
    ) -> List[ToolDefinition]:
        """List tools with optional filters"""
        tools = list(self.tools.values())

        if status_filter:
            tools = [t for t in tools if t.status == status_filter]

        if maker_mode_filter:
            tools = [t for t in tools if t.maker_mode == maker_mode_filter]

        return sorted(tools, key=lambda t: t.created_at, reverse=True)

    def search_tools(self, query: str) -> List[ToolDefinition]:
        """Search tools by name or description"""
        query_lower = query.lower()
        return [
            tool for tool in self.tools.values()
            if query_lower in tool.name.lower() or query_lower in tool.description.lower()
        ]

    def increment_usage(self, tool_id: str) -> bool:
        """Increment tool usage counter"""
        with self.lock:
            if tool_id not in self.tools:
                return False

            self.tools[tool_id].usage_count += 1
            self._save_repository()
            return True

    def _load_repository(self):
        """Load repository from storage"""
        try:
            import os
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.tools = {
                        tool_id: ToolDefinition.from_dict(tool_data)
                        for tool_id, tool_data in data.get('tools', {}).items()
                    }
                logger.info(f"Loaded {len(self.tools)} tools from repository")
        except (OSError, IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load tool repository: {e}")
            self.tools = {}

    def _save_repository(self):
        """Save repository to storage"""
        try:
            data = {
                'tools': {
                    tool_id: tool.to_dict()
                    for tool_id, tool in self.tools.items()
                },
                'last_updated': datetime.now().isoformat()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except (OSError, IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to save tool repository: {e}")


# =============================================================================
# CREWAI DELEGATION MANAGER
# =============================================================================

class CrewAIDelegationManager:
    """
    Manages delegation of tasks to CrewAI workflow system.

    Provides:
    - Task delegation to CrewAI tickets
    - Status tracking of delegated tasks
    - Result synchronization
    - Progress monitoring
    """

    def __init__(self, crewai_manager: Optional[CrewAIIntegrationManager] = None):
        self.crewai_manager = crewai_manager
        self.delegations: Dict[str, CrewAIDelegation] = {}
        self.lock = threading.Lock()

    def delegate_maker_run(
        self,
        run_id: str,
        title: str,
        description: str,
        initial_state: Any,
        maker_config: MakerConfig,
        workflow_epic_id: Optional[str] = None
    ) -> Optional[CrewAIDelegation]:
        """Delegate a MAKER run to CrewAI workflow"""
        if not self.crewai_manager or not CREWAI_INTEGRATION_AVAILABLE:
            logger.warning("CrewAI integration not available for delegation")
            return None

        try:
            # Create ticket in CrewAI
            ticket_id = self.crewai_manager.sync_maker_run(
                run_id=run_id,
                initial_state=initial_state,
                config=maker_config,
                workflow_epic_id=workflow_epic_id
            )

            if not ticket_id:
                return None

            # Create delegation record
            delegation_id = f"del_{int(time.time() * 1000)}"
            delegation = CrewAIDelegation(
                delegation_id=delegation_id,
                task_id=ticket_id,
                title=title,
                description=description,
                status=DelegationStatus.ASSIGNED,
                delegation_type="maker_run",
                workflow_epic_id=workflow_epic_id,
                metadata={"run_id": run_id}
            )

            with self.lock:
                self.delegations[delegation_id] = delegation

            logger.info(f"Delegated MAKER run {run_id} to CrewAI workflow ticket {ticket_id}")
            return delegation

        except (RuntimeError, ConnectionError, ValueError) as e:
            logger.error(f"Failed to delegate MAKER run: {e}")
            return None

    def delegate_tool_execution(
        self,
        tool_id: str,
        tool_name: str,
        input_data: Dict[str, Any],
        workflow_epic_id: Optional[str] = None
    ) -> Optional[CrewAIDelegation]:
        """Delegate a tool execution to CrewAI workflow"""
        if not self.crewai_manager or not CREWAI_INTEGRATION_AVAILABLE:
            return None

        try:
            # Create ticket for tool execution
            description = f"""
Execute Tool: {tool_name}
Tool ID: {tool_id}

Input Data:
{json.dumps(input_data, indent=2)}

This task represents the execution of a tool created by the Maker Engine.
            """.strip()

            ticket_id = self.crewai_manager.client.create_ticket(
                title=f"Execute Tool: {tool_name}",
                description=description,
                ticket_type=TicketType.TASK,
                labels=["tool-execution", f"tool-{tool_id}"]
            )

            if not ticket_id:
                return None

            # Create delegation record
            delegation_id = f"del_{int(time.time() * 1000)}"
            delegation = CrewAIDelegation(
                delegation_id=delegation_id,
                task_id=ticket_id,
                title=f"Execute Tool: {tool_name}",
                description=description,
                status=DelegationStatus.PENDING,
                delegation_type="custom_tool",
                tool_id=tool_id,
                workflow_epic_id=workflow_epic_id,
                metadata={"input_data": input_data}
            )

            with self.lock:
                self.delegations[delegation_id] = delegation

            logger.info(f"Delegated tool execution {tool_id} to CrewAI workflow ticket {ticket_id}")
            return delegation

        except (RuntimeError, ConnectionError, ValueError) as e:
            logger.error(f"Failed to delegate tool execution: {e}")
            return None

    def update_delegation_status(
        self,
        delegation_id: str,
        status: DelegationStatus,
        result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update status of a delegation"""
        with self.lock:
            if delegation_id not in self.delegations:
                return False

            delegation = self.delegations[delegation_id]
            delegation.status = status
            delegation.updated_at = datetime.now().isoformat()

            if result:
                delegation.result = result

            if status == DelegationStatus.COMPLETE:
                delegation.completed_at = datetime.now().isoformat()

            # Sync to CrewAI workflow if available
            if self.crewai_manager and CREWAI_INTEGRATION_AVAILABLE:
                try:
                    ticket_status = self._map_delegation_to_ticket_status(status)
                    self.crewai_manager.client.update_ticket(
                        ticket_id=delegation.task_id,
                        status=ticket_status
                    )
                except (RuntimeError, ConnectionError, ValueError) as e:
                    logger.error(f"Failed to sync status to CrewAI: {e}")

            return True

    def get_delegation(self, delegation_id: str) -> Optional[CrewAIDelegation]:
        """Get a delegation by ID"""
        return self.delegations.get(delegation_id)

    def list_delegations(
        self,
        status_filter: Optional[DelegationStatus] = None,
        delegation_type_filter: Optional[str] = None
    ) -> List[CrewAIDelegation]:
        """List delegations with optional filters"""
        delegations = list(self.delegations.values())

        if status_filter:
            delegations = [d for d in delegations if d.status == status_filter]

        if delegation_type_filter:
            delegations = [d for d in delegations if d.delegation_type == delegation_type_filter]

        return sorted(delegations, key=lambda d: d.created_at, reverse=True)

    def sync_from_crewai(self) -> int:
        """Sync delegation statuses from CrewAI workflow"""
        if not self.crewai_manager or not CREWAI_INTEGRATION_AVAILABLE:
            return 0

        synced = 0
        for delegation in self.delegations.values():
            try:
                # Get ticket from CrewAI
                ticket = self.crewai_manager.client.get_ticket(delegation.task_id)
                if ticket:
                    # Map ticket status to delegation status
                    ticket_status = ticket.get('status')
                    new_status = self._map_ticket_to_delegation_status(ticket_status)

                    if new_status != delegation.status:
                        delegation.status = new_status
                        delegation.updated_at = datetime.now().isoformat()
                        synced += 1

            except (RuntimeError, ConnectionError, ValueError) as e:
                logger.error(f"Failed to sync delegation {delegation.delegation_id}: {e}")

        if synced > 0:
            logger.info(f"Synced {synced} delegations from CrewAI workflow")

        return synced

    def _map_delegation_to_ticket_status(self, status: DelegationStatus) -> TicketStatus:
        """Map delegation status to CrewAI ticket status"""
        mapping = {
            DelegationStatus.PENDING: TicketStatus.TODO,
            DelegationStatus.ASSIGNED: TicketStatus.IN_PROGRESS,
            DelegationStatus.IN_PROGRESS: TicketStatus.IN_PROGRESS,
            DelegationStatus.REVIEW: TicketStatus.IN_REVIEW,
            DelegationStatus.COMPLETE: TicketStatus.DONE,
            DelegationStatus.FAILED: TicketStatus.BLOCKED
        }
        return mapping.get(status, TicketStatus.TODO)

    def _map_ticket_to_delegation_status(self, ticket_status: str) -> DelegationStatus:
        """Map CrewAI ticket status to delegation status"""
        mapping = {
            TicketStatus.TODO.value: DelegationStatus.PENDING,
            TicketStatus.IN_PROGRESS.value: DelegationStatus.IN_PROGRESS,
            TicketStatus.IN_REVIEW.value: DelegationStatus.REVIEW,
            TicketStatus.DONE.value: DelegationStatus.COMPLETE,
            TicketStatus.BLOCKED.value: DelegationStatus.FAILED
        }
        return mapping.get(ticket_status, DelegationStatus.PENDING)


# =============================================================================
# MAKER WORKFLOW MANAGER
# =============================================================================

class MakerWorkflowManager:
    """
    Manages Maker workflows with UI visualization and CrewAI integration.

    This is the main coordinator for:
    - Tool creation workflows
    - Tool execution
    - Progress tracking
    - Result visualization
    """

    def __init__(
        self,
        tool_repository: Optional[ToolRepository] = None,
        delegation_manager: Optional[CrewAIDelegationManager] = None
    ):
        self.tool_repository = tool_repository or ToolRepository()
        self.delegation_manager = delegation_manager
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_results: Dict[str, ToolExecutionResult] = {}

    def create_tool_workflow(
        self,
        name: str,
        description: str,
        task: str,
        maker_mode: str = "recursive",
        k_ahead: int = 3,
        max_depth: int = 5,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[ToolDefinition], Optional[str]]:
        """
        Create a new tool using Maker workflow.

        Returns:
            Tuple of (tool_definition, error_message)
        """
        if not MAKER_AVAILABLE:
            return None, "Maker Engine not available"

        try:
            # Create MAKER config
            config = create_maker_config(
                mode=maker_mode,
                k_ahead=k_ahead,
                max_depth=max_depth
            )

            # Store workflow state
            workflow_id = f"workflow_{int(time.time() * 1000)}"
            self.active_workflows[workflow_id] = {
                "name": name,
                "description": description,
                "task": task,
                "mode": maker_mode,
                "config": config,
                "context": context or {},
                "status": "initializing",
                "created_at": datetime.now().isoformat()
            }

            # Create tool definition (draft status)
            tool = self.tool_repository.register_tool(
                name=name,
                description=description,
                maker_mode=maker_mode,
                config=config,
                metadata={"workflow_id": workflow_id}
            )

            self.active_workflows[workflow_id]["tool_id"] = tool.tool_id
            self.active_workflows[workflow_id]["status"] = "created"

            return tool, None

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Failed to create tool workflow: {e}")
            return None, str(e)

    def execute_tool_workflow(
        self,
        tool_id: str,
        input_data: Dict[str, Any],
        delegate_to_crewai: bool = False
    ) -> Tuple[Optional[ToolExecutionResult], Optional[str]]:
        """
        Execute a tool workflow.

        Returns:
            Tuple of (execution_result, error_message)
        """
        if not MAKER_AVAILABLE:
            return None, "Maker Engine not available"

        try:
            # Get tool definition
            tool = self.tool_repository.get_tool(tool_id)
            if not tool:
                return None, f"Tool {tool_id} not found"

            # Delegate to CrewAI workflow if requested
            delegation = None
            if delegate_to_crewai and self.delegation_manager:
                delegation = self.delegation_manager.delegate_tool_execution(
                    tool_id=tool_id,
                    tool_name=tool.name,
                    input_data=input_data
                )

            # Execute using MAKER
            start_time = time.time()

            config = MAKERIntegrationConfig(**tool.config)
            bridge = MAKERIntegrationBridge(config)

            result = bridge.solve(
                task=input_data.get("task", tool.description),
                context=input_data.get("context", {})
            )

            execution_time = time.time() - start_time

            # Create execution result
            execution_result = ToolExecutionResult(
                tool_id=tool_id,
                execution_id=f"exec_{int(time.time() * 1000)}",
                input_data=input_data,
                output_data=result,
                execution_time=execution_time,
                success=result.get("success", False),
                error_message=result.get("error"),
                metrics=result.get("metrics"),
                CREWAI_ticket_id=delegation.task_id if delegation else None
            )

            # Store result
            self.workflow_results[execution_result.execution_id] = execution_result

            # Update tool usage
            self.tool_repository.increment_usage(tool_id)

            # Update delegation status
            if delegation:
                status = DelegationStatus.COMPLETE if result.get("success") else DelegationStatus.FAILED
                self.delegation_manager.update_delegation_status(
                    delegation.delegation_id,
                    status,
                    {"result": result}
                )

            return execution_result, None

        except (RuntimeError, ValueError, ConnectionError) as e:
            logger.error(f"Failed to execute tool workflow: {e}")
            return None, str(e)

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active workflow"""
        return self.active_workflows.get(workflow_id)

    def get_execution_result(self, execution_id: str) -> Optional[ToolExecutionResult]:
        """Get execution result by ID"""
        return self.workflow_results.get(execution_id)

    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all active workflows"""
        return list(self.active_workflows.values())


# =============================================================================
# BUBBLELABS UI COMPONENTS
# =============================================================================

class BubbleLabsMakerUI:
    """
    Streamlit UI components for BubbleLabs + Maker + CrewAI integration.

    Provides:
    - Tool creation wizard
    - Tool repository browser
    - Tool execution interface
    - CrewAI delegation tracker
    - Workflow progress visualization
    """

    def __init__(self):
        self.tool_repository = ToolRepository()
        self.delegation_manager = CrewAIDelegationManager()
        self.workflow_manager = MakerWorkflowManager(
            tool_repository=self.tool_repository,
            delegation_manager=self.delegation_manager
        )

    def render_maker_studio(self):
        """Render the main Maker Studio interface"""
        st.header("🛠️ Maker Studio")

        # Create tabs
        tabs = st.tabs([
            "🔨 Tool Creator",
            "📦 Tool Repository",
            "⚡ Tool Executor",
            "📋 CrewAI Workflow Tracker",
            "📊 Workflow Analytics"
        ])

        with tabs[0]:
            self._render_tool_creator()

        with tabs[1]:
            self._render_tool_repository()

        with tabs[2]:
            self._render_tool_executor()

        with tabs[3]:
            self._render_CREWAI_tracker()

        with tabs[4]:
            self._render_workflow_analytics()

    def _render_tool_creator(self):
        """Render tool creation wizard"""
        st.subheader("Create New Tool with Maker")

        # Step 1: Basic Information
        st.markdown("### Step 1: Basic Information")
        col1, col2 = st.columns(2)

        with col1:
            tool_name = st.text_input(
                "Tool Name",
                placeholder="e.g., Data Analyzer, Code Generator",
                key="maker_tool_name"
            )

        with col2:
            maker_mode = st.selectbox(
                "Maker Mode",
                options=["sequential", "recursive", "hybrid"],
                index=1,
                help="sequential: For step-by-step tasks | recursive: For decomposition tasks | hybrid: Combined approach",
                key="maker_mode"
            )

        tool_description = st.text_area(
            "Tool Description",
            placeholder="Describe what this tool does...",
            key="maker_tool_desc",
            height=100
        )

        # Step 2: Task Definition
        st.markdown("### Step 2: Task Definition")
        task_description = st.text_area(
            "Task to Solve",
            placeholder="Describe the task this tool should solve...",
            key="maker_task",
            height=150
        )

        # Step 3: Maker Configuration
        st.markdown("### Step 3: Maker Configuration")

        col3, col4, col5 = st.columns(3)

        with col3:
            k_ahead = st.number_input(
                "K-Ahead (Voting Threshold)",
                min_value=1,
                max_value=10,
                value=3,
                help="Number of votes needed for consensus",
                key="maker_k_ahead"
            )

        with col4:
            max_depth = st.number_input(
                "Max Decomposition Depth",
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum recursion depth for decomposition",
                key="maker_max_depth"
            )

        with col5:
            enable_red_flagging = st.checkbox(
                "Enable Red-Flagging",
                value=True,
                help="Filter out unreliable responses",
                key="maker_red_flag"
            )

        # Additional context
        with st.expander("Advanced Configuration"):
            context_json = st.text_area(
                "Additional Context (JSON)",
                placeholder='{"key": "value"}',
                key="maker_context",
                height=100
            )

        # Create button
        col6, col7, col8 = st.columns([1, 2, 1])

        with col7:
            create_button = st.button(
                "🚀 Create Tool",
                type="primary",
                use_container_width=True
            )

        if create_button:
            if not tool_name or not task_description:
                st.error("Please fill in all required fields")
            else:
                with st.spinner("Creating tool with Maker Engine..."):
                    # Parse context
                    context = {}
                    if context_json:
                        try:
                            context = json.loads(context_json)
                        except json.JSONDecodeError:
                            st.error("Invalid JSON in context field")
                            return

                    # Create tool
                    tool, error = self.workflow_manager.create_tool_workflow(
                        name=tool_name,
                        description=tool_description,
                        task=task_description,
                        maker_mode=maker_mode,
                        k_ahead=k_ahead,
                        max_depth=max_depth,
                        context=context
                    )

                    if tool:
                        st.success(f"[OK] Tool '{tool_name}' created successfully!")
                        st.json(tool.to_dict())
                    else:
                        st.error(f"[FAIL] Failed to create tool: {error}")

    def _render_tool_repository(self):
        """Render tool repository browser"""
        st.subheader("Tool Repository")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            status_filter = st.selectbox(
                "Filter by Status",
                options=["All"] + [s.value for s in ToolStatus],
                key="repo_status_filter"
            )

        with col2:
            mode_filter = st.selectbox(
                "Filter by Mode",
                options=["All", "sequential", "recursive", "hybrid"],
                key="repo_mode_filter"
            )

        with col3:
            search_query = st.text_input(
                "Search Tools",
                placeholder="Search by name or description...",
                key="repo_search"
            )

        # Get tools
        tools = self.tool_repository.list_tools()

        # Apply filters
        if status_filter != "All":
            tools = [t for t in tools if t.status.value == status_filter]
        if mode_filter != "All":
            tools = [t for t in tools if t.maker_mode == mode_filter]
        if search_query:
            tools = [t for t in tools if search_query.lower() in t.name.lower() or
                     search_query.lower() in t.description.lower()]

        # Display tools
        if not tools:
            st.info("📭 No tools found. Create your first tool in the Tool Creator tab!")
        else:
            st.info(f"📦 Found {len(tools)} tool(s)")

            for tool in tools:
                with st.expander(f"🔧 {tool.name} ({tool.version}) - {tool.status.value.title()}"):
                    # Tool details
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**ID:** `{tool.tool_id}`")
                        st.markdown(f"**Mode:** {tool.maker_mode}")
                        st.markdown(f"**Created:** {tool.created_at}")
                        st.markdown(f"**Usage Count:** {tool.usage_count}")

                    with col2:
                        st.markdown(f"**Status:** {tool.status.value.title()}")
                        if tool.test_results:
                            st.markdown("**Test Results:** [OK] Passed")
                        else:
                            st.markdown("**Test Results:** ⏳ Not tested")

                    st.markdown(f"**Description:** {tool.description}")

                    # Action buttons
                    col3, col4, col5 = st.columns(3)

                    with col3:
                        if st.button(f"▶️ Test", key=f"test_{tool.tool_id}"):
                            st.session_state[f"test_tool_{tool.tool_id}"] = True

                    with col4:
                        if st.button(f"📊 Details", key=f"details_{tool.tool_id}"):
                            st.json(tool.to_dict())

                    with col5:
                        if tool.status == ToolStatus.DRAFT:
                            if st.button(f"[OK] Validate", key=f"validate_{tool.tool_id}"):
                                self.tool_repository.update_tool(
                                    tool.tool_id,
                                    status=ToolStatus.VALIDATED
                                )
                                st.rerun()

    def _render_tool_executor(self):
        """Render tool execution interface"""
        st.subheader("Execute Tool")

        # Select tool
        tools = self.tool_repository.list_tools(status_filter=ToolStatus.VALIDATED)

        if not tools:
            st.warning("[WARN] No validated tools available. Please validate a tool first.")
            return

        tool_options = {f"{t.name} ({t.tool_id})": t for t in tools}
        selected = st.selectbox(
            "Select Tool to Execute",
            options=list(tool_options.keys()),
            key="exec_tool_select"
        )

        if selected:
            tool = tool_options[selected]

            # Display tool info
            st.info(f"**Description:** {tool.description}")
            st.info(f"**Mode:** {tool.maker_mode} | **Usage Count:** {tool.usage_count}")

            # Input data
            st.markdown("### Input Data")

            input_json = st.text_area(
                "Input (JSON)",
                placeholder='{"task": "Your task here", "context": {}}',
                key="exec_input",
                height=150
            )

            # Options
            col1, col2 = st.columns(2)

            with col1:
                delegate_to_crewai = st.checkbox(
                    "Delegate to CrewAI",
                    value=False,
                    help="Create a CrewAI workflow ticket for this execution",
                    key="exec_delegate"
                )

            with col2:
                show_progress = st.checkbox(
                    "Show Progress",
                    value=True,
                    help="Display real-time progress updates",
                    key="exec_show_progress"
                )

            # Execute button
            if st.button("⚡ Execute", type="primary", key="exec_button"):
                try:
                    input_data = json.loads(input_json) if input_json else {}

                    with st.spinner(f"Executing {tool.name}..."):
                        result, error = self.workflow_manager.execute_tool_workflow(
                            tool_id=tool.tool_id,
                            input_data=input_data,
                            delegate_to_crewai=delegate_to_crewai
                        )

                        if result:
                            # Display result
                            st.success(f"[OK] Execution completed in {result.execution_time:.2f}s")

                            if result.CREWAI_ticket_id:
                                st.info(f"📋 CrewAI Ticket: {result.CREWAI_ticket_id}")

                            # Tabs for result details
                            tabs = st.tabs(["Output", "Metrics", "Input"])

                            with tabs[0]:
                                st.json(result.output_data)

                            with tabs[1]:
                                if result.metrics:
                                    st.json(result.metrics)
                                else:
                                    st.info("No metrics available")

                            with tabs[2]:
                                st.json(result.input_data)

                        else:
                            st.error(f"[FAIL] Execution failed: {error}")

                except json.JSONDecodeError:
                    st.error("Invalid JSON input")

    def _render_crewai_tracker(self):
        """Render CrewAI workflow delegation tracker"""
        st.subheader("📋 CrewAI Workflow Tracker")

        if not CREWAI_INTEGRATION_AVAILABLE:
            st.warning("[WARN] CrewAI integration not available")
            return

        # Status filter
        col1, col2 = st.columns(2)

        with col1:
            status_filter = st.selectbox(
                "Filter by Status",
                options=["All"] + [s.value for s in DelegationStatus],
                key="heph_status_filter"
            )

        with col2:
            type_filter = st.selectbox(
                "Filter by Type",
                options=["All", "maker_run", "custom_tool", "mdap_task"],
                key="heph_type_filter"
            )

        # Sync button
        if st.button("🔄 Sync Delegations", key="crewai_sync"):
            with st.spinner("Syncing..."):
                synced = self.delegation_manager.sync_from_crewai()
                st.success(f"Synced {synced} delegation(s)")

        # Get delegations
        delegations = self.delegation_manager.list_delegations()

        # Apply filters
        if status_filter != "All":
            delegations = [d for d in delegations if d.status.value == status_filter]
        if type_filter != "All":
            delegations = [d for d in delegations if d.delegation_type == type_filter]

        # Display
        if not delegations:
            st.info("📭 No delegations found")
        else:
            st.info(f"📋 Found {len(delegations)} delegation(s)")

            for delegation in delegations:
                status_emoji = {
                    DelegationStatus.PENDING: "⏳",
                    DelegationStatus.ASSIGNED: "📋",
                    DelegationStatus.IN_PROGRESS: "⚙️",
                    DelegationStatus.REVIEW: "👀",
                    DelegationStatus.COMPLETE: "[OK]",
                    DelegationStatus.FAILED: "[FAIL]"
                }.get(delegation.status, "❓")

                with st.expander(f"{status_emoji} {delegation.title} - {delegation.status.value.title()}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**Delegation ID:** `{delegation.delegation_id}`")
                        st.markdown(f"**Ticket ID:** `{delegation.task_id}`")
                        st.markdown(f"**Type:** {delegation.delegation_type}")

                    with col2:
                        st.markdown(f"**Created:** {delegation.created_at}")
                        st.markdown(f"**Updated:** {delegation.updated_at}")
                        if delegation.completed_at:
                            st.markdown(f"**Completed:** {delegation.completed_at}")

                    st.markdown(f"**Description:** {delegation.description}")

                    if delegation.result:
                        with st.expander("📊 Result"):
                            st.json(delegation.result)

    def _render_workflow_analytics(self):
        """Render workflow analytics dashboard"""
        st.subheader("📊 Workflow Analytics")

        # Get statistics
        tools = self.tool_repository.list_tools()
        delegations = self.delegation_manager.list_delegations()

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Tools", len(tools))

        with col2:
            validated_tools = [t for t in tools if t.status == ToolStatus.VALIDATED]
            st.metric("Validated Tools", len(validated_tools))

        with col3:
            total_usage = sum(t.usage_count for t in tools)
            st.metric("Total Executions", total_usage)

        with col4:
            active_delegations = [d for d in delegations if d.status in
                                [DelegationStatus.PENDING, DelegationStatus.ASSIGNED, DelegationStatus.IN_PROGRESS]]
            st.metric("Active Delegations", len(active_delegations))

        # Tool status distribution
        st.markdown("### Tool Status Distribution")

        status_counts = {}
        for tool in tools:
            status_counts[tool.status.value] = status_counts.get(tool.status.value, 0) + 1

        if status_counts:
            st.bar_chart(status_counts)
        else:
            st.info("No tools to display")

        # Most used tools
        st.markdown("### Most Used Tools")

        top_tools = sorted(tools, key=lambda t: t.usage_count, reverse=True)[:5]

        if top_tools:
            for i, tool in enumerate(top_tools, 1):
                st.markdown(f"{i}. **{tool.name}** - {tool.usage_count} uses")
        else:
            st.info("No usage data yet")

        # Recent delegations
        st.markdown("### Recent Delegations")

        recent_delegations = sorted(delegations, key=lambda d: d.created_at, reverse=True)[:5]

        if recent_delegations:
            for delegation in recent_delegations:
                status_emoji = {
                    DelegationStatus.PENDING: "⏳",
                    DelegationStatus.ASSIGNED: "📋",
                    DelegationStatus.IN_PROGRESS: "⚙️",
                    DelegationStatus.REVIEW: "👀",
                    DelegationStatus.COMPLETE: "[OK]",
                    DelegationStatus.FAILED: "[FAIL]"
                }.get(delegation.status, "❓")

                st.markdown(f"{status_emoji} **{delegation.title}** - {delegation.created_at}")
        else:
            st.info("No delegations yet")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_bubblelabs_maker_integration(
    crewai_manager: Optional[CrewAIIntegrationManager] = None
) -> BubbleLabsMakerUI:
    """
    Create and initialize BubbleLabs Maker integration.

    Args:
        crewai_manager: Optional CrewAI integration manager

    Returns:
        BubbleLabsMakerUI instance
    """
    ui = BubbleLabsMakerUI()

    if crewai_manager:
        ui.delegation_manager.crewai_manager = crewai_manager

    return ui


def get_integration_status() -> Dict[str, Any]:
    """Get status of all integrations"""
    return {
        "maker_available": MAKER_AVAILABLE,
        "crewai_integration_available": CREWAI_INTEGRATION_AVAILABLE,
        "roma_available": ROMA_AVAILABLE,
        "features": {
            "tool_creation": MAKER_AVAILABLE,
            "tool_execution": MAKER_AVAILABLE,
            "crewai_delegation": CREWAI_INTEGRATION_AVAILABLE,
            "workflow_tracking": True,
            "tool_repository": True,
            "progress_visualization": True
        }
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "BubbleLabsMakerUI",
    "ToolRepository",
    "CrewAIDelegationManager",
    "MakerWorkflowManager",

    # Data structures
    "ToolDefinition",
    "CrewAIDelegation",
    "ToolExecutionResult",
    "ToolStatus",
    "DelegationStatus",

    # Helper functions
    "create_bubblelabs_maker_integration",
    "get_integration_status"
]
