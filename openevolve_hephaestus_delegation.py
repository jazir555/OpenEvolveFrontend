"""
OpenEvolve-Hephaestus Delegation Integration

This module provides PROPER integration between OpenEvolve and Hephaestus by DELEGATING
workflow orchestration to Hephaestus instead of just syncing tickets.

Architecture:
- OpenEvolve defines the problem decomposition logic and specialized solving strategies
- Hephaestus manages the workflow orchestration, agent spawning, and task coordination
- OpenEvolve stages map to Hephaestus phases
- Hephaestus agents call back into OpenEvolve for specialized processing

Usage:
    from openevolve_hephaestus_delegation import OpenEvolveHephaestusDelegator

    delegator = OpenEvolveHephaestusDelegator(
        hephaestus_config=config,
        working_directory="/path/to/project"
    )

    # Start a decomposition workflow
    workflow_id = await delegator.start_decomposition_workflow(
        problem_statement="Solve the traveling salesman problem",
        launch_params={}
    )

    # Monitor progress
    status = await delegator.get_workflow_status(workflow_id)
"""

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime
import requests

# Add Hephaestus SDK to path
HEPHAEUSTUS_SDK_PATH = Path(__file__).parent / "Hephaestus"
if str(HEPHAEUSTUS_SDK_PATH) not in sys.path:
    sys.path.insert(0, str(HEPHAEUSTUS_SDK_PATH))

from src.sdk import HephaestusSDK
from src.sdk.models import (
    Phase,
    WorkflowConfig,
    WorkflowDefinition,
    LaunchTemplate,
    LaunchParameter,
    TaskStatus,
    WorkflowExecution,
)
from src.sdk.config import HephaestusConfig

# Import OpenEvolve structures
from workflow_structures import (
    DecompositionPlan,
    SubProblem,
    SolutionAttempt,
    CritiqueReport,
    VerificationReport,
    WorkflowState,
)

# =============================================================================
# PHASE DEFINITIONS - Mapping OpenEvolve stages to Hephaestus phases
# =============================================================================

# Phase 1: Content Analysis & Problem Decomposition
PHASE_1_DECOMPOSITION = Phase(
    id=1,
    name="problem_decomposition",
    description="Analyze the problem statement and decompose into solvable sub-problems",
    done_definitions=[
        "Problem statement fully analyzed",
        "Domain and complexity identified",
        "Sub-problems identified and documented",
        "Dependencies mapped between sub-problems",
        "Teams and gauntlets assigned to each sub-problem",
        "Task marked as done"
    ],
    working_directory=".",
    additional_notes="""
MISSION: Decompose the complex problem into solvable components

STEP 1: Analyze the problem statement
- Identify the problem domain
- Estimate complexity and resource requirements
- Identify key constraints and requirements

STEP 2: Decompose into sub-problems
- Break down the problem into 5-15 solvable sub-problems
- Each sub-problem should be independently solvable
- Map dependencies between sub-problems
- Estimate complexity for each sub-problem (1-10)

STEP 3: Assign resources
- Assign Blue Teams to solve each sub-problem
- Assign Red Team gauntlets for critique
- Assign Gold Team gauntlets for verification
- Specify any specialized requirements

STEP 4: Create follow-up tasks
- For each sub-problem, create a Phase 2 task
- Use create_task() for each sub-problem
- Mark your task as done when complete
"""
)

# Phase 2: Sub-Problem Solving (Blue Team)
PHASE_2_SOLVING = Phase(
    id=2,
    name="sub_problem_solving",
    description="Generate solutions for assigned sub-problems",
    done_definitions=[
        "Solution approach designed",
        "Implementation completed",
        "Solution documented",
        "Phase 3 critique task created",
        "Task marked as done"
    ],
    working_directory=".",
    additional_notes="""
MISSION: Solve ONE assigned sub-problem completely

You are assigned to ONE specific sub-problem. Do not work on other sub-problems.

STEP 1: Understand the sub-problem
- Read the sub-problem description carefully
- Identify constraints and requirements
- Review any dependencies

STEP 2: Design solution approach
- Choose appropriate algorithms/methods
- Consider edge cases
- Plan implementation steps

STEP 3: Implement solution
- Write clean, well-documented code
- Follow best practices
- Handle errors appropriately

STEP 4: Create critique task
- Create a Phase 3 task for Red Team critique
- Include solution details in the task description
- Mark your task as done
"""
)

# Phase 3: Solution Critique (Red Team)
PHASE_3_CRITIQUE = Phase(
    id=3,
    name="solution_critique",
    description="Critique solutions using adversarial testing",
    done_definitions=[
        "Security vulnerabilities identified",
        "Edge cases tested",
        "Performance issues documented",
        "Improvement suggestions provided",
        "Phase 4 verification task created if approved",
        "Task marked as done"
    ],
    working_directory=".",
    additional_notes="""
MISSION: Find flaws in the solution through adversarial analysis

STEP 1: Understand the solution
- Review the implementation
- Understand the approach taken

STEP 2: Attack the solution
- Test edge cases
- Attempt to break the solution
- Identify security vulnerabilities
- Test performance limits

STEP 3: Document findings
- List all issues found
- Severity: Critical/High/Medium/Low
- Provide reproduction steps
- Suggest improvements

STEP 4: Make approval decision
- If solution is acceptable: Create Phase 4 verification task
- If solution needs work: Do NOT create Phase 4 task (solution goes back to Phase 2)
- Mark your task as done
"""
)

# Phase 4: Solution Verification (Gold Team)
PHASE_4_VERIFICATION = Phase(
    id=4,
    name="solution_verification",
    description="Verify solution correctness and completeness",
    done_definitions=[
        "Correctness verified",
        "Completeness checked",
        "Quality standards met",
        "Documentation reviewed",
        "Task marked as done"
    ],
    working_directory=".",
    additional_notes="""
MISSION: Verify the solution meets all requirements

STEP 1: Verify correctness
- Test the solution thoroughly
- Verify outputs match expected results
- Check mathematical correctness if applicable

STEP 2: Verify completeness
- All requirements met
- All edge cases handled
- Documentation complete

STEP 3: Quality assessment
- Code quality standards met
- Performance acceptable
- Security adequate

STEP 4: Final decision
- If approved: Solution is complete
- If issues found: Document issues for rework
- Mark your task as done
"""
)

# Phase 5: Solution Reassembly & Integration
PHASE_5_REASSEMBLY = Phase(
    id=5,
    name="solution_reassembly",
    description="Integrate verified sub-problem solutions into final solution",
    done_definitions=[
        "All solutions integrated",
        "Interface compatibility verified",
        "Dependency conflicts resolved",
        "Integrated solution tested",
        "Phase 6 verification task created",
        "Task marked as done"
    ],
    working_directory=".",
    additional_notes="""
MISSION: Assemble the final solution from verified components

STEP 1: Gather all verified solutions
- Collect outputs from all sub-problems
- Verify all are approved

STEP 2: Resolve interfaces
- Map data flows between components
- Resolve any type mismatches
- Handle dependencies correctly

STEP 3: Integration
- Combine components into final solution
- Ensure proper ordering
- Handle inter-component communication

STEP 4: Test integration
- Test end-to-end functionality
- Verify no regressions
- Check performance

STEP 5: Create final verification task
- Create Phase 6 task for final verification
- Mark your task as done
"""
)

# Phase 6: Final Verification & Self-Healing
PHASE_6_FINAL = Phase(
    id=6,
    name="final_verification",
    description="Final verification, testing, and self-healing of complete solution",
    done_definitions=[
        "Complete solution tested",
        "All requirements verified",
        "Quality standards met",
        "Documentation complete",
        "Knowledge artifacts extracted",
        "Task marked as done"
    ],
    working_directory=".",
    additional_notes="""
MISSION: Final verification of the complete solution

STEP 1: Comprehensive testing
- Unit tests pass
- Integration tests pass
- System tests pass
- Performance tests pass
- Security tests pass

STEP 2: Requirements verification
- All original requirements met
- Edge cases handled
- Constraints satisfied

STEP 3: Quality assessment
- Code quality reviewed
- Documentation complete
- Maintainability assessed

STEP 4: Self-healing
- If issues found: Create appropriate tasks to fix
- Allow for automatic fixes where possible
- Re-test after fixes

STEP 5: Knowledge extraction
- Document solution patterns
- Record performance metrics
- Save lessons learned

STEP 6: Complete workflow
- Submit final result
- Mark task as done
"""
)

# All phases for OpenEvolve decomposition workflow
OPENEVOLVE_PHASES = [
    PHASE_1_DECOMPOSITION,
    PHASE_2_SOLVING,
    PHASE_3_CRITIQUE,
    PHASE_4_VERIFICATION,
    PHASE_5_REASSEMBLY,
    PHASE_6_FINAL,
]

# =============================================================================
# WORKFLOW CONFIGURATION
# =============================================================================

OPENEVOLVE_WORKFLOW_CONFIG = WorkflowConfig(
    has_result=True,
    result_criteria="All sub-problems solved, verified, integrated, and final solution tested",
    on_result_found="stop_all",
    enable_tickets=True,
    board_config={
        "columns": [
            {"id": "pending", "name": "Pending", "limit": None},
            {"id": "in_progress", "name": "In Progress", "limit": None},
            {"id": "critique", "name": "Under Critique", "limit": None},
            {"id": "verified", "name": "Verified", "limit": None},
            {"id": "done", "name": "Done", "limit": None},
            {"id": "failed", "name": "Failed", "limit": None},
        ]
    }
)

# =============================================================================
# LAUNCH TEMPLATE
# =============================================================================

OPENEVOLVE_LAUNCH_TEMPLATE = LaunchTemplate(
    parameters=[
        LaunchParameter(
            name="problem_statement",
            label="Problem Statement",
            type="textarea",
            required=True,
            description="Describe the problem you want to solve in detail"
        ),
        LaunchParameter(
            name="problem_domain",
            label="Problem Domain",
            type="dropdown",
            required=False,
            options=["General", "Software Development", "Mathematics", "Data Science", "System Design", "Research", "Other"],
            description="What domain does this problem belong to?"
        ),
        LaunchParameter(
            name="complexity_level",
            label="Expected Complexity",
            type="dropdown",
            required=False,
            options=["Low (1-3)", "Medium (4-7)", "High (8-10)"],
            description="How complex is this problem?"
        ),
        LaunchParameter(
            name="max_sub_problems",
            label="Maximum Sub-Problems",
            type="number",
            required=False,
            description="Maximum number of sub-problems to create (default: 15)"
        ),
        LaunchParameter(
            name="auto_approve_decomposition",
            label="Auto-Approve Decomposition",
            type="boolean",
            required=False,
            description="Automatically approve the AI-generated decomposition plan"
        ),
    ],
    phase_1_task_prompt="""OpenEvolve Problem Decomposition Workflow

**Problem Domain:** {problem_domain}
**Expected Complexity:** {complexity_level}
**Maximum Sub-Problems:** {max_sub_problems}
**Auto-Approve:** {auto_approve_decomposition}

## Problem Statement
{problem_statement}

Your task:
1. Analyze this problem statement comprehensively
2. Decompose it into solvable sub-problems (max: {max_sub_problems})
3. Map dependencies between sub-problems
4. Assign appropriate teams and gauntlets
5. Create Phase 2 tasks for each sub-problem

Use the OpenEvolve MCP tools to create tasks and tickets for tracking.
"""
)

# =============================================================================
# WORKFLOW DEFINITION
# =============================================================================

OPENEVOLVE_WORKFLOW_DEFINITION = WorkflowDefinition(
    id="openevolve-decomposition",
    name="OpenEvolve Decomposition Workflow",
    phases=OPENEVOLVE_PHASES,
    config=OPENEVOLVE_WORKFLOW_CONFIG,
    description="Sovereign-grade problem decomposition with multi-agent teams and adversarial validation",
    launch_template=OPENEVOLVE_LAUNCH_TEMPLATE,
)

# =============================================================================
# DELEGATION CLIENT
# =============================================================================

@dataclass
class WorkflowMetrics:
    """Metrics for a running workflow"""
    workflow_id: str
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    in_progress_tasks: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "running"

    @property
    def duration_seconds(self) -> float:
        """Get workflow duration in seconds"""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def completion_percentage(self) -> float:
        """Get completion percentage"""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100


class OpenEvolveHephaestusDelegator:
    """
    Main delegator class that integrates OpenEvolve with Hephaestus.

    This class delegates workflow orchestration to Hephaestus while providing
    OpenEvolve's specialized decomposition and solving logic.

    Architecture:
    - HephaestusSDK manages workflow lifecycle, agent spawning, task coordination
    - OpenEvolve provides domain-specific logic for decomposition and solving
    - Callback system allows Hephaestus agents to invoke OpenEvolve functions
    """

    def __init__(
        self,
        hephaestus_config: Optional[HephaestusConfig] = None,
        working_directory: str = ".",
        auto_start: bool = False,
        **config_kwargs,
    ):
        """
        Initialize the OpenEvolve-Hephaestus delegator.

        Args:
            hephaestus_config: Pre-configured HephaestusConfig (optional)
            working_directory: Working directory for the workflow
            auto_start: Automatically start Hephaestus services
            **config_kwargs: Additional Hephaestus configuration parameters
        """
        self.working_directory = working_directory
        self.running = False
        self.sdk: Optional[HephaestusSDK] = None
        self.active_workflows: Dict[str, WorkflowMetrics] = {}

        # Create or use provided config
        if hephaestus_config:
            self.config = hephaestus_config
            # Override with kwargs
            for key, value in config_kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        else:
            # Default config with working directory
            default_config = {
                "working_directory": working_directory,
                "main_repo_path": working_directory,
                "project_root": working_directory,
            }
            default_config.update(config_kwargs)
            self.config = HephaestusConfig(**default_config)

        # Initialize SDK (but don't start yet)
        self._initialize_sdk()

        # Auto-start if requested
        if auto_start:
            self.start()

    def _initialize_sdk(self):
        """Initialize the Hephaestus SDK with OpenEvolve workflow definition"""
        self.sdk = HephaestusSDK(
            workflow_definitions=[OPENEVOLVE_WORKFLOW_DEFINITION],
            config=self.config,
            auto_start=False,
        )

    def start(self, timeout: int = 30, enable_tui: bool = False) -> bool:
        """
        Start Hephaestus services.

        Args:
            timeout: Maximum seconds to wait for services to become healthy
            enable_tui: Enable TUI mode (default: False, headless)

        Returns:
            True if all services are healthy

        Raises:
            Exception: If services don't start within timeout
        """
        if self.running:
            print("[OpenEvolve] Hephaestus is already running")
            return True

        print("[OpenEvolve] Starting Hephaestus services...")
        success = self.sdk.start(timeout=timeout, enable_tui=enable_tui)

        if success:
            self.running = True
            print("[OpenEvolve] ✓ Hephaestus services ready")
            print(f"[OpenEvolve] API endpoint: {self.config.api_base_url}")
        else:
            print("[OpenEvolve] ✗ Failed to start Hephaestus services")

        return success

    async def start_decomposition_workflow(
        self,
        problem_statement: str,
        problem_domain: str = "General",
        complexity_level: str = "Medium (4-7)",
        max_sub_problems: int = 15,
        auto_approve_decomposition: bool = False,
    ) -> str:
        """
        Start a new OpenEvolve decomposition workflow in Hephaestus.

        This method creates a new workflow execution in Hephaestus, which will:
        1. Create Phase 1 task for problem decomposition
        2. Spawn agents to work on tasks
        3. Coordinate task execution across phases
        4. Track progress and status

        Args:
            problem_statement: The problem to solve
            problem_domain: Domain of the problem
            complexity_level: Expected complexity
            max_sub_problems: Maximum sub-problems to create
            auto_approve_decomposition: Auto-approve decomposition plan

        Returns:
            workflow_id: The ID of the created workflow

        Raises:
            Exception: If workflow creation fails
        """
        if not self.running:
            raise RuntimeError("Hephaestus is not running. Call start() first.")

        # Prepare launch parameters
        launch_params = {
            "problem_statement": problem_statement,
            "problem_domain": problem_domain,
            "complexity_level": complexity_level,
            "max_sub_problems": str(max_sub_problems),
            "auto_approve_decomposition": "true" if auto_approve_decomposition else "false",
        }

        # Start workflow
        print(f"[OpenEvolve] Starting decomposition workflow...")
        workflow_id = self.sdk.start_workflow(
            definition_id="openevolve-decomposition",
            description=f"Decompose: {problem_statement[:100]}...",
            working_directory=self.working_directory,
            launch_params=launch_params,
        )

        # Track workflow
        self.active_workflows[workflow_id] = WorkflowMetrics(
            workflow_id=workflow_id,
            status="active",
        )

        print(f"[OpenEvolve] ✓ Workflow started: {workflow_id}")
        return workflow_id

    async def get_workflow_status(self, workflow_id: str) -> WorkflowExecution:
        """
        Get the status of a workflow execution.

        Args:
            workflow_id: The workflow execution ID

        Returns:
            WorkflowExecution object with status details

        Raises:
            Exception: If workflow not found
        """
        if not self.running:
            raise RuntimeError("Hephaestus is not running. Call start() first.")

        execution = self.sdk.get_workflow_execution(workflow_id)
        if execution is None:
            raise ValueError(f"Workflow {workflow_id} not found")

        # Update metrics
        if workflow_id in self.active_workflows:
            metrics = self.active_workflows[workflow_id]
            metrics.total_tasks = execution.total_tasks
            metrics.completed_tasks = execution.done_tasks
            metrics.failed_tasks = execution.failed_tasks
            metrics.in_progress_tasks = execution.active_tasks
            metrics.status = execution.status

            if execution.status in ["completed", "failed", "paused"]:
                metrics.end_time = time.time()

        return execution

    async def list_workflows(self, status: str = "all") -> List[WorkflowExecution]:
        """
        List all workflow executions.

        Args:
            status: Filter by status ("all", "active", "completed", "paused", "failed")

        Returns:
            List of WorkflowExecution objects
        """
        if not self.running:
            raise RuntimeError("Hephaestus is not running. Call start() first.")

        return self.sdk.list_workflow_executions(status=status)

    async def create_sub_problem_task(
        self,
        workflow_id: str,
        sub_problem: SubProblem,
        phase_id: int = 2,
        priority: str = "medium",
    ) -> str:
        """
        Create a task for solving a specific sub-problem.

        This is called by Phase 1 (decomposition) to create Phase 2 tasks
        for each sub-problem.

        Args:
            workflow_id: The workflow execution ID
            sub_problem: The sub-problem to solve
            phase_id: Which phase this task belongs to (default: 2 for solving)
            priority: Task priority

        Returns:
            task_id: The ID of the created task
        """
        if not self.running:
            raise RuntimeError("Hephaestus is not running. Call start() first.")

        # Build task description from sub-problem
        description = f"""Solve Sub-Problem: {sub_problem.id}

**Description:** {sub_problem.description}

**Complexity Score:** {sub_problem.ai_suggested_complexity_score}/10

**Dependencies:** {', '.join(sub_problem.dependencies) if sub_problem.dependencies else 'None'}

**Requirements:**
{chr(10).join(f'- {req}' for req in sub_problem.acceptance_criteria) if sub_problem.acceptance_criteria else '- Solve the problem completely'}

**Constraints:**
{chr(10).join(f'- {c}' for c in sub_problem.specific_constraints) if sub_problem.specific_constraints else '- None specified'}
"""

        # Create ticket first (for tracking)
        # Note: In production, you'd call the ticket creation endpoint
        ticket_id = f"ticket-{sub_problem.id}"

        # Create task in Hephaestus
        task_id = self.sdk.create_task_in_workflow(
            workflow_id=workflow_id,
            description=description,
            phase_id=phase_id,
            priority=priority,
            ticket_id=ticket_id,
        )

        print(f"[OpenEvolve] Created task {task_id} for sub-problem {sub_problem.id}")
        return task_id

    async def get_tasks(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        phase_id: Optional[int] = None,
    ) -> List[TaskStatus]:
        """
        Get tasks with optional filtering.

        Args:
            workflow_id: Filter by workflow (optional, for future use)
            status: Filter by status ("pending", "in_progress", "done", "failed")
            phase_id: Filter by phase ID

        Returns:
            List of TaskStatus objects
        """
        if not self.running:
            raise RuntimeError("Hephaestus is not running. Call start() first.")

        return self.sdk.get_tasks(status=status, phase_id=phase_id)

    async def monitor_workflow(
        self,
        workflow_id: str,
        callback: Optional[callable] = None,
        poll_interval: int = 5,
    ) -> WorkflowExecution:
        """
        Monitor a workflow execution until completion.

        Args:
            workflow_id: The workflow execution ID
            callback: Optional callback function called with status updates
            poll_interval: Seconds between status checks

        Returns:
            Final WorkflowExecution object
        """
        print(f"[OpenEvolve] Monitoring workflow {workflow_id}...")

        while True:
            execution = await self.get_workflow_status(workflow_id)

            # Call callback if provided
            if callback:
                await callback(execution)

            # Print status
            metrics = self.active_workflows.get(workflow_id)
            if metrics:
                print(f"[OpenEvolve] Status: {execution.status} | "
                      f"Tasks: {execution.done_tasks}/{execution.total_tasks} | "
                      f"Agents: {execution.active_agents} | "
                      f"Progress: {metrics.completion_percentage:.1f}%")

            # Check if complete
            if execution.status in ["completed", "failed"]:
                print(f"[OpenEvolve] Workflow {workflow_id} {execution.status}")
                return execution

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    def get_metrics(self, workflow_id: str) -> Optional[WorkflowMetrics]:
        """
        Get metrics for a workflow.

        Args:
            workflow_id: The workflow execution ID

        Returns:
            WorkflowMetrics object or None if not found
        """
        return self.active_workflows.get(workflow_id)

    def is_healthy(self) -> Dict[str, bool]:
        """
        Check health of Hephaestus services.

        Returns:
            Dictionary with component health status
        """
        if not self.running or not self.sdk:
            return {
                "running": False,
                "overall": False,
            }

        health = self.sdk.is_healthy()
        health["running"] = True
        return health

    def shutdown(self, graceful: bool = True, timeout: int = 10):
        """
        Shutdown Hephaestus services.

        Args:
            graceful: Use graceful shutdown vs force kill
            timeout: Maximum seconds to wait for graceful shutdown
        """
        if not self.running:
            return

        print("[OpenEvolve] Shutting down Hephaestus...")

        if self.sdk:
            self.sdk.shutdown(graceful=graceful, timeout=timeout)

        self.running = False
        print("[OpenEvolve] ✓ Shutdown complete")

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
        return False


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_openevolve_delegator(
    working_directory: str = ".",
    database_path: str = "./openevolve_hephaestus.db",
    qdrant_url: str = "http://localhost:6333",
    mcp_port: int = 8000,
    llm_provider: str = "anthropic",
    auto_start: bool = False,
    **kwargs,
) -> OpenEvolveHephaestusDelegator:
    """
    Factory function to create an OpenEvolve-Hephaestus delegator.

    Args:
        working_directory: Working directory for workflows
        database_path: Path to Hephaestus SQLite database
        qdrant_url: URL for Qdrant vector store
        mcp_port: Port for Hephaestus MCP server
        llm_provider: LLM provider to use
        auto_start: Automatically start Hephaestus services
        **kwargs: Additional configuration parameters

    Returns:
        OpenEvolveHephaestusDelegator instance
    """
    config = HephaestusConfig(
        database_path=database_path,
        qdrant_url=qdrant_url,
        mcp_port=mcp_port,
        llm_provider=llm_provider,
        working_directory=working_directory,
        main_repo_path=working_directory,
        project_root=working_directory,
        **kwargs,
    )

    return OpenEvolveHephaestusDelegator(
        hephaestus_config=config,
        working_directory=working_directory,
        auto_start=auto_start,
    )


# =============================================================================
# MAIN EXAMPLE
# =============================================================================

async def main():
    """Example usage of the OpenEvolve-Hephaestus delegator"""

    # Create delegator
    delegator = create_openevolve_delegator(
        working_directory="/path/to/project",
        auto_start=True,
    )

    try:
        # Start a decomposition workflow
        workflow_id = await delegator.start_decomposition_workflow(
            problem_statement="Design a scalable URL shortening service that can handle 10M requests per second",
            problem_domain="Software Development",
            complexity_level="High (8-10)",
            max_sub_problems=10,
        )

        # Monitor until completion
        execution = await delegator.monitor_workflow(
            workflow_id,
            poll_interval=10,
        )

        # Get final metrics
        metrics = delegator.get_metrics(workflow_id)
        print(f"\n[OpenEvolve] Final Metrics:")
        print(f"  Duration: {metrics.duration_seconds:.1f}s")
        print(f"  Tasks: {metrics.completed_tasks}/{metrics.total_tasks}")
        print(f"  Status: {metrics.status}")

    finally:
        # Shutdown
        delegator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
