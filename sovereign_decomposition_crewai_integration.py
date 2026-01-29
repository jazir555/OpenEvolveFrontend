"""
Sovereign-Grade Decomposition Workflow - CrewAI Integration Module

This module provides the complete integration between OpenEvolve's Sovereign-Grade Decomposition
workflow and CrewAI's agentic framework. It replaces the AGPL-licensed Hephaestus integration
with MIT-licensed CrewAI.

The integration enables:
- Automatic creation of CrewAI tasks for each sub-problem in the decomposition plan
- Bidirectional synchronization of solution status between OpenEvolve and CrewAI workflows
- Mapping of OpenEvolve teams to CrewAI agents
- Real-time monitoring of both systems through unified interfaces
- Self-healing loops that trigger new work items when issues are discovered
- Full MDAP/MAKER zero-error workflow support

This replaces sovereign_decomposition_hephaestus_integration.py with MIT-licensed CrewAI.

License: MIT (replaces AGPL Hephaestus)
Author: OpenEvolve Team
Date: 2026-01-21
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

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
    SolutionAttempt,
    CritiqueReport,
    VerificationReport,
    StateManager,
)

# Import OpenEvolve structures
try:
    from workflow_structures import (
        WorkflowState as OpenEvolveWorkflowState,
        SubProblem as OpenEvolveSubProblem,
        SolutionAttempt as OpenEvolveSolutionAttempt,
        CritiqueReport as OpenEvolveCritiqueReport,
        VerificationReport as OpenEvolveVerificationReport,
        Team,
        GauntletDefinition,
        ModelConfig,
    )
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False

# Import MDAP and MAKER components
try:
    from mdap_engine import MDAPTask, MDAPStep, MDAPConfig, MDAPOrchestrator
    from maker_engine import MakerConfig, MakerEngine, MakerState
    from mdap_maker_complete import MAKEREngine, RecursiveMAKERSolver, MAKERRunMetrics
    MDAP_MAKER_AVAILABLE = True
except ImportError:
    MDAP_MAKER_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SGDStage(Enum):
    """Stages in the Sovereign-Grade Decomposition workflow"""
    CONTENT_ANALYSIS = "Content Analysis"
    AI_DECOMPOSITION = "AI-Assisted Decomposition"
    MANUAL_REVIEW = "Manual Review & Override"
    SUBPROBLEM_SOLVING = "Sub-Problem Solving Loop"
    REASSEMBLY = "Configurable Reassembly"
    FINAL_VERIFICATION = "Final Verification & Self-Healing Loop"
    KNOWLEDGE_EXTRACTION = "Knowledge Extraction & Learning"


class SovereignDecompositionCrewAIIntegration:
    """
    Complete integration manager for the Sovereign-Grade Decomposition Workflow with CrewAI.

    This implementation uses MIT-licensed CrewAI instead of AGPL-licensed Hephaestus.
    It maintains full functional parity with the original Hephaestus integration while
    providing better license compatibility.

    Key Features:
    - CrewAI workflow orchestration (replaces Hephaestus)
    - Full MDAP/MAKER zero-error workflow support
    - Real-time synchronization between OpenEvolve and CrewAI
    - Team-to-agent mapping
    - Self-healing loops
    - Comprehensive metrics and monitoring
    """

    def __init__(
        self,
        working_directory: str = ".",
        state_storage_dir: str = "./crewai_states",
        zero_error_config: Optional[ZeroErrorConfig] = None,
    ):
        """
        Initialize the Sovereign Decomposition - CrewAI integration.

        Args:
            working_directory: Root directory for workflows
            state_storage_dir: Directory for CrewAI state storage
            zero_error_config: Optional zero-error workflow config
        """
        self.working_directory = working_directory
        self.state_manager = StateManager(state_storage_dir)
        self.active_workflows: Dict[str, WorkflowState] = {}
        self.workflow_counter = 0

        # Initialize zero-error workflow
        self.zero_error_config = zero_error_config or create_zero_error_config()
        self.zero_error_workflow = create_zero_error_workflow(self.zero_error_config)

        if not OPENEVOLVE_AVAILABLE:
            logger.warning("OpenEvolve structures not available - some features limited")

        if not MDAP_MAKER_AVAILABLE:
            logger.warning("MDAP/MAKER components not available - zero-error workflow disabled")

        logger.info("Sovereign Decomposition - CrewAI Integration initialized (MIT-licensed)")

    # ========================================================================
    # WORKFLOW INITIALIZATION
    # ========================================================================

    def initialize_sovereign_workflow(
        self,
        workflow_state: OpenEvolveWorkflowState
    ) -> bool:
        """
        Initialize the complete Sovereign-Grade Decomposition workflow in CrewAI.

        This creates the main workflow epic and individual tasks for each sub-problem,
        mirroring the functionality of the original Hephaestus integration.

        Args:
            workflow_state: OpenEvolve workflow state

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Create CrewAI workflow
            crewai_workflow_id = self._create_crewai_workflow(
                problem_statement=workflow_state.problem_statement,
                workflow_type="sovereign_decomposition",
            )

            # Ensure all mappings are properly set
            if workflow_state.decomposition_plan:
                # Create CrewAI tasks for each sub-problem
                for sub_problem in workflow_state.decomposition_plan.sub_problems:
                    task_id = self._create_subproblem_task(workflow_state, sub_problem)
                    if task_id:
                        workflow_state.id_to_ticket_id_map[sub_problem.id] = task_id
                        workflow_state.ticket_id_to_subproblem_id_map[task_id] = sub_problem.id

                        # Set up dependencies in CrewAI
                        if sub_problem.dependencies:
                            self._create_task_dependencies(task_id, sub_problem.dependencies, workflow_state)

            # Update workflow state to reflect CrewAI integration
            workflow_state.current_stage = SGDStage.MANUAL_REVIEW.value
            workflow_state.status = "awaiting_user_input"

            logger.info(f"Initialized Sovereign-Grade workflow {workflow_state.workflow_id} in CrewAI")
            return True

        except Exception as e:
            logger.error(f"Error initializing sovereign workflow: {e}")
            return False

    def _create_crewai_workflow(
        self,
        problem_statement: str,
        workflow_type: str = "sovereign_decomposition",
    ) -> str:
        """Create a CrewAI workflow for tracking"""
        self.workflow_counter += 1
        workflow_id = f"SGD-{self.workflow_counter:06d}"

        # Create workflow state
        crewai_workflow_state = WorkflowState(
            workflow_id=workflow_id,
            problem_statement=problem_statement[:200],
            execution_method="traditional",
            phase=1,
            status="pending",
        )

        # Save state
        self.state_manager.save_state(workflow_id, crewai_workflow_state)
        self.active_workflows[workflow_id] = crewai_workflow_state

        logger.info(f"Created CrewAI workflow {workflow_id}")
        return workflow_id

    def _create_subproblem_task(
        self,
        workflow_state: OpenEvolveWorkflowState,
        sub_problem: OpenEvolveSubProblem,
    ) -> Optional[str]:
        """
        Create a CrewAI task for a specific sub-problem.

        Args:
            workflow_state: OpenEvolve workflow state
            sub_problem: Sub-problem to create task for

        Returns:
            CrewAI task ID or None if failed
        """
        try:
            # Build task description
            task_description = f"""
Sovereign-Grade Decomposition Sub-Problem

Problem ID: {sub_problem.id}
Dependencies: {', '.join(sub_problem.dependencies) or 'None'}
AI Suggested Complexity: {sub_problem.ai_suggested_complexity_score}/10
AI Suggested Evolution Mode: {sub_problem.ai_suggested_evolution_mode}
AI Suggested Evaluation Prompt: {sub_problem.ai_suggested_evaluation_prompt}

Original Sub-Problem Description:
{sub_problem.description}

OpenEvolve Workflow ID: {workflow_state.workflow_id}

Assigned Teams:
- Solver Team: {sub_problem.solver_team_name}
- Patcher Team: {sub_problem.patcher_team_name}
- Red Team Gauntlet: {sub_problem.red_team_gauntlet_name}
- Gold Team Gauntlet: {sub_problem.gold_team_gauntlet_name}

Evolution Parameters: {json.dumps(sub_problem.evolution_params, indent=2)}

This task represents one sub-problem in a sovereign-grade decomposition workflow.
            """.strip()

            # Create CrewAI task through zero-error workflow
            task_id = self.zero_error_workflow.create_task(
                description=task_description,
                task_type="decomposition",
                priority="high" if sub_problem.ai_suggested_complexity_score >= 8 else "medium",
                metadata={
                    "openevolve": True,
                    "sovereign_grade": True,
                    "subproblem_id": sub_problem.id,
                    "complexity": sub_problem.ai_suggested_complexity_score,
                    "evolution_mode": sub_problem.ai_suggested_evolution_mode,
                },
            )

            if task_id:
                logger.info(f"Created CrewAI task {task_id} for sub-problem {sub_problem.id}")

            return task_id

        except Exception as e:
            logger.error(f"Failed to create task for sub-problem {sub_problem.id}: {e}")
            return None

    def _create_task_dependencies(
        self,
        task_id: str,
        dependency_ids: List[str],
        workflow_state: OpenEvolveWorkflowState,
    ) -> bool:
        """Create dependencies between CrewAI tasks based on sub-problem dependencies"""
        try:
            # Map dependency IDs to task IDs
            blocking_task_ids = []
            for dep_id in dependency_ids:
                if dep_id in workflow_state.id_to_ticket_id_map:
                    blocking_task_ids.append(workflow_state.id_to_ticket_id_map[dep_id])

            if blocking_task_ids:
                # Set task dependencies in CrewAI
                success = self.zero_error_workflow.set_task_dependencies(
                    task_id=task_id,
                    blocking_task_ids=blocking_task_ids,
                )

                if success:
                    logger.info(f"Created dependencies for task {task_id}: blocked by {blocking_task_ids}")
                    return True

            return True  # Return True if no dependencies to set

        except Exception as e:
            logger.error(f"Failed to create task dependencies: {e}")
            return False

    # ========================================================================
    # SOLUTION SYNCHRONIZATION
    # ========================================================================

    def sync_solution_to_crewai_task(
        self,
        workflow_state: OpenEvolveWorkflowState,
        sub_problem_id: str,
        solution: OpenEvolveSolutionAttempt,
    ) -> bool:
        """
        Sync a solution from OpenEvolve to its corresponding CrewAI task.

        Args:
            workflow_state: OpenEvolve workflow state
            sub_problem_id: Sub-problem ID
            solution: Solution attempt to sync

        Returns:
            True if sync successful, False otherwise
        """
        try:
            task_id = workflow_state.id_to_ticket_id_map.get(sub_problem_id)
            if not task_id:
                logger.warning(f"No CrewAI task found for sub-problem {sub_problem_id}")
                # Create task on-demand if it doesn't exist
                sub_problem = next(
                    (sp for sp in workflow_state.decomposition_plan.sub_problems
                     if sp.id == sub_problem_id),
                    None
                )
                if sub_problem:
                    task_id = self._create_subproblem_task(workflow_state, sub_problem)
                    if task_id:
                        workflow_state.id_to_ticket_id_map[sub_problem_id] = task_id
                        workflow_state.ticket_id_to_subproblem_id_map[task_id] = sub_problem_id

                if not task_id:
                    logger.error(f"Could not create task for sub-problem {sub_problem_id}")
                    return False

            # Sync solution to CrewAI task
            success = self.zero_error_workflow.sync_solution(
                task_id=task_id,
                solution_content=solution.solution_content,
                confidence=solution.confidence_score,
                metadata={
                    "approach": solution.approach_description,
                    "subproblem_id": sub_problem_id,
                },
            )

            if success:
                logger.info(f"Synced solution to task {task_id} for sub-problem {sub_problem_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to sync solution to CrewAI task: {e}")
            return False

    def sync_critique_to_crewai_task(
        self,
        workflow_state: OpenEvolveWorkflowState,
        sub_problem_id: str,
        critique: OpenEvolveCritiqueReport,
    ) -> bool:
        """
        Sync a critique report from OpenEvolve to its corresponding CrewAI task.

        Args:
            workflow_state: OpenEvolve workflow state
            sub_problem_id: Sub-problem ID
            critique: Critique report to sync

        Returns:
            True if sync successful, False otherwise
        """
        try:
            task_id = workflow_state.id_to_ticket_id_map.get(sub_problem_id)
            if not task_id:
                logger.warning(f"No CrewAI task found for sub-problem {sub_problem_id}")
                return False

            # Sync critique to CrewAI task
            success = self.zero_error_workflow.sync_critique(
                task_id=task_id,
                critique_content=critique.summary,
                issues_found=len(critique.identified_issues) if critique.identified_issues else 0,
                approved=critique.approved,
                metadata={
                    "subproblem_id": sub_problem_id,
                    "improvements": critique.suggested_improvements,
                },
            )

            if success:
                logger.info(f"Synced critique to task {task_id} for sub-problem {sub_problem_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to sync critique to CrewAI task: {e}")
            return False

    def sync_verification_to_crewai_task(
        self,
        workflow_state: OpenEvolveWorkflowState,
        sub_problem_id: str,
        verification: OpenEvolveVerificationReport,
    ) -> bool:
        """
        Sync a verification report from OpenEvolve to its corresponding CrewAI task.

        Args:
            workflow_state: OpenEvolve workflow state
            sub_problem_id: Sub-problem ID
            verification: Verification report to sync

        Returns:
            True if sync successful, False otherwise
        """
        try:
            task_id = workflow_state.id_to_ticket_id_map.get(sub_problem_id)
            if not task_id:
                logger.warning(f"No CrewAI task found for sub-problem {sub_problem_id}")
                return False

            # Sync verification to CrewAI task
            success = self.zero_error_workflow.sync_verification(
                task_id=task_id,
                verification_passed=verification.verification_passed,
                correctness_score=verification.correctness_score,
                completeness_score=verification.completeness_score,
                metadata={
                    "subproblem_id": sub_problem_id,
                    "test_results": verification.test_results,
                },
            )

            if success:
                logger.info(f"Synced verification to task {task_id} for sub-problem {sub_problem_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to sync verification to CrewAI task: {e}")
            return False

    def sync_solution_status_to_crewai_task(
        self,
        workflow_state: OpenEvolveWorkflowState,
        sub_problem_id: str,
        new_status: str,
        solution_content: Optional[str] = None,
    ) -> bool:
        """
        Sync the status of a sub-problem from OpenEvolve to its corresponding CrewAI task.

        Args:
            workflow_state: OpenEvolve workflow state
            sub_problem_id: Sub-problem ID
            new_status: New status to sync
            solution_content: Optional solution content

        Returns:
            True if sync successful, False otherwise
        """
        try:
            task_id = workflow_state.id_to_ticket_id_map.get(sub_problem_id)
            if not task_id:
                logger.warning(f"No CrewAI task found for sub-problem {sub_problem_id}")
                # Try to create task on-demand if it doesn't exist
                sub_problem = next(
                    (sp for sp in workflow_state.decomposition_plan.sub_problems
                     if sp.id == sub_problem_id),
                    None
                )
                if sub_problem:
                    task_id = self._create_subproblem_task(workflow_state, sub_problem)
                    if task_id:
                        workflow_state.id_to_ticket_id_map[sub_problem_id] = task_id
                        workflow_state.ticket_id_to_subproblem_id_map[task_id] = sub_problem_id

                if not task_id:
                    logger.error(f"Could not create task for sub-problem {sub_problem_id}")
                    return False

            # Update task status in CrewAI
            success = self.zero_error_workflow.update_task_status(
                task_id=task_id,
                status=new_status,
                content=solution_content,
            )

            if success:
                logger.info(f"Synced status {new_status} to task {task_id} for sub-problem {sub_problem_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to sync solution status to CrewAI task: {e}")
            return False

    # ========================================================================
    # WORKFLOW MANAGEMENT
    # ========================================================================

    def close_workflow_in_crewai(
        self,
        workflow_state: OpenEvolveWorkflowState,
    ) -> bool:
        """
        Close the workflow in CrewAI when the OpenEvolve workflow completes.

        Args:
            workflow_state: OpenEvolve workflow state

        Returns:
            True if close successful, False otherwise
        """
        try:
            # Close all associated tasks
            for task_id in workflow_state.id_to_ticket_id_map.values():
                self.zero_error_workflow.close_task(task_id)

            logger.info(f"Closed workflow in CrewAI for {workflow_state.workflow_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to close workflow in CrewAI: {e}")
            return False

    # ========================================================================
    # TEAM-TO-AGENT MAPPING
    # ========================================================================

    def map_openevolve_team_to_crewai_agent(
        self,
        team: Team,
        sub_problem: OpenEvolveSubProblem,
    ) -> str:
        """
        Map an OpenEvolve team to a CrewAI agent based on team role.

        Args:
            team: OpenEvolve team
            sub_problem: Sub-problem context

        Returns:
            CrewAI agent type string
        """
        team_mapping = {
            "Blue-Solvers": "ImplementationAgent",
            "Blue-Patchers": "FixAgent",
            "Blue-Assemblers": "IntegrationAgent",
            "Blue-Optimizers": "OptimizationAgent",
            "Red-Security": "SecurityValidationAgent",
            "Red-Logic": "LogicValidationAgent",
            "Red-EdgeCase": "EdgeCaseAgent",
            "Gold-Accuracy": "AccuracyAgent",
            "Gold-Completeness": "CompletenessAgent",
            "Gold-Efficiency": "PerformanceAgent"
        }

        # Default mapping based on team role
        if team.role == "Blue":
            if team.sub_role == "Patcher":
                return "FixAgent"
            elif team.sub_role == "Assembler":
                return "IntegrationAgent"
            else:
                return "ImplementationAgent"
        elif team.role == "Red":
            return "ValidationAgent"
        elif team.role == "Gold":
            return "QualityAssuranceAgent"
        else:
            return "GenericAgent"

    # ========================================================================
    # METRICS AND MONITORING
    # ========================================================================

    def get_openevolve_metrics_from_crewai_agents(
        self,
        workflow_id: str,
    ) -> Dict[str, Any]:
        """
        Extract OpenEvolve metrics from CrewAI agent performance.

        Args:
            workflow_id: CrewAI workflow ID

        Returns:
            Dictionary of metrics
        """
        try:
            # Get workflow metrics from CrewAI
            crewai_metrics = self.zero_error_workflow.get_workflow_metrics(workflow_id)

            metrics = {
                "agent_performance": crewai_metrics.get("agent_performance", {}),
                "task_completion_rate": crewai_metrics.get("task_completion_rate", 0),
                "average_resolution_time": crewai_metrics.get("average_resolution_time", 0),
                "quality_scores": crewai_metrics.get("quality_scores", []),
            }

            logger.info(f"Extracted metrics from CrewAI for workflow {workflow_id}")
            return metrics

        except Exception as e:
            logger.error(f"Failed to extract metrics from CrewAI: {e}")
            return {}

    def update_openevolve_with_crewai_feedback(
        self,
        workflow_state: OpenEvolveWorkflowState,
        feedback_metrics: Dict[str, Any],
    ):
        """
        Update OpenEvolve workflow state with feedback from CrewAI agents.

        Args:
            workflow_state: OpenEvolve workflow state
            feedback_metrics: Metrics from CrewAI
        """
        try:
            # Update performance metrics in the workflow state
            if 'openevolve_metrics' not in workflow_state.__dict__:
                workflow_state.__dict__['openevolve_metrics'] = {}

            # Add CrewAI-derived metrics
            workflow_state.openevolve_metrics.update({
                'crewai_feedback_time': time.time(),
                'agent_performance_metrics': feedback_metrics.get('agent_performance', {}),
                'task_completion_rate': feedback_metrics.get('task_completion_rate', 0),
                'average_resolution_time': feedback_metrics.get('average_resolution_time', 0)
            })

            logger.info(f"Updated OpenEvolve workflow with CrewAI feedback for {workflow_state.workflow_id}")

        except Exception as e:
            logger.error(f"Failed to update OpenEvolve with CrewAI feedback: {e}")

    # ========================================================================
    # SELF-HEALING
    # ========================================================================

    def trigger_self_healing_from_agent_discoveries(
        self,
        workflow_state: OpenEvolveWorkflowState,
    ) -> bool:
        """
        Trigger self-healing in OpenEvolve based on discoveries made by CrewAI agents.

        Args:
            workflow_state: OpenEvolve workflow state

        Returns:
            True if self-healing triggered, False otherwise
        """
        try:
            # Check for issues discovered by CrewAI agents
            workflow_id = workflow_state.crewai_workflow_id if hasattr(workflow_state, 'crewai_workflow_id') else None
            if not workflow_id:
                return False

            issues = self.zero_error_workflow.get_discovered_issues(workflow_id)

            if issues:
                # Create new sub-problems in OpenEvolve for each issue discovered
                for issue in issues:
                    sub_problem_id = issue.get('subproblem_id')
                    if sub_problem_id:
                        # Mark the original sub-problem as needing rework
                        workflow_state.rejected_sub_problems[sub_problem_id] = {
                            'timestamp': time.time(),
                            'reason': f"Issue discovered by CrewAI agent: {issue.get('status')}",
                            'details': issue.get('description', '')
                        }

                logger.info(f"Triggered self-healing for {len(issues)} issues discovered by CrewAI agents")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to trigger self-healing from agent discoveries: {e}")
            return False

    # ========================================================================
    # MDAP INTEGRATION
    # ========================================================================

    def initialize_mdap_subproblem_solve(
        self,
        workflow_state: OpenEvolveWorkflowState,
        sub_problem: OpenEvolveSubProblem,
        team: Team,
        mdap_config: Optional[MDAPConfig] = None,
    ) -> Optional[str]:
        """
        Initialize MDAP-based solving for a sub-problem with CrewAI tracking.

        Args:
            workflow_state: OpenEvolve workflow state
            sub_problem: Sub-problem to solve
            team: Team to use
            mdap_config: Optional MDAP config

        Returns:
            CrewAI task ID or None if failed
        """
        if not MDAP_MAKER_AVAILABLE:
            logger.warning("MDAP not available for sub-problem solve")
            return None

        try:
            # Create MDAP steps from sub-problem
            mdap_steps = [
                MDAPStep(
                    step_id=f"{sub_problem.id}-decompose",
                    prompt=f"Decompose: {sub_problem.description}",
                    task_type="decomposition",
                    priority=1,
                    metadata={"sub_problem_id": sub_problem.id}
                ),
                MDAPStep(
                    step_id=f"{sub_problem.id}-solve",
                    prompt=f"Solve: {sub_problem.description}",
                    task_type="solve",
                    priority=2,
                    metadata={"sub_problem_id": sub_problem.id}
                ),
                MDAPStep(
                    step_id=f"{sub_problem.id}-verify",
                    prompt=f"Verify solution for: {sub_problem.description}",
                    task_type="verification",
                    priority=3,
                    metadata={"sub_problem_id": sub_problem.id}
                )
            ]

            # Create MDAP task
            mdap_task = MDAPTask(
                task_id=f"mdap-{sub_problem.id}",
                description=f"MDAP solve for sub-problem: {sub_problem.description}",
                steps=mdap_steps,
                max_retries=sub_problem.ai_suggested_complexity_score // 3 + 1,
                target_success_rate=0.95,
                metadata={
                    "sub_problem_id": sub_problem.id,
                    "workflow_id": workflow_state.workflow_id,
                    "complexity_score": sub_problem.ai_suggested_complexity_score
                }
            )

            # Sync MDAP task to CrewAI
            mdap_task_id = self.zero_error_workflow.sync_mdap_task(
                mdap_task=mdap_task,
                workflow_id=workflow_state.crewai_workflow_id if hasattr(workflow_state, 'crewai_workflow_id') else None,
            )

            if mdap_task_id:
                logger.info(f"Initialized MDAP for sub-problem {sub_problem.id} with task {mdap_task_id}")
                # Store MDAP task reference in workflow state
                if not hasattr(workflow_state, 'mdap_tasks'):
                    workflow_state.mdap_tasks = {}
                workflow_state.mdap_tasks[sub_problem.id] = mdap_task

                return mdap_task_id

            return None

        except Exception as e:
            logger.error(f"Failed to initialize MDAP for sub-problem {sub_problem.id}: {e}")
            return None

    # ========================================================================
    # MAKER INTEGRATION
    # ========================================================================

    def initialize_maker_subproblem_solve(
        self,
        workflow_state: OpenEvolveWorkflowState,
        sub_problem: OpenEvolveSubProblem,
        team: Team,
        maker_config: Optional[MakerConfig] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Initialize MAKER-based solving for a sub-problem with CrewAI tracking.

        Args:
            workflow_state: OpenEvolve workflow state
            sub_problem: Sub-problem to solve
            team: Team to use
            maker_config: Optional MAKER config
            initial_state: Optional initial state

        Returns:
            CrewAI task ID or None if failed
        """
        if not MDAP_MAKER_AVAILABLE:
            logger.warning("MAKER not available for sub-problem solve")
            return None

        try:
            # Create MAKER config
            config = maker_config or MakerConfig(
                k_min=2,
                k_max=max(5, sub_problem.ai_suggested_complexity_score),
                max_votes_per_step=30 + sub_problem.ai_suggested_complexity_score * 5,
                max_steps=100 + sub_problem.ai_suggested_complexity_score * 10
            )

            # Create initial state
            state = initial_state or {
                "sub_problem_id": sub_problem.id,
                "description": sub_problem.description,
                "complexity": sub_problem.ai_suggested_complexity_score,
                "evolution_mode": sub_problem.ai_suggested_evolution_mode,
                "status": "initialized"
            }

            # Sync MAKER run to CrewAI
            maker_task_id = self.zero_error_workflow.sync_maker_run(
                run_id=f"maker-{sub_problem.id}",
                initial_state=state,
                config=config,
                workflow_id=workflow_state.crewai_workflow_id if hasattr(workflow_state, 'crewai_workflow_id') else None,
            )

            if maker_task_id:
                logger.info(f"Initialized MAKER for sub-problem {sub_problem.id} with task {maker_task_id}")

                # Store MAKER config and state in workflow state
                if not hasattr(workflow_state, 'maker_configs'):
                    workflow_state.maker_configs = {}
                if not hasattr(workflow_state, 'maker_initial_states'):
                    workflow_state.maker_initial_states = {}

                workflow_state.maker_configs[sub_problem.id] = config
                workflow_state.maker_initial_states[sub_problem.id] = state

                return maker_task_id

            return None

        except Exception as e:
            logger.error(f"Failed to initialize MAKER for sub-problem {sub_problem.id}: {e}")
            return None


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================_sgd_crewai_integration = None

def get_sgd_crewai_integration(
    working_directory: str = ".",
    state_storage_dir: str = "./crewai_states",
) -> Optional[SovereignDecompositionCrewAIIntegration]:
    """
    Get the singleton instance of the Sovereign Decomposition - CrewAI integration.

    Args:
        working_directory: Working directory for workflows
        state_storage_dir: Directory for state storage

    Returns:
        SovereignDecompositionCrewAIIntegration instance
    """
    global _sgd_crewai_integration
    if _sgd_crewai_integration is None:
        _sgd_crewai_integration = SovereignDecompositionCrewAIIntegration(
            working_directory=working_directory,
            state_storage_dir=state_storage_dir,
        )
    return _sgd_crewai_integration


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SovereignDecompositionCrewAIIntegration",
    "SGDStage",
    "get_sgd_crewai_integration",
]


# Module initialization
if __name__ == "__main__":
    print("Sovereign Decomposition - CrewAI Integration Module (MIT-licensed)")
    print(f"OpenEvolve Available: {OPENEVOLVE_AVAILABLE}")
    print(f"MDAP/MAKER Available: {MDAP_MAKER_AVAILABLE}")
    print("\nClasses:")
    print("  - SovereignDecompositionCrewAIIntegration")
    print("\nEnums:")
    print("  - SGDStage")
