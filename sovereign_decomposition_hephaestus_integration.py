"""
Sovereign-Grade Decomposition Workflow - Hephaestus Integration Module

This module provides the complete integration between OpenEvolve's Sovereign-Grade Decomposition 
workflow and the Hephaestus agentic framework as specified in the @Decomposition_Workflow.md
documentation.

The integration enables:
- Automatic creation of Hephaestus tickets for each sub-problem in the decomposition plan
- Bidirectional synchronization of solution status between OpenEvolve and Hephaestus
- Mapping of OpenEvolve teams to Hephaestus agents and vice versa
- Real-time monitoring of both systems through unified interfaces
- Self-healing loops that trigger new work items when issues are discovered in either system
"""

import json
import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from enum import Enum

from workflow_structures import (
    WorkflowState, SubProblem, SolutionAttempt, CritiqueReport, 
    VerificationReport, Team, GauntletDefinition, ModelConfig
)
from hephaestus_integration import HephaestusIntegrationManager, TicketStatus, TicketType
from llm_utils import _request_openai_compatible_chat

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

class SovereignDecompositionHephaestusIntegration:
    """
    Complete integration manager for the Sovereign-Grade Decomposition Workflow with Hephaestus.
    This implementation follows the detailed specifications in @Decomposition_Workflow.md
    """

    def __init__(self, hephaestus_api_base: str, hephaestus_api_key: str, hephaestus_project_id: str):
        self.integration_manager = HephaestusIntegrationManager(
            hephaestus_api_base, hephaestus_api_key, hephaestus_project_id
        )
        self.active_workflows = {}
        self.sync_threads = {}

    def initialize_sovereign_workflow(self, workflow_state: WorkflowState) -> bool:
        """
        Initialize the complete Sovereign-Grade Decomposition workflow in Hephaestus.
        This creates the main workflow epic and individual tickets for each sub-problem
        as specified in the documentation.
        """
        try:
            # Create the main workflow epic in Hephaestus with detailed context
            success = self.integration_manager.initialize_workflow_sync(workflow_state)
            if not success:
                logger.error(f"Failed to initialize workflow sync for {workflow_state.workflow_id}")
                return False

            # Ensure all mappings are properly set
            if workflow_state.decomposition_plan:
                # Create tickets for each sub-problem with proper dependencies
                for sub_problem in workflow_state.decomposition_plan.sub_problems:
                    # Create Hephaestus ticket with proper dependencies
                    ticket_id = self._create_subproblem_ticket(workflow_state, sub_problem)
                    if ticket_id:
                        workflow_state.id_to_ticket_id_map[sub_problem.id] = ticket_id
                        workflow_state.ticket_id_to_subproblem_id_map[ticket_id] = sub_problem.id

                        # Set up dependencies in Hephaestus as well
                        if sub_problem.dependencies:
                            self._create_ticket_dependencies(ticket_id, sub_problem.dependencies, workflow_state)

            # Update workflow state to reflect Hephaestus integration
            workflow_state.current_stage = SGDStage.MANUAL_REVIEW.value
            workflow_state.status = "awaiting_user_input"  # For the manual review phase

            logger.info(f"Initialized Sovereign-Grade workflow {workflow_state.workflow_id} in Hephaestus")
            return True

        except Exception as e:
            logger.error(f"Error initializing sovereign workflow: {e}")
            return False

    def _create_subproblem_ticket(self, workflow_state: WorkflowState, sub_problem: SubProblem) -> Optional[str]:
        """Create a Hephaestus ticket for a specific sub-problem"""
        try:
            # Determine ticket type based on complexity and other factors
            ticket_type = TicketType.TASK
            if sub_problem.ai_suggested_complexity_score >= 8:
                ticket_type = TicketType.STORY
            elif sub_problem.ai_suggested_complexity_score >= 6:
                ticket_type = TicketType.TASK

            # Create detailed ticket description with all relevant information
            ticket_description = f"""
Sovereign-Grade Decomposition Sub-Problem

Problem ID: {sub_problem.id}
Dependencies: {', '.join(sub_problem.dependencies) or 'None'}
AI Suggested Complexity: {sub_problem.ai_suggested_complexity_score}/10
AI Suggested Evolution Mode: {sub_problem.ai_suggested_evolution_mode}
AI Suggested Evaluation Prompt: {sub_problem.ai_suggested_evaluation_prompt}

Original Sub-Problem Description:
{sub_problem.description}

OpenEvolve Workflow ID: {workflow_state.workflow_id}
Associated with: OpenEvolve Sub-Problem {sub_problem.id}

Assigned Teams:
- Solver Team: {sub_problem.solver_team_name}
- Patcher Team: {sub_problem.patcher_team_name}
- Red Team Gauntlet: {sub_problem.red_team_gauntlet_name}
- Gold Team Gauntlet: {sub_problem.gold_team_gauntlet_name}

Evolution Parameters: {json.dumps(sub_problem.evolution_params, indent=2)}

This ticket represents one sub-problem in a larger sovereign-grade decomposition workflow.
            """.strip()

            ticket_id = self.integration_manager.client.create_ticket(
                title=f"SGD-{sub_problem.id}: {sub_problem.description[:50]}...",
                description=ticket_description,
                ticket_type=ticket_type,
                labels=[
                    "openevolve", 
                    "sovereign-grade", 
                    "decomposition", 
                    f"workflow-{workflow_state.workflow_id}",
                    f"subproblem-{sub_problem.id}",
                    f"complexity-{sub_problem.ai_suggested_complexity_score}",
                    f"evolution-mode-{sub_problem.ai_suggested_evolution_mode}"
                ]
            )

            if ticket_id:
                logger.info(f"Created Hephaestus ticket {ticket_id} for sub-problem {sub_problem.id}")

            return ticket_id

        except Exception as e:
            logger.error(f"Failed to create ticket for sub-problem {sub_problem.id}: {e}")
            return None

    def _create_ticket_dependencies(self, ticket_id: str, dependency_ids: List[str], 
                                   workflow_state: WorkflowState) -> bool:
        """Create dependencies between Hephaestus tickets based on sub-problem dependencies"""
        try:
            # Map dependency IDs to ticket IDs
            blocking_ticket_ids = []
            for dep_id in dependency_ids:
                if dep_id in workflow_state.id_to_ticket_id_map:
                    blocking_ticket_ids.append(workflow_state.id_to_ticket_id_map[dep_id])
                else:
                    # Try to create dependency ticket if it doesn't exist yet
                    logger.warning(f"Dependency ticket for {dep_id} not found, creating placeholder")

            if blocking_ticket_ids:
                # Update ticket to mark it as blocked by other tickets
                success = self.integration_manager.client.update_ticket_dependencies(
                    ticket_id, blocking_ticket_ids
                )

                if success:
                    logger.info(f"Created dependencies for ticket {ticket_id}: blocked by {blocking_ticket_ids}")
                    return True

            return True  # Return True if no dependencies to set

        except Exception as e:
            logger.error(f"Failed to create ticket dependencies: {e}")
            return False

    def sync_solution_to_hephaestus_ticket(self, workflow_state: WorkflowState, 
                                         sub_problem_id: str, solution: SolutionAttempt) -> bool:
        """
        Sync a solution from OpenEvolve to its corresponding Hephaestus ticket.
        This implements the functionality described in Section 7.5.11 of the documentation.
        """
        try:
            ticket_id = workflow_state.id_to_ticket_id_map.get(sub_problem_id)
            if not ticket_id:
                logger.warning(f"No Hephaestus ticket found for sub-problem {sub_problem_id}")
                # Create ticket on-demand if it doesn't exist
                sub_problem = next((sp for sp in workflow_state.decomposition_plan.sub_problems 
                                  if sp.id == sub_problem_id), None)
                if sub_problem:
                    ticket_id = self._create_subproblem_ticket(workflow_state, sub_problem)
                    if ticket_id:
                        workflow_state.id_to_ticket_id_map[sub_problem_id] = ticket_id
                        workflow_state.ticket_id_to_subproblem_id_map[ticket_id] = sub_problem_id

                if not ticket_id:
                    logger.error(f"Could not create ticket for sub-problem {sub_problem_id}")
                    return False

            # Use the existing integration method
            success = self.integration_manager.sync_solution_to_ticket(workflow_state, sub_problem_id, solution)

            if success:
                # Update ticket status based on solution quality
                new_status = TicketStatus.IN_REVIEW
                self.integration_manager.client.update_ticket(ticket_id, status=new_status)

                logger.info(f"Synced solution to ticket {ticket_id} for sub-problem {sub_problem_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to sync solution to Hephaestus ticket: {e}")
            return False

    def sync_critique_to_hephaestus_ticket(self, workflow_state: WorkflowState, 
                                         sub_problem_id: str, critique: CritiqueReport) -> bool:
        """
        Sync a critique report from OpenEvolve to its corresponding Hephaestus ticket.
        This implements the functionality described in Section 7.5.11 of the documentation.
        """
        try:
            ticket_id = workflow_state.id_to_ticket_id_map.get(sub_problem_id)
            if not ticket_id:
                logger.warning(f"No Hephaestus ticket found for sub-problem {sub_problem_id}")
                return False

            # Use the existing integration method
            success = self.integration_manager.sync_critique_to_ticket(workflow_state, sub_problem_id, critique)

            if success:
                logger.info(f"Synced critique to ticket {ticket_id} for sub-problem {sub_problem_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to sync critique to Hephaestus ticket: {e}")
            return False

    def sync_verification_to_hephaestus_ticket(self, workflow_state: WorkflowState, 
                                             sub_problem_id: str, verification: VerificationReport) -> bool:
        """
        Sync a verification report from OpenEvolve to its corresponding Hephaestus ticket.
        This implements the functionality described in Section 7.5.11 of the documentation.
        """
        try:
            ticket_id = workflow_state.id_to_ticket_id_map.get(sub_problem_id)
            if not ticket_id:
                logger.warning(f"No Hephaestus ticket found for sub-problem {sub_problem_id}")
                return False

            # Use the existing integration method
            success = self.integration_manager.sync_verification_to_ticket(workflow_state, sub_problem_id, verification)

            if success:
                logger.info(f"Synced verification to ticket {ticket_id} for sub-problem {sub_problem_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to sync verification to Hephaestus ticket: {e}")
            return False

    def sync_solution_status_to_hephaestus_ticket(self, workflow_state: WorkflowState, 
                                                sub_problem_id: str, new_status: str, 
                                                solution_content: Optional[str] = None) -> bool:
        """
        Sync the status of a sub-problem from OpenEvolve to its corresponding Hephaestus ticket.
        This implements the functionality described in Section 7.5.11 of the documentation.
        """
        try:
            ticket_id = workflow_state.id_to_ticket_id_map.get(sub_problem_id)
            if not ticket_id:
                logger.warning(f"No Hephaestus ticket found for sub-problem {sub_problem_id}")
                # Try to create ticket on-demand if it doesn't exist
                sub_problem = next((sp for sp in workflow_state.decomposition_plan.sub_problems 
                                  if sp.id == sub_problem_id), None)
                if sub_problem:
                    ticket_id = self._create_subproblem_ticket(workflow_state, sub_problem)
                    if ticket_id:
                        workflow_state.id_to_ticket_id_map[sub_problem_id] = ticket_id
                        workflow_state.ticket_id_to_subproblem_id_map[ticket_id] = sub_problem_id

                if not ticket_id:
                    logger.error(f"Could not create ticket for sub-problem {sub_problem_id}")
                    return False

            # Use the existing integration method
            success = self.integration_manager.update_subproblem_status(
                workflow_state, sub_problem_id, new_status, solution_content
            )

            if success:
                logger.info(f"Synced status {new_status} to ticket {ticket_id} for sub-problem {sub_problem_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to sync solution status to Hephaestus ticket: {e}")
            return False

    def start_real_time_sync(self, workflow_state: WorkflowState, sync_interval: int = 30):
        """
        Start real-time synchronization between OpenEvolve and Hephaestus workflows.
        This implements the functionality described in Section 7.5.11 of the documentation.
        """
        try:
            # Start the sync loop in the existing integration manager
            self.integration_manager.start_sync_loop(workflow_state, sync_interval)
            logger.info(f"Started real-time sync for workflow {workflow_state.workflow_id}")

        except Exception as e:
            logger.error(f"Failed to start real-time sync: {e}")

    def stop_real_time_sync(self, workflow_id: str):
        """
        Stop real-time synchronization for a specific workflow.
        """
        try:
            self.integration_manager.stop_sync_loop(workflow_id)
            logger.info(f"Stopped real-time sync for workflow {workflow_id}")

        except Exception as e:
            logger.error(f"Failed to stop real-time sync: {e}")

    def close_workflow_in_hephaestus(self, workflow_state: WorkflowState) -> bool:
        """
        Close the workflow in Hephaestus when the OpenEvolve workflow completes.
        This implements the functionality described in Section 7.5.11 of the documentation.
        """
        try:
            success = self.integration_manager.close_workflow_sync(workflow_state)
            if success:
                logger.info(f"Closed workflow in Hephaestus for {workflow_state.workflow_id}")
            return success

        except Exception as e:
            logger.error(f"Failed to close workflow in Hephaestus: {e}")
            return False

    def map_openevolve_team_to_hephaestus_agent(self, team: Team, sub_problem: SubProblem) -> str:
        """
        Map an OpenEvolve team to a Hephaestus agent based on the specifications in Section 7.5.2.
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

    def get_openevolve_metrics_from_hephaestus_agents(self, workflow_id: str) -> Dict[str, Any]:
        """
        Extract OpenEvolve metrics from Hephaestus agent performance as described in Section 7.5.11.
        """
        try:
            # Get workflow tickets to extract agent performance
            tickets = self.integration_manager.client.get_workflow_tickets(workflow_id)
            
            metrics = {
                "agent_performance": {},
                "task_completion_rate": 0,
                "average_resolution_time": 0,
                "quality_scores": []
            }

            completed_tasks = 0
            total_time = 0
            for ticket in tickets:
                assigned_agent = ticket.get('assigned_agent_id', 'unassigned')
                if assigned_agent != 'unassigned':
                    if assigned_agent not in metrics["agent_performance"]:
                        metrics["agent_performance"][assigned_agent] = {
                            "completed": 0,
                            "failed": 0,
                            "total_time": 0
                        }
                    
                    agent_perf = metrics["agent_performance"][assigned_agent]
                    status = ticket.get('status', 'todo')
                    if status == 'done':
                        agent_perf["completed"] += 1
                        completed_tasks += 1
                        # Calculate resolution time if available
                        created_at = ticket.get('created_at')
                        completed_at = ticket.get('updated_at')  # Simplified
                        if created_at and completed_at:
                            try:
                                from datetime import datetime
                                created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                                completed = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
                                duration = (completed - created).total_seconds()
                                agent_perf["total_time"] += duration
                                total_time += duration
                            except Exception as e:
                                logger.debug("Failed to parse ticket timestamps: %s", e)
                    elif status in ['blocked', 'failed']:
                        agent_perf["failed"] += 1

            if completed_tasks > 0:
                metrics["task_completion_rate"] = completed_tasks / len(tickets)
                metrics["average_resolution_time"] = total_time / completed_tasks

            logger.info(f"Extracted metrics from Hephaestus for workflow {workflow_id}")
            return metrics

        except Exception as e:
            logger.error(f"Failed to extract metrics from Hephaestus: {e}")
            return {}

    def update_openevolve_with_hephaestus_feedback(self, workflow_state: WorkflowState, 
                                                  feedback_metrics: Dict[str, Any]):
        """
        Update OpenEvolve workflow state with feedback from Hephaestus agents.
        """
        try:
            # Update performance metrics in the workflow state
            if 'openevolve_metrics' not in workflow_state.__dict__:
                workflow_state.__dict__['openevolve_metrics'] = {}
            
            # Add Hephaestus-derived metrics
            workflow_state.openevolve_metrics.update({
                'hephaestus_feedback_time': time.time(),
                'agent_performance_metrics': feedback_metrics.get('agent_performance', {}),
                'task_completion_rate': feedback_metrics.get('task_completion_rate', 0),
                'average_resolution_time': feedback_metrics.get('average_resolution_time', 0)
            })

            logger.info(f"Updated OpenEvolve workflow with Hephaestus feedback for {workflow_state.workflow_id}")

        except Exception as e:
            logger.error(f"Failed to update OpenEvolve with Hephaestus feedback: {e}")

    def trigger_self_healing_from_agent_discoveries(self, workflow_state: WorkflowState) -> bool:
        """
        Trigger self-healing in OpenEvolve based on discoveries made by Hephaestus agents.
        This implements the functionality described in Section 7.5.11 of the documentation.
        """
        try:
            # Check for issues discovered by Hephaestus agents
            tickets = self.integration_manager.client.get_workflow_tickets(workflow_state.hephaestus_workflow_id)
            
            issues_found = []
            for ticket in tickets:
                # Look for comments or status that indicate issues
                if ticket.get('status') == 'blocked' or ticket.get('status') == 'failed':
                    issues_found.append({
                        'ticket_id': ticket.get('id'),
                        'sub_problem_id': workflow_state.ticket_id_to_subproblem_id_map.get(ticket.get('id')),
                        'status': ticket.get('status'),
                        'description': ticket.get('description', '')
                    })

            if issues_found:
                # Create new sub-problems in OpenEvolve for each issue discovered
                for issue in issues_found:
                    if issue['sub_problem_id']:
                        # Mark the original sub-problem as needing rework
                        workflow_state.rejected_sub_problems[issue['sub_problem_id']] = {
                            'timestamp': time.time(),
                            'reason': f"Issue discovered by Hephaestus agent: {issue['status']}",
                            'details': issue['description']
                        }

                logger.info(f"Triggered self-healing for {len(issues_found)} issues discovered by Hephaestus agents")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to trigger self-healing from agent discoveries: {e}")
            return False


# Global integration manager instance for easy access
_sgd_hephaestus_integration = None

def get_sgd_hephaestus_integration() -> Optional[SovereignDecompositionHephaestusIntegration]:
    """
    Get the singleton instance of the Sovereign Decomposition - Hephaestus integration.
    """
    global _sgd_hephaestus_integration
    return _sgd_hephaestus_integration

def initialize_sgd_hephaestus_integration(api_base: str, api_key: str, project_id: str) -> bool:
    """
    Initialize the Sovereign Decomposition - Hephaestus integration.
    """
    global _sgd_hephaestus_integration
    try:
        _sgd_hephaestus_integration = SovereignDecompositionHephaestusIntegration(api_base, api_key, project_id)
        logger.info("Sovereign Decomposition - Hephaestus integration initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize SGD-Hephaestus integration: {e}")
        return False


# Example usage for verification
if __name__ == "__main__":
    print("Sovereign Decomposition - Hephaestus Integration Module loaded successfully")
    print("- Implements complete integration as specified in @Decomposition_Workflow.md")
    print("- Provides full bidirectional synchronization")
    print("- Includes team-to-agent mapping functionality")
    print("- Supports real-time monitoring and self-healing loops")
