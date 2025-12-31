"""
Hephaestus Integration for OpenEvolve
Provides production-ready integration with the Hephaestus project management system.
This module handles synchronization between OpenEvolve workflow and Hephaestus tickets.
"""
import json
import time
import requests
import hashlib
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from enum import Enum

from workflow_structures import WorkflowState, SubProblem, SolutionAttempt, CritiqueReport, VerificationReport
from llm_utils import _request_openai_compatible_chat

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TicketStatus(Enum):
    """Status values for Hephaestus tickets"""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    BLOCKED = "blocked"
    DONE = "done"

class TicketType(Enum):
    """Types of tickets in Hephaestus system"""
    TASK = "task"
    BUG = "bug"
    STORY = "story"
    EPIC = "epic"

class HephaestusClient:
    """Client for interacting with the Hephaestus API"""
    
    def __init__(self, api_base: str, api_key: str, project_id: str):
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.project_id = project_id
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def create_ticket(self, title: str, description: str, ticket_type: TicketType = TicketType.TASK, 
                     assignee: Optional[str] = None, labels: Optional[List[str]] = None) -> Optional[str]:
        """Create a new ticket in Hephaestus"""
        try:
            payload = {
                'title': title,
                'description': description,
                'type': ticket_type.value,
                'status': TicketStatus.TODO.value,
                'project_id': self.project_id,
                'assignee': assignee,
                'labels': labels or [],
                'created_at': datetime.utcnow().isoformat()
            }
            
            response = self.session.post(f"{self.api_base}/tickets", json=payload)
            response.raise_for_status()
            
            ticket_data = response.json()
            ticket_id = ticket_data.get('id')
            
            logger.info(f"Created Hephaestus ticket {ticket_id} for: {title}")
            return ticket_id
            
        except requests.RequestException as e:
            logger.error(f"Failed to create ticket: {e}")
            return None
    
    def update_ticket(self, ticket_id: str, status: Optional[TicketStatus] = None, 
                     assignee: Optional[str] = None, description: Optional[str] = None) -> bool:
        """Update an existing ticket in Hephaestus"""
        try:
            payload = {}
            if status:
                payload['status'] = status.value
            if assignee:
                payload['assignee'] = assignee
            if description:
                payload['description'] = description
            
            response = self.session.patch(f"{self.api_base}/tickets/{ticket_id}", json=payload)
            response.raise_for_status()
            
            logger.info(f"Updated Hephaestus ticket {ticket_id} with status: {status.value if status else 'unchanged'}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to update ticket {ticket_id}: {e}")
            return False
    
    def get_ticket(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a ticket from Hephaestus"""
        try:
            response = self.session.get(f"{self.api_base}/tickets/{ticket_id}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get ticket {ticket_id}: {e}")
            return None
    
    def get_tickets_by_label(self, label: str) -> List[Dict[str, Any]]:
        """Get tickets with a specific label from Hephaestus"""
        try:
            response = self.session.get(f"{self.api_base}/tickets", params={'label': label})
            response.raise_for_status()
            return response.json().get('tickets', [])
        except requests.RequestException as e:
            logger.error(f"Failed to get tickets by label {label}: {e}")
            return []

class HephaestusWorkflowSync:
    """Manages synchronization between OpenEvolve workflows and Hephaestus tickets"""
    
    def __init__(self, hephaestus_client: HephaestusClient):
        self.client = hephaestus_client
        self.sync_lock = threading.Lock()
        
    def create_workflow_in_hephaestus(self, workflow_state: WorkflowState) -> Optional[str]:
        """Create the main workflow ticket in Hephaestus to represent the entire OpenEvolve workflow"""
        try:
            # Create an epic ticket for the entire workflow
            workflow_description = f"""
OpenEvolve Sovereign-Grade Decomposition Workflow
Problem Statement: {workflow_state.problem_statement}
Workflow ID: {workflow_state.workflow_id}
Started: {datetime.fromtimestamp(workflow_state.start_time).isoformat()}

This epic represents the entire workflow and contains sub-tasks for each sub-problem.
            """.strip()
            
            epic_ticket_id = self.client.create_ticket(
                title=f"OpenEvolve Workflow: {workflow_state.workflow_id}",
                description=workflow_description,
                ticket_type=TicketType.EPIC,
                labels=["openevolve", "workflow", f"workflow-{workflow_state.workflow_id}"]
            )
            
            if epic_ticket_id:
                logger.info(f"Created main workflow epic ticket {epic_ticket_id}")
                return epic_ticket_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create workflow in Hephaestus: {e}")
            return None
    
    def create_subproblem_tickets(self, workflow_id: str, sub_problems: List[SubProblem], 
                                 workflow_epic_id: Optional[str] = None) -> Dict[str, str]:
        """Create individual tickets for each sub-problem in Hephaestus"""
        id_to_ticket_map = {}
        
        for sub_problem in sub_problems:
            try:
                # Create ticket description
                ticket_description = f"""
Sub-Problem ID: {sub_problem.id}
Dependencies: {', '.join(sub_problem.dependencies) or 'None'}
AI Suggested Complexity: {sub_problem.ai_suggested_complexity_score}/10
AI Suggested Evolution Mode: {sub_problem.ai_suggested_evolution_mode}
AI Suggested Evaluation Prompt: {sub_problem.ai_suggested_evaluation_prompt}

Original Sub-Problem Description:
{sub_problem.description}
                """.strip()
                
                # Determine ticket type based on complexity
                ticket_type = TicketType.TASK
                if sub_problem.ai_suggested_complexity_score >= 8:
                    ticket_type = TicketType.STORY
                elif sub_problem.ai_suggested_complexity_score >= 6:
                    ticket_type = TicketType.TASK
                
                ticket_id = self.client.create_ticket(
                    title=f"Sub-Problem {sub_problem.id}: {sub_problem.description[:50]}...",
                    description=ticket_description,
                    ticket_type=ticket_type,
                    assignee=sub_problem.assigned_team, # Assign the ticket
                    labels=["openevolve", "sub-problem", f"workflow-{workflow_id}", f"subproblem-{sub_problem.id}"] + 
                           ([f"team-{sub_problem.assigned_team}"] if sub_problem.assigned_team else [])
                )
                
                if ticket_id:
                    id_to_ticket_map[sub_problem.id] = ticket_id
                    logger.info(f"Created sub-problem ticket {ticket_id} for sub-problem {sub_problem.id}")
                    
                    # Link to workflow epic if provided
                    if workflow_epic_id:
                        # In a real system, we'd link these tickets to the epic
                        # For now, just log the relationship
                        logger.info(f"Linked sub-problem ticket {ticket_id} to workflow epic {workflow_epic_id}")
                
            except Exception as e:
                logger.error(f"Failed to create ticket for sub-problem {sub_problem.id}: {e}")
        
        return id_to_ticket_map
    
    def sync_subproblem_status(self, sub_problem_id: str, ticket_id: str, 
                              status: str, solution_content: Optional[str] = None) -> bool:
        """Sync the status of a sub-problem with its corresponding ticket in Hephaestus"""
        try:
            # Map OpenEvolve status to Hephaestus status
            hephaestus_status = self._map_status_to_hephaestus(status)
            
            # If solution content is provided, append it to the ticket description
            description_update = None
            if solution_content:
                description_update = f"Updated with solution content at {datetime.utcnow().isoformat()}:\n{solution_content[:200]}..."
            
            success = self.client.update_ticket(
                ticket_id=ticket_id, 
                status=hephaestus_status,
                description=description_update
            )
            
            if success:
                logger.info(f"Synced status for sub-problem {sub_problem_id} (ticket {ticket_id}) to {hephaestus_status.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to sync sub-problem status for {sub_problem_id}: {e}")
            return False
    
    def _map_status_to_hephaestus(self, openevolve_status: str) -> TicketStatus:
        """Map OpenEvolve sub-problem status to Hephaestus ticket status"""
        status_mapping = {
            'pending': TicketStatus.TODO,
            'in_progress': TicketStatus.IN_PROGRESS,
            'solved': TicketStatus.DONE,
            'failed': TicketStatus.BLOCKED,
            'requires_rework': TicketStatus.IN_REVIEW
        }
        return status_mapping.get(openevolve_status.lower(), TicketStatus.IN_PROGRESS)
    
    def get_sync_metrics(self, workflow_state: WorkflowState) -> Dict[str, Any]:
        """Get metrics about the synchronization status"""
        total_subproblems = len(workflow_state.decomposition_plan.sub_problems) if workflow_state.decomposition_plan else 0
        synced_subproblems = len([k for k, v in workflow_state.id_to_ticket_id_map.items() if v])
        
        return {
            'workflow_id': workflow_state.workflow_id,
            'total_subproblems': total_subproblems,
            'synced_subproblems': synced_subproblems,
            'sync_percentage': (synced_subproblems / total_subproblems * 100) if total_subproblems > 0 else 0,
            'hephaestus_workflow_id': workflow_state.hephaestus_workflow_id
        }

class HephaestusIntegrationManager:
    """Main integration manager that coordinates the flow between OpenEvolve and Hephaestus"""
    
    def __init__(self, api_base: str, api_key: str, project_id: str):
        self.client = HephaestusClient(api_base, api_key, project_id)
        self.sync_manager = HephaestusWorkflowSync(self.client)
        self.active_synchronizations = {}
    
    def initialize_workflow_sync(self, workflow_state: WorkflowState) -> bool:
        """Initialize synchronization for a new workflow"""
        try:
            # Create the main workflow epic in Hephaestus
            workflow_epic_id = self.sync_manager.create_workflow_in_hephaestus(workflow_state)
            if not workflow_epic_id:
                logger.error(f"Failed to create workflow epic for {workflow_state.workflow_id}")
                return False
            
            # Update workflow state with Hephaestus workflow ID
            workflow_state.hephaestus_workflow_id = workflow_epic_id
            
            # Create tickets for all sub-problems
            if workflow_state.decomposition_plan:
                id_to_ticket_map = self.sync_manager.create_subproblem_tickets(
                    workflow_state.workflow_id,
                    workflow_state.decomposition_plan.sub_problems,
                    workflow_epic_id
                )
                
                # Update the workflow state mappings
                workflow_state.id_to_ticket_id_map.update(id_to_ticket_map)
                
                # Create reverse mapping
                for sub_problem_id, ticket_id in id_to_ticket_map.items():
                    workflow_state.ticket_id_to_subproblem_id_map[ticket_id] = sub_problem_id
            
            logger.info(f"Initialized Hephaestus sync for workflow {workflow_state.workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize workflow sync: {e}")
            return False
    
    def update_subproblem_status(self, workflow_state: WorkflowState, sub_problem_id: str, 
                                new_status: str, solution_content: Optional[str] = None) -> bool:
        """Update the status of a sub-problem in Hephaestus"""
        try:
            ticket_id = workflow_state.id_to_ticket_id_map.get(sub_problem_id)
            if not ticket_id:
                logger.warning(f"No Hephaestus ticket found for sub-problem {sub_problem_id}")
                # Try to create ticket on-demand if it doesn't exist
                sub_problem = None
                if workflow_state.decomposition_plan:
                    sub_problem = next((sp for sp in workflow_state.decomposition_plan.sub_problems 
                                      if sp.id == sub_problem_id), None)
                
                if sub_problem:
                    ticket_id = self.client.create_ticket(
                        title=f"Sub-Problem {sub_problem.id}: {sub_problem.description[:50]}...",
                        description=f"Auto-created ticket for sub-problem {sub_problem.id}",
                        labels=["openevolve", "sub-problem", f"workflow-{workflow_state.workflow_id}"]
                    )
                    if ticket_id:
                        workflow_state.id_to_ticket_id_map[sub_problem_id] = ticket_id
                        workflow_state.ticket_id_to_subproblem_id_map[ticket_id] = sub_problem_id
                
                if not ticket_id:
                    logger.error(f"Could not create ticket for sub-problem {sub_problem_id}")
                    return False
            
            success = self.sync_manager.sync_subproblem_status(
                sub_problem_id, 
                ticket_id, 
                new_status, 
                solution_content
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update sub-problem status: {e}")
            return False
    
    def sync_solution_to_ticket(self, workflow_state: WorkflowState, sub_problem_id: str, 
                               solution: SolutionAttempt) -> bool:
        """Sync a completed solution to its corresponding Hephaestus ticket"""
        try:
            ticket_id = workflow_state.id_to_ticket_id_map.get(sub_problem_id)
            if not ticket_id:
                logger.warning(f"No Hephaestus ticket found for sub-problem {sub_problem_id}")
                return False
            
            # Update ticket with solution content
            success = self.client.update_ticket(
                ticket_id=ticket_id,
                status=TicketStatus.IN_REVIEW,
                description=f"""
Solution for Sub-Problem {sub_problem_id}:

Content:
{solution.content}

Generated by model: {solution.generated_by_model}
Timestamp: {datetime.fromtimestamp(solution.timestamp).isoformat()}

Please review the solution and update status accordingly.
                """.strip()
            )
            
            if success:
                logger.info(f"Synced solution to ticket {ticket_id} for sub-problem {sub_problem_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to sync solution to ticket: {e}")
            return False
    
    def sync_critique_to_ticket(self, workflow_state: WorkflowState, sub_problem_id: str, 
                               critique: CritiqueReport) -> bool:
        """Sync a critique report to its corresponding Hephaestus ticket"""
        try:
            ticket_id = workflow_state.id_to_ticket_id_map.get(sub_problem_id)
            if not ticket_id:
                logger.warning(f"No Hephaestus ticket found for sub-problem {sub_problem_id}")
                return False
            
            # Format critique information
            critique_info = f"""
Critique Report for Sub-Problem {sub_problem_id}:

Gauntlet: {critique.gauntlet_name}
Approved: {critique.is_approved}
Overall Score: {critique.overall_score}

Summary: {critique.summary}

Flaws Identified: {len(critique.identified_flaws)}
Suggested Improvements: {len(critique.suggested_improvements)}

Detailed reports from {len(critique.reports_by_judge)} judges.
                """.strip()
            
            # Update ticket status based on critique result
            new_status = TicketStatus.IN_REVIEW if critique.is_approved else TicketStatus.BLOCKED
            
            success = self.client.update_ticket(
                ticket_id=ticket_id,
                status=new_status,
                description=critique_info
            )
            
            if success:
                logger.info(f"Synced critique to ticket {ticket_id} for sub-problem {sub_problem_id}, new status: {new_status.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to sync critique to ticket: {e}")
            return False
    
    def sync_verification_to_ticket(self, workflow_state: WorkflowState, sub_problem_id: str, 
                                   verification: VerificationReport) -> bool:
        """Sync a verification report to its corresponding Hephaestus ticket"""
        try:
            ticket_id = workflow_state.id_to_ticket_id_map.get(sub_problem_id)
            if not ticket_id:
                logger.warning(f"No Hephaestus ticket found for sub-problem {sub_problem_id}")
                return False
            
            # Format verification information
            verification_info = f"""
Verification Report for Sub-Problem {sub_problem_id}:

Gauntlet: {verification.gauntlet_name}
Approved: {verification.is_approved}
Average Score: {verification.average_score}
Score Variance: {verification.score_variance}

Summary: {verification.summary}

Criteria Met: {verification.criteria_met}
Criteria Not Met: {verification.criteria_not_met}

Targeted Feedback: {verification.targeted_feedback or 'None'}
                """.strip()
            
            # Update ticket status based on verification result
            new_status = TicketStatus.DONE if verification.is_approved else TicketStatus.IN_REVIEW
            
            success = self.client.update_ticket(
                ticket_id=ticket_id,
                status=new_status,
                description=verification_info
            )
            
            if success:
                logger.info(f"Synced verification to ticket {ticket_id} for sub-problem {sub_problem_id}, new status: {new_status.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to sync verification to ticket: {e}")
            return False
    
    def get_workflow_sync_status(self, workflow_state: WorkflowState) -> Dict[str, Any]:
        """Get the current synchronization status for a workflow"""
        return self.sync_manager.get_sync_metrics(workflow_state)
    
    def close_workflow_sync(self, workflow_state: WorkflowState) -> bool:
        """Close the workflow sync by updating the main epic ticket"""
        if not workflow_state.hephaestus_workflow_id:
            logger.warning(f"No Hephaestus workflow ID for {workflow_state.workflow_id}")
            return False
        
        try:
            # Update the main epic ticket with final status
            status = TicketStatus.DONE if workflow_state.status == "completed" else TicketStatus.IN_REVIEW
            
            # Add final summary to description
            summary = f"""
Final Status: {workflow_state.status}
Progress: {workflow_state.progress:.2%}
Start Time: {datetime.fromtimestamp(workflow_state.start_time).isoformat()}
End Time: {datetime.fromtimestamp(workflow_state.end_time).isoformat() if workflow_state.end_time else 'N/A'}

Final Solution Available: {'Yes' if workflow_state.final_solution else 'No'}
Total Sub-problems: {len(workflow_state.decomposition_plan.sub_problems) if workflow_state.decomposition_plan else 0}
Solved: {len(workflow_state.solved_sub_problem_ids)}
Rejected: {len(workflow_state.rejected_sub_problems)}
Refinement Loops: {workflow_state.refinement_loop_count}

This workflow has been completed in OpenEvolve.
            """.strip()
            
            success = self.client.update_ticket(
                ticket_id=workflow_state.hephaestus_workflow_id,
                status=status,
                description=summary
            )
            
            if success:
                logger.info(f"Closed workflow sync for {workflow_state.workflow_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to close workflow sync: {e}")
            return False

    def start_sync_loop(self, workflow_state: WorkflowState, interval: int = 60):
        """Starts a background thread to periodically sync status from Hephaestus to OpenEvolve."""
        if workflow_state.workflow_id in self.active_synchronizations:
            logger.warning(f"Sync loop already active for workflow {workflow_state.workflow_id}")
            return

        stop_event = threading.Event()
        sync_thread = threading.Thread(
            target=self._sync_loop_worker,
            args=(workflow_state, interval, stop_event),
            daemon=True
        )
        sync_thread.start()
        self.active_synchronizations[workflow_state.workflow_id] = {
            "thread": sync_thread,
            "stop_event": stop_event
        }
        logger.info(f"Started Hephaestus sync loop for workflow {workflow_state.workflow_id}")

    def stop_sync_loop(self, workflow_id: str):
        """Stops the background sync thread for a given workflow."""
        sync_info = self.active_synchronizations.pop(workflow_id, None)
        if sync_info:
            sync_info["stop_event"].set()
            sync_info["thread"].join(timeout=5) # Give it a moment to stop
            if sync_info["thread"].is_alive():
                logger.warning(f"Hephaestus sync thread for {workflow_id} did not terminate gracefully.")
            logger.info(f"Stopped Hephaestus sync loop for workflow {workflow_id}")
        else:
            logger.warning(f"No active sync loop found for workflow {workflow_id}")

    def _sync_loop_worker(self, workflow_state: WorkflowState, interval: int, stop_event: threading.Event):
        """Worker function for the background sync loop."""
        while not stop_event.is_set():
            try:
                self._sync_status_from_hephaestus(workflow_state)
            except Exception as e:
                logger.error(f"Error during Hephaestus sync for workflow {workflow_state.workflow_id}: {e}")
            stop_event.wait(interval)

    def _sync_status_from_hephaestus(self, workflow_state: WorkflowState):
        """Fetches ticket statuses from Hephaestus and updates OpenEvolve WorkflowState."""
        if not workflow_state.hephaestus_workflow_id:
            return

        # Get all tickets associated with this workflow epic
        tickets = self.client.get_tickets_by_label(f"workflow-{workflow_state.workflow_id}")
        for ticket in tickets:
            ticket_id = ticket.get('id')
            hephaestus_status = ticket.get('status')
            sub_problem_id = workflow_state.ticket_id_to_subproblem_id_map.get(ticket_id)

            if sub_problem_id and hephaestus_status:
                # Map Hephaestus status back to OpenEvolve status
                openevolve_status = self._map_hephaestus_status_to_openevolve(hephaestus_status)
                if openevolve_status:
                    workflow_state.update_subproblem_status(sub_problem_id, openevolve_status)
                    logger.info(f"Hephaestus -> OpenEvolve: Sub-problem {sub_problem_id} (ticket {ticket_id}) status changed to {openevolve_status}")

    def _map_hephaestus_status_to_openevolve(self, hephaestus_status: str) -> Optional[str]:
        """Map Hephaestus ticket status to OpenEvolve sub-problem status."""
        status_mapping = {
            TicketStatus.TODO.value: "pending",
            TicketStatus.IN_PROGRESS.value: "in_progress",
            TicketStatus.IN_REVIEW.value: "requires_rework",
            TicketStatus.BLOCKED.value: "blocked",
            TicketStatus.DONE.value: "solved"
        }
        return status_mapping.get(hephaestus_status.lower())


# Helper function to initialize Hephaestus integration for a workflow
def setup_hephaestus_integration(workflow_state: WorkflowState, api_base: str, api_key: str, project_id: str) -> Optional[HephaestusIntegrationManager]:
    """
    Set up Hephaestus integration for a workflow
    
    Args:
        workflow_state: The workflow state to integrate with Hephaestus
        api_base: Base URL for the Hephaestus API
        api_key: API key for authentication
        project_id: Project ID in Hephaestus to associate tickets with
        
    Returns:
        HephaestusIntegrationManager instance or None if setup fails
    """
    try:
        integration_manager = HephaestusIntegrationManager(api_base, api_key, project_id)
        success = integration_manager.initialize_workflow_sync(workflow_state)
        
        if success:
            logger.info(f"Successfully set up Hephaestus integration for workflow {workflow_state.workflow_id}")
            return integration_manager
        else:
            logger.error(f"Failed to set up Hephaestus integration for workflow {workflow_state.workflow_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error setting up Hephaestus integration: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Example usage would require actual API credentials
    print("Hephaestus Integration Module loaded successfully")
    print("- Contains HephaestusClient for API communication")
    print("- Contains HephaestusWorkflowSync for workflow synchronization") 
    print("- Contains HephaestusIntegrationManager for complete workflow management")
    print("- Provides setup_hephaestus_integration() helper function")