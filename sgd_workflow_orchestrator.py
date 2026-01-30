"""
Sovereign-Grade Decomposition (SGD) Workflow Orchestrator for CREWAI

This module implements the full Sovereign-Grade Decomposition workflow within CREWAI,
integrating with OpenEvolve for team and gauntlet management.
"""

import asyncio
import logging
import threading
import time
import json
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

from openevolve_structures import (
    WorkflowState, 
    DecompositionPlan, 
    SubProblem, 
    SolutionAttempt,
    CritiqueReport,
    VerificationReport
)

logger = logging.getLogger(__name__)

class SGDWorkflowStatus(Enum):
    """Status values for SGD workflows"""
    PENDING = "pending"
    CONTENT_ANALYSIS = "content_analysis"
    PLANNING = "planning"
    DECOMPOSITION = "decomposition"
    SUB_PROBLEM_SOLVING = "sub_problem_solving"
    REASSEMBLY = "reassembly"
    FINAL_VERIFICATION = "final_verification"
    COMPLETED = "completed"
    FAILED = "failed"

class SGDWorkflowOrchestrator:
    """
    Orchestrates the complete Sovereign-Grade Decomposition Workflow within CREWAI
    """
    
    def __init__(self, CREWAI_api_base: str = "http://localhost:8002", 
                 openevolve_api_base: str = "http://localhost:8000"):
        self.CREWAI_api_base = CREWAI_api_base
        self.openevolve_api_base = openevolve_api_base
        self.active_workflows: Dict[str, WorkflowState] = {}
        self.running = True
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def create_workflow(self, problem_statement: str, 
                       content_analyzer_team: str,
                       planner_team: str,
                       solver_team: str,
                       patcher_team: str,
                       assembler_team: str,
                       sub_problem_red_gauntlet: str,
                       sub_problem_gold_gauntlet: str,
                       final_red_gauntlet: str,
                       final_gold_gauntlet: str,
                       mdap_enabled: bool = False,
                       mdap_config: Optional[Dict[str, Any]] = None,
                       maker_enabled: bool = False,
                       maker_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new SGD workflow with the specified team and gauntlet configurations
        """
        workflow_id = f"sgd_{int(time.time())}_{problem_statement[:10].replace(' ', '_')}"
        
        # Create initial workflow state
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            workflow_type="sovereign_decomposition",
            problem_statement=problem_statement,
            current_stage=SGDWorkflowStatus.PENDING.value,
            status="created",
            mdap_enabled=mdap_enabled,
            mdap_config=mdap_config or {},
            maker_enabled=maker_enabled,
            maker_config=maker_config or {}
        )
        
        # Store the team and gauntlet configurations for this workflow
        workflow_state.content_analyzer_team_name = content_analyzer_team
        workflow_state.planner_team_name = planner_team
        workflow_state.solver_team_name = solver_team
        workflow_state.patcher_team_name = patcher_team
        workflow_state.assembler_team_name = assembler_team
        workflow_state.sub_problem_red_gauntlet_name = sub_problem_red_gauntlet
        workflow_state.sub_problem_gold_gauntlet_name = sub_problem_gold_gauntlet
        workflow_state.final_red_gauntlet_name = final_red_gauntlet
        workflow_state.final_gold_gauntlet_name = final_gold_gauntlet
        
        self.active_workflows[workflow_id] = workflow_state
        logger.info(f"Created SGD workflow: {workflow_id}")
        
        return workflow_id

    async def run_workflow(self, workflow_id: str):
        """
        Run the complete SGD workflow from start to finish
        """
        if workflow_id not in self.active_workflows:
            logger.error(f"Workflow {workflow_id} not found")
            return False

        workflow_state = self.active_workflows[workflow_id]
        workflow_state.status = "running"
        workflow_state.current_stage = SGDWorkflowStatus.CONTENT_ANALYSIS.value
        workflow_state.start_time = time.time()

        try:
            # Stage 0: Content Analysis
            logger.info(f"Starting content analysis for workflow {workflow_id}")
            workflow_state.current_stage = SGDWorkflowStatus.CONTENT_ANALYSIS.value
            analyzed_context = await self._perform_content_analysis(workflow_state)
            if not analyzed_context:
                workflow_state.status = "failed"
                workflow_state.current_stage = SGDWorkflowStatus.FAILED.value
                return False

            # Update workflow state with analyzed context
            # (In a real implementation, we'd call the OpenEvolve content analyzer here)

            # Stage 1: Decomposition Planning
            logger.info(f"Starting decomposition planning for workflow {workflow_id}")
            workflow_state.current_stage = SGDWorkflowStatus.PLANNING.value
            decomposition_plan = await self._generate_decomposition_plan(workflow_state, analyzed_context)
            if not decomposition_plan:
                workflow_state.status = "failed"
                workflow_state.current_stage = SGDWorkflowStatus.FAILED.value
                return False

            workflow_state.decomposition_plan = decomposition_plan

            # Stage 2: Manual Review (In a real implementation, this would have UI interaction)
            logger.info(f"Manual review stage for workflow {workflow_id}")
            workflow_state.current_stage = SGDWorkflowStatus.DECOMPOSITION.value
            # In a real implementation, this would pause and wait for human review/approval

            # Stage 3: Sub-Problem Solving
            logger.info(f"Starting sub-problem solving for workflow {workflow_id}")
            workflow_state.current_stage = SGDWorkflowStatus.SUB_PROBLEM_SOLVING.value
            sub_problem_solutions = await self._solve_sub_problems(workflow_state)
            if not sub_problem_solutions:
                workflow_state.status = "failed"
                workflow_state.current_stage = SGDWorkflowStatus.FAILED.value
                return False

            workflow_state.sub_problem_solutions = sub_problem_solutions

            # Stage 4: Reassembly
            logger.info(f"Starting reassembly for workflow {workflow_id}")
            workflow_state.current_stage = SGDWorkflowStatus.REASSEMBLY.value
            final_solution = await self._reassemble_solution(workflow_state)
            if not final_solution:
                workflow_state.status = "failed"
                workflow_state.current_stage = SGDWorkflowStatus.FAILED.value
                return False

            workflow_state.final_solution = final_solution

            # Stage 5: Final Verification
            logger.info(f"Starting final verification for workflow {workflow_id}")
            workflow_state.current_stage = SGDWorkflowStatus.FINAL_VERIFICATION.value
            verification_passed = await self._verify_final_solution(workflow_state)
            
            if verification_passed:
                workflow_state.status = "completed"
                workflow_state.current_stage = SGDWorkflowStatus.COMPLETED.value
                workflow_state.end_time = time.time()
                workflow_state.progress = 1.0
                logger.info(f"Workflow {workflow_id} completed successfully")
            else:
                workflow_state.status = "failed_final_verification"
                workflow_state.current_stage = SGDWorkflowStatus.FAILED.value
                workflow_state.end_time = time.time()
                logger.warning(f"Workflow {workflow_id} failed final verification")
                
        except Exception as e:
            logger.error(f"Error running workflow {workflow_id}: {e}")
            workflow_state.status = "error"
            workflow_state.current_stage = SGDWorkflowStatus.FAILED.value
            workflow_state.end_time = time.time()
            return False

        return True

    async def _perform_content_analysis(self, workflow_state: WorkflowState) -> Dict[str, Any]:
        """
        Perform content analysis using the configured content analyzer team
        """
        try:
            # In a real implementation, call the OpenEvolve content analysis API
            # For now, return a basic analysis
            return {
                "complexity": "high",
                "domain": "software",
                "required_skills": ["python", "algorithms"],
                "estimated_sub_problems": 3
            }
        except Exception as e:
            logger.error(f"Error in content analysis: {e}")
            return {}

    async def _generate_decomposition_plan(self, workflow_state: WorkflowState, analyzed_context: Dict[str, Any]) -> Optional[DecompositionPlan]:
        """
        Generate decomposition plan using the configured planner team
        """
        try:
            # In a real implementation, call the OpenEvolve planner API
            # For now, create a basic decomposition plan
            sub_problems = [
                SubProblem(
                    id=f"sp_{i}",
                    description=f"Sub-problem {i} for the main problem",
                    dependencies=[],
                    solver_team_name=workflow_state.solver_team_name,
                    red_team_gauntlet_name=workflow_state.sub_problem_red_gauntlet_name,
                    gold_team_gauntlet_name=workflow_state.sub_problem_gold_gauntlet_name,
                    patcher_team_name=workflow_state.patcher_team_name
                )
                for i in range(1, 4)  # Create 3 sub-problems as example
            ]
            
            return DecompositionPlan(
                problem_statement=workflow_state.problem_statement,
                analyzed_context=analyzed_context,
                sub_problems=sub_problems,
                mdap_enabled=getattr(workflow_state, "mdap_enabled", False),
                mdap_config=getattr(workflow_state, "mdap_config", {}),
                maker_enabled=getattr(workflow_state, "maker_enabled", False),
                maker_config=getattr(workflow_state, "maker_config", {}),
                content_analyzer_team_name=workflow_state.content_analyzer_team_name,
                planner_team_name=workflow_state.planner_team_name,
                assembler_team_name=workflow_state.assembler_team_name,
                final_red_team_gauntlet_name=workflow_state.final_red_gauntlet_name,
                final_gold_team_gauntlet_name=workflow_state.final_gold_gauntlet_name
            )
        except Exception as e:
            logger.error(f"Error in decomposition planning: {e}")
            return None

    async def _solve_sub_problems(self, workflow_state: WorkflowState) -> Dict[str, SolutionAttempt]:
        """
        Solve all sub-problems using CREWAI tickets and OpenEvolve verification
        """
        solutions = {}
        
        if not workflow_state.decomposition_plan:
            return solutions

        for sub_problem in workflow_state.decomposition_plan.sub_problems:
            logger.info(f"Creating ticket for sub-problem {sub_problem.id}")
            
            # Create a CREWAI ticket for the sub-problem
            ticket_data = {
                "title": f"Sub-problem {sub_problem.id}: {sub_problem.description[:50]}...",
                "description": f"""
Sub-problem ID: {sub_problem.id}
Description: {sub_problem.description}

This ticket was automatically created as part of the Sovereign-Grade Decomposition workflow.

VALIDATION PROTOCOL:
Upon completion, this task will be validated using the following gauntlets:
- Red Team Gauntlet: '{sub_problem.red_team_gauntlet_name}'
- Gold Team Gauntlet: '{sub_problem.gold_team_gauntlet_name}'
                """.strip(),
                "workflow_id": workflow_state.workflow_id,
                "red_team_gauntlet_name": sub_problem.red_team_gauntlet_name,
                "gold_team_gauntlet_name": sub_problem.gold_team_gauntlet_name
            }
            
            try:
                # Create ticket in CREWAI
                response = self.session.post(
                    f"{self.CREWAI_api_base}/tickets/create",
                    json=ticket_data
                )
                response.raise_for_status()
                
                ticket_result = response.json()
                ticket_id = ticket_result.get("ticket", {}).get("id")
                
                if ticket_id:
                    # In a real implementation, we'd wait for the ticket to be completed
                    # and then extract the solution content
                    # For now, we'll simulate completion after a delay
                    await asyncio.sleep(2)  # Simulate work time
                    
                    # Get the ticket status after "completion"
                    ticket_response = self.session.get(f"{self.CREWAI_api_base}/tickets/{ticket_id}")
                    if ticket_response.status_code == 200:
                        ticket = ticket_response.json().get("ticket")
                        if ticket and ticket.get("verification_status") in ["verified", "failed_verification"]:
                            # Create solution attempt
                            solution = SolutionAttempt(
                                sub_problem_id=sub_problem.id,
                                content=ticket.get("solution_content", f"Solution for {sub_problem.id}"),
                                generated_by_model="automated_agent",
                                timestamp=time.time(),
                                status="generated"
                            )
                            
                            # Add verification reports if available
                            verification_reports = ticket.get("verification_reports", [])
                            for report_data in verification_reports:
                                if report_data.get("type") == "gold_team":
                                    verification_report = VerificationReport(
                                        solution_attempt_id=solution.sub_problem_id,
                                        gauntlet_name=report_data.get("gauntlet_name", ""),
                                        is_approved=report_data.get("result", {}).get("is_approved", False),
                                        reports_by_judge=[report_data.get("result", {})],
                                        summary=report_data.get("result", {}).get("report_summary", "")
                                    )
                                    solution.verification_reports.append(verification_report)
                                elif report_data.get("type") == "red_team":
                                    critique_report = CritiqueReport(
                                        solution_attempt_id=solution.sub_problem_id,
                                        gauntlet_name=report_data.get("gauntlet_name", ""),
                                        is_approved=report_data.get("result", {}).get("is_approved", True),
                                        reports_by_judge=[report_data.get("result", {})],
                                        summary=report_data.get("result", {}).get("report_summary", "")
                                    )
                                    solution.critique_reports.append(critique_report)
                            
                            solutions[sub_problem.id] = solution
                            
            except Exception as e:
                logger.error(f"Error creating ticket for sub-problem {sub_problem.id}: {e}")
                # Create a failed solution attempt
                solutions[sub_problem.id] = SolutionAttempt(
                    sub_problem_id=sub_problem.id,
                    content=f"Error creating ticket: {str(e)}",
                    generated_by_model="error_handler",
                    timestamp=time.time(),
                    status="failed"
                )

        return solutions

    async def _reassemble_solution(self, workflow_state: WorkflowState) -> Optional[SolutionAttempt]:
        """
        Reassemble the final solution from completed sub-problems using the assembler team
        """
        try:
            # In a real implementation, call the OpenEvolve assembler API
            # For now, create a basic reassembled solution
            reassembled_content = f"""
Final Solution for: {workflow_state.problem_statement}

Reassembled from {len(workflow_state.sub_problem_solutions)} sub-problem solutions:

"""
            for sp_id, solution in workflow_state.sub_problem_solutions.items():
                reassembled_content += f"\n### {sp_id}:\n{solution.content}\n---\n"

            return SolutionAttempt(
                sub_problem_id="final_solution",
                content=reassembled_content,
                generated_by_model=workflow_state.assembler_team_name,
                timestamp=time.time(),
                status="reassembled"
            )
        except Exception as e:
            logger.error(f"Error in reassembly: {e}")
            return None

    async def _verify_final_solution(self, workflow_state: WorkflowState) -> bool:
        """
        Verify the final solution using the configured final verification gauntlets
        """
        if not workflow_state.final_solution:
            logger.error("No final solution to verify")
            return False

        try:
            # In a real implementation, call the OpenEvolve verification API
            # For now, simulate verification
            
            # Run final red team gauntlet if configured
            if workflow_state.final_red_gauntlet_name:
                red_verification = self._run_gauntlet(
                    solution_content=workflow_state.final_solution.content,
                    gauntlet_name=workflow_state.final_red_gauntlet_name
                )
                
                if not red_verification.get("is_approved", False):
                    logger.info("Final solution failed Red Team verification")
                    return False

            # Run final gold team gauntlet if configured
            if workflow_state.final_gold_gauntlet_name:
                gold_verification = self._run_gauntlet(
                    solution_content=workflow_state.final_solution.content,
                    gauntlet_name=workflow_state.final_gold_gauntlet_name
                )
                
                return gold_verification.get("is_approved", False)
                
            # If no gauntlets configured, consider as verified
            return True

        except Exception as e:
            logger.error(f"Error verifying final solution: {e}")
            return False

    def _run_gauntlet(self, solution_content: str, gauntlet_name: str) -> Dict[str, Any]:
        """
        Run a gauntlet against the OpenEvolve API
        """
        try:
            validation_payload = {
                "solution_content": solution_content,
                "gauntlet_name": gauntlet_name,
                "context": {
                    "validation_type": "final_solution_verification",
                    "target": "complete_solution"
                }
            }
            
            response = self.session.post(
                f"{self.openevolve_api_base}/run_gauntlet",
                json=validation_payload,
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Gauntlet API returned status {response.status_code}")
                return {"is_approved": False, "error": f"Status {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error running gauntlet: {e}")
            return {"is_approved": False, "error": str(e)}

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific workflow
        """
        if workflow_id not in self.active_workflows:
            return None

        workflow_state = self.active_workflows[workflow_id]
        
        return {
            "workflow_id": workflow_state.workflow_id,
            "status": workflow_state.status,
            "current_stage": workflow_state.current_stage,
            "progress": workflow_state.progress,
            "start_time": workflow_state.start_time,
            "end_time": workflow_state.end_time,
            "solved_sub_problems": len(workflow_state.solved_sub_problem_ids),
            "total_sub_problems": len(workflow_state.decomposition_plan.sub_problems) if workflow_state.decomposition_plan else 0
        }

    def stop_workflow(self, workflow_id: str) -> bool:
        """
        Stop a running workflow
        """
        if workflow_id not in self.active_workflows:
            return False

        workflow_state = self.active_workflows[workflow_id]
        workflow_state.status = "stopped"
        workflow_state.end_time = time.time()
        return True

    def list_workflows(self) -> List[Dict[str, Any]]:
        """
        List all active workflows
        """
        return [
            {
                "workflow_id": ws.workflow_id,
                "status": ws.status,
                "current_stage": ws.current_stage,
                "problem_statement": ws.problem_statement[:50] + "..." if len(ws.problem_statement) > 50 else ws.problem_statement,
                "progress": ws.progress
            }
            for ws in self.active_workflows.values()
        ]

    async def run_continuous_monitoring(self):
        """
        Run continuous monitoring of all workflows
        """
        while self.running:
            try:
                # Update workflow progress metrics
                for workflow_id, workflow_state in self.active_workflows.items():
                    if workflow_state.status == "running":
                        # Calculate progress based on completion of sub-problems
                        if workflow_state.decomposition_plan:
                            total_sub_problems = len(workflow_state.decomposition_plan.sub_problems)
                            if total_sub_problems > 0:
                                workflow_state.progress = len(workflow_state.solved_sub_problem_ids) / total_sub_problems
                                # Cap progress at 0.9 until final verification
                                if workflow_state.current_stage != SGDWorkflowStatus.COMPLETED.value:
                                    workflow_state.progress = min(workflow_state.progress, 0.9)

                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(5)

    def shutdown(self) -> None:
        """
        Shutdown the orchestrator
        """
        self.running = False
        logger.info("SGD Workflow Orchestrator shutdown initiated")


# Example usage function
async def example_usage():
    """
    Example of how to use the SGD Workflow Orchestrator
    """
    orchestrator = SGDWorkflowOrchestrator()
    
    # Create a workflow
    workflow_id = orchestrator.create_workflow(
        problem_statement="Implement a secure user authentication system",
        content_analyzer_team="content_analysis_team",
        planner_team="planning_team", 
        solver_team="solver_team",
        patcher_team="patcher_team",
        assembler_team="assembler_team",
        sub_problem_red_gauntlet="sub_problem_red_gauntlet",
        sub_problem_gold_gauntlet="sub_problem_gold_gauntlet",
        final_red_gauntlet="final_red_gauntlet",
        final_gold_gauntlet="final_gold_gauntlet",
        mdap_enabled=True,
        mdap_config={"k_min": 2, "k_max": 6},
        maker_enabled=False,
        maker_config={}
    )
    
    print(f"Created workflow: {workflow_id}")
    
    # Run the workflow in the background
    asyncio.create_task(orchestrator.run_workflow(workflow_id))
    
    # Monitor progress
    while True:
        status = orchestrator.get_workflow_status(workflow_id)
        if status:
            print(f"Workflow {workflow_id}: {status['status']} - {status['current_stage']}")
            if status['status'] in ['completed', 'failed', 'error']:
                break
        await asyncio.sleep(5)

if __name__ == "__main__":
    # Example usage
    asyncio.run(example_usage())
