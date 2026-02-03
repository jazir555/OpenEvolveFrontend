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
from typing import Dict, Any, Optional, List, Tuple
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

# **ACTUAL INTEGRATION**: Alerting and knowledge for SGD workflow orchestration
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

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
    
    ICR Integration:
    - Stores workflow patterns for learning
    - Recommends optimal team/gauntlet configuration
    - Predicts workflow success probability
    - Learns from workflow outcomes
    """
    
    def __init__(
        self, 
        CREWAI_api_base: str = "http://localhost:8002", 
        openevolve_api_base: str = "http://localhost:8000",
        enable_icr: bool = True
    ):
        self.CREWAI_api_base = CREWAI_api_base
        self.openevolve_api_base = openevolve_api_base
        self.active_workflows: Dict[str, WorkflowState] = {}
        self.completed_workflows: List[Dict] = []  # For ICR learning
        self.running = True
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        # ICR Integration: Pattern storage
        self.enable_icr = enable_icr
        self.icr_patterns: Dict[str, Any] = {
            'problem_type_patterns': {},  # problem_type -> success patterns
            'complexity_patterns': {},  # complexity_range -> patterns
            'team_config_patterns': {},  # team_config_hash -> success_rate
            'gauntlet_config_patterns': {},  # gauntlet_config -> pass_rate
            'stage_duration_patterns': {},  # stage_name -> avg_duration
        }
        
        # ICR: Workflow outcome predictions
        self._prediction_cache: Dict[str, Dict] = {}

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

                # **ACTUAL INTEGRATION**: Extract knowledge and track performance for successful workflow
                duration = (workflow_state.end_time - workflow_state.start_time) if workflow_state.start_time else 0
                num_sub_problems = len(workflow_state.decomposition_plan.sub_problems) if workflow_state.decomposition_plan else 0
                self._extract_sgd_knowledge(workflow_id, workflow_state, True)
                self._track_sgd_performance(workflow_id, True, duration, num_sub_problems)
            else:
                workflow_state.status = "failed_final_verification"
                workflow_state.current_stage = SGDWorkflowStatus.FAILED.value
                workflow_state.end_time = time.time()
                logger.warning(f"Workflow {workflow_id} failed final verification")

                # **ACTUAL INTEGRATION**: Trigger alert, extract knowledge, and track performance
                duration = (workflow_state.end_time - workflow_state.start_time) if workflow_state.start_time else 0
                num_sub_problems = len(workflow_state.decomposition_plan.sub_problems) if workflow_state.decomposition_plan else 0
                self._trigger_sgd_alerts(workflow_id, False, "final_verification", "Failed final verification", {"duration": duration})
                self._extract_sgd_knowledge(workflow_id, workflow_state, False)
                self._track_sgd_performance(workflow_id, False, duration, num_sub_problems)

        except Exception as e:
            logger.error(f"Error running workflow {workflow_id}: {e}")
            workflow_state.status = "error"
            workflow_state.current_stage = SGDWorkflowStatus.FAILED.value
            workflow_state.end_time = time.time()

            # **ACTUAL INTEGRATION**: Trigger alert, extract knowledge, and track performance
            duration = (workflow_state.end_time - workflow_state.start_time) if workflow_state.start_time else 0
            num_sub_problems = len(workflow_state.decomposition_plan.sub_problems) if workflow_state.decomposition_plan else 0
            self._trigger_sgd_alerts(workflow_id, False, workflow_state.current_stage, str(e), {"duration": duration})
            self._extract_sgd_knowledge(workflow_id, workflow_state, False)
            self._track_sgd_performance(workflow_id, False, duration, num_sub_problems)

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
    
    # =========================================================================
    # ICR INTEGRATION METHODS
    # =========================================================================
    
    def _analyze_problem_complexity(self, problem_statement: str) -> Tuple[str, int]:
        """
        Analyze problem statement to determine complexity and type.
        
        Returns:
            Tuple of (problem_type, complexity_score)
        """
        problem_lower = problem_statement.lower()
        
        # Determine problem type
        if any(kw in problem_lower for kw in ['implement', 'build', 'create', 'develop']):
            problem_type = "implementation"
        elif any(kw in problem_lower for kw in ['design', 'architecture', 'plan']):
            problem_type = "design"
        elif any(kw in problem_lower for kw in ['optimize', 'improve', 'enhance']):
            problem_type = "optimization"
        elif any(kw in problem_lower for kw in ['fix', 'debug', 'resolve', 'repair']):
            problem_type = "debugging"
        elif any(kw in problem_lower for kw in ['research', 'analyze', 'investigate']):
            problem_type = "research"
        else:
            problem_type = "general"
        
        # Estimate complexity (1-10)
        complexity = 5  # Base complexity
        
        # Length indicator
        if len(problem_statement) > 500:
            complexity += 2
        elif len(problem_statement) > 200:
            complexity += 1
        
        # Keyword indicators
        if any(kw in problem_lower for kw in ['distributed', 'microservices', 'scalable']):
            complexity += 2
        if any(kw in problem_lower for kw in ['machine learning', 'ai', 'neural']):
            complexity += 2
        if any(kw in problem_lower for kw in ['security', 'encryption', 'authentication']):
            complexity += 1
        
        # Cap at 10
        complexity = min(10, max(1, complexity))
        
        return problem_type, complexity
    
    def predict_workflow_success(
        self,
        problem_statement: str,
        team_config: Dict[str, str],
        gauntlet_config: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Predict workflow success probability using ICR patterns.
        
        Args:
            problem_statement: Description of the problem
            team_config: Team configuration (content_analyzer_team, planner_team, etc.)
            gauntlet_config: Gauntlet configuration (sub_problem_red_gauntlet, etc.)
            
        Returns:
            Prediction dictionary with success probability and confidence
        """
        if not self.enable_icr:
            return {
                'success_probability': 0.5,
                'confidence': 0.0,
                'reason': 'ICR disabled'
            }
        
        problem_type, complexity = self._analyze_problem_complexity(problem_statement)
        
        # Create cache key
        cache_key = f"{problem_type}_{complexity}_{hash(json.dumps(team_config, sort_keys=True))}"
        
        # Check cache
        if cache_key in self._prediction_cache:
            cached = self._prediction_cache[cache_key]
            if time.time() - cached['timestamp'] < 3600:  # Cache for 1 hour
                return cached['prediction']
        
        # Get patterns for this problem type
        type_patterns = self.icr_patterns['problem_type_patterns'].get(problem_type, [])
        complexity_patterns = self.icr_patterns['complexity_patterns'].get(f"{complexity // 2}", [])
        
        # Calculate base success probability
        if type_patterns:
            passed = sum(1 for p in type_patterns if p.get('success', False))
            type_success_rate = passed / len(type_patterns)
        else:
            type_success_rate = 0.5
        
        if complexity_patterns:
            passed = sum(1 for p in complexity_patterns if p.get('success', False))
            complexity_success_rate = passed / len(complexity_patterns)
        else:
            complexity_success_rate = 0.5
        
        # Weight the factors
        success_probability = (type_success_rate * 0.4 + complexity_success_rate * 0.4 + 0.2)
        
        # Adjust for team configuration if we have patterns
        team_hash = hash(json.dumps(team_config, sort_keys=True))
        team_patterns = self.icr_patterns['team_config_patterns'].get(team_hash, [])
        
        if team_patterns:
            team_success_rate = sum(1 for p in team_patterns if p.get('success', False)) / len(team_patterns)
            # Blend team pattern with base rate
            success_probability = success_probability * 0.7 + team_success_rate * 0.3
        
        # Adjust for gauntlet configuration
        gauntlet_hash = hash(json.dumps(gauntlet_config, sort_keys=True))
        gauntlet_patterns = self.icr_patterns['gauntlet_config_patterns'].get(gauntlet_hash, [])
        
        if gauntlet_patterns:
            gauntlet_pass_rate = sum(1 for p in gauntlet_patterns if p.get('passed', False)) / len(gauntlet_patterns)
            success_probability = success_probability * 0.8 + gauntlet_pass_rate * 0.2
        
        # Calculate confidence based on pattern count
        total_patterns = len(type_patterns) + len(complexity_patterns)
        if total_patterns >= 20:
            confidence = 0.9
        elif total_patterns >= 10:
            confidence = 0.7
        elif total_patterns >= 5:
            confidence = 0.5
        else:
            confidence = 0.25
        
        # Generate risk factors
        risk_factors = []
        if complexity > 7:
            risk_factors.append("High complexity problem")
        if complexity > 5 and not team_patterns:
            risk_factors.append("No historical data for this complexity level")
        if not type_patterns:
            risk_factors.append(f"No historical data for {problem_type} problems")
        
        prediction = {
            'success_probability': max(0.0, min(1.0, success_probability)),
            'confidence': confidence,
            'problem_type': problem_type,
            'estimated_complexity': complexity,
            'risk_factors': risk_factors,
            'recommendations': self._get_recommendations(problem_type, complexity, team_config)
        }
        
        # Cache prediction
        self._prediction_cache[cache_key] = {
            'timestamp': time.time(),
            'prediction': prediction
        }
        
        return prediction
    
    def _get_recommendations(
        self,
        problem_type: str,
        complexity: int,
        team_config: Dict[str, str]
    ) -> List[str]:
        """Generate recommendations based on patterns"""
        recommendations = []
        
        # Get patterns for this problem type
        type_patterns = self.icr_patterns['problem_type_patterns'].get(problem_type, [])
        
        if not type_patterns:
            recommendations.append(f"No historical data for {problem_type} problems - monitor closely")
        else:
            # Find successful team configurations
            successful_configs = [p for p in type_patterns if p.get('success', False)]
            
            if successful_configs:
                # Recommend teams from successful workflows
                for sp in successful_configs[:3]:
                    if sp.get('content_analyzer_team') != team_config.get('content_analyzer_team'):
                        recommendations.append(f"Consider using '{sp.get('content_analyzer_team')}' for content analysis")
                        break
        
        # Complexity-based recommendations
        if complexity > 7:
            recommendations.append("High complexity - consider additional refinement cycles")
            recommendations.append("Consider using MDAP for better decomposition")
        
        if complexity > 5:
            recommendations.append("Consider using more thorough gauntlet validation")
        
        return recommendations
    
    def store_workflow_pattern(
        self,
        workflow_id: str,
        problem_statement: str,
        team_config: Dict[str, str],
        gauntlet_config: Dict[str, str],
        success: bool,
        duration_seconds: float,
        stages_completed: List[str],
        final_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store workflow pattern for ICR learning.
        
        Args:
            workflow_id: Unique workflow identifier
            problem_statement: Original problem statement
            team_config: Team configuration used
            gauntlet_config: Gauntlet configuration used
            success: Whether workflow succeeded
            duration_seconds: Total duration in seconds
            stages_completed: List of completed stages
            final_metrics: Optional metrics from the workflow
        """
        if not self.enable_icr:
            return
        
        logger.info(f"Storing workflow pattern for {workflow_id}")
        
        problem_type, complexity = self._analyze_problem_complexity(problem_statement)
        
        # Create pattern record
        pattern = {
            'workflow_id': workflow_id,
            'problem_type': problem_type,
            'complexity': complexity,
            'problem_statement': problem_statement[:200],  # Truncate for storage
            'team_config': team_config,
            'gauntlet_config': gauntlet_config,
            'success': success,
            'duration_seconds': duration_seconds,
            'stages_completed': stages_completed,
            'final_metrics': final_metrics or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Store by problem type
        if problem_type not in self.icr_patterns['problem_type_patterns']:
            self.icr_patterns['problem_type_patterns'][problem_type] = []
        
        patterns = self.icr_patterns['problem_type_patterns'][problem_type]
        patterns.append(pattern)
        if len(patterns) > 100:
            patterns.pop(0)  # Keep last 100
        
        # Store by complexity
        complexity_key = str(complexity // 2)
        if complexity_key not in self.icr_patterns['complexity_patterns']:
            self.icr_patterns['complexity_patterns'][complexity_key] = []
        
        complexity_patterns = self.icr_patterns['complexity_patterns'][complexity_key]
        complexity_patterns.append(pattern)
        if len(complexity_patterns) > 100:
            complexity_patterns.pop(0)
        
        # Store by team configuration
        team_hash = hash(json.dumps(team_config, sort_keys=True))
        if team_hash not in self.icr_patterns['team_config_patterns']:
            self.icr_patterns['team_config_patterns'][team_hash] = []
        
        team_patterns = self.icr_patterns['team_config_patterns'][team_hash]
        team_patterns.append(pattern)
        if len(team_patterns) > 50:
            team_patterns.pop(0)
        
        # Store by gauntlet configuration
        gauntlet_hash = hash(json.dumps(gauntlet_config, sort_keys=True))
        if gauntlet_hash not in self.icr_patterns['gauntlet_config_patterns']:
            self.icr_patterns['gauntlet_config_patterns'][gauntlet_hash] = []
        
        gauntlet_patterns = self.icr_patterns['gauntlet_config_patterns'][gauntlet_hash]
        gauntlet_patterns.append(pattern)
        if len(gauntlet_patterns) > 50:
            gauntlet_patterns.pop(0)
        
        # Add to completed workflows
        self.completed_workflows.append(pattern)
        if len(self.completed_workflows) > 200:
            self.completed_workflows.pop(0)
        
        logger.info(f"Workflow pattern stored: success={success}, type={problem_type}, complexity={complexity}")
    
    def recommend_optimal_config(
        self,
        problem_statement: str,
        complexity_hint: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Recommend optimal team and gauntlet configuration based on ICR patterns.
        
        Args:
            problem_statement: Description of the problem
            complexity_hint: Optional complexity hint (1-10)
            
        Returns:
            Recommended configuration dictionary
        """
        if not self.enable_icr:
            return {
                'content_analyzer_team': 'default_team',
                'planner_team': 'default_team',
                'solver_team': 'default_team',
                'patcher_team': 'default_team',
                'assembler_team': 'default_team',
                'sub_problem_red_gauntlet': 'coherence',
                'sub_problem_gold_gauntlet': 'completeness',
                'final_red_gauntlet': 'coherence',
                'final_gold_gauntlet': 'completeness',
                'mdap_enabled': False,
                'reason': 'ICR disabled - using defaults'
            }
        
        problem_type, complexity = self._analyze_problem_complexity(problem_statement)
        if complexity_hint:
            complexity = min(10, max(1, complexity_hint))
        
        # Find successful patterns for this problem type
        type_patterns = self.icr_patterns['problem_type_patterns'].get(problem_type, [])
        successful_patterns = [p for p in type_patterns if p.get('success', False)]
        
        if not successful_patterns:
            # Fall back to complexity patterns
            complexity_patterns = self.icr_patterns['complexity_patterns'].get(str(complexity // 2), [])
            successful_patterns = [p for p in complexity_patterns if p.get('success', False)]
        
        if not successful_patterns:
            # No patterns available - return defaults with explanation
            return {
                'content_analyzer_team': 'content_analysis_team',
                'planner_team': 'planning_team',
                'solver_team': 'solver_team',
                'patcher_team': 'patcher_team',
                'assembler_team': 'assembler_team',
                'sub_problem_red_gauntlet': 'coherence',
                'sub_problem_gold_gauntlet': 'completeness',
                'final_red_gauntlet': 'coherence',
                'final_gold_gauntlet': 'completeness',
                'mdap_enabled': complexity > 6,
                'maker_enabled': complexity > 8,
                'reason': f'No historical data for {problem_type} problems - using recommended defaults'
            }
        
        # Find most successful team configuration
        team_scores: Dict[str, float] = {}
        team_counts: Dict[str, int] = {}
        
        for pattern in successful_patterns:
            team_config = pattern.get('team_config', {})
            for team_role, team_name in team_config.items():
                if team_name not in team_scores:
                    team_scores[team_name] = 0.0
                    team_counts[team_name] = 0
                team_scores[team_name] += pattern.get('duration_seconds', 0) / 1000  # Lower duration = better
                team_counts[team_name] += 1
        
        # Normalize scores (lower is better, so invert)
        for team_name in team_scores:
            if team_counts[team_name] > 0:
                team_scores[team_name] = team_counts[team_name] / max(1, team_scores[team_name])
        
        # Get best teams for each role
        role_teams: Dict[str, str] = {}
        for pattern in successful_patterns:
            team_config = pattern.get('team_config', {})
            for role, team in team_config.items():
                if role not in role_teams:
                    role_teams[role] = team
        
        # Recommend gauntlets based on complexity
        if complexity <= 3:
            recommended_red = 'coherence'
            recommended_gold = 'completeness'
        elif complexity <= 6:
            recommended_red = 'feasibility'
            recommended_gold = 'dependency'
        else:
            recommended_red = 'adaptive'
            recommended_gold = 'hierarchical'
        
        # Check for gauntlet patterns
        complexity_key = str(complexity // 2)
        complexity_patterns = self.icr_patterns['complexity_patterns'].get(complexity_key, [])
        successful_gauntlets = [p for p in complexity_patterns if p.get('success', False)]
        
        if successful_gauntlets:
            # Find most successful gauntlet configuration
            gauntlet_pass_rates: Dict[str, float] = {}
            gauntlet_counts: Dict[str, int] = {}
            
            for pattern in successful_gauntlets:
                gauntlet_config = pattern.get('gauntlet_config', {})
                for key, value in gauntlet_config.items():
                    if 'gauntlet' in key:
                        if value not in gauntlet_pass_rates:
                            gauntlet_pass_rates[value] = 0.0
                            gauntlet_counts[value] = 0
                        # Calculate pass rate
                        patterns_for_gauntlet = [
                            p for p in self.icr_patterns['gauntlet_config_patterns'].get(
                                hash(json.dumps(pattern.get('gauntlet_config', {}), sort_keys=True)), []
                            )
                        ]
                        if patterns_for_gauntlet:
                            passed = sum(1 for p in patterns_for_gauntlet if p.get('passed', False))
                            gauntlet_pass_rates[value] = passed / len(patterns_for_gauntlet)
                            gauntlet_counts[value] = len(patterns_for_gauntlet)
        
        return {
            'content_analyzer_team': role_teams.get('content_analyzer_team', 'content_analysis_team'),
            'planner_team': role_teams.get('planner_team', 'planning_team'),
            'solver_team': role_teams.get('solver_team', 'solver_team'),
            'patcher_team': role_teams.get('patcher_team', 'patcher_team'),
            'assembler_team': role_teams.get('assembler_team', 'assembler_team'),
            'sub_problem_red_gauntlet': successful_patterns[0].get('gauntlet_config', {}).get('sub_problem_red_gauntlet', recommended_red),
            'sub_problem_gold_gauntlet': successful_patterns[0].get('gauntlet_config', {}).get('sub_problem_gold_gauntlet', recommended_gold),
            'final_red_gauntlet': successful_patterns[0].get('gauntlet_config', {}).get('final_red_gauntlet', recommended_red),
            'final_gold_gauntlet': successful_patterns[0].get('gauntlet_config', {}).get('final_gold_gauntlet', recommended_gold),
            'mdap_enabled': complexity > 6 or any(p.get('team_config', {}).get('mdap_enabled', False) for p in successful_patterns),
            'maker_enabled': complexity > 8,
            'reason': f'Based on {len(successful_patterns)} successful {problem_type} workflows',
            'confidence': min(0.9, 0.3 + len(successful_patterns) * 0.1),
            'estimated_success_rate': len(successful_patterns) / max(1, len(type_patterns)) if type_patterns else 0.5
        }
    
    def get_icr_statistics(self) -> Dict[str, Any]:
        """Get ICR-related statistics"""
        if not self.enable_icr:
            return {'icr_enabled': False}
        
        total_workflows = len(self.completed_workflows)
        
        # Calculate success rates
        success_counts = {'total': 0}
        for pattern in self.completed_workflows:
            ptype = pattern.get('problem_type', 'unknown')
            if ptype not in success_counts:
                success_counts[ptype] = {'total': 0, 'success': 0}
            success_counts['total'] += 1
            success_counts[ptype]['total'] += 1
            if pattern.get('success', False):
                success_counts[ptype]['success'] += 1
        
        # Calculate success rates
        success_rates = {}
        for ptype, counts in success_counts.items():
            if ptype != 'total' and counts['total'] > 0:
                success_rates[ptype] = counts['success'] / counts['total']
        
        # Calculate average duration
        durations = [p.get('duration_seconds', 0) for p in self.completed_workflows]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            'icr_enabled': True,
            'total_workflows': total_workflows,
            'overall_success_rate': success_counts['total'] / max(1, total_workflows),
            'success_rates_by_type': success_rates,
            'patterns_by_problem_type': {
                ptype: len(patterns) 
                for ptype, patterns in self.icr_patterns['problem_type_patterns'].items()
            },
            'patterns_by_complexity': {
                complexity: len(patterns)
                for complexity, patterns in self.icr_patterns['complexity_patterns'].items()
            },
            'average_duration_seconds': avg_duration,
            'unique_team_configs': len(self.icr_patterns['team_config_patterns']),
            'unique_gauntlet_configs': len(self.icr_patterns['gauntlet_config_patterns'])
        }
    
    def clear_icr_patterns(self) -> None:
        """Clear all stored ICR patterns"""
        if not self.enable_icr:
            return
        
        logger.info("Clearing all ICR patterns")
        
        self.icr_patterns = {
            'problem_type_patterns': {},
            'complexity_patterns': {},
            'team_config_patterns': {},
            'gauntlet_config_patterns': {},
            'stage_duration_patterns': {},
        }
        self.completed_workflows.clear()
        self._prediction_cache.clear()
    
    def _learn_from_completed_workflows(self) -> None:
        """Learn patterns from completed workflows (for periodic re-training)"""
        if not self.enable_icr or len(self.completed_workflows) < 5:
            return
        
        # Calculate success rates by configuration
        for team_hash, patterns in self.icr_patterns['team_config_patterns'].items():
            if len(patterns) >= 3:
                passed = sum(1 for p in patterns if p.get('success', False))
                success_rate = passed / len(patterns)
                
                # Update team config patterns with success rate
                for pattern in patterns:
                    pattern['calculated_success_rate'] = success_rate
        
        logger.info(f"Learned from {len(self.completed_workflows)} completed workflows")

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for SGD Orchestrator
    # =========================================================================

    def _trigger_sgd_alerts(
        self,
        workflow_id: str,
        success: bool,
        stage: str,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for SGD workflow failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            if not success:
                severity = AlertSeverity.HIGH

                alert_manager.create_alert(
                    title=f"SGD Workflow Failed: {workflow_id}",
                    description=f"SGD workflow '{workflow_id}' failed at stage '{stage}'. " +
                                 (f"Error: {error}" if error else ""),
                    severity=severity.value,
                    source="sgd_workflow_orchestrator",
                    component="sgd_orchestrator",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger SGD alert: {e}")

    def _extract_sgd_knowledge(
        self,
        workflow_id: str,
        workflow_state: 'WorkflowState',
        success: bool
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract SGD workflow knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"sgd_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="sgd_workflow_execution",
                source_component="sgd_workflow_orchestrator",
                title=f"SGD Workflow: {workflow_id}",
                content={
                    "workflow_id": workflow_id,
                    "problem_statement": workflow_state.problem_statement,
                    "status": workflow_state.status,
                    "current_stage": workflow_state.current_stage,
                    "success": success,
                    "num_sub_problems": len(workflow_state.decomposition_plan.sub_problems) if workflow_state.decomposition_plan else 0,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "start_time": workflow_state.start_time,
                    "end_time": workflow_state.end_time,
                    "duration": (workflow_state.end_time - workflow_state.start_time) if workflow_state.start_time and workflow_state.end_time else None,
                    "teams": {
                        "content_analyzer": workflow_state.content_analyzer_team_name,
                        "planner": workflow_state.planner_team_name,
                        "solver": workflow_state.solver_team_name,
                        "assembler": workflow_state.assembler_team_name
                    }
                },
                tags=["sgd", "workflow", "sovereign_grade", "success" if success else "failure"]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted SGD knowledge for {workflow_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract SGD knowledge: {e}")
            return False

    def _track_sgd_performance(
        self,
        workflow_id: str,
        success: bool,
        duration: float,
        num_sub_problems: int = 0
    ):
        """**ACTUAL INTEGRATION**: Track SGD workflow performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            performance_data = StrategyPerformanceData(
                strategy_name="sgd_workflow_orchestrator",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=1.0 if success else 0.0,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={"workflow_id": workflow_id, "duration": duration, "num_sub_problems": num_sub_problems}
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked SGD performance: {workflow_id}")

        except Exception as e:
            logger.error(f"Failed to track SGD performance: {e}")


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
