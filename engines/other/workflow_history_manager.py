from __future__ import annotations

import json
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from workflow_structures import WorkflowState, DecompositionPlan, SubProblem, SolutionAttempt, CritiqueReport, VerificationReport, ModelConfig, Team, GauntletDefinition, GauntletRoundRule
import dataclasses

# **ACTUAL INTEGRATION**: Alerting and knowledge for Workflow History Manager
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

class EnhancedJSONEncoder(json.JSONEncoder):
    """
    A JSON encoder that can handle dataclass objects.
    """

    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)

class WorkflowHistoryManager:
    """
    Manages the persistent storage and retrieval of WorkflowState objects.
    Stores workflow history in a JSON file.
    """
    def __init__(self, history_file: str = "workflow_history.json"):
        self.history_file = os.path.join(os.getcwd(), history_file)
        self._load_history()

    def _load_history(self) -> None:
        """
        Loads workflow history from the JSON file.
        """
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                try:
                    raw_history = json.load(f)
                    self.history: Dict[str, WorkflowState] = {}
                    for wf_id, wf_data in raw_history.items():
                        # Reconstruct WorkflowState and its nested dataclasses
                        # This requires careful handling of nested dataclasses
                        try:
                            # Reconstruct ModelConfig
                            if 'content_analyzer_team' in wf_data and wf_data['content_analyzer_team']:
                                wf_data['content_analyzer_team']['members'] = [ModelConfig(**m) for m in wf_data['content_analyzer_team']['members']]
                                wf_data['content_analyzer_team'] = Team(**wf_data['content_analyzer_team'])
                            if 'planner_team' in wf_data and wf_data['planner_team']:
                                wf_data['planner_team']['members'] = [ModelConfig(**m) for m in wf_data['planner_team']['members']]
                                wf_data['planner_team'] = Team(**wf_data['planner_team'])
                            if 'solver_team' in wf_data and wf_data['solver_team']:
                                wf_data['solver_team']['members'] = [ModelConfig(**m) for m in wf_data['solver_team']['members']]
                                wf_data['solver_team'] = Team(**wf_data['solver_team'])
                            if 'patcher_team' in wf_data and wf_data['patcher_team']:
                                wf_data['patcher_team']['members'] = [ModelConfig(**m) for m in wf_data['patcher_team']['members']]
                                wf_data['patcher_team'] = Team(**wf_data['patcher_team'])
                            if 'assembler_team' in wf_data and wf_data['assembler_team']:
                                wf_data['assembler_team']['members'] = [ModelConfig(**m) for m in wf_data['assembler_team']['members']]
                                wf_data['assembler_team'] = Team(**wf_data['assembler_team'])

                            # Reconstruct GauntletDefinition
                            if 'sub_problem_red_gauntlet' in wf_data and wf_data['sub_problem_red_gauntlet']:
                                wf_data['sub_problem_red_gauntlet']['rounds'] = [GauntletRoundRule(**r) for r in wf_data['sub_problem_red_gauntlet']['rounds']]
                                wf_data['sub_problem_red_gauntlet'] = GauntletDefinition(**wf_data['sub_problem_red_gauntlet'])
                            if 'sub_problem_gold_gauntlet' in wf_data and wf_data['sub_problem_gold_gauntlet']:
                                wf_data['sub_problem_gold_gauntlet']['rounds'] = [GauntletRoundRule(**r) for r in wf_data['sub_problem_gold_gauntlet']['rounds']]
                                wf_data['sub_problem_gold_gauntlet'] = GauntletDefinition(**wf_data['sub_problem_gold_gauntlet'])
                            if 'final_red_gauntlet' in wf_data and wf_data['final_red_gauntlet']:
                                wf_data['final_red_gauntlet']['rounds'] = [GauntletRoundRule(**r) for r in wf_data['final_red_gauntlet']['rounds']]
                                wf_data['final_red_gauntlet'] = GauntletDefinition(**wf_data['final_red_gauntlet'])
                            if 'final_gold_gauntlet' in wf_data and wf_data['final_gold_gauntlet']:
                                wf_data['final_gold_gauntlet']['rounds'] = [GauntletRoundRule(**r) for r in wf_data['final_gold_gauntlet']['rounds']]
                                wf_data['final_gold_gauntlet'] = GauntletDefinition(**wf_data['final_gold_gauntlet'])
                            if 'solver_generation_gauntlet' in wf_data and wf_data['solver_generation_gauntlet']:
                                wf_data['solver_generation_gauntlet']['rounds'] = [GauntletRoundRule(**r) for r in wf_data['solver_generation_gauntlet']['rounds']]
                                wf_data['solver_generation_gauntlet'] = GauntletDefinition(**wf_data['solver_generation_gauntlet'])

                            # Reconstruct DecompositionPlan
                            if 'decomposition_plan' in wf_data and wf_data['decomposition_plan']:
                                if 'sub_problems' in wf_data['decomposition_plan'] and wf_data['decomposition_plan']['sub_problems']:
                                    wf_data['decomposition_plan']['sub_problems'] = [SubProblem(**sp) for sp in wf_data['decomposition_plan']['sub_problems']]
                                wf_data['decomposition_plan'] = DecompositionPlan(**wf_data['decomposition_plan'])

                            # Reconstruct SolutionAttempt
                            if 'sub_problem_solutions' in wf_data and wf_data['sub_problem_solutions']:
                                wf_data['sub_problem_solutions'] = {k: SolutionAttempt(**v) for k, v in wf_data['sub_problem_solutions'].items()}
                            if 'final_solution' in wf_data and wf_data['final_solution']:
                                wf_data['final_solution'] = SolutionAttempt(**wf_data['final_solution'])

                            # Reconstruct CritiqueReport and VerificationReport
                            if 'all_critique_reports' in wf_data and wf_data['all_critique_reports']:
                                wf_data['all_critique_reports'] = [CritiqueReport(**cr) for cr in wf_data['all_critique_reports']]
                            if 'all_verification_reports' in wf_data and wf_data['all_verification_reports']:
                                wf_data['all_verification_reports'] = [VerificationReport(**vr) for vr in wf_data['all_verification_reports']]

                            # Reconstruct WorkflowState
                            self.history[wf_id] = WorkflowState(**wf_data)
                        except Exception as e:
                            print(f"Error reconstructing workflow {wf_id}: {e}. Skipping this entry.")
                    print(f"Loaded {len(self.history)} workflow history entries.")
                except json.JSONDecodeError:
                    print(f"Error decoding workflow history file: {self.history_file}. Starting with empty history.")
                    self.history = {}
        else:
            self.history = {}

    def _save_history(self) -> None:
        """
        Saves the current workflow history to the JSON file.
        """
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, cls=EnhancedJSONEncoder)

    def add_workflow_to_history(self, workflow_state: WorkflowState) -> None:
        """
        Adds a completed, failed, or cancelled workflow to the history.
        """
        try:
            self.history[workflow_state.workflow_id] = workflow_state
            self._save_history()

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance
            self._extract_history_knowledge(workflow_state, "add_to_history")
            self._track_history_performance("add_to_history", True)

        except Exception as e:
            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            self._trigger_history_alerts("add_to_history", False, workflow_state.workflow_id, str(e))
            self._track_history_performance("add_to_history", False)
            logger.error(f"Failed to add workflow to history: {e}")
            raise

    def get_all_historical_workflows(self) -> List[WorkflowState]:
        """
        Retrieves all historical workflow states.
        """
        return list(self.history.values())

    def get_historical_workflow(self, workflow_id: str) -> Optional[WorkflowState]:
        """
        Retrieves a specific historical workflow state by ID.
        """
        return self.history.get(workflow_id)

    def clear_history(self) -> None:
        """
        Clears all workflow history.
        """
        self.history = {}
        self._save_history()
    
    def get_openevolve_metrics_history(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get OpenEvolve metrics from workflow history.
        
        Args:
            workflow_id: Optional specific workflow ID, or None for all workflows
            
        Returns:
            Dictionary with OpenEvolve metrics aggregated from history
        """
        if workflow_id:
            workflow = self.get_historical_workflow(workflow_id)
            if not workflow:
                return {}
            workflows = [workflow]
        else:
            workflows = self.get_all_historical_workflows()
        
        metrics = {
            "total_workflows": len(workflows),
            "workflows_with_openevolve": 0,
            "total_evolution_calls": 0,
            "total_quality_diversity_calls": 0,
            "total_ensemble_calls": 0,
            "average_fitness_improvement": 0.0,
            "average_diversity_score": 0.0,
            "by_workflow": {}
        }
        
        fitness_improvements = []
        diversity_scores = []
        
        for wf in workflows:
            wf_metrics = self._extract_openevolve_metrics_from_workflow(wf)
            if wf_metrics["has_openevolve_data"]:
                metrics["workflows_with_openevolve"] += 1
                metrics["total_evolution_calls"] += wf_metrics["evolution_calls"]
                metrics["total_quality_diversity_calls"] += wf_metrics["quality_diversity_calls"]
                metrics["total_ensemble_calls"] += wf_metrics["ensemble_calls"]
                
                if wf_metrics["fitness_improvement"] is not None:
                    fitness_improvements.append(wf_metrics["fitness_improvement"])
                if wf_metrics["diversity_score"] is not None:
                    diversity_scores.append(wf_metrics["diversity_score"])
                
                metrics["by_workflow"][wf.workflow_id] = wf_metrics
        
        if fitness_improvements:
            metrics["average_fitness_improvement"] = sum(fitness_improvements) / len(fitness_improvements)
        if diversity_scores:
            metrics["average_diversity_score"] = sum(diversity_scores) / len(diversity_scores)
        
        return metrics
    
    def _extract_openevolve_metrics_from_workflow(self, workflow: WorkflowState) -> Dict[str, Any]:
        """
        Extract OpenEvolve metrics from a single workflow.
        
        Args:
            workflow: Workflow state to extract metrics from
            
        Returns:
            Dictionary with extracted metrics
        """
        metrics = {
            "workflow_id": workflow.workflow_id,
            "has_openevolve_data": False,
            "evolution_calls": 0,
            "quality_diversity_calls": 0,
            "ensemble_calls": 0,
            "fitness_improvement": None,
            "diversity_score": None,
            "metrics_by_stage": {}
        }
        
        # Check decomposition plan metrics
        if workflow.decomposition_plan and hasattr(workflow.decomposition_plan, 'openevolve_metrics'):
            if workflow.decomposition_plan.openevolve_metrics:
                metrics["has_openevolve_data"] = True
                metrics["metrics_by_stage"]["decomposition"] = workflow.decomposition_plan.openevolve_metrics
        
        # Check sub-problem solutions metrics
        if workflow.sub_problem_solutions:
            for sp_id, solution in workflow.sub_problem_solutions.items():
                if hasattr(solution, 'openevolve_metrics') and solution.openevolve_metrics:
                    metrics["has_openevolve_data"] = True
                    metrics["evolution_calls"] += 1
                    metrics["metrics_by_stage"][f"subproblem_{sp_id}"] = solution.openevolve_metrics
                    
                    # Extract fitness improvement if available
                    if "fitness_improvement" in solution.openevolve_metrics:
                        if metrics["fitness_improvement"] is None:
                            metrics["fitness_improvement"] = solution.openevolve_metrics["fitness_improvement"]
                        else:
                            metrics["fitness_improvement"] = (metrics["fitness_improvement"] + solution.openevolve_metrics["fitness_improvement"]) / 2
        
        # Check final solution metrics
        if workflow.final_solution and hasattr(workflow.final_solution, 'openevolve_metrics'):
            if workflow.final_solution.openevolve_metrics:
                metrics["has_openevolve_data"] = True
                metrics["metrics_by_stage"]["final_solution"] = workflow.final_solution.openevolve_metrics
        
        # Check critique reports for quality diversity metrics
        if workflow.all_critique_reports:
            for report in workflow.all_critique_reports:
                if hasattr(report, 'openevolve_metrics') and report.openevolve_metrics:
                    metrics["has_openevolve_data"] = True
                    metrics["quality_diversity_calls"] += 1
                    
                    if "diversity_score" in report.openevolve_metrics:
                        if metrics["diversity_score"] is None:
                            metrics["diversity_score"] = report.openevolve_metrics["diversity_score"]
                        else:
                            metrics["diversity_score"] = (metrics["diversity_score"] + report.openevolve_metrics["diversity_score"]) / 2
        
        # Check verification reports for ensemble metrics
        if workflow.all_verification_reports:
            for report in workflow.all_verification_reports:
                if hasattr(report, 'openevolve_metrics') and report.openevolve_metrics:
                    metrics["has_openevolve_data"] = True
                    metrics["ensemble_calls"] += 1
        
        return metrics
    
    def aggregate_metrics_by_timeframe(self, days: int = 30) -> Dict[str, Any]:
        """
        Aggregate OpenEvolve metrics for workflows within a timeframe.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Aggregated metrics dictionary
        """
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_workflows = []
        
        for workflow in self.get_all_historical_workflows():
            # Assuming workflow has a timestamp field
            if hasattr(workflow, 'created_at'):
                try:
                    wf_date = datetime.fromisoformat(workflow.created_at)
                    if wf_date >= cutoff_date:
                        recent_workflows.append(workflow)
                except (ValueError, TypeError, AttributeError):
                    # If date parsing fails, include the workflow
                    recent_workflows.append(workflow)
            else:
                # If no timestamp, include the workflow
                recent_workflows.append(workflow)
        
        # Aggregate metrics for recent workflows
        metrics = {
            "timeframe_days": days,
            "total_workflows": len(recent_workflows),
            "workflows_with_openevolve": 0,
            "total_evolution_calls": 0,
            "total_quality_diversity_calls": 0,
            "total_ensemble_calls": 0,
            "average_fitness_improvement": 0.0,
            "average_diversity_score": 0.0
        }
        
        fitness_improvements = []
        diversity_scores = []
        
        for wf in recent_workflows:
            wf_metrics = self._extract_openevolve_metrics_from_workflow(wf)
            if wf_metrics["has_openevolve_data"]:
                metrics["workflows_with_openevolve"] += 1
                metrics["total_evolution_calls"] += wf_metrics["evolution_calls"]
                metrics["total_quality_diversity_calls"] += wf_metrics["quality_diversity_calls"]
                metrics["total_ensemble_calls"] += wf_metrics["ensemble_calls"]
                
                if wf_metrics["fitness_improvement"] is not None:
                    fitness_improvements.append(wf_metrics["fitness_improvement"])
                if wf_metrics["diversity_score"] is not None:
                    diversity_scores.append(wf_metrics["diversity_score"])
        
        if fitness_improvements:
            metrics["average_fitness_improvement"] = sum(fitness_improvements) / len(fitness_improvements)
        if diversity_scores:
            metrics["average_diversity_score"] = sum(diversity_scores) / len(diversity_scores)
        
        return metrics
    
    def query_workflows_by_metrics(self, min_fitness: Optional[float] = None, 
                                   min_diversity: Optional[float] = None) -> List[WorkflowState]:
        """
        Query workflows that meet specified metric thresholds.
        
        Args:
            min_fitness: Minimum fitness improvement threshold
            min_diversity: Minimum diversity score threshold
            
        Returns:
            List of workflows meeting the criteria
        """
        matching_workflows = []
        
        for workflow in self.get_all_historical_workflows():
            wf_metrics = self._extract_openevolve_metrics_from_workflow(workflow)
            
            if not wf_metrics["has_openevolve_data"]:
                continue
            
            meets_criteria = True
            
            if min_fitness is not None:
                if wf_metrics["fitness_improvement"] is None or wf_metrics["fitness_improvement"] < min_fitness:
                    meets_criteria = False
            
            if min_diversity is not None:
                if wf_metrics["diversity_score"] is None or wf_metrics["diversity_score"] < min_diversity:
                    meets_criteria = False
            
            if meets_criteria:
                matching_workflows.append(workflow)

        return matching_workflows

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for History Manager
    # =========================================================================

    def _trigger_history_alerts(
        self,
        operation: str,
        success: bool,
        workflow_id: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for history operation failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            if not success:
                severity = AlertSeverity.MEDIUM

                alert_manager.create_alert(
                    title=f"Workflow History Operation Failed: {operation}",
                    description=f"History operation '{operation}' failed" +
                                 (f" for workflow '{workflow_id}'" if workflow_id else "") +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=severity.value,
                    source="workflow_history_manager",
                    component="history_tracking",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger History alert: {e}")

    def _extract_history_knowledge(
        self,
        workflow_state: WorkflowState,
        operation: str = "add_to_history"
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract workflow history knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"history_{workflow_state.workflow_id}_{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="workflow_history_entry",
                source_component="workflow_history_manager",
                title=f"Workflow History: {workflow_state.workflow_id} ({operation})",
                content={
                    "workflow_id": workflow_state.workflow_id,
                    "operation": operation,
                    "status": workflow_state.status,
                    "current_stage": workflow_state.current_stage,
                    "total_progress": workflow_state.total_progress,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "num_sub_problems": len(workflow_state.sub_problem_solutions) if workflow_state.sub_problem_solutions else 0,
                    "has_final_solution": workflow_state.final_solution is not None,
                    "has_decomposition_plan": workflow_state.decomposition_plan is not None
                },
                tags=["workflow", "history", operation, "archival"]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted History knowledge for {workflow_state.workflow_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract History knowledge: {e}")
            return False

    def _track_history_performance(
        self,
        operation: str,
        success: bool
    ):
        """**ACTUAL INTEGRATION**: Track history operation performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            performance_data = StrategyPerformanceData(
                strategy_name=f"history_manager_{operation}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=1.0 if success else 0.0,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={"operation": operation}
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked History performance for {operation}")

        except Exception as e:
            logger.error(f"Failed to track History performance: {e}")
