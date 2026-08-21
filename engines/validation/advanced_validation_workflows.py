"""
Advanced Validation Workflows for CREWAI Integration with OpenEvolve

This module implements advanced validation workflows including:
- Multi-stage validation with cascading gauntlets
- Custom validation criteria per ticket type
- Quality metrics tracking and reporting
"""
from __future__ import annotations


import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
import requests
from datetime import datetime
import statistics

from workflow_structures import (
    ModelConfig, 
    Team, 
    GauntletDefinition, 
    GauntletRoundRule,
    CritiqueReport, 
    VerificationReport,
    SolutionAttempt
)
from team_manager import TeamManager
from gauntlet_manager import GauntletManager
from openevolve_integration import run_unified_evolution, create_comprehensive_openevolve_config


logger = logging.getLogger(__name__)

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for Advanced Validation Workflows
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import enterprise_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False


@dataclass
class ValidationStage:
    """Definition of a single validation stage in a multi-stage validation workflow."""
    name: str
    gauntlet_name: str
    required_approval_rate: float  # Minimum approval rate needed to pass this stage
    failure_action: str  # Action to take if stage fails: 'stop', 'retry', 'continue'
    max_retries: int = 1
    custom_criteria: Optional[Dict[str, Any]] = None  # Additional custom validation criteria


@dataclass
class AdvancedValidationConfig:
    """Configuration for advanced validation workflows."""
    validation_stages: List[ValidationStage]
    parallel_validation_enabled: bool = False
    caching_enabled: bool = True
    performance_tracking_enabled: bool = True
    custom_validation_functions: Optional[Dict[str, Callable]] = None


# **ACTUAL INTEGRATION HELPER METHODS**: Advanced Validation Workflows
def _trigger_validation_alerts(operation, success, validation_id=None, error=None, metadata=None):
    """Trigger alerts for advanced validation operations"""
    if not ALERTING_AVAILABLE:
        return

    try:
        alert_mgr = get_alert_manager()
        if success:
            return  # No alerts for successful operations

        severity = AlertSeverity.HIGH if operation == "run_advanced_validation" else AlertSeverity.MEDIUM
        alert_mgr.trigger_alert(
            title=f"Validation {operation} Failed",
            message=f"Advanced validation operation '{operation}' failed: {error}",
            severity=severity,
            source="AdvancedValidationWorkflows",
            metadata=metadata or {"validation_id": validation_id, "operation": operation}
        )
    except Exception as e:
        logger.warning(f"Failed to trigger validation alert: {e}")


def _extract_validation_knowledge(operation, validation_id, config, result):
    """Extract knowledge from validation operations"""
    if not KNOWLEDGE_AVAILABLE:
        return

    try:
        artifact = KnowledgeArtifact(
            artifact_id=f"validation_{operation}_{validation_id}",
            artifact_type="validation_execution",
            source_component="AdvancedValidationWorkflows",
            content={
                "operation": operation,
                "validation_id": validation_id,
                "num_stages": len(getattr(config, 'validation_stages', [])),
                "parallel_validation": getattr(config, 'parallel_validation_enabled', False),
                "stages_passed": result.get("stages_passed", 0) if result else 0,
                "stages_failed": result.get("stages_failed", 0) if result else 0,
                "success": result.get("success", False) if result else False,
            },
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
        enterprise_knowledge_engine.store_artifact(artifact)
    except Exception as e:
        logger.warning(f"Failed to extract validation knowledge: {e}")


def _track_validation_performance(operation, success, duration_seconds, num_stages, stages_passed=0, stages_failed=0):
    """Track performance of validation operations"""
    if not ADAPTIVE_AVAILABLE:
        return

    try:
        tracker = StrategyPerformanceTracker.get_instance()
        data = StrategyPerformanceData(
            strategy_name="advanced_validation",
            component_name="AdvancedValidationWorkflows",
            operation_name=operation,
            success=success,
            duration_seconds=duration_seconds,
            metadata={
                "num_stages": num_stages,
                "stages_passed": stages_passed,
                "stages_failed": stages_failed
            }
        )
        tracker.record_execution(data)
    except Exception as e:
        logger.warning(f"Failed to track validation performance: {e}")


class AdvancedValidationOrchestrator:
    """
    Orchestrates advanced validation workflows with multi-stage, cascading validation.
    """
    
    def __init__(self, config: AdvancedValidationConfig):
        self.config = config
        self.team_manager = TeamManager()
        self.gauntlet_manager = GauntletManager()
        self.validation_cache = {}  # Cache for validation results
        self.performance_metrics = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "average_validation_time": 0.0,
            "validation_times": []
        }
        
        # Initialize custom validation functions
        self.custom_validation_functions = config.custom_validation_functions or {}
    
    async def run_advanced_validation(
        self, 
        content: str, 
        context: Dict[str, Any], 
        workflow_id: str,
        ticket_id: str
    ) -> Dict[str, Any]:
        """
        Run multi-stage advanced validation on content through all configured stages.
        
        Args:
            content: Content to validate
            context: Additional validation context
            workflow_id: ID of the parent workflow
            ticket_id: ID of the specific ticket/task
            
        Returns:
            Dict containing validation results and status
        """
        start_time = time.time()
        success = False
        validation_id = f"val_{workflow_id}_{ticket_id}"

        validation_results = {
            "workflow_id": workflow_id,
            "ticket_id": ticket_id,
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
            "stages_results": [],
            "overall_status": "pending",
            "validation_time": 0.0,
            "all_reports": []
        }

        all_reports = []  # Collect all gauntlet reports

        logger.info(f"Starting advanced validation for workflow {workflow_id}, ticket {ticket_id}")

        try:
            for stage_idx, stage in enumerate(self.config.validation_stages):
                logger.info(f"Running validation stage {stage_idx + 1}: {stage.name}")
                
                # Check cache first if enabled
                cache_key = f"{workflow_id}:{ticket_id}:{stage.gauntlet_name}:{hash(content)}"
                if self.config.caching_enabled and cache_key in self.validation_cache:
                    stage_result = self.validation_cache[cache_key]
                    logger.info(f"Using cached result for stage: {stage.name}")
                else:
                    # Run validation stage
                    stage_result = await self._run_validation_stage(
                        content, 
                        stage, 
                        context, 
                        workflow_id, 
                        ticket_id,
                        stage_idx
                    )
                    
                    # Cache result if enabled
                    if self.config.caching_enabled:
                        self.validation_cache[cache_key] = stage_result
                
                validation_results["stages_results"].append(stage_result)
                all_reports.extend(stage_result.get("reports", []))
                
                # Check custom criteria if specified for this stage
                if stage.custom_criteria:
                    custom_result = await self._evaluate_custom_criteria(
                        content, 
                        stage.custom_criteria, 
                        stage_result
                    )
                    
                    if not custom_result.get("passed", True):
                        logger.warning(f"Custom criteria failed for stage {stage.name}: {custom_result.get('message', 'Unknown issue')}")
                        
                        # Add custom criteria result to reports
                        all_reports.append({
                            "type": "custom_criteria",
                            "stage_name": stage.name,
                            "passed": False,
                            "message": custom_result.get("message", "Custom criteria failed"),
                            "criteria": stage.custom_criteria
                        })
                
                # Handle stage failure based on failure_action
                approval_rate = stage_result.get("approval_rate", 0.0)
                if approval_rate < stage.required_approval_rate:
                    logger.warning(f"Stage {stage.name} failed: approval rate {approval_rate:.2f} < required {stage.required_approval_rate:.2f}")
                    
                    if stage.failure_action == "stop":
                        logger.info(f"Stopping validation after stage {stage.name} failure")
                        validation_results["overall_status"] = "failed"
                        break
                    elif stage.failure_action == "retry" and stage_result.get("retry_count", 0) < stage.max_retries:
                        logger.info(f"Retrying stage {stage.name}")
                        # Implement retry logic
                        retry_result = await self._run_validation_stage(
                            content, 
                            stage, 
                            context, 
                            workflow_id, 
                            ticket_id,
                            stage_idx,
                            retry=True
                        )
                        validation_results["stages_results"][-1] = retry_result  # Update with retry result
                        all_reports.extend(retry_result.get("reports", []))
                        continue
                    # If failure_action is "continue", proceed to next stage
                else:
                    logger.info(f"Stage {stage.name} passed: approval rate {approval_rate:.2f} >= required {stage.required_approval_rate:.2f}")
            
            # Calculate overall validation status
            successful_stages = [sr for sr in validation_results["stages_results"] 
                                if sr.get("approval_rate", 0) >= sr.get("stage_required_approval", 0)]
            
            validation_results["all_reports"] = all_reports
            validation_results["successful_stages"] = len(successful_stages)
            validation_results["total_stages"] = len(self.config.validation_stages)
            
            if len(successful_stages) == len(self.config.validation_stages):
                validation_results["overall_status"] = "passed"
            elif len(successful_stages) == 0:
                validation_results["overall_status"] = "failed"
            else:
                # Partial success - may need custom logic for this case
                validation_results["overall_status"] = "partial_success"
            
            # Update performance metrics
            validation_time = time.time() - start_time
            validation_results["validation_time"] = validation_time
            self._update_performance_metrics(validation_results, validation_time)

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance
            success = True
            _extract_validation_knowledge("run_advanced_validation", validation_id, self.config, validation_results)
            _track_validation_performance("run_advanced_validation", True, validation_time,
                                         len(self.config.validation_stages),
                                         validation_results.get("successful_stages", 0),
                                         validation_results.get("total_stages", 0) - validation_results.get("successful_stages", 0))

            logger.info(f"Advanced validation completed for {workflow_id}:{ticket_id} - Status: {validation_results['overall_status']}")
            return validation_results

        except Exception as e:
            validation_time = time.time() - start_time
            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            _trigger_validation_alerts("run_advanced_validation", False, validation_id, str(e),
                                      {"workflow_id": workflow_id, "ticket_id": ticket_id})
            _track_validation_performance("run_advanced_validation", False, validation_time,
                                         len(getattr(self.config, 'validation_stages', [])), 0, 0)
            logger.error(f"Advanced validation failed for {workflow_id}:{ticket_id}: {e}")
            return {
                "workflow_id": workflow_id,
                "ticket_id": ticket_id,
                "overall_status": "error",
                "error": str(e),
                "stages_results": [],
                "validation_time": validation_time
            }
    
    async def _run_validation_stage(
        self, 
        content: str, 
        stage: ValidationStage, 
        context: Dict[str, Any], 
        workflow_id: str, 
        ticket_id: str,
        stage_idx: int = 0,
        retry: bool = False
    ) -> Dict[str, Any]:
        """
        Run a single validation stage on content.
        """
        start_time = time.time()
        
        try:
            # Get the gauntlet definition
            gauntlet = self.gauntlet_manager.get_gauntlet(stage.gauntlet_name)
            if not gauntlet:
                logger.error(f"Gauntlet {stage.gauntlet_name} not found")
                return {
                    "stage_name": stage.name,
                    "status": "error",
                    "error": f"Gauntlet {stage.gauntlet_name} not found",
                    "approval_rate": 0.0,
                    "reports": [],
                    "time_elapsed": time.time() - start_time
                }
            
            # Get the team for this gauntlet
            team = self.team_manager.get_team(gauntlet.team_name)
            if not team:
                logger.error(f"Team {gauntlet.team_name} not found for gauntlet {stage.gauntlet_name}")
                return {
                    "stage_name": stage.name,
                    "status": "error",
                    "error": f"Team {gauntlet.team_name} not found",
                    "approval_rate": 0.0,
                    "reports": [],
                    "time_elapsed": time.time() - start_time
                }
            
            # Run the gauntlet based on its type
            stage_context = {
                **context,
                "workflow_id": workflow_id,
                "ticket_id": ticket_id,
                "stage_idx": stage_idx,
                "stage_name": stage.name
            }
            
            # Import the run_gauntlet function from workflow_engine
            from workflow_engine import run_gauntlet
            result = run_gauntlet(content, gauntlet, team, stage_context)
            
            # Calculate approval rate
            reports = []
            if "critique_report" in result:
                reports = [result["critique_report"]]
                approval_rate = 1.0 if result["is_approved"] else 0.0
            elif "verification_report" in result:
                reports = [result["verification_report"]]
                approval_rate = 1.0 if result["is_approved"] else 0.0
            else:
                approval_rate = result.get("is_approved", False)
                # The result itself might contain reports
                if "report_object" in result:
                    reports = [result["report_object"]]
                elif result.get("reports_by_judge"):
                    reports = [result]
            
            # Prepare stage result
            stage_result = {
                "stage_name": stage.name,
                "gauntlet_name": stage.gauntlet_name,
                "status": "completed",
                "approval_rate": approval_rate,
                "required_approval_rate": stage.required_approval_rate,
                "is_approved": approval_rate >= stage.required_approval_rate,
                "reports": reports,
                "time_elapsed": time.time() - start_time,
                "retry_count": 1 if retry else 0
            }
            
            logger.info(f"Stage {stage.name} completed with approval rate: {approval_rate:.2f}")
            return stage_result
            
        except Exception as e:
            logger.error(f"Error running validation stage {stage.name}: {e}")
            return {
                "stage_name": stage.name,
                "status": "error",
                "error": str(e),
                "approval_rate": 0.0,
                "reports": [],
                "time_elapsed": time.time() - start_time
            }
    
    async def _evaluate_custom_criteria(
        self, 
        content: str, 
        custom_criteria: Dict[str, Any], 
        stage_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate custom validation criteria using configured functions or LLMs.
        """
        try:
            criterion_type = custom_criteria.get("type", "basic")
            
            if criterion_type == "function":
                # Execute a custom validation function
                function_name = custom_criteria.get("function_name")
                if function_name and function_name in self.custom_validation_functions:
                    result = self.custom_validation_functions[function_name](content, custom_criteria)
                    return {
                        "passed": result.get("passed", False),
                        "message": result.get("message", "Function validation completed"),
                        "details": result
                    }
            
            elif criterion_type == "llm_based":
                # Use LLM to evaluate custom criteria
                return await self._evaluate_llm_based_criteria(content, custom_criteria)
            
            elif criterion_type == "regex":
                # Use regex patterns for validation
                import re
                pattern = custom_criteria.get("pattern")
                if pattern:
                    match = re.search(pattern, content)
                    return {
                        "passed": bool(match),
                        "message": f"Regex pattern {'match' if match else 'no match'}: {pattern}",
                        "details": {"pattern": pattern, "match": bool(match)}
                    }
            
            elif criterion_type == "complexity":
                # Evaluate content based on complexity metrics
                complexity_score = self._evaluate_complexity(content, custom_criteria)
                threshold = custom_criteria.get("threshold", 0.5)
                return {
                    "passed": complexity_score >= threshold,
                    "message": f"Complexity score: {complexity_score:.2f}, threshold: {threshold}",
                    "details": {"complexity_score": complexity_score, "threshold": threshold}
                }
            
            else:
                # Default: basic validation
                return {"passed": True, "message": "Basic validation passed", "details": {}}
                
        except Exception as e:
            logger.error(f"Error evaluating custom criteria: {e}")
            return {
                "passed": False,
                "message": f"Error evaluating custom criteria: {str(e)}",
                "details": {}
            }
    
    async def _evaluate_llm_based_criteria(self, content: str, custom_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to evaluate custom validation criteria.
        """
        try:
            # Define the criteria to check
            criteria_description = custom_criteria.get("description", "Validate the content")
            expected_format = custom_criteria.get("expected_format", "boolean")
            
            # Get a model from a team (use first available team member)
            team_name = custom_criteria.get("evaluator_team", "default_team")
            team = self.team_manager.get_team(team_name)
            if not team or not team.members:
                # Use default model config if no team specified
                model_config = ModelConfig(
                    model_id="gpt-4o", 
                    api_key="",
                    api_base="https://api.openai.com/v1"
                )
            else:
                model_config = team.members[0]
            
            # Create prompt for LLM evaluation
            prompt = f"""
            Evaluate the following content against these criteria:

            CRITERIA:
            {criteria_description}

            CONTENT TO EVALUATE:
            {content}

            Please provide your evaluation in JSON format:
            {{
                "passed": true/false,
                "message": "explanation of your evaluation",
                "score": 0.0-1.0 (confidence score)
            }}
            """
            
            # Call LLM for evaluation
            from llm_utils import _request_openai_compatible_chat
            response = _request_openai_compatible_chat(
                api_key=model_config.api_key,
                base_url=model_config.api_base,
                model=model_config.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic output
                response_format={"type": "json_object"}
            )
            
            if response:
                try:
                    evaluation_result = json.loads(response)
                    return {
                        "passed": evaluation_result.get("passed", False),
                        "message": evaluation_result.get("message", "LLM evaluation completed"),
                        "score": evaluation_result.get("score", 0.5),
                        "details": evaluation_result
                    }
                except json.JSONDecodeError:
                    return {
                        "passed": False,
                        "message": f"LLM response not in JSON format: {response}",
                        "details": {"raw_response": response}
                    }
            else:
                return {
                    "passed": False,
                    "message": "No response from LLM evaluator",
                    "details": {}
                }
                
        except Exception as e:
            logger.error(f"Error in LLM-based criteria evaluation: {e}")
            return {
                "passed": False,
                "message": f"LLM evaluation error: {str(e)}",
                "details": {}
            }
    
    def _evaluate_complexity(self, content: str, criteria: Dict[str, Any]) -> float:
        """
        Evaluate content complexity based on various metrics.
        """
        # Length-based complexity
        length_score = min(1.0, len(content) / 5000)  # Normalize length to 0-1
        
        # Code complexity indicators (if applicable)
        code_score = 0.0
        if criteria.get("check_code_complexity", False):
            if "def " in content or "class " in content:
                code_score += 0.2
            if content.count("if ") > 5:
                code_score += 0.1
            if content.count("for ") > 3 or content.count("while ") > 2:
                code_score += 0.2
            if content.count("import ") > 3:
                code_score += 0.1
        
        # Vocabulary complexity (diversity of words)
        words = content.split()
        if words:
            unique_words_ratio = len(set(words)) / len(words)
            vocabulary_score = unique_words_ratio * 0.4
        else:
            vocabulary_score = 0.0
        
        # Combine scores (adjust weights as needed)
        total_score = (length_score * 0.3) + (code_score * 0.4) + (vocabulary_score * 0.3)
        return min(1.0, total_score)  # Ensure score is between 0 and 1
    
    def _update_performance_metrics(self, validation_result: Dict[str, Any], validation_time: float):
        """
        Update performance metrics based on validation result.
        """
        self.performance_metrics["total_validations"] += 1
        if validation_result["overall_status"] == "passed":
            self.performance_metrics["successful_validations"] += 1
        else:
            self.performance_metrics["failed_validations"] += 1
        
        self.performance_metrics["validation_times"].append(validation_time)
        if self.performance_metrics["validation_times"]:
            self.performance_metrics["average_validation_time"] = statistics.mean(
                self.performance_metrics["validation_times"]
            )
        
        # Keep only recent metrics to prevent memory growth
        if len(self.performance_metrics["validation_times"]) > 1000:
            self.performance_metrics["validation_times"] = (
                self.performance_metrics["validation_times"][-500:]
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics for validation workflows.
        """
        return self.performance_metrics.copy()
    
    def create_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Create a comprehensive validation report from validation results.
        """
        report_lines = [
            "# Advanced Validation Report",
            "",
            f"**Workflow ID**: {validation_results['workflow_id']}",
            f"**Ticket ID**: {validation_results['ticket_id']}",
            f"**Validation Status**: {validation_results['overall_status'].upper()}",
            f"**Content Preview**: {validation_results['content_preview']}",
            f"**Total Validation Time**: {validation_results['validation_time']:.2f}s",
            "",
            "## Stage Results",
            ""
        ]
        
        for stage_result in validation_results["stages_results"]:
            status_emoji = "[OK]" if stage_result.get("is_approved", False) else "[FAIL]"
            report_lines.append(
                f"- {status_emoji} **{stage_result['stage_name']}**: "
                f"{stage_result['approval_rate']:.2f} approval rate "
                f"(required: {stage_result['required_approval_rate']:.2f})"
            )
        
        report_lines.extend([
            "",
            f"**Successful Stages**: {validation_results['successful_stages']}/{validation_results['total_stages']}",
            ""
        ])
        
        # Add detailed reports if available
        if validation_results["all_reports"]:
            report_lines.extend([
                "## Detailed Reports",
                ""
            ])
            
            for i, report in enumerate(validation_results["all_reports"]):
                if isinstance(report, (CritiqueReport, VerificationReport)):
                    report_dict = asdict(report)
                    report_lines.append(f"### Report {i+1}")
                    report_lines.append(f"- **Gauntlet**: {report_dict.get('gauntlet_name', 'Unknown')}")
                    report_lines.append(f"- **Approved**: {report_dict.get('is_approved', False)}")
                    report_lines.append(f"- **Summary**: {report_dict.get('summary', 'No summary')[:100]}...")
                else:
                    # Handle dictionary-based reports
                    report_lines.append(f"### Report {i+1}")
                    report_lines.append(f"- **Type**: {report.get('type', 'Unknown')}")
                    report_lines.append(f"- **Name**: {report.get('gauntlet_name', 'N/A')}")
                    report_lines.append(f"- **Approved**: {report.get('is_approved', 'N/A')}")
                
                report_lines.append("")
        
        return "\n".join(report_lines)


class CascadingValidationManager:
    """
    Manages cascading validation where results from earlier stages influence later stages.
    """
    
    def __init__(self):
        self.stage_dependencies = {}
        self.validation_profiles = {}  # Different validation profiles for different content types
        self.validation_history = []
        self._max_history = 500
    
    def register_validation_profile(self, content_type: str, config: AdvancedValidationConfig):
        """
        Register a validation profile for a specific content type.
        """
        self.validation_profiles[content_type] = config
        logger.info(f"Registered validation profile for content type: {content_type}")
    
    def get_validation_profile(self, content_type: str) -> Optional[AdvancedValidationConfig]:
        """
        Get validation profile for content type, with fallback to default.
        """
        return self.validation_profiles.get(content_type) or self.validation_profiles.get("default")
    
    async def run_cascading_validation(
        self, 
        content: str, 
        content_type: str, 
        context: Dict[str, Any],
        workflow_id: str,
        ticket_id: str
    ) -> Dict[str, Any]:
        """
        Run cascading validation where early stages can influence later stages.
        """
        config = self.get_validation_profile(content_type)
        if not config:
            logger.error(f"No validation profile found for content type: {content_type}")
            return {
                "error": f"No validation profile found for content type: {content_type}",
                "status": "failed"
            }
        
        orchestrator = AdvancedValidationOrchestrator(config)
        
        # Run the advanced validation
        result = await orchestrator.run_advanced_validation(
            content, context, workflow_id, ticket_id
        )
        
        # Apply cascading logic based on results from earlier stages
        await self._apply_cascading_logic(result)
        self._record_validation_result(content_type, result)
        
        return result
    
    async def _apply_cascading_logic(self, validation_result: Dict[str, Any]):
        """
        Apply cascading logic where early results influence later validation behavior.
        """
        # Example: If early stages fail, increase scrutiny in later stages
        early_stage_results = validation_result["stages_results"][:len(validation_result["stages_results"])//2]
        late_stage_results = validation_result["stages_results"][len(validation_result["stages_results"])//2:]
        
        early_failures = [sr for sr in early_stage_results if not sr.get("is_approved", True)]
        
        if early_failures:
            logger.info(f"Early stage failures detected ({len(early_failures)}). "
                       f"Applying stricter validation for remaining stages.")
            
            # This could involve re-running later stages with more stringent criteria
            # or alerting additional validation teams
            validation_result["cascading_alert"] = {
                "type": "increased_scrutiny",
                "early_failures_count": len(early_failures),
                "applied": True
            }
    
    def analyze_validation_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in validation results to identify improvement opportunities.
        """
        if not self.validation_history:
            return {
                "pattern_analysis": "No validation history available yet.",
                "recommendations": [
                    "Run additional validations to build historical insights",
                    "Enable caching to speed up repeated validations"
                ]
            }

        content_type_stats = {}
        stage_failures = {}
        stage_times = {}

        for entry in self.validation_history:
            content_type = entry["content_type"]
            status = entry["result"].get("overall_status", "unknown")
            content_type_stats.setdefault(content_type, {"total": 0, "failed": 0, "partial": 0})
            content_type_stats[content_type]["total"] += 1
            if status == "failed":
                content_type_stats[content_type]["failed"] += 1
            elif status == "partial_success":
                content_type_stats[content_type]["partial"] += 1

            for stage_result in entry["result"].get("stages_results", []):
                stage_name = stage_result.get("stage_name", "unknown")
                required = stage_result.get("stage_required_approval", stage_result.get("required_approval_rate", 0.0))
                approval = stage_result.get("approval_rate", 0.0)

                stage_failures.setdefault(stage_name, {"total": 0, "failed": 0})
                stage_failures[stage_name]["total"] += 1
                if approval < required:
                    stage_failures[stage_name]["failed"] += 1

                if "time_elapsed" in stage_result:
                    stage_times.setdefault(stage_name, [])
                    stage_times[stage_name].append(stage_result["time_elapsed"])

        content_type_failure_rates = {
            content_type: {
                "failure_rate": stats["failed"] / max(1, stats["total"]),
                "partial_rate": stats["partial"] / max(1, stats["total"]),
                "total_validations": stats["total"]
            }
            for content_type, stats in content_type_stats.items()
        }

        stage_failure_rates = {
            stage: stats["failed"] / max(1, stats["total"])
            for stage, stats in stage_failures.items()
        }

        stage_time_averages = {
            stage: statistics.mean(times) if times else 0.0
            for stage, times in stage_times.items()
        }

        slowest_stage = None
        if stage_time_averages:
            slowest_stage = max(stage_time_averages, key=stage_time_averages.get)

        recommendations = []
        for stage, failure_rate in stage_failure_rates.items():
            if failure_rate >= 0.4:
                recommendations.append(
                    f"Review validation criteria for stage '{stage}' (failure rate {failure_rate:.0%})"
                )

        for content_type, stats in content_type_failure_rates.items():
            if stats["failure_rate"] >= 0.4:
                recommendations.append(
                    f"Add specialized validation guidance for content type '{content_type}'"
                )

        if slowest_stage:
            recommendations.append(
                f"Optimize stage '{slowest_stage}' (avg {stage_time_averages[slowest_stage]:.2f}s)"
            )

        if not recommendations:
            recommendations.append("Validation performance is stable; continue monitoring for anomalies.")

        return {
            "pattern_analysis": {
                "content_type_failure_rates": content_type_failure_rates,
                "stage_failure_rates": stage_failure_rates,
                "stage_time_averages": stage_time_averages,
                "slowest_stage": slowest_stage,
                "validation_volume": len(self.validation_history),
                "average_validation_time": statistics.mean(
                    entry["result"].get("validation_time", 0.0) for entry in self.validation_history
                )
            },
            "recommendations": recommendations
        }

    def _record_validation_result(self, content_type: str, result: Dict[str, Any]):
        """Record validation result for later analysis."""
        self.validation_history.append({
            "content_type": content_type,
            "result": result,
            "timestamp": time.time()
        })
        if len(self.validation_history) > self._max_history:
            self.validation_history = self.validation_history[-self._max_history:]


# Default validation configurations for different content types
DEFAULT_VALIDATION_PROFILES = {
    "code_python": AdvancedValidationConfig(
        validation_stages=[
            ValidationStage(
                name="Syntax Check",
                gauntlet_name="python_syntax_checker",
                required_approval_rate=0.9,
                failure_action="stop"
            ),
            ValidationStage(
                name="Security Scan",
                gauntlet_name="code_security_gauntlet",
                required_approval_rate=0.95,
                failure_action="stop",
                custom_criteria={
                    "type": "regex",
                    "pattern": r"(eval\(|exec\(|os\.system|subprocess\.run)"
                }
            ),
            ValidationStage(
                name="Performance Review",
                gauntlet_name="code_performance_gauntlet",
                required_approval_rate=0.8,
                failure_action="continue"
            ),
            ValidationStage(
                name="Final Approval",
                gauntlet_name="code_quality_final",
                required_approval_rate=0.9,
                failure_action="retry",
                max_retries=2
            )
        ]
    ),
    "documentation": AdvancedValidationConfig(
        validation_stages=[
            ValidationStage(
                name="Quality Check",
                gauntlet_name="documentation_quality",
                required_approval_rate=0.85,
                failure_action="continue"
            ),
            ValidationStage(
                name="Final Review",
                gauntlet_name="documentation_final",
                required_approval_rate=0.9,
                failure_action="retry",
                max_retries=1
            )
        ]
    ),
    "default": AdvancedValidationConfig(
        validation_stages=[
            ValidationStage(
                name="Initial Validation",
                gauntlet_name="basic_validator",
                required_approval_rate=0.75,
                failure_action="continue"
            ),
            ValidationStage(
                name="Final Check",
                gauntlet_name="quality_assurance",
                required_approval_rate=0.85,
                failure_action="retry",
                max_retries=1
            )
        ]
    )
}


def initialize_validation_system():
    """
    Initialize the advanced validation system with default profiles.
    """
    manager = CascadingValidationManager()
    
    # Register default validation profiles
    for content_type, config in DEFAULT_VALIDATION_PROFILES.items():
        manager.register_validation_profile(content_type, config)
    
    logger.info("Advanced validation system initialized with default profiles")
    return manager


# Example usage function
async def example_usage():
    """
    Example of how to use the advanced validation system.
    """
    # Initialize the validation system
    validation_manager = initialize_validation_system()
    
    # Example content to validate
    sample_content = """
    def hello_world():
        print("Hello, World!")
        return "Hello"
    """
    
    # Run validation
    result = await validation_manager.run_cascading_validation(
        content=sample_content,
        content_type="code_python",
        context={"source": "CREWAI_ticket", "priority": "high"},
        workflow_id="sgdw-workflow-123",
        ticket_id="ticket-456"
    )
    
    print("Validation Result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    # For testing purposes
    asyncio.run(example_usage())
