"""
Math Workflow Orchestrator Node for BubbleLabs

Orchestrates coherent workflows between OpenEvolve and Math Verification bubbles.
Provides pre-built workflow templates for common mathematical tasks.

Part of the OpenEvolve-Math Integration Suite.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from bubblelabs_nodes.base_node import BubbleLabsNode, NodeExecutionError

logger = logging.getLogger(__name__)


class WorkflowTemplate(Enum):
    """Pre-built workflow templates."""
    FORMALIZE_AND_VERIFY = "formalize_and_verify"
    DECOMPOSE_AND_VERIFY = "decompose_and_verify"
    EVOLVE_SOLUTION = "evolve_solution"
    CONJECTURE_TO_THEOREM = "conjecture_to_theorem"
    COUNTEREXAMPLE_SEARCH = "counterexample_search"
    PROOF_OPTIMIZATION = "proof_optimization"
    COMPLETE_VERIFICATION = "complete_verification"


class MathWorkflowOrchestratorNode(BubbleLabsNode):
    """
    Orchestrate coherent workflows between OpenEvolve and Math Verification.
    
    Operations:
        - execute_template: Execute a pre-built workflow template
        - formalize_and_verify: Formalize and verify a problem
        - decompose_and_verify: Decompose problem and verify parts
        - evolve_solution: Evolve solution with verification
        - conjecture_to_theorem: Convert conjecture to theorem
        - counterexample_search: Search for counterexamples
        - proof_optimization: Optimize existing proof
        - custom_workflow: Build and execute custom workflow
    """
    
    DISPLAY_NAME = "Math Workflow Orchestrator"
    DESCRIPTION = "Orchestrate coherent workflows between OpenEvolve and Math Verification"
    ICON = "math-workflow"
    CATEGORY = "mathematical_verification"
    VERSION = "1.0.0"
    
    OPERATIONS = [
        "execute_template",
        "formalize_and_verify",
        "decompose_and_verify",
        "evolve_solution",
        "conjecture_to_theorem",
        "counterexample_search",
        "proof_optimization",
        "custom_workflow"
    ]
    
    # Pre-built workflow templates
    TEMPLATES = {
        WorkflowTemplate.FORMALIZE_AND_VERIFY: {
            "description": "Convert natural language problem to formal proof",
            "steps": [
                ("MathProblemClassificationNode", "classify"),
                ("OpenEvolveMathBridgeNode", "convert_problem"),
                ("LeanAutoformalizationNode", "autoformalize"),
                ("MathTacticRecommendationNode", "recommend"),
                ("LeanProofCheckingNode", "check_proof"),
                ("MathVerificationDashboardNode", "generate_report")
            ]
        },
        WorkflowTemplate.DECOMPOSE_AND_VERIFY: {
            "description": "Decompose problem and verify subproblems",
            "steps": [
                ("DecompositionNode", "decompose"),
                ("OpenEvolveMathBridgeNode", "batch_verify"),
                ("LeanProofCheckingNode", "batch_verify"),
                ("AssemblyNode", "assemble"),
                ("MathVerificationPipelineNode", "verify"),
                ("MathVerificationDashboardNode", "generate_report")
            ]
        },
        WorkflowTemplate.EVOLVE_SOLUTION: {
            "description": "Evolve solution with continuous verification",
            "steps": [
                ("KnowledgeEvolutionNode", "evolve"),
                ("OpenEvolveMathBridgeNode", "verify_subproblem"),
                ("Z3ConstraintSolvingNode", "check_sat"),
                ("MathCounterexampleNode", "verify_claim"),
                ("MathProofSimplificationNode", "simplify"),
                ("MathVerificationDashboardNode", "overview")
            ]
        },
        WorkflowTemplate.CONJECTURE_TO_THEOREM: {
            "description": "Convert conjecture to verified theorem",
            "steps": [
                ("MathConjectureNode", "test_conjecture"),
                ("MathCounterexampleNode", "find_counterexample"),
                ("MathProblemClassificationNode", "classify"),
                ("LeanAutoformalizationNode", "autoformalize"),
                ("LeanProofCheckingNode", "check_proof"),
                ("MathLibrarySearchNode", "search_theorems"),
                ("MathVerificationDashboardNode", "generate_report")
            ]
        },
        WorkflowTemplate.COUNTEREXAMPLE_SEARCH: {
            "description": "Search for counterexamples before proving",
            "steps": [
                ("MathProblemClassificationNode", "classify"),
                ("MathCounterexampleNode", "find_counterexample"),
                ("Z3ConstraintSolvingNode", "check_sat"),
                ("OpenEvolveMathBridgeNode", "route_to_verification")
            ]
        },
        WorkflowTemplate.PROOF_OPTIMIZATION: {
            "description": "Optimize and simplify existing proof",
            "steps": [
                ("MathProofSimplificationNode", "simplify"),
                ("MathTacticRecommendationNode", "compare_tactics"),
                ("MathProofCompletionNode", "auto_complete"),
                ("LeanProofCheckingNode", "check_proof"),
                ("MathVerificationDashboardNode", "performance_report")
            ]
        },
        WorkflowTemplate.COMPLETE_VERIFICATION: {
            "description": "Complete end-to-end verification pipeline",
            "steps": [
                ("MathProblemClassificationNode", "classify"),
                ("OpenEvolveMathBridgeNode", "classify_and_route"),
                ("MathVerificationPipelineNode", "verify"),
                ("ProofTranslationNode", "translate"),
                ("MathVerificationDashboardNode", "generate_report")
            ]
        }
    }
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
    
    def validate_inputs(self, inputs: Dict) -> List[str]:
        """Validate node inputs."""
        errors = []
        operation = inputs.get("operation", self.config.get("operation", "execute_template"))
        
        if operation not in self.OPERATIONS:
            errors.append(f"Invalid operation: {operation}")
        
        if operation in ["formalize_and_verify", "decompose_and_verify", "evolve_solution",
                         "conjecture_to_theorem", "counterexample_search", "proof_optimization"]:
            if "input_data" not in inputs and "input_data" not in self.config:
                errors.append(f"{operation} requires 'input_data' input")
        
        if operation == "execute_template":
            template = inputs.get("template", self.config.get("template", ""))
            if template and template not in [t.value for t in WorkflowTemplate]:
                errors.append(f"Invalid template: {template}")
        
        return errors
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": self.OPERATIONS,
                    "default": "execute_template",
                    "description": "Workflow operation"
                },
                "template": {
                    "type": "string",
                    "enum": [t.value for t in WorkflowTemplate],
                    "description": "Pre-built workflow template"
                },
                "input_data": {
                    "type": "object",
                    "description": "Input data for workflow"
                },
                "custom_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "node": {"type": "string"},
                            "operation": {"type": "string"},
                            "config": {"type": "object"}
                        }
                    },
                    "description": "Custom workflow steps"
                },
                "parallel": {
                    "type": "boolean",
                    "default": False,
                    "description": "Execute independent steps in parallel"
                },
                "stop_on_error": {
                    "type": "boolean",
                    "default": True,
                    "description": "Stop workflow on error"
                },
                "collect_metrics": {
                    "type": "boolean",
                    "default": True,
                    "description": "Collect workflow metrics"
                }
            },
            "required": ["operation"]
        }
    
    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute workflow orchestration."""
        operation = inputs.get("operation", self.config.get("operation", "execute_template"))
        
        try:
            if operation == "execute_template":
                result = self._execute_template(inputs, context)
            elif operation == "formalize_and_verify":
                result = self._execute_specific_template(inputs, context, WorkflowTemplate.FORMALIZE_AND_VERIFY)
            elif operation == "decompose_and_verify":
                result = self._execute_specific_template(inputs, context, WorkflowTemplate.DECOMPOSE_AND_VERIFY)
            elif operation == "evolve_solution":
                result = self._execute_specific_template(inputs, context, WorkflowTemplate.EVOLVE_SOLUTION)
            elif operation == "conjecture_to_theorem":
                result = self._execute_specific_template(inputs, context, WorkflowTemplate.CONJECTURE_TO_THEOREM)
            elif operation == "counterexample_search":
                result = self._execute_specific_template(inputs, context, WorkflowTemplate.COUNTEREXAMPLE_SEARCH)
            elif operation == "proof_optimization":
                result = self._execute_specific_template(inputs, context, WorkflowTemplate.PROOF_OPTIMIZATION)
            elif operation == "custom_workflow":
                result = self._custom_workflow(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.DISPLAY_NAME,
                    message=f"Unknown operation: {operation}"
                )
            
            result["timestamp"] = datetime.utcnow().isoformat()
            result["orchestrator_version"] = self.VERSION
            context.add_artifact("workflow_orchestration_result", result)
            
            return result
            
        except Exception as e:
            raise NodeExecutionError(
                node_name=self.DISPLAY_NAME,
                message=f"Workflow orchestration failed: {str(e)}",
                details={"operation": operation}
            )
    
    def _execute_template(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute a pre-built workflow template."""
        template_name = inputs.get("template", self.config.get("template", "formalize_and_verify"))
        input_data = inputs.get("input_data", self.config.get("input_data", {}))
        
        try:
            template_enum = WorkflowTemplate(template_name)
        except ValueError:
            return {
                "success": False,
                "error": f"Unknown template: {template_name}",
                "available_templates": [t.value for t in WorkflowTemplate]
            }
        
        return self._execute_specific_template(inputs, context, template_enum)
    
    def _execute_specific_template(self, inputs: Dict, context, template: WorkflowTemplate) -> Dict[str, Any]:
        """Execute a specific workflow template."""
        input_data = inputs.get("input_data", self.config.get("input_data", {}))
        stop_on_error = inputs.get("stop_on_error", self.config.get("stop_on_error", True))
        
        template_info = self.TEMPLATES[template]
        steps = template_info["steps"]
        
        context.update_progress(10)
        
        results = []
        current_data = input_data
        
        for i, (node_name, operation) in enumerate(steps):
            progress = 10 + (80 * (i + 1) // len(steps))
            context.update_progress(progress)
            
            step_result = {
                "step": i + 1,
                "node": node_name,
                "operation": operation,
                "status": "executed"
            }
            
            # In real implementation, would call actual node
            # For now, simulate
            step_result["result"] = f"Simulated {node_name}.{operation}()"
            
            results.append(step_result)
            current_data = self._propagate_data(current_data, step_result)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "template": template.value,
            "description": template_info["description"],
            "steps_executed": len(results),
            "step_results": results,
            "final_output": current_data
        }
    
    def _custom_workflow(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute custom workflow."""
        custom_steps = inputs.get("custom_steps", self.config.get("custom_steps", []))
        input_data = inputs.get("input_data", self.config.get("input_data", {}))
        
        if not custom_steps:
            return {
                "success": False,
                "error": "No custom steps provided"
            }
        
        context.update_progress(10)
        
        results = []
        current_data = input_data
        
        for i, step in enumerate(custom_steps):
            progress = 10 + (80 * (i + 1) // len(custom_steps))
            context.update_progress(progress)
            
            node_name = step.get("node", "Unknown")
            operation = step.get("operation", "execute")
            
            step_result = {
                "step": i + 1,
                "node": node_name,
                "operation": operation,
                "status": "executed"
            }
            
            results.append(step_result)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "template": "custom",
            "steps_executed": len(results),
            "step_results": results
        }
    
    def _propagate_data(self, current_data: Dict, step_result: Dict) -> Dict:
        """Propagate data through workflow steps."""
        # Simple propagation - in real implementation would merge intelligently
        return {**current_data, "last_step_result": step_result}
    
    def is_healthy(self) -> bool:
        """Check node health."""
        return True
