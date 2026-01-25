"""
Sub-Problem Node for BubbleLabs Integration

Manages individual sub-problems and their dependencies.
"""

from typing import Dict, Any, List, Optional
from .base_node import BubbleLabsNode, NodeExecutionError


class SubProblemNode(BubbleLabsNode):
    """
    Manages and executes individual sub-problems with dependency tracking.

    Handles:
    - Dependency resolution
    - Priority queue management
    - Parallel execution coordination
    - Resource allocation
    """

    # Node metadata
    DISPLAY_NAME = "Sub-Problem Processor"
    DESCRIPTION = (
        "Process individual sub-problems with dependency resolution "
        "and resource management."
    )
    ICON = "subproblem"
    CATEGORY = "processing"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Import sub-problem manager (safe import)
        SubProblemManager = self.safe_import(
            'workflow_structures.SubProblemManager',
            fallback_value=None,
            error_msg="SubProblemManager not available for SubProblemNode"
        )

        if SubProblemManager:
            try:
                self.manager = SubProblemManager()
            except Exception as e:
                self.logger.warning(f"Could not instantiate SubProblemManager: {e}")
                self.manager = None
        else:
            self.manager = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required:
            - sub_problem: Dict or SubProblem object

        Optional:
            - available_resources: List[str]
            - priority: int
            - timeout: int
        """
        errors = []

        # Check required fields
        if 'sub_problem' not in inputs:
            errors.append("Missing required field: sub_problem")
        elif not isinstance(inputs['sub_problem'], (dict, object)):
            errors.append("sub_problem must be a dictionary or SubProblem object")

        # Validate available_resources
        if 'available_resources' in inputs:
            if not isinstance(inputs['available_resources'], list):
                errors.append("available_resources must be a list")
            elif not all(isinstance(r, str) for r in inputs['available_resources']):
                errors.append("All resources must be strings")

        # Validate priority
        if 'priority' in inputs:
            if not isinstance(inputs['priority'], int):
                errors.append("priority must be an integer")
            elif inputs['priority'] < 1 or inputs['priority'] > 10:
                errors.append("priority must be between 1 and 10")

        # Validate timeout
        if 'timeout' in inputs:
            if not isinstance(inputs['timeout'], int):
                errors.append("timeout must be an integer")
            elif inputs['timeout'] < 0:
                errors.append("timeout must be non-negative")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute a sub-problem with dependency resolution.

        Args:
            inputs: Must contain 'sub_problem' and optional execution parameters
            context: Workflow state for tracking

        Returns:
            Dict containing:
                - solution: The solution to the sub-problem
                - dependencies_resolved: Whether dependencies were resolved
                - execution_time: Time taken to execute
                - resource_usage: Resource consumption metrics
        """
        if not self.manager:
            return self._execute_simple(inputs, context)

        sub_problem = inputs['sub_problem']
        available_resources = inputs.get('available_resources', self.config.get('available_resources', []))
        priority = inputs.get('priority', self.config.get('priority', 5))
        timeout = inputs.get('timeout', self.config.get('timeout', 300))

        # Update progress
        context.update_progress(10, "Initializing sub-problem processor")
        self.logger.info(f"Processing sub-problem with priority {priority}")

        try:
            # Resolve dependencies
            context.update_progress(20, "Resolving dependencies")

            dependencies_result = self.manager.resolve_dependencies(
                sub_problem=sub_problem,
                context=context
            )

            dependencies_resolved = dependencies_result.all_resolved
            unresolved_deps = dependencies_result.unresolved

            if not dependencies_resolved:
                self.logger.warning(f"Sub-problem has {len(unresolved_deps)} unresolved dependencies")

            # Execute sub-problem
            context.update_progress(40, "Executing sub-problem solution")

            execution_result = self.manager.execute(
                sub_problem=sub_problem,
                resources=available_resources,
                priority=priority,
                timeout=timeout,
                callback=lambda p, m: context.update_progress(40 + p * 0.5, m)
            )

            # Update progress
            context.update_progress(90, "Processing execution results")

            # Extract and format results
            result = {
                'solution': execution_result.solution,
                'dependencies_resolved': dependencies_resolved,
                'unresolved_dependencies': [str(d) for d in unresolved_deps],
                'execution_time': execution_result.execution_time,
                'resource_usage': {
                    'cpu_time': execution_result.cpu_time,
                    'memory_peak': execution_result.memory_peak,
                    'resources_used': execution_result.resources_used
                },
                'status': execution_result.status,
                'quality_metrics': execution_result.quality_metrics,
                'sub_problem_id': getattr(sub_problem, 'id', 'unknown')
            }

            # Add artifacts to context
            context.add_artifact('subproblem_execution', {
                'result': result,
                'sub_problem': sub_problem,
                'priority': priority
            })

            context.update_progress(
                100,
                f"Sub-problem complete in {result['execution_time']:.2f}s, "
                f"status: {result['status']}"
            )

            self.logger.info(
                f"Sub-problem executed: {result['status']}, "
                f"time: {result['execution_time']:.2f}s, "
                f"deps resolved: {dependencies_resolved}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Sub-problem execution failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Sub-problem execution failed: {str(e)}",
                details={
                    'sub_problem_id': getattr(sub_problem, 'id', 'unknown'),
                    'priority': priority,
                    'timeout': timeout,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _execute_simple(self, inputs: Dict, context) -> Dict[str, Any]:
        """Simple execution fallback when manager not available"""
        sub_problem = inputs['sub_problem']
        priority = inputs.get('priority', self.config.get('priority', 5))

        context.update_progress(10, "Using simple execution (manager not available)")

        import time
        start_time = time.time()

        # Extract sub-problem data if it's an object
        if hasattr(sub_problem, 'title'):
            title = sub_problem.title
            description = sub_problem.description
        else:
            title = sub_problem.get('title', 'Unknown')
            description = sub_problem.get('description', '')

        context.update_progress(30, "Processing sub-problem")

        # Simulate solution generation
        solution = {
            'title': title,
            'description': description,
            'solution': f"Solution for: {title}",
            'approach': 'simple_fallback',
            'priority': priority,
            'note': 'Full manager not available, using simple processing'
        }

        execution_time = time.time() - start_time

        result = {
            'solution': solution,
            'dependencies_resolved': True,
            'unresolved_dependencies': [],
            'execution_time': execution_time,
            'resource_usage': {
                'cpu_time': execution_time,
                'memory_peak': 0,
                'resources_used': []
            },
            'status': 'completed',
            'quality_metrics': {
                'completeness': 0.5,
                'correctness': 0.5
            },
            'sub_problem_id': getattr(sub_problem, 'id', sub_problem.get('id', 'unknown'))
        }

        context.update_progress(100, f"Simple execution complete in {execution_time:.2f}s")
        return result

    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters"""
        return {
            "type": "object",
            "title": "Sub-Problem Configuration",
            "description": "Configure sub-problem processing parameters",
            "properties": {
                "priority": {
                    "type": "integer",
                    "title": "Priority",
                    "description": "Execution priority (1=lowest, 10=highest)",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5
                },
                "timeout": {
                    "type": "integer",
                    "title": "Timeout",
                    "description": "Maximum execution time in seconds",
                    "minimum": 0,
                    "maximum": 3600,
                    "default": 300
                },
                "available_resources": {
                    "type": "array",
                    "title": "Available Resources",
                    "description": "List of available resource identifiers",
                    "items": {
                        "type": "string"
                    },
                    "uniqueItems": True,
                    "default": []
                },
                "parallel_execution": {
                    "type": "boolean",
                    "title": "Enable Parallel Execution",
                    "description": "Allow parallel execution with other sub-problems",
                    "default": True
                }
            },
            "required": ["priority", "timeout"]
        }
