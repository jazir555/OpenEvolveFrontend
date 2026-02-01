"""
Workflow Orchestration Node for BubbleLabs

Orchestrates complex multi-step knowledge workflows with agent teams.
Supports CrewAI and LoongFlow integrations for agent orchestration,
workflow state management, dependency tracking, branching, loops,
and comprehensive execution monitoring.
"""

import uuid
import asyncio
import time
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime
from enum import Enum
from collections import deque

from .base_node import BubbleLabsNode, NodeExecutionError


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class StepStatus(Enum):
    """Individual step execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"  # Waiting for dependencies


class WorkflowOrchestrationNode(BubbleLabsNode):
    """
    Workflow Orchestration Node for BubbleLabs.

    Orchestrates complex multi-step knowledge workflows with agent teams:
    - Define complex multi-step workflows with DAG structure
    - Orchestrate agent teams via CrewAI/LoongFlow integrations
    - Manage workflow state, checkpoints, and dependencies
    - Handle workflow branching (conditional execution)
    - Support workflow loops (iteration)
    - Monitor workflow execution in real-time
    - Retry failed steps with configurable policies

    This node provides a unified interface for workflow orchestration
    across multiple backend systems.
    """

    # Node metadata
    DISPLAY_NAME = "Workflow Orchestration"
    DESCRIPTION = "Orchestrate complex multi-step knowledge workflows with agent teams"
    ICON = "orchestration"
    CATEGORY = "integration"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports with fallbacks
        self.CrewAIIntegration = self.safe_import(
            'knowledge_engine.integrations.crewai_integration.CrewAIIntegration',
            fallback_value=None,
            error_msg="CrewAIIntegration not available"
        )
        self.LoongFlowIntegration = self.safe_import(
            'knowledge_engine.integrations.loongflow_integration.LoongFlowIntegration',
            fallback_value=None,
            error_msg="LoongFlowIntegration not available"
        )
        self.WorkflowOrchestrator = self.safe_import(
            'knowledge_engine.workflow_orchestrator.WorkflowOrchestrator',
            fallback_value=None,
            error_msg="WorkflowOrchestrator not available"
        )

        # Initialize integrations
        self.crewai = None
        self.loongflow = None
        self.orchestrator = None

        if self.CrewAIIntegration:
            try:
                self.crewai = self.CrewAIIntegration()
                self.logger.info("CrewAI integration initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize CrewAI integration: {e}")

        if self.LoongFlowIntegration:
            try:
                self.loongflow = self.LoongFlowIntegration()
                self.logger.info("LoongFlow integration initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LoongFlow integration: {e}")

        # Internal workflow storage
        self._workflows: Dict[str, Dict[str, Any]] = {}
        self._step_handlers: Dict[str, Callable] = {}
        self._execution_logs: Dict[str, List[Dict]] = {}

        # Register built-in step handlers
        self._register_builtin_handlers()

        self.logger.info("WorkflowOrchestrationNode initialized")

    def _register_builtin_handlers(self):
        """Register built-in step type handlers"""
        self._step_handlers = {
            'crewai_task': self._handle_crewai_task,
            'loongflow_task': self._handle_loongflow_task,
            'conditional': self._handle_conditional,
            'loop': self._handle_loop,
            'parallel': self._handle_parallel,
            'sequential': self._handle_sequential,
            'custom': self._handle_custom,
            'noop': self._handle_noop,
        }

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters based on operation type.

        Required fields depend on operation:
        - define: workflow_definition
        - execute: workflow_definition or workflow_id
        - monitor: workflow_id
        - pause/resume/cancel/retry: workflow_id
        """
        errors = []

        operation = inputs.get('operation', self.config.get('operation', 'execute'))
        valid_operations = ['define', 'execute', 'monitor', 'pause', 'resume', 'cancel', 'retry']

        if operation not in valid_operations:
            errors.append(f"Invalid operation: {operation}. Must be one of: {', '.join(valid_operations)}")

        # Operation-specific validation
        if operation == 'define':
            if 'workflow_definition' not in inputs and 'workflow_definition' not in self.config:
                errors.append("Missing required field: workflow_definition for define operation")

        elif operation == 'execute':
            has_definition = 'workflow_definition' in inputs or 'workflow_definition' in self.config
            has_id = 'workflow_id' in inputs or 'workflow_id' in self.config
            if not has_definition and not has_id:
                errors.append("Execute operation requires either workflow_definition or workflow_id")

        elif operation in ['monitor', 'pause', 'resume', 'cancel', 'retry']:
            has_id = 'workflow_id' in inputs or 'workflow_id' in self.config
            if not has_id:
                errors.append(f"Missing required field: workflow_id for {operation} operation")

        # Validate workflow definition structure if provided
        if 'workflow_definition' in inputs:
            wf_def = inputs['workflow_definition']
            if not isinstance(wf_def, dict):
                errors.append("workflow_definition must be an object")
            else:
                if 'steps' not in wf_def:
                    errors.append("workflow_definition must contain 'steps' array")
                elif not isinstance(wf_def['steps'], list):
                    errors.append("workflow_definition.steps must be an array")
                elif len(wf_def['steps']) == 0:
                    errors.append("workflow_definition.steps cannot be empty")

        # Validate agents if provided
        if 'agents' in inputs:
            agents = inputs['agents']
            if not isinstance(agents, list):
                errors.append("agents must be an array")
            else:
                for i, agent in enumerate(agents):
                    if not isinstance(agent, dict):
                        errors.append(f"Agent at index {i} must be an object")
                    elif 'id' not in agent:
                        errors.append(f"Agent at index {i} must have an 'id' field")

        # Validate dependencies if provided
        if 'dependencies' in inputs:
            deps = inputs['dependencies']
            if not isinstance(deps, list):
                errors.append("dependencies must be an array")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute workflow orchestration based on operation type.

        Args:
            inputs: Input data containing operation and operation-specific parameters
            context: Workflow state for tracking progress

        Returns:
            Dict containing operation results with workflow status and metrics
        """
        operation = inputs.get('operation', self.config.get('operation', 'execute'))

        try:
            context.update_progress(10, f"Starting {operation} operation")

            if operation == 'define':
                result = self._execute_define(inputs, context)
            elif operation == 'execute':
                result = self._execute_execute(inputs, context)
            elif operation == 'monitor':
                result = self._execute_monitor(inputs, context)
            elif operation == 'pause':
                result = self._execute_pause(inputs, context)
            elif operation == 'resume':
                result = self._execute_resume(inputs, context)
            elif operation == 'cancel':
                result = self._execute_cancel(inputs, context)
            elif operation == 'retry':
                result = self._execute_retry(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': ['define', 'execute', 'monitor', 'pause', 'resume', 'cancel', 'retry']}
                )

            context.update_progress(100, f"{operation.capitalize()} operation completed")

            # Add artifact to context
            context.add_artifact('workflow_orchestration', {
                'operation': operation,
                'workflow_id': result.get('workflow_id'),
                'status': result.get('status'),
                'success': result.get('status') not in ['failed', 'cancelled']
            })

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Workflow orchestration {operation} failed: {e}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"{operation.capitalize()} operation failed: {str(e)}",
                details={
                    'operation': operation,
                    'inputs': {k: v for k, v in inputs.items() if k != 'workflow_definition'},
                    'exception_type': type(e).__name__
                }
            ) from e

    def _execute_define(self, inputs: Dict, context) -> Dict[str, Any]:
        """Define a new workflow without executing it."""
        workflow_def = inputs.get('workflow_definition', self.config.get('workflow_definition', {}))
        workflow_id = workflow_def.get('id', workflow_def.get('workflow_id', f"wf_{uuid.uuid4().hex[:12]}"))

        context.update_progress(30, "Validating workflow definition")

        # Validate workflow structure
        validation_result = self._validate_workflow_definition(workflow_def)
        if not validation_result['valid']:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message="Invalid workflow definition",
                details={'validation_errors': validation_result['errors']}
            )

        context.update_progress(60, "Storing workflow definition")

        # Store workflow
        self._workflows[workflow_id] = {
            'definition': workflow_def,
            'status': WorkflowStatus.PENDING.value,
            'created_at': datetime.now().isoformat(),
            'steps': {},
            'execution_log': []
        }

        # Initialize step statuses
        for step in workflow_def.get('steps', []):
            step_id = step.get('id', f"step_{uuid.uuid4().hex[:8]}")
            self._workflows[workflow_id]['steps'][step_id] = {
                'status': StepStatus.PENDING.value,
                'retries': 0,
                'started_at': None,
                'completed_at': None,
                'output': None,
                'error': None
            }

        context.update_progress(100, "Workflow defined successfully")

        return {
            'workflow_id': workflow_id,
            'status': WorkflowStatus.PENDING.value,
            'message': 'Workflow defined successfully',
            'total_steps': len(workflow_def.get('steps', [])),
            'validation': validation_result
        }

    def _execute_execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute a workflow (either new or existing)."""
        workflow_id = inputs.get('workflow_id', self.config.get('workflow_id'))
        workflow_def = inputs.get('workflow_definition', self.config.get('workflow_definition'))
        agents = inputs.get('agents', self.config.get('agents', []))
        dependencies = inputs.get('dependencies', self.config.get('dependencies', []))

        # Get execution parameters
        max_retries = inputs.get('max_retries', self.config.get('max_retries', 3))
        retry_delay = inputs.get('retry_delay', self.config.get('retry_delay', 5))
        timeout = inputs.get('timeout', self.config.get('timeout', 300))
        parallel_steps = inputs.get('parallel_steps', self.config.get('parallel_steps', True))
        checkpoint_interval = inputs.get('checkpoint_interval', self.config.get('checkpoint_interval', 10))

        # If workflow_id provided, use existing workflow
        if workflow_id and workflow_id in self._workflows:
            workflow = self._workflows[workflow_id]
            if workflow_def is None:
                workflow_def = workflow['definition']
        elif workflow_def:
            # Define new workflow first
            define_result = self._execute_define({
                'workflow_definition': workflow_def,
                'agents': agents,
                'dependencies': dependencies
            }, context)
            workflow_id = define_result['workflow_id']
        else:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message="No workflow definition or valid workflow_id provided",
                details={'hint': 'Provide either workflow_definition or a valid workflow_id'}
            )

        context.update_progress(20, f"Starting workflow execution: {workflow_id}")

        # Update workflow status
        self._workflows[workflow_id]['status'] = WorkflowStatus.RUNNING.value
        self._workflows[workflow_id]['started_at'] = datetime.now().isoformat()
        self._workflows[workflow_id]['execution_config'] = {
            'max_retries': max_retries,
            'retry_delay': retry_delay,
            'timeout': timeout,
            'parallel_steps': parallel_steps,
            'checkpoint_interval': checkpoint_interval
        }

        # Execute workflow
        try:
            result = self._run_workflow(
                workflow_id=workflow_id,
                context=context,
                parallel=parallel_steps,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                checkpoint_interval=checkpoint_interval
            )

            return result

        except Exception as e:
            self._workflows[workflow_id]['status'] = WorkflowStatus.FAILED.value
            self._workflows[workflow_id]['error'] = str(e)
            raise

    def _run_workflow(
        self,
        workflow_id: str,
        context,
        parallel: bool = True,
        timeout: int = 300,
        max_retries: int = 3,
        retry_delay: int = 5,
        checkpoint_interval: int = 10
    ) -> Dict[str, Any]:
        """Run the workflow execution logic."""
        workflow = self._workflows[workflow_id]
        workflow_def = workflow['definition']
        steps = workflow_def.get('steps', [])
        step_map = {step.get('id', f"step_{i}"): step for i, step in enumerate(steps)}

        start_time = time.time()
        checkpoint_counter = 0

        context.update_progress(30, "Building execution graph")

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(steps)
        execution_queue = deque(self._get_initial_steps(dependency_graph))
        running_steps: Set[str] = set()
        completed: Set[str] = set()
        failed: Set[str] = set()

        context.update_progress(40, f"Executing {len(steps)} steps")

        try:
            while execution_queue or running_steps:
                # Check timeout
                if time.time() - start_time > timeout:
                    workflow['status'] = WorkflowStatus.FAILED.value
                    raise NodeExecutionError(
                        node_name=self.get_display_name(),
                        message=f"Workflow execution timed out after {timeout} seconds",
                        details={'workflow_id': workflow_id, 'completed_steps': len(completed)}
                    )

                # Check if workflow was cancelled
                if workflow.get('status') == WorkflowStatus.CANCELLED.value:
                    return {
                        'workflow_id': workflow_id,
                        'status': WorkflowStatus.CANCELLED.value,
                        'completed_steps': len(completed),
                        'failed_steps': len(failed),
                        'message': 'Workflow was cancelled'
                    }

                # Check if workflow was paused
                if workflow.get('status') == WorkflowStatus.PAUSED.value:
                    return {
                        'workflow_id': workflow_id,
                        'status': WorkflowStatus.PAUSED.value,
                        'completed_steps': len(completed),
                        'failed_steps': len(failed),
                        'message': 'Workflow was paused'
                    }

                # Process queued steps
                while execution_queue and (parallel or not running_steps):
                    step_id = execution_queue.popleft()
                    step_def = step_map.get(step_id)

                    if not step_def:
                        continue

                    # Update step status
                    workflow['steps'][step_id]['status'] = StepStatus.RUNNING.value
                    workflow['steps'][step_id]['started_at'] = datetime.now().isoformat()
                    running_steps.add(step_id)

                    # Execute step synchronously
                    step_result = self._execute_step(workflow_id, step_id, step_def, context)
                    
                    # Process result
                    if step_result['success']:
                        workflow['steps'][step_id]['status'] = StepStatus.COMPLETED.value
                        workflow['steps'][step_id]['output'] = step_result.get('output')
                        workflow['steps'][step_id]['completed_at'] = datetime.now().isoformat()
                        completed.add(step_id)
                        running_steps.discard(step_id)

                        # Add dependent steps to queue
                        for dependent_id, deps in dependency_graph.items():
                            if step_id in deps and dependent_id not in completed and dependent_id not in failed:
                                if all(d in completed for d in deps):
                                    execution_queue.append(dependent_id)
                    else:
                        workflow['steps'][step_id]['status'] = StepStatus.FAILED.value
                        workflow['steps'][step_id]['error'] = step_result.get('error')
                        workflow['steps'][step_id]['completed_at'] = datetime.now().isoformat()
                        failed.add(step_id)
                        running_steps.discard(step_id)

                        # Retry logic
                        if workflow['steps'][step_id]['retries'] < max_retries:
                            workflow['steps'][step_id]['retries'] += 1
                            workflow['steps'][step_id]['status'] = StepStatus.PENDING.value
                            workflow['status'] = WorkflowStatus.RETRYING.value
                            time.sleep(retry_delay)
                            execution_queue.append(step_id)
                            failed.discard(step_id)

                # In parallel mode, process running steps
                if parallel and running_steps:
                    time.sleep(0.1)  # Brief pause to allow async operations
                    # Process any completed parallel steps
                    for step_id in list(running_steps):
                        step_status = workflow['steps'][step_id]['status']
                        if step_status in [StepStatus.COMPLETED.value, StepStatus.FAILED.value, StepStatus.SKIPPED.value]:
                            running_steps.discard(step_id)
                            if step_status == StepStatus.COMPLETED.value:
                                completed.add(step_id)
                                # Add dependent steps
                                for dependent_id, deps in dependency_graph.items():
                                    if step_id in deps and dependent_id not in completed and dependent_id not in failed:
                                        if all(d in completed for d in deps):
                                            execution_queue.append(dependent_id)
                            elif step_status == StepStatus.FAILED.value:
                                failed.add(step_id)
                                # Retry logic
                                if workflow['steps'][step_id]['retries'] < max_retries:
                                    workflow['steps'][step_id]['retries'] += 1
                                    workflow['steps'][step_id]['status'] = StepStatus.PENDING.value
                                    workflow['status'] = WorkflowStatus.RETRYING.value
                                    time.sleep(retry_delay)
                                    execution_queue.append(step_id)
                                    failed.discard(step_id)

                # Break if no more work to do
                if not execution_queue and not running_steps:
                    break

                # Create checkpoint if needed
                checkpoint_counter += 1
                if checkpoint_counter >= checkpoint_interval:
                    self._create_checkpoint(workflow_id, context)
                    checkpoint_counter = 0

                # Update progress
                progress = 40 + int((len(completed) / len(steps)) * 50) if steps else 40
                context.update_progress(min(progress, 90), f"Completed {len(completed)}/{len(steps)} steps")

            # Workflow completed
            final_status = WorkflowStatus.COMPLETED.value if not failed else WorkflowStatus.FAILED.value
            workflow['status'] = final_status
            workflow['completed_at'] = datetime.now().isoformat()

            return {
                'workflow_id': workflow_id,
                'status': final_status,
                'completed_steps': len(completed),
                'failed_steps': len(failed),
                'total_steps': len(steps),
                'execution_time': time.time() - start_time,
                'retries_used': sum(s['retries'] for s in workflow['steps'].values()),
                'steps': {k: {
                    'status': v['status'],
                    'retries': v['retries'],
                    'started_at': v['started_at'],
                    'completed_at': v['completed_at']
                } for k, v in workflow['steps'].items()}
            }

        except Exception as e:
            workflow['status'] = WorkflowStatus.FAILED.value
            raise

    def _build_dependency_graph(self, steps: List[Dict]) -> Dict[str, Set[str]]:
        """Build a dependency graph from step definitions."""
        graph = {}
        step_ids = {step.get('id', f"step_{i}") for i, step in enumerate(steps)}

        for i, step in enumerate(steps):
            step_id = step.get('id', f"step_{i}")
            deps = set()

            # Parse dependencies
            if 'depends_on' in step:
                if isinstance(step['depends_on'], list):
                    deps.update(step['depends_on'])
                elif isinstance(step['depends_on'], str):
                    deps.add(step['depends_on'])

            # Validate dependencies exist
            deps = deps.intersection(step_ids)
            graph[step_id] = deps

        return graph

    def _get_initial_steps(self, dependency_graph: Dict[str, Set[str]]) -> List[str]:
        """Get steps with no dependencies (can start immediately)."""
        return [step_id for step_id, deps in dependency_graph.items() if not deps]

    def _execute_step(
        self,
        workflow_id: str,
        step_id: str,
        step_def: Dict,
        context
    ) -> Dict[str, Any]:
        """Execute a single step."""
        step_type = step_def.get('type', 'custom')
        handler = self._step_handlers.get(step_type, self._handle_custom)

        try:
            result = handler(workflow_id, step_id, step_def, context)
            return {'success': True, 'output': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _execute_step_async(
        self,
        workflow_id: str,
        step_id: str,
        step_def: Dict,
        context
    ):
        """Execute a step asynchronously."""
        result = self._execute_step(workflow_id, step_id, step_def, context)
        workflow = self._workflows[workflow_id]

        if result['success']:
            workflow['steps'][step_id]['status'] = StepStatus.COMPLETED.value
            workflow['steps'][step_id]['output'] = result.get('output')
        else:
            workflow['steps'][step_id]['status'] = StepStatus.FAILED.value
            workflow['steps'][step_id]['error'] = result.get('error')

        workflow['steps'][step_id]['completed_at'] = datetime.now().isoformat()

    def _process_step_result(
        self,
        workflow_id: str,
        step_id: str,
        result: Dict,
        completed: Set[str],
        failed: Set[str],
        execution_queue: deque,
        dependency_graph: Dict,
        step_map: Dict
    ):
        """Process the result of a step execution."""
        workflow = self._workflows[workflow_id]

        if result['success']:
            workflow['steps'][step_id]['status'] = StepStatus.COMPLETED.value
            workflow['steps'][step_id]['output'] = result.get('output')
            completed.add(step_id)

            # Add dependent steps to queue
            for dependent_id, deps in dependency_graph.items():
                if step_id in deps and dependent_id not in completed and dependent_id not in failed:
                    if all(d in completed for d in deps):
                        execution_queue.append(dependent_id)
        else:
            workflow['steps'][step_id]['status'] = StepStatus.FAILED.value
            workflow['steps'][step_id]['error'] = result.get('error')
            failed.add(step_id)

            # Check if we should retry
            max_retries = workflow['execution_config'].get('max_retries', 3)
            if workflow['steps'][step_id]['retries'] < max_retries:
                workflow['steps'][step_id]['retries'] += 1
                workflow['steps'][step_id]['status'] = StepStatus.PENDING.value
                workflow['status'] = WorkflowStatus.RETRYING.value
                time.sleep(workflow['execution_config'].get('retry_delay', 5))
                execution_queue.append(step_id)

        workflow['steps'][step_id]['completed_at'] = datetime.now().isoformat()

    def _create_checkpoint(self, workflow_id: str, context):
        """Create a workflow checkpoint."""
        workflow = self._workflows[workflow_id]
        checkpoint_id = f"checkpoint_{uuid.uuid4().hex[:12]}"

        checkpoint = {
            'checkpoint_id': checkpoint_id,
            'workflow_id': workflow_id,
            'created_at': datetime.now().isoformat(),
            'status': workflow['status'],
            'steps': workflow['steps'].copy(),
            'execution_context': context.__dict__ if hasattr(context, '__dict__') else {}
        }

        if 'checkpoints' not in workflow:
            workflow['checkpoints'] = []
        workflow['checkpoints'].append(checkpoint)

        self.logger.info(f"Created checkpoint {checkpoint_id} for workflow {workflow_id}")

    def _execute_monitor(self, inputs: Dict, context) -> Dict[str, Any]:
        """Monitor a running or completed workflow."""
        workflow_id = inputs.get('workflow_id', self.config.get('workflow_id'))

        if not workflow_id or workflow_id not in self._workflows:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Workflow not found: {workflow_id}",
                details={'hint': 'Define or execute a workflow first'}
            )

        workflow = self._workflows[workflow_id]
        steps = workflow.get('steps', {})

        # Calculate statistics
        step_statuses = [s['status'] for s in steps.values()]
        completed = sum(1 for s in step_statuses if s == StepStatus.COMPLETED.value)
        failed = sum(1 for s in step_statuses if s == StepStatus.FAILED.value)
        running = sum(1 for s in step_statuses if s == StepStatus.RUNNING.value)
        pending = sum(1 for s in step_statuses if s == StepStatus.PENDING.value)

        context.update_progress(100, "Monitoring data retrieved")

        return {
            'workflow_id': workflow_id,
            'status': workflow.get('status'),
            'created_at': workflow.get('created_at'),
            'started_at': workflow.get('started_at'),
            'completed_at': workflow.get('completed_at'),
            'total_steps': len(steps),
            'completed_steps': completed,
            'failed_steps': failed,
            'running_steps': running,
            'pending_steps': pending,
            'progress_percentage': int((completed / len(steps)) * 100) if steps else 0,
            'steps': steps,
            'checkpoints': workflow.get('checkpoints', []),
            'execution_log': workflow.get('execution_log', [])
        }

    def _execute_pause(self, inputs: Dict, context) -> Dict[str, Any]:
        """Pause a running workflow."""
        workflow_id = inputs.get('workflow_id', self.config.get('workflow_id'))

        if not workflow_id or workflow_id not in self._workflows:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Workflow not found: {workflow_id}",
                details={}
            )

        workflow = self._workflows[workflow_id]

        if workflow['status'] != WorkflowStatus.RUNNING.value:
            return {
                'workflow_id': workflow_id,
                'status': workflow['status'],
                'message': f"Cannot pause workflow in {workflow['status']} state"
            }

        workflow['status'] = WorkflowStatus.PAUSED.value
        workflow['paused_at'] = datetime.now().isoformat()

        self._create_checkpoint(workflow_id, context)

        return {
            'workflow_id': workflow_id,
            'status': WorkflowStatus.PAUSED.value,
            'message': 'Workflow paused successfully',
            'paused_at': workflow['paused_at']
        }

    def _execute_resume(self, inputs: Dict, context) -> Dict[str, Any]:
        """Resume a paused workflow."""
        workflow_id = inputs.get('workflow_id', self.config.get('workflow_id'))
        checkpoint_id = inputs.get('checkpoint_id')

        if not workflow_id or workflow_id not in self._workflows:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Workflow not found: {workflow_id}",
                details={}
            )

        workflow = self._workflows[workflow_id]

        if workflow['status'] != WorkflowStatus.PAUSED.value:
            return {
                'workflow_id': workflow_id,
                'status': workflow['status'],
                'message': f"Cannot resume workflow in {workflow['status']} state"
            }

        # If checkpoint specified, restore from it
        if checkpoint_id:
            checkpoint = next(
                (c for c in workflow.get('checkpoints', []) if c['checkpoint_id'] == checkpoint_id),
                None
            )
            if checkpoint:
                workflow['steps'] = checkpoint['steps']

        workflow['status'] = WorkflowStatus.RUNNING.value
        workflow['resumed_at'] = datetime.now().isoformat()

        # Resume execution
        config = workflow.get('execution_config', {})
        return self._run_workflow(
            workflow_id=workflow_id,
            context=context,
            parallel=config.get('parallel_steps', True),
            timeout=config.get('timeout', 300),
            max_retries=config.get('max_retries', 3),
            retry_delay=config.get('retry_delay', 5),
            checkpoint_interval=config.get('checkpoint_interval', 10)
        )

    def _execute_cancel(self, inputs: Dict, context) -> Dict[str, Any]:
        """Cancel a running or paused workflow."""
        workflow_id = inputs.get('workflow_id', self.config.get('workflow_id'))

        if not workflow_id or workflow_id not in self._workflows:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Workflow not found: {workflow_id}",
                details={}
            )

        workflow = self._workflows[workflow_id]

        if workflow['status'] not in [WorkflowStatus.RUNNING.value, WorkflowStatus.PAUSED.value, WorkflowStatus.RETRYING.value]:
            return {
                'workflow_id': workflow_id,
                'status': workflow['status'],
                'message': f"Cannot cancel workflow in {workflow['status']} state"
            }

        workflow['status'] = WorkflowStatus.CANCELLED.value
        workflow['cancelled_at'] = datetime.now().isoformat()

        return {
            'workflow_id': workflow_id,
            'status': WorkflowStatus.CANCELLED.value,
            'message': 'Workflow cancelled successfully',
            'cancelled_at': workflow['cancelled_at']
        }

    def _execute_retry(self, inputs: Dict, context) -> Dict[str, Any]:
        """Retry failed steps in a workflow."""
        workflow_id = inputs.get('workflow_id', self.config.get('workflow_id'))
        step_ids = inputs.get('step_ids')  # Optional: specific steps to retry

        if not workflow_id or workflow_id not in self._workflows:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Workflow not found: {workflow_id}",
                details={}
            )

        workflow = self._workflows[workflow_id]
        steps = workflow.get('steps', {})

        # Find failed steps to retry
        if step_ids:
            steps_to_retry = [s for s in step_ids if s in steps and steps[s]['status'] == StepStatus.FAILED.value]
        else:
            steps_to_retry = [s_id for s_id, s in steps.items() if s['status'] == StepStatus.FAILED.value]

        if not steps_to_retry:
            return {
                'workflow_id': workflow_id,
                'message': 'No failed steps to retry',
                'retried_steps': []
            }

        # Reset failed steps to pending
        for step_id in steps_to_retry:
            steps[step_id]['status'] = StepStatus.PENDING.value
            steps[step_id]['error'] = None
            steps[step_id]['retries'] += 1

        workflow['status'] = WorkflowStatus.RETRYING.value

        # Re-run workflow
        config = workflow.get('execution_config', {})
        return self._run_workflow(
            workflow_id=workflow_id,
            context=context,
            parallel=config.get('parallel_steps', True),
            timeout=config.get('timeout', 300),
            max_retries=config.get('max_retries', 3),
            retry_delay=config.get('retry_delay', 5),
            checkpoint_interval=config.get('checkpoint_interval', 10)
        )

    # Step handlers

    def _handle_crewai_task(self, workflow_id: str, step_id: str, step_def: Dict, context) -> Any:
        """Handle CrewAI task execution."""
        if not self.crewai:
            raise Exception("CrewAI integration not available")

        task_config = step_def.get('config', {})
        agent_id = task_config.get('agent_id')
        task_description = task_config.get('description', step_def.get('name', 'Unnamed task'))

        # Execute via CrewAI integration
        try:
            result = self.crewai.execute_task(
                agent_id=agent_id,
                task=task_description,
                context=task_config.get('context', {})
            )
            return result
        except Exception as e:
            self.logger.error(f"CrewAI task failed: {e}")
            raise

    def _handle_loongflow_task(self, workflow_id: str, step_id: str, step_def: Dict, context) -> Any:
        """Handle LoongFlow task execution."""
        if not self.loongflow:
            raise Exception("LoongFlow integration not available")

        task_config = step_def.get('config', {})

        try:
            result = self.loongflow.execute_task(
                task_type=task_config.get('task_type', 'default'),
                parameters=task_config.get('parameters', {})
            )
            return result
        except Exception as e:
            self.logger.error(f"LoongFlow task failed: {e}")
            raise

    def _handle_conditional(self, workflow_id: str, step_id: str, step_def: Dict, context) -> Any:
        """Handle conditional branching."""
        config = step_def.get('config', {})
        condition = config.get('condition')
        true_branch = config.get('true_branch')
        false_branch = config.get('false_branch')

        # Evaluate condition (simplified - production would use expression evaluator)
        condition_result = self._evaluate_condition(condition, workflow_id, context)

        if condition_result and true_branch:
            return {'branch_taken': 'true', 'next_steps': true_branch}
        elif not condition_result and false_branch:
            return {'branch_taken': 'false', 'next_steps': false_branch}

        return {'branch_taken': 'none'}

    def _handle_loop(self, workflow_id: str, step_id: str, step_def: Dict, context) -> Any:
        """Handle loop iteration."""
        config = step_def.get('config', {})
        loop_type = config.get('loop_type', 'for')  # for, while, foreach
        iterations = config.get('iterations', [])
        loop_body = config.get('body', [])

        results = []

        if loop_type == 'foreach' and iterations:
            for item in iterations:
                for body_step in loop_body:
                    result = self._execute_step(workflow_id, f"{step_id}_iter", body_step, context)
                    results.append(result)
        elif loop_type == 'for':
            count = config.get('count', 1)
            for i in range(count):
                for body_step in loop_body:
                    result = self._execute_step(workflow_id, f"{step_id}_iter_{i}", body_step, context)
                    results.append(result)

        return {'iterations': len(results), 'results': results}

    def _handle_parallel(self, workflow_id: str, step_id: str, step_def: Dict, context) -> Any:
        """Handle parallel step execution."""
        config = step_def.get('config', {})
        parallel_steps = config.get('steps', [])

        results = []
        for p_step in parallel_steps:
            p_step_id = p_step.get('id', f"{step_id}_parallel_{uuid.uuid4().hex[:6]}")
            result = self._execute_step(workflow_id, p_step_id, p_step, context)
            results.append(result)

        return {'parallel_results': results}

    def _handle_sequential(self, workflow_id: str, step_id: str, step_def: Dict, context) -> Any:
        """Handle sequential step execution."""
        config = step_def.get('config', {})
        sequential_steps = config.get('steps', [])

        results = []
        for s_step in sequential_steps:
            s_step_id = s_step.get('id', f"{step_id}_seq_{uuid.uuid4().hex[:6]}")
            result = self._execute_step(workflow_id, s_step_id, s_step, context)
            results.append(result)

        return {'sequential_results': results}

    def _handle_custom(self, workflow_id: str, step_id: str, step_def: Dict, context) -> Any:
        """Handle custom step execution."""
        config = step_def.get('config', {})
        action = config.get('action', 'noop')

        # Custom action handling
        if action == 'log':
            message = config.get('message', 'Custom log message')
            self.logger.info(f"[Workflow {workflow_id}] {message}")
            return {'logged': message}
        elif action == 'transform':
            input_data = config.get('input', {})
            transform_type = config.get('transform_type', 'identity')
            return {'transformed': input_data, 'type': transform_type}

        return {'action': action, 'status': 'completed'}

    def _handle_noop(self, workflow_id: str, step_id: str, step_def: Dict, context) -> Any:
        """Handle no-op step."""
        return {'status': 'skipped', 'message': 'No operation performed'}

    def _evaluate_condition(self, condition: Any, workflow_id: str, context) -> bool:
        """Evaluate a condition expression."""
        if condition is None:
            return True

        if isinstance(condition, bool):
            return condition

        if isinstance(condition, str):
            # Simple string evaluation - production would use proper expression parser
            condition = condition.lower()
            if condition in ('true', 'yes', '1'):
                return True
            if condition in ('false', 'no', '0'):
                return False

        return bool(condition)

    def _validate_workflow_definition(self, workflow_def: Dict) -> Dict[str, Any]:
        """Validate workflow definition structure."""
        errors = []
        warnings = []

        steps = workflow_def.get('steps', [])

        # Check for duplicate step IDs
        step_ids = [step.get('id', f"step_{i}") for i, step in enumerate(steps)]
        if len(step_ids) != len(set(step_ids)):
            errors.append("Duplicate step IDs found")

        # Validate each step
        for i, step in enumerate(steps):
            step_id = step.get('id', f"step_{i}")

            if 'type' not in step:
                warnings.append(f"Step {step_id}: No type specified, using 'custom'")

            if step.get('type') == 'crewai_task' and not self.crewai:
                warnings.append(f"Step {step_id}: CrewAI task but CrewAI not available")

            if step.get('type') == 'loongflow_task' and not self.loongflow:
                warnings.append(f"Step {step_id}: LoongFlow task but LoongFlow not available")

        # Check for circular dependencies
        dependency_graph = self._build_dependency_graph(steps)
        if self._detect_cycle(dependency_graph):
            errors.append("Circular dependency detected in workflow")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def _detect_cycle(self, graph: Dict[str, Set[str]]) -> bool:
        """Detect cycles in dependency graph using DFS."""
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True

        return False

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns schema for UI configuration with all operation types and parameters.
        """
        return {
            "type": "object",
            "title": "Workflow Orchestration Configuration",
            "description": "Configure complex multi-step workflow orchestration with agent teams",
            "required": ["operation"],
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "The workflow operation to perform",
                    "enum": ["define", "execute", "monitor", "pause", "resume", "cancel", "retry"],
                    "enumNames": [
                        "Define - Define a new workflow without executing",
                        "Execute - Execute a workflow (define if needed)",
                        "Monitor - Get status of a running/completed workflow",
                        "Pause - Pause a running workflow",
                        "Resume - Resume a paused workflow",
                        "Cancel - Cancel a running or paused workflow",
                        "Retry - Retry failed steps in a workflow"
                    ],
                    "default": "execute"
                },
                "workflow_definition": {
                    "type": "object",
                    "title": "Workflow Definition",
                    "description": "DAG definition of the workflow steps and their dependencies",
                    "properties": {
                        "id": {
                            "type": "string",
                            "title": "Workflow ID",
                            "description": "Optional unique identifier for the workflow"
                        },
                        "name": {
                            "type": "string",
                            "title": "Workflow Name",
                            "description": "Human-readable name for the workflow"
                        },
                        "description": {
                            "type": "string",
                            "title": "Description",
                            "description": "Description of what the workflow does"
                        },
                        "steps": {
                            "type": "array",
                            "title": "Workflow Steps",
                            "description": "Array of workflow steps",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {
                                        "type": "string",
                                        "title": "Step ID"
                                    },
                                    "name": {
                                        "type": "string",
                                        "title": "Step Name"
                                    },
                                    "type": {
                                        "type": "string",
                                        "enum": ["crewai_task", "loongflow_task", "conditional", "loop", "parallel", "sequential", "custom", "noop"],
                                        "title": "Step Type"
                                    },
                                    "depends_on": {
                                        "type": ["string", "array"],
                                        "title": "Dependencies",
                                        "description": "Step ID(s) that must complete before this step"
                                    },
                                    "config": {
                                        "type": "object",
                                        "title": "Step Configuration"
                                    }
                                }
                            }
                        }
                    }
                },
                "workflow_id": {
                    "type": "string",
                    "title": "Workflow ID",
                    "description": "ID of an existing workflow (for monitor, pause, resume, cancel, retry operations)"
                },
                "agents": {
                    "type": "array",
                    "title": "Agent Configurations",
                    "description": "Configuration for agent teams",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "title": "Agent ID"
                            },
                            "type": {
                                "type": "string",
                                "enum": ["crewai", "loongflow"],
                                "title": "Agent Type"
                            },
                            "role": {
                                "type": "string",
                                "title": "Agent Role"
                            },
                            "config": {
                                "type": "object",
                                "title": "Agent Configuration"
                            }
                        }
                    },
                    "default": []
                },
                "dependencies": {
                    "type": "array",
                    "title": "Step Dependencies",
                    "description": "Explicit step dependencies",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_id": {
                                "type": "string",
                                "title": "Step ID"
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                                "title": "Depends On"
                            }
                        }
                    },
                    "default": []
                },
                "max_retries": {
                    "type": "integer",
                    "title": "Max Retries",
                    "description": "Maximum number of retries for failed steps",
                    "minimum": 0,
                    "maximum": 10,
                    "default": 3
                },
                "retry_delay": {
                    "type": "integer",
                    "title": "Retry Delay",
                    "description": "Delay in seconds between retries",
                    "minimum": 1,
                    "maximum": 300,
                    "default": 5
                },
                "timeout": {
                    "type": "integer",
                    "title": "Timeout",
                    "description": "Maximum execution time in seconds",
                    "minimum": 10,
                    "maximum": 3600,
                    "default": 300
                },
                "parallel_steps": {
                    "type": "boolean",
                    "title": "Parallel Steps",
                    "description": "Enable parallel execution of independent steps",
                    "default": True
                },
                "checkpoint_interval": {
                    "type": "integer",
                    "title": "Checkpoint Interval",
                    "description": "Create checkpoint every N steps",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                }
            }
        }

    def is_healthy(self) -> bool:
        """Check if the node is healthy and ready to execute."""
        try:
            # Check if at least one integration is available
            has_crewai = self.crewai is not None
            has_loongflow = self.loongflow is not None

            # Basic functionality requires neither, but having at least one is better
            return True
        except Exception:
            return False

    def get_available_integrations(self) -> Dict[str, bool]:
        """Get status of available integrations."""
        return {
            'crewai': self.crewai is not None,
            'loongflow': self.loongflow is not None,
            'workflow_orchestrator': self.WorkflowOrchestrator is not None
        }
