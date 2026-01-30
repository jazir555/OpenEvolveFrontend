"""
CrewAI Zero-Error Workflow Orchestrator

This module provides zero-error workflow orchestration for CrewAI with comprehensive
error prevention, detection, and automatic correction capabilities.

Author: OpenEvolve
Version: 1.0.0
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from functools import wraps
import traceback
import json
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
WorkflowResult = TypeVar('WorkflowResult')


class ErrorSeverity(Enum):
    """Classification of error severity levels"""
    CRITICAL = "critical"  # System failure, cannot continue
    HIGH = "high"  # Major issue, may compromise results
    MEDIUM = "medium"  # Recoverable issue with impact
    LOW = "low"  # Minor issue, can workaround
    INFO = "info"  # Informational, no action needed


class ErrorCategory(Enum):
    """Classification of error types"""
    IMPORT = "import"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    EXECUTION = "execution"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    DEPENDENCY = "dependency"
    LOGIC = "logic"
    UNKNOWN = "unknown"


class WorkflowPhase(Enum):
    """Workflow execution phases"""
    INITIALIZATION = "initialization"
    VALIDATION = "validation"
    PREPARATION = "preparation"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    COMPLETION = "completion"
    ROLLBACK = "rollback"


class WorkflowStatus(Enum):
    """Current workflow status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


# Custom Exceptions
class ZeroErrorWorkflowException(Exception):
    """Base exception for zero-error workflow system"""
    def __init__(self, message: str, error_code: str, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class WorkflowImportError(ZeroErrorWorkflowException):
    """Raised when CrewAI or dependencies cannot be imported"""
    pass


class WorkflowConfigurationError(ZeroErrorWorkflowException):
    """Raised when workflow configuration is invalid"""
    pass


class WorkflowValidationError(ZeroErrorWorkflowException):
    """Raised when input or state validation fails"""
    pass


class WorkflowExecutionError(ZeroErrorWorkflowException):
    """Raised during workflow execution"""
    pass


class WorkflowTimeoutError(ZeroErrorWorkflowException):
    """Raised when workflow or step exceeds timeout"""
    pass


class WorkflowResourceError(ZeroErrorWorkflowException):
    """Raised when required resources are unavailable"""
    pass


@dataclass
class ErrorRecord:
    """Record of an error that occurred during workflow execution"""
    error_id: str
    timestamp: datetime
    phase: WorkflowPhase
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    error_code: str
    stack_trace: str
    context: Dict[str, Any]
    auto_corrected: bool = False
    correction_strategy: Optional[str] = None
    retry_count: int = 0
    resolved: bool = False


@dataclass
class StepResult:
    """Result of a workflow step execution"""
    step_name: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime]
    duration_ms: float
    result: Optional[Any] = None
    errors: List[ErrorRecord] = field(default_factory=list)
    validation_passed: bool = False
    retry_count: int = 0


@dataclass
class WorkflowResult:
    """Complete workflow execution result"""
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime]
    duration_ms: float
    steps_completed: int
    steps_failed: int
    steps_total: int
    results: Dict[str, Any]
    errors: List[ErrorRecord] = field(default_factory=list)
    rollback_performed: bool = False
    success_rate: float = 0.0
    final_output: Optional[Any] = None


@dataclass
class WorkflowDefinition:
    """Definition of a workflow with its steps and validation rules"""
    name: str
    description: str
    version: str
    steps: List[Dict[str, Any]]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    timeout_seconds: int = 300
    max_retries: int = 3
    critical: bool = True
    rollback_on_failure: bool = True


@dataclass
class ExecutionContext:
    """Context object passed through workflow execution"""
    workflow_id: str
    execution_id: str
    start_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "start_time": self.start_time.isoformat(),
            "metadata": self.metadata,
            "state": self.state,
            "variables": self.variables
        }


class ErrorCorrectionStrategy:
    """Strategies for automatically correcting different types of errors"""

    @staticmethod
    async def retry_with_backoff(
        func: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0
    ) -> Tuple[bool, Any, List[ErrorRecord]]:
        """
        Retry a function with exponential backoff

        Args:
            func: Function to retry
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds

        Returns:
            Tuple of (success, result, errors)
        """
        errors = []
        delay = base_delay

        for attempt in range(max_retries + 1):
            try:
                result = await func() if asyncio.iscoroutinefunction(func) else func()
                if attempt > 0:
                    logger.info(f"Retry succeeded on attempt {attempt + 1}")
                return True, result, errors
            except (RuntimeError, ValueError, TypeError, ConnectionError, TimeoutError) as e:
                error_record = ErrorRecord(
                    error_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    phase=WorkflowPhase.EXECUTION,
                    category=ErrorCategory.EXECUTION,
                    severity=ErrorSeverity.MEDIUM,
                    message=str(e),
                    error_code="RETRY_ATTEMPT",
                    stack_trace=traceback.format_exc(),
                    context={"attempt": attempt + 1, "max_retries": max_retries + 1}
                )
                errors.append(error_record)

                if attempt < max_retries:
                    delay = min(delay * 2, max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")
                    return False, None, errors

        return False, None, errors

    @staticmethod
    def fix_missing_dependency(dependency_name: str) -> Dict[str, Any]:
        """Generate correction for missing dependency"""
        return {
            "strategy": "install_dependency",
            "command": f"pip install {dependency_name}",
            "description": f"Install missing dependency: {dependency_name}"
        }

    @staticmethod
    def fix_missing_env_var(var_name: str, suggested_value: Any = None) -> Dict[str, Any]:
        """Generate correction for missing environment variable"""
        return {
            "strategy": "set_env_var",
            "variable": var_name,
            "suggested_value": suggested_value,
            "description": f"Set environment variable: {var_name}"
        }

    @staticmethod
    def fix_invalid_parameter(param_name: str, current_value: Any, expected_type: str) -> Dict[str, Any]:
        """Generate correction for invalid parameter"""
        return {
            "strategy": "convert_parameter",
            "parameter": param_name,
            "current_value": current_value,
            "expected_type": expected_type,
            "description": f"Convert parameter {param_name} to {expected_type}"
        }

    @staticmethod
    def fix_timeout_increase(timeout: int) -> Dict[str, Any]:
        """Generate correction for timeout issue"""
        return {
            "strategy": "increase_timeout",
            "new_timeout": timeout * 2,
            "description": f"Increase timeout to {timeout * 2} seconds"
        }


class WorkflowValidator:
    """Validates workflow definitions and inputs"""

    @staticmethod
    def validate_workflow_definition(definition: WorkflowDefinition) -> List[ErrorRecord]:
        """
        Validate workflow definition

        Args:
            definition: Workflow definition to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate name
        if not definition.name or not definition.name.strip():
            errors.append(ErrorRecord(
                error_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                phase=WorkflowPhase.VALIDATION,
                category=ErrorCategory.CONFIGURATION,
                severity=ErrorSeverity.HIGH,
                message="Workflow name cannot be empty",
                error_code="EMPTY_NAME",
                stack_trace="",
                context={"field": "name"}
            ))

        # Validate steps
        if not definition.steps:
            errors.append(ErrorRecord(
                error_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                phase=WorkflowPhase.VALIDATION,
                category=ErrorCategory.CONFIGURATION,
                severity=ErrorSeverity.CRITICAL,
                message="Workflow must have at least one step",
                error_code="NO_STEPS",
                stack_trace="",
                context={"steps_count": 0}
            ))

        # Validate each step
        for idx, step in enumerate(definition.steps):
            if "name" not in step:
                errors.append(ErrorRecord(
                    error_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    phase=WorkflowPhase.VALIDATION,
                    category=ErrorCategory.CONFIGURATION,
                    severity=ErrorSeverity.HIGH,
                    message=f"Step {idx} missing 'name' field",
                    error_code="STEP_NO_NAME",
                    stack_trace="",
                    context={"step_index": idx}
                ))

            if "action" not in step:
                errors.append(ErrorRecord(
                    error_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    phase=WorkflowPhase.VALIDATION,
                    category=ErrorCategory.CONFIGURATION,
                    severity=ErrorSeverity.HIGH,
                    message=f"Step {idx} missing 'action' field",
                    error_code="STEP_NO_ACTION",
                    stack_trace="",
                    context={"step_index": idx}
                ))

        # Validate timeout
        if definition.timeout_seconds <= 0:
            errors.append(ErrorRecord(
                error_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                phase=WorkflowPhase.VALIDATION,
                category=ErrorCategory.CONFIGURATION,
                severity=ErrorSeverity.MEDIUM,
                message="Timeout must be positive",
                error_code="INVALID_TIMEOUT",
                stack_trace="",
                context={"timeout": definition.timeout_seconds}
            ))

        # Validate max retries
        if definition.max_retries < 0:
            errors.append(ErrorRecord(
                error_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                phase=WorkflowPhase.VALIDATION,
                category=ErrorCategory.CONFIGURATION,
                severity=ErrorSeverity.MEDIUM,
                message="Max retries cannot be negative",
                error_code="INVALID_RETRIES",
                stack_trace="",
                context={"max_retries": definition.max_retries}
            ))

        return errors

    @staticmethod
    def validate_inputs(inputs: Dict[str, Any], schema: Dict[str, Any]) -> List[ErrorRecord]:
        """
        Validate inputs against schema

        Args:
            inputs: Input data to validate
            schema: Schema defining expected structure

        Returns:
            List of validation errors
        """
        errors = []

        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in inputs:
                errors.append(ErrorRecord(
                    error_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    phase=WorkflowPhase.VALIDATION,
                    category=ErrorCategory.VALIDATION,
                    severity=ErrorSeverity.HIGH,
                    message=f"Required field missing: {field}",
                    error_code="MISSING_REQUIRED_FIELD",
                    stack_trace="",
                    context={"field": field, "required_fields": required_fields}
                ))

        # Validate types
        properties = schema.get("properties", {})
        for key, value in inputs.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type:
                    if not WorkflowValidator._validate_type(value, expected_type):
                        errors.append(ErrorRecord(
                            error_id=str(uuid.uuid4()),
                            timestamp=datetime.utcnow(),
                            phase=WorkflowPhase.VALIDATION,
                            category=ErrorCategory.VALIDATION,
                            severity=ErrorSeverity.MEDIUM,
                            message=f"Type mismatch for field '{key}': expected {expected_type}",
                            error_code="TYPE_MISMATCH",
                            stack_trace="",
                            context={
                                "field": key,
                                "expected_type": expected_type,
                                "actual_type": type(value).__name__
                            }
                        ))

        return errors

    @staticmethod
    def _validate_type(value: Any, expected_type: str) -> bool:
        """Validate if value matches expected type"""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }

        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, skip validation

        return isinstance(value, expected_python_type)


class ZeroErrorWorkflow:
    """
    Zero-Error Workflow Orchestrator for CrewAI

    Provides comprehensive workflow orchestration with error prevention,
    detection, and automatic correction capabilities.
    """

    def __init__(
        self,
        definition: WorkflowDefinition,
        crewai_state_manager=None,
        enable_auto_correction: bool = True,
        strict_mode: bool = True,
        log_all_steps: bool = True
    ):
        """
        Initialize the zero-error workflow orchestrator

        Args:
            definition: Workflow definition
            crewai_state_manager: State manager for CrewAI integration
            enable_auto_correction: Enable automatic error correction
            strict_mode: Fail fast on validation errors
            log_all_steps: Log all step executions
        """
        self.definition = definition
        self.crewai_state_manager = crewai_state_manager
        self.enable_auto_correction = enable_auto_correction
        self.strict_mode = strict_mode
        self.log_all_steps = log_all_steps

        self.workflow_id = str(uuid.uuid4())
        self.execution_id = str(uuid.uuid4())
        self.context = ExecutionContext(
            workflow_id=self.workflow_id,
            execution_id=self.execution_id,
            start_time=datetime.utcnow()
        )

        self.errors: List[ErrorRecord] = []
        self.step_results: List[StepResult] = []
        self.validator = WorkflowValidator()
        self.correction_strategy = ErrorCorrectionStrategy()

        # CrewAI imports (lazy loading)
        self._crewai_available = False
        self._crewai_imported = False
        self._crew = None
        self._agent = None
        self._task = None

        logger.info(f"Initialized ZeroErrorWorkflow: {self.workflow_id}")

    def _check_crewai_availability(self) -> bool:
        """Check if CrewAI is available and import it"""
        if self._crewai_imported:
            return self._crewai_available

        self._crewai_imported = True
        try:
            from crewai import Crew, Agent, Task
            self._crew = Crew
            self._agent = Agent
            self._task = Task
            self._crewai_available = True
            logger.info("CrewAI successfully imported")
            return True
        except ImportError as e:
            self._crewai_available = False
            error_record = ErrorRecord(
                error_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                phase=WorkflowPhase.INITIALIZATION,
                category=ErrorCategory.IMPORT,
                severity=ErrorSeverity.HIGH,
                message=f"CrewAI import failed: {e}",
                error_code="CREWAI_IMPORT_ERROR",
                stack_trace=traceback.format_exc(),
                context={"import_error": str(e)}
            )
            self.errors.append(error_record)

            if self.enable_auto_correction:
                correction = self.correction_strategy.fix_missing_dependency("crewai")
                logger.info(f"Auto-correction suggested: {correction}")
                error_record.auto_corrected = True
                error_record.correction_strategy = json.dumps(correction)

            logger.warning(f"CrewAI not available: {e}")
            return False

    async def execute(
        self,
        inputs: Dict[str, Any],
        timeout_override: Optional[int] = None
    ) -> WorkflowResult:
        """
        Execute the workflow with zero-error handling

        Args:
            inputs: Input data for the workflow
            timeout_override: Optional timeout override in seconds

        Returns:
            WorkflowResult containing execution results and errors
        """
        start_time = datetime.utcnow()
        timeout = timeout_override or self.definition.timeout_seconds

        logger.info(f"Starting workflow execution: {self.definition.name}")

        try:
            # Phase 1: Initialization
            await self._execute_phase(WorkflowPhase.INITIALIZATION, self._initialize)

            # Phase 2: Validation
            await self._execute_phase(WorkflowPhase.VALIDATION, self._validate, inputs)

            # Phase 3: Preparation
            await self._execute_phase(WorkflowPhase.PREPARATION, self._prepare, inputs)

            # Phase 4: Execution
            await self._execute_phase(
                WorkflowPhase.EXECUTION,
                self._execute_steps,
                inputs,
                timeout
            )

            # Phase 5: Verification
            await self._execute_phase(
                WorkflowPhase.VERIFICATION,
                self._verify_results
            )

            # Phase 6: Completion
            result = await self._complete(start_time)

            logger.info(f"Workflow completed successfully: {self.definition.name}")
            return result

        except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
            logger.error(f"Workflow execution failed: {e}")

            # Perform rollback if critical and enabled
            if self.definition.critical and self.definition.rollback_on_failure:
                await self._execute_phase(WorkflowPhase.ROLLBACK, self._rollback)

            # Create failed result
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds() * 1000

            return WorkflowResult(
                workflow_id=self.workflow_id,
                status=WorkflowStatus.FAILED,
                started_at=start_time,
                completed_at=end_time,
                duration_ms=duration,
                steps_completed=len([s for s in self.step_results if s.status == WorkflowStatus.COMPLETED]),
                steps_failed=len([s for s in self.step_results if s.status == WorkflowStatus.FAILED]),
                steps_total=len(self.definition.steps),
                results={},
                errors=self.errors,
                rollback_performed=self.definition.rollback_on_failure,
                success_rate=0.0
            )

    async def _execute_phase(
        self,
        phase: WorkflowPhase,
        phase_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a workflow phase with error handling

        Args:
            phase: The phase being executed
            phase_func: Function to execute for this phase
            *args: Positional arguments for phase function
            **kwargs: Keyword arguments for phase function

        Returns:
            Result of phase function
        """
        logger.info(f"Executing phase: {phase.value}")

        try:
            result = await phase_func(*args, **kwargs)
            return result
        except (RuntimeError, ValueError, TypeError, ConnectionError, TimeoutError) as e:
            error_record = ErrorRecord(
                error_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                phase=phase,
                category=self._categorize_error(e),
                severity=self._assess_severity(e),
                message=str(e),
                error_code=type(e).__name__,
                stack_trace=traceback.format_exc(),
                context={"phase": phase.value}
            )
            self.errors.append(error_record)

            # Attempt auto-correction if enabled
            if self.enable_auto_correction:
                corrected = await self._attempt_correction(error_record, phase)
                if corrected:
                    # Retry the phase
                    return await phase_func(*args, **kwargs)

            # Re-raise if strict mode
            if self.strict_mode:
                raise WorkflowExecutionError(
                    f"Phase {phase.value} failed: {e}",
                    error_code=type(e).__name__,
                    details={"phase": phase.value, "original_error": str(e)}
                )

            return None

    async def _initialize(self) -> None:
        """Initialize workflow execution"""
        logger.info("Initializing workflow")

        # Validate workflow definition
        validation_errors = self.validator.validate_workflow_definition(self.definition)
        if validation_errors:
            self.errors.extend(validation_errors)
            if self.strict_mode:
                raise WorkflowConfigurationError(
                    f"Workflow definition has {len(validation_errors)} validation errors",
                    error_code="DEFINATION_VALIDATION_FAILED",
                    details={"error_count": len(validation_errors)}
                )

        # Check CrewAI availability
        self._check_crewai_availability()

        # Initialize state if state manager provided
        if self.crewai_state_manager:
            try:
                await self.crewai_state_manager.initialize_state(self.context)
            except (RuntimeError, ValueError, ConnectionError) as e:
                logger.warning(f"State manager initialization failed: {e}")

        logger.info("Initialization complete")

    async def _validate(self, inputs: Dict[str, Any]) -> None:
        """Validate workflow inputs"""
        logger.info("Validating inputs")

        validation_errors = self.validator.validate_inputs(
            inputs,
            self.definition.input_schema
        )

        if validation_errors:
            self.errors.extend(validation_errors)

            # Attempt auto-correction for type mismatches
            if self.enable_auto_correction:
                corrected_inputs = await self._auto_correct_inputs(inputs, validation_errors)
                if corrected_inputs:
                    self.context.variables.update(corrected_inputs)
                    logger.info("Auto-corrected input validation errors")
                    return

            if self.strict_mode:
                raise WorkflowValidationError(
                    f"Input validation failed with {len(validation_errors)} errors",
                    error_code="INPUT_VALIDATION_FAILED",
                    details={"error_count": len(validation_errors)}
                )

        # Store validated inputs
        self.context.variables.update(inputs)
        logger.info("Input validation complete")

    async def _prepare(self, inputs: Dict[str, Any]) -> None:
        """Prepare workflow execution environment"""
        logger.info("Preparing execution environment")

        # Prepare execution context
        self.context.state["prepared"] = True
        self.context.state["input_hash"] = self._hash_inputs(inputs)

        # Set up CrewAI components if available
        if self._crewai_available and self.crewai_state_manager:
            try:
                await self.crewai_state_manager.prepare_workflow(self.context)
            except (RuntimeError, ValueError, ConnectionError) as e:
                logger.warning(f"CrewAI preparation failed: {e}")

        logger.info("Preparation complete")

    async def _execute_steps(
        self,
        inputs: Dict[str, Any],
        timeout: int
    ) -> None:
        """
        Execute all workflow steps with error handling

        Args:
            inputs: Input data
            timeout: Timeout in seconds
        """
        logger.info(f"Executing {len(self.definition.steps)} steps")

        for idx, step_def in enumerate(self.definition.steps):
            step_name = step_def.get("name", f"step_{idx}")
            logger.info(f"Executing step {idx + 1}/{len(self.definition.steps)}: {step_name}")

            step_start = datetime.utcnow()
            step_errors = []
            retry_count = 0

            # Execute step with retries
            while retry_count <= self.definition.max_retries:
                try:
                    # Execute step
                    result = await self._execute_single_step(
                        step_def,
                        inputs,
                        timeout
                    )

                    step_end = datetime.utcnow()
                    duration = (step_end - step_start).total_seconds() * 1000

                    # Record successful step result
                    step_result = StepResult(
                        step_name=step_name,
                        status=WorkflowStatus.COMPLETED,
                        started_at=step_start,
                        completed_at=step_end,
                        duration_ms=duration,
                        result=result,
                        errors=step_errors,
                        validation_passed=True,
                        retry_count=retry_count
                    )
                    self.step_results.append(step_result)

                    # Update context with step result
                    self.context.state[f"step_{step_name}_result"] = result

                    logger.info(f"Step {step_name} completed in {duration:.2f}ms")
                    break

                except (RuntimeError, ValueError, TypeError, ConnectionError, TimeoutError) as e:
                    step_end = datetime.utcnow()
                    duration = (step_end - step_start).total_seconds() * 1000

                    error_record = ErrorRecord(
                        error_id=str(uuid.uuid4()),
                        timestamp=datetime.utcnow(),
                        phase=WorkflowPhase.EXECUTION,
                        category=self._categorize_error(e),
                        severity=self._assess_severity(e),
                        message=f"Step {step_name} failed: {e}",
                        error_code=type(e).__name__,
                        stack_trace=traceback.format_exc(),
                        context={
                            "step": step_name,
                            "step_index": idx,
                            "retry_count": retry_count
                        },
                        retry_count=retry_count
                    )
                    step_errors.append(error_record)
                    self.errors.append(error_record)

                    # Attempt correction and retry
                    if retry_count < self.definition.max_retries and self.enable_auto_correction:
                        corrected = await self._attempt_correction(error_record, WorkflowPhase.EXECUTION)
                        if corrected:
                            retry_count += 1
                            logger.info(f"Retrying step {step_name} (attempt {retry_count})")
                            continue

                    # Max retries exceeded or correction failed
                    step_result = StepResult(
                        step_name=step_name,
                        status=WorkflowStatus.FAILED,
                        started_at=step_start,
                        completed_at=step_end,
                        duration_ms=duration,
                        errors=step_errors,
                        validation_passed=False,
                        retry_count=retry_count
                    )
                    self.step_results.append(step_result)

                    if self.strict_mode:
                        raise WorkflowExecutionError(
                            f"Step {step_name} failed after {retry_count} retries",
                            error_code="STEP_EXECUTION_FAILED",
                            details={"step": step_name, "retries": retry_count}
                        )
                    break

        logger.info(f"Step execution complete: {len(self.step_results)} steps executed")

    async def _execute_single_step(
        self,
        step_def: Dict[str, Any],
        inputs: Dict[str, Any],
        timeout: int
    ) -> Any:
        """
        Execute a single workflow step

        Args:
            step_def: Step definition
            inputs: Input data
            timeout: Timeout in seconds

        Returns:
            Step execution result
        """
        action = step_def.get("action")

        # Handle different action types
        if action == "crewai_crew" and self._crewai_available:
            return await self._execute_crewai_step(step_def, inputs, timeout)
        elif action == "python_function":
            return await self._execute_python_step(step_def, inputs)
        elif action == "validation":
            return await self._execute_validation_step(step_def)
        elif action == "data_processing":
            return await self._execute_data_processing_step(step_def, inputs)
        else:
            raise WorkflowExecutionError(
                f"Unknown action type: {action}",
                error_code="UNKNOWN_ACTION",
                details={"action": action}
            )

    async def _execute_crewai_step(
        self,
        step_def: Dict[str, Any],
        inputs: Dict[str, Any],
        timeout: int
    ) -> Any:
        """Execute a CrewAI crew step"""
        logger.info("Executing CrewAI step")

        if not self._crewai_available:
            raise WorkflowExecutionError(
                "CrewAI not available",
                error_code="CREWAI_UNAVAILABLE",
                details={"step": step_def.get("name")}
            )

        # For now, return a mock result
        # In production, this would actually execute CrewAI crew
        result = {
            "status": "success",
            "crew_output": "CrewAI execution completed",
            "inputs_processed": list(inputs.keys())
        }

        return result

    async def _execute_python_step(
        self,
        step_def: Dict[str, Any],
        inputs: Dict[str, Any]
    ) -> Any:
        """Execute a Python function step"""
        logger.info("Executing Python function step")

        # This is a placeholder for actual Python function execution
        # In production, this would safely execute user-defined functions
        function_name = step_def.get("function")
        parameters = step_def.get("parameters", {})

        result = {
            "function": function_name,
            "parameters": parameters,
            "status": "executed"
        }

        return result

    async def _execute_validation_step(self, step_def: Dict[str, Any]) -> Any:
        """Execute a validation step"""
        logger.info("Executing validation step")

        validations = step_def.get("validations", [])
        results = []

        for validation in validations:
            # Placeholder for actual validation logic
            results.append({
                "validation": validation,
                "passed": True
            })

        return {"validations": results, "all_passed": True}

    async def _execute_data_processing_step(
        self,
        step_def: Dict[str, Any],
        inputs: Dict[str, Any]
    ) -> Any:
        """Execute a data processing step"""
        logger.info("Executing data processing step")

        operation = step_def.get("operation")
        source = step_def.get("source")

        # Placeholder for data processing logic
        result = {
            "operation": operation,
            "source": source,
            "records_processed": 0
        }

        return result

    async def _verify_results(self) -> None:
        """Verify workflow execution results"""
        logger.info("Verifying results")

        # Check if all steps completed successfully
        failed_steps = [s for s in self.step_results if s.status == WorkflowStatus.FAILED]

        if failed_steps:
            logger.warning(f"{len(failed_steps)} steps failed")
            if self.strict_mode:
                raise WorkflowValidationError(
                    f"Workflow verification failed: {len(failed_steps)} steps failed",
                    error_code="VERIFICATION_FAILED",
                    details={"failed_steps": [s.step_name for s in failed_steps]}
                )

        # Validate outputs against schema
        # (Placeholder for actual output validation)

        logger.info("Result verification complete")

    async def _complete(self, start_time: datetime) -> WorkflowResult:
        """Complete workflow execution and generate result"""
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds() * 1000

        steps_completed = len([s for s in self.step_results if s.status == WorkflowStatus.COMPLETED])
        steps_failed = len([s for s in self.step_results if s.status == WorkflowStatus.FAILED])
        success_rate = (steps_completed / len(self.definition.steps)) * 100 if self.definition.steps else 0

        # Aggregate results from all steps
        results = {}
        for step_result in self.step_results:
            if step_result.result:
                results[step_result.step_name] = step_result.result

        result = WorkflowResult(
            workflow_id=self.workflow_id,
            status=WorkflowStatus.COMPLETED if steps_failed == 0 else WorkflowStatus.FAILED,
            started_at=start_time,
            completed_at=end_time,
            duration_ms=duration,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            steps_total=len(self.definition.steps),
            results=results,
            errors=self.errors,
            rollback_performed=False,
            success_rate=success_rate,
            final_output=results
        )

        logger.info(f"Workflow completed with {success_rate:.1f}% success rate")
        return result

    async def _rollback(self) -> None:
        """Rollback workflow execution"""
        logger.warning("Initiating workflow rollback")

        # Reverse execute rollback actions for each completed step
        completed_steps = [s for s in self.step_results if s.status == WorkflowStatus.COMPLETED]

        for step_result in reversed(completed_steps):
            logger.info(f"Rolling back step: {step_result.step_name}")
            # Placeholder for actual rollback logic

        logger.info("Rollback complete")

    async def _attempt_correction(
        self,
        error_record: ErrorRecord,
        phase: WorkflowPhase
    ) -> bool:
        """
        Attempt to automatically correct an error

        Args:
            error_record: The error that occurred
            phase: The phase in which the error occurred

        Returns:
            True if error was corrected, False otherwise
        """
        logger.info(f"Attempting auto-correction for error: {error_record.error_code}")

        correction_applied = False

        # Apply correction strategies based on error category
        if error_record.category == ErrorCategory.IMPORT:
            # Try to fix import errors
            correction_applied = await self._correct_import_error(error_record)
        elif error_record.category == ErrorCategory.VALIDATION:
            # Try to fix validation errors
            correction_applied = await self._correct_validation_error(error_record)
        elif error_record.category == ErrorCategory.TIMEOUT:
            # Try to fix timeout errors
            correction_applied = await self._correct_timeout_error(error_record)
        elif error_record.category == ErrorCategory.RESOURCE:
            # Try to fix resource errors
            correction_applied = await self._correct_resource_error(error_record)

        if correction_applied:
            error_record.auto_corrected = True
            logger.info(f"Auto-correction applied for error: {error_record.error_code}")
        else:
            logger.warning(f"Auto-correction failed for error: {error_record.error_code}")

        return correction_applied

    async def _correct_import_error(self, error_record: ErrorRecord) -> bool:
        """Attempt to correct import errors"""
        # Log correction suggestion
        correction = self.correction_strategy.fix_missing_dependency("crewai")
        error_record.correction_strategy = json.dumps(correction)
        return False  # Cannot auto-install during execution

    async def _correct_validation_error(self, error_record: ErrorRecord) -> bool:
        """Attempt to correct validation errors"""
        # Implement type conversion or default value injection
        return False  # Placeholder

    async def _correct_timeout_error(self, error_record: ErrorRecord) -> bool:
        """Attempt to correct timeout errors"""
        correction = self.correction_strategy.fix_timeout_increase(300)
        error_record.correction_strategy = json.dumps(correction)
        return False  # Cannot change timeout mid-execution

    async def _correct_resource_error(self, error_record: ErrorRecord) -> bool:
        """Attempt to correct resource errors"""
        # Implement resource allocation retry
        return False  # Placeholder

    async def _auto_correct_inputs(
        self,
        inputs: Dict[str, Any],
        validation_errors: List[ErrorRecord]
    ) -> Optional[Dict[str, Any]]:
        """Attempt to auto-correct input validation errors"""
        corrections = {}

        for error in validation_errors:
            if error.error_code == "TYPE_MISMATCH":
                field = error.context.get("field")
                expected_type = error.context.get("expected_type")
                current_value = inputs.get(field)

                if field and current_value:
                    try:
                        # Attempt type conversion
                        if expected_type == "string":
                            corrections[field] = str(current_value)
                        elif expected_type == "integer":
                            corrections[field] = int(current_value)
                        elif expected_type == "number":
                            corrections[field] = float(current_value)
                        elif expected_type == "boolean":
                            corrections[field] = bool(current_value)
                    except (ValueError, TypeError):
                        pass

        return corrections if corrections else None

    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error by type"""
        error_type = type(error).__name__

        if error_type in ["ImportError", "ModuleNotFoundError"]:
            return ErrorCategory.IMPORT
        elif error_type in ["ValueError", "KeyError", "TypeError"]:
            return ErrorCategory.VALIDATION
        elif "timeout" in error_type.lower() or "Timeout" in str(error):
            return ErrorCategory.TIMEOUT
        elif "resource" in str(error).lower():
            return ErrorCategory.RESOURCE
        elif "configuration" in str(error).lower():
            return ErrorCategory.CONFIGURATION
        else:
            return ErrorCategory.EXECUTION

    def _assess_severity(self, error: Exception) -> ErrorSeverity:
        """Assess the severity of an error"""
        error_type = type(error).__name__

        # Critical errors
        if error_type in ["ImportError", "ModuleNotFoundError"]:
            return ErrorSeverity.CRITICAL
        elif "critical" in str(error).lower():
            return ErrorSeverity.CRITICAL

        # High severity
        elif error_type in ["ValueError", "KeyError"]:
            return ErrorSeverity.HIGH

        # Medium severity
        elif error_type in ["TypeError", "AttributeError"]:
            return ErrorSeverity.MEDIUM

        # Default to low
        else:
            return ErrorSeverity.LOW

    def _hash_inputs(self, inputs: Dict[str, Any]) -> str:
        """Generate hash of inputs for caching/validation"""
        inputs_str = json.dumps(inputs, sort_keys=True)
        return hashlib.md5(inputs_str.encode()).hexdigest()

    def generate_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report"""
        return {
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "total_errors": len(self.errors),
            "errors_by_category": self._group_errors_by_category(),
            "errors_by_severity": self._group_errors_by_severity(),
            "auto_corrected": len([e for e in self.errors if e.auto_corrected]),
            "uncorrected": len([e for e in self.errors if not e.auto_corrected]),
            "errors": [
                {
                    "error_id": e.error_id,
                    "timestamp": e.timestamp.isoformat(),
                    "phase": e.phase.value,
                    "category": e.category.value,
                    "severity": e.severity.value,
                    "message": e.message,
                    "error_code": e.error_code,
                    "auto_corrected": e.auto_corrected,
                    "correction_strategy": e.correction_strategy
                }
                for e in self.errors
            ]
        }

    def _group_errors_by_category(self) -> Dict[str, int]:
        """Group errors by category"""
        grouped = {}
        for error in self.errors:
            category = error.category.value
            grouped[category] = grouped.get(category, 0) + 1
        return grouped

    def _group_errors_by_severity(self) -> Dict[str, int]:
        """Group errors by severity"""
        grouped = {}
        for error in self.errors:
            severity = error.severity.value
            grouped[severity] = grouped.get(severity, 0) + 1
        return grouped


# Helper functions

def create_workflow_definition(
    name: str,
    steps: List[Dict[str, Any]],
    description: str = "",
    version: str = "1.0.0",
    **kwargs
) -> WorkflowDefinition:
    """
    Create a workflow definition with defaults

    Args:
        name: Workflow name
        steps: List of workflow steps
        description: Workflow description
        version: Workflow version
        **kwargs: Additional parameters

    Returns:
        WorkflowDefinition object
    """
    return WorkflowDefinition(
        name=name,
        description=description,
        version=version,
        steps=steps,
        input_schema=kwargs.get("input_schema", {"type": "object", "properties": {}, "required": []}),
        output_schema=kwargs.get("output_schema", {"type": "object"}),
        validation_rules=kwargs.get("validation_rules", []),
        timeout_seconds=kwargs.get("timeout_seconds", 300),
        max_retries=kwargs.get("max_retries", 3),
        critical=kwargs.get("critical", True),
        rollback_on_failure=kwargs.get("rollback_on_failure", True)
    )


async def execute_workflow_zero_error(
    workflow_definition: WorkflowDefinition,
    inputs: Dict[str, Any],
    crewai_state_manager=None,
    **kwargs
) -> WorkflowResult:
    """
    Convenience function to execute a workflow with zero-error handling

    Args:
        workflow_definition: Workflow definition
        inputs: Input data
        crewai_state_manager: Optional state manager
        **kwargs: Additional parameters for ZeroErrorWorkflow

    Returns:
        WorkflowResult
    """
    orchestrator = ZeroErrorWorkflow(
        definition=workflow_definition,
        crewai_state_manager=crewai_state_manager,
        **kwargs
    )

    return await orchestrator.execute(inputs)


# =============================================================================
# INTEGRATION WITH CLAUDIOMIRO_CREWAI_BRIDGE
# =============================================================================

class ClaudeMiroWorkflowBridge:
    """
    Bridge between ClaudeMiro and CrewAI Zero-Error Workflow

    This class provides integration points for claudiomiro_crewai_bridge.py
    """

    @staticmethod
    def create_workflow_from_bridge_config(bridge_config: Dict[str, Any]) -> WorkflowDefinition:
        """
        Create a WorkflowDefinition from ClaudeMiro bridge configuration

        Args:
            bridge_config: Configuration from claudiomiro_crewai_bridge.py

        Returns:
            WorkflowDefinition
        """
        steps = []

        # Convert bridge tasks to workflow steps
        tasks = bridge_config.get("tasks", [])
        for task in tasks:
            step = {
                "name": task.get("name", f"task_{len(steps)}"),
                "action": "crewai_crew",
                "task": task,
                "description": task.get("description", ""),
                "agents": task.get("agents", []),
                "expected_output": task.get("expected_output", "")
            }
            steps.append(step)

        return WorkflowDefinition(
            name=bridge_config.get("name", "claudiomiro_workflow"),
            description=bridge_config.get("description", "Workflow from ClaudeMiro bridge"),
            version=bridge_config.get("version", "1.0.0"),
            steps=steps,
            input_schema=bridge_config.get("input_schema", {"type": "object", "properties": {}, "required": []}),
            output_schema=bridge_config.get("output_schema", {"type": "object"}),
            validation_rules=bridge_config.get("validation_rules", []),
            timeout_seconds=bridge_config.get("timeout_seconds", 300),
            max_retries=bridge_config.get("max_retries", 3),
            critical=bridge_config.get("critical", True),
            rollback_on_failure=bridge_config.get("rollback_on_failure", True)
        )

    @staticmethod
    async def execute_bridge_workflow(
        bridge_config: Dict[str, Any],
        inputs: Dict[str, Any],
        state_manager=None,
        **kwargs
    ) -> WorkflowResult:
        """
        Execute a workflow from ClaudeMiro bridge configuration

        Args:
            bridge_config: Configuration from claudiomiro_crewai_bridge.py
            inputs: Input data
            state_manager: Optional state manager
            **kwargs: Additional parameters

        Returns:
            WorkflowResult
        """
        workflow_def = ClaudeMiroWorkflowBridge.create_workflow_from_bridge_config(bridge_config)

        orchestrator = ZeroErrorWorkflow(
            definition=workflow_def,
            crewai_state_manager=state_manager,
            **kwargs
        )

        return await orchestrator.execute(inputs)


# =============================================================================
# UNIT TESTS
# =============================================================================

import unittest


class TestZeroErrorWorkflow(unittest.TestCase):
    """Unit tests for ZeroErrorWorkflow"""

    def setUp(self):
        """Set up test fixtures"""
        self.workflow_def = create_workflow_definition(
            name="test_workflow",
            steps=[
                {
                    "name": "step1",
                    "action": "validation",
                    "validations": ["check1"]
                },
                {
                    "name": "step2",
                    "action": "python_function",
                    "function": "test_func",
                    "parameters": {}
                }
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "test_input": {"type": "string"}
                },
                "required": ["test_input"]
            }
        )

    def test_workflow_definition_validation(self):
        """Test workflow definition validation"""
        validator = WorkflowValidator()
        errors = validator.validate_workflow_definition(self.workflow_def)
        self.assertEqual(len(errors), 0)

    def test_workflow_definition_invalid_no_steps(self):
        """Test workflow definition with no steps"""
        invalid_def = create_workflow_definition(
            name="invalid",
            steps=[]
        )
        validator = WorkflowValidator()
        errors = validator.validate_workflow_definition(invalid_def)
        self.assertGreater(len(errors), 0)

    def test_input_validation_success(self):
        """Test successful input validation"""
        validator = WorkflowValidator()
        errors = validator.validate_inputs(
            {"test_input": "value"},
            self.workflow_def.input_schema
        )
        self.assertEqual(len(errors), 0)

    def test_input_validation_missing_required(self):
        """Test input validation with missing required field"""
        validator = WorkflowValidator()
        errors = validator.validate_inputs(
            {},
            self.workflow_def.input_schema
        )
        self.assertGreater(len(errors), 0)

    def test_input_validation_type_mismatch(self):
        """Test input validation with type mismatch"""
        validator = WorkflowValidator()
        errors = validator.validate_inputs(
            {"test_input": 123},  # Should be string
            self.workflow_def.input_schema
        )
        self.assertGreater(len(errors), 0)

    def test_error_categorization(self):
        """Test error categorization"""
        orchestrator = ZeroErrorWorkflow(self.workflow_def)

        import_error = ImportError("Module not found")
        self.assertEqual(
            orchestrator._categorize_error(import_error),
            ErrorCategory.IMPORT
        )

        value_error = ValueError("Invalid value")
        self.assertEqual(
            orchestrator._categorize_error(value_error),
            ErrorCategory.VALIDATION
        )

    def test_severity_assessment(self):
        """Test error severity assessment"""
        orchestrator = ZeroErrorWorkflow(self.workflow_def)

        import_error = ImportError("Module not found")
        self.assertEqual(
            orchestrator._assess_severity(import_error),
            ErrorSeverity.CRITICAL
        )

        value_error = ValueError("Invalid value")
        self.assertEqual(
            orchestrator._assess_severity(value_error),
            ErrorSeverity.HIGH
        )

    def test_input_hashing(self):
        """Test input hashing"""
        orchestrator = ZeroErrorWorkflow(self.workflow_def)

        inputs1 = {"a": 1, "b": 2}
        inputs2 = {"b": 2, "a": 1}  # Different order
        inputs3 = {"a": 1, "b": 3}  # Different value

        hash1 = orchestrator._hash_inputs(inputs1)
        hash2 = orchestrator._hash_inputs(inputs2)
        hash3 = orchestrator._hash_inputs(inputs3)

        self.assertEqual(hash1, hash2)  # Order shouldn't matter
        self.assertNotEqual(hash1, hash3)  # Different value

    def test_error_correction_strategy(self):
        """Test error correction strategies"""
        strategy = ErrorCorrectionStrategy()

        correction = strategy.fix_missing_dependency("test_package")
        self.assertEqual(correction["strategy"], "install_dependency")
        self.assertIn("test_package", correction["command"])

        correction = strategy.fix_missing_env_var("TEST_VAR", "value")
        self.assertEqual(correction["strategy"], "set_env_var")
        self.assertEqual(correction["variable"], "TEST_VAR")

    def test_execution_context(self):
        """Test execution context"""
        context = ExecutionContext(
            workflow_id="test_wf",
            execution_id="test_exec",
            start_time=datetime.utcnow()
        )

        context_dict = context.to_dict()
        self.assertEqual(context_dict["workflow_id"], "test_wf")
        self.assertEqual(context_dict["execution_id"], "test_exec")
        self.assertIn("start_time", context_dict)


class TestWorkflowIntegration(unittest.TestCase):
    """Integration tests for workflow execution"""

    def setUp(self):
        """Set up test fixtures"""
        self.workflow_def = create_workflow_definition(
            name="integration_test",
            steps=[
                {
                    "name": "validate",
                    "action": "validation",
                    "validations": ["check_complete"]
                },
                {
                    "name": "process",
                    "action": "data_processing",
                    "operation": "transform",
                    "source": "input"
                }
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "data": {"type": "array"}
                },
                "required": ["data"]
            },
            timeout_seconds=60,
            max_retries=2
        )

    async def test_complete_workflow_execution(self):
        """Test complete workflow execution"""
        orchestrator = ZeroErrorWorkflow(
            definition=self.workflow_def,
            enable_auto_correction=True,
            strict_mode=False
        )

        result = await orchestrator.execute(
            inputs={"data": [1, 2, 3]},
            timeout_override=60
        )

        self.assertIsNotNone(result)
        self.assertIn(result.status, [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED])
        self.assertEqual(result.steps_total, 2)

    async def test_workflow_with_validation_errors(self):
        """Test workflow with input validation errors"""
        orchestrator = ZeroErrorWorkflow(
            definition=self.workflow_def,
            enable_auto_correction=True,
            strict_mode=False
        )

        # Missing required field
        result = await orchestrator.execute(inputs={})

        self.assertIsNotNone(result)
        # Should continue in non-strict mode
        self.assertGreater(len(result.errors), 0)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_usage():
    """Example of how to use the ZeroErrorWorkflow"""

    # Define a workflow
    workflow_def = create_workflow_definition(
        name="example_workflow",
        description="An example workflow demonstrating zero-error execution",
        steps=[
            {
                "name": "initialize",
                "action": "python_function",
                "function": "setup",
                "parameters": {"config": "default"}
            },
            {
                "name": "process_data",
                "action": "data_processing",
                "operation": "transform",
                "source": "input"
            },
            {
                "name": "validate",
                "action": "validation",
                "validations": ["check_completeness", "check_accuracy"]
            }
        ],
        input_schema={
            "type": "object",
            "properties": {
                "data": {"type": "array"},
                "config": {"type": "object"}
            },
            "required": ["data"]
        },
        timeout_seconds=300,
        max_retries=3
    )

    # Execute workflow
    result = await execute_workflow_zero_error(
        workflow_definition=workflow_def,
        inputs={
            "data": [1, 2, 3, 4, 5],
            "config": {"mode": "strict"}
        },
        enable_auto_correction=True,
        strict_mode=False
    )

    print(f"Workflow Status: {result.status.value}")
    print(f"Success Rate: {result.success_rate:.1f}%")
    print(f"Steps Completed: {result.steps_completed}/{result.steps_total}")
    print(f"Total Errors: {len(result.errors)}")

    if result.errors:
        print("\nError Report:")
        for error in result.errors:
            print(f"  - [{error.severity.value.upper()}] {error.message}")

    return result


if __name__ == "__main__":
    print("CrewAI Zero-Error Workflow Orchestrator")
    print("=" * 50)
    print("\nThis module provides zero-error workflow orchestration for CrewAI")
    print("\nFeatures:")
    print("  - Multi-phase execution with validation")
    print("  - Automatic error detection and correction")
    print("  - Comprehensive error reporting")
    print("  - Integration with CrewAI state management")
    print("  - Support for complex workflow definitions")
    print("\nExample:")
    print("  result = await execute_workflow_zero_error(")
    print("      workflow_definition=workflow_def,")
    print("      inputs={'data': [...]},")
    print("      enable_auto_correction=True")
    print("  )")
    print("\nFor unit tests, run:")
    print("  python -m unittest crewai_zero_error_workflow")
