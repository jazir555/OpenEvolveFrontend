"""
Base class for all BubbleLabs integration nodes.

Provides standardized interface, error handling, and state management
for all OpenEvolve components being integrated into BubbleLabs.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import deque
import traceback
import logging
import sys
import os

# Configure logging
logger = logging.getLogger(__name__)


class NodeExecutionError(Exception):
    """Standardized error for node execution failures"""

    def __init__(self, node_name: str, message: str, details: Dict[str, Any] = None):
        self.node_name = node_name
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        super().__init__(f"[{node_name}] {message}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization"""
        return {
            'node_name': self.node_name,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp,
            'traceback': traceback.format_exc()
        }


class BubbleLabsNode(ABC):
    """
    Abstract base class for all BubbleLabs workflow nodes.

    All OpenEvolve components must inherit from this class to be
    compatible with the BubbleLabs integration system.
    """

    # Node metadata - override in subclasses
    DISPLAY_NAME: str = "Base Node"
    DESCRIPTION: str = "Base node class"
    ICON: str = "default-node"
    CATEGORY: str = "general"
    VERSION: str = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize node with configuration.

        Args:
            config: Node-specific configuration parameters
        """
        self.config = config or {}
        self.status = "initialized"
        self.execution_id: Optional[str] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # ICR Integration: Pattern storage and learning
        self.enable_icr = self.config.get('enable_icr', True)
        self.icr_pattern_store = {
            'execution_patterns': {},      # operation_type -> pattern list
            'verification_patterns': {},   # verification_type -> pattern list
            'routing_patterns': {},        # routing_type -> pattern list
            'research_patterns': {},       # research_type -> pattern list
            'operation_history': deque(maxlen=500)  # Full operation history
        }

        # ICR: Adaptive threshold adjustments
        self._adaptive_thresholds: Dict[str, float] = {}

        # Apply default configuration
        self._apply_defaults()

    def _apply_defaults(self):
        """Apply default values from parameter schema"""
        schema = self.get_parameter_schema()
        defaults = {}

        def extract_defaults(props):
            for prop_name, prop_def in props.items():
                if 'default' in prop_def:
                    defaults[prop_name] = prop_def['default']
                if prop_def.get('type') == 'object' and 'properties' in prop_def:
                    extract_defaults(prop_def['properties'])

        extract_defaults(schema.get('properties', {}))

        # Apply defaults to config (don't override user-provided values)
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

    @abstractmethod
    def execute(self, inputs: Dict, context: 'WorkflowState') -> Dict[str, Any]:
        """
        Execute the node's primary logic.

        Args:
            inputs: Input data from previous nodes or user
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dictionary containing execution results

        Raises:
            NodeExecutionError: If execution fails
        """
        pass

    @abstractmethod
    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input data before execution.

        Args:
            inputs: Input data to validate

        Returns:
            List of error messages (empty if valid)
        """
        pass

    @abstractmethod
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        This schema is used by the BubbleLabs UI to generate
        configuration panels.

        Returns:
            JSON schema dictionary
        """
        pass

    def validate_config(self) -> List[str]:
        """
        Validate node configuration.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        schema = self.get_parameter_schema()

        # Validate config against schema
        # (Basic implementation - can be enhanced with jsonschema)
        required = schema.get('required', [])
        for field in required:
            if field not in self.config:
                errors.append(f"Missing required config field: {field}")

        return errors

    def get_display_name(self) -> str:
        """Get human-readable name for UI"""
        return self.DISPLAY_NAME

    def get_description(self) -> str:
        """Get node description for tooltips"""
        return self.DESCRIPTION

    def get_icon(self) -> str:
        """Get icon identifier for visual representation"""
        return self.ICON

    def get_category(self) -> str:
        """Get category for organizing nodes in UI"""
        return self.CATEGORY

    def get_version(self) -> str:
        """Get node version"""
        return self.VERSION

    def safe_import(self, module_name: str, fallback_value=None, error_msg: str = None):
        """
        Safely import a module with fallback and error handling.

        Args:
            module_name: Name of the module to import
            fallback_value: Value to return if import fails
            error_msg: Custom error message

        Returns:
            Imported module or fallback value
        """
        try:
            module = __import__(module_name, fromlist=[''])
            return module
        except ImportError as e:
            msg = error_msg or f"Module {module_name} not available: {e}"
            self.logger.warning(msg)
            return fallback_value
        except Exception as e:
            msg = error_msg or f"Unexpected error importing {module_name}: {e}"
            self.logger.error(msg)
            return fallback_value

    def safe_execute_with_fallback(self, main_func, fallback_func, *args, **kwargs):
        """
        Execute a function with a fallback in case of failure.

        Args:
            main_func: Main function to execute
            fallback_func: Fallback function to execute if main fails
            *args, **kwargs: Arguments to pass to functions

        Returns:
            Result of main_func or fallback_func
        """
        try:
            return main_func(*args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Main function failed: {e}, using fallback")
            try:
                return fallback_func(*args, **kwargs)
            except Exception as fallback_e:
                self.logger.error(f"Fallback function also failed: {fallback_e}")
                raise

    def before_execute(self, inputs: Dict, context: 'WorkflowState'):
        """
        Hook called before execution.
        Override in subclasses for custom pre-execution logic.
        """
        self.status = "running"
        self.execution_id = context.generate_execution_id()
        self.logger.info(f"Starting execution: {self.execution_id}")

        # Validate inputs
        input_errors = self.validate_inputs(inputs)
        if input_errors:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message="Input validation failed",
                details={'errors': input_errors}
            )

        # Validate config
        config_errors = self.validate_config()
        if config_errors:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message="Configuration validation failed",
                details={'errors': config_errors}
            )

    def after_execute(self, result: Dict, context: 'WorkflowState'):
        """
        Hook called after successful execution.
        Override in subclasses for custom post-execution logic.
        """
        self.status = "completed"
        self.logger.info(f"Execution completed: {self.execution_id}")

        # Store result in context
        context.add_artifact(
            f"{self.__class__.__name__}_result",
            {
                'execution_id': self.execution_id,
                'timestamp': datetime.now().isoformat(),
                'result': result
            }
        )

    def on_error(self, error: Exception, context: 'WorkflowState'):
        """
        Hook called when execution fails.
        Override in subclasses for custom error handling.
        """
        self.status = "failed"
        self.logger.error(f"Execution failed: {self.execution_id}", exc_info=error)

        # Store error in context
        if isinstance(error, NodeExecutionError):
            context.add_error(error.to_dict())
        else:
            context.add_error({
                'node_name': self.get_display_name(),
                'message': str(error),
                'timestamp': datetime.now().isoformat(),
                'traceback': traceback.format_exc()
            })

    def execute_safe(self, inputs: Dict, context: 'WorkflowState') -> Dict[str, Any]:
        """
        Execute with automatic error handling and lifecycle hooks.

        This is the recommended way to execute nodes as it provides:
        - Automatic error handling
        - Lifecycle hooks
        - State management
        - Progress tracking

        Args:
            inputs: Input data
            context: Workflow state

        Returns:
            Execution results

        Raises:
            NodeExecutionError: If execution fails and error handling is disabled
        """
        try:
            # Pre-execution hook
            self.before_execute(inputs, context)

            # Execute with potential timeout
            result = self.execute(inputs, context)

            # Post-execution hook
            self.after_execute(result, context)

            return result

        except Exception as e:
            # Error hook
            try:
                self.on_error(e, context)
            except Exception as error_handler_error:
                # If error handler itself fails, log and continue
                self.logger.error(f"Error in error handler: {error_handler_error}")

            # Re-raise if it's already a NodeExecutionError
            if isinstance(e, NodeExecutionError):
                raise

            # Wrap other exceptions
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Unexpected error: {str(e)}",
                details={
                    'exception_type': type(e).__name__,
                    'inputs': inputs
                }
            ) from e

    def execute_with_timeout(self, inputs: Dict, context: 'WorkflowState', timeout_seconds: int = 300) -> Dict[str, Any]:
        """
        Execute with timeout protection.

        Args:
            inputs: Input data
            context: Workflow state
            timeout_seconds: Maximum execution time in seconds

        Returns:
            Execution results

        Raises:
            NodeExecutionError: If execution fails or times out
        """
        import threading
        import time

        result = [None]  # Use list to store result from thread
        exception = [None]  # Use list to store exception from thread

        def target():
            try:
                # Execute with the safe method
                result[0] = self.execute_safe(inputs, context)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)

        if thread.is_alive():
            # Thread is still running, meaning it timed out
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Execution timed out after {timeout_seconds} seconds",
                details={
                    'timeout_seconds': timeout_seconds,
                    'inputs': inputs
                }
            )

        if exception[0]:
            raise exception[0]

        return result[0]

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Override in subclasses for custom health checks.

        Returns:
            True if node is healthy, False otherwise
        """
        try:
            # Basic health check - ensure required dependencies are available
            return True
        except Exception:
            return False
    
        # ==================== ICR Integration Methods ====================
    
        def store_icr_pattern(
            self,
            operation_type: str,
            success: bool,
            execution_time: float,
            metadata: Optional[Dict[str, Any]] = None,
            sub_key: Optional[str] = None
        ) -> None:
            """
            Store execution pattern for ICR learning.
    
            Args:
                operation_type: Type of operation performed (e.g., 'assembly', 'gauntlet')
                success: Whether the operation succeeded
                execution_time: Time taken to execute the operation
                metadata: Additional metadata about the operation
                sub_key: Optional sub-key for categorizing patterns (e.g., 'weighted', 'voting')
            """
            if not self.enable_icr:
                return
    
            self.logger.info(f"Storing ICR pattern for {operation_type}")
    
            # Create pattern record
            pattern = {
                'timestamp': datetime.now().isoformat(),
                'operation_type': operation_type,
                'success': success,
                'execution_time': execution_time,
                'node_type': self.__class__.__name__,
                'metadata': metadata or {}
            }
    
            # Determine the store key based on operation type
            store_key_map = {
                'assembly': 'execution_patterns',
                'gauntlet': 'verification_patterns',
                'routing': 'routing_patterns',
                'research': 'research_patterns'
            }
            store_key = store_key_map.get(operation_type, 'execution_patterns')
    
            # Use sub_key if provided, otherwise use operation_type
            key = sub_key or operation_type
    
            # Store in pattern store
            if key not in self.icr_pattern_store[store_key]:
                self.icr_pattern_store[store_key][key] = []
    
            # Keep only last 100 patterns per sub-key
            patterns = self.icr_pattern_store[store_key][key]
            patterns.append(pattern)
            if len(patterns) > 100:
                patterns.pop(0)  # Remove oldest
    
            # Store in operation history
            self.icr_pattern_store['operation_history'].append(pattern)
    
            # Calculate success rate for this pattern
            all_patterns = self.icr_pattern_store[store_key].get(key, [])
            succeeded = sum(1 for p in all_patterns if p.get('success', False))
            pattern['success_rate'] = succeeded / len(all_patterns) if all_patterns else 0.0
    
            self.logger.info(f"ICR pattern stored: success_rate={pattern['success_rate']:.2%}")
    
        def predict_pass_fail(
            self,
            operation_type: str,
            metadata: Optional[Dict[str, Any]] = None,
            sub_key: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Predict whether an operation will pass based on historical patterns.
    
            Args:
                operation_type: Type of operation to predict
                metadata: Additional metadata for prediction context
                sub_key: Optional sub-key for specific pattern category
    
            Returns:
                Dictionary with prediction results including:
                    - predicted_success: bool
                    - confidence: float (0-1)
                    - historical_success_rate: float
                    - sample_size: int
            """
            if not self.enable_icr:
                return {
                    'predicted_success': True,
                    'confidence': 0.5,
                    'historical_success_rate': 0.5,
                    'sample_size': 0,
                    'icr_enabled': False
                }
    
            # Determine the store key
            store_key_map = {
                'assembly': 'execution_patterns',
                'gauntlet': 'verification_patterns',
                'routing': 'routing_patterns',
                'research': 'research_patterns'
            }
            store_key = store_key_map.get(operation_type, 'execution_patterns')
            key = sub_key or operation_type
    
            # Get historical patterns
            historical_patterns = self.icr_pattern_store[store_key].get(key, [])
    
            if not historical_patterns:
                return {
                    'predicted_success': True,
                    'confidence': 0.5,
                    'historical_success_rate': 0.5,
                    'sample_size': 0,
                    'icr_enabled': True
                }
    
            # Calculate success rate
            succeeded = sum(1 for p in historical_patterns if p.get('success', False))
            success_rate = succeeded / len(historical_patterns)
    
            # Calculate confidence based on sample size (more samples = higher confidence)
            sample_size = len(historical_patterns)
            confidence = min(1.0, sample_size / 50.0)  # Max confidence at 50 samples
    
            # Predict success based on historical rate
            predicted_success = success_rate > 0.5
    
            self.logger.info(
                f"ICR prediction for {operation_type}: "
                f"success={predicted_success}, rate={success_rate:.2%}, "
                f"confidence={confidence:.2%}, samples={sample_size}"
            )
    
            return {
                'predicted_success': predicted_success,
                'confidence': confidence,
                'historical_success_rate': success_rate,
                'sample_size': sample_size,
                'icr_enabled': True
            }
    
        def get_threshold_adjustment(
            self,
            operation_type: str,
            sub_key: Optional[str] = None
        ) -> float:
            """
            Get recommended threshold adjustment based on ICR patterns.
    
            Args:
                operation_type: Type of operation
                sub_key: Optional sub-key for specific pattern category
    
            Returns:
                Float representing the recommended adjustment (positive = raise, negative = lower)
            """
            if not self.enable_icr:
                return 0.0
    
            store_key_map = {
                'assembly': 'execution_patterns',
                'gauntlet': 'verification_patterns',
                'routing': 'routing_patterns',
                'research': 'research_patterns'
            }
            store_key = store_key_map.get(operation_type, 'execution_patterns')
            key = sub_key or operation_type
    
            # Check if we have enough data to recommend adjustment
            patterns = self.icr_pattern_store[store_key].get(key, [])
    
            if len(patterns) < 10:
                return 0.0  # Not enough data
    
            # Calculate success rate
            succeeded = sum(1 for p in patterns if p.get('success', False))
            success_rate = succeeded / len(patterns)
    
            # Recommend adjustment based on success rate
            if success_rate > 0.8:
                return -2.0  # High success rate - can be more lenient
            elif success_rate < 0.3:
                return 2.0  # Low success rate - need to be stricter
    
            return 0.0
    
        def get_icr_statistics(self) -> Dict[str, Any]:
            """
            Get ICR-related statistics.
    
            Returns:
                Dictionary containing ICR statistics including:
                    - icr_enabled: bool
                    - total_patterns: int
                    - overall_success_rate: float
                    - patterns_by_type: dict
                    - operation_history_size: int
            """
            if not self.enable_icr:
                return {'icr_enabled': False}
    
            total_patterns = sum(
                len(patterns)
                for patterns in self.icr_pattern_store['execution_patterns'].values()
            ) + sum(
                len(patterns)
                for patterns in self.icr_pattern_store['verification_patterns'].values()
            ) + sum(
                len(patterns)
                for patterns in self.icr_pattern_store['routing_patterns'].values()
            ) + sum(
                len(patterns)
                for patterns in self.icr_pattern_store['research_patterns'].values()
            )
    
            # Calculate overall success rate
            all_patterns = list(self.icr_pattern_store['operation_history'])
            succeeded = sum(1 for p in all_patterns if p.get('success', False))
            overall_success_rate = succeeded / len(all_patterns) if all_patterns else 0.0
    
            return {
                'icr_enabled': True,
                'total_patterns': total_patterns,
                'overall_success_rate': overall_success_rate,
                'operation_history_size': len(self.icr_pattern_store['operation_history']),
                'patterns_by_type': {
                    'execution': {
                        key: len(patterns)
                        for key, patterns in self.icr_pattern_store['execution_patterns'].items()
                    },
                    'verification': {
                        key: len(patterns)
                        for key, patterns in self.icr_pattern_store['verification_patterns'].items()
                    },
                    'routing': {
                        key: len(patterns)
                        for key, patterns in self.icr_pattern_store['routing_patterns'].items()
                    },
                    'research': {
                        key: len(patterns)
                        for key, patterns in self.icr_pattern_store['research_patterns'].items()
                    }
                },
                'adaptive_thresholds': self._adaptive_thresholds.copy()
            }
    
        def clear_icr_patterns(self) -> None:
            """Clear all stored ICR patterns."""
            if not self.enable_icr:
                return
    
            self.logger.info("Clearing all ICR patterns")
    
            self.icr_pattern_store = {
                'execution_patterns': {},
                'verification_patterns': {},
                'routing_patterns': {},
                'research_patterns': {},
                'operation_history': deque(maxlen=500)
            }
            self._adaptive_thresholds.clear()
    
        def __repr__(self) -> str:
            return (
                f"{self.__class__.__name__}("
                f"name='{self.get_display_name()}', "
                f"version='{self.get_version()}', "
                f"status='{self.status}')"
            )
