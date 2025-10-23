"""
Comprehensive Error Handling System for OpenEvolve
Provides centralized error handling, recovery, and reporting
"""

import logging
import traceback
import time
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import functools


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    IMPORT_ERROR = "import_error"
    CONFIGURATION_ERROR = "configuration_error"
    API_ERROR = "api_error"
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    NETWORK_ERROR = "network_error"
    FILE_ERROR = "file_error"
    AUTHENTICATION_ERROR = "authentication_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ErrorInfo:
    """Structured error information"""
    error_type: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: float
    traceback_str: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    recovery_suggestions: Optional[List[str]] = None


class ErrorHandler:
    """Centralized error handling system"""
    
    def __init__(self, logger_name: str = "OpenEvolve.ErrorHandler"):
        self.logger = logging.getLogger(logger_name)
        self.error_history: List[ErrorInfo] = []
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
    def register_recovery_strategy(self, category: ErrorCategory, strategy: Callable):
        """Register a recovery strategy for a specific error category"""
        self.recovery_strategies[category] = strategy
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: Optional[ErrorCategory] = None,
        recovery_suggestions: Optional[List[str]] = None
    ) -> ErrorInfo:
        """
        Handle an error with comprehensive logging and recovery
        
        Args:
            error: The exception that occurred
            context: Additional context information
            severity: Error severity level
            category: Error category for classification
            recovery_suggestions: Suggested recovery actions
            
        Returns:
            ErrorInfo object with structured error information
        """
        # Classify error if not provided
        if category is None:
            category = self._classify_error(error)
        
        # Create error info
        error_info = ErrorInfo(
            error_type=type(error).__name__,
            message=str(error),
            category=category,
            severity=severity,
            timestamp=time.time(),
            traceback_str=traceback.format_exc(),
            context=context or {},
            recovery_suggestions=recovery_suggestions or self._get_default_suggestions(category)
        )
        
        # Log error
        self._log_error(error_info)
        
        # Store in history
        self.error_history.append(error_info)
        
        # Attempt recovery if strategy available
        if category in self.recovery_strategies:
            try:
                self.recovery_strategies[category](error_info)
                self.logger.info(f"Recovery strategy executed for {category.value}")
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy failed: {recovery_error}")
        
        return error_info
    
    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error based on type and message"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        if isinstance(error, ImportError):
            return ErrorCategory.IMPORT_ERROR
        elif isinstance(error, ValueError) and ('config' in error_message or 'parameter' in error_message):
            return ErrorCategory.CONFIGURATION_ERROR
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK_ERROR
        elif isinstance(error, FileNotFoundError):
            return ErrorCategory.FILE_ERROR
        elif 'api' in error_message or 'request' in error_message:
            return ErrorCategory.API_ERROR
        elif 'validation' in error_message or 'invalid' in error_message:
            return ErrorCategory.VALIDATION_ERROR
        elif 'timeout' in error_message:
            return ErrorCategory.TIMEOUT_ERROR
        elif 'auth' in error_message or 'permission' in error_message:
            return ErrorCategory.AUTHENTICATION_ERROR
        else:
            return ErrorCategory.UNKNOWN_ERROR
    
    def _get_default_suggestions(self, category: ErrorCategory) -> List[str]:
        """Get default recovery suggestions for error category"""
        suggestions = {
            ErrorCategory.IMPORT_ERROR: [
                "Check if required packages are installed",
                "Verify Python environment and dependencies",
                "Try reinstalling missing packages"
            ],
            ErrorCategory.CONFIGURATION_ERROR: [
                "Check configuration parameters",
                "Verify API keys and settings",
                "Use default configuration as fallback"
            ],
            ErrorCategory.API_ERROR: [
                "Check API key validity",
                "Verify network connection",
                "Try again with exponential backoff"
            ],
            ErrorCategory.NETWORK_ERROR: [
                "Check internet connection",
                "Verify API endpoints are accessible",
                "Try again after a short delay"
            ],
            ErrorCategory.FILE_ERROR: [
                "Check file path exists",
                "Verify file permissions",
                "Create missing directories"
            ],
            ErrorCategory.TIMEOUT_ERROR: [
                "Increase timeout duration",
                "Check network stability",
                "Retry with smaller batch size"
            ]
        }
        return suggestions.get(category, ["Contact support for assistance"])
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level"""
        log_message = f"[{error_info.category.value.upper()}] {error_info.error_type}: {error_info.message}"
        
        if error_info.context:
            log_message += f" | Context: {error_info.context}"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def get_error_summary(self, last_n: int = 10) -> Dict[str, Any]:
        """Get summary of recent errors"""
        recent_errors = self.error_history[-last_n:] if self.error_history else []
        
        category_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "last_error": recent_errors[-1].__dict__ if recent_errors else None
        }


def with_error_handling(
    category: Optional[ErrorCategory] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    fallback_value: Any = None,
    retry_count: int = 0,
    retry_delay: float = 1.0
):
    """
    Decorator for automatic error handling with retry logic
    
    Args:
        category: Error category for classification
        severity: Error severity level
        fallback_value: Value to return on error
        retry_count: Number of retries on failure
        retry_delay: Delay between retries in seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = get_global_error_handler()
            
            for attempt in range(retry_count + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < retry_count:
                        error_handler.logger.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}")
                        time.sleep(retry_delay)
                        continue
                    
                    # Final attempt failed, handle error
                    context = {
                        "function": func.__name__,
                        "args": str(args)[:100],  # Truncate for logging
                        "kwargs": str(kwargs)[:100],
                        "attempt": attempt + 1
                    }
                    
                    error_handler.handle_error(
                        error=e,
                        context=context,
                        category=category,
                        severity=severity
                    )
                    
                    return fallback_value
            
        return wrapper
    return decorator


# Global error handler instance
_global_error_handler = None


def get_global_error_handler() -> ErrorHandler:
    """Get or create global error handler instance"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def handle_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: Optional[ErrorCategory] = None
) -> ErrorInfo:
    """Convenience function for error handling"""
    return get_global_error_handler().handle_error(error, context, severity, category)


def setup_recovery_strategies():
    """Setup default recovery strategies"""
    error_handler = get_global_error_handler()
    
    def import_error_recovery(error_info: ErrorInfo):
        """Recovery strategy for import errors"""
        error_handler.logger.info("Attempting import error recovery...")
        # Could implement automatic package installation or fallback imports
    
    def config_error_recovery(error_info: ErrorInfo):
        """Recovery strategy for configuration errors"""
        error_handler.logger.info("Attempting configuration error recovery...")
        # Could implement default configuration loading
    
    def api_error_recovery(error_info: ErrorInfo):
        """Recovery strategy for API errors"""
        error_handler.logger.info("Attempting API error recovery...")
        # Could implement API key validation or endpoint switching
    
    error_handler.register_recovery_strategy(ErrorCategory.IMPORT_ERROR, import_error_recovery)
    error_handler.register_recovery_strategy(ErrorCategory.CONFIGURATION_ERROR, config_error_recovery)
    error_handler.register_recovery_strategy(ErrorCategory.API_ERROR, api_error_recovery)


# Initialize recovery strategies on import
setup_recovery_strategies()