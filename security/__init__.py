"""
RESE Security Module

Comprehensive security hardening for production deployment.

Components:
- Input validation and sanitization
- Error handling and recovery
- Resource limiting and monitoring
- Security auditing and testing
- Rate limiting and circuit breakers

Usage:
    from security import (
        InputValidator,
        ErrorHandler,
        ResourceMonitor,
        SecurityAuditor
    )

    # Validate input
    validator = InputValidator()
    is_valid, issues = validator.validate_problem_input(...)

    # Handle errors
    error_handler = ErrorHandler()
    error_handler.handle_error(exception, context)

    # Monitor resources
    monitor = ResourceMonitor()
    usage = monitor.get_current_usage()

    # Run security audit
    auditor = SecurityAuditor(target_path)
    report = auditor.run_full_audit()

Author: Agent M2 (Security and Reliability Specialist)
Created: 2025-12-31
"""

__version__ = "1.0.0"
__author__ = "Agent M2"

# Import all security components
from .input_validator import (
    InputValidator,
    SchemaValidator,
    SecurityIssue,
    SecuritySeverity,
    validate_input,
    sanitize_input
)

from .error_handler import (
    RESEError,
    ValidationError,
    ExecutionError,
    ResourceError,
    DependencyError,
    TimeoutError,
    SecurityError,
    ErrorCategory,
    ErrorSeverity,
    ErrorContext,
    ErrorDetails,
    ErrorHandler,
    handle_errors,
    safe_execute,
    retry_on_error,
    error_context,
    graceful_degradation,
    CircuitBreaker
)

from .resource_limiter import (
    ResourceLimits,
    ResourceMonitor,
    TimeoutManager,
    TimeoutException,
    QueuePriority,
    TaskQueue,
    RateLimiter,
    MemoryLimiter
)

from .security_audit import (
    Vulnerability,
    VulnerabilitySeverity,
    VulnerabilityCategory,
    SecurityAuditReport,
    StaticAnalyzer,
    DependencyScanner,
    PenetrationTester,
    SecurityAuditor
)

# Module-level convenience functions
def create_security_suite(
    strict_mode: bool = True,
    enable_monitoring: bool = True,
    log_file: str = None
) -> dict:
    """
    Create complete security suite with all components.

    Args:
        strict_mode: If True, reject on any validation issue
        enable_monitoring: If True, enable resource monitoring
        log_file: Optional log file path

    Returns:
        Dictionary with all security components
    """
    suite = {
        'input_validator': InputValidator(strict_mode=strict_mode),
        'error_handler': ErrorHandler(log_file=log_file),
        'resource_monitor': ResourceMonitor() if enable_monitoring else None,
        'rate_limiter': RateLimiter(rate_per_minute=60, burst_size=10),
        'timeout_manager': TimeoutManager(),
        'memory_limiter': MemoryLimiter(max_memory_mb=4096),
        'circuit_breaker': CircuitBreaker(failure_threshold=5, recovery_timeout=60.0),
    }

    if enable_monitoring and suite['resource_monitor']:
        suite['resource_monitor'].record_sample()

    return suite


def validate_request(
    description: str,
    constraints: list,
    variables: dict,
    suite: dict = None
) -> tuple:
    """
    Validate API request with security suite.

    Args:
        description: Problem description
        constraints: List of constraints
        variables: Problem variables
        suite: Optional security suite (creates if None)

    Returns:
        Tuple of (is_valid, issues, suite)
    """
    if suite is None:
        suite = create_security_suite()

    validator = suite['input_validator']
    is_valid, issues = validator.validate_problem_input(
        description, constraints, variables
    )

    return is_valid, issues, suite


def check_rate_limit(client_id: str, suite: dict = None) -> tuple:
    """
    Check if request is within rate limit.

    Args:
        client_id: Client identifier
        suite: Optional security suite

    Returns:
        Tuple of (allowed, remaining_tokens)
    """
    if suite is None:
        suite = create_security_suite()

    rate_limiter = suite['rate_limiter']
    allowed = rate_limiter.is_allowed(client_id)
    remaining = rate_limiter.get_remaining_tokens(client_id)

    return allowed, remaining


def execute_with_protection(
    func,
    *args,
    timeout_seconds: float = 300.0,
    max_retries: int = 3,
    suite: dict = None,
    **kwargs
):
    """
    Execute function with full security protection.

    Provides:
    - Timeout enforcement
    - Automatic retries
    - Error handling
    - Resource monitoring

    Args:
        func: Function to execute
        *args: Function arguments
        timeout_seconds: Maximum execution time
        max_retries: Maximum retry attempts
        suite: Security suite
        **kwargs: Function keyword arguments

    Returns:
        Function result or None on failure
    """
    if suite is None:
        suite = create_security_suite()

    # Apply retry decorator
    @retry_on_error(max_retries=max_retries, backoff_factor=1.0)
    def protected_func():
        # Execute with timeout
        timeout_mgr = suite['timeout_manager']
        return timeout_mgr.execute_with_timeout(
            func,
            timeout_seconds,
            graceful=True,
            *args,
            **kwargs
        )

    try:
        return protected_func()
    except Exception as e:
        # Handle error through error handler
        from security.error_handler import ErrorContext
        context = ErrorContext(
            component="execute_with_protection",
            operation=str(func.__name__)
        )
        suite['error_handler'].handle_error(e, context)
        return None


def run_security_audit(target_path: str, output_file: str = None) -> dict:
    """
    Run complete security audit on target path.

    Args:
        target_path: Path to audit
        output_file: Optional JSON output file

    Returns:
        Audit report dictionary
    """
    from pathlib import Path
    import json

    auditor = SecurityAuditor(Path(target_path))
    report = auditor.run_full_audit()

    report_dict = report.to_dict()

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2)

    return report_dict


def get_security_status(suite: dict = None) -> dict:
    """
    Get current security status from suite.

    Args:
        suite: Security suite (creates if None)

    Returns:
        Dictionary with security status
    """
    if suite is None:
        suite = create_security_suite()

    status = {
        'input_validator': {
            'strict_mode': suite['input_validator'].strict_mode,
            'issues_count': len(suite['input_validator'].issues)
        },
        'error_handler': suite['error_handler'].get_error_statistics(),
        'rate_limiter': {
            'active_clients': len(suite['rate_limiter'].client_buckets)
        },
        'circuit_breaker': {
            'state': suite['circuit_breaker'].state,
            'failure_count': suite['circuit_breaker'].failure_count
        }
    }

    if suite['resource_monitor']:
        status['resources'] = suite['resource_monitor'].get_current_usage()

    if suite['memory_limiter']:
        status['memory'] = suite['memory_limiter'].get_memory_usage()

    return status


# Export all components
__all__ = [
    # Version
    '__version__',
    '__author__',

    # Input validation
    'InputValidator',
    'SchemaValidator',
    'SecurityIssue',
    'SecuritySeverity',
    'validate_input',
    'sanitize_input',

    # Error handling
    'RESEError',
    'ValidationError',
    'ExecutionError',
    'ResourceError',
    'DependencyError',
    'TimeoutError',
    'SecurityError',
    'ErrorCategory',
    'ErrorSeverity',
    'ErrorContext',
    'ErrorDetails',
    'ErrorHandler',
    'handle_errors',
    'safe_execute',
    'retry_on_error',
    'error_context',
    'graceful_degradation',
    'CircuitBreaker',

    # Resource limiting
    'ResourceLimits',
    'ResourceMonitor',
    'TimeoutManager',
    'TimeoutException',
    'QueuePriority',
    'TaskQueue',
    'RateLimiter',
    'MemoryLimiter',

    # Security audit
    'Vulnerability',
    'VulnerabilitySeverity',
    'VulnerabilityCategory',
    'SecurityAuditReport',
    'StaticAnalyzer',
    'DependencyScanner',
    'PenetrationTester',
    'SecurityAuditor',

    # Convenience functions
    'create_security_suite',
    'validate_request',
    'check_rate_limit',
    'execute_with_protection',
    'run_security_audit',
    'get_security_status',
]
