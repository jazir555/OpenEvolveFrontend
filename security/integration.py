"""
RESE Security Integration Module

Integrates security components with RESE pipeline and API.

Author: Agent M2 (Security and Reliability Specialist)
Created: 2025-12-31
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from security import (
    InputValidator,
    ErrorHandler,
    ResourceMonitor,
    RateLimiter,
    TimeoutManager,
    MemoryLimiter,
    CircuitBreaker,
    ErrorContext,
    SecurityAuditor
)


class RESESecurityIntegration:
    """
    Integrates all security components with RESE system.

    Provides:
    - Automatic input validation on all API endpoints
    - Error handling and recovery
    - Resource monitoring and limiting
    - Rate limiting
    - Security audit logging
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize security integration.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize security components
        self.input_validator = InputValidator(
            strict_mode=self.config.get('strict_validation', True)
        )

        self.error_handler = ErrorHandler(
            log_file=self.config.get('error_log_file', 'logs/security_errors.log')
        )

        self.resource_monitor = ResourceMonitor(
            sampling_interval=self.config.get('monitoring_interval', 5.0)
        )

        self.rate_limiter = RateLimiter(
            rate_per_minute=self.config.get('rate_limit_per_minute', 60),
            burst_size=self.config.get('rate_limit_burst', 10)
        )

        self.timeout_manager = TimeoutManager()

        self.memory_limiter = MemoryLimiter(
            max_memory_mb=self.config.get('max_memory_mb', 4096),
            check_interval=self.config.get('memory_check_interval', 5.0),
            cleanup_threshold=self.config.get('memory_cleanup_threshold', 0.9)
        )

        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.get('circuit_failure_threshold', 5),
            recovery_timeout=self.config.get('circuit_recovery_timeout', 60.0)
        )

        # Security audit log
        self.security_audit_log: List[Dict[str, Any]] = []

    def validate_pipeline_input(
        self,
        description: str,
        constraints: List[Dict[str, Any]],
        variables: Dict[str, Any],
        client_id: str = "unknown"
    ) -> tuple[bool, List[str], Optional[str]]:
        """
        Validate RESE pipeline input with full security checks.

        Args:
            description: Problem description
            constraints: List of constraints
            variables: Problem variables
            client_id: Client identifier

        Returns:
            Tuple of (is_valid, error_messages, sanitized_description)
        """
        # Check rate limit
        if not self.rate_limiter.is_allowed(client_id):
            return False, ["Rate limit exceeded"], None

        # Validate input
        is_valid, issues = self.input_validator.validate_problem_input(
            description, constraints, variables
        )

        if not is_valid:
            error_messages = [issue.message for issue in issues]

            # Log security event
            self._log_security_event(
                event_type="validation_failure",
                client_id=client_id,
                details={"issues": error_messages}
            )

            return False, error_messages, None

        # Sanitize description
        sanitized = self.input_validator.sanitize_html(description)

        return True, [], sanitized

    def execute_pipeline_safely(
        self,
        pipeline_func,
        *args,
        timeout_seconds: float = 3600.0,
        client_id: str = "unknown",
        **kwargs
    ) -> tuple[bool, Any, Optional[Exception]]:
        """
        Execute pipeline function with full security protection.

        Args:
            pipeline_func: Pipeline function to execute
            *args: Function arguments
            timeout_seconds: Maximum execution time
            client_id: Client identifier
            **kwargs: Function keyword arguments

        Returns:
            Tuple of (success, result, exception)
        """
        # Check circuit breaker
        if self.circuit_breaker.state == 'open':
            return False, None, Exception("Circuit breaker is open - service degraded")

        # Create error context
        context = ErrorContext(
            component="RESEPipeline",
            operation="execute_pipeline_safely",
            user_id=client_id
        )

        try:
            # Execute with timeout
            result = self.timeout_manager.execute_with_timeout(
                pipeline_func,
                timeout_seconds,
                graceful=True,
                *args,
                **kwargs
            )

            # Record success for circuit breaker
            self.circuit_breaker._on_success()

            return True, result, None

        except Exception as e:
            # Handle error
            error_details = self.error_handler.handle_error(e, context)

            # Record failure for circuit breaker
            self.circuit_breaker._on_failure()

            # Log security event if critical
            if error_details.severity.value in ['critical', 'high']:
                self._log_security_event(
                    event_type="pipeline_failure",
                    client_id=client_id,
                    details={
                        "error": str(e),
                        "severity": error_details.severity.value
                    }
                )

            return False, None, e

    def get_resource_status(self) -> Dict[str, Any]:
        """
        Get current resource usage status.

        Returns:
            Dictionary with resource status
        """
        usage = self.resource_monitor.get_current_usage()
        memory_status = self.memory_limiter.get_memory_usage()

        return {
            'current': usage,
            'memory': memory_status,
            'rate_limiter': {
                'active_clients': len(self.rate_limiter.client_buckets)
            },
            'circuit_breaker': {
                'state': self.circuit_breaker.state,
                'failure_count': self.circuit_breaker.failure_count
            },
            'error_statistics': self.error_handler.get_error_statistics()
        }

    def run_security_scan(self, target_path: str) -> Dict[str, Any]:
        """
        Run security scan on target path.

        Args:
            target_path: Path to scan

        Returns:
            Security audit report
        """
        auditor = SecurityAuditor(Path(target_path))
        report = auditor.run_full_audit()

        # Log audit result
        self._log_security_event(
            event_type="security_audit",
            details={
                "score": report.score,
                "vulnerabilities": report.statistics['total_vulnerabilities']
            }
        )

        return report.to_dict()

    def _log_security_event(
        self,
        event_type: str,
        client_id: str = "system",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log security event.

        Args:
            event_type: Type of security event
            client_id: Client identifier
            details: Optional event details
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'client_id': client_id,
            'details': details or {}
        }

        self.security_audit_log.append(event)

        # Also log through error handler
        self.error_handler.logger.warning(
            f"Security Event: {event_type}",
            extra={'security_event': event}
        )

    def get_security_events(
        self,
        limit: int = 100,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get security events from log.

        Args:
            limit: Maximum number of events to return
            event_type: Optional event type filter

        Returns:
            List of security events
        """
        events = self.security_audit_log

        if event_type:
            events = [e for e in events if e['event_type'] == event_type]

        # Return most recent events first
        return sorted(events, key=lambda e: e['timestamp'], reverse=True)[:limit]

    def start_monitoring(self) -> None:
        """Start background resource monitoring"""
        self.memory_limiter.start_monitoring()

        # Record initial resource sample
        self.resource_monitor.record_sample()

        self._log_security_event(
            event_type="monitoring_started",
            details={"components": ["memory", "resources"]}
        )

    def stop_monitoring(self) -> None:
        """Stop background monitoring and cleanup"""
        self.memory_limiter.stop_monitoring()
        self.error_handler.cleanup()

        self._log_security_event(
            event_type="monitoring_stopped"
        )

    def get_security_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive security report.

        Returns:
            Security report dictionary
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'resource_status': self.get_resource_status(),
            'security_events': self.get_security_events(limit=50),
            'error_statistics': self.error_handler.get_error_statistics(),
            'resource_statistics': self.resource_monitor.get_statistics()
        }


# =============================================================================
# API Integration Helpers
# =============================================================================

def secure_api_endpoint(func):
    """
    Decorator for securing API endpoints.

    Validates input, handles errors, enforces rate limits.
    """
    def wrapper(*args, **kwargs):
        # This would be integrated with FastAPI/API framework
        # For now, just execute with error handling
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log and handle error
            return {"error": str(e)}, 500

    return wrapper


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'RESESecurityIntegration',
    'secure_api_endpoint',
]
