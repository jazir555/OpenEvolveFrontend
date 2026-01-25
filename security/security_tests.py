"""
RESE Security: Comprehensive Test Suite

Security tests for all RESE components including:
- Input validation tests
- Error handling tests
- Resource limit tests
- Fault tolerance tests
- Recovery tests

Author: Agent M2 (Security and Reliability Specialist)
Created: 2025-12-31
"""

import pytest
import time
import threading
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from security.input_validator import (
    InputValidator,
    SecurityIssue,
    SecuritySeverity,
    validate_input
)
from security.error_handler import (
    ErrorHandler,
    ErrorContext,
    ErrorCategory,
    ErrorSeverity,
    RESEError,
    ValidationError,
    ResourceError,
    CircuitBreaker,
    retry_on_error,
    handle_errors
)
from security.resource_limiter import (
    ResourceMonitor,
    ResourceLimits,
    TimeoutManager,
    TimeoutException,
    TaskQueue,
    QueuePriority,
    RateLimiter,
    MemoryLimiter
)
from security.security_audit import (
    SecurityAuditor,
    StaticAnalyzer,
    SecurityAuditReport
)


# =============================================================================
# Input Validation Tests
# =============================================================================

class TestInputValidator:
    """Test suite for InputValidator"""

    def test_valid_input(self):
        """Test validation of valid input"""
        validator = InputValidator(strict_mode=False)

        description = "Solve this problem"
        constraints = [
            {
                'id': 'constraint_1',
                'type': 'HARD',
                'description': 'Test constraint'
            }
        ]
        variables = {'x': 10, 'y': 20}

        is_valid, issues = validator.validate_problem_input(
            description, constraints, variables
        )

        assert is_valid
        assert len(issues) == 0

    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection"""
        validator = InputValidator(strict_mode=False)

        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users--"
        ]

        for malicious_input in malicious_inputs:
            is_valid, issues = validator.validate_problem_input(
                malicious_input, [], {}
            )

            sql_issues = [i for i in issues if i.category == 'sql_injection']
            assert len(sql_issues) > 0, f"Failed to detect SQL injection in: {malicious_input}"

    def test_xss_detection(self):
        """Test XSS pattern detection"""
        validator = InputValidator(strict_mode=False)

        malicious_inputs = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')"
        ]

        for malicious_input in malicious_inputs:
            is_valid, issues = validator.validate_problem_input(
                malicious_input, [], {}
            )

            xss_issues = [i for i in issues if i.category == 'xss']
            assert len(xss_issues) > 0, f"Failed to detect XSS in: {malicious_input}"

    def test_code_injection_detection(self):
        """Test code injection pattern detection"""
        validator = InputValidator(strict_mode=False)

        malicious_inputs = [
            "__import__('os').system('rm -rf /')",
            "eval('print(1)')",
            "exec('import os')"
        ]

        for malicious_input in malicious_inputs:
            is_valid, issues = validator.validate_problem_input(
                malicious_input, [], {}
            )

            code_injection_issues = [i for i in issues if i.category == 'code_injection']
            assert len(code_injection_issues) > 0, f"Failed to detect code injection in: {malicious_input}"

    def test_max_length_validation(self):
        """Test maximum length enforcement"""
        validator = InputValidator(strict_mode=False)

        long_string = "a" * 20000  # Exceeds MAX_STRING_LENGTH

        is_valid, issues = validator.validate_problem_input(
            long_string, [], {}
        )

        length_issues = [i for i in issues if i.category == 'length_validation']
        assert len(length_issues) > 0

    def test_null_byte_detection(self):
        """Test null byte detection"""
        validator = InputValidator(strict_mode=False)

        malicious_input = "test\x00value"

        is_valid, issues = validator.validate_problem_input(
            malicious_input, [], {}
        )

        null_byte_issues = [i for i in issues if i.category == 'null_byte_injection']
        assert len(null_byte_issues) > 0

    def test_path_traversal_detection(self):
        """Test path traversal pattern detection"""
        validator = InputValidator(strict_mode=False)

        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "%2e%2e%2f"
        ]

        for malicious_path in malicious_paths:
            is_valid, issues = validator.validate_problem_input(
                malicious_path, [], {}
            )

            path_issues = [i for i in issues if i.category == 'path_traversal']
            assert len(path_issues) > 0, f"Failed to detect path traversal in: {malicious_path}"

    def test_identifier_validation(self):
        """Test identifier validation"""
        validator = InputValidator(strict_mode=True)

        # Valid identifiers
        valid_identifiers = ['test_var', '_private', 'Var123', 'test_123']
        for identifier in valid_identifiers:
            validator._validate_identifier(identifier, "test_field")
            assert len([i for i in validator.issues if 'identifier_validation' in i.category]) == 0
            validator.issues.clear()

        # Invalid identifiers
        invalid_identifiers = ['123test', 'test-var', 'test.var', 'test var']
        for identifier in invalid_identifiers:
            validator._validate_identifier(identifier, "test_field")
            identifier_issues = [i for i in validator.issues if 'identifier_validation' in i.category]
            assert len(identifier_issues) > 0
            validator.issues.clear()


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandler:
    """Test suite for ErrorHandler"""

    def test_error_classification(self):
        """Test error classification"""
        handler = ErrorHandler()
        context = ErrorContext(
            component="test_component",
            operation="test_operation"
        )

        # Test ValidationError
        validation_error = ValidationError("Invalid input", field="test_field")
        details = handler._classify_error(validation_error, context)
        assert details.category == ErrorCategory.VALIDATION

        # Test ResourceError
        resource_error = ResourceError("Out of memory", resource_type="memory")
        details = handler._classify_error(resource_error, context)
        assert details.category == ErrorCategory.RESOURCE

    def test_error_recovery_registration(self):
        """Test recovery strategy registration"""
        handler = ErrorHandler()
        context = ErrorContext(component="test", operation="test")

        recovery_called = []

        def recovery_strategy(error_details):
            recovery_called.append(True)
            return True

        handler.register_recovery_strategy(ErrorCategory.VALIDATION, recovery_strategy)

        error = ValidationError("Test error")
        handler.handle_error(error, context)

        assert len(recovery_called) > 0

    def test_circuit_breaker(self):
        """Test circuit breaker pattern"""
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0
        )

        # Successful call
        def success_func():
            return "success"

        result = circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == 'closed'

        # Failing calls
        def failing_func():
            raise ValueError("Test failure")

        for _ in range(3):
            try:
                circuit_breaker.call(failing_func)
            except:
                pass

        # Circuit should be open
        assert circuit_breaker.state == 'open'

        # Should raise exception when circuit is open
        try:
            circuit_breaker.call(success_func)
            assert False, "Should have raised exception"
        except Exception as e:
            assert "Circuit breaker is OPEN" in str(e)

    def test_retry_decorator(self):
        """Test retry decorator"""
        attempt_count = []

        @retry_on_error(max_retries=3, backoff_factor=0.1, retry_on=(ValueError,))
        def failing_function():
            attempt_count.append(1)
            if len(attempt_count) < 3:
                raise ValueError("Not yet")
            return "success"

        result = failing_function()
        assert result == "success"
        assert len(attempt_count) == 3


# =============================================================================
# Resource Limit Tests
# =============================================================================

class TestResourceMonitor:
    """Test suite for ResourceMonitor"""

    def test_get_current_usage(self):
        """Test getting current resource usage"""
        monitor = ResourceMonitor()
        usage = monitor.get_current_usage()

        assert 'memory_mb' in usage
        assert 'cpu_percent' in usage
        assert 'num_threads' in usage
        assert usage['memory_mb'] > 0

    def test_check_limits(self):
        """Test limit checking"""
        monitor = ResourceMonitor()
        limits = ResourceLimits(
            max_memory_mb=1000000,  # Very high limit
            max_cpu_percent=100,
            max_threads=1000
        )

        within_limits, violations = monitor.check_limits(limits)
        assert within_limits
        assert len(violations) == 0

    def test_record_sample(self):
        """Test recording usage samples"""
        monitor = ResourceMonitor()

        # Record multiple samples
        for _ in range(10):
            monitor.record_sample()

        assert len(monitor.history) == 10

        # Check statistics
        stats = monitor.get_statistics()
        assert 'memory_mb' in stats
        assert 'cpu_percent' in stats
        assert stats['sample_count'] == 10


class TestTimeoutManager:
    """Test suite for TimeoutManager"""

    def test_successful_execution(self):
        """Test successful execution within timeout"""
        timeout_mgr = TimeoutManager()

        def quick_function():
            return "success"

        result = timeout_mgr.execute_with_timeout(
            quick_function,
            timeout_seconds=5.0
        )

        assert result == "success"

    def test_timeout_exception(self):
        """Test timeout exception"""
        timeout_mgr = TimeoutManager()

        def slow_function():
            time.sleep(10)
            return "success"

        with pytest.raises(TimeoutException):
            timeout_mgr.execute_with_timeout(
                slow_function,
                timeout_seconds=1.0,
                graceful=False
            )

    def test_graceful_timeout(self):
        """Test graceful timeout (returns None)"""
        timeout_mgr = TimeoutManager()

        def slow_function():
            time.sleep(10)
            return "success"

        result = timeout_mgr.execute_with_timeout(
            slow_function,
            timeout_seconds=1.0,
            graceful=True
        )

        assert result is None


class TestRateLimiter:
    """Test suite for RateLimiter"""

    def test_rate_limiting(self):
        """Test rate limiting enforcement"""
        limiter = RateLimiter(rate_per_minute=10, burst_size=5)

        client_id = "test_client"

        # Should allow first burst
        for _ in range(5):
            assert limiter.is_allowed(client_id) == True

        # Exhaust tokens
        for _ in range(5):
            limiter.is_allowed(client_id)

        # Should be rate limited now
        assert limiter.is_allowed(client_id) == False

    def test_token_refill(self):
        """Test token refill over time"""
        limiter = RateLimiter(rate_per_minute=60, burst_size=5)

        client_id = "test_client"

        # Exhaust tokens
        for _ in range(5):
            limiter.is_allowed(client_id)

        # Should be limited
        assert limiter.is_allowed(client_id) == False

        # Wait for refill
        time.sleep(2)

        # Should allow some requests
        assert limiter.is_allowed(client_id) == True

    def test_client_isolation(self):
        """Test client isolation"""
        limiter = RateLimiter(rate_per_minute=10, burst_size=2)

        # Exhaust client1
        for _ in range(2):
            limiter.is_allowed("client1")

        # client1 should be limited
        assert limiter.is_allowed("client1") == False

        # client2 should still be allowed
        assert limiter.is_allowed("client2") == True


class TestMemoryLimiter:
    """Test suite for MemoryLimiter"""

    def test_memory_monitoring(self):
        """Test memory usage monitoring"""
        limiter = MemoryLimiter(
            max_memory_mb=1000,
            check_interval=0.1
        )

        usage = limiter.get_memory_usage()

        assert 'memory_mb' in usage
        assert 'max_memory_mb' in usage
        assert 'usage_percent' in usage
        assert usage['max_memory_mb'] == 1000


# =============================================================================
# Security Audit Tests
# =============================================================================

class TestStaticAnalyzer:
    """Test suite for StaticAnalyzer"""

    def test_sql_injection_detection(self):
        """Test SQL injection detection in code"""
        analyzer = StaticAnalyzer()

        # Create temporary file with SQL injection vulnerability
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return execute(query)
""")
            temp_file = Path(f.name)

        try:
            vulnerabilities = analyzer.analyze_file(temp_file)

            sql_injection_vulns = [
                v for v in vulnerabilities
                if v.category.name == 'INJECTION'
            ]

            assert len(sql_injection_vulns) > 0
        finally:
            temp_file.unlink()

    def test_hardcoded_secret_detection(self):
        """Test hardcoded secret detection"""
        analyzer = StaticAnalyzer()

        # Create temporary file with hardcoded secret
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
API_KEY = "sk-1234567890abcdef"
password = "admin123"
""")
            temp_file = Path(f.name)

        try:
            vulnerabilities = analyzer.analyze_file(temp_file)

            secret_vulns = [
                v for v in vulnerabilities
                if v.category.name == 'CONFIGURATION' and 'secret' in v.title.lower()
            ]

            assert len(secret_vulns) > 0
        finally:
            temp_file.unlink()

    def test_weak_cryptography_detection(self):
        """Test weak cryptography detection"""
        analyzer = StaticAnalyzer()

        # Create temporary file with weak crypto
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import hashlib
hash = hashlib.md5(data).hexdigest()
""")
            temp_file = Path(f.name)

        try:
            vulnerabilities = analyzer.analyze_file(temp_file)

            crypto_vulns = [
                v for v in vulnerabilities
                if v.category.name == 'CRYPTOGRAPHY'
            ]

            assert len(crypto_vulns) > 0
        finally:
            temp_file.unlink()


class TestSecurityAuditor:
    """Test suite for SecurityAuditor"""

    def test_full_audit(self):
        """Test complete security audit"""
        # Create temporary directory with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create vulnerable file
            (temp_path / "vulnerable.py").write_text("""
def process(user_input):
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    return execute(query)
""")

            # Create requirements.txt
            (temp_path / "requirements.txt").write_text("""
flask==0.12.0
jinja2==2.10
""")

            # Run audit
            auditor = SecurityAuditor(temp_path)
            report = auditor.run_full_audit()

            # Verify report structure
            assert report.scan_id is not None
            assert report.target == str(temp_path)
            assert isinstance(report.vulnerabilities, list)
            assert isinstance(report.statistics, dict)
            assert isinstance(report.score, float)
            assert 0 <= report.score <= 100

            # Should have found vulnerabilities
            assert len(report.vulnerabilities) > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestSecurityIntegration:
    """Integration tests for security components"""

    def test_validation_to_error_handling(self):
        """Test validation errors propagate to error handler"""
        error_handler = ErrorHandler()
        input_validator = InputValidator(strict_mode=True)

        context = ErrorContext(
            component="test",
            operation="validate_input"
        )

        # Malicious input
        is_valid, issues = input_validator.validate_problem_input(
            "'; DROP TABLE users; --",
            [],
            {}
        )

        assert not is_valid

        # Handle as error
        if not is_valid:
            error = ValidationError(
                f"Input validation failed with {len(issues)} issues",
                context=context
            )
            details = error_handler.handle_error(error, context)

            assert details.error_type == 'ValidationError'
            assert details.category == ErrorCategory.VALIDATION

    def test_resource_limits_with_timeout(self):
        """Test resource limits combined with timeout"""
        timeout_mgr = TimeoutManager()
        rate_limiter = RateLimiter(rate_per_minute=10)

        client_id = "integration_test"

        # Check rate limit first
        if not rate_limiter.is_allowed(client_id):
            pytest.skip("Rate limited")

        # Execute with timeout
        def quick_task():
            return "completed"

        result = timeout_mgr.execute_with_timeout(
            quick_task,
            timeout_seconds=5.0
        )

        assert result == "completed"


# =============================================================================
# Test Runner
# =============================================================================

def run_security_tests():
    """Run all security tests"""
    print("=" * 80)
    print("RESE Security Test Suite")
    print("=" * 80)

    # Run tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--color=yes'
    ])


if __name__ == '__main__':
    run_security_tests()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'TestInputValidator',
    'TestErrorHandler',
    'TestResourceMonitor',
    'TestTimeoutManager',
    'TestRateLimiter',
    'TestMemoryLimiter',
    'TestStaticAnalyzer',
    'TestSecurityAuditor',
    'TestSecurityIntegration',
    'run_security_tests',
]
