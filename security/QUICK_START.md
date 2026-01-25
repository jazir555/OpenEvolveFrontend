# RESE Security Quick Start Guide

**Author:** Agent M2 (Security and Reliability Specialist)
**Version:** 1.0.0

---

## Quick Setup

### 1. Basic Usage

```python
# Import security components
from security import (
    InputValidator,
    ErrorHandler,
    ResourceMonitor,
    RateLimiter,
    SecurityAuditor
)

# Validate input
validator = InputValidator(strict_mode=True)
is_valid, issues = validator.validate_problem_input(
    description="Solve this problem",
    constraints=[{'id': 'c1', 'type': 'HARD', 'description': 'Test'}],
    variables={'x': 10}
)

if not is_valid:
    for issue in issues:
        print(f"[{issue.severity.value}] {issue.message}")
```

### 2. Error Handling

```python
from security import ErrorHandler, ErrorContext

# Create error handler
error_handler = ErrorHandler(log_file='logs/errors.log')

# Handle errors
context = ErrorContext(
    component="Phase1",
    operation="validate_constraints"
)

try:
    result = process_constraints(constraints)
except Exception as e:
    error_handler.handle_error(e, context)
```

### 3. Resource Monitoring

```python
from security import ResourceMonitor, ResourceLimits

# Create monitor
monitor = ResourceMonitor()

# Check usage
usage = monitor.get_current_usage()
print(f"Memory: {usage['memory_mb']:.1f} MB")
print(f"CPU: {usage['cpu_percent']:.1f}%")

# Check against limits
limits = ResourceLimits(max_memory_mb=4096)
within_limits, violations = monitor.check_limits(limits)
```

### 4. Rate Limiting

```python
from security import RateLimiter

# Create rate limiter
limiter = RateLimiter(rate_per_minute=60, burst_size=10)

# Check if allowed
if limiter.is_allowed(client_id):
    # Process request
    pass
else:
    # Rate limited
    return "Too many requests", 429
```

### 5. Security Audit

```python
from security import SecurityAuditor
from pathlib import Path

# Run audit
auditor = SecurityAuditor(Path("rese"))
report = auditor.run_full_audit()

# Print results
print(f"Security Score: {report.score}/100")
print(f"Vulnerabilities: {report.statistics['total_vulnerabilities']}")
```

---

## Integration with RESE Pipeline

### Protected Pipeline Execution

```python
from security.integration import RESESecurityIntegration

# Create security integration
security = RESESecurityIntegration(config={
    'strict_validation': True,
    'rate_limit_per_minute': 60,
    'max_memory_mb': 4096
})

# Validate input
is_valid, errors, sanitized = security.validate_pipeline_input(
    description=problem_description,
    constraints=constraints,
    variables=variables,
    client_id=user_id
)

if not is_valid:
    return {"error": errors}, 400

# Execute pipeline safely
success, result, error = security.execute_pipeline_safely(
    pipeline.run,
    problem,
    timeout_seconds=3600,
    client_id=user_id
)

if not success:
    return {"error": str(error)}, 500

return {"result": result}, 200
```

---

## Running Security Tests

### Run All Tests
```bash
cd rese/security
python -m pytest security_tests.py -v
```

### Run Specific Tests
```bash
# Input validation tests
python -m pytest security_tests.py::TestInputValidator -v

# Error handling tests
python -m pytest security_tests.py::TestErrorHandler -v

# Security audit tests
python -m pytest security_tests.py::TestSecurityAuditor -v
```

### With Coverage
```bash
python -m pytest security_tests.py --cov=. --cov-report=html
```

---

## Security Scanning

### Static Analysis
```python
from security import StaticAnalyzer

analyzer = StaticAnalyzer()
vulnerabilities = analyzer.analyze_directory(Path("rese"))

for vuln in vulnerabilities:
    print(f"[{vuln.severity.value}] {vuln.title}")
    print(f"  Location: {vuln.location}")
    print(f"  Fix: {vuln.remediation}")
```

### Full Security Audit
```bash
python -c "
from security import run_security_audit
report = run_security_audit('rese', 'security_report.json')
print(f'Security Score: {report[\"score\"]}/100')
"
```

---

## Common Security Patterns

### 1. Validate All Input
```python
from security import validate_input

is_valid, issues = validate_input(
    description, constraints, variables
)
if not is_valid:
    raise ValidationError(f"Invalid input: {issues}")
```

### 2. Execute with Timeout
```python
from security import TimeoutManager, TimeoutException

timeout_mgr = TimeoutManager()
try:
    result = timeout_mgr.execute_with_timeout(
        func, timeout_seconds=30.0
    )
except TimeoutException:
    print("Operation timed out")
```

### 3. Retry with Backoff
```python
from security import retry_on_error

@retry_on_error(max_retries=3, backoff_factor=1.0)
def unreliable_operation():
    # May fail temporarily
    pass
```

### 4. Circuit Breaker
```python
from security import CircuitBreaker

cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
try:
    result = cb.call(risky_function)
except Exception:
    result = fallback_function()
```

### 5. Rate Limit API Endpoint
```python
from security import RateLimiter

limiter = RateLimiter(rate_per_minute=60)

@app.post("/api/pipeline")
def run_pipeline(request):
    client_id = request.headers.get("X-Client-ID", "anonymous")

    if not limiter.is_allowed(client_id):
        return {"error": "Rate limit exceeded"}, 429

    # Process request
    return process_request(request)
```

---

## Configuration Examples

### Development
```python
config = {
    'strict_validation': False,
    'rate_limit_per_minute': 120,
    'max_memory_mb': 8192,
    'enable_monitoring': True
}
```

### Production
```python
config = {
    'strict_validation': True,
    'rate_limit_per_minute': 60,
    'rate_limit_burst': 10,
    'max_memory_mb': 4096,
    'max_time_seconds': 3600,
    'max_threads': 32,
    'circuit_failure_threshold': 5,
    'circuit_recovery_timeout': 60.0,
    'enable_monitoring': True,
    'error_log_file': 'logs/security_errors.log'
}
```

---

## Monitoring and Alerts

### Get Security Status
```python
from security import create_security_suite

suite = create_security_suite()
status = get_security_status(suite)

print(f"Active Clients: {status['rate_limiter']['active_clients']}")
print(f"Circuit State: {status['circuit_breaker']['state']}")
print(f"Memory Usage: {status['resources']['memory_mb']:.1f} MB")
```

### Get Security Events
```python
from security.integration import RESESecurityIntegration

security = RESESecurityIntegration()
events = security.get_security_events(limit=100, event_type="validation_failure")

for event in events:
    print(f"{event['timestamp']}: {event['event_type']}")
```

---

## Emergency Procedures

### 1. Disable System (Security Incident)
```python
# Set circuit breaker to open
security.circuit_breaker.state = 'open'
security.circuit_breaker.failure_count = 100
```

### 2. Block Client
```python
# Drain rate limit tokens
for _ in range(1000):
    security.rate_limiter.is_allowed(blocked_client_id)
```

### 3. Clear Rate Limits
```python
# Reset rate limit for client
security.rate_limiter.reset_client(client_id)
```

### 4. Emergency Cleanup
```python
# Clear caches, release resources
security.memory_limiter._trigger_cleanup()
security.resource_monitor.history.clear()
```

---

## Useful Commands

### Check Dependencies
```bash
pip install safety
safety check

pip install pip-audit
pip-audit
```

### Code Quality
```bash
pip install bandit
bandit -r rese/

pip install pyflakes
pyflakes rese/
```

### Type Checking
```bash
pip install mypy
mypy rese/security/
```

### Security Test
```bash
cd rese/security
python security_tests.py
```

---

## Troubleshooting

### Issue: Rate Limiting Too Aggressive
**Solution:** Increase rate limit or burst size
```python
limiter = RateLimiter(rate_per_minute=120, burst_size=20)
```

### Issue: Memory Exhaustion
**Solution:** Increase memory limit or enable cleanup
```python
memory_limiter = MemoryLimiter(max_memory_mb=8192)
memory_limiter.start_monitoring()
```

### Issue: Timeouts Too Short
**Solution:** Increase timeout
```python
result = timeout_mgr.execute_with_timeout(
    func, timeout_seconds=7200  # 2 hours
)
```

### Issue: Circuit Breaker Tripping
**Solution:** Increase failure threshold or recovery timeout
```python
cb = CircuitBreaker(failure_threshold=10, recovery_timeout=120.0)
```

---

## Support

- **Documentation:** `rese/security/SECURITY_HARDENING_GUIDE.md`
- **Tests:** `rese/security/security_tests.py`
- **Examples:** See test files for usage examples
- **Issues:** Report to security team

---

**Version:** 1.0.0
**Last Updated:** 2025-12-31
**Author:** Agent M2 (Security and Reliability Specialist)
