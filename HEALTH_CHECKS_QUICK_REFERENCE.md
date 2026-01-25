# Health Checks Quick Reference Guide

## Quick Start

### Basic Usage (Legacy API - Backward Compatible)

```python
from health_checks import (
    is_llm_service_available,
    check_database_connectivity,
    check_cache_health
)

# Quick boolean checks
llm_ok = is_llm_service_available()
db_ok = check_database_connectivity()
cache_ok = check_cache_health()

print(f"LLM: {llm_ok}, DB: {db_ok}, Cache: {cache_ok}")
```

---

### Modern API (Recommended)

```python
from health_checks import HealthChecker, get_health_checker

# Get global instance
checker = get_health_checker()

# Check all services
results = checker.check_all_services()

for service, result in results.items():
    print(f"{service}: {result.status.value} - {result.message}")
```

---

## API Reference

### HealthChecker Class

#### Constructor

```python
HealthChecker(config: Optional[HealthCheckConfig] = None)
```

**Parameters:**
- `config`: Optional configuration object

#### Methods

##### check_llm_service()

```python
def check_llm_service(
    timeout_seconds: float = 30.0
) -> HealthCheckResult
```

Performs comprehensive LLM health check with actual API call.

**Returns:** `HealthCheckResult` with:
- `status`: HEALTHY, DEGRADED, UNHEALTHY, or UNKNOWN
- `message`: Descriptive status message
- `response_time_ms`: Response time in milliseconds
- `metadata`: Additional diagnostic info

##### check_database_health()

```python
def check_database_health(
    test_query: Optional[str] = None
) -> HealthCheckResult
```

Performs database health check with query execution and timing.

**Parameters:**
- `test_query`: Optional custom test query (default: "SELECT 1 as test")

**Returns:** `HealthCheckResult` with database metadata

##### check_cache_health()

```python
def check_cache_health() -> HealthCheckResult
```

Performs cache health check with write/read/integrity verification.

**Returns:** `HealthCheckResult` with cache timing info

##### check_all_services()

```python
def check_all_services() -> Dict[str, HealthCheckResult]
```

Runs all health checks and returns results.

**Returns:** Dictionary mapping service names to results

##### get_system_health()

```python
def get_system_health() -> Dict[str, Any]
```

Returns comprehensive system health summary.

**Returns:** Dictionary with:
- `overall_status`: System-wide status
- `total_services`: Number of services checked
- `healthy_services`: Count of healthy services
- `degraded_services`: Count of degraded services
- `unhealthy_services`: Count of unhealthy services
- `services`: Detailed service results
- `metrics`: All service metrics
- `recent_alerts`: Recent alerts
- `timestamp`: ISO timestamp

##### get_metrics()

```python
def get_metrics(service_name: str) -> Optional[ServiceMetrics]
```

Gets metrics for a specific service.

**Returns:** `ServiceMetrics` or None

##### get_all_metrics()

```python
def get_all_metrics() -> Dict[str, Dict[str, Any]]
```

Gets metrics for all services.

**Returns:** Dictionary of metric data

##### get_recent_alerts()

```python
def get_recent_alerts(
    limit: int = 10,
    severity: Optional[AlertSeverity] = None,
    service: Optional[str] = None
) -> List[Alert]
```

Gets recent alerts with optional filtering.

**Parameters:**
- `limit`: Maximum number of alerts
- `severity`: Filter by severity
- `service`: Filter by service name

**Returns:** List of Alert objects

---

## Data Models

### HealthStatus

```python
class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
```

### AlertSeverity

```python
class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
```

### HealthCheckResult

```python
@dataclass
class HealthCheckResult:
    component: str                    # Service name
    status: HealthStatus              # Status level
    message: str                      # Descriptive message
    response_time_ms: float           # Response time
    timestamp: datetime               # Check time
    metadata: Dict[str, Any]          # Additional info
    error: Optional[str]              # Error if failed
```

### ServiceMetrics

```python
@dataclass
class ServiceMetrics:
    service_name: str
    total_checks: int
    successful_checks: int
    failed_checks: int
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    uptime_percentage: float
    consecutive_failures: int
    last_check_time: Optional[datetime]
    last_failure_time: Optional[datetime]
```

### Alert

```python
@dataclass
class Alert:
    alert_id: str
    service: str
    severity: AlertSeverity
    message: str
    triggered_at: datetime
    resolved: bool
    resolved_at: Optional[datetime]
    metadata: Dict[str, Any]
```

---

## Configuration

### HealthCheckConfig

```python
@dataclass
class HealthCheckConfig:
    # Response time thresholds (milliseconds)
    llm_response_time_warning: float = 5000.0
    llm_response_time_critical: float = 10000.0
    db_query_time_warning: float = 1000.0
    db_query_time_critical: float = 5000.0
    cache_response_time_warning: float = 100.0
    cache_response_time_critical: float = 500.0

    # Failure thresholds
    max_consecutive_failures: int = 3
    uptime_warning_threshold: float = 95.0
    uptime_critical_threshold: float = 90.0

    # Test parameters
    llm_test_prompt: str = "Respond with 'OK' if you receive this."
    db_test_query: str = "SELECT 1 as test"

    # Alert callbacks
    alert_callbacks: List[Callable[[Alert], None]]

    # Enable/disable checks
    enable_llm_checks: bool = True
    enable_db_checks: bool = True
    enable_cache_checks: bool = True
```

---

## Common Patterns

### Pattern 1: Quick Health Check

```python
from health_checks import get_health_checker

checker = get_health_checker()
health = checker.get_system_health()

if health['overall_status'] == 'healthy':
    print("System is healthy")
else:
    print(f"System has issues: {health['overall_status']}")
```

---

### Pattern 2: Custom Thresholds

```python
from health_checks import HealthChecker, HealthCheckConfig

config = HealthCheckConfig(
    llm_response_time_warning=3000.0,   # Stricter
    db_query_time_critical=2000.0,       # Stricter
    uptime_warning_threshold=98.0        # Higher
)

checker = HealthChecker(config=config)
```

---

### Pattern 3: Alert Callbacks

```python
def slack_alert(alert):
    print(f"Slack: {alert.message}")

def email_alert(alert):
    if alert.severity == AlertSeverity.CRITICAL:
        print(f"Email: {alert.message}")

config = HealthCheckConfig(
    alert_callbacks=[slack_alert, email_alert]
)
```

---

### Pattern 4: Metrics Analysis

```python
from health_checks import get_health_checker
import time

checker = get_health_checker()

# Run periodic checks
for _ in range(10):
    checker.check_all_services()
    time.sleep(60)

# Analyze metrics
metrics = checker.get_all_metrics()
for service, data in metrics.items():
    print(f"{service}:")
    print(f"  Uptime: {data['uptime_percentage']:.1f}%")
    print(f"  Avg Time: {data['avg_response_time_ms']:.1f}ms")
```

---

### Pattern 5: HTTP Health Endpoint

```python
from flask import Flask, jsonify
from health_checks import get_health_checker

app = Flask(__name__)

@app.route('/health')
def health():
    checker = get_health_checker()
    return jsonify(checker.get_system_health())

@app.route('/health/<service>')
def service_health(service):
    checker = get_health_checker()

    if service == 'llm':
        result = checker.check_llm_service()
    elif service == 'database':
        result = checker.check_database_health()
    elif service == 'cache':
        result = checker.check_cache_health()
    else:
        return jsonify({'error': 'Unknown service'}), 404

    return jsonify({
        'service': result.component,
        'status': result.status.value,
        'message': result.message,
        'response_time_ms': result.response_time_ms
    })
```

---

### Pattern 6: Conditional Execution

```python
from health_checks import HealthChecker, HealthCheckConfig, HealthStatus

config = HealthCheckConfig(
    enable_llm_checks=False,  # Disable expensive checks
    enable_db_checks=True,
    enable_cache_checks=True
)

checker = HealthChecker(config=config)

# Only proceed if healthy
results = checker.check_all_services()
if results['database'].status == HealthStatus.HEALTHY:
    # Safe to proceed with database operations
    pass
else:
    # Handle database issue
    logger.error("Database not healthy")
```

---

## Troubleshooting

### Health Check Fails

**Symptom:** All health checks return UNHEALTHY

**Diagnosis:**
```python
# Check individual services
checker = get_health_checker()

# Test database
try:
    from sovereign_persistence import SovereignDatabase
    db = SovereignDatabase()
    print(f"DB health: {db.check_health()}")
except Exception as e:
    print(f"DB error: {e}")

# Test LLM client
try:
    from openevolve_client import get_client, OPENEVOLVE_AVAILABLE
    client = get_client()
    print(f"LLM available: {OPENEVOLVE_AVAILABLE}")
    print(f"Client exists: {client is not None}")
except Exception as e:
    print(f"LLM error: {e}")
```

---

### Slow Response Times

**Symptom:** Response times exceed thresholds

**Solutions:**

1. **Increase thresholds:**
```python
config = HealthCheckConfig(
    llm_response_time_critical=20000.0,  # 20 seconds
    db_query_time_critical=10000.0       # 10 seconds
)
```

2. **Use ping-only check:**
```python
# Use lightweight check instead of full API call
result = checker.check_llm_service_ping_only()
```

3. **Check actual service performance:**
```python
result = checker.check_database_health()
print(f"Query time: {result.metadata['query_time_ms']:.2f}ms")
print(f"DB stats: {result.metadata['db_stats']}")
```

---

### Too Many Alerts

**Symptom:** Excessive alert notifications

**Solutions:**

1. **Adjust thresholds:**
```python
config = HealthCheckConfig(
    max_consecutive_failures=5,        # Require more failures
    uptime_warning_threshold=90.0,     # Lower bar
    llm_response_time_warning=10000.0  # Higher bar
)
```

2. **Filter alerts by severity:**
```python
# Only get critical alerts
critical_alerts = checker.get_recent_alerts(
    severity=AlertSeverity.CRITICAL
)
```

3. **Clear old alerts:**
```python
from datetime import timedelta

# Clear alerts older than 1 hour
checker.clear_alerts(older_than=timedelta(hours=1))
```

---

## Performance Tips

### 1. Use Ping Checks for Frequent Monitoring

```python
# For frequent checks (every few seconds)
ping_result = checker.check_llm_service_ping_only()

# For full checks (every few minutes)
full_result = checker.check_llm_service()
```

### 2. Run Checks in Parallel

```python
from concurrent.futures import ThreadPoolExecutor

checker = get_health_checker()

with ThreadPoolExecutor(max_workers=3) as executor:
    future_llm = executor.submit(checker.check_llm_service)
    future_db = executor.submit(checker.check_database_health)
    future_cache = executor.submit(checker.check_cache_health)

    results = {
        'llm': future_llm.result(),
        'database': future_db.result(),
        'cache': future_cache.result()
    }
```

### 3. Disable Expensive Checks

```python
config = HealthCheckConfig(
    enable_llm_checks=False  # Skip expensive LLM calls
)
```

---

## Best Practices

### 1. Start Simple

```python
# Start with legacy API
if is_llm_service_available():
    # Use LLM service
    pass
```

### 2. Add Detail as Needed

```python
# Upgrade to detailed checks when you need more info
checker = get_health_checker()
result = checker.check_llm_service()

if result.status == HealthStatus.HEALTHY:
    # Use service
    pass
elif result.status == HealthStatus.DEGRADED:
    # Use with caution
    logger.warning(f"LLM degraded: {result.response_time_ms:.2f}ms")
else:
    # Don't use
    logger.error(f"LLM unavailable: {result.error}")
```

### 3. Monitor Trends

```python
# Track metrics over time
metrics = checker.get_metrics('llm_service')

if metrics.consecutive_failures >= 2:
    # Alert before it becomes critical
    send_warning("LLM service unstable")
```

### 4. Set Appropriate Thresholds

```python
# Development: Lenient thresholds
dev_config = HealthCheckConfig(
    llm_response_time_critical=30000.0,
    uptime_critical_threshold=80.0
)

# Production: Strict thresholds
prod_config = HealthCheckConfig(
    llm_response_time_critical=10000.0,
    uptime_critical_threshold=99.0
)
```

### 5. Use Alerts Wisely

```python
# Only alert on critical issues in production
def production_alert_filter(alert):
    return alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]

config = HealthCheckConfig(
    alert_callbacks=[production_alert_filter]
)
```

---

## Complete Example

```python
#!/usr/bin/env python3
"""
Complete health monitoring example
"""

import logging
from health_checks import (
    HealthChecker,
    HealthCheckConfig,
    AlertSeverity,
    HealthStatus
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Custom alert callback
def alert_handler(alert):
    """Handle alerts."""
    print(f"\n🚨 ALERT [{alert.severity.value.upper()}]")
    print(f"   Service: {alert.service}")
    print(f"   Message: {alert.message}")
    print()

# Configuration
config = HealthCheckConfig(
    # Thresholds
    llm_response_time_warning=5000.0,
    llm_response_time_critical=10000.0,
    db_query_time_warning=1000.0,
    db_query_time_critical=5000.0,

    # Failure tolerance
    max_consecutive_failures=3,
    uptime_warning_threshold=95.0,
    uptime_critical_threshold=90.0,

    # Alert callback
    alert_callbacks=[alert_handler],

    # Enable all checks
    enable_llm_checks=True,
    enable_db_checks=True,
    enable_cache_checks=True
)

# Initialize
checker = HealthChecker(config=config)

# Check all services
print("=" * 60)
print("HEALTH CHECK RESULTS")
print("=" * 60)

results = checker.check_all_services()

for service, result in results.items():
    status_icon = {
        HealthStatus.HEALTHY: "✅",
        HealthStatus.DEGRADED: "⚠️",
        HealthStatus.UNHEALTHY: "❌",
        HealthStatus.UNKNOWN: "❓"
    }.get(result.status, "❓")

    print(f"\n{status_icon} {service.upper()}")
    print(f"   Status: {result.status.value}")
    print(f"   Message: {result.message}")
    print(f"   Response Time: {result.response_time_ms:.2f}ms")

    if result.metadata:
        print(f"   Metadata:")
        for key, value in result.metadata.items():
            if key != 'db_stats':  # Skip verbose stats
                print(f"     {key}: {value}")

# System summary
print("\n" + "=" * 60)
print("SYSTEM SUMMARY")
print("=" * 60)

system_health = checker.get_system_health()
print(f"\nOverall Status: {system_health['overall_status'].upper()}")
print(f"Healthy: {system_health['healthy_services']}/{system_health['total_services']}")
print(f"Degraded: {system_health['degraded_services']}")
print(f"Unhealthy: {system_health['unhealthy_services']}")

# Metrics
print("\n" + "=" * 60)
print("METRICS")
print("=" * 60)

metrics = system_health['metrics']
for service_name, service_metrics in metrics.items():
    if service_metrics['total_checks'] > 0:
        print(f"\n{service_name.upper()}")
        print(f"   Total Checks: {service_metrics['total_checks']}")
        print(f"   Success Rate: {service_metrics['uptime_percentage']:.1f}%")
        print(f"   Avg Response: {service_metrics['avg_response_time_ms']:.2f}ms")
        print(f"   Min Response: {service_metrics['min_response_time_ms']:.2f}ms")
        print(f"   Max Response: {service_metrics['max_response_time_ms']:.2f}ms")

# Recent alerts
if system_health['recent_alerts']:
    print("\n" + "=" * 60)
    print("RECENT ALERTS")
    print("=" * 60)

    for alert in system_health['recent_alerts']:
        print(f"\n[{alert['severity'].upper()}] {alert['service']}")
        print(f"   {alert['message']}")
else:
    print("\n✅ No alerts")

print("\n" + "=" * 60)
print("Health check complete!")
print("=" * 60)
```

---

## Quick Reference Card

### Import Options

```python
# Option 1: Legacy API (backward compatible)
from health_checks import is_llm_service_available, check_database_connectivity

# Option 2: Modern API (recommended)
from health_checks import HealthChecker, get_health_checker

# Option 3: Full import
from health_checks import *
```

### Common Operations

```python
# Quick check
checker = get_health_checker()
health = checker.get_system_health()

# Check specific service
result = checker.check_llm_service()
result = checker.check_database_health()
result = checker.check_cache_health()

# View metrics
metrics = checker.get_all_metrics()

# View alerts
alerts = checker.get_recent_alerts(limit=10)
```

### Status Values

- `HEALTHY` ✅: Service operating normally
- `DEGRADED` ⚠️: Service working but slow
- `UNHEALTHY` ❌: Service failed
- `UNKNOWN` ❓: Unable to determine

### Alert Severities

- `INFO`: Informational
- `WARNING`: Warning
- `ERROR`: Error
- `CRITICAL`: Critical

---

**Need More Info?** See `HEALTH_CHECKS_ENHANCEMENT_REPORT.md` for detailed documentation.
