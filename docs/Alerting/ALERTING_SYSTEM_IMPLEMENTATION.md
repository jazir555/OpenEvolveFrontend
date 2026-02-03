# Alerting System Implementation

**Status**: ✅ COMPLETED
**Date**: 2026-02-02

## What Was Implemented

Created comprehensive alerting system for OpenEvolve Frontend with multi-channel notifications and persistent storage.

## Features Implemented

### 1. Alert Data Model
- **Alert** dataclass with comprehensive fields:
  - ID, title, description
  - Severity levels (info, warning, error, critical)
  - Status tracking (open, acknowledged, resolved, escalated)
  - Timestamps (created, updated, acknowledged, resolved)
  - Source and component tracking
  - Tags and metadata
  - Occurrence counting for deduplication

### 2. Alert Storage
Implemented multiple storage backends:

**InMemoryAlertStore**:
- Thread-safe in-memory storage
- Fast access and updates
- Perfect for testing and development

**PersistentAlertStore**:
- JSON file-based persistence
- Automatic save/load
- Survives restarts
- Thread-safe operations

### 3. NotificationService
Multi-channel notification support:

**Email Notifications**:
- SMTP configuration (Gmail, custom servers)
- TLS/STARTTLS support
- HTML and plain text emails
- Multiple recipients

**Slack Notifications**:
- Webhook-based integration
- Rich formatting with colors
- Severity-based color coding
- Structured fields display

**Generic Webhooks**:
- POST requests to any endpoint
- Custom headers support
- JSON payload format

**Console Logging**:
- Emoji-based severity indicators
- Structured logging
- Always available fallback

### 4. AlertManager
Main alert management system:

**Alert Creation**:
- Automatic ID generation
- Timestamp tracking
- Deduplication (configurable window)
- Occurrence counting

**Alert Operations**:
- Create new alerts
- Acknowledge alerts (with user tracking)
- Resolve alerts (with user tracking)
- Query alerts with filters

**Alert Analytics**:
- Statistics by severity
- Statistics by status
- Statistics by component
- Total counts

### 5. Alert Deduplication
- Configurable deduplication window (default: 5 minutes)
- Occurrence counting for repeated alerts
- First seen / last seen tracking
- Automatic cache cleanup on resolution

### 6. Graceful Degradation
When optional services unavailable:
- Email: Skips if credentials not configured
- Slack: Skips if webhook URL not configured
- Webhook: Skips if URL not configured
- Console: Always available

## Usage

### Basic Alert Creation

```python
from alerting_system import get_alert_manager, NotificationChannel

# Get alert manager
alert_manager = get_alert_manager()

# Create a simple alert
alert = alert_manager.create_alert(
    title="High memory usage detected",
    description="Memory usage exceeded 80% threshold",
    severity="warning",
    source="monitoring",
    component="system",
    tags=["memory", "resource"],
    notify_channels=[NotificationChannel.CONSOLE]
)

print(f"Alert created: {alert.id}")
```

### Email Notifications

```python
from alerting_system import get_alert_manager, NotificationConfig, NotificationChannel

# Configure email
config = NotificationConfig(
    email_enabled=True,
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    smtp_username="your-email@gmail.com",
    smtp_password="your-app-password",
    email_from="alerts@example.com",
    email_to=["ops@example.com", "admin@example.com"]
)

alert_manager = get_alert_manager(config=config)

# Create alert with email notification
alert = alert_manager.create_alert(
    title="Database connection failed",
    description="Unable to connect to production database",
    severity="critical",
    component="database",
    notify_channels=[NotificationChannel.EMAIL, NotificationChannel.CONSOLE]
)
```

### Slack Notifications

```python
from alerting_system import get_alert_manager, NotificationConfig, NotificationChannel

# Configure Slack
config = NotificationConfig(
    slack_enabled=True,
    slack_webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
)

alert_manager = get_alert_manager(config=config)

# Create alert with Slack notification
alert = alert_manager.create_alert(
    title="Deployment failed",
    description="Production deployment failed at step 3/5",
    severity="error",
    component="deployment",
    tags=["deployment", "production"],
    notify_channels=[NotificationChannel.SLACK]
)
```

### Alert Lifecycle Management

```python
# Acknowledge an alert
alert_manager.acknowledge_alert(alert_id="alert-20260202...", acknowledged_by="john.doe")

# Resolve an alert
alert_manager.resolve_alert(alert_id="alert-20260202...", resolved_by="john.doe")

# Query alerts
all_alerts = alert_manager.get_all_alerts()
critical_alerts = alert_manager.get_all_alerts(severity="critical")
open_alerts = alert_manager.get_all_alerts(status="open")
component_alerts = alert_manager.get_all_alerts(component="database")

# Get statistics
stats = alert_manager.get_alert_stats()
print(f"Total alerts: {stats['total']}")
print(f"By severity: {stats['by_severity']}")
print(f"By status: {stats['by_status']}")
```

### Custom Storage Backend

```python
from alerting_system import AlertManager, PersistentAlertStore

# Use persistent storage
storage = PersistentAlertStore(storage_path="alerts.json")
alert_manager = AlertManager(storage=storage)

# Alerts will persist across restarts
alert = alert_manager.create_alert(
    title="Test alert",
    description="This alert will persist",
    severity="info"
)
```

## Integration with KnowledgeAlertingNode

The knowledge_alerting_node.py has been updated to use the new alerting system:

```python
# In KnowledgeAlertingNode.__init__
alerting_module = self.safe_import(
    'alerting_system',
    fallback_value=None,
    error_msg="AlertingSystem not available"
)

if alerting_module:
    self.AlertManager = getattr(alerting_module, 'AlertManager', None)
    if self.AlertManager:
        self.alert_manager = self.AlertManager()
        self.logger.info("AlertManager initialized")
```

## Alert Deduplication Example

```python
# Create first alert
alert1 = alert_manager.create_alert(
    title="CPU usage high",
    description="CPU usage at 85%",
    severity="warning",
    component="system"
)
print(f"Alert {alert1.id} - occurrences: {alert1.occurrences}")  # occurrences: 1

# Create duplicate alert within deduplication window (5 minutes)
time.sleep(2)
alert2 = alert_manager.create_alert(
    title="CPU usage high",
    description="CPU usage at 87%",
    severity="warning",
    component="system"
)
print(f"Alert {alert2.id} - occurrences: {alert2.occurrences}")  # occurrences: 2
# Same alert ID, occurrences incremented
```

## Storage Format

Alerts are stored as JSON:

```json
[
  {
    "id": "alert-20260202143022-a1b2c3d4",
    "title": "High memory usage",
    "description": "Memory usage exceeded threshold",
    "severity": "warning",
    "status": "open",
    "source": "monitoring",
    "component": "system",
    "created_at": "2026-02-02T14:30:22",
    "updated_at": "2026-02-02T14:30:22",
    "acknowledged_at": null,
    "resolved_at": null,
    "acknowledged_by": null,
    "resolved_by": null,
    "metadata": {},
    "tags": ["memory", "resource"],
    "occurrences": 1,
    "first_seen": "2026-02-02T14:30:22",
    "last_seen": "2026-02-02T14:30:22"
  }
]
```

## Configuration

Environment variables or direct configuration:

```python
import os
from alerting_system import NotificationConfig

config = NotificationConfig(
    # Email settings
    email_enabled=os.getenv('ALERT_EMAIL_ENABLED', 'false').lower() == 'true',
    smtp_server=os.getenv('ALERT_SMTP_SERVER', 'smtp.gmail.com'),
    smtp_port=int(os.getenv('ALERT_SMTP_PORT', '587')),
    smtp_username=os.getenv('ALERT_SMTP_USERNAME'),
    smtp_password=os.getenv('ALERT_SMTP_PASSWORD'),
    email_from=os.getenv('ALERT_EMAIL_FROM'),
    email_to=os.getenv('ALERT_EMAIL_TO', '').split(','),

    # Slack settings
    slack_enabled=os.getenv('ALERT_SLACK_ENABLED', 'false').lower() == 'true',
    slack_webhook_url=os.getenv('ALERT_SLACK_WEBHOOK_URL'),

    # Webhook settings
    webhook_enabled=os.getenv('ALERT_WEBHOOK_ENABLED', 'false').lower() == 'true',
    webhook_url=os.getenv('ALERT_WEBHOOK_URL'),

    # Deduplication
    deduplication_window=int(os.getenv('ALERT_DEDUP_WINDOW', '300'))
)
```

## Files Created

1. `alerting_system.py` - Main alerting system implementation (650+ lines)
2. `bubblelabs_nodes/knowledge_alerting_node.py` - Updated with alerting integration

## Dependencies

### Required
- Python 3.8+
- `requests` - For webhook notifications
- `smtplib` - Standard library (email)
- `json` - Standard library (persistence)
- `dataclasses` - Standard library (data models)
- `threading` - Standard library (thread safety)
- `hashlib` - Standard library (ID generation)

### Optional
- SMTP server access (for email notifications)
- Slack webhook URL (for Slack notifications)
- Webhook endpoint (for generic webhooks)

## Testing

Test the alerting system:

```python
from alerting_system import get_alert_manager, NotificationChannel

# Test basic alert creation
manager = get_alert_manager()
alert = manager.create_alert(
    title="Test alert",
    description="This is a test",
    severity="info"
)

# Test persistence
from alerting_system import PersistentAlertStore
storage = PersistentAlertStore("test_alerts.json")
manager = get_alert_manager(storage=storage)

# Create alert
alert = manager.create_alert(title="Persistence test", description="...", severity="info")

# Reload
manager2 = get_alert_manager(storage=storage)
alerts = manager2.get_all_alerts()
print(f"Persisted alerts: {len(alerts)}")

# Cleanup
import os
os.remove("test_alerts.json")
```

## Next Steps

To further enhance the alerting system:

1. **Database Backend**: Add PostgreSQL/MongoDB support for distributed systems
2. **Alert Escalation**: Implement automatic escalation after timeout
3. **Alert Rules Engine**: Create DSL for complex alert conditions
4. **Alert Dashboard**: Build UI for viewing and managing alerts
5. **Alert Aggregation**: Group related alerts into incidents
6. **Metrics Integration**: Export alert metrics to Prometheus/Grafana

## Related Documentation

- `bubblelabs_nodes/knowledge_alerting_node.py` - Knowledge alerting node
- `INTEGRATION_PROGRESS_REPORT.md` - Overall integration progress

## Security Considerations

1. **Credentials**: Never hardcode SMTP passwords or API keys
2. **Environment Variables**: Use environment variables for sensitive data
3. **HTTPS**: Always use HTTPS for webhook URLs
4. **TLS**: Use STARTTLS for SMTP connections
5. **Rate Limiting**: Implement rate limiting for external APIs
6. **Input Validation**: Validate all alert data before storage

## Troubleshooting

**Email not sending**:
- Check SMTP credentials are correct
- Verify SMTP server and port
- Check firewall allows outbound connections
- For Gmail, use App Passwords not account password

**Slack webhook failing**:
- Verify webhook URL is correct
- Check workspace allows incoming webhooks
- Ensure JSON payload format is correct

**Persistence not working**:
- Check file permissions on storage path
- Verify disk space available
- Check JSON serialization errors
