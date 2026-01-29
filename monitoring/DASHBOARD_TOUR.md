# BubbleLab Dashboard Tour

Comprehensive walkthrough of all BubbleLab monitoring dashboards and their features.

## Table of Contents

1. [Dashboard Overview](#dashboard-overview)
2. [Overview Dashboard](#overview-dashboard)
3. [Per-Bubble Dashboard](#per-bubble-dashboard)
4. [Security Dashboard](#security-dashboard)
5. [Performance Dashboard](#performance-dashboard)
6. [Business Metrics Dashboard](#business-metrics-dashboard)
7. [Dashboard Customization](#dashboard-customization)
8. [Common Use Cases](#common-use-cases)

## Dashboard Overview

BubbleLab provides 5 pre-configured Grafana dashboards:

1. **Overview Dashboard**: System-wide metrics and health
2. **Per-Bubble Dashboard**: Individual bubble metrics
3. **Security Dashboard**: Security threats and authentication
4. **Performance Dashboard**: Latency and resource usage
5. **Business Metrics Dashboard**: Operational metrics

### Accessing Dashboards

1. Navigate to http://localhost:3000
2. Login with credentials (default: admin/changeme)
3. Find dashboards in **Dashboards > Browse**
4. Search for "BubbleLab"

## Overview Dashboard

**Purpose**: High-level view of system health and performance
**Auto-refresh**: 30 seconds
**Time range**: Last 1 hour (default)

### Key Panels

#### 1. Total Operations Per Second

**Type**: Time series graph
**Metrics**: Successful vs failed operations
**Use case**: Monitor system throughput

**What to look for**:
- Sudden drops indicate issues
- Spikes may indicate load increases
- Gap between success/failure shows error rate

**Queries**:
```promql
# Successful
sum(rate(bubble_operation_total{status="success"}[5m]))

# Failed
sum(rate(bubble_operation_total{status="error"}[5m]))
```

#### 2. Error Rate

**Type**: Gauge
**Metric**: Percentage of failed operations
**Thresholds**:
- Green: < 5%
- Yellow: 5-10%
- Red: > 10%

**What to look for**:
- Red zone requires immediate attention
- Increasing trend needs investigation
- Correlate with deployment times

**Query**:
```promql
sum(rate(bubble_error_total[5m])) / sum(rate(bubble_operation_total[5m])) * 100
```

#### 3. Average Operation Duration

**Type**: Time series graph
**Metrics**: P50, P95, P99 latency
**Use case**: Monitor performance degradation

**What to look for**:
- P95/99 spikes indicate outliers
- Gradual increase needs capacity planning
- Sudden spikes may indicate external issues

**Queries**:
```promql
# P50
histogram_quantile(0.50, sum(rate(bubble_operation_duration_seconds_bucket[5m])) by (le))

# P95
histogram_quantile(0.95, sum(rate(bubble_operation_duration_seconds_bucket[5m])) by (le))

# P99
histogram_quantile(0.99, sum(rate(bubble_operation_duration_seconds_bucket[5m])) by (le))
```

#### 4. Active Circuit Breakers

**Type**: Stat
**Metric**: Circuit breaker states
**Use case**: Identify failing services

**What to look for**:
- Red (Open): Service is failing, requests blocked
- Yellow (Half-Open): Testing if service recovered
- Green (Closed): Normal operation

**Query**:
```promql
circuit_breaker_state
```

#### 5. Rate Limit Violations

**Type**: Bar chart
**Metric**: Count of rate limit breaches
**Use case**: Detect abuse or capacity issues

**What to look for**:
- Sudden spikes may indicate abuse
- Continuous violations indicate need for scaling
- Pattern by bubble identifies problematic services

**Query**:
```promql
sum(increase(rate_limit_exceeded_total[5m])) by (bubble)
```

#### 6. Memory Usage by Bubble

**Type**: Time series graph
**Metric**: Memory consumption
**Use case**: Capacity planning and leak detection

**What to look for**:
- Steady increase suggests memory leak
- Sudden spikes may indicate load changes
- Consistent high usage needs scaling

**Query**:
```promql
sum(bubble_memory_usage_bytes) by (bubble)
```

## Per-Bubble Dashboard

**Purpose**: Detailed metrics for individual bubbles
**Template Variable**: Select bubble from dropdown
**Auto-refresh**: 30 seconds

### Template Variables

**Bubble**: Dropdown to select specific bubble
- Defaults to "All"
- Options populated from metrics
- Select specific bubble for detailed view

### Key Panels

#### 1. Operation Duration by Type

**Type**: Time series graph
**Metrics**: P50, P95, P99 by operation
**Use case**: Identify slow operations

**What to look for**:
- Compare operation types
- Find outliers
- Track performance over time

**Query**:
```promql
histogram_quantile(0.95,
  sum(rate(bubble_operation_duration_seconds_bucket{bubble="$bubble"}[5m])) by (le, operation)
)
```

#### 2. Error Rate by Operation Type

**Type**: Gauge
**Metric**: Error percentage by operation
**Use case**: Identify problematic operations

**What to look for**:
- Operations with highest error rate
- Correlate with deployment times
- Focus improvement efforts

**Query**:
```promql
sum(rate(bubble_error_total{bubble="$bubble"}[5m])) by (operation)
/ sum(rate(bubble_operation_total{bubble="$bubble"}[5m])) by (operation) * 100
```

#### 3. Throughput Over Time

**Type**: Time series graph
**Metric**: Operations per second
**Use case**: Monitor traffic patterns

**What to look for**:
- Peak usage times
- Traffic trends
- Capacity planning

**Query**:
```promql
sum(rate(bubble_operation_total{bubble="$bubble", status="success"}[1m]))
```

#### 4. Circuit Breaker Status

**Type**: Stat
**Metric**: Current circuit breaker state
**Use case**: Quick health check

**What to look for**:
- State transitions
- Time in open state
- Recovery patterns

**Query**:
```promql
circuit_breaker_state{bubble="$bubble"}
```

#### 5. Retry Rate by Operation

**Type**: Bar chart
**Metric**: Retry count by operation
**Use case**: Identify unreliable operations

**What to look for**:
- High retry operations need attention
- May indicate external dependency issues
- Consider circuit breaker configuration

**Query**:
```promql
sum(rate(bubble_operation_retry_total{bubble="$bubble"}[5m])) by (operation)
```

#### 6. Operations Distribution

**Type**: Pie chart
**Metric**: Operation count by type
**Use case**: Understand workload composition

**What to look for**:
- Most common operations
- Optimization targets
- Resource allocation

**Query**:
```promql
sum(bubble_operation_total{bubble="$bubble"}) by (operation)
```

## Security Dashboard

**Purpose**: Monitor security threats and authentication
**Auto-refresh**: 30 seconds

### Key Panels

#### 1. Authentication Failures

**Type**: Bar chart
**Metric**: Failed authentication attempts
**Use case**: Detect brute force attacks

**What to look for**:
- Sudden spikes indicate attacks
- Continuous failures need investigation
- Check for compromised credentials

**Query**:
```promql
sum(increase(bubble_authentication_error_total[5m])) by (bubble)
```

#### 2. Rate Limit Violations

**Type**: Bar chart
**Metric**: Rate limit breaches
**Use case**: Detect abuse or DoS attempts

**What to look for**:
- Sudden spikes may indicate attacks
- Patterns by bubble identify targets
- Consider adjusting rate limits

**Query**:
```promql
sum(increase(rate_limit_exceeded_total[5m])) by (bubble)
```

#### 3. Validation Errors by Type

**Type**: Pie chart
**Metric**: Validation error breakdown
**Use case**: Identify input quality issues

**What to look for**:
- Common validation failures
- Integration issues
- API documentation gaps

**Query**:
```promql
sum(bubble_validation_error_total) by (bubble, validation_error_type)
```

#### 4. Security Threats Blocked

**Type**: Time series graph
**Metric**: SQL injection and XSS attempts blocked
**Use case**: Verify security controls

**What to look for**:
- Blocked attempts show controls working
- Patterns indicate attack types
- Consider IP blocking for persistent attackers

**Queries**:
```promql
# SQL Injection
sum(increase(sql_injection_blocked_total[5m])) by (bubble)

# XSS
sum(increase(xss_blocked_total[5m])) by (bubble)
```

#### 5. Unauthorized Access Attempts

**Type**: Bar chart
**Metric**: Unauthorized access attempts
**Use case**: Detect permission issues or attacks

**What to look for**:
- Sudden spikes indicate attacks
- May indicate misconfigured permissions
- Check user access logs

**Query**:
```promql
sum(increase(unauthorized_access_total[5m])) by (bubble)
```

#### 6. Real-time Authentication Failure Rate

**Type**: Gauge
**Metric**: Auth failures per minute
**Use case**: Immediate threat detection

**What to look for**:
- Red zone requires immediate action
- Correlate with other security events
- May need to block IPs

**Query**:
```promql
sum(rate(bubble_authentication_error_total[1m]))
```

## Performance Dashboard

**Purpose**: Monitor latency and resource usage
**Auto-refresh**: 30 seconds

### Key Panels

#### 1. P95 and P99 Latency

**Type**: Time series graph
**Metric**: Percentile latency
**Use case**: Performance optimization

**What to look for**:
- P99 shows worst-case latency
- P95 shows typical worst-case
- Trends indicate performance changes

**Queries**:
```promql
# P95
histogram_quantile(0.95, sum(rate(bubble_operation_duration_seconds_bucket[5m])) by (le))

# P99
histogram_quantile(0.99, sum(rate(bubble_operation_duration_seconds_bucket[5m])) by (le))
```

#### 2. Memory Usage Trends

**Type**: Time series graph
**Metric**: Memory consumption over time
**Use case**: Capacity planning and leak detection

**What to look for**:
- Steady increase = memory leak
- Sudden spikes = load changes
- Consistent high usage = scaling needed

**Query**:
```promql
sum(bubble_memory_usage_bytes) by (bubble)
```

#### 3. CPU Usage Trends

**Type**: Time series graph
**Metric**: CPU utilization
**Use case**: Capacity planning

**What to look for**:
- Consistent high usage needs scaling
- Spikes correlate with traffic
- Identify CPU-intensive operations

**Query**:
```promql
rate(process_cpu_seconds_total[5m])
```

#### 4. Database Connection Pool Usage

**Type**: Time series graph
**Metric**: Connection pool utilization
**Use case**: Detect pool exhaustion

**What to look for**:
- Near 100% = pool exhausted
- May cause performance issues
- Consider increasing pool size

**Query**:
```promql
db_connection_pool_usage
```

#### 5. Request/Response Size Distribution

**Type**: Time series graph
**Metric**: P95 request and response sizes
**Use case**: Detect payload changes

**What to look for**:
- Sudden increases may indicate issues
- Large responses need optimization
- Consider compression

**Queries**:
```promql
# Request size
histogram_quantile(0.95, sum(rate(bubble_request_size_bytes_bucket[5m])) by (le, bubble))

# Response size
histogram_quantile(0.95, sum(rate(bubble_response_size_bytes_bucket[5m])) by (le, bubble))
```

#### 6. Operation Duration Heatmap

**Type**: Heatmap
**Metric**: Latency distribution
**Use case**: Visualize performance patterns

**What to look for**:
- Hot spots = slow operations
- Distribution shape shows consistency
- Compare across operations

**Query**:
```promql
sum(rate(bubble_operation_duration_seconds_bucket[5m])) by (le, bubble, operation)
```

## Business Metrics Dashboard

**Purpose**: Track operational KPIs and business metrics
**Auto-refresh**: 30 seconds

### Key Panels

#### 1. Requests Per Hour

**Type**: Bar chart
**Metric**: Hourly request volume
**Use case**: Understand traffic patterns

**What to look for**:
- Peak hours
- Daily patterns
- Growth trends

**Query**:
```promql
sum(increase(bubble_operation_total[1h])) by (bubble)
```

#### 2. Active Workflows

**Type**: Stat
**Metric**: Current active workflow count
**Use case**: Monitor system activity

**What to look for**:
- Unusual changes
- Capacity limits
- Workflow patterns

**Query**:
```promql
sum(active_workflows)
```

#### 3. Successful vs Failed Operations

**Type**: Pie chart
**Metric**: Operation breakdown
**Use case**: Overall system health

**What to look for**:
- Success ratio
- Identify problematic bubbles
- Track improvements

**Query**:
```promql
sum(bubble_operation_total) by (bubble, status)
```

#### 4. Top 10 Slowest Operations

**Type**: Bar chart
**Metric**: Average duration
**Use case**: Identify optimization targets

**What to look for**:
- Focus optimization efforts
- Compare performance
- Track improvements over time

**Query**:
```promql
topk(10, avg(bubble_operation_duration_seconds) by (bubble, operation))
```

#### 5. Top 10 Error-Prone Operations

**Type**: Bar chart
**Metric**: Error count
**Use case**: Prioritize fixes

**What to look for**:
- High-impact issues
- Focus debugging efforts
- Track fix effectiveness

**Query**:
```promql
topk(10, sum(rate(bubble_error_total[5m])) by (bubble, operation))
```

#### 6. Active Operations by Bubble

**Type**: Time series graph
**Metric**: Concurrent operations
**Use case**: Monitor load distribution

**What to look for**:
- Load imbalance
- Capacity utilization
- Scaling opportunities

**Query**:
```promql
sum(bubble_active_operations) by (bubble)
```

## Dashboard Customization

### Changing Time Range

1. Click time range selector (top right)
2. Choose preset or custom range
3. Common ranges: Last 1h, 6h, 24h, 7d

### Adjusting Refresh Rate

1. Click refresh interval (top right)
2. Choose: Off, 5s, 10s, 30s, 1m, 5m
3. Recommendation: 30s for most dashboards

### Adding Annotations

1. Hover over graph
2. Click to add annotation
3. Add deployment, incident, or change markers

### Exporting Panels

1. Click panel options (top right of panel)
2. Select "Export"
3. Choose format: CSV, JSON, PNG

### Sharing Dashboards

1. Click share (top right)
2. Copy link or embed code
3. Set time range and variables

## Common Use Cases

### Incident Response

1. **Start**: Overview Dashboard
2. **Check**: Error rate and circuit breakers
3. **Drill down**: Per-Bubble Dashboard for affected service
4. **Investigate**: Check Security Dashboard if related to auth
5. **Analyze**: Performance Dashboard for latency issues

### Performance Investigation

1. **Start**: Performance Dashboard
2. **Check**: P95/P99 latency trends
3. **Drill down**: Per-Bubble Dashboard for specific bubble
4. **Analyze**: Operation duration by type
5. **Identify**: Top slowest operations

### Capacity Planning

1. **Review**: Business Metrics Dashboard
2. **Analyze**: Traffic trends (30 days)
3. **Check**: Memory usage trends
4. **Evaluate**: CPU usage patterns
5. **Plan**: Scale-up timeline

### Security Monitoring

1. **Start**: Security Dashboard
2. **Review**: Authentication failures
3. **Check**: Blocked threats
4. **Analyze**: Rate limit violations
5. **Investigate**: Unauthorized access attempts

### Post-Deployment Verification

1. **Start**: Overview Dashboard
2. **Check**: Error rate (should not increase)
3. **Verify**: Latency (should not degrade)
4. **Confirm**: Circuit breakers closed
5. **Monitor**: For 1 hour after deployment

## Keyboard Shortcuts

- `t`: Open time range picker
- `h`: Hide all panels
- `Ctrl + S`: Save dashboard
- `Ctrl + R`: Refresh dashboard
- `Ctrl + Z`: Undo view changes
- `Esc`: Clear all selections

## Next Steps

- Import dashboards to your Grafana instance
- Customize for your environment
- Set up alerts based on dashboard thresholds
- Create custom dashboards for specific use cases

## Additional Resources

- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Querying](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/)
