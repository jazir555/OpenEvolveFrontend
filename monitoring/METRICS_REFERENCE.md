# BubbleLab Metrics Reference

Complete catalog of all metrics collected by BubbleLab monitoring infrastructure.

## Table of Contents

1. [Operation Metrics](#operation-metrics)
2. [Circuit Breaker Metrics](#circuit-breaker-metrics)
3. [Rate Limiting Metrics](#rate-limiting-metrics)
4. [Error Metrics](#error-metrics)
5. [Security Metrics](#security-metrics)
6. [Performance Metrics](#performance-metrics)
7. [Business Metrics](#business-metrics)
8. [System Metrics](#system-metrics)

## Operation Metrics

### bubble_operation_duration_seconds

**Type**: Histogram
**Description**: Duration of bubble operations in seconds
**Labels**:
- `bubble`: Name of the bubble
- `operation`: Operation type (e.g., "createPayment", "sendEmail")
- `status`: Operation status ("success", "error")

**Buckets**: 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60, 300

**Example Queries**:
```promql
# Average operation duration
rate(bubble_operation_duration_seconds_sum[5m])
/ rate(bubble_operation_duration_seconds_count[5m])

# P95 latency
histogram_quantile(0.95,
  sum(rate(bubble_operation_duration_seconds_bucket[5m])) by (le, bubble)
)

# Slowest operations
topk(10, avg(bubble_operation_duration_seconds) by (bubble, operation))
```

### bubble_operation_total

**Type**: Counter
**Description**: Total number of bubble operations
**Labels**:
- `bubble`: Name of the bubble
- `operation`: Operation type
- `status`: Operation status ("success", "error")

**Example Queries**:
```promql
# Operations per second
sum(rate(bubble_operation_total[5m])) by (bubble)

# Success rate
sum(rate(bubble_operation_total{status="success"}[5m]))
/ sum(rate(bubble_operation_total[5m]))

# Total operations in last hour
increase(bubble_operation_total[1h])
```

### bubble_operation_retry_total

**Type**: Counter
**Description**: Total number of operation retries
**Labels**:
- `bubble`: Name of the bubble
- `operation`: Operation type

**Example Queries**:
```promql
# Retry rate
sum(rate(bubble_operation_retry_total[5m])) by (bubble)

# Retry percentage
sum(rate(bubble_operation_retry_total[5m]))
/ sum(rate(bubble_operation_total[5m]))
```

## Circuit Breaker Metrics

### circuit_breaker_state

**Type**: Gauge
**Description**: Current state of circuit breakers
**Labels**:
- `bubble`: Name of the bubble
- `state`: Circuit breaker state (0=closed, 1=open, 2=half_open)

**Example Queries**:
```promql
# All open circuit breakers
circuit_breaker_state{state="open"} == 1

# Circuit breaker status summary
count(circuit_breaker_state) by (bubble, state)
```

### circuit_breaker_failure_total

**Type**: Counter
**Description**: Total number of circuit breaker failures
**Labels**:
- `bubble`: Name of the bubble

**Example Queries**:
```promql
# Failure rate
sum(rate(circuit_breaker_failure_total[5m])) by (bubble)

# Recent failures
increase(circuit_breaker_failure_total[1h])
```

### circuit_breaker_success_total

**Type**: Counter
**Description**: Total number of circuit breaker successes
**Labels**:
- `bubble`: Name of the bubble

**Example Queries**:
```promql
# Success rate
sum(rate(circuit_breaker_success_total[5m])) by (bubble)

# Success/failure ratio
sum(rate(circuit_breaker_success_total[5m]))
/ sum(rate(circuit_breaker_failure_total[5m]))
```

## Rate Limiting Metrics

### rate_limit_exceeded_total

**Type**: Counter
**Description**: Total number of rate limit violations
**Labels**:
- `bubble`: Name of the bubble

**Example Queries**:
```promql
# Rate limit violations per second
sum(rate(rate_limit_exceeded_total[5m])) by (bubble)

# Recent violations
increase(rate_limit_exceeded_total[1h])
```

### rate_limit_remaining

**Type**: Gauge
**Description**: Remaining rate limit quota
**Labels**:
- `bubble`: Name of the bubble

**Example Queries**:
```promql
# Current quota
sum(rate_limit_remaining) by (bubble)

# Bubbles near quota limit
rate_limit_remaining < 100
```

## Error Metrics

### bubble_error_total

**Type**: Counter
**Description**: Total number of bubble errors
**Labels**:
- `bubble`: Name of the bubble
- `error_type`: Type of error (e.g., "database_error", "network_error")
- `operation`: Operation where error occurred

**Example Queries**:
```promql
# Error rate
sum(rate(bubble_error_total[5m])) by (bubble)

# Errors by type
sum(rate(bubble_error_total[5m])) by (error_type)

# Error percentage
sum(rate(bubble_error_total[5m])) / sum(rate(bubble_operation_total[5m]))
```

### bubble_validation_error_total

**Type**: Counter
**Description**: Total number of validation errors
**Labels**:
- `bubble`: Name of the bubble
- `validation_error_type`: Type of validation error

**Example Queries**:
```promql
# Validation error rate
sum(rate(bubble_validation_error_total[5m])) by (bubble)

# Validation errors by type
sum(rate(bubble_validation_error_total[5m])) by (validation_error_type)
```

### bubble_authentication_error_total

**Type**: Counter
**Description**: Total number of authentication errors
**Labels**:
- `bubble`: Name of the bubble

**Example Queries**:
```promql
# Authentication failure rate
sum(rate(bubble_authentication_error_total[1m])) by (bubble)

# Detect authentication spikes
rate(bubble_authentication_error_total[1m]) > 10
```

## Security Metrics

### sql_injection_blocked_total

**Type**: Counter
**Description**: Total number of blocked SQL injection attempts
**Labels**:
- `bubble`: Name of the bubble

**Example Queries**:
```promql
# SQL injection attempts
sum(increase(sql_injection_blocked_total[5m])) by (bubble)

# Bubbles under attack
rate(sql_injection_blocked_total[5m]) > 0
```

### xss_blocked_total

**Type**: Counter
**Description**: Total number of blocked XSS attempts
**Labels**:
- `bubble`: Name of the bubble

**Example Queries**:
```promql
# XSS attempts
sum(increase(xss_blocked_total[5m])) by (bubble)

# Total attacks blocked
sum(sql_injection_blocked_total) + sum(xss_blocked_total)
```

### unauthorized_access_total

**Type**: Counter
**Description**: Total number of unauthorized access attempts
**Labels**:
- `bubble`: Name of the bubble

**Example Queries**:
```promql
# Unauthorized access rate
sum(rate(unauthorized_access_total[5m])) by (bubble)

# Suspicious activity
rate(unauthorized_access_total[1m]) > 5
```

## Performance Metrics

### bubble_request_size_bytes

**Type**: Histogram
**Description**: Size of requests in bytes
**Labels**:
- `bubble`: Name of the bubble
- `operation`: Operation type

**Buckets**: 100, 1000, 10000, 100000, 1000000, 10000000

**Example Queries**:
```promql
# Average request size
rate(bubble_request_size_bytes_sum[5m])
/ rate(bubble_request_size_bytes_count[5m])

# P95 request size
histogram_quantile(0.95,
  sum(rate(bubble_request_size_bytes_bucket[5m])) by (le, bubble)
)
```

### bubble_response_size_bytes

**Type**: Histogram
**Description**: Size of responses in bytes
**Labels**:
- `bubble`: Name of the bubble
- `operation`: Operation type

**Buckets**: 100, 1000, 10000, 100000, 1000000, 10000000

**Example Queries**:
```promql
# Average response size
rate(bubble_response_size_bytes_sum[5m])
/ rate(bubble_response_size_bytes_count[5m])

# P95 response size
histogram_quantile(0.95,
  sum(rate(bubble_response_size_bytes_bucket[5m])) by (le, bubble)
)
```

### bubble_memory_usage_bytes

**Type**: Gauge
**Description**: Memory usage in bytes
**Labels**:
- `bubble`: Name of the bubble

**Example Queries**:
```promql
# Memory usage by bubble
sum(bubble_memory_usage_bytes) by (bubble)

# Total memory usage
sum(bubble_memory_usage_bytes)

# Memory usage percentage
sum(bubble_memory_usage_bytes) / sum(bubble_memory_limit_bytes)
```

### db_connection_pool_usage

**Type**: Gauge
**Description**: Database connection pool usage
**Labels**:
- `bubble`: Name of the bubble
- `pool_type`: Type of pool ("active", "idle", "max")

**Example Queries**:
```promql
# Connection pool utilization
sum(db_connection_pool_usage{pool_type="active"}) by (bubble)
/ sum(db_connection_pool_usage{pool_type="max"}) by (bubble)

# Available connections
sum(db_connection_pool_usage{pool_type="idle"}) by (bubble)
```

## Business Metrics

### bubble_active_operations

**Type**: Gauge
**Description**: Number of currently active operations
**Labels**:
- `bubble`: Name of the bubble

**Example Queries**:
```promql
# Active operations by bubble
sum(bubble_active_operations) by (bubble)

# Total active operations
sum(bubble_active_operations)

# Operations backlog
bubble_active_operations > 1000
```

### bubble_throughput_per_second

**Type**: Gauge
**Description**: Operations per second
**Labels**:
- `bubble`: Name of the bubble

**Example Queries**:
```promql
# Throughput by bubble
sum(bubble_throughput_per_second) by (bubble)

# Total throughput
sum(bubble_throughput_per_second)

# Low throughput detection
bubble_throughput_per_second < 1
```

### active_workflows

**Type**: Gauge
**Description**: Number of active workflows
**Labels**:
- `bubble`: Name of the bubble

**Example Queries**:
```promql
# Active workflows
sum(active_workflows) by (bubble)

# Total active workflows
sum(active_workflows)

# Workflow distribution
count(active_workflows) by (bubble)
```

## System Metrics

### process_cpu_seconds_total

**Type**: Counter
**Description**: Total CPU time in seconds

**Example Queries**:
```promql
# CPU usage percentage
rate(process_cpu_seconds_total[5m]) * 100
```

### node_memory_MemAvailable_bytes

**Type**: Gauge
**Description**: Available memory in bytes

**Example Queries**:
```promql
# Memory usage percentage
(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100
```

### node_filesystem_avail_bytes

**Type**: Gauge
**Description**: Available filesystem space in bytes

**Example Queries**:
```promql
# Disk usage percentage
(1 - node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100
```

## Useful Query Patterns

### Error Analysis

```promql
# Top 10 error types
topk(10, sum(rate(bubble_error_total[5m])) by (error_type))

# Error rate by bubble
sum(rate(bubble_error_total[5m])) by (bubble)
/ sum(rate(bubble_operation_total[5m])) by (bubble)

# Errors in the last hour
increase(bubble_error_total[1h])
```

### Performance Analysis

```promql
# Slowest operations (P99)
topk(10,
  histogram_quantile(0.99,
    sum(rate(bubble_operation_duration_seconds_bucket[5m])) by (le, bubble, operation)
  )
)

# Average operation time
rate(bubble_operation_duration_seconds_sum[5m])
/ rate(bubble_operation_duration_seconds_count[5m])
```

### Capacity Planning

```promql
# Predict growth (linear regression)
predict_linear(bubble_operation_total[30m:5m], 3600)

# Memory trend
deriv(bubble_memory_usage_bytes[1h])

# Throughput trend
deriv(bubble_throughput_per_second[1h])
```

## Alert Queries

### Critical Alerts

```promql
# High error rate
sum(rate(bubble_error_total[5m])) / sum(rate(bubble_operation_total[5m])) > 0.1

# Circuit breaker open
circuit_breaker_state{state="open"} == 1

# Memory exhaustion
sum(bubble_memory_usage_bytes) / sum(bubble_memory_limit_bytes) > 0.95

# Service down
up == 0
```

### Warning Alerts

```promql
# High latency
histogram_quantile(0.95,
  sum(rate(bubble_operation_duration_seconds_bucket[5m])) by (le, bubble)
) > 30

# High memory usage
sum(bubble_memory_usage_bytes) / sum(bubble_memory_limit_bytes) > 0.8

# Rate limit breaches
increase(rate_limit_exceeded_total[5m]) > 100
```

## Label Reference

### Common Labels

- `bubble`: Name of the bubble
- `operation`: Operation type
- `status`: Operation status
- `error_type`: Type of error
- `instance`: Instance address
- `job`: Job name

### Label Values

#### Bubble Names
- stripe-bubble
- slack-bubble
- postgresql-bubble
- http-bubble
- apify-bubble
- etc.

#### Operation Types
- createPayment
- sendMessage
- executeQuery
- sendRequest
- scrapeWeb
- etc.

#### Error Types
- database_error
- network_error
- validation_error
- authentication_error
- rate_limit_error
- timeout_error
- etc.

## Dashboard Panel Queries

### Overview Dashboard

```promql
# Operations per second
sum(rate(bubble_operation_total{status="success"}[5m]))

# Error rate
sum(rate(bubble_error_total[5m])) / sum(rate(bubble_operation_total[5m])) * 100

# P95 latency
histogram_quantile(0.95, sum(rate(bubble_operation_duration_seconds_bucket[5m])) by (le))
```

### Per-Bubble Dashboard

```promql
# Operation duration by type
histogram_quantile(0.95,
  sum(rate(bubble_operation_duration_seconds_bucket{bubble="$bubble"}[5m])) by (le, operation)
)

# Error rate by operation
sum(rate(bubble_error_total{bubble="$bubble"}[5m])) by (operation)
/ sum(rate(bubble_operation_total{bubble="$bubble"}[5m])) by (operation)
```

### Security Dashboard

```promql
# Authentication failures
sum(increase(bubble_authentication_error_total[5m])) by (bubble)

# Security threats blocked
sum(increase(sql_injection_blocked_total[5m])) by (bubble) +
sum(increase(xss_blocked_total[5m])) by (bubble)
```

## Next Steps

- Set up alerts using [Alerting Guide](ALERTING_GUIDE.md)
- Explore dashboards in [Dashboard Tour](DASHBOARD_TOUR.md)
- Troubleshoot issues in [Troubleshooting Guide](TROUBLESHOOTING.md)
