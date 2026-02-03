# Metrics Catalog

Complete reference of all Prometheus metrics exposed by the OpenEvolve Glue Layer monitoring system.

## Metric Naming Convention

All metrics follow the pattern: `{prefix}_{metric_name}`

- **Prefix**: Configurable via `METRICS_PREFIX` environment variable (default: `openvolve_`)
- **Metric Name**: Descriptive name using snake_case

Example: `openvolve_http_request_duration_seconds`

## Metric Types

### Counter
A cumulative metric that represents a single monotonically increasing counter whose value can only increase or be reset to zero.

### Gauge
A metric that represents a single numerical value that can arbitrarily go up and down.

### Histogram
A histogram samples observations (usually things like request durations or response sizes) and counts them in configurable buckets.

---

## HTTP Metrics

### `openvolve_http_request_duration_seconds`
**Type:** Histogram
**Labels:** `service`, `operation`, `status`

Request latency in seconds.

**Buckets:** [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]

**Example:**
```promql
# P95 latency for crm-adapter
histogram_quantile(0.95, rate(openvolve_http_request_duration_seconds_bucket{service="crm-adapter"}[5m]))
```

### `openvolve_http_requests_total`
**Type:** Counter
**Labels:** `service`, `operation`, `status`

Total number of HTTP requests.

**Example:**
```promql
# Request rate by service
rate(openvolve_http_requests_total[5m])

# Error rate (4xx and 5xx)
rate(openvolve_http_requests_total{status=~"[45].."}[5m])
```

### `openvolve_http_requests_in_progress`
**Type:** Gauge
**Labels:** `service`

Current number of HTTP requests in progress.

**Example:**
```promql
# Active requests by service
openvolve_http_requests_in_progress
```

---

## Error Metrics

### `openvolve_errors_total`
**Type:** Counter
**Labels:** `service`, `error_type`

Total number of errors.

**Example:**
```promql
# Error rate by type
rate(openvolve_errors_total[5m])

# Errors by service
sum by (service) (rate(openvolve_errors_total[5m]))
```

### `openvolve_errors_by_type_total`
**Type:** Counter
**Labels:** `service`, `operation`, `error_type`

Errors categorized by type and operation.

**Example:**
```promql
# Top error types
topk(10, sum by (error_type) (rate(openvolve_errors_by_type_total[5m])))
```

---

## Circuit Breaker Metrics

### `openvolve_circuit_breaker_state`
**Type:** Gauge
**Labels:** `service`, `circuit`

Current state of circuit breaker.

**Values:**
- `0` = CLOSED (normal operation)
- `1` = HALF_OPEN (testing recovery)
- `2` = OPEN (failing, blocking requests)

**Example:**
```promql
# Open circuits
openvolve_circuit_breaker_state{service="crm-adapter"} == 2
```

### `openvolve_circuit_breaker_failures_total`
**Type:** Counter
**Labels:** `service`, `circuit`

Total circuit breaker failures.

**Example:**
```promql
# Failure rate by circuit
rate(openvolve_circuit_breaker_failures_total[5m])
```

### `openvolve_circuit_breaker_successes_total`
**Type:** Counter
**Labels:** `service`, `circuit`

Total circuit breaker successes.

**Example:**
```promql
# Success rate by circuit
rate(openvolve_circuit_breaker_successes_total[5m])
```

### `openvolve_circuit_breaker_rejects_total`
**Type:** Counter
**Labels:** `service`, `circuit`

Total requests rejected by circuit breaker.

**Example:**
```promql
# Reject rate (requests blocked by open circuit)
rate(openvolve_circuit_breaker_rejects_total[5m])
```

---

## Adapter Health Metrics

### `openvolve_adapter_health`
**Type:** Gauge
**Labels:** `adapter`

Adapter health status.

**Values:**
- `0` = UNHEALTHY
- `1` = DEGRADED
- `2` = HEALTHY

**Example:**
```promql
# Unhealthy adapters
openvolve_adapter_health < 2

# All adapters status
openvolve_adapter_health
```

### `openvolve_adapter_last_success_timestamp`
**Type:** Gauge
**Labels:** `adapter`

Unix timestamp of last successful operation.

**Example:**
```promql
# Time since last success
time() - openvolve_adapter_last_success_timestamp
```

### `openvolve_adapter_last_failure_timestamp`
**Type:** Gauge
**Labels:** `adapter`

Unix timestamp of last failed operation.

**Example:**
```promql
# Time since last failure
time() - openvolve_adapter_last_failure_timestamp

# Adapters with recent failures (last 5 minutes)
time() - openvolve_adapter_last_failure_timestamp < 300
```

---

## Knowledge Extraction Metrics

### `openvolve_knowledge_extraction_total`
**Type:** Counter
**Labels:** `source`, `method`, `entity_type`, `success`

Total knowledge extraction operations.

**Example:**
```promql
# Extraction rate by source
rate(openvolve_knowledge_extraction_total[5m])

# Success rate
sum by (source) (rate(openvolve_knowledge_extraction_total{success="true"}[5m])) /
sum by (source) (rate(openvolve_knowledge_extraction_total[5m]))
```

### `openvolve_knowledge_extraction_duration_seconds`
**Type:** Histogram
**Labels:** `source`, `method`

Duration of knowledge extraction operations in seconds.

**Buckets:** [0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300]

**Example:**
```promql
# Average extraction duration
rate(openvolve_knowledge_extraction_duration_seconds_sum[5m]) /
rate(openvolve_knowledge_extraction_duration_seconds_count[5m])

# P95 extraction duration
histogram_quantile(0.95, rate(openvolve_knowledge_extraction_duration_seconds_bucket[5m]))
```

### `openvolve_knowledge_extraction_entities`
**Type:** Gauge
**Labels:** `source`, `entity_type`

Number of entities extracted.

**Example:**
```promql
# Total entities by source
sum by (source) (openvolve_knowledge_extraction_entities)

# Entities by type
sum by (entity_type) (openvolve_knowledge_extraction_entities)
```

### `openvolve_knowledge_extraction_relations`
**Type:** Gauge
**Labels:** `source`

Number of relations extracted.

**Example:**
```promql
# Total relations by source
openvolve_knowledge_extraction_relations

# Relations to entities ratio
openvolve_knowledge_extraction_relations /
openvolve_knowledge_extraction_entities
```

---

## Event Bus Metrics

### `openvolve_events_processed_total`
**Type:** Counter
**Labels:** `event_type`, `status`

Total events processed.

**Example:**
```promql
# Event processing rate
rate(openvolve_events_processed_total[5m])

# Success rate
sum by (event_type) (rate(openvolve_events_processed_total{status="success"}[5m])) /
sum by (event_type) (rate(openvolve_events_processed_total[5m]))
```

### `openvolve_event_processing_duration_seconds`
**Type:** Histogram
**Labels:** `event_type`

Duration of event processing in seconds.

**Buckets:** [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]

**Example:**
```promql
# Average processing duration by event type
rate(openvolve_event_processing_duration_seconds_sum[5m]) /
rate(openvolve_event_processing_duration_seconds_count[5m])

# P95 processing duration
histogram_quantile(0.95, rate(openvolve_event_processing_duration_seconds_bucket[5m]))
```

### `openvolve_events_in_queue`
**Type:** Gauge
**Labels:** `queue_name`

Number of events currently in queue.

**Example:**
```promql
# Queue depth
openvolve_events_in_queue

# Growing queues (increasing trend)
rate(openvolve_events_in_queue[5m]) > 0
```

---

## Retry Metrics

### `openvolve_retry_attempts_total`
**Type:** Counter
**Labels:** `service`, `operation`

Total retry attempts.

**Example:**
```promql
# Retry rate
rate(openvolve_retry_attempts_total[5m])

# Operations with most retries
topk(10, sum by (operation) (rate(openvolve_retry_attempts_total[5m])))
```

### `openvolve_retry_success_total`
**Type:** Counter
**Labels:** `service`, `operation`

Total successful retries.

**Example:**
```promql
# Successful retry rate
rate(openvolve_retry_success_total[5m])
```

### `openvolve_retry_failure_total`
**Type:** Counter
**Labels:** `service`, `operation`

Total failed retries after all attempts.

**Example:**
```promql
# Failed retry rate
rate(openvolve_retry_failure_total[5m])

# Retry success rate
sum by (service) (rate(openvolve_retry_success_total[5m])) /
sum by (service) (rate(openvolve_retry_attempts_total[5m]))
```

---

## Default Prometheus Metrics

The following default Node.js metrics are also collected with the configured prefix:

### Process Metrics
- `openvolve_nodejs_heap_size_total_bytes`
- `openvolve_nodejs_heap_size_used_bytes`
- `openvolve_nodejs_heap_size_external_bytes`
- `openvolve_nodejs_heap_spaces_size_bytes`
- `openvolve_nodejs_heap_spaces_size_used_bytes`
- `openvolve_nodejs_heap_spaces_size_available_bytes`

### Event Loop Metrics
- `openvolve_nodejs_eventloop_lag_seconds`
- `openvolve_nodejs_eventloop_lag_p50_seconds`
- `openvolve_nodejs_eventloop_lag_p95_seconds`
- `openvolve_nodejs_eventloop_lag_p99_seconds`

### GC Metrics
- `openvolve_nodejs_gc_duration_seconds`
- `openvolve_nodejs_gc_reclaimed_bytes`

### Process CPU and Memory
- `openvolve_process_cpu_seconds_total`
- `openvolve_process_cpu_percent_usage`
- `openvolve_process_resident_memory_bytes`
- `openvolve_process_heap_size_bytes`

---

## Useful Queries

### Service Overview
```promql
# Request rate by service
sum by (service) (rate(openvolve_http_requests_total[5m]))

# Error rate by service
sum by (service) (rate(openvolve_http_requests_total{status=~"5.."}[5m]))

# P95 latency by service
histogram_quantile(0.95, sum by (service, le) (rate(openvolve_http_request_duration_seconds_bucket[5m])))
```

### Circuit Breaker Status
```promql
# All circuit breaker states
openvolve_circuit_breaker_state

# Services with open circuits
openvolve_circuit_breaker_state == 2

# Circuit breaker reject rate
rate(openvolve_circuit_breaker_rejects_total[5m])
```

### Health Status
```promql
# Unhealthy adapters
openvolve_adapter_health == 0

# Degraded adapters
openvolve_adapter_health == 1

# All adapter health
openvolve_adapter_health
```

### Knowledge Extraction
```promql
# Extraction rate by source
sum by (source) (rate(openvolve_knowledge_extraction_total[5m]))

# Total entities extracted
sum by (source) (openvolve_knowledge_extraction_entities)

# Average extraction duration
rate(openvolve_knowledge_extraction_duration_seconds_sum[5m]) /
rate(openvolve_knowledge_extraction_duration_seconds_count[5m])
```

### Alerts

#### High Error Rate
```promql
rate(openvolve_errors_total[5m]) > 0.1
```

#### High Latency (P95 > 1s)
```promql
histogram_quantile(0.95, rate(openvolve_http_request_duration_seconds_bucket[5m])) > 1
```

#### Unhealthy Service
```promql
openvolve_adapter_health < 2
```

#### Circuit Breaker Open
```promql
openvolve_circuit_breaker_state == 2
```

#### Queue Growing
```promql
rate(openvolve_events_in_queue[5m]) > 10
```

---

## Grafana Dashboard JSON

```json
{
  "dashboard": {
    "title": "OpenEvolve Glue Layer",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "sum by (service) (rate(openvolve_http_requests_total[5m]))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "sum by (service) (rate(openvolve_http_requests_total{status=~\"5..\"}[5m]))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "P95 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum by (service, le) (rate(openvolve_http_request_duration_seconds_bucket[5m])))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Circuit Breaker States",
        "targets": [
          {
            "expr": "openvolve_circuit_breaker_state"
          }
        ],
        "type": "stat"
      },
      {
        "title": "Adapter Health",
        "targets": [
          {
            "expr": "openvolve_adapter_health"
          }
        ],
        "type": "stat"
      }
    ]
  }
}
```

---

## Label Best Practices

### Low Cardinality Labels (Recommended)
- `service`: Service name (e.g., "crm-adapter", "user-service")
- `operation`: Operation name (e.g., "fetch-users", "create-user")
- `status`: HTTP status category (e.g., "2xx", "4xx", "5xx")
- `error_type`: Error type (e.g., "timeout", "connection_error", "validation_error")
- `adapter`: Adapter name (e.g., "github", "slack", "jira")
- `event_type`: Event type (e.g., "user.created", "user.updated")
- `source`: Data source (e.g., "github", "gitlab")

### High Cardinality Labels (Avoid)
- `user_id`: Individual user IDs
- `request_id`: Unique request IDs
- `correlation_id`: Correlation IDs (use in logs, not metrics)
- `url`: Full URLs (use operation name instead)

---

## Metric Retention

Recommended retention policies:

| Metric Type | Retention | Resolution |
|-------------|-----------|------------|
| Counter/Rate | 30 days | 1 minute |
| Gauge | 7 days | 1 minute |
| Histogram | 30 days | 1 minute |
| Default Metrics | 7 days | 1 minute |

---

## Performance Considerations

1. **Label Cardinality**: Keep label values to a minimum. High cardinality can impact Prometheus performance.

2. **Scrape Interval**: Default 15 seconds is suitable for most use cases. Increase to 30s or 60s for large-scale deployments.

3. **Metric Count**: Monitor total metric count. Consider aggregation for high-cardinality metrics.

4. **Recording Rules**: Use recording rules for complex queries to improve dashboard performance.

Example recording rules:
```yaml
groups:
  - name: openevolve
    interval: 30s
    rules:
      - record: job:openvolve_http_requests_total:rate5m
        expr: sum by (job) (rate(openvolve_http_requests_total[5m]))

      - record: job:openvolve_http_request_latency:p95
        expr: histogram_quantile(0.95, sum by (job, le) (rate(openvolve_http_request_duration_seconds_bucket[5m])))
```
