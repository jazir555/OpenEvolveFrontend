# Trace Visualization with Jaeger

Complete guide for visualizing and analyzing traces using Jaeger UI.

## Table of Contents

1. [Running Jaeger](#running-jaeger)
2. [Accessing Jaeger UI](#accessing-jaeger-ui)
3. [Searching Traces](#searching-traces)
4. [Analyzing Trace Details](#analyzing-trace-details)
5. [Common Queries](#common-queries)
6. [Trace Timeline View](#trace-timeline-view)
7. [Service Dependency Graph](#service-dependency-graph)
8. [Compare Traces](#compare-traces)

## Running Jaeger

### Docker (Recommended)

```bash
docker run -d --name jaeger \
  -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 5775:5775/udp \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 5778:5778 \
  -p 16686:16686 \
  -p 14268:14268 \
  -p 14250:14250 \
  -p 9411:9411 \
  jaegertracing/all-in-one:latest
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: jaeger
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
      - COLLECTOR_OTLP_ENABLED=true
    ports:
      - "5775:5775/udp"
      - "6831:6831/udp"
      - "6832:6832/udp"
      - "5778:5778"
      - "16686:16686"  # Jaeger UI
      - "14268:14268"
      - "14250:14250"
      - "9411:9411"
    restart: unless-stopped
```

Start with:

```bash
docker-compose up -d jaeger
```

## Accessing Jaeger UI

Once Jaeger is running:

1. Open your browser
2. Navigate to http://localhost:16686
3. You should see the Jaeger UI

### Jaeger UI Overview

- **Search**: Find traces by service, operation, or tags
- **Trace Timeline**: Visual representation of trace spans
- **Trace Details**: Detailed information about each span
- **Dependencies**: Service dependency graph
- **Compare**: Compare multiple traces side-by-side

## Searching Traces

### Basic Search

1. Select a service from the dropdown (e.g., `bubble-lab`)
2. Select an operation (e.g., `bubble.execution`)
3. Adjust time range
4. Click "Find Traces"

### Search Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| Service   | Service name to search | `bubble-lab` |
| Operation | Operation name | `bubble.execution` |
| Tags      | Key-value pairs | `correlation.id=abc123` |
| Lookback  | Time range | `1h`, `12h`, `2d` |
| Min Duration | Minimum duration | `100ms` |
| Max Duration | Maximum duration | `5s` |

### Advanced Search with Tags

Find traces by custom attributes:

```bash
# Search by correlation ID
correlation.id=abc123

# Search by bubble name
bubble.name=postgresql

# Search by execution ID
execution.id=550e8400-e29b-41d4-a716-446655440000

# Search by error
error.type=ValidationError

# Combine multiple tags
bubble.name=ai-agent&error.type=*
```

### Search Examples

```bash
# Find slow operations ( > 3 seconds)
Min Duration: 3s

# Find failed operations
Tags: error.type=*

# Find specific bubble traces
Service: bubble-lab
Operation: bubble.execution
Tags: bubble.name=stripe-bubble

# Find by time range
Lookback: Last Hour
Min Duration: 1s

# Find by correlation ID
Tags: correlation.id=your-correlation-id

# Find errors for specific bubble
Service: bubble-lab
Operation: bubble.execution
Tags: bubble.name=slack&error.type=*
```

## Analyzing Trace Details

### Trace Timeline View

When you click on a trace, you'll see:

1. **Trace Overview**
   - Total duration
   - Number of spans
   - Service name
   - Start/end timestamps

2. **Span Hierarchy**
   - Waterfall view of all spans
   - Parent-child relationships
   - Duration bars with color coding

3. **Span Details**
   - Click any span to see details
   - Attributes and tags
   - Logs and events
   - References (links to other spans)

### Understanding Span Colors

| Color | Meaning |
|-------|---------|
| 🔵 Blue | Normal operation |
| 🔴 Red | Error occurred |
| 🟡 Yellow | Warning or slow operation |
| 🟢 Green | Fast operation |

### Span Information Panel

For each span, you can view:

- **Span ID**: Unique identifier
- **Operation Name**: Name of the operation
- **Duration**: Execution time
- **Start Time**: When the span started
- **Tags**: Custom attributes
- **Logs**: Timestamped events
- **References**: Links to parent/child spans

## Common Queries

### Find All Operations by Bubble

```
Service: bubble-lab
Tags: bubble.name=postgresql
Lookback: Last Hour
```

### Find Slow Operations

```
Service: bubble-lab
Min Duration: 3000000  # 3 seconds in microseconds
Lookback: Last Hour
```

### Find Failed Operations

```
Service: bubble-lab
Tags: error.type=*
Lookback: Last Hour
```

### Find by Correlation ID

```
Service: bubble-lab
Tags: correlation.id=abc123
Lookback: Last 6 Hours
```

### Find AI Agent Operations

```
Service: bubble-lab
Operation: bubble.execution
Tags: bubble.name=ai-agent
Lookback: Last Hour
```

### Find Database Queries

```
Service: bubble-lab
Operation: bubble.database.query
Tags: db.system=postgresql
Lookback: Last Hour
```

### Find HTTP Requests

```
Service: bubble-lab
Operation: bubble.http.request
Tags: http.method=POST
Lookback: Last Hour
```

## Trace Timeline View

### Understanding the Waterfall

The timeline view shows spans as horizontal bars:

```
┌─────────────────────────────────────────────────────────┐
│ HTTP Request (root span)                                │
│ └─ Input Validation                                     │
│ └─ Authentication                                        │
│ └─ Business Logic                                       │
│    ├─ Database Query                                    │
│    │  ├─ Connection Pool                               │
│    │  └─ Query Execution                               │
│    └─ External API Call                                 │
│       ├─ DNS Lookup                                    │
│       ├─ TCP Connect                                   │
│       ├─ TLS Handshake                                 │
│       └─ Request/Response                              │
│ └─ Output Formatting                                     │
└─────────────────────────────────────────────────────────┘
```

### Span Hierarchy Example

```
HTTP Request (root span)
└── Bubble.performAction
    ├── Input Validation
    ├── Authentication
    ├── Rate Limit Check
    ├── Business Logic
    │   ├── External API Call
    │   │   └── DNS Lookup
    │   │   ├── TCP Connect
    │   │   ├── TLS Handshake
    │   │   └── Request/Response
    │   └── Database Query
    │       ├── Connection Pool
    │       └── Query Execution
    ├── Output Sanitization
    └── Response Logging
```

### Critical Path Analysis

Jaeger automatically highlights the critical path (the longest path through the trace):

- Darker/larger spans are on the critical path
- Lighter/smaller spans can run in parallel
- Focus optimization efforts on the critical path

## Service Dependency Graph

### View Dependencies

1. Click the "Dependencies" tab in Jaeger UI
2. See a graph of service relationships
3. Arrow thickness indicates call frequency

### Dependency Graph Example

```
        ┌──────────────┐
        │ BubbleLab API │
        └──────┬───────┘
               │
       ┌───────┴────────┬────────────┐
       ▼                ▼            ▼
┌────────────┐  ┌──────────┐  ┌──────────┐
│ PostgreSQL │  │  Slack   │  │  AI Agent │
└────────────┘  └──────────┘  └──────────┘
```

### Understanding Relationships

- **Caller → Callee**: Direction of the arrow
- **Arrow Thickness**: Number of calls
- **Node Size**: Amount of traffic
- **Colors**: Error rates or latency

## Compare Traces

### Side-by-Side Comparison

1. Select multiple traces using checkboxes
2. Click "Compare Traces"
3. View differences in:
   - Duration
   - Number of spans
   - Error rates
   - Tag values

### Use Cases

- Compare successful vs failed requests
- Compare performance before/after optimization
- Compare different input sizes
- Compare different user flows

## System Metrics

### Jaeger Metrics

Jaeger exposes Prometheus metrics at `http://localhost:16686/metrics`:

- `jaeger_tracer_finished_spans`: Total spans processed
- `jaeger_tracer_started_spans`: Total spans started
- `jaeger_tracer_reported_spans`: Total spans reported

### System Metrics

Monitor Jaeger itself:

```bash
# Check Jaeger health
curl http://localhost:16686/api/status

# Get trace count
curl http://localhost:16686/api/traces?service=bubble-lab

# Get services
curl http://localhost:16686/api/services
```

## Best Practices

### Search Strategy

1. **Start broad**: Search by service and time range
2. **Narrow down**: Add tags and duration filters
3. **Focus on errors**: Search for `error.type=*` first
4. **Analyze outliers**: Sort by duration descending

### Trace Analysis Workflow

1. Find the slowest traces
2. Click into the trace details
3. Identify the critical path
4. Look for bottlenecks (red/yellow spans)
5. Check error logs
6. Examine span attributes
7. Identify optimization opportunities

### Performance Investigation

When investigating performance issues:

1. **Find slow traces**: Set Min Duration filter
2. **Check critical path**: Identify longest operations
3. **Look for patterns**: Same operation always slow?
4. **Check external calls**: HTTP requests, DB queries
5. **Verify caching**: Cache misses, cache hits
6. **Analyze errors**: Errors causing retries?

### Debugging Errors

When debugging errors:

1. **Find failed traces**: Search `error.type=*`
2. **Check error logs**: Look at span events
3. **Examine stack traces**: In error attributes
4. **Trace the flow**: Follow parent-child relationships
5. **Identify root cause**: Where did error originate?

## Query Examples for Common Issues

### High Latency Investigation

```
# Find slow operations
Min Duration: 5000000  # 5 seconds
Lookback: Last Hour

# Analyze specific bubble
Service: bubble-lab
Tags: bubble.name=ai-agent
Min Duration: 1000000  # 1 second
```

### Error Investigation

```
# Find all errors
Tags: error.type=*
Lookback: Last Hour

# Find specific error type
Tags: error.type=ValidationError
Lookback: Last 6 Hours

# Find errors in specific bubble
Tags: bubble.name=slack&error.type=*
```

### Throughput Investigation

```
# Check trace volume
Service: bubble-lab
Lookback: Last 15 minutes
Count traces displayed

# Compare with baseline
Service: bubble-lab
Lookback: Same time yesterday
Count traces displayed
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `t` | Focus search box |
| `Enter` | Search traces |
| `Esc` | Clear search |
| `←` `→` | Navigate trace list |
| `↑` `↓` | Navigate spans |
| `Enter` | Open selected span |

## API Access

### Query Traces Programmatically

```bash
# Get all traces for a service
curl "http://localhost:16686/api/traces?service=bubble-lab"

# Get traces by operation
curl "http://localhost:16686/api/traces?service=bubble-lab&operation=bubble.execution"

# Get traces by tag
curl "http://localhost:16686/api/traces?service=bubble-lab&tag=correlation.id:abc123"

# Get trace by ID
curl "http://localhost:16686/api/traces/{trace-id}"
```

## Next Steps

- [Performance Analysis Guide](./PERFORMANCE_ANALYSIS.md)
- [OpenTelemetry Setup](./OPENTELEMETRY_SETUP.md)
- [Troubleshooting Traces](./TROUBLESHOOTING_TRACES.md)
