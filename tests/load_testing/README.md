# Knowledge Graph Load Testing Framework

Comprehensive load testing framework for the Knowledge Graph system with support for multiple test scenarios, performance monitoring, and detailed reporting.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Test Scenarios](#test-scenarios)
- [Configuration](#configuration)
- [Locust Integration](#locust-integration)
- [Result Analysis](#result-analysis)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The load testing framework provides four main test scenarios:

1. **Read-Heavy Workload**: Tests search/query performance (90% reads, 10% writes)
2. **Write-Heavy Workload**: Tests knowledge addition performance (80% writes, 20% reads)
3. **Spike Test**: Validates system resilience to sudden traffic increases
4. **Endurance Test**: Checks for memory leaks and performance degradation over time

## Installation

### Requirements

```bash
# Python dependencies
pip install locust psutil pyyaml

# Optional: For advanced reporting
pip install matplotlib seaborn
```

### Setup

```bash
# Navigate to load testing directory
cd tests/load_testing

# Verify installation
python -c "from kg_load_tests import KnowledgeGraphLoadTest; print('OK')"
```

## Quick Start

### Run All Tests

```bash
python run_load_tests.py
```

### Run Specific Test

```bash
# Run read-heavy test
python run_load_tests.py --test read_heavy

# Run spike test with custom parameters
python run_load_tests.py --test spike --users 200 --duration 60
```

### Generate Report

```bash
python run_load_tests.py --analyze
```

## Test Scenarios

### 1. Read-Heavy Workload

Simulates typical production usage with heavy search/query operations.

**Characteristics:**
- 90% search operations
- 10% write operations
- Tests retrieval performance
- Validates caching effectiveness

**Usage:**

```python
from kg_load_tests import KnowledgeGraphLoadTest

load_test = KnowledgeGraphLoadTest(engine)
result = await load_test.run_read_heavy_test(
    num_users=100,
    spawn_rate=10,
    test_duration=60
)
```

### 2. Write-Heavy Workload

Tests system performance under heavy write operations.

**Characteristics:**
- 80% write operations
- 20% read operations
- Tests write scalability
- Validates batch processing

**Usage:**

```python
result = await load_test.run_write_heavy_test(
    num_users=50,
    spawn_rate=5,
    test_duration=60
)
```

### 3. Spike Test

Validates system resilience to sudden traffic increases.

**Process:**
1. Establish baseline with low traffic
2. Rapidly increase to spike level
3. Maintain spike load
4. Measure response time degradation

**Usage:**

```python
result = await load_test.run_spike_test(
    base_users=10,
    spike_users=100,
    spike_duration=30
)
```

### 4. Endurance Test

Tests for memory leaks and performance degradation over time.

**Monitors:**
- Memory growth
- Performance trends
- Connection stability
- Resource exhaustion

**Usage:**

```python
result = await load_test.run_endurance_test(
    num_users=20,
    test_duration=300  # 5 minutes
)
```

## Configuration

### Config File Structure

Edit `load_test_config.yaml` to customize tests:

```yaml
read_heavy:
  users: [10, 50, 100, 500]
  spawn_rate: 10
  duration: 60
  target_throughput: 100
  max_error_rate: 0.01

write_heavy:
  users: [5, 25, 50, 100]
  spawn_rate: 5
  duration: 60
  target_throughput: 50
  max_error_rate: 0.05
```

### Command-Line Overrides

```bash
# Override user count
python run_load_tests.py --users 200

# Override duration
python run_load_tests.py --duration 120

# Combine overrides
python run_load_tests.py --test spike --users 150 --duration 45
```

## Locust Integration

### Run Locust with Web UI

```bash
# Start Locust web interface
locust -f locustfile.py --host=http://localhost:8080

# Open browser to http://localhost:8089
```

### Run Headless Mode

```bash
# Run without web UI
locust -f locustfile.py \
  --host=http://localhost:8080 \
  --headless \
  -u 100 \
  -r 10 \
  --run-time 5m \
  --html locust_report.html
```

### Custom User Classes

The `locustfile.py` includes multiple user classes:

- **KnowledgeGraphUser**: Mixed workload (realistic usage)
- **KnowledgeGraphWriteUser**: Write-focused operations

Run specific user class:

```bash
locust -f locustfile.py KnowledgeGraphUser --host=http://localhost:8080
```

## Result Analysis

### Generate Comprehensive Report

```python
from analyze_results import LoadTestAnalyzer

analyzer = LoadTestAnalyzer("load_test_results.json")
analyzer.generate_report("report.txt")
```

### Programmatic Analysis

```python
# Analyze throughput
throughput = analyzer.analyze_throughput()
print(f"Average throughput: {throughput['average_throughput']:.2f} ops/sec")

# Identify bottlenecks
bottlenecks = analyzer.identify_bottlenecks()
for bottleneck in bottlenecks:
    print(f"[{bottleneck['severity']}] {bottleneck['issue']}")

# Estimate capacity
capacity = analyzer.estimate_capacity(target_response_time=1.0)
print(f"Max concurrent users: {capacity['estimated_max_concurrent_users']}")
```

### Result Metrics

Each test produces:

- **Throughput**: Operations per second
- **Error Rate**: Percentage of failed operations
- **Response Times**: Baseline and under load
- **Resource Usage**: Memory, CPU, connections
- **Performance Trends**: Degradation over time

## Best Practices

### 1. Test Environment

- Use dedicated testing environment
- Mirror production configuration
- Ensure realistic data volumes
- Monitor system resources

### 2. Test Design

- Start with small loads, gradually increase
- Test realistic user scenarios
- Include warm-up period
- Allow cool-down between tests

### 3. Data Management

- Clean test data between runs
- Use idempotent operations
- Test with production-like data distribution
- Archive test results for trend analysis

### 4. Monitoring

- Monitor system resources during tests
- Capture application logs
- Track database metrics
- Record network statistics

### 5. Analysis

- Compare results across multiple runs
- Identify performance regressions
- Track improvement over time
- Document findings and recommendations

## Troubleshooting

### High Error Rates

**Symptoms**: Error rate exceeds thresholds

**Possible Causes**:
- Database connection pool exhausted
- API rate limiting
- Network issues
- Resource exhaustion

**Solutions**:
```python
# Increase connection pool size
# Adjust rate limits
# Add retry logic
# Scale resources
```

### Slow Response Times

**Symptoms**: Response times exceed targets

**Possible Causes**:
- Inefficient queries
- Missing indexes
- Network latency
- CPU bottleneck

**Solutions**:
```python
# Add database indexes
# Optimize query performance
# Implement caching
# Scale horizontally
```

### Memory Issues

**Symptoms**: Continuous memory growth

**Possible Causes**:
- Memory leaks
- Unbounded caches
- Large result sets
- Connection not closed

**Solutions**:
```python
# Use weak references
# Implement cache eviction
# Paginate large results
# Ensure proper cleanup
```

### Test Failures

**Symptoms**: Tests fail to start or complete

**Possible Causes**:
- Port conflicts
- Missing dependencies
- Configuration errors
- Engine not available

**Solutions**:
```bash
# Check port availability
# Install dependencies
# Verify configuration
# Ensure engine is running
```

## Advanced Usage

### Custom Test Scenarios

```python
class CustomLoadTest(KnowledgeGraphLoadTest):
    async def run_custom_test(self, num_users=50):
        # Implement custom test logic
        pass
```

### Performance Monitoring

```python
import psutil

def monitor_resources():
    process = psutil.Process()
    return {
        "cpu_percent": process.cpu_percent(),
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "connections": len(process.connections())
    }
```

### Alert Integration

```python
def check_alerts(result: LoadTestResult):
    if result.metrics["error_rate"] > 0.05:
        send_alert(f"High error rate: {result.metrics['error_rate']:.2%}")

    if result.metrics["throughput_ops_per_sec"] < 50:
        send_alert(f"Low throughput: {result.metrics['throughput_ops_per_sec']:.2f}")
```

## Contributing

When adding new test scenarios:

1. Inherit from `KnowledgeGraphLoadTest`
2. Follow naming convention: `run_<scenario>_test`
3. Return `LoadTestResult` object
4. Add configuration to `load_test_config.yaml`
5. Update documentation

## License

MIT License - See LICENSE file for details
