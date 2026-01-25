# Load Testing Framework - Quick Start Guide

Get started with load testing the Knowledge Graph system in 5 minutes.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from kg_load_tests import KnowledgeGraphLoadTest; print('✓ Ready')"
```

## Basic Usage

### 1. Run All Tests

```bash
python run_load_tests.py
```

This will run:
- Read-heavy workload test
- Write-heavy workload test
- Spike test
- Endurance test

### 2. Run Specific Test

```bash
# Read-heavy test
python run_load_tests.py --test read_heavy

# Spike test with custom parameters
python run_load_tests.py --test spike --users 200 --duration 60

# Endurance test
python run_load_tests.py --test endurance --duration 600
```

### 3. Generate Report

```bash
python run_load_tests.py --analyze
```

## Using Locust (HTTP Load Testing)

### Start Locust Web Interface

```bash
locust -f locustfile.py --host=http://localhost:8080
```

Then open http://localhost:8089 in your browser.

### Run Headless (No UI)

```bash
locust -f locustfile.py \
  --host=http://localhost:8080 \
  --headless \
  -u 100 \
  -r 10 \
  --run-time 5m \
  --html locust_report.html
```

## Configuration

Edit `load_test_config.yaml` to customize:

```yaml
read_heavy:
  users: [10, 50, 100, 500]
  spawn_rate: 10
  duration: 60
  target_throughput: 100
  max_error_rate: 0.01

spike_test:
  base_users: 10
  spike_users: [50, 100, 200]
  spike_duration: 30
```

## Programmatic Usage

```python
import asyncio
from kg_load_tests import KnowledgeGraphLoadTest

async def main():
    # Initialize your knowledge engine
    from knowledge_engine.engine import KnowledgeEngine
    engine = KnowledgeEngine()

    # Create load tester
    load_test = KnowledgeGraphLoadTest(engine)

    # Run test
    result = await load_test.run_read_heavy_test(
        num_users=100,
        spawn_rate=10,
        test_duration=60
    )

    print(f"Status: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Throughput: {result.metrics['throughput_ops_per_sec']:.2f} ops/sec")

    # Save results
    load_test.save_results("results.json")

asyncio.run(main())
```

## Result Analysis

```python
from analyze_results import LoadTestAnalyzer

# Load and analyze results
analyzer = LoadTestAnalyzer("load_test_results.json")

# Get throughput analysis
throughput = analyzer.analyze_throughput()
print(f"Average throughput: {throughput['average_throughput']:.2f} ops/sec")

# Identify bottlenecks
bottlenecks = analyzer.identify_bottlenecks()
for bottleneck in bottlenecks:
    print(f"[{bottleneck['severity']}] {bottleneck['issue']}")

# Estimate capacity
capacity = analyzer.estimate_capacity(target_response_time=1.0)
print(f"Max concurrent users: {capacity['estimated_max_concurrent_users']}")

# Generate comprehensive report
analyzer.generate_report("report.txt")
```

## Common Scenarios

### Test Read Performance

```bash
python run_load_tests.py --test read_heavy --users 500 --duration 120
```

### Test Write Scalability

```bash
python run_load_tests.py --test write_heavy --users 100 --duration 60
```

### Test Traffic Spikes

```bash
python run_load_tests.py --test spike --users 200
```

### Test Long-running Stability

```bash
python run_load_tests.py --test endurance --duration 1800  # 30 minutes
```

## Understanding Results

### Test Metrics

- **Throughput**: Operations completed per second
- **Error Rate**: Percentage of failed operations
- **Response Time**: Time to complete operations
- **Concurrent Users**: Number of simultaneous users
- **Memory Growth**: Increase in memory usage over time

### Interpreting Results

✓ **Passed**: All thresholds met
✗ **Failed**: One or more thresholds exceeded
⚠️ **Warning**: Approaching threshold limits

### Typical Values

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Throughput | >100 ops/s | 50-100 ops/s | <50 ops/s |
| Error Rate | <1% | 1-5% | >5% |
| Response Time | <200ms | 200-500ms | >500ms |

## Troubleshooting

### Import Errors

```bash
# Ensure Python path includes parent directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Port Already in Use

```bash
# Use different port for Locust
locust -f locustfile.py --host=http://localhost:8080 -P 8090
```

### High Error Rates

- Check if knowledge engine is running
- Verify database connectivity
- Review error logs in `load_test.log`

### Slow Performance

- Reduce user count: `--users 50`
- Increase spawn rate delay
- Check system resources

## Next Steps

1. **Customize Tests**: Modify `load_test_config.yaml`
2. **Add Scenarios**: Create custom test methods in `kg_load_tests.py`
3. **Monitor Resources**: Use `monitor_resources.py` during tests
4. **Generate Reports**: Use `generate_report.py` for HTML reports
5. **Integrate CI/CD**: Add load tests to your pipeline

## Additional Resources

- Full documentation: `README.md`
- Example usage: `example_usage.py`
- Test examples: `test_load_tests.py`
- Run examples: `python example_usage.py`

## Support

For issues or questions:
1. Check logs: `load_test.log`
2. Review error messages
3. Consult full README.md
4. Run with verbose logging: `--log-level DEBUG`

## Tips

- Start with small loads and gradually increase
- Run tests multiple times for consistent results
- Monitor system resources during tests
- Save and compare results over time
- Test in production-like environment
- Allow cool-down between tests
- Use realistic data distributions

---

Happy Load Testing! 🚀
