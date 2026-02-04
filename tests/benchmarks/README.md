# Gauntlet Performance Benchmarking Suite

Comprehensive performance benchmarking suite for the OpenEvolve Gauntlet System with baseline metrics comparison, statistical significance testing, and CI/CD integration.

## Overview

This benchmarking suite provides extensive performance testing for all major gauntlet system components:

- **ML Optimizer**: Tests optimization speed, memory usage, convergence rate, and improvement percentage
- **Predictive Executor**: Measures prediction latency, accuracy, and cost savings
- **Adaptive Learner**: Evaluates training speed, memory usage, loss convergence, and prediction accuracy
- **Intelligent Orchestrator**: Benchmarks planning time, execution time, and resource utilization

## Features

- ✅ Baseline metrics comparison with PASS/FAIL criteria
- ✅ Statistical significance testing
- ✅ JSON output for CI/CD integration
- ✅ Performance targets and tolerances
- ✅ Memory usage tracking
- ✅ Convergence rate analysis
- ✅ Comprehensive logging
- ✅ Configurable test runs

## Installation

### Prerequisites

```bash
# Required Python packages
pip install numpy scipy

# Optional (for better JSON formatting)
# On Linux/Mac:
sudo apt-get install jq
# On Mac:
brew install jq
# On Windows:
# Download from https://stedolan.github.io/jq/
```

### Setup

```bash
# Make the benchmark runner executable (Linux/Mac)
chmod +x tests/benchmarks/run_benchmarks.sh
```

## Usage

### Quick Start

```bash
# Run all benchmarks with default settings
cd tests/benchmarks
./run_benchmarks.sh

# Or run directly with Python
python gauntlet_benchmarks.py
```

### Advanced Usage

```bash
# Custom output file and number of runs
./run_benchmarks.sh -o my_results.json -n 20

# Verbose mode for detailed logging
./run_benchmarks.sh --verbose

# Combine options
./run_benchmarks.sh -o results.json -n 50 -v
```

### Python API

```python
from gauntlet_benchmarks import GauntletBenchmarkSuite, BaselineMetrics

# Create custom baselines
custom_baselines = BaselineMetrics(
    ml_optimizer_iterations_per_second=60.0,
    prediction_latency_ms=80.0
)

# Run benchmarks
suite = GauntletBenchmarkSuite(
    baselines=custom_baselines,
    num_runs=20,
    confidence_level=0.95
)

results = suite.run_all_benchmarks()

# Save results
results.to_json("benchmark_results.json")

# Access summary
print(results.summary)
# {
#     'overall_status': 'PASS',
#     'pass_rate': '92.8%',
#     'performance_grade': 'A'
# }
```

## Benchmark Metrics

### ML Optimizer

| Metric | Description | Unit | Baseline | Target |
|--------|-------------|------|----------|--------|
| Optimization Speed | Iterations per second | iter/s | 50.0 | ≥ 40.0 |
| Memory Usage | Peak memory during optimization | MB | 50.0 | ≤ 65.0 |
| Convergence Rate | Improvement over iterations | rate | 0.95 | ≥ 0.86 |
| Improvement % | Score improvement over baseline | % | 15.0 | ≥ 10.0 |

### Predictive Executor

| Metric | Description | Unit | Baseline | Target |
|--------|-------------|------|----------|--------|
| Prediction Latency | Time to generate prediction | ms | 100.0 | ≤ 130.0 |
| Prediction Accuracy | Accuracy of predictions | ratio | 0.75 | ≥ 0.70 |
| Cost Savings | Savings from early termination | % | 20.0 | ≥ 15.0 |

### Adaptive Learner

| Metric | Description | Unit | Baseline | Target |
|--------|-------------|------|----------|--------|
| Training Speed | Episodes trained per second | eps | 10.0 | ≥ 7.5 |
| Training Memory | Peak memory during training | MB | 100.0 | ≤ 130.0 |
| Loss Convergence | Loss reduction over training | rate | 0.90 | ≥ 0.77 |
| Prediction Accuracy | Consistency of predictions | ratio | 0.70 | ≥ 0.63 |

### Intelligent Orchestrator

| Metric | Description | Unit | Baseline | Target |
|--------|-------------|------|----------|--------|
| Planning Time | Time to create orchestration plan | ms | 200.0 | ≤ 260.0 |
| Execution Ratio | Actual time vs estimated time | ratio | 0.85 | ≤ 1.02 |
| Resource Utilization | Allocation efficiency | ratio | 0.80 | ≥ 0.64 |

## Output Format

### JSON Output

Results are saved in JSON format for easy integration with CI/CD systems:

```json
{
  "suite_name": "Gauntlet System Performance Benchmarks",
  "start_time": "2026-02-03T12:00:00Z",
  "end_time": "2026-02-03T12:05:30Z",
  "duration_seconds": 330.5,
  "total_tests": 16,
  "passed": 15,
  "failed": 0,
  "warnings": 1,
  "skipped": 0,
  "results": [
    {
      "name": "ML Optimizer - Optimization Speed",
      "component": "ml_optimizer",
      "metric_name": "iterations_per_second",
      "value": 52.3,
      "baseline": 50.0,
      "unit": "iterations/second",
      "status": "PASS",
      "timestamp": "2026-02-03T12:01:00Z",
      "metadata": {
        "std_dev": 2.1,
        "runs": 10,
        "tolerance": 0.2
      }
    }
    // ... more results
  ],
  "summary": {
    "overall_status": "PASS",
    "pass_rate": "93.8%",
    "performance_grade": "A"
  },
  "statistical_significance": {
    "ml_optimizer": {
      "iterations_per_second": {
        "significant": true,
        "difference_percent": 4.6,
        "confidence_level": 0.95
      }
    }
  }
}
```

### Console Output

```
============================================
GAUNTLET BENCHMARK SUITE
============================================
✓ Python found: Python 3.11.0
✓ Benchmark script found
✓ All required packages installed
✓ Statistical testing available (scipy)

============================================
RUNNING BENCHMARKS
============================================
ℹ Configuration:
  Output file: benchmark_results.json
  Number of runs: 10
  Verbose mode: disabled

[benchmark execution logs...]

============================================
BENCHMARK RESULTS
============================================
✓ Results saved to: benchmark_results.json

Summary:
  Total Tests: 16
  Passed: 15
  Failed: 0
  Warnings: 1
  Pass Rate: 93.8%
  Grade: A
  Duration: 330.5s
```

## Performance Grades

The suite assigns an overall performance grade based on pass rate:

| Grade | Pass Rate | Description |
|-------|-----------|-------------|
| A | ≥ 95% | Excellent performance |
| B | ≥ 85% | Good performance |
| C | ≥ 70% | Acceptable performance |
| D | ≥ 50% | Marginal performance |
| F | < 50% | Poor performance |

## Baseline Metrics

Baseline metrics are defined in `BaselineMetrics` class. To update baselines after system improvements:

```python
from gauntlet_benchmarks import BaselineMetrics

new_baselines = BaselineMetrics(
    # Update values based on recent benchmark results
    ml_optimizer_iterations_per_second=55.0,
    prediction_latency_ms=90.0,
    # ... etc
)
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install numpy scipy

    - name: Run benchmarks
      run: |
        cd tests/benchmarks
        ./run_benchmarks.sh -o benchmark_results.json -n 20

    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: tests/benchmarks/benchmark_results.json

    - name: Check performance
      run: |
        # Fail if overall status is not PASS
        STATUS=$(jq -r '.summary.overall_status' tests/benchmarks/benchmark_results.json)
        if [ "$STATUS" != "PASS" ]; then
          echo "Performance benchmarks failed!"
          exit 1
        fi
```

### GitLab CI Example

```yaml
performance:
  stage: test
  script:
    - pip install numpy scipy
    - cd tests/benchmarks
    - ./run_benchmarks.sh -o benchmark_results.json
  artifacts:
    paths:
      - tests/benchmarks/benchmark_results.json
    reports:
      metrics: benchmark_results.json
```

## Statistical Significance

The suite performs statistical tests to determine if benchmark results are significantly different from baseline:

- **Confidence Level**: Default 95% (configurable)
- **Significance Threshold**: Differences > 10% are considered significant
- **Tests**: Performs t-tests comparing sample means against baseline

## Troubleshooting

### Import Errors

If you see import errors for gauntlet components:

```bash
# Set PYTHONPATH to include glue adapters
export PYTHONPATH="${PYTHONPATH}:$(pwd)/glue/adapters/gauntlet-adapter/src"
```

### Memory Issues

For memory-constrained environments, reduce the number of runs:

```bash
./run_benchmarks.sh -n 5
```

### Missing scipy

Statistical tests require scipy. Install with:

```bash
pip install scipy
```

The suite will still run without scipy, but statistical significance tests will be limited.

## Best Practices

1. **Run Regularly**: Execute benchmarks on every commit or at least daily
2. **Track Trends**: Store historical results to track performance over time
3. **Update Baselines**: Update baseline metrics only after intentional optimizations
4. **Investigate Failures**: Always investigate benchmark failures before merging
5. **Use Consistent Hardware**: Run on consistent hardware for reliable comparisons

## Extending the Suite

To add new benchmarks:

```python
def _benchmark_new_component(self):
    """Benchmark new component"""
    logger.info("Testing new component...")

    # Run benchmark multiple times
    measurements = []
    for _ in range(self.num_runs):
        start = time.time()
        # ... run component ...
        elapsed = time.time() - start
        measurements.append(elapsed)

    mean_value = np.mean(measurements)
    baseline = self.baselines.new_component_baseline

    # Create result
    result = BenchmarkResult(
        name="New Component - Performance",
        component="new_component",
        metric_name="performance",
        value=mean_value,
        baseline=baseline,
        unit="ms",
        status=BenchmarkStatus.PASS if mean_value <= baseline else BenchmarkStatus.FAIL
    )

    self.results.append(result)
```

## License

OpenEvolve Gauntlet System - 2026

## Support

For issues or questions:
- Check logs with `--verbose` flag
- Review JSON output for detailed metrics
- Compare against baseline metrics
- Consult system documentation
