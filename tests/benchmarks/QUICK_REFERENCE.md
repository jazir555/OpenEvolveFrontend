# Gauntlet Benchmark Suite - Quick Reference

## Quick Start

```bash
# Run all benchmarks
./run_benchmarks.sh

# With options
./run_benchmarks.sh -o results.json -n 20 -v
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o FILE` | Output JSON file | benchmark_results.json |
| `-n NUM` | Number of runs per benchmark | 10 |
| `-v` | Verbose output | disabled |
| `-h` | Help message | - |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All benchmarks passed |
| 1 | One or more benchmarks failed |
| 2 | Configuration error |

## Python API

### Basic Usage

```python
from gauntlet_benchmarks import GauntletBenchmarkSuite

suite = GauntletBenchmarkSuite()
results = suite.run_all_benchmarks()
results.to_json("output.json")
```

### Custom Configuration

```python
from gauntlet_benchmarks import (
    GauntletBenchmarkSuite,
    BaselineMetrics,
    PerformanceTargets
)

# Custom baselines
baselines = BaselineMetrics(
    ml_optimizer_iterations_per_second=60.0,
    prediction_latency_ms=80.0
)

# Custom targets
targets = PerformanceTargets(
    ml_optimizer_speed_tolerance=0.15,
    min_prediction_accuracy=0.80
)

# Create suite
suite = GauntletBenchmarkSuite(
    baselines=baselines,
    targets=targets,
    num_runs=20,
    confidence_level=0.99
)

results = suite.run_all_benchmarks()
```

### Access Results

```python
# Summary
print(results.summary)
# {'overall_status': 'PASS', 'pass_rate': '93.8%', 'performance_grade': 'A'}

# Individual results
for result in results.results:
    print(f"{result.name}: {result.value} {result.unit}")
    print(f"  Status: {result.status.value}")
    print(f"  Baseline: {result.baseline} {result.unit}")

# Filter by component
ml_results = [r for r in results.results if r.component == "ml_optimizer"]

# Filter by status
failures = [r for r in results.results if r.status == BenchmarkStatus.FAIL]

# Statistical significance
stats = results.statistical_significance
```

## Component Benchmarks

### ML Optimizer (4 tests)

- **Optimization Speed**: iterations/second
- **Memory Usage**: MB
- **Convergence Rate**: ratio (0-1)
- **Improvement %**: percent

### Predictive Executor (3 tests)

- **Prediction Latency**: ms
- **Prediction Accuracy**: ratio (0-1)
- **Cost Savings**: percent

### Adaptive Learner (4 tests)

- **Training Speed**: episodes/second
- **Training Memory**: MB
- **Loss Convergence**: ratio (0-1)
- **Prediction Accuracy**: ratio (0-1)

### Intelligent Orchestrator (3 tests)

- **Planning Time**: ms
- **Execution Time Ratio**: ratio
- **Resource Utilization**: ratio (0-1)

## Baseline Metrics

```
ML Optimizer:
  - Speed: 50.0 iter/s
  - Memory: 50.0 MB
  - Convergence: 0.95
  - Improvement: 15.0%

Predictive Executor:
  - Latency: 100.0 ms
  - Accuracy: 0.75
  - Savings: 20.0%

Adaptive Learner:
  - Speed: 10.0 eps
  - Memory: 100.0 MB
  - Convergence: 0.90
  - Accuracy: 0.70

Intelligent Orchestrator:
  - Planning: 200.0 ms
  - Exec Ratio: 0.85
  - Utilization: 0.80
```

## Performance Targets

Tolerances for PASS status:

| Component | Metric | Tolerance |
|-----------|--------|-----------|
| ML Optimizer | Speed | ±20% |
| ML Optimizer | Memory | +30% |
| Predictive | Latency | +30% |
| Adaptive | Speed | ±25% |
| Adaptive | Memory | +30% |
| Orchestrator | Planning | +30% |

Minimum thresholds:

| Metric | Minimum |
|--------|---------|
| Prediction Accuracy | 0.70 |
| Cost Savings | 15.0% |
| Improvement % | 10.0% |

## Status Codes

| Status | Meaning |
|--------|---------|
| PASS | Within tolerance |
| FAIL | Outside tolerance |
| WARNING | Below minimum but within tolerance |
| SKIPPED | Component not available |

## Performance Grades

| Grade | Pass Rate |
|-------|-----------|
| A | ≥ 95% |
| B | ≥ 85% |
| C | ≥ 70% |
| D | ≥ 50% |
| F | < 50% |

## JSON Structure

```json
{
  "suite_name": "string",
  "start_time": "ISO8601",
  "end_time": "ISO8601",
  "duration_seconds": float,
  "total_tests": int,
  "passed": int,
  "failed": int,
  "warnings": int,
  "skipped": int,
  "results": [
    {
      "name": "string",
      "component": "string",
      "metric_name": "string",
      "value": float,
      "baseline": float,
      "unit": "string",
      "status": "PASS|FAIL|WARNING|SKIPPED",
      "timestamp": "ISO8601",
      "metadata": {}
    }
  ],
  "summary": {
    "overall_status": "PASS|FAIL",
    "pass_rate": "string",
    "performance_grade": "A|B|C|D|F"
  },
  "statistical_significance": {}
}
```

## Troubleshooting

### Import Errors
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/glue/adapters/gauntlet-adapter/src"
```

### Memory Issues
```bash
./run_benchmarks.sh -n 5  # Fewer runs
```

### Missing Dependencies
```bash
pip install numpy scipy
```

### No jq Available
```bash
# Ubuntu/Debian
sudo apt-get install jq

# Mac
brew install jq
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Run benchmarks
  run: |
    cd tests/benchmarks
    ./run_benchmarks.sh -o results.json

- name: Check results
  run: |
    STATUS=$(jq -r '.summary.overall_status' tests/benchmarks/results.json)
    [ "$STATUS" = "PASS" ]
```

### GitLab CI

```yaml
benchmark:
  script:
    - cd tests/benchmarks
    - ./run_benchmarks.sh
  artifacts:
    paths:
      - tests/benchmarks/benchmark_results.json
```

## Best Practices

1. **Run Regularly**: Every commit or daily
2. **Consistent Hardware**: Use same machine for comparison
3. **Track History**: Store all benchmark results
4. **Update Baselines**: Only after intentional improvements
5. **Investigate Failures**: Never ignore benchmark failures

## Files

| File | Purpose |
|------|---------|
| `gauntlet_benchmarks.py` | Main benchmark suite |
| `run_benchmarks.sh` | Shell runner script |
| `baseline_config.json` | Baseline configuration |
| `example_usage.py` | Usage examples |
| `README.md` | Full documentation |
| `QUICK_REFERENCE.md` | This file |

## Support

For detailed documentation, see `README.md`

For examples, run `python example_usage.py`

For help: `./run_benchmarks.sh --help`
