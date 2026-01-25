# Benchmark Integration Guide

Complete guide for integrating the knowledge graph benchmark suite into your development workflow.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running Benchmarks](#running-benchmarks)
5. [CI/CD Integration](#cicd-integration)
6. [Performance Monitoring](#performance-monitoring)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Quick Start

### 1. Install Dependencies

```bash
cd tests/benchmarks
pip install -r requirements_benchmarks.txt
```

### 2. Run Quick Test

```bash
# Verify the system works
python test_benchmarks.py --smoke

# Run quick benchmark subset
python run_benchmarks.py --quick
```

### 3. View Results

Results are saved to `benchmark_results/`:
- `benchmark_report_*.md` - Human-readable report
- `benchmark_metrics_*.json` - Raw metrics data

## Installation

### Requirements

- Python 3.8+
- Knowledge Engine dependencies
- Benchmark monitoring tools

### Setup Steps

1. **Install system dependencies** (Ubuntu/Debian):
   ```bash
   sudo apt-get install python3-dev
   ```

2. **Install Python packages**:
   ```bash
   pip install -r tests/benchmarks/requirements_benchmarks.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import psutil, matplotlib; print('✓ Dependencies OK')"
   ```

## Configuration

### Basic Configuration

Edit `tests/benchmarks/benchmark_config.yaml`:

```yaml
knowledge_addition:
  num_artifacts: [100, 500, 1000]
  batch_sizes: [1, 10, 50]

knowledge_retrieval:
  num_queries: [10, 50, 100]
  query_types: [keyword, graph]
```

### Advanced Configuration

#### Performance Thresholds

Set acceptable performance limits:

```yaml
thresholds:
  knowledge_addition:
    min_throughput: 100
    max_memory: 2.0

  knowledge_retrieval:
    max_latency_p95: 500
```

#### Custom Test Data

Modify test data in `kg_performance_benchmarks.py`:

```python
class KnowledgeGraphPerformanceBenchmarks:
    def _init_test_data(self):
        self.sample_entities = [
            "YourEntity1", "YourEntity2", ...
        ]
        self.sample_relations = [
            "your_relation_1", "your_relation_2", ...
        ]
```

## Running Benchmarks

### Command-Line Interface

```bash
# Quick benchmarks (small datasets)
python run_benchmarks.py --quick

# All benchmarks (full suite)
python run_benchmarks.py --all

# Specific benchmark
python run_benchmarks.py --benchmark knowledge_addition

# Custom parameters
python run_benchmarks.py \
    --benchmark knowledge_addition \
    --num-artifacts 5000

# Custom config and output
python run_benchmarks.py \
    --config my_config.yaml \
    --output-dir my_results \
    --all
```

### Programmatic Usage

#### Basic Example

```python
import asyncio
from knowledge_engine.engine import KnowledgeEngine
from tests.benchmarks.kg_performance_benchmarks import (
    KnowledgeGraphPerformanceBenchmarks
)

async def run_benchmarks():
    engine = KnowledgeEngine()
    benchmarks = KnowledgeGraphPerformanceBenchmarks(engine)

    # Run benchmark
    result = await benchmarks.benchmark_knowledge_addition(
        num_artifacts=1000,
        batch_size=50
    )

    if result.success:
        print(f"Throughput: {result.metrics['artifacts_per_second']:.2f}/sec")

    # Generate report
    benchmarks.generate_report("my_report.md")

    await engine.cleanup_kggen_pipeline()

asyncio.run(run_benchmarks())
```

#### Advanced Example

```python
async def comprehensive_benchmark():
    engine = KnowledgeEngine()
    benchmarks = KnowledgeGraphPerformanceBenchmarks(engine)

    # Test multiple configurations
    configs = [
        {"num_artifacts": 100, "batch_size": 10},
        {"num_artifacts": 1000, "batch_size": 50},
        {"num_artifacts": 5000, "batch_size": 100}
    ]

    results = []
    for config in configs:
        result = await benchmarks.benchmark_knowledge_addition(**config)
        results.append(result)

    # Analyze results
    for result in results:
        throughput = result.metrics["artifacts_per_second"]
        print(f"Throughput: {throughput:.2f}/sec")

    await engine.cleanup_kggen_pipeline()
```

## CI/CD Integration

### GitHub Actions Example

Create `.github/workflows/benchmarks.yml`:

```yaml
name: Knowledge Graph Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    # Run daily at midnight UTC
    - cron: '0 0 * * *'

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r tests/benchmarks/requirements_benchmarks.txt

    - name: Run benchmarks
      run: |
        cd tests/benchmarks
        python run_benchmarks.py --quick

    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: tests/benchmarks/benchmark_results/

    - name: Generate PR comment
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const report = fs.readFileSync('tests/benchmarks/benchmark_results/benchmark_report_latest.md', 'utf8');

          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: `## Benchmark Results\n\n${report}`
          });
```

### GitLab CI Example

Create `.gitlab-ci.yml`:

```yaml
benchmark:
  stage: test
  image: python:3.10

  script:
    - pip install -r requirements.txt
    - pip install -r tests/benchmarks/requirements_benchmarks.txt
    - cd tests/benchmarks
    - python run_benchmarks.py --quick

  artifacts:
    paths:
      - tests/benchmarks/benchmark_results/
    reports:
      metrics: benchmark_metrics.txt

  only:
    - main
    - merge_requests
```

### Jenkins Pipeline Example

Create `Jenkinsfile`:

```groovy
pipeline {
    agent any

    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pip install -r tests/benchmarks/requirements_benchmarks.txt'
            }
        }

        stage('Benchmarks') {
            steps {
                sh '''
                    cd tests/benchmarks
                    python run_benchmarks.py --quick
                '''
            }
        }

        stage('Archive') {
            steps {
                archiveArtifacts artifacts: 'tests/benchmarks/benchmark_results/**'
            }
        }
    }

    post {
        always {
            publishHTML([
                reportDir: 'tests/benchmarks/benchmark_results',
                reportFiles: 'benchmark_report_*.md',
                reportName: 'Benchmark Report'
            ])
        }
    }
}
```

## Performance Monitoring

### Trend Analysis

Track performance over time:

```python
import json
from pathlib import Path
from datetime import datetime

def analyze_trends(results_dir="benchmark_results"):
    """Analyze performance trends across multiple runs."""

    results = []
    for json_file in Path(results_dir).glob("benchmark_metrics_*.json"):
        with open(json_file) as f:
            data = json.load(f)
            results.append(data)

    # Extract throughput over time
    throughput_trend = []
    for result in results:
        timestamp = result.get("timestamp")
        kg_addition = result.get("benchmarks", {}).get("knowledge_addition", {})
        throughput = kg_addition.get("metrics", {}).get("artifacts_per_second")
        if throughput:
            throughput_trend.append((timestamp, throughput))

    # Print trend
    print("Throughput Trend:")
    for timestamp, throughput in throughput_trend[-10:]:
        print(f"  {timestamp}: {throughput:.2f}/sec")
```

### Performance Regression Detection

Automatically detect performance regressions:

```python
def detect_regressions(current_results, baseline_results, threshold=0.1):
    """Detect if performance degraded beyond threshold."""

    regressions = []

    # Compare key metrics
    for benchmark_name in current_results.keys():
        if benchmark_name not in baseline_results:
            continue

        current = current_results[benchmark_name]["metrics"]
        baseline = baseline_results[benchmark_name]["metrics"]

        # Check throughput
        if "artifacts_per_second" in current:
            current_throughput = current["artifacts_per_second"]
            baseline_throughput = baseline["artifacts_per_second"]

            if current_throughput < baseline_throughput * (1 - threshold):
                regressions.append({
                    "benchmark": benchmark_name,
                    "metric": "artifacts_per_second",
                    "baseline": baseline_throughput,
                    "current": current_throughput,
                    "degradation": (1 - current_throughput / baseline_throughput) * 100
                })

    return regressions
```

### Dashboard Integration

Generate metrics for monitoring dashboards:

```python
def export_prometheus_metrics(benchmarks, output_path="metrics.txt"):
    """Export metrics in Prometheus format."""

    with open(output_path, 'w') as f:
        for name, result in benchmarks.results.items():
            if result.success:
                for metric_name, value in result.metrics.items():
                    if isinstance(value, (int, float)):
                        # Sanitize metric name
                        safe_name = f"openevolve_kg_{name}_{metric_name}"
                        safe_name = safe_name.replace("/", "_per_").replace("-", "_")

                        f.write(f"{safe_name} {value}\n")
```

## Troubleshooting

### Common Issues

#### Issue: High Memory Usage

**Symptoms**: Benchmarks fail with out-of-memory errors

**Solutions**:
```yaml
# Reduce dataset sizes in benchmark_config.yaml
knowledge_addition:
  num_artifacts: [100, 500]  # Instead of [100, 500, 1000, 5000]
```

#### Issue: Inconsistent Results

**Symptoms**: Results vary significantly between runs

**Solutions**:
1. Ensure system is idle during benchmarks
2. Run benchmarks multiple times
3. Use averages instead of single runs
4. Disable background services

```python
# Run multiple times and average
results = []
for i in range(5):
    result = await benchmarks.benchmark_knowledge_addition(num_artifacts=1000)
    results.append(result)

avg_throughput = mean(r.metrics["artifacts_per_second"] for r in results)
```

#### Issue: Import Errors

**Symptoms**: Module not found errors

**Solutions**:
```bash
# Ensure correct Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Reinstall dependencies
pip install --force-reinstall -r tests/benchmarks/requirements_benchmarks.txt
```

#### Issue: Timeouts

**Symptoms**: Benchmarks hang or timeout

**Solutions**:
```python
# Add timeouts to async operations
import asyncio

async def run_with_timeout():
    try:
        result = await asyncio.wait_for(
            benchmarks.benchmark_knowledge_addition(num_artifacts=10000),
            timeout=300  # 5 minutes
        )
    except asyncio.TimeoutError:
        print("Benchmark timed out")
```

## Best Practices

### 1. Baseline Establishment

Always establish a baseline before making changes:

```bash
# Run before changes
python run_benchmarks.py --quick --output-dir baseline_results

# Make changes...

# Run after changes
python run_benchmarks.py --quick --output-dir current_results

# Compare
python compare_results.py baseline_results current_results
```

### 2. Consistent Environment

Use consistent testing environments:

```bash
# Use Docker for consistency
docker build -t openevolve-benchmarks .
docker run openevolve-benchmarks
```

### 3. Automated Testing

Integrate into your test suite:

```python
# tests/test_performance.py
import pytest
from tests.benchmarks.kg_performance_benchmarks import (
    KnowledgeGraphPerformanceBenchmarks
)

@pytest.mark.performance
def test_knowledge_addition_throughput():
    """Test that knowledge addition meets performance requirements."""
    engine = KnowledgeEngine()
    benchmarks = KnowledgeGraphPerformanceBenchmarks(engine)

    result = asyncio.run(benchmarks.benchmark_knowledge_addition(
        num_artifacts=1000
    ))

    assert result.success
    assert result.metrics["artifacts_per_second"] >= 100  # Min requirement
```

### 4. Documentation

Document benchmark results and interpretations:

```markdown
# Performance Run 2025-01-07

## Environment
- CPU: Intel Xeon E5-2680 v4
- RAM: 32 GB
- Python: 3.10.0

## Results
- Knowledge Addition: 427.35 artifacts/sec
- Knowledge Retrieval: 45.23ms avg latency

## Interpretation
Performance improved by 15% compared to baseline.
Optimization in batch processing was effective.
```

### 5. Regular Monitoring

Run benchmarks regularly:

```bash
# Add to crontab for daily runs
0 0 * * * cd /path/to/project && python tests/benchmarks/run_benchmarks.py --quick
```

---

**Last Updated:** 2025-01-07
**Version:** 1.0.0
