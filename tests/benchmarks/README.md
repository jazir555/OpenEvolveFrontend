# Knowledge Graph Performance Benchmarks

Comprehensive performance benchmarking suite for the OpenEvolve Knowledge Graph system.

## Overview

This benchmark suite provides extensive performance testing capabilities for all knowledge graph components, measuring throughput, latency, memory usage, accuracy, and scalability under various conditions.

## Benchmarks

### 1. Knowledge Addition Throughput
Tests the performance of adding knowledge artifacts to the graph.

**Metrics:**
- Artifacts per second
- Batch addition efficiency
- Memory usage (GB)
- CPU utilization (%)
- Peak memory usage

**Parameters:**
- `num_artifacts`: Number of artifacts to add (default: 1000)
- `batch_size`: Size of batches (default: 10)

### 2. Knowledge Retrieval Latency
Measures search and query performance.

**Metrics:**
- Average query latency (ms)
- P50, P95, P99 latencies
- Queries per second
- Comparison across search types

**Parameters:**
- `num_queries`: Number of queries to execute (default: 100)
- `query_types`: Types of searches to test
  - `hybrid`: Combined vector + keyword
  - `vector`: Pure similarity search
  - `keyword`: Traditional keyword search
  - `graph`: Graph traversal search

### 3. Deduplication Performance
Tests entity deduplication accuracy and speed.

**Metrics:**
- Processing time (seconds)
- Accuracy (%)
- Precision, Recall, F1 Score
- Duplicate reduction rate
- Memory usage (MB)

**Parameters:**
- `num_entities`: Number of entities (default: 1000)
- `duplicate_rate`: Rate of duplicates 0.0-1.0 (default: 0.3)

### 4. Graph Algorithm Scalability
Benchmarks graph processing algorithms.

**Metrics:**
- Processing time vs graph size
- Memory usage vs graph size
- Scalability characteristics

**Parameters:**
- `graph_sizes`: List of node counts to test (default: [100, 500, 1000, 5000])

### 5. Concurrent Operations
Tests system behavior under concurrent load.

**Metrics:**
- Concurrent throughput (ops/sec)
- Resource contention
- Error rate under load

**Parameters:**
- `num_concurrent`: Number of concurrent clients (default: 10)
- `operations_per_client`: Operations per client (default: 50)

### 6. End-to-End Workflows
Realistic workflow scenario benchmarks.

**Scenarios:**
- `entity_relationship_workflow`: Add entities and relationships
- `batch_processing_workflow`: Batch processing performance
- `query_workflow`: Query and retrieval workflows
- `document_processing_workflow`: Document to knowledge graph
- `temporal_workflow`: Temporal knowledge operations
- `extraction_pipeline_workflow`: Multi-stage extraction

## Installation

```bash
# Install benchmark dependencies
pip install -r requirements_benchmarks.txt
```

## Usage

### Quick Start

Run a quick subset of benchmarks:

```bash
python run_benchmarks.py --quick
```

### Run All Benchmarks

Execute the complete benchmark suite:

```bash
python run_benchmarks.py --all
```

### Run Specific Benchmark

Run a single benchmark with custom parameters:

```bash
# Knowledge addition
python run_benchmarks.py --benchmark knowledge_addition --num-artifacts 5000

# Knowledge retrieval
python run_benchmarks.py --benchmark knowledge_retrieval --num-queries 500

# Deduplication
python run_benchmarks.py --benchmark deduplication

# Graph algorithms
python run_benchmarks.py --benchmark graph_algorithms

# Concurrent operations
python run_benchmarks.py --benchmark concurrent_operations

# End-to-end workflows
python run_benchmarks.py --benchmark end_to_end_workflows
```

### Use Custom Configuration

```bash
python run_benchmarks.py --config benchmark_config.yaml --all
```

### Specify Output Directory

```bash
python run_benchmarks.py --quick --output-dir my_benchmark_results
```

## Configuration

Edit `benchmark_config.yaml` to customize:

```yaml
knowledge_addition:
  num_artifacts: [100, 500, 1000, 5000]
  batch_sizes: [1, 10, 50, 100]

knowledge_retrieval:
  num_queries: [10, 50, 100, 500]
  query_types: [hybrid, vector, keyword, graph]

# ... more configuration
```

## Output

Benchmark results are saved to the output directory (default: `benchmark_results/`):

- **`benchmark_report_*.md`**: Human-readable markdown report
- **`benchmark_metrics_*.json`**: Raw metric data as JSON
- **`benchmark_execution.log`**: Execution log

### Report Format

The markdown report includes:

1. **Summary**: Overall benchmark statistics
2. **Detailed Results**: Per-benchmark metrics
3. **Performance Summary Table**: Key metrics comparison
4. **Raw Data**: Complete metric dumps (optional)

Example:

```markdown
## Summary

- **Successful:** 10
- **Failed:** 0
- **Success Rate:** 100.0%

## Detailed Results

### Knowledge Addition

**Status:** ✓ Success

- **duration_seconds:** 2.34
- **artifacts_per_second:** 427.35
- **memory_used_gb:** 0.15
- **peak_memory_gb:** 0.28
- **cpu_usage_percent:** 45.2

## Performance Summary

| Benchmark | Status | Key Metric | Value |
|-----------|--------|------------|-------|
| knowledge_addition | ✓ | Artifacts Per Second | 427.35 |
| knowledge_retrieval | ✓ | Avg Latency | 45.23ms |
...
```

## Programmatic Usage

You can also use benchmarks programmatically:

```python
import asyncio
from knowledge_engine.engine import KnowledgeEngine
from tests.benchmarks.kg_performance_benchmarks import (
    KnowledgeGraphPerformanceBenchmarks
)

async def run_custom_benchmarks():
    # Initialize
    engine = KnowledgeEngine()
    benchmarks = KnowledgeGraphPerformanceBenchmarks(engine)

    # Run specific benchmark
    result = await benchmarks.benchmark_knowledge_addition(
        num_artifacts=5000,
        batch_size=50
    )

    # Access results
    print(f"Throughput: {result.metrics['artifacts_per_second']:.2f}/sec")

    # Generate report
    benchmarks.generate_report("custom_report.md")

    # Cleanup
    await engine.cleanup_kggen_pipeline()

# Run
asyncio.run(run_custom_benchmarks())
```

## Performance Thresholds

Set performance thresholds in `benchmark_config.yaml`:

```yaml
thresholds:
  knowledge_addition:
    min_throughput: 100  # artifacts per second
    max_memory: 2.0      # GB

  knowledge_retrieval:
    max_latency_p95: 500  # milliseconds
    min_throughput: 50    # queries per second
```

These thresholds can be used to validate that performance meets requirements.

## Interpreting Results

### Good Performance Indicators

- **Throughput**: Higher is better (artifacts/queries per second)
- **Latency**: Lower is better (milliseconds)
- **Memory**: Lower is better (GB/MB)
- **Accuracy**: Higher is better (percentage)
- **F1 Score**: Higher is better (0-1)
- **Error Rate**: Lower is better (percentage)

### Performance Analysis

Compare results across:

1. **Different dataset sizes**: Identify scalability issues
2. **Different configurations**: Find optimal settings
3. **Different time periods**: Detect performance regressions
4. **Different environments**: Compare deployment options

## Troubleshooting

### Common Issues

**Issue**: High memory usage
- **Solution**: Reduce batch sizes or dataset sizes in configuration

**Issue**: Benchmarks timeout
- **Solution**: Reduce dataset sizes or increase timeout values

**Issue**: Import errors
- **Solution**: Ensure all dependencies are installed via `requirements_benchmarks.txt`

**Issue**: Inconsistent results
- **Solution**: Run benchmarks multiple times and average, ensure system is idle

## Best Practices

1. **Run on idle systems**: Close other applications for consistent results
2. **Multiple runs**: Execute benchmarks multiple times for reliability
3. **Baseline first**: Establish a baseline before making changes
4. **Monitor system resources**: Use tools like `htop` or `Task Manager`
5. **Save results**: Archive benchmark results for historical comparison
6. **Automate**: Integrate into CI/CD pipelines for regression testing

## Extending Benchmarks

To add custom benchmarks:

```python
class KnowledgeGraphPerformanceBenchmarks:
    async def benchmark_custom_operation(
        self,
        param1: int,
        param2: str
    ) -> BenchmarkResult:
        """
        Custom benchmark description.

        Metrics:
        - metric1: Description
        - metric2: Description
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARK: Custom Operation")
        print(f"{'='*60}")

        try:
            start_time = time.time()
            start_mem = self.psutil.virtual_memory().used

            # Your benchmark code here
            result = await self.perform_custom_operation(param1, param2)

            duration = time.time() - start_time
            mem_used = (self.psutil.virtual_memory().used - start_mem) / (1024**2)

            return BenchmarkResult(
                name="custom_operation",
                metrics={
                    "duration_seconds": duration,
                    "memory_mb": mem_used,
                    "result": result
                }
            )

        except Exception as e:
            return BenchmarkResult(
                name="custom_operation",
                metrics={},
                success=False,
                error=str(e)
            )
```

## Contributing

When contributing new benchmarks:

1. Follow existing naming conventions
2. Include comprehensive docstrings
3. Return `BenchmarkResult` objects
4. Handle errors gracefully
5. Add configuration options to `benchmark_config.yaml`
6. Update this README

## License

OpenEvolve Framework - See project LICENSE file.

## Support

For issues or questions:
- GitHub Issues: [Project Repository]
- Documentation: [Project Wiki]
- Email: [Support Email]

---

**Last Updated:** 2025-01-07
**Version:** 1.0.0
