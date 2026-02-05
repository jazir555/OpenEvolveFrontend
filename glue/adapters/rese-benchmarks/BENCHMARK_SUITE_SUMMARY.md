# RESE Benchmark Suite - Quick Start Summary

## Created Files

### Benchmark Scripts (6 total)

1. **`benchmark_phase1.py`** (12.7 KB)
   - Constraint hardening benchmarks
   - Assumption mining benchmarks
   - Red team protocol benchmarks
   - Test sizes: 10, 100, 1000 assumptions

2. **`benchmark_phase2.py`** (13.3 KB)
   - Isomorphic mapping benchmarks
   - I_mech score calculation benchmarks
   - Cross-domain pattern matching benchmarks
   - Test sizes: 1, 5, 10 target domains

3. **`benchmark_phase3.py`** (13.7 KB)
   - MCTS iteration benchmarks
   - Tree node creation benchmarks
   - Convergence detection benchmarks
   - Test sizes: 100, 1000, 10000 iterations

4. **`benchmark_phase4.py`** (14.1 KB)
   - Architecture assembly benchmarks
   - Knowledge integration benchmarks
   - Validation processing benchmarks
   - Test sizes: 1, 10, 100 paradigm shifts

5. **`benchmark_full_pipeline.py`** (19.6 KB)
   - End-to-end pipeline benchmarks
   - Per-phase timing breakdown
   - Memory usage tracking
   - Test complexities: simple, medium, complex

6. **`run_all_benchmarks.py`** (14.6 KB)
   - Orchestrator for all benchmarks
   - Baseline comparison
   - Combined report generation (JSON + Markdown)

### Supporting Files (2 total)

7. **`init_baseline.py`** (3.3 KB)
   - Initialize baseline performance metrics
   - Runs all benchmarks and saves results

8. **`BENCHMARKS_README.md`** (14.8 KB)
   - Comprehensive documentation
   - Usage instructions
   - Result interpretation guide
   - Troubleshooting section

## Key Features

### Statistical Analysis

All benchmarks calculate:
- **Min/Max**: Range of execution times
- **Mean**: Average execution time
- **Median**: Middle value (robust to outliers)
- **Std Dev**: Consistency measure
- **Throughput**: Operations per second

### Test Coverage

Each phase tests multiple scenarios:
- **Small/Medium/Large** data sizes
- **3-5 iterations** per benchmark
- **Multiple problem complexities**

### Output Formats

1. **JSON Results**: Machine-readable detailed metrics
2. **Markdown Report**: Human-readable summary
3. **Baseline Comparison**: Performance regression detection

## Quick Start

### Run Single Benchmark

```bash
cd glue/adapters/rese-benchmarks

# Run Phase I benchmarks
python benchmark_phase1.py

# Results saved to: results/phase1_benchmark_TIMESTAMP.json
```

### Run All Benchmarks

```bash
# Run everything and generate report
python run_all_benchmarks.py

# With baseline comparison
python run_all_benchmarks.py --compare-baseline

# Save as new baseline
python run_all_benchmarks.py --save-baseline
```

### Initialize Baseline

```bash
# First time: create baseline metrics
python init_baseline.py
```

## Directory Structure

```
glue/adapters/rese-benchmarks/
├── benchmark_phase1.py
├── benchmark_phase2.py
├── benchmark_phase3.py
├── benchmark_phase4.py
├── benchmark_full_pipeline.py
├── run_all_benchmarks.py
├── init_baseline.py
├── BENCHMARKS_README.md
└── results/
    ├── phase1_benchmark_*.json
    ├── phase2_benchmark_*.json
    ├── phase3_benchmark_*.json
    ├── phase4_benchmark_*.json
    ├── full_pipeline_benchmark_*.json
    ├── combined_benchmark_*.json
    ├── benchmark_report_*.md
    ├── baseline_comparison_*.json
    └── baseline.json
```

## Benchmark Metrics Summary

| Phase | Operation | Metric | Target |
|-------|-----------|--------|--------|
| I | Constraint Hardening | Time | <20ms |
| I | Assumption Mining | Throughput | >50 assumptions/sec |
| I | Red Team Protocol | Time | <150ms |
| II | Isomorphic Mapping | Time | <50ms |
| II | I_mech Score | Throughput | >1000 scores/sec |
| III | MCTS Iterations | Throughput | >1000 iters/sec |
| III | Tree Creation | Throughput | >5000 nodes/sec |
| IV | Assembly | Throughput | >10 shifts/sec |
| IV | Integration | Time | <150ms |
| Full Pipeline | Medium | Time | <10000ms |
| Full Pipeline | Complex | Time | <60000ms |

## Configuration

All benchmarks use environment variables for configuration. Set before running:

```bash
# Phase I
export PHASE1_TIMEOUT_MS=15000
export PHASE1_MAX_ASSUMPTIONS=100

# Phase II
export PHASE2_TIMEOUT_MS=20000
export PHASE2_IMECH_THRESHOLD=0.7

# Phase III
export PHASE3_ITERATIONS=1000
export PHASE3_UCB1_C=1.414

# Phase IV
export PHASE4_ASSEMBLY_TIMEOUT_MS=25000
export PHASE4_MIN_CONFIDENCE_THRESHOLD=0.6
```

## Result Interpretation

### Good Performance
- **Mean time** within target range
- **Low std dev** (consistent performance)
- **High throughput** values

### Performance Regression
- **Mean time** increased vs baseline (>10%)
- **Higher std dev** (less consistent)
- **Lower throughput** vs baseline

### Improvement Opportunities
- **High std dev** (>20% of mean): Optimize for consistency
- **Scaling issues**: Time grows faster than data size
- **Memory spikes**: Large memory usage in specific operations

## Troubleshooting

**Import errors**: Verify phase executors are in `../rese-phaseX/src/`
**Timeout errors**: Reduce test data sizes or increase timeout
**Memory errors**: Close other applications, reduce data sizes
**Inconsistent results**: Run more iterations, close background apps

## Next Steps

1. **Initialize baseline**: `python init_baseline.py`
2. **Run benchmarks**: `python run_all_benchmarks.py`
3. **Review report**: Check `results/benchmark_report_*.md`
4. **Optimize**: Focus on operations with highest std dev or slowest times
5. **Re-test**: Compare against baseline to measure improvement

## Technical Details

### Precision
- **Timing**: `time.perf_counter()` for nanosecond precision
- **Memory**: `tracemalloc` for MB-precision tracking
- **Statistics**: Python `statistics` module for accuracy

### Idempotency
All benchmarks are safe to run multiple times:
- Unique IDs generated for each test
- Results timestamped
- No side effects on production data

### CLAUDE.md Compliance
- **Configuration Explicitness**: All config via env vars
- **Circuit Breaker**: Timeouts prevent infinite hangs
- **Structured Logging**: JSON output for parsing
- **Idempotency**: Safe to run 100x

## Author

RESE Team
Created: 2026-02-04
Version: 1.0.0
