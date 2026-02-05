# RESE Performance Benchmark Suite

Comprehensive performance benchmark suite for the RESE (Recursive Epistemic Solvability Engine) framework.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Benchmark Descriptions](#benchmark-descriptions)
- [Result Interpretation](#result-interpretation)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

The RESE Benchmark Suite provides comprehensive performance testing for all four phases of the RESE pipeline:

- **Phase I**: Epistemic Audit benchmarks (constraint hardening, assumption mining, red team protocol)
- **Phase II**: Isomorphic Mapping benchmarks (mapping computation, I_mech scoring, pattern matching)
- **Phase III**: MCTS Search benchmarks (iteration throughput, tree creation, convergence detection)
- **Phase IV**: Architecture Assembly benchmarks (assembly time, knowledge integration, validation)
- **Full Pipeline**: End-to-end benchmarks with memory profiling

Each benchmark runs multiple iterations and calculates:
- **Min**: Minimum execution time
- **Max**: Maximum execution time
- **Mean**: Average execution time
- **Median**: Median execution time
- **Std Dev**: Standard deviation (measure of consistency)
- **Throughput**: Operations per second

---

## Directory Structure

```
glue/adapters/rese-benchmarks/
├── benchmark_phase1.py          # Phase I benchmarks
├── benchmark_phase2.py          # Phase II benchmarks
├── benchmark_phase3.py          # Phase III benchmarks
├── benchmark_phase4.py          # Phase IV benchmarks
├── benchmark_full_pipeline.py   # Full pipeline benchmarks
├── run_all_benchmarks.py        # Orchestrator script
├── BENCHMARKS_README.md         # This file
└── results/                     # Benchmark output directory
    ├── phase1_benchmark_*.json
    ├── phase2_benchmark_*.json
    ├── phase3_benchmark_*.json
    ├── phase4_benchmark_*.json
    ├── full_pipeline_benchmark_*.json
    ├── combined_benchmark_*.json
    ├── benchmark_report_*.md
    └── baseline.json             # Baseline results (optional)
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- All RESE phase dependencies installed
- (Optional) `memory_profiler` for detailed memory tracking

### Setup

1. Ensure all RESE phase adapters are available:
   ```bash
   cd glue/adapters
   ls rese-phase1 rese-phase2 rese-phase3 rese-phase4
   ```

2. The benchmark scripts use `sys.path` to import phase executors, so no additional installation needed.

3. (Optional) Install memory profiler:
   ```bash
   pip install memory_profiler
   ```

---

## Usage

### Run Individual Phase Benchmarks

Each phase benchmark can be run independently:

```bash
# Phase I: Epistemic Audit
cd glue/adapters/rese-benchmarks
python benchmark_phase1.py

# Phase II: Isomorphic Mapping
python benchmark_phase2.py

# Phase III: MCTS Search
python benchmark_phase3.py

# Phase IV: Architecture Assembly
python benchmark_phase4.py

# Full Pipeline
python benchmark_full_pipeline.py
```

Each script will:
1. Run all benchmarks for that phase
2. Display results to console
3. Save JSON results to `results/` directory with timestamp

### Run All Benchmarks (Orchestrator)

The orchestrator runs all benchmark suites and generates a combined report:

```bash
python run_all_benchmarks.py
```

**Options:**

```bash
# Run only specific phases
python run_all_benchmarks.py --phases phase1 phase3

# Compare against baseline (if exists)
python run_all_benchmarks.py --compare-baseline

# Save results as new baseline
python run_all_benchmarks.py --save-baseline
```

**Output:**

- `combined_benchmark_TIMESTAMP.json`: All benchmark results in JSON
- `benchmark_report_TIMESTAMP.md`: Human-readable Markdown report
- `baseline_comparison_TIMESTAMP.json`: Comparison against baseline (if `--compare-baseline`)

---

## Benchmark Descriptions

### Phase I Benchmarks (`benchmark_phase1.py`)

#### 1. Constraint Hardening

Measures the time to extract and harden constraints from problem descriptions.

- **Test Sizes**: small, medium, large (based on problem description length)
- **Metrics**:
  - Time to harden constraints (milliseconds)
  - Number of constraints extracted
  - Throughput (constraints per second)

#### 2. Assumption Mining

Measures the time to mine tacit assumptions from failure patterns.

- **Test Sizes**: 10, 100, 1000 failure patterns
- **Metrics**:
  - Time to mine assumptions (milliseconds)
  - Number of assumptions mined
  - Throughput (assumptions per second)

#### 3. Red Team Protocol

Measures the time to adversarially test hypotheses.

- **Test Sizes**: 10, 100, 1000 assumptions
- **Metrics**:
  - Time to execute red team protocol (milliseconds)
  - Number of falsifications detected
  - Throughput (assumptions tested per second)

---

### Phase II Benchmarks (`benchmark_phase2.py`)

#### 1. Isomorphic Mapping

Measures the time to find isomorphic mappings between domains.

- **Test Sizes**: 1, 5, 10 target domains
- **Metrics**:
  - Time to find mappings (milliseconds)
  - Number of mappings found
  - Throughput (mappings per second)

#### 2. I_mech Score Calculation

Measures the time to calculate mechanistic isomorphism scores.

- **Test Sizes**: 10, 20, 50 nodes per FDG
- **Metrics**:
  - Time to calculate I_mech score (microseconds)
  - Score values
  - Throughput (scores per second)

#### 3. Cross-Domain Pattern Matching

Measures the time to identify patterns across multiple domains.

- **Test Sizes**: 1, 5, 10 target domains
- **Metrics**:
  - Time to identify patterns (milliseconds)
  - Number of patterns found
  - Throughput (patterns per second)

---

### Phase III Benchmarks (`benchmark_phase3.py`)

#### 1. MCTS Iterations

Measures the throughput of Monte Carlo Tree Search iterations.

- **Test Sizes**: 100, 1000, 10000 iterations
- **Metrics**:
  - Time to complete iterations (milliseconds)
  - Tree statistics (nodes, depth)
  - Throughput (iterations per second)

#### 2. Tree Node Creation

Measures the rate of search tree node creation.

- **Test Sizes**: 100, 1000, 5000 nodes
- **Metrics**:
  - Time to create nodes (milliseconds)
  - Nodes created
  - Throughput (nodes per second)

#### 3. Convergence Detection

Measures the speed of ACI (Algorithmic Convergence Indicator) calculation.

- **Test Sizes**: 50, 100, 200 window sizes
- **Metrics**:
  - Time to check convergence (microseconds)
  - Convergence status
  - Throughput (checks per second)

---

### Phase IV Benchmarks (`benchmark_phase4.py`)

#### 1. Architecture Assembly

Measures the time to assemble paradigm shifts from patterns.

- **Test Sizes**: 1, 10, 100 paradigm shifts
- **Metrics**:
  - Time to assemble shifts (milliseconds)
  - Number of shifts assembled
  - Throughput (shifts per second)

#### 2. Knowledge Integration

Measures the time to integrate knowledge from all phases.

- **Test Sizes**: 1, 10, 100 paradigm shifts
- **Metrics**:
  - Time to integrate knowledge (milliseconds)
  - Integration confidence
  - Throughput (integrations per second)

#### 3. Validation Processing

Measures the time to validate architecture assemblies.

- **Test Sizes**: 1, 10, 100 paradigm shifts
- **Metrics**:
  - Time to validate (milliseconds)
  - Validation checks performed
  - Throughput (validations per second)

---

### Full Pipeline Benchmarks (`benchmark_full_pipeline.py`)

Measures end-to-end performance with all four phases.

#### Test Complexities

- **Simple**: Small problem, 10 assumptions, 100 MCTS iterations, 5 shifts
- **Medium**: Moderate problem, 100 assumptions, 1000 iterations, 20 shifts
- **Complex**: Large problem, 1000 assumptions, 5000 iterations, 50 shifts

#### Metrics

- **Per-Phase Timing**: Time for each phase (milliseconds)
- **Memory Usage**: Peak memory per phase (megabytes)
- **Total Time**: End-to-end pipeline time (milliseconds)
- **Peak Memory**: Maximum memory across all phases (megabytes)

---

## Result Interpretation

### JSON Output Format

Each benchmark produces a JSON file with the following structure:

```json
{
  "phase": "phase1_epistemic_audit",
  "timestamp": "2026-02-04T12:34:56.789Z",
  "system_info": {
    "python_version": "3.11.0",
    "platform": "win32"
  },
  "benchmarks": [
    {
      "benchmark": "constraint_hardening",
      "problem_size": "medium",
      "iterations": 5,
      "timings_ms": {
        "min": 12.34,
        "max": 15.67,
        "mean": 14.12,
        "median": 14.01,
        "stdev": 1.23
      },
      "throughput": {
        "constraints_per_second": 456.78
      }
    }
  ]
}
```

### Key Metrics

#### Timings

- **Mean**: Average execution time across all iterations. Use this for general performance assessment.
- **Median**: Middle value. Less sensitive to outliers than mean.
- **Std Dev**: Standard deviation. Lower values indicate more consistent performance.
- **Min/Max**: Range of execution times. Large gaps may indicate performance variability.

#### Throughput

Operations completed per second. Higher is better.

- Compare throughput across different problem sizes to assess scalability
- Look for linear or sub-linear scaling (ideal)
- Exponential scaling indicates potential performance issues

#### Baseline Comparison

When comparing against baseline:

- **✓ (negative change)**: Performance improved (faster)
- **✗ (positive >10%)**: Performance regressed (slower)
- **= (within ±10%)**: Performance stable

### Performance Targets

As a general guideline, aim for:

| Phase | Operation | Target Mean Time | Target Throughput |
|-------|-----------|------------------|-------------------|
| I | Constraint Hardening | <20ms | >100 constraints/sec |
| I | Assumption Mining | <100ms | >50 assumptions/sec |
| I | Red Team Protocol | <150ms | >30 assumptions/sec |
| II | Isomorphic Mapping | <50ms | >5 mappings/sec |
| II | I_mech Score | <1ms | >1000 scores/sec |
| III | MCTS Iterations | <1000ms/1000 iters | >1000 iters/sec |
| III | Tree Creation | <200ms/1000 nodes | >5000 nodes/sec |
| IV | Assembly | <200ms | >10 shifts/sec |
| IV | Integration | <150ms | >5 integrations/sec |
| Full Pipeline | Simple | <2000ms | - |
| Full Pipeline | Medium | <10000ms | - |
| Full Pipeline | Complex | <60000ms | - |

*Note: Targets are approximate and depend on hardware.*

---

## Configuration

### Environment Variables

Each phase executor uses environment variables for configuration. Set these before running benchmarks:

#### Phase I

```bash
export PHASE1_TIMEOUT_MS=15000
export PHASE1_MAX_ASSUMPTIONS=100
export PHASE1_MAX_CONSTRAINTS=1000
export PHASE1_ENABLE_TACIT_MINING=true
export PHASE1_ENABLE_RED_TEAM=true
```

#### Phase II

```bash
export PHASE2_TIMEOUT_MS=20000
export PHASE2_IMECH_THRESHOLD=0.7
export PHASE2_MAX_TARGET_DOMAINS=10
export PHASE2_ENABLE_CONSTRAINT_INVERSION=true
```

#### Phase III

```bash
export PHASE3_ITERATIONS=1000
export PHASE3_UCB1_C=1.414
export PHASE3_CONVERGENCE_THRESHOLD=0.001
export PHASE3_TIMEOUT_MS=30000
export PHASE3_MAX_DEPTH=20
```

#### Phase IV

```bash
export PHASE4_ASSEMBLY_TIMEOUT_MS=25000
export PHASE4_MIN_CONFIDENCE_THRESHOLD=0.6
export PHASE4_MAX_PARADIGM_SHIFTS=100
export PHASE4_VALIDATION_LEVEL=STANDARD
```

### Benchmark-Specific Configuration

Edit the benchmark scripts to change:

- **Iterations**: Number of times each benchmark runs (default: 3-5)
- **Problem sizes**: Test data sizes
- **Timeout values**: Maximum execution time

---

## Troubleshooting

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'phase1_executor'`

**Solution**:
- Verify phase executor files exist in `../rese-phaseX/src/`
- Check that you're running from the `rese-benchmarks/` directory
- Ensure all dependencies are installed

### Timeout Errors

**Error**: `Benchmark timed out`

**Solution**:
- Increase timeout in the orchestrator (`--timeout` argument)
- Reduce problem sizes (iteration counts, data sizes)
- Check for infinite loops in phase executors

### Memory Errors

**Error**: `MemoryError` or out of memory

**Solution**:
- Reduce test data sizes
- Close other applications
- Run benchmarks individually instead of all at once

### Inconsistent Results

**Observation**: High standard deviation across iterations

**Possible Causes**:
1. Background processes using CPU/memory
2. Variable data sizes
3. Non-deterministic algorithms (e.g., random sampling in MCTS)

**Solutions**:
- Close unnecessary applications
- Run multiple iterations (increase `iterations` parameter)
- Use median instead of mean for performance assessment

### No Results Generated

**Observation**: Scripts run but no JSON files in `results/`

**Solutions**:
- Check script output for errors
- Verify write permissions to `results/` directory
- Ensure `results/` directory exists

---

## Advanced Usage

### Custom Benchmarks

To create custom benchmarks:

1. Copy an existing benchmark script as template
2. Modify test data generators
3. Add new benchmark functions following the same pattern
4. Update the main runner to call your benchmarks

### Continuous Integration

Add benchmarks to CI/CD pipeline:

```yaml
# Example GitHub Actions
- name: Run RESE Benchmarks
  run: |
    cd glue/adapters/rese-benchmarks
    python run_all_benchmarks.py --save-baseline

- name: Compare with Baseline
  run: |
    cd glue/adapters/rese-benchmarks
    python run_all_benchmarks.py --compare-baseline
```

### Performance Profiling

For detailed profiling, use Python's built-in profilers:

```bash
# Profile specific phase
python -m cProfile -o profile.stats benchmark_phase1.py

# Visualize with snakeviz
pip install snakeviz
snakeviz profile.stats
```

---

## Contributing

When adding new benchmarks:

1. Follow existing naming conventions
2. Include comprehensive documentation
3. Calculate all statistics (min, max, mean, median, stdev)
4. Measure throughput (operations per second)
5. Test on multiple problem sizes
6. Update this README

---

## License

Part of the RESE (Recursive Epistemic Solvability Engine) project.

---

**Author**: RESE Team
**Created**: 2026-02-04
**Version**: 1.0.0
