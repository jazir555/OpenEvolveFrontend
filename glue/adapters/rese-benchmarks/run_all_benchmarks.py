#!/usr/bin/env python3
"""
RESE Benchmark Orchestrator

Runs all RESE benchmark suites and generates comprehensive reports:
- Phase I: Epistemic Audit benchmarks
- Phase II: Isomorphic Mapping benchmarks
- Phase III: MCTS Search benchmarks
- Phase IV: Architecture Assembly benchmarks
- Full Pipeline: End-to-end benchmarks

Generates:
- Combined JSON report with all results
- Markdown summary report
- Comparison against baseline (if exists)

Author: RESE Team
Created: 2026-02-04
"""

import sys
import os
import json
import subprocess
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path

# Benchmark script paths
BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "results"
BASELINE_FILE = RESULTS_DIR / "baseline.json"

BENCHMARK_SCRIPTS = {
    "phase1": BENCHMARK_DIR / "benchmark_phase1.py",
    "phase2": BENCHMARK_DIR / "benchmark_phase2.py",
    "phase3": BENCHMARK_DIR / "benchmark_phase3.py",
    "phase4": BENCHMARK_DIR / "benchmark_phase4.py",
    "full_pipeline": BENCHMARK_DIR / "benchmark_full_pipeline.py",
}


# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================

def run_benchmark_script(script_name: str) -> Optional[Dict[str, Any]]:
    """Run a single benchmark script.

    Args:
        script_name: Name of the benchmark (e.g., "phase1")

    Returns:
        Benchmark results dict or None if failed
    """
    script_path = BENCHMARK_SCRIPTS.get(script_name)
    if not script_path or not script_path.exists():
        print(f"Warning: Benchmark script not found: {script_name}")
        return None

    print(f"\n{'=' * 70}")
    print(f"Running {script_name.upper()} benchmarks...")
    print('=' * 70)

    try:
        # Run script and capture output
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BENCHMARK_DIR),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"Error running {script_name}:")
            print(result.stderr)
            return None

        print(result.stdout)

        # Find the most recent result file
        result_files = sorted(
            RESULTS_DIR.glob(f"{script_name}_benchmark_*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )

        if result_files:
            with open(result_files[0], 'r') as f:
                return json.load(f)

    except subprocess.TimeoutExpired:
        print(f"Error: {script_name} benchmark timed out")
    except Exception as e:
        print(f"Error running {script_name}: {e}")

    return None


def run_all_benchmarks() -> Dict[str, Any]:
    """Run all benchmark suites.

    Returns:
        Combined results from all benchmarks
    """
    print("=" * 70)
    print("RESE Benchmark Orchestrator")
    print("=" * 70)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")

    results = {
        "orchestration": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": sys.platform,
        },
        "benchmarks": {}
    }

    # Run each benchmark suite
    for benchmark_name in BENCHMARK_SCRIPTS.keys():
        benchmark_results = run_benchmark_script(benchmark_name)

        if benchmark_results:
            results["benchmarks"][benchmark_name] = benchmark_results
        else:
            print(f"Skipping {benchmark_name} due to errors")

    return results


# ============================================================================
# BASELINE COMPARISON
# ============================================================================

def load_baseline() -> Optional[Dict[str, Any]]:
    """Load baseline results for comparison.

    Returns:
        Baseline results or None if not found
    """
    if not BASELINE_FILE.exists():
        print(f"\nBaseline not found: {BASELINE_FILE}")
        print("Run with --save-baseline to create one")
        return None

    with open(BASELINE_FILE, 'r') as f:
        return json.load(f)


def compare_with_baseline(
    current_results: Dict[str, Any],
    baseline: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare current results with baseline.

    Args:
        current_results: Current benchmark results
        baseline: Baseline benchmark results

    Returns:
        Comparison report
    """
    print("\n" + "=" * 70)
    print("Baseline Comparison")
    print("=" * 70)

    comparison = {
        "baseline_timestamp": baseline.get("orchestration", {}).get("timestamp"),
        "current_timestamp": current_results.get("orchestration", {}).get("timestamp"),
        "comparisons": {}
    }

    current_benchmarks = current_results.get("benchmarks", {})
    baseline_benchmarks = baseline.get("benchmarks", {})

    for phase, current_data in current_benchmarks.items():
        if phase not in baseline_benchmarks:
            continue

        baseline_data = baseline_benchmarks[phase]
        phase_comparison = {}

        # Compare benchmark metrics
        current_benchmarks_list = current_data.get("benchmarks", [])
        baseline_benchmarks_list = baseline_data.get("benchmarks", [])

        for i, (curr, base) in enumerate(zip(current_benchmarks_list, baseline_benchmarks_list)):
            benchmark_name = curr.get("benchmark", f"benchmark_{i}")

            # Compare timing metrics
            curr_time = curr.get("timings_ms", {}).get("mean", 0)
            base_time = base.get("timings_ms", {}).get("mean", 0)

            if base_time > 0:
                time_change_pct = ((curr_time - base_time) / base_time) * 100
            else:
                time_change_pct = 0.0

            # Compare throughput
            curr_throughput = curr.get("throughput", {})
            base_throughput = base.get("throughput", {})

            throughput_comparison = {}
            for key in curr_throughput:
                curr_val = curr_throughput[key]
                base_val = base_throughput.get(key, 0)

                if base_val > 0:
                    throughput_comparison[key] = {
                        "current": curr_val,
                        "baseline": base_val,
                        "change_pct": round(((curr_val - base_val) / base_val) * 100, 2),
                    }

            phase_comparison[benchmark_name] = {
                "time_change_pct": round(time_change_pct, 2),
                "current_time_ms": curr_time,
                "baseline_time_ms": base_time,
                "throughput_comparison": throughput_comparison,
            }

            # Print comparison
            status = "[OK]" if time_change_pct < 0 else ("[FAIL]" if time_change_pct > 10 else "=")
            print(f"\n{phase.upper()} - {benchmark_name}:")
            print(f"  Time: {curr_time:.2f}ms vs {base_time:.2f}ms ({time_change_pct:+.1f}%) {status}")

        comparison["comparisons"][phase] = phase_comparison

    return comparison


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_markdown_report(
    results: Dict[str, Any],
    comparison: Optional[Dict[str, Any]] = None
) -> str:
    """Generate Markdown summary report.

    Args:
        results: Combined benchmark results
        comparison: Optional baseline comparison

    Returns:
        Markdown report string
    """
    lines = []
    lines.append("# RESE Performance Benchmark Report")
    lines.append("")
    lines.append(f"**Generated:** {results['orchestration']['timestamp']}")
    lines.append(f"**Python Version:** {results['orchestration']['python_version']}")
    lines.append(f"**Platform:** {results['orchestration']['platform']}")
    lines.append("")

    # Table of Contents
    lines.append("## Table of Contents")
    lines.append("")
    for phase in BENCHMARK_SCRIPTS.keys():
        if phase in results.get("benchmarks", {}):
            lines.append(f"- [{phase.upper()}](#{phase}-benchmarks)")
    if "full_pipeline" in results.get("benchmarks", {}):
        lines.append("- [Full Pipeline](#full_pipeline-benchmarks)")
    lines.append("")

    # Per-phase summaries
    for phase, phase_results in results.get("benchmarks", {}).items():
        lines.append(f"## {phase.upper()} Benchmarks")
        lines.append("")

        benchmarks = phase_results.get("benchmarks", [])

        for benchmark in benchmarks:
            benchmark_name = benchmark.get("benchmark", "unknown")
            lines.append(f"### {benchmark_name.replace('_', ' ').title()}")
            lines.append("")

            # Timing metrics
            timings = benchmark.get("timings_ms", benchmark.get("timings_us", {}))
            if timings:
                lines.append("**Timing Metrics:**")
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                for metric, value in timings.items():
                    unit = "ms" if "timings_ms" in benchmark else "μs"
                    lines.append(f"| {metric.title()} | {value} {unit} |")
                lines.append("")

            # Throughput
            throughput = benchmark.get("throughput", {})
            if throughput:
                lines.append("**Throughput:**")
                lines.append("")
                for metric, value in throughput.items():
                    lines.append(f"- {metric.replace('_', ' ').title()}: {value}")
                lines.append("")

    # Baseline comparison
    if comparison:
        lines.append("## Baseline Comparison")
        lines.append("")
        lines.append(f"**Baseline Timestamp:** {comparison.get('baseline_timestamp', 'N/A')}")
        lines.append("")

        for phase, phase_comparison in comparison.get("comparisons", {}).items():
            lines.append(f"### {phase.upper()}")
            lines.append("")
            lines.append("| Benchmark | Time Change | Current | Baseline |")
            lines.append("|-----------|-------------|---------|----------|")

            for benchmark_name, metrics in phase_comparison.items():
                time_change = metrics.get("time_change_pct", 0)
                current = metrics.get("current_time_ms", 0)
                baseline = metrics.get("baseline_time_ms", 0)

                status = "[OK]" if time_change < 0 else ("[FAIL]" if time_change > 10 else "=")
                lines.append(f"| {benchmark_name} | {time_change:+.1f}% {status} | {current:.2f}ms | {baseline:.2f}ms |")

            lines.append("")

    lines.append("---")
    lines.append("*Report generated by RESE Benchmark Orchestrator*")

    return "\n".join(lines)


def save_combined_report(
    results: Dict[str, Any],
    comparison: Optional[Dict[str, Any]] = None
):
    """Save combined JSON report and Markdown summary.

    Args:
        results: Combined benchmark results
        comparison: Optional baseline comparison
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save JSON report
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_file = RESULTS_DIR / f"combined_benchmark_{timestamp}.json"

    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nCombined results saved to: {json_file}")

    # Save Markdown report
    markdown = generate_markdown_report(results, comparison)
    md_file = RESULTS_DIR / f"benchmark_report_{timestamp}.md"

    with open(md_file, 'w') as f:
        f.write(markdown)

    print(f"Markdown report saved to: {md_file}")

    # Save comparison if available
    if comparison:
        comparison_file = RESULTS_DIR / f"baseline_comparison_{timestamp}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Baseline comparison saved to: {comparison_file}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    import argparse
    global BENCHMARK_SCRIPTS

    parser = argparse.ArgumentParser(
        description="RESE Benchmark Orchestrator"
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=list(BENCHMARK_SCRIPTS.keys()),
        help="Specific phases to benchmark (default: all)"
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare results against baseline"
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save results as new baseline"
    )

    args = parser.parse_args()

    # Filter benchmarks if specific phases requested
    if args.phases:
        BENCHMARK_SCRIPTS = {
            k: v for k, v in BENCHMARK_SCRIPTS.items()
            if k in args.phases
        }

    # Run all benchmarks
    results = run_all_benchmarks()

    if not results.get("benchmarks"):
        print("\nNo benchmark results collected. Exiting.")
        return 1

    # Compare with baseline if requested
    comparison = None
    if args.compare_baseline:
        baseline = load_baseline()
        if baseline:
            comparison = compare_with_baseline(results, baseline)

    # Save combined report
    save_combined_report(results, comparison)

    # Save as baseline if requested
    if args.save_baseline:
        with open(BASELINE_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved as baseline to: {BASELINE_FILE}")

    print("\n" + "=" * 70)
    print("Benchmark Orchestrator Complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
