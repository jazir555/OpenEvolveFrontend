"""
Benchmark Visualization Module

Generates charts and visualizations for benchmark results.

Author: OpenEvolve Framework
Date: 2025-01-07
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class BenchmarkVisualizer:
    """
    Generate visualizations from benchmark results.
    """

    def __init__(self, results_dir: str = "benchmark_results"):
        """
        Initialize visualizer.

        Args:
            results_dir: Directory containing benchmark results
        """
        self.results_dir = Path(results_dir)
        self.results = {}
        self._load_results()

    def _load_results(self):
        """Load benchmark results from JSON files."""
        json_files = list(self.results_dir.glob("benchmark_metrics_*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    self.results.update(data.get("benchmarks", {}))
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")

    def plot_throughput_comparison(
        self,
        save_path: Optional[str] = None
    ):
        """
        Plot throughput comparison across benchmarks.

        Args:
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        benchmark_names = []
        throughput_values = []

        for name, result in self.results.items():
            if result.get("success") and result.get("metrics"):
                metrics = result["metrics"]

                # Extract throughput metrics
                for key, value in metrics.items():
                    if "throughput" in key.lower() or "per_second" in key.lower():
                        if isinstance(value, (int, float)):
                            benchmark_names.append(f"{name}\n({key})")
                            throughput_values.append(value)

        if benchmark_names:
            # Create bar plot
            bars = ax.barh(benchmark_names, throughput_values, color='steelblue')

            # Add value labels
            for bar, value in zip(bars, throughput_values):
                ax.text(
                    value + max(throughput_values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{value:.2f}',
                    va='center',
                    fontsize=9
                )

            ax.set_xlabel('Throughput (Operations/Second)', fontsize=12, fontweight='bold')
            ax.set_title('Knowledge Graph Throughput Comparison', fontsize=14, fontweight='bold')
            ax.set_xscale('log')  # Log scale for better visualization

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Saved throughput chart to {save_path}")

            plt.show()

    def plot_latency_distribution(
        self,
        save_path: Optional[str] = None
    ):
        """
        Plot latency distribution from retrieval benchmarks.

        Args:
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Extract latency data
        latency_data = []
        labels = []

        for name, result in self.results.items():
            if "retrieval" in name.lower() and result.get("success"):
                metrics = result.get("metrics", {})
                latencies = metrics.get("latencies_ms", {})

                for query_type, latencies_list in latencies.items():
                    if isinstance(latencies_list, list):
                        latency_data.append(latencies_list)
                        labels.append(f"{query_type}")

        if latency_data:
            # Create box plot
            bp = ax.boxplot(latency_data, labels=labels, patch_artist=True)

            # Color the boxes
            colors = ['lightblue', 'lightgreen', 'salmon', 'plum']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)

            ax.set_ylabel('Latency (milliseconds)', fontsize=12, fontweight='bold')
            ax.set_title('Knowledge Retrieval Latency Distribution', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Saved latency chart to {save_path}")

            plt.show()

    def plot_memory_usage(
        self,
        save_path: Optional[str] = None
    ):
        """
        Plot memory usage across benchmarks.

        Args:
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        benchmark_names = []
        memory_values = []
        memory_units = []

        for name, result in self.results.items():
            if result.get("success") and result.get("metrics"):
                metrics = result["metrics"]

                # Extract memory metrics
                for key, value in metrics.items():
                    if "memory" in key.lower() and isinstance(value, (int, float)):
                        benchmark_names.append(name)
                        memory_values.append(value)

                        # Determine unit
                        if "gb" in key.lower() or value > 1024:
                            memory_units.append("GB")
                            if value > 1024:  # Convert MB to GB
                                memory_values[-1] = value / 1024
                        else:
                            memory_units.append("MB")

        if benchmark_names:
            # Create bar plot
            colors = ['coral' if unit == 'GB' else 'skyblue'
                     for unit in memory_units]

            bars = ax.bar(range(len(benchmark_names)), memory_values, color=colors)

            # Add value labels
            for bar, value, unit in zip(bars, memory_values, memory_units):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(memory_values) * 0.01,
                    f'{value:.2f} {unit}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

            ax.set_xticks(range(len(benchmark_names)))
            ax.set_xticklabels(benchmark_names, rotation=45, ha='right')
            ax.set_ylabel('Memory Usage', fontsize=12, fontweight='bold')
            ax.set_title('Memory Usage Across Benchmarks', fontsize=14, fontweight='bold')

            # Add legend
            gb_patch = mpatches.Patch(color='coral', label='Memory (GB)')
            mb_patch = mpatches.Patch(color='skyblue', label='Memory (MB)')
            ax.legend(handles=[gb_patch, mb_patch])

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Saved memory chart to {save_path}")

            plt.show()

    def plot_scalability(
        self,
        save_path: Optional[str] = None
    ):
        """
        Plot scalability analysis (performance vs dataset size).

        Args:
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Graph algorithms scalability
        graph_result = self.results.get("graph_algorithms")
        if graph_result and graph_result.get("success"):
            metrics = graph_result.get("metrics", {})

            sizes = []
            durations = []
            memories = []

            for size, data in metrics.items():
                try:
                    sizes.append(int(size))
                    durations.append(data.get("duration_seconds", 0))
                    memories.append(data.get("memory_mb", 0))
                except (ValueError, TypeError):
                    continue

            if sizes:
                # Duration plot
                ax1.plot(sizes, durations, marker='o', linewidth=2, markersize=8, color='steelblue')
                ax1.set_xlabel('Graph Size (nodes)', fontsize=11, fontweight='bold')
                ax1.set_ylabel('Duration (seconds)', fontsize=11, fontweight='bold')
                ax1.set_title('Processing Time vs Graph Size', fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.3)

                # Memory plot
                ax2.plot(sizes, memories, marker='s', linewidth=2, markersize=8, color='coral')
                ax2.set_xlabel('Graph Size (nodes)', fontsize=11, fontweight='bold')
                ax2.set_ylabel('Memory Usage (MB)', fontsize=11, fontweight='bold')
                ax2.set_title('Memory Usage vs Graph Size', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved scalability chart to {save_path}")

        plt.show()

    def plot_deduplication_accuracy(
        self,
        save_path: Optional[str] = None
    ):
        """
        Plot deduplication accuracy metrics.

        Args:
            save_path: Optional path to save figure
        """
        dedup_result = self.results.get("deduplication")

        if not dedup_result or not dedup_result.get("success"):
            print("No deduplication results found")
            return

        metrics = dedup_result.get("metrics", {})

        # Extract metrics
    accuracy = metrics.get("accuracy", 0) * 100
    precision = metrics.get("precision", 0) * 100
    recall = metrics.get("recall", 0) * 100
    f1_score = metrics.get("f1_score", 0) * 100

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_values = [accuracy, precision, recall, f1_score]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7)

    # Add value labels
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 1,
            f'{value:.1f}%',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Deduplication Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved deduplication accuracy chart to {save_path}")

    plt.show()

    def plot_concurrent_performance(
        self,
        save_path: Optional[str] = None
    ):
        """
        Plot concurrent operations performance.

        Args:
            save_path: Optional path to save figure
        """
        concurrent_result = self.results.get("concurrent_operations")

        if not concurrent_result or not concurrent_result.get("success"):
            print("No concurrent operations results found")
            return

        metrics = concurrent_result.get("metrics", {})

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Throughput plot
        num_concurrent = metrics.get("num_concurrent", 0)
        throughput = metrics.get("throughput_ops_per_sec", 0)

        ax1.bar(['Throughput'], [throughput], color='steelblue', alpha=0.7)
        ax1.set_ylabel('Operations/Second', fontsize=11, fontweight='bold')
        ax1.set_title(f'Concurrent Throughput\n({num_concurrent} concurrent clients)',
                     fontsize=12, fontweight='bold')
        ax1.text(0, throughput, f'{throughput:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Error rate plot
        error_rate = metrics.get("error_rate", 0) * 100
    total_ops = metrics.get("total_operations", 0)
    errors = metrics.get("errors", 0)

    ax2.bar(['Error Rate'], [error_rate], color='coral' if error_rate > 5 else 'green', alpha=0.7)
    ax2.set_ylabel('Error Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Error Rate Under Load\n({total_ops} operations, {errors} errors)',
                 fontsize=12, fontweight='bold')
    ax2.text(0, error_rate, f'{error_rate:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(error_rate * 1.2, 10))
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved concurrent performance chart to {save_path}")

    plt.show()

    def generate_all_charts(
        self,
        output_dir: Optional[str] = None
    ):
        """
        Generate all visualization charts.

        Args:
            output_dir: Directory to save charts (default: results_dir)
        """
        output_dir = Path(output_dir or self.results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*60)
        print("GENERATING VISUALIZATION CHARTS")
        print("="*60)

        try:
            self.plot_throughput_comparison(
                save_path=str(output_dir / "throughput_comparison.png")
            )
        except Exception as e:
            print(f"Warning: Could not generate throughput chart: {e}")

        try:
            self.plot_latency_distribution(
                save_path=str(output_dir / "latency_distribution.png")
            )
        except Exception as e:
            print(f"Warning: Could not generate latency chart: {e}")

        try:
            self.plot_memory_usage(
                save_path=str(output_dir / "memory_usage.png")
            )
        except Exception as e:
            print(f"Warning: Could not generate memory chart: {e}")

        try:
            self.plot_scalability(
                save_path=str(output_dir / "scalability_analysis.png")
            )
        except Exception as e:
            print(f"Warning: Could not generate scalability chart: {e}")

        try:
            self.plot_deduplication_accuracy(
                save_path=str(output_dir / "deduplication_accuracy.png")
            )
        except Exception as e:
            print(f"Warning: Could not generate deduplication chart: {e}")

        try:
            self.plot_concurrent_performance(
                save_path=str(output_dir / "concurrent_performance.png")
            )
        except Exception as e:
            print(f"Warning: Could not generate concurrent performance chart: {e}")

        print(f"\n✓ All charts saved to {output_dir}")


def main():
    """Example usage."""
    import sys

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "benchmark_results"

    visualizer = BenchmarkVisualizer(results_dir)
    visualizer.generate_all_charts()


if __name__ == "__main__":
    main()
