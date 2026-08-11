"""
ROMA Decomposition Comparison Tool

Provides utilities to compare ROMA decomposition against other strategies
to determine the best approach for different types of content.
"""

import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from problem_decomposition import (
    ProblemDecomposer,
    DecompositionStrategy,
    DecompositionResult,
)
from roma_config_helper import ROMAConfig, ROMAConfigPresets


@dataclass
class ComparisonMetrics:
    """Metrics from a single decomposition run"""
    strategy: str
    components_count: int
    quality_score: float
    decomposition_time: float
    avg_component_size: float
    complexity_distribution: Dict[str, float] = field(default_factory=dict)
    total_dependencies: int = 0
    avg_dependencies: float = 0.0
    high_complexity_count: int = 0


@dataclass
class ComparisonResult:
    """Result of comparing multiple decomposition strategies"""
    content_summary: str
    metrics: List[ComparisonMetrics] = field(default_factory=list)
    best_by_quality: ComparisonMetrics = None
    best_by_speed: ComparisonMetrics = None
    best_by_coverage: ComparisonMetrics = None
    recommendations: List[str] = field(default_factory=list)


class ROMAComparator:
    """
    Compare ROMA decomposition against other strategies.

    Provides automated benchmarking and recommendations for choosing
    the best decomposition strategy.
    """

    def __init__(self, auto_create_analyzer: bool = False):
        """
        Initialize comparator.

        Args:
            auto_create_analyzer: Whether to auto-create ProblemAnalyzer
        """
        self.decomposer = ProblemDecomposer(auto_create_analyzer=auto_create_analyzer)

    def compare_strategies(
        self,
        content: str,
        strategies: List[DecompositionStrategy] = None,
        max_components: int = 10,
        min_component_size: int = 50,
        runs: int = 1,
        **kwargs
    ) -> ComparisonResult:
        """
        Compare multiple decomposition strategies on the same content.

        Args:
            content: Content to decompose
            strategies: List of strategies to compare (default: all strategies)
            max_components: Max components for each decomposition
            min_component_size: Min component size
            runs: Number of runs to average (for performance consistency)
            **kwargs: Additional parameters passed to decompose_content

        Returns:
            ComparisonResult with metrics and recommendations
        """
        if strategies is None:
            strategies = [
                DecompositionStrategy.ROMA,
                DecompositionStrategy.SEMANTIC,
                DecompositionStrategy.HIERARCHICAL,
                DecompositionStrategy.FUNCTIONAL,
                DecompositionStrategy.COMPLEXITY_BASED,
            ]

        metrics_list = []
        content_summary = content[:100] + "..." if len(content) > 100 else content

        print(f"\nComparing {len(strategies)} strategies on: {content_summary}")
        print("=" * 70)

        for strategy in strategies:
            # Run multiple times for consistency
            times = []
            qualities = []
            component_counts = []

            for run in range(runs):
                result = self.decomposer.decompose_content(
                    content=content,
                    strategy=strategy,
                    max_components=max_components,
                    min_component_size=min_component_size,
                    use_problem_analyzer=False,
                    **kwargs
                )

                times.append(result.metadata.get('decomposition_time', 0))
                qualities.append(result.quality_score)
                component_counts.append(len(result.components))

            # Calculate averages
            avg_time = sum(times) / len(times)
            avg_quality = sum(qualities) / len(qualities)
            avg_components = sum(component_counts) / len(component_counts)

            # Get detailed metrics from last run
            complexity_dist = result.metadata.get('complexity_distribution', {})
            avg_size = result.metadata.get('avg_component_size', 0)

            # Calculate dependency metrics
            total_deps = sum(len(c.dependencies) for c in result.components)
            avg_deps = total_deps / len(result.components) if result.components else 0

            high_complexity = complexity_dist.get('high_complexity_count', 0)

            metric = ComparisonMetrics(
                strategy=strategy.value,
                components_count=int(avg_components),
                quality_score=avg_quality,
                decomposition_time=avg_time,
                avg_component_size=avg_size,
                complexity_distribution=complexity_dist,
                total_dependencies=total_deps,
                avg_dependencies=avg_deps,
                high_complexity_count=high_complexity,
            )
            metrics_list.append(metric)

            print(f"\n{strategy.value}:")
            print(f"  Quality: {avg_quality:.3f}")
            print(f"  Time: {avg_time:.3f}s")
            print(f"  Components: {avg_components:.0f}")

        # Find best strategies
        best_by_quality = max(metrics_list, key=lambda m: m.quality_score)
        best_by_speed = min(metrics_list, key=lambda m: m.decomposition_time)
        best_by_coverage = max(metrics_list, key=lambda m: m.components_count)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics_list, content)

        return ComparisonResult(
            content_summary=content_summary,
            metrics=metrics_list,
            best_by_quality=best_by_quality,
            best_by_speed=best_by_speed,
            best_by_coverage=best_by_coverage,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        metrics: List[ComparisonMetrics],
        content: str
    ) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []

        # Find best quality
        best_quality = max(metrics, key=lambda m: m.quality_score)
        if best_quality.quality_score > 0.7:
            recommendations.append(
                f"[OK] Use {best_quality.strategy} for best quality (score: {best_quality.quality_score:.3f})"
            )

        # Find fastest
        fastest = min(metrics, key=lambda m: m.decomposition_time)
        if fastest.decomposition_time < 1.0:
            recommendations.append(
                f"[OK] Use {fastest.strategy} for fastest decomposition ({fastest.decomposition_time:.3f}s)"
            )

        # Check ROMA specifically
        roma_metrics = [m for m in metrics if m.strategy == "roma"]
        if roma_metrics:
            roma = roma_metrics[0]
            if roma.quality_score > 0.6:
                recommendations.append(
                    f"[OK] ROMA provides good quality decomposition (score: {roma.quality_score:.3f})"
                )
            else:
                recommendations.append(
                    f"[WARN] ROMA quality is lower than expected. Consider using {best_quality.strategy}"
                )

        # Content-based recommendations
        content_lower = content.lower()
        if "implement" in content_lower or "code" in content_lower:
            if any(m.strategy == "functional" for m in metrics):
                recommendations.append(
                    "💡 For implementation tasks, consider FUNCTIONAL decomposition"
                )

        if len(content) > 2000:
            recommendations.append(
                "💡 For long content, consider increasing max_components or using COMPLEXITY_BASED strategy"
            )

        return recommendations

    def benchmark_roma_configs(
        self,
        content: str,
        configs: List[ROMAConfig] = None,
        max_components: int = 10,
    ) -> Dict[str, ComparisonMetrics]:
        """
        Benchmark different ROMA configurations.

        Args:
            content: Content to decompose
            configs: List of ROMA configs to test (default: presets)
            max_components: Max components

        Returns:
            Dict mapping config names to metrics
        """
        if configs is None:
            configs = [
                ROMAConfigPresets.fast(),
                ROMAConfigPresets.balanced(),
                ROMAConfigPresets.thorough(),
            ]

        results = {}

        print(f"\nBenchmarking {len(configs)} ROMA configurations")
        print("=" * 70)

        for i, config in enumerate(configs, 1):
            config_name = f"Config {i}"
            kwargs = config.to_kwargs()

            start = time.time()
            result = self.decomposer.decompose_content(
                content=content,
                strategy=DecompositionStrategy.ROMA,
                max_components=max_components,
                use_problem_analyzer=False,
                **kwargs
            )
            elapsed = time.time() - start

            metric = ComparisonMetrics(
                strategy=config_name,
                components_count=len(result.components),
                quality_score=result.quality_score,
                decomposition_time=elapsed,
                avg_component_size=result.metadata.get('avg_component_size', 0),
                complexity_distribution=result.metadata.get('complexity_distribution', {}),
            )

            results[config_name] = metric

            print(f"\n{config_name}:")
            print(f"  Max Depth: {kwargs.get('roma_max_depth', 'default')}")
            print(f"  Max Nodes: {kwargs.get('roma_max_nodes', 'default')}")
            print(f"  Quality: {metric.quality_score:.3f}")
            print(f"  Time: {metric.decomposition_time:.3f}s")
            print(f"  Components: {metric.components_count}")

        return results

    def find_optimal_config(
        self,
        content: str,
        objective: str = "balanced",
        max_components: int = 10,
    ) -> Tuple[ROMAConfig, ComparisonMetrics]:
        """
        Find optimal ROMA configuration for given content and objective.

        Args:
            content: Content to decompose
            objective: Optimization objective ("quality", "speed", "balanced")
            max_components: Max components

        Returns:
            Tuple of (best_config, metrics)
        """
        configs_to_try = [
            ROMAConfigPresets.fast(),
            ROMAConfigPresets.balanced(),
            ROMAConfigPresets.thorough(),
            ROMAConfigPresets.hierarchical(),
        ]

        results = self.benchmark_roma_configs(content, configs_to_try, max_components)

        # Select based on objective
        if objective == "quality":
            best_name = max(results.items(), key=lambda x: x[1].quality_score)[0]
        elif objective == "speed":
            best_name = min(results.items(), key=lambda x: x[1].decomposition_time)[0]
        else:  # balanced
            # Balance quality and speed
            best_name = max(
                results.items(),
                key=lambda x: x[1].quality_score / (x[1].decomposition_time + 0.1)
            )[0]

        best_config = configs_to_try[list(results.keys()).index(best_name)]
        best_metrics = results[best_name]

        return best_config, best_metrics


def print_comparison_table(result: ComparisonResult):
    """Print a formatted comparison table"""
    print("\n" + "=" * 100)
    print(f"DECOMPOSITION COMPARISON: {result.content_summary}")
    print("=" * 100)

    # Header
    print(f"\n{'Strategy':<20} {'Quality':<10} {'Time (s)':<10} {'Components':<12} "
          f"{'Avg Size':<10} {'Avg Deps':<10}")
    print("-" * 100)

    # Metrics
    for metric in result.metrics:
        print(f"{metric.strategy:<20} {metric.quality_score:<10.3f} "
              f"{metric.decomposition_time:<10.3f} {metric.components_count:<12} "
              f"{metric.avg_component_size:<10.0f} {metric.avg_dependencies:<10.2f}")

    # Best performers
    print("\n" + "-" * 100)
    print(f"Best Quality: {result.best_by_quality.strategy} "
          f"(score: {result.best_by_quality.quality_score:.3f})")
    print(f"Fastest: {result.best_by_speed.strategy} "
          f"(time: {result.best_by_speed.decomposition_time:.3f}s)")
    print(f"Most Components: {result.best_by_coverage.strategy} "
          f"(count: {result.best_by_coverage.components_count})")

    # Recommendations
    if result.recommendations:
        print("\n" + "-" * 100)
        print("Recommendations:")
        for rec in result.recommendations:
            print(f"  {rec}")

    print("=" * 100)


if __name__ == "__main__":
    # Example usage
    comparator = ROMAComparator(auto_create_analyzer=False)

    # Test content
    content = """
    Design and implement a microservices-based e-commerce platform with:
    - Product catalog service with search and filtering
    - Order processing service with payment integration
    - User management service with authentication
    - Inventory management service with real-time updates
    - Recommendation engine using machine learning
    - API gateway with rate limiting and caching
    """

    # Compare strategies
    result = comparator.compare_strategies(
        content=content,
        max_components=12,
        runs=1,
    )

    # Print results
    print_comparison_table(result)

    # Find optimal config
    print("\n" + "=" * 70)
    print("Finding Optimal ROMA Configuration")
    print("=" * 70)

    optimal_config, optimal_metrics = comparator.find_optimal_config(
        content=content,
        objective="balanced",
        max_components=12,
    )

    print(f"\nOptimal Configuration:")
    print(f"  Max Depth: {optimal_config.max_depth}")
    print(f"  Max Nodes: {optimal_config.max_nodes}")
    print(f"  Use Fractal: {optimal_config.use_fractal}")
    print(f"\nPerformance:")
    print(f"  Quality: {optimal_metrics.quality_score:.3f}")
    print(f"  Time: {optimal_metrics.decomposition_time:.3f}s")
    print(f"  Components: {optimal_metrics.components_count}")
