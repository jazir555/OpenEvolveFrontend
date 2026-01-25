"""
Knowledge Graph Benchmark Runner

This script executes the comprehensive benchmark suite for the knowledge graph system.
It can be configured via command-line arguments or a configuration file.

Usage:
    python run_benchmarks.py --all
    python run_benchmarks.py --benchmark knowledge_addition
    python run_benchmarks.py --config benchmark_config.yaml

Author: OpenEvolve Framework
Date: 2025-01-07
"""

import asyncio
import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge_engine.engine import KnowledgeEngine
from tests.benchmarks.kg_performance_benchmarks import KnowledgeGraphPerformanceBenchmarks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Benchmark runner with configuration and execution management.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: str = "benchmark_results"
    ):
        """
        Initialize benchmark runner.

        Args:
            config_path: Optional path to configuration file
            output_dir: Directory to save benchmark results
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = self._load_config()
        self.engine = None
        self.benchmarks = None

    def _load_config(self) -> dict:
        """Load configuration from file or use defaults."""
        if self.config_path and Path(self.config_path).exists():
            try:
                import yaml
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")

        # Default configuration
        return {
            "knowledge_addition": {
                "num_artifacts": [100, 500, 1000],
                "batch_sizes": [1, 10, 50]
            },
            "knowledge_retrieval": {
                "num_queries": [10, 50, 100],
                "query_types": ["keyword", "graph"]
            },
            "deduplication": {
                "num_entities": [100, 500, 1000],
                "duplicate_rates": [0.1, 0.3, 0.5]
            },
            "graph_algorithms": {
                "graph_sizes": [100, 500, 1000, 5000]
            },
            "concurrent_operations": {
                "num_concurrent": [5, 10, 20],
                "operations_per_client": [10, 50, 100]
            },
            "end_to_end_workflows": {
                "scenarios": [
                    "entity_relationship_workflow",
                    "batch_processing_workflow",
                    "query_workflow"
                ]
            },
            "output": {
                "report_format": "markdown",
                "include_charts": False,
                "save_raw_data": True
            }
        }

    async def initialize(self):
        """Initialize knowledge engine and benchmarks."""
        logger.info("Initializing Knowledge Engine...")

        try:
            self.engine = KnowledgeEngine()
            self.benchmarks = KnowledgeGraphPerformanceBenchmarks(self.engine)
            logger.info("✓ Initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise

    async def run_all_benchmarks(self):
        """Run all benchmarks with configured parameters."""
        logger.info("\n" + "="*60)
        logger.info("RUNNING ALL BENCHMARKS")
        logger.info("="*60)

        timestamp = asyncio.get_event_loop().time()

        # Knowledge Addition Benchmarks
        logger.info("\n▶ Knowledge Addition Benchmarks")
        for num_artifacts in self.config["knowledge_addition"]["num_artifacts"]:
            for batch_size in self.config["knowledge_addition"]["batch_sizes"]:
                await self.benchmarks.benchmark_knowledge_addition(
                    num_artifacts=num_artifacts,
                    batch_size=batch_size
                )

        # Knowledge Retrieval Benchmarks
        logger.info("\n▶ Knowledge Retrieval Benchmarks")
        for num_queries in self.config["knowledge_retrieval"]["num_queries"]:
            await self.benchmarks.benchmark_knowledge_retrieval(
                num_queries=num_queries,
                query_types=self.config["knowledge_retrieval"]["query_types"]
            )

        # Deduplication Benchmarks
        logger.info("\n▶ Deduplication Benchmarks")
        for num_entities in self.config["deduplication"]["num_entities"]:
            for dup_rate in self.config["deduplication"]["duplicate_rates"]:
                await self.benchmarks.benchmark_deduplication(
                    num_entities=num_entities,
                    duplicate_rate=dup_rate
                )

        # Graph Algorithm Benchmarks
        logger.info("\n▶ Graph Algorithm Benchmarks")
        await self.benchmarks.benchmark_graph_algorithms(
            graph_sizes=self.config["graph_algorithms"]["graph_sizes"]
        )

        # Concurrent Operations Benchmarks
        logger.info("\n▶ Concurrent Operations Benchmarks")
        for num_concurrent in self.config["concurrent_operations"]["num_concurrent"]:
            for ops_per_client in self.config["concurrent_operations"]["operations_per_client"]:
                await self.benchmarks.benchmark_concurrent_operations(
                    num_concurrent=num_concurrent,
                    operations_per_client=ops_per_client
                )

        # End-to-End Workflow Benchmarks
        logger.info("\n▶ End-to-End Workflow Benchmarks")
        await self.benchmarks.benchmark_end_to_end_workflows(
            scenarios=self.config["end_to_end_workflows"]["scenarios"]
        )

        duration = asyncio.get_event_loop().time() - timestamp
        logger.info(f"\n✓ All benchmarks completed in {duration:.2f}s")

    async def run_specific_benchmark(
        self,
        benchmark_name: str,
        **kwargs
    ):
        """
        Run a specific benchmark.

        Args:
            benchmark_name: Name of benchmark to run
            **kwargs: Additional parameters for the benchmark
        """
        logger.info(f"\n▶ Running benchmark: {benchmark_name}")

        benchmark_map = {
            "knowledge_addition": self.benchmarks.benchmark_knowledge_addition,
            "knowledge_retrieval": self.benchmarks.benchmark_knowledge_retrieval,
            "deduplication": self.benchmarks.benchmark_deduplication,
            "graph_algorithms": self.benchmarks.benchmark_graph_algorithms,
            "concurrent_operations": self.benchmarks.benchmark_concurrent_operations,
            "end_to_end_workflows": self.benchmarks.benchmark_end_to_end_workflows
        }

        if benchmark_name not in benchmark_map:
            logger.error(f"Unknown benchmark: {benchmark_name}")
            logger.info(f"Available benchmarks: {list(benchmark_map.keys())}")
            return

        try:
            result = await benchmark_map[benchmark_name](**kwargs)
            logger.info(f"✓ Benchmark '{benchmark_name}' complete")

            if result.success:
                logger.info(f"  Metrics: {json.dumps(result.metrics, indent=2, default=str)}")
            else:
                logger.error(f"  Error: {result.error}")

        except Exception as e:
            logger.error(f"Benchmark '{benchmark_name}' failed: {e}")

    async def run_quick_benchmarks(self):
        """Run a quick subset of benchmarks for rapid testing."""
        logger.info("\n" + "="*60)
        logger.info("RUNNING QUICK BENCHMARKS")
        logger.info("="*60)

        # Smaller dataset sizes for quick testing
        await self.benchmarks.benchmark_knowledge_addition(
            num_artifacts=100,
            batch_size=10
        )

        await self.benchmarks.benchmark_knowledge_retrieval(
            num_queries=20
        )

        await self.benchmarks.benchmark_deduplication(
            num_entities=100,
            duplicate_rate=0.3
        )

        await self.benchmarks.benchmark_graph_algorithms(
            graph_sizes=[100, 500]
        )

        await self.benchmarks.benchmark_concurrent_operations(
            num_concurrent=5,
            operations_per_client=10
        )

        await self.benchmarks.benchmark_end_to_end_workflows()

        logger.info("\n✓ Quick benchmarks complete")

    def generate_reports(self):
        """Generate benchmark reports."""
        logger.info("\n" + "="*60)
        logger.info("GENERATING REPORTS")
        logger.info("="*60)

        timestamp_str = asyncio.get_event_loop().time() if hasattr(asyncio, 'get_event_loop') else "now"
        report_path = self.output_dir / f"benchmark_report_{timestamp_str}.md"

        self.benchmarks.generate_report(
            output_path=str(report_path),
            include_raw_data=self.config["output"]["save_raw_data"]
        )

        # Save metrics
        metrics_path = self.output_dir / f"benchmark_metrics_{timestamp_str}.json"
        self.benchmarks.save_metrics(output_path=str(metrics_path))

        logger.info(f"\n✓ Reports saved to {self.output_dir}")

    async def cleanup(self):
        """Cleanup resources."""
        if self.engine:
            try:
                await self.engine.cleanup_kggen_pipeline()
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OpenEvolve Knowledge Graph Benchmark Runner"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark subset"
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        choices=[
            "knowledge_addition",
            "knowledge_retrieval",
            "deduplication",
            "graph_algorithms",
            "concurrent_operations",
            "end_to_end_workflows"
        ],
        help="Run specific benchmark"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results"
    )

    parser.add_argument(
        "--num-artifacts",
        type=int,
        help="Number of artifacts for addition benchmark"
    )

    parser.add_argument(
        "--num-queries",
        type=int,
        help="Number of queries for retrieval benchmark"
    )

    args = parser.parse_args()

    # Print banner
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "KNOWLEDGE GRAPH BENCHMARKS" + " "*16 + "║")
    print("╚" + "="*58 + "╝")

    # Initialize runner
    runner = BenchmarkRunner(
        config_path=args.config,
        output_dir=args.output_dir
    )

    try:
        await runner.initialize()

        # Run benchmarks based on arguments
        if args.all:
            await runner.run_all_benchmarks()
        elif args.quick:
            await runner.run_quick_benchmarks()
        elif args.benchmark:
            # Build kwargs from command-line args
            kwargs = {}
            if args.num_artifacts:
                kwargs["num_artifacts"] = args.num_artifacts
            if args.num_queries:
                kwargs["num_queries"] = args.num_queries

            await runner.run_specific_benchmark(args.benchmark, **kwargs)
        else:
            # Default to quick benchmarks
            logger.info("No benchmark specified, running quick benchmarks...")
            await runner.run_quick_benchmarks()

        # Generate reports
        runner.generate_reports()

        print("\n" + "="*60)
        print("ALL BENCHMARKS COMPLETE!")
        print("="*60)

    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        raise
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nBenchmarks interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
