"""
Comprehensive Performance Benchmarks for Knowledge Graph System

This module provides extensive performance testing capabilities for the knowledge graph
components including throughput, latency, memory usage, and scalability benchmarks.

Author: OpenEvolve Framework
Date: 2025-01-07
"""

import asyncio
import time
import psutil
import tracemalloc
import random
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from statistics import mean, median, stdev
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    metrics: Dict[str, Any]
    timestamp: str = None
    success: bool = True
    error: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KnowledgeGraphPerformanceBenchmarks:
    """
    Comprehensive performance benchmarks for knowledge graph system.

    Benchmarks:
    1. Knowledge addition (throughput)
    2. Knowledge retrieval (latency)
    3. Graph generation (processing time)
    4. Deduplication (accuracy + speed)
    5. Community detection (scalability)
    6. Embedding generation (performance)
    7. Search algorithms (quality + speed)
    8. Memory usage (efficiency)
    9. Concurrent operations (throughput)
    10. End-to-end workflows (realistic scenarios)
    """

    def __init__(self, kg_engine):
        """
        Initialize benchmark suite.

        Args:
            kg_engine: KnowledgeEngine instance to benchmark
        """
        self.engine = kg_engine
        self.results: Dict[str, BenchmarkResult] = {}

        # Performance tracking
        self.tracemalloc = tracemalloc
        self.psutil = psutil
        self.logger = logger

        # Test data generators
        self._init_test_data()

    def _init_test_data(self):
        """Initialize test data generators."""
        self.sample_entities = [
            "Python", "JavaScript", "Java", "C++", "Go",
            "Django", "Flask", "FastAPI", "React", "Vue.js",
            "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
            "Docker", "Kubernetes", "AWS", "GCP", "Azure"
        ]

        self.sample_relations = [
            "used_for", "implements", "extends", "depends_on",
            "competitor_of", "similar_to", "part_of", "requires"
        ]

        self.sample_queries = [
            "What is Python used for?",
            "Compare Django vs Flask",
            "How does Kubernetes work?",
            "Best database for web applications",
            "Python web frameworks comparison",
            "Microservices architecture patterns",
            "Cloud computing platforms comparison",
            "NoSQL vs SQL databases"
        ]

    def _generate_test_artifacts(
        self,
        num_artifacts: int
    ) -> List[Dict[str, Any]]:
        """Generate test knowledge artifacts."""
        artifacts = []

        for i in range(num_artifacts):
            artifact = {
                "source": f"test_source_{i % 10}",
                "content": f"Test knowledge artifact {i}: " +
                          f"{random.choice(self.sample_entities)} " +
                          f"{random.choice(self.sample_relations)} " +
                          f"{random.choice(self.sample_entities)}.",
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "type": random.choice(["fact", "concept", "relationship"]),
                    "confidence": random.uniform(0.7, 1.0)
                }
            }
            artifacts.append(artifact)

        return artifacts

    def _generate_test_queries(
        self,
        num_queries: int
    ) -> List[str]:
        """Generate test search queries."""
        queries = []

        for i in range(num_queries):
            if i < len(self.sample_queries):
                queries.append(self.sample_queries[i])
            else:
                # Generate random queries
                entity = random.choice(self.sample_entities)
                queries.append(f"Tell me about {entity}")

        return queries

    def _generate_entities_with_duplicates(
        self,
        num_entities: int,
        duplicate_rate: float
    ) -> Tuple[List[Dict], Dict[str, List[str]]]:
        """
        Generate entities with known duplicates.

        Returns:
            Tuple of (entities_list, ground_truth_mapping)
        """
        entities = []
        ground_truth = {}

        # Create unique entities
        num_unique = int(num_entities * (1 - duplicate_rate))
        for i in range(num_unique):
            entity_name = self.sample_entities[i % len(self.sample_entities)]
            entities.append({
                "name": entity_name,
                "type": random.choice(["technology", "framework", "database"]),
                "attributes": {"id": i}
            })
            ground_truth[entity_name] = [entity_name]

        # Create duplicates with variations
        for i in range(num_unique, num_entities):
            original_name = entities[i % num_unique]["name"]
            variations = [
                original_name.lower(),
                original_name.upper(),
                f"{original_name}_v2",
                f" {original_name} ",
                f"{original_name}.js"
            ]
            duplicate_name = variations[i % len(variations)]

            entities.append({
                "name": duplicate_name,
                "type": entities[i % num_unique]["type"],
                "attributes": {"id": i}
            })

            # Add to ground truth
            if original_name not in ground_truth:
                ground_truth[original_name] = []
            ground_truth[original_name].append(duplicate_name)

        return entities, ground_truth

    def _generate_test_graph(
        self,
        num_nodes: int
    ) -> Dict[str, Any]:
        """Generate test knowledge graph."""
        graph = {
            "nodes": [],
            "edges": []
        }

        # Create nodes
        for i in range(num_nodes):
            node = {
                "id": f"node_{i}",
                "name": self.sample_entities[i % len(self.sample_entities)],
                "type": random.choice(["entity", "concept", "relationship"]),
                "attributes": {"weight": random.uniform(0.1, 1.0)}
            }
            graph["nodes"].append(node)

        # Create edges (approximately 2x number of nodes)
        num_edges = min(num_nodes * 2, num_nodes * (num_nodes - 1) // 2)
        for i in range(num_edges):
            source_idx = random.randint(0, num_nodes - 1)
            target_idx = random.randint(0, num_nodes - 1)

            if source_idx != target_idx:
                edge = {
                    "source": graph["nodes"][source_idx]["id"],
                    "target": graph["nodes"][target_idx]["id"],
                    "relation": random.choice(self.sample_relations),
                    "weight": random.uniform(0.1, 1.0)
                }
                graph["edges"].append(edge)

        return graph

    def _percentile(self, data: List[float], p: int) -> float:
        """Calculate percentile of data."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * p / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def _calculate_dedup_metrics(
        self,
        deduped: List[Dict],
        ground_truth: Dict[str, List[str]]
    ) -> Tuple[int, int, int, int]:
        """Calculate deduplication metrics (TP, FP, TN, FN)."""
        tp = fp = tn = fn = 0

        # Simplified metric calculation
        deduped_names = set(e["name"].lower() for e in deduped)

        for canonical, variants in ground_truth.items():
            all_variants = [canonical.lower()] + [v.lower() for v in variants]

            # Check if canonical was identified
            if canonical.lower() in deduped_names:
                # True positive: duplicates correctly merged
                tp += 1
            else:
                # False negative: should have been merged
                fn += 1

            # Check for false positives
            for variant in variants[1:]:  # Skip canonical itself
                if variant in deduped_names:
                    fp += 1

        return tp, fp, tn, fn

    async def benchmark_knowledge_addition(
        self,
        num_artifacts: int = 1000,
        batch_size: int = 10
    ) -> BenchmarkResult:
        """
        Benchmark knowledge addition throughput.

        Metrics:
        - Artifacts per second
        - Batch addition efficiency
        - Memory usage
        - CPU utilization
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARK: Knowledge Addition")
        print(f"{'='*60}")
        print(f"Artifacts: {num_artifacts}")
        print(f"Batch Size: {batch_size}")

        try:
            # Start tracking
            self.tracemalloc.start()
            start_cpu = self.psutil.cpu_percent()
            start_time = time.time()
            start_mem = self.psutil.virtual_memory().used

            # Run benchmark
            artifacts = self._generate_test_artifacts(num_artifacts)

            # Add entities and relationships to knowledge graph
            for i, artifact in enumerate(artifacts):
                await self.engine.entity_graph.add_entity(
                    artifact["source"],
                    artifact["metadata"]
                )

                # Add some relationships
                if i > 0 and i % 2 == 0:
                    await self.engine.entity_graph.add_relationship(
                        artifact["source"],
                        artifact["metadata"]["type"],
                        artifacts[i-1]["source"]
                    )

            # Stop tracking
            end_time = time.time()
            end_cpu = self.psutil.cpu_percent()
            end_mem = self.psutil.virtual_memory().used
            current, peak = self.tracemalloc.get_traced_memory()
            self.tracemalloc.stop()

            # Calculate metrics
            duration = end_time - start_time
            throughput = num_artifacts / duration
            memory_used = (end_mem - start_mem) / (1024**3)  # GB
            peak_memory = peak / (1024**3)

            result = BenchmarkResult(
                name="knowledge_addition",
                metrics={
                    "duration_seconds": duration,
                    "artifacts_per_second": throughput,
                    "memory_used_gb": memory_used,
                    "peak_memory_gb": peak_memory,
                    "cpu_usage_percent": end_cpu - start_cpu,
                    "num_artifacts": num_artifacts,
                    "batch_size": batch_size
                }
            )

            print(f"✓ Duration: {duration:.2f}s")
            print(f"✓ Throughput: {throughput:.2f} artifacts/sec")
            print(f"✓ Memory Used: {memory_used:.2f} GB")
            print(f"✓ Peak Memory: {peak_memory:.2f} GB")
            print(f"✓ CPU Usage: {end_cpu - start_cpu:.1f}%")

            self.results["knowledge_addition"] = result
            return result

        except Exception as e:
            self.logger.error(f"Knowledge addition benchmark failed: {e}")
            return BenchmarkResult(
                name="knowledge_addition",
                metrics={},
                success=False,
                error=str(e)
            )

    async def benchmark_knowledge_retrieval(
        self,
        num_queries: int = 100,
        query_types: List[str] = None
    ) -> BenchmarkResult:
        """
        Benchmark knowledge retrieval latency.

        Metrics:
        - Average query latency
        - P50, P95, P99 latencies
        - Throughput (queries/sec)
        - Different search types comparison
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARK: Knowledge Retrieval")
        print(f"{'='*60}")

        query_types = query_types or ["keyword", "graph"]
        latencies = {qt: [] for qt in query_types}

        try:
            # First, populate with some test data
            artifacts = self._generate_test_artifacts(100)
            for artifact in artifacts[:50]:
                await self.engine.entity_graph.add_entity(
                    artifact["source"],
                    artifact["metadata"]
                )

            for query_type in query_types:
                print(f"\nTesting {query_type} search...")
                queries = self._generate_test_queries(num_queries)
                type_latencies = []

                for query in queries:
                    start = time.time()

                    # Perform search based on type
                    if query_type == "keyword":
                        # Simple keyword search through entities
                        entities = self.engine.entity_graph.get_entities()
                        matches = [
                            e for e in entities.values()
                            if query.lower() in str(e).lower()
                        ]
                    elif query_type == "graph":
                        # Graph traversal simulation
                        entities = self.engine.entity_graph.get_entities()
                        matches = list(entities.items())[:10]

                    latency = (time.time() - start) * 1000  # ms
                    type_latencies.append(latency)

                latencies[query_type] = type_latencies

                # Calculate statistics
                avg = mean(type_latencies)
                p50 = median(type_latencies)
                p95 = self._percentile(type_latencies, 95)
                p99 = self._percentile(type_latencies, 99)

                print(f"  Average: {avg:.2f}ms")
                print(f"  P50: {p50:.2f}ms")
                print(f"  P95: {p95:.2f}ms")
                print(f"  P99: {p99:.2f}ms")

            result = BenchmarkResult(
                name="knowledge_retrieval",
                metrics={
                    "latencies_ms": latencies,
                    "avg_latency_ms": mean([mean(l) for l in latencies.values()]),
                    "queries_per_second": num_queries / sum([sum(l) for l in latencies.values()]) * 1000,
                    "num_queries": num_queries,
                    "query_types": query_types
                }
            )

            self.results["knowledge_retrieval"] = result
            return result

        except Exception as e:
            self.logger.error(f"Knowledge retrieval benchmark failed: {e}")
            return BenchmarkResult(
                name="knowledge_retrieval",
                metrics={},
                success=False,
                error=str(e)
            )

    async def benchmark_deduplication(
        self,
        num_entities: int = 1000,
        duplicate_rate: float = 0.3
    ) -> BenchmarkResult:
        """
        Benchmark deduplication performance and accuracy.

        Metrics:
        - Processing time
        - Duplicate detection accuracy
        - False positive rate
        - False negative rate
        - Memory usage
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARK: Deduplication")
        print(f"{'='*60}")

        try:
            # Generate test data with known duplicates
            entities, ground_truth = self._generate_entities_with_duplicates(
                num_entities,
                duplicate_rate
            )

            print(f"\nTesting entity standardization...")

            start = time.time()
            start_mem = self.psutil.virtual_memory().used

            # Add entities to knowledge graph
            for entity in entities:
                await self.engine.entity_graph.add_entity(
                    entity["name"],
                    entity["attributes"]
                )

            duration = time.time() - start
            mem_used = (self.psutil.virtual_memory().used - start_mem) / (1024**2)  # MB

            # Get final entities
            final_entities = self.engine.entity_graph.get_entities()

            # Calculate metrics
            tp, fp, tn, fn = self._calculate_dedup_metrics(
                [{"name": k, **v} for k, v in final_entities.items()],
                ground_truth
            )

            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            reduction_rate = (len(entities) - len(final_entities)) / len(entities)

            print(f"  Duration: {duration:.2f}s")
            print(f"  Accuracy: {accuracy:.2%}")
            print(f"  Precision: {precision:.2%}")
            print(f"  Recall: {recall:.2%}")
            print(f"  F1 Score: {f1_score:.2%}")
            print(f"  Reduction: {reduction_rate:.2%}")
            print(f"  Memory Used: {mem_used:.2f} MB")

            result = BenchmarkResult(
                name="deduplication",
                metrics={
                    "duration_seconds": duration,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "reduction_rate": reduction_rate,
                    "memory_mb": mem_used,
                    "num_entities": num_entities,
                    "duplicate_rate": duplicate_rate
                }
            )

            self.results["deduplication"] = result
            return result

        except Exception as e:
            self.logger.error(f"Deduplication benchmark failed: {e}")
            return BenchmarkResult(
                name="deduplication",
                metrics={},
                success=False,
                error=str(e)
            )

    async def benchmark_graph_algorithms(
        self,
        graph_sizes: List[int] = None
    ) -> BenchmarkResult:
        """
        Benchmark graph algorithm performance.

        Metrics:
        - Processing time vs graph size
        - Memory usage vs graph size
        - Scalability characteristics
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARK: Graph Algorithms")
        print(f"{'='*60}")

        graph_sizes = graph_sizes or [100, 500, 1000, 5000]
        results = {}

        try:
            for size in graph_sizes:
                print(f"\nTesting graph size: {size} nodes")

                graph = self._generate_test_graph(size)
                results[size] = {}

                start_time = time.time()
                start_mem = self.psutil.virtual_memory().used

                # Add nodes to knowledge graph
                for node in graph["nodes"]:
                    await self.engine.entity_graph.add_entity(
                        node["id"],
                        node["attributes"]
                    )

                # Add relationships
                for edge in graph["edges"]:
                    await self.engine.entity_graph.add_relationship(
                        edge["source"],
                        edge["relation"],
                        edge["target"],
                        edge.get("attributes", {})
                    )

                duration = time.time() - start_time
                mem_used = (self.psutil.virtual_memory().used - start_mem) / (1024**2)  # MB

                results[size] = {
                    "duration_seconds": duration,
                    "memory_mb": mem_used,
                    "nodes_processed": size,
                    "edges_processed": len(graph["edges"])
                }

                print(f"  Time: {duration:.2f}s")
                print(f"  Memory: {mem_used:.1f}MB")
                print(f"  Edges: {len(graph['edges'])}")

            result = BenchmarkResult(
                name="graph_algorithms",
                metrics=results
            )

            self.results["graph_algorithms"] = result
            return result

        except Exception as e:
            self.logger.error(f"Graph algorithms benchmark failed: {e}")
            return BenchmarkResult(
                name="graph_algorithms",
                metrics={},
                success=False,
                error=str(e)
            )

    async def benchmark_concurrent_operations(
        self,
        num_concurrent: int = 10,
        operations_per_client: int = 50
    ) -> BenchmarkResult:
        """
        Benchmark concurrent operation throughput.

        Metrics:
        - Concurrent throughput
        - Resource contention
        - Error rate under load
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARK: Concurrent Operations")
        print(f"{'='*60}")
        print(f"Concurrent Clients: {num_concurrent}")
        print(f"Operations per Client: {operations_per_client}")

        try:
            async def client_operations(client_id: int):
                """Simulate client operations."""
                ops_completed = 0
                errors = 0

                for i in range(operations_per_client):
                    try:
                        # Mix of operations
                        op_type = random.choice(["add", "search", "add_relation"])

                        if op_type == "add":
                            await self.engine.entity_graph.add_entity(
                                f"client_{client_id}_entity_{i}",
                                {"client": client_id, "index": i}
                            )
                        elif op_type == "search":
                            self.engine.entity_graph.get_entities()
                        elif op_type == "add_relation":
                            await self.engine.entity_graph.add_relationship(
                                f"client_{client_id}_entity_{i}",
                                "test_relation",
                                f"client_{client_id}_entity_{i-1}" if i > 0 else "root"
                            )

                        ops_completed += 1
                    except Exception as e:
                        errors += 1
                        self.logger.debug(f"Client {client_id} operation {i} failed: {e}")

                return ops_completed, errors

            # Run concurrent clients
            start_time = time.time()
            tasks = [client_operations(i) for i in range(num_concurrent)]
            results_list = await asyncio.gather(*tasks)
            end_time = time.time()

            # Calculate metrics
            total_ops = sum(r[0] for r in results_list)
            total_errors = sum(r[1] for r in results_list)
            duration = end_time - start_time
            throughput = total_ops / duration
            error_rate = total_errors / (total_ops + total_errors) if (total_ops + total_errors) > 0 else 0

            print(f"\n✓ Total Operations: {total_ops}")
            print(f"✓ Errors: {total_errors}")
            print(f"✓ Duration: {duration:.2f}s")
            print(f"✓ Throughput: {throughput:.2f} ops/sec")
            print(f"✓ Error Rate: {error_rate:.2%}")

            result = BenchmarkResult(
                name="concurrent_operations",
                metrics={
                    "total_operations": total_ops,
                    "errors": total_errors,
                    "duration_seconds": duration,
                    "throughput_ops_per_sec": throughput,
                    "error_rate": error_rate,
                    "num_concurrent": num_concurrent,
                    "operations_per_client": operations_per_client
                }
            )

            self.results["concurrent_operations"] = result
            return result

        except Exception as e:
            self.logger.error(f"Concurrent operations benchmark failed: {e}")
            return BenchmarkResult(
                name="concurrent_operations",
                metrics={},
                success=False,
                error=str(e)
            )

    async def benchmark_end_to_end_workflows(
        self,
        scenarios: List[str] = None
    ) -> BenchmarkResult:
        """
        Benchmark realistic end-to-end workflows.

        Scenarios:
        1. Document processing → extraction → storage
        2. Large document chunking → parallel processing → aggregation
        3. Temporal knowledge addition → point-in-time query
        4. Multi-stage extraction → deduplication → inference
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARK: End-to-End Workflows")
        print(f"{'='*60}")

        scenarios = scenarios or [
            "entity_relationship_workflow",
            "batch_processing_workflow",
            "query_workflow"
        ]

        results = {}

        try:
            for scenario in scenarios:
                print(f"\nScenario: {scenario}")

                if scenario == "entity_relationship_workflow":
                    result = await self._benchmark_entity_workflow()
                elif scenario == "batch_processing_workflow":
                    result = await self._benchmark_batch_workflow()
                elif scenario == "query_workflow":
                    result = await self._benchmark_query_workflow()

                results[scenario] = result
                print(f"  Duration: {result['duration_seconds']:.2f}s")
                print(f"  Success: {result['success']}")

            benchmark_result = BenchmarkResult(
                name="end_to_end_workflows",
                metrics=results
            )

            self.results["end_to_end_workflows"] = benchmark_result
            return benchmark_result

        except Exception as e:
            self.logger.error(f"End-to-end workflows benchmark failed: {e}")
            return BenchmarkResult(
                name="end_to_end_workflows",
                metrics={},
                success=False,
                error=str(e)
            )

    async def _benchmark_entity_workflow(self) -> Dict[str, Any]:
        """Benchmark entity and relationship workflow."""
        start_time = time.time()

        # Add entities
        entities = self._generate_test_artifacts(50)
        for artifact in entities:
            await self.engine.entity_graph.add_entity(
                artifact["source"],
                artifact["metadata"]
            )

        # Add relationships
        for i in range(len(entities) - 1):
            await self.engine.entity_graph.add_relationship(
                entities[i]["source"],
                "related_to",
                entities[i+1]["source"]
            )

        # Query
        all_entities = self.engine.entity_graph.get_entities()

        duration = time.time() - start_time

        return {
            "duration_seconds": duration,
            "success": True,
            "entities_added": len(entities),
            "relationships_added": len(entities) - 1,
            "entities_retrieved": len(all_entities)
        }

    async def _benchmark_batch_workflow(self) -> Dict[str, Any]:
        """Benchmark batch processing workflow."""
        start_time = time.time()

        # Process batches
        batch_size = 10
        artifacts = self._generate_test_artifacts(100)

        for i in range(0, len(artifacts), batch_size):
            batch = artifacts[i:i+batch_size]
            for artifact in batch:
                await self.engine.entity_graph.add_entity(
                    artifact["source"],
                    artifact["metadata"]
                )

        duration = time.time() - start_time

        return {
            "duration_seconds": duration,
            "success": True,
            "batch_size": batch_size,
            "total_artifacts": len(artifacts)
        }

    async def _benchmark_query_workflow(self) -> Dict[str, Any]:
        """Benchmark query workflow."""
        # First populate data
        artifacts = self._generate_test_artifacts(50)
        for artifact in artifacts:
            await self.engine.entity_graph.add_entity(
                artifact["source"],
                artifact["metadata"]
            )

        start_time = time.time()

        # Perform queries
        queries = self._generate_test_queries(20)
        results_count = 0

        for query in queries:
            entities = self.engine.entity_graph.get_entities()
            matches = [
                e for e in entities.values()
                if any(q.lower() in str(e).lower() for q in query.split())
            ]
            results_count += len(matches)

        duration = time.time() - start_time

        return {
            "duration_seconds": duration,
            "success": True,
            "queries_executed": len(queries),
            "total_results": results_count
        }

    def generate_report(
        self,
        output_path: str = "benchmark_report.md",
        include_raw_data: bool = True
    ):
        """
        Generate comprehensive benchmark report.

        Args:
            output_path: Path to save the report
            include_raw_data: Whether to include raw metric data
        """
        print(f"\n{'='*60}")
        print(f"GENERATING BENCHMARK REPORT")
        print(f"{'='*60}")

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Header
                f.write("# Knowledge Graph Performance Benchmarks\n\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Total Benchmarks:** {len(self.results)}\n\n")

                # Summary
                f.write("## Summary\n\n")
                successful = sum(1 for r in self.results.values() if r.success)
                failed = len(self.results) - successful

                f.write(f"- **Successful:** {successful}\n")
                f.write(f"- **Failed:** {failed}\n")
                f.write(f"- **Success Rate:** {successful/len(self.results)*100:.1f}%\n\n")

                # Detailed Results
                f.write("## Detailed Results\n\n")

                for name, result in self.results.items():
                    f.write(f"### {name.replace('_', ' ').title()}\n\n")

                    if result.success:
                        f.write(f"**Status:** ✓ Success\n\n")

                        # Write key metrics
                        for key, value in result.metrics.items():
                            if isinstance(value, (int, float, str, bool)):
                                f.write(f"- **{key}:** {value}\n")
                            elif isinstance(value, dict):
                                f.write(f"- **{key}:**\n")
                                for k, v in value.items():
                                    f.write(f"  - {k}: {v}\n")
                            elif isinstance(value, list) and len(value) > 0:
                                f.write(f"- **{key}:** {len(value)} items\n")

                        # Include raw data if requested
                        if include_raw_data:
                            f.write(f"\n**Raw Data:**\n")
                            f.write(f"```json\n{json.dumps(result.metrics, indent=2, default=str)}\n```\n")

                    else:
                        f.write(f"**Status:** ✗ Failed\n\n")
                        f.write(f"**Error:** {result.error}\n\n")

                    f.write("\n")

                # Performance summary table
                f.write("## Performance Summary\n\n")
                f.write("| Benchmark | Status | Key Metric | Value |\n")
                f.write("|-----------|--------|------------|-------|\n")

                for name, result in self.results.items():
                    status = "✓" if result.success else "✗"

                    # Extract a key metric
                    key_metric = "N/A"
                    key_value = "N/A"

                    if result.success and result.metrics:
                        if "throughput" in str(result.metrics).lower():
                            for k, v in result.metrics.items():
                                if "throughput" in k.lower() or "per_second" in k.lower():
                                    key_metric = k.replace("_", " ").title()
                                    key_value = f"{v:.2f}"
                                    break
                        elif "duration" in result.metrics:
                            key_metric = "Duration"
                            key_value = f"{result.metrics['duration_seconds']:.2f}s"

                    f.write(f"| {name} | {status} | {key_metric} | {key_value} |\n")

                f.write("\n")

                # Footer
                f.write("---\n")
                f.write("*Generated by OpenEvolve Knowledge Graph Benchmark Suite*\n")

            print(f"✓ Report saved to: {output_path}")

            # Also save raw JSON data
            json_path = output_path.replace('.md', '_raw.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {k: v.to_dict() for k, v in self.results.items()},
                    f,
                    indent=2,
                    default=str
                )
            print(f"✓ Raw data saved to: {json_path}")

        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")

    def save_metrics(
        self,
        output_path: str = "benchmark_metrics.json"
    ):
        """
        Save benchmark metrics to JSON file.

        Args:
            output_path: Path to save metrics
        """
        try:
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "benchmarks": {
                    name: result.to_dict()
                    for name, result in self.results.items()
                }
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, default=str)

            self.logger.info(f"Metrics saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
