"""
Load Testing Framework for Knowledge Graph System

This module provides comprehensive load testing capabilities for the knowledge
graph system, including read-heavy, write-heavy, spike, and endurance tests.

Test Scenarios:
1. Read-heavy workload (search queries)
2. Write-heavy workload (knowledge addition)
3. Mixed workload (realistic usage)
4. Spike test (sudden traffic increase)
5. Endurance test (sustained load)
"""

import asyncio
import time
import random
import statistics
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LoadTestResult:
    """Results from a load test execution."""
    test_name: str
    metrics: Dict
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    passed: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "errors": self.errors
        }


class KnowledgeGraphLoadTest:
    """
    Load testing framework for knowledge graph system.

    Provides multiple test scenarios to validate system performance
    under various load conditions.
    """

    def __init__(self, kg_engine):
        """
        Initialize load tester.

        Args:
            kg_engine: Knowledge graph engine instance
        """
        self.engine = kg_engine
        self.metrics = {
            "response_times": [],
            "success_count": 0,
            "error_count": 0,
            "throughput": []
        }
        self.test_results: List[LoadTestResult] = []

    async def run_read_heavy_test(
        self,
        num_users: int = 100,
        spawn_rate: int = 10,
        test_duration: int = 60,
        config: Optional[Dict] = None
    ) -> LoadTestResult:
        """
        Simulate read-heavy workload (mostly search queries).

        Characteristics:
        - 90% search operations
        - 10% write operations
        - Tests retrieval performance under load

        Args:
            num_users: Number of concurrent users to simulate
            spawn_rate: Users spawned per second
            test_duration: Test duration in seconds
            config: Optional test configuration

        Returns:
            LoadTestResult with metrics and analysis
        """
        config = config or {}
        target_throughput = config.get("target_throughput", 100)
        max_error_rate = config.get("max_error_rate", 0.01)

        print(f"\n{'='*60}")
        print(f"LOAD TEST: Read-Heavy Workload")
        print(f"{'='*60}")
        print(f"Users: {num_users}")
        print(f"Spawn Rate: {spawn_rate} users/sec")
        print(f"Duration: {test_duration} sec")
        print(f"Target Throughput: {target_throughput} ops/sec")
        print(f"Max Error Rate: {max_error_rate:.1%}")

        warnings = []
        errors = []

        async def user_simulation(user_id: int) -> Tuple[int, int]:
            """Simulate a user performing read operations."""
            start_time = time.time()
            ops_completed = 0
            user_errors = 0

            while time.time() - start_time < test_duration:
                try:
                    # 90% reads
                    if random.random() < 0.9:
                        await self.engine.search(
                            f"test query {user_id}",
                            search_type="hybrid"
                        )
                    else:
                        # 10% writes
                        await self.engine.add_knowledge(
                            source=f"user_{user_id}",
                            content=f"Test content {ops_completed}"
                        )

                    ops_completed += 1
                    await asyncio.sleep(random.uniform(0.1, 0.5))  # Think time
                except Exception as e:
                    user_errors += 1
                    logger.warning(f"User {user_id} operation failed: {e}")

            return ops_completed, user_errors

        # Spawn users gradually
        start_time = time.time()
        tasks = []
        spawned = 0

        while spawned < num_users:
            batch_size = min(spawn_rate, num_users - spawned)
            batch_tasks = [
                user_simulation(i)
                for i in range(spawned, spawned + batch_size)
            ]
            tasks.extend(batch_tasks)
            spawned += batch_size

            if spawned < num_users:
                await asyncio.sleep(1)  # Spawn rate delay

        # Wait for all users to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        # Calculate metrics
        valid_results = [r for r in results if isinstance(r, tuple)]
        total_ops = sum(r[0] for r in valid_results)
        total_errors = sum(r[1] for r in valid_results)
        duration = end_time - start_time
        throughput = total_ops / duration
        error_rate = total_errors / (total_ops + total_errors) if (total_ops + total_errors) > 0 else 0

        # Validate results
        passed = True
        if throughput < target_throughput:
            passed = False
            errors.append(
                f"Throughput {throughput:.2f} ops/sec below target {target_throughput} ops/sec"
            )
        else:
            logger.info(f"✓ Throughput target met: {throughput:.2f} ops/sec")

        if error_rate > max_error_rate:
            passed = False
            errors.append(
                f"Error rate {error_rate:.2%} exceeds maximum {max_error_rate:.1%}"
            )
        else:
            logger.info(f"✓ Error rate acceptable: {error_rate:.2%}")

        print(f"\n{'='*60}")
        print(f"RESULTS: Read-Heavy Workload")
        print(f"{'='*60}")
        print(f"✓ Total Operations: {total_ops}")
        print(f"✓ Throughput: {throughput:.2f} ops/sec")
        print(f"✓ Error Rate: {error_rate:.2%}")
        print(f"✓ Duration: {duration:.2f}s")
        print(f"✓ Status: {'PASSED' if passed else 'FAILED'}")

        result = LoadTestResult(
            test_name="read_heavy",
            metrics={
                "total_operations": total_ops,
                "duration_seconds": duration,
                "throughput_ops_per_sec": throughput,
                "error_rate": error_rate,
                "concurrent_users": num_users,
                "spawn_rate": spawn_rate
            },
            passed=passed,
            warnings=warnings,
            errors=errors
        )

        self.test_results.append(result)
        return result

    async def run_write_heavy_test(
        self,
        num_users: int = 50,
        spawn_rate: int = 5,
        test_duration: int = 60,
        config: Optional[Dict] = None
    ) -> LoadTestResult:
        """
        Simulate write-heavy workload (knowledge addition).

        Characteristics:
        - 80% write operations
        - 20% read operations
        - Tests write performance and scalability

        Args:
            num_users: Number of concurrent users
            spawn_rate: Users spawned per second
            test_duration: Test duration in seconds
            config: Optional test configuration

        Returns:
            LoadTestResult with metrics
        """
        config = config or {}
        target_throughput = config.get("target_throughput", 50)
        max_error_rate = config.get("max_error_rate", 0.05)

        print(f"\n{'='*60}")
        print(f"LOAD TEST: Write-Heavy Workload")
        print(f"{'='*60}")
        print(f"Users: {num_users}")
        print(f"Spawn Rate: {spawn_rate} users/sec")
        print(f"Duration: {test_duration} sec")
        print(f"Target Throughput: {target_throughput} ops/sec")
        print(f"Max Error Rate: {max_error_rate:.1%}")

        warnings = []
        errors = []

        async def user_simulation(user_id: int) -> Tuple[int, int]:
            """Simulate write-heavy user."""
            start_time = time.time()
            ops_completed = 0
            user_errors = 0

            while time.time() - start_time < test_duration:
                try:
                    # 80% writes
                    if random.random() < 0.8:
                        await self.engine.add_knowledge(
                            source=f"user_{user_id}",
                            content=f"Test content {ops_completed}",
                            metadata={"batch": user_id, "timestamp": time.time()}
                        )
                    else:
                        # 20% reads
                        await self.engine.search(f"query {ops_completed}")

                    ops_completed += 1
                    await asyncio.sleep(random.uniform(0.05, 0.2))
                except Exception as e:
                    user_errors += 1
                    logger.warning(f"User {user_id} write failed: {e}")

            return ops_completed, user_errors

        # Spawn and execute
        start_time = time.time()
        tasks = []
        spawned = 0

        while spawned < num_users:
            batch_size = min(spawn_rate, num_users - spawned)
            batch_tasks = [
                user_simulation(i)
                for i in range(spawned, spawned + batch_size)
            ]
            tasks.extend(batch_tasks)
            spawned += batch_size

            if spawned < num_users:
                await asyncio.sleep(1)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        # Calculate metrics
        valid_results = [r for r in results if isinstance(r, tuple)]
        total_ops = sum(r[0] for r in valid_results)
        total_errors = sum(r[1] for r in valid_results)
        duration = end_time - start_time
        throughput = total_ops / duration
        error_rate = total_errors / (total_ops + total_errors) if (total_ops + total_errors) > 0 else 0

        # Validate
        passed = True
        if throughput < target_throughput:
            passed = False
            errors.append(
                f"Throughput {throughput:.2f} ops/sec below target {target_throughput} ops/sec"
            )

        if error_rate > max_error_rate:
            passed = False
            errors.append(
                f"Error rate {error_rate:.2%} exceeds maximum {max_error_rate:.1%}"
            )

        print(f"\n{'='*60}")
        print(f"RESULTS: Write-Heavy Workload")
        print(f"{'='*60}")
        print(f"✓ Total Operations: {total_ops}")
        print(f"✓ Throughput: {throughput:.2f} ops/sec")
        print(f"✓ Error Rate: {error_rate:.2%}")
        print(f"✓ Duration: {duration:.2f}s")
        print(f"✓ Status: {'PASSED' if passed else 'FAILED'}")

        result = LoadTestResult(
            test_name="write_heavy",
            metrics={
                "total_operations": total_ops,
                "duration_seconds": duration,
                "throughput_ops_per_sec": throughput,
                "error_rate": error_rate,
                "concurrent_users": num_users,
                "spawn_rate": spawn_rate
            },
            passed=passed,
            warnings=warnings,
            errors=errors
        )

        self.test_results.append(result)
        return result

    async def run_spike_test(
        self,
        base_users: int = 10,
        spike_users: int = 100,
        spike_duration: int = 30,
        config: Optional[Dict] = None
    ) -> LoadTestResult:
        """
        Spike test: Simulate sudden traffic increase.

        Process:
        1. Start with base_users
        2. Ramp up to spike_users rapidly
        3. Maintain spike load
        4. Measure system resilience

        Args:
            base_users: Initial number of users
            spike_users: Peak number of users during spike
            spike_duration: Duration of spike in seconds
            config: Optional test configuration

        Returns:
            LoadTestResult with spike metrics
        """
        config = config or {}
        max_response_degradation = config.get("max_response_time_degradation", 0.5)

        print(f"\n{'='*60}")
        print(f"LOAD TEST: Spike Test")
        print(f"{'='*60}")
        print(f"Base Users: {base_users}")
        print(f"Spike Users: {spike_users}")
        print(f"Spike Duration: {spike_duration} sec")
        print(f"Max Response Time Degradation: {max_response_degradation:.1%}")

        warnings = []
        errors = []

        # Measure baseline response time
        print(f"\nPhase 1: Measuring baseline ({base_users} users, 10 sec)")
        baseline_times = []

        async def baseline_user(user_id: int):
            start = time.time()
            try:
                await self.engine.search(f"baseline query {user_id}")
                baseline_times.append(time.time() - start)
            except Exception as e:
                logger.warning(f"Baseline query failed: {e}")

        baseline_tasks = [baseline_user(i) for i in range(base_users)]
        await asyncio.gather(*baseline_tasks)

        baseline_avg = statistics.mean(baseline_times) if baseline_times else 0
        print(f"✓ Baseline response time: {baseline_avg:.3f}s")

        await asyncio.sleep(2)

        # Phase 2: Spike!
        print(f"\nPhase 2: SPIKE! (+{spike_users - base_users} users)")

        spike_times = []
        spike_ops = [0]
        spike_errors = [0]

        async def spike_user(user_id: int):
            start_time = time.time()
            user_ops = 0
            user_errors = 0

            while time.time() - start_time < spike_duration:
                try:
                    op_start = time.time()
                    await self.engine.search(f"spike query {user_id}")
                    spike_times.append(time.time() - op_start)
                    user_ops += 1
                except Exception as e:
                    user_errors += 1
                    logger.warning(f"Spike query failed: {e}")

                await asyncio.sleep(random.uniform(0.1, 0.3))

            spike_ops[0] += user_ops
            spike_errors[0] += user_errors

        spike_tasks = [spike_user(i) for i in range(spike_users)]
        await asyncio.gather(*spike_tasks)

        spike_avg = statistics.mean(spike_times) if spike_times else 0
        degradation = (spike_avg - baseline_avg) / baseline_avg if baseline_avg > 0 else 0

        print(f"\n{'='*60}")
        print(f"RESULTS: Spike Test")
        print(f"{'='*60}")
        print(f"✓ Operations completed: {spike_ops[0]}")
        print(f"✓ Baseline response time: {baseline_avg:.3f}s")
        print(f"✓ Spike response time: {spike_avg:.3f}s")
        print(f"✓ Response time degradation: {degradation:.1%}")
        print(f"✓ Errors during spike: {spike_errors[0]}")

        # Validate
        passed = True
        if degradation > max_response_degradation:
            passed = False
            errors.append(
                f"Response time degradation {degradation:.1%} exceeds maximum {max_response_degradation:.1%}"
            )
        else:
            logger.info(f"✓ Response time degradation acceptable")

        if spike_errors[0] > spike_ops[0] * 0.05:  # 5% error threshold
            passed = False
            errors.append(f"High error rate during spike: {spike_errors[0]} errors")

        print(f"✓ Status: {'PASSED' if passed else 'FAILED'}")

        result = LoadTestResult(
            test_name="spike_test",
            metrics={
                "baseline_users": base_users,
                "spike_users": spike_users,
                "spike_duration_seconds": spike_duration,
                "operations_completed": spike_ops[0],
                "errors": spike_errors[0],
                "baseline_response_time": baseline_avg,
                "spike_response_time": spike_avg,
                "response_time_degradation": degradation
            },
            passed=passed,
            warnings=warnings,
            errors=errors
        )

        self.test_results.append(result)
        return result

    async def run_endurance_test(
        self,
        num_users: int = 20,
        test_duration: int = 300,
        config: Optional[Dict] = None
    ) -> LoadTestResult:
        """
        Endurance test: Sustained load over time.

        Tests:
        - Memory leak detection
        - Performance degradation
        - Connection stability

        Args:
            num_users: Number of concurrent users
            test_duration: Test duration in seconds (default 5 min)
            config: Optional test configuration

        Returns:
            LoadTestResult with endurance metrics
        """
        config = config or {}
        max_memory_growth = config.get("max_memory_growth", 0.5)  # 500 MB
        max_performance_degradation = config.get("max_performance_degradation", 0.2)  # 20%

        print(f"\n{'='*60}")
        print(f"LOAD TEST: Endurance Test")
        print(f"{'='*60}")
        print(f"Users: {num_users}")
        print(f"Duration: {test_duration} sec ({test_duration//60} min)")
        print(f"Max Memory Growth: {max_memory_growth:.2f} GB")
        print(f"Max Performance Degradation: {max_performance_degradation:.1%}")

        warnings = []
        errors = []

        # Import monitoring libraries
        try:
            import tracemalloc
            import psutil

            tracemalloc.start()
            process = psutil.Process()
            start_mem = process.memory_info().rss / (1024**3)  # GB
        except ImportError:
            logger.warning("psutil or tracemalloc not available - skipping memory tracking")
            start_mem = 0
            tracemalloc = None

        # Track performance over time
        performance_samples = []
        sample_interval = test_duration // 10  # 10 samples

        async def endurance_user(user_id: int):
            """User running for entire duration."""
            start_time = time.time()
            ops_completed = 0
            user_errors = 0
            response_times = []

            while time.time() - start_time < test_duration:
                try:
                    op_start = time.time()

                    # Mix of operations
                    op_type = random.choice(["search", "add", "analyze"])

                    if op_type == "search":
                        await self.engine.search(f"endurance query {user_id}")
                    elif op_type == "add":
                        await self.engine.add_knowledge(
                            source=f"endurance_user_{user_id}",
                            content=f"Endurance test content {ops_completed}"
                        )
                    elif op_type == "analyze":
                        # Lightweight analysis
                        try:
                            await self.engine.get_graph_stats()
                        except Exception:
                            pass  # Skip if not available

                    response_times.append(time.time() - op_start)
                    ops_completed += 1

                    # Sample performance periodically
                    elapsed = time.time() - start_time
                    if int(elapsed) % sample_interval == 0 and len(response_times) > 0:
                        avg_response = statistics.mean(response_times[-10:])
                        performance_samples.append({
                            "time": elapsed,
                            "user_id": user_id,
                            "avg_response_time": avg_response
                        })

                    await asyncio.sleep(random.uniform(0.2, 0.5))

                except Exception as e:
                    user_errors += 1
                    logger.warning(f"Endurance user {user_id} error: {e}")

            return ops_completed, user_errors, response_times

        # Run endurance test
        start_time = time.time()
        tasks = [endurance_user(i) for i in range(num_users)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        # Calculate final metrics
        valid_results = [r for r in results if isinstance(r, tuple)]
        total_ops = sum(r[0] for r in valid_results)
        total_errors = sum(r[1] for r in valid_results)
        duration = end_time - start_time
        throughput = total_ops / duration

        # Memory analysis
        mem_growth = 0
        peak_mem = 0

        if tracemalloc:
            try:
                end_mem = process.memory_info().rss / (1024**3)
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                mem_growth = end_mem - start_mem
                peak_mem = peak / (1024**3)

                print(f"\n{'='*60}")
                print(f"MEMORY ANALYSIS")
                print(f"{'='*60}")
                print(f"✓ Start Memory: {start_mem:.3f} GB")
                print(f"✓ End Memory: {end_mem:.3f} GB")
                print(f"✓ Memory Growth: {mem_growth:.3f} GB")
                print(f"✓ Peak Memory: {peak_mem:.2f} GB")
            except Exception as e:
                logger.warning(f"Memory analysis failed: {e}")

        # Performance degradation analysis
        perf_degradation = 0
        if len(performance_samples) >= 5:
            first_half = performance_samples[:len(performance_samples)//2]
            second_half = performance_samples[len(performance_samples)//2:]

            first_avg = statistics.mean([s["avg_response_time"] for s in first_half])
            second_avg = statistics.mean([s["avg_response_time"] for s in second_half])

            if first_avg > 0:
                perf_degradation = (second_avg - first_avg) / first_avg

            print(f"\n{'='*60}")
            print(f"PERFORMANCE ANALYSIS")
            print(f"{'='*60}")
            print(f"✓ First half avg response: {first_avg:.3f}s")
            print(f"✓ Second half avg response: {second_avg:.3f}s")
            print(f"✓ Performance degradation: {perf_degradation:.1%}")

        # Validate
        passed = True

        if mem_growth > max_memory_growth:
            passed = False
            errors.append(
                f"Memory growth {mem_growth:.3f} GB exceeds maximum {max_memory_growth:.3f} GB"
            )
        else:
            logger.info(f"✓ Memory usage stable")

        if perf_degradation > max_performance_degradation:
            passed = False
            errors.append(
                f"Performance degradation {perf_degradation:.1%} exceeds maximum {max_performance_degradation:.1%}"
            )
        else:
            logger.info(f"✓ Performance degradation acceptable")

        print(f"\n{'='*60}")
        print(f"RESULTS: Endurance Test")
        print(f"{'='*60}")
        print(f"✓ Total Operations: {total_ops}")
        print(f"✓ Throughput: {throughput:.2f} ops/sec")
        print(f"✓ Errors: {total_errors}")
        print(f"✓ Duration: {duration:.2f}s")
        print(f"✓ Status: {'PASSED' if passed else 'FAILED'}")

        result = LoadTestResult(
            test_name="endurance",
            metrics={
                "concurrent_users": num_users,
                "duration_seconds": duration,
                "total_operations": total_ops,
                "errors": total_errors,
                "throughput_ops_per_sec": throughput,
                "memory_growth_gb": mem_growth,
                "peak_memory_gb": peak_mem,
                "performance_degradation": perf_degradation,
                "performance_samples": len(performance_samples)
            },
            passed=passed,
            warnings=warnings,
            errors=errors
        )

        self.test_results.append(result)
        return result

    async def _user_ops(self, user_id: int, duration: int) -> Tuple[int, int]:
        """
        Simulate user operations for specified duration.

        Args:
            user_id: User identifier
            duration: Duration in seconds

        Returns:
            Tuple of (operations_completed, errors)
        """
        start_time = time.time()
        ops_completed = 0
        errors = 0

        while time.time() - start_time < duration:
            try:
                # Mix of operations
                op_type = random.choice(["add", "search", "analyze"])

                if op_type == "add":
                    await self.engine.add_knowledge(
                        source=f"user_{user_id}",
                        content=f"Load test content {ops_completed}"
                    )
                elif op_type == "search":
                    await self.engine.search(f"load test query {user_id}")
                elif op_type == "analyze":
                    try:
                        await self.engine.get_graph_stats()
                    except Exception:
                        pass  # Skip if not available

                ops_completed += 1
                await asyncio.sleep(random.uniform(0.1, 0.3))
            except Exception as e:
                errors += 1
                logger.warning(f"User {user_id} operation failed: {e}")

        return ops_completed, errors

    def save_results(self, filepath: str):
        """
        Save test results to JSON file.

        Args:
            filepath: Path to save results
        """
        results_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "tests": [r.to_dict() for r in self.test_results]
        }

        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Results saved to {filepath}")

    def get_summary(self) -> Dict:
        """
        Get summary of all test results.

        Returns:
            Summary dictionary
        """
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)

        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "tests": [
                {
                    "name": r.test_name,
                    "passed": r.passed,
                    "errors": r.errors
                }
                for r in self.test_results
            ]
        }
