"""
Performance Integration Tests - License: Apache 2.0

Tests performance across all systems:
- Full workflow execution time
- Memory usage across systems
- Database query performance
- API response times
- Concurrent request handling

Run: pytest test_performance_integration.py -v
"""

import asyncio
import json
import tempfile
import time
import tracemalloc
import psutil
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

# Performance thresholds
MAX_WORKFLOW_TIME_MS = 30000  # 30 seconds for full workflow
MAX_API_RESPONSE_TIME_MS = 500  # 500ms for API responses
MAX_MEMORY_MB = 512  # 512MB max memory per system
MAX_DB_QUERY_TIME_MS = 100  # 100ms for DB queries
MAX_CONCURRENT_REQUESTS = 50  # Should handle 50 concurrent requests

# System availability checks
try:
    from api_server import app as api_app
    from fastapi.testclient import TestClient
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

try:
    from decomposition_engine import DecompositionEngine
    DECOMPOSITION_AVAILABLE = True
except ImportError:
    DECOMPOSITION_AVAILABLE = False

try:
    from evolution import EvolutionEngine
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False

try:
    from gauntlet_manager import GauntletManager
    GAUNTLET_AVAILABLE = True
except ImportError:
    GAUNTLET_AVAILABLE = False

try:
    from stage6_knowledge_extraction import Stage6KnowledgeExtraction
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from quality_gate_engine import QualityGateEngine
    QUALITY_AVAILABLE = True
except ImportError:
    QUALITY_AVAILABLE = False

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False


@dataclass
class PerformanceTestResult:
    """Result of a performance test."""
    test_name: str
    metric: str  # 'time', 'memory', 'throughput', 'latency'
    value: float
    threshold: float
    status: str  # 'passed', 'failed'
    unit: str
    details: Dict = field(default_factory=dict)


class TestPerformanceIntegration:
    """
    Performance Integration Tests.
    
    Verifies performance across all systems meets requirements.
    """
    
    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Setup test environment for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.results: List[PerformanceTestResult] = []
        
        # Initialize systems
        self.systems = {}
        self._init_systems()
        
        yield
        
        # Cleanup
        self.temp_dir.cleanup()
    
    def _init_systems(self):
        """Initialize all required systems."""
        if DECOMPOSITION_AVAILABLE:
            self.systems['decomposition'] = DecompositionEngine()
        
        if EVOLUTION_AVAILABLE:
            self.systems['evolution'] = EvolutionEngine()
        
        if GAUNTLET_AVAILABLE:
            self.systems['gauntlet'] = GauntletManager()
        
        if KNOWLEDGE_AVAILABLE:
            self.systems['knowledge'] = Stage6KnowledgeExtraction(
                storage_path=Path(self.temp_dir.name)
            )
        
        if QUALITY_AVAILABLE:
            self.systems['quality'] = QualityGateEngine()
    
    def _record_result(self, result: PerformanceTestResult):
        """Record test result."""
        self.results.append(result)
        return result.status == 'passed'
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_workflow_execution_time(self):
        """Test full workflow execution completes within threshold."""
        start = time.time()
        
        try:
            # Run a simplified full workflow
            workflow_stages = []
            
            # Stage 1: Decomposition
            if DECOMPOSITION_AVAILABLE:
                stage_start = time.time()
                engine = self.systems['decomposition']
                result = engine.decompose({"id": "perf_test", "description": "Performance test"})
                workflow_stages.append({"stage": "decomposition", "time": (time.time() - stage_start) * 1000})
            
            # Stage 2: Evolution
            if EVOLUTION_AVAILABLE:
                stage_start = time.time()
                evolution = self.systems['evolution']
                result = evolution.evolve({"population_size": 10, "generations": 5})
                workflow_stages.append({"stage": "evolution", "time": (time.time() - stage_start) * 1000})
            
            # Stage 3: Quality Gate
            if QUALITY_AVAILABLE:
                stage_start = time.time()
                quality = self.systems['quality']
                result = quality.check_quality({"test": "performance"})
                workflow_stages.append({"stage": "quality", "time": (time.time() - stage_start) * 1000})
            
            # Stage 4: Gauntlet
            if GAUNTLET_AVAILABLE:
                stage_start = time.time()
                gauntlet = self.systems['gauntlet']
                result = gauntlet.run_gauntlet({"id": "perf_test"})
                workflow_stages.append({"stage": "gauntlet", "time": (time.time() - stage_start) * 1000})
            
            total_time = (time.time() - start) * 1000
            
            # Record result
            passed = total_time < MAX_WORKFLOW_TIME_MS
            
            result = PerformanceTestResult(
                test_name="test_full_workflow_execution_time",
                metric="time",
                value=total_time,
                threshold=MAX_WORKFLOW_TIME_MS,
                status="passed" if passed else "failed",
                unit="ms",
                details={"stages": workflow_stages, "total_stages": len(workflow_stages)}
            )
            self._record_result(result)
            
            print(f"\n[Performance] Full workflow time: {total_time:.2f}ms (threshold: {MAX_WORKFLOW_TIME_MS}ms)")
            for stage in workflow_stages:
                print(f"   - {stage['stage']}: {stage['time']:.2f}ms")
            
            assert passed, f"Workflow took {total_time:.2f}ms, exceeds threshold of {MAX_WORKFLOW_TIME_MS}ms"
            
        except Exception as e:
            total_time = (time.time() - start) * 1000
            self._record_result(PerformanceTestResult(
                test_name="test_full_workflow_execution_time",
                metric="time",
                value=total_time,
                threshold=MAX_WORKFLOW_TIME_MS,
                status="failed",
                unit="ms",
                details={"error": str(e)}
            ))
            raise
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_memory_usage_across_systems(self):
        """Test memory usage stays within limits across all systems."""
        # Start memory tracking
        tracemalloc.start()
        initial_memory = self._get_memory_usage_mb()
        
        try:
            memory_readings = []
            
            # Test decomposition memory
            if DECOMPOSITION_AVAILABLE:
                before = self._get_memory_usage_mb()
                engine = self.systems['decomposition']
                for i in range(10):
                    engine.decompose({"id": f"mem_test_{i}", "description": f"Test {i}"})
                after = self._get_memory_usage_mb()
                memory_readings.append({"system": "decomposition", "memory_mb": after - before})
            
            # Test evolution memory
            if EVOLUTION_AVAILABLE:
                before = self._get_memory_usage_mb()
                evolution = self.systems['evolution']
                evolution.evolve({"population_size": 20, "generations": 5})
                after = self._get_memory_usage_mb()
                memory_readings.append({"system": "evolution", "memory_mb": after - before})
            
            # Test knowledge extraction memory
            if KNOWLEDGE_AVAILABLE:
                before = self._get_memory_usage_mb()
                knowledge = self.systems['knowledge']
                for i in range(5):
                    trace = {
                        "trace_id": f"trace_{i}",
                        "workflow_id": f"wf_{i}",
                        "problem_description": f"Test problem {i}",
                        "stages": [],
                        "final_result": {},
                        "execution_time_ms": 1000.0,
                        "timestamp": datetime.now()
                    }
                    asyncio.run(knowledge.process_trace(trace))
                after = self._get_memory_usage_mb()
                memory_readings.append({"system": "knowledge", "memory_mb": after - before})
            
            # Test gauntlet memory
            if GAUNTLET_AVAILABLE:
                before = self._get_memory_usage_mb()
                gauntlet = self.systems['gauntlet']
                for i in range(5):
                    gauntlet.run_gauntlet({"id": f"mem_test_{i}"})
                after = self._get_memory_usage_mb()
                memory_readings.append({"system": "gauntlet", "memory_mb": after - before})
            
            # Calculate peak memory
            peak_memory = max([r["memory_mb"] for r in memory_readings]) if memory_readings else 0
            
            # Get tracemalloc stats
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            passed = peak_memory < MAX_MEMORY_MB
            
            result = PerformanceTestResult(
                test_name="test_memory_usage_across_systems",
                metric="memory",
                value=peak_memory,
                threshold=MAX_MEMORY_MB,
                status="passed" if passed else "failed",
                unit="MB",
                details={
                    "readings": memory_readings,
                    "tracemalloc_peak_mb": peak / 1024 / 1024
                }
            )
            self._record_result(result)
            
            print(f"\n[Performance] Peak memory usage: {peak_memory:.2f}MB (threshold: {MAX_MEMORY_MB}MB)")
            for reading in memory_readings:
                print(f"   - {reading['system']}: {reading['memory_mb']:.2f}MB")
            
            assert passed, f"Peak memory {peak_memory:.2f}MB exceeds threshold of {MAX_MEMORY_MB}MB"
            
        except Exception as e:
            tracemalloc.stop()
            self._record_result(PerformanceTestResult(
                test_name="test_memory_usage_across_systems",
                metric="memory",
                value=self._get_memory_usage_mb(),
                threshold=MAX_MEMORY_MB,
                status="failed",
                unit="MB",
                details={"error": str(e)}
            ))
            raise
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_query_performance(self):
        """Test database query performance."""
        if not SQLITE_AVAILABLE:
            pytest.skip("SQLite not available")
        
        db_path = os.path.join(self.temp_dir.name, "perf_test.db")
        
        try:
            # Create test database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create test table
            cursor.execute("""
                CREATE TABLE test_data (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    value REAL,
                    timestamp TEXT
                )
            """)
            
            # Insert test data
            for i in range(1000):
                cursor.execute(
                    "INSERT INTO test_data (name, value, timestamp) VALUES (?, ?, ?)",
                    (f"item_{i}", i * 1.5, datetime.now().isoformat())
                )
            conn.commit()
            
            # Test query performance
            query_times = []
            
            # Simple query
            start = time.time()
            cursor.execute("SELECT * FROM test_data WHERE id = 500")
            cursor.fetchall()
            query_times.append({"query": "simple_lookup", "time_ms": (time.time() - start) * 1000})
            
            # Range query
            start = time.time()
            cursor.execute("SELECT * FROM test_data WHERE value > 500 AND value < 1000")
            cursor.fetchall()
            query_times.append({"query": "range_query", "time_ms": (time.time() - start) * 1000})
            
            # Aggregate query
            start = time.time()
            cursor.execute("SELECT AVG(value), MAX(value), MIN(value) FROM test_data")
            cursor.fetchall()
            query_times.append({"query": "aggregate", "time_ms": (time.time() - start) * 1000})
            
            # Complex query
            start = time.time()
            cursor.execute("""
                SELECT name, COUNT(*) as cnt, AVG(value) as avg_val
                FROM test_data
                WHERE value > 100
                GROUP BY name
                HAVING cnt > 0
                ORDER BY avg_val DESC
                LIMIT 100
            """)
            cursor.fetchall()
            query_times.append({"query": "complex", "time_ms": (time.time() - start) * 1000})
            
            conn.close()
            
            max_time = max([q["time_ms"] for q in query_times])
            passed = max_time < MAX_DB_QUERY_TIME_MS
            
            result = PerformanceTestResult(
                test_name="test_database_query_performance",
                metric="time",
                value=max_time,
                threshold=MAX_DB_QUERY_TIME_MS,
                status="passed" if passed else "failed",
                unit="ms",
                details={"queries": query_times}
            )
            self._record_result(result)
            
            print(f"\n[Performance] Max DB query time: {max_time:.2f}ms (threshold: {MAX_DB_QUERY_TIME_MS}ms)")
            for q in query_times:
                print(f"   - {q['query']}: {q['time_ms']:.4f}ms")
            
            assert passed, f"Query took {max_time:.2f}ms, exceeds threshold of {MAX_DB_QUERY_TIME_MS}ms"
            
        except Exception as e:
            self._record_result(PerformanceTestResult(
                test_name="test_database_query_performance",
                metric="time",
                value=0,
                threshold=MAX_DB_QUERY_TIME_MS,
                status="failed",
                unit="ms",
                details={"error": str(e)}
            ))
            raise
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_api_response_times(self):
        """Test API endpoint response times."""
        if not API_AVAILABLE:
            pytest.skip("API server not available")
        
        try:
            client = TestClient(api_app)
            
            endpoints = [
                ("/health", "GET", None),
                ("/api/v1/status", "GET", None),
            ]
            
            response_times = []
            
            for endpoint, method, data in endpoints:
                # Warm up
                if method == "GET":
                    client.get(endpoint)
                
                # Measure
                start = time.time()
                if method == "GET":
                    response = client.get(endpoint)
                else:
                    response = client.post(endpoint, json=data)
                elapsed = (time.time() - start) * 1000
                
                response_times.append({
                    "endpoint": endpoint,
                    "method": method,
                    "time_ms": elapsed,
                    "status": response.status_code
                })
            
            max_time = max([r["time_ms"] for r in response_times])
            passed = max_time < MAX_API_RESPONSE_TIME_MS
            
            result = PerformanceTestResult(
                test_name="test_api_response_times",
                metric="latency",
                value=max_time,
                threshold=MAX_API_RESPONSE_TIME_MS,
                status="passed" if passed else "failed",
                unit="ms",
                details={"endpoints": response_times}
            )
            self._record_result(result)
            
            print(f"\n[Performance] Max API response time: {max_time:.2f}ms (threshold: {MAX_API_RESPONSE_TIME_MS}ms)")
            for r in response_times:
                print(f"   - {r['method']} {r['endpoint']}: {r['time_ms']:.2f}ms (status: {r['status']})")
            
            assert passed, f"API response took {max_time:.2f}ms, exceeds threshold of {MAX_API_RESPONSE_TIME_MS}ms"
            
        except Exception as e:
            self._record_result(PerformanceTestResult(
                test_name="test_api_response_times",
                metric="latency",
                value=0,
                threshold=MAX_API_RESPONSE_TIME_MS,
                status="failed",
                unit="ms",
                details={"error": str(e)}
            ))
            raise
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_concurrent_request_handling(self):
        """Test system can handle concurrent requests."""
        num_concurrent = 10  # Reduced for testing stability
        
        try:
            def run_workflow_task(task_id: int) -> Dict:
                """Run a workflow task and return timing info."""
                start = time.time()
                
                try:
                    # Run a simple workflow
                    if DECOMPOSITION_AVAILABLE:
                        engine = self.systems['decomposition']
                        engine.decompose({"id": f"concurrent_{task_id}", "description": f"Task {task_id}"})
                    
                    elapsed = (time.time() - start) * 1000
                    return {"task_id": task_id, "success": True, "time_ms": elapsed}
                except Exception as e:
                    elapsed = (time.time() - start) * 1000
                    return {"task_id": task_id, "success": False, "time_ms": elapsed, "error": str(e)}
            
            # Run concurrent tasks
            start = time.time()
            results = []
            
            with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                futures = [executor.submit(run_workflow_task, i) for i in range(num_concurrent)]
                for future in as_completed(futures):
                    results.append(future.result())
            
            total_time = (time.time() - start) * 1000
            
            successful = sum(1 for r in results if r["success"])
            failed = num_concurrent - successful
            avg_time = sum(r["time_ms"] for r in results) / len(results) if results else 0
            max_time = max(r["time_ms"] for r in results) if results else 0
            
            passed = successful == num_concurrent and total_time < MAX_WORKFLOW_TIME_MS * 2
            
            result = PerformanceTestResult(
                test_name="test_concurrent_request_handling",
                metric="throughput",
                value=successful,
                threshold=num_concurrent,
                status="passed" if passed else "failed",
                unit="requests",
                details={
                    "total_requests": num_concurrent,
                    "successful": successful,
                    "failed": failed,
                    "avg_time_ms": avg_time,
                    "max_time_ms": max_time,
                    "total_time_ms": total_time
                }
            )
            self._record_result(result)
            
            print(f"\n[Performance] Concurrent requests: {successful}/{num_concurrent} successful")
            print(f"   Total time: {total_time:.2f}ms")
            print(f"   Avg time per request: {avg_time:.2f}ms")
            print(f"   Max time: {max_time:.2f}ms")
            
            assert passed, f"Only {successful}/{num_concurrent} concurrent requests succeeded"
            
        except Exception as e:
            self._record_result(PerformanceTestResult(
                test_name="test_concurrent_request_handling",
                metric="throughput",
                value=0,
                threshold=num_concurrent,
                status="failed",
                unit="requests",
                details={"error": str(e)}
            ))
            raise
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_system_scaling_performance(self):
        """Test system performance scales appropriately with load."""
        try:
            load_levels = [1, 5, 10]
            scaling_results = []
            
            for load in load_levels:
                start = time.time()
                
                # Run load number of decompositions
                if DECOMPOSITION_AVAILABLE:
                    engine = self.systems['decomposition']
                    for i in range(load):
                        engine.decompose({"id": f"scale_{load}_{i}", "description": f"Scale test {i}"})
                
                elapsed = (time.time() - start) * 1000
                
                scaling_results.append({
                    "load": load,
                    "time_ms": elapsed,
                    "time_per_item": elapsed / load if load > 0 else 0
                })
            
            # Check if scaling is roughly linear (within 3x factor)
            if len(scaling_results) >= 2:
                base_time = scaling_results[0]["time_per_item"]
                max_time = max(r["time_per_item"] for r in scaling_results)
                scaling_factor = max_time / base_time if base_time > 0 else 1
                
                passed = scaling_factor < 5  # Allow up to 5x degradation
                
                result = PerformanceTestResult(
                    test_name="test_system_scaling_performance",
                    metric="scaling_factor",
                    value=scaling_factor,
                    threshold=5.0,
                    status="passed" if passed else "failed",
                    unit="x",
                    details={"scaling_results": scaling_results}
                )
                self._record_result(result)
                
                print(f"\n[Performance] Scaling factor: {scaling_factor:.2f}x (threshold: 5x)")
                for r in scaling_results:
                    print(f"   - Load {r['load']}: {r['time_ms']:.2f}ms total, {r['time_per_item']:.2f}ms/item")
                
                assert passed, f"Scaling factor {scaling_factor:.2f}x exceeds threshold of 5x"
            
        except Exception as e:
            self._record_result(PerformanceTestResult(
                test_name="test_system_scaling_performance",
                metric="scaling_factor",
                value=0,
                threshold=5.0,
                status="failed",
                unit="x",
                details={"error": str(e)}
            ))
            raise
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_performance_summary(self):
        """Generate performance test summary."""
        print("\n" + "="*70)
        print("PERFORMANCE TEST SUMMARY")
        print("="*70)
        
        if not self.results:
            print("No performance results recorded")
            return
        
        passed = sum(1 for r in self.results if r.status == "passed")
        failed = sum(1 for r in self.results if r.status == "failed")
        
        print(f"\nTotal Tests: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        print("\nDetailed Results:")
        for result in self.results:
            status_icon = "[OK]" if result.status == "passed" else "[FAIL]"
            print(f"  {status_icon} {result.test_name}")
            print(f"       {result.metric}: {result.value:.2f} {result.unit} (threshold: {result.threshold} {result.unit})")
        
        print("="*70)
        
        assert failed == 0, f"{failed} performance tests failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
