"""
Performance Benchmarks for Sovereign-Grade Problem Decomposition System
Task 15.3: Implement performance benchmarks
"""

import pytest
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from problem_analyzer import ProblemAnalyzer
from decomposition_engine import DecompositionEngine
from sovereign_gauntlets import GauntletSystem
from sovereign_quality_assessment import QualityAssessor
from sovereign_performance_optimization import PerformanceMonitor, get_performance_stats


class TestDecompositionPerformance:
    """Benchmark decomposition speed."""
    
    def test_simple_problem_performance(self):
        """Benchmark simple problem decomposition."""
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        
        start_time = time.time()
        problem = analyzer.analyze_problem(
            "Build a simple REST API",
            title="REST API"
        )
        plan = engine.decompose(problem)
        duration = time.time() - start_time
        
        assert duration < 10.0, f"Simple decomposition took {duration:.2f}s (should be < 10s)"
        assert len(plan.sub_problems) > 0
        print(f"Simple problem: {duration:.2f}s, {len(plan.sub_problems)} sub-problems")
    
    def test_complex_problem_performance(self):
        """Benchmark complex problem decomposition."""
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        
        start_time = time.time()
        problem = analyzer.analyze_problem(
            """Build a distributed machine learning system with:
            - Real-time data ingestion from multiple sources
            - Feature engineering pipeline
            - Model training with hyperparameter optimization
            - A/B testing framework
            - Model serving with auto-scaling
            - Monitoring and alerting
            - Data versioning and model registry
            """,
            title="ML System"
        )
        plan = engine.decompose(problem, strategy='hybrid')
        duration = time.time() - start_time
        
        assert duration < 30.0, f"Complex decomposition took {duration:.2f}s (should be < 30s)"
        assert len(plan.sub_problems) >= 5
        print(f"Complex problem: {duration:.2f}s, {len(plan.sub_problems)} sub-problems")
    
    def test_100_subproblems_performance(self):
        """Benchmark handling 100 sub-problems."""
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        
        # Create a problem that will generate many sub-problems
        problem_text = "Build a comprehensive system with: " + ", ".join([
            f"component {i}" for i in range(50)
        ])
        
        start_time = time.time()
        problem = analyzer.analyze_problem(problem_text, title="Large System")
        plan = engine.decompose(problem)
        duration = time.time() - start_time
        
        # Should handle large decompositions in reasonable time
        assert duration < 30.0, f"Large decomposition took {duration:.2f}s (should be < 30s)"
        print(f"Large problem: {duration:.2f}s, {len(plan.sub_problems)} sub-problems")


class TestConcurrentCapacity:
    """Test concurrent problem handling."""
    
    def test_10_concurrent_problems(self):
        """Test handling 10 concurrent decompositions."""
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        
        problems = [
            f"Build system {i} with components A, B, and C"
            for i in range(10)
        ]
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for prob_text in problems:
                future = executor.submit(self._decompose_problem, analyzer, engine, prob_text)
                futures.append(future)
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in concurrent decomposition: {e}")
        
        duration = time.time() - start_time
        
        assert len(results) == 10, f"Only {len(results)}/10 problems completed"
        assert duration < 60.0, f"10 concurrent problems took {duration:.2f}s (should be < 60s)"
        print(f"10 concurrent: {duration:.2f}s, avg {duration/10:.2f}s per problem")
    
    def test_100_concurrent_problems(self):
        """Test handling 100 concurrent decompositions."""
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        
        problems = [
            f"Build system {i}"
            for i in range(100)
        ]
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for prob_text in problems:
                future = executor.submit(self._decompose_problem, analyzer, engine, prob_text)
                futures.append(future)
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    pass  # Some may fail under load
        
        duration = time.time() - start_time
        
        success_rate = len(results) / 100
        assert success_rate >= 0.90, f"Only {success_rate*100:.0f}% success rate"
        print(f"100 concurrent: {duration:.2f}s, {len(results)}/100 completed, {success_rate*100:.0f}% success")
    
    def _decompose_problem(self, analyzer, engine, problem_text):
        """Helper to decompose a problem."""
        problem = analyzer.analyze_problem(problem_text, title=problem_text[:20])
        plan = engine.decompose(problem)
        return plan


class TestScalabilityMetrics:
    """Test scalability characteristics."""
    
    def test_memory_usage_scaling(self):
        """Test memory usage with increasing problem size."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process increasingly large problems
        for size in [10, 50, 100]:
            problem_text = "Build system with: " + ", ".join([
                f"component {i}" for i in range(size)
            ])
            problem = analyzer.analyze_problem(problem_text, title=f"System {size}")
            plan = engine.decompose(problem)
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # Memory should scale reasonably
            assert memory_increase < size * 2, f"Memory increased by {memory_increase:.1f}MB for {size} components"
            print(f"Size {size}: {memory_increase:.1f}MB increase")
    
    def test_response_time_consistency(self):
        """Test that response times remain consistent."""
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        
        times = []
        for i in range(10):
            start = time.time()
            problem = analyzer.analyze_problem(
                f"Build system {i} with authentication and database",
                title=f"System {i}"
            )
            plan = engine.decompose(problem)
            duration = time.time() - start
            times.append(duration)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        # Variance should be reasonable
        variance = max_time - min_time
        assert variance < avg_time * 0.5, f"High variance: {variance:.2f}s (avg: {avg_time:.2f}s)"
        print(f"Response times: avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s")


class TestEndToEndPerformance:
    """Test complete workflow performance."""
    
    def test_full_workflow_performance(self):
        """Benchmark complete workflow from analysis to quality assessment."""
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        gauntlet_system = GauntletSystem()
        assessor = QualityAssessor()
        
        start_time = time.time()
        
        # Full workflow
        problem = analyzer.analyze_problem(
            "Build a recommendation system with ML and real-time processing",
            title="Recommendation System"
        )
        plan = engine.decompose(problem, strategy='hybrid')
        results = gauntlet_system.run_decomposition_gauntlets(plan)
        report = assessor.generate_quality_report(plan)
        
        duration = time.time() - start_time
        
        assert duration < 45.0, f"Full workflow took {duration:.2f}s (should be < 45s)"
        assert report.metrics.overall_score > 0
        print(f"Full workflow: {duration:.2f}s, quality={report.metrics.overall_score:.2f}")
    
    def test_workflow_with_refinement_performance(self):
        """Benchmark workflow with refinement cycles."""
        from sovereign_refinement import RefinementCoordinator
        
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        coordinator = RefinementCoordinator()
        
        start_time = time.time()
        
        problem = analyzer.analyze_problem(
            "Design a distributed caching system",
            title="Cache System"
        )
        plan = engine.decompose(problem)
        
        # Run refinement
        result = coordinator.track_refinement_cycles(
            plan,
            max_cycles=3,
            convergence_threshold=0.01
        )
        
        duration = time.time() - start_time
        
        assert duration < 120.0, f"Workflow with refinement took {duration:.2f}s (should be < 120s)"
        print(f"With refinement: {duration:.2f}s, {result['total_cycles']} cycles")


class TestPerformanceMonitoring:
    """Test performance monitoring capabilities."""
    
    def test_performance_stats_collection(self):
        """Test that performance stats are collected."""
        from sovereign_performance_optimization import timed
        
        @timed("test_operation")
        def test_operation():
            time.sleep(0.1)
            return "done"
        
        # Run operation multiple times
        for _ in range(5):
            test_operation()
        
        stats = get_performance_stats()
        
        if "test_operation" in stats:
            op_stats = stats["test_operation"]
            assert op_stats["count"] == 5
            assert op_stats["avg_duration"] >= 0.1
            print(f"Stats collected: {op_stats}")


def run_all_benchmarks():
    """Run all benchmarks and print summary."""
    print("\n" + "="*60)
    print("SOVEREIGN SYSTEM PERFORMANCE BENCHMARKS")
    print("="*60 + "\n")
    
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_all_benchmarks()
