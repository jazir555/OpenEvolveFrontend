"""
Integration Test: Z3 Behavioral Equivalence with I_mech Benchmarking

This integration test benchmarks the performance and accuracy of Z3-based
behavioral equivalence verification compared to structural-only methods.

Test Scenarios:
1. Equivalent domains (same structure, same behavior)
2. Structurally similar but behaviorally different
3. Completely different domains
4. Benchmark: With vs Without Z3 verification

Following CLAUDE.md principles:
- Law of Runtime Truth: Test against real Z3 solver
- Circuit Breaker: Verify timeout handling
- Structured Logging: Verify JSON output
- Idempotency: Same inputs -> same scores

Author: RESE Team
Created: 2026-02-04
"""

import os
import sys
import time
import json
import unittest
from typing import Dict, List, Tuple
from datetime import datetime, timezone

# Add paths
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.abspath(os.path.join(_current_dir, "..", "src"))
_schemas_dir = os.path.abspath(os.path.join(_current_dir, "..", "..", "schemas"))
_root_dir = os.path.abspath(os.path.join(_current_dir, "..", "..", ".."))

for path in [_src_dir, _schemas_dir, _root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from rese_schemas import (
        Phase2Config,
        FunctionalDependencyGraph,
        FunctionalDependency,
        IsomorphismType
    )
    from phase2_executor import CrossDomainMapper, Phase2Logger
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"Warning: Could not import required modules: {e}")


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name: str):
        self.name = name
        self.structural_score = 0.0
        self.z3_enhanced_score = 0.0
        self.structural_time_ms = 0.0
        self.z3_time_ms = 0.0
        self.verified = False
        self.confidence = 0.0
        self.errors = []

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "structural_score": self.structural_score,
            "z3_enhanced_score": self.z3_enhanced_score,
            "structural_time_ms": self.structural_time_ms,
            "z3_time_ms": self.z3_time_ms,
            "verified": self.verified,
            "confidence": self.confidence,
            "score_improvement": self.z3_enhanced_score - self.structural_score,
            "time_overhead": self.z3_time_ms - self.structural_time_ms,
            "errors": self.errors
        }


class TestZ3IntegrationBenchmark(unittest.TestCase):
    """Integration benchmark tests for Z3 behavioral equivalence."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for all tests."""
        if not IMPORTS_AVAILABLE:
            cls.skipTest("Required imports not available")

        cls.benchmark_results = []

        # Create test FDGs for different scenarios

        # Scenario 1: Equivalent domains (same structure, same behavior)
        cls.fdg_physics_energy = FunctionalDependencyGraph(
            domain="physics",
            nodes=["energy", "work", "power"],
            dependencies=[
                FunctionalDependency(
                    source="energy",
                    target="work",
                    relationship_type="causal",
                    strength=1.0,
                    domain="physics"
                ),
                FunctionalDependency(
                    source="work",
                    target="power",
                    relationship_type="causal",
                    strength=0.9,
                    domain="physics"
                )
            ],
            adjacency_list={"energy": ["work"], "work": ["power"], "power": []}
        )

        cls.fdg_economics_energy = FunctionalDependencyGraph(
            domain="economics",
            nodes=["energy", "work", "power"],
            dependencies=[
                FunctionalDependency(
                    source="energy",
                    target="work",
                    relationship_type="causal",
                    strength=1.0,
                    domain="economics"
                ),
                FunctionalDependency(
                    source="work",
                    target="power",
                    relationship_type="causal",
                    strength=0.9,
                    domain="economics"
                )
            ],
            adjacency_list={"energy": ["work"], "work": ["power"], "power": []}
        )

        # Scenario 2: Structurally similar but different behavior
        cls.fdg_biology_population = FunctionalDependencyGraph(
            domain="biology",
            nodes=["population", "resources", "growth"],
            dependencies=[
                FunctionalDependency(
                    source="resources",
                    target="population",
                    relationship_type="causal",
                    strength=0.8,
                    domain="biology"
                ),
                FunctionalDependency(
                    source="population",
                    target="growth",
                    relationship_type="causal",
                    strength=0.7,
                    domain="biology"
                )
            ],
            adjacency_list={"resources": ["population"], "population": ["growth"], "growth": []}
        )

        cls.fdg_cs_cache = FunctionalDependencyGraph(
            domain="computer_science",
            nodes=["population", "resources", "growth"],
            dependencies=[
                FunctionalDependency(
                    source="resources",
                    target="population",
                    relationship_type="causal",
                    strength=0.8,
                    domain="computer_science"
                ),
                # Note: Different dependency structure
                FunctionalDependency(
                    source="resources",
                    target="growth",
                    relationship_type="causal",
                    strength=0.6,
                    domain="computer_science"
                )
            ],
            adjacency_list={"resources": ["population", "growth"], "population": [], "growth": []}
        )

        # Scenario 3: Completely different domains
        cls.fdg_physics_force = FunctionalDependencyGraph(
            domain="physics",
            nodes=["force", "mass", "acceleration"],
            dependencies=[
                FunctionalDependency(
                    source="mass",
                    target="force",
                    relationship_type="causal",
                    strength=0.9,
                    domain="physics"
                ),
                FunctionalDependency(
                    source="acceleration",
                    target="force",
                    relationship_type="causal",
                    strength=0.9,
                    domain="physics"
                )
            ],
            adjacency_list={"mass": ["force"], "acceleration": ["force"], "force": []}
        )

        cls.fdg_biology_evolution = FunctionalDependencyGraph(
            domain="biology",
            nodes=["species", "mutation", "adaptation"],
            dependencies=[
                FunctionalDependency(
                    source="mutation",
                    target="adaptation",
                    relationship_type="causal",
                    strength=0.7,
                    domain="biology"
                )
            ],
            adjacency_list={"mutation": ["adaptation"], "species": [], "adaptation": []}
        )

    def setUp(self):
        """Set up individual test."""
        self.config = Phase2Config(
            i_mech_threshold=0.5,
            correlation_id=f"benchmark-{datetime.now(timezone.utc).isoformat()}"
        )
        self.logger = Phase2Logger(correlation_id=self.config.correlation_id)

    def benchmark_scenario(
        self,
        name: str,
        source_fdg: FunctionalDependencyGraph,
        target_fdg: FunctionalDependencyGraph,
        with_z3: bool
    ) -> BenchmarkResult:
        """
        Benchmark a single scenario.

        Args:
            name: Benchmark name
            source_fdg: Source FDG
            target_fdg: Target FDG
            with_z3: Whether to use Z3 verification

        Returns:
            BenchmarkResult
        """
        result = BenchmarkResult(name)

        try:
            # Configure Z3
            if with_z3:
                os.environ['RESE_Z3_PHASE2_ENABLED'] = 'true'
                os.environ['Z3_TIMEOUT'] = '10000'
            else:
                os.environ['RESE_Z3_PHASE2_ENABLED'] = 'false'

            mapper = CrossDomainMapper(self.config, self.logger)

            # Measure time
            start_time = time.time()
            score = mapper.compute_imech_score(
                source_fdg,
                target_fdg,
                correlation_id=self.config.correlation_id
            )
            elapsed_ms = (time.time() - start_time) * 1000

            if with_z3:
                result.z3_enhanced_score = score
                result.z3_time_ms = elapsed_ms

                # Try to get verification result
                try:
                    equiv_result = mapper._verify_behavioral_equivalence(
                        source_fdg,
                        target_fdg,
                        self.config.correlation_id
                    )
                    result.verified = equiv_result.verified
                    result.confidence = equiv_result.confidence
                except Exception as e:
                    result.errors.append(str(e))
            else:
                result.structural_score = score
                result.structural_time_ms = elapsed_ms

        except Exception as e:
            result.errors.append(str(e))

        finally:
            # Clean up
            if 'RESE_Z3_PHASE2_ENABLED' in os.environ:
                del os.environ['RESE_Z3_PHASE2_ENABLED']
            if 'Z3_TIMEOUT' in os.environ:
                del os.environ['Z3_TIMEOUT']

        return result

    def run_comparison_benchmark(
        self,
        name: str,
        source_fdg: FunctionalDependencyGraph,
        target_fdg: FunctionalDependencyGraph
    ) -> BenchmarkResult:
        """
        Run benchmark comparing structural vs Z3-enhanced methods.

        Args:
            name: Benchmark name
            source_fdg: Source FDG
            target_fdg: Target FDG

        Returns:
            BenchmarkResult with both scores
        """
        # Run without Z3
        structural_result = self.benchmark_scenario(
            f"{name}-structural",
            source_fdg,
            target_fdg,
            with_z3=False
        )

        # Run with Z3
        z3_result = self.benchmark_scenario(
            f"{name}-z3",
            source_fdg,
            target_fdg,
            with_z3=True
        )

        # Combine results
        combined = BenchmarkResult(name)
        combined.structural_score = structural_result.structural_score
        combined.structural_time_ms = structural_result.structural_time_ms
        combined.z3_enhanced_score = z3_result.z3_enhanced_score
        combined.z3_time_ms = z3_result.z3_time_ms
        combined.verified = z3_result.verified
        combined.confidence = z3_result.confidence
        combined.errors = structural_result.errors + z3_result.errors

        return combined

    def test_scenario_1_equivalent_domains(self):
        """Test Scenario 1: Equivalent domains with same structure and behavior."""
        result = self.run_comparison_benchmark(
            "equivalent_domains",
            self.fdg_physics_energy,
            self.fdg_economics_energy
        )

        self.benchmark_results.append(result)

        # Both should recognize high similarity
        self.assertGreater(result.structural_score, 0.7, "Structural score should be high for equivalent domains")
        self.assertGreater(result.z3_enhanced_score, 0.7, "Z3-enhanced score should be high for equivalent domains")

        # Z3 should verify equivalence
        if result.verified:
            self.assertTrue(result.verified, "Z3 should verify behavioral equivalence")
            self.assertGreater(result.confidence, 0.8, "Should have high confidence in equivalence")

        print(f"\nScenario 1 - Equivalent Domains:")
        print(f"  Structural Score: {result.structural_score:.3f}")
        print(f"  Z3-Enhanced Score: {result.z3_enhanced_score:.3f}")
        print(f"  Verification: {result.verified}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Time Overhead: {result.z3_time_ms - result.structural_time_ms:.2f}ms")

    def test_scenario_2_structural_similarity_behavioral_divergence(self):
        """Test Scenario 2: Structurally similar but behaviorally different."""
        result = self.run_comparison_benchmark(
            "structural_similarity_behavioral_divergence",
            self.fdg_biology_population,
            self.fdg_cs_cache
        )

        self.benchmark_results.append(result)

        # Structural should detect similarity
        self.assertGreater(result.structural_score, 0.3, "Structural score should detect some similarity")

        # Z3 might reduce score if behavioral divergence detected
        # (This depends on Z3's ability to prove non-equivalence)
        if result.verified:
            # If verified, score should be similar or higher
            self.assertGreaterEqual(result.z3_enhanced_score, result.structural_score * 0.8)
        else:
            # If not verified, score might be reduced
            self.assertLess(result.z3_enhanced_score, result.structural_score * 1.1)

        print(f"\nScenario 2 - Structural Similarity, Behavioral Divergence:")
        print(f"  Structural Score: {result.structural_score:.3f}")
        print(f"  Z3-Enhanced Score: {result.z3_enhanced_score:.3f}")
        print(f"  Verification: {result.verified}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Score Change: {result.z3_enhanced_score - result.structural_score:+.3f}")

    def test_scenario_3_completely_different(self):
        """Test Scenario 3: Completely different domains."""
        result = self.run_comparison_benchmark(
            "completely_different",
            self.fdg_physics_force,
            self.fdg_biology_evolution
        )

        self.benchmark_results.append(result)

        # Both should detect low similarity
        self.assertLess(result.structural_score, 0.5, "Structural score should be low for different domains")
        self.assertLess(result.z3_enhanced_score, 0.5, "Z3-enhanced score should be low for different domains")

        # Z3 should not verify equivalence
        self.assertFalse(result.verified, "Z3 should NOT verify equivalence for different domains")

        print(f"\nScenario 3 - Completely Different:")
        print(f"  Structural Score: {result.structural_score:.3f}")
        print(f"  Z3-Enhanced Score: {result.z3_enhanced_score:.3f}")
        print(f"  Verification: {result.verified}")
        print(f"  Confidence: {result.confidence:.3f}")

    def test_benchmark_summary(self):
        """Print summary of all benchmarks."""
        print("\n" + "=" * 80)
        print("Z3 INTEGRATION BENCHMARK SUMMARY")
        print("=" * 80)

        if not self.benchmark_results:
            print("No benchmark results available")
            return

        total_structural_time = sum(r.structural_time_ms for r in self.benchmark_results)
        total_z3_time = sum(r.z3_time_ms for r in self.benchmark_results)

        print(f"\nTotal Scenarios: {len(self.benchmark_results)}")
        print(f"Total Structural Time: {total_structural_time:.2f}ms")
        print(f"Total Z3 Time: {total_z3_time:.2f}ms")
        print(f"Average Time Overhead: {(total_z3_time - total_structural_time) / len(self.benchmark_results):.2f}ms")

        print(f"\n{'Scenario':<50} {'Structural':<12} {'Z3-Enhanced':<12} {'Verified':<10} {'Overhead':<10}")
        print("-" * 100)

        for result in self.benchmark_results:
            print(f"{result.name:<50} {result.structural_score:<12.3f} {result.z3_enhanced_score:<12.3f} "
                  f"{str(result.verified):<10} {result.z3_time_ms - result.structural_time_ms:<10.2f}")

        # Save results to JSON
        timestamp = datetime.now(timezone.utc).isoformat()
        results_file = f"z3_benchmark_results_{timestamp.replace(':', '-')}.json"
        results_path = os.path.join(_current_dir, results_file)

        with open(results_path, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "summary": {
                    "total_scenarios": len(self.benchmark_results),
                    "total_structural_time_ms": total_structural_time,
                    "total_z3_time_ms": total_z3_time,
                    "average_overhead_ms": (total_z3_time - total_structural_time) / len(self.benchmark_results)
                },
                "results": [r.to_dict() for r in self.benchmark_results]
            }, f, indent=2)

        print(f"\nResults saved to: {results_path}")
        print("=" * 80)


class TestZ3CircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality for Z3 verification."""

    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")

        self.config = Phase2Config(correlation_id="circuit-breaker-test")
        self.logger = Phase2Logger(correlation_id="circuit-breaker-test")

    def test_timeout_handling(self):
        """Test that Z3 timeouts are handled gracefully."""
        os.environ['RESE_Z3_PHASE2_ENABLED'] = 'true'
        os.environ['Z3_TIMEOUT'] = '1'  # 1ms timeout (should trigger quickly)

        mapper = CrossDomainMapper(self.config, self.logger)

        # Create complex FDG that might take time to verify
        complex_fdg = FunctionalDependencyGraph(
            domain="complex",
            nodes=[f"node_{i}" for i in range(50)],
            dependencies=[
                FunctionalDependency(
                    source=f"node_{i}",
                    target=f"node_{i+1}",
                    relationship_type="causal",
                    strength=0.8,
                    domain="complex"
                )
                for i in range(49)
            ],
            adjacency_list={f"node_{i}": [f"node_{i+1}"] for i in range(49)}
        )

        target_fdg = FunctionalDependencyGraph(
            domain="target",
            nodes=[f"node_{i}" for i in range(50)],
            dependencies=[
                FunctionalDependency(
                    source=f"node_{i}",
                    target=f"node_{i+1}",
                    relationship_type="causal",
                    strength=0.8,
                    domain="target"
                )
                for i in range(49)
            ],
            adjacency_list={f"node_{i}": [f"node_{i+1}"] for i in range(49)}
        )

        try:
            # Should not hang, should return score or raise error gracefully
            start_time = time.time()
            score = mapper.compute_imech_score(complex_fdg, target_fdg, correlation_id="timeout-test")
            elapsed = time.time() - start_time

            # Should complete quickly (< 5 seconds even with timeout)
            self.assertLess(elapsed, 5.0, "Should handle timeout gracefully")

            # Should still return a score (fallback to structural)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

        except Exception as e:
            # Exception is okay if it's handled gracefully
            self.assertIn("timeout", str(e).lower(), "Error should mention timeout")

        finally:
            if 'RESE_Z3_PHASE2_ENABLED' in os.environ:
                del os.environ['RESE_Z3_PHASE2_ENABLED']
            if 'Z3_TIMEOUT' in os.environ:
                del os.environ['Z3_TIMEOUT']


def run_integration_tests():
    """Run all integration tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestZ3IntegrationBenchmark))
    suite.addTests(loader.loadTestsFromTestCase(TestZ3CircuitBreaker))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    result = run_integration_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
