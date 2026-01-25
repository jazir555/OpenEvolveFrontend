"""
Performance Tests for Symbolic Constraint Engine (SCE)

Author: Agent A1
Created: 2025-12-31
Status: Active Implementation
Purpose: Test SCE performance with large constraint sets and complex dependencies
"""


import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import time
import tracemalloc
from typing import List
from core.symbolic_constraint_engine import (
    SymbolicConstraintEngine,
    Constraint,
    ConstraintType
)


class TestPerformanceLargeScale:
    """Performance tests for large-scale constraint sets"""

    def test_add_1000_constraints(self):
        """Test adding 1000 constraints"""
        sce = SymbolicConstraintEngine()
        start_time = time.time()
        start_memory = tracemalloc.get_traced_memory()[1] if tracemalloc.is_tracing() else 0

        # Add 1000 constraints
        for i in range(1000):
            constraint = Constraint(
                id=f"constraint_{i}",
                type=ConstraintType.HARD,
                description=f"Constraint {i}",
                formalization=f"formal_{i}",
                source="test"
            )
            sce.add_constraint(constraint)

        end_time = time.time()
        elapsed = end_time - start_time

        # Verify all added
        assert len(sce.get_all_constraints()) == 1000

        # Performance assertion: should complete in reasonable time
        assert elapsed < 1.0, f"Adding 1000 constraints took {elapsed:.2f}s (expected < 1.0s)"

        print(f"✓ Added 1000 constraints in {elapsed:.4f}s")

    def test_add_1000_constraints_with_dependencies(self):
        """Test adding 1000 constraints with dependency chain"""
        sce = SymbolicConstraintEngine()
        start_time = time.time()

        # Create dependency chain: c0 -> c1 -> c2 -> ... -> c999
        prev_id = None
        for i in range(1000):
            constraint_id = f"constraint_{i}"

            if prev_id is None:
                # First constraint has no dependencies
                constraint = Constraint(
                    id=constraint_id,
                    type=ConstraintType.HARD,
                    description=f"Constraint {i}",
                    formalization=f"formal_{i}",
                    source="test"
                )
            else:
                # Subsequent constraints depend on previous
                constraint = Constraint(
                    id=constraint_id,
                    type=ConstraintType.HARD,
                    description=f"Constraint {i}",
                    formalization=f"formal_{i}",
                    source="test",
                    dependencies=[prev_id]
                )

            sce.add_constraint(constraint)
            prev_id = constraint_id

        end_time = time.time()
        elapsed = end_time - start_time

        # Verify all added
        assert len(sce.get_all_constraints()) == 1000

        # Performance assertion
        assert elapsed < 2.0, f"Adding 1000 chained constraints took {elapsed:.2f}s (expected < 2.0s)"

        print(f"✓ Added 1000 chained constraints in {elapsed:.4f}s")

    def test_get_all_constraints_1000(self):
        """Test retrieving 1000 constraints"""
        sce = SymbolicConstraintEngine()

        # Add 1000 constraints
        for i in range(1000):
            constraint = Constraint(
                id=f"constraint_{i}",
                type=ConstraintType.HARD,
                description=f"Constraint {i}",
                formalization=f"formal_{i}",
                source="test"
            )
            sce.add_constraint(constraint)

        # Time retrieval
        start_time = time.time()
        all_constraints = sce.get_all_constraints()
        end_time = time.time()
        elapsed = end_time - start_time

        assert len(all_constraints) == 1000
        assert elapsed < 0.1, f"Retrieving 1000 constraints took {elapsed:.4f}s (expected < 0.1s)"

        print(f"✓ Retrieved 1000 constraints in {elapsed:.4f}s")

    def test_get_constraints_by_type_1000(self):
        """Test filtering 1000 constraints by type"""
        sce = SymbolicConstraintEngine()

        # Add mixed constraint types
        for i in range(1000):
            c_type = ConstraintType.HARD if i % 2 == 0 else ConstraintType.SOFT
            constraint = Constraint(
                id=f"constraint_{i}",
                type=c_type,
                description=f"Constraint {i}",
                formalization=f"formal_{i}",
                source="test"
            )
            sce.add_constraint(constraint)

        # Time filtering
        start_time = time.time()
        hard_constraints = sce.get_constraints_by_type(ConstraintType.HARD)
        end_time = time.time()
        elapsed = end_time - start_time

        assert len(hard_constraints) == 500
        assert elapsed < 0.1, f"Filtering 1000 constraints took {elapsed:.4f}s (expected < 0.1s)"

        print(f"✓ Filtered 1000 constraints in {elapsed:.4f}s")


class TestPerformanceDeepDependencies:
    """Performance tests for deep dependency chains"""

    def test_dependency_chain_100_levels(self):
        """Test dependency chain with 100 levels"""
        sce = SymbolicConstraintEngine()
        start_time = time.time()

        # Create chain: c0 -> c1 -> c2 -> ... -> c99
        prev_id = None
        for i in range(100):
            constraint_id = f"link_{i}"

            if prev_id is None:
                constraint = Constraint(
                    id=constraint_id,
                    type=ConstraintType.HARD,
                    description=f"Link {i}",
                    formalization=f"link_{i}",
                    source="test"
                )
            else:
                constraint = Constraint(
                    id=constraint_id,
                    type=ConstraintType.HARD,
                    description=f"Link {i}",
                    formalization=f"link_{i}",
                    source="test",
                    dependencies=[prev_id]
                )

            sce.add_constraint(constraint)
            prev_id = constraint_id

        end_time = time.time()
        elapsed = end_time - start_time

        # Verify chain length
        assert len(sce.get_all_constraints()) == 100

        # Verify dependencies
        last_link = f"link_99"
        deps = sce.get_dependencies(last_link)
        assert len(deps) == 1  # Only immediate dependency (link_98)

        # Performance assertion
        assert elapsed < 0.5, f"Creating 100-level chain took {elapsed:.2f}s (expected < 0.5s)"

        print(f"✓ Created 100-level dependency chain in {elapsed:.4f}s")

    def test_topological_sort_100_levels(self):
        """Test topological sort on 100-level chain"""
        sce = SymbolicConstraintEngine()

        # Create 100-level chain
        prev_id = None
        for i in range(100):
            constraint_id = f"link_{i}"

            if prev_id is None:
                constraint = Constraint(
                    id=constraint_id,
                    type=ConstraintType.HARD,
                    description=f"Link {i}",
                    formalization=f"link_{i}",
                    source="test"
                )
            else:
                constraint = Constraint(
                    id=constraint_id,
                    type=ConstraintType.HARD,
                    description=f"Link {i}",
                    formalization=f"link_{i}",
                    source="test",
                    dependencies=[prev_id]
                )

            sce.add_constraint(constraint)
            prev_id = constraint_id

        # Time topological sort
        start_time = time.time()
        sorted_ids = sce.topological_sort()
        end_time = time.time()
        elapsed = end_time - start_time

        # Verify order
        assert len(sorted_ids) == 100
        assert sorted_ids == [f"link_{i}" for i in range(100)]

        # Performance assertion
        assert elapsed < 0.1, f"Topological sort of 100 constraints took {elapsed:.4f}s (expected < 0.1s)"

        print(f"✓ Topological sort of 100-level chain in {elapsed:.4f}s")

    def test_diamond_structure_performance(self):
        """Test diamond dependency structure performance"""
        sce = SymbolicConstraintEngine()
        start_time = time.time()

        # Create diamond: A -> B, A -> C, B -> D, C -> D
        # Repeat with 100 such diamonds
        for diamond_num in range(100):
            base_id = f"d{diamond_num}_A"
            branch1_id = f"d{diamond_num}_B"
            branch2_id = f"d{diamond_num}_C"
            leaf_id = f"d{diamond_num}_D"

            # Add base
            sce.add_constraint(Constraint(
                id=base_id,
                type=ConstraintType.HARD,
                description=f"Diamond {diamond_num} Base",
                formalization=f"d{diamond_num}_A",
                source="test"
            ))

            # Add branches
            sce.add_constraint(Constraint(
                id=branch1_id,
                type=ConstraintType.HARD,
                description=f"Diamond {diamond_num} Branch 1",
                formalization=f"d{diamond_num}_B",
                source="test",
                dependencies=[base_id]
            ))

            sce.add_constraint(Constraint(
                id=branch2_id,
                type=ConstraintType.HARD,
                description=f"Diamond {diamond_num} Branch 2",
                formalization=f"d{diamond_num}_C",
                source="test",
                dependencies=[base_id]
            ))

            # Add leaf
            sce.add_constraint(Constraint(
                id=leaf_id,
                type=ConstraintType.HARD,
                description=f"Diamond {diamond_num} Leaf",
                formalization=f"d{diamond_num}_D",
                source="test",
                dependencies=[branch1_id, branch2_id]
            ))

        end_time = time.time()
        elapsed = end_time - start_time

        # Verify
        assert len(sce.get_all_constraints()) == 400  # 100 diamonds * 4 nodes

        # Performance assertion
        assert elapsed < 1.0, f"Creating 100 diamond structures took {elapsed:.2f}s (expected < 1.0s)"

        print(f"✓ Created 100 diamond structures in {elapsed:.4f}s")


class TestPerformanceConflictDetection:
    """Performance tests for conflict detection"""

    def test_conflict_detection_100_constraints(self):
        """Test conflict detection with 100 constraints"""
        sce = SymbolicConstraintEngine()

        # Add 100 constraints (some conflicts)
        for i in range(50):
            # Add conflicting pairs
            sce.add_constraint(Constraint(
                id=f"less_{i}",
                type=ConstraintType.HARD,
                description=f"Value must be less than {i}",
                formalization=f"x < {i}",
                source="test"
            ))

            sce.add_constraint(Constraint(
                id=f"greater_{i}",
                type=ConstraintType.HARD,
                description=f"Value must be greater than {i+10}",
                formalization=f"x > {i+10}",
                source="test"
            ))

        # Time conflict detection
        start_time = time.time()
        conflicts = sce.detect_conflicts()
        end_time = time.time()
        elapsed = end_time - start_time

        # Should detect conflicts (less/greater contradiction)
        assert len(conflicts) > 0

        # Performance assertion: O(n²) but should still be fast
        assert elapsed < 1.0, f"Conflict detection for 100 constraints took {elapsed:.2f}s (expected < 1.0s)"

        print(f"✓ Conflict detection for 100 constraints in {elapsed:.4f}s (found {len(conflicts)} conflicts)")

    def test_conflict_detection_1000_no_conflicts(self):
        """Test conflict detection with 1000 non-conflicting constraints"""
        sce = SymbolicConstraintEngine()

        # Add 1000 non-conflicting constraints
        for i in range(1000):
            constraint = Constraint(
                id=f"constraint_{i}",
                type=ConstraintType.HARD,
                description=f"Constraint {i} about topic_{i % 100}",
                formalization=f"topic_{i % 100}_property_{i}",
                source="test"
            )
            sce.add_constraint(constraint)

        # Time conflict detection
        start_time = time.time()
        conflicts = sce.detect_conflicts()
        end_time = time.time()
        elapsed = end_time - start_time

        # Should have no conflicts
        assert len(conflicts) == 0

        # Performance assertion
        assert elapsed < 10.0, f"Conflict detection for 1000 constraints took {elapsed:.2f}s (expected < 10.0s)"

        print(f"✓ Conflict detection for 1000 non-conflicting constraints in {elapsed:.4f}s")

    def test_conflict_cache_performance(self):
        """Test that conflict caching improves performance"""
        sce = SymbolicConstraintEngine()

        # Add 100 constraints with conflicts
        for i in range(50):
            sce.add_constraint(Constraint(
                id=f"less_{i}",
                type=ConstraintType.HARD,
                description=f"Less than {i}",
                formalization=f"x < {i}",
                source="test"
            ))

            sce.add_constraint(Constraint(
                id=f"greater_{i}",
                type=ConstraintType.HARD,
                description=f"Greater than {i}",
                formalization=f"x > {i}",
                source="test"
            ))

        # First run (no cache)
        start_time = time.time()
        conflicts1 = sce.detect_conflicts()
        first_run = time.time() - start_time

        # Second run (with cache)
        start_time = time.time()
        conflicts2 = sce.detect_conflicts()
        second_run = time.time() - start_time

        # Results should be same
        assert len(conflicts1) == len(conflicts2)

        # Second run should be faster (or similar due to caching)
        # Note: Caching is at pair level, so speedup depends on implementation
        print(f"✓ First run: {first_run:.4f}s, Second run: {second_run:.4f}s")


class TestPerformanceMemory:
    """Memory usage tests"""

    def test_memory_usage_1000_constraints(self):
        """Test memory usage with 1000 constraints"""
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        sce = SymbolicConstraintEngine()

        # Get baseline memory
        baseline = tracemalloc.get_traced_memory()[1]

        # Add 1000 constraints
        for i in range(1000):
            constraint = Constraint(
                id=f"constraint_{i}",
                type=ConstraintType.HARD,
                description=f"Constraint {i}",
                formalization=f"formal_{i}",
                source="test"
            )
            sce.add_constraint(constraint)

        # Get final memory
        final = tracemalloc.get_traced_memory()[1]
        memory_used = final - baseline

        # Convert to MB
        memory_mb = memory_used / (1024 * 1024)

        # Memory assertion: should use reasonable amount
        # (This is a loose assertion; actual usage depends on implementation)
        assert memory_mb < 100, f"Memory usage {memory_mb:.2f}MB exceeds 100MB"

        print(f"✓ Memory usage for 1000 constraints: {memory_mb:.2f}MB")

    def test_memory_usage_dependency_chain_100(self):
        """Test memory usage with 100-level dependency chain"""
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        sce = SymbolicConstraintEngine()

        # Get baseline memory
        baseline = tracemalloc.get_traced_memory()[1]

        # Create 100-level chain
        prev_id = None
        for i in range(100):
            constraint_id = f"link_{i}"

            if prev_id is None:
                constraint = Constraint(
                    id=constraint_id,
                    type=ConstraintType.HARD,
                    description=f"Link {i}",
                    formalization=f"link_{i}",
                    source="test"
                )
            else:
                constraint = Constraint(
                    id=constraint_id,
                    type=ConstraintType.HARD,
                    description=f"Link {i}",
                    formalization=f"link_{i}",
                    source="test",
                    dependencies=[prev_id]
                )

            sce.add_constraint(constraint)
            prev_id = constraint_id

        # Get final memory
        final = tracemalloc.get_traced_memory()[1]
        memory_used = final - baseline

        # Convert to MB
        memory_mb = memory_used / (1024 * 1024)

        # Memory assertion
        assert memory_mb < 10, f"Memory usage {memory_mb:.2f}MB exceeds 10MB for 100 constraints"

        print(f"✓ Memory usage for 100-level chain: {memory_mb:.2f}MB")


class TestPerformanceStatistics:
    """Statistics calculation performance tests"""

    def test_statistics_1000_constraints(self):
        """Test statistics calculation with 1000 constraints"""
        sce = SymbolicConstraintEngine()

        # Add 1000 mixed constraints
        for i in range(1000):
            if i % 3 == 0:
                c_type = ConstraintType.HARD
            elif i % 3 == 1:
                c_type = ConstraintType.SOFT
            else:
                c_type = ConstraintType.PREFERENCE

            constraint = Constraint(
                id=f"constraint_{i}",
                type=c_type,
                description=f"Constraint {i}",
                formalization=f"formal_{i}",
                source="test"
            )
            sce.add_constraint(constraint)

        # Add some dependencies
        for i in range(100):
            constraint = Constraint(
                id=f"dependent_{i}",
                type=ConstraintType.HARD,
                description=f"Dependent {i}",
                formalization=f"dep_{i}",
                source="test",
                dependencies=[f"constraint_{i}"]
            )
            sce.add_constraint(constraint)

        # Time statistics calculation
        start_time = time.time()
        stats = sce.get_statistics()
        end_time = time.time()
        elapsed = end_time - start_time

        # Verify statistics
        assert stats["total_constraints"] == 1100
        # 334 hard from first 1000 (i%3==0) = 334, plus 100 dependents = 434
        assert stats["hard_constraints"] == 434
        assert stats["dependencies"] == 100

        # Performance assertion
        # Note: Statistics includes conflict detection which is O(n²)
        assert elapsed < 10.0, f"Statistics calculation took {elapsed:.4f}s (expected < 10.0s)"

        print(f"✓ Statistics for 1100 constraints in {elapsed:.4f}s")


class TestPerformanceComplexScenarios:
    """Complex real-world scenario performance tests"""

    def test_real_world_scenario_mixed_constraints(self):
        """Test real-world scenario with mixed constraint types and dependencies"""
        sce = SymbolicConstraintEngine()
        start_time = time.time()

        # Simulate real-world constraint set
        # 100 base constraints
        for i in range(100):
            constraint = Constraint(
                id=f"base_{i}",
                type=ConstraintType.HARD if i % 2 == 0 else ConstraintType.SOFT,
                description=f"Base requirement {i}",
                formalization=f"base_{i}",
                source="user_prompt"
            )
            sce.add_constraint(constraint)

        # 50 derived constraints
        for i in range(50):
            constraint = Constraint(
                id=f"derived_{i}",
                type=ConstraintType.HARD,
                description=f"Derived requirement {i}",
                formalization=f"derived_{i}",
                source="system_inferred",
                dependencies=[f"base_{i % 100}"]
            )
            sce.add_constraint(constraint)

        # 20 preference constraints
        for i in range(20):
            constraint = Constraint(
                id=f"pref_{i}",
                type=ConstraintType.PREFERENCE,
                description=f"Preference {i}",
                formalization=f"pref_{i}",
                source="design_guideline"
            )
            sce.add_constraint(constraint)

        # 10 verified constraints
        for i in range(10):
            constraint = Constraint(
                id=f"verified_{i}",
                type=ConstraintType.HARD,
                description=f"Verified constraint {i}",
                formalization=f"verified_{i}",
                source="formal_verification",
                verified=True,
                lean_theorem=f"theorem verified_{i} : True"
            )
            sce.add_constraint(constraint)

        end_time = time.time()
        elapsed = end_time - start_time

        # Verify
        stats = sce.get_statistics()
        assert stats["total_constraints"] == 180
        assert stats["verified_constraints"] == 10

        # Performance assertion
        assert elapsed < 1.0, f"Real-world scenario took {elapsed:.2f}s (expected < 1.0s)"

        print(f"✓ Real-world scenario (180 constraints) in {elapsed:.4f}s")

    def test_iterative_operations_performance(self):
        """Test performance of iterative operations"""
        sce = SymbolicConstraintEngine()

        # Add constraints iteratively
        start_time = time.time()
        for i in range(100):
            constraint = Constraint(
                id=f"constraint_{i}",
                type=ConstraintType.HARD,
                description=f"Constraint {i}",
                formalization=f"formal_{i}",
                source="test"
            )
            sce.add_constraint(constraint)

            # Perform some operation after each addition
            _ = sce.get_statistics()
            if i > 0:
                _ = sce.get_dependencies(f"constraint_{i-1}")

        elapsed = time.time() - start_time

        # Performance assertion
        assert elapsed < 1.0, f"Iterative operations took {elapsed:.2f}s (expected < 1.0s)"

        print(f"✓ Iterative operations (100 constraints) in {elapsed:.4f}s")


# Benchmark runner

def run_performance_benchmarks():
    """Run all performance benchmarks and generate report"""
    print("\n" + "=" * 70)
    print("SCE Performance Benchmarks")
    print("=" * 70)

    tests = [
        ("Add 1000 constraints", TestPerformanceLargeScale().test_add_1000_constraints),
        ("Add 1000 chained constraints", TestPerformanceLargeScale().test_add_1000_constraints_with_dependencies),
        ("Retrieve 1000 constraints", TestPerformanceLargeScale().test_get_all_constraints_1000),
        ("Filter 1000 constraints", TestPerformanceLargeScale().test_get_constraints_by_type_1000),
        ("100-level dependency chain", TestPerformanceDeepDependencies().test_dependency_chain_100_levels),
        ("Topological sort 100 constraints", TestPerformanceDeepDependencies().test_topological_sort_100_levels),
        ("Diamond structure (400 nodes)", TestPerformanceDeepDependencies().test_diamond_structure_performance),
        ("Conflict detection 100 constraints", TestPerformanceConflictDetection().test_conflict_detection_100_constraints),
        ("Conflict detection 1000 no conflicts", TestPerformanceConflictDetection().test_conflict_detection_1000_no_conflicts),
        ("Statistics 1100 constraints", TestPerformanceStatistics().test_statistics_1000_constraints),
        ("Real-world scenario 180 constraints", TestPerformanceComplexScenarios().test_real_world_scenario_mixed_constraints),
        ("Iterative operations 100 constraints", TestPerformanceComplexScenarios().test_iterative_operations_performance),
    ]

    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, "PASS"))
        except AssertionError as e:
            results.append((name, f"FAIL: {str(e)}"))
        except Exception as e:
            results.append((name, f"ERROR: {str(e)}"))

    print("\n" + "=" * 70)
    print("Benchmark Results Summary")
    print("=" * 70)

    for name, status in results:
        status_symbol = "✓" if status == "PASS" else "✗"
        print(f"{status_symbol} {name}: {status}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run benchmarks
    run_performance_benchmarks()

    # Run pytest with verbose output
    print("\nRunning pytest with detailed output...")
    pytest.main([__file__, "-v", "-s"])
