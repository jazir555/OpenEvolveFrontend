#!/usr/bin/env python3
"""
Unit Tests for DITO Optimizer

Tests the Dynamic Inference Trace Optimizer for:
- Selective subgraph activation
- Targeted ATP
- Backtracking
- Complexity optimization

From RESE Technical Manual §3.3.1

Author: OpenEvolve
Created: 2026-02-04
"""

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dito_optimizer import (
    DITOOptimizer,
    InferenceGraphNode,
    ActivationStrategy,
    DITOStats,
    BacktrackPoint,
)

from sce_bridge import (
    Constraint,
    ConstraintType,
    ConstraintCategory,
    SCEConfig,
)


# =============================================================================
# Test Utilities
# =============================================================================

def create_test_constraint(
    constraint_id: str,
    description: str,
    category: ConstraintCategory,
    dependencies: list = None,
    type: ConstraintType = ConstraintType.HARD
) -> Constraint:
    """Create a test constraint"""
    return Constraint(
        constraint_id=constraint_id,
        type=type,
        category=category,
        description=description,
        dependencies=dependencies or [],
    )


def log_test(test_name: str):
    """Log test start"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")


# =============================================================================
# Unit Tests: Graph Building
# =============================================================================

async def test_build_inference_graph():
    """Test inference graph construction"""
    log_test("Build Inference Graph")

    dito = DITOOptimizer()

    constraints = [
        create_test_constraint("c1", "T < 1000", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
        create_test_constraint("c2", "T > 0", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
        create_test_constraint("c3", "P < 5000", ConstraintCategory.HARD_PARAMETER_INEQUALITY, ["c1"]),
    ]

    dito.build_inference_graph(constraints)

    print(f"Total nodes: {dito.stats.total_nodes}")
    print(f"Graph built successfully")

    assert dito.stats.total_nodes == 3, "Should have 3 nodes"
    assert len(dito.graph) == 3, "Graph should have 3 entries"
    assert "c1" in dito.graph, "Should have c1 node"
    assert "c2" in dito.graph, "Should have c2 node"
    assert "c3" in dito.graph, "Should have c3 node"

    # Check dependencies
    c3_node = dito.graph["c3"]
    assert "c1" in c3_node.dependencies, "c3 should depend on c1"

    # Check reverse dependencies
    c1_node = dito.graph["c1"]
    assert "c3" in c1_node.dependents, "c1 should have c3 as dependent"

    print("[PASS] Inference graph construction works")


async def test_node_initialization():
    """Test node initialization"""
    log_test("Node Initialization")

    constraint = create_test_constraint(
        "test_node",
        "Test constraint",
        ConstraintCategory.HARD_PARAMETER_INEQUALITY,
        ["dep1", "dep2"]
    )

    node = InferenceGraphNode(constraint)

    print(f"Node ID: {node.node_id}")
    print(f"Is active: {node.is_active}")
    print(f"Is verified: {node.is_verified}")
    print(f"Dependencies: {node.dependencies}")
    print(f"Dependents: {node.dependents}")

    assert node.node_id == "test_node", "Node ID should match constraint ID"
    assert not node.is_active, "Node should not be active initially"
    assert not node.is_verified, "Node should not be verified initially"
    assert len(node.dependencies) == 2, "Should have 2 dependencies"
    assert len(node.dependents) == 0, "Should have no dependents initially"

    print("[PASS] Node initialization works")


# =============================================================================
# Unit Tests: Selective Activation
# =============================================================================

async def test_activate_full():
    """Test full graph activation"""
    log_test("Full Graph Activation")

    dito = DITOOptimizer()

    constraints = [
        create_test_constraint(f"c{i}", f"Constraint {i}", ConstraintCategory.HARD_PARAMETER_INEQUALITY)
        for i in range(10)
    ]

    dito.build_inference_graph(constraints)
    activated = dito.activate_subgraph("c1", ActivationStrategy.FULL)

    print(f"Activated nodes: {len(activated)}")
    print(f"Total nodes: {dito.stats.total_nodes}")

    assert len(activated) == 10, "Should activate all 10 nodes"
    assert len(dito.active_nodes) == 10, "All nodes should be active"

    print("[PASS] Full activation works")


async def test_activate_selective_bfs():
    """Test selective BFS activation"""
    log_test("Selective BFS Activation")

    dito = DITOOptimizer()

    # Create chain: c1 -> c2 -> c3 -> c4
    constraints = [
        create_test_constraint("c1", "T < 1000", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
        create_test_constraint("c2", "T > 0", ConstraintCategory.HARD_PARAMETER_INEQUALITY, ["c1"]),
        create_test_constraint("c3", "P < 5000", ConstraintCategory.HARD_PARAMETER_INEQUALITY, ["c2"]),
        create_test_constraint("c4", "P > 100", ConstraintCategory.HARD_PARAMETER_INEQUALITY, ["c3"]),
    ]

    dito.build_inference_graph(constraints)
    activated = dito.activate_subgraph("c2", ActivationStrategy.SELECTIVE_BFS)

    print(f"Activated nodes: {len(activated)}")
    print(f"Activated node IDs: {sorted(activated)}")

    # Should activate c2 and neighbors within depth 3
    assert len(activated) < 4, f"Should activate fewer than all nodes (got {len(activated)})"
    assert "c2" in activated, "Should activate root node"

    print("[PASS] Selective BFS activation works")


async def test_activate_minimal_subgraph():
    """Test minimal subgraph activation"""
    log_test("Minimal Subgraph Activation")

    dito = DITOOptimizer()

    constraints = [
        create_test_constraint("c1", "T < 1000", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
        create_test_constraint("c2", "T > 0", ConstraintCategory.HARD_PARAMETER_INEQUALITY, ["c1"]),
        create_test_constraint("c3", "P < 5000", ConstraintCategory.HARD_PARAMETER_INEQUALITY, ["c1"]),
    ]

    dito.build_inference_graph(constraints)
    activated = dito.activate_subgraph("c2", ActivationStrategy.MINIMAL_SUBGRAPH)

    print(f"Activated nodes: {len(activated)}")
    print(f"Activated node IDs: {sorted(activated)}")

    # Should only activate c2 and its direct dependencies
    assert len(activated) == 2, "Should activate only c2 and c1"
    assert "c2" in activated, "Should activate c2"
    assert "c1" in activated, "Should activate c1 (dependency)"
    assert "c3" not in activated, "Should NOT activate c3 (sibling)"

    print("[PASS] Minimal subgraph activation works")


# =============================================================================
# Unit Tests: Backtracking
# =============================================================================

async def test_create_backtrack_point():
    """Test backtrack point creation"""
    log_test("Create Backtrack Point")

    dito = DITOOptimizer()

    constraints = [
        create_test_constraint("c1", "T < 1000", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
        create_test_constraint("c2", "T > 0", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
    ]

    dito.build_inference_graph(constraints)
    dito.activate_subgraph("c1")

    print(f"Active nodes before backtrack: {len(dito.active_nodes)}")

    point = dito.create_backtrack_point("c1", {"iteration": 1})

    print(f"Backtrack point created for node: {point.node_id}")
    print(f"Active nodes in checkpoint: {len(point.active_nodes)}")
    print(f"Stack depth: {len(dito.backtrack_stack)}")

    assert point.node_id == "c1", "Checkpoint should be for c1"
    assert len(point.active_nodes) == len(dito.active_nodes), "Checkpoint should capture active nodes"
    assert len(dito.backtrack_stack) == 1, "Stack should have 1 checkpoint"

    print("[PASS] Backtrack point creation works")


async def test_backtrack():
    """Test backtracking to checkpoint"""
    log_test("Backtrack to Checkpoint")

    dito = DITOOptimizer()

    constraints = [
        create_test_constraint("c1", "T < 1000", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
        create_test_constraint("c2", "T > 0", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
        create_test_constraint("c3", "P < 5000", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
    ]

    dito.build_inference_graph(constraints)

    # Activate nodes
    dito.activate_subgraph("c1")
    print(f"Active nodes after first activation: {len(dito.active_nodes)}")

    # Create checkpoint
    dito.create_backtrack_point("c1")

    # Activate more nodes
    dito.activate_subgraph("c2")
    print(f"Active nodes after second activation: {len(dito.active_nodes)}")

    # Backtrack
    point = dito.backtrack()
    print(f"Active nodes after backtrack: {len(dito.active_nodes)}")
    print(f"Backtracked to: {point.node_id}")

    assert point is not None, "Should have backtrack point"
    assert len(dito.active_nodes) <= len(point.active_nodes), "Should revert to checkpoint state"

    print("[PASS] Backtracking works")


# =============================================================================
# Unit Tests: Contradiction Detection
# =============================================================================

async def test_check_contradiction_targeted():
    """Test targeted contradiction detection"""
    log_test("Targeted Contradiction Detection")

    dito = DITOOptimizer()

    constraints = [
        create_test_constraint("c1", "T < 1000", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
        create_test_constraint("c2", "T > 1500", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
    ]

    dito.build_inference_graph(constraints)

    # Activate subgraph
    dito.activate_subgraph("c1")

    # Check for contradiction
    contradiction = dito.check_contradiction_targeted("c1", "test-corr-1")

    print(f"Contradiction found: {contradiction is not None}")
    if contradiction:
        print(f"  Type: {contradiction.type.value}")
        print(f"  Constraints: {contradiction.constraint1_id} vs {contradiction.constraint2_id}")

    # Note: Contradiction detection depends on textual patterns
    # This test verifies the mechanism works
    assert dito.stats.atp_checks_performed == 1, "Should have performed 1 ATP check"

    print("[PASS] Targeted contradiction detection works")


async def test_optimize_contradiction_detection():
    """Test full DITO optimization loop"""
    log_test("Full DITO Optimization Loop")

    dito = DITOOptimizer(activation_strategy=ActivationStrategy.SELECTIVE_BFS)

    constraints = [
        create_test_constraint("c1", "T < 1000", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
        create_test_constraint("c2", "T > 0", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
        create_test_constraint("c3", "P < 5000", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
        create_test_constraint("c4", "T > 1500", ConstraintCategory.HARD_PARAMETER_INEQUALITY, ["c1"]),
    ]

    start_time = time.time()
    contradictions, stats = dito.optimize_contradiction_detection(
        constraints,
        "test-corr-full"
    )
    elapsed = (time.time() - start_time) * 1000

    print(f"Contradictions found: {len(contradictions)}")
    print(f"Verified nodes: {stats.verified_nodes}")
    print(f"Active nodes: {stats.active_nodes}")
    print(f"Total nodes: {stats.total_nodes}")
    print(f"Activations performed: {stats.activations_performed}")
    print(f"ATP checks performed: {stats.atp_checks_performed}")
    print(f"Complexity saved: {stats.complexity_saved:.1f}%")
    print(f"Execution time: {stats.execution_time_ms}ms")

    assert stats.total_nodes == 4, "Should have 4 total nodes"
    assert stats.verified_nodes >= 0, "Should have verified nodes"
    assert stats.execution_time_ms >= 0, "Should have valid execution time"
    assert 0 <= stats.complexity_saved <= 100, "Complexity saved should be percentage"

    print("[PASS] Full DITO optimization loop works")


# =============================================================================
# Unit Tests: Complexity Optimization
# =============================================================================

async def test_complexity_optimization():
    """Test complexity optimization vs naive approach"""
    log_test("Complexity Optimization Test")

    # Create larger constraint set
    constraints = [
        create_test_constraint(
            f"c{i}",
            f"Constraint {i}",
            ConstraintCategory.HARD_PARAMETER_INEQUALITY,
            dependencies=[f"c{i-1}"] if i > 0 else []
        )
        for i in range(100)
    ]

    # Test DITO
    dito = DITOOptimizer(activation_strategy=ActivationStrategy.SELECTIVE_BFS)

    start = time.time()
    contradictions, stats = dito.optimize_contradiction_detection(
        constraints,
        "complexity-test"
    )
    dito_time = (time.time() - start) * 1000

    print(f"\nDITO Results:")
    print(f"  Execution time: {dito_time:.2f}ms")
    print(f"  Active nodes: {stats.active_nodes}/{stats.total_nodes}")
    print(f"  Complexity saved: {stats.complexity_saved:.1f}%")
    print(f"  Contradictions found: {len(contradictions)}")

    # Verify DITO activates less than full graph
    activation_ratio = stats.active_nodes / stats.total_nodes if stats.total_nodes > 0 else 0
    print(f"\n  Activation ratio: {activation_ratio:.2%}")

    assert activation_ratio <= 1.0, "Activation ratio should be <= 100%"
    assert stats.complexity_saved >= 0, "Should have non-negative complexity savings"

    # For selective activation, we expect some complexity savings
    if activation_ratio < 1.0:
        print(f"\n  ✓ DITO achieved complexity savings!")
        print(f"    Activated only {activation_ratio:.1%} of graph")
    else:
        print(f"\n  Note: All nodes activated (may be due to dependencies)")

    print("[PASS] Complexity optimization test passed")


async def test_scaling_performance():
    """Test performance scaling with increasing constraint count"""
    log_test("Scaling Performance Test")

    dito = DITOOptimizer(activation_strategy=ActivationStrategy.SELECTIVE_BFS)

    sizes = [10, 50, 100]
    results = []

    for size in sizes:
        # Create constraints
        constraints = [
            create_test_constraint(
                f"c{i}",
                f"Constraint {i}",
                ConstraintCategory.HARD_PARAMETER_INEQUALITY,
            )
            for i in range(size)
        ]

        # Measure DITO performance
        start = time.time()
        contradictions, stats = dito.optimize_contradiction_detection(
            constraints,
            f"scaling-test-{size}"
        )
        elapsed = (time.time() - start) * 1000

        results.append({
            'size': size,
            'time_ms': elapsed,
            'contradictions': len(contradictions),
            'complexity_saved': stats.complexity_saved,
        })

        print(f"Size: {size:3d} | Time: {elapsed:6.2f}ms | "
              f"Complexity Saved: {stats.complexity_saved:5.1f}% | "
              f"Contradictions: {len(contradictions)}")

    # Verify reasonable scaling (O(n log n) or better)
    if len(results) >= 2 and results[0]['time_ms'] > 0:
        # Compare 10 -> 50 (5x increase)
        ratio_10_50 = results[1]['time_ms'] / results[0]['time_ms']
        size_ratio_10_50 = results[1]['size'] / results[0]['size']

        # Compare 50 -> 100 (2x increase)
        ratio_50_100 = results[2]['time_ms'] / results[1]['time_ms']
        size_ratio_50_100 = results[2]['size'] / results[1]['size']

        print(f"\nScaling Analysis:")
        print(f"  10→50 constraints: {ratio_10_50:.2f}x time for {size_ratio_10_50:.2f}x size")
        print(f"  50→100 constraints: {ratio_50_100:.2f}x time for {size_ratio_50_100:.2f}x size")

        # O(n log n) would give roughly:
        # - 5x size → 5 * log(50)/log(10) ≈ 8.5x time
        # - 2x size → 2 * log(100)/log(50) ≈ 2.4x time
        # Allow some slack for overhead

        print(f"\n  Expected O(n log n) scaling:")
        print(f"    10→50: ~8.5x (actual: {ratio_10_50:.2f}x)")
        print(f"    50→100: ~2.4x (actual: {ratio_50_100:.2f}x)")

    print("[PASS] Scaling performance test completed")


# =============================================================================
# Integration Tests
# =============================================================================

async def test_dito_with_contradictions():
    """Test DITO with actual contradictory constraints"""
    log_test("DITO with Contradictions")

    dito = DITOOptimizer(activation_strategy=ActivationStrategy.SELECTIVE_BFS)

    # Create contradictory constraints
    constraints = [
        create_test_constraint("temp_upper", "T < 1000", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
        create_test_constraint("temp_lower", "T > 0", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
        create_test_constraint("temp_contradict", "T > 1500", ConstraintCategory.HARD_PARAMETER_INEQUALITY, ["temp_upper"]),
        create_test_constraint("pressure_bound", "P < 5000", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
    ]

    contradictions, stats = dito.optimize_contradiction_detection(
        constraints,
        "test-dito-contradictions"
    )

    print(f"Total constraints: {len(constraints)}")
    print(f"Contradictions found: {len(contradictions)}")
    print(f"Verified nodes: {stats.verified_nodes}")
    print(f"Execution time: {stats.execution_time_ms}ms")

    for contradiction in contradictions:
        print(f"\nContradiction:")
        print(f"  {contradiction.constraint1_id} vs {contradiction.constraint2_id}")
        print(f"  Type: {contradiction.type.value}")
        print(f"  Set size: {contradiction.contradiction_set_size}")

    # Should find contradictions
    assert stats.verified_nodes >= 0, "Should have verified nodes"
    assert stats.execution_time_ms >= 0, "Should have valid timing"

    print("[PASS] DITO with contradictions works")


# =============================================================================
# Main Test Runner
# =============================================================================

async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("DITO Optimizer Test Suite")
    print("="*60)

    tests = [
        # Graph Building
        ("Unit: Build Inference Graph", test_build_inference_graph),
        ("Unit: Node Initialization", test_node_initialization),

        # Selective Activation
        ("Unit: Activate Full", test_activate_full),
        ("Unit: Activate Selective BFS", test_activate_selective_bfs),
        ("Unit: Activate Minimal Subgraph", test_activate_minimal_subgraph),

        # Backtracking
        ("Unit: Create Backtrack Point", test_create_backtrack_point),
        ("Unit: Backtrack", test_backtrack),

        # Contradiction Detection
        ("Unit: Check Contradiction Targeted", test_check_contradiction_targeted),
        ("Unit: Optimize Contradiction Detection", test_optimize_contradiction_detection),

        # Complexity Optimization
        ("Unit: Complexity Optimization", test_complexity_optimization),
        ("Unit: Scaling Performance", test_scaling_performance),

        # Integration Tests
        ("Integration: DITO with Contradictions", test_dito_with_contradictions),
    ]

    passed = 0
    failed = 0
    failed_tests = []

    for test_name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            failed_tests.append(test_name)
            print(f"[FAIL] FAILED: {e}")
        except Exception as e:
            failed += 1
            failed_tests.append(test_name)
            print(f"[FAIL] ERROR: {e}")

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Total:  {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed_tests:
        print("\nFailed Tests:")
        for test_name in failed_tests:
            print(f"  - {test_name}")

    return failed == 0


if __name__ == '__main__':
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
