"""
Test script for LeanAide MCTS Workflow Integration

This script demonstrates and tests the MCTS workflow integration.
"""

import asyncio
import sys
import io

# Set UTF-8 encoding for output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Test imports
try:
    from leanaide_mcts_workflow import (
        MCTSWorkflowIntegrator,
        MCTSSubProblemSolver,
        MCTSProofRefiner,
        MCTSWorkflowMonitor,
        MCTSWorkflowConfig,
        MCTSStrategy,
        MCTSSearchSpace,
        MCTSProgress,
        add_mcts_config_to_workflow_state,
        extract_mcts_config_from_workflow_state,
        solve_with_mcts_approach,
        MCTS_AVAILABLE,
        LEANAIDE_AVAILABLE,
        EVOLUTIONARY_AVAILABLE,
        WORKFLOW_AVAILABLE
    )
    print("[PASS] All imports successful")
except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)


def test_configuration():
    """Test MCTS configuration."""
    print("\n=== Testing Configuration ===")

    config = MCTSWorkflowConfig(
        lean_mcts_enabled=True,
        lean_mcts_strategy=MCTSStrategy.ADAPTIVE,
        lean_mcts_iterations=100,
        lean_mcts_time_budget=60.0,
        lean_mcts_c_param=1.414
    )

    print(f"  MCTS Enabled: {config.lean_mcts_enabled}")
    print(f"  Strategy: {config.lean_mcts_strategy.value}")
    print(f"  Iterations: {config.lean_mcts_iterations}")
    print(f"  Time Budget: {config.lean_mcts_time_budget}s")
    print(f"  C Param: {config.lean_mcts_c_param}")

    print("[PASS] Configuration test passed")
    return config


def test_search_space_analysis():
    """Test search space analysis."""
    print("\n=== Testing Search Space Analysis ===")

    if not WORKFLOW_AVAILABLE:
        print("  ⊘ Workflow structures not available, skipping")
        return None

    from workflow_structures import SubProblem

    config = MCTSWorkflowConfig()
    integrator = MCTSWorkflowIntegrator(config=config)

    # Create test sub-problem
    sub_problem = SubProblem(
        id="test_001",
        description="Prove that for all natural numbers n, n + 0 = n by induction",
        dependencies=[],
        ai_suggested_complexity_score=7
    )

    # Analyze search space
    search_space = integrator.analyze_search_space(sub_problem)

    print(f"  Branching Factor: {search_space.branching_factor}")
    print(f"  Estimated Depth: {search_space.estimated_depth}")
    print(f"  Has Heuristics: {search_space.has_heuristics}")
    print(f"  Tactic Diversity: {search_space.tactic_diversity:.2f}")
    print(f"  Applicability Score: {search_space.calculate_applicability_score():.2f}")
    print(f"  Is Applicable: {search_space.is_applicable}")

    print("[OK] Search space analysis test passed")
    return search_space


def test_monitor():
    """Test MCTS monitor."""
    print("\n=== Testing MCTS Monitor ===")

    config = MCTSWorkflowConfig()
    monitor = MCTSWorkflowMonitor(config)

    # Simulate progress updates
    for i in range(0, 100, 20):
        monitor.update_progress(
            sub_problem_id="test_001",
            iteration=i,
            best_score=0.5 + (i / 200),
            current_best_proof=f"Proof at iteration {i}",
            tree_size=i * 5,
            nodes_explored=i * 2
        )

    # Get progress
    progress = monitor.get_progress("test_001")
    print(f"  Iterations: {progress['iterations']}")
    print(f"  Best Score: {progress['best_score']:.3f}")
    print(f"  Tree Size: {progress['tree_size']}")
    print(f"  Nodes Explored: {progress['nodes_explored']}")

    # Get statistics
    stats = monitor.get_statistics("test_001")
    if stats:
        print(f"  Average Score: {stats.get('avg_score', 0):.3f}")
        print(f"  Max Score: {stats.get('max_score', 0):.3f}")

    # Test early termination
    should_terminate = monitor.should_early_terminate("test_001")
    print(f"  Should Terminate: {should_terminate}")

    print("[OK] Monitor test passed")


def test_strategies():
    """Test different MCTS strategies."""
    print("\n=== Testing MCTS Strategies ===")

    strategies = [
        MCTSStrategy.STANDARD,
        MCTSStrategy.UCT,
        MCTSStrategy.HYBRID_EVOLUTION,
        MCTSStrategy.HYBRID_ADVERSARIAL,
        MCTSStrategy.ADAPTIVE
    ]

    for strategy in strategies:
        config = MCTSWorkflowConfig(
            lean_mcts_strategy=strategy,
            lean_mcts_iterations=50
        )
        print(f"  {strategy.value:20s} - iterations={config.lean_mcts_iterations}")

    print("[OK] Strategies test passed")


def test_workflow_state_integration():
    """Test WorkflowState integration."""
    print("\n=== Testing WorkflowState Integration ===")

    if not WORKFLOW_AVAILABLE:
        print("  ⊘ Workflow structures not available, skipping")
        return

    from workflow_structures import WorkflowState

    # Create workflow state
    workflow_state = WorkflowState(
        workflow_id="test_workflow",
        problem_statement="Test problem",
        analyzed_context={}
    )

    # Add MCTS config
    config = MCTSWorkflowConfig(
        lean_mcts_enabled=True,
        lean_mcts_strategy=MCTSStrategy.ADAPTIVE,
        lean_mcts_iterations=500
    )

    updated_state = add_mcts_config_to_workflow_state(workflow_state, config)

    # Extract config
    extracted_config = extract_mcts_config_from_workflow_state(updated_state)

    print(f"  Original enabled: {config.lean_mcts_enabled}")
    print(f"  Extracted enabled: {extracted_config.lean_mcts_enabled}")
    print(f"  Original strategy: {config.lean_mcts_strategy.value}")
    print(f"  Extracted strategy: {extracted_config.lean_mcts_strategy.value}")
    print(f"  Original iterations: {config.lean_mcts_iterations}")
    print(f"  Extracted iterations: {extracted_config.lean_mcts_iterations}")

    assert config.lean_mcts_enabled == extracted_config.lean_mcts_enabled
    assert config.lean_mcts_strategy == extracted_config.lean_mcts_strategy
    assert config.lean_mcts_iterations == extracted_config.lean_mcts_iterations

    print("[OK] WorkflowState integration test passed")


async def test_integrator_creation():
    """Test MCTSWorkflowIntegrator creation."""
    print("\n=== Testing MCTSWorkflowIntegrator ===")

    config = MCTSWorkflowConfig(
        lean_mcts_enabled=True,
        lean_mcts_strategy=MCTSStrategy.STANDARD,
        lean_mcts_iterations=50
    )

    integrator = MCTSWorkflowIntegrator(config=config)

    print(f"  Config: {integrator.config.lean_mcts_strategy.value}")
    print(f"  Monitor created: {integrator.monitor is not None}")

    print("[OK] Integrator creation test passed")


def test_proof_refiner():
    """Test MCTSProofRefiner."""
    print("\n=== Testing MCTSProofRefiner ===")

    config = MCTSWorkflowConfig(lean_mcts_refinement_iterations=50)
    refiner = MCTSProofRefiner(config)

    print(f"  Refiner created: {refiner is not None}")
    print(f"  Config iterations: {refiner.config.lean_mcts_refinement_iterations}")

    print("[OK] Proof refiner test passed")


def test_subproblem_solver():
    """Test MCTSSubProblemSolver."""
    print("\n=== Testing MCTSSubProblemSolver ===")

    config = MCTSWorkflowConfig()
    integrator = MCTSWorkflowIntegrator(config=config)
    solver = MCTSSubProblemSolver(integrator, config)

    print(f"  Solver created: {solver is not None}")
    print(f"  Integrator linked: {solver.integrator is integrator}")

    print("[OK] Subproblem solver test passed")


def test_availability():
    """Test component availability."""
    print("\n=== Testing Component Availability ===")

    print(f"  MCTS Available: {MCTS_AVAILABLE}")
    print(f"  LeanAide Available: {LEANAIDE_AVAILABLE}")
    print(f"  Evolutionary Available: {EVOLUTIONARY_AVAILABLE}")
    print(f"  Workflow Available: {WORKFLOW_AVAILABLE}")

    print("[OK] Availability test passed")


async def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("LeanAide MCTS Workflow Integration Tests")
    print("=" * 60)

    try:
        # Test availability
        test_availability()

        # Test configuration
        test_configuration()

        # Test strategies
        test_strategies()

        # Test search space analysis
        test_search_space_analysis()

        # Test monitor
        test_monitor()

        # Test integrator creation
        await test_integrator_creation()

        # Test proof refiner
        test_proof_refiner()

        # Test subproblem solver
        test_subproblem_solver()

        # Test WorkflowState integration
        test_workflow_state_integration()

        print("\n" + "=" * 60)
        print("All tests passed! [OK]")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
