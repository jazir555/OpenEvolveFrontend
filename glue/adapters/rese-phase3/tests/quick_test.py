"""
Quick test script for RESE Phase III.

Tests basic functionality without running full test suite.
"""

import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "schemas"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Set environment variables
os.environ["PHASE3_ITERATIONS"] = "20"
os.environ["PHASE3_UCB1_C"] = "1.414"
os.environ["PHASE3_CONVERGENCE_THRESHOLD"] = "0.01"
os.environ["PHASE3_TIMEOUT_MS"] = "30000"
os.environ["PHASE3_MAX_DEPTH"] = "5"
os.environ["PHASE3_MAX_CHILDREN"] = "3"
os.environ["PHASE3_MIN_VISITS"] = "2"
os.environ["PHASE3_SIG_THRESHOLD"] = "0.05"
os.environ["PHASE3_CONFIDENCE_INTERVAL"] = "0.95"
os.environ["PHASE3_MIN_SAMPLE_SIZE"] = "10"
os.environ["PHASE3_ACI_WINDOW"] = "10"
os.environ["PHASE3_ACI_STABILITY"] = "0.01"
os.environ["PHASE3_DEDUP_ENABLED"] = "true"
os.environ["PHASE3_CACHE_SIZE"] = "1000"
os.environ["PHASE3_CB_THRESHOLD"] = "5"
os.environ["PHASE3_CB_TIMEOUT"] = "60000"

try:
    print("=" * 60)
    print("RESE Phase III Quick Test")
    print("=" * 60)
    print()

    # Test 1: Imports
    print("Test 1: Importing modules...")
    from rese_schemas import Hypothesis
    from phase3_executor import (
        MCTSSearchExecutor,
        Phase3Config,
    )
    from phase3_adapter import Phase3Adapter
    print("  ✓ All imports successful")
    print()

    # Test 2: Configuration
    print("Test 2: Loading configuration...")
    config = Phase3Config.from_env()
    assert config.iterations == 20
    assert config.max_depth == 5
    print(f"  ✓ Configuration loaded: {config.iterations} iterations, max depth {config.max_depth}")
    print()

    # Test 3: Executor Initialization
    print("Test 3: Initializing executor...")
    executor = MCTSSearchExecutor(config)
    print("  ✓ Executor initialized successfully")
    print()

    # Test 4: Simple Search
    print("Test 4: Executing simple search...")
    root_hypothesis = Hypothesis(
        statement="Test root hypothesis for quick test",
        type="test",
        domain="test_domain",
        confidence=0.5,
    )

    def hypothesis_generator():
        import random
        random.seed(42)
        children = []
        for i in range(3):
            child = Hypothesis(
                statement=f"Child hypothesis {i}",
                type="test",
                domain="test_domain",
                confidence=0.5 + random.uniform(0.0, 0.3),
                source_hypotheses=[root_hypothesis.hypothesis_id],
            )
            children.append(child)
        return children

    def reward_function(hypothesis):
        import random
        random.seed(hash(hypothesis.hypothesis_id) % 1000)
        return hypothesis.confidence + random.uniform(-0.05, 0.05)

    search_result, error = executor.execute_search(
        root_hypothesis=root_hypothesis,
        hypothesis_generator=hypothesis_generator,
        reward_function=reward_function,
    )

    if error:
        print(f"  ✗ Search failed: {error}")
        sys.exit(1)

    print(f"  ✓ Search completed successfully")
    print(f"    - Search ID: {search_result.search_id}")
    print(f"    - Iterations: {search_result.iterations}")
    print(f"    - Total nodes: {search_result.total_nodes}")
    print(f"    - Max depth: {search_result.max_depth}")
    print(f"    - Best confidence: {search_result.best_hypothesis.confidence:.3f}")
    print(f"    - Converged: {search_result.convergence_reached}")
    print(f"    - Execution time: {search_result.execution_time_ms:.1f}ms")
    print()

    # Test 5: Adapter
    print("Test 5: Testing adapter...")
    adapter = Phase3Adapter(config)

    request = {
        "root_hypothesis": {
            "statement": "Adapter test hypothesis",
            "type": "test",
            "domain": "test",
            "confidence": 0.5,
        },
        "num_children": 3,
    }

    result = adapter.search(request)

    if not result.get("success"):
        print(f"  ✗ Adapter search failed: {result.get('error')}")
        sys.exit(1)

    print(f"  ✓ Adapter search completed successfully")
    print(f"    - Search ID: {result['search_id']}")
    print(f"    - Best confidence: {result['best_confidence']:.3f}")
    print()

    # Test 6: Health Check
    print("Test 6: Health check...")
    health = adapter.get_health()
    print(f"  ✓ Health status: {health['status']}")
    print(f"    - Circuit breaker: {health['circuit_breaker_state']}")
    print(f"    - DLQ size: {health['dlq_size']}")
    print()

    # Summary
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print()
    print("RESE Phase III MCTS Search Executor is functional!")
    print()
    print("Components verified:")
    print("  ✓ Configuration")
    print("  ✓ Executor initialization")
    print("  ✓ MCTS search execution")
    print("  ✓ Adapter interface")
    print("  ✓ Health monitoring")
    print()
    print("Ready for integration with DEE and LLTL.")
    print()

except Exception as e:
    print()
    print("=" * 60)
    print("TEST FAILED ✗")
    print("=" * 60)
    print()
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
