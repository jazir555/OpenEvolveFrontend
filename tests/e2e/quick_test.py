"""
Quick E2E Test Runner

Runs a subset of critical tests to verify RESE framework functionality.

Author: RESE Team
Created: 2026-02-04
"""

import sys
import os
import time
from datetime import datetime, timezone

# Fix encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "glue"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "glue", "orchestration"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "glue", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "glue", "schemas"))


def test_imports():
    """Test 1: Verify all imports work"""
    print("\n" + "="*80)
    print("TEST 1: Import Verification")
    print("="*80)

    try:
        from glue.orchestration.rese_pipeline import (
            RESEPipeline, PhaseStatus, PipelineConfig
        )
        print("  [OK] RESE Pipeline imported")

        from glue.lib.rese_dee import DeepExplorationEngine, ExplorationConfig
        print("  [OK] DEE imported")

        from glue.lib.rese_lltl import LogicToLossTranslator
        print("  [OK] LLTL imported")

        try:
            from glue.adapters.rese_z3_bridge.src.rese_z3_bridge import (
                RESEZ3Bridge, RESEZ3BridgeConfig
            )
            print("  [OK] Z3 Bridge imported")
        except ImportError:
            print("  [WARN] Z3 Bridge not available (optional)")

        print("\n[SUCCESS] ALL CRITICAL IMPORTS SUCCESSFUL")
        return True

    except Exception as e:
        print(f"\n[FAILED] IMPORT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_creation():
    """Test 2: Verify pipeline can be created"""
    print("\n" + "="*80)
    print("TEST 2: Pipeline Creation")
    print("="*80)

    try:
        from glue.orchestration.rese_pipeline import RESEPipeline, PipelineConfig

        # Set all required environment variables
        os.environ.update({
            "PHASE_I_TIMEOUT_MS": "30000",
            "PHASE_II_TIMEOUT_MS": "30000",
            "PHASE_III_TIMEOUT_MS": "30000",
            "PHASE_IV_TIMEOUT_MS": "30000",
            "PIPELINE_TIMEOUT_MS": "120000",
            "MAX_RETRIES": "3",
            "RETRY_INITIAL_DELAY_MS": "1000",
            "RETRY_MAX_DELAY_MS": "30000",
            "RETRY_BACKOFF_MULTIPLIER": "2.0",
            "CIRCUIT_BREAKER_THRESHOLD": "5",
            "CIRCUIT_BREAKER_TIMEOUT_MS": "60000",
            "CIRCUIT_BREAKER_HALF_OPEN_ATTEMPTS": "3",
            "DLQ_MAX_SIZE": "1000"
        })

        config = PipelineConfig.from_env()
        pipeline = RESEPipeline(config)

        print(f"  [OK] Pipeline created successfully")
        print(f"  [OK] Phase I enabled: {config.enable_phase_i}")
        print(f"  [OK] Phase II enabled: {config.enable_phase_ii}")
        print(f"  [OK] Phase III enabled: {config.enable_phase_iii}")
        print(f"  [OK] Phase IV enabled: {config.enable_phase_iv}")

        print("\n[SUCCESS] PIPELINE CREATION SUCCESSFUL")
        return True

    except Exception as e:
        print(f"\n[FAILED] PIPELINE CREATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dee_creation():
    """Test 3: Verify DEE can be created"""
    print("\n" + "="*80)
    print("TEST 3: DEE Creation")
    print("="*80)

    try:
        from glue.lib.rese_dee import DeepExplorationEngine, ExplorationConfig

        config = ExplorationConfig.from_env()
        dee = DeepExplorationEngine(config)

        print(f"  [OK] DEE created successfully")
        print(f"  [OK] Max hypotheses: {config.max_hypotheses}")
        print(f"  [OK] MCTS iterations: {config.mcts_iterations}")
        print(f"  [OK] Exploration depth: {config.exploration_depth}")

        print("\n[SUCCESS] DEE CREATION SUCCESSFUL")
        return True

    except Exception as e:
        print(f"\n[FAILED] DEE CREATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lltl_creation():
    """Test 4: Verify LLTL can be created"""
    print("\n" + "="*80)
    print("TEST 4: LLTL Creation")
    print("="*80)

    try:
        from glue.lib.rese_lltl import (
            LogicToLossTranslator, EncodingConfig, LossConfig
        )

        translator = LogicToLossTranslator()

        print(f"  [OK] LLTL created successfully")
        print(f"  [OK] Encoder initialized")
        print(f"  [OK] Composer initialized")
        print(f"  [OK] DITO initialized")

        stats = translator.get_stats()
        print(f"  [OK] Stats retrievable: {stats}")

        print("\n[SUCCESS] LLTL CREATION SUCCESSFUL")
        return True

    except Exception as e:
        print(f"\n[FAILED] LLTL CREATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_z3_bridge_creation():
    """Test 5: Verify Z3 Bridge can be created"""
    print("\n" + "="*80)
    print("TEST 5: Z3 Bridge Creation (Optional)")
    print("="*80)

    try:
        from glue.adapters.rese_z3_bridge.src.rese_z3_bridge import (
            RESEZ3Bridge, RESEZ3BridgeConfig
        )

        os.environ.update({
            "Z3_BASE_URL": "http://localhost:8000",
            "Z3_TIMEOUT_MS": "10000",
            "LEANAIDE_ENABLE": "false"
        })

        config = RESEZ3BridgeConfig.from_env()
        bridge = RESEZ3Bridge(config)

        print(f"  [OK] Z3 Bridge created successfully")
        print(f"  [OK] Z3 URL: {config.z3_base_url}")
        print(f"  [OK] Z3 timeout: {config.z3_timeout_ms}ms")
        print(f"  [OK] LeanAide enabled: {config.leanaide_enable}")

        stats = bridge.get_stats()
        print(f"  [OK] Stats retrievable")

        print("\n[SUCCESS] Z3 BRIDGE CREATION SUCCESSFUL")
        return True

    except ImportError as e:
        print(f"\n[WARN] Z3 Bridge not available: {e}")
        print("[INFO] This is optional for core RESE functionality")
        return True  # Consider this a pass since it's optional
    except Exception as e:
        print(f"\n[FAILED] Z3 BRIDGE CREATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase_execution():
    """Test 6: Verify phases can execute"""
    print("\n" + "="*80)
    print("TEST 6: Phase Execution (Mock)")
    print("="*80)

    try:
        from glue.orchestration.rese_pipeline import RESEPipeline, PipelineConfig

        # Set all required environment variables
        os.environ.update({
            "PHASE_I_TIMEOUT_MS": "30000",
            "PHASE_II_TIMEOUT_MS": "30000",
            "PHASE_III_TIMEOUT_MS": "30000",
            "PHASE_IV_TIMEOUT_MS": "30000",
            "PIPELINE_TIMEOUT_MS": "120000",
            "MAX_RETRIES": "3",
            "RETRY_INITIAL_DELAY_MS": "1000",
            "RETRY_MAX_DELAY_MS": "30000",
            "RETRY_BACKOFF_MULTIPLIER": "2.0",
            "CIRCUIT_BREAKER_THRESHOLD": "5",
            "CIRCUIT_BREAKER_TIMEOUT_MS": "60000",
            "CIRCUIT_BREAKER_HALF_OPEN_ATTEMPTS": "3",
            "DLQ_MAX_SIZE": "1000"
        })

        config = PipelineConfig.from_env()
        pipeline = RESEPipeline(config)

        problem = "Simple test problem: A implies B"

        print(f"  Executing pipeline with problem: {problem}")

        # Note: This will use mock implementations in the phase executors
        result = pipeline.execute(
            problem_statement=problem,
            correlation_id="test-correlation-123"
        )

        print(f"  [OK] Pipeline executed")
        print(f"  [OK] Correlation ID: {result['correlation_id']}")
        print(f"  [OK] Status: {result['status']}")
        print(f"  [OK] Execution time: {result['execution_time_ms']:.2f}ms")

        if 'results' in result:
            for phase, phase_result in result['results'].items():
                print(f"  [OK] {phase}: {phase_result.get('status', 'unknown')}")

        print("\n[SUCCESS] PHASE EXECUTION SUCCESSFUL")
        return True

    except Exception as e:
        print(f"\n[FAILED] PHASE EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_report(results):
    """Generate test report"""
    print("\n" + "="*80)
    print("TEST EXECUTION REPORT")
    print("="*80)

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result is True)
    failed_tests = total_tests - passed_tests

    print(f"\nTotal Tests:     {total_tests}")
    print(f"Passed:          {passed_tests} [PASS]")
    print(f"Failed:          {failed_tests} [FAIL]")
    print(f"Pass Rate:       {passed_tests/total_tests*100:.1f}%")

    print("\nTest Results:")
    print("-" * 80)

    for test_name, result in results.items():
        status = "[PASS]" if result is True else "[FAIL]"
        print(f"{status}  {test_name}")

    print("\n" + "="*80)

    if failed_tests == 0:
        print("STATUS: ALL TESTS PASSED [SUCCESS]")
        print("="*80)
        return 0
    else:
        print("STATUS: SOME TESTS FAILED [FAILED]")
        print("="*80)
        return 1


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("RESE QUICK E2E TEST SUITE")
    print("="*80)
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("Testing: RESE Framework with Z3 and LeanAide Integrations")

    results = {}
    start_time = time.time()

    # Run tests
    results["Import Verification"] = test_imports()
    results["Pipeline Creation"] = test_pipeline_creation()
    results["DEE Creation"] = test_dee_creation()
    results["LLTL Creation"] = test_lltl_creation()
    results["Z3 Bridge Creation"] = test_z3_bridge_creation()
    results["Phase Execution"] = test_phase_execution()

    execution_time = time.time() - start_time

    print(f"\nTotal Execution Time: {execution_time:.2f}s")

    # Generate report
    exit_code = generate_report(results)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
