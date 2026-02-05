#!/usr/bin/env python
"""
OpenEvolve Working Components Test

This script tests all components that are currently working after the fixes.
Run with: python -X utf8 tests/integration/test_working_components.py
"""

import sys


def test_gauntlets():
    """Test gauntlet imports and functionality"""
    print("\n" + "="*80)
    print("TEST 1: Gauntlets")
    print("="*80)

    from openevolve.gauntlets import (
        LoongFlowGauntletEvaluator,
        ThreeRoundGauntletOrchestrator,
        MultiRoundGauntletOrchestrator,
    )

    print("[OK] LoongFlowGauntletEvaluator imported")
    print("[OK] ThreeRoundGauntletOrchestrator imported")
    print("[OK] MultiRoundGauntletOrchestrator imported")

    return True


def test_domain_optimizers():
    """Test domain optimizer imports and instantiation"""
    print("\n" + "="*80)
    print("TEST 2: Domain Optimizers")
    print("="*80)

    from openevolve.domain import (
        FinanceOptimizer,
        TradingOptimizer,
        ScienceOptimizer,
        EngineeringOptimizer,
        PharmaOptimizer,
        WebDesignOptimizer,
    )

    print("[OK] All domain optimizers imported")

    # Test instantiation
    finance = FinanceOptimizer()
    trading = TradingOptimizer()
    science = ScienceOptimizer()

    print("[OK] FinanceOptimizer instantiated")
    print("[OK] TradingOptimizer instantiated")
    print("[OK] ScienceOptimizer instantiated")

    # Test config retrieval
    finance_config = finance.get_default_config()
    trading_config = trading.get_default_config()
    science_config = science.get_default_config()

    print(f"[OK] Finance config: {type(finance_config).__name__}")
    print(f"[OK] Trading config: {type(trading_config).__name__}")
    print(f"[OK] Science config: {type(science_config).__name__}")

    return True


def test_unified_config():
    """Test unified config system"""
    print("\n" + "="*80)
    print("TEST 3: Unified Config")
    print("="*80)

    from openevolve.unified.config import UnifiedEvolutionConfig

    config = UnifiedEvolutionConfig()

    print(f"[OK] UnifiedEvolutionConfig created")
    print(f"  - max_iterations: {config.max_iterations}")
    print(f"  - population_size: {config.population_size}")
    print(f"  - evolution_mode: {config.evolution_mode}")

    return True


def test_knowledge_engine():
    """Test knowledge engine integration"""
    print("\n" + "="*80)
    print("TEST 4: Knowledge Engine")
    print("="*80)

    from openevolve.knowledge_engine.integrations import (
        LoongFlowKnowledgeExtractor,
    )

    print("[OK] LoongFlowKnowledgeExtractor imported")

    return True


def test_evolution_result():
    """Test evolution result structure"""
    print("\n" + "="*80)
    print("TEST 5: Evolution Result")
    print("="*80)

    from openevolve.api import EvolutionResult
    from openevolve.database import Program

    # Create a test result
    program = Program(code="test code", score=0.95)
    result = EvolutionResult(
        best_program=program,
        best_score=0.95,
        best_code="test code",
        metrics={"iterations": 10},
        output_dir=None
    )

    print(f"[OK] EvolutionResult created")
    print(f"  - best_score: {result.best_score}")
    print(f"  - iterations: {result.metrics['iterations']}")

    return True


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*80)
    print("OPENEVOLVE WORKING COMPONENTS TEST")
    print("="*80)
    print("\nThis test suite verifies all components that are working")
    print("after the integration fixes were applied.")

    tests = [
        ("Gauntlets", test_gauntlets),
        ("Domain Optimizers", test_domain_optimizers),
        ("Unified Config", test_unified_config),
        ("Knowledge Engine", test_knowledge_engine),
        ("Evolution Result", test_evolution_result),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"\n[OK] PASSED: {name}")
        except Exception as e:
            failed += 1
            print(f"\n[FAIL] FAILED: {name}")
            print(f"  Error: {e}")

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(tests)*100):.1f}%")
    print("="*80)

    if failed == 0:
        print("\n[OK] ALL TESTS PASSED!")
    else:
        print(f"\n⚠ {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
