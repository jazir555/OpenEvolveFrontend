"""
Simple Integration Tests for OpenEvolve

Tests that verify the core imports work correctly.
"""

import sys
import os

# Add paths to PYTHONPATH
openevolve_root = '/c/Users/mmeadow/Documents/OpenEvolve/Frontend/openevolve'
if openevolve_root not in sys.path:
    sys.path.insert(0, openevolve_root)

front_end_root = '/c/Users/mmeadow/Documents/OpenEvolve/Frontend'
if front_end_root not in sys.path:
    sys.path.insert(0, front_end_root)

def test_unified_api():
    """Test unified API imports"""
    print("\n=== Testing Unified API ===")
    from openevolve.api import run_evolution, EvolutionResult
    from openevolve.unified.config import UnifiedEvolutionConfig
    print("[OK] Unified API imports successful")


def test_gauntlets():
    """Test gauntlet imports"""
    print("\n=== Testing Gauntlets ===")
    from gauntlets import (
        LoongFlowGauntletEvaluator,
        ThreeRoundGauntletOrchestrator,
        MultiRoundGauntletOrchestrator,
    )
    print("[OK] Gauntlet imports successful")


def test_domain_optimizers():
    """Test domain optimizer imports"""
    print("\n=== Testing Domain Optimizers ===")
    from domain import (
        FinanceOptimizer,
        TradingOptimizer,
        ScienceOptimizer,
        EngineeringOptimizer,
        PharmaOptimizer,
        WebDesignOptimizer,
    )
    print("[OK] Domain optimizer imports successful")


def test_knowledge_engine():
    """Test knowledge engine imports"""
    print("\n=== Testing Knowledge Engine ===")
    from knowledge_engine.integrations import (
        LoongFlowKnowledgeExtractor,
        UnifiedEvolutionKnowledgeExtractor,
    )
    print("[OK] Knowledge engine imports successful")


def test_instantiation():
    """Test that classes can be instantiated"""
    print("\n=== Testing Instantiation ===")

    from domain import FinanceOptimizer, TradingOptimizer, ScienceOptimizer
    from gauntlets import LoongFlowGauntletEvaluator

    # Test domain optimizers
    finance = FinanceOptimizer()
    trading = TradingOptimizer()
    science = ScienceOptimizer()

    print("[OK] Domain optimizers instantiated")

    # Test config retrieval
    finance_config = finance.get_default_config()
    trading_config = trading.get_default_config()
    science_config = science.get_default_config()

    assert finance_config is not None
    assert trading_config is not None
    assert science_config is not None

    print("[OK] Configs retrieved successfully")


def test_unified_config():
    """Test unified config"""
    print("\n=== Testing Unified Config ===")

    from openevolve.unified import (
        get_finance_config,
        get_trading_config,
        get_scientific_config,
        UnifiedEvolutionConfig
    )

    config = UnifiedEvolutionConfig(
        max_iterations=100,
        population_size=50,
        mutation_rate=0.15,
        crossover_rate=0.75,
    )

    assert config.max_iterations == 100
    assert config.population_size == 50

    print("[OK] Unified config works correctly")


def run_all_tests():
    """Run all tests"""
    print("="*80)
    print("OPENEVOLVE SIMPLE INTEGRATION TESTS")
    print("="*80)

    tests = [
        ("Unified API", test_unified_api),
        ("Gauntlets", test_gauntlets),
        ("Domain Optimizers", test_domain_optimizers),
        ("Knowledge Engine", test_knowledge_engine),
        ("Instantiation", test_instantiation),
        ("Unified Config", test_unified_config),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"\n[OK] {name}: PASSED")
        except Exception as e:
            failed += 1
            print(f"\n[FAIL] {name}: FAILED - {e}")

    print("\n" + "="*80)
    print(f"Total: {len(tests)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(tests)*100):.1f}%")
    print("="*80)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
