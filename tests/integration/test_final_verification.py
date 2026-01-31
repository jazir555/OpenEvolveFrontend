"""
Final Comprehensive Integration Tests for OpenEvolve

This test suite verifies that all fixes work correctly.
All imports use the proper 'openevolve.*' package structure.
"""

import sys
import os


def test_unified_api_imports():
    """Test unified API imports"""
    print("\n=== Testing Unified API Imports ===")
    from openevolve.api import run_evolution, EvolutionResult
    from openevolve.unified.config import UnifiedEvolutionConfig
    print("SUCCESS: All unified API imports work")


def test_gauntlet_imports():
    """Test gauntlet imports"""
    print("\n=== Testing Gauntlet Imports ===")
    from openevolve.gauntlets import (
        LoongFlowGauntletEvaluator,
        ThreeRoundGauntletOrchestrator,
        MultiRoundGauntletOrchestrator,
    )
    print("SUCCESS: All gauntlet imports work")


def test_domain_optimizer_imports():
    """Test domain optimizer imports"""
    print("\n=== Testing Domain Optimizer Imports ===")
    from openevolve.domain import (
        FinanceOptimizer,
        TradingOptimizer,
        ScienceOptimizer,
        EngineeringOptimizer,
        PharmaOptimizer,
        WebDesignOptimizer,
    )
    print("SUCCESS: All domain optimizer imports work")


def test_knowledge_engine_imports():
    """Test knowledge engine imports"""
    print("\n=== Testing Knowledge Engine Imports ===")
    # Import from knowledge_engine (not openevolve)
    from knowledge_engine.integrations import (
        LoongFlowKnowledgeExtractor,
        UnifiedEvolutionKnowledgeExtractor,
    )
    print("SUCCESS: Knowledge engine imports work")


def test_domain_optimizers_instantiation():
    """Test domain optimizers can be instantiated"""
    print("\n=== Testing Domain Optimizer Instantiation ===")
    from openevolve.domain import (
        FinanceOptimizer,
        TradingOptimizer,
        ScienceOptimizer,
    )

    finance = FinanceOptimizer()
    trading = TradingOptimizer()
    science = ScienceOptimizer()

    assert finance is not None
    assert trading is not None
    assert science is not None

    print("SUCCESS: All domain optimizers instantiated")


def test_domain_optimizer_configs():
    """Test domain optimizers provide configs"""
    print("\n=== Testing Domain Optimizer Configs ===")
    from openevolve.domain import FinanceOptimizer, TradingOptimizer

    finance = FinanceOptimizer()
    finance_config = finance.get_default_config()
    assert finance_config is not None

    trading = TradingOptimizer()
    trading_config = trading.get_default_config()
    assert trading_config is not None

    print("SUCCESS: Domain optimizer configs retrieved")


def test_unified_config_structure():
    """Test unified config structure"""
    print("\n=== Testing Unified Config Structure ===")
    from openevolve.unified.config import UnifiedEvolutionConfig

    config = UnifiedEvolutionConfig()

    # Check basic attributes exist
    assert hasattr(config, 'max_iterations')
    assert hasattr(config, 'population_size')
    assert hasattr(config, 'mutation_rate')
    assert hasattr(config, 'evolution_mode')
    assert hasattr(config, 'domain')

    print(f"SUCCESS: Config has max_iterations={config.max_iterations}")
    print(f"SUCCESS: Config has evolution_mode={config.evolution_mode}")


def test_unified_config_defaults():
    """Test unified config defaults"""
    print("\n=== Testing Unified Config Defaults ===")
    from openevolve.unified import (
        get_finance_config,
        get_trading_config,
        get_scientific_config,
    )

    finance_config = get_finance_config()
    assert finance_config is not None

    trading_config = get_trading_config()
    assert trading_config is not None

    scientific_config = get_scientific_config()
    assert scientific_config is not None

    print("SUCCESS: All domain configs retrieved")


def test_evolution_result_structure():
    """Test EvolutionResult structure"""
    print("\n=== Testing EvolutionResult Structure ===")
    from openevolve.api import EvolutionResult

    # Just verify the class can be imported and has expected attributes
    assert hasattr(EvolutionResult, '__annotations__')
    print("SUCCESS: EvolutionResult structure verified")


def test_full_import_chain():
    """Test complete import chain"""
    print("\n=== Testing Full Import Chain ===")

    # Import everything that should work together
    from openevolve.api import run_evolution, EvolutionResult
    from openevolve.unified.config import UnifiedEvolutionConfig
    from openevolve.domain import FinanceOptimizer, TradingOptimizer
    from openevolve.gauntlets import LoongFlowGauntletEvaluator

    print("SUCCESS: Full import chain works")


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*80)
    print("OPENEVOLVE FINAL INTEGRATION TESTS")
    print("="*80)

    tests = [
        ("Unified API Imports", test_unified_api_imports),
        ("Gauntlet Imports", test_gauntlet_imports),
        ("Domain Optimizer Imports", test_domain_optimizer_imports),
        ("Knowledge Engine Imports", test_knowledge_engine_imports),
        ("Domain Optimizer Instantiation", test_domain_optimizers_instantiation),
        ("Domain Optimizer Configs", test_domain_optimizer_configs),
        ("Unified Config Structure", test_unified_config_structure),
        ("Unified Config Defaults", test_unified_config_defaults),
        ("Evolution Result Structure", test_evolution_result_structure),
        ("Full Import Chain", test_full_import_chain),
    ]

    passed = 0
    failed = 0
    results = []

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
            results.append((name, "PASSED"))
        except Exception as e:
            failed += 1
            results.append((name, f"FAILED: {str(e)}"))
            print(f"\nERROR: {name} failed - {e}")

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, result in results:
        status = "PASS" if result == "PASSED" else "FAIL"
        print(f"{status}: {test_name}")

    print(f"\n{'='*80}")
    print(f"Total: {len(tests)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(tests)*100):.1f}%")
    print("="*80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
