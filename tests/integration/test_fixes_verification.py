"""
Comprehensive Integration Tests for OpenEvolve Fixes

This test suite verifies that all fixes applied to the OpenEvolve codebase
are working correctly, including:
- Unified API imports
- Gauntlet imports
- Domain optimizer imports
- Knowledge engine imports
- Basic functionality
- Integration tests
"""

import pytest
import asyncio
from typing import Any, Dict

# =============================================================================
# TEST SUITE 1: Import Tests
# =============================================================================

def test_unified_api_imports():
    """Test that unified API imports work correctly"""
    print("\n=== Testing Unified API Imports ===")

    # Test main API imports - use the actual API structure
    try:
        from openevolve.api import run_evolution, EvolutionResult
        print("[OK] Successfully imported run_evolution and EvolutionResult")
    except ImportError as e:
        pytest.fail(f"Failed to import unified API: {e}")

    # Test unified evolution config
    try:
        from openevolve.unified.config import UnifiedEvolutionConfig
        print("[OK] Successfully imported UnifiedEvolutionConfig")
    except ImportError as e:
        pytest.fail(f"Failed to import UnifiedEvolutionConfig: {e}")

    # Test that run_evolution is callable
    from openevolve.api import run_evolution
    assert callable(run_evolution), "run_evolution function should be callable"
    print("[OK] run_evolution function is callable")


def test_gauntlet_imports():
    """Test that gauntlet imports work correctly"""
    print("\n=== Testing Gauntlet Imports ===")

    try:
        # Gauntlets are in the top-level openevolve package
        import sys
        import os
        openevolve_path = '/c/Users/mmeadow/Documents/OpenEvolve/Frontend/openevolve'
        if openevolve_path not in sys.path:
            sys.path.insert(0, openevolve_path)
        from gauntlets import (
            LoongFlowGauntletEvaluator,
            ThreeRoundGauntletOrchestrator,
            MultiRoundGauntletOrchestrator,
        )
        print("[OK] Successfully imported all gauntlet classes")
    except ImportError as e:
        pytest.fail(f"Failed to import gauntlets: {e}")

    # Verify classes exist
    assert LoongFlowGauntletEvaluator is not None
    assert ThreeRoundGauntletOrchestrator is not None
    assert MultiRoundGauntletOrchestrator is not None
    print("[OK] All gauntlet classes are available")


def test_domain_optimizer_imports():
    """Test that domain optimizer imports work correctly"""
    print("\n=== Testing Domain Optimizer Imports ===")

    try:
        # Domain optimizers are in the top-level openevolve package
        import sys
        sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend/openevolve')
        from domain import (
            FinanceOptimizer,
            TradingOptimizer,
            ScienceOptimizer,
            EngineeringOptimizer,
            PharmaOptimizer,
            WebDesignOptimizer,
        )
        print("[OK] Successfully imported all domain optimizers")
    except ImportError as e:
        pytest.fail(f"Failed to import domain optimizers: {e}")

    # Verify all optimizers exist
    assert FinanceOptimizer is not None
    assert TradingOptimizer is not None
    assert ScienceOptimizer is not None
    assert EngineeringOptimizer is not None
    assert PharmaOptimizer is not None
    assert WebDesignOptimizer is not None
    print("[OK] All domain optimizer classes are available")


def test_knowledge_engine_imports():
    """Test that knowledge engine imports work correctly"""
    print("\n=== Testing Knowledge Engine Imports ===")

    try:
        from knowledge_engine.integrations import (
            LoongFlowKnowledgeExtractor,
            UnifiedEvolutionKnowledgeExtractor,
        )
        print("[OK] Successfully imported knowledge engine integrations")
    except ImportError as e:
        pytest.fail(f"Failed to import knowledge engine: {e}")

    # Verify extractors exist
    assert LoongFlowKnowledgeExtractor is not None
    assert UnifiedEvolutionKnowledgeExtractor is not None
    print("[OK] All knowledge extractor classes are available")


# =============================================================================
# TEST SUITE 2: Basic Functionality Tests
# =============================================================================

def test_domain_optimizers_instantiation():
    """Test that domain optimizers can be instantiated"""
    print("\n=== Testing Domain Optimizer Instantiation ===")

    import sys
    sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend/openevolve')
    from domain import (
        FinanceOptimizer,
        TradingOptimizer,
        ScienceOptimizer,
        EngineeringOptimizer,
        PharmaOptimizer,
        WebDesignOptimizer,
    )

    optimizers = {
        "Finance": FinanceOptimizer(),
        "Trading": TradingOptimizer(),
        "Science": ScienceOptimizer(),
        "Engineering": EngineeringOptimizer(),
        "Pharma": PharmaOptimizer(),
        "WebDesign": WebDesignOptimizer(),
    }

    for name, optimizer in optimizers.items():
        assert optimizer is not None, f"{name} optimizer should be instantiated"
        print(f"[OK] {name} optimizer instantiated successfully")


def test_domain_optimizer_configs():
    """Test that domain optimizers can provide default configs"""
    print("\n=== Testing Domain Optimizer Configs ===")

    import sys
    sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend/openevolve')
    from domain import (
        FinanceOptimizer,
        TradingOptimizer,
        ScienceOptimizer,
    )

    # Test a few key optimizers
    finance = FinanceOptimizer()
    finance_config = finance.get_default_config()
    assert finance_config is not None, "Finance config should not be None"
    print("[OK] Finance optimizer config retrieved")

    trading = TradingOptimizer()
    trading_config = trading.get_default_config()
    assert trading_config is not None, "Trading config should not be None"
    print("[OK] Trading optimizer config retrieved")

    science = ScienceOptimizer()
    science_config = science.get_default_config()
    assert science_config is not None, "Science config should not be None"
    print("[OK] Science optimizer config retrieved")


def test_gauntlet_evaluator_imports():
    """Test that gauntlet evaluators can be imported"""
    print("\n=== Testing Gauntlet Evaluator Imports ===")

    import sys
    sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend/openevolve')
    from gauntlets import LoongFlowGauntletEvaluator

    # Just verify the class can be imported (instantiation may need config)
    assert LoongFlowGauntletEvaluator is not None
    print("[OK] LoongFlowGauntletEvaluator class exists and can be imported")


# =============================================================================
# TEST SUITE 3: Integration Tests
# =============================================================================

def test_unified_config_structure():
    """Test that UnifiedEvolutionConfig structure is correct"""
    print("\n=== Testing UnifiedEvolutionConfig Structure ===")

    from openevolve.unified.config import UnifiedEvolutionConfig

    # Create a config to verify structure
    config = UnifiedEvolutionConfig(
        max_iterations=100,
        population_size=50,
        mutation_rate=0.15,
        crossover_rate=0.75,
    )

    assert config.max_iterations == 100
    assert config.population_size == 50
    assert config.mutation_rate == 0.15
    assert config.crossover_rate == 0.75

    print("[OK] UnifiedEvolutionConfig structure verified")


def test_evolution_result_structure():
    """Test that EvolutionResult structure is correct"""
    print("\n=== Testing EvolutionResult Structure ===")

    from openevolve.api import EvolutionResult
    from openevolve.database import Program

    # Create a dummy result to verify structure
    result = EvolutionResult(
        best_program=Program(code="test code", score=0.95),
        best_score=0.95,
        best_code="test code",
        metrics={"iterations": 10},
        output_dir=None
    )

    assert result.best_score == 0.95
    assert result.best_code == "test code"
    assert result.metrics["iterations"] == 10

    print("[OK] EvolutionResult structure verified")


def test_unified_config_defaults():
    """Test that unified config defaults work"""
    print("\n=== Testing Unified Config Defaults ===")

    from openevolve.unified import get_finance_config, get_trading_config, get_scientific_config

    # Test finance config
    finance_config = get_finance_config()
    assert finance_config is not None
    print("[OK] Finance config default retrieved")

    # Test trading config
    trading_config = get_trading_config()
    assert trading_config is not None
    print("[OK] Trading config default retrieved")

    # Test scientific config
    scientific_config = get_scientific_config()
    assert scientific_config is not None
    print("[OK] Scientific config default retrieved")


# =============================================================================
# TEST SUITE 4: Error Handling Tests
# =============================================================================

def test_import_error_handling():
    """Test that import errors are handled gracefully"""
    print("\n=== Testing Import Error Handling ===")

    # Test that we can catch import errors
    try:
        from openevolve.api import run_evolution
        print("[OK] No import errors detected")
    except ImportError as e:
        print(f"[FAIL] Import error detected: {e}")
        raise


def test_domain_optimizer_error_handling():
    """Test that domain optimizers handle errors gracefully"""
    print("\n=== Testing Domain Optimizer Error Handling ===")

    import sys
    sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend/openevolve')
    from domain import FinanceOptimizer

    optimizer = FinanceOptimizer()

    # Test that optimizer can handle missing optional parameters
    try:
        config = optimizer.get_default_config()
        assert config is not None
        print("[OK] FinanceOptimizer handles default config correctly")
    except Exception as e:
        pytest.fail(f"FinanceOptimizer failed to provide default config: {e}")


# =============================================================================
# TEST SUITE 5: Integration Smoke Tests
# =============================================================================

def test_full_import_chain():
    """Test the full import chain from top to bottom"""
    print("\n=== Testing Full Import Chain ===")

    # This simulates a real usage scenario
    try:
        # 1. Import the main API
        from openevolve.api import run_evolution, EvolutionResult
        print("[OK] Step 1: Main API imported")

        # 2. Import config
        from openevolve.unified.config import UnifiedEvolutionConfig
        print("[OK] Step 2: Config imported")

        # 3. Import domain optimizers
        import sys
        sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend/openevolve')
        from domain import FinanceOptimizer, TradingOptimizer
        print("[OK] Step 3: Domain optimizers imported")

        # 4. Import gauntlets
        import sys
        sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend/openevolve')
        from gauntlets import LoongFlowGauntletEvaluator
        print("[OK] Step 4: Gauntlets imported")

        # 5. Import knowledge engine
        from knowledge_engine.integrations import LoongFlowKnowledgeExtractor
        print("[OK] Step 5: Knowledge engine imported")

        print("[OK] Full import chain successful")

    except ImportError as e:
        pytest.fail(f"Full import chain failed: {e}")


def test_domain_optimizer_methods():
    """Test that domain optimizers have required methods"""
    print("\n=== Testing Domain Optimizer Methods ===")

    import sys
    sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend/openevolve')
    from domain import FinanceOptimizer

    optimizer = FinanceOptimizer()

    # Check for required methods
    assert hasattr(optimizer, 'get_default_config'), "Should have get_default_config method"
    assert hasattr(optimizer, 'validate_config'), "Should have validate_config method"
    assert hasattr(optimizer, 'optimize'), "Should have optimize method"

    print("[OK] Domain optimizer has all required methods")


def test_gauntlet_orchestrator_methods():
    """Test that gauntlet orchestrators have required methods"""
    print("\n=== Testing Gauntlet Orchestrator Methods ===")

    import sys
    sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend/openevolve')
    from gauntlets import ThreeRoundGauntletOrchestrator

    # Check for required methods on the class
    assert hasattr(ThreeRoundGauntletOrchestrator, 'run_gauntlet'), "Should have run_gauntlet method"
    assert hasattr(ThreeRoundGauntletOrchestrator, 'evaluate'), "Should have evaluate method"

    print("[OK] Gauntlet orchestrator has all required methods")


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*80)
    print("OPENEVOLVE COMPREHENSIVE INTEGRATION TESTS")
    print("="*80)

    tests = [
        ("Import Tests", [
            test_unified_api_imports,
            test_gauntlet_imports,
            test_domain_optimizer_imports,
            test_knowledge_engine_imports,
        ]),
        ("Functionality Tests", [
            test_domain_optimizers_instantiation,
            test_domain_optimizer_configs,
            test_gauntlet_evaluator_imports,
        ]),
        ("Error Handling Tests", [
            test_import_error_handling,
            test_domain_optimizer_error_handling,
        ]),
        ("Integration Smoke Tests", [
            test_full_import_chain,
            test_unified_config_structure,
            test_unified_config_defaults,
            test_evolution_result_structure,
            test_domain_optimizer_methods,
            test_gauntlet_orchestrator_methods,
        ]),
    ]

    results = []
    for suite_name, suite_tests in tests:
        print(f"\n{'='*80}")
        print(f"Running: {suite_name}")
        print(f"{'='*80}")

        for test_func in suite_tests:
            try:
                test_func()
                results.append((test_func.__name__, "PASSED"))
            except Exception as e:
                results.append((test_func.__name__, f"FAILED: {str(e)}"))
                print(f"\n[FAIL] {test_func.__name__} FAILED: {e}")

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result == "PASSED")
    failed = sum(1 for _, result in results if result != "PASSED")

    for test_name, result in results:
        status = "[OK]" if result == "PASSED" else "[FAIL]"
        print(f"{status} {test_name}: {result}")

    print(f"\n{'='*80}")
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(results)*100):.1f}%")
    print("="*80)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
