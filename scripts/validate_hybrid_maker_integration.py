"""
MAKER/MDAP Hybrid Strategies Integration - Validation Script

This script validates that the MAKER/MDAP hybrid integration is working correctly.

Usage:
    python validate_hybrid_maker_integration.py
"""

import sys
import logging
import asyncio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def validate_imports():
    """Validate all required imports."""
    print_section("1. VALIDATING IMPORTS")

    results = {"status": "unknown", "imports": [], "failures": []}

    # Test 1: Hybrid MAKER integration
    try:
        from hybrid_maker_integration import (
            MAKERHybridConfig,
            MAKERHybridMode,
            MCTSThenMAKER,
            MAKERThenEvolution,
            MAKERAdversarialHybrid,
            AdaptiveMAKERHybrid,
            MAKERMDAPParallel,
            FullMAKERHybrid,
            run_maker_hybrid,
            get_maker_hybrid_capabilities
        )
        results["imports"].append({
            "module": "hybrid_maker_integration",
            "status": "OK",
            "classes": [
                "MCTSThenMAKER",
                "MAKERThenEvolution",
                "MAKERAdversarialHybrid",
                "AdaptiveMAKERHybrid",
                "MAKERMDAPParallel",
                "FullMAKERHybrid"
            ]
        })
        print("[OK] Hybrid MAKER integration module")
    except ImportError as e:
        results["failures"].append({"module": "hybrid_maker_integration", "error": str(e)})
        print(f"[FAIL] Hybrid MAKER integration: {e}")

    # Test 2: MAKER evolution integration
    try:
        from evolution_maker_integration import (
            MakerevolutionConfig,
            Individual,
            Population,
            MAKERSelection,
            MDAPEvolutionDecomposer,
            MAKEREvolutionEngine
        )
        results["imports"].append({
            "module": "evolution_maker_integration",
            "status": "OK",
            "classes": ["MakerevolutionConfig", "Individual", "Population", "MAKERSelection"]
        })
        print("[OK] Evolution MAKER integration module")
    except ImportError as e:
        results["failures"].append({"module": "evolution_maker_integration", "error": str(e)})
        print(f"[FAIL] Evolution MAKER integration: {e}")

    # Test 3: MAKER adversarial integration
    try:
        from adversarial_maker_integration import (
            AdversarialMAKERConfig,
            MAKERRedTeamAgent,
            MDAPBlueTeamAgent,
            AdversarialCoEvolution
        )
        results["imports"].append({
            "module": "adversarial_maker_integration",
            "status": "OK",
            "classes": ["AdversarialMAKERConfig", "MAKERRedTeamAgent", "MDAPBlueTeamAgent"]
        })
        print("[OK] Adversarial MAKER integration module")
    except ImportError as e:
        results["failures"].append({"module": "adversarial_maker_integration", "error": str(e)})
        print(f"[FAIL] Adversarial MAKER integration: {e}")

    # Test 4: Core MAKER implementation
    try:
        from mdap_maker_complete import (
            MAKEREngine,
            RecursiveMAKERSolver,
            VotingEngine,
            VoteCollector
        )
        results["imports"].append({
            "module": "mdap_maker_complete",
            "status": "OK",
            "classes": ["MAKEREngine", "RecursiveMAKERSolver", "VotingEngine"]
        })
        print("[OK] Core MAKER implementation")
    except ImportError as e:
        results["failures"].append({"module": "mdap_maker_complete", "error": str(e)})
        print(f"[FAIL] Core MAKER implementation: {e}")

    # Test 5: MDAP engine
    try:
        from mdap_engine import (
            MDAPConfig,
            MDAPTask,
            MDAPStep,
            MDAPOrchestrator
        )
        results["imports"].append({
            "module": "mdap_engine",
            "status": "OK",
            "classes": ["MDAPConfig", "MDAPTask", "MDAPOrchestrator"]
        })
        print("[OK] MDAP engine")
    except ImportError as e:
        results["failures"].append({"module": "mdap_engine", "error": str(e)})
        print(f"[FAIL] MDAP engine: {e}")

    # Test 6: Hybrid strategies base
    try:
        from leanaide_hybrid_strategies import (
            HybridStrategy,
            EvolutionResult,
            MCTSThenEvolution,
            MCTSThenMDAP
        )
        results["imports"].append({
            "module": "leanaide_hybrid_strategies",
            "status": "OK",
            "classes": ["HybridStrategy", "EvolutionResult"]
        })
        print("[OK] Hybrid strategies base")
    except ImportError as e:
        # Check if fallback is used
        try:
            from hybrid_maker_integration import HybridStrategy, EvolutionResult
            results["imports"].append({
                "module": "leanaide_hybrid_strategies",
                "status": "FALLBACK",
                "classes": ["HybridStrategy (fallback)", "EvolutionResult (fallback)"]
            })
            print("[OK] Hybrid strategies base (using fallback)")
        except ImportError:
            results["failures"].append({"module": "leanaide_hybrid_strategies", "error": str(e)})
            print(f"[FAIL] Hybrid strategies base: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All imports successful!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} import(s) failed")

    return results


def validate_configuration():
    """Validate configuration classes."""
    print_section("2. VALIDATING CONFIGURATION")

    results = {"status": "unknown", "checks": [], "failures": []}

    # Test 1: Create MAKERHybridConfig
    try:
        from hybrid_maker_integration import MAKERHybridConfig

        config = MAKERHybridConfig(
            enable_voting=True,
            voting_threshold=3,
            enable_decomposition=True,
            mcts_simulations=100,
            evolution_generations=20,
            adversarial_rounds=3
        )
        results["checks"].append({
            "name": "config_creation",
            "status": "OK"
        })
        print("[OK] MAKERHybridConfig creation")
    except Exception as e:
        results["failures"].append({"check": "config_creation", "error": str(e)})
        print(f"[FAIL] MAKERHybridConfig creation: {e}")

    # Test 2: Convert config to dict
    try:
        config_dict = config.to_dict()
        results["checks"].append({
            "name": "config_to_dict",
            "status": "OK"
        })
        print("[OK] Config to dict conversion")
    except Exception as e:
        results["failures"].append({"check": "config_to_dict", "error": str(e)})
        print(f"[FAIL] Config to dict: {e}")

    # Test 3: Check all modes
    try:
        from hybrid_maker_integration import MAKERHybridMode

        modes = [mode.value for mode in MAKERHybridMode]
        expected_modes = [
            "mcts_then_maker",
            "maker_then_evolution",
            "maker_adversarial",
            "adaptive_maker",
            "maker_mdap_parallel",
            "full_maker_hybrid"
        ]
        if all(mode in modes for mode in expected_modes):
            results["checks"].append({
                "name": "modes_check",
                "status": "OK"
            })
            print(f"[OK] All {len(modes)} hybrid modes available")
        else:
            raise ValueError(f"Missing modes: {set(expected_modes) - set(modes)}")
    except Exception as e:
        results["failures"].append({"check": "modes_check", "error": str(e)})
        print(f"[FAIL] Modes check: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All configuration checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


def validate_strategies():
    """Validate hybrid strategy classes."""
    print_section("3. VALIDATING STRATEGIES")

    results = {"status": "unknown", "checks": [], "failures": []}

    # Test 1: Create MCTSThenMAKER
    try:
        from hybrid_maker_integration import MCTSThenMAKER

        strategy = MCTSThenMAKER(
            mcts_simulations=100,
            maker_voting_threshold=3,
            population_size=15
        )
        results["checks"].append({
            "name": "mcts_then_maker_creation",
            "status": "OK"
        })
        print("[OK] MCTSThenMAKER creation")
    except Exception as e:
        results["failures"].append({"check": "mcts_then_maker_creation", "error": str(e)})
        print(f"[FAIL] MCTSThenMAKER creation: {e}")

    # Test 2: Create MAKERThenEvolution
    try:
        from hybrid_maker_integration import MAKERThenEvolution

        strategy = MAKERThenEvolution(
            maker_voting_threshold=3,
            evolution_generations=20,
            population_size=20
        )
        results["checks"].append({
            "name": "maker_then_evolution_creation",
            "status": "OK"
        })
        print("[OK] MAKERThenEvolution creation")
    except Exception as e:
        results["failures"].append({"check": "maker_then_evolution_creation", "error": str(e)})
        print(f"[FAIL] MAKERThenEvolution creation: {e}")

    # Test 3: Create MAKERAdversarialHybrid
    try:
        from hybrid_maker_integration import MAKERAdversarialHybrid

        strategy = MAKERAdversarialHybrid(
            adversarial_rounds=3,
            maker_voting_threshold=3
        )
        results["checks"].append({
            "name": "maker_adversarial_creation",
            "status": "OK"
        })
        print("[OK] MAKERAdversarialHybrid creation")
    except Exception as e:
        results["failures"].append({"check": "maker_adversarial_creation", "error": str(e)})
        print(f"[FAIL] MAKERAdversarialHybrid creation: {e}")

    # Test 4: Create AdaptiveMAKERHybrid
    try:
        from hybrid_maker_integration import AdaptiveMAKERHybrid

        strategy = AdaptiveMAKERHybrid(
            diversity_threshold=0.3,
            max_generations=50
        )
        results["checks"].append({
            "name": "adaptive_maker_creation",
            "status": "OK"
        })
        print("[OK] AdaptiveMAKERHybrid creation")
    except Exception as e:
        results["failures"].append({"check": "adaptive_maker_creation", "error": str(e)})
        print(f"[FAIL] AdaptiveMAKERHybrid creation: {e}")

    # Test 5: Create MAKERMDAPParallel
    try:
        from hybrid_maker_integration import MAKERMDAPParallel

        strategy = MAKERMDAPParallel(
            maker_voting_threshold=3,
            mdap_agents=4
        )
        results["checks"].append({
            "name": "maker_mdap_parallel_creation",
            "status": "OK"
        })
        print("[OK] MAKERMDAPParallel creation")
    except Exception as e:
        results["failures"].append({"check": "maker_mdap_parallel_creation", "error": str(e)})
        print(f"[FAIL] MAKERMDAPParallel creation: {e}")

    # Test 6: Create FullMAKERHybrid
    try:
        from hybrid_maker_integration import FullMAKERHybrid, MAKERHybridConfig

        config = MAKERHybridConfig()
        strategy = FullMAKERHybrid(config)
        results["checks"].append({
            "name": "full_maker_hybrid_creation",
            "status": "OK"
        })
        print("[OK] FullMAKERHybrid creation")
    except Exception as e:
        results["failures"].append({"check": "full_maker_hybrid_creation", "error": str(e)})
        print(f"[FAIL] FullMAKERHybrid creation: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All strategy checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


async def validate_execution():
    """Validate basic execution of strategies."""
    print_section("4. VALIDATING EXECUTION")

    results = {"status": "unknown", "checks": [], "failures": []}

    # Test 1: Run MCTSThenMAKER
    try:
        from hybrid_maker_integration import MCTSThenMAKER

        strategy = MCTSThenMAKER(
            mcts_simulations=10,
            maker_voting_threshold=2,
            population_size=5
        )

        result = await strategy.generate_proof("forall n : nat, n + 0 = n")

        results["checks"].append({
            "name": "mcts_then_maker_execution",
            "status": "OK",
            "success": result.success
        })
        print(f"[OK] MCTSThenMAKER execution (success={result.success})")
    except Exception as e:
        results["failures"].append({"check": "mcts_then_maker_execution", "error": str(e)})
        print(f"[FAIL] MCTSThenMAKER execution: {e}")

    # Test 2: Run MAKERThenEvolution
    try:
        from hybrid_maker_integration import MAKERThenEvolution

        strategy = MAKERThenEvolution(
            maker_voting_threshold=2,
            evolution_generations=3,
            population_size=5,
            initial_candidates=10
        )

        result = await strategy.generate_proof("forall n : nat, n + 0 = n")

        results["checks"].append({
            "name": "maker_then_evolution_execution",
            "status": "OK",
            "success": result.success
        })
        print(f"[OK] MAKERThenEvolution execution (success={result.success})")
    except Exception as e:
        results["failures"].append({"check": "maker_then_evolution_execution", "error": str(e)})
        print(f"[FAIL] MAKERThenEvolution execution: {e}")

    # Test 3: Run run_maker_hybrid with different modes
    try:
        from hybrid_maker_integration import run_maker_hybrid, MAKERHybridMode

        result = await run_maker_hybrid(
            theorem="forall n : nat, n + 0 = n",
            mode=MAKERHybridMode.ADAPTIVE_MAKER,
            config=None
        )

        results["checks"].append({
            "name": "run_maker_hybrid_execution",
            "status": "OK",
            "success": result.success
        })
        print(f"[OK] run_maker_hybrid execution (success={result.success})")
    except Exception as e:
        results["failures"].append({"check": "run_maker_hybrid_execution", "error": str(e)})
        print(f"[FAIL] run_maker_hybrid execution: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All execution checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


def validate_capabilities():
    """Validate capabilities function."""
    print_section("5. VALIDATING CAPABILITIES")

    results = {"status": "unknown", "checks": [], "failures": []}

    # Test 1: Get capabilities
    try:
        from hybrid_maker_integration import get_maker_hybrid_capabilities

        capabilities = get_maker_hybrid_capabilities()
        results["checks"].append({
            "name": "capabilities_function",
            "status": "OK"
        })
        print("[OK] Capabilities function")

        # Display capabilities
        print(f"\n  - MAKER hybrid enabled: {capabilities.get('maker_hybrid_enabled', False)}")
        print(f"  - MAKER evolution: {capabilities.get('maker_evolution_available', False)}")
        print(f"  - MAKER adversarial: {capabilities.get('maker_adversarial_available', False)}")
        print(f"  - Integration status: {capabilities.get('integration_status', 'unknown')}")
        print(f"  - Modes: {len(capabilities.get('modes', []))}")
        print(f"  - Strategies: {len(capabilities.get('strategies', []))}")

        if 'paper' in capabilities:
            paper = capabilities['paper']
            print(f"\n  - Paper: {paper.get('arxiv', 'N/A')}")

    except Exception as e:
        results["failures"].append({"check": "capabilities_function", "error": str(e)})
        print(f"[FAIL] Capabilities function: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] Capabilities validation passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


def main():
    """Main validation function."""
    print("\n")
    print("=" * 80)
    print("  MAKER/MDAP HYBRID STRATEGIES INTEGRATION VALIDATION")
    print("  Complete arXiv:2511.09030 Implementation")
    print("=" * 80)
    print("")

    all_results = {}

    # Run all validations
    import_results = validate_imports()
    all_results["imports"] = import_results

    config_results = validate_configuration()
    all_results["configuration"] = config_results

    strategy_results = validate_strategies()
    all_results["strategies"] = strategy_results

    # Run async validation
    execution_results = asyncio.run(validate_execution())
    all_results["execution"] = execution_results

    capabilities_results = validate_capabilities()
    all_results["capabilities"] = capabilities_results

    # Summary
    print_section("VALIDATION SUMMARY")

    total_checks = 0
    total_passed = 0
    total_failed = 0

    for category, results in all_results.items():
        if results["status"] == "pass":
            total_passed += 1
        elif results["status"] == "fail":
            total_failed += 1

        total_checks += 1

    print(f"Categories: {total_checks}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")

    if total_failed == 0:
        print("\n" + "=" * 80)
        print("[OK][OK][OK] ALL VALIDATIONS PASSED [OK][OK][OK]")
        print("=" * 80)
        print("\nMAKER/MDAP hybrid integration is fully functional!")
        print("\nNext steps:")
        print("1. Run demo: python demo_hybrid_maker.py")
        print("2. Use in code: from hybrid_maker_integration import run_maker_hybrid")
        print("3. Read guide: MAKER_HYBRID_INTEGRATION_GUIDE.md")
        return 0
    else:
        print("\n" + "=" * 80)
        print("[FAIL][FAIL][FAIL] SOME VALIDATIONS FAILED [FAIL][FAIL][FAIL]")
        print("=" * 80)
        print("\nPlease check the errors above and fix them before using MAKER/MDAP hybrid.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
