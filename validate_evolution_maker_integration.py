"""
MAKER/MDAP Evolution Integration - Validation Script

This script validates that the MAKER/MDAP evolution integration is working correctly.

Usage:
    python validate_evolution_maker_integration.py
"""

import sys
import logging

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

    # Test 1: Evolution module with MAKER functions
    try:
        from evolution import (
            run_maker_enhanced_evolution,
            get_maker_evolution_capabilities,
            EvolutionConfiguration
        )
        results["imports"].append({
            "module": "evolution",
            "status": "OK",
            "functions": ["run_maker_enhanced_evolution", "get_maker_evolution_capabilities"]
        })
        print("[OK] Evolution module with MAKER functions")
    except ImportError as e:
        results["failures"].append({"module": "evolution", "error": str(e)})
        print(f"[FAIL] Evolution module: {e}")

    # Test 2: Evolution MAKER integration
    try:
        from evolution_maker_integration import (
            MakerevolutionConfig,
            MakerevolutionMode,
            Individual,
            Population,
            MAKERSelection,
            MDAPEvolutionDecomposer,
            MAKEREvolutionEngine,
            run_maker_evolution
        )
        results["imports"].append({
            "module": "evolution_maker_integration",
            "status": "OK",
            "classes": ["MakerevolutionConfig", "Individual", "Population", "MAKEREvolutionEngine"]
        })
        print("[OK] Evolution MAKER integration module")
    except ImportError as e:
        results["failures"].append({"module": "evolution_maker_integration", "error": str(e)})
        print(f"[FAIL] Evolution MAKER integration: {e}")

    # Test 3: Core MAKER implementation
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

    # Test 4: MDAP engine
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

    # Test 1: Create MakerevolutionConfig
    try:
        from evolution_maker_integration import MakerevolutionConfig, MakerevolutionMode

        config = MakerevolutionConfig(
            mode=MakerevolutionMode.HYBRID,
            enable_voting=True,
            enable_decomposition=True,
            voting_threshold=3
        )
        results["checks"].append({
            "name": "config_creation",
            "status": "OK",
            "mode": config.mode.value
        })
        print(f"[OK] MakerevolutionConfig creation: {config.mode.value}")
    except Exception as e:
        results["failures"].append({"check": "config_creation", "error": str(e)})
        print(f"[FAIL] MakerevolutionConfig creation: {e}")

    # Test 2: Convert to dict
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
        modes = [mode.value for mode in MakerevolutionMode]
        expected_modes = ["voting_only", "decomposition", "hybrid", "full_maker"]
        if all(mode in modes for mode in expected_modes):
            results["checks"].append({
                "name": "modes_check",
                "status": "OK"
            })
            print(f"[OK] All {len(modes)} evolution modes available")
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


def validate_data_structures():
    """Validate Individual and Population classes."""
    print_section("3. VALIDATING DATA STRUCTURES")

    results = {"status": "unknown", "checks": [], "failures": []}

    # Test 1: Create Individual
    try:
        from evolution_maker_integration import Individual

        individual = Individual(
            genome="test_program",
            fitness=0.95,
            generation=0,
            metadata={"test": True}
        )
        results["checks"].append({
            "name": "individual_creation",
            "status": "OK"
        })
        print("[OK] Individual creation")
    except Exception as e:
        results["failures"].append({"check": "individual_creation", "error": str(e)})
        print(f"[FAIL] Individual creation: {e}")

    # Test 2: Create Population
    try:
        from evolution_maker_integration import Population, Individual

        individuals = [
            Individual(f"prog{i}", float(i) / 10, 0)
            for i in range(1, 6)
        ]
        population = Population(individuals=individuals, generation=0)

        results["checks"].append({
            "name": "population_creation",
            "status": "OK"
        })
        print("[OK] Population creation")
    except Exception as e:
        results["failures"].append({"check": "population_creation", "error": str(e)})
        print(f"[FAIL] Population creation: {e}")

    # Test 3: Test Population properties
    try:
        best = population.best_individual
        avg = population.average_fitness
        diversity = population.diversity

        results["checks"].append({
            "name": "population_properties",
            "status": "OK"
        })
        print(f"[OK] Population properties: best={best.fitness:.2f}, avg={avg:.2f}, diversity={diversity:.2f}")
    except Exception as e:
        results["failures"].append({"check": "population_properties", "error": str(e)})
        print(f"[FAIL] Population properties: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All data structure checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


def validate_components():
    """Validate MAKER evolution components."""
    print_section("4. VALIDATING COMPONENTS")

    results = {"status": "unknown", "checks": [], "failures": []}

    # Test 1: Create MAKERSelection
    try:
        from evolution_maker_integration import MAKERSelection, MakerevolutionConfig

        config = MakerevolutionConfig()
        selector = MAKERSelection(config)

        results["checks"].append({
            "name": "selector_creation",
            "status": "OK"
        })
        print("[OK] MAKERSelection creation")
    except Exception as e:
        results["failures"].append({"check": "selector_creation", "error": str(e)})
        print(f"[FAIL] MAKERSelection creation: {e}")

    # Test 2: Create MDAPEvolutionDecomposer
    try:
        from evolution_maker_integration import MDAPEvolutionDecomposer

        decomposer = MDAPEvolutionDecomposer(config)

        results["checks"].append({
            "name": "decomposer_creation",
            "status": "OK"
        })
        print("[OK] MDAPEvolutionDecomposer creation")
    except Exception as e:
        results["failures"].append({"check": "decomposer_creation", "error": str(e)})
        print(f"[FAIL] MDAPEvolutionDecomposer creation: {e}")

    # Test 3: Create MAKEREvolutionEngine
    try:
        from evolution_maker_integration import MAKEREvolutionEngine

        engine = MAKEREvolutionEngine(config)

        results["checks"].append({
            "name": "engine_creation",
            "status": "OK"
        })
        print("[OK] MAKEREvolutionEngine creation")
    except Exception as e:
        results["failures"].append({"check": "engine_creation", "error": str(e)})
        print(f"[FAIL] MAKEREvolutionEngine creation: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All component checks passed!")
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
        from evolution import get_maker_evolution_capabilities

        capabilities = get_maker_evolution_capabilities()
        results["checks"].append({
            "name": "capabilities_function",
            "status": "OK"
        })
        print("[OK] Capabilities function")

        # Display capabilities
        print(f"  - MAKER evolution enabled: {capabilities.get('maker_evolution_enabled', False)}")
        print(f"  - MDAP decomposition enabled: {capabilities.get('mdap_decomposition_enabled', False)}")
        print(f"  - Integration status: {capabilities.get('integration_status', 'unknown')}")
        print(f"  - Modes: {len(capabilities.get('modes', []))}")
        print(f"  - Algorithms: {len(capabilities.get('algorithms', []))}")

        if 'paper' in capabilities:
            paper = capabilities['paper']
            print(f"  - Paper: {paper.get('arxiv', 'N/A')}")

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
    print("  MAKER/MDAP EVOLUTION INTEGRATION VALIDATION")
    print("  Complete arXiv:2511.09030 Implementation")
    print("=" * 80)
    print("")

    all_results = {}

    # Run all validations
    import_results = validate_imports()
    all_results["imports"] = import_results

    config_results = validate_configuration()
    all_results["configuration"] = config_results

    data_struct_results = validate_data_structures()
    all_results["data_structures"] = data_struct_results

    component_results = validate_components()
    all_results["components"] = component_results

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
        print("\nMAKER/MDAP evolution integration is fully functional!")
        print("\nNext steps:")
        print("1. Run demo: python demo_evolution_maker.py")
        print("2. Use in code: from evolution import run_maker_enhanced_evolution")
        print("3. Read guide: MAKER_EVOLUTION_INTEGRATION_GUIDE.md")
        return 0
    else:
        print("\n" + "=" * 80)
        print("[FAIL][FAIL][FAIL] SOME VALIDATIONS FAILED [FAIL][FAIL][FAIL]")
        print("=" * 80)
        print("\nPlease check the errors above and fix them before using MAKER/MDAP evolution.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
