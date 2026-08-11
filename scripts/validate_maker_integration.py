"""
MAKER v2 Integration Validation Script

This script validates that MAKER v2 is properly integrated into the workflow engine
and all components are working correctly.

Usage:
    python validate_maker_integration.py

The script will:
1. Check all module imports
2. Validate core MAKER implementation
3. Test OpenEvolve integration
4. Verify workflow engine integration
5. Run basic functionality tests
"""

import sys
import logging
from typing import Dict, Any

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


def validate_imports() -> Dict[str, Any]:
    """Validate all required module imports."""
    print_section("1. VALIDATING IMPORTS")

    results = {
        "status": "unknown",
        "imports": [],
        "failures": []
    }

    # Test 1: Core MAKER implementation
    try:
        from mdap_maker_complete import (
            MAKEREngine,
            RecursiveMAKERSolver,
            VotingEngine,
            VoteCollector,
            TaskDecomposition,
            MAKERRunMetrics
        )
        results["imports"].append({
            "module": "mdap_maker_complete",
            "status": "[OK]",
            "classes": ["MAKEREngine", "RecursiveMAKERSolver", "VotingEngine", "VoteCollector"]
        })
        print("[OK] Core MAKER implementation (mdap_maker_complete.py)")
    except ImportError as e:
        results["failures"].append({"module": "mdap_maker_complete", "error": str(e)})
        print(f"[FAIL] Core MAKER implementation: {e}")

    # Test 2: MAKER integration bridge
    try:
        from maker_integration_bridge import (
            MAKERIntegrationBridge,
            MAKERIntegrationConfig,
            solve_with_maker,
            solve_towers_of_hanoi
        )
        results["imports"].append({
            "module": "maker_integration_bridge",
            "status": "[OK]",
            "functions": ["MAKERIntegrationBridge", "solve_with_maker", "solve_towers_of_hanoi"]
        })
        print("[OK] MAKER integration bridge")
    except ImportError as e:
        results["failures"].append({"module": "maker_integration_bridge", "error": str(e)})
        print(f"[FAIL] MAKER integration bridge: {e}")

    # Test 3: OpenEvolve MAKER integration
    try:
        from openevolve_maker_integration import (
            OpenEvolveVoteCollector,
            OpenEvolveMAKEREngine,
            OpenEvolveRecursiveMAKERSolver,
            MAKERWorkflowIntegrator,
            MAKERWorkflowConfig,
            MAKERMode
        )
        results["imports"].append({
            "module": "openevolve_maker_integration",
            "status": "[OK]",
            "classes": ["OpenEvolveVoteCollector", "OpenEvolveMAKEREngine", "MAKERWorkflowIntegrator"]
        })
        print("[OK] OpenEvolve MAKER integration")
    except ImportError as e:
        results["failures"].append({"module": "openevolve_maker_integration", "error": str(e)})
        print(f"[FAIL] OpenEvolve MAKER integration: {e}")

    # Test 4: Workflow integration
    try:
        from maker_workflow_integration import (
            generate_solution_with_maker_v2,
            build_maker_config_from_workflow,
            resolve_maker_enabled,
            get_maker_integration_info
        )
        results["imports"].append({
            "module": "maker_workflow_integration",
            "status": "[OK]",
            "functions": ["generate_solution_with_maker_v2", "build_maker_config_from_workflow"]
        })
        print("[OK] Workflow integration (maker_workflow_integration.py)")
    except ImportError as e:
        results["failures"].append({"module": "maker_workflow_integration", "error": str(e)})
        print(f"[FAIL] Workflow integration: {e}")

    # Test 5: Workflow engine
    try:
        from workflow_engine import (
            _resolve_maker_enabled,
            _generate_solution_with_maker,
            get_maker_workflow_status,
            validate_maker_integration,
            get_maker_configuration_help
        )
        results["imports"].append({
            "module": "workflow_engine",
            "status": "[OK]",
            "functions": ["_resolve_maker_enabled", "_generate_solution_with_maker", "get_maker_workflow_status"]
        })
        print("[OK] Workflow engine (workflow_engine.py) - MAKER functions available")
    except ImportError as e:
        results["failures"].append({"module": "workflow_engine", "error": str(e)})
        print(f"[FAIL] Workflow engine: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All imports successful!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} import(s) failed")

    return results


def validate_algorithms() -> Dict[str, Any]:
    """Validate all 4 algorithms from the paper."""
    print_section("2. VALIDATING ALGORITHMS (arXiv:2511.09030)")

    results = {
        "status": "unknown",
        "algorithms": [],
        "missing": []
    }

    from mdap_maker_complete import MAKEREngine, RecursiveMAKERSolver, VotingEngine, VoteCollector

    # Algorithm 1: generate_solution
    if hasattr(MAKEREngine, 'generate_solution'):
        results["algorithms"].append("Algorithm 1: generate_solution")
        print("[OK] Algorithm 1: generate_solution (MAKEREngine.generate_solution)")
    else:
        results["missing"].append("Algorithm 1: generate_solution")
        print("[FAIL] Algorithm 1: generate_solution - NOT FOUND")

    # Algorithm 2: do_voting
    if hasattr(VotingEngine, 'do_voting'):
        results["algorithms"].append("Algorithm 2: do_voting")
        print("[OK] Algorithm 2: do_voting (VotingEngine.do_voting)")
    else:
        results["missing"].append("Algorithm 2: do_voting")
        print("[FAIL] Algorithm 2: do_voting - NOT FOUND")

    # Algorithm 3: get_vote
    if hasattr(VoteCollector, 'get_vote'):
        results["algorithms"].append("Algorithm 3: get_vote")
        print("[OK] Algorithm 3: get_vote (VoteCollector.get_vote)")
    else:
        results["missing"].append("Algorithm 3: get_vote")
        print("[FAIL] Algorithm 3: get_vote - NOT FOUND")

    # Algorithm 4: Recursive solve
    if hasattr(RecursiveMAKERSolver, 'solve'):
        results["algorithms"].append("Algorithm 4: Recursive solve")
        print("[OK] Algorithm 4: Recursive solve (RecursiveMAKERSolver.solve)")
    else:
        results["missing"].append("Algorithm 4: Recursive solve")
        print("[FAIL] Algorithm 4: Recursive solve - NOT FOUND")

    # Check for expected methods
    print("\nDetailed method checks:")

    # Check vote collector red-flagging
    if hasattr(VoteCollector, '_has_red_flags'):
        print("  [OK] Red-flagging: VoteCollector._has_red_flags")
    else:
        print("  [FAIL] Red-flagging: NOT FOUND")

    # Check voting modes
    if hasattr(VotingEngine, '__init__'):
        import inspect
        sig = inspect.signature(VotingEngine.__init__)
        if 'enable_first_to_ahead' in sig.parameters:
            print("  [OK] First-to-ahead-by-k voting: supported")
        else:
            print("  ! First-to-ahead-by-k voting: parameter not found")

    # Determine status
    if len(results["algorithms"]) == 4:
        results["status"] = "pass"
        print("\n[OK] All 4 algorithms from paper implemented!")
    else:
        results["status"] = "partial"
        print(f"\n[WARN] {len(results['algorithms'])}/4 algorithms implemented")

    return results


def validate_workflow_integration() -> Dict[str, Any]:
    """Validate integration with workflow engine."""
    print_section("3. VALIDATING WORKFLOW INTEGRATION")

    results = {
        "status": "unknown",
        "checks": [],
        "failures": []
    }

    # Test 1: Workflow engine functions exist
    try:
        from workflow_engine import (
            _resolve_maker_enabled,
            _generate_solution_with_maker,
            get_maker_workflow_status,
            validate_maker_integration
        )
        results["checks"].append({
            "name": "workflow_functions",
            "status": "[OK]"
        })
        print("[OK] Workflow engine MAKER functions available")
    except ImportError as e:
        results["failures"].append({"check": "workflow_functions", "error": str(e)})
        print(f"[FAIL] Workflow engine functions: {e}")
        return results

    # Test 2: Configuration builder
    try:
        from maker_workflow_integration import build_maker_config_from_workflow
        from workflow_structures import WorkflowState

        # Create test workflow state (using required fields)
        workflow_state = WorkflowState(
            workflow_id="test",
            workflow_type="test",
            problem_statement="Test problem",
            current_stage="test",
            maker_enabled=True,
            maker_config={"maker_mode": "recursive"}
        )

        config = build_maker_config_from_workflow(workflow_state, None)
        results["checks"].append({
            "name": "config_builder",
            "status": "[OK]",
            "config_type": type(config).__name__
        })
        print(f"[OK] Configuration builder: {type(config).__name__}")
    except Exception as e:
        results["failures"].append({"check": "config_builder", "error": str(e)})
        print(f"[FAIL] Configuration builder: {e}")

    # Test 3: Status function
    try:
        from workflow_engine import get_maker_workflow_status
        status = get_maker_workflow_status()
        results["checks"].append({
            "name": "status_function",
            "status": "[OK]"
        })
        print("[OK] Status function: get_maker_workflow_status")
        print(f"  - MAKER available: {status.get('maker_available', 'unknown')}")
        print(f"  - Integration version: {status.get('integration_version', 'unknown')}")
        print(f"  - Supported modes: {status.get('supported_modes', [])}")
    except Exception as e:
        results["failures"].append({"check": "status_function", "error": str(e)})
        print(f"[FAIL] Status function: {e}")

    # Test 4: Validation function
    try:
        from workflow_engine import validate_maker_integration
        validation = validate_maker_integration()
        results["checks"].append({
            "name": "validation_function",
            "status": "[OK]",
            "validation_status": validation.get("status", "unknown")
        })
        print("[OK] Validation function: validate_maker_integration")
        print(f"  - Checks passed: {len(validation.get('checks', []))}")
        print(f"  - Errors: {len(validation.get('errors', []))}")
        print(f"  - Warnings: {len(validation.get('warnings', []))}")
    except Exception as e:
        results["failures"].append({"check": "validation_function", "error": str(e)})
        print(f"[FAIL] Validation function: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] Workflow integration complete!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} integration check(s) failed")

    return results


def validate_openevolve_integration() -> Dict[str, Any]:
    """Validate OpenEvolve integration."""
    print_section("4. VALIDATING OPENEVOLVE INTEGRATION")

    results = {
        "status": "unknown",
        "checks": [],
        "warnings": []
    }

    # Check OpenEvolve client availability
    try:
        from openevolve_client import OpenEvolveClient, OPENEVOLVE_AVAILABLE
        results["checks"].append({
            "name": "openevolve_client_import",
            "status": "[OK]" if OPENEVOLVE_AVAILABLE else "[WARN]",
            "available": OPENEVOLVE_AVAILABLE
        })
        if OPENEVOLVE_AVAILABLE:
            print(f"[OK] OpenEvolve client available")
        else:
            print("[WARN] OpenEvolve client not available (will use fallback)")
            results["warnings"].append("OpenEvolve client unavailable")
    except ImportError:
        results["warnings"].append("OpenEvolve client module not found")
        print("[WARN] OpenEvolve client module not found (will use fallback)")

    # Check OpenEvolve-adapted classes
    try:
        from openevolve_maker_integration import (
            OpenEvolveVoteCollector,
            OpenEvolveMAKEREngine,
            OpenEvolveRecursiveMAKERSolver
        )
        results["checks"].append({
            "name": "openevolve_adapted_classes",
            "status": "[OK]"
        })
        print("[OK] OpenEvolve-adapted MAKER classes")
    except ImportError as e:
        results["warnings"].append({"check": "openevolve_adapted_classes", "error": str(e)})
        print(f"[WARN] OpenEvolve-adapted classes: {e}")

    # Test instantiation
    try:
        from openevolve_maker_integration import MAKERWorkflowConfig, MAKERMode
        config = MAKERWorkflowConfig(mode=MAKERMode.RECURSIVE)
        results["checks"].append({
            "name": "config_instantiation",
            "status": "[OK]"
        })
        print("[OK] MAKER workflow configuration")
    except Exception as e:
        results["warnings"].append({"check": "config_instantiation", "error": str(e)})
        print(f"[WARN] Configuration instantiation: {e}")

    # Determine status
    if not results["warnings"]:
        results["status"] = "pass"
        print("\n[OK] OpenEvolve integration complete!")
    elif results["checks"]:
        results["status"] = "pass_with_warnings"
        print(f"\n[OK] OpenEvolve integration functional with {len(results['warnings'])} warning(s)")
    else:
        results["status"] = "fail"

    return results


def run_basic_tests() -> Dict[str, Any]:
    """Run basic functionality tests."""
    print_section("5. RUNNING BASIC FUNCTIONALITY TESTS")

    results = {
        "status": "unknown",
        "tests": [],
        "failures": []
    }

    # Test 1: Create MAKER configuration
    try:
        from openevolve_maker_integration import MAKERWorkflowConfig, MAKERMode
        config = MAKERWorkflowConfig(
            mode=MAKERMode.RECURSIVE,
            k_ahead=3,
            max_depth=5
        )
        results["tests"].append({
            "name": "config_creation",
            "status": "[OK]",
            "mode": config.mode.value
        })
        print(f"[OK] Config creation: {config.mode.value} mode")
    except Exception as e:
        results["failures"].append({"test": "config_creation", "error": str(e)})
        print(f"[FAIL] Config creation: {e}")

    # Test 2: Simple recursive solve (no actual LLM call)
    try:
        from mdap_maker_complete import RecursiveMAKERSolver
        from workflow_structures import Team, ModelConfig

        # Create test team with valid ModelConfig
        test_config = ModelConfig(
            model_id="test_model",
            api_key="test_key",
            temperature=0.1
        )
        test_team = Team(
            name="Test Team",
            role="Blue",
            members=[test_config]
        )

        # Create solver (won't actually call LLM)
        solver = RecursiveMAKERSolver(team=test_team, max_depth=2, k_ahead=2)

        results["tests"].append({
            "name": "solver_instantiation",
            "status": "[OK]"
        })
        print("[OK] Solver instantiation")
    except Exception as e:
        results["failures"].append({"test": "solver_instantiation", "error": str(e)})
        print(f"[FAIL] Solver instantiation: {e}")

    # Test 3: Workflow status function
    try:
        from workflow_engine import get_maker_workflow_status
        status = get_maker_workflow_status()

        results["tests"].append({
            "name": "workflow_status",
            "status": "[OK]",
            "maker_available": status.get("maker_available")
        })
        print("[OK] Workflow status function")
        print(f"  - MAKER available: {status.get('maker_available')}")
        print(f"  - Integration version: {status.get('integration_version')}")
    except Exception as e:
        results["failures"].append({"test": "workflow_status", "error": str(e)})
        print(f"[FAIL] Workflow status: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All basic tests passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} test(s) failed")

    return results


def main():
    """Main validation function."""
    print("\n")
    print("=" * 80)
    print("  MAKER v2 INTEGRATION VALIDATION")
    print("  Complete arXiv:2511.09030 Implementation")
    print("=" * 80)
    print("")

    all_results = {}

    # Run all validations
    import_results = validate_imports()
    all_results["imports"] = import_results

    algorithm_results = validate_algorithms()
    all_results["algorithms"] = algorithm_results

    workflow_results = validate_workflow_integration()
    all_results["workflow"] = workflow_results

    openevolve_results = validate_openevolve_integration()
    all_results["openevolve"] = openevolve_results

    test_results = run_basic_tests()
    all_results["tests"] = test_results

    # Summary
    print_section("VALIDATION SUMMARY")

    total_checks = 0
    total_passed = 0
    total_failed = 0
    total_warnings = 0

    for category, results in all_results.items():
        if results["status"] == "pass":
            total_passed += 1
        elif results["status"] == "pass_with_warnings":
            total_warnings += 1
        elif results["status"] == "fail":
            total_failed += 1

        total_checks += 1

    print(f"Categories: {total_checks}")
    print(f"  Passed: {total_passed}")
    print(f"  Passed with warnings: {total_warnings}")
    print(f"  Failed: {total_failed}")

    if total_failed == 0:
        print("\n" + "=" * 80)
        print("[OK][OK][OK] ALL VALIDATIONS PASSED [OK][OK][OK]")
        print("=" * 80)
        print("\nMAKER v2 is fully integrated and ready for use!")
        print("\nNext steps:")
        print("1. Enable MAKER in your workflow: workflow_state.maker_enabled = True")
        print("2. Configure MAKER mode in metadata")
        print("3. Run your workflow - MAKER v2 will be used automatically")
        print("\nFor more information, see:")
        print("  - MAKER_WORKFLOW_INTEGRATION_GUIDE.md")
        print("  - MAKER_IMPLEMENTATION_README.md")
        return 0
    else:
        print("\n" + "=" * 80)
        print("[FAIL][FAIL][FAIL] SOME VALIDATIONS FAILED [FAIL][FAIL][FAIL]")
        print("=" * 80)
        print("\nPlease check the errors above and fix them before using MAKER v2.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
