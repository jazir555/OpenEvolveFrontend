"""
Generic MAKER/MDAP Integration - Validation Script

Validates that the generic MAKER integration works for any task type.

Usage:
    python validate_generic_maker_integration.py
"""

import sys
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def validate_imports():
    """Validate imports"""
    print_section("1. VALIDATING IMPORTS")

    results = {"status": "unknown", "imports": [], "failures": []}

    # Test 1: Generic MAKER integration
    try:
        from generic_maker_integration import (
            run_generic_maker,
            GenericMAKERSolver,
            GenericEvaluator,
            GenericTask,
            GenericSolution,
            MAKERConfig,
            TaskType
        )
        results["imports"].append({
            "module": "generic_maker_integration",
            "status": "OK"
        })
        print("[OK] Generic MAKER integration module")
    except ImportError as e:
        results["failures"].append({"module": "generic_maker_integration", "error": str(e)})
        print(f"[FAIL] Generic MAKER integration: {e}")

    # Test 2: Core MAKER
    try:
        from mdap_maker_complete import MAKEREngine, VotingEngine
        results["imports"].append({
            "module": "mdap_maker_complete",
            "status": "OK"
        })
        print("[OK] Core MAKER module")
    except ImportError as e:
        results["failures"].append({"module": "mdap_maker_complete", "error": str(e)})
        print(f"[FAIL] Core MAKER: {e}")

    # Test 3: MDAP engine
    try:
        from mdap_engine import MDAPOrchestrator, MDAPConfig
        results["imports"].append({
            "module": "mdap_engine",
            "status": "OK"
        })
        print("[OK] MDAP engine module")
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


def validate_types():
    """Validate type definitions"""
    print_section("2. VALIDATING TYPE DEFINITIONS")

    results = {"status": "unknown", "checks": [], "failures": []}

    # Test 1: TaskType enum
    try:
        from generic_maker_integration import TaskType

        types = [t.value for t in TaskType]
        expected_types = [
            "code_generation",
            "code_refactoring",
            "document_processing",
            "text_summarization",
            "data_analysis",
            "workflow_orchestration",
            "optimization",
            "custom"
        ]

        if all(t in types for t in expected_types):
            results["checks"].append({
                "name": "task_types",
                "status": "OK",
                "count": len(types)
            })
            print(f"[OK] All {len(types)} task types available")
        else:
            raise ValueError(f"Missing task types")

    except Exception as e:
        results["failures"].append({"check": "task_types", "error": str(e)})
        print(f"[FAIL] Task types: {e}")

    # Test 2: GenericTask
    try:
        from generic_maker_integration import GenericTask, TaskType

        task = GenericTask(
            task_id="test_task",
            description="Test description",
            task_type=TaskType.CUSTOM
        )

        task_dict = task.to_dict()

        results["checks"].append({
            "name": "generic_task",
            "status": "OK"
        })
        print("[OK] GenericTask creation and serialization")

    except Exception as e:
        results["failures"].append({"check": "generic_task", "error": str(e)})
        print(f"[FAIL] GenericTask: {e}")

    # Test 3: GenericSolution
    try:
        from generic_maker_integration import GenericSolution

        solution = GenericSolution(
            task_id="test_task",
            solution="Test solution",
            quality_score=0.85
        )

        solution_dict = solution.to_dict()

        results["checks"].append({
            "name": "generic_solution",
            "status": "OK"
        })
        print("[OK] GenericSolution creation and serialization")

    except Exception as e:
        results["failures"].append({"check": "generic_solution", "error": str(e)})
        print(f"[FAIL] GenericSolution: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All type definitions validated!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


def validate_configuration():
    """Validate configuration"""
    print_section("3. VALIDATING CONFIGURATION")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from generic_maker_integration import MAKERConfig

        # Test default config
        config = MAKERConfig()

        results["checks"].append({
            "name": "default_config",
            "status": "OK"
        })
        print("[OK] Default MAKERConfig creation")

        # Test custom config
        custom_config = MAKERConfig(
            enable_voting=True,
            voting_threshold=5,
            enable_decomposition=True,
            max_generations=100
        )

        config_dict = custom_config.to_dict()

        results["checks"].append({
            "name": "custom_config",
            "status": "OK"
        })
        print("[OK] Custom MAKERConfig creation")

        # Verify config values
        if custom_config.voting_threshold == 5:
            results["checks"].append({
                "name": "config_values",
                "status": "OK"
            })
            print("[OK] Configuration values set correctly")
        else:
            raise ValueError(f"Config value mismatch: expected 5, got {custom_config.voting_threshold}")

    except Exception as e:
        results["failures"].append({"check": "configuration", "error": str(e)})
        print(f"[FAIL] Configuration: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All configuration checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


async def validate_execution():
    """Validate basic execution"""
    print_section("4. VALIDATING EXECUTION")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from generic_maker_integration import (
            run_generic_maker,
            GenericEvaluator,
            TaskType
        )

        # Create a simple test evaluator
        class TestEvaluator(GenericEvaluator):
            def evaluate(self, solution: str, task) -> float:
                # Simple heuristic: prefer longer solutions
                return min(1.0, len(solution) / 100.0)

            def get_evaluation_details(self):
                return {"test": "evaluator"}

        evaluator = TestEvaluator()

        # Test 1: Code generation task
        print("  Testing code generation task...")
        result = await run_generic_maker(
            task_description="Generate a function to add two numbers",
            evaluator=evaluator,
            task_type=TaskType.CODE_GENERATION,
            config=None  # Use defaults
        )

        if result and result.solution:
            results["checks"].append({
                "name": "code_generation",
                "status": "OK",
                "quality": result.quality_score
            })
            print(f"[OK] Code generation: quality={result.quality_score:.3f}")
        else:
            raise ValueError("Code generation failed")

        # Test 2: Custom task
        print("  Testing custom task...")
        result2 = await run_generic_maker(
            task_description="Optimize this process",
            evaluator=evaluator,
            task_type=TaskType.CUSTOM,
            config=None
        )

        if result2 and result2.solution:
            results["checks"].append({
                "name": "custom_task",
                "status": "OK",
                "quality": result2.quality_score
            })
            print(f"[OK] Custom task: quality={result2.quality_score:.3f}")
        else:
            raise ValueError("Custom task failed")

    except Exception as e:
        results["failures"].append({"check": "execution", "error": str(e)})
        print(f"[FAIL] Execution: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All execution checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


def validate_capabilities():
    """Validate capabilities function"""
    print_section("5. VALIDATING CAPABILITIES")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from generic_maker_integration import get_generic_maker_capabilities

        capabilities = get_generic_maker_capabilities()

        # Display capabilities
        print("Generic MAKER Capabilities:")
        print(f"  - MAKER enabled: {capabilities.get('generic_maker_enabled', False)}")
        print(f"  - MDAP available: {capabilities.get('mdap_available', False)}")
        print(f"  - Integration status: {capabilities.get('integration_status', 'unknown')}")

        print(f"\n  Supported Task Types ({len(capabilities.get('supported_task_types', []))}):")
        for task_type in capabilities.get('supported_task_types', []):
            print(f"    - {task_type}")

        print(f"\n  Features:")
        for feature, desc in capabilities.get('features', {}).items():
            print(f"    - {feature}: {desc}")

        if 'paper' in capabilities:
            paper = capabilities['paper']
            print(f"\n  Paper: {paper.get('arxiv', 'N/A')}")

        results["checks"].append({
            "name": "capabilities",
            "status": "OK"
        })
        print("\n[OK] Capabilities function working!")

    except Exception as e:
        results["failures"].append({"check": "capabilities", "error": str(e)})
        print(f"[FAIL] Capabilities: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] Capabilities validation passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


def main():
    """Main validation function"""
    print("\n")
    print("=" * 80)
    print("  GENERIC MAKER/MDAP INTEGRATION VALIDATION")
    print("  Complete arXiv:2511.09030 Implementation")
    print("  Works with ANY task type!")
    print("=" * 80)
    print("")

    all_results = {}

    # Run validations
    import_results = validate_imports()
    all_results["imports"] = import_results

    type_results = validate_types()
    all_results["types"] = type_results

    config_results = validate_configuration()
    all_results["configuration"] = config_results

    execution_results = asyncio.run(validate_execution())
    all_results["execution"] = execution_results

    capabilities_results = validate_capabilities()
    all_results["capabilities"] = capabilities_results

    # Summary
    print_section("VALIDATION SUMMARY")

    total_checks = len(all_results)
    total_passed = sum(1 for r in all_results.values() if r["status"] == "pass")
    total_failed = sum(1 for r in all_results.values() if r["status"] == "fail")

    print(f"Categories: {total_checks}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")

    if total_failed == 0:
        print("\n" + "=" * 80)
        print("[OK][OK][OK] ALL VALIDATIONS PASSED [OK][OK][OK]")
        print("=" * 80)
        print("\nGeneric MAKER integration is fully functional!")
        print("\nNext steps:")
        print("1. Run demo: python demo_generic_maker.py")
        print("2. Use in your code: from generic_maker_integration import run_generic_maker")
        print("3. Read guide: GENERIC_MAKER_INTEGRATION_GUIDE.md")
        print("4. Apply to ANY task: code, text, data, workflows, etc.")
        return 0
    else:
        print("\n" + "=" * 80)
        print("[FAIL][FAIL][FAIL] SOME VALIDATIONS FAILED [FAIL][FAIL][FAIL]")
        print("=" * 80)
        print("\nPlease check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
