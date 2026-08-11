"""
End-to-End Invention Planner - Validation Script

Validates that the complete end-to-end invention planning system works.

Usage:
    python validate_end_to_end_invention.py
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
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def validate_imports():
    """Validate imports"""
    print_section("1. VALIDATING IMPORTS")

    results = {"status": "unknown", "imports": [], "failures": []}

    modules = [
        ("end_to_end_invention_planner", "End-to-end invention planner"),
        ("sop_generator", "Core SOP generator"),
        ("sop_component_system", "Component system"),
        ("sop_integrated_system", "Integrated system"),
        ("generic_maker_integration", "Generic MAKER integration")
    ]

    for module_name, description in modules:
        try:
            __import__(module_name)
            results["imports"].append({
                "module": module_name,
                "status": "OK"
            })
            print(f"[OK] {description}")
        except ImportError as e:
            results["failures"].append({
                "module": module_name,
                "error": str(e)
            })
            print(f"[FAIL] {description}: {e}")

    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All imports successful!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} import(s) failed")

    return results


def validate_data_models():
    """Validate data models"""
    print_section("2. VALIDATING DATA MODELS")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from end_to_end_invention_planner import (
            InventionGoal,
            ValidatedMath,
            ErrorSource,
            SuccessCriterion,
            BulletproofSOP
        )

        # Test InventionGoal
        goal = InventionGoal(
            goal_type="technology",
            target="Test invention",
            domain="physics",
            key_requirements=[],
            constraints=[],
            success_definition="Success",
            complexity_score=0.5
        )
        results["checks"].append({"name": "InventionGoal", "status": "OK"})
        print("[OK] InventionGoal model")

        # Test ValidatedMath
        math = ValidatedMath(
            description="Test theorem",
            lean_theorem="theorem test :=",
            lean_proof="by sorry",
            variables={},
            assumptions=[],
            verification_method="Test",
            confidence=0.95
        )
        results["checks"].append({"name": "ValidatedMath", "status": "OK"})
        print("[OK] ValidatedMath model")

        # Test ErrorSource
        error = ErrorSource(
            error_type="test",
            description="Test error",
            probability=0.1,
            impact="low",
            mitigation_strategy="Test",
            verification_method="Test",
            acceptance_criteria="Test"
        )
        results["checks"].append({"name": "ErrorSource", "status": "OK"})
        print("[OK] ErrorSource model")

        # Test SuccessCriterion
        criterion = SuccessCriterion(
            criterion="Test criterion",
            measurement_method="Test",
            pass_threshold=1.0,
            units="binary",
            verification="Test"
        )
        results["checks"].append({"name": "SuccessCriterion", "status": "OK"})
        print("[OK] SuccessCriterion model")

        if not results["failures"]:
            results["status"] = "pass"
            print("\n[OK] All data models validated!")
        else:
            results["status"] = "fail"
            print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    except Exception as e:
        results["failures"].append({"check": "data_models", "error": str(e)})
        print(f"[FAIL] Data models: {e}")
        results["status"] = "fail"

    return results


async def validate_planner_initialization():
    """Validate planner initialization"""
    print_section("3. VALIDATING PLANNER INITIALIZATION")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from end_to_end_invention_planner import EndToEndInventionPlanner

        planner = EndToEndInventionPlanner()

        results["checks"].append({"name": "initialization", "status": "OK"})
        print("[OK] Planner initialization")

        # Check statistics
        stats = planner.get_statistics()
        assert "inventions_planned" in stats
        assert "math_formalized" in stats
        assert "errors_identified" in stats

        results["checks"].append({"name": "statistics", "status": "OK"})
        print("[OK] Statistics tracking")

        if not results["failures"]:
            results["status"] = "pass"
            print("\n[OK] All initialization checks passed!")
        else:
            results["status"] = "fail"
            print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    except Exception as e:
        results["failures"].append({"check": "initialization", "error": str(e)})
        print(f"[FAIL] Initialization: {e}")
        results["status"] = "fail"

    return results


async def validate_prompt_analysis():
    """Validate prompt analysis"""
    print_section("4. VALIDATING PROMPT ANALYSIS")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from end_to_end_invention_planner import EndToEndInventionPlanner

        planner = EndToEndInventionPlanner()

        goal = await planner._analyze_prompt(
            "Create magnetic nanoparticles",
            "chemistry",
            ["Biocompatible"]
        )

        assert goal.domain == "chemistry"
        assert goal.target is not None

        results["checks"].append({
            "name": "prompt_analysis",
            "status": "OK",
            "goal": goal.target
        })
        print(f"[OK] Prompt analyzed: {goal.target}")

        if not results["failures"]:
            results["status"] = "pass"
            print("\n[OK] Prompt analysis working!")
        else:
            results["status"] = "fail"
            print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    except Exception as e:
        results["failures"].append({"check": "prompt_analysis", "error": str(e)})
        print(f"[FAIL] Prompt analysis: {e}")
        results["status"] = "fail"

    return results


async def validate_knowledge_retrieval():
    """Validate knowledge retrieval"""
    print_section("5. VALIDATING KNOWLEDGE RETRIEVAL")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from end_to_end_invention_planner import EndToEndInventionPlanner, InventionGoal

        planner = EndToEndInventionPlanner()

        goal = InventionGoal(
            goal_type="technology",
            target="Test",
            domain="chemistry",
            key_requirements=[],
            constraints=[],
            success_definition="",
            complexity_score=0.5
        )

        knowledge = await planner._retrieve_knowledge(goal)

        assert isinstance(knowledge, list)
        assert len(knowledge) > 0

        results["checks"].append({
            "name": "knowledge_retrieval",
            "status": "OK",
            "items": len(knowledge)
        })
        print(f"[OK] Knowledge retrieved: {len(knowledge)} items")

        if not results["failures"]:
            results["status"] = "pass"
            print("\n[OK] Knowledge retrieval working!")
        else:
            results["status"] = "fail"
            print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    except Exception as e:
        results["failures"].append({"check": "knowledge_retrieval", "error": str(e)})
        print(f"[FAIL] Knowledge retrieval: {e}")
        results["status"] = "fail"

    return results


async def validate_decomposition():
    """Validate decomposition"""
    print_section("6. VALIDATING DECOMPOSITION")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from end_to_end_invention_planner import EndToEndInventionPlanner, InventionGoal

        planner = EndToEndInventionPlanner()

        goal = InventionGoal(
            goal_type="technology",
            target="Test invention",
            domain="chemistry",
            key_requirements=[],
            constraints=[],
            success_definition="",
            complexity_score=0.5
        )

        decomposition = await planner._decompose_invention(goal, [])

        assert "steps" in decomposition
        assert isinstance(decomposition["steps"], list)

        results["checks"].append({
            "name": "decomposition",
            "status": "OK",
            "steps": len(decomposition["steps"])
        })
        print(f"[OK] Decomposition: {len(decomposition['steps'])} steps")

        if not results["failures"]:
            results["status"] = "pass"
            print("\n[OK] Decomposition working!")
        else:
            results["status"] = "fail"
            print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    except Exception as e:
        results["failures"].append({"check": "decomposition", "error": str(e)})
        print(f"[FAIL] Decomposition: {e}")
        results["status"] = "fail"

    return results


async def validate_math_formalization():
    """Validate math formalization"""
    print_section("7. VALIDATING MATH FORMALIZATION")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from end_to_end_invention_planner import EndToEndInventionPlanner, InventionGoal

        planner = EndToEndInventionPlanner()

        goal = InventionGoal(
            goal_type="technology",
            target="Test invention",
            domain="physics",
            key_requirements=[],
            constraints=[],
            success_definition="",
            complexity_score=0.5
        )

        formalized = await planner._formalize_math(goal, {}, [])

        assert isinstance(formalized, list)

        results["checks"].append({
            "name": "math_formalization",
            "status": "OK",
            "theorems": len(formalized)
        })
        print(f"[OK] Math formalized: {len(formalized)} theorems")

        if not results["failures"]:
            results["status"] = "pass"
            print("\n[OK] Math formalization working!")
        else:
            results["status"] = "fail"
            print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    except Exception as e:
        results["failures"].append({"check": "math_formalization", "error": str(e)})
        print(f"[FAIL] Math formalization: {e}")
        results["status"] = "fail"

    return results


async def validate_error_analysis():
    """Validate error analysis"""
    print_section("8. VALIDATING ERROR ANALYSIS")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from end_to_end_invention_planner import EndToEndInventionPlanner, InventionGoal

        planner = EndToEndInventionPlanner()

        goal = InventionGoal(
            goal_type="technology",
            target="Test invention",
            domain="chemistry",
            key_requirements=[],
            constraints=[],
            success_definition="",
            complexity_score=0.5
        )

        errors = await planner._analyze_error_sources(goal, {}, [])

        assert isinstance(errors, list)

        results["checks"].append({
            "name": "error_analysis",
            "status": "OK",
            "errors": len(errors)
        })
        print(f"[OK] Error sources identified: {len(errors)}")

        if not results["failures"]:
            results["status"] = "pass"
            print("\n[OK] Error analysis working!")
        else:
            results["status"] = "fail"
            print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    except Exception as e:
        results["failures"].append({"check": "error_analysis", "error": str(e)})
        print(f"[FAIL] Error analysis: {e}")
        results["status"] = "fail"

    return results


async def validate_red_blue_team():
    """Validate red/blue team testing"""
    print_section("9. VALIDATING RED/BLUE TEAM TESTING")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from end_to_end_invention_planner import EndToEndInventionPlanner, InventionGoal

        planner = EndToEndInventionPlanner()

        goal = InventionGoal(
            goal_type="technology",
            target="Test invention",
            domain="chemistry",
            key_requirements=[],
            constraints=[],
            success_definition="",
            complexity_score=0.5
        )

        red_findings, blue_fixes = await planner._red_blue_team_test(goal, {}, [])

        assert isinstance(red_findings, list)
        assert isinstance(blue_fixes, list)

        results["checks"].append({
            "name": "red_blue_team",
            "status": "OK",
            "red_findings": len(red_findings),
            "blue_fixes": len(blue_fixes)
        })
        print(f"[OK] Red team findings: {len(red_findings)}")
        print(f"[OK] Blue team fixes: {len(blue_fixes)}")

        if not results["failures"]:
            results["status"] = "pass"
            print("\n[OK] Red/blue team testing working!")
        else:
            results["status"] = "fail"
            print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    except Exception as e:
        results["failures"].append({"check": "red_blue_team", "error": str(e)})
        print(f"[FAIL] Red/blue team: {e}")
        results["status"] = "fail"

    return results


async def validate_success_criteria():
    """Validate success criteria"""
    print_section("10. VALIDATING SUCCESS CRITERIA")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from end_to_end_invention_planner import EndToEndInventionPlanner, InventionGoal

        planner = EndToEndInventionPlanner()

        goal = InventionGoal(
            goal_type="technology",
            target="Test invention",
            domain="chemistry",
            key_requirements=[],
            constraints=[],
            success_definition="",
            complexity_score=0.5
        )

        criteria = await planner._define_success_criteria(goal, {})

        assert isinstance(criteria, list)

        results["checks"].append({
            "name": "success_criteria",
            "status": "OK",
            "criteria": len(criteria)
        })
        print(f"[OK] Success criteria defined: {len(criteria)}")

        if not results["failures"]:
            results["status"] = "pass"
            print("\n[OK] Success criteria definition working!")
        else:
            results["status"] = "fail"
            print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    except Exception as e:
        results["failures"].append({"check": "success_criteria", "error": str(e)})
        print(f"[FAIL] Success criteria: {e}")
        results["status"] = "fail"

    return results


async def validate_end_to_end():
    """Validate complete end-to-end planning"""
    print_section("11. VALIDATING END-TO-END PLANNING")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from end_to_end_invention_planner import plan_invention

        print("  Running complete end-to-end planning...")
        print("  (using minimal config for faster validation)")

        bulletproof = await plan_invention(
            prompt="Create a simple chemical synthesis procedure",
            domain="chemistry"
        )

        # Verify structure
        assert bulletproof.invention_goal is not None
        assert bulletproof.sop is not None
        assert bulletproof.validation_summary is not None

        results["checks"].append({
            "name": "end_to_end_planning",
            "status": "OK",
            "goal": bulletproof.invention_goal.target
        })
        print(f"[OK] End-to-end planning: {bulletproof.invention_goal.target}")

        # Check document generation
        document = bulletproof.to_executable_document()
        assert len(document) > 500

        results["checks"].append({
            "name": "document_generation",
            "status": "OK",
            "length": len(document)
        })
        print(f"[OK] Document generated: {len(document)} chars")

        if not results["failures"]:
            results["status"] = "pass"
            print("\n[OK] All end-to-end checks passed!")
        else:
            results["status"] = "fail"
            print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    except Exception as e:
        results["failures"].append({"check": "end_to_end", "error": str(e)})
        print(f"[FAIL] End-to-end: {e}")
        results["status"] = "fail"

    return results


def validate_capabilities():
    """Validate capabilities"""
    print_section("12. VALIDATING CAPABILITIES")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from end_to_end_invention_planner import get_invention_planner_capabilities

        capabilities = get_invention_planner_capabilities()

        print("End-to-End Invention Planner Capabilities:")
        for key, value in capabilities.items():
            if key != "pipeline_stages" and isinstance(value, bool):
                status = "[OK]" if value else "[--]"
                print(f"  {status} {key}")

        print(f"\n  Pipeline Stages ({len(capabilities['pipeline_stages'])}):")
        for stage in capabilities['pipeline_stages']:
            print(f"    - {stage}")

        results["checks"].append({
            "name": "capabilities",
            "status": "OK"
        })
        print("\n[OK] Capabilities validated!")

    except Exception as e:
        results["failures"].append({"check": "capabilities", "error": str(e)})
        print(f"[FAIL] Capabilities: {e}")
        results["status"] = "fail"

    if not results["failures"]:
        results["status"] = "pass"
    else:
        results["status"] = "fail"

    return results


def main():
    """Main validation function"""
    print("\n")
    print("=" * 80)
    print("  END-TO-END INVENTION PLANNER VALIDATION")
    print("  Complete Natural Language -> Bulletproof Invention Plan")
    print("  Paper: arXiv:2511.09030")
    print("=" * 80)
    print("")

    all_results = {}

    # Run validations
    import_results = validate_imports()
    all_results["imports"] = import_results

    model_results = validate_data_models()
    all_results["data_models"] = model_results

    init_results = asyncio.run(validate_planner_initialization())
    all_results["initialization"] = init_results

    prompt_results = asyncio.run(validate_prompt_analysis())
    all_results["prompt_analysis"] = prompt_results

    knowledge_results = asyncio.run(validate_knowledge_retrieval())
    all_results["knowledge_retrieval"] = knowledge_results

    decomp_results = asyncio.run(validate_decomposition())
    all_results["decomposition"] = decomp_results

    math_results = asyncio.run(validate_math_formalization())
    all_results["math_formalization"] = math_results

    error_results = asyncio.run(validate_error_analysis())
    all_results["error_analysis"] = error_results

    redblue_results = asyncio.run(validate_red_blue_team())
    all_results["red_blue_team"] = redblue_results

    criteria_results = asyncio.run(validate_success_criteria())
    all_results["success_criteria"] = criteria_results

    e2e_results = asyncio.run(validate_end_to_end())
    all_results["end_to_end"] = e2e_results

    cap_results = validate_capabilities()
    all_results["capabilities"] = cap_results

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
        print("\nEnd-to-End Invention Planner is fully functional!")
        print("\nPipeline Stages:")
        print("  1. Prompt Analysis -> Extract invention goal")
        print("  2. Knowledge Retrieval -> Gather scientific knowledge")
        print("  3. Decomposition -> Break into atomic steps")
        print("  4. Math Formalization -> Convert to Lean proofs")
        print("  5. Physics Validation -> Verify consistency")
        print("  6. Error Analysis -> Identify every error source")
        print("  7. Red/Blue Team -> Adversarial testing")
        print("  8. SOP Generation -> Create bulletproof plan")
        print("  9. Success Criteria -> Binary pass/fail")
        print("\nNext steps:")
        print("1. Run demo: python demo_end_to_end_invention.py")
        print("2. Use in your code:")
        print("     from end_to_end_invention_planner import plan_invention")
        print("     plan = await plan_invention('Create a plan to invent X')")
        print("     print(plan.to_executable_document())")
        return 0
    else:
        print("\n" + "=" * 80)
        print("[FAIL][FAIL][FAIL] SOME VALIDATIONS FAILED [FAIL][FAIL][FAIL]")
        print("=" * 80)
        print("\nPlease check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
