"""
SOP Generator - Validation Script

Validates that the SOP generator system works correctly.

Usage:
    python validate_sop_generator.py
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
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def validate_imports():
    """Validate imports"""
    print_section("1. VALIDATING IMPORTS")

    results = {"status": "unknown", "imports": [], "failures": []}

    # Test 1: SOP generator module
    try:
        from sop_generator import (
            SOPGenerator,
            SOPParameter,
            SOPStep,
            StandardOperatingProcedure,
            SOPEvaluator,
            generate_sop,
            refine_sop,
            get_sop_capabilities
        )
        results["imports"].append({
            "module": "sop_generator",
            "status": "OK"
        })
        print("[OK] SOP generator module")
    except ImportError as e:
        results["failures"].append({"module": "sop_generator", "error": str(e)})
        print(f"[FAIL] SOP generator: {e}")

    # Test 2: Generic MAKER integration
    try:
        from generic_maker_integration import (
            run_generic_maker,
            GenericEvaluator,
            GenericTask,
            GenericSolution,
            TaskType,
            MAKERConfig
        )
        results["imports"].append({
            "module": "generic_maker_integration",
            "status": "OK"
        })
        print("[OK] Generic MAKER integration module")
    except ImportError as e:
        results["failures"].append({"module": "generic_maker_integration", "error": str(e)})
        print(f"[FAIL] Generic MAKER integration: {e}")

    # Determine status
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

    # Test 1: SOPParameter
    try:
        from sop_generator import SOPParameter

        param = SOPParameter(
            name="Temperature",
            value=25.0,
            unit="°C",
            tolerance=2.0,
            verification_method="Thermometer",
            critical=True,
            rationale="Temperature affects kinetics"
        )

        spec = param.format_spec()
        assert "25.0" in spec
        assert "°C" in spec
        assert "±" in spec

        results["checks"].append({
            "name": "sop_parameter",
            "status": "OK"
        })
        print("[OK] SOPParameter creation and formatting")

    except Exception as e:
        results["failures"].append({"check": "sop_parameter", "error": str(e)})
        print(f"[FAIL] SOPParameter: {e}")

    # Test 2: SOPStep
    try:
        from sop_generator import SOPStep

        step = SOPStep(
            step_number=1,
            action="Mix the solution",
            duration=300.0,
            duration_tolerance=30.0,
            verification_method="Visual inspection",
            acceptance_criteria="Clear solution",
            contingency_action="Add solvent if cloudy",
            substeps=["Add reagent A", "Add reagent B", "Stir"]
        )

        formatted = step.format_step()
        assert "Step 1" in formatted
        assert "Mix the solution" in formatted
        assert "minutes" in formatted or "seconds" in formatted

        results["checks"].append({
            "name": "sop_step",
            "status": "OK"
        })
        print("[OK] SOPStep creation and formatting")

    except Exception as e:
        results["failures"].append({"check": "sop_step", "error": str(e)})
        print(f"[FAIL] SOPStep: {e}")

    # Test 3: StandardOperatingProcedure
    try:
        from sop_generator import StandardOperatingProcedure
        from datetime import datetime

        sop = StandardOperatingProcedure(
            title="Test SOP",
            version="1.0",
            status="DRAFT",
            effective_date=datetime.now().strftime("%Y-%m-%d"),
            description="Test description"
        )

        # Add some content
        sop.environmental_conditions = {
            "Temperature": param
        }
        sop.protocols = [step]
        sop.quality_control = ["Check 1", "Check 2"]
        sop.safety_protocols = ["Safety 1", "Safety 2"]

        # Test to_markdown
        markdown = sop.to_markdown()
        assert "Test SOP" in markdown
        assert "Environmental Conditions" in markdown
        assert "Protocols" in markdown

        # Test to_dict
        sop_dict = sop.to_dict()
        assert sop_dict["title"] == "Test SOP"
        assert sop_dict["version"] == "1.0"

        results["checks"].append({
            "name": "standard_operating_procedure",
            "status": "OK"
        })
        print("[OK] StandardOperatingProcedure creation, export, serialization")

    except Exception as e:
        results["failures"].append({"check": "standard_operating_procedure", "error": str(e)})
        print(f"[FAIL] StandardOperatingProcedure: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All data models validated!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


def validate_evaluator():
    """Validate SOP evaluator"""
    print_section("3. VALIDATING SOP EVALUATOR")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from sop_generator import SOPEvaluator, GenericTask

        evaluator = SOPEvaluator(
            domain="chemistry",
            constraints=["Temperature < 100°C"],
            equipment=["Magnetic stirrer"]
        )

        # Test evaluation with good SOP
        good_sop = """
        # Chemical Mixing SOP

        ## Environmental Conditions

        ### Temperature
        · Target: 25.0 °C ± 2.0 °C
        · Verification: Calibrated thermometer

        ## Equipment

        ### Magnetic Stirrer
        · Model: SuperSpinner 5000

        ## Protocols

        ### General Protocol

        **Step 1:** Mix the chemicals

        · Duration: 5.0 minutes
        · Verification: Visual inspection
        · Acceptance: Clear solution

        ## Quality Control

        · Check temperature before starting
        · Verify clarity

        ## Safety

        · Wear safety glasses
        · Use fume hood
        · Emergency eyewash available
        """

        task = GenericTask(
            task_id="test",
            description="Generate a chemical mixing SOP",
            task_type="document_processing"
        )

        score = evaluator.evaluate(good_sop, task)

        results["checks"].append({
            "name": "evaluator_good_sop",
            "status": "OK",
            "score": score
        })
        print(f"[OK] Evaluator scores good SOP: {score:.3f}")

        # Test evaluation with bad SOP
        bad_sop = """
        # Chemical Mixing SOP

        Mix the chemicals as appropriate.
        Heat as needed.
        """

        bad_score = evaluator.evaluate(bad_sop, task)

        results["checks"].append({
            "name": "evaluator_bad_sop",
            "status": "OK",
            "score": bad_score
        })
        print(f"[OK] Evaluator scores bad SOP: {bad_score:.3f}")

        # Verify good SOP scores higher
        if score > bad_score:
            results["checks"].append({
                "name": "evaluator_discrimination",
                "status": "OK"
            })
            print("[OK] Evaluator correctly discriminates good vs bad SOPs")
        else:
            raise ValueError("Evaluator did not discriminate: good={score}, bad={bad_score}")

        # Test evaluation details
        details = evaluator.get_evaluation_details()
        assert "criteria" in details
        assert len(details["criteria"]) == 5

        results["checks"].append({
            "name": "evaluator_details",
            "status": "OK"
        })
        print("[OK] Evaluator provides evaluation details")

    except Exception as e:
        results["failures"].append({"check": "evaluator", "error": str(e)})
        print(f"[FAIL] Evaluator: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All evaluator checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


def validate_generator():
    """Validate SOP generator"""
    print_section("4. VALIDATING SOP GENERATOR")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from sop_generator import SOPGenerator, MAKERConfig

        # Test initialization
        config = MAKERConfig(
            enable_voting=True,
            voting_threshold=3,
            enable_decomposition=True,
            max_generations=10,  # Low for faster testing
            population_size=10
        )

        generator = SOPGenerator(config=config)

        results["checks"].append({
            "name": "generator_initialization",
            "status": "OK"
        })
        print("[OK] SOPGenerator initialization")

        # Test statistics
        assert "sops_generated" in generator.statistics
        assert "sops_refined" in generator.statistics
        assert "average_quality" in generator.statistics

        results["checks"].append({
            "name": "generator_statistics",
            "status": "OK"
        })
        print("[OK] Generator statistics tracking")

        # Test task creation
        task = generator._create_generation_task(
            requirement="Create a mixing protocol",
            domain="chemistry",
            constraints=["Temperature < 50°C"],
            equipment=["Beaker", "Stirrer"]
        )

        assert "mixing protocol" in task.description.lower()
        assert "chemistry" in task.description.lower()

        results["checks"].append({
            "name": "generator_task_creation",
            "status": "OK"
        })
        print("[OK] Generator task creation")

    except Exception as e:
        results["failures"].append({"check": "generator", "error": str(e)})
        print(f"[FAIL] Generator: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All generator checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


async def validate_end_to_end():
    """Validate end-to-end execution"""
    print_section("5. VALIDATING END-TO-END EXECUTION")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from sop_generator import SOPGenerator, SOPEvaluator
        from generic_maker_integration import MAKERConfig

        # Create generator with minimal config for fast testing
        config = MAKERConfig(
            enable_voting=True,
            voting_threshold=2,  # Lower for speed
            enable_decomposition=True,
            max_generations=5,   # Low for speed
            population_size=5
        )

        generator = SOPGenerator(config=config)

        print("  Testing SOP generation...")
        print("  (using minimal config for faster validation)")

        # Generate a simple SOP
        sop = await generator.generate_sop(
            requirement_description="Create a protocol for measuring liquid volume",
            domain="chemistry",
            constraints=["Use standard laboratory equipment"],
            equipment_available=["Graduated cylinder", "Beaker"]
        )

        # Verify SOP structure
        assert sop.title is not None
        assert sop.version is not None
        assert sop.status is not None
        assert len(sop.revision_history) > 0

        results["checks"].append({
            "name": "sop_generation",
            "status": "OK",
            "title": sop.title
        })
        print(f"[OK] SOP generated: {sop.title}")

        # Verify it can export to markdown
        markdown = sop.to_markdown()
        assert len(markdown) > 100
        assert sop.title in markdown

        results["checks"].append({
            "name": "sop_export",
            "status": "OK",
            "markdown_length": len(markdown)
        })
        print(f"[OK] SOP exports to Markdown: {len(markdown)} chars")

        # Verify it can serialize
        sop_dict = sop.to_dict()
        assert sop_dict["title"] == sop.title

        results["checks"].append({
            "name": "sop_serialization",
            "status": "OK"
        })
        print("[OK] SOP serializes to dictionary")

    except Exception as e:
        results["failures"].append({"check": "end_to_end", "error": str(e)})
        print(f"[FAIL] End-to-end: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All end-to-end checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


def validate_capabilities():
    """Validate capabilities function"""
    print_section("6. VALIDATING CAPABILITIES")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from sop_generator import get_sop_capabilities

        capabilities = get_sop_capabilities()

        # Display capabilities
        print("SOP Generator Capabilities:")
        print(f"  - SOP generation: {capabilities.get('sop_generation_enabled', False)}")
        print(f"  - SOP refinement: {capabilities.get('sop_refinement_enabled', False)}")

        print(f"\n  Supported Domains ({len(capabilities.get('supported_domains', []))}):")
        for domain in capabilities.get('supported_domains', []):
            print(f"    - {domain}")

        print(f"\n  Features:")
        for feature, desc in capabilities.get('features', {}).items():
            print(f"    - {feature}: {desc}")

        if 'paper' in capabilities:
            paper = capabilities['paper']
            print(f"\n  Paper: {paper.get('arxiv', 'N/A')}")

        # Verify structure
        assert capabilities.get('sop_generation_enabled') == True
        assert capabilities.get('sop_refinement_enabled') == True
        assert len(capabilities.get('supported_domains', [])) > 0
        assert len(capabilities.get('features', {})) > 0
        assert 'paper' in capabilities

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
    print("  SOP GENERATOR VALIDATION")
    print("  MAKER-Based Standard Operating Procedure System")
    print("  Paper: arXiv:2511.09030")
    print("=" * 80)
    print("")

    all_results = {}

    # Run validations
    import_results = validate_imports()
    all_results["imports"] = import_results

    model_results = validate_data_models()
    all_results["data_models"] = model_results

    evaluator_results = validate_evaluator()
    all_results["evaluator"] = evaluator_results

    generator_results = validate_generator()
    all_results["generator"] = generator_results

    e2e_results = asyncio.run(validate_end_to_end())
    all_results["end_to_end"] = e2e_results

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
        print("\nSOP Generator is fully functional!")
        print("\nNext steps:")
        print("1. Run demo: python demo_sop_generator.py")
        print("2. Use in your code: from sop_generator import generate_sop")
        print("3. Read guide: SOP_GENERATOR_GUIDE.md")
        print("\nKey features:")
        print("  - Generate complete SOPs from requirements")
        print("  - Refine existing SOPs based on feedback")
        print("  - Zero-error guarantees through MAKER voting")
        print("  - Turnkey-ready protocols with all parameters specified")
        print("  - Works with any domain (chemistry, manufacturing, software, etc.)")
        return 0
    else:
        print("\n" + "=" * 80)
        print("[FAIL][FAIL][FAIL] SOME VALIDATIONS FAILED [FAIL][FAIL][FAIL]")
        print("=" * 80)
        print("\nPlease check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
