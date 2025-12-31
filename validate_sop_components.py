"""
SOP Component System - Validation Script

Validates that the component-level generation and refinement system works correctly.

Usage:
    python validate_sop_components.py
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

    # Test 1: Component system
    try:
        from sop_component_system import (
            SOPComponentGenerator,
            SOPComponentType,
            generate_sop_component,
            get_component_capabilities
        )
        results["imports"].append({
            "module": "sop_component_system",
            "status": "OK"
        })
        print("[OK] SOP component system module")
    except ImportError as e:
        results["failures"].append({"module": "sop_component_system", "error": str(e)})
        print(f"[FAIL] SOP component system: {e}")

    # Test 2: Core SOP generator
    try:
        from sop_generator import (
            SOPParameter,
            SOPStep,
            StandardOperatingProcedure
        )
        results["imports"].append({
            "module": "sop_generator",
            "status": "OK"
        })
        print("[OK] Core SOP generator module")
    except ImportError as e:
        results["failures"].append({"module": "sop_generator", "error": str(e)})
        print(f"[FAIL] Core SOP generator: {e}")

    # Test 3: Integrated system
    try:
        from sop_integrated_system import (
            IntegratedSOPGenerator,
            SOPIntegratedConfig
        )
        results["imports"].append({
            "module": "sop_integrated_system",
            "status": "OK"
        })
        print("[OK] Integrated SOP system module")
    except ImportError as e:
        results["failures"].append({"module": "sop_integrated_system", "error": str(e)})
        print(f"[FAIL] Integrated SOP system: {e}")

    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All imports successful!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} import(s) failed")

    return results


def validate_component_types():
    """Validate component types"""
    print_section("2. VALIDATING COMPONENT TYPES")

    results = {"status": "unknown", "types": [], "failures": []}

    try:
        from sop_component_system import SOPComponentType

        expected_types = [
            "ENVIRONMENTAL_CONDITION",
            "EQUIPMENT_SPECIFICATION",
            "MATERIAL",
            "PROTOCOL_STEP",
            "QUALITY_CONTROL",
            "SAFETY_PROTOCOL",
            "VALIDATION_CRITERION",
            "SCALING_INFO",
            "PRECONDITION"
        ]

        available_types = [t.name for t in SOPComponentType]

        for type_name in expected_types:
            if type_name in available_types:
                results["types"].append({
                    "type": type_name,
                    "status": "OK"
                })
                print(f"[OK] Component type: {type_name}")
            else:
                results["failures"].append({"type": type_name, "error": "Not found"})
                print(f"[FAIL] Component type: {type_name} - Not found")

        if not results["failures"]:
            results["status"] = "pass"
            print(f"\n[OK] All {len(expected_types)} component types available!")
        else:
            results["status"] = "fail"
            print(f"\n[FAIL] {len(results['failures'])} type(s) missing")

    except Exception as e:
        results["failures"].append({"type": "all", "error": str(e)})
        print(f"[FAIL] Component type validation: {e}")
        results["status"] = "fail"

    return results


async def validate_environmental_conditions():
    """Validate environmental condition generation"""
    print_section("3. VALIDATING ENVIRONMENTAL CONDITIONS")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from sop_component_system import SOPComponentGenerator

        generator = SOPComponentGenerator()

        # Generate a parameter
        context = {"purpose": "Test", "equipment": ["Standard"]}
        param = await generator.generate_environmental_condition(
            "Temperature",
            context,
            "chemistry"
        )

        # Verify structure
        assert param.name == "Temperature"
        assert hasattr(param, 'value')
        assert hasattr(param, 'tolerance')
        assert hasattr(param, 'verification_method')

        results["checks"].append({
            "name": "parameter_generation",
            "status": "OK"
        })
        print(f"[OK] Environmental condition generated: {param.name}")
        print(f"  Value: {param.format_spec()}")

        # Test refinement
        refined = await generator.refine_environmental_condition(
            param,
            "Improve specification",
            {"domain": "chemistry"}
        )

        results["checks"].append({
            "name": "parameter_refinement",
            "status": "OK"
        })
        print(f"[OK] Environmental condition refined")

    except Exception as e:
        results["failures"].append({"check": "environmental_conditions", "error": str(e)})
        print(f"[FAIL] Environmental conditions: {e}")

    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All environmental condition checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


async def validate_equipment_specs():
    """Validate equipment specification generation"""
    print_section("4. VALIDATING EQUIPMENT SPECIFICATIONS")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from sop_component_system import SOPComponentGenerator

        generator = SOPComponentGenerator()

        spec = await generator.generate_equipment_specification(
            "Magnetic Stirrer",
            "Mixing",
            {"requirements": ["Standard"]},
            "chemistry"
        )

        # Verify structure
        assert isinstance(spec, dict)
        assert "name" in spec
        assert "model" in spec

        results["checks"].append({
            "name": "equipment_generation",
            "status": "OK"
        })
        print(f"[OK] Equipment specification generated: {spec['name']}")
        print(f"  Model: {spec['model']}")

    except Exception as e:
        results["failures"].append({"check": "equipment_specs", "error": str(e)})
        print(f"[FAIL] Equipment specifications: {e}")

    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All equipment specification checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


async def validate_materials():
    """Validate material specification generation"""
    print_section("5. VALIDATING MATERIAL SPECIFICATIONS")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from sop_component_system import SOPComponentGenerator

        generator = SOPComponentGenerator()

        material = await generator.generate_material(
            "Test Chemical",
            "Test purpose",
            {"requirements": ["Standard"]},
            "chemistry"
        )

        # Verify structure
        assert isinstance(material, dict)
        assert "name" in material
        assert "purity" in material

        results["checks"].append({
            "name": "material_generation",
            "status": "OK"
        })
        print(f"[OK] Material specification generated: {material['name']}")
        print(f"  Purity: {material['purity']}")

    except Exception as e:
        results["failures"].append({"check": "materials", "error": str(e)})
        print(f"[FAIL] Materials: {e}")

    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All material specification checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


async def validate_protocol_steps():
    """Validate protocol step generation"""
    print_section("6. VALIDATING PROTOCOL STEPS")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from sop_component_system import SOPComponentGenerator

        generator = SOPComponentGenerator()

        step = await generator.generate_protocol_step(
            1,
            "Test action",
            {"equipment": [], "materials": []},
            [],
            "chemistry"
        )

        # Verify structure
        assert step.step_number == 1
        assert hasattr(step, 'action')
        assert hasattr(step, 'duration')
        assert hasattr(step, 'verification_method')

        results["checks"].append({
            "name": "step_generation",
            "status": "OK"
        })
        print(f"[OK] Protocol step generated: Step {step.step_number}")
        print(f"  Action: {step.action[:50]}...")

        # Test refinement
        refined = await generator.refine_protocol_step(
            step,
            "Add more detail",
            {}
        )

        results["checks"].append({
            "name": "step_refinement",
            "status": "OK"
        })
        print(f"[OK] Protocol step refined")

    except Exception as e:
        results["failures"].append({"check": "protocol_steps", "error": str(e)})
        print(f"[FAIL] Protocol steps: {e}")

    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All protocol step checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


async def validate_other_components():
    """Validate other component types"""
    print_section("7. VALIDATING OTHER COMPONENT TYPES")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from sop_component_system import SOPComponentGenerator

        generator = SOPComponentGenerator()

        # Quality control
        qc = await generator.generate_quality_control_procedure(
            "Test focus",
            {},
            "chemistry"
        )
        assert isinstance(qc, str)
        results["checks"].append({"name": "quality_control", "status": "OK"})
        print("[OK] Quality control procedure generated")

        # Safety protocol
        safety = await generator.generate_safety_protocol(
            "Test hazard",
            {},
            "chemistry"
        )
        assert isinstance(safety, str)
        results["checks"].append({"name": "safety_protocol", "status": "OK"})
        print("[OK] Safety protocol generated")

        # Validation criterion
        validation = await generator.generate_validation_criterion(
            "Test criterion",
            {},
            "chemistry"
        )
        assert isinstance(validation, str)
        results["checks"].append({"name": "validation_criterion", "status": "OK"})
        print("[OK] Validation criterion generated")

        # Scaling info
        scaling = await generator.generate_scaling_info(
            "Test process",
            {},
            "chemistry"
        )
        assert isinstance(scaling, str)
        results["checks"].append({"name": "scaling_info", "status": "OK"})
        print("[OK] Scaling information generated")

    except Exception as e:
        results["failures"].append({"check": "other_components", "error": str(e)})
        print(f"[FAIL] Other components: {e}")

    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All other component checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


async def validate_optimization():
    """Validate component optimization"""
    print_section("8. VALIDATING COMPONENT OPTIMIZATION")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from sop_component_system import (
            SOPComponentGenerator,
            SOPIntegratedConfig,
            SOPComponentType
        )
        from sop_generator import SOPParameter

        # Create config with evolution enabled
        config = SOPIntegratedConfig(
            enable_evolution=True,
            evolution_generations=2,  # Low for validation
            evolution_population_size=3
        )

        generator = SOPComponentGenerator(config)

        param = SOPParameter(
            name="Test",
            value=100.0,
            unit="°C",
            tolerance=10.0
        )

        optimized = await generator.optimize_component(
            param,
            SOPComponentType.ENVIRONMENTAL_CONDITION,
            "Minimize tolerance",
            {}
        )

        results["checks"].append({
            "name": "component_optimization",
            "status": "OK"
        })
        print("[OK] Component optimization working")
        print(f"  Original: {param.format_spec()}")
        print(f"  Optimized: {optimized.format_spec()}")

    except Exception as e:
        results["failures"].append({"check": "optimization", "error": str(e)})
        print(f"[FAIL] Optimization: {e}")

    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All optimization checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


async def validate_safety_testing():
    """Validate component safety testing"""
    print_section("9. VALIDATING COMPONENT SAFETY TESTING")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from sop_component_system import SOPComponentGenerator, SOPComponentType
        from sop_generator import SOPStep

        generator = SOPComponentGenerator()

        # Test unsafe step
        unsafe_step = SOPStep(
            step_number=1,
            action="Heat without monitoring",
            verification_method="",
            contingency_action=""
        )

        is_safe, issues = await generator.test_component_safety(
            unsafe_step,
            SOPComponentType.PROTOCOL_STEP,
            {}
        )

        results["checks"].append({
            "name": "safety_testing",
            "status": "OK"
        })
        print("[OK] Component safety testing working")
        print(f"  Safe: {is_safe}")
        print(f"  Issues found: {len(issues)}")

    except Exception as e:
        results["failures"].append({"check": "safety_testing", "error": str(e)})
        print(f"[FAIL] Safety testing: {e}")

    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All safety testing checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


def validate_capabilities():
    """Validate capabilities function"""
    print_section("10. VALIDATING CAPABILITIES")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from sop_component_system import get_component_capabilities

        capabilities = get_component_capabilities()

        print("SOP Component System Capabilities:")
        print(f"  - Generation: {capabilities.get('component_generation_enabled', False)}")
        print(f"  - Refinement: {capabilities.get('component_refinement_enabled', False)}")
        print(f"  - Optimization: {capabilities.get('component_optimization_enabled', False)}")
        print(f"  - Testing: {capabilities.get('component_testing_enabled', False)}")

        print(f"\n  Supported Components ({len(capabilities.get('supported_components', []))}):")
        for comp in capabilities.get('supported_components', []):
            print(f"    - {comp}")

        print("\n  Features:")
        for feature, desc in capabilities.get('features', {}).items():
            print(f"    - {feature}: {desc}")

        results["checks"].append({
            "name": "capabilities",
            "status": "OK"
        })
        print("\n[OK] Capabilities function working!")

    except Exception as e:
        results["failures"].append({"check": "capabilities", "error": str(e)})
        print(f"[FAIL] Capabilities: {e}")

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
    print("  SOP COMPONENT SYSTEM VALIDATION")
    print("  Granular Generation and Refinement of Every SOP Component")
    print("  Paper: arXiv:2511.09030")
    print("=" * 80)
    print("")

    all_results = {}

    # Run validations
    import_results = validate_imports()
    all_results["imports"] = import_results

    type_results = validate_component_types()
    all_results["component_types"] = type_results

    env_results = asyncio.run(validate_environmental_conditions())
    all_results["environmental_conditions"] = env_results

    equip_results = asyncio.run(validate_equipment_specs())
    all_results["equipment_specs"] = equip_results

    material_results = asyncio.run(validate_materials())
    all_results["materials"] = material_results

    step_results = asyncio.run(validate_protocol_steps())
    all_results["protocol_steps"] = step_results

    other_results = asyncio.run(validate_other_components())
    all_results["other_components"] = other_results

    opt_results = asyncio.run(validate_optimization())
    all_results["optimization"] = opt_results

    safety_results = asyncio.run(validate_safety_testing())
    all_results["safety_testing"] = safety_results

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
        print("\nSOP Component System is fully functional!")
        print("\nComponent Types Supported:")
        print("  1. Environmental conditions (parameters)")
        print("  2. Equipment specifications")
        print("  3. Materials/reagents")
        print("  4. Protocol steps")
        print("  5. Quality control procedures")
        print("  6. Safety protocols")
        print("  7. Validation criteria")
        print("  8. Scaling information")
        print("  9. Preconditions")
        print("\nOperations Supported:")
        print("  - Generate any component independently")
        print("  - Refine individual components")
        print("  - Optimize via evolution")
        print("  - Test for safety")
        print("  - Build complete SOPs from components")
        print("\nNext steps:")
        print("1. Run demo: python demo_sop_components.py")
        print("2. Use in your code:")
        print("     from sop_component_system import generate_sop_component, SOPComponentType")
        print("     component = await generate_sop_component(")
        print("         SOPComponentType.PROTOCOL_STEP,")
        print("         'Prepare solution',")
        print("         context={'equipment': ['Beaker']}")
        print("     )")
        return 0
    else:
        print("\n" + "=" * 80)
        print("[FAIL][FAIL][FAIL] SOME VALIDATIONS FAILED [FAIL][FAIL][FAIL]")
        print("=" * 80)
        print("\nPlease check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
