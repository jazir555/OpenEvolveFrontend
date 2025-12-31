"""
SOP Component System - Demo Script

Demonstrates granular generation and refinement of individual SOP components:
- Environmental conditions (parameters)
- Equipment specifications
- Materials
- Protocol steps
- Quality control procedures
- Safety protocols
- Validation criteria
- Scaling information

Usage:
    python demo_sop_components.py
"""

import asyncio
import logging
from datetime import datetime

from sop_component_system import (
    SOPComponentGenerator,
    SOPComponentType,
    generate_sop_component,
    get_component_capabilities
)

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


async def demo_1_capabilities():
    """Demo 1: Component system capabilities"""
    print_section("DEMO 1: COMPONENT SYSTEM CAPABILITIES")

    capabilities = get_component_capabilities()

    print("SOP Component System Status:")
    print(f"  - Component Generation: {capabilities['component_generation_enabled']}")
    print(f"  - Component Refinement: {capabilities['component_refinement_enabled']}")
    print(f"  - Component Optimization: {capabilities['component_optimization_enabled']}")
    print(f"  - Component Testing: {capabilities['component_testing_enabled']}")

    print(f"\n  Supported Components ({len(capabilities['supported_components'])}):")
    for component in capabilities['supported_components']:
        print(f"    - {component}")

    print("\n  Features:")
    for feature, description in capabilities['features'].items():
        print(f"    - {feature}: {description}")


async def demo_2_environmental_conditions():
    """Demo 2: Generate and refine environmental conditions"""
    print_section("DEMO 2: ENVIRONMENTAL CONDITIONS")

    generator = SOPComponentGenerator()

    context = {
        "purpose": "Nanoparticle synthesis",
        "equipment": ["Temperature controller", "Hotplate", "Thermometer"],
        "domain": "chemistry"
    }

    # Generate temperature parameter
    print("Generating temperature parameter...")
    temp_param = await generator.generate_environmental_condition(
        parameter_name="Temperature",
        context=context,
        domain="chemistry"
    )

    print(f"\n[OK] Generated: {temp_param.name}")
    print(f"  Value: {temp_param.format_spec()}")
    print(f"  Verification: {temp_param.verification_method}")
    print(f"  Critical: {temp_param.critical}")
    print(f"  Rationale: {temp_param.rationale}")

    # Refine the parameter
    print("\nRefining parameter (tighten tolerance)...")
    refined_temp = await generator.refine_environmental_condition(
        param=temp_param,
        refinement_goal="Tighten tolerance from ±2°C to ±1°C for better control",
        context={"domain": "chemistry", "equipment": ["Precision temperature controller"]}
    )

    print(f"\n[OK] Refined: {refined_temp.name}")
    print(f"  Original: {temp_param.format_spec()}")
    print(f"  Refined: {refined_temp.format_spec()}")

    # Generate humidity parameter
    print("\nGenerating humidity parameter...")
    humidity_param = await generator.generate_environmental_condition(
        parameter_name="Humidity",
        context=context,
        domain="chemistry"
    )

    print(f"\n[OK] Generated: {humidity_param.name}")
    print(f"  Value: {humidity_param.format_spec()}")

    return [temp_param, humidity_param]


async def demo_3_equipment_specifications():
    """Demo 3: Generate equipment specifications"""
    print_section("DEMO 3: EQUIPMENT SPECIFICATIONS")

    generator = SOPComponentGenerator()

    # Generate magnetic stirrer specification
    context = {
        "purpose": "Mixing chemical solutions",
        "requirements": [
            "Precise temperature control",
            "Variable speed control",
            "Corrosion-resistant"
        ]
    }

    print("Generating magnetic stirrer specification...")
    stirrer_spec = await generator.generate_equipment_specification(
        equipment_name="Magnetic Stirrer",
        purpose="Mixing and heating chemical solutions",
        context=context,
        domain="chemistry"
    )

    print(f"\n[OK] Generated: {stirrer_spec['name']}")
    print(f"  Model: {stirrer_spec['model']}")
    print(f"  Specifications: {stirrer_spec['specifications']}")
    print(f"  Features: {stirrer_spec['features']}")
    print(f"  Calibration: {stirrer_spec['calibration']}")
    print(f"  Maintenance: {stirrer_spec['maintenance']}")

    return stirrer_spec


async def demo_4_materials():
    """Demo 4: Generate material specifications"""
    print_section("DEMO 4: MATERIAL SPECIFICATIONS")

    generator = SOPComponentGenerator()

    context = {
        "purpose": "Precursor for nanoparticle synthesis",
        "requirements": [
            "High purity",
            "Anhydrous",
            "Storable"
        ]
    }

    print("Generating iron chloride material specification...")
    material = await generator.generate_material(
        material_name="Iron(III) chloride hexahydrate",
        purpose="Precursor for iron oxide nanoparticle synthesis",
        context=context,
        domain="chemistry"
    )

    print(f"\n[OK] Generated: {material['name']}")
    print(f"  Purity: {material['purity']}")
    print(f"  Grade: {material['grade']}")
    print(f"  Amount: {material['amount']} ± {material['tolerance']} {material['unit']}")
    print(f"  Storage: {material['storage']}")
    print(f"  Safety: {material['safety']}")

    return material


async def demo_5_protocol_steps():
    """Demo 5: Generate and refine protocol steps"""
    print_section("DEMO 5: PROTOCOL STEPS")

    generator = SOPComponentGenerator()

    context = {
        "equipment": ["Magnetic stirrer", "Hotplate", "Thermometer"],
        "materials": ["Iron chloride", "Water"],
        "environmental_conditions": {"Temperature": "75°C"}
    }

    # Generate a step
    print("Generating protocol step...")
    step = await generator.generate_protocol_step(
        step_number=1,
        action_description="Prepare iron chloride solution",
        context=context,
        domain="chemistry"
    )

    print(f"\n[OK] Generated: Step {step.step_number}")
    print(f"  Action: {step.action}")
    print(f"  Duration: {step.duration/60:.1f} ± {step.duration_tolerance/60:.1f} minutes")
    print(f"  Verification: {step.verification_method}")
    print(f"  Acceptance: {step.acceptance_criteria}")
    print(f"  Contingency: {step.contingency_action}")

    if step.substeps:
        print("  Substeps:")
        for i, substep in enumerate(step.substeps, 1):
            print(f"    {i}. {substep}")

    # Refine the step
    print("\nRefining step (add more detail)...")
    refined_step = await generator.refine_protocol_step(
        step=step,
        refinement_goal="Add more specific verification method and acceptance criteria",
        context={"domain": "chemistry"}
    )

    print(f"\n[OK] Refined: Step {refined_step.step_number}")
    print(f"  Original verification: {step.verification_method}")
    print(f"  Refined verification: {refined_step.verification_method}")

    return [step, refined_step]


async def demo_6_quality_control():
    """Demo 6: Generate quality control procedures"""
    print_section("DEMO 6: QUALITY CONTROL PROCEDURES")

    generator = SOPComponentGenerator()

    context = {
        "process": "Nanoparticle synthesis",
        "critical_parameters": ["Particle size", "Magnetic properties"]
    }

    print("Generating quality control procedure...")
    qc = await generator.generate_quality_control_procedure(
        qc_focus="Particle size verification",
        context=context,
        domain="chemistry"
    )

    print(f"\n[OK] Generated QC Procedure:")
    print(qc)

    return qc


async def demo_7_safety_protocols():
    """Demo 7: Generate safety protocols"""
    print_section("DEMO 7: SAFETY PROTOCOLS")

    generator = SOPComponentGenerator()

    context = {
        "chemicals": ["Iron chloride", "Sodium hydroxide"],
        "equipment": ["Hotplate", "Fume hood"]
    }

    print("Generating safety protocol...")
    safety = await generator.generate_safety_protocol(
        hazard_type="Handling corrosive chemicals and heating operations",
        context=context,
        domain="chemistry"
    )

    print(f"\n[OK] Generated Safety Protocol:")
    print(safety)

    return safety


async def demo_8_validation_criteria():
    """Demo 8: Generate validation criteria"""
    print_section("DEMO 8: VALIDATION CRITERIA")

    generator = SOPComponentGenerator()

    context = {
        "target": "Particle size 10-15 nm",
        "method": "Dynamic light scattering"
    }

    print("Generating validation criterion...")
    validation = await generator.generate_validation_criterion(
        criterion_focus="Particle size distribution",
        context=context,
        domain="chemistry"
    )

    print(f"\n[OK] Generated Validation Criterion:")
    print(validation)

    return validation


async def demo_9_scaling_info():
    """Demo 9: Generate scaling information"""
    print_section("DEMO 9: SCALING INFORMATION")

    generator = SOPComponentGenerator()

    context = {
        "base_scale": "100 mL reaction",
        "limitations": ["Heat transfer", "Mixing efficiency"]
    }

    print("Generating scaling information...")
    scaling = await generator.generate_scaling_info(
        base_process="Nanoparticle synthesis",
        context=context,
        domain="chemistry"
    )

    print(f"\n[OK] Generated Scaling Information:")
    print(scaling)

    return scaling


async def demo_10_component_optimization():
    """Demo 10: Optimize components using evolution"""
    print_section("DEMO 10: COMPONENT OPTIMIZATION")

    from sop_component_system import SOPComponentGenerator, SOPIntegratedConfig, SOPComponentType
    from sop_generator import SOPParameter

    # Create config with evolution enabled
    config = SOPIntegratedConfig(
        enable_evolution=True,
        evolution_generations=5,  # Low for demo
        evolution_population_size=8
    )

    generator = SOPComponentGenerator(config)

    # Create a parameter to optimize
    param = SOPParameter(
        name="Temperature",
        value=75.0,
        unit="°C",
        tolerance=5.0,  # Wide tolerance to start
        verification_method="Thermometer reading",
        critical=True,
        rationale="Controls particle size"
    )

    print(f"Initial parameter: {param.format_spec()}")
    print(f"  Tolerance: ±{param.tolerance}°C")

    # Optimize for tighter tolerance
    print("\nOptimizing for tighter tolerance...")
    optimized = await generator.optimize_component(
        component=param,
        component_type=SOPComponentType.ENVIRONMENTAL_CONDITION,
        optimization_goal="Minimize tolerance while maintaining achievability",
        context={"domain": "chemistry", "equipment": ["Standard temperature controller"]}
    )

    print(f"\n[OK] Optimized parameter: {optimized.format_spec()}")
    print(f"  Original tolerance: ±{param.tolerance}°C")
    print(f"  Optimized tolerance: ±{optimized.tolerance}°C")
    print(f"  Improvement: {(1 - optimized.tolerance/param.tolerance)*100:.1f}% tighter")


async def demo_11_component_testing():
    """Demo 11: Test components for safety issues"""
    print_section("DEMO 11: COMPONENT SAFETY TESTING")

    from sop_component_system import SOPComponentGenerator, SOPComponentType
    from sop_generator import SOPStep

    generator = SOPComponentGenerator()

    # Test a protocol step
    step = SOPStep(
        step_number=1,
        action="Heat the solution to 100°C without monitoring",
        duration=600.0,
        verification_method="",  # Missing verification
        acceptance_criteria="",
        contingency_action=""  # Missing contingency
    )

    print(f"Testing step for safety issues:")
    print(f"  Action: {step.action}")
    print(f"  Verification: {step.verification_method if step.verification_method else '(missing)'}")
    print(f"  Contingency: {step.contingency_action if step.contingency_action else '(missing)'}")

    is_safe, issues = await generator.test_component_safety(
        component=step,
        component_type=SOPComponentType.PROTOCOL_STEP,
        context={"domain": "chemistry"}
    )

    print(f"\nSafety Test Result: {'SAFE' if is_safe else 'ISSUES FOUND'}")
    if issues:
        print("\nIssues:")
        for issue in issues:
            print(f"  [!] {issue}")


async def demo_12_complete_sop_from_components():
    """Demo 12: Build complete SOP from individual components"""
    print_section("DEMO 12: BUILD COMPLETE SOP FROM COMPONENTS")

    from sop_generator import StandardOperatingProcedure

    generator = SOPComponentGenerator()
    sop = StandardOperatingProcedure(
        title="Magneto-Chemical Assembly of Iron Oxide Nanoparticles",
        version="1.0",
        status="DRAFT",
        effective_date=datetime.now().strftime("%Y-%m-%d"),
        description="Complete protocol synthesized from individual components",
        classification="TURNKEY"
    )

    print("Building complete SOP from individual components...")

    # Environmental conditions
    print("\n[1/8] Generating environmental conditions...")
    temp = await generator.generate_environmental_condition(
        "Temperature",
        {"purpose": "Nanoparticle synthesis"},
        "chemistry"
    )
    sop.environmental_conditions["Temperature"] = temp

    # Equipment
    print("[2/8] Generating equipment specifications...")
    stirrer = await generator.generate_equipment_specification(
        "Magnetic Stirrer",
        "Mixing and heating",
        {"requirements": ["Temperature control"]},
        "chemistry"
    )
    sop.equipment.append(stirrer)

    # Materials
    print("[3/8] Generating materials...")
    iron_cl = await generator.generate_material(
        "Iron(III) chloride",
        "Precursor",
        {"requirements": ["High purity"]},
        "chemistry"
    )
    sop.materials.append(iron_cl)

    # Protocol steps
    print("[4/8] Generating protocol steps...")
    step1 = await generator.generate_protocol_step(
        1,
        "Prepare precursor solution",
        {"equipment": ["Beaker", "Stirrer"]},
        [],
        "chemistry"
    )
    sop.protocols.append(step1)

    step2 = await generator.generate_protocol_step(
        2,
        "Heat to reaction temperature",
        {"equipment": ["Hotplate", "Thermometer"]},
        [step1],
        "chemistry"
    )
    sop.protocols.append(step2)

    # Quality control
    print("[5/8] Generating quality control...")
    qc = await generator.generate_quality_control_procedure(
        "Particle size verification",
        {},
        "chemistry"
    )
    sop.quality_control.append(qc)

    # Safety
    print("[6/8] Generating safety protocols...")
    safety = await generator.generate_safety_protocol(
        "Handling corrosive chemicals",
        {},
        "chemistry"
    )
    sop.safety_protocols.append(safety)

    # Validation
    print("[7/8] Generating validation criteria...")
    validation = await generator.generate_validation_criterion(
        "Particle size",
        {},
        "chemistry"
    )
    sop.validation_criteria.append(validation)

    # Scaling
    print("[8/8] Generating scaling information...")
    scaling = await generator.generate_scaling_info(
        "Nanoparticle synthesis",
        {},
        "chemistry"
    )
    sop.scaling_info.append(scaling)

    # Display results
    print(f"\n[OK] Complete SOP built from {generator.statistics['total_operations']} individual components")
    print(f"\nSOP Structure:")
    print(f"  - Environmental conditions: {len(sop.environmental_conditions)}")
    print(f"  - Equipment items: {len(sop.equipment)}")
    print(f"  - Materials: {len(sop.materials)}")
    print(f"  - Protocol steps: {len(sop.protocols)}")
    print(f"  - Quality control items: {len(sop.quality_control)}")
    print(f"  - Safety protocols: {len(sop.safety_protocols)}")
    print(f"  - Validation criteria: {len(sop.validation_criteria)}")
    print(f"  - Scaling info: {len(sop.scaling_info)}")

    # Show statistics
    print(f"\nComponent Statistics:")
    for operation_type, components in generator.statistics.items():
        if operation_type != "total_operations" and isinstance(components, dict) and components:
            print(f"  - {operation_type}: {len(components)} component types")

    return sop


async def main():
    """Run all demos"""
    print("\n")
    print("=" * 80)
    print("  SOP COMPONENT SYSTEM - DEMONSTRATION")
    print("  Granular Generation and Refinement of Every SOP Component")
    print("  Paper: arXiv:2511.09030")
    print("=" * 80)

    try:
        await demo_1_capabilities()
        await demo_2_environmental_conditions()
        await demo_3_equipment_specifications()
        await demo_4_materials()
        await demo_5_protocol_steps()
        await demo_6_quality_control()
        await demo_7_safety_protocols()
        await demo_8_validation_criteria()
        await demo_9_scaling_info()
        await demo_10_component_optimization()
        await demo_11_component_testing()
        await demo_12_complete_sop_from_components()

        print_section("DEMO COMPLETE")

        print("[OK] All component demos completed successfully!")
        print("\nKey Capabilities Demonstrated:")
        print("  1. Generate any SOP component independently")
        print("  2. Refine individual components based on feedback")
        print("  3. Optimize components through evolution")
        print("  4. Test components for safety issues")
        print("  5. Build complete SOPs from individual components")
        print("\nComponent Types Supported:")
        print("  - Environmental conditions (parameters)")
        print("  - Equipment specifications")
        print("  - Materials/reagents")
        print("  - Protocol steps")
        print("  - Quality control procedures")
        print("  - Safety protocols")
        print("  - Validation criteria")
        print("  - Scaling information")
        print("\nNext Steps:")
        print("1. Run validation: python validate_sop_components.py")
        print("2. Use in your code:")
        print("     from sop_component_system import generate_sop_component, SOPComponentType")
        print("     component = await generate_sop_component(")
        print("         SOPComponentType.ENVIRONMENTAL_CONDITION,")
        print("         'Temperature',")
        print("         context={'purpose': 'synthesis'}")
        print("     )")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n[ERROR] Demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
