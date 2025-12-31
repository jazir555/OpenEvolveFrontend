"""
SOP Generator - Demo Script

Demonstrates generating and refining Standard Operating Procedures (SOPs)
using the MAKER framework for zero-error, turnkey-ready protocols.

Usage:
    python demo_sop_generator.py
"""

import asyncio
import logging
from datetime import datetime

from sop_generator import (
    SOPGenerator,
    StandardOperatingProcedure,
    SOPParameter,
    SOPStep,
    generate_sop,
    refine_sop,
    get_sop_capabilities
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
    """Demo 1: Check SOP generator capabilities"""
    print_section("DEMO 1: SOP GENERATOR CAPABILITIES")

    capabilities = get_sop_capabilities()

    print("SOP Generator Status:")
    print(f"  - Generation Enabled: {capabilities['sop_generation_enabled']}")
    print(f"  - Refinement Enabled: {capabilities['sop_refinement_enabled']}")

    print(f"\n  Supported Domains ({len(capabilities['supported_domains'])}):")
    for domain in capabilities['supported_domains']:
        print(f"    - {domain}")

    print("\n  Features:")
    for feature, description in capabilities['features'].items():
        print(f"    - {feature}: {description}")

    print("\n  Research Paper:")
    paper = capabilities['paper']
    print(f"    - {paper['title']}")
    print(f"    - arXiv: {paper['arxiv']}")


async def demo_2_simple_generation():
    """Demo 2: Generate a simple SOP"""
    print_section("DEMO 2: SIMPLE SOP GENERATION")

    generator = SOPGenerator()

    requirement = """
    Create a protocol for mixing two chemical solutions in a laboratory setting.

    The process should:
    1. Ensure proper safety precautions
    2. Specify exact mixing ratios
    3. Define temperature requirements
    4. Include verification steps
    """

    print("Requirement:")
    print(requirement)

    print("\nGenerating SOP... (this may take a moment)")

    sop = await generator.generate_sop(
        requirement_description=requirement,
        domain="chemistry",
        constraints=["Must use standard lab equipment"],
        equipment_available=["Magnetic stirrer", "Thermometer", "Beakers", "Safety glasses"]
    )

    print(f"\n✓ SOP Generated: {sop.title}")
    print(f"  Version: {sop.version}")
    print(f"  Status: {sop.status}")
    print(f"  Classification: {sop.classification}")

    # Show key sections
    print(f"\n  Environmental Conditions ({len(sop.environmental_conditions)}):")
    for name, param in list(sop.environmental_conditions.items())[:3]:
        print(f"    - {name}: {param.format_spec()}")

    print(f"\n  Protocols ({len(sop.protocols)} steps):")
    for step in sop.protocols[:3]:
        print(f"    - Step {step.step_number}: {step.action[:60]}...")

    print(f"\n  Quality Control ({len(sop.quality_control)} items)")
    print(f"  Safety Protocols ({len(sop.safety_protocols)} items)")

    return sop


async def demo_3_detailed_chemistry_sop():
    """Demo 3: Generate detailed chemistry SOP"""
    print_section("DEMO 3: DETAILED CHEMISTRY SOP")

    requirement = """
    Magneto-chemical assembly of iron oxide nanoparticles for biomedical applications.

    Process Overview:
    1. Prepare precursor solutions (Fe2+ and Fe3+ salts)
    2. Mix under controlled atmosphere (nitrogen)
    3. Heat to specific temperature with precise ramping
    4. Hold at reaction temperature for specified duration
    5. Cool with controlled cooling rate
    6. Wash and purify nanoparticles
    7. Characterize size and magnetic properties

    Critical Parameters:
    - Temperature control is critical (affects particle size)
    - Mixing ratio determines magnetic properties
    - Reaction time influences yield
    - Cooling rate affects crystallinity
    """

    print("Requirement: Magneto-chemical assembly of iron oxide nanoparticles")
    print("\nCritical Requirements:")
    print("  - Precise temperature control ±2°C")
    print("  - Controlled atmosphere (nitrogen)")
    print("  - Specific stoichiometric ratios")
    print("  - Controlled heating/cooling rates")

    print("\nGenerating detailed SOP...")

    sop = await generate_sop(
        requirement=requirement,
        domain="chemistry",
        constraints=[
            "Temperature must be controlled within ±2°C",
            "Must use nitrogen atmosphere",
            "Particle size target: 10-15 nm",
            "Must be reproducible"
        ],
        equipment=[
            "Three-neck round bottom flask",
            "Condenser",
            "Magnetic stirrer with hotplate",
            "Temperature controller",
            "Nitrogen gas supply",
            "Thermocouple",
            "Syringe pump"
        ]
    )

    print(f"\n✓ SOP Generated: {sop.title}")
    print(f"  Version: {sop.version}")

    # Show structured content
    if sop.environmental_conditions:
        print("\n  Environmental Conditions:")
        for name, param in sop.environmental_conditions.items():
            if param.critical:
                print(f"    [CRITICAL] {name}:")
                print(f"              Target: {param.format_spec()}")
                if param.verification_method:
                    print(f"              Verify: {param.verification_method}")
                if param.rationale:
                    print(f"              Why: {param.rationale}")

    if sop.protocols:
        print(f"\n  Protocol Steps (showing first 3 of {len(sop.protocols)}):")
        for step in sop.protocols[:3]:
            print(f"\n    Step {step.step_number}: {step.action}")
            if step.duration:
                mins = step.duration / 60
                print(f"      Duration: {mins:.1f} min ± {step.duration_tolerance/60:.1f} min")
            if step.verification_method:
                print(f"      Verify: {step.verification_method}")
            if step.acceptance_criteria:
                print(f"      Accept: {step.acceptance_criteria}")
            if step.contingency_action:
                print(f"      Contingency: {step.contingency_action}")

    return sop


async def demo_4_refinement():
    """Demo 4: Refine an existing SOP"""
    print_section("DEMO 4: SOP REFINEMENT")

    # Create a simple SOP first
    sop = StandardOperatingProcedure(
        title="Basic Chemical Mixing Protocol",
        version="1.0",
        status="DRAFT",
        effective_date=datetime.now().strftime("%Y-%m-%d"),
        description="Basic protocol for mixing chemicals",
        classification="DRAFT"
    )

    # Add some incomplete content
    sop.environmental_conditions = {
        "Temperature": SOPParameter(
            name="Temperature",
            value=25.0,
            unit="°C",
            tolerance=0.0,  # No tolerance - this is an issue!
            verification_method="",
            critical=True
        )
    }

    sop.protocols = [
        SOPStep(
            step_number=1,
            action="Mix the chemicals together",
            duration=None,  # No duration - this is an issue!
            verification_method="",
            acceptance_criteria="",  # No acceptance criteria - this is an issue!
            contingency_action=""  # No contingency - this is an issue!
        ),
        SOPStep(
            step_number=2,
            action="Heat the mixture",
            duration=300.0,
            verification_method="Visual inspection",
            acceptance_criteria="Should be uniform",
            contingency_action=""
        )
    ]

    print("Original SOP: Basic Chemical Mixing Protocol (v1.0)")
    print("\nIssues identified:")
    print("  - Temperature has no tolerance specified")
    print("  - Step 1 has no duration")
    print("  - Step 1 has no acceptance criteria")
    print("  - Step 1 has no contingency action")
    print("  - Step 2 has no contingency action")

    print("\nRefining SOP based on feedback...")

    refined_sop = await refine_sop(
        requirement="Add missing tolerances, durations, and acceptance criteria",
        existing_sop=sop,
        feedback=[
            "All parameters need realistic tolerances",
            "All steps need duration estimates",
            "All steps need acceptance criteria",
            "All steps need contingency actions"
        ]
    )

    print(f"\n✓ Refined SOP: {refined_sop.title}")
    print(f"  Version: {refined_sop.version} (updated from {sop.version})")
    print(f"  Revisions: {len(refined_sop.revision_history)}")

    print("\nRevision History:")
    for revision in refined_sop.revision_history:
        print(f"  - {revision['date']}: {revision['change']}")
        print(f"    Previous: {revision['previous_version']}")


async def demo_5_markdown_export():
    """Demo 5: Export SOP as Markdown"""
    print_section("DEMO 5: MARKDOWN EXPORT")

    # Create a sample SOP
    sop = StandardOperatingProcedure(
        title="Sample Laboratory Protocol",
        version="1.0",
        status="ACTIVE",
        effective_date="2025-01-15",
        description="A sample protocol demonstrating SOP structure",
        classification="TURNKEY"
    )

    # Add content
    sop.preconditions = [
        "Laboratory temperature must be 22±2°C",
        "Operator must have completed chemical safety training",
        "All equipment must be calibrated within last 30 days"
    ]

    sop.environmental_conditions = {
        "Temperature": SOPParameter(
            name="Temperature",
            value=22.0,
            unit="°C",
            tolerance=2.0,
            verification_method="Calibrated digital thermometer",
            critical=True,
            rationale="Temperature affects reaction kinetics"
        ),
        "Humidity": SOPParameter(
            name="Humidity",
            value=45.0,
            unit="%",
            tolerance=5.0,
            verification_method="Hygrometer",
            critical=False,
            rationale="Humidity control prevents moisture contamination"
        )
    }

    sop.equipment = [
        {
            "name": "Magnetic Stirrer",
            "model": "ThermoFisher SuperSpinner 5000",
            "specifications": "Speed: 100-2000 RPM, Temperature: RT-250°C"
        },
        {
            "name": "Digital Thermometer",
            "model": "Fluke 51 II",
            "specifications": "Range: -50°C to 160°C, Accuracy: ±0.1°C"
        }
    ]

    sop.materials = [
        {
            "name": "Iron(II) chloride tetrahydrate",
            "purity": "≥99%",
            "grade": "ACS reagent grade",
            "amount": "10.0 g ± 0.1 g"
        },
        {
            "name": "Deionized water",
            "purity": "18.2 MΩ·cm",
            "grade": "Type I",
            "amount": "500 mL ± 5 mL"
        }
    ]

    sop.protocols = [
        SOPStep(
            step_number=1,
            action="Prepare solution by dissolving FeCl2 in deionized water",
            duration=600.0,
            duration_tolerance=60.0,
            verification_method="Visual check - solution should be clear pale green",
            acceptance_criteria="No visible precipitate",
            contingency_action="If precipitate forms, add small amount of HCl and stir",
            substeps=["Weigh 10.0 g FeCl2", "Add to 400 mL water", "Stir until dissolved"]
        ),
        SOPStep(
            step_number=2,
            action="Heat solution to 80°C with continuous stirring",
            duration=900.0,
            duration_tolerance=120.0,
            verification_method="Thermometer reading",
            acceptance_criteria="Temperature stable at 80±2°C for 2 minutes",
            contingency_action="If temperature exceeds 82°C, remove from heat immediately"
        )
    ]

    sop.quality_control = [
        "Verify solution clarity before heating",
        "Record temperature every 60 seconds",
        "Confirm final temperature stability",
        "Document any deviations from protocol"
    ]

    sop.safety_protocols = [
        "Wear safety glasses, lab coat, and nitrile gloves",
        "Work in fume hood",
        "Have spill kit readily available",
        "Emergency eyewash station must be accessible"
    ]

    sop.validation_criteria = [
        "Solution temperature reaches 80±2°C",
        "No precipitate formation",
        "Heating rate within specification"
    ]

    sop.scaling_info = [
        "For 2x scale: double all materials, same temperature profile",
        "For 5x scale: use larger vessel, increase heating time by 50%",
        "Maximum scale: 10x due to heat transfer limitations"
    ]

    print("Exporting SOP to Markdown format...\n")

    markdown = sop.to_markdown()

    # Show first 100 lines
    lines = markdown.split('\n')
    preview_lines = lines[:100]
    print('\n'.join(preview_lines))

    if len(lines) > 100:
        print(f"\n... ({len(lines) - 100} more lines)")

    print(f"\n✓ Full Markdown export: {len(markdown)} characters")

    return sop


async def demo_6_comparison():
    """Demo 6: Compare with/without MAKER"""
    print_section("DEMO 6: MAKER BENEFITS")

    print("MAKER Framework Benefits for SOP Generation:\n")

    print("1. ZERO-ERROR GUARANTEE")
    print("   - First-to-ahead-by-k voting ensures consensus")
    print("   - k=3 provides 99% confidence")
    print("   - Eliminates ambiguous or incomplete SOP sections")

    print("\n2. TASK DECOMPOSITION")
    print("   - Complex SOPs broken into manageable sections")
    print("   - Each section generated with appropriate specificity")
    print("   - Parallel generation of independent sections")

    print("\n3. EVOLUTIONARY OPTIMIZATION")
    print("   - Iterative improvement of SOP content")
    print("   - Parameter optimization based on quality metrics")
    print("   - Convergence to highest-quality SOP")

    print("\n4. QUALITY EVALUATION")
    print("   - Completeness: All sections present")
    print("   - Specificity: All parameters with tolerances")
    print("   - Realism: Achievable tolerances, verification methods")
    print("   - Clarity: Unambiguous instructions")
    print("   - Safety: Comprehensive protocols")

    print("\n5. CONTINUOUS IMPROVEMENT")
    print("   - Refine SOPs based on execution feedback")
    print("   - Optimize parameters based on performance data")
    print("   - Learn from successful and failed executions")


async def main():
    """Run all demos"""
    print("\n")
    print("=" * 80)
    print("  SOP GENERATOR - DEMONSTRATION")
    print("  MAKER-Based Standard Operating Procedure Generation")
    print("  arXiv:2511.09030")
    print("=" * 80)

    try:
        await demo_1_capabilities()
        await demo_2_simple_generation()
        await demo_3_detailed_chemistry_sop()
        await demo_4_refinement()
        await demo_5_markdown_export()
        await demo_6_comparison()

        print_section("DEMO COMPLETE")

        print("✓ All demos completed successfully!")
        print("\nNext Steps:")
        print("1. Run validation: python validate_sop_generator.py")
        print("2. Read guide: SOP_GENERATOR_GUIDE.md")
        print("3. Use in your code:")
        print("     from sop_generator import generate_sop")
        print("     sop = await generate_sop('your requirement here')")
        print("\nKey Features:")
        print("  - Generate complete SOPs from requirements")
        print("  - Refine existing SOPs based on feedback")
        print("  - Zero-error guarantees through MAKER voting")
        print("  - Turnkey-ready protocols with all parameters specified")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
