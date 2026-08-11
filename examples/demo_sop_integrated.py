"""
SOP Integrated System - Demo Script

Demonstrates the unified SOP generation system with all integrations:
- MAKER/MDAP (core zero-error generation)
- LeanAide (formal verification)
- Evolution (evolutionary optimization)
- Adversarial (red/blue team safety testing)
- MCTS (protocol exploration)

Usage:
    python demo_sop_integrated.py
"""

import asyncio
import logging
from datetime import datetime

from sop_integrated_system import (
    IntegratedSOPGenerator,
    SOPIntegratedConfig,
    SOPIntegrationMode,
    generate_integrated_sop,
    get_integrated_capabilities
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
    """Demo 1: Check integrated system capabilities"""
    print_section("DEMO 1: INTEGRATED SYSTEM CAPABILITIES")

    capabilities = get_integrated_capabilities()

    print("SOP Integrated System Status:")
    print(f"  - SOP Generation: {capabilities['sop_generation_enabled']}")

    print("\n  Integrations:")
    for name, info in capabilities['integrations'].items():
        status = "OK" if info['enabled'] else "MISSING"
        print(f"    [{status}] {name.upper()}: {info['description']}")

    print(f"\n  Supported Domains ({len(capabilities['supported_domains'])}):")
    for domain in capabilities['supported_domains']:
        print(f"    - {domain}")

    print(f"\n  Integration Modes ({len(capabilities['modes'])}):")
    for mode in capabilities['modes']:
        print(f"    - {mode}")

    print("\n  Research Paper:")
    paper = capabilities['paper']
    print(f"    - {paper['title']}")
    print(f"    - arXiv: {paper['arxiv']}")


async def demo_2_basic_generation():
    """Demo 2: Basic SOP generation (MAKER/MDAP only)"""
    print_section("DEMO 2: BASIC SOP GENERATION (MAKER/MDAP)")

    config = SOPIntegratedConfig(
        mode=SOPIntegrationMode.BASIC,
        enable_leanaide=False,
        enable_evolution=False,
        enable_adversarial=False,
        enable_mcts=False
    )

    generator = IntegratedSOPGenerator(config)

    requirement = "Create a protocol for measuring liquid volume in a laboratory"

    print(f"Requirement: {requirement}")
    print("Mode: BASIC (MAKER/MDAP only)")
    print("\nGenerating SOP...")

    sop = await generator.generate_sop(
        requirement=requirement,
        domain="chemistry",
        constraints=["Use standard laboratory equipment"],
        equipment=["Graduated cylinder", "Beaker", "Pipette"]
    )

    print(f"\n[OK] SOP Generated: {sop.title}")
    print(f"  Version: {sop.version}")
    print(f"  Status: {sop.status}")

    # Show key statistics
    stats = generator.get_statistics()
    print(f"\n  Generation Statistics:")
    print(f"    - SOPs generated: {stats['sops_generated']}")
    print(f"    - Total time: {stats['total_generation_time']:.1f}s")

    return sop


async def demo_3_formal_verification():
    """Demo 3: SOP with LeanAide formal verification"""
    print_section("DEMO 3: SOP WITH FORMAL VERIFICATION")

    config = SOPIntegratedConfig(
        mode=SOPIntegrationMode.FORMAL,
        enable_leanaide=True,
        verify_mathematical_steps=True,
        leanaide_confidence_threshold=0.7
    )

    generator = IntegratedSOPGenerator(config)

    requirement = """
    Create a protocol for preparing a precise chemical solution.

    The protocol must include:
    1. Calculation of required mass based on desired molarity
    2. Verification of concentration using formula
    3. Dilution calculations if needed
    """

    print("Requirement: Precise solution preparation with calculations")
    print("Mode: FORMAL (MAKER/MDAP + LeanAide)")
    print("\nGenerating SOP with formal verification...")

    sop = await generator.generate_sop(
        requirement=requirement,
        domain="chemistry",
        constraints=["All calculations must be verified"],
        equipment=["Balance", "Volumetric flask", "Calculator"]
    )

    print(f"\n[OK] SOP Generated: {sop.title}")
    print(f"  Version: {sop.version}")

    # Show formal verification statistics
    stats = generator.get_statistics()
    print(f"\n  Verification Statistics:")
    print(f"    - Steps formally verified: {stats['formal_verifications']}")
    print(f"    - Total time: {stats['total_generation_time']:.1f}s")

    # Show steps with verification
    print("\n  Steps with Formal Verification:")
    for step in sop.protocols:
        if "formal verification" in step.acceptance_criteria.lower():
            print(f"    - Step {step.step_number}: {step.action[:50]}...")
            print(f"      {step.acceptance_criteria}")

    return sop


async def demo_4_evolutionary_optimization():
    """Demo 4: SOP with evolutionary optimization"""
    print_section("DEMO 4: SOP WITH EVOLUTIONARY OPTIMIZATION")

    config = SOPIntegratedConfig(
        mode=SOPIntegrationMode.EVOLUTIONARY,
        enable_evolution=True,
        evolution_generations=10,  # Low for demo
        evolution_population_size=10,
        evolution_mutation_rate=0.15
    )

    generator = IntegratedSOPGenerator(config)

    requirement = """
    Create a protocol for magnetic nanoparticle synthesis.

    Key parameters to optimize:
    - Temperature
    - Reaction time
    - Mixing ratio
    - pH level
    """

    print("Requirement: Magnetic nanoparticle synthesis")
    print("Mode: EVOLUTIONARY (MAKER/MDAP + Evolution)")
    print("  - Generations: 10")
    print("  - Population: 10")
    print("  - Mutation rate: 15%")
    print("\nGenerating and optimizing SOP...")

    sop = await generator.generate_sop(
        requirement=requirement,
        domain="chemistry",
        constraints=["Temperature must stay below 80°C"],
        equipment=["Magnetic stirrer", "Hotplate", "Thermometer", "pH meter"]
    )

    print(f"\n[OK] SOP Generated: {sop.title}")
    print(f"  Version: {sop.version}")

    # Show evolution statistics
    stats = generator.get_statistics()
    print(f"\n  Evolution Statistics:")
    print(f"    - Evolutionary runs: {stats['evolutionary_optimizations']}")
    print(f"    - Total time: {stats['total_generation_time']:.1f}s")

    # Show optimized parameters
    print("\n  Optimized Parameters:")
    for param_name, param in list(sop.environmental_conditions.items())[:3]:
        print(f"    - {param_name}: {param.format_spec()}")
        if param.verification_method:
            print(f"      Verification: {param.verification_method}")

    return sop


async def demo_5_adversarial_testing():
    """Demo 5: SOP with adversarial testing"""
    print_section("DEMO 5: SOP WITH ADVERSARIAL TESTING")

    config = SOPIntegratedConfig(
        mode=SOPIntegrationMode.ADVERSARIAL,
        enable_adversarial=True,
        red_team_agents=3,
        blue_team_agents=2,
        adversarial_rounds=2
    )

    generator = IntegratedSOPGenerator(config)

    requirement = """
    Create a protocol for handling hazardous chemicals.

    Must include comprehensive safety protocols.
    """

    print("Requirement: Hazardous chemical handling")
    print("Mode: ADVERSARIAL (MAKER/MDAP + Red/Blue Team)")
    print("  - Red team agents: 3")
    print("  - Blue team agents: 2")
    print("  - Rounds: 2")
    print("\nGenerating and testing SOP...")

    sop = await generator.generate_sop(
        requirement=requirement,
        domain="chemistry",
        constraints=["Must include emergency procedures"],
        equipment=["Fume hood", "Safety glasses", "Gloves", "Eyewash station"]
    )

    print(f"\n[OK] SOP Generated: {sop.title}")
    print(f"  Version: {sop.version}")

    # Show adversarial statistics
    stats = generator.get_statistics()
    print(f"\n  Adversarial Testing Statistics:")
    print(f"    - Adversarial tests run: {stats['adversarial_tests']}")
    print(f"    - Total time: {stats['total_generation_time']:.1f}s")

    # Show safety protocols (likely improved by adversarial testing)
    print("\n  Safety Protocols (after adversarial testing):")
    for i, protocol in enumerate(sop.safety_protocols[:5], 1):
        print(f"    {i}. {protocol}")

    # Show revision history
    if len(sop.revision_history) > 1:
        print("\n  Revision History:")
        for revision in sop.revision_history:
            print(f"    - {revision['date']}: {revision['change']}")

    return sop


async def demo_6_full_integration():
    """Demo 6: Full integration (all systems)"""
    print_section("DEMO 6: FULL INTEGRATION")

    config = SOPIntegratedConfig(
        mode=SOPIntegrationMode.FULL,
        enable_leanaide=True,
        enable_evolution=True,
        enable_adversarial=True,
        enable_mcts=True,
        evolution_generations=5,  # Lower for demo
        evolution_population_size=8,
        red_team_agents=2,
        adversarial_rounds=2
    )

    generator = IntegratedSOPGenerator(config)

    requirement = """
    Magneto-chemical assembly of iron oxide nanoparticles.

    Process:
    1. Prepare precursor solutions
    2. Mix under nitrogen atmosphere
    3. Heat to specific temperature
    4. Hold for reaction time
    5. Cool and wash nanoparticles

    Key requirements:
    - Temperature must be controlled precisely
    - Calculations for concentrations needed
    - Safety protocols for handling chemicals
    - Quality control for particle size
    """

    print("Requirement: Magneto-chemical assembly (complex)")
    print("Mode: FULL (All integrations enabled)")
    print("\n  Integrations:")
    print("    [OK] MAKER/MDAP (zero-error generation)")
    print("    [OK] LeanAide (formal verification)")
    print("    [OK] Evolution (parameter optimization)")
    print("    [OK] Adversarial (safety testing)")
    print("    [OK] MCTS (protocol exploration)")
    print("\nGenerating SOP with full integration...")

    sop = await generator.generate_sop(
        requirement=requirement,
        domain="chemistry",
        constraints=[
            "Temperature < 80°C",
            "Particle size: 10-15 nm",
            "Must include emergency procedures"
        ],
        equipment=[
            "Three-neck flask",
            "Condenser",
            "Magnetic stirrer with hotplate",
            "Temperature controller",
            "Nitrogen gas supply",
            "Thermometer"
        ]
    )

    print(f"\n[OK] SOP Generated: {sop.title}")
    print(f"  Version: {sop.version}")
    print(f"  Status: {sop.status}")
    print(f"  Classification: {sop.classification}")

    # Show comprehensive statistics
    stats = generator.get_statistics()
    print(f"\n  Comprehensive Statistics:")
    print(f"    - SOPs generated: {stats['sops_generated']}")
    print(f"    - Formal verifications: {stats['formal_verifications']}")
    print(f"    - Evolutionary optimizations: {stats['evolutionary_optimizations']}")
    print(f"    - Adversarial tests: {stats['adversarial_tests']}")
    print(f"    - MCTS explorations: {stats['mcts_explorations']}")
    print(f"    - Total time: {stats['total_generation_time']:.1f}s")

    print("\n  Integration Status:")
    for name, enabled in stats['integrations_enabled'].items():
        status = "OK" if enabled else "MISSING"
        print(f"    [{status}] {name.upper()}")

    # Show key sections
    print(f"\n  SOP Structure:")
    print(f"    - Environmental conditions: {len(sop.environmental_conditions)}")
    print(f"    - Equipment items: {len(sop.equipment)}")
    print(f"    - Materials: {len(sop.materials)}")
    print(f"    - Protocol steps: {len(sop.protocols)}")
    print(f"    - Quality control items: {len(sop.quality_control)}")
    print(f"    - Safety protocols: {len(sop.safety_protocols)}")

    # Show sample environmental conditions
    if sop.environmental_conditions:
        print(f"\n  Sample Environmental Conditions:")
        for name, param in list(sop.environmental_conditions.items())[:2]:
            print(f"    - {name}: {param.format_spec()}")
            if param.verification_method:
                print(f"      Verification: {param.verification_method}")

    # Show sample protocol steps
    if sop.protocols:
        print(f"\n  Sample Protocol Steps (first 2):")
        for step in sop.protocols[:2]:
            print(f"\n    Step {step.step_number}: {step.action}")
            if step.duration:
                print(f"      Duration: {step.duration/60:.1f} min ± {step.duration_tolerance/60:.1f} min")
            if step.acceptance_criteria:
                print(f"      Acceptance: {step.acceptance_criteria[:80]}...")

    return sop


async def demo_7_comparison():
    """Demo 7: Comparison of different modes"""
    print_section("DEMO 7: MODE COMPARISON")

    requirement = "Create a protocol for mixing two solutions"

    modes_to_compare = [
        (SOPIntegrationMode.BASIC, "Basic (MAKER/MDAP only)"),
        (SOPIntegrationMode.FORMAL, "Formal (+LeanAide)"),
        (SOPIntegrationMode.EVOLUTIONARY, "Evolutionary (+Optimization)"),
        (SOPIntegrationMode.ADVERSARIAL, "Adversarial (+Safety Testing)"),
    ]

    results = []

    for mode, description in modes_to_compare:
        print(f"\nTesting {description}...")

        config = SOPIntegratedConfig(
            mode=mode,
            evolution_generations=3,  # Low for quick demo
            evolution_population_size=5
        )

        generator = IntegratedSOPGenerator(config)

        import time
        start = time.time()
        sop = await generator.generate_sop(
            requirement=requirement,
            domain="chemistry",
            constraints=["Use standard equipment"],
            equipment=["Beaker", "Stirrer"]
        )
        elapsed = time.time() - start

        stats = generator.get_statistics()
        results.append({
            "mode": description,
            "time": elapsed,
            "version": sop.version,
            "revisions": len(sop.revision_history)
        })

        print(f"  Time: {elapsed:.1f}s, Version: {sop.version}, Revisions: {len(sop.revision_history)}")

    # Summary table
    print("\n  Comparison Summary:")
    print("  " + "-" * 70)
    print(f"  {'Mode':<30} {'Time (s)':<12} {'Version':<10} {'Revisions':<10}")
    print("  " + "-" * 70)
    for result in results:
        print(f"  {result['mode']:<30} {result['time']:<12.1f} {result['version']:<10} {result['revisions']:<10}")
    print("  " + "-" * 70)


async def main():
    """Run all demos"""
    print("\n")
    print("=" * 80)
    print("  SOP INTEGRATED SYSTEM - DEMONSTRATION")
    print("  Unified Integration: MAKER/MDAP + LeanAide + Evolution + Adversarial + MCTS")
    print("  Paper: arXiv:2511.09030")
    print("=" * 80)

    try:
        await demo_1_capabilities()
        await demo_2_basic_generation()
        await demo_3_formal_verification()
        await demo_4_evolutionary_optimization()
        await demo_5_adversarial_testing()
        await demo_6_full_integration()
        await demo_7_comparison()

        print_section("DEMO COMPLETE")

        print("[OK] All demos completed successfully!")
        print("\nKey Integrations Demonstrated:")
        print("  1. MAKER/MDAP - Zero-error generation through voting")
        print("  2. LeanAide - Formal verification of mathematical procedures")
        print("  3. Evolution - Evolutionary optimization of parameters")
        print("  4. Adversarial - Red/blue team safety testing")
        print("  5. MCTS - Protocol exploration and optimization")
        print("\nNext Steps:")
        print("1. Run validation: python validate_sop_integrated.py")
        print("2. Read guide: SOP_INTEGRATED_GUIDE.md")
        print("3. Use in your code:")
        print("     from sop_integrated_system import generate_integrated_sop")
        print("     sop = await generate_integrated_sop('your requirement')")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n[FAIL] Demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
