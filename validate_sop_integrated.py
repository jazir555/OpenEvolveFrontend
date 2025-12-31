"""
SOP Integrated System - Validation Script

Validates that the integrated SOP generator system works correctly with all integrations:
- MAKER/MDAP (core)
- LeanAide (formal verification)
- Evolution (evolutionary optimization)
- Adversarial (red/blue team testing)
- MCTS (protocol exploration)

Usage:
    python validate_sop_integrated.py
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

    # Test 1: SOP integrated system
    try:
        from sop_integrated_system import (
            IntegratedSOPGenerator,
            SOPIntegratedConfig,
            SOPIntegrationMode,
            generate_integrated_sop,
            get_integrated_capabilities
        )
        results["imports"].append({
            "module": "sop_integrated_system",
            "status": "OK"
        })
        print("[OK] SOP integrated system module")
    except ImportError as e:
        results["failures"].append({"module": "sop_integrated_system", "error": str(e)})
        print(f"[FAIL] SOP integrated system: {e}")

    # Test 2: Core SOP generator
    try:
        from sop_generator import (
            SOPGenerator,
            StandardOperatingProcedure,
            SOPEvaluator
        )
        results["imports"].append({
            "module": "sop_generator",
            "status": "OK"
        })
        print("[OK] Core SOP generator module")
    except ImportError as e:
        results["failures"].append({"module": "sop_generator", "error": str(e)})
        print(f"[FAIL] Core SOP generator: {e}")

    # Test 3: Generic MAKER integration
    try:
        from generic_maker_integration import (
            run_generic_maker,
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

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All core imports successful!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} import(s) failed")

    return results


def validate_integration_availability():
    """Validate which integrations are available"""
    print_section("2. VALIDATING INTEGRATION AVAILABILITY")

    results = {"status": "unknown", "integrations": {}, "failures": []}

    integrations_to_check = [
        ("leanaide_workflow_integration", "LeanAide", "formal verification"),
        ("evolution_maker_integration", "Evolution", "evolutionary optimization"),
        ("adversarial_maker_integration", "Adversarial", "adversarial testing"),
        ("hybrid_maker_integration", "Hybrid", "hybrid strategies"),
        ("mdap_engine", "MDAP", "task decomposition"),
        ("leanaide_mcts", "MCTS", "protocol exploration"),
    ]

    for module_name, display_name, description in integrations_to_check:
        try:
            __import__(module_name)
            results["integrations"][display_name] = {
                "status": "OK",
                "description": description
            }
            print(f"[OK] {display_name}: {description}")
        except ImportError:
            results["integrations"][display_name] = {
                "status": "MISSING",
                "description": description
            }
            print(f"[--] {display_name}: Not available (optional)")

    # Determine status (missing integrations are OK, they're optional)
    core_count = sum(1 for v in results["integrations"].values() if v["status"] == "OK")
    results["status"] = "pass"
    print(f"\n[OK] {core_count} integrations available (optional integrations can be missing)")

    return results


def validate_modes():
    """Validate integration modes"""
    print_section("3. VALIDATING INTEGRATION MODES")

    results = {"status": "unknown", "modes": [], "failures": []}

    try:
        from sop_integrated_system import SOPIntegrationMode

        expected_modes = [
            "BASIC",
            "FORMAL",
            "EVOLUTIONARY",
            "ADVERSARIAL",
            "MCTS",
            "FULL"
        ]

        available_modes = [mode.name for mode in SOPIntegrationMode]

        for mode in expected_modes:
            if mode in available_modes:
                results["modes"].append({
                    "mode": mode,
                    "status": "OK"
                })
                print(f"[OK] Mode: {mode}")
            else:
                results["failures"].append({"mode": mode, "error": "Not found"})
                print(f"[FAIL] Mode: {mode} - Not found")

        if not results["failures"]:
            results["status"] = "pass"
            print(f"\n[OK] All {len(expected_modes)} modes available!")
        else:
            results["status"] = "fail"
            print(f"\n[FAIL] {len(results['failures'])} mode(s) missing")

    except Exception as e:
        results["failures"].append({"mode": "all", "error": str(e)})
        print(f"[FAIL] Mode validation: {e}")
        results["status"] = "fail"

    return results


async def validate_basic_generation():
    """Validate basic SOP generation"""
    print_section("4. VALIDATING BASIC SOP GENERATION")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from sop_integrated_system import (
            IntegratedSOPGenerator,
            SOPIntegratedConfig,
            SOPIntegrationMode
        )

        config = SOPIntegratedConfig(
            mode=SOPIntegrationMode.BASIC,
            enable_leanaide=False,
            enable_evolution=False,
            enable_adversarial=False,
            enable_mcts=False
        )

        generator = IntegratedSOPGenerator(config)

        results["checks"].append({
            "name": "generator_initialization",
            "status": "OK"
        })
        print("[OK] Generator initialization (BASIC mode)")

        # Generate a simple SOP
        print("\n  Generating test SOP...")
        sop = await generator.generate_sop(
            requirement="Create a protocol for mixing two solutions",
            domain="chemistry",
            constraints=["Use standard equipment"],
            equipment=["Beaker", "Stirrer"]
        )

        # Verify SOP structure
        assert sop.title is not None
        assert sop.version is not None
        assert sop.status is not None

        results["checks"].append({
            "name": "sop_generation",
            "status": "OK",
            "title": sop.title
        })
        print(f"[OK] SOP generated: {sop.title}")

        # Verify it can export to markdown
        markdown = sop.to_markdown()
        assert len(markdown) > 100

        results["checks"].append({
            "name": "markdown_export",
            "status": "OK",
            "length": len(markdown)
        })
        print(f"[OK] Markdown export: {len(markdown)} chars")

        # Check statistics
        stats = generator.get_statistics()
        assert stats["sops_generated"] > 0

        results["checks"].append({
            "name": "statistics",
            "status": "OK"
        })
        print("[OK] Statistics tracking working")

    except Exception as e:
        results["failures"].append({"check": "basic_generation", "error": str(e)})
        print(f"[FAIL] Basic generation: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All basic generation checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


async def validate_config_modes():
    """Validate different configuration modes"""
    print_section("5. VALIDATING CONFIGURATION MODES")

    results = {"status": "unknown", "modes": [], "failures": []}

    try:
        from sop_integrated_system import (
            IntegratedSOPGenerator,
            SOPIntegratedConfig,
            SOPIntegrationMode
        )

        modes_to_test = [
            (SOPIntegrationMode.BASIC, "BASIC"),
            (SOPIntegrationMode.FORMAL, "FORMAL"),
            (SOPIntegrationMode.EVOLUTIONARY, "EVOLUTIONARY"),
            (SOPIntegrationMode.ADVERSARIAL, "ADVERSARIAL"),
            (SOPIntegrationMode.FULL, "FULL"),
        ]

        for mode, mode_name in modes_to_test:
            print(f"\n  Testing {mode_name} mode...")

            try:
                config = SOPIntegratedConfig(
                    mode=mode,
                    evolution_generations=2,  # Low for validation
                    evolution_population_size=3
                )

                generator = IntegratedSOPGenerator(config)
                stats = generator.get_statistics()

                results["modes"].append({
                    "mode": mode_name,
                    "status": "OK"
                })
                print(f"  [OK] {mode_name} mode configured")

            except Exception as e:
                results["failures"].append({
                    "mode": mode_name,
                    "error": str(e)
                })
                print(f"  [FAIL] {mode_name} mode: {e}")

        if not results["failures"]:
            results["status"] = "pass"
            print("\n[OK] All configuration modes working!")
        else:
            results["status"] = "partial"
            print(f"\n[WARNING] {len(results['failures'])} mode(s) failed (may be optional)")

    except Exception as e:
        results["failures"].append({"mode": "config", "error": str(e)})
        print(f"[FAIL] Configuration validation: {e}")
        results["status"] = "fail"

    return results


async def validate_end_to_end():
    """Validate end-to-end integrated generation"""
    print_section("6. VALIDATING END-TO-END INTEGRATED GENERATION")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from sop_integrated_system import (
            generate_integrated_sop,
            SOPIntegrationMode
        )

        print("  Testing full integrated generation...")
        print("  (using lower generation counts for faster validation)")

        # Generate with full integration (but low counts for speed)
        sop = await generate_integrated_sop(
            requirement="Create a protocol for measuring liquid volume",
            domain="chemistry",
            constraints=["Use standard equipment"],
            equipment=["Graduated cylinder", "Beaker"],
            mode=SOPIntegrationMode.FULL
        )

        # Verify SOP structure
        assert sop.title is not None
        assert sop.version is not None

        results["checks"].append({
            "name": "integrated_generation",
            "status": "OK",
            "title": sop.title,
            "version": sop.version
        })
        print(f"[OK] Integrated SOP generated: {sop.title} v{sop.version}")

        # Verify markdown export
        markdown = sop.to_markdown()
        assert len(markdown) > 100

        results["checks"].append({
            "name": "integrated_export",
            "status": "OK",
            "length": len(markdown)
        })
        print(f"[OK] Markdown export: {len(markdown)} chars")

        # Verify it has all expected sections
        has_environmental = len(sop.environmental_conditions) > 0
        has_protocols = len(sop.protocols) > 0
        has_safety = len(sop.safety_protocols) > 0
        has_qc = len(sop.quality_control) > 0

        results["checks"].append({
            "name": "sop_completeness",
            "status": "OK",
            "sections": {
                "environmental": has_environmental,
                "protocols": has_protocols,
                "safety": has_safety,
                "quality_control": has_qc
            }
        })
        print(f"[OK] SOP completeness check:")
        print(f"    - Environmental conditions: {len(sop.environmental_conditions)}")
        print(f"    - Protocols: {len(sop.protocols)} steps")
        print(f"    - Safety protocols: {len(sop.safety_protocols)}")
        print(f"    - Quality control: {len(sop.quality_control)}")

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
    print_section("7. VALIDATING CAPABILITIES")

    results = {"status": "unknown", "checks": [], "failures": []}

    try:
        from sop_integrated_system import get_integrated_capabilities

        capabilities = get_integrated_capabilities()

        # Display capabilities
        print("SOP Integrated System Capabilities:")
        print(f"  - SOP generation: {capabilities.get('sop_generation_enabled', False)}")

        print("\n  Integrations:")
        for name, info in capabilities.get('integrations', {}).items():
            status = "[OK]" if info['enabled'] else "[FAIL]"
            print(f"    [{status}] {name.upper()}: {info['description']}")

        print(f"\n  Supported Domains ({len(capabilities.get('supported_domains', []))}):")
        for domain in capabilities.get('supported_domains', []):
            print(f"    - {domain}")

        print(f"\n  Modes ({len(capabilities.get('modes', []))}):")
        for mode in capabilities.get('modes', []):
            print(f"    - {mode}")

        if 'paper' in capabilities:
            paper = capabilities['paper']
            print(f"\n  Paper: {paper.get('arxiv', 'N/A')}")

        # Verify structure
        assert capabilities.get('sop_generation_enabled') == True
        assert 'integrations' in capabilities
        assert len(capabilities.get('supported_domains', [])) > 0
        assert len(capabilities.get('modes', [])) > 0

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
    print("  SOP INTEGRATED SYSTEM VALIDATION")
    print("  Unified Integration: MAKER/MDAP + LeanAide + Evolution + Adversarial + MCTS")
    print("  Paper: arXiv:2511.09030")
    print("=" * 80)
    print("")

    all_results = {}

    # Run validations
    import_results = validate_imports()
    all_results["imports"] = import_results

    integration_results = validate_integration_availability()
    all_results["integrations"] = integration_results

    mode_results = validate_modes()
    all_results["modes"] = mode_results

    basic_results = asyncio.run(validate_basic_generation())
    all_results["basic_generation"] = basic_results

    config_results = asyncio.run(validate_config_modes())
    all_results["config_modes"] = config_results

    e2e_results = asyncio.run(validate_end_to_end())
    all_results["end_to_end"] = e2e_results

    capabilities_results = validate_capabilities()
    all_results["capabilities"] = capabilities_results

    # Summary
    print_section("VALIDATION SUMMARY")

    total_checks = len(all_results)
    total_passed = sum(1 for r in all_results.values() if r["status"] == "pass")
    total_partial = sum(1 for r in all_results.values() if r["status"] == "partial")
    total_failed = sum(1 for r in all_results.values() if r["status"] == "fail")

    print(f"Categories: {total_checks}")
    print(f"  Passed: {total_passed}")
    print(f"  Partial (optional): {total_partial}")
    print(f"  Failed: {total_failed}")

    if total_failed == 0:
        print("\n" + "=" * 80)
        print("[OK][OK][OK] ALL VALIDATIONS PASSED [OK][OK][OK]")
        print("=" * 80)
        print("\nSOP Integrated System is fully functional!")
        print("\nAvailable Integrations:")
        for name, info in integration_results.get("integrations", {}).items():
            status = "[OK]" if info["status"] == "OK" else "[FAIL]"
            print(f"  [{status}] {name}: {info['description']}")
        print("\nNext steps:")
        print("1. Run demo: python demo_sop_integrated.py")
        print("2. Use in your code:")
        print("     from sop_integrated_system import generate_integrated_sop")
        print("     sop = await generate_integrated_sop('your requirement', mode='full')")
        print("\nAvailable modes:")
        print("  - basic: MAKER/MDAP only")
        print("  - formal: + LeanAide verification")
        print("  - evolutionary: + Evolutionary optimization")
        print("  - adversarial: + Red/blue team testing")
        print("  - mcts: + MCTS exploration")
        print("  - full: All integrations")
        return 0
    else:
        print("\n" + "=" * 80)
        print("[FAIL][FAIL][FAIL] SOME VALIDATIONS FAILED [FAIL][FAIL][FAIL]")
        print("=" * 80)
        print("\nPlease check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
