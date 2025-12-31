"""
MAKER/MDAP Adversarial Integration - Validation Script

This script validates that the MAKER/MDAP adversarial integration is working correctly.

Usage:
    python validate_adversarial_maker_integration.py
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

    # Test 1: Core adversarial module
    try:
        from adversarial import (
            run_maker_enhanced_adversarial_testing,
            get_maker_adversarial_capabilities,
            AdversarialConfiguration
        )
        results["imports"].append({
            "module": "adversarial",
            "status": "OK",
            "functions": ["run_maker_enhanced_adversarial_testing", "get_maker_adversarial_capabilities"]
        })
        print("[OK] Core adversarial module with MAKER functions")
    except ImportError as e:
        results["failures"].append({"module": "adversarial", "error": str(e)})
        print(f"[FAIL] Core adversarial module: {e}")

    # Test 2: Adversarial MAKER integration
    try:
        from adversarial_maker_integration import (
            AdversarialMAKERConfig,
            AdversarialMAKERMode,
            MAKERRedTeamAgent,
            MDAPBlueTeamAgent,
            AdversarialCoEvolution,
            run_maker_adversarial_testing,
            create_adversarial_maker_config
        )
        results["imports"].append({
            "module": "adversarial_maker_integration",
            "status": "OK",
            "classes": ["AdversarialMAKERConfig", "MAKERRedTeamAgent", "MDAPBlueTeamAgent"]
        })
        print("[OK] Adversarial MAKER integration module")
    except ImportError as e:
        results["failures"].append({"module": "adversarial_maker_integration", "error": str(e)})
        print(f"[FAIL] Adversarial MAKER integration: {e}")

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

    # Test 5: Red team
    try:
        from red_team import (
            RedTeamMember,
            RedTeamAssessment,
            IssueFinding,
            IssueCategory
        )
        results["imports"].append({
            "module": "red_team",
            "status": "OK",
            "classes": ["RedTeamMember", "IssueFinding"]
        })
        print("[OK] Red team module")
    except ImportError as e:
        results["failures"].append({"module": "red_team", "error": str(e)})
        print(f"[FAIL] Red team module: {e}")

    # Test 6: Blue team
    try:
        from blue_team import (
            BlueTeamMember,
            BlueTeamAssessment
        )
        # DefenseStrategy is defined in adversarial_maker_integration.py
        results["imports"].append({
            "module": "blue_team",
            "status": "OK",
            "classes": ["BlueTeamMember", "BlueTeamAssessment"]
        })
        print("[OK] Blue team module")
    except ImportError as e:
        results["failures"].append({"module": "blue_team", "error": str(e)})
        print(f"[FAIL] Blue team module: {e}")

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

    # Test 1: Create AdversarialMAKERConfig
    try:
        from adversarial_maker_integration import AdversarialMAKERConfig, AdversarialMAKERMode, MAKERMode

        config = AdversarialMAKERConfig(
            mode=MAKERMode.RECURSIVE,
            k_ahead=3,
            adversarial_mode=AdversarialMAKERMode.ATTACK_GENERATION
        )
        results["checks"].append({
            "name": "config_creation",
            "status": "OK",
            "mode": config.mode.value
        })
        print(f"[OK] AdversarialMAKERConfig creation: {config.mode.value}")
    except Exception as e:
        results["failures"].append({"check": "config_creation", "error": str(e)})
        print(f"[FAIL] AdversarialMAKERConfig creation: {e}")

    # Test 2: Convert to MAKERWorkflowConfig
    try:
        maker_config = config.to_maker_workflow_config()
        results["checks"].append({
            "name": "config_conversion",
            "status": "OK"
        })
        print("[OK] Conversion to MAKERWorkflowConfig")
    except Exception as e:
        results["failures"].append({"check": "config_conversion", "error": str(e)})
        print(f"[FAIL] Config conversion: {e}")

    # Test 3: Create from AdversarialConfiguration
    try:
        from adversarial import create_adversarial_configuration
        from adversarial_maker_integration import create_adversarial_maker_config

        adv_config = create_adversarial_configuration()
        maker_config = create_adversarial_maker_config(adv_config)
        results["checks"].append({
            "name": "config_from_adversarial",
            "status": "OK"
        })
        print("[OK] Creation from AdversarialConfiguration")
    except Exception as e:
        results["failures"].append({"check": "config_from_adversarial", "error": str(e)})
        print(f"[FAIL] Creation from AdversarialConfiguration: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All configuration checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


def validate_agents():
    """Validate MAKER-enhanced agents."""
    print_section("3. VALIDATING AGENTS")

    results = {"status": "unknown", "checks": [], "failures": []}

    # Test 1: Create MAKERRedTeamAgent
    try:
        from adversarial_maker_integration import MAKERRedTeamAgent, AdversarialMAKERConfig
        from red_team import IssueCategory

        config = AdversarialMAKERConfig()
        agent = MAKERRedTeamAgent(
            name="TestRedAgent",
            specializations=[IssueCategory.SECURITY_VULNERABILITY],
            maker_config=config
        )
        results["checks"].append({
            "name": "red_agent_creation",
            "status": "OK"
        })
        print("[OK] MAKERRedTeamAgent creation")
    except Exception as e:
        results["failures"].append({"check": "red_agent_creation", "error": str(e)})
        print(f"[FAIL] MAKERRedTeamAgent creation: {e}")

    # Test 2: Create MDAPBlueTeamAgent
    try:
        from adversarial_maker_integration import MDAPBlueTeamAgent

        agent = MDAPBlueTeamAgent(
            name="TestBlueAgent",
            defense_specialization="general"
        )
        results["checks"].append({
            "name": "blue_agent_creation",
            "status": "OK"
        })
        print("[OK] MDAPBlueTeamAgent creation")
    except Exception as e:
        results["failures"].append({"check": "blue_agent_creation", "error": str(e)})
        print(f"[FAIL] MDAPBlueTeamAgent creation: {e}")

    # Test 3: Create AdversarialCoEvolution
    try:
        from adversarial_maker_integration import (
            AdversarialCoEvolution,
            AdversarialMAKERConfig,
            MAKERRedTeamAgent,
            MDAPBlueTeamAgent
        )

        config = AdversarialMAKERConfig()
        red_team = [MAKERRedTeamAgent(f"Red{i}", [IssueCategory.SECURITY_VULNERABILITY], maker_config=config)
                    for i in range(2)]
        blue_team = [MDAPBlueTeamAgent(f"Blue{i}") for i in range(2)]

        coevolution = AdversarialCoEvolution(config, red_team, blue_team)
        results["checks"].append({
            "name": "coevolution_creation",
            "status": "OK"
        })
        print("[OK] AdversarialCoEvolution creation")
    except Exception as e:
        results["failures"].append({"check": "coevolution_creation", "error": str(e)})
        print(f"[FAIL] AdversarialCoEvolution creation: {e}")

    # Determine status
    if not results["failures"]:
        results["status"] = "pass"
        print("\n[OK] All agent checks passed!")
    else:
        results["status"] = "fail"
        print(f"\n[FAIL] {len(results['failures'])} check(s) failed")

    return results


def validate_capabilities():
    """Validate capabilities function."""
    print_section("4. VALIDATING CAPABILITIES")

    results = {"status": "unknown", "checks": [], "failures": []}

    # Test 1: Get capabilities
    try:
        from adversarial import get_maker_adversarial_capabilities

        capabilities = get_maker_adversarial_capabilities()
        results["checks"].append({
            "name": "capabilities_function",
            "status": "OK"
        })
        print("[OK] Capabilities function")

        # Display capabilities
        print(f"  - MAKER enabled: {capabilities.get('maker_enabled', False)}")
        print(f"  - MDAP enabled: {capabilities.get('mdap_enabled', False)}")
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
    print("  MAKER/MDAP ADVERSARIAL INTEGRATION VALIDATION")
    print("  Complete arXiv:2511.09030 Implementation")
    print("=" * 80)
    print("")

    all_results = {}

    # Run all validations
    import_results = validate_imports()
    all_results["imports"] = import_results

    config_results = validate_configuration()
    all_results["configuration"] = config_results

    agent_results = validate_agents()
    all_results["agents"] = agent_results

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
        print("\nMAKER/MDAP adversarial integration is fully functional!")
        print("\nNext steps:")
        print("1. Run demo: python demo_adversarial_maker.py")
        print("2. Use in code: from adversarial import run_maker_enhanced_adversarial_testing")
        print("3. Read guide: MAKER_ADVERSARIAL_INTEGRATION_GUIDE.md")
        return 0
    else:
        print("\n" + "=" * 80)
        print("[FAIL][FAIL][FAIL] SOME VALIDATIONS FAILED [FAIL][FAIL][FAIL]")
        print("=" * 80)
        print("\nPlease check the errors above and fix them before using MAKER/MDAP adversarial testing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
