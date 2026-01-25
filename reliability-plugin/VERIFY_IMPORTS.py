#!/usr/bin/env python3
"""
Import verification script for reliability-plugin.
Tests all imports from core projects and dependencies.

Run this script to verify that all components of the reliability plugin
can be imported successfully. This helps identify missing dependencies
or configuration issues.

Usage:
    python reliability-plugin/VERIFY_IMPORTS.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add paths
plugin_root = Path(__file__).parent.parent
sys.path.insert(0, str(plugin_root))
sys.path.insert(0, str(plugin_root / "ROMA" / "src"))
sys.path.insert(0, str(plugin_root / "reliability-plugin"))


def test_reliability_imports() -> Dict[str, Any]:
    """Test reliability core imports"""
    print("Testing Reliability Core Imports...")
    results = {"passed": [], "failed": [], "skipped": []}

    # Test LMQL adapter
    try:
        from reliability.lmql_adapter import (
            LMQLAdapter, get_default_adapter,
            Constraint, GenerationResult
        )
        results["passed"].append("LMQL adapter")
        print("  ✅ LMQL adapter")
    except ImportError as e:
        results["failed"].append(("LMQL adapter", str(e)))
        print(f"  ❌ LMQL adapter: {e}")

    # Test Guardrails adapter
    try:
        from reliability.guardrails_adapter import (
            GuardrailsAdapter, create_adapter,
            ValidationResult
        )
        results["passed"].append("Guardrails adapter")
        print("  ✅ Guardrails adapter")
    except ImportError as e:
        results["failed"].append(("Guardrails adapter", str(e)))
        print(f"  ❌ Guardrails adapter: {e}")

    # Test config
    try:
        from reliability.config import (
            get_config, ReliabilityConfig,
            check_layer_health
        )
        results["passed"].append("Config")
        print("  ✅ Config")
    except ImportError as e:
        results["failed"].append(("Config", str(e)))
        print(f"  ❌ Config: {e}")

    # Test unified bridge
    try:
        from reliability.unified_bridge import (
            UnifiedReliabilityBridge,
            generate, generate_with_retry
        )
        results["passed"].append("Unified bridge")
        print("  ✅ Unified bridge")
    except ImportError as e:
        results["failed"].append(("Unified bridge", str(e)))
        print(f"  ❌ Unified bridge: {e}")

    # Test validation layer
    try:
        from reliability.validation_layer import (
            ValidationLayer, validate_result,
            validate_constraints, validate_quality
        )
        results["passed"].append("Validation layer")
        print("  ✅ Validation layer")
    except ImportError as e:
        results["failed"].append(("Validation layer", str(e)))
        print(f"  ❌ Validation layer: {e}")

    return results


def test_roma_core_imports() -> Dict[str, Any]:
    """Test ROMA core imports"""
    print("\nTesting ROMA Core Imports...")
    results = {"passed": [], "failed": [], "skipped": []}

    # Test ROMA core
    try:
        from roma_dspy import (
            RecursiveSolver, solve,
            Atomizer, Planner, Executor,
            TaskNode, TaskDAG, SubTask
        )
        results["passed"].append("ROMA core modules")
        print("  ✅ ROMA core modules")
    except ImportError as e:
        results["failed"].append(("ROMA core", str(e)))
        print(f"  ❌ ROMA core: {e}")

    # Test ROMA types
    try:
        from roma_dspy.types import (
            TaskType, NodeType, TaskStatus,
            AgentType, PredictionStrategy
        )
        results["passed"].append("ROMA types")
        print("  ✅ ROMA types")
    except ImportError as e:
        results["failed"].append(("ROMA types", str(e)))
        print(f"  ❌ ROMA types: {e}")

    # Test ROMA configuration
    try:
        from roma_dspy.config import (
            RomaConfig, get_roma_config
        )
        results["passed"].append("ROMA config")
        print("  ✅ ROMA config")
    except ImportError as e:
        results["failed"].append(("ROMA config", str(e)))
        print(f"  ❌ ROMA config: {e}")

    return results


def test_mdap_core_imports() -> Dict[str, Any]:
    """Test MDAP core imports"""
    print("\nTesting MDAP Core Imports...")
    results = {"passed": [], "failed": [], "skipped": []}

    # Test MDAP engine
    try:
        from mdap_engine import (
            MDAPOrchestrator, MDAPConfig,
            RedFlagger, RedFlagRules,
            validate_schema, canonicalize_candidate
        )
        results["passed"].append("MDAP engine")
        print("  ✅ MDAP engine")
    except ImportError as e:
        results["failed"].append(("MDAP engine", str(e)))
        print(f"  ❌ MDAP engine: {e}")

    # Test MAKER engine
    try:
        from maker_engine import (
            MakerEngine, MakerConfig,
            MakerState, MakerRunResult
        )
        results["passed"].append("MAKER engine")
        print("  ✅ MAKER engine")
    except ImportError as e:
        results["failed"].append(("MAKER engine", str(e)))
        print(f"  ❌ MAKER engine: {e}")

    # Test ROMA-MDAP-MAKER integration
    try:
        from roma_mdap_maker_engine import (
            ROMAMDAPMakerEngine, ROMAMDAPMakerConfig,
            ROMARedFlagger, HierarchicalVotingStrategy
        )
        results["passed"].append("ROMA-MDAP-MAKER integration")
        print("  ✅ ROMA-MDAP-MAKER integration")
    except ImportError as e:
        results["failed"].append(("ROMA-MDAP-MAKER", str(e)))
        print(f"  ❌ ROMA-MDAP-MAKER: {e}")

    return results


def test_adapter_imports() -> Dict[str, Any]:
    """Test adapter imports"""
    print("\nTesting Adapter Imports...")
    results = {"passed": [], "failed": [], "skipped": []}

    # Test ROMA adapter
    try:
        from reliability_plugin.adapters.roma import (
            RomaReliabilityAdapter,
            solve_with_constraints,
            create_roma_adapter
        )
        results["passed"].append("ROMA adapter")
        print("  ✅ ROMA adapter")
    except ImportError as e:
        results["failed"].append(("ROMA adapter", str(e)))
        print(f"  ❌ ROMA adapter: {e}")

    # Test MDAP adapter
    try:
        from reliability_plugin.adapters.mdap import (
            MDAPReliabilityAdapter,
            solve_with_guardrails,
            verify_vote
        )
        results["passed"].append("MDAP adapter")
        print("  ✅ MDAP adapter")
    except ImportError as e:
        results["failed"].append(("MDAP adapter", str(e)))
        print(f"  ❌ MDAP adapter: {e}")

    # Test MAKER adapter
    try:
        from reliability_plugin.adapters.maker import (
            MakerReliabilityAdapter,
            invent_with_reliability,
            track_invention_progress
        )
        results["passed"].append("MAKER adapter")
        print("  ✅ MAKER adapter")
    except ImportError as e:
        results["failed"].append(("MAKER adapter", str(e)))
        print(f"  ❌ MAKER adapter: {e}")

    return results


def test_mcp_tool_imports() -> Dict[str, Any]:
    """Test MCP tool imports"""
    print("\nTesting MCP Tool Imports...")
    results = {"passed": [], "failed": [], "skipped": []}

    # Test LMQL MCP tools
    try:
        import lmql_mcp_tools
        results["passed"].append("LMQL MCP tools")
        print("  ✅ LMQL MCP tools")
    except ImportError as e:
        results["failed"].append(("LMQL MCP tools", str(e)))
        print(f"  ❌ LMQL MCP tools: {e}")

    # Test Guardrails MCP tools
    try:
        import guardrails_mcp_tools
        results["passed"].append("Guardrails MCP tools")
        print("  ✅ Guardrails MCP tools")
    except ImportError as e:
        results["failed"].append(("Guardrails MCP tools", str(e)))
        print(f"  ❌ Guardrails MCP tools: {e}")

    # Test reliability MCP tools
    try:
        import reliability_mcp_tools
        results["passed"].append("Reliability MCP tools")
        print("  ✅ Reliability MCP tools")
    except ImportError as e:
        results["failed"].append(("Reliability MCP tools", str(e)))
        print(f"  ❌ Reliability MCP tools: {e}")

    return results


def test_schema_imports() -> Dict[str, Any]:
    """Test schema imports"""
    print("\nTesting Schema Imports...")
    results = {"passed": [], "failed": [], "skipped": []}

    # Test canonical models
    try:
        from reliability_plugin.schemas.canonical_models import (
            BaseResult, ValidationResult,
            GenerationResult, LayerStatus,
            RomaDecompositionResult,
            MDAPSolveResult,
            Constraint, ConstraintType
        )
        results["passed"].append("Canonical schemas")
        print("  ✅ Canonical schemas")
    except ImportError as e:
        results["failed"].append(("Canonical schemas", str(e)))
        print(f"  ❌ Canonical schemas: {e}")

    return results


def test_integration_imports() -> Dict[str, Any]:
    """Test integration imports"""
    print("\nTesting Integration Imports...")
    results = {"passed": [], "failed": [], "skipped": []}

    # Test unified orchestrator
    try:
        from reliability_plugin.integrations.unified_orchestrator import (
            UnifiedOrchestrator,
            solve_decompose_invent,
            orchestrate_workflow
        )
        results["passed"].append("Unified orchestrator")
        print("  ✅ Unified orchestrator")
    except ImportError as e:
        results["failed"].append(("Unified orchestrator", str(e)))
        print(f"  ❌ Unified orchestrator: {e}")

    # Test reliability bridge
    try:
        from reliability_plugin.integrations.reliability_bridge import (
            ReliabilityBridge,
            bridge_all_systems
        )
        results["passed"].append("Reliability bridge")
        print("  ✅ Reliability bridge")
    except ImportError as e:
        results["failed"].append(("Reliability bridge", str(e)))
        print(f"  ❌ Reliability bridge: {e}")

    return results


def print_detailed_results(all_results: Dict[str, Dict[str, Any]]) -> None:
    """Print detailed test results"""
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)

    for category, results in all_results.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        print(f"  Passed: {len(results['passed'])}")
        print(f"  Failed: {len(results['failed'])}")
        print(f"  Skipped: {len(results['skipped'])}")

        if results['passed']:
            print("\n  ✅ Successful imports:")
            for item in results['passed']:
                print(f"     - {item}")

        if results['failed']:
            print("\n  ❌ Failed imports:")
            for name, error in results['failed']:
                print(f"     - {name}")
                print(f"       Error: {error}")


def print_summary(all_results: Dict[str, Dict[str, Any]]) -> int:
    """Print summary and return exit code"""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_passed = sum(len(r["passed"]) for r in all_results.values())
    total_failed = sum(len(r["failed"]) for r in all_results.values())
    total_skipped = sum(len(r["skipped"]) for r in all_results.values())
    total = total_passed + total_failed + total_skipped

    print(f"\nTotal Tests: {total}")
    print(f"  ✅ Passed: {total_passed}")
    print(f"  ❌ Failed: {total_failed}")
    print(f"  ⏭️  Skipped: {total_skipped}")

    # Print recommendations
    if total_failed > 0:
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        print("\nTo fix failed imports:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Ensure ROMA is properly configured: cd ROMA && pip install -e .")
        print("3. Check Python path includes project directories")
        print("4. Verify environment variables are set")
        print("5. Review error messages above for specific issues")

    # Print final status
    print("\n" + "=" * 70)
    if total_failed == 0:
        print("✅ ALL IMPORTS SUCCESSFUL!")
        print("The reliability plugin is ready to use.")
    else:
        print(f"⚠️  {total_failed} import(s) failed")
        print("Some features may be unavailable.")
    print("=" * 70)

    return 0 if total_failed == 0 else 1


def main():
    """Run all import tests"""
    print("=" * 70)
    print("RELIABILITY PLUGIN IMPORT VERIFICATION")
    print("=" * 70)
    print("\nThis script tests all imports required for the reliability plugin.")
    print("It helps identify missing dependencies or configuration issues.\n")

    all_results = {}

    # Run all tests
    try:
        all_results["reliability"] = test_reliability_imports()
        all_results["roma_core"] = test_roma_core_imports()
        all_results["mdap_core"] = test_mdap_core_imports()
        all_results["adapters"] = test_adapter_imports()
        all_results["mcp_tools"] = test_mcp_tool_imports()
        all_results["schemas"] = test_schema_imports()
        all_results["integrations"] = test_integration_imports()
    except Exception as e:
        print(f"\n❌ Fatal error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print detailed results
    print_detailed_results(all_results)

    # Print summary and exit
    return print_summary(all_results)


if __name__ == "__main__":
    sys.exit(main())
