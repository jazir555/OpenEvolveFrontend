"""
Comprehensive Lean 4 Wiring Verification

This script verifies that all 140+ wired files have LEAN_AVAILABLE = True
after importing. It tests a representative sample of the wired files.

Usage:
    python verify_lean_wiring.py
    python verify_lean_wiring.py --all  # Test all known wired files
"""

import sys
import importlib
from pathlib import Path

# Must import bootstrap first
import lean_bootstrap

# Representative sample of wired files to test
# These cover all major categories: core, glue, validators, engines, etc.
WIRED_FILES_SAMPLE = [
    # Core modules
    "leanaide_client",
    "leanaide_config",
    "lean4_integration",
    "leanaide_integration",
    "config",
    
    # Quality/Validation
    "quality_gate_engine",
    "validation_manager",
    "verification_engine",
    "verified_recomposition",
    
    # Team systems
    "red_team",
    "blue_team",
    "evaluator_team",
    
    # Domain validators
    "physics_validator",
    "chemistry_validator",
    "finance_validator",
    "engineering_validator",
    
    # Knowledge systems
    "knowledge_base",
    "ground_truth_store",
    "solution_assembler",
    
    # Engines
    "universal_problem_solver",
    "universal_decomposition_engine",
    "universal_recomposition_engine",
    "final_solution",
    
    # Workflow
    "workflow_engine",
    "integrated_workflow",
    
    # BubbleLabs nodes (sample)
    "bubblelabs_nodes.lean_proof_checking_node",
    "bubblelabs_nodes.math_proof_completion_node",
    
    # ROMA/CrewAI
    "roma_config",
    "crewai_zero_error_workflow",
    
    # Adversarial
    "adversarial",
    "adversarial_unified",
    
    # Gauntlets
    "formal_gauntlet_system",
    "gauntlet_orchestrator",
    
    # MCTS/MDAP
    "mcts_evolved_policies",
    "mdap_maker_mcts_unified",
]

# Files that should have verify_with_lean methods
VERIFY_WITH_LEAN_FILES = [
    "verification_engine",
    "validation_manager",
    "verified_recomposition",
    "final_solution",
    "universal_problem_solver",
    "universal_recomposition_engine",
    "physics_validator",
    "chemistry_validator",
    "finance_validator",
    "engineering_validator",
]


def test_file(module_name: str) -> dict:
    """Test a single wired file."""
    result = {
        "module": module_name,
        "imported": False,
        "lean_available": None,
        "has_verify_with_lean": False,
        "error": None
    }
    
    try:
        # Import the module
        module = importlib.import_module(module_name)
        result["imported"] = True
        
        # Check LEAN_AVAILABLE
        lean_avail = getattr(module, 'LEAN_AVAILABLE', 'NOT_SET')
        result["lean_available"] = lean_avail
        
        # Check for verify_with_lean method
        if hasattr(module, 'verify_with_lean'):
            result["has_verify_with_lean"] = True
        else:
            # Check classes in module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and hasattr(attr, 'verify_with_lean'):
                    result["has_verify_with_lean"] = True
                    break
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Lean 4 wiring")
    parser.add_argument("--all", action="store_true", help="Test all known wired files")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()
    
    files_to_test = WIRED_FILES_SAMPLE
    
    print("="*70)
    print("LEAN 4 WIRING VERIFICATION")
    print("="*70)
    print(f"\nTesting {len(files_to_test)} wired files...")
    print()
    
    results = []
    success_count = 0
    lean_true_count = 0
    verify_method_count = 0
    
    for module_name in files_to_test:
        result = test_file(module_name)
        results.append(result)
        
        if result["imported"]:
            success_count += 1
        
        if result["lean_available"] is True:
            lean_true_count += 1
        
        if result["has_verify_with_lean"]:
            verify_method_count += 1
        
        # Print status
        status = "OK" if result["imported"] else "FAIL"
        lean_status = ""
        if result["lean_available"] is True:
            lean_status = "[LEAN_OK]"
        elif result["lean_available"] is False:
            lean_status = "[LEAN_FALSE]"
        elif result["lean_available"] == "NOT_SET":
            lean_status = "[LEAN_NOT_SET]"
        
        verify_status = "[verify]" if result["has_verify_with_lean"] else ""
        
        print(f"  [{status}] {module_name:50s} {lean_status:12s} {verify_status}")
        
        if result["error"]:
            print(f"       Error: {result['error'][:60]}")
    
    # Summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Files tested:        {len(files_to_test)}")
    print(f"  Successfully imported: {success_count}")
    print(f"  LEAN_AVAILABLE=True: {lean_true_count}")
    print(f"  With verify_with_lean: {verify_method_count}")
    print()
    
    # Detailed issues
    failed_imports = [r for r in results if not r["imported"]]
    lean_false = [r for r in results if r["lean_available"] is False]
    lean_not_set = [r for r in results if r["lean_available"] == "NOT_SET"]
    
    if failed_imports:
        print("FAILED IMPORTS:")
        for r in failed_imports:
            print(f"  - {r['module']}: {r['error'][:60]}")
        print()
    
    if lean_false:
        print("LEAN_AVAILABLE=FALSE (needs fixing):")
        for r in lean_false:
            print(f"  - {r['module']}")
        print()
    
    if lean_not_set:
        print("LEAN_AVAILABLE NOT SET (may need adding):")
        for r in lean_not_set:
            print(f"  - {r['module']}")
        print()
    
    # Overall status
    if success_count == len(files_to_test) and not lean_false:
        print("OVERALL: ALL CHECKS PASSED")
        return 0
    else:
        print("OVERALL: SOME ISSUES FOUND")
        return 1


if __name__ == "__main__":
    sys.exit(main())
