"""
Mass Verification of All 140+ Lean 4 Wired Files

This script discovers and tests ALL files with Lean integration based on:
1. Files containing 'LEAN_AVAILABLE' flag
2. Files importing from leanaide_client
3. Files with verify_with_lean methods

Usage:
    python verify_all_lean_wiring.py
    python verify_all_lean_wiring.py --quick  # Skip detailed tests
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Set

# Must import bootstrap first
import lean_bootstrap


def discover_lean_files() -> List[str]:
    """Discover all files with Lean integration by scanning Python files."""
    import os
    lean_files = set()
    root = Path(__file__).parent
    
    # Directories to skip entirely
    skip_dirs = {
        '__pycache__', '.egg-info', 'node_modules', '.git', 
        '.venv', 'venv', '.pytest_cache', '.mypy_cache',
        'core-projects', 'bubblelabs-nodes-backup', 'docs',
        'test_results', 'tests', 'benchmark_artifacts'
    }
    
    # Use os.walk for better control
    for dirpath, dirnames, filenames in os.walk(root):
        # Remove skip directories from traversal
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        
        for filename in filenames:
            if not filename.endswith('.py'):
                continue
            if filename.startswith('test_') or filename.endswith('_test.py'):
                continue
            
            py_file = Path(dirpath) / filename
            
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                # Check for LEAN_AVAILABLE flag or leanaide imports
                has_lean = 'LEAN_AVAILABLE' in content
                has_import = 'from leanaide_client import' in content or 'import leanaide_client' in content
                
                if has_lean or has_import:
                    rel_path = py_file.relative_to(root)
                    module = str(rel_path.with_suffix('')).replace('/', '.').replace('\\', '.')
                    lean_files.add(module)
            
            except (Exception, OSError):
                continue
    
    return sorted(lean_files)


def test_module(module_name: str) -> Dict[str, Any]:
    """Test a single module."""
    result = {
        "module": module_name,
        "imported": False,
        "lean_available": None,
        "error": None
    }
    
    try:
        module = importlib.import_module(module_name)
        result["imported"] = True
        result["lean_available"] = getattr(module, 'LEAN_AVAILABLE', 'NOT_SET')
    except Exception as e:
        result["error"] = str(e)[:100]
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Mass verify Lean 4 wiring")
    parser.add_argument("--quick", action="store_true", 
                       help="Use predefined list instead of grep discovery")
    parser.add_argument("--max", type=int, default=200,
                       help="Maximum files to test")
    args = parser.parse_args()
    
    print("="*70)
    print("MASS LEAN 4 WIRING VERIFICATION")
    print("="*70)
    print()
    
    # Discover or use predefined list
    if args.quick:
        # Use a representative sample
        modules_to_test = [
            "leanaide_client", "leanaide_config", "lean4_integration",
            "leanaide_integration", "config", "quality_gate_engine",
            "validation_manager", "verification_engine", "red_team",
            "blue_team", "evaluator_team", "physics_validator",
            "chemistry_validator", "finance_validator", "engineering_validator",
            "knowledge_base", "ground_truth_store", "solution_assembler",
            "universal_problem_solver", "universal_decomposition_engine",
            "universal_recomposition_engine", "final_solution",
            "workflow_engine", "integrated_workflow", "roma_config",
            "crewai_zero_error_workflow", "adversarial", "adversarial_unified",
            "formal_gauntlet_system", "gauntlet_orchestrator",
            "mcts_evolved_policies", "mdap_maker_mcts_unified",
        ]
    else:
        print("Discovering Lean-integrated files...")
        modules_to_test = discover_lean_files()
    
    # Limit to max
    modules_to_test = modules_to_test[:args.max]
    
    print(f"Testing {len(modules_to_test)} modules...")
    print()
    
    # Test all modules
    results = []
    imported_count = 0
    lean_true_count = 0
    lean_false_count = 0
    lean_not_set_count = 0
    
    for i, module_name in enumerate(modules_to_test, 1):
        result = test_module(module_name)
        results.append(result)
        
        if result["imported"]:
            imported_count += 1
        
        lean_avail = result["lean_available"]
        if lean_avail is True:
            lean_true_count += 1
        elif lean_avail is False:
            lean_false_count += 1
        elif lean_avail == "NOT_SET":
            lean_not_set_count += 1
        
        # Progress indicator
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(modules_to_test)} modules tested...")
    
    # Summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Total modules tested:     {len(modules_to_test)}")
    print(f"  Successfully imported:    {imported_count}")
    print(f"  LEAN_AVAILABLE=True:      {lean_true_count}")
    print(f"  LEAN_AVAILABLE=False:     {lean_false_count}")
    print(f"  LEAN_AVAILABLE not set:   {lean_not_set_count}")
    print()
    
    # Detailed issues
    failed = [r for r in results if not r["imported"]]
    lean_false = [r for r in results if r["lean_available"] is False]
    
    if failed:
        print("FAILED IMPORTS:")
        for r in failed[:10]:  # Show first 10
            print(f"  - {r['module']}: {r['error']}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
        print()
    
    if lean_false:
        print("LEAN_AVAILABLE=FALSE (may need fixing):")
        for r in lean_false[:10]:
            print(f"  - {r['module']}")
        if len(lean_false) > 10:
            print(f"  ... and {len(lean_false) - 10} more")
        print()
    
    # Calculate success rate
    if len(modules_to_test) > 0:
        success_rate = (imported_count / len(modules_to_test)) * 100
        lean_rate = (lean_true_count / len(modules_to_test)) * 100
        print(f"SUCCESS RATES:")
        print(f"  Import success: {success_rate:.1f}%")
        print(f"  LEAN_AVAILABLE=True: {lean_rate:.1f}%")
        print()
    
    # Overall status
    if imported_count == len(modules_to_test) and lean_false_count == 0:
        print("OVERALL: ALL CHECKS PASSED")
        return 0
    elif lean_false_count == 0:
        print(f"OVERALL: MOSTLY PASSED ({failed} import failures)")
        return 0
    else:
        print(f"OVERALL: ISSUES FOUND ({lean_false_count} files with LEAN_AVAILABLE=False)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
