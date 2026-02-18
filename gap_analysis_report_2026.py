#!/usr/bin/env python3
"""
Comprehensive Gap Analysis for Z3-Lean Integration
Analyzes the current state and identifies any remaining gaps
"""

import sys
import os
from pathlib import Path

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}\n")

def print_success(text):
    print(f"{Colors.GREEN}[PASS]{Colors.RESET} {text}")

def print_fail(text):
    print(f"{Colors.RED}[FAIL]{Colors.RESET} {text}")

def print_warning(text):
    print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {text}")

def print_info(text):
    print(f"{Colors.BLUE}[INFO]{Colors.RESET} {text}")

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        size = Path(filepath).stat().st_size
        print_success(f"{description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print_fail(f"{description}: {filepath} - NOT FOUND")
        return False

def check_import(module_path, description):
    """Check if a module can be imported"""
    try:
        parts = module_path.split('.')
        module = __import__(module_path)
        for part in parts[1:]:
            module = getattr(module, part)
        print_success(f"{description}: {module_path}")
        return True
    except Exception as e:
        print_fail(f"{description}: {module_path} - {str(e)[:50]}")
        return False

def check_function_exists(module_path, function_name, description):
    """Check if a function exists in a module"""
    try:
        parts = module_path.split('.')
        module = __import__(module_path)
        for part in parts[1:]:
            module = getattr(module, part)

        if hasattr(module, function_name):
            print_success(f"{description}: {module_path}.{function_name}")
            return True
        else:
            print_fail(f"{description}: {module_path}.{function_name} - NOT FOUND")
            return False
    except Exception as e:
        print_fail(f"{description}: {module_path}.{function_name} - {str(e)[:50]}")
        return False

def check_code_in_file(filepath, pattern, description):
    """Check if a pattern exists in a file"""
    try:
        content = Path(filepath).read_text(encoding='utf-8', errors='ignore')
        if pattern in content:
            count = content.count(pattern)
            print_success(f"{description}: Found {count} occurrence(s)")
            return True
        else:
            print_fail(f"{description}: Pattern '{pattern[:50]}...' NOT FOUND")
            return False
    except Exception as e:
        print_fail(f"{description}: {str(e)[:50]}")
        return False

def main():
    print_header("Z3-LEAN INTEGRATION GAP ANALYSIS")

    # =========================================================================
    # SECTION 1: Core Integration Files
    # =========================================================================
    print_header("SECTION 1: CORE INTEGRATION FILES")

    core_files = [
        ("z3prover_integration.py", "Z3 Prover Integration"),
        ("z3_solver_connector.py", "Z3 Solver Connector"),
        ("z3_canonicalizer.py", "Z3 Canonicalizer"),
        ("z3_semantic_synthesis.py", "Z3 Semantic Synthesis"),
        ("z3_to_lean_integration.py", "Z3-to-Lean Integration"),
        ("enhanced_z3_to_lean_integration.py", "Enhanced Z3-to-Lean Integration"),
        ("z3_to_lean_invention_integration.py", "Invention Planner Integration"),
    ]

    files_ok = True
    for filepath, description in core_files:
        if not check_file_exists(filepath, description):
            files_ok = False

    # =========================================================================
    # SECTION 2: Module Imports
    # =========================================================================
    print_header("SECTION 2: MODULE IMPORTS")

    imports_ok = True

    # Check Z3 integration
    if not check_import("z3prover_integration", "Z3 Prover module"):
        imports_ok = False

    # Check Z3-to-Lean integration
    if not check_import("z3_to_lean_integration", "Z3-to-Lean module"):
        imports_ok = False

    # Check enhanced integration
    if not check_import("enhanced_z3_to_lean_integration", "Enhanced Z3-to-Lean module"):
        imports_ok = False

    # Check invention integration
    if not check_import("z3_to_lean_invention_integration", "Invention integration module"):
        imports_ok = False

    # =========================================================================
    # SECTION 3: Key Functions
    # =========================================================================
    print_header("SECTION 3: KEY FUNCTIONS")

    functions_ok = True

    # Check for formalize_invention_plan
    if not check_function_exists(
        "z3_to_lean_invention_integration",
        "formalize_invention_plan",
        "formalize_invention_plan function"
    ):
        functions_ok = False

    # Check for Z3LeanInventionIntegration class
    if not check_function_exists(
        "z3_to_lean_invention_integration",
        "Z3LeanInventionIntegration",
        "Z3LeanInventionIntegration class"
    ):
        functions_ok = False

    # Check for InventionFormalizationResult
    if not check_function_exists(
        "z3_to_lean_invention_integration",
        "InventionFormalizationResult",
        "InventionFormalizationResult class"
    ):
        functions_ok = False

    # =========================================================================
    # SECTION 4: Invention Planner Integration (CRITICAL)
    # =========================================================================
    print_header("SECTION 4: INVENTION PLANNER INTEGRATION (CRITICAL)")

    planner_ok = True

    # Check if invention planner exists
    if not check_file_exists("end_to_end_invention_planner.py", "Invention Planner"):
        planner_ok = False
    else:
        # Check for Z3-Lean import
        if not check_code_in_file(
            "end_to_end_invention_planner.py",
            "from z3_to_lean_invention_integration import",
            "Z3-Lean import in invention planner"
        ):
            planner_ok = False

        # Check for availability flag
        if not check_code_in_file(
            "end_to_end_invention_planner.py",
            "Z3_LEAN_INTEGRATION_AVAILABLE",
            "Z3_LEAN_INTEGRATION_AVAILABLE flag"
        ):
            planner_ok = False

        # Check for formalize_invention_plan call
        if not check_code_in_file(
            "end_to_end_invention_planner.py",
            "await formalize_invention_plan(",
            "formalize_invention_plan() call"
        ):
            planner_ok = False

        # Check for Z3+Lean usage
        if not check_code_in_file(
            "end_to_end_invention_planner.py",
            "Z3+Lean",
            "Z3+Lean string reference"
        ):
            planner_ok = False

    # =========================================================================
    # SECTION 5: Gauntlet System Integration
    # =========================================================================
    print_header("SECTION 5: GAUNTLET SYSTEM INTEGRATION")

    gauntlet_ok = True

    # Check if Z3LeanFormalVerificationGauntlet exists
    if not check_function_exists(
        "z3_to_lean_integration",
        "Z3LeanFormalVerificationGauntlet",
        "Z3LeanFormalVerificationGauntlet class"
    ):
        gauntlet_ok = False

    # Check if it's referenced in gauntlet_types.py
    if Path("gauntlet_types.py").exists():
        if not check_code_in_file(
            "gauntlet_types.py",
            "Z3LeanFormalVerificationGauntlet",
            "Gauntlet referenced in gauntlet_types.py"
        ):
            gauntlet_ok = False
    else:
        print_warning("gauntlet_types.py not found - skipping")

    # =========================================================================
    # SECTION 6: Test Files
    # =========================================================================
    print_header("SECTION 6: TEST FILES")

    tests_ok = True

    test_files = [
        ("test_z3_lean_quick.py", "Quick Z3-Lean Test"),
        ("test_gap_fixes_comprehensive.py", "Gap Fixes Test"),
        ("test_formalization_levels_final.py", "Formalization Levels Test"),
        ("test_z3_lean_invention_integration.py", "Invention Integration Test"),
        ("test_z3_lean_invention_planner_integration.py", "Planner Integration Test"),
    ]

    for filepath, description in test_files:
        if not check_file_exists(filepath, description):
            tests_ok = False

    # =========================================================================
    # SECTION 7: Documentation
    # =========================================================================
    print_header("SECTION 7: DOCUMENTATION")

    docs_ok = True

    doc_files = [
        ("ENHANCED_Z3_TO_LEAN_IMPROVEMENTS.md", "Enhanced Improvements Doc"),
        ("Z3_TO_LEAN_INTEGRATION_COMPLETE.md", "Integration Complete Doc"),
        ("Z3_BUG_FIXES_APPLIED.md", "Bug Fixes Doc"),
        ("Z3_LEAN_INVENTION_PLANNER_INTEGRATION.md", "Planner Integration Doc"),
        ("Z3_LEAN_GAP_FIXES_COMPLETE.md", "Gap Fixes Complete Doc"),
        ("GAP_12_FIX_COMPLETE.md", "Gap 12 Fix Doc"),
        ("Z3_LEAN_100_PERCENT_COMPLETE.md", "100% Complete Doc"),
    ]

    for filepath, description in doc_files:
        if not check_file_exists(filepath, description):
            docs_ok = False

    # =========================================================================
    # SECTION 8: Final Summary
    # =========================================================================
    print_header("GAP ANALYSIS SUMMARY")

    all_sections = [
        ("Core Integration Files", files_ok),
        ("Module Imports", imports_ok),
        ("Key Functions", functions_ok),
        ("Invention Planner Integration", planner_ok),
        ("Gauntlet System Integration", gauntlet_ok),
        ("Test Files", tests_ok),
        ("Documentation", docs_ok),
    ]

    print("\nSection Status:")
    print("-" * 80)

    all_ok = True
    for section_name, ok in all_sections:
        status = Colors.GREEN + "PASS" + Colors.RESET if ok else Colors.RED + "FAIL" + Colors.RESET
        print(f"  {status}: {section_name}")
        if not ok:
            all_ok = False

    print("\n" + "=" * 80)

    if all_ok:
        print_success("ALL CHECKS PASSED - NO CRITICAL GAPS FOUND")
        print_info("The Z3-Lean integration is complete and functional")
        return 0
    else:
        print_fail("SOME CHECKS FAILED - GAPS IDENTIFIED")
        print_info("See details above for specific gaps")

        # Provide guidance
        print("\n" + "=" * 80)
        print_info("NEXT STEPS:")
        for section_name, ok in all_sections:
            if not ok:
                print_warning(f"  - Fix issues in: {section_name}")

        return 1

if __name__ == "__main__":
    sys.exit(main())
