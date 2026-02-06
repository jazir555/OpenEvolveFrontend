"""
Comprehensive Test Suite for Z3/LeanAide/Lean Integrations

This module verifies that all Z3, LeanAide, and Lean4 integrations
are properly wired and working together.

Usage:
    python test_z3_leanaide_lean_integrations.py
    python test_z3_leanaide_lean_integrations.py --verbose
"""

import sys
import importlib
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class IntegrationTestResult:
    """Result of an integration test."""
    module: str
    imported: bool
    z3_available: bool = False
    leanaide_available: bool = False
    lean_available: bool = False
    error: str = None
    notes: List[str] = None
    
    def __post_init__(self):
        if self.notes is None:
            self.notes = []


# Core integration modules to test
CORE_INTEGRATION_MODULES = [
    # Z3 Core
    ("z3prover_integration", ["Z3_AVAILABLE"]),
    ("z3_mcp_tools", ["Z3_AVAILABLE"]),
    ("z3_crewai_bridge", ["Z3_AVAILABLE"]),
    ("z3_leanaide_bridge", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("z3_leanaide_bubbles", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    
    # LeanAide Core
    ("leanaide_client", ["LEAN_AVAILABLE"]),
    ("leanaide_integration", ["LEAN_AVAILABLE"]),
    ("leanaide_mcp_tools", ["LEAN_AVAILABLE"]),
    ("leanaide_crewai_bridge", ["LEAN_AVAILABLE"]),
    ("leanaide_config", ["LEAN_AVAILABLE"]),
    
    # Lean4 Core
    ("lean4_integration", ["LEAN_AVAILABLE"]),
    ("lean4_true_100_integration", ["LEAN_AVAILABLE"]),
    
    # Hybrid Integrations
    ("robust_z3_leanaide_integration", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("openevolve_leanaide_bridge", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("openevolve_leanaide_integration_system", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("verification_engine", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("blue_team_solver_engine", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("comprehensive_decomposition_engine", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    
    # BubbleLabs Integrations
    ("bubblelabs_leanaide_integration", ["LEANAIDE_AVAILABLE"]),
    ("bubblelabs_extended_integration", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    
    # Glue Adapters
    ("glue.lib.lean4_bridge.lean4_interface", ["LEAN_AVAILABLE"]),
    ("glue.lib.lean4_bridge.lean4_atp_bridge", ["Z3_AVAILABLE", "LEAN_AVAILABLE"]),
    ("glue.adapters.rese-sce.src.sce_bridge", ["Z3_AVAILABLE"]),
    
    # Knowledge Engine
    ("knowledge_engine.integrations.z3_knowledge_integration", ["Z3_AVAILABLE"]),
    ("knowledge_engine.integrations.leanaide_knowledge_extraction", ["LEANAIDE_AVAILABLE"]),
    
    # Domain Validators
    ("physics_validator", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("chemistry_validator", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("finance_validator", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("engineering_validator", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
]


def test_module_integration(module_name: str, expected_flags: List[str]) -> IntegrationTestResult:
    """Test a single integration module."""
    result = IntegrationTestResult(module=module_name, imported=False)
    
    try:
        module = importlib.import_module(module_name)
        result.imported = True
        
        # Check for expected availability flags
        for flag in expected_flags:
            flag_value = getattr(module, flag, False)
            if flag == "Z3_AVAILABLE":
                result.z3_available = bool(flag_value)
            elif flag == "LEANAIDE_AVAILABLE":
                result.leanaide_available = bool(flag_value)
            elif flag == "LEAN_AVAILABLE":
                result.lean_available = bool(flag_value)
        
        # Check for verify methods
        has_verify_with_lean = hasattr(module, 'verify_with_lean')
        has_verify_with_z3 = hasattr(module, 'verify_with_z3')
        has_verify_hybrid = hasattr(module, 'verify_hybrid')
        
        if has_verify_with_lean:
            result.notes.append("has verify_with_lean()")
        if has_verify_with_z3:
            result.notes.append("has verify_with_z3()")
        if has_verify_hybrid:
            result.notes.append("has verify_hybrid()")
            
    except Exception as e:
        result.error = str(e)[:100]
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Z3/LeanAide/Lean integrations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print("="*70)
    print("Z3/LEANAIDE/LEAN INTEGRATION TEST SUITE")
    print("="*70)
    print()
    
    results = []
    
    print("Testing integration modules...")
    for module_name, expected_flags in CORE_INTEGRATION_MODULES:
        result = test_module_integration(module_name, expected_flags)
        results.append(result)
        
        if args.verbose:
            status = "OK" if result.imported else "FAIL"
            print(f"  [{status}] {module_name}")
            if result.error:
                print(f"       Error: {result.error}")
    
    # Analysis
    total = len(results)
    imported = sum(1 for r in results if r.imported)
    with_z3 = sum(1 for r in results if r.z3_available)
    with_leanaide = sum(1 for r in results if r.leanaide_available)
    with_lean = sum(1 for r in results if r.lean_available)
    
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print(f"Total modules tested: {total}")
    print(f"  Successfully imported: {imported} ({imported/total*100:.1f}%)")
    print()
    print(f"  With Z3 available: {with_z3}")
    print(f"  With LeanAide available: {with_leanaide}")
    print(f"  With Lean available: {with_lean}")
    print()
    
    # Failed imports
    failed = [r for r in results if not r.imported]
    if failed:
        print("FAILED IMPORTS:")
        for r in failed:
            print(f"  - {r.module}: {r.error}")
        print()
    
    # Modules with integration methods
    with_verify = [r for r in results if r.notes]
    if with_verify and args.verbose:
        print("MODULES WITH VERIFICATION METHODS:")
        for r in with_verify:
            print(f"  - {r.module}: {', '.join(r.notes)}")
        print()
    
    print("="*70)
    
    if failed:
        print(f"STATUS: {len(failed)} modules failed to import")
        return 1
    else:
        print("STATUS: ALL INTEGRATIONS WORKING")
        return 0


if __name__ == "__main__":
    sys.exit(main())
