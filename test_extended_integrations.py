"""
Extended Integration Test - Check additional integration files
"""

import sys
import importlib

# Extended list of integration files to test
EXTENDED_INTEGRATION_MODULES = [
    # Analytics & Monitoring
    ("analytics_z3_connector", ["Z3_AVAILABLE"]),
    ("automated_proof_engine", ["Z3_AVAILABLE", "LEAN_AVAILABLE"]),
    
    # Audit & Validation
    ("brutal_audit", ["Z3_AVAILABLE"]),
    ("check_wiring_complete", ["Z3_AVAILABLE", "LEAN_AVAILABLE"]),
    
    # BubbleLabs
    ("bubblelabs_integration", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("bubblelabs_node_completion", ["Z3_AVAILABLE"]),
    ("bubblelabs_ui_component", ["Z3_AVAILABLE"]),
    
    # Chronicle & Memory
    ("chronicle_memory_z3_integration", ["Z3_AVAILABLE"]),
    
    # CrewAI
    ("crewai_zero_error_workflow", ["Z3_AVAILABLE", "LEAN_AVAILABLE"]),
    ("decomposition_crewai_bridge", ["Z3_AVAILABLE"]),
    
    # Domain Validators
    ("physics_validator", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("chemistry_validator", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("finance_validator", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("engineering_validator", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    
    # Gauntlet
    ("formal_gauntlet_system", ["Z3_AVAILABLE", "LEAN_AVAILABLE"]),
    ("gauntlet_orchestrator", ["Z3_AVAILABLE", "LEAN_AVAILABLE"]),
    ("gauntlet_types", ["Z3_AVAILABLE"]),
    
    # Ground Truth
    ("ground_truth_store", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    
    # Hybrid
    ("hybrid_mcts_framework", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    
    # Knowledge
    ("knowledge_context_assembler", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("knowledge_graph_reasoning_integration", ["Z3_AVAILABLE"]),
    
    # OpenEvolve
    ("openevolve_imports", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("openevolve_validation", ["Z3_AVAILABLE"]),
    
    # Universal
    ("universal_problem_solver", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("universal_decomposition_engine", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("universal_recomposition_engine", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    
    # Validation & Verification
    ("validation_manager", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("verification_engine", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("verified_recomposition", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    
    # Workflow
    ("workflow_enhanced_stages", ["Z3_AVAILABLE", "LEAN_AVAILABLE"]),
    ("workflow_lifecycle_controller", ["Z3_AVAILABLE", "LEAN_AVAILABLE"]),
    ("integrated_workflow", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
    ("working_integration_bridge", ["Z3_AVAILABLE", "LEANAIDE_AVAILABLE"]),
]


def test_import(module_name, expected_flags):
    """Test importing a module."""
    try:
        module = importlib.import_module(module_name)
        
        # Check flags
        results = {}
        for flag in expected_flags:
            results[flag] = getattr(module, flag, False)
        
        return True, results, None
    except Exception as e:
        return False, {}, str(e)[:100]


def main():
    print("="*70)
    print("EXTENDED INTEGRATION TEST")
    print("="*70)
    print()
    
    total = len(EXTENDED_INTEGRATION_MODULES)
    success = 0
    failed = []
    
    print(f"Testing {total} extended integration modules...")
    print()
    
    for i, (module, flags) in enumerate(EXTENDED_INTEGRATION_MODULES, 1):
        ok, results, error = test_import(module, flags)
        
        if ok:
            success += 1
            status = "OK"
        else:
            failed.append((module, error))
            status = "FAIL"
        
        if i % 10 == 0:
            print(f"  Progress: {i}/{total}...")
    
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print(f"Total: {total}")
    print(f"  Success: {success} ({success/total*100:.1f}%)")
    print(f"  Failed: {len(failed)}")
    
    if failed:
        print()
        print("FAILED MODULES:")
        for module, error in failed:
            print(f"  - {module}: {error}")
    
    print()
    print("="*70)
    
    if failed:
        print(f"STATUS: {len(failed)} modules failed")
        return 1
    else:
        print("STATUS: ALL EXTENDED INTEGRATIONS WORKING")
        return 0


if __name__ == "__main__":
    sys.exit(main())
