"""
Test script to verify all import fixes for Phase 1 (100% import success rate).

This script tests all the stub classes and functions that were added to fix import errors.
"""

import sys
import importlib
from typing import List, Tuple


def test_import(module_name: str, item_name: str = None) -> Tuple[bool, str]:
    """
    Test importing a module or specific item from a module.
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        module = importlib.import_module(module_name)
        if item_name:
            if not hasattr(module, item_name):
                return False, f"Module {module_name} does not have attribute {item_name}"
            item = getattr(module, item_name)
            if item is None:
                return False, f"{item_name} from {module_name} is None"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def main():
    """Run all import tests."""
    print("=" * 70)
    print("IMPORT FIXES VERIFICATION - PHASE 1")
    print("=" * 70)
    print()
    
    # List of all fixes to test
    fixes = [
        # Stub modules
        ("lean_type_theory", None),
        ("compositional_meta_rules", None),
        ("flexible_semantic_parsing", None),
        
        # Stub classes in existing modules
        ("crewai_zero_error_workflow", "CrewAIZeroErrorWorkflow"),
        ("roma_config", "CrewAIROMAConfig"),
        ("input_validation", "InputValidator"),
        ("sovereign_data_models", "WorkflowState"),
        ("sovereign_data_models", "ResourceEstimate"),
        ("sovereign_data_models", "SubProblemTeamAssignment"),
        ("decomposition_recomposition_integration", "DecompositionRecompositionPipeline"),
        ("bubblelabs_crewai_bridge", "BubbleLabsCREWAIBridge"),
        ("bubblelabs_crewai_bridge", "BubbleLabsCrewAIBridge"),
        ("decomposition_engine", "calculate_functional_weight"),
        ("decomposition_engine", "FlowBasedDecomposition"),
        ("decomposition_engine", "HierarchicalDecomposition"),
        ("leanaide_pes_handler", "enhance_lean_proof"),
        ("leanaide_autoformalization_mdap_maker", "LeanAideAutoformalizationEngine"),
        ("bubblelabs_analytics", "cleanup_all_databases"),
        ("reliability_config", "HEALTH_CHECK_CONFIG"),
        ("problem_decomposition", "get_recommended_strategy"),
        ("problem_decomposition", "get_roma_integration_status"),
        ("lean4_integration", "create_lean4_verification_engine"),
        ("associative_recomposition", "SolutionType"),
        ("openevolve_pes_integration", "PythonHandler"),
        
        # BubbleLabs nodes
        ("bubblelabs_nodes", "CircuitBreakerState"),
        ("bubblelabs_nodes", "CircuitBreakerStrategy"),
        ("bubblelabs_nodes", "FuzzInputGenerator"),
        ("bubblelabs_nodes", "ChangeTracker"),
        ("bubblelabs_nodes", "create_config"),
        ("bubblelabs_nodes.circuit_breakers", "CircuitBreakerStrategy"),
        
        # OpenEvolve unified package
        ("openevolve.unified", None),
        ("openevolve.unified", "UnifiedEvolutionAPI"),
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for module_name, item_name in fixes:
        success, message = test_import(module_name, item_name)
        
        if item_name:
            display_name = f"{module_name}.{item_name}"
        else:
            display_name = module_name
        
        if success:
            print(f"[OK] {display_name}")
            passed += 1
        else:
            print(f"[FAIL] {display_name}: {message}")
            failed += 1
            errors.append((display_name, message))
    
    print()
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(fixes)} total")
    print("=" * 70)
    
    if errors:
        print("\nFAILED IMPORTS:")
        for name, error in errors:
            print(f"  - {name}: {error}")
    
    # Return exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
