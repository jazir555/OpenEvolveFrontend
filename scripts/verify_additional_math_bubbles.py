#!/usr/bin/env python3
"""
Verification script for additional Mathematical Verification Bubbles.
Tests that all 7 new math bubbles can be imported and initialized.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ADDITIONAL_MATH_BUBBLES = [
    ("math_problem_classification_node", "MathProblemClassificationNode"),
    ("math_tactic_recommendation_node", "MathTacticRecommendationNode"),
    ("math_library_search_node", "MathLibrarySearchNode"),
    ("math_proof_simplification_node", "MathProofSimplificationNode"),
    ("math_counterexample_node", "MathCounterexampleNode"),
    ("math_induction_helper_node", "MathInductionHelperNode"),
    ("math_equivalence_node", "MathEquivalenceNode"),
    ("math_conjecture_node", "MathConjectureNode"),
    ("math_proof_completion_node", "MathProofCompletionNode"),
]

def verify_bubble(module_name: str, class_name: str) -> bool:
    """Verify a single bubble can be imported and initialized."""
    try:
        module = __import__(f"bubblelabs_nodes.{module_name}", fromlist=[class_name])
        cls = getattr(module, class_name)
        instance = cls(config={})
        
        assert hasattr(cls, 'DISPLAY_NAME'), f"{class_name} missing DISPLAY_NAME"
        assert hasattr(cls, 'DESCRIPTION'), f"{class_name} missing DESCRIPTION"
        assert hasattr(cls, 'CATEGORY'), f"{class_name} missing CATEGORY"
        assert hasattr(cls, 'VERSION'), f"{class_name} missing VERSION"
        
        assert hasattr(instance, 'execute'), f"{class_name} missing execute method"
        assert hasattr(instance, 'validate_inputs'), f"{class_name} missing validate_inputs method"
        assert hasattr(instance, 'get_parameter_schema'), f"{class_name} missing get_parameter_schema method"
        assert hasattr(instance, 'is_healthy'), f"{class_name} missing is_healthy method"
        
        assert instance.is_healthy(), f"{class_name} is not healthy"
        assert instance.CATEGORY == "mathematical_verification", f"{class_name} has wrong category"
        
        print(f"  [OK] {class_name}: {cls.DISPLAY_NAME}")
        return True
        
    except Exception as e:
        print(f"  [FAIL] {class_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("Additional Mathematical Verification Bubble Suite - Verification")
    print("=" * 70)
    print()
    
    success_count = 0
    total_size = 0
    
    for module_name, class_name in ADDITIONAL_MATH_BUBBLES:
        if verify_bubble(module_name, class_name):
            success_count += 1
            file_path = Path(__file__).parent / "bubblelabs_nodes" / f"{module_name}.py"
            if file_path.exists():
                total_size += file_path.stat().st_size
    
    print()
    print("=" * 70)
    print(f"Results: {success_count}/{len(ADDITIONAL_MATH_BUBBLES)} bubbles verified successfully")
    print(f"Total Code: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    print("=" * 70)
    
    if success_count == len(ADDITIONAL_MATH_BUBBLES):
        print("\n[SUCCESS] All additional mathematical verification bubbles are ready!")
        return 0
    else:
        print(f"\n[WARNING] {len(ADDITIONAL_MATH_BUBBLES) - success_count} bubble(s) failed verification")
        return 1

if __name__ == "__main__":
    sys.exit(main())
