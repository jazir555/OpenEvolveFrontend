#!/usr/bin/env python3
"""
Verification script for Mathematical Verification Bubble Suite.
Tests that all 8 math verification bubbles can be imported and initialized.
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

MATH_BUBBLES = [
    ("lean_autoformalization_node", "LeanAutoformalizationNode"),
    ("lean_proof_checking_node", "LeanProofCheckingNode"),
    ("z3_constraint_solving_node", "Z3ConstraintSolvingNode"),
    ("z3_theorem_proving_node", "Z3TheoremProvingNode"),
    ("math_verification_pipeline_node", "MathVerificationPipelineNode"),
    ("math_knowledge_extraction_node", "MathKnowledgeExtractionNode"),
    ("proof_translation_node", "ProofTranslationNode"),
    ("math_verification_dashboard_node", "MathVerificationDashboardNode"),
]

def verify_bubble(module_name: str, class_name: str) -> bool:
    """Verify a single bubble can be imported and initialized."""
    try:
        # Import the module
        module = __import__(f"bubblelabs_nodes.{module_name}", fromlist=[class_name])
        
        # Get the class
        cls = getattr(module, class_name)
        
        # Initialize with default config
        instance = cls(config={})
        
        # Check required attributes
        assert hasattr(cls, 'DISPLAY_NAME'), f"{class_name} missing DISPLAY_NAME"
        assert hasattr(cls, 'DESCRIPTION'), f"{class_name} missing DESCRIPTION"
        assert hasattr(cls, 'CATEGORY'), f"{class_name} missing CATEGORY"
        assert hasattr(cls, 'VERSION'), f"{class_name} missing VERSION"
        
        # Check methods
        assert hasattr(instance, 'execute'), f"{class_name} missing execute method"
        assert hasattr(instance, 'validate_inputs'), f"{class_name} missing validate_inputs method"
        assert hasattr(instance, 'get_parameter_schema'), f"{class_name} missing get_parameter_schema method"
        assert hasattr(instance, 'is_healthy'), f"{class_name} missing is_healthy method"
        
        # Check health
        assert instance.is_healthy(), f"{class_name} is not healthy"
        
        print(f"  [OK] {class_name}: {cls.DISPLAY_NAME}")
        return True
        
    except Exception as e:
        print(f"  [FAIL] {class_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("Mathematical Verification Bubble Suite - Verification")
    print("=" * 70)
    print()
    
    success_count = 0
    total_size = 0
    
    for module_name, class_name in MATH_BUBBLES:
        if verify_bubble(module_name, class_name):
            success_count += 1
            # Get file size
            file_path = Path(__file__).parent / "bubblelabs_nodes" / f"{module_name}.py"
            if file_path.exists():
                total_size += file_path.stat().st_size
    
    print()
    print("=" * 70)
    print(f"Results: {success_count}/{len(MATH_BUBBLES)} bubbles verified successfully")
    print(f"Total Code: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    print("=" * 70)
    
    if success_count == len(MATH_BUBBLES):
        print("\n[SUCCESS] All mathematical verification bubbles are ready!")
        return 0
    else:
        print(f"\n[WARN]  {len(MATH_BUBBLES) - success_count} bubble(s) failed verification")
        return 1

if __name__ == "__main__":
    sys.exit(main())
