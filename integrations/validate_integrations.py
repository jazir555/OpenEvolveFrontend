"""
Quick validation script for RESE-E2E Stage Integrations

Validates all integration modules can be imported and instantiated.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def validate_stage(stage_num: int) -> bool:
    """Validate a single stage integration"""
    try:
        if stage_num == 1:
            from integrations.stage1 import Stage1Integration
            integration = Stage1Integration()
            assert hasattr(integration, 'analyze_prompt')
        elif stage_num == 2:
            from integrations.stage2 import Stage2Integration
            integration = Stage2Integration()
            assert hasattr(integration, 'analyze_domains')
        elif stage_num == 3:
            from integrations.stage3 import Stage3Integration
            integration = Stage3Integration()
            assert hasattr(integration, 'search')
        elif stage_num == 5:
            from integrations.stage5 import Stage5Integration
            integration = Stage5Integration()
            # Stage5Integration exposes validate_candidate(); the earlier
            # 'validate_solution' name never existed on this class.
            assert hasattr(integration, 'validate_candidate')
        elif stage_num == 6:
            from integrations.stage6 import Stage6Integration
            integration = Stage6Integration()
            assert hasattr(integration, 'analyze_error')
        elif stage_num == 7:
            from integrations.stage7 import Stage7Integration
            integration = Stage7Integration()
            assert hasattr(integration, 'validate_adversarially')
        elif stage_num == 8:
            from integrations.stage8 import Stage8Integration
            integration = Stage8Integration()
            assert hasattr(integration, 'assemble_architecture')
        elif stage_num == 9:
            from integrations.stage9 import Stage9Integration
            integration = Stage9Integration()
            assert hasattr(integration, 'validate_final_solution')
        else:
            return False

        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False

def main():
    """Validate all stage integrations"""
    print("=" * 70)
    print("RESE-E2E Stage Integration Validation")
    print("=" * 70)
    print()

    stages = [1, 2, 3, 5, 6, 7, 8, 9]
    results = {}

    for stage_num in stages:
        print(f"Validating Stage {stage_num}...", end=" ")
        success = validate_stage(stage_num)
        results[stage_num] = success
        print("PASS" if success else "FAIL")

    print()
    print("=" * 70)
    print("Validation Summary")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print()
        print("SUCCESS: All stage integrations validated successfully!")
        print()
        print("Available Integrations:")
        for stage_num in sorted(stages):
            print(f"  - Stage {stage_num}: {'Valid' if results[stage_num] else 'Invalid'}")
        print()
        print("Next Steps:")
        print("  1. Run full test suite: python rese/integrations/test_e2e_pipeline.py")
        print("  2. Read integration guide: rese/integrations/INTEGRATION_GUIDE.md")
        print("  3. Start using integrations in your pipeline")
        return 0
    else:
        print()
        print("ERROR: Some integrations failed validation")
        print("Please check the errors above")
        return 1

if __name__ == '__main__':
    sys.exit(main())
