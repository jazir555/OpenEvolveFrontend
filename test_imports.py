"""
Test file to verify imports work correctly
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly"""
    try:
        from content_analyzer import ContentAnalyzer
        ContentAnalyzer()
        print("[PASS] content_analyzer imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import content_analyzer: {e}")
        return False
    
    try:
        from red_team import RedTeam
        RedTeam()
        print("[PASS] red_team imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import red_team: {e}")
        return False
        
    try:
        from blue_team import BlueTeam
        BlueTeam()
        print("[PASS] blue_team imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import blue_team: {e}")
        return False
        
    try:
        from evaluator_team import EvaluatorTeam
        EvaluatorTeam()
        print("[PASS] evaluator_team imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import evaluator_team: {e}")
        return False
        
    try:
        from quality_assessment import QualityAssessmentEngine
        QualityAssessmentEngine()
        print("[PASS] quality_assessment imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import quality_assessment: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if test_imports():
        print("\n[PASS] All imports successful!")
    else:
        print("\n[FAIL] Some imports failed!")
        sys.exit(1)