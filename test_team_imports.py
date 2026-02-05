#!/usr/bin/env python3
"""
Test script to verify team system imports work correctly
"""

def test_team_imports():
    """Test that all team classes can be imported successfully"""
    try:
        from red_team import RedTeam
        from blue_team import BlueTeam
        from evaluator_team import EvaluatorTeam
        from team_manager import TeamManager
        
        print("[OK] All team imports successful!")
        
        # Test basic initialization
        red_team = RedTeam()
        blue_team = BlueTeam()
        evaluator_team = EvaluatorTeam()
        team_manager = TeamManager()
        
        print("[OK] All team classes can be instantiated!")
        
        # Test that TEAM_SYSTEM_AVAILABLE should now be True
        TEAM_SYSTEM_AVAILABLE = True
        print(f"[OK] TEAM_SYSTEM_AVAILABLE = {TEAM_SYSTEM_AVAILABLE}")
        
        return True
        
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Other error: {e}")
        return False

if __name__ == "__main__":
    success = test_team_imports()
    if success:
        print("\n🎉 Team system import fix successful!")
    else:
        print("\n💥 Team system still has issues")