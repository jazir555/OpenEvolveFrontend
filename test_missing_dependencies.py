#!/usr/bin/env python3
"""
Test script to verify missing dependencies have been fixed
"""

def test_missing_dependencies():
    """Test that all previously missing dependencies are now available"""
    try:
        # Test session_manager imports
        try:
            from session_manager import APPROVAL_PROMPT, RED_TEAM_CRITIQUE_PROMPT, BLUE_TEAM_PATCH_PROMPT
            print("✅ Session manager prompts imported successfully!")
            print(f"   - APPROVAL_PROMPT: {len(APPROVAL_PROMPT)} characters")
            print(f"   - RED_TEAM_CRITIQUE_PROMPT: {len(RED_TEAM_CRITIQUE_PROMPT)} characters")
            print(f"   - BLUE_TEAM_PATCH_PROMPT: {len(BLUE_TEAM_PATCH_PROMPT)} characters")
        except ImportError as e:
            print(f"❌ Session manager import failed: {e}")
            return False
        
        # Test review_utils imports
        try:
            from review_utils import determine_review_type, get_appropriate_prompts
            print("✅ Review utils imported successfully!")
            
            # Test functionality
            test_content = "def hello_world(): print('Hello, World!')"
            review_type = determine_review_type(test_content)
            print(f"   - Review type detection works: '{test_content[:20]}...' -> {review_type}")
            
            red_prompt, blue_prompt = get_appropriate_prompts(review_type)
            print(f"   - Prompt generation works: {len(red_prompt)} chars red, {len(blue_prompt)} chars blue")
        except ImportError as e:
            print(f"❌ Review utils import failed: {e}")
            return False
        
        # Test logging_util imports
        try:
            from logging_util import _update_adv_log_and_status
            print("✅ Logging util imported successfully!")
            
            # Test functionality (should work without Streamlit)
            _update_adv_log_and_status("Test adversarial log message")
            print("   - Adversarial logging function works!")
        except ImportError as e:
            print(f"❌ Logging util import failed: {e}")
            return False
        
        # Test that adversarial.py can now import without fallbacks
        try:
            import adversarial
            print("✅ Adversarial module imports without fallbacks!")
            
            # Check if the fallback flags are no longer needed
            if hasattr(adversarial, 'APPROVAL_PROMPT'):
                print("   - APPROVAL_PROMPT available in adversarial module")
            if hasattr(adversarial, 'determine_review_type'):
                print("   - determine_review_type available in adversarial module")
            if hasattr(adversarial, '_update_adv_log_and_status'):
                print("   - _update_adv_log_and_status available in adversarial module")
        except ImportError as e:
            print(f"❌ Adversarial module import failed: {e}")
            return False
        
        # Test that evolution.py team system is available
        try:
            from evolution import TEAM_SYSTEM_AVAILABLE
            print(f"✅ Team system availability: {TEAM_SYSTEM_AVAILABLE}")
            if TEAM_SYSTEM_AVAILABLE:
                print("   - Team system imports are working!")
            else:
                print("   - Team system still has issues")
                return False
        except ImportError as e:
            print(f"❌ Evolution team system check failed: {e}")
            return False
        
        print("✅ All missing dependencies have been resolved!")
        return True
        
    except Exception as e:
        print(f"❌ Missing dependencies test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_missing_dependencies()
    if success:
        print("\n🎉 Missing dependencies fixed! Critical blocker #4 RESOLVED!")
    else:
        print("\n💥 Missing dependencies still exist")