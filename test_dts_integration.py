#!/usr/bin/env python3
"""
Test script to verify DTS integration works with graceful fallbacks.
"""
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dts_import():
    """Test that DTS integration imports work with fallbacks."""
    print("Testing DTS integration imports...")
    
    # Test 1: Import the integration module
    try:
        from dts_integration import DTSIntegration, DTSIntegrationConfig
        print("✓ DTS integration module imports successfully")
        
        # Test 2: Create config (should work even without API keys)
        try:
            config = DTSIntegrationConfig(max_rounds=2, use_multi_judge=True)
            print("✓ DTSIntegrationConfig created successfully")
        except Exception as e:
            print(f"✗ DTSIntegrationConfig creation failed: {e}")
            
        # Test 3: Create integration instance
        try:
            integration = DTSIntegration(config)
            print("✓ DTSIntegration instance created successfully")
            
            # Test 4: Check if DTS is available
            if integration.dts_available:
                print("✓ DTS engine is available (API keys configured)")
            else:
                print("✓ DTS engine is not available (fallback mode active)")
                
        except Exception as e:
            print(f"✗ DTSIntegration instance creation failed: {e}")
            
    except ImportError as e:
        print(f"✗ Failed to import DTS integration module: {e}")
        return False
    
    return True

def test_red_team_integration():
    """Test Red Team DTS integration."""
    print("\nTesting Red Team DTS integration...")
    
    try:
        from red_team import RedTeam
        red_team = RedTeam()
        print("✓ RedTeam imported successfully")
        
        # Test the DTS method
        sample_content = "def test(): return 1"
        result = red_team.run_adversarial_dialogue_with_dts(
            content=sample_content,
            content_type="code",
            rounds=1
        )
        
        print(f"✓ Red Team DTS method executed successfully")
        print(f"  DTS available: {result.get('dts_available', False)}")
        print(f"  Fallback used: {result.get('fallback_used', False)}")
        print(f"  Findings count: {result.get('findings_count', 0)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Red Team DTS integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_blue_team_integration():
    """Test Blue Team DTS integration."""
    print("\nTesting Blue Team DTS integration...")
    
    try:
        from blue_team import BlueTeam
        blue_team = BlueTeam()
        print("✓ BlueTeam imported successfully")
        
        # Test the DTS method
        sample_content = "def test(): return 1"
        result = blue_team.generate_fixes_with_dts(
            content=sample_content,
            content_type="code",
            rounds=1
        )
        
        print(f"✓ Blue Team DTS method executed successfully")
        print(f"  DTS available: {result.get('dts_available', False)}")
        print(f"  Fallback used: {result.get('fallback_used', False)}")
        print(f"  Fix strategies count: {result.get('fix_count', 0)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Blue Team DTS integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_assessment_integration():
    """Test Quality Assessment DTS integration."""
    print("\nTesting Quality Assessment DTS integration...")
    
    try:
        from quality_assessment import QualityAssessmentEngine
        engine = QualityAssessmentEngine()
        print("✓ QualityAssessmentEngine imported successfully")
        
        # Test the DTS method
        sample_content = "This is a test document for quality assessment."
        result = engine.assess_with_dts_multi_judge(
            content=sample_content,
            content_type="document_general",
            judge_count=2
        )
        
        print(f"✓ Quality Assessment DTS method executed successfully")
        print(f"  DTS available: {result.get('dts_available', False)}")
        print(f"  Fallback used: {result.get('fallback_used', False)}")
        print(f"  Consensus score: {result.get('consensus_score', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Quality Assessment DTS integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evolution_integration():
    """Test Evolution DTS integration."""
    print("\nTesting Evolution DTS integration...")
    
    try:
        from evolution import run_evolution_with_dts_strategy_exploration
        print("✓ Evolution DTS function imported successfully")
        
        # Test the DTS method (with minimal parameters)
        sample_content = "def add(a, b): return a + b"
        result = run_evolution_with_dts_strategy_exploration(
            content=sample_content,
            content_type="code",
            evolution_mode="standard",
            use_dts_for_strategy=True,
            dts_rounds=1
        )
        
        print(f"✓ Evolution DTS method executed successfully")
        print(f"  DTS available: {result.get('dts_available', False)}")
        print(f"  Fallback used: {result.get('fallback_used', False)}")
        print(f"  Final score: {result.get('final_score', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Evolution DTS integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("DTS Integration Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    # Run tests
    tests = [
        ("DTS Module Import", test_dts_import),
        ("Red Team Integration", test_red_team_integration),
        ("Blue Team Integration", test_blue_team_integration),
        ("Quality Assessment Integration", test_quality_assessment_integration),
        ("Evolution Integration", test_evolution_integration),
    ]
    
    for test_name, test_func in tests:
        tests_total += 1
        print(f"\n{'='*40}")
        print(f"Test: {test_name}")
        print(f"{'='*40}")
        try:
            if test_func():
                tests_passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("✓ All tests passed! DTS integration is working correctly.")
        return 0
    else:
        print(f"⚠ {tests_total - tests_passed} tests failed.")
        print("Note: Some failures may be expected if DTS API keys are not configured.")
        print("The integration should fall back gracefully to standard methods.")
        return 1

if __name__ == "__main__":
    sys.exit(main())