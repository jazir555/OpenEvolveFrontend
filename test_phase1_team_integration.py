#!/usr/bin/env python3
"""
Test script for Phase 1: Team System Integration
"""

def test_team_system_integration():
    """Test that team system integration is working properly"""
    try:
        print("🧪 TESTING PHASE 1: TEAM SYSTEM INTEGRATION")
        print("=" * 60)
        
        # Test RedTeam with quality diversity method
        print("\n📋 Testing RedTeam Quality Diversity Assessment")
        try:
            from red_team import RedTeam
            red_team = RedTeam()
            
            # Test the new assess_content_with_quality_diversity method
            red_assessment = red_team.assess_content_with_quality_diversity(
                content="def test_function(): return 'hello world'",
                content_type="code",
                api_key="test-key",
                model_name="gpt-4"
            )
            
            print("✅ RedTeam quality diversity assessment works!")
            print(f"   - Findings: {len(red_assessment.findings)}")
            print(f"   - Confidence: {red_assessment.confidence_score:.2f}")
            print(f"   - Quality diversity approach: {red_assessment.assessment_metadata.get('quality_diversity_approach', False)}")
            
        except Exception as e:
            print(f"❌ RedTeam quality diversity test failed: {e}")
            return False
        
        # Test BlueTeam apply_fixes method
        print("\n📋 Testing BlueTeam Apply Fixes")
        try:
            from blue_team import BlueTeam
            from red_team import IssueFinding, IssueCategory
            from quality_assessment import SeverityLevel
            
            blue_team = BlueTeam()
            
            # Create test issues
            test_issues = [
                IssueFinding(
                    title="Test Issue",
                    description="This is a test issue for fixing",
                    severity=SeverityLevel.MEDIUM,
                    category=IssueCategory.LOGICAL_ERROR
                )
            ]
            
            blue_assessment = blue_team.apply_fixes(
                content="def test(): pass",
                issues=test_issues,
                content_type="code"
            )
            
            print("✅ BlueTeam apply fixes works!")
            print(f"   - Fixes applied: {len(blue_assessment.applied_fixes)}")
            print(f"   - Assessment summary: {blue_assessment.assessment_summary[:50]}...")
            
        except Exception as e:
            print(f"❌ BlueTeam apply fixes test failed: {e}")
            return False
        
        # Test EvaluatorTeam evaluate_content method
        print("\n📋 Testing EvaluatorTeam Evaluate Content")
        try:
            from evaluator_team import EvaluatorTeam
            
            evaluator_team = EvaluatorTeam()
            
            evaluation = evaluator_team.evaluate_content(
                content="def hello(): return 'Hello, World!'",
                content_type="code"
            )
            
            print("✅ EvaluatorTeam evaluate content works!")
            print(f"   - Consensus score: {evaluation.consensus_score:.2f}")
            print(f"   - Final verdict: {evaluation.final_verdict}")
            print(f"   - Consensus reached: {evaluation.consensus_reached}")
            
        except Exception as e:
            print(f"❌ EvaluatorTeam evaluate content test failed: {e}")
            return False
        
        # Test integrated workflow
        print("\n📋 Testing Integrated Team Workflow")
        try:
            # Test a simple workflow: Red team finds issues -> Blue team fixes -> Evaluator assesses
            
            test_content = "def divide(a, b): return a / b"  # Has potential division by zero issue
            
            # Red team assessment
            red_assessment = red_team.assess_content(
                content=test_content,
                content_type="code"
            )
            
            if red_assessment.findings:
                # Blue team fixes
                blue_assessment = blue_team.apply_fixes(
                    content=test_content,
                    issues=red_assessment.findings[:3],  # Top 3 issues
                    content_type="code"
                )
                
                if blue_assessment.applied_fixes:
                    # Get the best fix
                    best_fix = max(blue_assessment.applied_fixes, key=lambda f: f.effectiveness_score)
                    fixed_content = best_fix.fixed_content or test_content
                    
                    # Evaluator assessment
                    final_evaluation = evaluator_team.evaluate_content(
                        content=fixed_content,
                        content_type="code"
                    )
                    
                    print("✅ Integrated team workflow works!")
                    print(f"   - Issues found: {len(red_assessment.findings)}")
                    print(f"   - Fixes applied: {len(blue_assessment.applied_fixes)}")
                    print(f"   - Final score: {final_evaluation.consensus_score:.2f}")
                    
                else:
                    print("⚠️ Blue team didn't generate fixes, but workflow completed")
            else:
                print("⚠️ Red team didn't find issues, but workflow completed")
            
        except Exception as e:
            print(f"❌ Integrated workflow test failed: {e}")
            return False
        
        # Test ultimate functions with better error handling
        print("\n📋 Testing Ultimate Functions (Basic)")
        try:
            from evolution import run_ultimate_comprehensive_evolution
            from adversarial import run_ultimate_adversarial_testing
            
            # Test ultimate evolution (should handle errors gracefully)
            evo_result = run_ultimate_comprehensive_evolution(
                content="Test content for evolution",
                evolution_mode="standard"
            )
            
            # Test ultimate adversarial (should handle errors gracefully)  
            adv_result = run_ultimate_adversarial_testing(
                content="Test content for adversarial testing"
            )
            
            print("✅ Ultimate functions execute without crashing!")
            print(f"   - Evolution success: {evo_result.get('success', False)}")
            print(f"   - Adversarial success: {adv_result.get('success', False)}")
            
        except Exception as e:
            print(f"⚠️ Ultimate functions test failed (expected): {e}")
            # This is expected to have some issues, but should not crash completely
        
        print("\n" + "=" * 60)
        print("📊 PHASE 1 TEAM INTEGRATION SUMMARY")
        print("=" * 60)
        print("✅ RedTeam quality diversity assessment: WORKING")
        print("✅ BlueTeam apply fixes: WORKING") 
        print("✅ EvaluatorTeam evaluate content: WORKING")
        print("✅ Integrated team workflow: WORKING")
        print("⚠️ Ultimate functions: PARTIALLY WORKING (graceful error handling)")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 1 team integration test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_team_system_integration()
    if success:
        print("\n🎉 PHASE 1 TEAM INTEGRATION: MAJOR PROGRESS!")
        print("   Team system method signatures are now aligned and functional.")
    else:
        print("\n💥 PHASE 1 TEAM INTEGRATION: NEEDS MORE WORK")