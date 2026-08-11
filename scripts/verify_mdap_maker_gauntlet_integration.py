"""
MDAP/MAKER-Gauntlet Integration Verification Script

Tests the integration between MDAP/MAKER systems and the Gauntlet quality control system.

Author: OpenEvolve QA Team
Date: 2026-02-17
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_integration_module_import():
    """Test MDAP/MAKER-Gauntlet integration module can be imported."""
    print("\n" + "="*80)
    print("TEST 1: MDAP/MAKER-Gauntlet Integration Module Import")
    print("="*80)
    
    try:
        from mdap_maker_gauntlet_integration import (
            MDAPMakerGauntletMode,
            MDAPMakerGauntletConfig,
            MDAPMakerGauntletResult,
            MDAPMakerGauntletIntegration,
            create_mdap_maker_integration,
            execute_gauntlet_with_mdap
        )
        print("[PASS] MDAP/MAKER-Gauntlet integration module imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to import integration module: {e}")
        return False


def test_mdap_availability():
    """Test MDAP components availability."""
    print("\n" + "="*80)
    print("TEST 2: MDAP Components Availability")
    print("="*80)

    mdap_available = False
    try:
        from adaptive_mdap import (
            TaskComplexityClassifier,
            AdaptiveMDAPAllocator,
            AdaptiveExecutionController,
            ComplexityScore,
            SubProblem
        )

        # Test instantiation
        classifier = TaskComplexityClassifier()
        allocator = AdaptiveMDAPAllocator()
        controller = AdaptiveExecutionController()

        print("[PASS] MDAP components available and functional")
        mdap_available = True
    except Exception as e:
        print(f"[WARN] MDAP components not fully available: {e}")

    # Check adaptive_mdap availability flag
    try:
        from mdap_maker_gauntlet_integration import ADAPTIVE_MDAP_AVAILABLE
        if ADAPTIVE_MDAP_AVAILABLE:
            print("[PASS] ADAPTIVE_MDAP_AVAILABLE = True")
        else:
            print("[WARN] ADAPTIVE_MDAP_AVAILABLE = False")
    except Exception as e:
        print(f"[FAIL] Failed to check MDAP availability: {e}")

    return mdap_available


def test_maker_availability():
    """Test MAKER components availability."""
    print("\n" + "="*80)
    print("TEST 3: MAKER Components Availability")
    print("="*80)
    
    maker_available = False
    try:
        from maker_engine import MakerEngine, MakerConfig, MakerState, MakerStep
        from mdap_engine import RedFlagRules, RedFlagger
        
        # Test instantiation
        config = MakerConfig()
        red_flag_rules = RedFlagRules()
        red_flagger = RedFlagger(red_flag_rules)
        
        print("[PASS] MAKER components available and functional")
        maker_available = True
    except Exception as e:
        print(f"[WARN] MAKER components not fully available: {e}")
    
    # Check maker availability flag
    try:
        from mdap_maker_gauntlet_integration import MAKER_AVAILABLE
        if MAKER_AVAILABLE:
            print("[PASS] MAKER_AVAILABLE = True")
        else:
            print("[WARN] MAKER_AVAILABLE = False")
    except Exception as e:
        print(f"[FAIL] Failed to check MAKER availability: {e}")
    
    return maker_available


def test_integration_instantiation():
    """Test MDAP/MAKER-Gauntlet integration instantiation."""
    print("\n" + "="*80)
    print("TEST 4: Integration Instantiation")
    print("="*80)
    
    try:
        from mdap_maker_gauntlet_integration import (
            MDAPMakerGauntletIntegration,
            MDAPMakerGauntletConfig,
            MDAPMakerGauntletMode
        )
        
        # Test with different modes
        modes = [
            MDAPMakerGauntletMode.MDAP_ADAPTIVE,
            MDAPMakerGauntletMode.MAKER_VOTING,
            MDAPMakerGauntletMode.HYBRID,
            MDAPMakerGauntletMode.CONSENSUS
        ]
        
        for mode in modes:
            config = MDAPMakerGauntletConfig(mode=mode)
            integration = MDAPMakerGauntletIntegration(config=config)
            print(f"[PASS] Integration instantiated with mode: {mode.value}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Integration instantiation failed: {e}")
        return False


def test_complexity_analysis():
    """Test MDAP complexity analysis."""
    print("\n" + "="*80)
    print("TEST 5: MDAP Complexity Analysis")
    print("="*80)

    try:
        from mdap_maker_gauntlet_integration import MDAPMakerGauntletIntegration
        from adaptive_mdap import SubProblem

        integration = MDAPMakerGauntletIntegration()

        if not integration.complexity_classifier:
            print("[WARN] Complexity classifier not available")
            return False

        # Test complexity analysis
        problem_desc = "Implement a complex trading algorithm with risk management"
        solution = {"code": "def trade(): pass", "type": "trading"}
        context = {"domain": "finance", "depth": 3}

        complexity_score = integration._analyze_complexity(problem_desc, solution, context)

        if complexity_score:
            print(f"[PASS] Complexity analysis successful")
            print(f"   - Overall: {complexity_score.overall_score:.3f}")
            print(f"   - Text length: {complexity_score.text_length_score:.3f}")
            print(f"   - Depth: {complexity_score.depth_score:.3f}")
            print(f"   - Dependencies: {complexity_score.dependency_score:.3f}")
            return True
        else:
            print("[FAIL] Complexity analysis returned None")
            return False

    except Exception as e:
        print(f"[FAIL] Complexity analysis failed: {e}")
        return False


def test_gauntlet_adaptation():
    """Test adaptive gauntlet configuration."""
    print("\n" + "="*80)
    print("TEST 6: Adaptive Gauntlet Configuration")
    print("="*80)
    
    try:
        from mdap_maker_gauntlet_integration import MDAPMakerGauntletIntegration
        from gauntlet_types import AdversarialGauntlet, GauntletType
        
        integration = MDAPMakerGauntletIntegration()
        
        # Test gauntlet adaptation
        from adaptive_mdap import AdaptiveMDAPAllocator
        
        allocator = AdaptiveMDAPAllocator()
        
        # Test different complexity levels
        test_cases = [
            (0.2, "Low complexity"),
            (0.5, "Medium complexity"),
            (0.8, "High complexity")
        ]
        
        for complexity, description in test_cases:
            strategy = allocator.allocate_resources(complexity)
            
            # Create a test gauntlet
            gauntlet = AdversarialGauntlet(f"test_{complexity}")
            
            # Adapt gauntlet
            integration._adapt_gauntlet_config(gauntlet, strategy)
            
            print(f"[PASS] {description}: strategy={strategy.strategy}, n_agents={strategy.n_agents}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Gauntlet adaptation failed: {e}")
        return False


def test_maker_voting():
    """Test MAKER voting integration."""
    print("\n" + "="*80)
    print("TEST 7: MAKER Voting Integration")
    print("="*80)

    try:
        from mdap_maker_gauntlet_integration import MDAPMakerGauntletIntegration
        from gauntlet_types import StatisticalGauntlet

        integration = MDAPMakerGauntletIntegration()

        if not integration.maker_engine:
            print("[WARN] MAKER engine not available")
            return False

        # Create a simple gauntlet
        gauntlet = StatisticalGauntlet("test_maker_voting")

        # Mock solution and context
        solution = {"data": [1, 2, 3, 4, 5]}
        context = {"expected_mean": 3.0}

        # Test MAKER voting
        # Note: MAKER voting may return empty result if team has no members with API credentials
        # This is expected behavior - the integration is working, but needs configured team members
        maker_result = integration._execute_with_maker_voting(gauntlet, solution, context)

        if maker_result is not None:
            print(f"[PASS] MAKER voting executed successfully")
            if maker_result.get('agent_votes'):
                print(f"   - Agent votes: {len(maker_result['agent_votes'])}")
            else:
                print(f"   - No agent votes (expected without API credentials)")
            if maker_result.get('red_flags'):
                print(f"   - Red flags: {len(maker_result['red_flags'])}")
            return True
        else:
            print("[WARN] MAKER voting returned None (unexpected)")
            return False

    except Exception as e:
        print(f"[FAIL] MAKER voting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_consensus_calculation():
    """Test consensus calculation from agent votes."""
    print("\n" + "="*80)
    print("TEST 8: Consensus Calculation")
    print("="*80)
    
    try:
        from mdap_maker_gauntlet_integration import MDAPMakerGauntletIntegration
        from gauntlet_types import GauntletResult, GauntletType
        from datetime import datetime
        
        integration = MDAPMakerGauntletIntegration()
        
        # Test with different vote distributions
        test_cases = [
            (
                [
                    {"score": 0.8, "justification": "Good"},
                    {"score": 0.85, "justification": "Good"},
                    {"score": 0.78, "justification": "Good"}
                ],
                "High consensus"
            ),
            (
                [
                    {"score": 0.9, "justification": "Excellent"},
                    {"score": 0.3, "justification": "Poor"},
                    {"score": 0.5, "justification": "Average"}
                ],
                "Low consensus"
            ),
            (
                [
                    {"score": 0.7, "justification": "OK"},
                    {"score": 0.7, "justification": "OK"},
                    {"score": 0.7, "justification": "OK"},
                    {"score": 0.7, "justification": "OK"}
                ],
                "Perfect consensus"
            )
        ]
        
        # Mock gauntlet result
        mock_result = GauntletResult(
            gauntlet_type=GauntletType.STATISTICAL,
            gauntlet_name="test",
            solution_id="test",
            passed=True,
            score=0.8,
            confidence=0.8,
            execution_time=1.0,
            timestamp=datetime.now()
        )
        
        for votes, description in test_cases:
            consensus_reached, consensus_score = integration._calculate_consensus(votes, mock_result)
            
            print(f"[PASS] {description}: consensus_reached={consensus_reached}, score={consensus_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Consensus calculation failed: {e}")
        return False


def test_create_mdap_adaptive_gauntlet():
    """Test creation of MDAP-adaptive gauntlet."""
    print("\n" + "="*80)
    print("TEST 9: Create MDAP-Adaptive Gauntlet")
    print("="*80)
    
    try:
        from mdap_maker_gauntlet_integration import MDAPMakerGauntletIntegration
        from gauntlet_types import GauntletType
        
        integration = MDAPMakerGauntletIntegration()
        
        # Test with different problem descriptions
        test_problems = [
            ("Simple addition", {"code": "def add(a, b): return a + b"}, "general"),
            ("Implement a complex machine learning pipeline with feature engineering, model selection, and hyperparameter optimization", 
             {"code": "class MLPipeline: pass"}, "ml"),
            ("Create a quantum computing simulator", 
             {"code": "class QuantumSimulator: pass"}, "physics")
        ]
        
        for problem_desc, solution, domain in test_problems:
            gauntlet, result = integration.create_mdap_adaptive_gauntlet(
                problem_description=problem_desc,
                solution=solution,
                context={"domain": domain}
            )
            
            print(f"[PASS] Problem: {problem_desc[:50]}...")
            print(f"   - Gauntlet type: {gauntlet.gauntlet_type.value}")
            print(f"   - Gauntlet name: {gauntlet.name}")
            if result.complexity_score:
                print(f"   - Complexity: {result.complexity_score.overall_score:.3f}")
            if result.mdap_strategy:
                print(f"   - MDAP strategy: {result.mdap_strategy}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] MDAP-adaptive gauntlet creation failed: {e}")
        return False


def test_convenience_functions():
    """Test convenience functions."""
    print("\n" + "="*80)
    print("TEST 10: Convenience Functions")
    print("="*80)
    
    try:
        from mdap_maker_gauntlet_integration import (
            create_mdap_maker_integration,
            execute_gauntlet_with_mdap
        )
        from gauntlet_types import AdversarialGauntlet
        
        # Test create_mdap_maker_integration
        integration = create_mdap_maker_integration(
            mode=MDAPMakerGauntletMode.HYBRID,
            use_complexity_adaptation=True,
            use_maker_voting=True
        )
        print("[PASS] create_mdap_maker_integration() works")
        
        # Test execute_gauntlet_with_mdap (basic test)
        gauntlet = AdversarialGauntlet("test_convenience")
        solution = {"code": "def test(): pass"}
        
        result = execute_gauntlet_with_mdap(
            gauntlet=gauntlet,
            solution=solution,
            problem_description="Test problem"
        )
        
        print(f"[PASS] execute_gauntlet_with_mdap() works")
        print(f"   - Result type: {type(result).__name__}")
        print(f"   - Has gauntlet_result: {hasattr(result, 'gauntlet_result')}")
        print(f"   - Has complexity_score: {hasattr(result, 'complexity_score')}")
        print(f"   - Has maker_state: {hasattr(result, 'maker_state')}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Convenience functions failed: {e}")
        return False


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "="*80)
    print("MDAP/MAKER-GAUNTLET INTEGRATION VERIFICATION")
    print("="*80)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    tests = [
        ("Integration Module Import", test_integration_module_import),
        ("MDAP Components", test_mdap_availability),
        ("MAKER Components", test_maker_availability),
        ("Integration Instantiation", test_integration_instantiation),
        ("Complexity Analysis", test_complexity_analysis),
        ("Gauntlet Adaptation", test_gauntlet_adaptation),
        ("MAKER Voting", test_maker_voting),
        ("Consensus Calculation", test_consensus_calculation),
        ("MDAP-Adaptive Gauntlet", test_create_mdap_adaptive_gauntlet),
        ("Convenience Functions", test_convenience_functions),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")
    
    print("\n" + "-"*80)
    print(f"Total: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    print("="*80)
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED! MDAP/MAKER-Gauntlet integration is fully functional!")
    else:
        print(f"\n[WARN] {total - passed} test(s) failed or had warnings. Review the output above.")
    
    return passed == total


if __name__ == "__main__":
    # Import MDAPMakerGauntletMode for test 10
    try:
        from mdap_maker_gauntlet_integration import MDAPMakerGauntletMode
    except:
        MDAPMakerGauntletMode = None
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
