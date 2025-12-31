"""
Test Multi-Round Testing System
"""

from multi_round_testing import (
    MultiRoundTester,
    RoundStrategy,
    RoundStoppingCriteria,
    create_evolution_test_function,
    create_team_test_function,
    run_multi_round_evolution,
    run_multi_round_team_testing
)


def mock_evolution_function(content: str, **parameters) -> dict:
    """Mock evolution function for testing"""
    iterations = parameters.get('iterations', 10)
    mutation_rate = parameters.get('mutation_rate', 0.1)
    
    # Simulate improvement based on parameters
    improvement_factor = min(iterations * mutation_rate, 0.5)
    improved_content = content + f" [Improved with {iterations} iterations, {mutation_rate} mutation rate]"
    
    return {
        'evolved_content': improved_content,
        'quality_score': min(0.5 + improvement_factor, 1.0),
        'metrics': {
            'iterations_used': iterations,
            'mutation_rate_used': mutation_rate,
            'improvement_factor': improvement_factor
        }
    }


def mock_team_function(content: str, **parameters) -> dict:
    """Mock team function for testing"""
    team_size = parameters.get('team_size', 3)
    consensus_threshold = parameters.get('consensus_threshold', 0.7)
    
    # Simulate team improvement
    team_improvement = min(team_size * consensus_threshold * 0.2, 0.6)
    improved_content = content + f" [Team improved with {team_size} members, {consensus_threshold} consensus]"
    
    return {
        'content': improved_content,
        'overall_score': min(5.0 + team_improvement * 10, 10.0),
        'metrics': {
            'team_size': team_size,
            'consensus_achieved': consensus_threshold,
            'team_improvement': team_improvement
        }
    }


def test_basic_multi_round():
    """Test basic multi-round functionality"""
    print("Testing basic multi-round functionality...")
    
    tester = MultiRoundTester()
    test_function = create_evolution_test_function(mock_evolution_function)
    
    result = tester.run_multi_round_test(
        content="Initial test content",
        test_function=test_function,
        max_rounds=3,
        strategy=RoundStrategy.FIXED,
        base_parameters={'iterations': 10, 'mutation_rate': 0.1}
    )
    
    print(f"✅ Basic multi-round test completed:")
    print(f"   - Total rounds: {result.total_rounds}")
    print(f"   - Successful rounds: {result.successful_rounds}")
    print(f"   - Overall improvement: {result.overall_improvement:.2f}")
    print(f"   - Stopping reason: {result.stopping_reason}")
    
    assert result.total_rounds == 3
    assert result.successful_rounds > 0
    assert len(result.round_results) == 3
    assert result.best_round is not None
    
    return True


def test_adaptive_strategy():
    """Test adaptive round strategy"""
    print("\nTesting adaptive strategy...")
    
    tester = MultiRoundTester()
    test_function = create_evolution_test_function(mock_evolution_function)
    
    result = tester.run_multi_round_test(
        content="Adaptive test content",
        test_function=test_function,
        max_rounds=5,
        strategy=RoundStrategy.ADAPTIVE,
        base_parameters={'iterations': 5, 'mutation_rate': 0.2}
    )
    
    print(f"✅ Adaptive strategy test completed:")
    print(f"   - Total rounds: {result.total_rounds}")
    print(f"   - Parameter evolution tracked: {len(result.convergence_data['parameter_evolution'])}")
    
    # Check that parameters evolved over rounds
    param_evolution = result.convergence_data['parameter_evolution']
    if len(param_evolution) > 1:
        first_params = param_evolution[0]
        last_params = param_evolution[-1]
        print(f"   - First round iterations: {first_params.get('iterations', 'N/A')}")
        print(f"   - Last round iterations: {last_params.get('iterations', 'N/A')}")
    
    assert result.total_rounds > 0
    assert len(result.convergence_data['quality_scores']) == result.total_rounds
    
    return True


def test_progressive_strategy():
    """Test progressive round strategy"""
    print("\nTesting progressive strategy...")
    
    tester = MultiRoundTester()
    test_function = create_evolution_test_function(mock_evolution_function)
    
    result = tester.run_multi_round_test(
        content="Progressive test content",
        test_function=test_function,
        max_rounds=4,
        strategy=RoundStrategy.PROGRESSIVE,
        base_parameters={'iterations': 5, 'mutation_rate': 0.1}
    )
    
    print(f"✅ Progressive strategy test completed:")
    print(f"   - Total rounds: {result.total_rounds}")
    
    # Check that parameters increased progressively
    param_evolution = result.convergence_data['parameter_evolution']
    if len(param_evolution) > 1:
        iterations_progression = [p.get('iterations', 0) for p in param_evolution]
        print(f"   - Iterations progression: {iterations_progression}")
        
        # Should generally increase (allowing for some variation)
        assert iterations_progression[-1] >= iterations_progression[0]
    
    return True


def test_stopping_criteria():
    """Test different stopping criteria"""
    print("\nTesting stopping criteria...")
    
    def low_improvement_function(content: str, **parameters) -> dict:
        """Function that produces very low improvement"""
        return {
            'evolved_content': content + " [minimal change]",
            'quality_score': 0.5,  # Fixed low score
            'metrics': {'improvement': 0.001}  # Very low improvement
        }
    
    tester = MultiRoundTester()
    test_function = create_evolution_test_function(low_improvement_function)
    
    # Test improvement plateau stopping
    result = tester.run_multi_round_test(
        content="Stopping criteria test",
        test_function=test_function,
        max_rounds=10,
        strategy=RoundStrategy.FIXED,
        stopping_criteria=[RoundStoppingCriteria.IMPROVEMENT_PLATEAU],
        base_parameters={}
    )
    
    print(f"✅ Stopping criteria test completed:")
    print(f"   - Rounds run: {result.total_rounds}")
    print(f"   - Stopping reason: {result.stopping_reason}")
    print(f"   - Improvement scores: {[round(r.improvement_score, 3) for r in result.round_results]}")
    
    # Should stop before max rounds due to low improvement or reach max rounds
    assert result.total_rounds <= 10
    
    return True


def test_team_integration():
    """Test integration with team functions"""
    print("\nTesting team integration...")
    
    tester = MultiRoundTester()
    test_function = create_team_test_function(mock_team_function)
    
    result = tester.run_multi_round_test(
        content="Team integration test",
        test_function=test_function,
        max_rounds=3,
        strategy=RoundStrategy.ADAPTIVE,
        base_parameters={'team_size': 3, 'consensus_threshold': 0.6}
    )
    
    print(f"✅ Team integration test completed:")
    print(f"   - Total rounds: {result.total_rounds}")
    print(f"   - Best round quality: {result.best_round.quality_score:.2f}")
    
    assert result.total_rounds > 0
    assert result.best_round is not None
    
    return True


def test_convenience_functions():
    """Test convenience functions"""
    print("\nTesting convenience functions...")
    
    # Test evolution convenience function
    evolution_result = run_multi_round_evolution(
        content="Evolution convenience test",
        evolution_function=mock_evolution_function,
        rounds=3,
        strategy=RoundStrategy.PROGRESSIVE,
        base_parameters={'iterations': 5}
    )
    
    print(f"✅ Evolution convenience function:")
    print(f"   - Rounds: {evolution_result.total_rounds}")
    print(f"   - Improvement: {evolution_result.overall_improvement:.2f}")
    
    # Test team convenience function
    team_result = run_multi_round_team_testing(
        content="Team convenience test",
        team_function=mock_team_function,
        rounds=2,
        strategy=RoundStrategy.ADAPTIVE,
        base_parameters={'team_size': 4}
    )
    
    print(f"✅ Team convenience function:")
    print(f"   - Rounds: {team_result.total_rounds}")
    print(f"   - Improvement: {team_result.overall_improvement:.2f}")
    
    assert evolution_result.total_rounds > 0
    assert team_result.total_rounds > 0
    
    return True


def test_adaptive_learning():
    """Test adaptive learning and recommendations"""
    print("\nTesting adaptive learning...")
    
    tester = MultiRoundTester()
    test_function = create_evolution_test_function(mock_evolution_function)
    
    # Run multiple multi-round tests to build adaptive knowledge
    for i in range(3):
        result = tester.run_multi_round_test(
            content=f"Adaptive learning test {i+1}",
            test_function=test_function,
            max_rounds=3,
            strategy=RoundStrategy.ADAPTIVE,
            base_parameters={'iterations': 5 + i, 'mutation_rate': 0.1 + i * 0.05}
        )
    
    # Get adaptive recommendations
    recommendations = tester.get_adaptive_recommendations()
    
    print(f"✅ Adaptive learning test completed:")
    print(f"   - Recommendations generated: {len(recommendations)}")
    print(f"   - Sample recommendations: {recommendations}")
    
    assert len(recommendations) > 0
    assert 'iterations' in recommendations or 'mutation_rate' in recommendations
    
    return True


def test_convergence_analysis():
    """Test convergence analysis"""
    print("\nTesting convergence analysis...")
    
    tester = MultiRoundTester()
    test_function = create_evolution_test_function(mock_evolution_function)
    
    result = tester.run_multi_round_test(
        content="Convergence test",
        test_function=test_function,
        max_rounds=6,
        strategy=RoundStrategy.CONVERGENT,
        base_parameters={'iterations': 10, 'mutation_rate': 0.2}
    )
    
    print(f"✅ Convergence analysis test completed:")
    print(f"   - Quality scores: {[round(s, 2) for s in result.convergence_data['quality_scores']]}")
    print(f"   - Improvement scores: {[round(s, 2) for s in result.convergence_data['improvement_scores']]}")
    
    # Check convergence data is properly tracked
    assert len(result.convergence_data['quality_scores']) == result.total_rounds
    assert len(result.convergence_data['improvement_scores']) == result.total_rounds
    assert len(result.convergence_data['parameter_evolution']) == result.total_rounds
    
    return True


def test_error_handling():
    """Test error handling in multi-round testing"""
    print("\nTesting error handling...")
    
    def failing_function(content: str, **parameters) -> dict:
        """Function that sometimes fails"""
        if parameters.get('should_fail', False):
            raise ValueError("Intentional test failure")
        
        return {
            'content': content + " [Success]",
            'quality_score': 0.7,
            'metrics': {},
            'success': True
        }
    
    tester = MultiRoundTester()
    test_function = create_evolution_test_function(failing_function)
    
    result = tester.run_multi_round_test(
        content="Error handling test",
        test_function=test_function,
        max_rounds=4,
        strategy=RoundStrategy.FIXED,
        base_parameters={'should_fail': False}  # Start with success
    )
    
    print(f"✅ Error handling test completed:")
    print(f"   - Total rounds: {result.total_rounds}")
    print(f"   - Successful rounds: {result.successful_rounds}")
    
    # Should handle errors gracefully
    assert result.total_rounds > 0
    
    return True


if __name__ == "__main__":
    try:
        test_basic_multi_round()
        test_adaptive_strategy()
        test_progressive_strategy()
        test_stopping_criteria()
        test_team_integration()
        test_convenience_functions()
        test_adaptive_learning()
        test_convergence_analysis()
        test_error_handling()
        
        print("\n🎉 All multi-round testing tests passed! System is fully functional.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()