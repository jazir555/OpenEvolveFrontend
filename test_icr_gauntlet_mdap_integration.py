"""
Test Suite for Gauntlet-ICR and MDAP-ICR Integrations

Tests the new ICR integration methods added to:
- GauntletSystem (sovereign_gauntlets.py)
- AdaptiveMDAPAllocator (adaptive_mdap/allocators/resource_allocator.py)
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from typing import Dict, Any, List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestGauntletICRIntegration(unittest.TestCase):
    """Test GauntletSystem ICR integration methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        from sovereign_gauntlets import GauntletSystem, DecompositionPlan
        from sovereign_data_models import SubProblem, generate_id
        
        # Create mock openevolve client
        self.mock_client = Mock()
        
        # Create mock decomposition plan
        sub_problem = SubProblem(
            id=generate_id("sub"),
            parent_id=generate_id("parent"),
            title="Test Sub-problem",
            description="Test sub-problem for gauntlet validation",
            type="ANALYSIS",
            complexity_score={"overall_complexity": 5.0}
        )
        
        self.mock_plan = Mock(spec=DecompositionPlan)
        self.mock_plan.id = generate_id("plan")
        self.mock_plan.sub_problems = [sub_problem]
        
        # Create gauntlet system with ICR integration
        self.gauntlet_system = GauntletSystem(
            openevolve_client=self.mock_client,
            track_patterns=True
        )
    
    def test_gauntlet_system_with_icr_init(self):
        """Test GauntletSystem initialization with ICR parameters."""
        from sovereign_gauntlets import GauntletSystem
        
        # Test with ICR parameters
        gs = GauntletSystem(
            openevolve_client=self.mock_client,
            refinement_coordinator=Mock(),
            track_patterns=True
        )
        
        self.assertIsNotNone(gs.refinement_coordinator)
        self.assertTrue(gs.track_patterns)
        self.assertEqual(gs._gauntlet_patterns, {})
        self.assertEqual(gs._gauntlet_metrics, {})
    
    def test_store_gauntlet_pattern(self):
        """Test storing gauntlet execution patterns."""
        from sovereign_gauntlets import ValidationResult
        
        # Create mock results
        results = {
            'coherence': Mock(
                passed=True,
                score=0.8,
                feedback="Test feedback"
            ),
            'completeness': Mock(
                passed=False,
                score=0.5,
                feedback="Missing coverage"
            )
        }
        
        # Store pattern
        self.gauntlet_system._store_gauntlet_pattern(
            self.mock_plan.id,
            results
        )
        
        # Verify pattern was stored
        failed_key = ('completeness',)
        self.assertIn(failed_key, self.gauntlet_system._gauntlet_patterns)
        
        # Verify metrics were updated
        self.assertIn('coherence', self.gauntlet_system._gauntlet_metrics)
        self.assertIn('completeness', self.gauntlet_system._gauntlet_metrics)
        
        # Check coherence metrics
        coherence_metrics = self.gauntlet_system._gauntlet_metrics['coherence']
        self.assertEqual(coherence_metrics['total_runs'], 1)
        self.assertEqual(coherence_metrics['pass_count'], 1)
        self.assertEqual(coherence_metrics['fail_count'], 0)
    
    def test_get_gauntlet_effectiveness(self):
        """Test getting gauntlet effectiveness metrics."""
        from sovereign_gauntlets import ValidationResult
        
        # Store some patterns first
        results = {
            'coherence': Mock(passed=True, score=0.8),
            'feasibility': Mock(passed=False, score=0.4)
        }
        self.gauntlet_system._store_gauntlet_pattern("plan1", results)
        
        results2 = {
            'coherence': Mock(passed=True, score=0.9),
            'feasibility': Mock(passed=True, score=0.7)
        }
        self.gauntlet_system._store_gauntlet_pattern("plan2", results2)
        
        # Get effectiveness
        effectiveness = self.gauntlet_system.get_gauntlet_effectiveness()
        
        self.assertIn('coherence', effectiveness)
        self.assertIn('feasibility', effectiveness)
        
        # Coherence should have 100% pass rate
        self.assertEqual(effectiveness['coherence']['pass_rate'], 1.0)
        self.assertEqual(effectiveness['coherence']['avg_score'], 0.85)
        
        # Feasibility should have 50% pass rate
        self.assertEqual(effectiveness['feasibility']['pass_rate'], 0.5)
    
    def test_get_failure_patterns(self):
        """Test getting learned failure patterns."""
        from sovereign_gauntlets import ValidationResult
        
        # Store patterns with different failure combinations
        results1 = {
            'coherence': Mock(passed=False, score=0.3),
            'completeness': Mock(passed=True, score=0.8)
        }
        self.gauntlet_system._store_gauntlet_pattern("plan1", results1)
        
        results2 = {
            'coherence': Mock(passed=False, score=0.4),
            'completeness': Mock(passed=True, score=0.9)
        }
        self.gauntlet_system._store_gauntlet_pattern("plan2", results2)
        
        patterns = self.gauntlet_system.get_failure_patterns()
        
        # Should have pattern for failed gauntlets
        failed_key = ('coherence',)
        self.assertIn(failed_key, patterns)
        self.assertEqual(len(patterns[failed_key]), 2)
    
    def test_suggest_optimal_gauntlets(self):
        """Test suggesting optimal gauntlet configuration."""
        from sovereign_gauntlets import ValidationResult
        
        # Store patterns to build up history
        results = {
            'coherence': Mock(passed=False, score=0.4),
            'feasibility': Mock(passed=True, score=0.8)
        }
        for i in range(5):
            self.gauntlet_system._store_gauntlet_pattern(f"plan{i}", results)
        
        # Get suggestions for different complexity levels
        low_complexity_suggestions = self.gauntlet_system.suggest_optimal_gauntlets(
            complexity=0.3
        )
        high_complexity_suggestions = self.gauntlet_system.suggest_optimal_gauntlets(
            complexity=0.8
        )
        
        # Should always include base gauntlets
        self.assertIn('coherence', low_complexity_suggestions)
        self.assertIn('completeness', high_complexity_suggestions)
        
        # High complexity should include adaptive/hierarchical
        self.assertIn('adaptive', high_complexity_suggestions)
        self.assertIn('hierarchical', high_complexity_suggestions)
    
    def test_adapt_gauntlet_config(self):
        """Test adapting gauntlet configuration based on patterns."""
        from sovereign_gauntlets import ValidationResult
        
        # Store patterns showing high pass rate
        results = {
            'coherence': Mock(passed=True, score=0.95),
        }
        for i in range(10):
            self.gauntlet_system._store_gauntlet_pattern(f"plan{i}", results)
        
        # Adapt config for lenient gauntlet
        config = self.gauntlet_system.adapt_gauntlet_config(
            'coherence',
            {'complexity': 0.5}
        )
        
        # Should raise min_score since gauntlet is too lenient
        self.assertIn('min_score', config)
        self.assertGreater(config['min_score'], 0.5)
    
    def test_clear_patterns(self):
        """Test clearing stored patterns and metrics."""
        from sovereign_gauntlets import ValidationResult
        
        # Store some patterns
        results = {'coherence': Mock(passed=True, score=0.8)}
        self.gauntlet_system._store_gauntlet_pattern("plan1", results)
        
        # Verify they exist
        self.assertTrue(len(self.gauntlet_system._gauntlet_patterns) > 0)
        self.assertTrue(len(self.gauntlet_system._gauntlet_metrics) > 0)
        
        # Clear
        self.gauntlet_system.clear_patterns()
        
        # Verify cleared
        self.assertEqual(self.gauntlet_system._gauntlet_patterns, {})
        self.assertEqual(self.gauntlet_system._gauntlet_metrics, {})
    
    def test_run_with_icr_refinement_basic(self):
        """Test basic run_with_icr_refinement functionality."""
        from sovereign_gauntlets import ValidationResult
        
        # Mock gauntlet execution
        with patch.object(self.gauntlet_system, 'run_decomposition_gauntlets') as mock_run:
            mock_run.return_value = {
                'coherence': Mock(passed=True, score=0.85),
                'completeness': Mock(passed=True, score=0.80)
            }
            
            result = self.gauntlet_system.run_with_icr_refinement(
                self.mock_plan,
                max_refinement_cycles=2
            )
            
            # Should converge immediately since all gauntlets pass
            self.assertEqual(result['total_cycles'], 1)
            self.assertTrue(result['converged'])
            self.assertTrue(result['final_quality'] > 0.8)
    
    def test_run_with_icr_refinement_with_refinement(self):
        """Test run_with_icr_refinement when refinement is triggered."""
        from sovereign_gauntlets import ValidationResult
        
        refinement_coordinator = Mock()
        refinement_coordinator.generate_smart_refinement_strategy.return_value = {
            'strategy_type': 'test'
        }
        refinement_coordinator.generate_refinement_plan.return_value = Mock(
            id='test_plan',
            issues=[],
            improvements=['Fix test issue']
        )
        refinement_coordinator.execute_refinement.return_value = (
            self.mock_plan,
            Mock(quality_improvement=0.1, issues_resolved=1)
        )
        
        self.gauntlet_system.refinement_coordinator = refinement_coordinator
        
        call_count = [0]
        def create_results():
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: some gauntlets fail
                return {
                    'coherence': Mock(passed=False, score=0.4),
                    'completeness': Mock(passed=True, score=0.8)
                }
            else:
                # Second call: all pass
                return {
                    'coherence': Mock(passed=True, score=0.85),
                    'completeness': Mock(passed=True, score=0.80)
                }
        
        with patch.object(self.gauntlet_system, 'run_decomposition_gauntlets', side_effect=create_results):
            with patch.object(self.gauntlet_system, 'process_gauntlet_feedback', return_value=[]):
                result = self.gauntlet_system.run_with_icr_refinement(
                    self.mock_plan,
                    max_refinement_cycles=2
                )
        
        # Should have refined and converged
        self.assertTrue(result['converged'])
        self.assertGreaterEqual(result['total_cycles'], 2)
        
        # Verify refinement coordinator was called
        refinement_coordinator.generate_smart_refinement_strategy.assert_called()
        refinement_coordinator.execute_refinement.assert_called()


class TestMDAPICRIntegration(unittest.TestCase):
    """Test AdaptiveMDAPAllocator ICR integration methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator
        from adaptive_mdap.core.types import SolveStrategy
        
        self.allocator = AdaptiveMDAPAllocator(
            enable_learning=True,
            enable_context_aware=True
        )
    
    def test_detect_strategy_patterns_insufficient_data(self):
        """Test pattern detection with insufficient data."""
        patterns = self.allocator.detect_strategy_patterns()
        
        self.assertFalse(patterns['has_enough_data'])
        self.assertIn('Need at least 10', patterns['message'])
    
    def test_detect_strategy_patterns_with_data(self):
        """Test pattern detection with sufficient learning data."""
        from adaptive_mdap.core.types import SolveStrategy
        
        # Record some outcomes
        for i in range(15):
            complexity = 0.3 + (i % 5) * 0.15  # Various complexity scores
            success = i % 4 != 0  # 75% success rate
            self.allocator.record_outcome(
                complexity_score=complexity,
                strategy=SolveStrategy.MDAP_LIGHT,
                success=success,
                cost=3.0,
                quality=0.7 if success else 0.4
            )
        
        patterns = self.allocator.detect_strategy_patterns()
        
        self.assertTrue(patterns['has_enough_data'])
        self.assertEqual(patterns['total_samples'], 15)
        self.assertIn('complexity_ranges', patterns)
        self.assertIn('strategy_effectiveness', patterns)
    
    def test_record_outcome(self):
        """Test recording strategy execution outcomes."""
        from adaptive_mdap.core.types import SolveStrategy
        
        initial_count = len(self.allocator._learning_data)
        
        self.allocator.record_outcome(
            complexity_score=0.5,
            strategy=SolveStrategy.MDAP_MEDIUM,
            success=True,
            cost=5.0,
            quality=0.85
        )
        
        self.assertEqual(len(self.allocator._learning_data), initial_count + 1)
        
        # Verify data structure
        last_record = self.allocator._learning_data[-1]
        self.assertEqual(last_record['complexity_score'], 0.5)
        self.assertEqual(last_record['strategy'], 'MDAP_MEDIUM')
        self.assertTrue(last_record['success'])
        self.assertEqual(last_record['quality'], 0.85)
    
    def test_adapt_thresholds_from_patterns(self):
        """Test adapting thresholds based on patterns."""
        from adaptive_mdap.core.types import SolveStrategy
        
        # Record outcomes showing DIRECT works well in medium complexity
        for i in range(10):
            self.allocator.record_outcome(
                complexity_score=0.35,  # medium-low band
                strategy=SolveStrategy.DIRECT,
                success=True,
                cost=1.0,
                quality=0.85
            )
        
        # Record outcomes showing MAKER struggles in high complexity
        for i in range(10):
            self.allocator.record_outcome(
                complexity_score=0.85,  # high band
                strategy=SolveStrategy.MAKER_FULL,
                success=False,  # Low success
                cost=7.5,
                quality=0.45
            )
        
        old_thresholds = list(self.allocator.thresholds)
        
        new_thresholds, changes = self.allocator.adapt_thresholds_from_patterns()
        
        # Should have made some changes
        self.assertTrue(len(changes) > 0)
        
        # Verify thresholds were updated
        self.assertNotEqual(old_thresholds, new_thresholds)
    
    def test_get_strategy_for_context(self):
        """Test comprehensive strategy recommendation."""
        from adaptive_mdap.core.types import SolveStrategy
        
        result = self.allocator.get_strategy_for_context(
            complexity_score=0.5,
            context=None,
            use_icr_patterns=True
        )
        
        # Should have all expected fields
        self.assertIn('complexity_score', result)
        self.assertIn('recommended_strategy', result)
        self.assertIn('n_agents', result)
        self.assertIn('k_ahead', result)
        self.assertIn('reasoning', result)
        
        # Without enough data, icr_insights should indicate no data
        if result.get('icr_insights') is not None:
            self.assertIn('sample_count', result['icr_insights'])
    
    def test_record_gauntlet_feedback(self):
        """Test recording gauntlet feedback for MDAP learning."""
        from adaptive_mdap.core.types import SolveStrategy
        
        initial_count = len(self.allocator._learning_data)
        
        gauntlet_results = {
            'coherence': {'score': 0.8, 'passed': True},
            'completeness': {'score': 0.6, 'passed': True},
            'feasibility': {'score': 0.4, 'passed': False}
        }
        
        self.allocator.record_gauntlet_feedback(
            complexity_score=0.5,
            strategy=SolveStrategy.MDAP_MEDIUM,
            gauntlet_results=gauntlet_results,
            refinement_applied=True
        )
        
        # Should have recorded an outcome
        self.assertEqual(len(self.allocator._learning_data), initial_count + 1)
        
        # Check the recorded data
        last_record = self.allocator._learning_data[-1]
        self.assertEqual(last_record['complexity_score'], 0.5)
        self.assertEqual(last_record['strategy'], 'MDAP_MEDIUM')
        # Should be considered success since 2/3 gauntlets passed
        self.assertTrue(last_record['success'])
        # Quality should be average of scores
        self.assertAlmostEqual(last_record['quality'], (0.8 + 0.6 + 0.4) / 3)
    
    def test_enable_learning_flag(self):
        """Test that learning can be disabled."""
        from adaptive_mdap.core.types import SolveStrategy
        
        # Create allocator without learning
        allocator_no_learning = AdaptiveMDAPAllocator(enable_learning=False)
        
        # Record outcome - should be ignored
        allocator_no_learning.record_outcome(
            complexity_score=0.5,
            strategy=SolveStrategy.DIRECT,
            success=True,
            cost=1.0,
            quality=0.8
        )
        
        # No data should be recorded
        self.assertEqual(len(allocator_no_learning._learning_data), 0)


class TestICRIntegrationWorkflow(unittest.TestCase):
    """Test end-to-end ICR integration workflows."""
    
    def test_gauntlet_to_mdap_feedback_loop(self):
        """Test the feedback loop from GauntletSystem to MDAP allocator."""
        from sovereign_gauntlets import GauntletSystem, ValidationResult
        from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator
        from adaptive_mdap.core.types import SolveStrategy
        from sovereign_data_models import SubProblem, generate_id
        
        # Set up GauntletSystem with mock client
        mock_client = Mock()
        gauntlet_system = GauntletSystem(
            openevolve_client=mock_client,
            track_patterns=True
        )
        
        # Set up MDAP allocator with learning enabled
        mdap_allocator = AdaptiveMDAPAllocator(enable_learning=True)
        
        # Create mock plan
        sub_problem = SubProblem(
            id=generate_id("sub"),
            parent_id=generate_id("parent"),
            title="Test Problem",
            description="Integration test problem",
            type="ANALYSIS",
            complexity_score={"overall_complexity": 6.0}
        )
        mock_plan = Mock()
        mock_plan.id = generate_id("plan")
        mock_plan.sub_problems = [sub_problem]
        
        # Simulate gauntlet run with results
        gauntlet_results = {
            'coherence': Mock(passed=True, score=0.85),
            'completeness': Mock(passed=False, score=0.45),  # Failed
            'feasibility': Mock(passed=True, score=0.75),
            'dependency': Mock(passed=True, score=0.70)
        }
        
        # Record gauntlet feedback to MDAP allocator
        mdap_allocator.record_gauntlet_feedback(
            complexity_score=0.6,
            strategy=SolveStrategy.MDAP_MEDIUM,
            gauntlet_results=gauntlet_results,
            refinement_applied=True
        )
        
        # Verify MDAP allocator recorded the outcome
        self.assertEqual(len(mdap_allocator._learning_data), 1)
        
        # Get strategy recommendation with ICR insights
        strategy_rec = mdap_allocator.get_strategy_for_context(
            complexity_score=0.6,
            use_icr_patterns=True
        )
        
        self.assertIn('recommended_strategy', strategy_rec)
        self.assertIn('icr_insights', strategy_rec)
    
    def test_threshold_adaptation_workflow(self):
        """Test the threshold adaptation workflow."""
        from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator
        from adaptive_mdap.core.types import SolveStrategy
        
        allocator = AdaptiveMDAPAllocator(enable_learning=True)
        
        # Simulate learning data showing DIRECT is effective in medium complexity
        for _ in range(10):
            allocator.record_outcome(
                complexity_score=0.35,  # Just above t1
                strategy=SolveStrategy.DIRECT,
                success=True,
                cost=1.0,
                quality=0.9
            )
        
        # Simulate learning data showing MAKER_FULL is struggling in high complexity
        for _ in range(10):
            allocator.record_outcome(
                complexity_score=0.85,  # High complexity
                strategy=SolveStrategy.MAKER_FULL,
                success=False,
                cost=7.5,
                quality=0.4
            )
        
        # Detect patterns
        patterns = allocator.detect_strategy_patterns()
        
        self.assertTrue(patterns['has_enough_data'])
        
        # Adapt thresholds
        new_thresholds, changes = allocator.adapt_thresholds_from_patterns(patterns)
        
        # Verify changes were made
        self.assertIsInstance(new_thresholds, list)
        self.assertEqual(len(new_thresholds), 4)  # 4 thresholds for 5 bands
        self.assertTrue(len(changes) > 0)
        
        # Verify thresholds are valid (strictly increasing)
        for i in range(len(new_thresholds) - 1):
            self.assertLess(new_thresholds[i], new_thresholds[i + 1])


if __name__ == '__main__':
    unittest.main()
