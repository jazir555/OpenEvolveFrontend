"""Tests for BudgetEnforcer - verifies budget enforcement functionality.

These tests verify that:
1. BudgetEnforcer correctly tracks budget status
2. Warning threshold (70%) logs but allows continuation
3. Critical threshold (90%) stops execution
4. Budget exceeded (100%+) stops execution
5. Stop requests work correctly
6. Integration with execution monitor works
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openevolve_pes_enhanced.budget_enforcer import (
    BudgetEnforcer,
    BudgetCheckResult,
    BudgetEnforcedResult
)
from openevolve_pes_enhanced.cost_optimizer import BudgetTracker, BudgetStatus


class TestBudgetEnforcer(unittest.TestCase):
    """Test BudgetEnforcer functionality."""
    
    def test_initialization(self):
        """Test BudgetEnforcer initializes correctly."""
        tracker = BudgetTracker(
            max_cost_usd=10.0,
            max_tokens=100000,
            max_time_ms=300000
        )
        
        enforcer = BudgetEnforcer(
            budget_tracker=tracker,
            warning_threshold=0.70,
            critical_threshold=0.90
        )
        
        self.assertEqual(enforcer.warning_threshold, 0.70)
        self.assertEqual(enforcer.critical_threshold, 0.90)
        self.assertFalse(enforcer._stop_requested)
    
    def test_check_budget_ok(self):
        """Test budget check returns OK when under threshold."""
        tracker = BudgetTracker(max_cost_usd=10.0)
        enforcer = BudgetEnforcer(tracker)
        
        # Budget is at 0%, should be OK
        can_continue, reason = enforcer.check_budget()
        
        self.assertTrue(can_continue)
        self.assertIn("OK", reason)
    
    def test_warning_threshold(self):
        """Test warning threshold (70%) logs but allows continuation."""
        tracker = BudgetTracker(
            max_cost_usd=10.0,
            warning_threshold=0.70,
            critical_threshold=0.90
        )
        
        # Simulate 75% cost usage (above warning, below critical)
        tracker.cost_used = 7.5
        
        enforcer = BudgetEnforcer(
            tracker,
            warning_threshold=0.70,
            critical_threshold=0.90
        )
        
        with self.assertLogs(level='WARNING') as log_context:
            can_continue, reason = enforcer.check_budget()
        
        # Should allow continuation but log warning
        self.assertTrue(can_continue)
        self.assertIn("OK", reason)
        
        # Warning should have been logged
        self.assertTrue(
            any("WARNING" in msg or "warning" in msg.lower() for msg in log_context.output)
        )
    
    def test_critical_threshold(self):
        """Test critical threshold (90%) stops execution."""
        tracker = BudgetTracker(
            max_cost_usd=10.0,
            warning_threshold=0.70,
            critical_threshold=0.90
        )
        
        # Simulate 95% cost usage (above critical)
        tracker.cost_used = 9.5
        
        enforcer = BudgetEnforcer(
            tracker,
            warning_threshold=0.70,
            critical_threshold=0.90
        )
        
        with self.assertLogs(level='WARNING') as log_context:
            can_continue, reason = enforcer.check_budget()
        
        # Should stop execution
        self.assertFalse(can_continue)
        self.assertIn("CRITICAL", reason)
        self.assertIn("cost 95.0%", reason)
    
    def test_budget_exceeded(self):
        """Test budget exceeded (100%+) stops execution."""
        tracker = BudgetTracker(max_cost_usd=10.0)
        
        # Simulate 110% cost usage (over budget)
        tracker.cost_used = 11.0
        
        enforcer = BudgetEnforcer(tracker)
        
        can_continue, reason = enforcer.check_budget()
        
        # Should stop execution
        self.assertFalse(can_continue)
        self.assertIn("exceeded", reason.lower())
    
    def test_tokens_critical_threshold(self):
        """Test critical threshold based on token usage."""
        tracker = BudgetTracker(
            max_cost_usd=10.0,
            max_tokens=100000,
            warning_threshold=0.70,
            critical_threshold=0.90
        )
        
        # Simulate 95% token usage (above critical)
        tracker.tokens_used = 95000
        
        enforcer = BudgetEnforcer(
            tracker,
            warning_threshold=0.70,
            critical_threshold=0.90
        )
        
        can_continue, reason = enforcer.check_budget()
        
        # Should stop execution
        self.assertFalse(can_continue)
        self.assertIn("CRITICAL", reason)
        self.assertIn("tokens 95.0%", reason)
    
    def test_time_critical_threshold(self):
        """Test critical threshold based on time usage."""
        tracker = BudgetTracker(
            max_cost_usd=10.0,
            max_time_ms=100000,
            warning_threshold=0.70,
            critical_threshold=0.90
        )
        
        enforcer = BudgetEnforcer(
            tracker,
            warning_threshold=0.70,
            critical_threshold=0.90
        )
        
        # Manually set start time to simulate time elapsed
        import time
        enforcer.budget_tracker.start_time = (time.time() * 1000) - 95000  # 95% elapsed
        
        can_continue, reason = enforcer.check_budget()
        
        # Should stop execution
        self.assertFalse(can_continue)
        self.assertIn("CRITICAL", reason)
        self.assertIn("time 95.0%", reason)
    
    def test_request_stop(self):
        """Test manual stop request."""
        tracker = BudgetTracker(max_cost_usd=10.0)
        enforcer = BudgetEnforcer(tracker)
        
        # Request stop
        enforcer.request_stop("Test stop reason")
        
        # Should stop immediately
        can_continue, reason = enforcer.check_budget()
        
        self.assertFalse(can_continue)
        self.assertIn("Stop requested", reason)
        self.assertIn("Test stop reason", reason)
    
    def test_request_stop_with_execution_monitor(self):
        """Test stop request propagates to execution monitor."""
        tracker = BudgetTracker(max_cost_usd=10.0)
        
        # Mock execution monitor with early stopping
        mock_early_stopping = MagicMock()
        mock_early_stopping.stopped = False
        mock_early_stopping.stop_reason = None
        
        mock_monitor = MagicMock()
        mock_monitor.early_stopping = mock_early_stopping
        
        enforcer = BudgetEnforcer(tracker, execution_monitor=mock_monitor)
        
        # Request stop
        enforcer.request_stop("Budget exhausted")
        
        # Should propagate to early stopping controller
        self.assertTrue(mock_early_stopping.stopped)
        self.assertIn("Budget", mock_early_stopping.stop_reason)
    
    def test_get_status(self):
        """Test get_status returns detailed information."""
        tracker = BudgetTracker(max_cost_usd=10.0)
        tracker.cost_used = 5.0  # 50% used
        
        enforcer = BudgetEnforcer(tracker)
        
        status = enforcer.get_status()
        
        self.assertIsInstance(status, BudgetCheckResult)
        self.assertTrue(status.can_continue)
        self.assertEqual(status.status, "ok")
        self.assertEqual(status.percent_used, 0.5)
    
    def test_create_callback(self):
        """Test create_callback returns callable function."""
        tracker = BudgetTracker(max_cost_usd=10.0)
        enforcer = BudgetEnforcer(tracker)
        
        callback = enforcer.create_callback()
        
        self.assertTrue(callable(callback))
        
        # Callback should return tuple
        result = callback()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
    
    def test_no_budget_tracker(self):
        """Test behavior when no budget tracker provided."""
        enforcer = BudgetEnforcer(budget_tracker=None)
        
        can_continue, reason = enforcer.check_budget()
        
        # Should allow continuation
        self.assertTrue(can_continue)
        self.assertIn("No budget tracking", reason)
    
    def test_warning_logged_only_once(self):
        """Test warning is only logged once when threshold crossed."""
        tracker = BudgetTracker(
            max_cost_usd=10.0,
            warning_threshold=0.70,
            critical_threshold=0.90
        )
        tracker.cost_used = 7.5  # 75% - above warning
        
        enforcer = BudgetEnforcer(
            tracker,
            warning_threshold=0.70,
            critical_threshold=0.90
        )
        
        # First check should log warning
        with self.assertLogs(level='WARNING') as log_context:
            enforcer.check_budget()
        
        warning_count = len(log_context.output)
        
        # Second check should NOT log another warning
        with self.assertLogs(level='WARNING') as log_context:
            enforcer.check_budget()
            # Need to emit at least one log to avoid assertion error
            import logging
            logging.getLogger().warning("Dummy log")
        
        # Should only have the dummy log, not a budget warning
        self.assertEqual(len(log_context.output), 1)
        self.assertIn("Dummy log", log_context.output[0])


class TestBudgetEnforcedResult(unittest.TestCase):
    """Test BudgetEnforcedResult data class."""
    
    def test_initialization(self):
        """Test BudgetEnforcedResult initializes correctly."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.best_fitness = 0.95
        mock_result.total_evaluations = 100
        mock_result.code = "def test(): pass"
        
        budget_status = BudgetCheckResult(
            can_continue=False,
            reason="Budget exceeded",
            status="exceeded",
            percent_used=1.1
        )
        
        result = BudgetEnforcedResult(
            original_result=mock_result,
            stopped_early=True,
            stop_reason="Budget exceeded",
            final_budget_status=budget_status,
            iterations_completed=5
        )
        
        self.assertEqual(result.original_result, mock_result)
        self.assertTrue(result.stopped_early)
        self.assertEqual(result.stop_reason, "Budget exceeded")
        self.assertEqual(result.iterations_completed, 5)
        
        # Should copy attributes from original result
        self.assertTrue(result.success)
        self.assertEqual(result.best_fitness, 0.95)
        self.assertEqual(result.total_evaluations, 100)
    
    def test_to_dict(self):
        """Test to_dict conversion."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.best_fitness = 0.95
        mock_result.total_evaluations = 100
        
        budget_status = BudgetCheckResult(
            can_continue=False,
            reason="Budget exceeded",
            status="exceeded",
            percent_used=1.1
        )
        
        result = BudgetEnforcedResult(
            original_result=mock_result,
            stopped_early=True,
            stop_reason="Budget exceeded",
            final_budget_status=budget_status,
            iterations_completed=5
        )
        
        d = result.to_dict()
        
        self.assertIn("success", d)
        self.assertIn("stopped_early", d)
        self.assertIn("stop_reason", d)
        self.assertIn("iterations_completed", d)
        self.assertIn("best_fitness", d)
        self.assertIn("budget_status", d)
        
        self.assertTrue(d["success"])
        self.assertTrue(d["stopped_early"])
        self.assertEqual(d["stop_reason"], "Budget exceeded")
        self.assertEqual(d["iterations_completed"], 5)


class TestBudgetEnforcerIntegration(unittest.TestCase):
    """Test BudgetEnforcer integration scenarios."""
    
    def test_typical_evolution_scenario(self):
        """Test typical evolution scenario with budget enforcement."""
        tracker = BudgetTracker(
            max_cost_usd=10.0,
            max_tokens=100000,
            max_time_ms=300000,
            warning_threshold=0.70,
            critical_threshold=0.90
        )
        
        enforcer = BudgetEnforcer(tracker)
        
        # Simulate evolution iterations
        iteration = 0
        max_iterations = 10
        
        while iteration < max_iterations:
            # Check budget before each iteration
            can_continue, reason = enforcer.check_budget()
            
            if not can_continue:
                break
            
            # Simulate work
            tracker.record_tokens(5000, 2000)  # Use some tokens
            iteration += 1
        
        # Should complete all iterations (budget not exceeded)
        self.assertEqual(iteration, max_iterations)
    
    def test_budget_stop_during_evolution(self):
        """Test evolution stops when budget exceeded."""
        tracker = BudgetTracker(
            max_cost_usd=1.0,  # Very low budget
            warning_threshold=0.70,
            critical_threshold=0.90
        )
        
        enforcer = BudgetEnforcer(tracker)
        
        # Simulate evolution with high token usage
        iteration = 0
        max_iterations = 10
        
        while iteration < max_iterations:
            can_continue, reason = enforcer.check_budget()
            
            if not can_continue:
                break
            
            # Use lots of tokens (will exceed budget quickly)
            tracker.record_tokens(50000, 20000)
            iteration += 1
        
        # Should have stopped before completing all iterations
        self.assertLess(iteration, max_iterations)
    
    def test_multiple_budget_metrics(self):
        """Test enforcement considers all metrics (cost, tokens, time)."""
        tracker = BudgetTracker(
            max_cost_usd=10.0,
            max_tokens=1000,  # Very low token limit
            max_time_ms=1000000,
            warning_threshold=0.70,
            critical_threshold=0.90
        )
        
        enforcer = BudgetEnforcer(tracker)
        
        # Use tokens to exceed limit (but not cost)
        tracker.record_tokens(500, 500)  # 100% of tokens
        
        can_continue, reason = enforcer.check_budget()
        
        # Should stop due to tokens even though cost is low
        self.assertFalse(can_continue)
        self.assertIn("tokens", reason.lower())


if __name__ == "__main__":
    unittest.main()
