"""
End-to-end tests for the full Adaptive MDAP system integration.
"""

import pytest
from sub_problem_solver import SubProblemSolver
from sovereign_data_models import SubProblem, SubProblemType, ComplexityScore

def test_full_system_adaptive_solve():
    """Test the full system from SubProblemSolver down to Adaptive MDAP."""
    # Mock OpenEvolveClient
    class MockClient:
        def evolve(self, **kwargs):
            class MockResult:
                success = True
                best_code = "print('Hello World')"
            return MockResult()
    
    # Initialize solver with adaptive enabled
    solver = SubProblemSolver(openevolve_client=MockClient(), enable_adaptive_allocation=True)
    
    # Verify adaptive is enabled
    assert solver.enable_adaptive_allocation is True
    assert solver.adaptive_integration is not None
    
    # Create a sub-problem
    sp = SubProblem(
        id="e2e-test-1",
        parent_id="root",
        title="Test Subproblem",
        description="A simple task to verify the full integration chain.",
        type=SubProblemType.IMPLEMENTATION,
        complexity_score=ComplexityScore(
            explanation="Test",
            cognitive_complexity=2.0,
            computational_complexity=1.0,
            domain_complexity=1.0,
            integration_complexity=1.0,
            overall_complexity=1.5
        ),
        dependencies=[]
    )
    
    # Solve it
    attempt = solver.solve(sp)
    
    # Verify the results
    assert attempt is not None
    assert attempt.sub_problem_id == sp.id
    assert "adaptive" in attempt.approach
    assert attempt.status == "solved"
    assert "strategy" in attempt.solution_content

def test_adaptive_fallback_to_standard():
    """Test that it falls back to standard solve if adaptive fails."""
    # Initialize solver but break adaptive integration
    solver = SubProblemSolver(openevolve_client=None, enable_adaptive_allocation=True)
    
    # Manually break it
    solver.adaptive_integration = None
    
    # It should still be able to try solving (and fail due to missing client, but show it got past adaptive)
    sp = SubProblem(
        id="fallback-test",
        parent_id="root",
        title="Test Fallback",
        description="Test description",
        type=SubProblemType.IMPLEMENTATION,
        complexity_score=ComplexityScore(
            explanation="Test",
            cognitive_complexity=2.0,
            computational_complexity=1.0,
            domain_complexity=1.0,
            integration_complexity=1.0,
            overall_complexity=1.5
        ),
        dependencies=[]
    )
    
    # Since we have @with_error_handling, it should return a failed SolutionAttempt
    attempt = solver.solve(sp)
    
    assert attempt is not None
    assert attempt.approach == "failed"
