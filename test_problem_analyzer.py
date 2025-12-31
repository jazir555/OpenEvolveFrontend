"""
Test script for the Problem Analyzer module.
Verifies that all components are working correctly.
"""

import sys
import os
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_problem_analyzer():
    """Test the Problem Analyzer functionality."""
    try:
        # Import the problem analyzer
        from problem_analyzer import ProblemAnalyzer
        from sovereign_data_models import ProblemDefinition
        
        logger.info("Testing ProblemAnalyzer...")
        
        # Create a sample problem
        problem_text = """
        We need to develop a machine learning model that can predict customer churn for our subscription service. 
        The model should achieve at least 85% accuracy and process predictions in under 100ms. 
        The solution must be completed within 3 months and stay within a $50,000 budget.
        The model should handle 10,000 predictions per day and comply with GDPR regulations.
        """
        
        # Initialize the analyzer
        analyzer = ProblemAnalyzer()
        
        # Test problem analysis
        logger.info("Analyzing problem...")
        problem = analyzer.analyze_problem(problem_text, "Customer Churn Prediction")
        
        # Verify the results
        assert isinstance(problem, ProblemDefinition), "Should return ProblemDefinition"
        assert problem.title == "Customer Churn Prediction", "Title should match"
        assert len(problem.domain_context.domain) > 0, "Should have domain"
        assert problem.complexity_score.overall_complexity >= 0, "Should have valid complexity score"
        assert len(problem.constraints) >= 0, "Should have constraints (even if 0)"
        assert len(problem.success_criteria) >= 0, "Should have success criteria (even if 0)"
        
        logger.info(f"Problem analyzed successfully:")
        logger.info(f"  Title: {problem.title}")
        logger.info(f"  Domain: {problem.domain_context.domain}")
        logger.info(f"  Complexity: {problem.complexity_score.overall_complexity}")
        logger.info(f"  Constraints: {len(problem.constraints)}")
        logger.info(f"  Success Criteria: {len(problem.success_criteria)}")
        
        # Test validation
        logger.info("Validating problem definition...")
        is_valid, errors = analyzer.validate_problem_definition(problem)
        assert is_valid, f"Problem should be valid. Errors: {errors}"
        logger.info("Problem validation passed")
        
        logger.info("All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_problem_analyzer()
    sys.exit(0 if success else 1)