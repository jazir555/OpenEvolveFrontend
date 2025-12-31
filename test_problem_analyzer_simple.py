"""
Simple test script for the Problem Analyzer module.
Verifies core functionality without external dependencies.
"""

import sys
import os
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock the missing dependencies
class MockOpenEvolveClient:
    def __init__(self, *args, **kwargs):
        pass
    
    def evolve(self, **kwargs):
        # Mock successful evolution result
        class MockResult:
            def __init__(self):
                self.success = True
                self.best_code = "Mock successful analysis result"
                self.best_score = 0.95
                self.iterations_completed = 1
                self.metrics = {"mock": True}
                self.error = None
        return MockResult()

# Replace the imports that are causing issues
sys.modules['opentelemetry'] = type('MockModule', (), {})()
sys.modules['opentelemetry.trace'] = type('MockModule', (), {})()

def test_problem_analyzer_basic():
    """Test basic Problem Analyzer functionality without external dependencies."""
    try:
        # Mock the OpenEvolve client
        import problem_analyzer
        problem_analyzer.OpenEvolveClient = MockOpenEvolveClient
        problem_analyzer.OPENEVOLVE_AVAILABLE = True
        
        # Import the problem analyzer
        from problem_analyzer import ProblemAnalyzer
        from sovereign_data_models import ProblemDefinition, ProblemType, DomainContext, ComplexityScore
        
        logger = logging.getLogger(__name__)
        logger.info("Testing ProblemAnalyzer with mocked dependencies...")
        
        # Create a sample problem
        problem_text = """
        We need to develop a machine learning model that can predict customer churn for our subscription service. 
        The model should achieve at least 85% accuracy and process predictions in under 100ms. 
        The solution must be completed within 3 months and stay within a $50,000 budget.
        The model should handle 10,000 predictions per day and comply with GDPR regulations.
        """
        
        # Initialize the analyzer with mocked client
        analyzer = ProblemAnalyzer(openevolve_client=MockOpenEvolveClient())
        
        # Test that the analyzer was created correctly
        assert analyzer is not None, "ProblemAnalyzer should be created"
        assert analyzer.openevolve_client is not None, "OpenEvolve client should be set"
        
        logger.info("ProblemAnalyzer created successfully with mocked client")
        
        # Test fallback functionality
        logger.info("Testing fallback domain context extraction...")
        domain_context = analyzer._extract_domain_context_fallback(problem_text)
        assert domain_context is not None, "Fallback domain context should be created"
        assert domain_context.domain is not None, "Domain should be identified"
        logger.info(f"Fallback domain context: {domain_context.domain}")
        
        # Test fallback problem type classification
        logger.info("Testing fallback problem type classification...")
        problem_type = analyzer._classify_problem_type_fallback(problem_text)
        assert problem_type is not None, "Fallback problem type should be classified"
        logger.info(f"Fallback problem type: {problem_type}")
        
        # Test fallback complexity assessment
        logger.info("Testing fallback complexity assessment...")
        # Create a mock problem definition for complexity assessment
        mock_problem = ProblemDefinition(
            id="test_problem_123",
            title="Customer Churn Prediction",
            description=problem_text,
            problem_type=ProblemType.IMPLEMENTATION,
            domain_context=DomainContext(
                domain="machine_learning",
                subdomain="predictive_modeling",
                related_domains=["data_science", "business_analytics"],
                domain_knowledge={
                    "key_concepts": ["machine learning", "predictive modeling", "customer analytics"],
                    "extraction_method": "mock",
                    "domain_complexity": 7.0,
                    "required_expertise": ["ML engineer", "data scientist"]
                }
            ),
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Initial mock assessment"
            ),
            constraints=[],
            success_criteria=[],
            stakeholders=[],
            resources_available={}
        )
        
        complexity_score = analyzer._assess_complexity_fallback(mock_problem)
        assert complexity_score is not None, "Fallback complexity score should be calculated"
        assert 0 <= complexity_score.overall_complexity <= 10, "Overall complexity should be between 0-10"
        logger.info(f"Fallback complexity score: {complexity_score.overall_complexity}")
        
        # Test fallback constraint identification
        logger.info("Testing fallback constraint identification...")
        constraints = analyzer._identify_constraints_fallback(problem_text)
        assert isinstance(constraints, list), "Fallback constraints should return a list"
        logger.info(f"Fallback constraints identified: {len(constraints)}")
        
        # Test fallback success criteria generation
        logger.info("Testing fallback success criteria generation...")
        success_criteria = analyzer._generate_criteria_fallback(mock_problem)
        assert isinstance(success_criteria, list), "Fallback success criteria should return a list"
        assert len(success_criteria) > 0, "Should generate at least one success criterion"
        logger.info(f"Fallback success criteria generated: {len(success_criteria)}")
        
        logger.info("All basic tests passed!")
        return True
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = test_problem_analyzer_basic()
    sys.exit(0 if success else 1)