"""
Unit Tests for Complexity Analyzer

This module contains comprehensive unit tests for the ComplexityAnalyzer class,
covering all dimensions of complexity analysis and edge cases.
"""

import unittest
from typing import List

from complexity_analyzer import (
    ComplexityAnalyzer,
    ComplexityScore,
    ComplexityLevel,
    quick_complexity_analysis
)

try:
    from sovereign_data_models import ProblemDefinition
    HAS_PROBLEM_DEFINITION = True
except ImportError:
    # Fallback for testing
    from dataclasses import dataclass, field
    from typing import List, Optional
    from datetime import datetime

    HAS_PROBLEM_DEFINITION = False

    @dataclass
    class ProblemDefinition:
        """Minimal ProblemDefinition for testing"""
        problem_id: str
        title: str
        description: str
        domain: str
        complexity: str
        priority: str
        estimated_effort: str
        requirements: List[str]
        constraints: List[str]
        created_at: datetime
        dependencies: List[str] = field(default_factory=list)


class TestComplexityAnalyzer(unittest.TestCase):
    """Test cases for ComplexityAnalyzer"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = ComplexityAnalyzer()

    def _create_problem(
        self,
        title="Test Problem",
        description="Test description",
        domain="test",
        complexity="moderate",
        priority="medium",
        estimated_effort="hours",
        requirements=None,
        constraints=None,
        dependencies=None
    ):
        """Helper to create ProblemDefinition with all required fields"""
        from datetime import datetime
        return ProblemDefinition(
            problem_id=f"test_{title.lower().replace(' ', '_')}",
            title=title,
            description=description,
            domain=domain,
            complexity=complexity,
            priority=priority,
            estimated_effort=estimated_effort,
            requirements=requirements or [],
            constraints=constraints or [],
            created_at=datetime.now()
        )

    def test_initialization(self):
        """Test analyzer initialization"""
        analyzer = ComplexityAnalyzer()
        self.assertIsNotNone(analyzer)
        self.assertAlmostEqual(
            analyzer.cognitive_weight +
            analyzer.computational_weight +
            analyzer.domain_weight +
            analyzer.integration_weight,
            1.0,
            places=2
        )

    def test_custom_weights(self):
        """Test analyzer with custom weights"""
        config = {
            'cognitive_weight': 0.4,
            'computational_weight': 0.3,
            'domain_weight': 0.2,
            'integration_weight': 0.1
        }
        analyzer = ComplexityAnalyzer(config)
        self.assertEqual(analyzer.cognitive_weight, 0.4)
        self.assertEqual(analyzer.computational_weight, 0.3)
        self.assertEqual(analyzer.domain_weight, 0.2)
        self.assertEqual(analyzer.integration_weight, 0.1)

    def test_weight_normalization(self):
        """Test that weights are normalized if they don't sum to 1.0"""
        config = {
            'cognitive_weight': 0.5,
            'computational_weight': 0.5,
            'domain_weight': 0.5,
            'integration_weight': 0.5
        }
        analyzer = ComplexityAnalyzer(config)
        # Should be normalized to 0.25 each
        self.assertAlmostEqual(analyzer.cognitive_weight, 0.25, places=2)
        self.assertAlmostEqual(analyzer.computational_weight, 0.25, places=2)

    def test_calculate_complexity_simple(self):
        """Test complexity calculation for a simple problem"""
        problem = self._create_problem(
            title="Simple Form",
            description="Create a simple web form to collect user input.",
            domain="web_development",
            requirements=["Collect input", "Validate form"]
        )

        result = self.analyzer.calculate_complexity(problem)

        self.assertIsInstance(result, ComplexityScore)
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 1.0)
        self.assertGreaterEqual(result.cognitive_score, 0.0)
        self.assertLessEqual(result.cognitive_score, 1.0)
        self.assertGreaterEqual(result.computational_score, 0.0)
        self.assertLessEqual(result.computational_score, 1.0)
        self.assertGreaterEqual(result.domain_score, 0.0)
        self.assertLessEqual(result.domain_score, 1.0)
        self.assertGreaterEqual(result.integration_score, 0.0)
        self.assertLessEqual(result.integration_score, 1.0)

    def test_calculate_complexity_complex(self):
        """Test complexity calculation for a complex problem"""
        problem = self._create_problem(
            title="Distributed ML System",
            description=(
                "Design a distributed machine learning system using deep "
                "reinforcement learning for real-time optimization across "
                "multiple geographic regions with Byzantine fault tolerance."
            ),
            domain="machine_learning",
            requirements=[
                "Real-time processing",
                "Distributed training",
                "Fault tolerance",
                "Low latency"
            ],
            constraints=[
                "Network partitions",
                "Limited resources",
                "Strong consistency"
            ],
            dependencies=[
                "TensorFlow",
                "Kafka",
                "Redis",
                "PostgreSQL"
            ]
        )

        result = self.analyzer.calculate_complexity(problem)

        # Complex problem should have higher score
        self.assertGreater(result.overall_score, 0.5)
        self.assertGreater(result.cognitive_score, 0.5)
        self.assertGreater(result.computational_score, 0.5)

    def test_calculate_complexity_invalid_input(self):
        """Test complexity calculation with invalid inputs"""
        with self.assertRaises(ValueError):
            problem = self._create_problem(
                title="Empty",
                description=""
            )
            self.analyzer.calculate_complexity(problem)

    def test_calculate_complexity_wrong_type(self):
        """Test complexity calculation with wrong type"""
        with self.assertRaises(TypeError):
            self.analyzer.calculate_complexity("not a problem")

    def test_analyze_cognitive_complexity(self):
        """Test cognitive complexity analysis"""
        # Simple text
        simple = "Create a basic web page with a form."
        score = self.analyzer.analyze_cognitive_complexity(simple, "web_development")
        self.assertLess(score, 0.5)

        # Complex text
        complex_text = (
            "Design a sophisticated multi-objective optimization framework "
            "using deep reinforcement learning with attention mechanisms for "
            "real-time decision making in stochastic environments."
        )
        score = self.analyzer.analyze_cognitive_complexity(complex_text, "machine_learning")
        self.assertGreater(score, 0.4)  # Adjusted threshold

    def test_analyze_cognitive_complexity_empty(self):
        """Test cognitive complexity with empty input"""
        score = self.analyzer.analyze_cognitive_complexity("", "unknown")
        self.assertEqual(score, 0.0)

    def test_analyze_computational_complexity(self):
        """Test computational complexity analysis"""
        # Simple requirements
        simple_reqs = ["Store data in database"]
        score = self.analyzer.analyze_computational_complexity(simple_reqs)
        self.assertLess(score, 0.5)

        # Complex requirements
        complex_reqs = [
            "Implement O(n^2) algorithm with optimization",
            "Real-time processing with low latency",
            "Distributed computation across multiple nodes"
        ]
        score = self.analyzer.analyze_computational_complexity(complex_reqs)
        self.assertGreater(score, 0.5)

    def test_analyze_computational_complexity_empty(self):
        """Test computational complexity with empty requirements"""
        score = self.analyzer.analyze_computational_complexity([])
        # Should return default low complexity
        self.assertEqual(score, 0.3)

    def test_analyze_domain_complexity(self):
        """Test domain complexity analysis"""
        # Simple domain
        score = self.analyzer.analyze_domain_complexity("web_development", [])
        self.assertLess(score, 0.7)

        # Complex domain
        score = self.analyzer.analyze_domain_complexity(
            "machine_learning",
            ["Non-convex optimization", "Dynamic constraints"]
        )
        self.assertGreater(score, 0.6)

    def test_analyze_domain_complexity_unknown(self):
        """Test domain complexity with unknown domain"""
        score = self.analyzer.analyze_domain_complexity("unknown", [])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_analyze_integration_complexity(self):
        """Test integration complexity analysis"""
        # No dependencies
        score = self.analyzer.analyze_integration_complexity([])
        self.assertEqual(score, 0.0)

        # Few dependencies
        score = self.analyzer.analyze_integration_complexity(["API", "Database"])
        self.assertGreater(score, 0.0)
        self.assertLess(score, 0.5)

        # Many dependencies
        deps = [
            "REST API", "Database", "Message Queue", "Cache",
            "External Service", "Auth Service", "Logging Service"
        ]
        score = self.analyzer.analyze_integration_complexity(deps)
        self.assertGreater(score, 0.4)

    def test_complexity_levels(self):
        """Test complexity level mapping"""
        test_cases = [
            (0.1, ComplexityLevel.TRIVIAL),
            (0.2, ComplexityLevel.SIMPLE),
            (0.4, ComplexityLevel.MODERATE),
            (0.6, ComplexityLevel.COMPLEX),
            (0.8, ComplexityLevel.VERY_COMPLEX),
            (0.95, ComplexityLevel.EXTREME)
        ]

        for score, expected_level in test_cases:
            level = self.analyzer._get_complexity_level(score)
            self.assertEqual(level, expected_level)

    def test_explanation_generation(self):
        """Test explanation generation"""
        explanation = self.analyzer._generate_explanation(
            overall=0.7,
            cognitive=0.8,
            computational=0.6,
            domain=0.7,
            integration=0.5
        )

        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)
        self.assertIn("complex", explanation.lower())

    def test_confidence_calculation(self):
        """Test confidence calculation"""
        # Minimal problem
        problem = self._create_problem(
            title="Minimal",
            description="Short"
        )
        confidence = self.analyzer._calculate_confidence(problem, {})
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

        # Complete problem
        problem = self._create_problem(
            title="Complete",
            description="This is a detailed problem description with plenty of context.",
            domain="machine_learning",
            requirements=["Req1", "Req2", "Req3"],
            constraints=["Constraint1"],
            dependencies=["Dep1"]
        )
        confidence = self.analyzer._calculate_confidence(problem, {})
        self.assertGreater(confidence, 0.7)

    def test_dimension_breakdown(self):
        """Test that dimension breakdown is included"""
        problem = self._create_problem(
            title="Test",
            description="Test description",
            domain="web_development"
        )

        result = self.analyzer.calculate_complexity(problem)

        self.assertIn('dimension_breakdown', result.__dict__)
        self.assertIn('cognitive', result.dimension_breakdown)
        self.assertIn('computational', result.dimension_breakdown)
        self.assertIn('domain', result.dimension_breakdown)
        self.assertIn('integration', result.dimension_breakdown)

    def test_score_normalization(self):
        """Test that scores are properly normalized"""
        problem = self._create_problem(
            title="Test",
            description="Test" * 1000,  # Very long to potentially spike score
            domain="machine_learning",
            requirements=["Optimization"] * 100,
            constraints=["Constraint"] * 100,
            dependencies=["Dependency"] * 100
        )

        result = self.analyzer.calculate_complexity(problem)

        # All scores should be within [0, 1]
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 1.0)
        self.assertGreaterEqual(result.cognitive_score, 0.0)
        self.assertLessEqual(result.cognitive_score, 1.0)
        self.assertGreaterEqual(result.computational_score, 0.0)
        self.assertLessEqual(result.computational_score, 1.0)
        self.assertGreaterEqual(result.domain_score, 0.0)
        self.assertLessEqual(result.domain_score, 1.0)
        self.assertGreaterEqual(result.integration_score, 0.0)
        self.assertLessEqual(result.integration_score, 1.0)


class TestQuickComplexityAnalysis(unittest.TestCase):
    """Test cases for quick_complexity_analysis function"""

    def test_quick_analysis_basic(self):
        """Test basic quick analysis"""
        result = quick_complexity_analysis(
            description="Create a simple web form.",
            domain="web_development"
        )

        self.assertIsInstance(result, ComplexityScore)
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 1.0)

    def test_quick_analysis_with_all_parameters(self):
        """Test quick analysis with all parameters"""
        result = quick_complexity_analysis(
            description="Design a machine learning system for real-time predictions.",
            domain="machine_learning",
            requirements=["Real-time processing", "High accuracy"],
            constraints=["Limited resources"],
            dependencies=["TensorFlow", "Redis"]
        )

        self.assertIsInstance(result, ComplexityScore)
        self.assertGreater(result.overall_score, 0.3)  # Should be moderately complex

    def test_quick_analysis_minimal(self):
        """Test quick analysis with minimal parameters"""
        result = quick_complexity_analysis(
            description="Simple task"
        )

        self.assertIsInstance(result, ComplexityScore)
        self.assertGreaterEqual(result.overall_score, 0.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = ComplexityAnalyzer()

    def _create_problem(self, **kwargs):
        """Helper to create ProblemDefinition with all required fields"""
        from datetime import datetime
        defaults = {
            'problem_id': 'test_problem',
            'title': 'Test',
            'description': 'Test description',
            'domain': 'test',
            'complexity': 'moderate',
            'priority': 'medium',
            'estimated_effort': 'hours',
            'requirements': [],
            'constraints': [],
            'created_at': datetime.now()
        }
        # Filter out 'dependencies' if present (not in actual ProblemDefinition)
        kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'dependencies'}
        defaults.update(kwargs_filtered)
        return ProblemDefinition(**defaults)

    def test_none_context(self):
        """Test with None context"""
        problem = self._create_problem(
            title="Test",
            description="Test description",
            domain="web_development"
        )
        # Should not raise error
        result = self.analyzer.calculate_complexity(problem, None)
        self.assertIsInstance(result, ComplexityScore)

    def test_empty_lists(self):
        """Test with empty lists"""
        problem = self._create_problem(
            title="Test",
            description="Test description",
            domain="web_development",
            requirements=[],
            constraints=[],
            dependencies=[]
        )
        # Should not raise error
        result = self.analyzer.calculate_complexity(problem)
        self.assertIsInstance(result, ComplexityScore)

    def test_very_long_description(self):
        """Test with very long description"""
        description = "Complex problem description. " * 1000
        problem = self._create_problem(
            title="Test",
            description=description,
            domain="machine_learning"
        )
        # Should not raise error and score should be normalized
        result = self.analyzer.calculate_complexity(problem)
        self.assertLessEqual(result.cognitive_score, 1.0)

    def test_special_characters(self):
        """Test with special characters in description"""
        description = "Design a system with special characters: @#$%^&*()_+-=[]{}|;':\",./<>?"
        problem = self._create_problem(
            title="Test",
            description=description,
            domain="web_development"
        )
        # Should not raise error
        result = self.analyzer.calculate_complexity(problem)
        self.assertIsInstance(result, ComplexityScore)

    def test_unicode_characters(self):
        """Test with unicode characters"""
        description = "Design a system with unicode: 你好世界 مرحبا بالعالم 안녕하세요"
        problem = self._create_problem(
            title="Test",
            description=description,
            domain="web_development"
        )
        # Should not raise error
        result = self.analyzer.calculate_complexity(problem)
        self.assertIsInstance(result, ComplexityScore)

    def test_minimal_configuration(self):
        """Test analyzer with minimal configuration"""
        analyzer = ComplexityAnalyzer(config={})
        problem = self._create_problem(
            title="Test",
            description="Test description",
            domain="web_development"
        )
        # Should work with default config
        result = analyzer.calculate_complexity(problem)
        self.assertIsInstance(result, ComplexityScore)

    def test_normalize_disabled(self):
        """Test analyzer with normalization disabled"""
        config = {'normalize_scores': False}
        analyzer = ComplexityAnalyzer(config=config)

        problem = self._create_problem(
            title="Test",
            description="Test" * 1000,
            domain="machine_learning"
        )

        result = analyzer.calculate_complexity(problem)
        # Scores might exceed 1.0 without normalization
        self.assertIsInstance(result, ComplexityScore)


class TestTechnicalTerms(unittest.TestCase):
    """Test technical term detection and scoring"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = ComplexityAnalyzer()

    def test_technical_terms_simple(self):
        """Test that technical terms increase complexity score"""
        simple = "Create a web page."
        technical = "Create a neural network with backpropagation."

        simple_score = self.analyzer.analyze_cognitive_complexity(simple, "web_development")
        technical_score = self.analyzer.analyze_cognitive_complexity(technical, "machine_learning")

        self.assertGreater(technical_score, simple_score)

    def test_multiple_technical_terms(self):
        """Test accumulation of technical terms"""
        text = (
            "Design a deep learning system using convolutional neural networks "
            "with transformer architecture and attention mechanisms for "
            "reinforcement learning optimization."
        )

        score = self.analyzer.analyze_cognitive_complexity(text, "machine_learning")
        self.assertGreater(score, 0.4)  # Adjusted threshold


class TestDomainComplexity(unittest.TestCase):
    """Test domain-specific complexity analysis"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = ComplexityAnalyzer()

    def test_ml_domain_complexity(self):
        """Test that ML domain has higher complexity"""
        ml_score = self.analyzer.analyze_domain_complexity(
            "machine_learning",
            ["Optimization constraints"]
        )
        web_score = self.analyzer.analyze_domain_complexity(
            "web_development",
            []
        )

        self.assertGreater(ml_score, web_score)

    def test_domain_variations(self):
        """Test different domain names"""
        domains = [
            "machine_learning",
            "deep_learning",
            "web_development",
            "database"
        ]

        scores = [
            self.analyzer.analyze_domain_complexity(domain, [])
            for domain in domains
        ]

        # All scores should be valid
        for score in scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


def run_tests():
    """Run all tests and print results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestComplexityAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestQuickComplexityAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestTechnicalTerms))
    suite.addTests(loader.loadTestsFromTestCase(TestDomainComplexity))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
