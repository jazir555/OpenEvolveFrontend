"""
Unit Tests for Sub-Problem Classifier

Comprehensive test suite for the subproblem_classifier module.
Tests include:
- Basic classification functionality
- Confidence scoring
- Edge cases (mixed-type, ambiguous, empty descriptions)
- Custom patterns
- Batch classification
- Error handling
"""

import pytest
from datetime import datetime, timezone
from subproblem_classifier import (
    SubProblemType,
    ClassificationResult,
    KeywordPattern,
    ProblemClassifier,
    classify_problem_quick,
    classify_with_confidence,
    batch_classify_descriptions,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_subproblem_dict():
    """Create a sample subproblem as dictionary."""
    return {
        'sub_problem_id': 'test_001',
        'title': 'Implement User Authentication',
        'description': 'Create a secure user authentication system with JWT tokens and password hashing.',
        'status': 'pending',
        'confidence': 0.0,
        'assigned_agent': None,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'completed_at': None
    }


@pytest.fixture
def sample_classifier():
    """Create a sample ProblemClassifier instance."""
    return ProblemClassifier()


@pytest.fixture
def implementation_problems():
    """Sample implementation-type problems."""
    return [
        {'title': 'Create API Endpoint', 'description': 'Implement a REST API endpoint for user management.'},
        {'title': 'Build Database', 'description': 'Create database schema and migrations.'},
        {'title': 'Write Function', 'description': 'Develop a function to process user input.'},
    ]


@pytest.fixture
def analysis_problems():
    """Sample analysis-type problems."""
    return [
        {'title': 'Analyze Bug', 'description': 'Investigate the root cause of the authentication failure.'},
        {'title': 'Review Code', 'description': 'Examine the existing codebase to understand the architecture.'},
        {'title': 'Evaluate Performance', 'description': 'Study the system performance under load.'},
    ]


@pytest.fixture
def validation_problems():
    """Sample validation-type problems."""
    return [
        {'title': 'Test Login', 'description': 'Verify that user login works correctly.'},
        {'title': 'Validate Inputs', 'description': 'Ensure all user inputs are properly validated.'},
        {'title': 'Check Coverage', 'description': 'Confirm test coverage meets requirements.'},
    ]


@pytest.fixture
def ambiguous_problems():
    """Sample ambiguous/mixed-type problems."""
    return [
        {'title': 'Fix and Test', 'description': 'Analyze the bug and implement a fix with tests.'},
        {'title': 'Create Validation', 'description': 'Build a validation system and verify it works.'},
        {'title': 'Review Implementation', 'description': 'Examine the implemented feature for quality.'},
    ]


# ============================================================================
# TEST SUBPROBLEMTYPE ENUM
# ============================================================================

class TestSubProblemType:
    """Test SubProblemType enum functionality."""

    def test_enum_values(self):
        """Test that enum has correct values."""
        assert SubProblemType.IMPLEMENTATION.value == "implementation"
        assert SubProblemType.ANALYSIS.value == "analysis"
        assert SubProblemType.VALIDATION.value == "validation"

    def test_string_conversion(self):
        """Test string conversion."""
        assert str(SubProblemType.IMPLEMENTATION) == "implementation"
        assert str(SubProblemType.ANALYSIS) == "analysis"
        assert str(SubProblemType.VALIDATION) == "validation"

    def test_from_string_valid(self):
        """Test from_string with valid values."""
        assert SubProblemType.from_string("implementation") == SubProblemType.IMPLEMENTATION
        assert SubProblemType.from_string("IMPLEMENTATION") == SubProblemType.IMPLEMENTATION
        assert SubProblemType.from_string("Implementation") == SubProblemType.IMPLEMENTATION
        assert SubProblemType.from_string("analysis") == SubProblemType.ANALYSIS
        assert SubProblemType.from_string("validation") == SubProblemType.VALIDATION

    def test_from_string_invalid(self):
        """Test from_string with invalid values."""
        with pytest.raises(ValueError):
            SubProblemType.from_string("invalid")

        with pytest.raises(ValueError):
            SubProblemType.from_string("research")


# ============================================================================
# TEST CLASSIFICATIONRESULT
# ============================================================================

class TestClassificationResult:
    """Test ClassificationResult dataclass."""

    def test_creation(self):
        """Test creating a ClassificationResult."""
        result = ClassificationResult(
            problem_type=SubProblemType.IMPLEMENTATION,
            confidence=0.85,
            keyword_scores={'implementation': 2.0, 'analysis': 0.5, 'validation': 0.0},
            reasoning="Clear implementation indicators",
            alternative_types=[(SubProblemType.ANALYSIS, 0.5)]
        )

        assert result.problem_type == SubProblemType.IMPLEMENTATION
        assert result.confidence == 0.85
        assert result.keyword_scores['implementation'] == 2.0

    def test_string_type_conversion(self):
        """Test automatic string to enum conversion."""
        result = ClassificationResult(
            problem_type="implementation",  # String instead of enum
            confidence=0.75,
            keyword_scores={}
        )

        assert isinstance(result.problem_type, SubProblemType)
        assert result.problem_type == SubProblemType.IMPLEMENTATION

    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence
        result = ClassificationResult(
            problem_type=SubProblemType.IMPLEMENTATION,
            confidence=0.5,
            keyword_scores={}
        )
        assert result.confidence == 0.5

        # Invalid confidence (too high)
        with pytest.raises(ValueError):
            ClassificationResult(
                problem_type=SubProblemType.IMPLEMENTATION,
                confidence=1.5,
                keyword_scores={}
            )

        # Invalid confidence (negative)
        with pytest.raises(ValueError):
            ClassificationResult(
                problem_type=SubProblemType.IMPLEMENTATION,
                confidence=-0.1,
                keyword_scores={}
            )

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = ClassificationResult(
            problem_type=SubProblemType.ANALYSIS,
            confidence=0.90,
            keyword_scores={'analysis': 3.0, 'implementation': 0.5},
            reasoning="Strong analysis indicators",
            alternative_types=[(SubProblemType.IMPLEMENTATION, 0.5)]
        )

        data = result.to_dict()

        assert data['problem_type'] == 'analysis'
        assert data['confidence'] == 0.90
        assert 'keyword_scores' in data
        assert 'reasoning' in data
        assert len(data['alternative_types']) == 1

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            'problem_type': 'validation',
            'confidence': 0.70,
            'keyword_scores': {'validation': 2.5, 'analysis': 1.0},
            'reasoning': 'Validation keywords found',
            'alternative_types': [
                {'type': 'analysis', 'score': 1.0}
            ],
            'classification_metadata': {'test': True}
        }

        result = ClassificationResult.from_dict(data)

        assert result.problem_type == SubProblemType.VALIDATION
        assert result.confidence == 0.70
        assert len(result.alternative_types) == 1
        assert result.classification_metadata['test'] is True


# ============================================================================
# TEST KEYWORDPATTERN
# ============================================================================

class TestKeywordPattern:
    """Test KeywordPattern functionality."""

    def test_simple_pattern_match(self):
        """Test simple word matching."""
        pattern = KeywordPattern(
            keywords=['implement', 'create', 'build'],
            weight=1.0,
            pattern_type='simple'
        )

        matches, score = pattern.matches("We need to implement a new feature")

        assert matches is True
        assert score > 0

    def test_simple_pattern_no_match(self):
        """Test simple pattern with no match."""
        pattern = KeywordPattern(
            keywords=['implement', 'create', 'build'],
            weight=1.0,
            pattern_type='simple'
        )

        matches, score = pattern.matches("We should analyze the requirements")

        assert matches is False
        assert score == 0

    def test_regex_pattern_match(self):
        """Test regex pattern matching."""
        pattern = KeywordPattern(
            keywords=[r'implement\s+\w+', r'create\s+\w+'],
            weight=1.5,
            pattern_type='regex'
        )

        matches, score = pattern.matches("We should implement authentication")

        assert matches is True
        assert score > 0

    def test_phrase_pattern_match(self):
        """Test phrase pattern matching."""
        pattern = KeywordPattern(
            keywords=['unit test', 'integration test', 'test coverage'],
            weight=1.2,
            pattern_type='phrase'
        )

        matches, score = pattern.matches("Write unit test for authentication")

        assert matches is True
        assert score > 0


# ============================================================================
# TEST PROBLEMCLASSIFIER
# ============================================================================

class TestProblemClassifier:
    """Test ProblemClassifier main functionality."""

    def test_initialization(self):
        """Test classifier initialization."""
        classifier = ProblemClassifier()

        assert classifier.confidence_threshold == ProblemClassifier.MEDIUM_CONFIDENCE_THRESHOLD
        assert classifier.enable_nlp_patterns is True
        assert classifier.handle_mixed_types is True

    def test_custom_initialization(self):
        """Test classifier with custom settings."""
        classifier = ProblemClassifier(
            confidence_threshold=0.8,
            enable_nlp_patterns=False,
            handle_mixed_types=False
        )

        assert classifier.confidence_threshold == 0.8
        assert classifier.enable_nlp_patterns is False
        assert classifier.handle_mixed_types is False

    def test_classify_implementation_problem(self, sample_classifier, implementation_problems):
        """Test classifying implementation problems."""
        for problem in implementation_problems:
            result = sample_classifier.classify_problem(problem, return_details=True)

            assert result.problem_type == SubProblemType.IMPLEMENTATION
            assert result.confidence >= 0.5

    def test_classify_analysis_problem(self, sample_classifier, analysis_problems):
        """Test classifying analysis problems."""
        for problem in analysis_problems:
            result = sample_classifier.classify_problem(problem, return_details=True)

            assert result.problem_type == SubProblemType.ANALYSIS
            assert result.confidence >= 0.5

    def test_classify_validation_problem(self, sample_classifier, validation_problems):
        """Test classifying validation problems."""
        for problem in validation_problems:
            result = sample_classifier.classify_problem(problem, return_details=True)

            assert result.problem_type == SubProblemType.VALIDATION
            assert result.confidence >= 0.5

    def test_classify_without_details(self, sample_classifier):
        """Test classification without detailed result."""
        problem = {
            'title': 'Feature Implementation',
            'description': 'Implement a new user management feature with CRUD operations.'
        }

        result = sample_classifier.classify_problem(problem, return_details=False)

        assert isinstance(result, SubProblemType)
        assert result == SubProblemType.IMPLEMENTATION

    def test_classify_ambiguous_problem(self, sample_classifier, ambiguous_problems):
        """Test classifying ambiguous/mixed-type problems."""
        for problem in ambiguous_problems:
            result = sample_classifier.classify_problem(problem, return_details=True)

            # Should still classify, but with lower confidence
            assert result.problem_type in SubProblemType
            # May have lower confidence for mixed problems

    def test_analyze_keywords(self, sample_classifier):
        """Test keyword analysis."""
        description = "Implement a new feature and test it thoroughly"

        scores = sample_classifier.analyze_keywords(description)

        assert 'implementation' in scores
        assert 'validation' in scores
        assert 'analysis' in scores
        # Should have scores for both implementation and validation
        assert scores['implementation'] > 0
        assert scores['validation'] > 0

    def test_determine_type(self, sample_classifier):
        """Test type determination from scores."""
        # Clear implementation score
        scores = {
            'implementation': 3.0,
            'analysis': 0.5,
            'validation': 0.0
        }

        problem_type = sample_classifier.determine_type(scores)

        assert problem_type == SubProblemType.IMPLEMENTATION

    def test_get_confidence_score(self, sample_classifier):
        """Test confidence score calculation."""
        # High confidence (one dominant type)
        scores_high = {
            'implementation': 3.0,
            'analysis': 0.0,
            'validation': 0.0
        }

        confidence_high = sample_classifier.get_confidence_score(scores_high)

        assert confidence_high >= 0.8

        # Lower confidence (multiple types)
        scores_low = {
            'implementation': 1.5,
            'analysis': 1.4,
            'validation': 0.0
        }

        confidence_low = sample_classifier.get_confidence_score(scores_low)

        assert confidence_low < confidence_high

    def test_add_custom_pattern(self, sample_classifier):
        """Test adding custom patterns."""
        custom_pattern = KeywordPattern(
            keywords=['custom', 'specialized'],
            weight=2.0,
            category='custom_test',
            pattern_type='simple'
        )

        sample_classifier.add_custom_pattern(
            SubProblemType.IMPLEMENTATION,
            ['custom', 'specialized'],
            weight=2.0,
            category='custom_test'
        )

        # Verify pattern was added
        assert len(sample_classifier.patterns[SubProblemType.IMPLEMENTATION]) > 3

    def test_batch_classify(self, sample_classifier):
        """Test batch classification."""
        problems = [
            {'title': 'P1', 'description': 'Implement feature X'},
            {'title': 'P2', 'description': 'Analyze the bug'},
            {'title': 'P3', 'description': 'Test the system'},
        ]

        results = sample_classifier.classify_batch(problems, return_details=True)

        assert len(results) == 3
        assert results[0].problem_type == SubProblemType.IMPLEMENTATION
        assert results[1].problem_type == SubProblemType.ANALYSIS
        assert results[2].problem_type == SubProblemType.VALIDATION

    def test_get_type_distribution(self, sample_classifier):
        """Test getting type distribution."""
        problems = [
            {'title': 'P1', 'description': 'Implement feature X'},
            {'title': 'P2', 'description': 'Create another feature'},
            {'title': 'P3', 'description': 'Test the system'},
        ]

        distribution = sample_classifier.get_type_distribution(problems)

        assert distribution[SubProblemType.IMPLEMENTATION] == 2
        assert distribution[SubProblemType.VALIDATION] == 1
        assert distribution[SubProblemType.ANALYSIS] == 0

    def test_get_statistics(self, sample_classifier):
        """Test getting classifier statistics."""
        stats = sample_classifier.get_statistics()

        assert 'confidence_threshold' in stats
        assert 'nlp_patterns_enabled' in stats
        assert 'available_types' in stats
        assert 'patterns_count' in stats

    def test_error_handling_empty_description(self, sample_classifier):
        """Test error handling for empty description."""
        with pytest.raises(ValueError):
            sample_classifier.classify_problem({'description': ''})

    def test_error_handling_short_description(self, sample_classifier):
        """Test error handling for very short description."""
        with pytest.raises(ValueError):
            sample_classifier.classify_problem({'description': 'ab'})

    def test_error_handling_invalid_description(self, sample_classifier):
        """Test error handling for invalid description."""
        with pytest.raises(ValueError):
            sample_classifier.classify_problem({'description': None})


# ============================================================================
# TEST CONVENIENCE FUNCTIONS
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_classify_problem_quick(self):
        """Test quick classification function."""
        result = classify_problem_quick(
            "Implement a secure authentication system",
            "Auth System"
        )

        assert isinstance(result, SubProblemType)
        assert result == SubProblemType.IMPLEMENTATION

    def test_classify_with_confidence(self):
        """Test classification with confidence."""
        problem_type, confidence = classify_with_confidence(
            "Verify the authentication works correctly",
            "Auth Tests"
        )

        assert isinstance(problem_type, SubProblemType)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert problem_type == SubProblemType.VALIDATION

    def test_batch_classify_descriptions(self):
        """Test batch classification of descriptions."""
        descriptions = [
            ("Create API", "Implement REST API"),
            ("Analyze Bug", "Investigate the issue"),
            ("Test Login", "Verify login works"),
        ]

        results = batch_classify_descriptions(descriptions)

        assert len(results) == 3
        assert results[0][1] == SubProblemType.IMPLEMENTATION
        assert results[1][1] == SubProblemType.ANALYSIS
        assert results[2][1] == SubProblemType.VALIDATION


# ============================================================================
# TEST EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_zeros_keywords(self):
        """Test when all keyword scores are zero."""
        classifier = ProblemClassifier()

        # Description with no matching keywords
        scores = classifier.analyze_keywords("xyz abc def")

        problem_type = classifier.determine_type(scores)

        # Should still return a type (default to IMPLEMENTATION)
        assert problem_type in SubProblemType

    def test_tied_scores(self):
        """Test handling of tied scores."""
        classifier = ProblemClassifier()

        # Create tied scores
        scores = {
            'implementation': 1.5,
            'analysis': 1.5,
            'validation': 0.0
        }

        problem_type = classifier.determine_type(scores)

        # Should break tie using preference order (IMPLEMENTATION first)
        assert problem_type == SubProblemType.IMPLEMENTATION

    def test_mixed_type_detection(self):
        """Test mixed-type problem detection."""
        classifier = ProblemClassifier(handle_mixed_types=True)

        # Mixed implementation and validation
        result = classifier.classify_problem(
            {'description': 'Analyze the bug and implement a fix with comprehensive tests'},
            return_details=True
        )

        # Check metadata for mixed type flag
        assert 'is_mixed_type' in result.classification_metadata

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        result = classify_problem_quick(
            "Implémenter un système d'authentification 安全",
            "Auth System"
        )

        assert isinstance(result, SubProblemType)

    def test_very_long_description(self):
        """Test handling of very long descriptions."""
        long_desc = "Implement " + "feature " * 1000

        result = classify_problem_quick(long_desc)

        assert result == SubProblemType.IMPLEMENTATION

    def test_special_characters(self):
        """Test handling of special characters."""
        result = classify_problem_quick(
            "Implement the @#$%^&*() system with <tags> and {symbols}",
            "Special Chars"
        )

        assert isinstance(result, SubProblemType)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
