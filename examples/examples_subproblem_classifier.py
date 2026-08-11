"""
Usage Examples for Sub-Problem Classifier

This file demonstrates various ways to use the subproblem_classifier module
with real-world examples and integration patterns.
"""

from subproblem_classifier import (
    SubProblemType,
    ClassificationResult,
    ProblemClassifier,
    classify_problem_quick,
    classify_with_confidence,
    batch_classify_descriptions,
)
from sovereign_data_models import SubProblem, ProblemStatus
from datetime import datetime


# ============================================================================
# EXAMPLE 1: BASIC CLASSIFICATION
# ============================================================================

def example_1_basic_classification():
    """
    Example 1: Basic classification of a single problem.
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Classification")
    print("=" * 80)

    # Simple classification
    result = classify_problem_quick(
        description="Implement a secure user authentication system with JWT tokens",
        title="User Authentication"
    )

    print(f"Problem Type: {result}")
    print(f"Type Value: {result.value}")
    print()


# ============================================================================
# EXAMPLE 2: CLASSIFICATION WITH CONFIDENCE
# ============================================================================

def example_2_classification_with_confidence():
    """
    Example 2: Get both type and confidence score.
    """
    print("=" * 80)
    print("EXAMPLE 2: Classification with Confidence Score")
    print("=" * 80)

    descriptions = [
        ("Build a REST API for user management", "API Development"),
        ("Investigate the performance bottleneck in the database", "Performance Analysis"),
        ("Write unit tests for the authentication module", "Testing"),
        ("Analyze the root cause of the login failure", "Bug Investigation"),
    ]

    for desc, title in descriptions:
        problem_type, confidence = classify_with_confidence(desc, title)

        confidence_level = (
            "HIGH" if confidence >= 0.75 else
            "MEDIUM" if confidence >= 0.50 else
            "LOW"
        )

        print(f"Title: {title}")
        print(f"  Type: {problem_type.value}")
        print(f"  Confidence: {confidence:.2f} ({confidence_level})")
        print()


# ============================================================================
# EXAMPLE 3: DETAILED CLASSIFICATION RESULT
# ============================================================================

def example_3_detailed_result():
    """
    Example 3: Get detailed classification result with reasoning.
    """
    print("=" * 80)
    print("EXAMPLE 3: Detailed Classification Result")
    print("=" * 80)

    classifier = ProblemClassifier()

    problem = {
        'title': 'Authentication System',
        'description': 'Implement JWT-based authentication with refresh tokens and secure password hashing'
    }

    result = classifier.classify_problem(problem, return_details=True)

    print(f"Problem Type: {result.problem_type.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"\nReasoning:\n{result.reasoning}")
    print(f"\nKeyword Scores:")
    for type_name, score in result.keyword_scores.items():
        if score > 0:
            print(f"  {type_name}: {score:.2f}")

    if result.alternative_types:
        print(f"\nAlternative Types:")
        for alt_type, alt_score in result.alternative_types:
            print(f"  {alt_type.value}: {alt_score:.2f}")

    print()


# ============================================================================
# EXAMPLE 4: WORKING WITH SUBPROBLEM MODEL
# ============================================================================

def example_4_subproblem_model():
    """
    Example 4: Classify SubProblem model from sovereign_data_models.
    """
    print("=" * 80)
    print("EXAMPLE 4: Working with SubProblem Model")
    print("=" * 80)

    # Create SubProblem instances
    subproblems = [
        SubProblem(
            sub_problem_id="sp_001",
            parent_id=None,
            title="Create User Model",
            description="Implement a database model for storing user information",
            status=ProblemStatus.PENDING,
            confidence=0.0,
            assigned_agent=None,
            created_at=datetime.utcnow(),
            completed_at=None
        ),
        SubProblem(
            sub_problem_id="sp_002",
            parent_id=None,
            title="Analyze Performance",
            description="Investigate slow query performance in the user dashboard",
            status=ProblemStatus.PENDING,
            confidence=0.0,
            assigned_agent=None,
            created_at=datetime.utcnow(),
            completed_at=None
        ),
        SubProblem(
            sub_problem_id="sp_003",
            parent_id=None,
            title="Test Authentication",
            description="Verify that JWT authentication works correctly across all endpoints",
            status=ProblemStatus.PENDING,
            confidence=0.0,
            assigned_agent=None,
            created_at=datetime.utcnow(),
            completed_at=None
        ),
    ]

    classifier = ProblemClassifier()

    for sp in subproblems:
        result = classifier.classify_problem(sp, return_details=True)

        print(f"ID: {sp.sub_problem_id}")
        print(f"  Title: {sp.title}")
        print(f"  Type: {result.problem_type.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print()


# ============================================================================
# EXAMPLE 5: BATCH CLASSIFICATION
# ============================================================================

def example_5_batch_classification():
    """
    Example 5: Classify multiple problems at once.
    """
    print("=" * 80)
    print("EXAMPLE 5: Batch Classification")
    print("=" * 80)

    descriptions = [
        ("Implement caching layer", "Add Redis caching for frequently accessed data"),
        ("Analyze memory usage", "Investigate high memory consumption in worker processes"),
        ("Write integration tests", "Create end-to-end tests for the payment flow"),
        ("Design database schema", "Create tables and relationships for order management"),
        ("Verify data integrity", "Ensure referential integrity across all tables"),
    ]

    results = batch_classify_descriptions(descriptions)

    print("Classification Results:")
    print("-" * 80)
    for title, problem_type, confidence in results:
        print(f"{title:30s} | {problem_type.value:15s} | {confidence:.2f}")
    print()


# ============================================================================
# EXAMPLE 6: TYPE DISTRIBUTION ANALYSIS
# ============================================================================

def example_6_distribution_analysis():
    """
    Example 6: Analyze distribution of problem types.
    """
    print("=" * 80)
    print("EXAMPLE 6: Type Distribution Analysis")
    print("=" * 80)

    problems = [
        {'title': 'P1', 'description': 'Implement feature X'},
        {'title': 'P2', 'description': 'Create component Y'},
        {'title': 'P3', 'description': 'Analyze issue Z'},
        {'title': 'P4', 'description': 'Build module A'},
        {'title': 'P5', 'description': 'Test feature X'},
        {'title': 'P6', 'description': 'Verify component Y'},
        {'title': 'P7', 'description': 'Investigate bug B'},
        {'title': 'P8', 'description': 'Implement feature C'},
    ]

    classifier = ProblemClassifier()
    distribution = classifier.get_type_distribution(problems)

    print("Problem Type Distribution:")
    print("-" * 40)
    for ptype, count in distribution.items():
        percentage = (count / len(problems)) * 100 if problems else 0
        bar = "#" * int(percentage / 5)
        print(f"{ptype.value:15s} | {count:2d} | {percentage:5.1f}% | {bar}")
    print()


# ============================================================================
# EXAMPLE 7: CUSTOM CLASSIFICATION RULES
# ============================================================================

def example_7_custom_patterns():
    """
    Example 7: Add custom classification patterns for domain-specific terms.
    """
    print("=" * 80)
    print("EXAMPLE 7: Custom Classification Patterns")
    print("=" * 80)

    # Create classifier with custom patterns using add_custom_pattern method
    classifier = ProblemClassifier()

    # Add custom pattern for ML implementation
    classifier.add_custom_pattern(
        problem_type=SubProblemType.IMPLEMENTATION,
        keywords=['train', 'model', 'neural', 'deep learning', 'ml'],
        weight=1.2,
        category='ml_implementation',
        pattern_type='simple'
    )

    # Add custom pattern for QA testing
    classifier.add_custom_pattern(
        problem_type=SubProblemType.VALIDATION,
        keywords=['qa', 'quality check', 'smoke test', 'regression'],
        weight=1.3,
        category='qa_testing',
        pattern_type='simple'
    )

    # Test custom patterns
    test_problems = [
        {'description': 'Train a neural network model for image classification'},
        {'description': 'Perform smoke testing on the deployment pipeline'},
    ]

    print("Testing Custom Patterns:")
    for problem in test_problems:
        result = classifier.classify_problem(problem, return_details=True)
        print(f"Description: {problem['description']}")
        print(f"  Type: {result.problem_type.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print()


# ============================================================================
# EXAMPLE 8: HANDLING AMBIGUOUS PROBLEMS
# ============================================================================

def example_8_ambiguous_problems():
    """
    Example 8: Handle ambiguous or mixed-type problems.
    """
    print("=" * 80)
    print("EXAMPLE 8: Handling Ambiguous Problems")
    print("=" * 80)

    ambiguous_problems = [
        {
            'title': 'Fix and Verify',
            'description': 'Analyze the authentication bug, implement a fix, and verify it works'
        },
        {
            'title': 'Code Review',
            'description': 'Review the implemented code for quality and best practices'
        },
        {
            'title': 'Create Tests',
            'description': 'Write implementation code with comprehensive test coverage'
        },
    ]

    classifier = ProblemClassifier(handle_mixed_types=True)

    print("Ambiguous Problem Classification:")
    for problem in ambiguous_problems:
        result = classifier.classify_problem(problem, return_details=True)

        print(f"\nTitle: {problem['title']}")
        print(f"Description: {problem['description']}")
        print(f"Primary Type: {result.problem_type.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Mixed Type: {result.classification_metadata.get('is_mixed_type', False)}")

        if result.alternative_types:
            print("Alternative Types:")
            for alt_type, alt_score in result.alternative_types[:2]:
                print(f"  - {alt_type.value}: {alt_score:.2f}")


# ============================================================================
# EXAMPLE 9: INTEGRATION WITH WORKFLOW
# ============================================================================

def example_9_workflow_integration():
    """
    Example 9: Integrate with workflow decomposition.
    """
    print("=" * 80)
    print("EXAMPLE 9: Workflow Integration")
    print("=" * 80)

    # Simulate decomposing a high-level task
    main_task = "Build a secure authentication system"

    subtasks = [
        "Design database schema for users",
        "Implement user registration endpoint",
        "Analyze security requirements",
        "Create unit tests for authentication",
        "Verify JWT token generation",
        "Investigate OAuth integration options",
    ]

    classifier = ProblemClassifier()

    print(f"Main Task: {main_task}")
    print("\nSubtask Classification:")
    print("-" * 80)

    for i, subtask in enumerate(subtasks, 1):
        result = classifier.classify_problem(
            {'description': subtask, 'title': f'Subtask {i}'},
            return_details=True
        )

        # Map type to workflow stage
        stage_mapping = {
            SubProblemType.IMPLEMENTATION: 'Development',
            SubProblemType.ANALYSIS: 'Planning',
            SubProblemType.VALIDATION: 'Testing',
        }

        stage = stage_mapping.get(result.problem_type, 'Unknown')

        print(f"{i}. {subtask}")
        print(f"   Type: {result.problem_type.value:15s} | Stage: {stage:12s} | Conf: {result.confidence:.2f}")

    print()


# ============================================================================
# EXAMPLE 10: SERIALIZATION AND PERSISTENCE
# ============================================================================

def example_10_serialization():
    """
    Example 10: Serialize and deserialize classification results.
    """
    print("=" * 80)
    print("EXAMPLE 10: Serialization and Persistence")
    print("=" * 80)

    # Create a classification
    classifier = ProblemClassifier()
    result = classifier.classify_problem(
        {'description': 'Implement user authentication', 'title': 'Auth'},
        return_details=True
    )

    # Serialize to dictionary
    result_dict = result.to_dict()

    print("Serialized Result:")
    print(f"  Type: {result_dict['problem_type']}")
    print(f"  Confidence: {result_dict['confidence']}")
    print(f"  Metadata Keys: {list(result_dict['classification_metadata'].keys())}")

    # Deserialize back
    restored_result = ClassificationResult.from_dict(result_dict)

    print("\nRestored Result:")
    print(f"  Type: {restored_result.problem_type.value}")
    print(f"  Confidence: {restored_result.confidence}")

    # Verify restoration
    assert result.problem_type == restored_result.problem_type
    assert result.confidence == restored_result.confidence
    print("\n[OK] Serialization round-trip successful!")
    print()


# ============================================================================
# EXAMPLE 11: CONFIDENCE THRESHOLD FILTERING
# ============================================================================

def example_11_confidence_filtering():
    """
    Example 11: Filter classifications by confidence threshold.
    """
    print("=" * 80)
    print("EXAMPLE 11: Confidence Threshold Filtering")
    print("=" * 80)

    problems = [
        {'title': 'Clear Implementation', 'description': 'Implement a REST API endpoint'},
        {'title': 'Ambiguous Task', 'description': 'Review and analyze the system'},
        {'title': 'Mixed Activity', 'description': 'Fix the bug and add tests'},
    ]

    classifier = ProblemClassifier()

    print("All Classifications:")
    for problem in problems:
        result = classifier.classify_problem(problem, return_details=True)
        print(f"  {problem['title']:25s} | Type: {result.problem_type.value:15s} | Conf: {result.confidence:.2f}")

    # Filter high confidence results
    print("\nHigh Confidence Results (>= 0.70):")
    for problem in problems:
        result = classifier.classify_problem(problem, return_details=True)
        if result.confidence >= 0.70:
            print(f"  [OK] {problem['title']:25s} | {result.problem_type.value}")

    print()


# ============================================================================
# EXAMPLE 12: STATISTICS AND REPORTING
# ============================================================================

def example_12_statistics():
    """
    Example 12: Get classifier statistics and configuration.
    """
    print("=" * 80)
    print("EXAMPLE 12: Classifier Statistics")
    print("=" * 80)

    classifier = ProblemClassifier(
        confidence_threshold=0.65,
        enable_nlp_patterns=True,
        handle_mixed_types=True
    )

    # Classify some problems
    problems = [
        {'description': 'Implement feature X'},
        {'description': 'Analyze bug Y'},
        {'description': 'Test component Z'},
    ]

    for problem in problems:
        classifier.classify_problem(problem)

    # Get statistics
    stats = classifier.get_statistics()

    print("Classifier Configuration:")
    print(f"  Confidence Threshold: {stats['confidence_threshold']}")
    print(f"  NLP Patterns Enabled: {stats['nlp_patterns_enabled']}")
    print(f"  Available Types: {len(stats['available_types'])}")

    print("\nAvailable Types:")
    for t in stats['available_types']:
        print(f"  - {t}")

    print("\nPattern Count by Type:")
    for ptype, count in stats['patterns_count'].items():
        print(f"  {ptype:15s}: {count} patterns")

    print()


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run all examples."""
    examples = [
        example_1_basic_classification,
        example_2_classification_with_confidence,
        example_3_detailed_result,
        example_4_subproblem_model,
        example_5_batch_classification,
        example_6_distribution_analysis,
        example_7_custom_patterns,
        example_8_ambiguous_problems,
        example_9_workflow_integration,
        example_10_serialization,
        example_11_confidence_filtering,
        example_12_statistics,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
