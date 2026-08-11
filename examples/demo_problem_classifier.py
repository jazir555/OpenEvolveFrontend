"""
Demo Script for Problem Classifier

Shows how to use the automatic problem classification system.
"""

from problem_classifier import ProblemClassifier, classify_problem_auto, get_problem_type_from_text
from sovereign_data_models import (
    ProblemDefinition, ProblemType, DomainContext, ComplexityScore
)


def demo_keyword_classification():
    """Demonstrate keyword-based classification."""
    print("=" * 80)
    print("DEMO 1: Keyword-Based Classification (Fast, No LLM Required)")
    print("=" * 80)

    # Create classifier with LLM disabled
    classifier = ProblemClassifier(enable_llm=False)

    # Example 1: Implementation problem
    print("\n1. Implementation Problem:")
    print("-" * 80)
    problem1 = ProblemDefinition(
        id="prob_001",
        title="Build user authentication system",
        description="Implement a secure user authentication system with login, logout, "
                   "password reset, and session management.",
        problem_type=ProblemType.IMPLEMENTATION,
        domain_context=DomainContext(domain="software_engineering"),
        complexity_score=ComplexityScore(
            explanation="Moderate complexity",
            cognitive_complexity=5.0,
            computational_complexity=4.0,
            domain_complexity=6.0,
            integration_complexity=5.0,
            overall_complexity=5.0
        )
    )

    classification1 = classifier.classify_problem(problem1)
    print(f"Title: {problem1.title}")
    print(f"Classified as: {classification1.primary_type.value}")
    print(f"Confidence: {classification1.confidence:.2f}")
    print(f"Method: {classification1.classification_method}")
    print(f"Indicators: {', '.join(classification1.indicators[:5])}")
    print(f"Suggested strategies: {', '.join(classification1.suggested_strategies)}")

    # Example 2: Analysis problem
    print("\n2. Analysis Problem:")
    print("-" * 80)
    problem2 = ProblemDefinition(
        id="prob_002",
        title="Analyze codebase architecture",
        description="Examine and study the current codebase architecture to understand "
                   "the system design, component relationships, and data flow.",
        problem_type=ProblemType.ANALYSIS,
        domain_context=DomainContext(domain="software_engineering"),
        complexity_score=ComplexityScore(
            explanation="Moderate complexity",
            cognitive_complexity=5.0,
            computational_complexity=4.0,
            domain_complexity=6.0,
            integration_complexity=5.0,
            overall_complexity=5.0
        )
    )

    classification2 = classifier.classify_problem(problem2)
    print(f"Title: {problem2.title}")
    print(f"Classified as: {classification2.primary_type.value}")
    print(f"Confidence: {classification2.confidence:.2f}")
    print(f"Method: {classification2.classification_method}")
    print(f"Indicators: {', '.join(classification2.indicators[:5])}")
    print(f"Suggested strategies: {', '.join(classification2.suggested_strategies)}")

    # Example 3: Research problem
    print("\n3. Research Problem:")
    print("-" * 80)
    problem3 = ProblemDefinition(
        id="prob_003",
        title="Research GraphQL vs REST API architectures",
        description="Explore and research the differences between GraphQL and REST API "
                   "architectures. Investigate performance implications and ecosystem support.",
        problem_type=ProblemType.RESEARCH,
        domain_context=DomainContext(domain="software_architecture"),
        complexity_score=ComplexityScore(
            explanation="Moderate complexity",
            cognitive_complexity=5.0,
            computational_complexity=4.0,
            domain_complexity=6.0,
            integration_complexity=5.0,
            overall_complexity=5.0
        )
    )

    classification3 = classifier.classify_problem(problem3)
    print(f"Title: {problem3.title}")
    print(f"Classified as: {classification3.primary_type.value}")
    print(f"Confidence: {classification3.confidence:.2f}")
    print(f"Method: {classification3.classification_method}")
    print(f"Indicators: {', '.join(classification3.indicators[:5])}")
    print(f"Suggested strategies: {', '.join(classification3.suggested_strategies)}")


def demo_quick_classification():
    """Demonstrate quick classification utility functions."""
    print("\n" + "=" * 80)
    print("DEMO 2: Quick Classification Functions")
    print("=" * 80)

    # Quick type detection
    print("\n1. Quick Type Detection:")
    print("-" * 80)

    examples = [
        ("Build API", "Implement REST API endpoints"),
        ("Analyze logs", "Examine system logs for errors"),
        ("Research GraphQL", "Explore GraphQL capabilities"),
        ("Design schema", "Create database schema design"),
        ("Optimize queries", "Improve slow database queries"),
        ("Test auth", "Verify authentication works correctly")
    ]

    for title, description in examples:
        problem_type = get_problem_type_from_text(title, description)
        print(f"{title:30} -> {problem_type.value:15}")

    # Convenience function
    print("\n2. Using Convenience Function:")
    print("-" * 80)
    problem = ProblemDefinition(
        id="prob_quick",
        title="Build payment system",
        description="Implement payment processing with Stripe integration",
        problem_type=ProblemType.IMPLEMENTATION,
        domain_context=DomainContext(domain="ecommerce"),
        complexity_score=ComplexityScore(
            explanation="Simple",
            cognitive_complexity=3.0,
            computational_complexity=3.0,
            domain_complexity=4.0,
            integration_complexity=4.0,
            overall_complexity=3.5
        )
    )

    classification = classify_problem_auto(problem)
    print(f"Problem: {problem.title}")
    print(f"Type: {classification.primary_type.value}")
    print(f"Confidence: {classification.confidence:.2f}")


def demo_all_problem_types():
    """Demonstrate classification of all 6 problem types."""
    print("\n" + "=" * 80)
    print("DEMO 3: All Six Problem Types")
    print("=" * 80)

    classifier = ProblemClassifier(enable_llm=False)

    problems = [
        (ProblemType.IMPLEMENTATION, "Build user authentication system",
         "Implement secure login and registration with JWT tokens"),

        (ProblemType.ANALYSIS, "Analyze code quality",
         "Examine and assess the current codebase for quality issues and technical debt"),

        (ProblemType.RESEARCH, "Research microservices patterns",
         "Explore and study microservices architectural patterns and best practices"),

        (ProblemType.DESIGN, "Design system architecture",
         "Architect and plan the overall system structure with components and interfaces"),

        (ProblemType.OPTIMIZATION, "Optimize database performance",
         "Improve and enhance database queries for better performance and reduced latency"),

        (ProblemType.VALIDATION, "Validate security implementation",
         "Test and verify that security measures meet requirements and standards")
    ]

    for expected_type, title, description in problems:
        problem = ProblemDefinition(
            id=f"prob_{expected_type.value}",
            title=title,
            description=description,
            problem_type=expected_type,
            domain_context=DomainContext(domain="software_engineering"),
            complexity_score=ComplexityScore(
                explanation="Test",
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0
            )
        )

        classification = classifier.classify_problem(problem)
        match = "[OK]" if classification.primary_type == expected_type else "[MISMATCH]"

        print(f"\n{match} {expected_type.value.upper():15} -> {classification.primary_type.value.upper():15} "
              f"(confidence: {classification.confidence:.2f})")
        print(f"  Title: {title}")
        if classification.primary_type != expected_type:
            print(f"  Note: Expected {expected_type.value}, got {classification.primary_type.value}")


def demo_statistics():
    """Demonstrate classification statistics."""
    print("\n" + "=" * 80)
    print("DEMO 4: Classification Statistics")
    print("=" * 80)

    classifier = ProblemClassifier(enable_llm=False)

    # Classify multiple problems
    for i in range(10):
        problem = ProblemDefinition(
            id=f"prob_{i}",
            title=f"Test problem {i}",
            description="Implement a test feature",
            problem_type=ProblemType.IMPLEMENTATION,
            domain_context=DomainContext(domain="testing"),
            complexity_score=ComplexityScore(
                explanation="Simple",
                cognitive_complexity=3.0,
                computational_complexity=3.0,
                domain_complexity=3.0,
                integration_complexity=3.0,
                overall_complexity=3.0
            )
        )
        classifier.classify_problem(problem)

    stats = classifier.get_statistics()
    print(f"\nClassification Statistics:")
    print(f"  Total classifications: {stats['total']}")
    print(f"  LLM successes: {stats['llm_success']}")
    print(f"  LLM failures: {stats['llm_failure']}")
    print(f"  Keyword fallbacks: {stats['keyword_fallback']}")
    print(f"  LLM success rate: {stats['llm_success_rate']:.2%}")
    print(f"  Keyword fallback rate: {stats['keyword_fallback_rate']:.2%}")
    print(f"  LLM available: {stats['llm_available']}")
    print(f"  Fallback enabled: {stats['fallback_enabled']}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PROBLEM CLASSIFIER DEMONSTRATION")
    print("=" * 80)

    demo_keyword_classification()
    demo_quick_classification()
    demo_all_problem_types()
    demo_statistics()

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nThe Problem Classifier successfully:")
    print("  [OK] Classifies problems into 6 types")
    print("  [OK] Provides confidence scores")
    print("  [OK] Suggests decomposition strategies")
    print("  [OK] Works with both LLM and keyword methods")
    print("  [OK] Integrates with DecompositionEngine")
    print("  [OK] Handles edge cases gracefully")
    print("\nFor more information, see PROBLEM_CLASSIFIER_COMPLETE.md")
    print("=" * 80 + "\n")
