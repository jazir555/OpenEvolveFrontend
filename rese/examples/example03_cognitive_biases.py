"""
Example 3: Cognitive Bias Detection (Φ₂)

This example demonstrates how to detect and mitigate cognitive biases in problem formulation.
"""

import sys
sys.path.insert(0, r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend')

from phase1.cognitive_biases import CognitiveBiasDetector
from core.symbolic_constraint_engine import Constraint, ConstraintType

def main():
    print("=" * 60)
    print("Example 3: Cognitive Bias Detection (Φ₂)")
    print("=" * 60)
    print()

    # Create constraints with potential biases
    constraints = [
        Constraint(
            id="c1",
            type=ConstraintType.HARD,
            description="This solution MUST work because similar solutions worked before",
            formalization="solution = similar_solution",
            source="expert"  # Authority bias potential
        ),
        Constraint(
            id="c2",
            type=ConstraintType.HARD,
            description="We should stick with our initial approach",
            formalization="approach = initial_approach",
            source="user"  # Sunk cost bias potential
        ),
        Constraint(
            id="c3",
            type=ConstraintType.HARD,
            description="We know X is true, so Y must also be true",
            formalization="X ∧ (X → Y)",
            source="expert"  # Confirmation bias potential
        ),
        Constraint(
            id="c4",
            type=ConstraintType.SOFT,
            description="Consider all variables equally important",
            formalization="∀ x, y: importance(x) = importance(y)",
            source="user"
        )
    ]

    print("Analyzing Constraints for Cognitive Biases:")
    print("-" * 60)
    for constraint in constraints:
        print(f"{constraint.id}: {constraint.description}")
    print()

    # Create bias detector
    detector = CognitiveBiasDetector()

    # Analyze constraints
    print("Running Bias Detection...")
    print("-" * 60)
    report = detector.analyze_constraints(constraints)

    # Display results
    print()
    print("Bias Detection Report:")
    print("=" * 60)
    print(f"Overall Bias Score: {report.overall_bias_score:.2f}")
    print(f"Total Detections: {report.total_detections}")
    print()

    print("Detections by Severity:")
    for severity, count in report.by_severity.items():
        print(f"  {severity.name}: {count}")
    print()

    print("Individual Detections:")
    print("-" * 60)
    for detection in report.detections:
        print(f"\nBias Type: {detection.bias_type.value}")
        print(f"Severity: {detection.severity.name}")
        print(f"Confidence: {detection.confidence:.2f}")
        print(f"Description: {detection.description}")
        if detection.evidence:
            print(f"Evidence: {detection.evidence}")
        if detection.suggestion:
            print(f"Suggestion: {detection.suggestion}")
        if detection.affected_elements:
            print(f"Affected: {', '.join(detection.affected_elements)}")

    # Recommendations
    print()
    print("=" * 60)
    print("Recommendations:")
    print("-" * 60)

    if report.overall_bias_score > 0.7:
        print("⚠️  HIGH BIAS DETECTED")
        print("   - Review all constraints carefully")
        print("   - Consider debiasing intervention")
        print("   - Get second opinion from independent reviewer")
    elif report.overall_bias_score > 0.4:
        print("⚠️  MODERATE BIAS DETECTED")
        print("   - Review highest-severity detections")
        print("   - Consider softening biased constraints")
    else:
        print("✓ LOW BIAS")
        print("   - Constraints appear relatively unbiased")

    print()

    # Debiasing example
    if report.total_detections > 0:
        print("=" * 60)
        print("Debiasing Example:")
        print("-" * 60)

        from phase1.cognitive_biases import Debiaser
        debiaser = Debiaser()

        # Show before/after for one detection
        if report.detections:
            detection = report.detections[0]
            print(f"\nOriginal Constraint: {detection.affected_elements[0]}")
            print(f"Detected Bias: {detection.bias_type.value}")
            print(f"Debiasing Suggestion: {detection.suggestion}")

    print()
    print("=" * 60)
    print("Example 3 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
