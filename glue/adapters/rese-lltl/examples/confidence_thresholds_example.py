#!/usr/bin/env python3
"""
Example: Formal Propositional Commitments with Confidence Thresholds

Demonstrates the DEE -> SCE translation with confidence threshold tracking.

Following RESE Technical Manual §2.2:
"DEE -> SCE (Auditability): The DEE's statistical results are converted
into auditable Formal Propositional Commitments by assigning explicit
Confidence Thresholds that the SCE can integrate into its logic graph
for contradiction detection."

Author: RESE Team
Created: 2026-02-04
"""

import os
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "lib"))

from datetime import datetime, timezone

# Import confidence tracking modules
try:
    from confidence_tracker import ConfidenceTracker, ConfidenceLevel
    from formal_commitments import (
        FormalCommitmentsHandler,
        FormalCommitment,
        CommitmentStatus
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Error: Required modules not available: {e}")
    MODULES_AVAILABLE = False
    sys.exit(1)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def example_1_basic_threshold_calculation():
    """Example 1: Basic confidence threshold calculation."""
    print_section("Example 1: Basic Confidence Threshold Calculation")

    # Initialize confidence tracker
    tracker = ConfidenceTracker()

    # Calculate thresholds for different confidence levels
    test_confidences = [0.98, 0.85, 0.70, 0.50]

    for confidence in test_confidences:
        threshold = tracker.calculate_threshold(
            confidence=confidence,
            derivation_method="tiered",
            correlation_id=f"example-1-{confidence}"
        )

        print(f"\nConfidence: {confidence:.2f}")
        print(f"  -> Threshold: {threshold.threshold:.2f}")
        print(f"  -> Level: {threshold.level.value.upper()}")
        print(f"  -> Significance: {threshold.significance_level:.3f}")


def example_2_formal_commitment_creation():
    """Example 2: Create formal commitment from DEE result."""
    print_section("Example 2: Formal Commitment Creation")

    # Initialize handler
    tracker = ConfidenceTracker()
    handler = FormalCommitmentsHandler(confidence_tracker=tracker)

    # Simulate DEE statistical result
    dee_result = {
        'hypothesis_statement': 'Lattice confinement enables LENR',
        'confidence': 0.85,
        'p_value': 0.02,
        'confidence_interval': (0.78, 0.92),
        'expected_value': 0.85
    }

    print("\nDEE Statistical Result:")
    print(f"  Hypothesis: {dee_result['hypothesis_statement']}")
    print(f"  Confidence: {dee_result['confidence']:.2f}")
    print(f"  P-value: {dee_result['p_value']:.3f}")
    print(f"  Confidence Interval: [{dee_result['confidence_interval'][0]:.2f}, {dee_result['confidence_interval'][1]:.2f}]")

    # Convert to formal commitment
    commitment, error = handler.create_commitment(
        statistical_result=dee_result,
        source_hypothesis='hypothesis-lenr-001',
        derivation_method='mcts_validation',
        correlation_id='example-2-001'
    )

    if error:
        print(f"\nError: {error}")
        return

    print("\nFormal Propositional Commitment:")
    print(f"  Proposition ID: {commitment.proposition_id}")
    print(f"  Status: {commitment.status.value.upper()}")
    print(f"  Confidence Threshold: {commitment.confidence_threshold:.2f}")
    print(f"\n  Formal Statement:")
    print(f"    {commitment.statement}")

    print(f"\n  Statistical Evidence:")
    for key, value in commitment.statistical_evidence.items():
        print(f"    {key}: {value}")

    # Convert to SCE constraint format
    sce_constraint = commitment.to_sce_constraint()
    print(f"\n  SCE Constraint Format:")
    print(f"    Type: {sce_constraint['type']}")
    print(f"    Constraint ID: {sce_constraint['constraint_id']}")


def example_3_audit_trail():
    """Example 3: Audit trail tracking."""
    print_section("Example 3: Audit Trail Tracking")

    # Initialize handler
    tracker = ConfidenceTracker()
    handler = FormalCommitmentsHandler(confidence_tracker=tracker)

    # Create multiple commitments
    hypotheses = [
        {'statement': 'x > 5', 'confidence': 0.90},
        {'statement': 'y < 10', 'confidence': 0.75},
        {'statement': 'z >= 3', 'confidence': 0.60}
    ]

    for i, hyp in enumerate(hypotheses):
        dee_result = {
            'hypothesis_statement': hyp['statement'],
            'confidence': hyp['confidence'],
            'p_value': 0.03,
            'confidence_interval': (hyp['confidence'] - 0.05, hyp['confidence'] + 0.05),
            'expected_value': hyp['confidence']
        }

        commitment, _ = handler.create_commitment(
            statistical_result=dee_result,
            source_hypothesis=f'hypothesis-{i}',
            derivation_method='test',
            correlation_id=f'example-3-{i}'
        )

        print(f"\nCreated commitment {i+1}:")
        print(f"  Statement: {hyp['statement']}")
        print(f"  Confidence: {hyp['confidence']:.2f}")
        print(f"  Threshold: {commitment.confidence_threshold:.2f}")

    # Get audit trail
    print("\n" + "-" * 80)
    print("Confidence Tracker History:")
    history = tracker.get_history(limit=10)

    for entry in history:
        print(f"\n  Proposition: {entry.proposition_id}")
        print(f"  Input Confidence: {entry.input_confidence:.2f}")
        print(f"  Calculated Threshold: {entry.calculated_threshold.threshold:.2f}")
        print(f"  Level: {entry.calculated_threshold.level.value}")

    # Get handler stats
    print("\n" + "-" * 80)
    print("Handler Statistics:")
    stats = handler.get_stats()
    print(f"  Total Commitments: {stats['commitments']['total']}")
    print(f"  By Status: {stats['commitments']['by_status']}")


def example_4_contradiction_detection():
    """Example 4: Contradiction detection with confidence thresholds."""
    print_section("Example 4: Contradiction Detection")

    # Initialize handler
    tracker = ConfidenceTracker()
    handler = FormalCommitmentsHandler(confidence_tracker=tracker)

    # Create contradictory commitments
    print("\nCreating potentially contradictory commitments...")

    # Commitment 1: x > 5 (high confidence)
    result1 = {
        'hypothesis_statement': 'x > 5',
        'confidence': 0.90,
        'p_value': 0.01,
        'confidence_interval': (0.85, 0.95),
        'expected_value': 0.90
    }

    commitment1, _ = handler.create_commitment(
        statistical_result=result1,
        source_hypothesis='hypothesis-1',
        derivation_method='test',
        correlation_id='example-4-1'
    )

    print(f"\nCommitment 1:")
    print(f"  Statement: {result1['hypothesis_statement']}")
    print(f"  Confidence: {result1['confidence']:.2f}")
    print(f"  Threshold: {commitment1.confidence_threshold:.2f}")

    # Commitment 2: x < 3 (high confidence) - contradictory
    result2 = {
        'hypothesis_statement': 'x < 3',
        'confidence': 0.85,
        'p_value': 0.02,
        'confidence_interval': (0.78, 0.92),
        'expected_value': 0.85
    }

    commitment2, _ = handler.create_commitment(
        statistical_result=result2,
        source_hypothesis='hypothesis-2',
        derivation_method='test',
        correlation_id='example-4-2'
    )

    print(f"\nCommitment 2:")
    print(f"  Statement: {result2['hypothesis_statement']}")
    print(f"  Confidence: {result2['confidence']:.2f}")
    print(f"  Threshold: {commitment2.confidence_threshold:.2f}")

    # Mark both as integrated
    handler.update_commitment_status(commitment1.proposition_id, CommitmentStatus.INTEGRATED)
    handler.update_commitment_status(commitment2.proposition_id, CommitmentStatus.INTEGRATED)

    # Detect contradictions
    print("\n" + "-" * 80)
    print("Detecting contradictions...")
    contradictions = handler.detect_contradictions(correlation_id='example-4')

    if contradictions:
        print(f"\nFound {len(contradictions)} contradiction(s):")
        for report in contradictions:
            print(f"\n  Report ID: {report.report_id}")
            print(f"  Type: {report.contradiction_type}")
            print(f"  Reason: {report.reason}")
            print(f"  Contradicted Commitments: {report.contradicted_commitments}")
    else:
        print("\nNo contradictions detected (or detection not triggered)")


def example_5_strategies():
    """Example 5: Different threshold calculation strategies."""
    print_section("Example 5: Threshold Calculation Strategies")

    # Test confidence
    test_confidence = 0.75

    # Tiered strategy
    tracker_tiered = ConfidenceTracker(config={"calculation_strategy": "tiered"})
    threshold_tiered = tracker_tiered.calculate_threshold(
        confidence=test_confidence,
        derivation_method="tiered"
    )

    # Linear strategy
    tracker_linear = ConfidenceTracker(config={"calculation_strategy": "linear"})
    threshold_linear = tracker_linear.calculate_threshold(
        confidence=test_confidence,
        derivation_method="linear"
    )

    print(f"\nInput Confidence: {test_confidence:.2f}")
    print("\n" + "-" * 80)

    print("\nTiered Strategy:")
    print(f"  Threshold: {threshold_tiered.threshold:.2f}")
    print(f"  Level: {threshold_tiered.level.value}")

    print("\nLinear Strategy:")
    print(f"  Threshold: {threshold_linear.threshold:.2f}")
    print(f"  Level: {threshold_linear.level.value}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("  FORMAL PROPOSITIONAL COMMITMENTS WITH CONFIDENCE THRESHOLDS")
    print("  DEE -> SCE Translation Layer Examples")
    print("=" * 80)

    try:
        example_1_basic_threshold_calculation()
        example_2_formal_commitment_creation()
        example_3_audit_trail()
        example_4_contradiction_detection()
        example_5_strategies()

        print("\n" + "=" * 80)
        print("  ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
