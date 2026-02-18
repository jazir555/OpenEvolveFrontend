#!/usr/bin/env python3
"""
Example: ICR Pattern Learning and Prediction

This example demonstrates the ICR (Iterative Contextual Refinements) system's
ability to learn from past executions and predict outcomes.

Usage:
    cd examples
    python example_icr_learning.py
"""

import os
import sys
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# Set environment variables
os.environ.setdefault("ADAPTIVE_MDAP_TIMEOUT_MS", "5000")

from src import get_advanced_icr_integration, ICRPatternType


def main():
    """Demonstrate ICR pattern learning and prediction."""
    print("=" * 70)
    print("  EXAMPLE: ICR Pattern Learning and Prediction")
    print("=" * 70)
    print(f"\nStart Time: {datetime.now(timezone.utc).isoformat()}\n")

    # Get advanced ICR integration
    icr = get_advanced_icr_integration()

    # Phase 1: Store patterns from past executions
    print("Phase 1: Storing Patterns from Past Executions")
    print("-" * 70)

    pattern_ids = []

    # Simulate storing 20 workflow execution patterns
    print("\nStoring workflow execution patterns...")

    for i in range(20):
        passed = i % 3 != 0  # 2/3 pass rate for this domain
        domain = ["security", "ml", "distributed_systems"][i % 3]

        pattern_id = icr.store_pattern_advanced(
            pattern_type=ICRPatternType.WORKFLOW_EXECUTION,
            passed=passed,
            context={
                "domain": domain,
                "complexity": 0.5 + (i % 5) * 0.1,
                "workflow_type": "evolution" if i % 2 == 0 else "sovereign"
            },
            metrics={
                "execution_time_ms": 1000 + i * 100,
                "memory_used_mb": 512 + i * 50,
                "agent_count": 3 + (i % 4)
            }
        )
        pattern_ids.append(pattern_id)

        if i < 5 or i == 19:
            status = "[PASS]" if passed else "[FAIL]"
            print(f"  Pattern {i+1:2d}: {status} {domain} - {pattern_id}")

    print(f"\nStored {len(pattern_ids)} patterns")

    # Phase 2: Get pattern insights
    print("\n" + "=" * 70)
    print("Phase 2: Pattern Insights")
    print("=" * 70)

    insights = icr.get_pattern_insights()

    print(f"\nICR Available: {insights.get('available', False)}")
    print(f"Pattern Types Tracked: {len(insights.get('pattern_types', {}))}")

    if insights.get('available'):
        print("\nPattern Statistics:")
        print("-" * 70)

        for ptype, stats in insights.get('pattern_types', {}).items():
            print(f"\n{ptype}:")
            print(f"  Count: {stats.get('count', 0)}")
            print(f"  Pass Rate: {stats.get('pass_rate', 0):.1%}")
            print(f"  Confidence: {stats.get('confidence', 0):.1%}")

            if stats.get('recent_patterns'):
                print(f"  Recent Patterns: {len(stats['recent_patterns'])}")

    # Phase 3: Get adaptive threshold
    print("\n" + "=" * 70)
    print("Phase 3: Adaptive Threshold")
    print("=" * 70)

    threshold = icr.get_adaptive_threshold(
        ICRPatternType.WORKFLOW_EXECUTION,
        default=0.5
    )

    print(f"\nAdaptive Threshold: {threshold:.3f}")
    print("  (Threshold adapts based on historical performance)")

    # Phase 4: Find similar patterns
    print("\n" + "=" * 70)
    print("Phase 4: Pattern Similarity Search")
    print("=" * 70)

    similar = icr.find_similar_patterns(
        pattern_type=ICRPatternType.WORKFLOW_EXECUTION,
        context={
            "domain": "security",
            "complexity": 0.7
        },
        limit=5,
        similarity_threshold=0.5
    )

    print(f"\nFound {len(similar.get('similar_patterns', []))} similar patterns")

    for pattern, similarity in similar.get('similar_patterns', [])[:3]:
        status = "[PASS]" if pattern.passed else "[FAIL]"
        print(f"\n  {status} Similarity: {similarity:.1%}")
        print(f"    Domain: {pattern.context.get('domain')}")
        print(f"    Complexity: {pattern.context.get('complexity'):.3f}")

    # Phase 5: Predict outcome
    print("\n" + "=" * 70)
    print("Phase 5: Outcome Prediction")
    print("=" * 70)

    prediction = icr.predict_with_confidence(
        pattern_type=ICRPatternType.WORKFLOW_EXECUTION,
        context={
            "domain": "security",
            "complexity": 0.7,
            "workflow_type": "sovereign"
        },
        min_confidence=0.6
    )

    print(f"\nPrediction Results:")
    print(f"  Predicted Outcome: {prediction.predicted_outcome}")
    print(f"  Confidence: {prediction.confidence:.1%}")
    print(f"  Recommended Action: {prediction.recommended_action}")
    print(f"  Pattern Count: {prediction.pattern_count}")
    print(f"  Timestamp: {prediction.timestamp}")

    # Phase 6: Export patterns
    print("\n" + "=" * 70)
    print("Phase 6: Export Patterns")
    print("=" * 70)

    export_path = "/tmp/icr_patterns_export.json"
    icr.export_patterns(
        pattern_type=ICRPatternType.WORKFLOW_EXECUTION,
        filepath=export_path
    )

    print(f"\nExported patterns to: {export_path}")

    # Try to read and display count
    try:
        import json
        with open(export_path, 'r') as f:
            data = json.load(f)
            print(f"  Total patterns in export: {len(data.get('patterns', []))}")
    except Exception as e:
        print(f"  (Could not verify export: {e})")

    print("\n" + "=" * 70)
    print("  EXAMPLE COMPLETE")
    print("=" * 70)
    print(f"\nEnd Time: {datetime.now(timezone.utc).isoformat()}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
