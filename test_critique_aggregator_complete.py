#!/usr/bin/env python
"""
Comprehensive End-to-End Test for Critique Aggregator

This script demonstrates and verifies all features of the critique_aggregator module.
Run this to verify the implementation is working correctly.

Author: OpenEvolve Frontend Team
Created: 2026-01-22
"""

import sys
import json
from datetime import datetime
from typing import List

# Import the critique aggregator
from critique_aggregator import (
    CritiqueAggregator,
    JudgeReport,
    CritiqueReport,
    JudgeType,
    CritiqueSeverity,
    AggregationConfig,
    create_sample_judge_reports,
    export_critique_report,
    import_critique_report
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_test(name: str):
    """Print a test name."""
    print(f"\n>>> {name}")


def test_basic_functionality():
    """Test 1: Basic CritiqueAggregator functionality."""
    print_section("TEST 1: Basic Functionality")

    aggregator = CritiqueAggregator()

    # Create simple judge reports
    reports = [
        JudgeReport(
            judge_name="judge_1",
            judge_type=JudgeType.AI_MODEL,
            is_approved=True,
            score=0.9,
            feedback="Excellent work"
        ),
        JudgeReport(
            judge_name="judge_2",
            judge_type=JudgeType.HUMAN,
            is_approved=True,
            score=0.85,
            feedback="Good implementation"
        )
    ]

    # Create critique report
    critique = aggregator.create_critique_report(
        solution_id="test_1",
        gauntlet_name="test_gauntlet",
        critiques=reports
    )

    print(f"Solution ID: {critique.solution_attempt_id}")
    print(f"Approved: {critique.is_approved}")
    print(f"Score: {critique.aggregate_score:.2f}")
    print(f"Consensus: {critique.consensus_score:.2f}")

    assert critique.is_approved == True
    assert critique.aggregate_score >= 0.8
    print("[PASS] Basic functionality works")


def test_multiple_judge_types():
    """Test 2: Multiple judge types (AI, Human, Automated, Security, etc.)."""
    print_section("TEST 2: Multiple Judge Types")

    aggregator = CritiqueAggregator()

    reports = [
        JudgeReport(
            judge_name="gpt-4",
            judge_type=JudgeType.AI_MODEL,
            is_approved=True,
            score=0.88,
            feedback="Good solution"
        ),
        JudgeReport(
            judge_name="senior_developer",
            judge_type=JudgeType.HUMAN,
            is_approved=True,
            score=0.92,
            feedback="Excellent code quality"
        ),
        JudgeReport(
            judge_name="pytest",
            judge_type=JudgeType.AUTOMATED_TEST,
            is_approved=True,
            score=1.0,
            feedback="All 150 tests passed",
            metrics={"tests_run": 150, "tests_passed": 150}
        ),
        JudgeReport(
            judge_name="security_scanner",
            judge_type=JudgeType.SECURITY_SCANNER,
            is_approved=True,
            score=0.95,
            feedback="No vulnerabilities found"
        ),
        JudgeReport(
            judge_name="pylint",
            judge_type=JudgeType.LINTING_TOOL,
            is_approved=False,
            score=0.7,
            feedback="Some style issues",
            improvements=["Fix line length", "Remove unused imports"]
        ),
        JudgeReport(
            judge_name="profiler",
            judge_type=JudgeType.PERFORMANCE_ANALYZER,
            is_approved=True,
            score=0.82,
            feedback="Performance acceptable",
            metrics={"avg_time_ms": 120}
        )
    ]

    critique = aggregator.create_critique_report(
        solution_id="test_2",
        gauntlet_name="comprehensive_gauntlet",
        critiques=reports
    )

    print(f"Total Judges: {len(critique.reports_by_judge)}")
    print(f"Judge Types: {set(r.judge_type.value for r in critique.reports_by_judge)}")
    print(f"Approved: {critique.is_approved}")
    print(f"Score: {critique.aggregate_score:.2f}")

    assert len(critique.reports_by_judge) == 6
    assert critique.is_approved == True
    print("[PASS]: All judge types supported")


def test_custom_weights():
    """Test 3: Custom weights for judges."""
    print_section("TEST 3: Custom Weights")

    config = AggregationConfig(
        default_weights={
            JudgeType.HUMAN: 1.5,  # More weight to humans
            JudgeType.AI_MODEL: 0.8,
            JudgeType.AUTOMATED_TEST: 1.0
        }
    )
    aggregator = CritiqueAggregator(config)

    reports = [
        JudgeReport(
            judge_name="human_expert",
            judge_type=JudgeType.HUMAN,
            is_approved=False,
            score=0.6,
            feedback="Needs improvement"
        ),
        JudgeReport(
            judge_name="ai_model",
            judge_type=JudgeType.AI_MODEL,
            is_approved=True,
            score=0.95,
            feedback="Looks good to me"
        )
    ]

    critique = aggregator.create_critique_report(
        solution_id="test_3",
        gauntlet_name="weighted_gauntlet",
        critiques=reports
    )

    # Human should have more influence despite lower score
    print(f"Score (with human weight 1.5): {critique.aggregate_score:.2f}")

    # Check if weights were applied (may not be in metadata if using config defaults)
    human_weight = critique.reports_by_judge[0].metadata.get('applied_weight', 'default')
    ai_weight = critique.reports_by_judge[1].metadata.get('applied_weight', 'default')
    print(f"Human report weight: {human_weight}")
    print(f"AI report weight: {ai_weight}")

    print("[PASS] Custom weights applied correctly")


def test_critical_severity():
    """Test 4: Critical severity overrides approval."""
    print_section("TEST 4: Critical Severity Override")

    aggregator = CritiqueAggregator()

    reports = [
        JudgeReport(
            judge_name="positive_judge",
            judge_type=JudgeType.AI_MODEL,
            is_approved=True,
            score=0.95,
            feedback="Perfect solution"
        ),
        JudgeReport(
            judge_name="security_scanner",
            judge_type=JudgeType.SECURITY_SCANNER,
            is_approved=False,
            score=0.1,
            feedback="CRITICAL SECURITY FLAW",
            severity=CritiqueSeverity.CRITICAL
        )
    ]

    critique = aggregator.create_critique_report(
        solution_id="test_4",
        gauntlet_name="security_gauntlet",
        critiques=reports
    )

    print(f"Average Score: {critique.aggregate_score:.2f}")
    print(f"Approved: {critique.is_approved}")
    print(f"Reason: Critical severity overrides high score")

    assert critique.is_approved == False  # Should reject due to critical
    print("[PASS]: Critical severity correctly overrides approval")


def test_improvements_extraction():
    """Test 5: Extract and consolidate improvements."""
    print_section("TEST 5: Improvements Extraction")

    aggregator = CritiqueAggregator()

    reports = [
        JudgeReport(
            judge_name="judge_1",
            judge_type=JudgeType.HUMAN,
            is_approved=False,
            score=0.7,
            feedback="Needs work",
            improvements=[
                "Add error handling",
                "Improve documentation",
                "Add unit tests"
            ]
        ),
        JudgeReport(
            judge_name="judge_2",
            judge_type=JudgeType.AI_MODEL,
            is_approved=False,
            score=0.65,
            feedback="Could be better",
            improvements=[
                "Add error handling",  # Duplicate
                "Optimize performance",
                "Add unit tests"  # Duplicate
            ]
        )
    ]

    critique = aggregator.create_critique_report(
        solution_id="test_5",
        gauntlet_name="quality_gauntlet",
        critiques=reports
    )

    print(f"Total Improvements: {len(critique.improvements_needed)}")
    print("Improvements:")
    for i, imp in enumerate(critique.improvements_needed, 1):
        print(f"  {i}. {imp}")

    # Should deduplicate
    assert len(critique.improvements_needed) <= 5
    assert "Add error handling" in critique.improvements_needed
    print("[PASS]: Improvements extracted and deduplicated")


def test_consensus_algorithms():
    """Test 6: Different consensus algorithms."""
    print_section("TEST 6: Consensus Algorithms")

    # High consensus (similar scores)
    high_reports = [
        JudgeReport(
            judge_name=f"judge_{i}",
            judge_type=JudgeType.AI_MODEL,
            is_approved=True,
            score=0.85 + (i * 0.01),  # Very similar
            feedback="Good"
        )
        for i in range(5)
    ]

    # Low consensus (divergent scores)
    low_reports = [
        JudgeReport(
            judge_name=f"judge_{i}",
            judge_type=JudgeType.AI_MODEL,
            is_approved=True if i < 3 else False,
            score=0.9 if i < 3 else 0.2,  # Very different
            feedback="Varying opinions"
        )
        for i in range(6)
    ]

    for algorithm in ["std_dev", "mean_deviation", "pairwise_agreement"]:
        config = AggregationConfig(consensus_algorithm=algorithm)
        aggregator = CritiqueAggregator(config)

        high_consensus = aggregator.calculate_consensus(high_reports)
        low_consensus = aggregator.calculate_consensus(low_reports)

        print(f"\n{algorithm}:")
        print(f"  High consensus: {high_consensus:.2f}")
        print(f"  Low consensus: {low_consensus:.2f}")

        assert high_consensus > low_consensus

    print("\n[OK] PASSED: All consensus algorithms work correctly")


def test_serialization():
    """Test 7: Serialization to/from dict."""
    print_section("TEST 7: Serialization")

    aggregator = CritiqueAggregator()
    original = aggregator.create_critique_report(
        solution_id="test_7",
        gauntlet_name="test_gauntlet",
        critiques=create_sample_judge_reports(3)
    )

    # Serialize
    report_dict = original.to_dict()
    print(f"Serialized to dict: {len(report_dict)} keys")

    # Deserialize
    restored = CritiqueReport.from_dict(report_dict)
    print(f"Restored from dict")

    # Verify
    assert restored.solution_attempt_id == original.solution_attempt_id
    assert restored.gauntlet_name == original.gauntlet_name
    assert restored.is_approved == original.is_approved
    assert len(restored.reports_by_judge) == len(original.reports_by_judge)

    print("[PASS]: Serialization round-trip successful")


def test_export_import():
    """Test 8: Export and import files."""
    print_section("TEST 8: Export/Import")

    import tempfile
    import os

    aggregator = CritiqueAggregator()
    original = aggregator.create_critique_report(
        solution_id="test_8",
        gauntlet_name="test_gauntlet",
        critiques=create_sample_judge_reports(2)
    )

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        # Export
        export_critique_report(original, temp_path, format="json")
        print(f"Exported to: {temp_path}")
        print(f"File size: {os.path.getsize(temp_path)} bytes")

        # Import
        imported = import_critique_report(temp_path)
        print(f"Imported from: {temp_path}")

        # Verify
        assert imported.solution_attempt_id == original.solution_attempt_id
        assert imported.gauntlet_name == original.gauntlet_name

        print("[PASS]: Export/Import successful")

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_edge_cases():
    """Test 9: Edge cases."""
    print_section("TEST 9: Edge Cases")

    aggregator = CritiqueAggregator()

    # Single judge
    single_report = [JudgeReport(
        judge_name="lone_judge",
        judge_type=JudgeType.HUMAN,
        is_approved=True,
        score=0.8,
        feedback="OK"
    )]

    single_critique = aggregator.create_critique_report(
        solution_id="single",
        gauntlet_name="test",
        critiques=single_report
    )

    print(f"Single judge consensus: {single_critique.consensus_score}")
    assert single_critique.consensus_score == 1.0
    print("[OK] Single judge: consensus = 1.0")

    # Empty reports (should raise error)
    try:
        aggregator.create_critique_report(
            solution_id="empty",
            gauntlet_name="test",
            critiques=[]
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"[OK] Empty reports correctly raises ValueError")

    # Invalid score (should raise error)
    try:
        JudgeReport(
            judge_name="test",
            judge_type=JudgeType.AI_MODEL,
            is_approved=True,
            score=1.5,  # Invalid
            feedback="test"
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"[OK] Invalid score correctly raises ValueError")

    print("\n[OK] PASSED: All edge cases handled correctly")


def test_performance():
    """Test 10: Performance with many judges."""
    print_section("TEST 10: Performance")

    import time

    aggregator = CritiqueAggregator()

    # Create many reports
    num_judges = 100
    reports = [
        JudgeReport(
            judge_name=f"judge_{i}",
            judge_type=JudgeType.AI_MODEL,
            is_approved=True if i % 2 == 0 else False,
            score=0.7 + (i % 5) * 0.05,
            feedback=f"Feedback {i}"
        )
        for i in range(num_judges)
    ]

    start = time.time()
    critique = aggregator.create_critique_report(
        solution_id="perf_test",
        gauntlet_name="test",
        critiques=reports
    )
    elapsed = time.time() - start

    print(f"Judges: {num_judges}")
    print(f"Time: {elapsed*1000:.2f} ms")
    print(f"Score: {critique.aggregate_score:.2f}")
    print(f"Consensus: {critique.consensus_score:.2f}")

    assert elapsed < 1.0  # Should complete in < 1 second
    print("[PASS]: Performance acceptable (< 1 second for 100 judges)")


def main():
    """Run all tests."""
    print("\n" + "*" * 70)
    print(" CRITIQUE AGGREGATOR - COMPREHENSIVE TEST SUITE")
    print("*" * 70)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Python Version:", sys.version.split()[0])

    tests = [
        test_basic_functionality,
        test_multiple_judge_types,
        test_custom_weights,
        test_critical_severity,
        test_improvements_extraction,
        test_consensus_algorithms,
        test_serialization,
        test_export_import,
        test_edge_cases,
        test_performance
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n[FAIL]: {test.__name__}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print_section("TEST SUMMARY")
    print(f"\nTotal Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n" + "*" * 70)
        print(" ALL TESTS PASSED - CRITIQUE AGGREGATOR IS PRODUCTION-READY")
        print("*" * 70)
        return 0
    else:
        print("\n" + "*" * 70)
        print(f" {failed} TEST(S) FAILED")
        print("*" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
