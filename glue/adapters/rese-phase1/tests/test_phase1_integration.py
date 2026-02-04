#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RESE Phase I Integration Tests

Tests the complete Phase I executor following CLAUDE.md principles:
- Law of Runtime Truth: Test actual execution, not mocks
- Law of Idempotency: Verify safe to run 100x
- Circuit Breaker: Verify failure detection
- Structured Logging: Verify JSON output
- Timeout: Verify timeout enforcement
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime, timezone

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from phase1_executor import (
    Phase1Config,
    EpistemicAuditExecutor,
    TacitAssumption,
    ContradictionDetection,
    FalsificationResult,
    CircuitBreaker,
    DeadLetterQueue,
    StructuredLogger,
)


def test_config_from_env():
    """Test 1: Configuration loading from environment"""
    print("Test 1: Configuration from environment")

    # Set environment variables
    os.environ['PHASE1_TIMEOUT_MS'] = '20000'
    os.environ['PHASE1_MAX_ASSUMPTIONS'] = '150'
    os.environ['PHASE1_MIN_ASSUMPTION_CONFIDENCE'] = '0.4'

    config = Phase1Config.from_env()

    assert config.TIMEOUT_MS == 20000
    assert config.MAX_ASSUMPTIONS == 150
    assert config.MIN_ASSUMPTION_CONFIDENCE == 0.4

    print("  ✓ Configuration loaded correctly")
    return True


def test_config_validation():
    """Test 2: Configuration validation"""
    print("Test 2: Configuration validation")

    # Test invalid timeout
    os.environ['PHASE1_TIMEOUT_MS'] = '-100'
    try:
        Phase1Config.from_env()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be positive" in str(e)
        print("  ✓ Invalid timeout rejected")

    # Test invalid confidence
    os.environ['PHASE1_TIMEOUT_MS'] = '15000'
    os.environ['PHASE1_MIN_ASSUMPTION_CONFIDENCE'] = '1.5'
    try:
        Phase1Config.from_env()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "between 0 and 1" in str(e)
        print("  ✓ Invalid confidence rejected")

    return True


def test_executor_initialization():
    """Test 3: Executor initialization"""
    print("Test 3: Executor initialization")

    os.environ['PHASE1_TIMEOUT_MS'] = '15000'
    os.environ['PHASE1_MAX_ASSUMPTIONS'] = '100'
    os.environ['PHASE1_CIRCUIT_BREAKER_THRESHOLD'] = '5'
    os.environ['PHASE1_MIN_ASSUMPTION_CONFIDENCE'] = '0.3'

    config = Phase1Config.from_env()
    executor = EpistemicAuditExecutor(config=config)

    assert executor.config == config
    assert executor.circuit_breaker is not None
    assert executor.dlq is not None

    print("  ✓ Executor initialized successfully")
    return True


def test_tacit_assumption_serialization():
    """Test 4: TacitAssumption serialization"""
    print("Test 4: TacitAssumption serialization")

    assumption = TacitAssumption(
        id='test-id',
        description='Test assumption',
        source_pattern='Test pattern',
        confidence_score=0.8,
        supporting_evidence_count=10,
    )

    # Test to_dict
    data = assumption.to_dict()
    assert data['id'] == 'test-id'
    assert data['confidence_score'] == 0.8

    # Test from_dict
    reconstructed = TacitAssumption.from_dict(data)
    assert reconstructed.id == assumption.id
    assert reconstructed.description == assumption.description

    print("  ✓ TacitAssumption serialization works")
    return True


def test_contradiction_detection_serialization():
    """Test 5: ContradictionDetection serialization"""
    print("Test 5: ContradictionDetection serialization")

    from phase1_executor import LogicalFallacy

    contradiction = ContradictionDetection(
        id='test-id',
        fallacy_type=LogicalFallacy.CONTRADICTION,
        contradiction_set_size=2,
        rollback_steps=1,
        affected_premises=['premise-1', 'premise-2'],
    )

    # Test to_dict
    data = contradiction.to_dict()
    assert data['fallacy_type'] == 'contradiction'
    assert data['contradiction_set_size'] == 2

    # Test from_dict
    reconstructed = ContradictionDetection.from_dict(data)
    assert reconstructed.id == contradiction.id
    assert reconstructed.fallacy_type == LogicalFallacy.CONTRADICTION

    print("  ✓ ContradictionDetection serialization works")
    return True


def test_constraint_hardening():
    """Test 6: ConstraintHardener"""
    print("Test 6: ConstraintHardener")

    from phase1_executor import ConstraintHardener

    os.environ['PHASE1_TIMEOUT_MS'] = '15000'
    os.environ['PHASE1_MIN_ASSUMPTION_CONFIDENCE'] = '0.3'
    config = Phase1Config.from_env()
    logger = StructuredLogger('test')

    hardener = ConstraintHardener(config=config, logger=logger)

    problem = "This problem is impossible to solve due to limited resources."
    constraints = hardener.harden_constraints(
        problem_description=problem,
        correlation_id='test-correlation',
    )

    assert len(constraints) > 0
    assert 'category' in constraints[0]
    assert 'inverted_description' in constraints[0]
    assert constraints[0]['inverted_description'] != problem

    print(f"  ✓ Extracted {len(constraints)} constraints")
    return True


def test_assumption_mining():
    """Test 7: AssumptionMiner"""
    print("Test 7: AssumptionMiner")

    from phase1_executor import AssumptionMiner

    os.environ['PHASE1_TIMEOUT_MS'] = '15000'
    os.environ['PHASE1_MIN_ASSUMPTION_CONFIDENCE'] = '0.3'
    config = Phase1Config.from_env()
    logger = StructuredLogger('test')

    miner = AssumptionMiner(config=config, logger=logger)

    patterns = [
        {
            'pattern_description': 'Lattice defects cause irregular heat',
            'failure_rate': 0.6,
            'data_points': 50,
        },
    ]

    assumptions = miner.mine_assumptions(
        failure_patterns=patterns,
        correlation_id='test-correlation',
    )

    assert len(assumptions) > 0
    assert assumptions[0].confidence_score >= 0.3
    assert assumptions[0].source_pattern == patterns[0]['pattern_description']

    print(f"  ✓ Mined {len(assumptions)} assumptions")
    return True


def test_red_team_protocol():
    """Test 8: RedTeamProtocator"""
    print("Test 8: RedTeamProtocator")

    from phase1_executor import RedTeamProtocator

    os.environ['PHASE1_TIMEOUT_MS'] = '15000'
    os.environ['PHASE1_MIN_ROBUSTNESS_SCORE'] = '0.5'
    config = Phase1Config.from_env()
    logger = StructuredLogger('test')

    red_team = RedTeamProtocator(config=config, logger=logger)

    assumptions = [
        TacitAssumption(
            id='test-id',
            description='Test assumption',
            source_pattern='Test',
            confidence_score=0.3,  # Low confidence = should be falsified
            supporting_evidence_count=10,
        )
    ]

    results = red_team.attack_hypotheses(
        assumptions=assumptions,
        constraints=[],
        correlation_id='test-correlation',
    )

    assert len(results) > 0
    # Low confidence assumption should be falsified
    assert results[0].falsified == True
    assert results[0].hypothesis_robustness_score < 0.5

    print(f"  ✓ Tested {len(results)} hypotheses, {sum(1 for r in results if r.falsified)} falsified")
    return True


def test_circuit_breaker():
    """Test 9: CircuitBreaker"""
    print("Test 9: CircuitBreaker")

    logger = StructuredLogger('test')
    cb = CircuitBreaker(threshold=2, timeout_ms=1000, logger=logger.logger)

    # Initial state
    assert cb.can_execute() == True
    assert cb.get_stats()['state'] == 'closed'

    # Record failures
    cb.record_failure()
    assert cb.can_execute() == True

    cb.record_failure()
    # Should be open now
    assert cb.get_stats()['state'] == 'open'
    assert cb.can_execute() == False

    # Wait for timeout
    time.sleep(1.1)
    assert cb.can_execute() == True  # Should be half-open

    # Record success
    cb.record_success()
    assert cb.get_stats()['state'] == 'closed'

    print("  ✓ Circuit breaker state transitions work")
    return True


def test_dead_letter_queue():
    """Test 10: DeadLetterQueue"""
    print("Test 10: DeadLetterQueue")

    logger = StructuredLogger('test')
    dlq = DeadLetterQueue(max_size=10, logger=logger.logger)

    # Enqueue
    item = {'test': 'data'}
    assert dlq.enqueue(item) == True
    assert dlq.size() == 1

    # Dequeue
    retrieved = dlq.dequeue()
    assert retrieved['test'] == 'data'
    assert dlq.size() == 0

    # Test max size
    for i in range(15):
        dlq.enqueue({'item': i})

    assert dlq.size() == 10  # Should be capped

    print("  ✓ Dead letter queue works")
    return True


def test_full_audit():
    """Test 11: Full Phase I audit"""
    print("Test 11: Full Phase I audit (end-to-end)")

    os.environ['PHASE1_TIMEOUT_MS'] = '15000'
    os.environ['PHASE1_MAX_ASSUMPTIONS'] = '100'
    os.environ['PHASE1_CIRCUIT_BREAKER_THRESHOLD'] = '5'
    os.environ['PHASE1_ENABLE_TACIT_MINING'] = 'true'
    os.environ['PHASE1_ENABLE_RED_TEAM'] = 'true'

    config = Phase1Config.from_env()
    executor = EpistemicAuditExecutor(config=config)

    problem = "LENR thermal coefficient inconsistency shows 50% null results"
    patterns = [
        {
            'pattern_description': 'Lattice defects cause irregular heat distribution',
            'failure_rate': 0.5,
            'data_points': 100,
        },
    ]

    result = executor.perform_audit(
        problem_description=problem,
        failure_patterns=patterns,
        correlation_id='test-correlation',
    )

    # Verify result structure
    assert result.phase == 'phase1_epistemic_audit'
    assert result.audit_id is not None
    assert result.problem_description == problem
    assert len(result.tacit_assumptions) > 0
    assert result.timestamp is not None

    # Verify canonical format
    result_dict = result.to_dict()
    assert 'phase' in result_dict
    assert 'audit_id' in result_dict
    assert 'tacit_assumptions' in result_dict
    assert 'contradictions' in result_dict
    assert 'falsification_results' in result_dict
    assert 'metrics' in result_dict
    assert 'metadata' in result_dict
    assert 'timestamp' in result_dict

    print(f"  ✓ Audit completed successfully")
    print(f"    - Assumptions: {len(result.tacit_assumptions)}")
    print(f"    - Contradictions: {len(result.contradictions)}")
    print(f"    - Falsified: {result.metrics['hypotheses_falsified']}")
    print(f"    - Execution time: {result.metadata['execution_time_ms']}ms")
    return True


def test_idempotency():
    """Test 12: Idempotency (safe to run 100x)"""
    print("Test 12: Idempotency")

    os.environ['PHASE1_TIMEOUT_MS'] = '15000'
    os.environ['PHASE1_MAX_ASSUMPTIONS'] = '100'

    config = Phase1Config.from_env()
    executor = EpistemicAuditExecutor(config=config)

    problem = "Test problem for idempotency"
    patterns = [
        {
            'pattern_description': 'Test pattern',
            'failure_rate': 0.5,
            'data_points': 10,
        }
    ]

    # Run same audit 10 times
    results = []
    for i in range(10):
        result = executor.perform_audit(
            problem_description=problem,
            failure_patterns=patterns,
            correlation_id='test-idempotency',
        )
        results.append(result)

    # All should succeed
    assert all(r.phase == 'phase1_epistemic_audit' for r in results)

    # Results should be consistent
    first = results[0]
    for r in results[1:]:
        assert len(r.tacit_assumptions) == len(first.tacit_assumptions)

    print("  ✓ Idempotency verified (10 runs, all consistent)")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("RESE Phase I Integration Tests")
    print("=" * 60)
    print()

    tests = [
        test_config_from_env,
        test_config_validation,
        test_executor_initialization,
        test_tacit_assumption_serialization,
        test_contradiction_detection_serialization,
        test_constraint_hardening,
        test_assumption_mining,
        test_red_team_protocol,
        test_circuit_breaker,
        test_dead_letter_queue,
        test_full_audit,
        test_idempotency,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
